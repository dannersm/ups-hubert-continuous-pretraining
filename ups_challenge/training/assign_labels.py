"""Phase 2: Stream tars sequentially, extract chunks, compute MFCCs, fit k-means.

Reads the chunk index produced by build_chunk_index.py (sorted by tar_number),
streams tars one at a time via WebDataset, decodes audio chunks, computes
39-dim MFCCs, fits MiniBatchKMeans incrementally (partial_fit), and assigns
labels on-the-fly.

The process is fully resumable: a checkpoint is saved every --save_every_tars
tars containing the KMeans model, running normalisation stats, completed tars,
and labels assigned so far.

Usage:
    python -m ups_challenge.training.assign_labels \\
        --index ./data/chunk_index_100h.pkl \\
        --n_clusters 100 --output_dir ./data \\
        --hf_token $HF_TOKEN

Output files:
    data/pretraining_index_<N>h.pkl   -- chunk entries + 'labels' field
    data/kmeans_100.pkl               -- fitted k-means model
    data/kmeans_norm_100.pkl          -- {mean, std, kmeans} for reproducibility
"""

import argparse
import os
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torchaudio
import webdataset as wds
from sklearn.cluster import MiniBatchKMeans
from torchcodec.decoders import AudioDecoder
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoConfig, AutoModel


# ---------------------------------------------------------------------------
# Audio / MFCC
# ---------------------------------------------------------------------------

def extract_mfcc(waveform, sample_rate=16000, n_mfcc=13, hop_length=320):
    """39-dim MFCC (13 + delta + delta²), center=False → 499 frames per 10 s."""
    mfcc_t = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft": 400, "hop_length": hop_length,
                   "n_mels": 23, "center": False},
    )
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    mfcc = mfcc_t(waveform)
    delta = torchaudio.functional.compute_deltas(mfcc)
    delta2 = torchaudio.functional.compute_deltas(delta)
    return torch.cat([mfcc, delta, delta2], dim=1).squeeze(0).T.numpy()


# ---------------------------------------------------------------------------
# Running statistics (Welford's algorithm)
# ---------------------------------------------------------------------------

class RunningStats:
    """Incremental mean/std over feature dimensions using Welford's algorithm."""

    def __init__(self, n_features=39):
        self.n = 0
        self.mean = np.zeros(n_features, dtype=np.float64)
        self.M2 = np.zeros(n_features, dtype=np.float64)

    def update(self, frames: np.ndarray):
        """Update with a batch of frames (N_frames, n_features)."""
        for x in frames:
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            delta2 = x - self.mean
            self.M2 += delta * delta2

    def update_batch(self, frames: np.ndarray):
        """Batch update — faster than row-by-row for large arrays."""
        batch_n = frames.shape[0]
        if batch_n == 0:
            return
        batch_mean = frames.mean(axis=0)
        batch_var = frames.var(axis=0)
        total_n = self.n + batch_n
        delta = batch_mean - self.mean
        self.mean = (self.n * self.mean + batch_n * batch_mean) / total_n
        self.M2 += batch_n * batch_var + delta ** 2 * self.n * batch_n / total_n
        self.n = total_n

    @property
    def std(self):
        if self.n < 2:
            return np.ones_like(self.mean)
        return np.sqrt(self.M2 / self.n)

    def state_dict(self):
        return {"n": self.n, "mean": self.mean.copy(), "M2": self.M2.copy()}

    @classmethod
    def from_state_dict(cls, d):
        obj = cls(n_features=len(d["mean"]))
        obj.n = d["n"]
        obj.mean = d["mean"]
        obj.M2 = d["M2"]
        return obj


# ---------------------------------------------------------------------------
# Embedding extractor
# ---------------------------------------------------------------------------

class EmbeddingExtractor:
    """GPU-batched hidden-state extraction from a HuggingFace speech model."""

    def __init__(self, model_name, layer_idx, device, batch_size=32):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        self.model = (
            AutoModel.from_pretrained(model_name, config=config)
            .to(device)
            .eval()
        )
        self.layer_idx = layer_idx
        self.device = device
        self.batch_size = batch_size
        self.hidden_size = config.hidden_size

    def extract(self, waveforms: list) -> list:
        """Batch-GPU extract hidden states from the specified layer.

        Args:
            waveforms: list of numpy arrays, each shape (T_i,).

        Returns:
            list of (T_i_enc, hidden_size) numpy arrays.
        """
        results = []
        for i in range(0, len(waveforms), self.batch_size):
            chunk = waveforms[i: i + self.batch_size]
            inputs = self.feature_extractor(
                chunk,
                sampling_rate=16000,
                padding=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                out = self.model(
                    inputs.input_values.to(self.device),
                    attention_mask=inputs.get("attention_mask", None),
                    output_hidden_states=True,
                )
            hidden = out.hidden_states[self.layer_idx]  # [B, T, D]
            for j, wav in enumerate(chunk):
                wav_len = len(wav)
                enc_len = self.model._get_feat_extract_output_lengths(wav_len)
                if isinstance(enc_len, torch.Tensor):
                    enc_len = enc_len.item()
                results.append(hidden[j, :enc_len].cpu().numpy())
        return results


# ---------------------------------------------------------------------------
# Tar download
# ---------------------------------------------------------------------------

def _download_tar(tar_number: str, hf_token: str, cache_dir: str) -> str:
    """Download a tar to cache_dir if not already cached, return local path."""
    tn_str = str(tar_number).zfill(6)
    dest = os.path.join(cache_dir, f"{tn_str}.tar")
    if os.path.exists(dest):
        return dest
    folder = "audio" if int(tar_number) <= 5000 else "audio2"
    url = (
        f"https://huggingface.co/datasets/MLCommons/"
        f"unsupervised_peoples_speech/resolve/main/{folder}/{tn_str}.tar?download=True"
    )
    temp = dest + f".tmp{os.getpid()}"
    import subprocess
    subprocess.run(
        ["curl", "-s", "-L", "-o", temp, "-H", f"Authorization:Bearer {hf_token}", url],
        check=True,
    )
    os.rename(temp, dest)
    return dest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Assign k-means labels to a Phase 1 chunk index"
    )
    parser.add_argument("--index", type=str, required=True,
                        help="Path to chunk index produced by build_chunk_index.py")
    parser.add_argument("--n_clusters", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="./data")
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default="./data/tar_cache",
                        help="Directory to cache downloaded tars (reused on subsequent runs)")
    parser.add_argument("--target_sr", type=int, default=16000)
    parser.add_argument("--save_every_tars", type=int, default=5,
                        help="Save checkpoint every N tars processed")
    parser.add_argument("--feature_type", type=str, default="mfcc",
                        choices=["mfcc", "embedding"],
                        help="Feature type for k-means: mfcc (default) or embedding")
    parser.add_argument("--embedding_model", type=str, default=None,
                        help="HF model name for embedding extraction (required when "
                             "feature_type=embedding)")
    parser.add_argument("--embedding_layer", type=int, default=9,
                        help="0-indexed hidden layer to extract (default: 9, "
                             "per WavLM paper section 5)")
    parser.add_argument("--embedding_batch_size", type=int, default=32,
                        help="Chunks per GPU batch for embedding extraction (default: 32)")
    parser.add_argument("--embedding_device", type=str, default=None,
                        help="Device for embedding model: 'cuda' or 'cpu' "
                             "(default: auto-detect)")
    args = parser.parse_args()

    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("Set HF_TOKEN env var or pass --hf_token")

    # Feature extraction setup
    embedding_extractor = None
    if args.feature_type == "embedding":
        if not args.embedding_model:
            raise ValueError("--embedding_model is required when --feature_type=embedding")
        emb_device = args.embedding_device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading embedding model: {args.embedding_model} "
              f"(layer {args.embedding_layer}, device {emb_device})")
        embedding_extractor = EmbeddingExtractor(
            model_name=args.embedding_model,
            layer_idx=args.embedding_layer,
            device=emb_device,
            batch_size=args.embedding_batch_size,
        )
        n_features = embedding_extractor.hidden_size
        print(f"  Embedding hidden size: {n_features}")
    else:
        n_features = 39

    # ------------------------------------------------------------------
    # Load index
    # ------------------------------------------------------------------
    print(f"Loading index from {args.index}...")
    with open(args.index, "rb") as f:
        index_entries = pickle.load(f)
    print(f"  {len(index_entries):,} entries")

    # Build lookup: (tar_number, key) -> list[entry_index]
    lookup: dict[tuple, list[int]] = defaultdict(list)
    for i, entry in enumerate(index_entries):
        lookup[(entry["tar_number"], entry["key"])].append(i)

    # Get ordered list of unique tars (index is already tar-sorted)
    seen = set()
    ordered_tars = []
    for entry in index_entries:
        t = entry["tar_number"]
        if t not in seen:
            ordered_tars.append(t)
            seen.add(t)
    print(f"  {len(ordered_tars)} unique tars, streaming sequentially")

    # ------------------------------------------------------------------
    # Resume from checkpoint or start fresh
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, "assign_labels_checkpoint.pkl")

    completed_tars: set = set()
    labels_dict: dict[int, np.ndarray] = {}  # idx -> labels array
    stats = RunningStats(n_features=n_features)
    kmeans = MiniBatchKMeans(n_clusters=args.n_clusters, batch_size=1024,
                             random_state=42, n_init=3)
    kmeans_initialized = False
    collected_count = 0

    if os.path.exists(ckpt_path):
        print(f"Resuming from {ckpt_path}...")
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        completed_tars = ckpt["completed_tars"]
        labels_dict = ckpt["labels_dict"]
        stats = RunningStats.from_state_dict(ckpt["running_stats"])
        kmeans = ckpt["kmeans"]
        kmeans_initialized = ckpt.get("kmeans_initialized", True)
        collected_count = len(labels_dict)
        print(f"  Restored {collected_count:,} labeled chunks from "
              f"{len(completed_tars)} tars")
        print(f"  Running stats: {stats.n:,} frames seen")

    remaining_tars = [t for t in ordered_tars if t not in completed_tars]
    print(f"  {len(remaining_tars)} tars remaining (of {len(ordered_tars)} total)")

    def _save_checkpoint():
        """Save lightweight checkpoint: KMeans + running stats + labels."""
        tmp = ckpt_path + f".tmp{os.getpid()}"
        with open(tmp, "wb") as f:
            pickle.dump({
                "completed_tars": completed_tars,
                "labels_dict": labels_dict,
                "running_stats": stats.state_dict(),
                "kmeans": kmeans,
                "kmeans_initialized": kmeans_initialized,
            }, f, protocol=4)
        os.rename(tmp, ckpt_path)

    # ------------------------------------------------------------------
    # Single-pass: extract features, partial_fit KMeans, predict labels
    # ------------------------------------------------------------------
    pbar = tqdm(total=len(index_entries), initial=collected_count,
                desc="Chunks processed", unit="chunk")
    tars_since_save = 0

    for tar_idx, tar_number in enumerate(remaining_tars):
        chunks_in_tar = sum(1 for e in index_entries if e["tar_number"] == tar_number)
        t0 = time.time()
        first_chunk_logged = False
        pbar.write(f"\n[{len(completed_tars)+1}/{len(ordered_tars)}] Tar {tar_number} "
                   f"({chunks_in_tar:,} chunks expected) ...")

        # Build per-tar lookup
        tar_lookup: dict[str, list[int]] = defaultdict(list)
        for (tn, key), idxs in lookup.items():
            if tn == tar_number:
                tar_lookup[key].extend(idxs)

        if not tar_lookup:
            completed_tars.add(tar_number)
            continue

        cache_dir = Path(args.cache_dir) if args.cache_dir else None
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

        local_tar = _download_tar(tar_number, hf_token, str(cache_dir))
        pbar.write(f"  Using {local_tar}")

        dataset = (
            wds.WebDataset(local_tar, shardshuffle=False,
                           handler=wds.handlers.ignore_and_continue)
            .to_tuple("mp3", "__key__", "__url__",
                      handler=wds.handlers.ignore_and_continue)
        )

        # Process features in micro-batches to avoid OOM on large tars
        BATCH_SIZE = 2000  # chunks per micro-batch (~300 MB peak)
        # Stores (idx, feature_array_or_waveform) depending on feature_type
        pending_chunks: list[tuple[int, np.ndarray]] = []

        def _flush_batch():
            """Process accumulated micro-batch: stats, partial_fit, predict."""
            nonlocal kmeans_initialized, collected_count
            if not pending_chunks:
                return

            if embedding_extractor is not None:
                # Embedding mode: run GPU batched extraction on raw waveforms
                waveforms_batch = [w for _, w in pending_chunks]
                features_list = embedding_extractor.extract(waveforms_batch)
                all_frames = np.concatenate(features_list, axis=0)
                per_chunk = [(idx, feat) for (idx, _), feat in zip(pending_chunks, features_list)]
            else:
                # MFCC mode: features are already computed
                all_frames = np.concatenate([m for _, m in pending_chunks], axis=0)
                per_chunk = pending_chunks

            stats.update_batch(all_frames)

            current_std = stats.std + 1e-8
            normalized = (all_frames - stats.mean) / current_std

            kmeans.partial_fit(normalized)
            kmeans_initialized = True

            offset = 0
            for idx, feat in per_chunk:
                n_frames = feat.shape[0]
                chunk_norm = normalized[offset:offset + n_frames]
                labels_dict[idx] = kmeans.predict(chunk_norm).astype(np.int16)
                offset += n_frames
                collected_count += 1
                pbar.update(1)

            del all_frames, normalized
            pending_chunks.clear()

        for mp3_bytes, key, _url in dataset:
            idxs = tar_lookup.get(os.path.basename(key))
            if idxs is None:
                continue

            try:
                decoder = AudioDecoder(source=mp3_bytes,
                                       sample_rate=args.target_sr,
                                       num_channels=1)
                full_audio = decoder.get_all_samples().data.squeeze(0)
            except Exception:
                continue

            sr = args.target_sr
            for idx in idxs:
                entry = index_entries[idx]
                start_sample = int(entry["start_sec"] * sr)
                end_sample = int(entry["end_sec"] * sr)
                waveform = full_audio[start_sample:end_sample]
                if waveform.shape[0] == 0:
                    continue

                if embedding_extractor is not None:
                    # Store raw waveform as numpy; embedding extracted in _flush_batch
                    wav_np = waveform.numpy() if isinstance(waveform, torch.Tensor) else np.asarray(waveform, dtype=np.float32)
                    pending_chunks.append((idx, wav_np))
                else:
                    mfcc = extract_mfcc(waveform, sample_rate=sr)
                    pending_chunks.append((idx, mfcc))

                if not first_chunk_logged:
                    pbar.write(f"  First chunk arrived after {time.time()-t0:.1f}s")
                    first_chunk_logged = True

            if len(pending_chunks) >= BATCH_SIZE:
                _flush_batch()

        # Flush remaining chunks for this tar
        _flush_batch()

        completed_tars.add(tar_number)
        tars_since_save += 1

        if tars_since_save >= args.save_every_tars:
            pbar.write(f"  Saving checkpoint ({collected_count:,} chunks, "
                       f"{len(completed_tars)} tars)...")
            _save_checkpoint()
            tars_since_save = 0

    # Final checkpoint
    _save_checkpoint()
    pbar.close()
    print(f"\nProcessed {collected_count:,} / {len(index_entries):,} chunks")

    # ------------------------------------------------------------------
    # Build labeled index
    # ------------------------------------------------------------------
    labeled_entries = []
    for i, entry in enumerate(index_entries):
        if i in labels_dict:
            entry["labels"] = labels_dict[i]
            labeled_entries.append(entry)

    if not labeled_entries:
        print("ERROR: No labels produced.")
        return

    print(f"Labeled entries: {len(labeled_entries):,}")

    # ------------------------------------------------------------------
    # Save final outputs
    # ------------------------------------------------------------------
    n_hours = len(labeled_entries) * 10 / 3600
    suffix = f"{round(n_hours)}h"

    idx_path = os.path.join(args.output_dir, f"pretraining_index_{suffix}.pkl")
    with open(idx_path, "wb") as f:
        pickle.dump(labeled_entries, f, protocol=4)
    print(f"Saved index  → {idx_path}  ({len(labeled_entries):,} entries)")

    km_path = os.path.join(args.output_dir, f"kmeans_{args.n_clusters}.pkl")
    with open(km_path, "wb") as f:
        pickle.dump(kmeans, f, protocol=4)
    print(f"Saved kmeans → {km_path}")

    mean = stats.mean.astype(np.float32)
    std = stats.std.astype(np.float32)
    norm_path = os.path.join(args.output_dir,
                             f"kmeans_norm_{args.n_clusters}.pkl")
    with open(norm_path, "wb") as f:
        pickle.dump({"mean": mean, "std": std, "kmeans": kmeans}, f, protocol=4)
    print(f"Saved kmeans + norm stats → {norm_path}")

    # Cleanup checkpoint
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        print(f"Removed checkpoint {ckpt_path}")

    # Summary
    unique_files = len({(e["tar_number"], e["key"]) for e in labeled_entries})
    unique_tars = len({e["tar_number"] for e in labeled_entries})
    label_lens = [len(e["labels"]) for e in labeled_entries]
    lang_counts = Counter(e.get("language", "unknown") for e in labeled_entries)
    print(f"\nFinal summary:")
    print(f"  Entries:      {len(labeled_entries):,}")
    print(f"  Unique files: {unique_files:,}")
    print(f"  Unique tars:  {unique_tars}")
    print(f"  Label lengths: min={min(label_lens)}, max={max(label_lens)}, "
          f"mode={Counter(label_lens).most_common(1)[0]}")
    print(f"  Running stats: mean range [{mean.min():.2f}, {mean.max():.2f}], "
          f"std range [{std.min():.4f}, {std.max():.4f}]")
    print(f"  Languages ({len(lang_counts)}):")
    for lang, cnt in sorted(lang_counts.items(), key=lambda x: -x[1]):
        print(f"    {lang:6s}  {cnt:6,} chunks")


if __name__ == "__main__":
    main()
