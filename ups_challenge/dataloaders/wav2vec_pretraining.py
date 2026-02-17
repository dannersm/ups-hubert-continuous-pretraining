"""Dataloader for wav2vec 2.0 self-supervised pre-training.

Label-free variant of masked_pretraining.py — reads a chunk index
(from build_chunk_index.py) that maps (tar_number, key) -> {start_sec, end_sec}.
No k-means labels are needed since wav2vec2 uses internal Gumbel-softmax
quantization.
"""

import io
import os
import pickle
import re

import numpy as np
import torch
import torchaudio
import webdataset as wds

from ups_challenge.dataloaders.masked_pretraining import (
    _build_lookup,
    _build_tar_urls,
    _download_tar,
    _flatten_list,
    _warn_and_continue,
)

# ---------------------------------------------------------------------------
# Error tracking
# ---------------------------------------------------------------------------
_error_counts = {"decode": 0}


def _decode_chunks(sample, lookup, target_sr=16000):
    """Decode audio chunks from a tar sample. Returns waveform only (no labels).

    Returns list of dicts with 'waveform' (np.ndarray).
    """
    mp3_bytes, key, url = sample

    m = re.search(r"/(\d+)\.tar", url)
    if m is None:
        return []
    tar_number = m.group(1).zfill(6)

    entries = lookup.get((tar_number, os.path.basename(key)))
    if entries is None:
        return []

    results = []
    try:
        # Load MP3 from bytes
        with io.BytesIO(mp3_bytes) as f:
            full_audio, sr = torchaudio.load(f, format="mp3")

        # Resample if necessary
        if sr != target_sr:
            full_audio = torchaudio.functional.resample(full_audio, sr, target_sr)

        # Convert to mono if necessary
        if full_audio.shape[0] > 1:
            full_audio = full_audio.mean(dim=0, keepdim=True)
        
        # Squeeze channel dim: [1, T] -> [T]
        full_audio = full_audio.squeeze(0)

        for entry in entries:
            start_sample = int(entry["start_sec"] * target_sr)
            end_sample = int(entry["end_sec"] * target_sr)
            
            # Simple bounds check
            start_sample = max(0, start_sample)
            end_sample = min(len(full_audio), end_sample)
            
            if end_sample > start_sample:
                waveform = full_audio[start_sample:end_sample]
                results.append({"waveform": waveform.numpy()})

    except Exception as e:
        _error_counts["decode"] += 1
        n = _error_counts["decode"]
        if n <= 3 or (n <= 100 and n % 10 == 0) or n % 500 == 0:
            print(f"  [decode] error #{n} ({os.path.basename(key)}): "
                  f"{type(e).__name__}: {e}")
        return []

    return results


def collate_fn(batch):
    """Pad waveforms -> [B, T], create attention_mask -> [B, T]."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    waveforms = [torch.from_numpy(b["waveform"]) for b in batch]

    max_wav_len = max(w.shape[0] for w in waveforms)
    padded_wav = torch.zeros(len(waveforms), max_wav_len)
    attention_mask = torch.zeros(len(waveforms), max_wav_len, dtype=torch.long)
    for i, w in enumerate(waveforms):
        padded_wav[i, : w.shape[0]] = w
        attention_mask[i, : w.shape[0]] = 1

    return {
        "waveform": padded_wav,
        "attention_mask": attention_mask,
    }


def build_wav2vec_dataset(
    index_path, hf_token=None, target_sr=16000, cache_dir="./data/tar_cache",
):
    """Build a WebDataset that yields {waveform} from a chunk index.

    Args:
        index_path: Path to the pickle chunk index (from build_chunk_index.py).
        hf_token: HuggingFace token (falls back to HF_TOKEN env var).
        target_sr: Target sample rate for audio decoding.
        cache_dir: Directory to cache downloaded tars.
    """
    if hf_token is None:
        hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        raise ValueError("HF_TOKEN is not set")

    with open(index_path, "rb") as f:
        index_entries = pickle.load(f)

    lookup = _build_lookup(index_entries)
    tar_numbers = {e["tar_number"] for e in index_entries}
    urls = _build_tar_urls(tar_numbers, hf_token, cache_dir)

    dataset = (
        wds.WebDataset(urls, shardshuffle=False, handler=_warn_and_continue)
        .to_tuple("mp3", "__key__", "__url__", handler=_warn_and_continue)
        .map(lambda s: _decode_chunks(s, lookup, target_sr))
        .compose(_flatten_list)
        .shuffle(5000)
    )
    return dataset
