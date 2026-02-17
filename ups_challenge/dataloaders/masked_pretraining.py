"""Dataloader for HuBERT masked pre-training.

Reads a pre-built index (from prepare_pretraining_index.py) that maps
(tar_number, key) -> {start_sec, end_sec, labels, language}.  Streams audio
from HuggingFace tars via WebDataset, extracts the specific chunk for each
matching file, and returns (waveform, labels) pairs.

Language stratification is baked into the chunk index; the dataloader uses
a single WebDataset over all tars for finite, deterministic epochs.
"""

import os
import pickle
import re
import subprocess
from collections import defaultdict

import numpy as np
import torch
import webdataset as wds
from torchcodec.decoders import AudioDecoder

# ---------------------------------------------------------------------------
# Error tracking (per-worker counters, logged with rate limiting)
# ---------------------------------------------------------------------------
_error_counts = {"decode": 0, "wds": 0}


def _warn_and_continue(exn):
    """WebDataset error handler that logs instead of silently skipping."""
    _error_counts["wds"] += 1
    n = _error_counts["wds"]
    if n <= 3 or (n <= 100 and n % 10 == 0) or n % 500 == 0:
        print(f"  [WebDataset] error #{n}: {type(exn).__name__}: {exn}")
    return True


def _download_tar(tar_number, hf_token, cache_dir):
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
    os.makedirs(cache_dir, exist_ok=True)
    temp = dest + f".tmp{os.getpid()}"
    subprocess.run(
        ["curl", "-s", "-L", "-o", temp, "-H", f"Authorization:Bearer {hf_token}", url],
        check=True,
    )
    os.rename(temp, dest)
    return dest


def _build_tar_urls(tar_numbers, hf_token, cache_dir=None):
    """Build local paths (cached) or pipe:curl URLs for tar numbers."""
    if cache_dir:
        return [_download_tar(tn, hf_token, cache_dir) for tn in sorted(tar_numbers)]
    token = f"Authorization:Bearer {hf_token}"
    urls = []
    for tn in sorted(tar_numbers):
        tn_str = str(tn).zfill(6)
        folder = "audio" if int(tn) <= 5000 else "audio2"
        raw = (
            f"https://huggingface.co/datasets/MLCommons/"
            f"unsupervised_peoples_speech/resolve/main/{folder}/{tn_str}.tar?download=True"
        )
        urls.append(f"pipe:curl -s -L {raw} -H {token}")
    return urls


def _build_lookup(entries):
    """Build (tar_number, key) -> list[entry] lookup from index entries."""
    lookup = {}
    for entry in entries:
        k = (entry["tar_number"], entry["key"])
        if k not in lookup:
            lookup[k] = []
        lookup[k].append(entry)
    return lookup


def _decode_pretraining(sample, lookup, target_sr=16000):
    """Decode a sample and pair with pre-computed labels if it is in the index.

    Returns list of dicts with 'waveform' (np.ndarray) and 'labels' (np.ndarray).
    """
    mp3_bytes, key, url = sample

    # Extract tar number from URL
    m = re.search(r"/(\d+)\.tar", url)
    if m is None:
        return []
    tar_number = m.group(1).zfill(6)

    entries = lookup.get((tar_number, os.path.basename(key)))
    if entries is None:
        return []

    results = []
    try:
        decoder = AudioDecoder(source=mp3_bytes, sample_rate=target_sr, num_channels=1)
        full_audio = decoder.get_all_samples().data.squeeze(0)

        for entry in entries:
            start_sample = int(entry["start_sec"] * target_sr)
            end_sample = int(entry["end_sec"] * target_sr)
            waveform = full_audio[start_sample:end_sample]

            results.append({
                "waveform": waveform.numpy(),
                "labels": entry["labels"],
            })

    except Exception as e:
        _error_counts["decode"] += 1
        n = _error_counts["decode"]
        if n <= 3 or (n <= 100 and n % 10 == 0) or n % 500 == 0:
            print(f"  [decode] error #{n} ({os.path.basename(key)}): "
                  f"{type(e).__name__}: {e}")
        return []

    return results


def collate_pretraining(batch):
    """Stack waveforms -> [B, T], pad labels -> [B, max_frames] (pad=-100)."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    waveforms = [torch.from_numpy(b["waveform"]) for b in batch]
    labels = [torch.from_numpy(b["labels"].astype(np.int64)) for b in batch]

    # Pad waveforms
    max_wav_len = max(w.shape[0] for w in waveforms)
    padded_wav = torch.zeros(len(waveforms), max_wav_len)
    attention_mask = torch.zeros(len(waveforms), max_wav_len, dtype=torch.long)
    for i, w in enumerate(waveforms):
        padded_wav[i, : w.shape[0]] = w
        attention_mask[i, : w.shape[0]] = 1

    # Pad labels (use -100 for ignore_index in cross-entropy)
    max_label_len = max(l.shape[0] for l in labels)
    padded_labels = torch.full((len(labels), max_label_len), -100, dtype=torch.long)
    for i, l in enumerate(labels):
        padded_labels[i, : l.shape[0]] = l

    return {
        "waveform": padded_wav,
        "attention_mask": attention_mask,
        "labels": padded_labels,
    }


# Helper to flatten list of lists
def _flatten_list(stream):
    for sample in stream:
        if isinstance(sample, list):
            for x in sample:
                yield x
        else:
            yield sample


def build_pretraining_dataset(index_path, hf_token=None, target_sr=16000, cache_dir="./data/tar_cache"):
    """Build a WebDataset that yields {waveform, labels} from a pretraining index.

    Uses a single WebDataset over all tars for finite epochs. If entries have
    a "language" field, prints the language distribution for visibility.

    Args:
        index_path: Path to the pickle index produced by prepare_pretraining_index.py.
        hf_token: HuggingFace token (falls back to HF_TOKEN env var).
        target_sr: Target sample rate for audio decoding.
    """
    if hf_token is None:
        hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        raise ValueError("HF_TOKEN is not set")

    with open(index_path, "rb") as f:
        index_entries = pickle.load(f)

    # Check if entries have language info
    has_language = any("language" in e for e in index_entries)

    if has_language:
        return _build_multilang_dataset(index_entries, hf_token, target_sr, cache_dir)
    else:
        return _build_single_dataset(index_entries, hf_token, target_sr, cache_dir)


def _build_single_dataset(index_entries, hf_token, target_sr, cache_dir=None):
    """Build a single WebDataset over all entries (legacy path)."""
    lookup = _build_lookup(index_entries)
    tar_numbers = {e["tar_number"] for e in index_entries}
    urls = _build_tar_urls(tar_numbers, hf_token, cache_dir)

    dataset = (
        wds.WebDataset(urls, shardshuffle=False, handler=_warn_and_continue)
        .to_tuple("mp3", "__key__", "__url__", handler=_warn_and_continue)
        .map(lambda s: _decode_pretraining(s, lookup, target_sr))
        .compose(_flatten_list)
        .shuffle(1000)
    )

    return dataset


def _build_multilang_dataset(index_entries, hf_token, target_sr, cache_dir=None):
    """Build a single WebDataset from multilang index entries.

    Language stratification is already in the chunk index. We use one
    WebDataset over all tars → finite epochs, no RandomMix.
    """
    # Print language distribution for visibility
    entries_by_lang = defaultdict(int)
    for entry in index_entries:
        entries_by_lang[entry.get("language", "unknown")] += 1

    print(f"Building dataset: {len(entries_by_lang)} languages, "
          f"{len(index_entries)} entries")
    for lang in sorted(entries_by_lang):
        print(f"  {lang}: {entries_by_lang[lang]} entries")

    lookup = _build_lookup(index_entries)
    tar_numbers = {e["tar_number"] for e in index_entries}
    urls = _build_tar_urls(tar_numbers, hf_token, cache_dir)

    dataset = (
        wds.WebDataset(urls, shardshuffle=False, handler=_warn_and_continue)
        .to_tuple("mp3", "__key__", "__url__", handler=_warn_and_continue)
        .map(lambda s: _decode_pretraining(s, lookup, target_sr))
        .compose(_flatten_list)
        .shuffle(5000)
    )
    return dataset
