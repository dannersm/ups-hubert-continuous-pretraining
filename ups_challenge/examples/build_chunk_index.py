"""Phase 1: Build a static chunk index from local VAD shards (no audio download).

Tar selection strategy (simple & predictable):
  1. Filter out tars below --min_vad_ratio (VAD density threshold).
  2. Score each remaining tar:
       tar_score = sum(lang_score(l) for l in tar_langs) * vad_density
     where lang_score is based on dataset hours:
       > 1000h  → 1 pt  (abundant, low priority)
       100-1000h → 2 pts
       10-100h  → 3 pts
       < 10h    → 4 pts  (scarce, high priority)
  3. Sort tars by score descending.
  4. Process tars in order, collecting chunks with vad_chunks_with_density(),
     until --total_hours is reached.

Output is sorted by (tar_number, file_order_in_shard, start_sec) so Phase 2
can load each tar exactly once in a single sequential pass.

Prerequisites:
  - data/vad_density_index.pkl   (build with build_vad_density_index.py)
  - data/lid_index.pkl
  - ups_challenge/examples/lang_speech_hours.json

Usage:
    python -m ups_challenge.examples.build_chunk_index \\
        --total_hours 100 \\
        --min_vad_ratio 0.5 \\
        --vad_base_dir ./data/vad_shards \\
        --vad_density_index ./data/vad_density_index.pkl \\
        --lid_index_path ./data/lid_index.pkl \\
        --lang_hours_path ./ups_challenge/examples/lang_speech_hours.json \\
        --output ./data/chunk_index_100h.pkl
"""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote

from tqdm import tqdm


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def vad_chunks_with_density(segments, chunk_sec=10.0, min_density=0.8):
    """Non-overlapping 10s chunks where speech fills >= min_density fraction.

    Anchors to each VAD segment end and scans leftward in chunk_sec steps.
    Captures both long segments and clusters of short segments that together
    fill the window.
    """
    if not segments:
        return []
    segments = sorted(segments)
    min_speech = min_density * chunk_sec

    def speech_in_window(start, end):
        return sum(
            min(se, end) - max(ss, start)
            for ss, se in segments
            if se > start and ss < end
        )

    chunks = []
    used_before = float('inf')

    for _, seg_end in reversed(segments):
        anchor = seg_end
        while anchor - chunk_sec >= 0 and anchor <= used_before:
            start = anchor - chunk_sec
            if speech_in_window(start, anchor) >= min_speech:
                chunks.append((start, anchor))
                used_before = start
                anchor -= chunk_sec
            else:
                anchor -= chunk_sec
    return sorted(chunks)


# ---------------------------------------------------------------------------
# Language scoring
# ---------------------------------------------------------------------------

def lang_score(hours: float) -> int:
    """Higher score = scarcer language = higher tar priority."""
    if hours > 1000:
        return 1
    elif hours > 100:
        return 2
    elif hours > 10:
        return 3
    else:
        return 4


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Build chunk index — simple scoring, no balancing"
    )
    parser.add_argument("--total_hours", type=float, default=100.0,
                        help="Stop collecting once this many hours are gathered")
    parser.add_argument("--chunk_sec", type=float, default=10.0)
    parser.add_argument("--min_chunk_density", type=float, default=0.8,
                        help="Min speech fraction within each 10s chunk window")
    parser.add_argument("--min_vad_ratio", type=float, default=0.5,
                        help="Exclude tars whose overall VAD density is below this")
    parser.add_argument("--vad_base_dir", type=str, default="./data/vad_shards")
    parser.add_argument("--vad_density_index", type=str,
                        default="./data/vad_density_index.pkl",
                        help="Pre-computed density index (build_vad_density_index.py)")
    parser.add_argument("--lid_index_path", type=str, default="./data/lid_index.pkl")
    parser.add_argument("--lang_hours_path", type=str,
                        default="./ups_challenge/examples/lang_speech_hours.json")
    parser.add_argument("--output", type=str, default="./data/chunk_index_{hours}h.pkl",
                        help="Output path; {hours} is replaced with --total_hours value")
    args = parser.parse_args()

    target_chunks = int(args.total_hours * 3600 / args.chunk_sec)
    print(f"Target: {args.total_hours} h = {target_chunks:,} chunks of {args.chunk_sec}s")

    # ------------------------------------------------------------------
    # Load support data
    # ------------------------------------------------------------------
    print("\nLoading vad_density_index...")
    density_path = Path(args.vad_density_index)
    if not density_path.exists():
        raise FileNotFoundError(
            f"VAD density index not found: {density_path}\n"
            "Run: python -m ups_challenge.examples.build_vad_density_index"
        )
    with open(density_path, "rb") as f:
        vad_density: dict[str, float] = pickle.load(f)
    print(f"  {len(vad_density)} tars in density index")

    print("Loading lid_index...")
    with open(args.lid_index_path, "rb") as f:
        lid_index_raw = pickle.load(f)

    # Normalize keys: (zero-padded tar str, filename with .mp3)
    lid_index: dict[tuple, str] = {}
    for (tar_str, filename), lang in lid_index_raw.items():
        tar_str = str(tar_str).zfill(6)
        if not filename.endswith(".mp3"):
            filename = filename + ".mp3"
        lid_index[(tar_str, filename)] = lang
    print(f"  {len(lid_index):,} entries")

    print("Loading lang_speech_hours...")
    with open(args.lang_hours_path) as f:
        lang_hours: dict[str, float] = json.load(f)

    # ------------------------------------------------------------------
    # Filter and score tars
    # ------------------------------------------------------------------
    # Build tar -> set of languages from lid_index
    tar_langs: dict[str, set] = defaultdict(set)
    for (tar_str, _), lang in lid_index.items():
        tar_langs[tar_str].add(lang)

    # Filter by VAD ratio
    all_tars = sorted(vad_density.keys())
    filtered = [t for t in all_tars if vad_density[t] >= args.min_vad_ratio]
    excluded = len(all_tars) - len(filtered)
    print(f"\nVAD filter ({args.min_vad_ratio:.0%}): "
          f"excluded {excluded}, kept {len(filtered)} tars")

    # Score: sum of lang scores weighted by VAD density
    def tar_score(tar: str) -> float:
        langs = tar_langs.get(tar, set())
        if not langs:
            return 0.0
        score = sum(lang_score(lang_hours.get(l, 0.0)) for l in langs)
        return score * vad_density[tar]

    scored = sorted(filtered, key=tar_score, reverse=True)
    print(f"Top-5 tars by score:")
    for t in scored[:5]:
        langs = tar_langs.get(t, set())
        s = lang_score
        ls = sum(lang_score(lang_hours.get(l, 0.0)) for l in langs)
        print(f"  {t}  density={vad_density[t]:.1%}  "
              f"lang_score_sum={ls}  "
              f"tar_score={ls * vad_density[t]:.2f}  "
              f"langs={len(langs)}")

    # ------------------------------------------------------------------
    # Process tars in score order until target_hours reached
    # ------------------------------------------------------------------
    vad_base = Path(args.vad_base_dir)
    all_chunks = []
    tars_used = []

    print(f"\nProcessing tars (stopping at {target_chunks:,} chunks)...")
    for tar in tqdm(scored, desc="Tars"):
        if len(all_chunks) >= target_chunks:
            break

        shard_path = vad_base / f"{tar}.pkl"
        if not shard_path.exists():
            continue

        with open(shard_path, "rb") as f:
            shard = pickle.load(f)

        tar_new = 0
        for file_order, (filename, vad_data) in enumerate(shard.items()):
            if not isinstance(vad_data, dict):
                continue
            segs = vad_data.get("segments") or []
            if not segs:
                continue

            # VAD shard filenames are URL-encoded; WDS keys are decoded UTF-8
            key = unquote(filename[:-4] if filename.endswith(".mp3") else filename)
            lang = lid_index.get((tar, filename), "unknown")

            for start, end in vad_chunks_with_density(segs, args.chunk_sec,
                                                       args.min_chunk_density):
                all_chunks.append({
                    "tar_number": tar,
                    "key": key,
                    "start_sec": start,
                    "end_sec": end,
                    "language": lang,
                    "_file_order": file_order,
                })
                tar_new += 1

        if tar_new > 0:
            tars_used.append(tar)

        if len(all_chunks) >= target_chunks:
            break

    # Filter out non-speech / unknown language tags
    all_chunks = [c for c in all_chunks
                  if c["language"] not in ("nospeech", "unknown")]

    # Trim to exactly target_chunks
    if len(all_chunks) > target_chunks:
        all_chunks = all_chunks[:target_chunks]

    print(f"\nCollected {len(all_chunks):,} chunks from {len(tars_used)} tars "
          f"({len(all_chunks) * args.chunk_sec / 3600:.1f} h)")

    # ------------------------------------------------------------------
    # Sort for sequential Phase 2 streaming: (tar, file_order, start)
    # ------------------------------------------------------------------
    all_chunks.sort(key=lambda c: (c["tar_number"], c["_file_order"], c["start_sec"]))

    # Strip internal _file_order field
    final_index = [{k: v for k, v in c.items() if k != "_file_order"}
                   for c in all_chunks]

    # ------------------------------------------------------------------
    # Download plan summary
    # ------------------------------------------------------------------
    from collections import Counter
    tar_counter = Counter(c["tar_number"] for c in final_index)
    lang_counter = Counter(c["language"] for c in final_index)
    unique_tars = sorted(tar_counter)
    total_h = len(final_index) * args.chunk_sec / 3600

    print("\n" + "=" * 52)
    print("=== DOWNLOAD PLAN FOR PHASE 2 ===")
    print(f"Tars needed:  {len(unique_tars)}")
    print(f"Total chunks: {len(final_index):,}  ({total_h:.1f} h)")
    print(f"Languages:    {len(lang_counter)}")
    print("Tar breakdown:")
    for tar in unique_tars:
        n = tar_counter[tar]
        h = n * args.chunk_sec / 3600
        tar_langs_set = {c["language"] for c in final_index if c["tar_number"] == tar}
        lang_str = ",".join(sorted(tar_langs_set)[:8])
        if len(tar_langs_set) > 8:
            lang_str += f",...(+{len(tar_langs_set)-8})"
        print(f"  {tar}  {n:>6,} chunks  {h:.1f}h  langs: {lang_str}")
    est_gb = len(unique_tars) * 4.5
    print(f"Estimated download: ~{est_gb:.1f} GB  (~4.5 GB/tar estimate)")
    print("=" * 52)

    print("\nLanguage breakdown:")
    for lang, cnt in sorted(lang_counter.items(), key=lambda x: -x[1])[:30]:
        print(f"  {lang:6s}  {cnt:6,} chunks  {cnt * args.chunk_sec / 3600:.2f} h")
    if len(lang_counter) > 30:
        print(f"  ... and {len(lang_counter)-30} more languages")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_path = Path(args.output.replace("{hours}", str(int(args.total_hours))))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(final_index, f, protocol=4)
    print(f"\nSaved chunk index → {out_path}  ({len(final_index):,} entries)")


if __name__ == "__main__":
    main()
