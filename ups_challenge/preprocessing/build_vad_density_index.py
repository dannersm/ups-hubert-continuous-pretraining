"""Compute per-tar VAD density index from local vad_shards (run once, cache result).

Reads all .pkl files in vad_shards/, computes for each tar:
  density = total_speech_sec / total_duration_sec

Saves a dict {tar_number: density} to data/vad_density_index.pkl.

Usage:
    python -m ups_challenge.preprocessing.build_vad_density_index \
        --vad_base_dir ./data/vad_shards \
        --output ./data/vad_density_index.pkl
"""

import argparse
import pickle
from pathlib import Path

from tqdm import tqdm


def compute_density(shard: dict) -> float:
    total_speech = 0.0
    total_dur = 0.0
    for vad_data in shard.values():
        if not isinstance(vad_data, dict):
            continue
        dur = vad_data.get("duration") or 0.0
        segs = vad_data.get("segments") or []
        total_dur += dur
        total_speech += sum(max(0.0, e - s) for s, e in segs)
    return total_speech / total_dur if total_dur > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Build per-tar VAD density index (run once)"
    )
    parser.add_argument("--vad_base_dir", default="./data/vad_shards")
    parser.add_argument("--output", default="./data/vad_density_index.pkl")
    args = parser.parse_args()

    vad_base = Path(args.vad_base_dir)
    shard_files = sorted(vad_base.glob("*.pkl"))
    print(f"Found {len(shard_files)} VAD shards in {vad_base}")

    density_index = {}
    for path in tqdm(shard_files, desc="Computing VAD density"):
        tar_number = path.stem
        with open(path, "rb") as f:
            shard = pickle.load(f)
        density_index[tar_number] = compute_density(shard)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(density_index, f, protocol=4)

    densities = list(density_index.values())
    print(f"\nSaved → {out_path}  ({len(density_index)} tars)")
    print(f"Density stats:  min={min(densities):.1%}  "
          f"median={sorted(densities)[len(densities)//2]:.1%}  "
          f"max={max(densities):.1%}")
    low = [(t, d) for t, d in density_index.items() if d < 0.5]
    print(f"Tars below 50% density: {len(low)} → {sorted(low, key=lambda x: x[1])[:10]}")


if __name__ == "__main__":
    main()
