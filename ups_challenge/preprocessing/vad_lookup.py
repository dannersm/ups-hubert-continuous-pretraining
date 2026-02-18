import json
import pickle
import os
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

VAD_SAMPLE_RATE = 16000

def _parse_speech_segments(timestamps_list):
    if not timestamps_list:
        return None
    segments = []
    for ts in timestamps_list:
        if "start" in ts and "end" in ts:
            segments.append((
                float(ts["start"]) / VAD_SAMPLE_RATE,
                float(ts["end"]) / VAD_SAMPLE_RATE,
            ))
    return segments if segments else None

def build_sharded_vad(
    jsonl_path="./data/vad_results.jsonl",
    output_dir="./data/vad_shards",
    flush_every=10_000):

    jsonl_path = Path(jsonl_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading from: {jsonl_path}")
    print(f"Writing shards to: {output_dir}")

    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"{jsonl_path} not found. Remember to download the vad_results.jsonl file first, "
            "you can delete it after process is completed."
        )
    
    print("Counting lines...")
    with open(jsonl_path, 'r') as f:
        total_lines = sum(1 for _ in f)
    
    vad_shards = defaultdict(dict)
    processed = 0
    kept_total = 0
    kept_with_speech = 0
    
    print(f"Processing {total_lines:,} lines...")
    with open(jsonl_path, 'r') as f:
        for line in tqdm(f, total=total_lines, desc="Building shards"):
            line = line.strip()
            if not line:
                continue
            
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            processed += 1
            
            for full_path, data in obj.items():
                filename = os.path.basename(full_path)
                if not filename.endswith(".mp3"):
                    filename += ".mp3"
                
                tar_number = data.get("tar_number")
                duration = data.get("duration", 0.0)
                timestamps = data.get("timestamps")
                
                if not tar_number:
                    continue
                
                if isinstance(tar_number, int):
                    tar_number = str(tar_number).zfill(6)
                elif isinstance(tar_number, str):
                    tar_number = tar_number.zfill(6)
                
                segments = _parse_speech_segments(timestamps)
                
                # store all files, even those without speech
                vad_shards[tar_number][filename] = {
                    "segments": segments,
                    "duration": duration
                }
                kept_total += 1
                
                if segments:  # Count files with speech
                    kept_with_speech += 1
            
            # Flush every N lines
            if processed % flush_every == 0:
                _flush_shards(vad_shards, output_dir)
                vad_shards.clear()
    
    # Final flush
    _flush_shards(vad_shards, output_dir)
    
    pkl_files = list(output_dir.glob("*.pkl"))
    print(f"Done!")
    print(f"   Processed: {processed:,} lines")
    print(f"   Kept total: {kept_total:,} files")
    print(f"   With speech: {kept_with_speech:,} files")
    print(f"   Without speech: {kept_total - kept_with_speech:,} files")
    print(f"   Created: {len(pkl_files)} shard files")

def _flush_shards(vad_shards, output_dir):
    for tar_number, shard in vad_shards.items():
        if not shard:
            continue
        
        shard_path = output_dir / f"{tar_number}.pkl"
        
        if shard_path.exists():
            with open(shard_path, "rb") as f:
                existing = pickle.load(f)
        else:
            existing = {}
        
        existing.update(shard)
        
        tmp_path = shard_path.with_suffix(".tmp")
        with open(tmp_path, "wb") as f:
            pickle.dump(existing, f, protocol=4)
        tmp_path.replace(shard_path)

if __name__ == "__main__":
    build_sharded_vad()