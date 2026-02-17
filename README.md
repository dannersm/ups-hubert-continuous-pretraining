# HuBERT Continued Pre-Training for the UPS Challenge

A **HuBERT Continued Pre-Training (CPT)** pipeline for the [MLCommons Unsupervised People's Speech (UPS) Challenge](https://mlcommons.org/datasets/peoples-speech/).

The full dataset contains ~750k hours of multilingual audio across 90+ languages. This pipeline works with **curated subsets** (~100-500h) selected via VAD density scoring and language-scarcity weighting, making it feasible to train on a single GPU.

## Prerequisites

- **Google Colab Pro** with a GPU runtime (A100 recommended, V100 or T4 also work)
- **HuggingFace account** with an access token that has permission to the [`MLCommons/unsupervised_peoples_speech`](https://huggingface.co/datasets/MLCommons/unsupervised_peoples_speech) dataset
- **Google Drive** with ~50+ GB of free space (for tar cache, indices, and checkpoints)

## Quick Start

1. **Open the notebook** — Upload `colab_hubert_pretraining.ipynb` to Google Colab (or open it directly from GitHub).

2. **Select a GPU runtime** — Go to `Runtime → Change runtime type → GPU` (A100 40GB is ideal).

3. **Set your HuggingFace token** — Either:
   - Add `HF_TOKEN` as a [Colab Secret](https://colab.research.google.com/) (recommended), or
   - Paste it directly in the notebook cell (Option 2).

4. **Run all cells sequentially** — The notebook handles everything: cloning the repo, mounting Drive, installing dependencies, and running each pipeline step. Each step is idempotent and skips if its output already exists.

## Pipeline Steps

The notebook runs the following steps in order:

| Step | Name | Description | Output |
|------|------|-------------|--------|
| 0 | **Setup** | Mount Drive, clone repo, create symlinks, install deps | Repo + environment ready |
| 1 | **Download JSONL** | Download `vad_results.jsonl` and `lang_id_results.jsonl` from HuggingFace | `data/*.jsonl` |
| 2 | **Build VAD shards** | Parse VAD results into per-tar `.pkl` shards | `data/vad_shards/*.pkl` |
| 3 | **Build VAD density index** | Compute speech density per tar | `data/vad_density_index.pkl` |
| 4 | **Build LID index** | Parse language ID results into an index with train/test splits | `data/lid_index.pkl` |
| 5 | **Build chunk index** | Select ~Nh of 10s chunks prioritizing scarce languages and high speech density | `data/chunk_index_{N}h.pkl` |
| 6 | **Assign labels** | Download tars, extract MFCCs, fit incremental k-means, assign cluster labels | `data/pretraining_index_{N}h.pkl` |
| 7 | **HuBERT pre-training** | Masked pre-training with cross-entropy loss on masked frames | `checkpoints/aligned/` |

Steps 5-6 are the core **curation strategy**: tars are scored by VAD density and language scarcity to build a balanced, high-quality subset without downloading the entire dataset.

## Resumability

Both **assign_labels** (Step 6) and **HuBERT pre-training** (Step 7) are fully resumable:

- **assign_labels** checkpoints every N tars. If the Colab session crashes, re-running the cell picks up from the last checkpoint.
- **HuBERT pre-training** saves `training_state.pt` every 500 optimizer steps. Uncomment the `--resume` flag in the training cell to resume from the last checkpoint.

## Batch Size by GPU

The notebook auto-detects VRAM and sets batch size accordingly:

| GPU | VRAM | batch_size | grad_accum | Effective batch |
|-----|------|-----------|------------|----------------|
| T4 | 16 GB | 8 | 4 | 32 |
| V100 | 16 GB | 12-16 | 2-4 | 32-64 |
| A100 | 40 GB | 32-48 | 1-2 | 32-96 |
| A100 | 80 GB | 64 | 1 | 64 |

## Project Structure

```
ups_challenge/
├── dataloaders/
│   └── masked_pretraining.py       # WebDataset dataloader (single stream + shuffle buffer)
├── inference/
│   ├── assign_labels.py            # Incremental k-means + label assignment (resumable)
│   └── hubert_pretraining.py       # HuBERT masked pre-training loop (resumable)
└── preprocessing/
    ├── build_chunk_index.py        # Curated chunk selection with VAD + language scoring
    ├── build_lang_index.py         # Builds LID index from lang_id_results.jsonl
    ├── build_vad_density_index.py  # Per-tar VAD density computation
    ├── lang_speech_hours.json      # Hours per language in the dataset
    └── vad_lookup.py               # Parses vad_results.jsonl into per-tar sharded .pkl

colab_hubert_pretraining.ipynb      # Main orchestrator notebook
```

## Data Storage

All data and checkpoints are stored on Google Drive via symlinks so they persist across Colab sessions:

```
Google Drive (MyDrive/ups-challenge/)
├── data/
│   ├── vad_results.jsonl          # ~GB, downloaded once
│   ├── lang_id_results.jsonl
│   ├── vad_shards/                # Per-tar VAD .pkl files
│   ├── tar_cache/                 # Downloaded tar files (reused across steps)
│   ├── vad_density_index.pkl
│   ├── lid_index.pkl
│   ├── chunk_index_{N}h.pkl
│   └── pretraining_index_{N}h.pkl
└── checkpoints/
    └── aligned/                   # Model checkpoints + loss curve
```

## Configurable Parameters

- **`TOTAL_HOURS`** (Step 5): How many hours of audio to curate (default: 100). Increase for more training data at the cost of more download time and disk space.
- **`--n_clusters`** (Step 6): Number of k-means clusters for pseudo-labels (default: 100).
- **`--num_epochs`** (Step 7): Training epochs (default: 4).
- **`--learning_rate`** (Step 7): Learning rate (default: 5e-5).
- **`--save_every_steps`** (Step 7): Checkpoint frequency (default: 500).

## Cleaning Up Space

The notebook includes a section (Step 8) to monitor Drive usage by component. Optionally, you can delete `vad_results.jsonl` after generating the VAD shards to save space.
