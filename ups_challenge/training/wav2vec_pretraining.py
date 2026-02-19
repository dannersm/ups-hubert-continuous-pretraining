"""Wav2Vec 2.0 continued pre-training of OmniASR-300M.

Self-supervised: uses Gumbel-softmax quantization + contrastive loss +
diversity loss from HF's Wav2Vec2ForPreTraining — no external k-means labels.

Usage:
    python -m ups_challenge.training.wav2vec_pretraining \
        --index_path ./data/chunk_index_100h.pkl \
        --batch_size 8 --num_epochs 1
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoFeatureExtractor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    _compute_mask_indices,
    _sample_negative_indices,
)
from tqdm import tqdm
from dotenv import load_dotenv

from ups_challenge.dataloaders.wav2vec_pretraining import (
    build_wav2vec_dataset,
    collate_fn,
)
from ups_challenge.models.omniasr import load_omniasr_for_pretraining

load_dotenv()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_loss_plot(loss_history, output_dir):
    """Save a loss-over-steps curve to output_dir/loss_curve.png."""
    if not loss_history:
        return
    steps, losses = zip(*loss_history)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, losses, linewidth=0.8, alpha=0.7, label="loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Contrastive + diversity loss")
    ax.set_title("Wav2Vec2 pre-training loss")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(output_dir, "loss_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Loss curve saved to {path}")


def _save_training_state(
    model, optimizer, scheduler, epoch, global_step,
    loss_history, gumbel_temp, out=None,
):
    """Save full training state for resumability."""
    if out is None:
        out = "training_state.pt"
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "loss_history": loss_history,
        "gumbel_temp": gumbel_temp,
    }
    torch.save(state, out)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_wav2vec(
    index_path="./data/chunk_index_100h.pkl",
    batch_size=16,
    num_workers=8,
    num_epochs=1,
    learning_rate=2e-5,
    device=None,
    hf_token=None,
    output_dir="./checkpoints/wav2vec",
    cache_dir="./data/tar_cache",
    grad_accum_steps=4,
    warmup_steps=1500,
    max_grad_norm=1.0,
    mask_time_prob=0.065,
    mask_time_length=10,
    gumbel_max=2.0,
    gumbel_min=0.5,
    gumbel_decay=0.999995,
    resume=False,
    weights_path=None,
    weights_cache_dir=None,
):
    # Device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Model
    print("Loading OmniASR wav2vec2 for pre-training...")
    model = load_omniasr_for_pretraining(
        weights_path=weights_path, cache_dir=weights_cache_dir,
    )
    # Override masking config if provided
    model.config.mask_time_prob = mask_time_prob
    model.config.mask_time_length = mask_time_length

    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    model.to(device)
    model.train()

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, float(step) / float(max(1, warmup_steps))),
    )

    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    start_epoch = 0
    global_step = 0
    loss_history = []
    gumbel_temp = gumbel_max

    # Resume
    resume_ckpt_path = output_dir / "training_state.pt"
    if resume and resume_ckpt_path.exists():
        print(f"Resuming from {resume_ckpt_path}...")
        ckpt = torch.load(resume_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt["global_step"]
        loss_history = ckpt.get("loss_history", [])
        gumbel_temp = ckpt.get("gumbel_temp", gumbel_max)
        print(f"  Resumed at epoch {start_epoch + 1}, global_step {global_step}, "
              f"gumbel_temp {gumbel_temp:.4f}")

    # Set initial Gumbel temperature
    model.set_gumbel_temperature(gumbel_temp)

    print(f"Starting training (epochs {start_epoch + 1}-{num_epochs})...")

    for epoch in range(start_epoch, num_epochs):
        data_loader = torch.utils.data.DataLoader(
            build_wav2vec_dataset(index_path=index_path, cache_dir=cache_dir),
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

        epoch_loss = 0.0
        num_batches = 0
        micro_step = 0
        accum_loss = 0.0

        pbar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch in pbar:
            if batch is None:
                continue

            # Feature-extract raw waveforms
            waveforms = [
                batch["waveform"][i].numpy() for i in range(batch["waveform"].shape[0])
            ]
            inputs = feature_extractor(
                waveforms,
                sampling_rate=feature_extractor.sampling_rate,
                padding=True,
                return_tensors="pt",
            )
            input_values = inputs.input_values.to(device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            # Compute mask indices explicitly (Wav2Vec2ForPreTraining needs these
            # to be passed in; otherwise loss is None)
            batch_size, raw_seq_len = input_values.shape
            seq_len = model._get_feat_extract_output_lengths(raw_seq_len).item()

            mask_time_indices = _compute_mask_indices(
                shape=(batch_size, seq_len),
                mask_prob=model.config.mask_time_prob,
                mask_length=model.config.mask_time_length,
            )
            sampled_negative_indices = _sample_negative_indices(
                features_shape=(batch_size, seq_len),
                num_negatives=model.config.num_negatives,
                mask_time_indices=mask_time_indices,
            )
            mask_time_indices = torch.tensor(mask_time_indices, device=device, dtype=torch.bool)
            sampled_negative_indices = torch.tensor(sampled_negative_indices, device=device, dtype=torch.long)

            # Forward
            outputs = model(
                input_values=input_values,
                attention_mask=attention_mask,
                mask_time_indices=mask_time_indices,
                sampled_negative_indices=sampled_negative_indices,
            )
            loss = outputs.loss / grad_accum_steps

            loss.backward()
            accum_loss += loss.item()
            micro_step += 1

            if micro_step % grad_accum_steps != 0:
                continue

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Gumbel temperature annealing
            global_step += 1
            gumbel_temp = max(gumbel_max * gumbel_decay ** global_step, gumbel_min)
            model.set_gumbel_temperature(gumbel_temp)

            loss_val = accum_loss
            accum_loss = 0.0
            epoch_loss += loss_val
            num_batches += 1
            loss_history.append((global_step, loss_val))

            pbar.set_postfix({
                "loss": f"{loss_val:.4f}",
                "avg": f"{epoch_loss / max(num_batches, 1):.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                "gumbel": f"{gumbel_temp:.3f}",
                "ppl": f"{outputs.codevector_perplexity.item():.1f}",
            })

        avg = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch + 1} done. avg_loss={avg:.4f}, "
              f"batches={num_batches}, global_step={global_step}")

        # Save checkpoint
        ckpt_path = output_dir / f"wav2vec_epoch_{epoch + 1}.pt"
        print(f"Saving epoch checkpoint to {ckpt_path}")
        torch.save(model.state_dict(), ckpt_path)
        _save_training_state(
            model=model, optimizer=optimizer, scheduler=scheduler,
            epoch=epoch + 1, global_step=global_step,
            loss_history=loss_history, gumbel_temp=gumbel_temp,
            out=output_dir / "training_state.pt",
        )
        _save_loss_plot(loss_history, output_dir)

    print("Training completed!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wav2Vec2 continued pre-training")
    parser.add_argument("--index_path", type=str,
                        default="./data/chunk_index_100h.pkl")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=1500)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mask_time_prob", type=float, default=0.065)
    parser.add_argument("--mask_time_length", type=int, default=10)
    parser.add_argument("--gumbel_max", type=float, default=2.0)
    parser.add_argument("--gumbel_min", type=float, default=0.5)
    parser.add_argument("--gumbel_decay", type=float, default=0.999995)
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default="./data/tar_cache")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/wav2vec")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from training_state.pt in output_dir")
    parser.add_argument("--weights_path", type=str, default=None,
                        help="Path to OmniASR checkpoint (downloads if not set)")
    parser.add_argument("--weights_cache_dir", type=str, default=None,
                        help="Cache dir for OmniASR weights (default: ~/.cache/omniasr)")
    args = parser.parse_args()

    hf_token = args.hf_token or os.getenv("HF_TOKEN")

    train_wav2vec(
        index_path=args.index_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        hf_token=hf_token,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        grad_accum_steps=args.grad_accum_steps,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        mask_time_prob=args.mask_time_prob,
        mask_time_length=args.mask_time_length,
        gumbel_max=args.gumbel_max,
        gumbel_min=args.gumbel_min,
        gumbel_decay=args.gumbel_decay,
        resume=args.resume,
        weights_path=args.weights_path,
        weights_cache_dir=args.weights_cache_dir,
    )
