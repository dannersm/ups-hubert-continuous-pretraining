"""HuBERT masked speech pre-training.

Uses a pre-built index with k-means labels (from prepare_pretraining_index.py).
Masks ~57 % of frames (span masking, p=0.08 starts × l=10 length) and trains a cross-entropy prediction head
on the masked positions.

Usage:
    python -m ups_challenge.examples.hubert_pretraining \
        --index_path ./data/pretraining_index_100h.pkl \
        --num_clusters 100 --batch_size 8 --max_steps 1000
"""

import argparse
import os

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, HubertConfig, HubertModel
from tqdm import tqdm

from dotenv import load_dotenv

from ups_challenge.dataloaders.masked_pretraining import (
    build_pretraining_dataset,
    collate_pretraining,
)

load_dotenv()


def compute_span_mask(shape, mask_prob, mask_length, attention_mask=None, device="cpu"):
    """
    Compute mask indices using strictly PyTorch operations.
    Strategy: Sample starting points with probability 'mask_prob', then expand to spans.
    
    Args:
        shape: (batch_size, seq_len)
        mask_prob: Probability of a token being the *start* of a span (e.g. 0.08).
        mask_length: Length of the span to mask (e.g. 10).
        attention_mask: (batch_size, seq_len) 1=valid, 0=padding.
        device: Torch device.
    """
    valid_mask = attention_mask if attention_mask is not None else torch.ones(shape)
    probs = torch.full(shape, mask_prob, device=device) * valid_mask,float()
    mask_starts = torch.bernoulli(probs) # [B, T] (float for conv)
    
    if mask_starts.sum() == 0:
        return torch.zeros(shape, dtype=torch.bool, device=device)

    # Reshape for conv: [B, 1, T]
    mask_starts_unsqueezed = mask_starts.unsqueeze(1)
    kernel = torch.ones(1, 1, mask_length, device=device)
    
    
    padded_starts = F.pad(mask_starts_unsqueezed, (mask_length - 1, 0)) # Pad left
    
    mask_expanded = F.conv1d(padded_starts, kernel) # [B, 1, T]
    
    mask = mask_expanded.squeeze(1) > 0
    
    if attention_mask is not None:
        mask = mask & attention_mask.bool()
        
    return mask


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class HubertForPreTraining(nn.Module):
    """HuBERT encoder + linear projection for masked-frame classification."""

    def __init__(
        self,
        model_name="facebook/hubert-base-ls960",
        num_clusters=100,
        mask_time_prob=0.08, # Paper standard: 8% of steps are starts
        mask_time_length=10,
    ):
        super().__init__()
        config = HubertConfig.from_pretrained(model_name)
        config.apply_spec_augment = False  # avoid double masking
        self.hubert = HubertModel.from_pretrained(model_name, config=config)
        self.projection = nn.Linear(config.hidden_size, num_clusters)
        self.mask_time_prob = mask_time_prob
        self.mask_time_length = mask_time_length

    def forward(self, input_values, labels, attention_mask=None):
        """
        Args:
            input_values:  [B, T]  processed waveforms
            labels:        [B, T'] k-means cluster IDs (-100 = padding)
            attention_mask:[B, T]  wav-level mask (1=real, 0=pad), optional
        Returns:
            loss   – scalar cross-entropy on masked positions
            logits – [B, T', K]
        """
        batch_size = input_values.shape[0]

        # Determine encoded sequence length directly from labels
        # We assume labels (from MFCC) are aligned with CNN output (stride 320)
        # This avoids running the CNN twice (once for shape, once for forward)
        seq_len = labels.shape[1]

        # Build encoder-level attention mask directly from labels
        # Since we padded labels with -100, valid tokens are != -100
        encoder_attention_mask = (labels != -100).long()
        
        # Compute span-based mask indices using transparent PyTorch logic
        mask_time_indices = compute_span_mask(
            shape=(batch_size, seq_len),
            mask_prob=self.mask_time_prob,
            mask_length=self.mask_time_length,
            attention_mask=encoder_attention_mask,
            device=input_values.device,
        )

        # Forward through HuBERT (masking is applied inside the model)
        outputs = self.hubert(
            input_values,
            attention_mask=attention_mask,
            mask_time_indices=mask_time_indices,
        )
        hidden_states = outputs.last_hidden_state  # [B, enc_len, D]

        # The CNN encoder may produce a slightly different length than labels
        # (e.g. ±2 frames due to kernel/stride math). Clip everything to the
        # shorter of the two so both directions are safe.
        enc_len = hidden_states.shape[1]
        common_len = min(enc_len, labels.shape[1])
        hidden_states = hidden_states[:, :common_len]
        mask_time_indices = mask_time_indices[:, :common_len]
        labels = labels[:, :common_len]

        # Project to cluster logits
        logits = self.projection(hidden_states)  # [B, common_len, K]

        # Loss only on masked positions
        masked_logits = logits[mask_time_indices]   # [N_masked, K]
        masked_labels = labels[mask_time_indices]   # [N_masked]
        loss = F.cross_entropy(masked_logits, masked_labels, ignore_index=-100)

        return loss, logits


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _save_loss_plot(loss_history, output_dir):
    """Save a loss-over-steps curve to output_dir/loss_curve.png."""
    if not loss_history:
        return
    steps, losses = zip(*loss_history)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, losses, linewidth=0.8, alpha=0.7, label="loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("HuBERT pre-training loss")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(output_dir, "loss_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Loss curve saved to {path}")


def _save_training_state(model: HubertForPreTraining,
                         optimizer: torch.optim.Optimizer,
                         scheduler: torch.optim.lr_scheduler.LRScheduler,
                         epoch: int,
                         global_step: int,
                         loss_history: list[dict[str, float]],
                         out: str | Path | None = None,
                         ):
    """Save full training state for resumability (epoch-level)."""
    if out is None:
        out = "training_state.pt"
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "loss_history": loss_history,
        "global_step": global_step,
    }
    torch.save(state, out)


def _setup_projection_phase(lr: float):
    """Phase 1: freeze hubert, train only projection with flat LR."""
    opt = torch.optim.AdamW(model.projection.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1.0)
    return opt, sched

def _setup_cpt_phase(lr: float):
    """Phase 2: unfreeze everything, train with warmup."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda step: min(1.0, float(step) / float(max(1, warmup_steps))),
    )
    return opt, sched

def train_hubert(
    model_name="facebook/hubert-base-ls960",
    index_path="./data/pretraining_index_100h.pkl",
    num_clusters=100,
    mask_time_prob=0.08,
    mask_time_length=10,
    batch_size=16,
    num_workers=8,
    num_epochs=1,
    learning_rate=2e-5,
    device=None,
    hf_token=None,
    max_steps=None,
    output_dir="./checkpoints",
    cache_dir="./data/tar_cache",
    grad_accum_steps=4,
    warmup_steps=1500,
    max_grad_norm=1.0,
    resume=False,
    projection_warmup_epochs=0,
    projection_lr=None,
):
    # Device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Model
    print(f"Loading model: {model_name}")
    model = HubertForPreTraining(
        model_name,
        num_clusters=num_clusters,
        mask_time_prob=mask_time_prob,
        mask_time_length=mask_time_length,
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model.to(device)
    model.train()
    model.hubert.requires_grad_(False)
    model.projection.requires_grad_(True)

    total_epochs = projection_warmup_epochs + num_epochs
    epoch = 0
    start_epoch = 0
    global_step = 0
    
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if projection_lr is None:
        projection_lr = learning_rate

    resume_ckpt_path = os.path.join(output_dir, "training_state.pt")
    if resume and os.path.exists(resume_ckpt_path):
        print(f"Resuming from {resume_ckpt_path}...")
        ckpt = torch.load(resume_ckpt_path, map_location=device, weights_only=False)
        epoch = ckpt["epoch"]
        global_step = ckpt["global_step"]
        start_epoch = epoch

        if start_epoch >= projection_warmup_epochs:
            model.hubert.requires_grad_(True)
            optimizer, scheduler = _setup_cpt_phase(lr=learning_rate)
        else:
            optimizer, scheduler = _setup_projection_phase(lr=projection_lr)

        model.load_state_dict(ckpt["model"])
        loss_history = ckpt.get("loss_history", [])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        print(f"  Resumed at epoch {start_epoch+1}, global_step {global_step}")

    # Train
    print(f"Starting training (epochs {start_epoch+1}-{total_epochs})...")

    for epoch in range(start_epoch, total_epochs):
        if epoch == projection_warmup_epochs and projection_warmup_epochs > 0 and start_epoch < projection_warmup_epochs:
            print(f"\n--- Phase 2: CPT (unfreezing all layers), "
                  f"lr={learning_rate:.2e}, warmup={warmup_steps} steps ---")
            model.hubert.requires_grad_(True)
            optimizer, scheduler = _setup_cpt_phase(learning_rate)
            global_step = 0  # reset step counter for CPT warmup
            model.zero_grad()

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_pretraining,
        )

        epoch_loss = 0.0
        num_batches = 0
        micro_step = 0
        accum_loss = 0.0

        phase_str = "proj" if epoch < projection_warmup_epochs else "CPT"
        pbar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{total_epochs} [{phase_str}]")
        for batch_idx, batch in enumerate(pbar):
            if batch is None:
                continue

            # Feature-extract raw waveforms -> normalized input_values
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
            wav_attention_mask = inputs.get("attention_mask")
            if wav_attention_mask is not None:
                wav_attention_mask = wav_attention_mask.to(device)

            labels = batch["labels"].to(device)

            # Forward
            loss, _ = model(input_values, labels, wav_attention_mask)
            loss = loss / grad_accum_steps

            # Backward (accumulate gradients)
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

            loss_val = accum_loss
            accum_loss = 0.0
            epoch_loss += loss_val
            num_batches += 1
            global_step += 1
            loss_history.append((global_step, loss_val))
            pbar.set_postfix({
                "loss": f"{loss_val:.4f}",
                "avg": f"{epoch_loss / max(num_batches, 1):.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

        avg = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch + 1} done. avg_loss={avg:.4f}, "
              f"batches={num_batches}, global_step={global_step}")

        # Save checkpoint at end of epoch
        ckpt_path = os.path.join(output_dir, f"hubert_epoch_{epoch + 1}.pt")
        print(f"Saving epoch checkpoint to {ckpt_path}")
        torch.save(model.state_dict(), ckpt_path)
        _save_training_state(epoch=epoch+1, model=model, optimizer=optimizer, scheduler=scheduler, global_step=global_step, out=output_dir/"training_state.pt", loss_history=loss_history)
        _save_loss_plot(loss_history, output_dir)

    print("Training completed!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HuBERT masked pre-training")
    parser.add_argument("--model_name", type=str, default="facebook/hubert-base-ls960")
    parser.add_argument("--index_path", type=str,
                        default="./data/pretraining_index_100h.pkl")
    parser.add_argument("--num_clusters", type=int, default=100)
    parser.add_argument("--mask_time_prob", type=float, default=0.08)
    parser.add_argument("--mask_time_length", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default="./data/tar_cache",
                        help="Directory to cache downloaded tars (reused across runs)")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from training_state.pt in output_dir")
    parser.add_argument("--projection_warmup_epochs", type=int, default=0,
                        help="Epochs to train only projection head before CPT (default: 0)")
    parser.add_argument("--projection_lr", type=float, default=None,
                        help="LR for projection warmup phase (default: same as learning_rate)")
    args = parser.parse_args()

    hf_token = args.hf_token or os.getenv("HF_TOKEN")

    train_hubert(
        model_name=args.model_name,
        index_path=args.index_path,
        num_clusters=args.num_clusters,
        mask_time_prob=args.mask_time_prob,
        mask_time_length=args.mask_time_length,
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
        resume=args.resume,
        projection_warmup_epochs=args.projection_warmup_epochs,
        projection_lr=args.projection_lr,
    )
