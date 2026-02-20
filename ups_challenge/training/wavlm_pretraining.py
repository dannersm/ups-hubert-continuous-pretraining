"""WavLM masked speech pre-training with intra-batch utterance noise mixing.

Uses a pre-built index with k-means labels (from assign_labels.py).
Masks ~57% of frames (span masking, p=0.08 starts × l=10 length) and trains a
cross-entropy prediction head on the masked positions.  Before feature
extraction, utterances can be mixed intra-batch at a random SNR so the model
learns to predict clean pseudo-labels even on noised inputs.

Usage:
    python -m ups_challenge.training.wavlm_pretraining \\
        --index_path ./data/pretraining_index_100h.pkl \\
        --num_clusters 100 --batch_size 8 --max_steps 1000
"""

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoFeatureExtractor
from tqdm import tqdm
from dotenv import load_dotenv

from ups_challenge.dataloaders.masked_pretraining import (
    build_pretraining_dataset,
    collate_fn,
)
from ups_challenge.models.wavlm import WavLMForPreTraining, mix_utterances_intra_batch
from ups_challenge.training.training_utils import (
    _save_loss_plot,
    _save_training_state,
    _setup_projection_phase,
    _setup_cpt_phase,
)

load_dotenv()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_wavlm(
    model_name="microsoft/wavlm-large",
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
    noise_prob=0.1,
    snr_min=-5.0,
    snr_max=5.0,
):
    # Device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Model
    print(f"Loading model: {model_name}")
    model = WavLMForPreTraining(
        model_name,
        num_clusters=num_clusters,
        mask_time_prob=mask_time_prob,
        mask_time_length=mask_time_length,
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model.to(device)
    model.train()
    model.wavlm.requires_grad_(False)
    model.projection.requires_grad_(True)

    total_epochs = projection_warmup_epochs + num_epochs
    epoch = 0
    start_epoch = 0
    global_step = 0
    loss_history = []

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

        if start_epoch > projection_warmup_epochs:
            model.wavlm.requires_grad_(True)
            optimizer, scheduler = _setup_cpt_phase(
                lr=learning_rate, warmup_steps=warmup_steps, model=model
            )
        else:
            optimizer, scheduler = _setup_projection_phase(
                lr=projection_lr, model=model
            )

        model.load_state_dict(ckpt["model"])
        loss_history = ckpt.get("loss_history", [])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        del ckpt
        print(f"  Resumed at epoch {start_epoch+1}, global_step {global_step}")
    elif projection_warmup_epochs > 0:
        optimizer, scheduler = _setup_projection_phase(lr=projection_lr, model=model)
    else:
        model.wavlm.requires_grad_(True)
        optimizer, scheduler = _setup_cpt_phase(lr=learning_rate, warmup_steps=warmup_steps, model=model)

    # Train
    print(f"Starting training (epochs {start_epoch+1}-{total_epochs})...")

    for epoch in range(start_epoch, total_epochs):
        if epoch == projection_warmup_epochs and projection_warmup_epochs > 0:
            print(f"\n--- Phase 2: CPT (unfreezing all layers), "
                  f"lr={learning_rate:.2e}, warmup={warmup_steps} steps ---")
            model.wavlm.requires_grad_(True)
            optimizer, scheduler = _setup_cpt_phase(
                lr=learning_rate, warmup_steps=warmup_steps, model=model
            )
            global_step = 0  # reset step counter for CPT warmup
            model.zero_grad()

        data_loader = torch.utils.data.DataLoader(
            build_pretraining_dataset(index_path=index_path, cache_dir=cache_dir),
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
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

            # Collect raw waveforms from batch
            B = batch["waveform"].shape[0]
            waveforms = [batch["waveform"][i].numpy() for i in range(B)]

            # Apply intra-batch noise mixing on raw waveforms (before normalization)
            if noise_prob > 0:
                waveforms, _ = mix_utterances_intra_batch(
                    waveforms,
                    batch.get("attention_mask"),
                    noise_prob,
                    snr_min,
                    snr_max,
                )

            # Feature-extract (normalize) raw waveforms → input_values
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
            del input_values, labels, wav_attention_mask, waveforms, inputs
            loss = loss / grad_accum_steps

            # Backward (accumulate gradients)
            loss.backward()
            accum_loss += loss.item()
            del loss
            micro_step += 1

            if micro_step % grad_accum_steps != 0:
                continue

            # Gradient clipping + optimizer step
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

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

            if max_steps is not None and global_step >= max_steps:
                break

        avg = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch + 1} done. avg_loss={avg:.4f}, "
              f"batches={num_batches}, global_step={global_step}")

        # Save checkpoint at end of epoch
        ckpt_path = os.path.join(output_dir, f"wavlm_epoch_{epoch + 1}.pt")
        print(f"Saving epoch checkpoint to {ckpt_path}")
        torch.save(model.state_dict(), ckpt_path)
        _save_training_state(
            epoch=epoch + 1,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=global_step,
            out=output_dir / "training_state.pt",
            loss_history=loss_history,
        )
        _save_loss_plot(loss_history, output_dir)

        if max_steps is not None and global_step >= max_steps:
            break

    print("Training completed!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WavLM masked pre-training")
    parser.add_argument("--model_name", type=str, default="microsoft/wavlm-large")
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
    # WavLM-specific noise mixing args
    parser.add_argument("--noise_prob", type=float, default=0.1,
                        help="Probability each utterance gets intra-batch noise overlay")
    parser.add_argument("--snr_min", type=float, default=-5.0,
                        help="Minimum SNR (dB) for noise mixing")
    parser.add_argument("--snr_max", type=float, default=5.0,
                        help="Maximum SNR (dB) for noise mixing")
    args = parser.parse_args()

    hf_token = args.hf_token or os.getenv("HF_TOKEN")

    train_wavlm(
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
        noise_prob=args.noise_prob,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        max_steps=args.max_steps,
    )
