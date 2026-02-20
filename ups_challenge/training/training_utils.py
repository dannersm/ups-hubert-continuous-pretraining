"""Shared training helpers for HuBERT and WavLM pre-training loops."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def _save_loss_plot(loss_history, output_dir):
    """Save a loss-over-steps curve to output_dir/loss_curve.png."""
    if not loss_history:
        return
    steps, losses = zip(*loss_history)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, losses, linewidth=0.8, alpha=0.7, label="loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Pre-training loss")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(output_dir, "loss_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Loss curve saved to {path}")


def _save_training_state(model, optimizer, scheduler, epoch, global_step,
                         loss_history, out=None):
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


def _setup_projection_phase(lr, model):
    """Phase 1: freeze backbone, train only projection with flat LR."""
    opt = torch.optim.AdamW(model.projection.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1.0)
    return opt, sched


def _setup_cpt_phase(lr, warmup_steps, model):
    """Phase 2: unfreeze everything, train with linear warmup."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda step: min(1.0, float(step) / float(max(1, warmup_steps))),
    )
    return opt, sched


def _setup_lora_cpt_phase(lr, warmup_steps, model):
    """Phase 2 (LoRA): train only LoRA adapters + projection with linear warmup."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda step: min(1.0, float(step) / float(max(1, warmup_steps))),
    )
    return opt, sched
