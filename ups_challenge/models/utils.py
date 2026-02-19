import torch
import torch.nn.functional as F


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
    valid_mask = attention_mask if attention_mask is not None else torch.ones(shape, device=device)
    probs = torch.full(shape, mask_prob, device=device) * valid_mask.float()
    mask_starts = torch.bernoulli(probs)  # [B, T] (float for conv)

    if mask_starts.sum() == 0:
        return torch.zeros(shape, dtype=torch.bool, device=device)

    # Reshape for conv: [B, 1, T]
    mask_starts_unsqueezed = mask_starts.unsqueeze(1)
    kernel = torch.ones(1, 1, mask_length, device=device)

    padded_starts = F.pad(mask_starts_unsqueezed, (mask_length - 1, 0))  # Pad left

    mask_expanded = F.conv1d(padded_starts, kernel)  # [B, 1, T]

    mask = mask_expanded.squeeze(1) > 0

    if attention_mask is not None:
        mask = mask & attention_mask.bool()

    return mask
