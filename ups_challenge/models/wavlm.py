import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import WavLMModel, WavLMConfig

from ups_challenge.models.utils import compute_span_mask


# ---------------------------------------------------------------------------
# Intra-batch noise mixing
# ---------------------------------------------------------------------------

def mix_utterances_intra_batch(
    waveforms: list,
    attention_mask,
    noise_prob: float,
    snr_min: float,
    snr_max: float,
) -> tuple:
    """Mix random utterances as noise into each waveform at a random SNR.

    Operates on raw waveforms (numpy float32 arrays or torch tensors) *before*
    feature-extractor normalization.

    Args:
        waveforms:      List of B numpy arrays (or 1-D tensors) of shape (T_i,).
        attention_mask: Optional [B, T] tensor (1=valid, 0=pad). If None, full
                        length is treated as valid for all utterances.
        noise_prob:     Probability each utterance gets a noise overlay.
        snr_min:        Minimum SNR (dB) for the noise mix.
        snr_max:        Maximum SNR (dB) for the noise mix.

    Returns:
        mixed_waveforms: Same structure as input, possibly modified in place.
        is_noised:       List[bool] of length B.
    """
    B = len(waveforms)
    is_noised = [False] * B

    # Convert to numpy if needed
    wavs = []
    for w in waveforms:
        if isinstance(w, torch.Tensor):
            wavs.append(w.numpy().astype(np.float32))
        else:
            wavs.append(np.asarray(w, dtype=np.float32))

    # Valid lengths from attention_mask or full length
    if attention_mask is not None:
        valid_lens = attention_mask.sum(dim=1).long().tolist()
    else:
        valid_lens = [len(w) for w in wavs]

    for i in range(B):
        if np.random.rand() >= noise_prob:
            continue

        # Pick a random noise source ≠ i
        j = np.random.randint(0, B - 1)
        if j >= i:
            j += 1

        len_i = int(valid_lens[i])
        len_j = int(valid_lens[j])

        # WavLM strategy:
        # 1. Noise length: random from [0, len_i // 2]
        #    (We enforce a min length of 10ms to avoid empty slices)
        min_noise_len = int(16000 * 0.01)  # 10ms at 16k
        max_noise_len = max(min_noise_len, len_i // 2)
        
        noise_len = np.random.randint(min_noise_len, max_noise_len + 1)
        
        # 2. Target insertion point: random from [0, len_i - noise_len]
        start_i = np.random.randint(0, len_i - noise_len + 1)
        end_i = start_i + noise_len

        target_segment = wavs[i][start_i:end_i]
        rms_target = np.sqrt(np.mean(target_segment ** 2) + 1e-8)

        # 3. Source crop: random from [0, len_j - noise_len]
        #    If source is too short, we loop/pad it.
        if len_j >= noise_len:
            start_j = np.random.randint(0, len_j - noise_len + 1)
            noise_segment = wavs[j][start_j : start_j + noise_len]
        else:
            # Source too short, repeat it to fill noise_len
            noise_segment = np.resize(wavs[j], (noise_len,))
        
        rms_noise = np.sqrt(np.mean(noise_segment ** 2) + 1e-8)

        # 4. Mix
        snr_db = np.random.uniform(snr_min, snr_max)
        scale = rms_target / (rms_noise * 10 ** (snr_db / 20.0))
        
        wavs[i][start_i:end_i] = target_segment + scale * noise_segment
        is_noised[i] = True

    return wavs, is_noised


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class WavLMForPreTraining(nn.Module):
    """WavLM encoder + linear projection for masked-frame classification."""

    def __init__(
        self,
        model_name="microsoft/wavlm-large",
        num_clusters=100,
        mask_time_prob=0.08,
        mask_time_length=10,
    ):
        super().__init__()
        config = WavLMConfig.from_pretrained(model_name)
        config.apply_spec_augment = False  # avoid double masking
        self.wavlm = WavLMModel.from_pretrained(model_name, config=config)
        self.projection = nn.Linear(config.hidden_size, num_clusters)
        self.mask_time_prob = mask_time_prob
        self.mask_time_length = mask_time_length

    def forward(self, input_values, labels, attention_mask=None):
        """
        Args:
            input_values:  [B, T]  processed waveforms (after feature extractor)
            labels:        [B, T'] k-means cluster IDs (-100 = padding)
            attention_mask:[B, T]  wav-level mask (1=real, 0=pad), optional
        Returns:
            loss   – scalar cross-entropy on masked positions
            logits – [B, T', K]
        """
        batch_size = input_values.shape[0]

        seq_len = labels.shape[1]

        # Build encoder-level attention mask from labels (valid = not padded)
        encoder_attention_mask = (labels != -100).long()

        # Span-based mask indices
        mask_time_indices = compute_span_mask(
            shape=(batch_size, seq_len),
            mask_prob=self.mask_time_prob,
            mask_length=self.mask_time_length,
            attention_mask=encoder_attention_mask,
            device=input_values.device,
        )

        outputs = self.wavlm(
            input_values,
            attention_mask=attention_mask,
            mask_time_indices=mask_time_indices,
        )
        hidden_states = outputs.last_hidden_state  # [B, enc_len, D]

        # Clip to common length (CNN stride rounding may differ by ±2 frames)
        enc_len = hidden_states.shape[1]
        common_len = min(enc_len, labels.shape[1])
        hidden_states = hidden_states[:, :common_len]
        mask_time_indices = mask_time_indices[:, :common_len]
        labels = labels[:, :common_len]

        logits = self.projection(hidden_states)  # [B, common_len, K]

        # Loss only on masked positions
        masked_logits = logits[mask_time_indices]   # [N_masked, K]
        masked_labels = labels[mask_time_indices]   # [N_masked]
        loss = F.cross_entropy(masked_logits, masked_labels, ignore_index=-100)

        return loss, logits
