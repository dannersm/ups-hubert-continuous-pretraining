import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import HubertModel, HubertConfig

from ups_challenge.models.utils import compute_span_mask


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
        use_lora=False,
        lora_rank=16,
        lora_alpha=32,
        lora_target_modules=None,
        use_rslora=False,
    ):
        super().__init__()
        config = HubertConfig.from_pretrained(model_name)
        config.apply_spec_augment = False  # avoid double masking
        self.hubert = HubertModel.from_pretrained(model_name, config=config)

        self.use_lora = use_lora
        if use_lora:
            from peft import LoraConfig, get_peft_model
            if lora_target_modules is None:
                lora_target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
            peft_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=0.05,
                bias="none",
                use_rslora=use_rslora,
            )
            self.hubert = get_peft_model(self.hubert, peft_config)
            self.hubert.print_trainable_parameters()

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

