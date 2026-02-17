import os
import urllib.request

import torch
from transformers import Wav2Vec2Config, Wav2Vec2ForPreTraining

_DEFAULT_URL = "https://dl.fbaipublicfiles.com/mms/omniASR-W2V-300M.pt"


def download_omniasr_weights(
    weights_path=None, cache_dir=None, model_size="300M",
):
    """Download OmniASR checkpoint, caching locally. Returns path to .pt file."""
    if weights_path and os.path.exists(weights_path):
        return weights_path

    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "omniasr")
    os.makedirs(cache_dir, exist_ok=True)

    filename = f"omniASR-W2V-{model_size}.pt"
    dest = os.path.join(cache_dir, filename)
    if os.path.exists(dest):
        return dest

    print(f"Downloading OmniASR-{model_size} weights to {dest}...")
    urllib.request.urlretrieve(_DEFAULT_URL, dest)
    print("Download complete.")
    return dest


# ---------------------------------------------------------------------------
# Encoder-only config (for Wav2Vec2Model inference)
# ---------------------------------------------------------------------------
_CONFIG = Wav2Vec2Config(
    hidden_size=1024,
    num_hidden_layers=24,
    num_attention_heads=16,
    intermediate_size=4096,
    conv_dim=[512] * 7,
    conv_kernel=[10, 3, 3, 3, 3, 2, 2],
    conv_stride=[5, 2, 2, 2, 2, 2, 2],
    conv_bias=True,
    feat_extract_norm="layer",
    do_stable_layer_norm=True,
    num_conv_pos_embeddings=128,
    num_conv_pos_embedding_groups=16,
)

# ---------------------------------------------------------------------------
# Pre-training config (encoder + quantizer + contrastive head)
# ---------------------------------------------------------------------------
_PRETRAINING_CONFIG = Wav2Vec2Config(
    # Encoder (same as _CONFIG)
    hidden_size=1024,
    num_hidden_layers=24,
    num_attention_heads=16,
    intermediate_size=4096,
    conv_dim=[512] * 7,
    conv_kernel=[10, 3, 3, 3, 3, 2, 2],
    conv_stride=[5, 2, 2, 2, 2, 2, 2],
    conv_bias=True,
    feat_extract_norm="layer",
    do_stable_layer_norm=True,
    num_conv_pos_embeddings=128,
    num_conv_pos_embedding_groups=16,
    # Quantizer (derived from checkpoint shapes)
    codevector_dim=768,
    num_codevector_groups=2,
    num_codevectors_per_group=320,
    proj_codevector_dim=768,
    # Pre-training (wav2vec 2.0 paper defaults)
    mask_time_prob=0.065,
    mask_time_length=10,
    num_negatives=100,
    diversity_loss_weight=0.1,
    contrastive_logits_temperature=0.1,
)


# ---------------------------------------------------------------------------
# Key conversion: fairseq2 → HuggingFace
# ---------------------------------------------------------------------------

def _convert_key(k: str) -> str | None:
    """Map a fairseq2 checkpoint key to its HuggingFace Wav2Vec2Model equivalent.

    Returns encoder-level keys (no 'wav2vec2.' prefix). Used for inference
    with Wav2Vec2Model.
    """
    # Feature extractor conv layers
    if k.startswith("encoder_frontend.feature_extractor.layers."):
        return k.replace(
            "encoder_frontend.feature_extractor.layers.",
            "feature_extractor.conv_layers.",
        )

    # Feature projection
    _proj = {
        "encoder_frontend.post_extract_layer_norm.": "feature_projection.layer_norm.",
        "encoder_frontend.model_dim_proj.": "feature_projection.projection.",
    }
    for src, dst in _proj.items():
        if k.startswith(src):
            return k.replace(src, dst)

    # Positional conv embedding (weight_norm keys)
    if k.startswith("encoder_frontend.pos_encoder.conv."):
        suffix = k.split("encoder_frontend.pos_encoder.conv.")[1]
        remap = {
            "weight_g": "parametrizations.weight.original0",
            "weight_v": "parametrizations.weight.original1",
            "bias": "bias",
        }
        if suffix in remap:
            return f"encoder.pos_conv_embed.conv.{remap[suffix]}"
        return None

    # Encoder transformer layers
    if k.startswith("encoder.layers."):
        k = k.replace(".self_attn_layer_norm.", ".layer_norm.")
        k = k.replace(".self_attn.", ".attention.")
        k = k.replace(".ffn_layer_norm.", ".final_layer_norm.")
        k = k.replace(".ffn.inner_proj.", ".feed_forward.intermediate_dense.")
        k = k.replace(".ffn.output_proj.", ".feed_forward.output_dense.")
        k = k.replace(".output_proj.", ".out_proj.")
        return k

    # Encoder final layer norm (keys already match)
    if k.startswith("encoder.layer_norm."):
        return k

    # SSL-only components handled by _convert_key_pretraining
    return None


def _convert_key_pretraining(k: str) -> str | None:
    """Map a fairseq2 checkpoint key to Wav2Vec2ForPreTraining equivalent.

    Handles both encoder keys (prefixed with 'wav2vec2.') and SSL-specific
    keys (quantizer, projections, masker).
    """
    # --- SSL-specific keys ---
    # Quantizer weight projection
    if k.startswith("quantizer.entry_proj."):
        return k.replace("quantizer.entry_proj.", "quantizer.weight_proj.")

    # Quantizer codebook entries
    if k == "quantizer.entries":
        return "quantizer.codevectors"

    # Final projection (context → hidden)
    if k.startswith("final_proj."):
        return k.replace("final_proj.", "project_hid.")

    # Final target projection (quantized → projected)
    if k.startswith("final_target_proj."):
        return k.replace("final_target_proj.", "project_q.")

    # Mask embedding
    if k == "masker.temporal_mask_embed":
        return "wav2vec2.masked_spec_embed"

    # --- Encoder keys (delegate, then prefix) ---
    encoder_key = _convert_key(k)
    if encoder_key is not None:
        return f"wav2vec2.{encoder_key}"

    return None


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _load_checkpoint(path: str) -> dict[str, torch.Tensor]:
    """Load fairseq2 checkpoint and convert to Wav2Vec2Model keys."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    src = ckpt["model"]
    converted = {}
    for k, v in src.items():
        new_k = _convert_key(k)
        if new_k is not None:
            converted[new_k] = v
    return converted


def _load_checkpoint_pretraining(path: str) -> dict[str, torch.Tensor]:
    """Load fairseq2 checkpoint and convert to Wav2Vec2ForPreTraining keys."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    src = ckpt["model"]
    converted = {}
    skipped = []
    for k, v in src.items():
        new_k = _convert_key_pretraining(k)
        if new_k is not None:
            converted[new_k] = v
        else:
            skipped.append(k)
    if skipped:
        print(f"Skipped {len(skipped)} fairseq2 keys: {skipped[:10]}{'...' if len(skipped) > 10 else ''}")
    return converted


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_omniasr_for_pretraining(
    weights_path=None, cache_dir=None,
) -> Wav2Vec2ForPreTraining:
    """Download OmniASR weights, convert, load into Wav2Vec2ForPreTraining."""
    path = download_omniasr_weights(weights_path, cache_dir)
    model = Wav2Vec2ForPreTraining(_PRETRAINING_CONFIG)
    state_dict = _load_checkpoint_pretraining(path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys ({len(missing)}): {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}): {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")
    return model
