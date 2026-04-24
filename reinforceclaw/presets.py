"""Per-profile training presets based on the Gemma-4-31B Modal sweep."""

from __future__ import annotations

from .profile import ModelProfile

ANCHOR_BALANCED = 5e-6  # Gemma-4-31B dense balanced (our sweep)

_SIZE_MULT = {
    "tiny":  1.6,   # <= 2B active, flatter logits
    "small": 1.4,   # 2-8B
    "mid":   1.2,   # 8-20B
    "large": 1.0,   # 20-45B   <-- anchor
    "xl":    0.8,   # 45-100B dense, peakier logits
}


def _scale_cut(total_b: float) -> float:
    """Softmax sharpening + router/depth variance at very large total sizes."""
    if total_b <= 100:
        return 1.0
    if total_b <= 250:   # Qwen3-235B region
        return 0.60
    if total_b <= 500:   # Maverick-400, Mistral-Large-3-675 (dense) sits just above
        return 0.45
    return 0.35          # DeepSeek-671, Kimi-1T, anything larger


def _moe_mult(kind: str, total_b: float, active_b: float) -> float:
    """Sparsity-driven cut on attention-LoRA LR."""
    if kind != "moe":
        return 1.0
    sparsity = total_b / max(active_b, 1.0)
    if sparsity <= 4:
        return 0.95
    if sparsity <= 10:
        return 0.85
    if sparsity <= 20:
        return 0.70
    return 0.55

_PRESET_LR_MULT = {"careful": 0.6, "balanced": 1.0, "aggressive": 1.6}
_PRESET_KL = {"careful": 0.0020, "balanced": 0.0010, "aggressive": 0.0005}
_PRESET_STEPS = {"careful": 24, "balanced": 32, "aggressive": 48}
_PRESET_TRAJ_CLIP = {
    "careful":    [0.996, 1.001],
    "balanced":   [0.994, 1.002],
    "aggressive": [0.992, 1.004],
}

BATCH_BY_BUCKET = {"tiny": 30, "small": 30, "mid": 32, "large": 36, "xl": 40}
BATCH_SIZE_BY_BUCKET = {"tiny": 8, "small": 6, "mid": 4, "large": 3, "xl": 2}
GRAD_ACCUM_BY_BUCKET = {"tiny": 1, "small": 1, "mid": 2, "large": 2, "xl": 4}
RANK_BY_BUCKET = {"tiny": 16, "small": 16, "mid": 16, "large": 16, "xl": 8}


def pick(profile: ModelProfile, preset: str = "balanced") -> dict:
    """Return a full training config for this (profile, preset)."""
    preset = preset if preset in _PRESET_LR_MULT else "balanced"
    bucket = profile.size_bucket if profile.size_bucket in _SIZE_MULT else "mid"
    lr = (ANCHOR_BALANCED
          * _SIZE_MULT[bucket]
          * _scale_cut(profile.total_b)
          * _moe_mult(profile.kind, profile.total_b, profile.active_b)
          * _PRESET_LR_MULT[preset])
    kl = _PRESET_KL[preset] * (0.5 if profile.kind == "moe" else 1.0)
    rank = RANK_BY_BUCKET[bucket] + (8 if profile.kind == "moe" else 0)
    rank = min(rank, 32)

    return {
        "lr": _round(lr),
        "kl_coeff": _round(kl, 6),
        "lora_rank": rank,
        "lora_alpha": rank,
        "lora_target": "attention",
        "batch_min": BATCH_BY_BUCKET[bucket],
        "batch_size": BATCH_SIZE_BY_BUCKET[bucket],
        "grad_accum": GRAD_ACCUM_BY_BUCKET[bucket],
        "steps": _PRESET_STEPS[preset],
        "traj_clip": list(_PRESET_TRAJ_CLIP[preset]),
        "token_clip": [0.5, 2.0],
        "pos_weight": 1.2 if preset != "careful" else 1.0,
        "replay_ratio": 0.0,
        "ema_decay": 0.99,
        "model_profile": profile.as_dict(),
        "tuning_mode": "auto",
    }


def _round(x: float, sig: int = 7) -> float:
    if x <= 0:
        return 0.0
    return float(f"{x:.{sig}g}")
