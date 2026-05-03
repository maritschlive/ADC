"""
dynamic_config.py
=================
Unified dynamic configuration for ADC training.

Provides:
- ConfigSchema: dataclass for type-safe config
- PRESETS: predefined training configurations
- load_config_from_env(): overlay environment variables
- apply_config_to_model(): apply config to Lightning module

Usage:
    from dynamic_config import PRESETS, load_config_from_env, apply_config_to_model

    config = PRESETS["scratch"]  # base preset
    config = load_config_from_env(config)  # override with env vars
    apply_config_to_model(model, config)  # apply to model
"""

import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import json


@dataclass
class LossConfig:
    """Loss weighting and variance parameters."""
    weight_mask: float = 1.0              # Mask decoder loss scale
    weight_image: float = 0.0             # Image decoder loss scale (0=Phase1, 1.0+=Phase2)
    weight_distill: float = 0.0           # Anatomy-aware distillation scale
    simple_weight: float = 1.0            # Diffusion MSE scaling
    elbo_weight: float = 0.1              # VLB (variational lower bound) regularizer
    learn_logvar: bool = True             # Enable learnable variance schedule


@dataclass
class ControlConfig:
    """Control network signal mixing."""
    weight_mask: float = 1.0              # Mask ControlNet signal strength
    weight_image: float = 0.25            # Image ControlNet signal strength (Phase 2)
    scales: List[float] = field(default_factory=lambda: [1.0] * 13)  # Per-block scales


@dataclass
class OptimizerConfig:
    """Optimizer and learning rate settings."""
    learning_rate: float = 1e-5           # Base LR for ControlNets
    decoder_lr_scale: float = 0.1         # Decoder LR multiplier
    weight_decay: float = 0.0             # L2 regularization
    ema_decay: float = 0.9999             # EMA smoothing for validation


@dataclass
class TrainingConfig:
    """Model training modes and steps."""
    train_mask_cn: bool = True            # Train mask ControlNet
    train_image_cn: bool = False          # Train image ControlNet (Phase 2)
    sd_locked: bool = True                # Freeze SD decoder
    unlock_last_n: int = 0                # Unlock last N decoder blocks
    max_steps: int = 20000                # Maximum training steps


@dataclass
class FullConfig:
    """Complete training configuration."""
    loss: LossConfig = field(default_factory=LossConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    name: str = "unnamed"                 # Configuration name for logging


# ═════════════════════════════════════════════════════════════════════════════
# PRESETS — Ready-to-use configurations
# ═════════════════════════════════════════════════════════════════════════════

PRESETS: Dict[str, Dict] = {
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1: Mask ControlNet Training
    # ─────────────────────────────────────────────────────────────────────────
    "scratch": {
        "name": "scratch",
        "loss": {
            "weight_mask": 1.0,
            "weight_image": 0.0,
            "weight_distill": 0.0,
            "learn_logvar": True,
        },
        "training": {
            "train_mask_cn": True,
            "train_image_cn": False,
            "sd_locked": True,
            "unlock_last_n": 0,
            "max_steps": 20000,
        },
        "optimizer": {
            "learning_rate": 1e-5,
            "decoder_lr_scale": 0.1,
        },
    },

    "scratch_unlocked": {
        "name": "scratch_unlocked",
        "loss": {
            "weight_mask": 1.0,
            "weight_image": 0.0,
            "weight_distill": 0.0,
        },
        "training": {
            "train_mask_cn": True,
            "train_image_cn": False,
            "sd_locked": False,
            "unlock_last_n": 3,
            "max_steps": 10000,
        },
        "optimizer": {
            "learning_rate": 5e-6,
            "decoder_lr_scale": 0.1,
        },
    },

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2: Multi-Path + Distillation
    # ─────────────────────────────────────────────────────────────────────────
    "polyp_stage2": {
        "name": "polyp_stage2",
        "loss": {
            "weight_mask": 1.0,           # Mask decoder as baseline
            "weight_image": 1.0,          # Image decoder denoising
            "weight_distill": 0.5,        # Anatomy-aware distillation
            "learn_logvar": True,
        },
        "control": {
            "weight_mask": 1.0,
            "weight_image": 0.25,
        },
        "training": {
            "train_mask_cn": False,       # Mask CN frozen (teacher)
            "train_image_cn": True,       # Image CN trainable (student)
            "sd_locked": False,
            "unlock_last_n": 3,
            "max_steps": 10000,
        },
        "optimizer": {
            "learning_rate": 5e-6,
            "decoder_lr_scale": 0.1,
            "ema_decay": 0.9999,
        },
    },

    "scratch_stage2": {
        "name": "scratch_stage2",
        "loss": {
            "weight_mask": 1.0,
            "weight_image": 1.0,
            "weight_distill": 0.5,
        },
        "training": {
            "train_mask_cn": False,
            "train_image_cn": True,
            "sd_locked": False,
            "unlock_last_n": 3,
            "max_steps": 10000,
        },
        "optimizer": {
            "learning_rate": 5e-6,
            "decoder_lr_scale": 0.1,
        },
    },
}


def load_preset(preset_name: str) -> FullConfig:
    """
    Load a preset and return FullConfig object.

    Args:
        preset_name: Key from PRESETS dict

    Returns:
        FullConfig with preset values
    """
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")

    preset_dict = PRESETS[preset_name]
    config = FullConfig(
        name=preset_dict.get("name", preset_name),
        loss=LossConfig(**preset_dict.get("loss", {})),
        control=ControlConfig(**preset_dict.get("control", {})),
        optimizer=OptimizerConfig(**preset_dict.get("optimizer", {})),
        training=TrainingConfig(**preset_dict.get("training", {})),
    )
    return config


def load_config_from_env(config: FullConfig, prefix: str = "ADC_") -> FullConfig:
    """
    Override config values with environment variables.

    Environment variables should be named like:
        ADC_LOSS_WEIGHT_MASK=1.5
        ADC_LEARNING_RATE=1e-6
        ADC_UNLOCK_LAST_N=5
        ADC_TRAIN_IMAGE_CN=1
        ADC_LEARN_LOGVAR=0

    Args:
        config: Base FullConfig to update
        prefix: Environment variable prefix

    Returns:
        Updated FullConfig
    """
    config_dict = asdict(config)

    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue

        # Strip prefix and convert to lowercase path
        env_key = key[len(prefix):].lower()
        parts = env_key.split("_")

        # Navigate to nested dict
        target = config_dict
        for part in parts[:-1]:
            if part not in target:
                print(f"  [!] Ignoring unknown env var: {key} (path not found)")
                break
            if not isinstance(target[part], dict):
                print(f"  [!] Ignoring {key} (intermediate not a dict)")
                break
            target = target[part]
        else:
            # Found valid path, now convert and set
            final_key = parts[-1]
            if final_key in target:
                old_val = target[final_key]
                old_type = type(old_val)

                try:
                    if old_type == bool:
                        new_val = value.lower() in ("1", "true", "yes", "on")
                    elif old_type == int:
                        new_val = int(value)
                    elif old_type == float:
                        new_val = float(value)
                    elif old_type == str:
                        new_val = value
                    else:
                        new_val = value
                    
                    target[final_key] = new_val
                    print(f"  ✓ {key}: {old_val} → {new_val}")
                except ValueError as e:
                    print(f"  [!] Failed to parse {key}={value}: {e}")

    # Reconstruct FullConfig from updated dict
    return FullConfig(
        name=config_dict["name"],
        loss=LossConfig(**config_dict["loss"]),
        control=ControlConfig(**config_dict["control"]),
        optimizer=OptimizerConfig(**config_dict["optimizer"]),
        training=TrainingConfig(**config_dict["training"]),
    )


def apply_config_to_model(model, config: FullConfig) -> None:
    """
    Apply FullConfig to a ControlLDM model instance.

    Sets model attributes for:
    - Loss weighting
    - Control signal mixing
    - Training mode flags
    - Learning rates (via configure_optimizers)

    Args:
        model: PyTorch Lightning ControlLDM module
        config: FullConfig to apply
    """
    # Loss weights
    model.loss_weight_mask = config.loss.weight_mask
    model.loss_weight_image = config.loss.weight_image
    model.loss_weight_distill = config.loss.weight_distill
    model.l_simple_weight = config.loss.simple_weight
    model.original_elbo_weight = config.loss.elbo_weight
    model.learn_logvar = config.loss.learn_logvar

    # Control mixing
    model.control_weight_mask = config.control.weight_mask
    model.control_weight_image = config.control.weight_image
    model.control_scales = config.control.scales

    # Training modes
    model.train_mask_cn = config.training.train_mask_cn
    model.train_image_cn = config.training.train_image_cn
    model.sd_locked = config.training.sd_locked
    model.unlock_last_n = config.training.unlock_last_n

    # Optimizer settings
    model.learning_rate = config.optimizer.learning_rate
    model.decoder_lr_scale = config.optimizer.decoder_lr_scale
    model.ema_decay = config.optimizer.ema_decay

    print(f"\n✓ Applied config '{config.name}' to model")


def config_to_dict(config: FullConfig) -> dict:
    """Convert FullConfig to plain dict (for JSON/YAML serialization)."""
    return asdict(config)


def config_to_json(config: FullConfig) -> str:
    """Convert FullConfig to JSON string."""
    return json.dumps(asdict(config), indent=2)


def save_config(config: FullConfig, filepath: str) -> None:
    """Save FullConfig to JSON file."""
    with open(filepath, "w") as f:
        json.dump(asdict(config), f, indent=2)
    print(f"✓ Saved config to {filepath}")


def load_config_from_json(filepath: str) -> FullConfig:
    """Load FullConfig from JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    return FullConfig(
        name=data.get("name", "loaded"),
        loss=LossConfig(**data["loss"]),
        control=ControlConfig(**data["control"]),
        optimizer=OptimizerConfig(**data["optimizer"]),
        training=TrainingConfig(**data["training"]),
    )


# ═════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Example 1: Load preset
    print("═" * 70)
    print("EXAMPLE 1: Load Preset")
    print("═" * 70)
    config = load_preset("scratch")
    print(config_to_json(config))

    # Example 2: Override with environment
    print("\n" + "═" * 70)
    print("EXAMPLE 2: Override with Environment Variables")
    print("═" * 70)
    os.environ["ADC_LEARNING_RATE"] = "5e-6"
    os.environ["ADC_LOSS_WEIGHT_IMAGE"] = "0.5"
    os.environ["ADC_UNLOCK_LAST_N"] = "3"
    config = load_preset("scratch")
    config = load_config_from_env(config)
    print(config_to_json(config))

    # Example 3: Save/load config
    print("\n" + "═" * 70)
    print("EXAMPLE 3: Save and Load Config")
    print("═" * 70)
    save_config(config, "my_config.json")
    loaded = load_config_from_json("my_config.json")
    print("Loaded:", loaded.training)
