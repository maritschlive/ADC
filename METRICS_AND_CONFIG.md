# ADC Model Metrics & Dynamic Configuration Guide

## Training Metrics (Logged to metrics.csv & TensorBoard)

### Loss Components
| Metric | Phase | Description |
|--------|-------|-------------|
| `train/loss` | Both | **Main training loss** — backprop target, combines all loss signals |
| `train/loss_simple` | Both | **Diffusion MSE loss** — L2 distance between predicted & actual noise |
| `train/loss_gamma` | Both | Weighted loss component (logged if `learn_logvar=True`) |
| `train/loss_vlb` | Both | **Variational Lower Bound loss** — KL divergence term for noise schedule |
| `val/loss` | Val | Validation main loss |
| `val/loss_simple` | Val | Validation diffusion MSE |
| `val/loss_gamma` | Val | Validation weighted loss (if `learn_logvar=True`) |
| `val/loss_vlb` | Val | Validation VLB loss |
| `val/loss_simple_ema` | Val | **EMA validation loss** — exponential moving average version |
| `val/loss_gamma_ema` | Val | EMA gamma loss |
| `val/loss_vlb_ema` | Val | EMA VLB loss |

### Model Parameters & Learning
| Metric | Description |
|--------|-------------|
| `logvar` | **Learned log-variance** of noise schedule (trainable parameter if `learn_logvar=True`) |
| `global_step` | Current training step counter |
| `lr_abs` | Current absolute learning rate (updated per step if scheduler active) |

## Evaluation Metrics (Post-Training Analysis)

Computed via [evaluate_adc.py](evaluate_adc.py):

| Metric | Range | Better | Notes |
|--------|-------|--------|-------|
| **FID** | 0-∞ | Lower | Fréchet Inception Distance — distribution-level image quality. Needs ≥2 images. Gold standard for generation. |
| **SSIM** | 0-1 | Higher | Structural Similarity Index — per-image structural match. Only meaningful if 1:1 image pairs. |
| **LPIPS** | 0-1 | Lower | Learned Perceptual Image Patch Similarity — per-image perceptual quality. Requires lpips package. |

---

## Training Hyperparameters (Dynamic Config)

### Loss Weighting
| Parameter | Type | Default | Phase | Description |
|-----------|------|---------|-------|-------------|
| `loss_weight_mask` (`w_mask`) | float | 1.0 | Phase 1-2 | Weight for **mask decoder denoising loss** |
| `loss_weight_image` (`w_image`) | float | 0.0 | Phase 1, 1.0+ Phase 2 | Weight for **image decoder denoising loss** |
| `loss_weight_distill` (`w_dist`) | float | 0.0 | Phase 1, 0.5+ Phase 2 | Weight for **anatomy-aware distillation** (mask→image) |
| `l_simple_weight` | float | 1.0 | Both | Scaling for simple diffusion loss |
| `original_elbo_weight` | float | 0.1 | Both | Weight for VLB term (ELBO regularizer) |
| `learn_logvar` | bool | True | Both | Enable learnable log-variance; disables if `False` |

### Learned Variance Schedule
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `logvar` | tensor | [0.1, 0.2, ... 0.9] | **Per-timestep log-variance** — learned during training if enabled |
| `lvlb_weights[t]` | tensor | Computed | **VLB weights per timestep** — schedule for noise weighting |

### Control Network Mixing
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `control_weight_mask` | float | 1.0 | Mixing weight for **mask ControlNet** signal |
| `control_weight_image` | float | 0.25 | Mixing weight for **image ControlNet** signal (Phase 2) |
| `control_scales[0:13]` | list | [1.0] × 13 | Per-layer control signal scaling (13 diffusion blocks) |

### Optimizer & Learning Rate
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr` / `learning_rate` | float | 1e-5 | **Base learning rate** for ControlNets |
| `decoder_lr_scale` | float | 0.1 | **Decoder LR multiplier** — decoder_lr = lr × decoder_lr_scale |
| `weight_decay` | float | 0.0 | L2 regularization (0.0 for decoder layers) |
| `ema_decay` | float | 0.9999 | **Exponential Moving Average decay** — EMA for stable validation metrics |

### Model Training Modes
| Parameter | Type | Default | Phase | Description |
|-----------|------|---------|-------|-------------|
| `train_mask_cn` | bool | True | Phase 1, False Phase 2 | Train **mask ControlNet** (frozen as teacher in Phase 2) |
| `train_image_cn` | bool | False | Phase 1, True Phase 2 | Train **image ControlNet** (Phase 2 only) |
| `sd_locked` | bool | True | Phase 1, False Phase 2+ | **Freeze/unfreeze Stable Diffusion decoder** |
| `unlock_last_n` | int | 0 | Phase 1, 3 Phase 2+ | Number of **last decoder blocks to unlock** (0=fully frozen, n=last n blocks trainable) |

---

## Configuration Presets (From tutorial_train_single_gpu.py)

### Phase 1: Mask ControlNet Training
```python
# scratch — from SD v1.5 baseline
"scratch": {
    "loss_weight_mask": 1.0,
    "loss_weight_image": 0.0,
    "loss_weight_distill": 0.0,
    "train_mask_cn": True,
    "train_image_cn": False,
    "sd_locked": True,
    "unlock_last_n": 0,
    "decoder_lr_scale": 0.1,
    "lr": 1e-5,
    "max_steps": 20000,
}

# scratch_unlocked — progressive unfreezing
"scratch_unlocked": {
    "loss_weight_mask": 1.0,
    "loss_weight_image": 0.0,
    "loss_weight_distill": 0.0,
    "train_mask_cn": True,
    "train_image_cn": False,
    "sd_locked": False,
    "unlock_last_n": 3,
    "decoder_lr_scale": 0.1,
    "lr": 5e-6,
    "max_steps": 10000,
}
```

### Phase 2: Multi-Path Training (Image CN + Distillation)
```python
# polyp_stage2 — full Phase 2
"polyp_stage2": {
    "loss_weight_mask": 1.0,      # Mask decoder baseline
    "loss_weight_image": 1.0,     # Image decoder denoising
    "loss_weight_distill": 0.5,   # Anatomy distillation
    "train_mask_cn": False,       # Mask CN frozen (teacher)
    "train_image_cn": True,       # Image CN trainable (student)
    "sd_locked": False,
    "unlock_last_n": 3,
    "decoder_lr_scale": 0.1,
    "lr": 5e-6,
    "max_steps": 10000,
}
```

---

## Dynamic Config Template (Python)

```python
# config.py or similar
CONFIG = {
    # ═══════════════════════════════════════════════════════════════════════
    # LOSS WEIGHTING (dynamic per training run)
    # ═══════════════════════════════════════════════════════════════════════
    "loss": {
        "weight_mask": 1.0,           # Mask decoder loss scale
        "weight_image": 0.0,          # Image decoder loss scale (0=disabled, 1.0+=Phase2)
        "weight_distill": 0.0,        # Anatomy distillation scale
        "simple_weight": 1.0,         # Diffusion MSE scaling
        "elbo_weight": 0.1,           # VLB regularizer
        "learn_logvar": True,         # Enable learnable variance
    },

    # ═══════════════════════════════════════════════════════════════════════
    # CONTROL NETWORK MIXING
    # ═══════════════════════════════════════════════════════════════════════
    "control": {
        "weight_mask": 1.0,           # Mask ControlNet signal strength
        "weight_image": 0.25,         # Image ControlNet signal strength
        "scales": [1.0] * 13,         # Per-block scales (13 diffusion blocks)
    },

    # ═══════════════════════════════════════════════════════════════════════
    # OPTIMIZER & LEARNING RATES
    # ═══════════════════════════════════════════════════════════════════════
    "optimizer": {
        "learning_rate": 1e-5,        # Base LR for ControlNets
        "decoder_lr_scale": 0.1,      # Decoder LR = lr × scale
        "weight_decay": 0.0,          # L2 regularization
        "ema_decay": 0.9999,          # EMA smooth factor for validation
    },

    # ═══════════════════════════════════════════════════════════════════════
    # MODEL TRAINING MODES
    # ═══════════════════════════════════════════════════════════════════════
    "training": {
        "train_mask_cn": True,        # Train mask ControlNet
        "train_image_cn": False,      # Train image ControlNet
        "sd_locked": True,            # Freeze SD decoder
        "unlock_last_n": 0,           # Unlock last N decoder blocks
        "max_steps": 20000,
    },

    # ═══════════════════════════════════════════════════════════════════════
    # LOGGING & MONITORING
    # ═══════════════════════════════════════════════════════════════════════
    "logging": {
        "log_every_t": 100,           # Log interval (timesteps)
        "save_interval": 1000,        # Save checkpoint interval
        "val_interval": 500,          # Validation step interval
    },
}
```

### Usage Example:
```python
# Load and apply config
model.loss_weight_mask = CONFIG["loss"]["weight_mask"]
model.loss_weight_image = CONFIG["loss"]["weight_image"]
model.loss_weight_distill = CONFIG["loss"]["weight_distill"]
model.learning_rate = CONFIG["optimizer"]["learning_rate"]
model.train_mask_cn = CONFIG["training"]["train_mask_cn"]
# ... etc
```

---

## Metric Dependencies & Relationships

```
┌─────────────────────────────────────────────┐
│        INPUT: Image + Mask                  │
└────────────────────┬────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
    ┌────▼────┐         ┌───────▼──────┐
    │ Mask CN │         │  Image CN    │
    └────┬────┘         └───────┬──────┘
         │                      │
    Loss 0: mask_simple  Loss 1: image_simple
    (w_mask × 1.0)      (w_image × 1.0)
         │                      │
         │         ┌────────────┘
         │         │
         │    Loss 2: distillation
         │    (mask → image)
         │    (w_distill × 1.0)
         │         │
         └────┬────┘
              │
         Total Loss = Loss0 + Loss1 + Loss2
              │
         ┌────┴─────────────────────┐
         │                          │
    loss_simple      loss_gamma + logvar
    (gradient)       (if learn_logvar)
         │                          │
         ├──────┬────────────────────┤
         │      │                    │
    loss_vlb (ELBO term)       ema-smoothed
         │                          │
         └──────┬─────────┬──────────┘
                │         │
        train/loss    val/loss_ema
```

---

## Recommended Metric Monitoring Strategy

### Training Phase 1 (Mask CN only):
```python
MONITOR = {
    "primary": "train/loss",           # Main optimization target
    "secondary": ["train/loss_simple", "train/loss_vlb"],
    "validation": ["val/loss_simple_ema", "logvar"],
}
```

### Training Phase 2 (Multi-path):
```python
MONITOR = {
    "primary": "train/loss",
    "secondary": [
        "train/loss_simple",          # Mask decoder
        "train/loss_image",           # Image decoder (if enabled)
        "train/loss_distill",         # Distillation
    ],
    "validation": ["val/loss_simple_ema", "val/loss_image"],
}
```

### Post-Training Evaluation:
```python
EVALUATE = {
    "fid": "Primary metric (distribution quality)",
    "ssim": "Secondary (structural similarity)",
    "lpips": "Tertiary (perceptual quality)",
}
```

---

## Environment Variable Overrides (from tutorial_train_single_gpu.py)

```bash
# Override LR
export LR=5e-6

# Override losses
export LOSS_MASK=1.0
export LOSS_IMAGE=0.0
export LOSS_DISTILL=0.0

# Override control weights
export CONTROL_WEIGHT_MASK=1.0
export CONTROL_WEIGHT_IMAGE=0.25

# Override decoder settings
export UNLOCK_LAST_N=3
export DECODER_LR_SCALE=0.1
```

---

## Quick Reference: All Changeable Parameters

| Category | Variable | Min | Max | Recommended |
|----------|----------|-----|-----|-------------|
| **Loss Weights** | loss_weight_mask | 0 | 2.0 | 1.0 |
| | loss_weight_image | 0 | 2.0 | 0.0 (Phase1), 1.0 (Phase2) |
| | loss_weight_distill | 0 | 2.0 | 0.0 (Phase1), 0.5 (Phase2) |
| | elbo_weight | 0 | 1.0 | 0.1 |
| **Control Mixing** | control_weight_mask | 0 | 2.0 | 1.0 |
| | control_weight_image | 0 | 2.0 | 0.25 |
| **Learning** | learning_rate | 1e-7 | 1e-3 | 1e-5 - 5e-6 |
| | decoder_lr_scale | 0 | 1.0 | 0.1 |
| | weight_decay | 0 | 1e-1 | 0.0 (decoder), 1e-2 (others) |
| | ema_decay | 0.9 | 0.9999 | 0.9999 |
| **Model Modes** | unlock_last_n | 0 | 36 | 0 (Phase1), 3 (Phase2) |

