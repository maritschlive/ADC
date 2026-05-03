"""
QUICK REFERENCE: All Metrics & Config Parameters
==================================================

Print-friendly cheat sheet for all changeable metrics and values.
"""

# ═════════════════════════════════════════════════════════════════════════════
# TRAINING METRICS (logged per step)
# ═════════════════════════════════════════════════════════════════════════════

TRAINING_METRICS = {
    "train/loss": "Main loss (all components)",
    "train/loss_simple": "Diffusion MSE component",
    "train/loss_gamma": "Weighted loss (if learn_logvar=True)",
    "train/loss_vlb": "Variational Lower Bound (KL term)",
    "logvar": "Learned log-variance of noise schedule",
    "global_step": "Step counter",
    "lr_abs": "Current learning rate",
}

VALIDATION_METRICS = {
    "val/loss": "Validation main loss",
    "val/loss_simple": "Validation MSE",
    "val/loss_gamma": "Validation gamma",
    "val/loss_vlb": "Validation VLB",
    "val/loss_simple_ema": "EMA smoothed validation",
    "val/loss_gamma_ema": "EMA smoothed gamma",
    "val/loss_vlb_ema": "EMA smoothed VLB",
}

# ═════════════════════════════════════════════════════════════════════════════
# EVALUATION METRICS (post-training)
# ═════════════════════════════════════════════════════════════════════════════

EVAL_METRICS = {
    "FID": {"range": (0, float('inf')), "better": "lower", "requires": "≥2 imgs"},
    "SSIM": {"range": (0, 1), "better": "higher", "requires": "1:1 pairs"},
    "LPIPS": {"range": (0, 1), "better": "lower", "requires": "1:1 pairs"},
}

# ═════════════════════════════════════════════════════════════════════════════
# LOSS WEIGHTING PARAMETERS
# ═════════════════════════════════════════════════════════════════════════════

LOSS_WEIGHTS = {
    "loss_weight_mask": {
        "type": "float",
        "range": (0, 2.0),
        "default": 1.0,
        "description": "Mask decoder denoising loss scale",
        "phase": "1,2",
    },
    "loss_weight_image": {
        "type": "float",
        "range": (0, 2.0),
        "default": "0.0 (Ph1), 1.0 (Ph2)",
        "description": "Image decoder denoising loss scale",
        "phase": "1,2",
    },
    "loss_weight_distill": {
        "type": "float",
        "range": (0, 2.0),
        "default": "0.0 (Ph1), 0.5 (Ph2)",
        "description": "Anatomy-aware distillation (mask→image)",
        "phase": "1,2",
    },
    "l_simple_weight": {
        "type": "float",
        "range": (0, 2.0),
        "default": 1.0,
        "description": "Diffusion MSE scaling",
    },
    "original_elbo_weight": {
        "type": "float",
        "range": (0, 1.0),
        "default": 0.1,
        "description": "VLB regularizer weight",
    },
    "learn_logvar": {
        "type": "bool",
        "default": True,
        "description": "Enable learnable log-variance schedule",
    },
}

# ═════════════════════════════════════════════════════════════════════════════
# CONTROL NETWORK PARAMETERS
# ═════════════════════════════════════════════════════════════════════════════

CONTROL_PARAMS = {
    "control_weight_mask": {
        "type": "float",
        "range": (0, 2.0),
        "default": 1.0,
        "description": "Mask ControlNet signal strength",
    },
    "control_weight_image": {
        "type": "float",
        "range": (0, 2.0),
        "default": 0.25,
        "description": "Image ControlNet signal strength (Phase 2)",
    },
    "control_scales": {
        "type": "list[float]",
        "length": 13,
        "default": "[1.0] * 13",
        "description": "Per-block control scaling (one per diffusion block)",
    },
}

# ═════════════════════════════════════════════════════════════════════════════
# OPTIMIZER PARAMETERS
# ═════════════════════════════════════════════════════════════════════════════

OPTIMIZER_PARAMS = {
    "learning_rate": {
        "type": "float",
        "range": (1e-7, 1e-3),
        "recommended": "1e-5 (Ph1), 5e-6 (Ph2)",
        "description": "Base LR for ControlNets",
    },
    "decoder_lr_scale": {
        "type": "float",
        "range": (0, 1.0),
        "default": 0.1,
        "description": "Decoder LR = lr × scale",
    },
    "weight_decay": {
        "type": "float",
        "range": (0, 0.1),
        "default": 0.0,
        "description": "L2 regularization (0 for decoder)",
    },
    "ema_decay": {
        "type": "float",
        "range": (0.9, 0.9999),
        "default": 0.9999,
        "description": "EMA smoothing for stable validation",
    },
}

# ═════════════════════════════════════════════════════════════════════════════
# TRAINING MODE FLAGS
# ═════════════════════════════════════════════════════════════════════════════

TRAINING_MODES = {
    "train_mask_cn": {
        "type": "bool",
        "default": "True (Ph1), False (Ph2)",
        "description": "Train mask ControlNet",
    },
    "train_image_cn": {
        "type": "bool",
        "default": "False (Ph1), True (Ph2)",
        "description": "Train image ControlNet",
    },
    "sd_locked": {
        "type": "bool",
        "default": "True (Ph1), False (Ph2)",
        "description": "Freeze/unfreeze SD decoder",
    },
    "unlock_last_n": {
        "type": "int",
        "range": (0, 36),
        "default": "0 (Ph1), 3 (Ph2)",
        "description": "# of last decoder blocks to unlock (0=frozen)",
    },
    "max_steps": {
        "type": "int",
        "default": 20000,
        "description": "Maximum training steps",
    },
}

# ═════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT VARIABLE OVERRIDES
# ═════════════════════════════════════════════════════════════════════════════

ENV_OVERRIDES = """
Set environment variables with prefix ADC_:

    ADC_LEARNING_RATE=5e-6          # float
    ADC_DECODER_LR_SCALE=0.15       # float
    ADC_LOSS_WEIGHT_MASK=1.2        # float
    ADC_LOSS_WEIGHT_IMAGE=0.8       # float
    ADC_LOSS_WEIGHT_DISTILL=0.4     # float
    ADC_UNLOCK_LAST_N=5             # int
    ADC_TRAIN_IMAGE_CN=1            # bool (1=True, 0=False)
    ADC_TRAIN_MASK_CN=0             # bool
    ADC_SD_LOCKED=0                 # bool
    ADC_LEARN_LOGVAR=1              # bool
    ADC_CONTROL_WEIGHT_MASK=1.0     # float
    ADC_CONTROL_WEIGHT_IMAGE=0.3    # float

Example usage:
    export ADC_LEARNING_RATE=1e-4
    export ADC_UNLOCK_LAST_N=5
    python tutorial_train_single_gpu.py
"""

# ═════════════════════════════════════════════════════════════════════════════
# TYPICAL CONFIGURATIONS
# ═════════════════════════════════════════════════════════════════════════════

TYPICAL_CONFIGS = {
    "Phase_1_Scratch": {
        "loss_weight_mask": 1.0,
        "loss_weight_image": 0.0,
        "loss_weight_distill": 0.0,
        "train_mask_cn": True,
        "train_image_cn": False,
        "sd_locked": True,
        "unlock_last_n": 0,
        "learning_rate": 1e-5,
        "max_steps": 20000,
    },
    "Phase_1_Transfer": {
        "loss_weight_mask": 1.0,
        "loss_weight_image": 0.0,
        "loss_weight_distill": 0.0,
        "train_mask_cn": True,
        "train_image_cn": False,
        "sd_locked": True,
        "unlock_last_n": 0,
        "learning_rate": 1e-5,
        "max_steps": 20000,
    },
    "Phase_1_Unlocked": {
        "loss_weight_mask": 1.0,
        "loss_weight_image": 0.0,
        "loss_weight_distill": 0.0,
        "train_mask_cn": True,
        "train_image_cn": False,
        "sd_locked": False,
        "unlock_last_n": 3,
        "learning_rate": 5e-6,
        "max_steps": 10000,
    },
    "Phase_2_Full": {
        "loss_weight_mask": 1.0,
        "loss_weight_image": 1.0,
        "loss_weight_distill": 0.5,
        "train_mask_cn": False,
        "train_image_cn": True,
        "sd_locked": False,
        "unlock_last_n": 3,
        "learning_rate": 5e-6,
        "max_steps": 10000,
    },
}

# ═════════════════════════════════════════════════════════════════════════════
# LOSS CALCULATION FORMULAS
# ═════════════════════════════════════════════════════════════════════════════

LOSS_FORMULAS = """
Loss Composition:

    loss_simple_mask   = w_mask × MSE(pred_mask, target)
    loss_simple_image  = w_image × MSE(pred_image, target)      [if enabled]
    loss_distill       = w_dist × MSE(pred_mask, pred_image)    [if enabled]
    
    loss_simple_total  = loss_simple_mask + loss_simple_image + loss_distill
    
    loss_weighted      = loss_simple_total / exp(logvar[t]) + logvar[t]
    
    loss_vlb           = lvlb_weights[t] × loss_simple_total    [KL regularizer]
    
    loss_total         = l_simple_weight × loss_weighted
                       + original_elbo_weight × loss_vlb

Gradient Flow:
    loss_total
        ↓
    [loss_simple_mask] ← weights on mask decoder
    [loss_simple_image] ← weights on image decoder
    [loss_distill] ← anatomy similarity constraint
        ↓
    ControlNets + Decoder (if unlocked)

Per-Pixel Weighting (anatomy-aware):
    weight = weights_polyp + weights_background
    - weights_polyp: emphasizes region gradients
    - weights_background: emphasizes region contrast
"""

# ═════════════════════════════════════════════════════════════════════════════
# METRICS DEPENDENCY TREE
# ═════════════════════════════════════════════════════════════════════════════

DEPENDENCY_TREE = """
train/loss ← depends on:
├─ train/loss_simple (MSE component)
│  ├─ w_mask, w_image, w_dist (loss weights)
│  └─ model_output shape
├─ train/loss_vlb (KL regularizer)
│  └─ original_elbo_weight, lvlb_weights
└─ logvar (learned variance)
   └─ learn_logvar flag

val/loss_ema ← depends on:
├─ train_image_cn (Phase 2 enables image metrics)
├─ ema_decay (smoothing strength)
└─ Validation batch

FID ← depends on:
├─ Generated image quality
├─ Dataset representativeness
└─ ≥2 samples required

Gradient Updates ← depend on:
├─ All loss components above
├─ optimizer: learning_rate, weight_decay
├─ Training modes: train_mask_cn, train_image_cn
└─ Decoder: unlock_last_n (which params trainable)
"""

if __name__ == "__main__":
    print("╔" + "═" * 78 + "╗")
    print("║" + "ADC METRICS & CONFIG REFERENCE".center(78) + "║")
    print("╚" + "═" * 78 + "╝\n")
    
    print("TRAINING METRICS:")
    for metric, desc in TRAINING_METRICS.items():
        print(f"  {metric:30s} {desc}")
    
    print("\n" + "─" * 80)
    print("LOSS WEIGHTING PARAMETERS:")
    for param, info in LOSS_WEIGHTS.items():
        default = info.get("default", "—")
        print(f"  {param:30s} default={default:20s} range={str(info.get('range', '—')):15s}")
    
    print("\n" + "─" * 80)
    print("OPTIMIZER PARAMETERS:")
    for param, info in OPTIMIZER_PARAMS.items():
        default = info.get("default", info.get("recommended", "—"))
        print(f"  {param:30s} default={default:20s} range={str(info.get('range', '—')):15s}")
    
    print("\n" + "─" * 80)
    print("TRAINING MODES:")
    for param, info in TRAINING_MODES.items():
        default = info.get("default", "—")
        print(f"  {param:30s} type={info['type']:10s} default={default}")
    
    print("\n" + "─" * 80)
    print("ENVIRONMENT OVERRIDES:")
    print(ENV_OVERRIDES)
