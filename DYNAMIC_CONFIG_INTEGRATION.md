"""
INTEGRATION GUIDE: Using dynamic_config.py with tutorial_train_single_gpu.py
================================================================================

This guide shows how to integrate the new dynamic_config.py module into
your existing training pipeline for maximum flexibility.

Three integration approaches (from simplest to most flexible):
"""

# ═════════════════════════════════════════════════════════════════════════════
# APPROACH 1: Minimal (swap presets, use env overrides)
# ═════════════════════════════════════════════════════════════════════════════

"""
FILE: tutorial_train_single_gpu.py (modified excerpt)

Replace the hardcoded PRESETS dict with dynamic loading:

    from dynamic_config import load_preset, load_config_from_env, apply_config_to_model

    # At the top of main():
    preset_name = os.environ.get("PRESET", "scratch")
    config = load_preset(preset_name)           # Load base preset
    config = load_config_from_env(config)       # Override with env vars
    apply_config_to_model(model, config)        # Apply to model

USAGE:
    # Base preset (unchanged)
    PRESET=scratch uv run python tutorial_train_single_gpu.py

    # Override learning rate
    PRESET=scratch ADC_LEARNING_RATE=5e-6 uv run python tutorial_train_single_gpu.py

    # Multi-override Phase 2
    PRESET=scratch_stage2 \\
        ADC_LOSS_WEIGHT_IMAGE=1.0 \\
        ADC_LOSS_WEIGHT_DISTILL=0.5 \\
        ADC_TRAIN_IMAGE_CN=1 \\
        ADC_UNLOCK_LAST_N=3 \\
        uv run python tutorial_train_single_gpu.py
"""


# ═════════════════════════════════════════════════════════════════════════════
# APPROACH 2: Config file (JSON)
# ═════════════════════════════════════════════════════════════════════════════

"""
FILE: configs/my_experiment.json

{
  "name": "phase2_high_lr",
  "loss": {
    "weight_mask": 1.0,
    "weight_image": 1.5,
    "weight_distill": 0.7,
    "learn_logvar": true
  },
  "control": {
    "weight_mask": 1.0,
    "weight_image": 0.3
  },
  "optimizer": {
    "learning_rate": 1e-5,
    "decoder_lr_scale": 0.15,
    "ema_decay": 0.9999
  },
  "training": {
    "train_mask_cn": false,
    "train_image_cn": true,
    "sd_locked": false,
    "unlock_last_n": 3,
    "max_steps": 10000
  }
}

USAGE in code:
    from dynamic_config import load_config_from_json, apply_config_to_model

    config = load_config_from_json("configs/my_experiment.json")
    apply_config_to_model(model, config)

OR via environment:
    CONFIG_PATH=configs/my_experiment.json uv run python tutorial_train_single_gpu.py
"""


# ═════════════════════════════════════════════════════════════════════════════
# APPROACH 3: Full Integration (recommended for production)
# ═════════════════════════════════════════════════════════════════════════════

"""
Modified tutorial_train_single_gpu.py excerpt:

```python
import os
import sys
from pathlib import Path

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

from share import *
from dynamic_config import load_preset, load_config_from_env, load_config_from_json, apply_config_to_model
from cldm.model import create_model, load_state_dict


def main():
    # ─────────────────────────────────────────────────────────────────────────
    # 1. Load base configuration
    # ─────────────────────────────────────────────────────────────────────────
    config_source = os.environ.get("CONFIG_PATH")
    preset_name = os.environ.get("PRESET", "scratch")

    if config_source:
        # Option A: Load from JSON file
        print(f"Loading config from file: {config_source}")
        config = load_config_from_json(config_source)
    else:
        # Option B: Load from preset
        print(f"Loading preset: {preset_name}")
        config = load_preset(preset_name)

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Apply environment overrides
    # ─────────────────────────────────────────────────────────────────────────
    print("Checking for environment overrides (ADC_*=value)...")
    config = load_config_from_env(config)

    print(f"\\nFinal Configuration:")
    from dynamic_config import config_to_json
    print(config_to_json(config))

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Create & configure model
    # ─────────────────────────────────────────────────────────────────────────
    model = create_model("models/cldm_v15.yaml")
    apply_config_to_model(model, config)  # ← Single line applies all settings

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Continue with training setup (unchanged)
    # ─────────────────────────────────────────────────────────────────────────
    # ... trainer, dataloader, logging setup ...
```
"""


# ═════════════════════════════════════════════════════════════════════════════
# EXAMPLE WORKFLOWS
# ═════════════════════════════════════════════════════════════════════════════

"""
Workflow 1: Quick experiment with environment overrides
───────────────────────────────────────────────────────

    # Start from Phase 1 scratch, test higher LR
    PRESET=scratch ADC_LEARNING_RATE=1e-4 ADC_LOSS_WEIGHT_MASK=1.2 \\
        uv run python tutorial_train_single_gpu.py

Workflow 2: Sweep over configurations (grid search)
────────────────────────────────────────────────────

    for lr in 1e-5 5e-6 1e-6; do
        for mask_w in 0.8 1.0 1.2; do
            echo "Testing LR=$lr, mask_weight=$mask_w"
            PRESET=scratch \\
                ADC_LEARNING_RATE=$lr \\
                ADC_LOSS_WEIGHT_MASK=$mask_w \\
                RUN_TAG="grid_${lr}_${mask_w}" \\
                uv run python tutorial_train_single_gpu.py
        done
    done

Workflow 3: Reproduce experiment from saved config
──────────────────────────────────────────────────

    # Save config after successful run
    cp configs/my_experiment.json runs/my_run_v1/config_used.json

    # Later, reproduce it exactly
    CONFIG_PATH=runs/my_run_v1/config_used.json \\
        RUN_TAG="my_run_v1_reproduce" \\
        uv run python tutorial_train_single_gpu.py

Workflow 4: Phase 1 → Phase 2 progression
─────────────────────────────────────────

    # Phase 1: Train mask CN from scratch (20k steps)
    PRESET=scratch uv run python tutorial_train_single_gpu.py

    # Phase 2: Load Phase 1 checkpoint, add image CN + distillation
    PRESET=scratch_stage2 \\
        RESUME_PATH=runs/scratch/version_0/checkpoints/last.ckpt \\
        RUN_TAG="phase2_from_scratch" \\
        uv run python tutorial_train_single_gpu.py

Workflow 5: Custom config with JSON
────────────────────────────────────

    # Create custom config
    cat > configs/my_custom.json << 'EOF'
    {
      "name": "custom_phase2",
      "loss": {"weight_mask": 1.0, "weight_image": 1.5, "weight_distill": 0.8},
      "training": {"train_mask_cn": false, "train_image_cn": true, "unlock_last_n": 3}
    }
    EOF

    CONFIG_PATH=configs/my_custom.json uv run python tutorial_train_single_gpu.py
"""


# ═════════════════════════════════════════════════════════════════════════════
# METRICS MONITORING WITH DYNAMIC CONFIG
# ═════════════════════════════════════════════════════════════════════════════

"""
The metrics logged depend on your config settings. Auto-select based on phase:

    from dynamic_config import load_preset

    config = load_preset("scratch_stage2")
    
    # Determine which metrics to monitor based on config
    if config.training.train_image_cn or config.loss.weight_image > 0:
        # Phase 2: multi-path
        MONITOR_METRICS = [
            "train/loss",
            "train/loss_simple",
            "val/loss_simple_ema",
            "logvar",
        ]
    else:
        # Phase 1: mask only
        MONITOR_METRICS = [
            "train/loss",
            "train/loss_simple",
            "val/loss_simple_ema",
        ]

    for metric in MONITOR_METRICS:
        trainer.add_metric_monitor(metric)
"""


# ═════════════════════════════════════════════════════════════════════════════
# SAVING & LOADING CONFIGURATIONS
# ═════════════════════════════════════════════════════════════════════════════

"""
Example: Always save the config used for training

    from dynamic_config import save_config, config_to_json
    
    # After setting up model with config:
    config_log_dir = os.path.join(save_dir, "config_used.json")
    save_config(config, config_log_dir)
    
    # Also log to tensorboard/wandb
    with open(config_log_dir, "w") as f:
        logger.log_dict({"config_json": config_to_json(config)})
"""


# ═════════════════════════════════════════════════════════════════════════════
# TESTING YOUR CONFIGURATIONS
# ═════════════════════════════════════════════════════════════════════════════

"""
Quick validation without training:

    python -c "
    from dynamic_config import load_preset, load_config_from_env, config_to_json
    import os
    
    os.environ['ADC_LEARNING_RATE'] = '5e-6'
    config = load_preset('scratch')
    config = load_config_from_env(config)
    print(config_to_json(config))
    "
"""


# ═════════════════════════════════════════════════════════════════════════════
# FAQ & TROUBLESHOOTING
# ═════════════════════════════════════════════════════════════════════════════

"""
Q: How do I see what config is being used?
A: Add this after apply_config_to_model():
   
   from dynamic_config import config_to_json
   print(f"\\nUsing config:\\n{config_to_json(config)}")

Q: Can I mix preset + JSON + env overrides?
A: Yes (in order of priority):
   1. Load JSON (if CONFIG_PATH set)
   2. Else load preset (PRESET env var)
   3. Apply env overrides (ADC_* vars)
   
   So: ADC_* vars always win over JSON/preset values.

Q: How do I know which metrics will be logged?
A: Check loss weights in your config:
   - If loss.weight_image == 0.0 → only mask decoder metrics logged
   - If loss.weight_image > 0.0 → image decoder metrics also logged
   - If loss.weight_distill > 0.0 → distillation metrics logged

Q: Can I use partial configs in JSON?
A: Yes, missing keys use FullConfig defaults. Only override what you need:
   
   {
     "name": "just_lr_change",
     "optimizer": {"learning_rate": 2e-5}
   }

Q: How to enable all losses for debugging?
A: Set env vars:
   
   ADC_LOSS_WEIGHT_MASK=1.0 \\
   ADC_LOSS_WEIGHT_IMAGE=1.0 \\
   ADC_LOSS_WEIGHT_DISTILL=1.0 \\
   ADC_LEARN_LOGVAR=1
"""
