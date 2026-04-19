# ADC Project Training Guide

This guide is the practical operator manual for this repository.

It explains:
- what the project does and how data flows through it,
- what each Slurm script does exactly,
- where to change values for fine-tuning and training strategies,
- how to run, monitor, control, resume, and evaluate experiments.

This guide is based on current code in this repository, not only README text.

## 1) Project in One Page

ADC (Adaptively Distilled ControlNet) is a diffusion-based medical image synthesis project.

Core idea in this codebase:
- condition on a binary anatomical mask,
- optionally add a second image ControlNet branch,
- train with phase-specific losses (mask denoising, image denoising, mask-to-image distillation),
- generate realistic laparoscopic liver-like images from masks.

Main training path:
1. Prepare dataset JSON and resized image/mask pairs.
2. Train with a preset in `tutorial_train_single_gpu.py`.
3. Optionally orchestrate all presets with `run_all.py`.
4. Run inference and evaluate outputs.

## 2) Repository Map (What Matters Most)

### Core training and orchestration
- `tutorial_train_single_gpu.py`: main training entrypoint for single preset.
- `run_all.py`: sequential multi-preset orchestration with dependency and completion checks.
- `cldm/cldm.py`: model forward, optimizer parameter groups, loss composition.
- `models/cldm_v15.yaml`: architecture config for ControlLDM + ControlNets.

### Data preparation and loading
- `prepare_liver_data.py`: converts raw images/masks into ADC format under `data/`.
- `tutorial_dataset.py`: training dataset loader (`data/prompt.json` by default).
- `tutorial_dataset_sample.py`: inference/test dataset loader (`data/test/prompt.json` by default).

### Setup, inference, evaluation
- `setup_adc.py`: one-command setup for dependencies/weights/checkpoints.
- `tutorial_inference_local.py`: MPS/CPU/CUDA inference.
- `evaluate_adc.py`: FID, SSIM, LPIPS evaluation.
- `analyze_runs.py`: post-training run summary report generation.
- `vram_calculator.py`: preset-specific VRAM estimate utility.

### Slurm scripts
- `slurm/setup.sh`: one-time setup job.
- `slurm/train.sh`: training job (single preset or all presets).
- `slurm/train_all.sh`: submit helper for all presets (single-job or split dependency mode).
- `slurm/submit_experiments.sh`: multi-strategy experiment sweeps with dependency chaining.
- `slurm/infer.sh`: inference job.

## 3) End-to-End Workflow

### Step A: One-time setup
- Install dependencies and download model weights.
- Generate `stable-diffusion-v1-5/control_sd15.ckpt` (training initialization checkpoint).

Use:
```bash
uv run python setup_adc.py
```

### Step B: Prepare liver data
Convert your raw dataset to expected ADC structure and prompt JSON files.

Use:
```bash
uv run python prepare_liver_data.py --src /path/to/raw --out ./data --pre-split --dsad --prompt "a laparoscopic image of the liver"
```

Outputs:
- `data/train/images/*.png`
- `data/train/masks/*.png`
- `data/train/prompt.json`
- same for `val` and optionally `test`
- combined `data/prompt.json`

### Step C: Train
Run one preset or run all presets.

Single preset:
```bash
PRESET=scratch TRAINING_TARGET=workstation uv run python tutorial_train_single_gpu.py
```

All presets with dependency logic:
```bash
TRAINING_TARGET=workstation uv run python run_all.py
```

### Step D: Inference
```bash
CKPT_PATH=./runs/scratch/version_0/checkpoints/last.ckpt uv run python tutorial_inference_local.py
```

### Step E: Evaluate
```bash
uv run python evaluate_adc.py --real data/train/images --generated generated_results/local_test/images --out eval_results.json
```

## 4) Data Contract and Format

Training dataset items are dictionaries with keys:
- `source`: mask image path
- `target`: RGB image path
- `prompt_target`: text prompt

Example item:
```json
{
  "source": "data/train/masks/000000.png",
  "target": "data/train/images/000000.png",
  "prompt_target": "a laparoscopic image of the liver"
}
```

Current format notes:
- `prepare_liver_data.py` writes standard JSON arrays to `prompt.json`.
- loaders in `tutorial_dataset.py` and `tutorial_dataset_sample.py` accept both:
  - JSON array (preferred),
  - legacy JSONL (one JSON object per line).

Loader behavior in `tutorial_dataset.py`:
- mask is loaded as grayscale, binarized, and used as `hint` condition,
- image is loaded as RGB target,
- random prompt dropout at 5 percent for classifier-free guidance (`prompt_target` empty sometimes),
- resize to 384x384.

## 5) Slurm Scripts: Exact Behavior

## `slurm/setup.sh`
Purpose:
- one-time cluster setup before training.

Resources:
- partition `workstations`
- qos `students_qos`
- `gpu:1`, `cpus-per-task=4`, `mem=32G`, `time=02:00:00`

What it does:
1. `cd "$HOME/ADC"`
2. ensures `uv` is available:
   - tries `curl` installer,
   - fallback `wget`,
   - fallback `python3 -m pip install --user uv`.
3. runs setup:
   - `uv run python3 setup_adc.py` if `python3` exists,
   - else `uv run python setup_adc.py`.

Exit behavior:
- exits with code 127 if installer tooling and `uv` are unavailable.

Typical submit command:
```bash
sbatch slurm/setup.sh
```

## `slurm/train.sh`
Purpose:
- run training in Slurm (single preset or all presets).

Resources:
- partition `workstations`
- qos `students_qos`
- `gpu:1`, `cpus-per-task=8`, `mem=64G`, `time=72:00:00`

What it does:
1. sets `PRESET=${PRESET:-all}`.
2. hard-sets `TRAINING_TARGET=workstation` for this job.
3. if `PRESET=all`:
   - runs `uv run python run_all.py`
4. else:
   - runs `uv run python tutorial_train_single_gpu.py`
5. prints end-of-job summary:
   - node and GPU info,
   - start/end timestamps,
   - exit code,
   - disk/checkpoint/image counts per run directory,
   - first 25 lines of `runs/training_report.md` if present.

Typical submit commands:
```bash
# default PRESET=all
sbatch slurm/train.sh

# single preset
PRESET=scratch sbatch slurm/train.sh
```

## `slurm/train_all.sh`
Purpose:
- convenience submit wrapper for all presets.

Modes:
1. default mode:
   - submits one job: `sbatch --export=ALL,PRESET=all slurm/train.sh`
   - this job runs `run_all.py` internally.
2. `--split` mode:
   - submits one job per preset with dependency chaining (`afterany`).

Important detail:
- `--split` default preset list currently is:
  - `scratch`
  - `polyp_transfer`
  - `scratch_unlocked`
  - `polyp_unlocked`
  - `polyp_stage2`
  - `scratch_stage2`
  - `polyp_stage2_from_unlocked`
- this list does not include `paper_faithful_polyp` or `paper_faithful_scratch` unless you pass them explicitly.

Typical usage:
```bash
# one job for all presets (recommended default)
bash slurm/train_all.sh

# split into dependent jobs
bash slurm/train_all.sh --split

# split with explicit custom list
bash slurm/train_all.sh --split scratch polyp_transfer polyp_stage2
```

## `slurm/infer.sh`
Purpose:
- run inference as Slurm job.

Resources:
- partition `workstations`
- qos `students_qos`
- `gpu:1`, `cpus-per-task=4`, `mem=32G`, `time=01:00:00`

What it does:
1. `cd "$HOME/ADC"`
2. runs `uv run python tutorial_inference_local.py`

Typical submit command:
```bash
CKPT_PATH=./runs/polyp_stage2/version_0/checkpoints/last.ckpt sbatch slurm/infer.sh
```

## 6) What Training Actually Trains

Training logic is controlled by preset values in `tutorial_train_single_gpu.py` and applied in `cldm/cldm.py`.

Potential trainable parts:
- mask ControlNet (`control_model`)
- image ControlNet (`image_control_model`)
- UNet decoder blocks (last `N` via `unlock_last_n`, plus output heads)

Optimizer setup (`cldm/cldm.py`):
- AdamW with parameter groups based on preset flags:
  - include mask CN params if `train_mask_cn=True`
  - include image CN params if `train_image_cn=True`
  - include decoder params if `sd_locked=False`
- decoder LR uses `lr * decoder_lr_scale`.

Loss composition (`cldm/cldm.py`):
- mask denoising loss weighted by `loss_weight_mask` (always 1.0 in training script)
- image denoising loss weighted by `loss_weight_image`
- distillation loss weighted by `loss_weight_distill`

Control branch mixing:
- `control_weight_mask`
- `control_weight_image`

## 7) Presets, Steps, and Dependency Chain

Defined in `tutorial_train_single_gpu.py`.

| Preset | Start checkpoint | Main training intent | LR | Max steps |
|---|---|---|---:|---:|
| scratch | `./stable-diffusion-v1-5/control_sd15.ckpt` | Phase 1 from SD base, train CN path | 1e-5 | 20000 |
| polyp_transfer | `./adc_weights/merged_pytorch_model.pth` | Phase 1 transfer from ADC polyp | 1e-5 | 20000 |
| scratch_unlocked | `$scratch` | Phase 1b, unlock last 3 decoder blocks | 5e-6 | 10000 |
| polyp_unlocked | `$polyp_transfer` | Phase 1b transfer + decoder unlock | 5e-6 | 10000 |
| polyp_stage2 | `$polyp_transfer` | Phase 2, image CN + distillation | 5e-6 | 10000 |
| scratch_stage2 | `$scratch_unlocked` | Phase 2 from scratch path | 5e-6 | 10000 |
| polyp_stage2_from_unlocked | `$polyp_unlocked` | Phase 2 from unlocked transfer path | 5e-6 | 10000 |
| paper_faithful_polyp | `./adc_weights/merged_pytorch_model.pth` | Paper-like dual CN + equal losses | 1e-5 | 24000 |
| paper_faithful_scratch | `./stable-diffusion-v1-5/control_sd15.ckpt` | Paper-like dual CN from SD base | 1e-5 | 24000 |

Dependency graph used by `run_all.py`:
- `scratch -> scratch_unlocked -> scratch_stage2`
- `polyp_transfer -> polyp_unlocked -> polyp_stage2_from_unlocked`
- `polyp_transfer -> polyp_stage2`
- `paper_faithful_*` are standalone

If all presets run from scratch (none already complete), total planned training steps are:
- `138000` steps

`run_all.py` completion detection checks:
1. metrics CSV last `step`
2. checkpoint filename step (`epoch=...-step=...`)
3. checkpoint metadata `global_step` fallback

## 8) Where to Experiment and Change Values

This is the most important section for strategy changes.

| What to change | File and symbol | Effect |
|---|---|---|
| Active preset | `tutorial_train_single_gpu.py` -> `PRESET_NAME` (env `PRESET`) | selects full training strategy |
| Hardware profile | `tutorial_train_single_gpu.py` -> `TRAINING_TARGET` (env) | precision, batch, grad accumulation, workers |
| Steps | `tutorial_train_single_gpu.py` -> per-preset `max_steps` or env `MAX_STEPS` | training duration per preset |
| Base LR | `tutorial_train_single_gpu.py` -> preset `lr` | learning speed/stability |
| Freeze/unfreeze decoder | `tutorial_train_single_gpu.py` -> `sd_locked` | whether UNet decoder is trainable |
| How many decoder blocks | `tutorial_train_single_gpu.py` -> `unlock_last_n` | amount of decoder adaptation |
| Train mask CN | `tutorial_train_single_gpu.py` -> `train_mask_cn` | include/exclude mask CN gradients |
| Train image CN | `tutorial_train_single_gpu.py` -> `train_image_cn` | include/exclude image CN gradients |
| Image loss weight | `tutorial_train_single_gpu.py` -> `image_loss` | strength of image branch supervision |
| Distillation weight | `tutorial_train_single_gpu.py` -> `distill_loss` | anatomy-aware transfer strength |
| Decoder LR scale | `tutorial_train_single_gpu.py` -> `decoder_lr_scale` | decoder LR relative to base LR |
| Branch mixing | `tutorial_train_single_gpu.py` -> `control_weight_mask`, `control_weight_image` | relative control influence |
| Dataset path | `tutorial_train_single_gpu.py` -> `DATA_ROOT` | which prompt file is trained |
| Resume checkpoint | env `RESUME_PATH` | resume from specific checkpoint |
| Log frequency | env `LOGGER_FREQ` | image/checkpoint cadence |
| Slurm resource limits | `slurm/*.sh` `#SBATCH` headers | runtime, memory, GPU/CPU allocation |
| Multi-preset order | `run_all.py` -> `DEFAULT_ORDER`, `PRESET_DEPS` | order and dependencies for all-preset runs |

## 9) Recommended Experiment Strategies

## Strategy 1: Safe baseline (new dataset)
1. run setup and data prep
2. train `scratch` first
3. optionally continue with `scratch_unlocked`
4. then try `scratch_stage2`

Commands:
```bash
PRESET=scratch sbatch slurm/train.sh
PRESET=scratch_unlocked sbatch slurm/train.sh
PRESET=scratch_stage2 sbatch slurm/train.sh
```

## Strategy 2: Transfer-first (faster domain adaptation)
1. start from `polyp_transfer`
2. optionally `polyp_unlocked`
3. phase-2 path: `polyp_stage2` or `polyp_stage2_from_unlocked`

Commands:
```bash
PRESET=polyp_transfer sbatch slurm/train.sh
PRESET=polyp_unlocked sbatch slurm/train.sh
PRESET=polyp_stage2 sbatch slurm/train.sh
```

## Strategy 3: Paper-like dual-CN training
Use one of:
- `paper_faithful_polyp`
- `paper_faithful_scratch`

This enables both ControlNets and equalized losses with decoders locked.

Command:
```bash
PRESET=paper_faithful_polyp sbatch slurm/train.sh
```

## 10) Command Cookbook (Slurm + Local)

### Setup
```bash
# local
uv run python setup_adc.py

# slurm
sbatch slurm/setup.sh
```

### Data prep
```bash
# pre-split dataset
uv run python prepare_liver_data.py --src ./data/dataset --out ./data --pre-split --dsad --prompt "a laparoscopic image of the liver"
```

### Training
```bash
# one preset
PRESET=scratch sbatch slurm/train.sh

# all presets in one job
bash slurm/train_all.sh

# all presets as dependent jobs
bash slurm/train_all.sh --split

# strategy sweeps (recommended for experiment batches)
bash slurm/submit_experiments.sh --profile quick
bash slurm/submit_experiments.sh --profile transfer
bash slurm/submit_experiments.sh --profile full
```

### Monitoring and control
```bash
# queue status
squeue -u $USER

# job details
scontrol show job <JOB_ID>

# watch logs
tail -f adc_train_<JOB_ID>.out
tail -f adc_train_<JOB_ID>.err

# cancel one
scancel <JOB_ID>

# cancel all yours
scancel -u $USER

# accounting/history
sacct -j <JOB_ID> --format=JobID,JobName,State,Elapsed,ExitCode
```

### Resume training
```bash
PRESET=scratch RESUME_PATH=./runs/scratch/version_0/checkpoints/last.ckpt sbatch slurm/train.sh
```

### Inference
```bash
CKPT_PATH=./runs/scratch/version_0/checkpoints/last.ckpt sbatch slurm/infer.sh
```

### Evaluation
```bash
uv run python evaluate_adc.py --real data/train/images --generated generated_results/local_test/images --out eval_results.json
```

## 11) Runtime and Logging Behavior

From `tutorial_train_single_gpu.py`:
- automatic 2-step sanity check before main training if not resuming,
- main trainer then runs to `max_steps`,
- checkpoint callback:
  - periodic save every `LOGGER_FREQ * 5` steps,
  - always keeps `last.ckpt`,
- logger frequency auto-scales from effective batch size if `LOGGER_FREQ=0`.
- logs metrics with CSV logger (for run_all/analyze_runs compatibility).
- also logs to TensorBoard when backend is available.

TensorBoard usage:
```bash
tensorboard --logdir runs --port 6006
```

From `train.sh`:
- after training, prints per-run disk/checkpoint/image summary,
- if `runs/training_report.md` exists, prints first 25 lines.

From `run_all.py`:
- runs each preset in a subprocess (cleaner GPU memory handling across presets),
- skips presets already complete,
- runs `analyze_runs.py` at the end.

## 12) Troubleshooting and Pitfalls

## Setup failures with exit code 127
Meaning:
- command not found in job environment.

Now mitigated in `slurm/setup.sh` by installer fallbacks (`curl`/`wget`/`python3 -m pip`).

## OOM in phase-2 presets
Cause:
- extra image branch and trainable components increase memory.

Mitigations:
- keep workstation phase-2 settings (`batch=1`, `grad_accum=4`),
- lower `MAX_STEPS` for initial smoke tests,
- check memory budget with:
```bash
uv run python vram_calculator.py polyp_stage2 --target workstation
```

## Missing source checkpoint for chained preset
If preset `ckpt_path` starts with `$`, the source preset must exist and have a checkpoint.

Example:
- `scratch_unlocked` requires a `scratch` checkpoint.

Use `run_all.py` or train dependencies first.

## Noisy FID with very small sample count
`evaluate_adc.py` notes FID needs larger sample sets for meaningful interpretation.

## Prompt file lint issues
If editor complains about JSON format, ensure files are valid JSON arrays.
Current pipeline writes JSON arrays; loaders still accept legacy JSONL.

## 13) Known Documentation Mismatches

Current code differs from some README sections:
- Slurm script names are `slurm/setup.sh`, `slurm/train.sh`, `slurm/infer.sh` (not `slurm/slurm_*.sh`).
- Current prompt files are standard JSON arrays, not JSONL-only.
- `slurm/train_all.sh --split` default list omits paper-faithful presets unless specified.

When in doubt, trust code in:
- `tutorial_train_single_gpu.py`
- `run_all.py`
- `slurm/*.sh`

## 14) Fast Start Recipes

### Recipe A: First successful training quickly
```bash
sbatch slurm/setup.sh
PRESET=scratch MAX_STEPS=2000 sbatch slurm/train.sh
CKPT_PATH=./runs/scratch/version_0/checkpoints/last.ckpt sbatch slurm/infer.sh
```

### Recipe B: Full sequential curriculum
```bash
sbatch slurm/setup.sh
bash slurm/train_all.sh
```

### Recipe C: Controlled split submissions
```bash
bash slurm/train_all.sh --split scratch polyp_transfer polyp_stage2
```

---

If you want this guide extended with your lab-specific defaults (paths, partition, QoS, expected runtime per preset on your exact GPUs), add a short appendix at the bottom and keep this file as the canonical base manual.
