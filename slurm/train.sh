#!/bin/bash
# Train ADC on liver data. Requires setup.sh to have been run first.
#
# Usage:
#   PRESET=scratch sbatch slurm/train.sh          # single preset
#   PRESET=all sbatch slurm/train.sh              # all presets, autodetect completion
#   sbatch slurm/train.sh                         # default: all presets
#
#SBATCH --partition=workstations
#SBATCH --qos=students_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --job-name=adc_train
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s-mtschi@haw-landshut.de
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$HOME/ADC"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if [[ -n "${UV_BIN:-}" && -x "${UV_BIN}" ]]; then
    UV_RUNNER="$UV_BIN"
elif command -v uv &>/dev/null; then
    UV_RUNNER="$(command -v uv)"
elif [[ -x "$HOME/.local/bin/uv" ]]; then
    UV_RUNNER="$HOME/.local/bin/uv"
elif [[ -x "$HOME/.cargo/bin/uv" ]]; then
    UV_RUNNER="$HOME/.cargo/bin/uv"
else
    echo "ERROR: uv not found in batch environment."
    echo "Tried PATH, $HOME/.local/bin/uv and $HOME/.cargo/bin/uv."
    echo "If setup was run under a different account, rerun: sbatch slurm/setup.sh"
    exit 127
fi

if command -v python3 &>/dev/null; then
    PY_BIN=python3
else
    PY_BIN=python
fi

START_TIME=$(date)
EXIT_CODE=0

# ── Preset selection ──
export PRESET=${PRESET:-all}

if command -v nvidia-smi &>/dev/null; then
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
else
    GPU_NAME="GPU unavailable"
fi

echo "Job $SLURM_JOB_ID on $(hostname) — ${GPU_NAME:-GPU unavailable}"
echo "Preset: $PRESET  |  Starting at $START_TIME"

if [[ "$PRESET" == "all" ]]; then
    TRAINING_TARGET=workstation "$UV_RUNNER" run "$PY_BIN" run_all.py || EXIT_CODE=$?
else
    TRAINING_TARGET=workstation "$UV_RUNNER" run "$PY_BIN" tutorial_train_single_gpu.py || EXIT_CODE=$?
fi

# ── Job summary (appears in .out file and SLURM email) ──
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  JOB SUMMARY — $SLURM_JOB_ID"
echo "═══════════════════════════════════════════════════════════"
echo "  Node:     $(hostname)"
echo "  GPU:      $(nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo 'unavailable')"
echo "  Started:  $START_TIME"
echo "  Finished: $(date)"
echo "  Exit:     $EXIT_CODE"
echo ""

# Show disk usage per preset
echo "  Disk usage:"
for d in runs/*/; do
    if [[ -d "$d" ]]; then
        size=$(du -sh "$d" 2>/dev/null | cut -f1)
        ckpts=$(find "$d" -name "*.ckpt" 2>/dev/null | wc -l | tr -d ' ')
        imgs=$(find "$d" -path "*/image_log/train/*.png" 2>/dev/null | wc -l | tr -d ' ')
        echo "    $(basename "$d"): ${size} (${ckpts} ckpts, ${imgs} images)"
    fi
done
echo ""

# Show training report summary if available
if [[ -f runs/training_report.md ]]; then
    echo "  Training report (first 25 lines):"
    head -25 runs/training_report.md | sed 's/^/    /'
fi

echo "═══════════════════════════════════════════════════════════"
exit $EXIT_CODE
