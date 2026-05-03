#!/bin/bash
# Run inference with a trained checkpoint.
# Usage: CKPT_PATH=./lightning_logs/.../last.ckpt sbatch slurm/infer.sh
#SBATCH --partition=workstations
#SBATCH --qos=students_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=adc_infer
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s-mtschi@haw-landshut.de
set -euo pipefail
export PYTHONUNBUFFERED=1
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

echo "Inference job $SLURM_JOB_ID — $(date)"
"$UV_RUNNER" run "$PY_BIN" tutorial_inference_local.py

echo "Done at $(date). Results: generated_results/"
