#!/bin/bash
# One-time setup: install deps, download weights, create control checkpoint.
# Usage: sbatch slurm/setup.sh
#SBATCH --partition=workstations
#SBATCH --qos=students_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=adc_setup
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s-mtschi@haw-landshut.de
set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/ADC"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv &>/dev/null; then
    echo "uv not found. Attempting installation..."
    if command -v curl &>/dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | bash
    elif command -v wget &>/dev/null; then
        wget -qO- https://astral.sh/uv/install.sh | bash
    elif command -v python3 &>/dev/null; then
        python3 -m pip install --user uv
    else
        echo "ERROR: Could not install uv (missing curl, wget, and python3)."
        exit 127
    fi
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
fi

if ! command -v uv &>/dev/null; then
    echo "ERROR: uv is still not available after installation attempt."
    exit 127
fi

if command -v python3 &>/dev/null; then
    uv run python3 setup_adc.py "$@"
else
    uv run python setup_adc.py "$@"
fi

echo "Setup done. Next: sbatch slurm/train.sh"
