#!/bin/bash
# Submit multi-strategy ADC experiments on Slurm with dependency-aware chaining.
#
# Usage examples:
#   bash slurm/submit_experiments.sh --profile quick
#   bash slurm/submit_experiments.sh --profile full
#   bash slurm/submit_experiments.sh --profile transfer --max-steps 8000
#   bash slurm/submit_experiments.sh --profile full --dry-run

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_SCRIPT="${ROOT_DIR}/slurm/train.sh"

PROFILE="full"
DATA_ROOT="./data/train/prompt.json"
GLOBAL_MAX_STEPS=""
AUTO_RESUME="0"
DRY_RUN=0
STAMP="$(date +%Y%m%d_%H%M%S)"

# Track job ids by logical experiment name for dependency chaining.
declare -A JOB_IDS
ORDER=()

usage() {
    cat <<'EOF'
Submit multi-strategy ADC experiments on Slurm.

Options:
  --profile <quick|full|transfer>  Experiment set to submit (default: full)
  --data-root <path>               Prompt file for training (default: ./data/train/prompt.json)
  --max-steps <int>                Override MAX_STEPS for all submitted jobs
  --dry-run                        Print planned sbatch commands only
  -h, --help                       Show this help

Profiles:
  quick     Fast smoke suite across best strategy families
  transfer  Transfer-focused strategy branch
  full      Broad strategy coverage (scratch, transfer, unlocked, stage2, paper-faithful)
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile)
            [[ $# -ge 2 ]] || die "--profile requires a value"
            PROFILE="$2"
            shift 2
            ;;
        --data-root)
            [[ $# -ge 2 ]] || die "--data-root requires a value"
            DATA_ROOT="$2"
            shift 2
            ;;
        --max-steps)
            [[ $# -ge 2 ]] || die "--max-steps requires a value"
            GLOBAL_MAX_STEPS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "Unknown argument: $1"
            ;;
    esac
done

case "$PROFILE" in
    quick|full|transfer)
        ;;
    *)
        die "Unknown --profile '$PROFILE' (expected quick|full|transfer)"
        ;;
esac

if [[ "$DATA_ROOT" = /* ]]; then
    DATA_ROOT_ABS="$DATA_ROOT"
else
    DATA_ROOT_ABS="${ROOT_DIR}/${DATA_ROOT#./}"
fi

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    die "Training script not found at $TRAIN_SCRIPT"
fi

if [[ $DRY_RUN -eq 0 && ! -f "$DATA_ROOT_ABS" ]]; then
    die "DATA_ROOT does not exist: $DATA_ROOT_ABS"
fi

if [[ -n "$GLOBAL_MAX_STEPS" && ! "$GLOBAL_MAX_STEPS" =~ ^[0-9]+$ ]]; then
    die "--max-steps must be an integer"
fi

submit_experiment() {
    local name="$1"
    local preset="$2"
    local depends_on="${3:-}"
    local profile_steps="${4:-}"
    local extra_exports="${5:-}"

    local run_tag="${STAMP}_${PROFILE}_${name}"
    local max_steps="$profile_steps"
    if [[ -n "$GLOBAL_MAX_STEPS" ]]; then
        max_steps="$GLOBAL_MAX_STEPS"
    fi

    local export_vars="ALL,PRESET=${preset},RUN_TAG=${run_tag},DATA_ROOT=${DATA_ROOT_ABS},AUTO_RESUME=${AUTO_RESUME}"
    if [[ -n "$max_steps" ]]; then
        export_vars+=",MAX_STEPS=${max_steps}"
    fi
    if [[ -n "$extra_exports" ]]; then
        export_vars+=",${extra_exports}"
    fi

    local dep_job=""
    if [[ -n "$depends_on" ]]; then
        dep_job="${JOB_IDS[$depends_on]:-}"
        [[ -n "$dep_job" ]] || die "Dependency '$depends_on' has no job id"
    fi

    ORDER+=("$name")

    if [[ $DRY_RUN -eq 1 ]]; then
        if [[ -n "$depends_on" ]]; then
            echo "[DRY] $name -> preset=$preset after=$depends_on($dep_job)"
        else
            echo "[DRY] $name -> preset=$preset"
        fi
        echo "      sbatch --export=${export_vars} ${TRAIN_SCRIPT}"
        JOB_IDS["$name"]="DRY_${name}"
        return
    fi

    local job_id
    if [[ -n "$dep_job" ]]; then
        job_id=$(sbatch --parsable --dependency=afterany:"$dep_job" --export="$export_vars" "$TRAIN_SCRIPT")
        echo "Submitted $name -> Job $job_id (after $depends_on: $dep_job)"
    else
        job_id=$(sbatch --parsable --export="$export_vars" "$TRAIN_SCRIPT")
        echo "Submitted $name -> Job $job_id"
    fi
    JOB_IDS["$name"]="$job_id"
}

echo "=== ADC Experiment Submission ==="
echo "Profile:     $PROFILE"
echo "Data root:   $DATA_ROOT_ABS"
echo "Max steps:   ${GLOBAL_MAX_STEPS:-profile defaults}"
echo "Auto resume: $AUTO_RESUME"
echo "Timestamp:   $STAMP"
echo

if [[ "$PROFILE" == "quick" ]]; then
    # Fast cross-family smoke tests.
    submit_experiment "scratch_base" "scratch" "" "5000"
    submit_experiment "scratch_unlocked" "scratch_unlocked" "scratch_base" "3000"

    submit_experiment "transfer_base" "polyp_transfer" "" "5000"
    submit_experiment "transfer_stage2" "polyp_stage2" "transfer_base" "4000"

    submit_experiment "paper_polyp" "paper_faithful_polyp" "" "8000"

elif [[ "$PROFILE" == "transfer" ]]; then
    # Transfer path with stage-2 variants.
    submit_experiment "transfer_base" "polyp_transfer" "" ""
    submit_experiment "transfer_unlocked" "polyp_unlocked" "transfer_base" ""

    submit_experiment "stage2_direct_default" "polyp_stage2" "transfer_base" ""
    submit_experiment "stage2_direct_high_distill" "polyp_stage2" "transfer_base" "" "DISTILL_LOSS_OVERRIDE=1.0,CONTROL_WEIGHT_IMAGE_OVERRIDE=0.5"

    submit_experiment "stage2_from_unlocked" "polyp_stage2_from_unlocked" "transfer_unlocked" ""

elif [[ "$PROFILE" == "full" ]]; then
    # Broad strategy coverage.
    submit_experiment "scratch_base" "scratch"
    submit_experiment "scratch_unlocked" "scratch_unlocked" "scratch_base"
    submit_experiment "scratch_stage2" "scratch_stage2" "scratch_unlocked"

    submit_experiment "transfer_base" "polyp_transfer"
    submit_experiment "transfer_unlocked" "polyp_unlocked" "transfer_base"
    submit_experiment "transfer_stage2_direct" "polyp_stage2" "transfer_base"
    submit_experiment "transfer_stage2_from_unlocked" "polyp_stage2_from_unlocked" "transfer_unlocked"

    submit_experiment "transfer_stage2_high_distill" "polyp_stage2" "transfer_base" "" "DISTILL_LOSS_OVERRIDE=1.0,CONTROL_WEIGHT_IMAGE_OVERRIDE=0.5"

    submit_experiment "paper_polyp" "paper_faithful_polyp"
    submit_experiment "paper_scratch" "paper_faithful_scratch"
fi

echo
echo "=== Submission Summary ==="
for name in "${ORDER[@]}"; do
    echo "  $name -> ${JOB_IDS[$name]}"
done

echo
if [[ $DRY_RUN -eq 1 ]]; then
    echo "Dry run completed: no jobs were submitted."
else
    echo "Monitor jobs with: squeue -u \$USER"
fi
