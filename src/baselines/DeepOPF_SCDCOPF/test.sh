#!/usr/bin/env bash
# Test DeepOPF_SCDCOPF on all test scenarios for one case
set -euo pipefail

WHICH_DATA_LIST=(
    "in_distribution"
    "load_extreme_high"
    "load_extreme_low"
    "fuel_price_up"
    "fuel_price_down"
    "geo_shift"
    "extreme_weather"
    "islanded_operation"
    "nk_contingencies"
    "reactive_scarcity"
)

# -------- Defaults --------
CASE="case14"
WHICH_MODEL="baseline"     # baseline or augmented
BATCH=3072
DEVICE="cpu"
HIDDEN=8
MAX_SAMPLES_TEST=1000
NO_PROJECTION_FLAG=""

# -------- Parse user arguments --------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --case)
            CASE="$2"
            shift 2
            ;;
        --which_model)
            WHICH_MODEL="$2"
            shift 2
            ;;
        --batch)
            BATCH="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --hidden)
            HIDDEN="$2"
            shift 2
            ;;
        --max_samples_test)
            MAX_SAMPLES_TEST="$2"
            shift 2
            ;;
        --no_projection)
            NO_PROJECTION_FLAG="--no_projection"
            shift 1
            ;;
        -h|--help)
            echo "Usage: $0 [--case CASE] [--which_model baseline|augmented] [--batch N] [--device DEVICE] [--hidden N] [--max_samples_test N] [--no_projection]"
            echo "Runs python -m src.baselines.DeepOPF_SCDCOPF.test_model for all scenario names in WHICH_DATA_LIST."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage."
            exit 1
            ;;
    esac
done

# -------- Validate which_model --------
if [[ "$WHICH_MODEL" != "baseline" && "$WHICH_MODEL" != "augmented" ]]; then
    echo "Error: --which_model must be either 'baseline' or 'augmented'"
    exit 1
fi

# -------- Run all scenarios --------
for WHICH_DATA in "${WHICH_DATA_LIST[@]}"; do
    echo "=== Running DeepOPF_SCDCOPF test: case='${CASE}', which_model='${WHICH_MODEL}', which_data='${WHICH_DATA}' ==="

    python -m src.baselines.DeepOPF_SCDCOPF.test_model \
        --case "$CASE" \
        --which_model "$WHICH_MODEL" \
        --which_data "$WHICH_DATA" \
        --max_samples_test "$MAX_SAMPLES_TEST" \
        --batch "$BATCH" \
        --device "$DEVICE" \
        --hidden "$HIDDEN" \
        $NO_PROJECTION_FLAG
done