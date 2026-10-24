#!/usr/bin/env bash
#test DeepOPF
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

# -------- Defaults  --------
CASE="case14"
GEN_DATA_FLAG=""          # only set if user passes --gen_data
WHICH_MODEL="baseline"   # baseline or augmented
WEIGHTS="src/baselines/DeepOPF/trained_model.pth"
BATCH=3072
DEVICE="cpu"

# -------- Parse user arguments  --------
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
        --gen_data)
            GEN_DATA_FLAG="--gen_data"
            shift 1
            ;;
        --batch)
            BATCH="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--case CASE] [--gen_data] [--weights PATH] [--batch N] [--device DEVICE]"
            echo "Runs python -m src.baselines.DeepOPF.test_model for all which_data values in WHICH_DATA_LIST."
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

# -------- Loop over all which_data values --------
for WHICH_DATA in "${WHICH_DATA_LIST[@]}"; do
    echo "=== Running DeepOPF test for which_data='${WHICH_DATA},  which_model='${WHICH_MODEL}' ==="

    python -m src.baselines.DeepOPF.test_model \
        --case "$CASE" \
        --which_model "$WHICH_MODEL" \
        $GEN_DATA_FLAG \
        --which_data "$WHICH_DATA" \
        --batch "$BATCH" \
        --device "$DEVICE"
done