#!/usr/bin/env bash
# test OPF-DNN

set -euo pipefail

# ---- sanity check: make sure we're inside a conda env ----
if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "Please 'conda activate env_opf_dnn' before running this script."
    exit 1
fi

# ---- arguments ----
CASE="${1:-case14}"                  # case14 | case118
WHICH_MODEL="${2:-baseline}"         # baseline | augmented
RETRAIN_FLAG="${3:-}"                # retrain   (want to train the model again)

DATADIR="src/baselines/OPF_DNN/datasets/$CASE"

# ---- choose Julia script based on WHICH_MODEL ----
if [[ "$WHICH_MODEL" == "baseline" ]]; then
    JL_SCRIPT="src/baselines/OPF_DNN/opf-dnn.jl"
elif [[ "$WHICH_MODEL" == "augmented" ]]; then
    JL_SCRIPT="src/baselines/OPF_DNN/opf-dnn-augmented.jl"
else
    echo "Invalid WHICH_MODEL: $WHICH_MODEL"
    echo "Usage: $0 [case14|case118] [baseline|augmented] [retrain]"
    exit 1
fi

# ---- other args if needed ----
EXTRA_ARGS=()
if [[ "$RETRAIN_FLAG" == "retrain" ]]; then
    EXTRA_ARGS+=(--retrain)
fi

# ---- keep Python inside this conda env ----
export PYTHONNOUSERSITE=1
export PYTHONWARNINGS="ignore"
export CODES_ACOPF_ROOT="$(pwd)"

for f in "$DATADIR"/test_data_*.json; do
    export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6:$CONDA_PREFIX/lib/libgcc_s.so.1"

    julia --project=src/baselines/OPF_DNN \
        "$JL_SCRIPT" \
        --testdata "$(basename "$f")" \
        --netname "$CASE" \
        "${EXTRA_ARGS[@]}"
done