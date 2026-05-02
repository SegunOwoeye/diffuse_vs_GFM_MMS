#!/bin/bash

# -------------------------
# [1] Activate Environment
# -------------------------
if [ ! -d ".venv" ]; then
    echo "[ERROR] .venv not found"
    exit 1
fi

source .venv/bin/activate

echo "[COMPARE] Running GFM vs DIM validation"

python src/graphing/plot_gfm_dim_1d.py \
    --convergence-test test2 \
    --overlay-n 200 \
    --include-convergence-overlay \
    --output-dir data/plots/1d_GFM_DIM_validation

echo "[COMPARE] Completed"
echo "-------------------------"

deactivate