#!/bin/bash

# -------------------------
# [1] Activate Environment
# -------------------------
if [ ! -d ".venv" ]; then
    echo "[ERROR] .venv not found"
    exit 1
fi

source .venv/bin/activate

echo "[COMPARE] Running rGFM vs Allaire 5eq validation"
echo "[COMPARE] Writing method-only validation figures; Fedkiw reference images are kept separate."

echo "[COMPARE] Writing 1D Fedkiw comparison figures"
python src/graphing/plot_gfm_dim_1d.py \
    --convergence-test test2 \
    --overlay-n 400 \
    --include-convergence-overlay \
    --output-dir data/plots/1d_rGFM_Allaire5eq_validation

echo "[COMPARE] Writing 2D Fedkiw reduction figures"
python src/graphing/plot_gfm_dim_2d.py \
    --overlay-n 400 \
    --output-dir data/plots/2d_rGFM_Allaire5eq_validation

echo "[COMPARE] Completed"
echo "-------------------------"

deactivate
