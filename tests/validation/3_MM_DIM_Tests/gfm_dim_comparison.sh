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

if compgen -G "data/csv/gfm/MM_1D_validation/gfm_FedkiwA/*.csv" > /dev/null &&
   compgen -G "data/csv/dim/MM_1D_validation/dim_FedkiwA/*.csv" > /dev/null; then
    echo "[COMPARE] Writing 1D Fedkiw comparison figures"
    python src/graphing/plot_gfm_dim_1d.py \
        --convergence-test test2 \
        --overlay-n 400 \
        --include-convergence-overlay \
        --output-dir data/plots/1d_rGFM_Allaire5eq_validation
else
    echo "[COMPARE] Skipping 1D comparison; 1D CSVs are missing"
fi

if compgen -G "data/csv/gfm/MM_2D_validation/gfm_FedkiwA/*.csv" > /dev/null &&
   compgen -G "data/csv/dim/MM_2D_validation/dim_FedkiwA/*.csv" > /dev/null; then
    echo "[COMPARE] Writing 2D Fedkiw reduction figures"
    python src/graphing/plot_gfm_dim_2d.py \
        --overlay-n 400 \
        --output-dir data/plots/2d_rGFM_Allaire5eq_validation
else
    echo "[COMPARE] Skipping 2D comparison; grid-aligned 2D CSVs are missing"
fi

if compgen -G "data/csv/gfm/MM_2D_validation/gfm_FedkiwA45/*.csv" > /dev/null &&
   compgen -G "data/csv/dim/MM_2D_validation/dim_FedkiwA45/*.csv" > /dev/null; then
    echo "[COMPARE] Writing oblique45 Fedkiw reduction figures"
    python src/graphing/plot_gfm_dim_2d.py \
        --overlay-n 200 \
        --two-d-name-suffix 45 \
        --slices oblique45 \
        --output-dir data/plots/2d_rGFM_Allaire5eq_oblique45_validation
else
    echo "[COMPARE] Skipping oblique45 comparison; oblique 2D CSVs are missing"
fi

echo "[COMPARE] Completed"
echo "-------------------------"

deactivate
