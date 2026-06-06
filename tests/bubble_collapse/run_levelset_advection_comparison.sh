#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

FLOW_CFG="configs/GFM/bubble_collapse/comparison/helium_bubble_2d_flow_weno.txt"
NORMAL_CFG="configs/GFM/bubble_collapse/comparison/helium_bubble_2d_normal_speed_weno.txt"

FLOW_OUT="data/csv/gfm/bubble_levelset_advection_comparison/gfm_helium_bubble_2d_flow_weno"
NORMAL_OUT="data/csv/gfm/bubble_levelset_advection_comparison/gfm_helium_bubble_2d_normal_speed_weno"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc)}"

echo "[setup] Compiling 2D multimaterial solver"
g++ -std=c++17 -O2 -fopenmp -I. -DAPP_DIM=2 src/app/multimaterial_main.cpp -o mm_main_2d

echo "[setup] OpenMP threads: ${OMP_NUM_THREADS}"

echo "[run] GFM bubble collapse with level_set_advection = flow"
./mm_main_2d "$FLOW_CFG"

echo "[run] GFM bubble collapse with level_set_advection = normal_speed"
./mm_main_2d "$NORMAL_CFG"

echo "[plot] Flow-advection schlieren and pressure/phi contours"
python src/graphing/plot_multid.py --schlieren --pressure-contours "$FLOW_OUT"

echo "[plot] Normal-speed-advection schlieren and pressure/phi contours"
python src/graphing/plot_multid.py --schlieren --pressure-contours "$NORMAL_OUT"

echo "[done] Outputs:"
echo "  $FLOW_OUT"
echo "  $NORMAL_OUT"
