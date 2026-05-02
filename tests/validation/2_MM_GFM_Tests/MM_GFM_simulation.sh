#!/bin/bash

cores=6

# -------------------------
# [1] 1D SIMULATIONS
# -------------------------
echo "[GFM 1D] Compiling..."
g++ -std=c++17 -O2 -fopenmp -I. -DAPP_DIM=1 src/app/multimaterial_main.cpp -o mm_main_1d

tests=("test1" "test2" "test3" "test4" "test5")

for t in "${tests[@]}"; do
    echo "[GFM 1D] Running $t"
    OMP_NUM_THREADS=$cores OMP_SCHEDULE=dynamic \
    ./mm_main_1d configs/GFM/MM_1D_validation/$t.txt || { echo "Solver failed"; continue; }
done

echo "[GFM 1D] Completed"
echo "-------------------------"


# -------------------------
# [2] 2D SIMULATIONS
# -------------------------
echo "[GFM 2D] Compiling..."
g++ -std=c++17 -O2 -fopenmp -I. -DAPP_DIM=2 src/app/multimaterial_main.cpp -o mm_main_2d

for t in "${tests[@]}"; do
    echo "[GFM 2D] Running $t"
    OMP_NUM_THREADS=$cores OMP_SCHEDULE=dynamic \
    ./mm_main_2d configs/GFM/MM_2D_validation/$t.txt || { echo "Solver failed"; continue; }
done

echo "[GFM 2D] Completed"
echo "-------------------------"

echo "[GFM] All simulations complete."


# RUN Commands
# cd diffuse_vs_GFM_MMS
# chmod +x tests/validation/2_MM_GFM_Tests/MM_GFM_simulation.sh
# source .venv/bin/activate
# ./tests/validation/2_MM_GFM_Tests/MM_GFM_simulation.sh