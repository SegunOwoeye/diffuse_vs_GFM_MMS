#!/bin/bash

cores=6
tests=("test1" "test2" "test3" "test4" "test5")

# -------------------------
# [1] 1D SIMULATIONS
# -------------------------
echo "[DIM 1D] Compiling..."
g++ -std=c++17 -O2 -fopenmp -I. -DAPP_DIM=1 src/app/multimaterial_main.cpp -o mm_main_1d

for t in "${tests[@]}"; do
    echo "[DIM 1D] Running $t"
    OMP_NUM_THREADS=$cores OMP_SCHEDULE=dynamic \
    ./mm_main_1d configs/DIM/MM_1D_validation/$t.txt || { echo "Solver failed"; continue; }
done

echo "[DIM 1D] Completed"
echo "-------------------------"


# -------------------------
# [2] 2D SIMULATIONS
# -------------------------
echo "[DIM 2D] Compiling..."
g++ -std=c++17 -O2 -fopenmp -I. -DAPP_DIM=2 src/app/multimaterial_main.cpp -o mm_main_2d

for t in "${tests[@]}"; do
    echo "[DIM 2D] Running $t"
    OMP_NUM_THREADS=$cores OMP_SCHEDULE=dynamic \
    ./mm_main_2d configs/DIM/MM_2D_validation/$t.txt || { echo "Solver failed"; continue; }
done

echo "[DIM 2D] Completed"
echo "-------------------------"

echo "[DIM] All simulations complete."


# RUN Commands
# cd diffuse_vs_DIM_MMS
# chmod +x tests/validation/2_MM_DIM_Tests/MM_DIM_simulation.sh
# source .venv/bin/activate
# ./tests/validation/2_MM_DIM_Tests/MM_DIM_simulation.sh