#!/bin/bash

# -------------------------
# [1] 1D SIMULATIONS
# -------------------------
echo "[1D] Compiling..."
g++ -std=c++17 -O2 -I. -DAPP_DIM=1 src/app/sm_main.cpp -o sm_main_1d

tests_1d=("test1" "test2" "test3" "test4" "test5")

for t in "${tests_1d[@]}"; do
    echo "[1D] Running Toro $t"
    ./sm_main_1d configs/toro/$t.txt
done

echo "[1D] Completed"
echo "-------------------------"


# -------------------------
# [2] 2D SIMULATIONS
# -------------------------
echo "[2D] Compiling..."
g++ -std=c++17 -O2 -I. -DAPP_DIM=2 src/app/sm_main.cpp -o sm_main_2d

tests_2d=("explosion1")

for t in "${tests_2d[@]}"; do
    echo "[2D] Running $t"
    ./sm_main_2d configs/toro/$t.txt
done

echo "[2D] Completed"
echo "-------------------------"


# -------------------------
# [3] 3D SIMULATIONS
# -------------------------
echo "[3D] Compiling..."
g++ -std=c++17 -O2 -I. -DAPP_DIM=3 src/app/sm_main.cpp -o sm_main_3d

tests_3d=("explosion2")

for t in "${tests_3d[@]}"; do
    echo "[3D] Running $t"
    ./sm_main_3d configs/toro/$t.txt
done

echo "[3D] Completed"
echo "-------------------------"

echo "All simulations complete."






# RUN Commands
# cd diffuse_vs_GFM_MMS
# chmod +x tests/validation/1_SM_Euler_Tests/SM_simulation.sh
# source .venv/bin/activate
# ./tests/validation/1_SM_Euler_Tests/SM_simulation.sh
