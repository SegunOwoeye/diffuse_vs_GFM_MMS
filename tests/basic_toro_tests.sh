#!/bin/bash

# -------------------------
# [1.1] Compile in 1D
# -------------------------
g++ -std=c++17 -O2 -I. -DAPP_DIM=1 src/app/sm_main.cpp -o sm_main_1d || exit 1

# [1.2] Tests
tests=("test1" "test2" "test3" "test4" "test5")

# [1.3] Run CPP Solver
for t in "${tests[@]}"; do
    echo "Running Toro $t 1D solver"

    ./sm_main_1d configs/toro/$t.txt || { echo "Solver failed"; continue; }
    

    echo "Completed Toro $t 1D solver"
    echo "-------------------------"
done

# [1.4] Results Post Processing
# [1.4.1]  Activate Virtual Environment
if [ ! -d ".venv" ]; then
    echo "[ERROR] .venv not found. Create the virtual environment first."
    exit 1
fi
source .venv/bin/activate

# [1.4.2] Run For loop
for t in "${tests[@]}"; do
    echo "Running Toro $t Postprocessing in 1D"

    case $t in
        test1) out="toro1" ;;
        test2) out="toro2" ;;
        test3) out="toro3" ;;
        test4) out="toro4" ;;
        test5) out="toro5" ;;
        *) echo "Unknown test: $t"; continue ;;
    esac

    python src/graphing/plot_1d.py toro/$out || { echo "Plot failed"; continue; }
    python src/graphing/compute_l1.py toro/$out || { echo "L1 failed"; continue; }

    echo "Completed Toro $t Postprocessing in 1D"
    echo "-------------------------"
done

# [1.4.3] Deactivate Environment
deactivate
echo "All 1D post-processing for Toro tests are complete."


# -------------------------
# [2.1] Compile in 2D
# -------------------------
g++ -std=c++17 -O2 -I. -DAPP_DIM=2 src/app/sm_main.cpp -o sm_main_2d || exit 1

# [2.2] Tests
tests=("explosion1")

# [2.3] Run CPP Solver
for t in "${tests[@]}"; do
    echo "Running $t 2D solver"

    ./sm_main_2d configs/toro/$t.txt || { echo "Solver failed"; continue; }
    

    echo "Completed Toro $t 2D solver"
    echo "-------------------------"
done

# [2.4] Results Post Processing
# [2.4.1]  Activate Virtual Environment
if [ ! -d ".venv" ]; then
    echo "[ERROR] .venv not found. Create the virtual environment first."
    exit 1
fi
source .venv/bin/activate

# [2.4.2] Run For loop
for t in "${tests[@]}"; do
    echo "Running Toro $t Postprocessing in 2D"

    case $t in
        test1) out="explosion1" ;;
        *) echo "Unknown test: $t"; continue ;;
    esac

    python src/graphing/plot_multid.py toro/$out || { echo "Plot failed"; continue; }
    python src/graphing/compute_l1.py toro/$out || { echo "L1 failed"; continue; }

    echo "Completed Toro $t Postprocessing in 2D"
    echo "-------------------------"
done

# [2.4.3] Deactivate Environment
deactivate
echo "All 2D post-processing for Toro tests are complete."


# -------------------------
# [3.1] Compile in 3D
# -------------------------
g++ -std=c++17 -O2 -I. -DAPP_DIM=3 src/app/sm_main.cpp -o sm_main_3d || exit 1

# [3.2] Tests
tests=("explosion2")

# [3.3] Run CPP Solver
for t in "${tests[@]}"; do
    echo "Running $t 3D solver"

    ./sm_main_3d configs/toro/$t.txt || { echo "Solver failed"; continue; }
    

    echo "Completed Toro $t 3D solver"
    echo "-------------------------"
done

# [3.4] Results Post Processing
# [3.4.1]  Activate Virtual Environment
if [ ! -d ".venv" ]; then
    echo "[ERROR] .venv not found. Create the virtual environment first."
    exit 1
fi
source .venv/bin/activate

# [3.4.2] Run For loop
for t in "${tests[@]}"; do
    echo "Running Toro $t Postprocessing in 3D"

    case $t in
        test1) out="explosion2" ;;
        *) echo "Unknown test: $t"; continue ;;
    esac

    python src/graphing/plot_multid.py toro/$out || { echo "Plot failed"; continue; }
    python src/graphing/compute_l1.py toro/$out || { echo "L1 failed"; continue; }

    echo "Completed Toro $t Postprocessing in 3D"
    echo "-------------------------"
done

# [3.4.3] Deactivate Environment
deactivate
echo "All 3D post-processing for Toro tests are complete."





# RUN Commands
# cd diffuse_vs_GFM_MMS
# chmod +x tests/basic_toro_tests.sh
# source .venv/bin/activate
# ./tests/basic_toro_tests.sh

