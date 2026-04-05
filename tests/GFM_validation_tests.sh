#!/bin/bash

# -------------------------
# [1.1] Compile in 1D - Fedwik 1999 Tests A-D
# -------------------------
g++ -std=c++17 -O2 -I. -DAPP_DIM=1 src/app/multimaterial_main.cpp -o mm_main_1d || exit 1
# [1.2] Tests
tests=("test1" "test2" "test3" "test4" "test5")

# [1.3] Run CPP Solver
for t in "${tests[@]}"; do
    echo "Running Fedwik $t 1D solver"
    ./mm_main_1d configs/GFM/MM_1D_validation/$t.txt || { echo "Solver failed"; continue; }
    

    echo "Completed Fedwik $t 1D solver"
    echo "-------------------------"
done

# [1.4] Results Post Processing
# [1.4.1] Activate Virtual Environment
if [ ! -d ".venv" ]; then
    echo "[ERROR] .venv not found. Create the virtual environment first."
    exit 1
fi
source .venv/bin/activate

# [1.4.2] Run For loop
for t in "${tests[@]}"; do
    echo "Running Fedwik $t Postprocessing in 1D"

    case $t in
        test1) out="gfm_FedkiwA" ;;
        test2) out="gfm_FedkiwB" ;;
        test3) out="gfm_FedkiwC" ;;
        test4) out="gfm_FedkiwD1" ;;
        test5) out="gfm_FedkiwD2" ;;
        *) echo "Unknown test: $t"; continue ;;
    esac
    
    python src/graphing/plot_1d.py gfm/MM_1D_validation/$out || { echo "Plot failed"; continue; }
    python src/graphing/compute_l1.py gfm/MM_1D_validation/$out || { echo "L1 failed"; continue; }

    echo "Completed Fedwik $t Postprocessing in 1D"
    echo "-------------------------"
done

# [1.4.3] Deactivate Environment
deactivate
echo "All 1D post-processing for Fedwik tests are complete."





# RUN Commands
# cd diffuse_vs_GFM_MMS
# chmod +x tests/GFM_validation_tests.sh
# source .venv/bin/activate
# ./tests/GFM_validation_tests.sh

