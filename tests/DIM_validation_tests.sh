#!/bin/bash

cores=6

# -------------------------
# [1.1] Compile in 1D - Fedwik 1999 Tests A-D
# -------------------------
g++ -std=c++17 -O2 -fopenmp -I. -DAPP_DIM=1 src/app/multimaterial_main.cpp -o mm_main_1d || exit 1
# [1.2] Tests
tests_1d=("test1" "test2" "test3" "test4" "test5")

# [1.3] Run CPP Solver
for t in "${tests_1d[@]}"; do
    echo "Running Fedwik $t 1D solver"
    OMP_NUM_THREADS=$cores OMP_SCHEDULE=dynamic ./mm_main_1d configs/DIM/MM_1D_validation/$t.txt || { echo "Solver failed"; continue; }
    

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
for t in "${tests_1d[@]}"; do
    echo "Running Fedwik $t Postprocessing in 1D"

    case $t in
        test1) out="dim_FedkiwA" ;;
        test2) out="dim_FedkiwB" ;;
        test3) out="dim_FedkiwC" ;;
        test4) out="dim_FedkiwD1" ;;
        test5) out="dim_FedkiwD2" ;;
        *) echo "Unknown test: $t"; continue ;;
    esac
    
    python src/graphing/plot_1d.py dim/MM_1D_validation/$out || { echo "Plot failed"; continue; }
    python src/graphing/compute_l1.py dim/MM_1D_validation/$out || { echo "L1 failed"; continue; }

    echo "Completed Fedwik $t Postprocessing in 1D"
    echo "-------------------------"
done

# [1.4.3] Deactivate Environment
deactivate
echo "All 1D post-processing for Fedwik tests are complete."

# [2] 1D GFM Vs DIM validation
# [2.1] Activate Virtual Environment
if [ ! -d ".venv" ]; then
    echo "[ERROR] .venv not found. Create the virtual environment first."
    exit 1
fi
source .venv/bin/activate

# [2.2] Run
echo "Running 1D Validation and Comparison tests for GFM and DIM"

python src/graphing/plot_gfm_dim_1d.py --convergence-test test2 --overlay-n 200 --include-convergence-overlay --output-dir data/plots/1d_GFM_DIM_validation || { echo "Plot failed"; continue; }

echo "Completed 1D Validation and Comparison Tests for GFM and DIM"
echo "-------------------------"

# [2.3] Deactivate Environment
deactivate
echo "All 1D Validation and Comparison tests are complete."

#############################################################################
# 2D
#############################################################################
g++ -std=c++17 -O2 -fopenmp -I. -DAPP_DIM=2 src/app/multimaterial_main.cpp -o mm_main_2d || exit 1
# [1.2] Tests
tests_2d=("test1" "test2" "test3" "test4" "test5" "test6")

# [1.3] Run CPP Solver
for t in "${tests_2d[@]}"; do
    echo "Running Fedwik $t 2D solver"
    OMP_NUM_THREADS=$cores OMP_SCHEDULE=dynamic ./mm_main_2d configs/DIM/MM_2D_validation/$t.txt || { echo "Solver failed"; continue; }
    

    echo "Completed Fedwik $t 2D solver"
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
for t in "${tests_2d[@]}"; do
    echo "Running Fedwik $t Postprocessing in 2D"

    case $t in
        test1) out="dim_FedkiwA" ;;
        test2) out="dim_FedkiwB" ;;
        test3) out="dim_FedkiwC" ;;
        test4) out="dim_FedkiwD1" ;;
        test5) out="dim_FedkiwD2" ;;
        test6) out="dim_helium_bubble_2d" ;;
        *) echo "Unknown test: $t"; continue ;;
    esac
    
    if [[ "$t" == "test6" ]]; then
        python src/graphing/plot_multid.py --schlieren dim/MM_2D_validation/$out || { echo "Plot failed"; continue; }
    else
        python src/graphing/plot_multid.py dim/MM_2D_validation/$out || { echo "Plot failed"; continue; }
        python src/graphing/compute_l1.py dim/MM_2D_validation/$out || { echo "L1 failed"; continue; }
    fi

    echo "Completed Fedwik $t Postprocessing in 2D"
    echo "-------------------------"
done

# [1.4.3] Deactivate Environment
deactivate
echo "All 2D post-processing for Fedwik tests are complete."


# SCALING STUDY






# RUN Commands
# cd diffuse_vs_GFM_MMS
# chmod +x tests/DIM_validation_tests.sh
# source .venv/bin/activate
# ./tests/DIM_validation_tests.sh

