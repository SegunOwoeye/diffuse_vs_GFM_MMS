#!/bin/bash

cores=6
RUN_DIMS="${VALIDATION_DIMS:-all}"
tests=("test1" "test2" "test3" "test4" "test5")

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --dims|--dim) shift; RUN_DIMS="$1" ;;
        --dims=*|--dim=*) RUN_DIMS="${1#*=}" ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

dimension_enabled() {
    local dim="$1"
    local dims

    dims=$(echo "$RUN_DIMS" | tr '[:upper:]' '[:lower:]' | tr -d ' ')

    case "$dims" in
        all|both|1,2|2,1) return 0 ;;
        1|1d) [[ "$dim" == "1" ]] ;;
        2|2d) [[ "$dim" == "2" ]] ;;
        *) echo "[ERROR] Unknown dimension selection: $RUN_DIMS"; exit 1 ;;
    esac
}

# -------------------------
# [1] 1D SIMULATIONS
# -------------------------
if dimension_enabled 1; then
    echo "[GFM 1D] Compiling..."
    g++ -std=c++17 -O2 -fopenmp -I. -DAPP_DIM=1 src/app/multimaterial_main.cpp -o mm_main_1d

    for t in "${tests[@]}"; do
        echo "[GFM 1D] Running $t"
        OMP_NUM_THREADS=$cores OMP_SCHEDULE=dynamic \
        ./mm_main_1d configs/GFM/MM_1D_validation/$t.txt || { echo "Solver failed"; continue; }
    done

    echo "[GFM 1D] Completed"
    echo "-------------------------"
fi


# -------------------------
# [2] 2D SIMULATIONS
# -------------------------
if dimension_enabled 2; then
    echo "[GFM 2D] Compiling..."
    g++ -std=c++17 -O2 -fopenmp -I. -DAPP_DIM=2 src/app/multimaterial_main.cpp -o mm_main_2d

    for t in "${tests[@]}"; do
        echo "[GFM 2D] Running $t"
        OMP_NUM_THREADS=$cores OMP_SCHEDULE=dynamic \
        ./mm_main_2d configs/GFM/MM_2D_validation/$t.txt || { echo "Solver failed"; continue; }
    done

    echo "[GFM 2D] Completed"
    echo "-------------------------"
fi

echo "[GFM] All simulations complete."


# RUN Commands
# cd diffuse_vs_GFM_MMS
# chmod +x tests/validation/2_MM_GFM_Tests/MM_GFM_simulation.sh
# source .venv/bin/activate
# ./tests/validation/2_MM_GFM_Tests/MM_GFM_simulation.sh
