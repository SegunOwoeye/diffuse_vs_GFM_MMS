#!/bin/bash

# -------------------------
# [1] Activate Environment
# -------------------------
if [ ! -d ".venv" ]; then
    echo "[ERROR] .venv not found"
    exit 1
fi

source .venv/bin/activate

RUN_DIMS="${VALIDATION_DIMS:-all}"
tests=("test1" "test2" "test3" "test4" "test5")

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --dims|--dim) shift; RUN_DIMS="$1" ;;
        --dims=*|--dim=*) RUN_DIMS="${1#*=}" ;;
        *) echo "Unknown option: $1"; deactivate; exit 1 ;;
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
        *) echo "[ERROR] Unknown dimension selection: $RUN_DIMS"; deactivate; exit 1 ;;
    esac
}

# -------------------------
# [2] 1D POSTPROCESSING
# -------------------------
if dimension_enabled 1; then
    for t in "${tests[@]}"; do
        echo "[DIM 1D] Postprocessing $t"

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
    done

    echo "[DIM 1D] Postprocessing complete"
    echo "-------------------------"
fi


# -------------------------
# [3] 2D POSTPROCESSING
# -------------------------
if dimension_enabled 2; then
    for t in "${tests[@]}"; do
        echo "[DIM 2D] Postprocessing $t"

        case $t in
            test1) out="dim_FedkiwA" ;;
            test2) out="dim_FedkiwB" ;;
            test3) out="dim_FedkiwC" ;;
            test4) out="dim_FedkiwD1" ;;
            test5) out="dim_FedkiwD2" ;;
            *) echo "Unknown test: $t"; continue ;;
        esac

        python src/graphing/plot_multid.py dim/MM_2D_validation/$out || { echo "Plot failed"; continue; }
        python src/graphing/compute_l1.py dim/MM_2D_validation/$out || { echo "L1 failed"; continue; }
    done

    echo "[DIM 2D] Postprocessing complete"
    echo "-------------------------"
fi

# -------------------------
# [4] Cleanup
# -------------------------
deactivate

echo "[DIM] All postprocessing complete."


# RUN Commands
# cd diffuse_vs_DIM_MMS
# chmod +x tests/validation/2_MM_DIM_Tests/MM_DIM_graphing.sh
# source .venv/bin/activate
# ./tests/validation/2_MM_DIM_Tests/MM_DIM_graphing.sh
