#!/bin/bash

# -------------------------
# [1] Activate Environment
# -------------------------
if [ ! -d ".venv" ]; then
    echo "[ERROR] .venv not found"
    exit 1
fi

source .venv/bin/activate

tests=("test1" "test2" "test3" "test4" "test5")

# -------------------------
# [2] 1D POSTPROCESSING
# -------------------------
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


# -------------------------
# [3] 2D POSTPROCESSING
# -------------------------
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