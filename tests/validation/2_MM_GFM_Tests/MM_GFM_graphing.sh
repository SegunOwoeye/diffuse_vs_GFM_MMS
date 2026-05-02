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
    echo "[GFM 1D] Postprocessing $t"

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
done

echo "[GFM 1D] Postprocessing complete"
echo "-------------------------"


# -------------------------
# [3] 2D POSTPROCESSING
# -------------------------
for t in "${tests[@]}"; do
    echo "[GFM 2D] Postprocessing $t"

    case $t in
        test1) out="gfm_FedkiwA" ;;
        test2) out="gfm_FedkiwB" ;;
        test3) out="gfm_FedkiwC" ;;
        test4) out="gfm_FedkiwD1" ;;
        test5) out="gfm_FedkiwD2" ;;
        *) echo "Unknown test: $t"; continue ;;
    esac

    python src/graphing/plot_multid.py gfm/MM_2D_validation/$out || { echo "Plot failed"; continue; }
    python src/graphing/compute_l1.py gfm/MM_2D_validation/$out || { echo "L1 failed"; continue; }
done

echo "[GFM 2D] Postprocessing complete"
echo "-------------------------"

# -------------------------
# [4] Cleanup
# -------------------------
deactivate

echo "[GFM] All postprocessing complete."


# RUN Commands
# cd diffuse_vs_GFM_MMS
# chmod +x tests/validation/2_MM_GFM_Tests/MM_GFM_graphing.sh
# source .venv/bin/activate
# ./tests/validation/2_MM_GFM_Tests/MM_GFM_graphing.sh