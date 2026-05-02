#!/bin/bash

# -------------------------
# [1] Activate Environment
# -------------------------
if [ ! -d ".venv" ]; then
    echo "[ERROR] .venv not found"
    exit 1
fi

source .venv/bin/activate


# -------------------------
# [2] 1D POSTPROCESSING
# -------------------------
tests_1d=("test1" "test2" "test3" "test4" "test5")

for t in "${tests_1d[@]}"; do
    echo "[1D] Postprocessing $t"

    case $t in
        test1) out="toro1" ;;
        test2) out="toro2" ;;
        test3) out="toro3" ;;
        test4) out="toro4" ;;
        test5) out="toro5" ;;
        *) echo "Unknown test: $t"; continue ;;
    esac

    python src/graphing/plot_1d.py toro/$out
    python src/graphing/compute_l1.py toro/$out
done

echo "[1D] Postprocessing complete"
echo "-------------------------"


# -------------------------
# [3] 2D POSTPROCESSING
# -------------------------
tests_2d=("explosion1")

for t in "${tests_2d[@]}"; do
    echo "[2D] Postprocessing $t"

    python src/graphing/plot_multid.py toro/$t
    python src/graphing/compute_l1.py toro/$t
done

echo "[2D] Postprocessing complete"
echo "-------------------------"


# -------------------------
# [4] 3D POSTPROCESSING
# -------------------------
tests_3d=("explosion2")

for t in "${tests_3d[@]}"; do
    echo "[3D] Postprocessing $t"

    python src/graphing/plot_multid.py toro/$t
    python src/graphing/compute_l1.py toro/$t
done

echo "[3D] Postprocessing complete"
echo "-------------------------"


# -------------------------
# [5] Cleanup
# -------------------------
deactivate

echo "All postprocessing complete."




# RUN Commands
# cd diffuse_vs_GFM_MMS
# chmod +x tests/validation/1_SM_Euler_Tests/SM_graphing.sh
# source .venv/bin/activate
# ./tests/validation/1_SM_Euler_Tests/SM_graphing.sh
