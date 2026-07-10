#!/bin/bash

set -euo pipefail

cores=${CORES:-6}
mode="${1:-both}"

g++ -std=c++17 -O2 -fopenmp -I. -DAPP_DIM=2 src/app/multimaterial_main.cpp -o mm_main_2d

if [ ! -d ".venv" ]; then
    echo "[ERROR] .venv not found. Create the virtual environment first."
    exit 1
fi

source .venv/bin/activate

run_case() {
    local label="$1"
    local config="$2"
    local output="$3"
    local output_dir="data/csv/$output"

    echo "Running helium bubble collapse ($label)"
    mkdir -p "$output_dir"
    rm -f "$output_dir"/*.csv "$output_dir"/*.png

    OMP_NUM_THREADS=$cores OMP_SCHEDULE=dynamic ./mm_main_2d "$config"
    python src/graphing/plot_multid.py --schlieren "$output"

    echo "Completed helium bubble collapse ($label)"
    echo "-------------------------"
}

case "$mode" in
    gfm|GFM)
        run_case "GFM" \
            "configs/GFM/MM_2D_validation/test6.txt" \
            "gfm/MM_2D_validation/gfm_helium_bubble_2d"
        ;;
    dim|DIM)
        run_case "DIM" \
            "configs/DIM/MM_2D_validation/test6.txt" \
            "dim/MM_2D_validation/dim_helium_bubble_2d"
        ;;
    both|all)
        run_case "GFM" \
            "configs/GFM/MM_2D_validation/test6.txt" \
            "gfm/MM_2D_validation/gfm_helium_bubble_2d"
        run_case "DIM" \
            "configs/DIM/MM_2D_validation/test6.txt" \
            "dim/MM_2D_validation/dim_helium_bubble_2d"
        ;;
    *)
        echo "Usage: $0 [gfm|dim|both]"
        deactivate
        exit 1
        ;;
esac

deactivate

echo "Helium bubble collapse runs complete."
