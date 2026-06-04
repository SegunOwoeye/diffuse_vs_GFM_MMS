#!/bin/bash
set -euo pipefail

run_full=false

usage() {
    cat <<EOF
Usage: $0 [--full]

Options:
  --full    Also run the Barton radial-pressure tensor benchmark 250x250 case.
EOF
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --full) run_full=true ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] Unknown option: $1"; usage; exit 1 ;;
    esac
    shift
done

cores="${CORES_SOLID:-${CORES:-${OMP_NUM_THREADS:-8}}}"

echo "[SOLID] Compiling solid_main with OpenMP..."
g++ -std=c++17 -O3 -fopenmp -I. src/app/solid_main.cpp -o solid_main

echo "[SOLID][Barton 1D] Running test1: Section 6.1, 0.8 km/s flyer plate"
OMP_NUM_THREADS="$cores" ./solid_main configs/solid/Barton_1D_validation/test1.txt

echo "[SOLID][Barton 1D] Running test2: Section 6.1, 2.0 km/s flyer plate"
OMP_NUM_THREADS="$cores" ./solid_main configs/solid/Barton_1D_validation/test2.txt

echo "[SOLID][Barton 2D] Running test1_debug: radial-pressure tensor, 50x50"
OMP_NUM_THREADS="$cores" ./solid_main configs/solid/Barton_2D_validation/test1_debug.txt

if [ "$run_full" = true ]; then
    echo "[SOLID][Barton 2D] Running test1: radial-pressure tensor, 250x250 paper grid"
    OMP_NUM_THREADS="$cores" ./solid_main configs/solid/Barton_2D_validation/test1.txt
fi

echo "[SOLID] Simulations complete."
