#!/usr/bin/env bash
set -euo pipefail

# [0] Defaults
DIMENSION="${MPI_DIM:-3}"
RANKS="${MPI_RANKS:-4}"
THREADS="${OMP_NUM_THREADS:-1}"

case "$DIMENSION" in
    1)
        DEFAULT_CONFIG="configs/toro/test1.txt"
        DEFAULT_RESOLUTION="100"
        ;;
    2)
        DEFAULT_CONFIG="configs/toro/explosion1.txt"
        DEFAULT_RESOLUTION="50x50"
        ;;
    3)
        DEFAULT_CONFIG="configs/toro/explosion2.txt"
        DEFAULT_RESOLUTION="50x50x50"
        ;;
    *)
        echo "MPI_DIM must be 1, 2, or 3" >&2
        exit 2
        ;;
esac

CONFIG="${1:-$DEFAULT_CONFIG}"
RESOLUTION="${MPI_RESOLUTION:-$DEFAULT_RESOLUTION}"
OUTPUT_DIR="${MPI_OUTPUT_DIR:-tmp/sm_mpi_${DIMENSION}d_probe}"
BINARY="sm_mpi_main_${DIMENSION}d"

# [1] Build
mpic++ -std=c++17 -O3 -march=native -fopenmp -DAPP_DIM="$DIMENSION" -I. \
    src/app/sm_mpi_main.cpp \
    -o "$BINARY"

# [2] Run
export OMP_NUM_THREADS="$THREADS"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

mpirun -np "$RANKS" "./$BINARY" "$CONFIG" \
    --resolution "$RESOLUTION" \
    --output-dir "$OUTPUT_DIR"

echo "MPI probe output: $OUTPUT_DIR"
