#!/usr/bin/env bash
#SBATCH -J oo338_SM_MPI_scaling
#SBATCH --partition=csc-mphil
#SBATCH --time=06:00:00
#SBATCH --output=sm_mpi_scaling_%A.out
#SBATCH --mem=16GB
#SBATCH --ntasks=32
#SBATCH --mail-type=ALL
#SBATCH --account=oo338
#SBATCH --clusters=CSC

set -euo pipefail

# [0] Environment
cd "${SLURM_SUBMIT_DIR:-$PWD}"

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install numpy pandas matplotlib openpyxl

# [1] Build Quant Suite
g++ -std=c++17 -O2 -fopenmp -I. tools/quant_suite.cpp -o tools/quant_suite

# [2] Run MPI Scaling
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

RESULT_ROOT="${RESULT_ROOT:-results/quantitative/sm_mpi_scaling_csc}"
MPI_SCALING_RESOLUTION="${MPI_SCALING_RESOLUTION:-50x50x50}"

./tools/quant_suite \
    --scaling mpi_ranks \
    --resolutions "$MPI_SCALING_RESOLUTION" \
    --omp-threads "$OMP_NUM_THREADS" \
    --benchmark-mode clean \
    --benchmark-warmups 1 \
    --benchmark-repeats 3 \
    --result-root "$RESULT_ROOT" \
    --overwrite

echo "SM MPI scaling results: $RESULT_ROOT"
