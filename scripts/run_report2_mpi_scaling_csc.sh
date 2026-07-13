#!/usr/bin/env bash
#SBATCH -J oo338_report2_mpi_scaling
#SBATCH --partition=lsc
#SBATCH --clusters=CSC
#SBATCH --time=12:00:00
#SBATCH --output=report2_mpi_scaling_%A.out
#SBATCH --mem=32GB
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --account=oo338

set -euo pipefail

# [0] Environment
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SLURM_SUBMIT_DIR:-$repo_root}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

# [1] Run Report 2 MPI Scaling
RESULT_ROOT="${RESULT_ROOT:-results/quantitative/report_selected_mpi_scaling}"
MPI_SCALING_RESOLUTION="${MPI_SCALING_RESOLUTION:-1300x178}"
BENCHMARK_WARMUPS="${BENCHMARK_WARMUPS:-0}"
BENCHMARK_REPEATS="${BENCHMARK_REPEATS:-1}"

scripts/run_quant_suite.sh \
    --scaling mpi_ranks \
    --case bubble \
    --methods SIM_MPI,DIM_MPI \
    --resolutions "$MPI_SCALING_RESOLUTION" \
    --omp-threads "$OMP_NUM_THREADS" \
    --benchmark-mode clean \
    --benchmark-warmups "$BENCHMARK_WARMUPS" \
    --benchmark-repeats "$BENCHMARK_REPEATS" \
    --result-root "$RESULT_ROOT" \
    --overwrite

echo "Report 2 MPI scaling results: $RESULT_ROOT"
