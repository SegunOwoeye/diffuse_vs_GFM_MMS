#!/usr/bin/env bash
#SBATCH --job-name=oo338_report2_openmp_scaling
#SBATCH --clusters=csc
#SBATCH --partition=csc-mphil
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/report2_openmp_scaling_%j.out
#SBATCH --error=logs/report2_openmp_scaling_%j.err
#SBATCH --mail-type=ALL

set -euo pipefail

# [0] Environment
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SLURM_SUBMIT_DIR:-$repo_root}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

# [1] Run Report 2 OpenMP Scaling
RESULT_ROOT="${RESULT_ROOT:-results/quantitative/report_selected_openmp_scaling}"
OPENMP_SCALING_RESOLUTION="${OPENMP_SCALING_RESOLUTION:-1300x178}"
BENCHMARK_WARMUPS="${BENCHMARK_WARMUPS:-0}"
BENCHMARK_REPEATS="${BENCHMARK_REPEATS:-1}"

scripts/run_quant_suite.sh \
    --scaling openmp_threads \
    --case bubble \
    --methods SIM,DIM \
    --resolutions "$OPENMP_SCALING_RESOLUTION" \
    --omp-threads "$OMP_NUM_THREADS" \
    --benchmark-mode clean \
    --benchmark-warmups "$BENCHMARK_WARMUPS" \
    --benchmark-repeats "$BENCHMARK_REPEATS" \
    --result-root "$RESULT_ROOT" \
    --overwrite

echo "Report 2 OpenMP scaling results: $RESULT_ROOT"
