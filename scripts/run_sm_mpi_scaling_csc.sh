#!/usr/bin/env bash
#SBATCH -J oo338_report2_openmp_scaling
#SBATCH --partition=lsc
#SBATCH --clusters=CSC
#SBATCH --time=12:00:00
#SBATCH --output=report2_openmp_scaling_%A.out
#SBATCH --mem=32GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mail-type=ALL
#SBATCH --account=oo338

set -euo pipefail

# [0] Compatibility Wrapper
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SLURM_SUBMIT_DIR:-$repo_root}"

echo "scripts/run_sm_mpi_scaling_csc.sh is retained for compatibility."
echo "Running the Report 2 2D helium bubble OpenMP scaling study instead."

exec scripts/run_report2_mpi_scaling_csc.sh
