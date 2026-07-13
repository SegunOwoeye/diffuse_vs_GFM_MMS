#!/usr/bin/env bash
#SBATCH --job-name=report2_csc
#SBATCH --partition=lsc
#SBATCH --clusters=CSC
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --account=oo338
#SBATCH --output=report2_csc-%j.out
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

omp_threads=1
mpi_ranks=32
result_root_base="results/quantitative"
conservation_interval=10
timeout_seconds=0
overwrite=true
skip_scaling=false
include_gorsse_3d=false
skip_3d=false
skip_postprocess=false
clean_results=false
organized_output_dir="results/report2_organized"
benchmark_repeats=1
benchmark_warmups=0
account_name="oo338"
bootstrap_python=true

print_help() {
    cat <<'EOF'
Run the Report 2 suite with CSC defaults.

Defaults:
  - 32 MPI ranks
  - MPI scaling included unless skipped
  - account name recorded as oo338

Usage:
  scripts/run_report2_csc.sh [options]

Options:
  --omp-threads N              OpenMP thread count per MPI rank.
  --mpi-ranks N                MPI rank count for MPI-backed runs.
  --result-root-base PATH      Root for report result directories.
  --conservation-interval N    Step interval for conservation CSV sampling.
  --timeout-seconds N          Timeout per solver run. Use 0 to disable.
  --no-overwrite               Do not clean per-run outputs before rerun.
  --include-scaling            Include the MPI rank scaling study.
  --skip-scaling               Explicitly skip MPI rank scaling.
  --skip-3d                    Skip the 3D report cases.
  --include-gorsse-3d          Legacy option retained for compatibility.
  --skip-postprocess           Skip plotting, tracker update, and organizer.
  --clean-results              Remove selected quantitative result roots before running.
  --organized-output-dir PATH  Destination for organized report outputs.
  --benchmark-repeats N        Measured repeats for scaling.
  --benchmark-warmups N        Warmup repeats for scaling.
  --account-name NAME          Record the cluster account/user name for notes.
  --no-bootstrap-python        Do not create/update the local Python venv.
  --help, -h                   Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --omp-threads)
            if [[ $# -lt 2 ]]; then
            echo "run_report2_csc.sh: --omp-threads requires a value" >&2
                exit 2
            fi
            omp_threads="$2"
            shift 2
            ;;
        --mpi-ranks)
            if [[ $# -lt 2 ]]; then
                echo "run_report2_csc.sh: --mpi-ranks requires a value" >&2
                exit 2
            fi
            mpi_ranks="$2"
            shift 2
            ;;
        --result-root-base)
            if [[ $# -lt 2 ]]; then
                echo "run_report2_csc.sh: --result-root-base requires a path" >&2
                exit 2
            fi
            result_root_base="$2"
            shift 2
            ;;
        --conservation-interval)
            if [[ $# -lt 2 ]]; then
                echo "run_report2_csc.sh: --conservation-interval requires a value" >&2
                exit 2
            fi
            conservation_interval="$2"
            shift 2
            ;;
        --timeout-seconds)
            if [[ $# -lt 2 ]]; then
                echo "run_report2_csc.sh: --timeout-seconds requires a value" >&2
                exit 2
            fi
            timeout_seconds="$2"
            shift 2
            ;;
        --no-overwrite)
            overwrite=false
            shift
            ;;
        --include-scaling)
            skip_scaling=false
            shift
            ;;
        --skip-scaling)
            skip_scaling=true
            shift
            ;;
        --include-gorsse-3d)
            include_gorsse_3d=true
            shift
            ;;
        --skip-3d)
            skip_3d=true
            shift
            ;;
        --skip-postprocess)
            skip_postprocess=true
            shift
            ;;
        --clean-results)
            clean_results=true
            shift
            ;;
        --organized-output-dir)
            if [[ $# -lt 2 ]]; then
                echo "run_report2_csc.sh: --organized-output-dir requires a path" >&2
                exit 2
            fi
            organized_output_dir="$2"
            shift 2
            ;;
        --benchmark-repeats)
            if [[ $# -lt 2 ]]; then
                echo "run_report2_csc.sh: --benchmark-repeats requires a value" >&2
                exit 2
            fi
            benchmark_repeats="$2"
            shift 2
            ;;
        --benchmark-warmups)
            if [[ $# -lt 2 ]]; then
                echo "run_report2_csc.sh: --benchmark-warmups requires a value" >&2
                exit 2
            fi
            benchmark_warmups="$2"
            shift 2
            ;;
        --account-name)
            if [[ $# -lt 2 ]]; then
                echo "run_report2_csc.sh: --account-name requires a value" >&2
                exit 2
            fi
            account_name="$2"
            shift 2
            ;;
        --no-bootstrap-python)
            bootstrap_python=false
            shift
            ;;
        --help|-h)
            print_help
            exit 0
            ;;
        *)
            echo "run_report2_csc.sh: unknown argument '$1'" >&2
            print_help >&2
            exit 2
            ;;
    esac
done

args=(
    --omp-threads "$omp_threads"
    --mpi-ranks "$mpi_ranks"
    --result-root-base "$result_root_base"
    --conservation-interval "$conservation_interval"
    --timeout-seconds "$timeout_seconds"
    --organized-output-dir "$organized_output_dir"
    --benchmark-repeats "$benchmark_repeats"
    --benchmark-warmups "$benchmark_warmups"
)

if [[ "$overwrite" == false ]]; then
    args+=("--no-overwrite")
fi

if [[ "$skip_scaling" == true ]]; then
    args+=("--skip-scaling")
fi

if [[ "$skip_3d" == true ]]; then
    args+=("--skip-3d")
fi

if [[ "$skip_postprocess" == true ]]; then
    args+=("--skip-plots" "--skip-organize")
fi

if [[ "$clean_results" == true ]]; then
    args+=("--clean-results")
fi

if [[ "$include_gorsse_3d" == true ]]; then
    echo "[csc] --include-gorsse-3d is retained for compatibility; the full selected suite includes 3D Gorsse unless --skip-3d is used"
fi

if [[ "$bootstrap_python" == true ]]; then
    echo "[csc] preparing Python environment"
    python3 -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install numpy pandas matplotlib openpyxl
fi

echo "[csc] account=${account_name} mpi_ranks=${mpi_ranks} omp_threads_per_rank=${omp_threads}"
exec scripts/run_report2_selected_suite.sh "${args[@]}"
