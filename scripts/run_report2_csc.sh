#!/usr/bin/env bash
#SBATCH --job-name=report2_production
#SBATCH --partition=lsc
#SBATCH --clusters=CSC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --time=48:00:00
#SBATCH --account=oo338
#SBATCH --output=report2_production-%j.out
set -euo pipefail

# [0] Repository
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SLURM_SUBMIT_DIR:-$repo_root}"

# [1] Help And Submission
show_help=false
list_only=false
for argument in "$@"; do
    case "$argument" in
        --help|-h)
            show_help=true
            ;;
        --list)
            list_only=true
            ;;
    esac
done

if [[ "$show_help" == true ]]; then
    cat <<'EOF'
Run the production studies on the university CSC machine.

When called from a login shell this script submits itself with sbatch. Inside an
existing Slurm allocation it executes immediately. It accepts the same suite and
method options as scripts/run_report2_local.sh.

Examples:
  scripts/run_report2_csc.sh --suite base --methods SIM
  scripts/run_report2_csc.sh --suite three-material,3d --methods SIM,DIM
  scripts/run_report2_csc.sh --suite performance --methods SIM --scaling-threads 1,2,4,8,16,32

Shared options:
EOF
    scripts/run_report2_local.sh --help
    exit 0
fi

if [[ "$list_only" == true && -z "${SLURM_JOB_ID:-}" ]]; then
    exec scripts/run_report2_local.sh "$@"
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if ! command -v sbatch >/dev/null 2>&1; then
        echo "run_report2_csc.sh: sbatch is unavailable; run this script on CSC" >&2
        exit 2
    fi
    exec sbatch "$0" "$@"
fi

# [2] Resource Validation
requested_omp_threads=32
scaling_threads="1,2,4,8,16,32"
args=("$@")
for ((index = 0; index < ${#args[@]}; ++index)); do
    case "${args[$index]}" in
        --omp-threads)
            if ((index + 1 >= ${#args[@]})); then
                echo "run_report2_csc.sh: --omp-threads requires a value" >&2
                exit 2
            fi
            requested_omp_threads="${args[$((index + 1))]}"
            ;;
        --scaling-threads)
            if ((index + 1 >= ${#args[@]})); then
                echo "run_report2_csc.sh: --scaling-threads requires a value" >&2
                exit 2
            fi
            scaling_threads="${args[$((index + 1))]}"
            ;;
    esac
done

allocated_cpus="${SLURM_CPUS_PER_TASK:-1}"
if [[ ! "$requested_omp_threads" =~ ^[0-9]+$ || "$requested_omp_threads" -lt 1 ]]; then
    echo "run_report2_csc.sh: --omp-threads must be a positive integer" >&2
    exit 2
fi
if ((requested_omp_threads > allocated_cpus)); then
    echo "run_report2_csc.sh: requested $requested_omp_threads OpenMP threads but Slurm allocated $allocated_cpus" >&2
    exit 2
fi

max_scaling_threads=0
IFS=',' read -r -a scaling_values <<< "$scaling_threads"
for value in "${scaling_values[@]}"; do
    if [[ ! "$value" =~ ^[0-9]+$ || "$value" -lt 1 ]]; then
        echo "run_report2_csc.sh: invalid scaling thread count '$value'" >&2
        exit 2
    fi
    ((value > max_scaling_threads)) && max_scaling_threads="$value"
done
if ((max_scaling_threads > allocated_cpus)); then
    echo "run_report2_csc.sh: scaling requests $max_scaling_threads threads but Slurm allocated $allocated_cpus" >&2
    exit 2
fi

# [3] OpenMP Environment
export OMP_NUM_THREADS="$requested_omp_threads"
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

echo "[csc] job=${SLURM_JOB_ID} cpus_per_task=${allocated_cpus} omp_threads=${requested_omp_threads}"
exec scripts/run_report2_local.sh "$@"
