#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

omp_threads=32
mpi_ranks=32
result_root_base="results/quantitative"
conservation_interval=10
timeout_seconds=0
overwrite=true
skip_scaling=true
include_gorsse_3d=false
skip_postprocess=false
organized_output_dir="results/report2_organized"
benchmark_repeats=1
benchmark_warmups=0

print_help() {
    cat <<'EOF'
Run the personal Report 2 suite with your preferred defaults.

This wrapper keeps the shared university runner intact, but defaults to:
  - 32 OpenMP threads
  - skipping the scaling study

Usage:
  scripts/run_report2_personal_suite.sh [options]

Options:
  --omp-threads N              OpenMP thread count for each solver process.
  --mpi-ranks N                Deprecated compatibility option; ignored by quant_suite.
  --result-root-base PATH      Root for report result directories.
  --conservation-interval N    Step interval for conservation CSV sampling.
  --timeout-seconds N          Timeout per solver run. Use 0 to disable.
  --no-overwrite               Do not clean per-run outputs before rerun.
  --include-scaling            Include the OpenMP thread scaling study.
  --skip-scaling               Explicitly skip OpenMP thread scaling.
  --include-gorsse-3d          Include optional 3D Gorsse TC9 water-air run.
  --skip-postprocess           Skip plotting, tracker update, and organizer.
  --organized-output-dir PATH  Destination for organized report outputs.
  --benchmark-repeats N        Measured repeats for scaling.
  --benchmark-warmups N        Warmup repeats for scaling.
  --help, -h                   Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --omp-threads)
            if [[ $# -lt 2 ]]; then
                echo "run_report2_personal_suite.sh: --omp-threads requires a value" >&2
                exit 2
            fi
            omp_threads="$2"
            shift 2
            ;;
        --mpi-ranks)
            if [[ $# -lt 2 ]]; then
                echo "run_report2_personal_suite.sh: --mpi-ranks requires a value" >&2
                exit 2
            fi
            mpi_ranks="$2"
            shift 2
            ;;
        --result-root-base)
            if [[ $# -lt 2 ]]; then
                echo "run_report2_personal_suite.sh: --result-root-base requires a path" >&2
                exit 2
            fi
            result_root_base="$2"
            shift 2
            ;;
        --conservation-interval)
            if [[ $# -lt 2 ]]; then
                echo "run_report2_personal_suite.sh: --conservation-interval requires a value" >&2
                exit 2
            fi
            conservation_interval="$2"
            shift 2
            ;;
        --timeout-seconds)
            if [[ $# -lt 2 ]]; then
                echo "run_report2_personal_suite.sh: --timeout-seconds requires a value" >&2
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
        --skip-postprocess)
            skip_postprocess=true
            shift
            ;;
        --organized-output-dir)
            if [[ $# -lt 2 ]]; then
                echo "run_report2_personal_suite.sh: --organized-output-dir requires a path" >&2
                exit 2
            fi
            organized_output_dir="$2"
            shift 2
            ;;
        --benchmark-repeats)
            if [[ $# -lt 2 ]]; then
                echo "run_report2_personal_suite.sh: --benchmark-repeats requires a value" >&2
                exit 2
            fi
            benchmark_repeats="$2"
            shift 2
            ;;
        --benchmark-warmups)
            if [[ $# -lt 2 ]]; then
                echo "run_report2_personal_suite.sh: --benchmark-warmups requires a value" >&2
                exit 2
            fi
            benchmark_warmups="$2"
            shift 2
            ;;
        --help|-h)
            print_help
            exit 0
            ;;
        *)
            echo "run_report2_personal_suite.sh: unknown argument '$1'" >&2
            print_help >&2
            exit 2
            ;;
    esac
done

args=(
    --omp-threads "$omp_threads"
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

if [[ "$include_gorsse_3d" == true ]]; then
    args+=("--include-gorsse-3d")
fi

if [[ "$skip_postprocess" == true ]]; then
    args+=("--skip-postprocess")
fi

exec scripts/run_report2_university_suite.sh "${args[@]}"
