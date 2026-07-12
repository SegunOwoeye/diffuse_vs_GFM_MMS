#!/usr/bin/env bash
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
skip_postprocess=false
organized_output_dir="results/report2_organized"
benchmark_repeats=1
benchmark_warmups=0

print_help() {
    cat <<'EOF'
Run the dissertation-critical missing Report 2 jobs for the university machine.

This targets the current missing evidence:
  1. He 2023 three-material 1D convergence
  2. He 2023 three-material 2D triple-point validation
  3. MPI rank performance scaling
  4. Optional 3D Gorsse TC9 water-air extension

Usage:
  scripts/run_report2_university_suite.sh [options]

Options:
  --omp-threads N              OpenMP thread count per MPI rank.
  --mpi-ranks N                MPI rank count for MPI-backed runs.
  --result-root-base PATH      Root for report result directories.
  --conservation-interval N    Step interval for conservation CSV sampling.
  --timeout-seconds N          Timeout per solver run. Use 0 to disable.
  --no-overwrite               Do not clean per-run outputs before rerun.
  --skip-scaling               Skip MPI rank scaling.
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
                echo "run_report2_university_suite.sh: --omp-threads requires a value" >&2
                exit 2
            fi
            omp_threads="$2"
            shift 2
            ;;
        --mpi-ranks)
            if [[ $# -lt 2 ]]; then
                echo "run_report2_university_suite.sh: --mpi-ranks requires a value" >&2
                exit 2
            fi
            mpi_ranks="$2"
            shift 2
            ;;
        --result-root-base)
            if [[ $# -lt 2 ]]; then
                echo "run_report2_university_suite.sh: --result-root-base requires a path" >&2
                exit 2
            fi
            result_root_base="$2"
            shift 2
            ;;
        --conservation-interval)
            if [[ $# -lt 2 ]]; then
                echo "run_report2_university_suite.sh: --conservation-interval requires a value" >&2
                exit 2
            fi
            conservation_interval="$2"
            shift 2
            ;;
        --timeout-seconds)
            if [[ $# -lt 2 ]]; then
                echo "run_report2_university_suite.sh: --timeout-seconds requires a value" >&2
                exit 2
            fi
            timeout_seconds="$2"
            shift 2
            ;;
        --no-overwrite)
            overwrite=false
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
                echo "run_report2_university_suite.sh: --organized-output-dir requires a path" >&2
                exit 2
            fi
            organized_output_dir="$2"
            shift 2
            ;;
        --benchmark-repeats)
            if [[ $# -lt 2 ]]; then
                echo "run_report2_university_suite.sh: --benchmark-repeats requires a value" >&2
                exit 2
            fi
            benchmark_repeats="$2"
            shift 2
            ;;
        --benchmark-warmups)
            if [[ $# -lt 2 ]]; then
                echo "run_report2_university_suite.sh: --benchmark-warmups requires a value" >&2
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
            echo "run_report2_university_suite.sh: unknown argument '$1'" >&2
            print_help >&2
            exit 2
            ;;
    esac
done

python_bin=".venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
    python_bin="python3"
fi

common_flags=("--omp-threads" "$omp_threads" "--mpi-ranks" "$mpi_ranks")
if [[ "$overwrite" == true ]]; then
    common_flags+=("--overwrite")
fi

run_quant() {
    local label="$1"
    shift
    echo
    echo "[university] $label"
    scripts/run_quant_suite.sh "$@" "${common_flags[@]}"
}

run_quant "He 2023 three-material 1D convergence" \
    --case he2023_three_material_1d \
    --methods SIM_MPI,DIM_MPI \
    --resolutions 100,200,400,800,2000 \
    --result-root "$result_root_base/report_selected_he2023_three_material_1d" \
    --conservation-interval "$conservation_interval" \
    --timeout-seconds "$timeout_seconds"

run_quant "He 2023 three-material triple-point 2D" \
    --case he2023_triple_point \
    --methods SIM_MPI,DIM_MPI \
    --resolutions 1400x600 \
    --result-root "$result_root_base/report_selected_he2023_three_material_triple_point_2d" \
    --conservation-interval "$conservation_interval" \
    --timeout-seconds "$timeout_seconds"

if [[ "$skip_scaling" != true ]]; then
    echo
    echo "[university] MPI rank scaling"
    scaling_flags=()
    if [[ "$overwrite" == true ]]; then
        scaling_flags+=("--overwrite")
    fi
    scripts/run_quant_suite.sh \
        --scaling mpi_ranks \
        --methods SM_MPI,SIM_MPI,DIM_MPI \
        --omp-threads 1 \
        --benchmark-mode clean \
        --benchmark-warmups "$benchmark_warmups" \
        --benchmark-repeats "$benchmark_repeats" \
        --result-root "$result_root_base/report_selected_mpi_scaling" \
        --conservation-interval "$conservation_interval" \
        --timeout-seconds "$timeout_seconds" \
        "${scaling_flags[@]}"
fi

if [[ "$include_gorsse_3d" == true ]]; then
    run_quant "3D Gorsse TC9 water-air bubble" \
        --case gorsse_tc9_3d \
        --methods SIM_MPI,DIM_MPI \
        --resolutions 240x200x200 \
        --result-root "$result_root_base/report_selected_gorsse_tc9_water_air_3d" \
        --conservation-interval "$conservation_interval" \
        --timeout-seconds "$timeout_seconds"
fi

if [[ "$skip_postprocess" != true ]]; then
    echo
    echo "[university] Generating report PNG figures"
    scripts/plot_report2_selected_suite.sh --result-root-base "$result_root_base"

    echo
    echo "[university] Updating run tracker"
    "$python_bin" scripts/update_report2_run_tracker.py

    echo
    echo "[university] Organizing report outputs"
    "$python_bin" scripts/organize_report2_outputs.py \
        --output-dir "$organized_output_dir" \
        --clean
fi

echo
echo "[university] Missing Report 2 suite complete"
