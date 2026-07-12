#!/usr/bin/env bash
#SBATCH --job-name=report2_3d_csc
#SBATCH --partition=lsc
#SBATCH --clusters=CSC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=36:00:00
#SBATCH --mem=128GB
#SBATCH --account=oo338
#SBATCH --output=report2_3d_csc-%j.out
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

omp_threads=32
result_root_base="results/quantitative"
conservation_interval=10
timeout_seconds=0
overwrite=true
run_gorsse_3d=true
run_bubble_3d_features=true
run_postprocess=false
organized_output_dir="results/report2_organized"

print_help() {
    cat <<'EOF'
Run the Report 2 3D suite on the CSC/LSC cluster.

This runs:
  1. 3D Euler explosion
  2. 3D contaminated helium shock-bubble, SIM and DIM
  3. 3D Gorsse TC9 water-air bubble, SIM and DIM

Usage:
  scripts/run_report2_3d_csc.sh [options]

Options:
  --omp-threads N              OpenMP thread count for solver runs.
  --result-root-base PATH      Root for report result directories.
  --conservation-interval N    Step interval for conservation CSV sampling.
  --timeout-seconds N          Timeout per solver run. Use 0 to disable.
  --no-overwrite               Do not clean per-run outputs before rerun.
  --skip-gorsse-3d             Skip the optional 3D Gorsse TC9 water-air run.
  --skip-bubble-3d-features    Skip 3D helium surface feature extraction.
  --postprocess                Generate report plots, tracker, and organized outputs.
  --organized-output-dir PATH  Destination for organized report outputs.
  --help, -h                   Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --omp-threads)
            if [[ $# -lt 2 ]]; then
                echo "run_report2_3d_csc.sh: --omp-threads requires a value" >&2
                exit 2
            fi
            omp_threads="$2"
            shift 2
            ;;
        --result-root-base)
            if [[ $# -lt 2 ]]; then
                echo "run_report2_3d_csc.sh: --result-root-base requires a path" >&2
                exit 2
            fi
            result_root_base="$2"
            shift 2
            ;;
        --conservation-interval)
            if [[ $# -lt 2 ]]; then
                echo "run_report2_3d_csc.sh: --conservation-interval requires a value" >&2
                exit 2
            fi
            conservation_interval="$2"
            shift 2
            ;;
        --timeout-seconds)
            if [[ $# -lt 2 ]]; then
                echo "run_report2_3d_csc.sh: --timeout-seconds requires a value" >&2
                exit 2
            fi
            timeout_seconds="$2"
            shift 2
            ;;
        --no-overwrite)
            overwrite=false
            shift
            ;;
        --skip-gorsse-3d)
            run_gorsse_3d=false
            shift
            ;;
        --skip-bubble-3d-features)
            run_bubble_3d_features=false
            shift
            ;;
        --postprocess)
            run_postprocess=true
            shift
            ;;
        --organized-output-dir)
            if [[ $# -lt 2 ]]; then
                echo "run_report2_3d_csc.sh: --organized-output-dir requires a path" >&2
                exit 2
            fi
            organized_output_dir="$2"
            shift 2
            ;;
        --help|-h)
            print_help
            exit 0
            ;;
        *)
            echo "run_report2_3d_csc.sh: unknown argument '$1'" >&2
            print_help >&2
            exit 2
            ;;
    esac
done

python_bin=".venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
    python_bin="python3"
fi

export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"
export OMP_PLACES="${OMP_PLACES:-cores}"

common_flags=("--omp-threads" "$omp_threads")
if [[ "$overwrite" == true ]]; then
    common_flags+=("--overwrite")
fi

conservation_flags=(
    "--conservation-interval" "$conservation_interval"
    "--timeout-seconds" "$timeout_seconds"
)

run_quant() {
    local label="$1"
    shift
    echo
    echo "[report2-3d] $label"
    scripts/run_quant_suite.sh "$@" "${common_flags[@]}"
}

run_quant "3D Euler explosion" \
    --case explosion3d \
    --method common \
    --resolutions 50x50x50,100x100x100,200x200x200 \
    --result-root "$result_root_base/report_selected_explosion_3d"

bubble_feature_flags=()
if [[ "$run_bubble_3d_features" == true ]]; then
    bubble_feature_flags=(
        "--bubble-3d-features"
        "--bubble-methods" "GFM" "DIM"
        "--bubble-result-root" "$result_root_base/report_selected_helium_bubble_3d/features"
    )
fi

run_quant "3D contaminated helium shock-bubble" \
    --case bubble3d \
    --methods sim,dim \
    --resolutions 325x45x45 \
    --result-root "$result_root_base/report_selected_helium_bubble_3d" \
    "${conservation_flags[@]}" \
    "${bubble_feature_flags[@]}"

if [[ "$run_gorsse_3d" == true ]]; then
    run_quant "3D Gorsse TC9 water-air bubble" \
        --case gorsse_tc9_3d \
        --methods sim,dim \
        --resolutions 240x200x200 \
        --result-root "$result_root_base/report_selected_gorsse_tc9_water_air_3d" \
        "${conservation_flags[@]}"
fi

if [[ "$run_postprocess" == true ]]; then
    echo
    echo "[report2-3d] Generating report PNG figures"
    scripts/plot_report2_selected_suite.sh --result-root-base "$result_root_base"

    echo
    echo "[report2-3d] Updating run tracker"
    "$python_bin" scripts/update_report2_run_tracker.py

    echo
    echo "[report2-3d] Organizing report outputs"
    "$python_bin" scripts/organize_report2_outputs.py \
        --output-dir "$organized_output_dir" \
        --clean
fi

echo
echo "[report2-3d] 3D suite complete"
