#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

overwrite=true
omp_threads=6
timeout_seconds=0
conservation_interval=10
result_root_base="results/quantitative"
skip_exact_refs=false
skip_3d=false
skip_scaling=false
skip_sensitivity=false
skip_reinit_sensitivity=false
skip_alpha_sensitivity=false
skip_plots=false
skip_organize=false
organized_output_dir="results/report2_organized"
benchmark_repeats=3
benchmark_warmups=1

print_help() {
    cat <<'EOF'
Run the selected Report 2 quantitative suite.

Usage:
  scripts/run_report2_selected_suite.sh [options]

Options:
  --no-overwrite             Do not pass --overwrite to quant_suite.
  --result-root-base PATH    Root for all selected result directories.
  --omp-threads N            OpenMP thread count for solver runs.
  --timeout-seconds N        Timeout for each solver run. Use 0 to disable.
  --conservation-interval N  Conservation sampling interval.
  --skip-exact-refs          Skip Fedkiw exact-reference generation.
  --skip-3d                  Skip 3D explosion, 3D helium bubble, and 3D Gorsse TC9.
  --skip-scaling             Skip OpenMP speedup study.
  --skip-sensitivity         Skip all interface sensitivity studies.
  --skip-reinit-sensitivity  Skip rGFM reinitialisation sensitivity.
  --skip-alpha-sensitivity   Skip DIM interface-thickness sensitivity.
  --skip-plots               Skip report PNG generation after solver runs.
  --skip-organize            Skip final report output organization.
  --organized-output-dir PATH
                              Destination for organized report outputs.
  --benchmark-repeats N      Measured repeats for speedup study.
  --benchmark-warmups N      Warmup repeats for speedup study.
  --help, -h                 Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-overwrite)
            overwrite=false
            shift
            ;;
        --result-root-base)
            if [[ $# -lt 2 ]]; then
                echo "run_report2_selected_suite.sh: --result-root-base requires a path" >&2
                exit 2
            fi
            result_root_base="$2"
            shift 2
            ;;
        --omp-threads)
            if [[ $# -lt 2 ]]; then
                echo "run_report2_selected_suite.sh: --omp-threads requires a value" >&2
                exit 2
            fi
            omp_threads="$2"
            shift 2
            ;;
        --timeout-seconds)
            if [[ $# -lt 2 ]]; then
                echo "run_report2_selected_suite.sh: --timeout-seconds requires a value" >&2
                exit 2
            fi
            timeout_seconds="$2"
            shift 2
            ;;
        --conservation-interval)
            if [[ $# -lt 2 ]]; then
                echo "run_report2_selected_suite.sh: --conservation-interval requires a value" >&2
                exit 2
            fi
            conservation_interval="$2"
            shift 2
            ;;
        --skip-exact-refs)
            skip_exact_refs=true
            shift
            ;;
        --skip-3d)
            skip_3d=true
            shift
            ;;
        --skip-scaling)
            skip_scaling=true
            shift
            ;;
        --skip-sensitivity)
            skip_sensitivity=true
            shift
            ;;
        --skip-reinit-sensitivity)
            skip_reinit_sensitivity=true
            shift
            ;;
        --skip-alpha-sensitivity)
            skip_alpha_sensitivity=true
            shift
            ;;
        --skip-plots)
            skip_plots=true
            shift
            ;;
        --skip-organize)
            skip_organize=true
            shift
            ;;
        --organized-output-dir)
            if [[ $# -lt 2 ]]; then
                echo "run_report2_selected_suite.sh: --organized-output-dir requires a path" >&2
                exit 2
            fi
            organized_output_dir="$2"
            shift 2
            ;;
        --benchmark-repeats)
            if [[ $# -lt 2 ]]; then
                echo "run_report2_selected_suite.sh: --benchmark-repeats requires a value" >&2
                exit 2
            fi
            benchmark_repeats="$2"
            shift 2
            ;;
        --benchmark-warmups)
            if [[ $# -lt 2 ]]; then
                echo "run_report2_selected_suite.sh: --benchmark-warmups requires a value" >&2
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
            echo "run_report2_selected_suite.sh: unknown argument '$1'" >&2
            print_help >&2
            exit 2
            ;;
    esac
done

python_bin=".venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
    python_bin="python3"
fi

common_flags=("--omp-threads" "$omp_threads")
if [[ "$overwrite" == true ]]; then
    common_flags+=("--overwrite")
fi

conservation_flags=("--conservation-interval" "$conservation_interval")
timeout_flags=("--timeout-seconds" "$timeout_seconds")

run_quant() {
    local label="$1"
    shift
    echo
    echo "[report2] $label"
    scripts/run_quant_suite.sh "$@" "${common_flags[@]}"
}

run_quant_with_conservation() {
    local label="$1"
    shift
    run_quant "$label" "$@" "${conservation_flags[@]}"
}

run_quant_with_timeout_and_conservation() {
    local label="$1"
    shift
    run_quant "$label" "$@" "${timeout_flags[@]}" "${conservation_flags[@]}"
}

if [[ "$skip_exact_refs" != true ]]; then
    if [[ -f tools/generate_fedkiw_exact_references.py ]]; then
        echo "[report2] Generating Fedkiw exact references"
        "$python_bin" tools/generate_fedkiw_exact_references.py
    else
        echo "[report2] Skipping exact references because tools/generate_fedkiw_exact_references.py is missing" >&2
    fi
fi

run_quant "Toro test 5 1D Euler check" \
    --case toro_1d,test5 \
    --method common \
    --resolutions 100,200,400,800 \
    --result-root "$result_root_base/report_selected_toro_test5"

run_quant "2D explosion Euler check" \
    --case explosion2d \
    --method common \
    --resolutions 100x100,200x200,400x400 \
    --result-root "$result_root_base/report_selected_explosion_2d"

if [[ "$skip_3d" != true ]]; then
    run_quant "3D explosion Euler check" \
        --case explosion3d \
        --method common \
        --resolutions 50x50x50,100x100x100,200x200x200 \
        --result-root "$result_root_base/report_selected_explosion_3d"
fi

run_quant_with_conservation "FedkiwD2 1D SIM versus DIM" \
    --case fedkiw_1d,test5 \
    --methods sim,dim \
    --resolutions 100,200,400,800 \
    --result-root "$result_root_base/report_selected_fedkiw_d2_1d"

run_quant_with_conservation "He 2023 three-material 1D convergence" \
    --case he2023_three_material_1d \
    --methods sim,dim \
    --resolutions 100,200,400,800,2000 \
    --result-root "$result_root_base/report_selected_he2023_three_material_1d"

run_quant_with_conservation "2D contaminated helium shock-bubble" \
    --case bubble \
    --methods sim,dim \
    --resolutions 325x45,650x89,1300x178 \
    --result-root "$result_root_base/report_selected_helium_bubble_2d"

if [[ "$skip_3d" != true ]]; then
    run_quant_with_conservation "3D contaminated helium shock-bubble" \
        --case bubble3d \
        --methods sim,dim \
        --resolutions 650x89x89 \
        --result-root "$result_root_base/report_selected_helium_bubble_3d"
fi

run_quant_with_timeout_and_conservation "2D Gorsse TC9 water-air bubble" \
    --case gorsse_tc9 \
    --methods sim,dim \
    --resolutions 240x200,480x400 \
    --result-root "$result_root_base/report_selected_gorsse_tc9_water_air_2d"

run_quant_with_timeout_and_conservation "He 2023 three-material triple-point 2D" \
    --case he2023_triple_point \
    --methods sim,dim \
    --resolutions 1400x600 \
    --result-root "$result_root_base/report_selected_he2023_three_material_triple_point_2d"

if [[ "$skip_3d" != true ]]; then
    run_quant_with_timeout_and_conservation "3D Gorsse TC9 water-air bubble" \
        --case gorsse_tc9_3d \
        --methods sim,dim \
        --resolutions 240x200x200 \
        --result-root "$result_root_base/report_selected_gorsse_tc9_water_air_3d"
fi

if [[ "$skip_sensitivity" != true && "$skip_reinit_sensitivity" != true ]]; then
    run_quant_with_conservation "rGFM reinitialisation sensitivity" \
        --sensitivity sim_reinit \
        --result-root "$result_root_base/report_selected_sim_reinit_sensitivity"
fi

if [[ "$skip_sensitivity" != true && "$skip_alpha_sensitivity" != true ]]; then
    run_quant_with_conservation "DIM interface-thickness sensitivity" \
        --sensitivity dim_epsilon \
        --result-root "$result_root_base/report_selected_dim_alpha_sensitivity"
fi

if [[ "$skip_scaling" != true ]]; then
    echo
    echo "[report2] OpenMP speedup study"
    scaling_flags=()
    if [[ "$overwrite" == true ]]; then
        scaling_flags+=("--overwrite")
    fi
    scripts/run_quant_suite.sh \
        --scaling openmp_threads \
        --benchmark-mode clean \
        --benchmark-warmups "$benchmark_warmups" \
        --benchmark-repeats "$benchmark_repeats" \
        --result-root "$result_root_base/report_selected_openmp_scaling" \
        "${scaling_flags[@]}"
fi

if [[ "$skip_plots" != true ]]; then
    echo
    echo "[report2] Generating report PNG figures"
    plot_args=("--result-root-base" "$result_root_base")
    scripts/plot_report2_selected_suite.sh "${plot_args[@]}"
fi

echo
echo "[report2] Updating run tracker"
"$python_bin" scripts/update_report2_run_tracker.py

if [[ "$skip_organize" != true ]]; then
    echo
    echo "[report2] Organizing report outputs"
    "$python_bin" scripts/organize_report2_outputs.py \
        --output-dir "$organized_output_dir" \
        --clean
fi

echo
echo "[report2] Selected suite complete"
