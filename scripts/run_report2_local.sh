#!/usr/bin/env bash
set -euo pipefail

# [0] Repository
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# [1] Defaults
suites=()
methods=("SIM" "DIM")
omp_threads="${OMP_NUM_THREADS:-6}"
scaling_threads="1,2,4,8,16,32"
benchmark_repeats=3
benchmark_warmups=1
result_root_base="results/quantitative"
conservation_interval=10
timeout_seconds=0
organized_output_dir="results/report2_organized"
cpu_list="${QUANT_CPU_LIST:-}"
resume=true
overwrite=false
dry_run=false
list_only=false
generate_exact_references=false
postprocess=false

print_help() {
    cat <<'EOF'
Run production studies locally or from an SSH shell.

Usage:
  scripts/run_report2_local.sh --suite SUITE[,SUITE...] [options]

Suites:
  base             Fedkiw D2 1D, Toro explosions 2D/3D, helium bubble 2D,
                   and Gorsse TC9 water-air bubble 2D.
  three-material   He three-material test in 1D and 2D.
  3d               Helium bubble 325x45x45 and Gorsse TC9 120x100x100.
  performance      Helium bubble 2D OpenMP scaling with runtime-only solver output.
  sensitivity      SIM reinitialisation and DIM tanh-alpha sensitivity.
  all              All production suites.

Options:
  --suite LIST                 Select one or more comma-separated suites. Required.
  --methods LIST               SIM, DIM, or SIM,DIM. Default: SIM,DIM.
  --omp-threads N              Threads for non-scaling runs. Default: OMP_NUM_THREADS or 6.
  --scaling-threads LIST       Scaling thread counts. Default: 1,2,4,8,16,32.
  --benchmark-repeats N        Measured performance repeats. Default: 3.
  --benchmark-warmups N        Performance warmups. Default: 1.
  --result-root-base PATH      Root for quantitative result directories.
  --conservation-interval N    Conservation sampling interval. Default: 10.
  --timeout-seconds N          Per-run timeout. Zero disables it.
  --cpu-list LIST              Pin local solver processes, for example 0-7.
  --resume                     Skip runs with successful metadata. Default.
  --overwrite                  Remove and rerun selected run directories.
  --generate-exact-references  Generate Fedkiw exact references before base runs.
  --postprocess                Plot, update the tracker, and organize outputs.
  --organized-output-dir PATH  Destination used by --postprocess.
  --list                       List the production suite contents and exit.
  --dry-run                    Print commands without compiling or running.
  --help, -h                   Show this help.

Examples:
  scripts/run_report2_local.sh --suite base --methods SIM
  scripts/run_report2_local.sh --suite three-material,3d --methods SIM,DIM --omp-threads 8
  scripts/run_report2_local.sh --suite performance --methods SIM --scaling-threads 1,2,4,8
EOF
}

print_suite_list() {
    cat <<'EOF'
base:
  common  Toro explosion 2D                100x100, 200x200, 400x400
  common  Toro explosion 3D                50x50x50, 100x100x100, 200x200x200
  SIM/DIM Fedkiw D2 1D                     100, 200, 400, 800
  SIM/DIM helium shock-bubble 2D           1300x178
  SIM/DIM Gorsse TC9 water-air bubble 2D   480x400
three-material:
  SIM/DIM He three-material 1D             100, 200, 400, 800, 2000
  SIM/DIM He three-material 2D             1400x600
3d:
  SIM/DIM helium shock-bubble 3D           325x45x45
  SIM/DIM Gorsse TC9 water-air bubble 3D   120x100x100
performance:
  SIM/DIM helium shock-bubble 2D           1300x178, selected OpenMP threads
sensitivity:
  SIM     helium bubble reinitialisation   intervals 1, 2, 5, 10, 20, never
  DIM     helium bubble tanh alpha         0.5, 1, 2, 4, 8
EOF
}

require_value() {
    local option="$1"
    local count="$2"
    if [[ "$count" -lt 2 ]]; then
        echo "run_report2_local.sh: $option requires a value" >&2
        exit 2
    fi
}

append_csv_values() {
    local value="$1"
    local -n destination="$2"
    local item
    IFS=',' read -r -a parsed_values <<< "$value"
    for item in "${parsed_values[@]}"; do
        [[ -n "$item" ]] && destination+=("$item")
    done
}

# [2] Arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --suite|--suites)
            require_value "$1" "$#"
            append_csv_values "$2" suites
            shift 2
            ;;
        --methods)
            require_value "$1" "$#"
            methods=()
            append_csv_values "$2" methods
            shift 2
            ;;
        --omp-threads)
            require_value "$1" "$#"
            omp_threads="$2"
            shift 2
            ;;
        --scaling-threads)
            require_value "$1" "$#"
            scaling_threads="$2"
            shift 2
            ;;
        --benchmark-repeats)
            require_value "$1" "$#"
            benchmark_repeats="$2"
            shift 2
            ;;
        --benchmark-warmups)
            require_value "$1" "$#"
            benchmark_warmups="$2"
            shift 2
            ;;
        --result-root-base)
            require_value "$1" "$#"
            result_root_base="$2"
            shift 2
            ;;
        --conservation-interval)
            require_value "$1" "$#"
            conservation_interval="$2"
            shift 2
            ;;
        --timeout-seconds)
            require_value "$1" "$#"
            timeout_seconds="$2"
            shift 2
            ;;
        --cpu-list)
            require_value "$1" "$#"
            cpu_list="$2"
            shift 2
            ;;
        --resume)
            resume=true
            overwrite=false
            shift
            ;;
        --overwrite)
            overwrite=true
            resume=false
            shift
            ;;
        --generate-exact-references)
            generate_exact_references=true
            shift
            ;;
        --postprocess)
            postprocess=true
            shift
            ;;
        --organized-output-dir)
            require_value "$1" "$#"
            organized_output_dir="$2"
            shift 2
            ;;
        --list)
            list_only=true
            shift
            ;;
        --dry-run)
            dry_run=true
            shift
            ;;
        --help|-h)
            print_help
            exit 0
            ;;
        *)
            echo "run_report2_local.sh: unknown argument '$1'" >&2
            print_help >&2
            exit 2
            ;;
    esac
done

if [[ "$list_only" == true ]]; then
    print_suite_list
    exit 0
fi

if [[ ${#suites[@]} -eq 0 ]]; then
    echo "run_report2_local.sh: --suite is required" >&2
    print_help >&2
    exit 2
fi

# [3] Validation
declare -A selected_suites=()
for suite in "${suites[@]}"; do
    suite="${suite,,}"
    case "$suite" in
        all)
            selected_suites[base]=1
            selected_suites[three-material]=1
            selected_suites[3d]=1
            selected_suites[performance]=1
            selected_suites[sensitivity]=1
            ;;
        base|three-material|3d|performance|sensitivity)
            selected_suites["$suite"]=1
            ;;
        *)
            echo "run_report2_local.sh: unknown suite '$suite'" >&2
            exit 2
            ;;
    esac
done

normalized_methods=()
for method in "${methods[@]}"; do
    method="${method^^}"
    case "$method" in
        SIM|DIM)
            if [[ " ${normalized_methods[*]} " != *" $method "* ]]; then
                normalized_methods+=("$method")
            fi
            ;;
        *)
            echo "run_report2_local.sh: unknown method '$method'; expected SIM or DIM" >&2
            exit 2
            ;;
    esac
done
if [[ ${#normalized_methods[@]} -eq 0 ]]; then
    echo "run_report2_local.sh: --methods requires SIM, DIM, or both" >&2
    exit 2
fi
methods_csv="$(IFS=','; echo "${normalized_methods[*]}")"

for numeric in "$omp_threads" "$benchmark_repeats" "$benchmark_warmups" "$conservation_interval" "$timeout_seconds"; do
    if [[ ! "$numeric" =~ ^[0-9]+$ ]]; then
        echo "run_report2_local.sh: numeric options require non-negative integers" >&2
        exit 2
    fi
done
if [[ "$omp_threads" -lt 1 || "$benchmark_repeats" -lt 1 || "$conservation_interval" -lt 1 ]]; then
    echo "run_report2_local.sh: threads, repeats, and conservation interval must be positive" >&2
    exit 2
fi
if [[ ! "$scaling_threads" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "run_report2_local.sh: --scaling-threads must be a comma-separated integer list" >&2
    exit 2
fi

if [[ -n "$cpu_list" ]]; then
    export QUANT_CPU_LIST="$cpu_list"
fi
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

# [4] Execution Helpers
python_bin=".venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
    python_bin="python3"
fi

common_flags=(
    --omp-threads "$omp_threads"
    --conservation-interval "$conservation_interval"
    --timeout-seconds "$timeout_seconds"
)
if [[ "$overwrite" == true ]]; then
    common_flags+=(--overwrite)
elif [[ "$resume" == true ]]; then
    common_flags+=(--resume)
fi

print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

run_command() {
    if [[ "$dry_run" == true ]]; then
        print_command "$@"
    else
        "$@"
    fi
}

run_quant() {
    local label="$1"
    shift
    echo
    echo "[production] $label"
    run_command scripts/run_quant_suite.sh "$@" "${common_flags[@]}"
}

method_selected() {
    local requested="$1"
    local method
    for method in "${normalized_methods[@]}"; do
        [[ "$method" == "$requested" ]] && return 0
    done
    return 1
}

# [5] Base Suite
if [[ -n "${selected_suites[base]:-}" ]]; then
    if [[ "$generate_exact_references" == true ]]; then
        echo
        echo "[production] Generating Fedkiw exact references"
        run_command "$python_bin" tools/generate_fedkiw_exact_references.py
    fi

    run_quant "Fedkiw D2 1D" \
        --case fedkiw_1d,test5 \
        --methods "$methods_csv" \
        --resolutions 100,200,400,800 \
        --result-root "$result_root_base/report_selected_fedkiw_d2_1d"

    run_quant "Toro explosion 2D" \
        --case explosion2d \
        --method common \
        --resolutions 100x100,200x200,400x400 \
        --result-root "$result_root_base/report_selected_explosion_2d"

    run_quant "Toro explosion 3D" \
        --case explosion3d \
        --method common \
        --resolutions 50x50x50,100x100x100,200x200x200 \
        --result-root "$result_root_base/report_selected_explosion_3d"

    run_quant "Helium shock-bubble 2D" \
        --case bubble \
        --methods "$methods_csv" \
        --resolutions 1300x178 \
        --result-root "$result_root_base/report_selected_helium_bubble_2d"

    run_quant "Gorsse TC9 water-air bubble 2D" \
        --case gorsse_tc9 \
        --methods "$methods_csv" \
        --resolutions 480x400 \
        --result-root "$result_root_base/report_selected_gorsse_tc9_water_air_2d"
fi

# [6] Three-Material Suite
if [[ -n "${selected_suites[three-material]:-}" ]]; then
    run_quant "He three-material 1D" \
        --case he2023_three_material_1d \
        --methods "$methods_csv" \
        --resolutions 100,200,400,800,2000 \
        --result-root "$result_root_base/report_selected_he2023_three_material_1d"

    run_quant "He three-material triple-point 2D" \
        --case he2023_triple_point \
        --methods "$methods_csv" \
        --resolutions 1400x600 \
        --result-root "$result_root_base/report_selected_he2023_three_material_triple_point_2d"
fi

# [7] Three-Dimensional Suite
if [[ -n "${selected_suites[3d]:-}" ]]; then
    run_quant "Helium shock-bubble 3D" \
        --case bubble3d \
        --methods "$methods_csv" \
        --resolutions 325x45x45 \
        --result-root "$result_root_base/report_selected_helium_bubble_3d"

    run_quant "Gorsse TC9 water-air bubble 3D" \
        --case gorsse_tc9_3d \
        --methods "$methods_csv" \
        --resolutions 120x100x100 \
        --result-root "$result_root_base/report_selected_gorsse_tc9_water_air_3d"
fi

# [8] Performance Suite
if [[ -n "${selected_suites[performance]:-}" ]]; then
    run_quant "Helium shock-bubble 2D OpenMP scaling" \
        --scaling openmp_threads \
        --case bubble \
        --methods "$methods_csv" \
        --resolutions 1300x178 \
        --scaling-threads "$scaling_threads" \
        --benchmark-mode clean \
        --benchmark-warmups "$benchmark_warmups" \
        --benchmark-repeats "$benchmark_repeats" \
        --result-root "$result_root_base/report_selected_openmp_scaling"
fi

# [9] Sensitivity Suite
if [[ -n "${selected_suites[sensitivity]:-}" ]]; then
    if method_selected SIM; then
        run_quant "SIM reinitialisation sensitivity" \
            --sensitivity sim_reinit_bubble \
            --result-root "$result_root_base/report_selected_sim_reinit_sensitivity"
    fi
    if method_selected DIM; then
        run_quant "DIM tanh-alpha sensitivity" \
            --sensitivity dim_alpha_bubble \
            --result-root "$result_root_base/report_selected_dim_alpha_sensitivity"
    fi
fi

# [10] Optional Postprocessing
if [[ "$postprocess" == true ]]; then
    echo
    echo "[production] Generating report figures"
    run_command scripts/plot_report2_selected_suite.sh --result-root-base "$result_root_base"

    echo
    echo "[production] Updating run tracker"
    run_command "$python_bin" scripts/update_report2_run_tracker.py

    echo
    echo "[production] Organizing report outputs"
    run_command "$python_bin" scripts/organize_report2_outputs.py \
        --output-dir "$organized_output_dir" \
        --clean
fi

echo
echo "[production] Selected suites complete"
