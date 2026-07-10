#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_bubble_features=false
run_bubble_3d_features=false
bubble_only=false
bubble_methods=("GFM" "DIM")
bubble_result_root="results/bubble_features"
bubble_3d_csv_root=""
quant_args=()

print_wrapper_help() {
    cat <<'EOF'
Additional run_quant_suite.sh wrapper options:
  --bubble-features              Run helium shock-bubble feature extraction after the C++ suite.
  --bubble-3d-features           Extract full 3D interface-surface features and surface plots.
  --bubble-only                  Skip the C++ suite and run only bubble feature extraction.
  --bubble-methods GFM DIM       Methods to process for bubble features (default: GFM DIM).
  --bubble-result-root PATH      Output root for feature results (default: results/bubble_features).
  --bubble-3d-csv-root PATH      Root containing 3D bubble CSVs (default: quant --result-root/raw, else data/csv).

All other arguments are passed through to tools/quant_suite.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bubble-features)
            run_bubble_features=true
            shift
            ;;
        --bubble-3d-features)
            run_bubble_3d_features=true
            shift
            ;;
        --bubble-only)
            bubble_only=true
            shift
            ;;
        --bubble-methods)
            shift
            bubble_methods=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                bubble_methods+=("$1")
                shift
            done
            if [[ ${#bubble_methods[@]} -eq 0 ]]; then
                echo "run_quant_suite.sh: --bubble-methods requires at least one method" >&2
                exit 2
            fi
            ;;
        --bubble-result-root)
            if [[ $# -lt 2 ]]; then
                echo "run_quant_suite.sh: --bubble-result-root requires a path" >&2
                exit 2
            fi
            bubble_result_root="$2"
            shift 2
            ;;
        --bubble-3d-csv-root)
            if [[ $# -lt 2 ]]; then
                echo "run_quant_suite.sh: --bubble-3d-csv-root requires a path" >&2
                exit 2
            fi
            bubble_3d_csv_root="$2"
            shift 2
            ;;
        --help|-h)
            print_wrapper_help
            exit 0
            ;;
        *)
            quant_args+=("$1")
            shift
            ;;
    esac
done

if [[ "$bubble_only" == true && "$run_bubble_features" != true && "$run_bubble_3d_features" != true ]]; then
    run_bubble_features=true
fi

python_bin=".venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
    python_bin="python3"
fi

quant_result_root_from_args() {
    local expect_value=false
    local arg
    for arg in "${quant_args[@]}"; do
        if [[ "$expect_value" == true ]]; then
            printf '%s\n' "$arg"
            return 0
        fi
        if [[ "$arg" == "--result-root" ]]; then
            expect_value=true
        fi
    done
    return 1
}

run_bubble_method() {
    local method="$1"
    local method_upper
    method_upper="$(printf '%s' "$method" | tr '[:lower:]' '[:upper:]')"

    case "$method_upper" in
        GFM|SIM)
            echo "[bubble_features] GFM/SIM"
            "$python_bin" src/bubble_features/batch_extract_bubble_features.py \
                --csv-dir data/csv/gfm/MM_2D_validation/gfm_helium_bubble_2d \
                --csv-glob "gfm_helium_bubble_2d_t*_N1300_N178.csv" \
                --image-dir data/bubble_collapse_validation/GFM \
                --image-prefix MM_2D_validation_gfm_helium_bubble_2d \
                --outdir "$bubble_result_root/GFM_piecewise_v4"
            "$python_bin" src/bubble_features/fit_bubble_feature_velocities.py \
                --positions-csv "$bubble_result_root/GFM_piecewise_v4/bubble_feature_positions_master.csv" \
                --output-csv "$bubble_result_root/GFM_piecewise_v4/bubble_feature_velocity_fits.csv"
            ;;
        DIM|ALLAIRE)
            echo "[bubble_features] DIM/Allaire"
            "$python_bin" src/bubble_features/batch_extract_bubble_features.py \
                --csv-dir data/csv/dim/MM_2D_validation/dim_helium_bubble_2d \
                --csv-glob "dim_helium_bubble_2d_t*_N1300_N178.csv" \
                --image-dir data/csv/dim/MM_2D_validation/dim_helium_bubble_2d \
                --image-prefix MM_2D_validation_dim_helium_bubble_2d \
                --outdir "$bubble_result_root/DIM_piecewise_v4"
            "$python_bin" src/bubble_features/fit_bubble_feature_velocities.py \
                --positions-csv "$bubble_result_root/DIM_piecewise_v4/bubble_feature_positions_master.csv" \
                --output-csv "$bubble_result_root/DIM_piecewise_v4/bubble_feature_velocity_fits.csv"
            ;;
        *)
            echo "run_quant_suite.sh: unknown bubble method '$method' (expected GFM/SIM or DIM/Allaire)" >&2
            return 2
            ;;
    esac
}

run_bubble_3d_method() {
    local method="$1"
    local method_upper
    local csv_root
    local csv_find_pattern
    local surface_out
    local quant_root
    local found_any=false

    method_upper="$(printf '%s' "$method" | tr '[:lower:]' '[:upper:]')"

    csv_root="$bubble_3d_csv_root"
    if [[ -z "$csv_root" ]]; then
        if quant_root="$(quant_result_root_from_args)"; then
            csv_root="$quant_root/raw"
        fi
    fi

    case "$method_upper" in
        GFM|SIM)
            [[ -n "$csv_root" ]] || csv_root="data/csv/gfm/MM_3D_validation"
            csv_find_pattern="*gfm*helium_bubble_3d*_t*_N*_N*_N*.csv"
            surface_out="$bubble_result_root/GFM_3D_surface"
            echo "[bubble_features_3d] GFM/SIM full 3D surface from $csv_root"
            ;;
        DIM|ALLAIRE)
            [[ -n "$csv_root" ]] || csv_root="data/csv/dim/MM_3D_validation"
            csv_find_pattern="*dim*helium_bubble_3d*_t*_N*_N*_N*.csv"
            surface_out="$bubble_result_root/DIM_3D_surface"
            echo "[bubble_features_3d] DIM/Allaire full 3D surface from $csv_root"
            ;;
        *)
            echo "run_quant_suite.sh: unknown 3D bubble method '$method' (expected GFM/SIM or DIM/Allaire)" >&2
            return 2
            ;;
    esac

    if [[ ! -d "$csv_root" ]]; then
        echo "run_quant_suite.sh: 3D CSV root does not exist: $csv_root" >&2
        return 2
    fi

    mkdir -p "$surface_out/pretty_surface"

    while IFS= read -r csv_file; do
        found_any=true
        echo "[bubble_features_3d] plotting $csv_file"

        "$python_bin" src/bubble_features/extract_3d_surface_features.py \
            --csv "$csv_file" \
            --outdir "$surface_out/pretty_surface" \
            --interactive
    done < <(find "$csv_root" -type f -name "$csv_find_pattern" | sort)

    if [[ "$found_any" != true ]]; then
        echo "run_quant_suite.sh: no 3D bubble CSVs found in $csv_root matching $csv_find_pattern" >&2
        return 2
    fi
}

mkdir -p tools

if [[ "$bubble_only" != true ]]; then
    echo "[compile] tools/quant_suite"
    g++ -std=c++17 -O2 -fopenmp -I. tools/quant_suite.cpp -o tools/quant_suite

    ./tools/quant_suite "${quant_args[@]}"
fi

if [[ "$run_bubble_features" == true ]]; then
    for method in "${bubble_methods[@]}"; do
        run_bubble_method "$method"
    done
fi

if [[ "$run_bubble_3d_features" == true ]]; then
    for method in "${bubble_methods[@]}"; do
        run_bubble_3d_method "$method"
    done
fi
