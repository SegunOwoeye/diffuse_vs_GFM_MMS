#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_bubble_features=false
bubble_only=false
bubble_methods=("GFM" "DIM")
bubble_result_root="results/bubble_features"
quant_args=()

print_wrapper_help() {
    cat <<'EOF'
Additional run_quant_suite.sh wrapper options:
  --bubble-features              Run helium shock-bubble feature extraction after the C++ suite.
  --bubble-only                  Skip the C++ suite and run only bubble feature extraction.
  --bubble-methods GFM DIM       Methods to process for bubble features (default: GFM DIM).
  --bubble-result-root PATH      Output root for feature results (default: results/bubble_features).

All other arguments are passed through to tools/quant_suite.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bubble-features)
            run_bubble_features=true
            shift
            ;;
        --bubble-only)
            bubble_only=true
            run_bubble_features=true
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
        --help|-h)
            quant_args+=("$1")
            print_wrapper_help
            shift
            ;;
        *)
            quant_args+=("$1")
            shift
            ;;
    esac
done

python_bin=".venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
    python_bin="python3"
fi

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
