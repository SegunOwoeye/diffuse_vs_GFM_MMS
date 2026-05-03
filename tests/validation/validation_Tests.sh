#!/bin/bash


# -------------------------
# [1] Defaults
# -------------------------
RUN_SIM=true
RUN_PLOT=true
ARCHIVE=false
CLEAN=true
RUN_METHODS="all"
RUN_DIMS="1"

DATA_DIR="data"
ARCHIVE_DIR="results/archive"

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --no-sim              Skip simulations
  --no-plot             Skip postprocessing
  --archive             Move existing data/ into results/archive/
  --clean               Delete existing data/
  --method VALUE        all, gfm, dim, both, or sm (default: all)
  --methods VALUE       Alias for --method
  --dims VALUE          all, 1, 2, or 1,2 for GFM/DIM (default: all)
  --dim VALUE           Alias for --dims
  -h, --help            Show this help

Examples:
  $0 --method gfm --dims 1
  $0 --method dim --dims 2
  $0 --method both --dims 1,2
EOF
}

method_enabled() {
    local method="$1"

    case "$RUN_METHODS" in
        all) return 0 ;;
        sm) [[ "$method" == "sm" ]] ;;
        gfm) [[ "$method" == "gfm" ]] ;;
        dim) [[ "$method" == "dim" ]] ;;
        both|gfm-dim|dim-gfm) [[ "$method" == "gfm" || "$method" == "dim" ]] ;;
        *) echo "[ERROR] Unknown method selection: $RUN_METHODS"; exit 1 ;;
    esac
}

dim_enabled() {
    local dim="$1"
    local dims

    dims=$(echo "$RUN_DIMS" | tr '[:upper:]' '[:lower:]' | tr -d ' ')

    case "$dims" in
        all|both|1,2|2,1) return 0 ;;
        1|1d) [[ "$dim" == "1" ]] ;;
        2|2d) [[ "$dim" == "2" ]] ;;
        *) echo "[ERROR] Unknown dimension selection: $RUN_DIMS"; exit 1 ;;
    esac
}

# -------------------------
# [2] Parse Flags
# -------------------------
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --no-sim) RUN_SIM=false ;;
        --no-plot) RUN_PLOT=false ;;
        --archive) ARCHIVE=true ;;
        --clean) CLEAN=true ;;
        --method|--methods) shift; RUN_METHODS="$1" ;;
        --method=*|--methods=*) RUN_METHODS="${1#*=}" ;;
        --dims|--dim) shift; RUN_DIMS="$1" ;;
        --dims=*|--dim=*) RUN_DIMS="${1#*=}" ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# -------------------------
# [3] Data Handling
# -------------------------
timestamp=$(date +"%Y%m%d_%H%M%S")

if [ -d "$DATA_DIR" ]; then
    if [ "$ARCHIVE" = true ]; then
        echo "[INFO] Archiving existing data..."
        mkdir -p "$ARCHIVE_DIR"
        mv "$DATA_DIR" "${ARCHIVE_DIR}/run_${timestamp}"
    elif [ "$CLEAN" = true ]; then
        echo "[INFO] Deleting existing data..."
        rm -rf "$DATA_DIR"
    else
        echo "[WARNING] Existing data folder detected. No action taken."
    fi
fi

# Ensure fresh directory exists
mkdir -p "$DATA_DIR"

# -------------------------
# [4] Run Simulations
# -------------------------
if [ "$RUN_SIM" = true ]; then
    echo "[INFO] Running simulations..."
    echo "[INFO] Method selection: $RUN_METHODS"
    echo "[INFO] GFM/DIM dimensions: $RUN_DIMS"

    # [4.1] Compile Files
    chmod +x tests/validation/1_SM_Euler_Tests/SM_simulation.sh
    chmod +x tests/validation/2_MM_GFM_Tests/MM_GFM_simulation.sh
    chmod +x tests/validation/3_MM_DIM_Tests/MM_DIM_simulation.sh

    # [4.2] Run Files
    if method_enabled sm; then
        ./tests/validation/1_SM_Euler_Tests/SM_simulation.sh
    fi

    if method_enabled gfm; then
        ./tests/validation/2_MM_GFM_Tests/MM_GFM_simulation.sh --dims "$RUN_DIMS"
    fi

    if method_enabled dim; then
        ./tests/validation/3_MM_DIM_Tests/MM_DIM_simulation.sh --dims "$RUN_DIMS"
    fi

else
    echo "[INFO] Skipping simulations"
fi

# -------------------------
# [5] Run Postprocessing
# -------------------------
if [ "$RUN_PLOT" = true ]; then
    echo "[INFO] Running postprocessing..."
    echo "[INFO] Method selection: $RUN_METHODS"
    echo "[INFO] GFM/DIM dimensions: $RUN_DIMS"

    # [5.1] Compile Files
    chmod +x tests/validation/1_SM_Euler_Tests/SM_graphing.sh
    chmod +x tests/validation/2_MM_GFM_Tests/MM_GFM_graphing.sh
    chmod +x tests/validation/3_MM_DIM_Tests/MM_DIM_graphing.sh
    chmod +x tests/validation/3_MM_DIM_Tests/gfm_dim_comparison.sh

    # [5.2] Run Files
    if method_enabled sm; then
        ./tests/validation/1_SM_Euler_Tests/SM_graphing.sh
    fi

    if method_enabled gfm; then
        ./tests/validation/2_MM_GFM_Tests/MM_GFM_graphing.sh --dims "$RUN_DIMS"
    fi

    if method_enabled dim; then
        ./tests/validation/3_MM_DIM_Tests/MM_DIM_graphing.sh --dims "$RUN_DIMS"
    fi

    if method_enabled gfm && method_enabled dim && dim_enabled 1; then
        ./tests/validation/3_MM_DIM_Tests/gfm_dim_comparison.sh
    fi

else
    echo "[INFO] Skipping postprocessing"
fi

echo "[INFO] Validation pipeline complete."
