#!/bin/bash


# -------------------------
# [1] Defaults
# -------------------------
RUN_SIM=true
RUN_PLOT=true
ARCHIVE=false
CLEAN=false
RUN_METHODS="sm"
RUN_DIMS="all"
RUN_CASES="all"
RUN_CONSERVATION=false
CONSERVATION_INTERVAL=1

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
  --method VALUE        all, gfm, dim, both, sm, or solid (default: all)
  --methods VALUE       Alias for --method
  --dims VALUE          all, 1, 2, or 1,2 for GFM/DIM (default: all)
  --dim VALUE           Alias for --dims
  --case VALUE          all or bubble (default: all)
  --cases VALUE         Alias for --case
  --conservation        Write conservation diagnostics during simulations
  --conservation-interval VALUE
                        Write conservation diagnostics every VALUE timesteps
  -h, --help            Show this help

Examples:
  $0 --method gfm --dims 1
  $0 --method dim --dims 2
  $0 --method both --dims 1,2
  $0 --case bubble --method both
  $0 --method both --dims 1 --conservation
EOF
}

method_enabled() {
    local method="$1"

    case "$RUN_METHODS" in
        all) return 0 ;;
        sm) [[ "$method" == "sm" ]] ;;
        solid) [[ "$method" == "solid" ]] ;;
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

run_validation_stage() {
    local stage_name="$1"
    shift
    local sim_cmd="$1"
    shift
    local plot_cmd="$1"
    shift

    echo "[INFO] ===== $stage_name ====="

    if [ "$RUN_SIM" = true ]; then
        echo "[INFO] Running $stage_name simulations..."
        eval "$sim_cmd"
    else
        echo "[INFO] Skipping $stage_name simulations"
    fi

    if [ "$RUN_PLOT" = true ]; then
        echo "[INFO] Running $stage_name postprocessing..."
        eval "$plot_cmd"
    else
        echo "[INFO] Skipping $stage_name postprocessing"
    fi

    echo "[INFO] Finished $stage_name"
    echo "-------------------------"
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
        --case|--cases) shift; RUN_CASES="$1" ;;
        --case=*|--cases=*) RUN_CASES="${1#*=}" ;;
        --conservation) RUN_CONSERVATION=true ;;
        --conservation-interval) shift; CONSERVATION_INTERVAL="$1" ;;
        --conservation-interval=*) CONSERVATION_INTERVAL="${1#*=}" ;;
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
        echo "[INFO] Deleting generated CSV and plot outputs..."
        rm -rf "$DATA_DIR/csv" "$DATA_DIR/plots"
    else
        echo "[WARNING] Existing data folder detected. No action taken."
    fi
fi

# Ensure fresh generated-output directories exist
mkdir -p "$DATA_DIR/csv" "$DATA_DIR/plots"

# -------------------------
# [4] Run staged simulations and postprocessing
# -------------------------
echo "[INFO] Method selection: $RUN_METHODS"
echo "[INFO] GFM/DIM dimensions: $RUN_DIMS"
echo "[INFO] Conservation diagnostics: $RUN_CONSERVATION"

if [ "$RUN_CONSERVATION" = true ]; then
    export SOLVER_CONSERVATION=1
    export SOLVER_CONSERVATION_INTERVAL="$CONSERVATION_INTERVAL"
else
    unset SOLVER_CONSERVATION
    unset SOLVER_CONSERVATION_INTERVAL
fi

# [4.1] Make stage scripts executable
chmod +x tests/validation/1_SM_Euler_Tests/SM_simulation.sh
chmod +x tests/validation/1_SM_Euler_Tests/SM_graphing.sh
chmod +x tests/validation/2_MM_GFM_Tests/MM_GFM_simulation.sh
chmod +x tests/validation/2_MM_GFM_Tests/MM_GFM_graphing.sh
chmod +x tests/validation/3_MM_DIM_Tests/MM_DIM_simulation.sh
chmod +x tests/validation/3_MM_DIM_Tests/MM_DIM_graphing.sh
chmod +x tests/validation/3_MM_DIM_Tests/gfm_dim_comparison.sh
chmod +x tests/validation/5_Elastoplastic_Tests/EP_simulation.sh
chmod +x tests/validation/5_Elastoplastic_Tests/EP_graphing.sh
chmod +x tests/bubble_collapse_tests.sh

if [ "$RUN_CASES" = "bubble" ]; then
    case "$RUN_METHODS" in
        gfm|GFM) bubble_mode="gfm" ;;
        dim|DIM) bubble_mode="dim" ;;
        both|all|gfm-dim|dim-gfm) bubble_mode="both" ;;
        *) echo "[ERROR] Bubble collapse supports --method gfm, dim, both, or all"; exit 1 ;;
    esac

    echo "[INFO] Running bubble collapse only: $bubble_mode"
    CORES="${CORES:-6}" ./tests/bubble_collapse_tests.sh "$bubble_mode"
    echo "[INFO] Bubble collapse pipeline complete."
    exit 0
elif [ "$RUN_CASES" != "all" ]; then
    echo "[ERROR] Unknown case selection: $RUN_CASES"
    exit 1
fi

# [4.2] Run each stage and postprocess before moving on
if method_enabled sm; then
    run_validation_stage \
        "SM all dimensions" \
        "./tests/validation/1_SM_Euler_Tests/SM_simulation.sh" \
        "./tests/validation/1_SM_Euler_Tests/SM_graphing.sh"
fi

if method_enabled solid; then
    run_validation_stage \
        "Solid elastoplastic" \
        "./tests/validation/5_Elastoplastic_Tests/EP_simulation.sh" \
        "./tests/validation/5_Elastoplastic_Tests/EP_graphing.sh"
fi

if method_enabled dim && dim_enabled 1; then
    run_validation_stage \
        "DIM 1D" \
        "./tests/validation/3_MM_DIM_Tests/MM_DIM_simulation.sh --dims 1" \
        "./tests/validation/3_MM_DIM_Tests/MM_DIM_graphing.sh --dims 1"
fi

if method_enabled gfm && dim_enabled 1; then
    run_validation_stage \
        "GFM 1D" \
        "./tests/validation/2_MM_GFM_Tests/MM_GFM_simulation.sh --dims 1" \
        "./tests/validation/2_MM_GFM_Tests/MM_GFM_graphing.sh --dims 1"
fi

if method_enabled gfm && dim_enabled 2 && ! dim_enabled 1 && [ "$RUN_SIM" = true ]; then
    echo "[INFO] Running GFM 1D baseline needed for 2D reduction validation..."
    ./tests/validation/2_MM_GFM_Tests/MM_GFM_simulation.sh --dims 1
    echo "-------------------------"
fi

if method_enabled dim && dim_enabled 2 && ! dim_enabled 1 && [ "$RUN_SIM" = true ]; then
    echo "[INFO] Running DIM 1D baseline needed for 2D reduction validation..."
    ./tests/validation/3_MM_DIM_Tests/MM_DIM_simulation.sh --dims 1
    echo "-------------------------"
fi

if method_enabled gfm && dim_enabled 2; then
    run_validation_stage \
        "GFM 2D" \
        "./tests/validation/2_MM_GFM_Tests/MM_GFM_simulation.sh --dims 2" \
        "./tests/validation/2_MM_GFM_Tests/MM_GFM_graphing.sh --dims 2"
fi

if method_enabled dim && dim_enabled 2; then
    run_validation_stage \
        "DIM 2D" \
        "./tests/validation/3_MM_DIM_Tests/MM_DIM_simulation.sh --dims 2" \
        "./tests/validation/3_MM_DIM_Tests/MM_DIM_graphing.sh --dims 2"
fi

if method_enabled gfm && method_enabled dim && [ "$RUN_PLOT" = true ]; then
    echo "[INFO] Running GFM vs DIM comparison after required data stages..."
    ./tests/validation/3_MM_DIM_Tests/gfm_dim_comparison.sh
    echo "-------------------------"
fi

echo "[INFO] Validation pipeline complete."
