#!/bin/bash


# -------------------------
# [1] Defaults
# -------------------------
RUN_SIM=false
RUN_PLOT=true
ARCHIVE=false
CLEAN=false

DATA_DIR="data"
ARCHIVE_DIR="results/archive"

# -------------------------
# [2] Parse Flags
# -------------------------
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --no-sim) RUN_SIM=false ;;
        --no-plot) RUN_PLOT=false ;;
        --archive) ARCHIVE=true ;;
        --clean) CLEAN=true ;;
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

    # [4.1] Compile Files
    chmod +x tests/validation/1_SM_Euler_Tests/SM_simulation.sh
    chmod +x tests/validation/2_MM_GFM_Tests/MM_GFM_simulation.sh
    chmod +x tests/validation/3_MM_DIM_Tests/MM_DIM_simulation.sh

    # [4.2] Run Files
    ./tests/validation/1_SM_Euler_Tests/SM_simulation.sh
    ./tests/validation/2_MM_GFM_Tests/MM_GFM_simulation.sh
    ./tests/validation/3_MM_DIM_Tests/MM_DIM_simulation.sh

else
    echo "[INFO] Skipping simulations"
fi

# -------------------------
# [5] Run Postprocessing
# -------------------------
if [ "$RUN_PLOT" = true ]; then
    echo "[INFO] Running postprocessing..."

    # [5.1] Compile Files
    chmod +x tests/validation/1_SM_Euler_Tests/SM_graphing.sh
    chmod +x tests/validation/2_MM_GFM_Tests/MM_GFM_graphing.sh
    chmod +x tests/validation/3_MM_DIM_Tests/MM_DIM_graphing.sh
    chmod +x tests/validation/3_MM_DIM_Tests/gfm_dim_comparison.sh

    # [5.2] Run Files
    ./tests/validation/1_SM_Euler_Tests/SM_graphing.sh
    ./tests/validation/2_MM_GFM_Tests/MM_GFM_graphing.sh
    ./tests/validation/3_MM_DIM_Tests/MM_DIM_graphing.sh
    ./tests/validation/3_MM_DIM_Tests/gfm_dim_comparison.sh

else
    echo "[INFO] Skipping postprocessing"
fi

echo "[INFO] Validation pipeline complete."