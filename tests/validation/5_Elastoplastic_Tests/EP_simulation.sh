#!/bin/bash
set -euo pipefail

run_full=false
validate_damage=false
validate_fluid_solid=false
run_fluid_solid_paper=false
run_fluid_solid_case43=false

usage() {
    cat <<EOF
Usage: $0 [--full] [--validate-damage] [--validate-fluid-solid-rgfm] [--fluid-solid-paper] [--fluid-solid-case43]

Options:
  --full                       Also run the Barton 2D 250x250 and 3D 100x100x100 paper-grid cases.
  --validate-damage            Run deterministic Johnson-Cook damage law validation.
  --validate-fluid-solid-rgfm  Run the 1D fluid-solid rGFM interface-state validation.
  --fluid-solid-paper          Run the 1D fluid-solid paper Case 4.1 and Case 4.2 configs.
  --fluid-solid-case43         Run the rotated 2D fluid-solid paper Case 4.3 config.
EOF
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --full) run_full=true ;;
        --validate-damage) validate_damage=true ;;
        --validate-fluid-solid-rgfm) validate_fluid_solid=true ;;
        --fluid-solid-paper) run_fluid_solid_paper=true ;;
        --fluid-solid-case43) run_fluid_solid_case43=true ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] Unknown option: $1"; usage; exit 1 ;;
    esac
    shift
done

cores="${CORES_SOLID:-${CORES:-${OMP_NUM_THREADS:-8}}}"

echo "[SOLID] Compiling solid_main with OpenMP..."
g++ -std=c++17 -O3 -fopenmp -I. src/app/solid_main.cpp -o solid_main

if [ "$validate_damage" = true ]; then
    echo "[SOLID][Barton] Validating Johnson-Cook damage law"
    ./solid_main --validate-johnson-cook-damage
fi

if [ "$validate_fluid_solid" = true ]; then
    echo "[SOLID][Barton] Validating fluid-solid rGFM interface states"
    ./solid_main --validate-fluid-solid-rgfm
fi

if [ "$run_fluid_solid_paper" = true ]; then
    echo "[SOLID][FSI] Running Liu-Chowdhury-Khoo Case 4.1 gas-solid"
    OMP_NUM_THREADS="$cores" ./solid_main configs/solid/FluidSolid_RGFM_paper/case4_1_gas_solid.txt

    echo "[SOLID][FSI] Running Liu-Chowdhury-Khoo Case 4.2 water-solid"
    OMP_NUM_THREADS="$cores" ./solid_main configs/solid/FluidSolid_RGFM_paper/case4_2_water_solid.txt
fi

if [ "$run_fluid_solid_case43" = true ]; then
    echo "[SOLID][FSI] Running Liu-Chowdhury-Khoo Case 4.3 rotated water-solid"
    OMP_NUM_THREADS="$cores" ./solid_main configs/solid/FluidSolid_RGFM_paper/case4_3_rotated_water_solid.txt
fi

echo "[SOLID][Barton 1D] Running test1: Section 6.1, 0.8 km/s flyer plate"
OMP_NUM_THREADS="$cores" ./solid_main configs/solid/Barton_1D_validation/test1.txt

echo "[SOLID][Barton 1D] Running test2: Section 6.1, 2.0 km/s flyer plate"
OMP_NUM_THREADS="$cores" ./solid_main configs/solid/Barton_1D_validation/test2.txt

if [ "$run_full" = true ]; then
    echo "[SOLID][Barton 2D] Running test1: radial-pressure tensor, 250x250 paper grid"
    OMP_NUM_THREADS="$cores" ./solid_main configs/solid/Barton_2D_validation/test1.txt

    echo "[SOLID][Barton 3D] Running test1: spherical-pressure tensor, 100x100x100 paper grid"
    OMP_NUM_THREADS="$cores" ./solid_main configs/solid/Barton_3D_validation/test1.txt
fi

echo "[SOLID] Simulations complete."
