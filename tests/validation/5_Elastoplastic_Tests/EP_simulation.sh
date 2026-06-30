#!/bin/bash
set -euo pipefail

run_full=false
validate_damage=false
validate_fluid_solid=false
run_fluid_solid_paper=false
run_fluid_solid_case43=false
run_favrie_comparison=false
run_primary=false
run_legacy_barton=false
run_barton_dim_case41=false
run_barton_dim_favrie_full=false

usage() {
    cat <<EOF
Usage: $0 [--primary] [--full] [--legacy-barton] [--validate-damage] [--validate-fluid-solid-rgfm] [--fluid-solid-paper] [--fluid-solid-case43] [--favrie-comparison] [--barton-dim-case41] [--barton-dim-favrie-full]

Options:
  --primary                    Run the mapped primary 1D rGFM case and Favrie-Gavrilyuk comparisons.
  --full                       Also run the Barton 2D 250x250 and 3D 100x100x100 paper-grid cases.
  --legacy-barton              Run the historical Barton 1D test1 and test2 cases.
  --validate-damage            Run deterministic Johnson-Cook damage law validation.
  --validate-fluid-solid-rgfm  Run the 1D fluid-solid rGFM interface-state validation.
  --fluid-solid-paper          Run the 1D fluid-solid paper Case 4.1 and Case 4.2 configs.
  --fluid-solid-case43         Run the rotated 2D fluid-solid paper Case 4.3 config.
  --favrie-comparison          Run matched Favrie-Gavrilyuk rGFM and Barton-DIM pairs, then compare outputs.
  --barton-dim-case41          Run the Barton-DIM counterpart of rGFM 1D Case 4.1.
  --barton-dim-favrie-full     Run the canonical 2D Barton-DIM Favrie-Gavrilyuk analysis case.
EOF
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --full) run_full=true ;;
        --primary) run_primary=true ;;
        --legacy-barton) run_legacy_barton=true ;;
        --validate-damage) validate_damage=true ;;
        --validate-fluid-solid-rgfm) validate_fluid_solid=true ;;
        --fluid-solid-paper) run_fluid_solid_paper=true ;;
        --fluid-solid-case43) run_fluid_solid_case43=true ;;
        --favrie-comparison) run_favrie_comparison=true ;;
        --barton-dim-case41) run_barton_dim_case41=true ;;
        --barton-dim-favrie-full) run_barton_dim_favrie_full=true ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] Unknown option: $1"; usage; exit 1 ;;
    esac
    shift
done

if [ "$run_primary" = true ]; then
    run_favrie_comparison=true
fi

cores="${CORES_SOLID:-${CORES:-${OMP_NUM_THREADS:-8}}}"
dim_1d_built=false
dim_2d_built=false

build_barton_dim_1d() {
    if [ "$dim_1d_built" = false ]; then
        echo "[SOLID] Compiling unified 1D multimaterial runner for Barton-DIM"
        g++ -std=c++20 -O3 -fopenmp -I. -DAPP_DIM=1 src/app/multimaterial_main.cpp -o multimaterial_main_1d
        dim_1d_built=true
    fi
}

build_barton_dim_2d() {
    if [ "$dim_2d_built" = false ]; then
        echo "[SOLID] Compiling unified 2D multimaterial runner for Barton-DIM"
        g++ -std=c++20 -O3 -fopenmp -I. -DAPP_DIM=2 src/app/multimaterial_main.cpp -o multimaterial_main_2d
        dim_2d_built=true
    fi
}

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

if [ "$run_primary" = true ]; then
    echo "[SOLID][PRIMARY][rGFM 1D] Running Liu-Chowdhury-Khoo Case 4.1 gas-solid"
    OMP_NUM_THREADS="$cores" ./solid_main configs/solid/FluidSolid_RGFM_paper/case4_1_gas_solid.txt
fi

if [ "$run_barton_dim_case41" = true ]; then
    build_barton_dim_1d
    echo "[SOLID][Barton-DIM 1D] Running Liu-Chowdhury-Khoo Case 4.1 counterpart"
    OMP_NUM_THREADS="$cores" ./multimaterial_main_1d \
        configs/solid/MM/DIM/1D_validation/Liu_Chowdhury_Khoo_2011/case4_1_gas_solid_barton_dim.txt
fi

if [ "$run_barton_dim_favrie_full" = true ]; then
    build_barton_dim_2d
    echo "[SOLID][Barton-DIM 2D] Running Favrie-Gavrilyuk analysis case"
    OMP_NUM_THREADS="$cores" ./multimaterial_main_2d \
        configs/solid/MM/DIM/2D_validation/Favrie_Gavrilyuk_2012/copper_projectile_plate_air_barton_dim.txt
fi

if [ "$run_fluid_solid_case43" = true ]; then
    echo "[SOLID][FSI] Running Liu-Chowdhury-Khoo Case 4.3 rotated water-solid"
    OMP_NUM_THREADS="$cores" ./solid_main configs/solid/FluidSolid_RGFM_paper/case4_3_rotated_water_solid.txt
fi

if [ "$run_favrie_comparison" = true ]; then
    if [ ! -x ".venv/bin/python" ]; then
        echo "[ERROR] .venv/bin/python is required for Favrie-Gavrilyuk comparison plotting"
        exit 1
    fi
    rgfm_config="configs/solid/Favrie_Gavrilyuk_2012/Comparison_2D_validation/copper_projectile_plate_air_rgfm_barton_pair.txt"
    dim_config="configs/solid/Favrie_Gavrilyuk_2012/Comparison_2D_validation/copper_projectile_plate_air_barton_dim_pair.txt"
    rgfm_csv="data/csv/fluid_solid/Favrie_Gavrilyuk_2012/Comparison_2D_validation/favrie_gavrilyuk_2012_copper_projectile_plate_air_rgfm_barton_pair/favrie_gavrilyuk_2012_copper_projectile_plate_air_rgfm_barton_pair_N16x16.csv"
    dim_csv="data/csv/fluid_solid/Favrie_Gavrilyuk_2012/Comparison_2D_validation/favrie_gavrilyuk_2012_copper_projectile_plate_air_barton_dim_pair_t2.000000em08.csv"
    compare_dir="data/plots/fluid_solid/Favrie_Gavrilyuk_2012/RGFM_DIM_comparison_pair"
    sm_1d_config="configs/solid/Favrie_Gavrilyuk_2012/Comparison_1D_validation/wilkins_flying_plate_sm_pair.txt"
    dim_1d_config="configs/solid/Favrie_Gavrilyuk_2012/Comparison_1D_validation/wilkins_flying_plate_barton_dim_pair.txt"
    sm_1d_csv="data/csv/solid/Favrie_Gavrilyuk_2012/Comparison_1D_validation/favrie_gavrilyuk_2012_wilkins_sm_pair/favrie_gavrilyuk_2012_wilkins_sm_pair_t0p500us_N200.csv"
    dim_1d_csv="data/csv/solid/Favrie_Gavrilyuk_2012/Comparison_1D_validation/favrie_gavrilyuk_2012_wilkins_barton_dim_pair_t5.000000em07.csv"
    compare_1d_dir="data/plots/solid/Favrie_Gavrilyuk_2012/SM_DIM_comparison_pair"

    build_barton_dim_1d
    build_barton_dim_2d

    echo "[SOLID][Favrie-Gavrilyuk] Running matched 1D Wilkins SM case"
    OMP_NUM_THREADS="$cores" ./solid_main "$sm_1d_config"

    echo "[SOLID][Favrie-Gavrilyuk] Running matched 1D Wilkins Barton-DIM case"
    OMP_NUM_THREADS="$cores" ./multimaterial_main_1d "$dim_1d_config"

    echo "[SOLID][Favrie-Gavrilyuk] Comparing 1D Wilkins output fields"
    .venv/bin/python src/graphing/compare_favrie_gavrilyuk_2012_1d.py \
        --sm "$sm_1d_csv" \
        --barton-dim "$dim_1d_csv" \
        --output-dir "$compare_1d_dir"

    echo "[SOLID][Favrie-Gavrilyuk] Running matched rGFM pair"
    OMP_NUM_THREADS="$cores" ./solid_main "$rgfm_config"

    echo "[SOLID][Favrie-Gavrilyuk] Running matched Barton-DIM pair"
    OMP_NUM_THREADS="$cores" ./multimaterial_main_2d "$dim_config"

    echo "[SOLID][Favrie-Gavrilyuk] Comparing matched output fields"
    .venv/bin/python src/graphing/compare_favrie_gavrilyuk_2012.py \
        --rgfm "$rgfm_csv" \
        --barton-dim "$dim_csv" \
        --output-dir "$compare_dir"
fi

if [ "$run_legacy_barton" = true ]; then
    echo "[SOLID][LEGACY][Barton 1D] Running test1: Section 6.1, 0.8 km/s flyer plate"
    OMP_NUM_THREADS="$cores" ./solid_main configs/solid/Barton_1D_validation/test1.txt

    echo "[SOLID][LEGACY][Barton 1D] Running test2: Section 6.1, 2.0 km/s flyer plate"
    OMP_NUM_THREADS="$cores" ./solid_main configs/solid/Barton_1D_validation/test2.txt
fi

if [ "$run_full" = true ]; then
    echo "[SOLID][Barton 2D] Running test1: radial-pressure tensor, 250x250 paper grid"
    OMP_NUM_THREADS="$cores" ./solid_main configs/solid/Barton_2D_validation/test1.txt

    echo "[SOLID][Barton 3D] Running test1: spherical-pressure tensor, 100x100x100 paper grid"
    OMP_NUM_THREADS="$cores" ./solid_main configs/solid/Barton_3D_validation/test1.txt
fi

echo "[SOLID] Simulations complete."
