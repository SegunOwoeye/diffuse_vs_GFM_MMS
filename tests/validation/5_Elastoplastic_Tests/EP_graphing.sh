#!/bin/bash
set -euo pipefail

if [ ! -d ".venv" ]; then
    echo "[ERROR] .venv not found"
    exit 1
fi

usage() {
    cat <<EOF
Usage: $0
EOF
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] Unknown option: $1"; usage; exit 1 ;;
    esac
    shift
done

source .venv/bin/activate

plot_barton_1d() {
    local prefix="$1"
    local ymin="$2"
    local out_dir="data/csv/solid/${prefix}"

    if [ ! -d "$out_dir" ]; then
        echo "[SOLID][WARN] Missing output directory: $out_dir"
        return
    fi

    python src/graphing/plot_solid_1d.py \
        "$out_dir/${prefix}_t0p500us_N500.csv" \
        "$out_dir/${prefix}_t1p000us_N500.csv" \
        "$out_dir/${prefix}_t2p000us_N500.csv" \
        "$out_dir/${prefix}_t3p000us_N500.csv" \
        "$out_dir/${prefix}_t4p000us_N500.csv" \
        "$out_dir/${prefix}_t5p000us_N500.csv" \
        --wilkins-stress-grid \
        --x-unit cm \
        --stress-ymin "$ymin" \
        --stress-ymax 1 \
        --out "$out_dir/${prefix}_stress_grid.png"
}

plot_barton_2d() {
    local prefix="$1"
    local cells="$2"
    local radial="$3"
    local out_dir="data/csv/solid/${prefix}"

    if [ ! -f "$out_dir/${prefix}_N${cells}.csv" ]; then
        echo "[SOLID][WARN] Missing 2D output: $out_dir/${prefix}_N${cells}.csv"
        return
    fi

    python src/graphing/plot_solid_2d.py \
        "$out_dir/${prefix}_N${cells}.csv" \
        --reference "$out_dir/${prefix}_cylindrical_N${radial}.csv" \
        --barton-radial-reference \
        --out "$out_dir/${prefix}_figure6.png"
}

plot_barton_3d() {
    local prefix="$1"
    local cells="$2"
    local radial="$3"
    local out_dir="data/csv/solid/${prefix}"

    if [ ! -f "$out_dir/${prefix}_N${cells}.csv" ]; then
        echo "[SOLID][WARN] Missing 3D output: $out_dir/${prefix}_N${cells}.csv"
        return
    fi

    python src/graphing/plot_solid_2d.py \
        "$out_dir/${prefix}_N${cells}.csv" \
        --reference "$out_dir/${prefix}_spherical_N${radial}.csv" \
        --barton-radial-reference \
        --scatter-label "3D" \
        --out "$out_dir/${prefix}_figure7.png"
}

plot_barton_material_points() {
    local prefix="$1"
    local out_dir="data/csv/solid/${prefix}"
    local initial="$out_dir/${prefix}_material_points_initial.vtp"
    local final="$out_dir/${prefix}_material_points_final.vtp"

    if [ ! -f "$initial" ] || [ ! -f "$final" ]; then
        echo "[SOLID][WARN] Missing material-point output for: $prefix"
        return
    fi

    python src/graphing/plot_solid_material_points.py \
        "$initial" \
        "$final" \
        --max-points 12000 \
        --scale 20 \
        --out "$out_dir/${prefix}_material_points.png"
}

plot_barton_damage() {
    local prefix="$1"
    local cells="$2"
    local out_dir="data/csv/solid/${prefix}"
    local csv="$out_dir/${prefix}_N${cells}.csv"

    if [ ! -f "$csv" ]; then
        echo "[SOLID][WARN] Missing damage output: $csv"
        return
    fi

    python src/graphing/plot_solid_damage.py \
        "$csv" \
        --out "$out_dir/${prefix}_damage.png"
}

echo "[SOLID][Graphing] Barton 1D test1"
plot_barton_1d "barton_test1_wilkins_flying_plate_0p8_1d" -7

echo "[SOLID][Graphing] Barton 1D test2"
plot_barton_1d "barton_test2_wilkins_flying_plate_2p0_1d" -20

echo "[SOLID][Graphing] Barton 2D test1, if available"
plot_barton_2d "barton_test1_tensor_radial_pressure_2d" "250x250" 500

echo "[SOLID][Graphing] Barton 3D test1, if available"
plot_barton_3d "barton_test1_tensor_spherical_pressure_3d" "100x100x100" 500

deactivate
echo "[SOLID] Graphing complete."
