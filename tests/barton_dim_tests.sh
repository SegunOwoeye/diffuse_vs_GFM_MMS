#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

g++ -std=c++20 -O2 -I. -DAPP_DIM=1 src/app/multimaterial_main.cpp -o multimaterial_main_1d
g++ -std=c++20 -O2 -I. -DAPP_DIM=2 src/app/multimaterial_main.cpp -o multimaterial_main_2d

./multimaterial_main_1d configs/barton_dim/pure_fluid_1d.txt
./multimaterial_main_1d configs/barton_dim/pure_solid_1d.txt
./multimaterial_main_1d configs/barton_dim/flying_plate_1d.txt
./multimaterial_main_2d configs/barton_dim/projectile_2d.txt
./multimaterial_main_2d configs/solid/MM/DIM/2D_validation/Favrie_Gavrilyuk_2012/copper_projectile_plate_air_barton_dim_pair.txt

python3 - <<'PY'
import csv
from pathlib import Path

paths = {
    "pure fluid": Path("results/barton_dim/pure_fluid_1d_t1.000000em04.csv"),
    "pure solid": Path("results/barton_dim/pure_solid_1d_t2.000000em07.csv"),
    "flying plate": Path("results/barton_dim/flying_plate_1d_t1.000000em06.csv"),
    "projectile": Path("results/barton_dim/projectile_2d_t2.000000em08.csv"),
}
for name, path in paths.items():
    if not path.exists():
        raise SystemExit(f"{name}: missing output {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{name}: empty output")
    for row in rows:
        for key in ("rho", "total_energy", "p_solid", "p_fluid"):
            value = float(row[key])
            if value != value or value in (float("inf"), float("-inf")):
                raise SystemExit(f"{name}: non-finite {key}")
        alpha = float(row["alpha_solid"])
        if not 0.0 <= alpha <= 1.0:
            raise SystemExit(f"{name}: alpha_solid outside [0, 1]")

pure_fluid = list(csv.DictReader(paths["pure fluid"].open(newline="")))
if any(abs(float(row["alpha_solid"])) > 1.0e-12 for row in pure_fluid):
    raise SystemExit("pure fluid: solid volume fraction changed")
if max(abs(float(row["rho"]) - 1.0) for row in pure_fluid) > 1.0e-9:
    raise SystemExit("pure fluid: uniform density was not preserved")

pure_solid = list(csv.DictReader(paths["pure solid"].open(newline="")))
if any(abs(float(row["alpha_solid"]) - 1.0) > 1.0e-12 for row in pure_solid):
    raise SystemExit("pure solid: fluid volume fraction appeared")
if max(abs(float(row["rho"]) - 2700.0) for row in pure_solid) > 1.0e-6:
    raise SystemExit("pure solid: uniform density was not preserved")

flying_plate = list(csv.DictReader(paths["flying plate"].open(newline="")))
velocities = [float(row["u0"]) for row in flying_plate]
if max(velocities) < 500.0 or min(velocities) > 100.0:
    raise SystemExit("flying plate: initial impact states were not retained")

projectile = list(csv.DictReader(paths["projectile"].open(newline="")))
alphas = [float(row["alpha_solid"]) for row in projectile]
if max(alphas) < 0.9 or min(alphas) > 0.1:
    raise SystemExit("projectile: both solid and fluid cells are required")

rgfm_compatible = Path(
    "data/csv/solid/MM/DIM/2D_validation/Favrie_Gavrilyuk_2012/"
    "favrie_gavrilyuk_2012_copper_projectile_plate_air_barton_dim_pair_t2.000000em08.csv"
)
with rgfm_compatible.open(newline="") as handle:
    rgfm_rows = list(csv.DictReader(handle))
if not rgfm_rows or {"fluid", "solid"} - {row["material"] for row in rgfm_rows}:
    raise SystemExit("rGFM-compatible Barton-DIM output is missing a phase")
if "sigma_nn" not in rgfm_rows[0] or "vn" not in rgfm_rows[0]:
    raise SystemExit("rGFM-compatible Barton-DIM output is missing required fields")
print("Barton-DIM verification passed")
PY
