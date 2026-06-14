"""Plot 1D fluid-solid rGFM paper cases with the shared MM plot style."""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_style import configure_profile_axis, plot_profile, save_figure


@dataclass(frozen=True)
class PaperCase:
    name: st
    csv_path: Path
    rho_l: float
    u_l: float
    p_l: float
    gamma: float
    p_inf: float
    rho_s: float
    u_s: float
    sigma_nn_s: float
    bulk: float
    shear: float
    interface_x: float = 5.0
    final_time: float = 4.45e-3
    domain_min: float = 0.0
    domain_max: float = 10.0


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def paper_cases(root: Path) -> dict[str, PaperCase]:
    data_root = root / "data" / "csv" / "fluid_solid"
    bulk_aisi = 1.651e6
    shear_aisi = 8.383e5
    return {
        "case4_1": PaperCase(
            name="case4_1_gas_solid_rgfm",
            csv_path=data_root / "case4_1_gas_solid_rgfm" / "case4_1_gas_solid_rgfm_N2200.csv",
            rho_l=0.05,
            u_l=50.0,
            p_l=10000.0,
            gamma=1.4,
            p_inf=0.0,
            rho_s=7.7,
            u_s=0.0,
            sigma_nn_s=-1.0,
            bulk=bulk_aisi,
            shear=shear_aisi,
        ),
        "case4_2": PaperCase(
            name="case4_2_water_solid_rgfm",
            csv_path=data_root / "case4_2_water_solid_rgfm" / "case4_2_water_solid_rgfm_N2200.csv",
            rho_l=1.0,
            u_l=30.0,
            p_l=25000.0,
            gamma=7.15,
            p_inf=3300.0,
            rho_s=7.7,
            u_s=-30.0,
            sigma_nn_s=-1.0,
            bulk=bulk_aisi,
            shear=shear_aisi,
        ),
    }


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Run the fluid-solid paper case first.")
    with csv_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def sound_speed(rho: float, p: float, gamma: float, p_inf: float) -> float:
    return math.sqrt(gamma * (p + p_inf) / rho)


def density_from_isentrope(p: float, rho0: float, p0: float, gamma: float, p_inf: float) -> float:
    invariant = (p0 + p_inf) / (rho0**gamma)
    return ((p + p_inf) / invariant) ** (1.0 / gamma)


def shock_impedance(p: float, rho0: float, p0: float, gamma: float, p_inf: float) -> float:
    if abs(p - p0) <= 1.0e-12 * max(1.0, abs(p0)):
        return rho0 * sound_speed(rho0, p0, gamma, p_inf)
    rho = density_from_isentrope(p, rho0, p0, gamma, p_inf)
    return math.sqrt(max((p - p0) / (1.0 / rho0 - 1.0 / rho), 1.0e-30))


def interface_state(case: PaperCase) -> dict[str, float]:
    c_s = math.sqrt((case.bulk + 4.0 * case.shear / 3.0) / case.rho_s)
    z_s = case.rho_s * c_s

    def solid_velocity(p: float) -> float:
        return case.u_s + (p + case.sigma_nn_s) / z_s

    def residual(p: float) -> float:
        w_l = shock_impedance(p, case.rho_l, case.p_l, case.gamma, case.p_inf)
        fluid_velocity = case.u_l - (p - case.p_l) / w_l
        return fluid_velocity - solid_velocity(p)

    lo = max(case.p_l, 1.0e-12)
    hi = max(2.0 * case.p_l, case.p_l + 1.0)
    while residual(hi) > 0.0:
        hi *= 2.0
        if hi > 1.0e12:
            raise RuntimeError("Failed to bracket fluid-solid reference pressure")

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0.0:
            lo = mid
        else:
            hi = mid

    p_star = 0.5 * (lo + hi)
    u_star = solid_velocity(p_star)
    w_star = shock_impedance(p_star, case.rho_l, case.p_l, case.gamma, case.p_inf)

    alpha2 = (case.bulk + 4.0 * case.shear / 3.0) / case.rho_s
    beta2 = case.shear / case.rho_s
    sigma_tt_star = ((alpha2 - 2.0 * beta2) / alpha2) * (-p_star)

    return {
        "u": u_star,
        "p": p_star,
        "sigma_nn": -p_star,
        "sigma_tt": sigma_tt_star,
        "fluid_front": case.interface_x + (case.u_l - w_star / case.rho_l) * case.final_time,
        "solid_front": case.interface_x + c_s * case.final_time,
    }


def exact_step_points(case: PaperCase, field: str) -> tuple[np.ndarray, np.ndarray]:
    state = interface_state(case)
    xf = max(case.domain_min, state["fluid_front"])
    xs = min(case.domain_max, state["solid_front"])
    xi = case.interface_x

    if field == "u":
        vals = (case.u_l, state["u"], state["u"], case.u_s)
    elif field == "normal":
        vals = (case.p_l, state["p"], -state["sigma_nn"], -case.sigma_nn_s)
    elif field == "stress_yy":
        vals = (0.0, 0.0, state["sigma_tt"], 0.0)
    elif field == "stress_xy":
        vals = (0.0, 0.0, 0.0, 0.0)
    else:
        raise ValueError(f"Unknown exact field {field}")

    points = [
        (case.domain_min, vals[0]), (xf, vals[0]), (xf, vals[1]),
        (xi, vals[1]), (xi, vals[2]), (xs, vals[2]),
        (xs, vals[3]), (case.domain_max, vals[3]),
    ]
    return np.array([p[0] for p in points]), np.array([p[1] for p in points])


def numeric_series(rows: list[dict[str, str]], field: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([float(row["x"]) for row in rows])
    values = []
    for row in rows:
        material = row["material"]
        if field == "u":
            values.append(float(row["u"]))
        elif field == "normal":
            values.append(float(row["p"]) if material == "fluid" else -float(row["sigma11"]))
        elif field == "stress_yy":
            values.append(0.0 if material == "fluid" else float(row["sigma22"]))
        elif field == "stress_xy":
            values.append(0.0)
        else:
            raise ValueError(f"Unknown numeric field {field}")
    return x, np.array(values)


def cell_label(csv_path: Path) -> str:
    match = re.search(r"_N(\d+)\.csv$", csv_path.name)
    if match:
        return f"{match.group(1)} cells"
    return "Numerical"


def configure_math_axis(ax, ylabel: str) -> None:
    configure_profile_axis(ax, None, x_label=r"$x$", show_title=False)
    ax.set_ylabel(ylabel)


def plot_panel(ax, case: PaperCase, rows: list[dict[str, str]], field: str, ylabel: str) -> None:
    exact_x, exact_y = exact_step_points(case, field)
    x, y = numeric_series(rows, field)
    plot_profile(ax, exact_x, exact_y, "Exact", index=0)
    plot_profile(ax, x, y, cell_label(case.csv_path), index=0)
    configure_math_axis(ax, ylabel)
    ax.set_xlim(case.domain_min - 0.08 * (case.domain_max - case.domain_min),
                case.domain_max + 0.08 * (case.domain_max - case.domain_min))


def render_pair_plot(case: PaperCase, rows, output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    plot_panel(axes[0], case, rows, "u", r"Velocity, $u$")
    plot_panel(axes[1], case, rows, "normal", r"$p$, $-\sigma_{11}$")
    fig.tight_layout()
    path = output_dir / f"{case.name}_paper_style_profiles.png"
    save_figure(fig, path)
    return path


def render_stress_plot(case: PaperCase, rows, output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    plot_panel(axes[0], case, rows, "stress_yy", r"$\sigma_{22}$")
    plot_panel(axes[1], case, rows, "stress_xy", r"$\sigma_{12}$")
    fig.tight_layout()
    path = output_dir / f"{case.name}_paper_style_stresses.png"
    save_figure(fig, path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["case4_1", "case4_2", "all"], default="all")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    root = repo_root()
    output_dir = args.output_dir or root / "data" / "plots" / "fluid_solid"
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = paper_cases(root)
    selected = cases.values() if args.case == "all" else [cases[args.case]]

    for case in selected:
        rows = read_rows(case.csv_path)
        print(render_pair_plot(case, rows, output_dir))
        print(render_stress_plot(case, rows, output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
