#!/usr/bin/env python3
"""Generate exact/reference CSVs for the Fedkiw multimaterial tests.

The quantitative runner prefers these generated references over old digitized
curves because they preserve each side's material parameters. The script reads
the existing validation configs, compiles the small exact Riemann driver, and
writes both raw driver output and report-ready field names.
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


TEST_NAMES = {
    "test1": "FedkiwA",
    "test2": "FedkiwB",
    "test3": "FedkiwC",
    "test4": "FedkiwD1",
    "test5": "FedkiwD2",
}


@dataclass(frozen=True)
class Material:
    material_id: int
    kind: str
    gamma: float
    p_inf: float = 0.0


@dataclass(frozen=True)
class Region:
    lower: float
    upper: float
    rho: float
    u: float
    p: float
    material_id: int


@dataclass(frozen=True)
class ConfigCase:
    x_min: float
    x_max: float
    tfinal: float
    materials: dict[int, Material]
    regions: list[Region]


def strip_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def parse_scalar_list(value: str) -> list[float]:
    text = value.strip()
    if not (text.startswith("[") and text.endswith("]")):
        raise ValueError(f"expected bracketed list, got {value!r}")
    return [float(item.strip()) for item in text[1:-1].split(",") if item.strip()]


def parse_key_values(text: str) -> dict[str, float | int | list[float]]:
    values: dict[str, float | int | list[float]] = {}
    for match in re.finditer(r"(\w+)=((?:\[[^\]]*\])|(?:[^,\s]+))", text):
        key = match.group(1)
        raw = match.group(2)
        if raw.startswith("["):
            values[key] = parse_scalar_list(raw)
        elif key == "material":
            values[key] = int(raw)
        else:
            values[key] = float(raw)
    return values


def parse_material(text: str) -> Material:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) < 3:
        raise ValueError(f"bad material line: {text}")
    material_id = int(parts[0])
    kind = parts[1]
    params: dict[str, float] = {}
    for part in parts[2:]:
        key, raw = [item.strip() for item in part.split("=", 1)]
        params[key] = float(raw)
    if "gamma" not in params:
        raise ValueError(f"material {material_id} is missing gamma")
    return Material(material_id, kind, params["gamma"], params.get("p_inf", 0.0))


def parse_region(text: str) -> Region:
    bounds = re.match(r"\s*(\[[^\]]+\])\s*,\s*(\[[^\]]+\])\s*,\s*(.*)$", text)
    if bounds is None:
        raise ValueError(f"bad region line: {text}")
    lower = parse_scalar_list(bounds.group(1))[0]
    upper = parse_scalar_list(bounds.group(2))[0]
    values = parse_key_values(bounds.group(3))
    vel = values.get("vel")
    if not isinstance(vel, list):
        raise ValueError(f"region is missing vel: {text}")
    return Region(
        lower=lower,
        upper=upper,
        rho=float(values["rho"]),
        u=float(vel[0]),
        p=float(values["p"]),
        material_id=int(values["material"]),
    )


def parse_config(path: Path) -> ConfigCase:
    """Read the 1D validation config subset needed by the exact solver."""

    x_min = x_max = tfinal = None
    materials: dict[int, Material] = {}
    regions: list[Region] = []

    for raw_line in path.read_text().splitlines():
        line = strip_comment(raw_line)
        if not line:
            continue
        if line.startswith("domain_min"):
            x_min = parse_scalar_list(line.split("=", 1)[1])[0]
        elif line.startswith("domain_max"):
            x_max = parse_scalar_list(line.split("=", 1)[1])[0]
        elif line.startswith("tfinal"):
            tfinal = float(line.split("=", 1)[1].strip())
        elif line.startswith("material"):
            material = parse_material(line.split("=", 1)[1])
            materials[material.material_id] = material
        elif line.startswith("region"):
            regions.append(parse_region(line.split("=", 1)[1]))

    if x_min is None or x_max is None or tfinal is None:
        raise ValueError(f"{path} is missing domain or final time")
    if len(regions) not in (2, 3):
        raise ValueError(f"{path} must have two regions or one incident shock plus an interface")
    if not materials:
        raise ValueError(f"{path} is missing materials")

    return ConfigCase(x_min, x_max, tfinal, materials, sorted(regions, key=lambda region: region.lower))


def shifted_pressure(p: float, material: Material) -> float:
    value = p + material.p_inf
    if value <= 0.0:
        raise ValueError("p + p_inf must be positive")
    return value


def sound_speed(state: Region, material: Material) -> float:
    return math.sqrt(material.gamma * shifted_pressure(state.p, material) / state.rho)


def wave_curve(p: float, state: Region, material: Material) -> float:
    gamma = material.gamma
    if p > state.p:
        a = 2.0 / ((gamma + 1.0) * state.rho)
        b = ((gamma - 1.0) / (gamma + 1.0)) * state.p
        b += ((2.0 * gamma) / (gamma + 1.0)) * material.p_inf
        return (p - state.p) * math.sqrt(a / (p + b))

    c = sound_speed(state, material)
    ratio = shifted_pressure(p, material) / shifted_pressure(state.p, material)
    exponent = (gamma - 1.0) / (2.0 * gamma)
    return (2.0 * c / (gamma - 1.0)) * (ratio**exponent - 1.0)


def wave_curve_derivative(p: float, state: Region, material: Material) -> float:
    gamma = material.gamma
    if p > state.p:
        a = 2.0 / ((gamma + 1.0) * state.rho)
        b = ((gamma - 1.0) / (gamma + 1.0)) * state.p
        b += ((2.0 * gamma) / (gamma + 1.0)) * material.p_inf
        denom = p + b
        root = math.sqrt(a / denom)
        return root * (1.0 - 0.5 * (p - state.p) / denom)

    c = sound_speed(state, material)
    ratio = shifted_pressure(p, material) / shifted_pressure(state.p, material)
    exponent = -(gamma + 1.0) / (2.0 * gamma)
    return (1.0 / (state.rho * c)) * ratio**exponent


def solve_star(left: Region, right: Region, mat_left: Material, mat_right: Material) -> tuple[float, float]:
    """Solve the two-material pressure/velocity star state used for sampling."""

    c_left = sound_speed(left, mat_left)
    c_right = sound_speed(right, mat_right)
    p_guess = 0.5 * (left.p + right.p)
    p_guess -= 0.125 * (right.u - left.u) * (left.rho + right.rho) * (c_left + c_right)
    floor = max(1.0e-10, -mat_left.p_inf + 1.0e-10, -mat_right.p_inf + 1.0e-10)

    def residual(p: float) -> float:
        return wave_curve(p, left, mat_left) + wave_curve(p, right, mat_right) + right.u - left.u

    lo = floor
    hi = max(p_guess, left.p, right.p, 1.0, 2.0 * floor)
    if residual(lo) > 0.0:
        raise ValueError("vacuum state detected")
    while residual(hi) <= 0.0:
        hi = max(2.0 * hi, hi + 1.0)

    p = min(max(p_guess, lo), hi)
    for _ in range(200):
        f = residual(p)
        if abs(f) <= 1.0e-10:
            break
        if f > 0.0:
            hi = p
        else:
            lo = p
        derivative = wave_curve_derivative(p, left, mat_left) + wave_curve_derivative(p, right, mat_right)
        candidate = p - f / derivative
        if not math.isfinite(candidate) or candidate <= lo or candidate >= hi:
            candidate = 0.5 * (lo + hi)
        p = candidate

    f_left = wave_curve(p, left, mat_left)
    f_right = wave_curve(p, right, mat_right)
    u_star = 0.5 * (left.u + right.u + f_right - f_left)
    return p, u_star


def rho_star(state: Region, material: Material, p_star: float) -> float:
    gamma = material.gamma
    ratio = shifted_pressure(p_star, material) / shifted_pressure(state.p, material)
    if p_star > state.p:
        numerator = ratio + (gamma - 1.0) / (gamma + 1.0)
        denominator = ((gamma - 1.0) / (gamma + 1.0)) * ratio + 1.0
        return state.rho * numerator / denominator
    return state.rho * ratio ** (1.0 / gamma)


def right_shock_speed(right: Region, material: Material, p_star: float) -> float:
    if p_star <= right.p:
        raise ValueError("incident wave is not a right shock")
    gamma = material.gamma
    ratio = shifted_pressure(p_star, material) / shifted_pressure(right.p, material)
    factor = ((gamma + 1.0) / (2.0 * gamma)) * ratio + ((gamma - 1.0) / (2.0 * gamma))
    return right.u + sound_speed(right, material) * math.sqrt(factor)


def interface_riemann_inputs(case: ConfigCase) -> tuple[Region, Region, float, float]:
    """Pick the states adjacent to the material interface, not the incident shock."""

    if len(case.regions) == 2:
        left, right = case.regions
        return left, right, left.upper, case.tfinal

    driver, unshocked, right = case.regions
    mat_driver = case.materials[driver.material_id]
    mat_unshocked = case.materials[unshocked.material_id]
    if driver.material_id != unshocked.material_id:
        raise ValueError("incident shock regions must use the same material")

    p_star, u_star = solve_star(driver, unshocked, mat_driver, mat_unshocked)
    shock_speed = right_shock_speed(unshocked, mat_unshocked, p_star)
    interface_x = unshocked.upper
    hit_time = (interface_x - driver.upper) / shock_speed
    tau = case.tfinal - hit_time
    if tau <= 0.0:
        raise ValueError("final time is before the incident shock reaches the material interface")

    post_shock = Region(
        lower=case.x_min,
        upper=interface_x,
        rho=rho_star(unshocked, mat_unshocked, p_star),
        u=u_star,
        p=p_star,
        material_id=unshocked.material_id,
    )
    return post_shock, right, interface_x, tau


def build_solver(repo_root: Path, solver: Path) -> None:
    solver.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-O2",
            "-I.",
            "src/app/exact_two_material_riemann.cpp",
            "-o",
            str(solver),
        ],
        cwd=repo_root,
        check=True,
    )


def run_solver(
    repo_root: Path,
    solver: Path,
    case: ConfigCase,
    left: Region,
    right: Region,
    x0: float,
    time: float,
    samples: int,
    raw_path: Path,
) -> None:
    mat_left = case.materials[left.material_id]
    mat_right = case.materials[right.material_id]
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(solver),
            str(left.rho),
            str(left.u),
            str(left.p),
            str(mat_left.gamma),
            str(right.rho),
            str(right.u),
            str(right.p),
            str(mat_right.gamma),
            str(x0),
            str(time),
            str(case.x_min),
            str(case.x_max),
            str(samples),
            str(raw_path),
            str(mat_left.p_inf),
            str(mat_right.p_inf),
        ],
        cwd=repo_root,
        check=True,
    )


def write_plot_reference(raw_path: Path, output_path: Path) -> None:
    df = pd.read_csv(raw_path)
    entropy = df["p"] / (df["rho"] ** df["gamma"])
    rows = [
        ["Density", "", "Velocity", "", "Entropy", "", "Pressure", ""],
        ["x", "y", "x", "y", "x", "y", "x", "y"],
    ]
    rows.extend(
        [x, rho, x, u, x, s, x, p]
        for x, rho, u, s, p in zip(df["x"], df["rho"], df["u"], entropy, df["p"])
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, header=False)


def generate_case(repo_root: Path, solver: Path, test_name: str, config_path: Path, output_root: Path, samples: int) -> None:
    case = parse_config(config_path)
    left, right, x0, time = interface_riemann_inputs(case)
    raw_path = output_root / f"{test_name}_exact_raw.csv"
    output_path = output_root / f"{test_name}_exact.csv"
    run_solver(repo_root, solver, case, left, right, x0, time, samples, raw_path)
    write_plot_reference(raw_path, output_path)
    print(f"{test_name}: wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Fedkiw exact references using the two-material exact Riemann solver."
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        choices=sorted(TEST_NAMES),
        default=sorted(TEST_NAMES),
    )
    parser.add_argument(
        "--config-root",
        type=Path,
        default=Path("configs/GFM/MM_1D_validation"),
        help="Folder containing test1.txt ... test5.txt 1D Fedkiw configs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/exact/generated_multimaterial"),
    )
    parser.add_argument(
        "--solver",
        type=Path,
        default=Path("/tmp/exact_two_material_riemann"),
    )
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()

    repo_root = Path.cwd()
    solver = args.solver
    if not args.no_build or not solver.exists():
        build_solver(repo_root, solver)

    for test_name in args.tests:
        generate_case(
            repo_root,
            solver,
            test_name,
            args.config_root / f"{test_name}.txt",
            args.output_root,
            args.samples,
        )


if __name__ == "__main__":
    main()
