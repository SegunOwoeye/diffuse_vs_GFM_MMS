#!/usr/bin/env python3
"""Generate 1D elastic reference Riemann data for solid shear tests.

This is a small-strain elastic five-wave reference solver. It is exact only for
the simplified linear acoustic-elastic Riemann problem with longitudinal and
shear impedances. It is intended as a diagnostic elastic-limit reference for
Barton/GPR rGFM solid-solid tests inspired by cases such as Gorsse et al. (2014)
TC4.

It is not an exact solver for the Barton/GPR tensor material law, and it is not
an exact solver for rate-dependent elastoplastic yielding. Once yield fronts
form, use this only as an elastic predictor/reference and compare the plastic
run against grid-converged Barton/GPR numerical references.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ElasticState:
    rho: float
    u: float
    v: float
    p: float
    sigma_xx: float
    sigma_xy: float


@dataclass(frozen=True)
class ElasticMaterial:
    rho0: float
    c_longitudinal: float
    c_shear: float
    gamma: float
    p_inf: float

    @property
    def z_longitudinal(self) -> float:
        return self.rho0 * self.c_longitudinal

    @property
    def z_shear(self) -> float:
        return self.rho0 * self.c_shear


@dataclass(frozen=True)
class ElasticRiemannSolution:
    left: ElasticState
    left_normal_star: ElasticState
    left_star: ElasticState
    right_star: ElasticState
    right_normal_star: ElasticState
    right: ElasticState
    left_material: ElasticMaterial
    right_material: ElasticMaterial
    x0: float
    u_star: float


def acoustic_star(
    velocity_l: float,
    velocity_r: float,
    traction_l: float,
    traction_r: float,
    impedance_l: float,
    impedance_r: float,
) -> tuple[float, float]:
    """Return compatible star velocity and traction.

    The pressure-like variable is q=-traction. This sign convention lets the
    same formula cover normal compression and shear traction.
    """
    q_l = -traction_l
    q_r = -traction_r
    denom = max(impedance_l + impedance_r, 1.0e-300)
    q_star = (
        impedance_r * q_l
        + impedance_l * q_r
        + impedance_l * impedance_r * (velocity_l - velocity_r)
    ) / denom
    velocity_star = (
        impedance_l * velocity_l
        + impedance_r * velocity_r
        + q_l
        - q_r
    ) / denom
    return velocity_star, -q_star


def density_after_longitudinal_wave(
    state: ElasticState,
    material: ElasticMaterial,
    sigma_xx_star: float,
) -> float:
    q = -state.sigma_xx
    q_star = -sigma_xx_star
    return max(state.rho + (q_star - q) / (material.c_longitudinal**2), 1.0e-12)


def isentropic_pressure(state: ElasticState, material: ElasticMaterial, rho: float) -> float:
    return (state.p + material.p_inf) * (rho / state.rho) ** material.gamma - material.p_inf


def build_solution(
    left: ElasticState,
    right: ElasticState,
    left_material: ElasticMaterial,
    right_material: ElasticMaterial,
    x0: float,
) -> ElasticRiemannSolution:
    u_star, sigma_xx_star = acoustic_star(
        left.u,
        right.u,
        left.sigma_xx,
        right.sigma_xx,
        left_material.z_longitudinal,
        right_material.z_longitudinal,
    )
    v_star, sigma_xy_star = acoustic_star(
        left.v,
        right.v,
        left.sigma_xy,
        right.sigma_xy,
        left_material.z_shear,
        right_material.z_shear,
    )

    rho_l_star = density_after_longitudinal_wave(left, left_material, sigma_xx_star)
    rho_r_star = density_after_longitudinal_wave(right, right_material, sigma_xx_star)
    p_l_star = isentropic_pressure(left, left_material, rho_l_star)
    p_r_star = isentropic_pressure(right, right_material, rho_r_star)

    left_normal_star = ElasticState(
        rho=rho_l_star,
        u=u_star,
        v=left.v,
        p=p_l_star,
        sigma_xx=sigma_xx_star,
        sigma_xy=left.sigma_xy,
    )
    left_star = ElasticState(
        rho=rho_l_star,
        u=u_star,
        v=v_star,
        p=p_l_star,
        sigma_xx=sigma_xx_star,
        sigma_xy=sigma_xy_star,
    )
    right_star = ElasticState(
        rho=rho_r_star,
        u=u_star,
        v=v_star,
        p=p_r_star,
        sigma_xx=sigma_xx_star,
        sigma_xy=sigma_xy_star,
    )
    right_normal_star = ElasticState(
        rho=rho_r_star,
        u=u_star,
        v=right.v,
        p=p_r_star,
        sigma_xx=sigma_xx_star,
        sigma_xy=right.sigma_xy,
    )
    return ElasticRiemannSolution(
        left=left,
        left_normal_star=left_normal_star,
        left_star=left_star,
        right_star=right_star,
        right_normal_star=right_normal_star,
        right=right,
        left_material=left_material,
        right_material=right_material,
        x0=x0,
        u_star=u_star,
    )


def sample_solution(solution: ElasticRiemannSolution, x: float, time: float) -> ElasticState:
    if time <= 0.0:
        return solution.left if x < solution.x0 else solution.right
    xi = (x - solution.x0) / time
    if xi < -solution.left_material.c_longitudinal:
        return solution.left
    if xi < -solution.left_material.c_shear:
        return solution.left_normal_star
    if xi < solution.u_star:
        return solution.left_star
    if xi < solution.right_material.c_shear:
        return solution.right_star
    if xi < solution.right_material.c_longitudinal:
        return solution.right_normal_star
    return solution.right


def gorsse_tc4_solution() -> ElasticRiemannSolution:
    rho = 8900.0
    chi = 5.0e10
    gamma = 4.22
    p_inf = 3.42e10
    def material_for_pressure(p: float) -> ElasticMaterial:
        c_shear = (2.0 * chi / rho) ** 0.5
        c_bulk = (gamma * (p + p_inf) / rho) ** 0.5
        c_longitudinal = (c_bulk**2 + c_shear**2) ** 0.5
        return ElasticMaterial(
            rho0=rho,
            c_longitudinal=c_longitudinal,
            c_shear=c_shear,
            gamma=gamma,
            p_inf=p_inf,
        )
    left = ElasticState(
        rho=rho,
        u=0.0,
        v=0.0,
        p=1.0e9,
        sigma_xx=-1.0e9,
        sigma_xy=0.0,
    )
    right = ElasticState(
        rho=rho,
        u=0.0,
        v=100.0,
        p=1.0e5,
        sigma_xx=-1.0e5,
        sigma_xy=0.0,
    )
    return build_solution(left, right, material_for_pressure(left.p), material_for_pressure(right.p), x0=0.5)


def write_csv(path: Path, solution: ElasticRiemannSolution, cells: int, time: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x_min = 0.0
    x_max = 1.0
    dx = (x_max - x_min) / cells
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "rho", "u", "v", "p", "sigma_xx", "sigma_xy", "time"])
        for i in range(cells):
            x = x_min + (i + 0.5) * dx
            state = sample_solution(solution, x, time)
            writer.writerow([
                f"{x:.17g}",
                f"{state.rho:.17g}",
                f"{state.u:.17g}",
                f"{state.v:.17g}",
                f"{state.p:.17g}",
                f"{state.sigma_xx:.17g}",
                f"{state.sigma_xy:.17g}",
                f"{time:.17g}",
            ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["gorsse_tc4"], default="gorsse_tc4")
    parser.add_argument("--cells", type=int, default=4000)
    parser.add_argument("--time", type=float, default=5.0e-5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/exact/solid/gorsse_2014_tc4_elastic_exact_N4000.csv"),
    )
    args = parser.parse_args()

    if args.case != "gorsse_tc4":
        raise ValueError(f"Unsupported case: {args.case}")
    solution = gorsse_tc4_solution()
    write_csv(args.output, solution, args.cells, args.time)
    print(f"Written exact elastic reference: {args.output}")


if __name__ == "__main__":
    main()
