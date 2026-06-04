#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def label_from_path(path: Path) -> str:
    match = re.search(r"_t([0-9p]+)us_", path.name)
    if match:
        return match.group(1).replace("p", ".") + r" $\mu$s"
    if re.search(r"_N[0-9]+$", path.stem):
        return "final"
    return path.stem


def time_label(path: Path) -> str:
    match = re.search(r"_t([0-9p]+)us_", path.name)
    if match:
        return match.group(1).replace("p", ".").rstrip("0").rstrip(".")
    return "final"


def wilkins_stress_limits(
    csv_paths: list[Path],
    ymin: float | None,
    ymax: float | None,
) -> tuple[float, float]:
    if ymin is not None and ymax is not None:
        return ymin, ymax

    stress_min = math.inf
    stress_max = -math.inf
    for csv_path in csv_paths:
        stress_gpa = pd.read_csv(csv_path, usecols=["normal_stress"])["normal_stress"] / 1.0e9
        stress_min = min(stress_min, float(stress_gpa.min()))
        stress_max = max(stress_max, float(stress_gpa.max()))

    if not math.isfinite(stress_min) or not math.isfinite(stress_max):
        raise ValueError("Could not determine stress limits from supplied CSV files")

    auto_ymin = math.floor(stress_min) if stress_min > -10.0 else 5.0 * math.floor(stress_min / 5.0)
    auto_ymax = max(1.0, math.ceil(stress_max))
    return (
        auto_ymin if ymin is None else ymin,
        auto_ymax if ymax is None else ymax,
    )


def plot_wilkins_stress_grid(
    csv_paths: list[Path],
    out_path: Path | None,
    stress_ymin: float | None,
    stress_ymax: float | None,
    x_unit: str,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(8.4, 9.2), sharex=True, sharey=True)
    axes = axes.flatten()
    ymin, ymax = wilkins_stress_limits(csv_paths, stress_ymin, stress_ymax)
    x_scale = 100.0 if x_unit == "cm" else 1000.0
    x_label = "x [cm]" if x_unit == "cm" else "x [mm]"
    x_max = 5.0 if x_unit == "cm" else 50.0

    for ax, csv_path in zip(axes, csv_paths):
        df = pd.read_csv(csv_path)
        ax.plot(
            x_scale * df["x"],
            df["normal_stress"] / 1.0e9,
            marker="+",
            linestyle="none",
            color="black",
            markersize=2.2,
            markeredgewidth=0.45,
        )
        ax.set_xlim(0.0, x_max)
        ax.set_ylim(ymin, ymax)
        ax.text(
            0.82,
            0.08,
            f"t={time_label(csv_path)}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
        )
        ax.tick_params(direction="in", top=True, right=False)

    for ax in axes[::2]:
        ax.set_ylabel(r"Stress $\sigma_{11}$ [GPa]")
    for ax in axes[-2:]:
        ax.set_xlabel(x_label)

    fig.tight_layout()
    if out_path is None:
        plt.show()
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved figure to {out_path}")


def plot_solid(
    csv_paths: list[Path],
    out_path: Path | None,
    paper_wilkins: bool,
    wilkins_stress_grid: bool,
    stress_ymin: float | None,
    stress_ymax: float | None,
    x_unit: str,
) -> None:
    if not csv_paths:
        raise ValueError("No CSV files supplied")

    if wilkins_stress_grid:
        plot_wilkins_stress_grid(csv_paths, out_path, stress_ymin, stress_ymax, x_unit)
        return

    df = pd.read_csv(csv_paths[-1])

    x_mm = 1000.0 * df["x"]
    pressure_gpa = df["p"] / 1.0e9
    velocity = df["u"]
    equivalent_stress_gpa = df["equivalent_stress"] / 1.0e9
    plastic_strain = df["plastic_strain"]
    longitudinal_stress_gpa = df["normal_stress"] / 1.0e9

    if paper_wilkins:
        fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.2), sharex=True)
        for csv_path in csv_paths:
            snapshot = pd.read_csv(csv_path)
            snapshot_x_mm = 1000.0 * snapshot["x"]
            axes[0].plot(
                snapshot_x_mm,
                snapshot["normal_stress"] / 1.0e9,
                linewidth=1.4,
                label=label_from_path(csv_path),
            )
            axes[1].plot(
                snapshot_x_mm,
                snapshot["rho"],
                linewidth=1.4,
                label=label_from_path(csv_path),
            )
        axes[0].set_ylabel(r"Longitudinal stress, $\sigma_{xx}$ (GPa)")
        axes[1].set_ylabel(r"Mass density, $\rho$ (kg/m$^3$)")
        axes[1].set_xlabel("x (mm)")
        for ax in axes:
            ax.grid(False)
            ax.legend(frameon=False, fontsize=8, loc="best")
        fig.tight_layout()
        if out_path is None:
            plt.show()
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved figure to {out_path}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.6), sharex=True)
    axes = axes.flatten()

    axes[0].plot(x_mm, df["rho"], color="black")
    axes[0].set_ylabel(r"Density, $\rho$ (kg/m$^3$)")

    axes[1].plot(x_mm, velocity, color="tab:blue")
    axes[1].set_ylabel(r"Velocity, $u$ (m/s)")

    axes[2].plot(x_mm, pressure_gpa, color="tab:red", label="pressure")
    axes[2].plot(x_mm, df["sxx"] / 1.0e9, color="tab:orange", label=r"$s_{xx}$")
    axes[2].plot(x_mm, longitudinal_stress_gpa, color="black", label=r"$\sigma_{xx}$")
    axes[2].set_ylabel("Stress (GPa)")
    axes[2].legend(frameon=False, loc="best")

    axes[3].plot(x_mm, equivalent_stress_gpa, color="tab:green", label="von Mises")
    axes[3].plot(x_mm, plastic_strain, color="tab:purple", label="plastic strain")
    if "Fp_xx" in df:
        axes[3].plot(x_mm, df["Fp_xx"], color="tab:brown", label=r"$F^p_{xx}$")
    axes[3].set_ylabel("Strength response")
    axes[3].legend(frameon=False, loc="best")

    for ax in axes[2:]:
        ax.set_xlabel("x (mm)")

    for ax in axes:
        ax.grid(False)

    fig.tight_layout()

    if out_path is None:
        plt.show()
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved figure to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 1D elastoplastic solid output.")
    parser.add_argument("csv", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--paper-wilkins",
        action="store_true",
        help="Plot longitudinal stress and density in the style of Miller-Colella Wilkins figures.",
    )
    parser.add_argument(
        "--wilkins-stress-grid",
        action="store_true",
        help="Plot six longitudinal-stress panels matching Miller-Colella Fig. 7/9 layout.",
    )
    parser.add_argument(
        "--stress-ymin",
        type=float,
        default=None,
        help="Lower y-axis limit in GPa for --wilkins-stress-grid. Defaults to data-based autoscaling.",
    )
    parser.add_argument(
        "--stress-ymax",
        type=float,
        default=None,
        help="Upper y-axis limit in GPa for --wilkins-stress-grid. Defaults to data-based autoscaling.",
    )
    parser.add_argument(
        "--x-unit",
        choices=["mm", "cm"],
        default="mm",
        help="x-axis unit for --wilkins-stress-grid. Barton Section 6.1 uses cm.",
    )
    args = parser.parse_args()
    plot_solid(
        args.csv,
        args.out,
        args.paper_wilkins,
        args.wilkins_stress_grid,
        args.stress_ymin,
        args.stress_ymax,
        args.x_unit,
    )


if __name__ == "__main__":
    main()
