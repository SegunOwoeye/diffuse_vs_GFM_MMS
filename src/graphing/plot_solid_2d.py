#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import apply_plot_style, configure_profile_axis


apply_plot_style()


def point_stride(count: int, target_points: int = 1600) -> int:
    return max(1, count // target_points)


def plot_section_8p5(
    csv_path: Path,
    out_path: Path | None,
    paper_limits: bool,
    reference_path: Path | None,
) -> None:
    df = pd.read_csv(csv_path)
    required = ["r", "rho", "ur", "srr", "Fp_rr", "kappa"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    ref = None
    if reference_path is not None:
        ref = pd.read_csv(reference_path)
        ref = ref.rename(columns={"R": "r"})
        ref = ref.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    bin_width = 0.10
    df = df.copy()
    df["r_bin"] = (df["r"] / bin_width).round().astype(int) * bin_width

    fields = [
        ("rho", r"Density $\rho$", (0.5, 5.5), "Fig. 11"),
        ("ur", r"Radial velocity $v_r$", (-0.6, 0.2), "Fig. 12"),
        ("srr", r"Radial stress $\sigma_{rr}$", (-8.0, 1.0), "Fig. 13"),
        ("Fp_rr", r"Plastic deformation $F^p_{rr}$", (0.7, 1.4), "Fig. 14"),
        ("kappa", r"Work hardening parameter $\kappa$", (-0.1, 1.0), "Fig. 15"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(8.6, 9.0))
    axes = axes.flatten()

    for ax, (field, ylabel, ylim, figure_label) in zip(axes, fields):
        grouped = df.groupby("r_bin", sort=True)[field]
        r_mean = grouped.mean().index.to_numpy()
        y_mean = grouped.mean().to_numpy()
        y_p10 = grouped.quantile(0.10).to_numpy()
        y_p90 = grouped.quantile(0.90).to_numpy()

        if ref is not None:
            ax.plot(ref["r"], ref[field], color="black", linewidth=1.0, label="1D")
            ax.plot(
                df["r"],
                df[field],
                marker=".",
                linestyle="none",
                color="black",
                markersize=1.2,
                markeredgewidth=0.0,
                label="2D",
            )
        else:
            ax.plot(
                df["r"],
                df[field],
                marker="+",
                linestyle="none",
                color="black",
                markersize=1.8,
                markeredgewidth=0.35,
            )
            ax.fill_between(
                r_mean,
                y_p10,
                y_p90,
                color="0.82",
                linewidth=0.0,
                zorder=0,
            )
            ax.plot(
                r_mean,
                y_mean,
                color="red",
                linewidth=1.0,
            )
        ax.set_xlim(0.0, 25.0)
        if paper_limits:
            ax.set_ylim(*ylim)
        ax.text(
            0.04,
            0.92,
            figure_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )
        ax.set_xlabel("R")
        ax.set_ylabel(ylabel)
        ax.tick_params(direction="in", top=True, right=True)
        if ref is not None:
            ax.legend(loc="upper right", frameon=False, fontsize=8)

    axes[-1].axis("off")
    fig.tight_layout()

    if out_path is None:
        plt.show()
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved figure to {out_path}")


def plot_barton_radial_reference(
    csv_path: Path,
    reference_path: Path,
    out_path: Path | None,
    scatter_label: str,
) -> None:
    df = pd.read_csv(csv_path)
    ref = pd.read_csv(reference_path)
    required = ["r", "rho", "ur", "srr", "T"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    ref = ref.replace([np.inf, -np.inf], np.nan).dropna(subset=required)

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.2))
    axes = axes.flatten()
    temperature_ylim = (340.0, 850.0) if scatter_label == "3D" else (340.0, 650.0)
    fields = [
        ("rho", r"$\rho$ (g/cm$^3$)", (9.20, 9.62)),
        ("ur", r"$u_r$ (km/s)", (-0.08, 0.08)),
        ("srr", r"$\sigma_{rr}$ (GPa)", (-15.0, -7.0)),
        ("T", "T (K)", temperature_ylim),
    ]
    for ax, (field, ylabel, ylim) in zip(axes, fields):
        ref_stride = point_stride(len(ref))
        solution_stride = point_stride(len(df))
        ax.plot(
            ref["r"].iloc[::ref_stride],
            ref[field].iloc[::ref_stride],
            marker="+",
            linestyle="none",
            color="black",
            markersize=2.6,
            markeredgewidth=0.6,
            label="1D reference",
        )
        ax.plot(
            df["r"].iloc[::solution_stride],
            df[field].iloc[::solution_stride],
            marker="o",
            linestyle="none",
            color="tab:blue",
            markersize=2.0,
            markerfacecolor="none",
            markeredgewidth=0.5,
            label=scatter_label,
        )
        ax.set_xlim(0.0, 10.0)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
        configure_profile_axis(ax, x_label="r (cm)", show_legend=True, show_title=False)

    fig.tight_layout()

    if out_path is None:
        plt.show()
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved figure to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 2D elastoplastic solid radial scatter output.")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--auto-limits",
        action="store_true",
        help="Use automatic y-limits instead of the Miller-Colella Fig. 11-15 axes.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Optional 1D cylindrical reference CSV to overlay as the paper-style solid line.",
    )
    parser.add_argument(
        "--barton-radial-reference",
        action="store_true",
        help="Plot Barton tensor radial scatter data against a 1D radial-reference solution.",
    )
    parser.add_argument(
        "--scatter-label",
        default="2D",
        help="Legend label for the Cartesian scatter points.",
    )
    args = parser.parse_args()
    if args.barton_radial_reference:
        if args.reference is None:
            raise SystemExit("--barton-radial-reference requires --reference")
        plot_barton_radial_reference(args.csv, args.reference, args.out, args.scatter_label)
        return
    plot_section_8p5(
        args.csv,
        args.out,
        paper_limits=not args.auto_limits,
        reference_path=args.reference,
    )


if __name__ == "__main__":
    main()
