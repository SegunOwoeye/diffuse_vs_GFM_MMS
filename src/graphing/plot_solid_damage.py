#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def radial_damage_profile(df: pd.DataFrame, bin_width: float) -> pd.DataFrame:
    work = df.copy()
    work["r_bin"] = (work["r"] / bin_width).round().astype(int) * bin_width
    grouped = work.groupby("r_bin", sort=True)
    return pd.DataFrame(
        {
            "r": grouped["r"].mean(),
            "damage_mean": grouped["damage"].mean(),
            "damage_max": grouped["damage"].max(),
            "eqps_mean": grouped["eqps"].mean(),
            "failed_fraction": grouped["failed"].mean(),
        }
    ).reset_index(drop=True)


def damage_slice(df: pd.DataFrame) -> pd.DataFrame:
    if "z" not in df.columns:
        return df
    z_values = np.sort(df["z"].unique())
    z0 = z_values[0]
    return df[np.isclose(df["z"], z0)]


def plot_damage(csv_path: Path, out_path: Path | None, bin_width: float) -> None:
    df = pd.read_csv(csv_path)
    required = ["x", "y", "r", "eqps", "damage", "failed"]
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(f"Damage plot requires missing CSV columns: {', '.join(missing)}")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    profile = radial_damage_profile(df, bin_width)
    sliced = damage_slice(df)
    max_damage = float(df["damage"].max())
    damage_vmax = max_damage if max_damage > 0.0 else 1.0

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.6))
    ax0, ax1, ax2, ax3 = axes.flatten()

    sc = ax0.scatter(
        sliced["x"],
        sliced["y"],
        c=sliced["damage"],
        s=6.0,
        cmap="inferno",
        vmin=0.0,
        vmax=damage_vmax,
        linewidths=0.0,
    )
    ax0.set_aspect("equal", adjustable="box")
    ax0.set_title("Damage map")
    ax0.set_xlabel("x (cm)")
    ax0.set_ylabel("y (cm)")
    cbar = fig.colorbar(sc, ax=ax0, fraction=0.046, pad=0.04)
    cbar.set_label("Johnson-Cook damage D")

    ax1.scatter(
        sliced["x"],
        sliced["y"],
        c=sliced["failed"],
        s=6.0,
        cmap="gray_r",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.0,
    )
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_title("Failed cells")
    ax1.set_xlabel("x (cm)")
    ax1.set_ylabel("y (cm)")

    ax2.plot(profile["r"], profile["damage_mean"], color="black", linewidth=1.0, label="mean D")
    ax2.plot(profile["r"], profile["damage_max"], color="red", linewidth=1.0, label="max D")
    if max_damage > 0.5:
        ax2.axhline(1.0, color="0.5", linewidth=0.8, linestyle="--")
    elif max_damage > 0.0:
        ax2.set_ylim(0.0, 1.1 * max_damage)
        ax2.text(
            0.02,
            0.95,
            "failure threshold D=1 is outside this axis",
            transform=ax2.transAxes,
            ha="left",
            va="top",
            fontsize=8,
        )
    ax2.set_title("Radial damage profile")
    ax2.set_xlabel("r (cm)")
    ax2.set_ylabel("damage D")
    ax2.legend(frameon=False)

    ax3.plot(profile["r"], profile["eqps_mean"], color="black", linewidth=1.0, label="mean eqps")
    ax3.plot(profile["r"], profile["failed_fraction"], color="red", linewidth=1.0, label="failed fraction")
    ax3.set_title("Plastic strain and failure")
    ax3.set_xlabel("r (cm)")
    ax3.set_ylabel("value")
    ax3.legend(frameon=False)

    for ax in axes.flatten():
        ax.tick_params(direction="in", top=True, right=True)

    fig.tight_layout()
    if out_path is None:
        plt.show()
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved figure to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Johnson-Cook damage output from Barton tensor solids.")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--bin-width", type=float, default=0.05, help="Radial bin width in cm.")
    args = parser.parse_args()
    plot_damage(args.csv, args.out, args.bin_width)


if __name__ == "__main__":
    main()
