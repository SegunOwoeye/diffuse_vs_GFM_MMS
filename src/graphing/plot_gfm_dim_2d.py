# [0] Import Libraries
from pathlib import Path
import argparse
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import (
    configure_profile_axis,
    save_figure,
)
from exact_reference import load_optional_exact_reference


# [1] Test / Folder Settings
TEST_FOLDERS = {
    "test1": "FedkiwA",
    "test2": "FedkiwB",
    "test3": "FedkiwC",
    "test4": "FedkiwD1",
    "test5": "FedkiwD2",
}

TEST_GAMMAS = {
    "test1": (1.4, 1.2),
    "test2": (1.4, 1.67),
    "test3": (1.4, 1.249),
    "test4": (1.4, 1.67),
    "test5": (1.4, 1.249),
}

TEST_TIMES = {
    "test1": 7.0e-4,
    "test2": 1.2e-3,
    "test3": 1.7e-3,
    "test4": 5.0e-4,
    "test5": 7.0e-4,
}

METHOD_STYLES = {
    "gfm": {
        "label": "rGFM",
        "slug": "rgfm",
    },
    "dim": {
        "label": "Allaire 5eq",
        "slug": "allaire5eq",
    },
}

FIELD_NAMES = [
    ("rho", "Density"),
    ("u0", "X-velocity"),
    ("p", "Pressure"),
    ("entropy", "Entropy"),
]

OBSOLETE_OUTPUT_PATTERNS = (
    "*side_by_side*.png",
    "*_2d_rgfm_allaire5eq_xslice_overlay_*.png",
    "*_2d_rgfm_xslice_convergence.png",
    "*_2d_allaire5eq_xslice_convergence.png",
)

TRANSVERSE_VELOCITY_REL_FLOOR = 1.0e-10
TRANSVERSE_VELOCITY_ABS_FLOOR = 1.0e-12


def format_time(value):
    return f"{value:.3g}"


# [2] Load CSV Data
def infer_material_ids(df):
    if "mat" in df.columns:
        return df["mat"].round().astype(int).to_numpy()

    alpha_columns = sorted([
        column for column in df.columns
        if column.startswith("alpha")
    ])

    if alpha_columns:
        return df[alpha_columns].to_numpy().argmax(axis=1)

    return None


def compute_entropy(df, test_name):
    material_ids = infer_material_ids(df)
    gammas = TEST_GAMMAS[test_name]

    if material_ids is None:
        gamma = gammas[0]
    else:
        gamma = np.asarray([gammas[int(mat)] for mat in material_ids])

    return df["p"].to_numpy() / (df["rho"].to_numpy() ** gamma)


def load_1d_solution(csv_path, test_name):
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found")

    df = pd.read_csv(csv_path)
    fields = {
        "rho": df["rho"].to_numpy(),
        "u0": df["u0"].to_numpy(),
        "p": df["p"].to_numpy(),
        "entropy": compute_entropy(df, test_name),
    }

    return df["x0"].to_numpy(), fields


def load_2d_solution(csv_path, test_name):
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found")

    df = pd.read_csv(csv_path)
    fields = {
        "rho": df["rho"].to_numpy(),
        "u0": df["u0"].to_numpy(),
        "u1": df["u1"].to_numpy(),
        "p": df["p"].to_numpy(),
        "entropy": compute_entropy(df, test_name),
    }

    return df["x0"].to_numpy(), df["x1"].to_numpy(), fields


def extract_centerline_x_slice(x, y, values, requested_y=None):
    unique_y = np.unique(y)

    if requested_y is None:
        requested_y = 0.5 * (float(unique_y.min()) + float(unique_y.max()))

    slice_y = unique_y[np.argmin(np.abs(unique_y - requested_y))]
    mask = np.isclose(y, slice_y)
    order = np.argsort(x[mask])

    return x[mask][order], values[mask][order], float(slice_y)


# [3] Build Paths
def fedkiw_name(test_name):
    return TEST_FOLDERS[test_name]


def solution_folder(data_root, method, dimension, test_name):
    name = fedkiw_name(test_name)
    return data_root / method / f"MM_{dimension}D_validation" / f"{method}_{name}"


def solution_path(data_root, method, dimension, test_name, resolution):
    name = fedkiw_name(test_name)
    folder = solution_folder(data_root, method, dimension, test_name)

    if dimension == 1:
        return folder / f"{method}_{name}_N{resolution}.csv"

    return folder / f"{method}_{name}_N{resolution}_N{resolution}.csv"


# [4] Resolution Selection
def extract_resolution(csv_path, dimension):
    if dimension == 1:
        match = re.search(r"_N(\d+)\.csv$", csv_path.name)
    else:
        match = re.search(r"_N(\d+)_N\1\.csv$", csv_path.name)

    if match is None:
        return None

    return int(match.group(1))


def available_resolutions(data_root, method, dimension, test_name):
    folder = solution_folder(data_root, method, dimension, test_name)

    if not folder.exists():
        return []

    resolutions = []

    for csv_path in folder.glob("*.csv"):
        resolution = extract_resolution(csv_path, dimension)

        if resolution is not None:
            resolutions.append(resolution)

    return sorted(set(resolutions))


def choose_resolution(data_root, method, test_name, requested_resolution):
    one_d = set(available_resolutions(data_root, method, 1, test_name))
    two_d = set(available_resolutions(data_root, method, 2, test_name))
    resolutions = sorted(one_d.intersection(two_d))

    if not resolutions:
        label = METHOD_STYLES[method]["label"]
        raise FileNotFoundError(f"No common 1D / 2D {label} resolutions found for {test_name}")

    if requested_resolution in resolutions:
        return requested_resolution

    higher = [resolution for resolution in resolutions if resolution > requested_resolution]

    if higher:
        chosen = min(higher)
    else:
        chosen = max(resolutions)

    print(
        f"{test_name} {METHOD_STYLES[method]['label']}: N={requested_resolution} "
        f"was not available in both 1D and 2D, using N={chosen} instead"
    )

    return chosen


# [5] Plot Helpers
def remove_obsolete_plot_outputs(output_dir):
    for pattern in OBSOLETE_OUTPUT_PATTERNS:
        for path in output_dir.glob(pattern):
            path.unlink()


def make_validation_figure():
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 5.8))
    axes = axes.flatten()

    for ax, (field_key, _) in zip(axes[:4], FIELD_NAMES):
        configure_profile_axis(ax, field_key, show_legend=False, show_title=False)

    configure_profile_axis(axes[4], "u1", show_legend=False, show_title=False)
    axes[1].set_ylabel(r"X-velocity, $u_x$")
    axes[4].set_ylabel(r"Y-velocity, $u_y$")
    axes[5].axis("off")

    return fig, axes


def marker_step(x, target_markers=28):
    if len(x) <= target_markers:
        return 1
    return max(1, len(x) // target_markers)


def transverse_velocity_plot_floor(ux_slice):
    scale = max(float(np.nanmax(np.abs(ux_slice))), 1.0)
    return max(TRANSVERSE_VELOCITY_ABS_FLOOR, TRANSVERSE_VELOCITY_REL_FLOOR * scale)


def zero_negligible_transverse_velocity(uy_slice, floor):
    values = np.asarray(uy_slice, dtype=float).copy()
    values[np.abs(values) <= floor] = 0.0
    return values


def plot_numeric_points(
    ax,
    x,
    values,
    label,
    color,
    marker,
    zorder=3,
):
    ax.plot(
        x,
        values,
        color=color,
        linestyle="None",
        linewidth=0.0,
        marker=marker,
        markersize=2.6,
        markerfacecolor="white",
        markeredgewidth=0.7,
        label=label,
        zorder=zorder,
    )


def plot_1d_vs_2d(data_root, output_dir, method, test_name, requested_resolution, exact_root):
    resolution = choose_resolution(data_root, method, test_name, requested_resolution)
    x_1d, fields_1d = load_1d_solution(
        solution_path(data_root, method, 1, test_name, resolution),
        test_name,
    )
    x_2d, y_2d, fields_2d = load_2d_solution(
        solution_path(data_root, method, 2, test_name, resolution),
        test_name,
    )

    fig, axes = make_validation_figure()
    selected_y = None
    exact_fields = load_optional_exact_reference(
        exact_root,
        test_name,
        context=f"{test_name} {METHOD_STYLES[method]['label']} 2D centerline",
    )

    for ax, (field_key, _) in zip(axes[:4], FIELD_NAMES):
        if exact_fields and field_key in exact_fields:
            exact_x, exact_y = exact_fields[field_key]
            ax.plot(
                exact_x,
                exact_y,
                color="black",
                linestyle="-",
                linewidth=1.2,
                label="Exact",
                zorder=4,
            )

        x_slice, values_slice, selected_y = extract_centerline_x_slice(
            x_2d,
            y_2d,
            fields_2d[field_key],
        )

        plot_numeric_points(
            ax,
            x_1d,
            fields_1d[field_key],
            label="1D",
            color="tab:blue",
            marker="D",
            zorder=2,
        )

        plot_numeric_points(
            ax,
            x_slice,
            values_slice,
            label="2D centerline",
            color="tab:red",
            marker="o",
            zorder=3,
        )
        ax.legend(frameon=False)

    x_slice, uy_slice, selected_y = extract_centerline_x_slice(
        x_2d,
        y_2d,
        fields_2d["u1"],
    )
    _, ux_slice, _ = extract_centerline_x_slice(
        x_2d,
        y_2d,
        fields_2d["u0"],
    )
    uy_floor = transverse_velocity_plot_floor(ux_slice)
    uy_plot = zero_negligible_transverse_velocity(uy_slice, uy_floor)

    axes[4].axhline(0.0, color="black", linewidth=0.9, label="zero")
    plot_numeric_points(
        axes[4],
        x_slice,
        uy_plot,
        label="2D centerline",
        color="tab:red",
        marker="o",
        zorder=3,
    )

    axes[4].legend(frameon=False)

    label = METHOD_STYLES[method]["label"]
    slug = METHOD_STYLES[method]["slug"]
    plt.tight_layout()

    output_path = output_dir / f"{test_name}_2d_validation_{slug}_1d_vs_2d_xslice_N{resolution}.png"
    save_figure(fig, output_path)
    return output_path


# [6] Script Entry
def main():
    parser = argparse.ArgumentParser(
        description="Plot representative Fedkiw 2D validation as 1D solution vs 2D centerline slice."
    )
    parser.add_argument(
        "--test",
        choices=sorted(TEST_FOLDERS.keys()),
        help="Single Fedkiw case to use for the 2D reduction validation.",
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        choices=sorted(TEST_FOLDERS.keys()),
        help="Fedkiw cases to use for the 2D reduction validation. Defaults to test1-test5.",
    )
    parser.add_argument(
        "--overlay-n",
        type=int,
        default=400,
        help="Requested resolution shared by the 1D and 2D solutions.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/csv"),
        help="Root folder containing the GFM and DIM CSV outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/plots/2d_rGFM_Allaire5eq_validation"),
        help="Folder where the 2D validation figures will be saved.",
    )
    parser.add_argument(
        "--exact-root",
        type=Path,
        default=Path("data/exact/fedkiw"),
        help="Folder containing optional digitised exact CSVs named test1_exact.csv, test2_exact.csv, ...",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=sorted(METHOD_STYLES.keys()),
        default=sorted(METHOD_STYLES.keys()),
        help="Methods to validate separately.",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    remove_obsolete_plot_outputs(args.output_dir)

    selected_tests = args.tests or ([args.test] if args.test else sorted(TEST_FOLDERS.keys()))

    for test_name in selected_tests:
        for method in args.methods:
            plot_1d_vs_2d(
                args.data_root,
                args.output_dir,
                method,
                test_name,
                args.overlay_n,
                args.exact_root,
            )


if __name__ == "__main__":
    main()


# RUN Commands
# cd diffuse_vs_GFM_MMS
# source .venv/bin/activate
# python src/graphing/plot_gfm_dim_2d.py --overlay-n 400
