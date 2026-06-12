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
from fedkiw_common import (
    TEST_FOLDERS,
    compute_entropy,
    solution_folder,
    solution_path,
    available_resolutions as common_available_resolutions,
)


# [1] Test / Folder Settings
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
    from fedkiw_common import infer_material_ids as common_infer_material_ids
    return common_infer_material_ids(df)


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


def build_structured_grid(x, y, values):
    df = pd.DataFrame({"x": x, "y": y, "value": values})
    grid = (
        df.pivot(index="y", columns="x", values="value")
        .sort_index()
        .sort_index(axis=1)
    )

    return (
        grid.columns.to_numpy(dtype=float),
        grid.index.to_numpy(dtype=float),
        grid.to_numpy(dtype=float),
    )


def bilinear_sample(x_unique, y_unique, grid, sample_x, sample_y):
    sample_x = np.asarray(sample_x, dtype=float)
    sample_y = np.asarray(sample_y, dtype=float)

    if len(x_unique) < 2 or len(y_unique) < 2:
        raise ValueError("bilinear_sample requires at least two grid points in each direction")

    if (
        np.any(sample_x < x_unique[0]) or
        np.any(sample_x > x_unique[-1]) or
        np.any(sample_y < y_unique[0]) or
        np.any(sample_y > y_unique[-1])
    ):
        raise ValueError("Requested oblique slice lies outside the 2D grid")

    ix = np.searchsorted(x_unique, sample_x, side="right") - 1
    iy = np.searchsorted(y_unique, sample_y, side="right") - 1
    ix = np.clip(ix, 0, len(x_unique) - 2)
    iy = np.clip(iy, 0, len(y_unique) - 2)

    x0 = x_unique[ix]
    x1 = x_unique[ix + 1]
    y0 = y_unique[iy]
    y1 = y_unique[iy + 1]

    tx = np.divide(sample_x - x0, x1 - x0, out=np.zeros_like(sample_x), where=(x1 != x0))
    ty = np.divide(sample_y - y0, y1 - y0, out=np.zeros_like(sample_y), where=(y1 != y0))

    q00 = grid[iy, ix]
    q10 = grid[iy, ix + 1]
    q01 = grid[iy + 1, ix]
    q11 = grid[iy + 1, ix + 1]

    return (
        (1.0 - tx) * (1.0 - ty) * q00 +
        tx * (1.0 - ty) * q10 +
        (1.0 - tx) * ty * q01 +
        tx * ty * q11
    )


def nearest_sample(x_unique, y_unique, grid, sample_x, sample_y):
    sample_x = np.asarray(sample_x, dtype=float)
    sample_y = np.asarray(sample_y, dtype=float)

    if (
        np.any(sample_x < x_unique[0]) or
        np.any(sample_x > x_unique[-1]) or
        np.any(sample_y < y_unique[0]) or
        np.any(sample_y > y_unique[-1])
    ):
        raise ValueError("Requested oblique slice lies outside the 2D grid")

    ix = np.searchsorted(x_unique, sample_x)
    iy = np.searchsorted(y_unique, sample_y)
    ix = np.clip(ix, 1, len(x_unique) - 1)
    iy = np.clip(iy, 1, len(y_unique) - 1)

    left_x = x_unique[ix - 1]
    right_x = x_unique[ix]
    down_y = y_unique[iy - 1]
    up_y = y_unique[iy]

    ix = np.where(np.abs(sample_x - left_x) <= np.abs(sample_x - right_x), ix - 1, ix)
    iy = np.where(np.abs(sample_y - down_y) <= np.abs(sample_y - up_y), iy - 1, iy)

    return grid[iy, ix]


def extract_diagonal_45_slice(x, y, values, num_samples=None):
    x_unique, y_unique, grid = build_structured_grid(x, y, values)
    start = max(float(x_unique[0]), float(y_unique[0]))
    stop = min(float(x_unique[-1]), float(y_unique[-1]))

    if stop <= start:
        raise ValueError("Cannot extract y=x diagonal slice from this 2D grid")

    if num_samples is None:
        num_samples = min(len(x_unique), len(y_unique))

    line_coord = np.linspace(start, stop, num_samples)
    sampled_values = bilinear_sample(
        x_unique,
        y_unique,
        grid,
        line_coord,
        line_coord,
    )

    return line_coord, sampled_values, "y=x"


def extract_oblique45_normal_slice(x, y, values, num_samples=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    values = np.asarray(values, dtype=float)

    x_unique = np.sort(np.unique(x))
    y_unique = np.sort(np.unique(y))
    if len(x_unique) < 2 or len(y_unique) < 2:
        raise ValueError("extract_oblique45_normal_slice requires a 2D grid")

    dx = float(np.min(np.diff(x_unique)))
    dy = float(np.min(np.diff(y_unique)))
    half_strip_width = 0.5 * min(dx, dy) / np.sqrt(2.0)

    normal_coord = (x + y) / np.sqrt(2.0)
    tangent_coord = (-x + y) / np.sqrt(2.0)
    mask = (
        (normal_coord >= 0.0) &
        (normal_coord <= 1.0) &
        (np.abs(tangent_coord) <= half_strip_width + 1e-14)
    )

    if not np.any(mask):
        raise ValueError("No cell centres found in the 45-degree diagnostic strip")

    order = np.argsort(normal_coord[mask])
    return normal_coord[mask][order], values[mask][order], "s=(x+y)/sqrt(2)"


# [3] Build Paths
# [4] Resolution Selection
def extract_resolution(csv_path, dimension):
    if dimension == 1:
        match = re.search(r"_N(\d+)\.csv$", csv_path.name)
    else:
        match = re.search(r"_N(\d+)_N\1\.csv$", csv_path.name)

    if match is None:
        return None

    return int(match.group(1))


def available_resolutions(data_root, method, dimension, test_name, name_suffix=""):
    return common_available_resolutions(data_root, method, dimension, test_name, name_suffix)


def choose_resolution(data_root, method, test_name, requested_resolution, two_d_name_suffix=""):
    one_d = set(available_resolutions(data_root, method, 1, test_name))
    two_d = set(available_resolutions(
        data_root,
        method,
        2,
        test_name,
        two_d_name_suffix,
    ))
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


def make_validation_figure(
    primary_velocity_label=r"X-velocity, $u_x$",
    transverse_velocity_label=r"Y-velocity, $u_y$",
    include_transverse=True,
):
    if include_transverse:
        fig, axes = plt.subplots(2, 3, figsize=(10.8, 5.8))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.2))

    axes = axes.flatten()

    for ax, (field_key, _) in zip(axes[:4], FIELD_NAMES):
        configure_profile_axis(ax, field_key, show_legend=False, show_title=False)

    axes[1].set_ylabel(primary_velocity_label)

    if include_transverse:
        configure_profile_axis(axes[4], "u1", show_legend=False, show_title=False)
        axes[4].set_ylabel(transverse_velocity_label)
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


def oblique45_velocity_fields(fields):
    transformed = dict(fields)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    u0 = fields["u0"]
    u1 = fields["u1"]
    transformed["u0"] = (u0 + u1) * inv_sqrt2
    transformed["u1"] = (-u0 + u1) * inv_sqrt2
    return transformed


def plot_1d_vs_2d(
    data_root,
    output_dir,
    method,
    test_name,
    requested_resolution,
    exact_root,
    slice_kind="centerline",
    two_d_name_suffix="",
):
    resolution = choose_resolution(
        data_root,
        method,
        test_name,
        requested_resolution,
        two_d_name_suffix,
    )
    x_1d, fields_1d = load_1d_solution(
        solution_path(data_root, method, 1, test_name, resolution),
        test_name,
    )
    x_2d, y_2d, fields_2d = load_2d_solution(
        solution_path(data_root, method, 2, test_name, resolution, two_d_name_suffix),
        test_name,
    )

    if slice_kind == "centerline":
        slice_extractor = extract_centerline_x_slice
        slice_label = "2D centerline"
        output_slice_slug = "xslice"
        context_slice_label = "2D centerline"
        primary_velocity_label = r"X-velocity, $u_x$"
        transverse_velocity_label = r"Y-velocity, $u_y$"
        x_axis_label = r"$x$"
    elif slice_kind == "diagonal45":
        slice_extractor = extract_diagonal_45_slice
        slice_label = r"2D 45$^\circ$ slice"
        output_slice_slug = "diagonal45"
        context_slice_label = "2D 45-degree slice"
        primary_velocity_label = r"X-velocity, $u_x$"
        transverse_velocity_label = r"Y-velocity, $u_y$"
        x_axis_label = r"$x$"
    elif slice_kind == "oblique45":
        slice_extractor = extract_oblique45_normal_slice
        slice_label = r"2D oblique 45$^\circ$"
        output_slice_slug = "oblique45"
        context_slice_label = "2D oblique 45-degree normal slice"
        primary_velocity_label = r"Normal velocity, $u_n$"
        transverse_velocity_label = r"Tangential velocity, $u_t$"
        x_axis_label = r"$s=(x+y)/\sqrt{2}$"
        fields_2d = oblique45_velocity_fields(fields_2d)
    else:
        raise ValueError("slice_kind must be 'centerline', 'diagonal45', or 'oblique45'")

    fig, axes = make_validation_figure(
        primary_velocity_label=primary_velocity_label,
        transverse_velocity_label=transverse_velocity_label,
        include_transverse=(slice_kind != "oblique45"),
    )
    selected_y = None
    exact_fields = load_optional_exact_reference(
        exact_root,
        test_name,
        context=f"{test_name} {METHOD_STYLES[method]['label']} {context_slice_label}",
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

        x_slice, values_slice, selected_y = slice_extractor(
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
            label=slice_label,
            color="tab:red",
            marker="o",
            zorder=3,
        )
        ax.legend(frameon=False)

    if slice_kind != "oblique45":
        x_slice, uy_slice, selected_y = slice_extractor(
            x_2d,
            y_2d,
            fields_2d["u1"],
        )
        _, ux_slice, _ = slice_extractor(
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
            label=slice_label,
            color="tab:red",
            marker="o",
            zorder=3,
        )

        axes[4].legend(frameon=False)

    x_label_axes = axes[:4] if slice_kind == "oblique45" else axes[:5]
    for ax in x_label_axes:
        ax.set_xlabel(x_axis_label)

    label = METHOD_STYLES[method]["label"]
    slug = METHOD_STYLES[method]["slug"]
    plt.tight_layout()

    suffix = f"_{two_d_name_suffix}" if two_d_name_suffix else ""
    output_path = output_dir / f"{test_name}{suffix}_2d_validation_{slug}_1d_vs_2d_{output_slice_slug}_N{resolution}.png"
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
        default=Path("data/exact/generated_multimaterial"),
        help="Folder containing generated multimaterial exact CSVs named test1_exact.csv, test2_exact.csv, ...",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=sorted(METHOD_STYLES.keys()),
        default=sorted(METHOD_STYLES.keys()),
        help="Methods to validate separately.",
    )
    parser.add_argument(
        "--slices",
        nargs="+",
        choices=("centerline", "diagonal45", "oblique45"),
        default=("centerline", "diagonal45"),
        help="2D slice reductions to plot.",
    )
    parser.add_argument(
        "--two-d-name-suffix",
        default="",
        help="Suffix appended to the 2D Fedkiw output folder names, e.g. 45 for gfm_FedkiwA45.",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    remove_obsolete_plot_outputs(args.output_dir)

    selected_tests = args.tests or ([args.test] if args.test else sorted(TEST_FOLDERS.keys()))

    for test_name in selected_tests:
        for method in args.methods:
            for slice_kind in args.slices:
                plot_1d_vs_2d(
                    args.data_root,
                    args.output_dir,
                    method,
                    test_name,
                    args.overlay_n,
                    args.exact_root,
                    slice_kind=slice_kind,
                    two_d_name_suffix=args.two_d_name_suffix,
                )


if __name__ == "__main__":
    main()


# RUN Commands
# cd diffuse_vs_GFM_MMS
# source .venv/bin/activate
# python src/graphing/plot_gfm_dim_2d.py --overlay-n 400
