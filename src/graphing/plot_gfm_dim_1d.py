# [0] Import Libraries
from pathlib import Path
import argparse
import re

import matplotlib.pyplot as plt
import pandas as pd

from plot_style import (
    configure_profile_axis,
    plot_profile,
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

METHOD_STYLES = {
    "gfm": {
        "label": "rGFM",
        "color": "tab:blue",
        "marker": "o",
    },
    "dim": {
        "label": "Allaire 5eq",
        "color": "tab:red",
        "marker": "s",
    },
}

FIELD_NAMES = [
    ("rho", "Density"),
    ("u0", "Velocity"),
    ("p", "Pressure"),
    ("entropy", "Entropy"),
]

LEGACY_OUTPUT_PATTERNS = (
    "*_gfm_convergence.png",
    "*_dim_convergence.png",
)


# [2] Load 1D CSV Data
def choose_energy_column(df, energy_column):
    if energy_column == "auto":
        return "e"

    if energy_column not in df.columns:
        raise KeyError(f"{energy_column} not found in CSV columns for energy plot")

    return energy_column


def infer_material_ids(df):
    if "mat" in df.columns:
        return df["mat"].round().astype(int)

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
        gamma = pd.Series(material_ids).map(lambda mat: gammas[int(mat)]).to_numpy()

    return df["p"].to_numpy() / (df["rho"].to_numpy() ** gamma)


def fourth_panel_label(panel):
    if panel == "energy":
        return "Specific internal energy"

    return "Entropy"


def load_solution_csv(csv_path, test_name, fourth_panel="entropy", energy_column="auto"):
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found")

    df = pd.read_csv(csv_path)

    if fourth_panel == "energy":
        energy_key = choose_energy_column(df, energy_column)
        fourth_values = df[energy_key].to_numpy()
    else:
        fourth_values = compute_entropy(df, test_name)

    x = df["x0"].to_numpy()
    fields = {
        "rho": df["rho"].to_numpy(),
        "u0": df["u0"].to_numpy(),
        "p": df["p"].to_numpy(),
        "entropy": fourth_values,
        "e_plot": fourth_values,
    }

    return x, fields


def plotted_field_names(fourth_panel):
    field_names = FIELD_NAMES.copy()
    field_names[-1] = (
        "e_plot" if fourth_panel == "energy" else "entropy",
        fourth_panel_label(fourth_panel),
    )
    return field_names


def remove_legacy_plot_outputs(output_dir):
    for pattern in LEGACY_OUTPUT_PATTERNS:
        for path in output_dir.glob(pattern):
            path.unlink()


# [3] Build Paths For Each Method
def method_folder(data_root, method, test_name):
    fedkiw_name = TEST_FOLDERS[test_name]
    folder_name = f"{method}_{fedkiw_name}"

    return data_root / method / "MM_1D_validation" / folder_name


def solution_path(data_root, method, test_name, resolution):
    fedkiw_name = TEST_FOLDERS[test_name]
    folder = method_folder(data_root, method, test_name)

    return folder / f"{method}_{fedkiw_name}_N{resolution}.csv"


# [4] Find Available Resolutions
def extract_resolution(csv_path):
    match = re.search(r"_N(\d+)\.csv$", csv_path.name)

    if match is None:
        return None

    return int(match.group(1))


def available_resolutions(data_root, method, test_name):
    folder = method_folder(data_root, method, test_name)

    if not folder.exists():
        return []

    resolutions = []

    for csv_path in folder.glob("*.csv"):
        resolution = extract_resolution(csv_path)

        if resolution is not None:
            resolutions.append(resolution)

    return sorted(set(resolutions))


def common_resolutions(data_root, test_name):
    gfm_resolutions = set(available_resolutions(data_root, "gfm", test_name))
    dim_resolutions = set(available_resolutions(data_root, "dim", test_name))

    return sorted(gfm_resolutions.intersection(dim_resolutions))


# [5] Pick The Overlay Resolution
def choose_overlay_resolution(data_root, test_name, requested_resolution):
    resolutions = common_resolutions(data_root, test_name)

    if not resolutions:
        raise FileNotFoundError(f"No common rGFM / Allaire 5eq resolutions found for {test_name}")

    if requested_resolution in resolutions:
        return requested_resolution

    higher = [resolution for resolution in resolutions if resolution > requested_resolution]

    if higher:
        chosen = min(higher)
    else:
        chosen = max(resolutions)

    print(
        f"{test_name}: N={requested_resolution} was not available for both methods, "
        f"using N={chosen} instead"
    )

    return chosen


# [6] Create The Standard Four-Panel Figure
def make_four_panel_figure(fourth_panel):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    axes = axes.flatten()
    field_names = plotted_field_names(fourth_panel)

    for ax, (field_key, _) in zip(axes, field_names):
        configure_profile_axis(ax, field_key, show_legend=False, show_title=False)

    return fig, axes


# [7] Plot One Resolution: rGFM vs Allaire
def plot_exact_reference(axes, field_names, exact_fields):
    if not exact_fields:
        return

    for ax, (field_key, _) in zip(axes, field_names):
        if field_key in exact_fields:
            exact_x, exact_y = exact_fields[field_key]
            plot_profile(ax, exact_x, exact_y, "Exact", index=0)


def plot_method_overlay(
    data_root,
    output_dir,
    test_name,
    requested_resolution,
    fourth_panel,
    energy_column,
    exact_root,
):
    resolution = choose_overlay_resolution(data_root, test_name, requested_resolution)
    field_names = plotted_field_names(fourth_panel)
    fig, axes = make_four_panel_figure(fourth_panel)
    exact_fields = load_optional_exact_reference(
        exact_root,
        test_name,
        context=f"{test_name} 1D overlay",
    )
    plot_exact_reference(axes, field_names, exact_fields)

    for method, style in METHOD_STYLES.items():
        csv_path = solution_path(data_root, method, test_name, resolution)
        x, fields = load_solution_csv(csv_path, test_name, fourth_panel, energy_column)

        for ax, (field_key, _) in zip(axes, field_names):
            ax.plot(
                x,
                fields[field_key],
                linestyle="None",
                linewidth=0.0,
                marker=style["marker"],
                markersize=2.8,
                markerfacecolor="none",
                markeredgewidth=0.8,
                color=style["color"],
                label=style["label"],
            )

    for ax in axes:
        ax.legend(frameon=False)

    output_path = output_dir / f"{test_name}_rgfm_allaire5eq_overlay_N{resolution}.png"
    plt.tight_layout()
    save_figure(fig, output_path)
    return output_path, resolution


# [8] Plot Convergence: One Method At A Time
def plot_method_convergence(
    data_root,
    output_dir,
    method,
    test_name,
    fourth_panel,
    energy_column,
    exact_root,
):
    resolutions = available_resolutions(data_root, method, test_name)

    if not resolutions:
        raise FileNotFoundError(f"No {METHOD_STYLES[method]['label']} resolutions found for {test_name}")

    field_names = plotted_field_names(fourth_panel)
    fig, axes = make_four_panel_figure(fourth_panel)
    exact_fields = load_optional_exact_reference(
        exact_root,
        test_name,
        context=f"{test_name} {METHOD_STYLES[method]['label']} convergence",
    )
    plot_exact_reference(axes, field_names, exact_fields)

    for index, resolution in enumerate(sorted(resolutions, reverse=True)):
        csv_path = solution_path(data_root, method, test_name, resolution)
        x, fields = load_solution_csv(csv_path, test_name, fourth_panel, energy_column)

        for ax, (field_key, _) in zip(axes, field_names):
            plot_profile(ax, x, fields[field_key], f"N={resolution}", index=index)

    for ax in axes:
        ax.legend(frameon=False)

    method_slug = "rgfm" if method == "gfm" else "allaire5eq"
    output_path = output_dir / f"{test_name}_{method_slug}_convergence.png"
    plt.tight_layout()
    save_figure(fig, output_path)


# [9] Script Entry
def main():
    parser = argparse.ArgumentParser(
        description="Plot 1D rGFM and Allaire five-equation method comparisons for Fedkiw validation tests."
    )
    parser.add_argument(
        "--convergence-test",
        choices=sorted(TEST_FOLDERS.keys()),
        default="test5",
        help="Test used for the full convergence comparison.",
    )
    parser.add_argument(
        "--overlay-n",
        type=int,
        default=200,
        help="Requested resolution for the single-resolution overlay plots.",
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
        default=Path("data/plots/1d_validation"),
        help="Folder where the comparison figures will be saved.",
    )
    parser.add_argument(
        "--exact-root",
        type=Path,
        default=Path("data/exact/fedkiw"),
        help="Folder containing optional digitised exact CSVs named test1_exact.csv, test2_exact.csv, ...",
    )
    parser.add_argument(
        "--include-convergence-overlay",
        action="store_true",
        help="Also make the single-resolution overlay for the convergence test.",
    )
    parser.add_argument(
        "--fourth-panel",
        choices=["entropy", "energy"],
        default="entropy",
        help="Quantity plotted in the fourth panel.",
    )
    parser.add_argument(
        "--energy-column",
        choices=["auto", "e",],
        default="auto",
        help="Column used when --fourth-panel energy is selected.",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    remove_legacy_plot_outputs(args.output_dir)

    for method in METHOD_STYLES:
        plot_method_convergence(
            args.data_root,
            args.output_dir,
            method,
            args.convergence_test,
            args.fourth_panel,
            args.energy_column,
            args.exact_root,
        )

    for test_name in sorted(TEST_FOLDERS.keys()):
        if test_name == args.convergence_test and not args.include_convergence_overlay:
            continue

        plot_method_overlay(
            args.data_root,
            args.output_dir,
            test_name,
            args.overlay_n,
            args.fourth_panel,
            args.energy_column,
            args.exact_root,
        )


if __name__ == "__main__":
    main()


# RUN Commands
# cd diffuse_vs_GFM_MMS
# source .venv/bin/activate
# python src/graphing/plot_gfm_dim_1d.py --convergence-test test5 --overlay-n 400 --include-convergence-overlay
