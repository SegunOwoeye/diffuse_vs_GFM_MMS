# [0] Import Libraries
from pathlib import Path
import argparse
import re

import matplotlib.pyplot as plt
import pandas as pd


# [1] Test / Folder Settings
TEST_FOLDERS = {
    "test1": "FedkiwA",
    "test2": "FedkiwB",
    "test3": "FedkiwC",
    "test4": "FedkiwD1",
    "test5": "FedkiwD2",
}

METHOD_STYLES = {
    "gfm": {
        "label": "GFM",
        "linestyle": "-",
    },
    "dim": {
        "label": "DIM",
        "linestyle": "--",
    },
}

FIELD_NAMES = [
    ("rho", "Density"),
    ("u0", "Velocity"),
    ("p", "Pressure"),
    ("e_plot", "Specific internal energy"),
]


# [2] Load 1D CSV Data
def choose_energy_column(df, energy_column):
    if energy_column == "auto":
        return "e"

    if energy_column not in df.columns:
        raise KeyError(f"{energy_column} not found in CSV columns for energy plot")

    return energy_column


def load_solution_csv(csv_path, energy_column="auto"):
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found")

    df = pd.read_csv(csv_path)
    energy_key = choose_energy_column(df, energy_column)

    x = df["x0"].to_numpy()
    fields = {
        "rho": df["rho"].to_numpy(),
        "u0": df["u0"].to_numpy(),
        "p": df["p"].to_numpy(),
        "e_plot": df[energy_key].to_numpy(),
    }

    return x, fields


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
        raise FileNotFoundError(f"No common GFM / DIM resolutions found for {test_name}")

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
def energy_label(energy_column):
    if energy_column == "e":
        return "Specific internal energy"

    return "Specific internal energy"


def make_four_panel_figure(title, energy_column):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()
    field_names = FIELD_NAMES.copy()
    field_names[-1] = ("e_plot", energy_label(energy_column))

    for ax, (_, field_name) in zip(axes, field_names):
        ax.set_xlabel("Position x")
        ax.set_ylabel(field_name)
        ax.grid(True, alpha=0.35)

    fig.suptitle(title)

    return fig, axes


# [7] Plot One Resolution: GFM vs DIM
def plot_method_overlay(data_root, output_dir, test_name, requested_resolution, energy_column):
    resolution = choose_overlay_resolution(data_root, test_name, requested_resolution)
    fig, axes = make_four_panel_figure(
        f"{test_name.upper()} GFM vs DIM at N={resolution}",
        energy_column,
    )

    for method, style in METHOD_STYLES.items():
        csv_path = solution_path(data_root, method, test_name, resolution)
        x, fields = load_solution_csv(csv_path, energy_column)

        for ax, (field_key, _) in zip(axes, FIELD_NAMES):
            ax.plot(
                x,
                fields[field_key],
                linestyle=style["linestyle"],
                linewidth=1.4,
                label=style["label"],
            )

    for ax in axes:
        ax.legend()

    output_path = output_dir / f"{test_name}_gfm_dim_overlay_N{resolution}.png"
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved figure to {output_path}")


# [8] Plot Convergence: One Method At A Time
def plot_method_convergence(data_root, output_dir, method, test_name, energy_column):
    resolutions = available_resolutions(data_root, method, test_name)

    if not resolutions:
        raise FileNotFoundError(f"No {method.upper()} resolutions found for {test_name}")

    method_label = METHOD_STYLES[method]["label"]
    fig, axes = make_four_panel_figure(
        f"{test_name.upper()} {method_label} convergence",
        energy_column,
    )

    for resolution in resolutions:
        csv_path = solution_path(data_root, method, test_name, resolution)
        x, fields = load_solution_csv(csv_path, energy_column)

        for ax, (field_key, _) in zip(axes, FIELD_NAMES):
            ax.plot(
                x,
                fields[field_key],
                linewidth=1.3,
                label=f"N={resolution}",
            )

    for ax in axes:
        ax.legend()

    output_path = output_dir / f"{test_name}_{method}_convergence.png"
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved figure to {output_path}")


# [9] Script Entry
def main():
    parser = argparse.ArgumentParser(
        description="Plot 1D GFM and DIM method comparisons for Fedkiw validation tests."
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
        "--include-convergence-overlay",
        action="store_true",
        help="Also make the single-resolution overlay for the convergence test.",
    )
    parser.add_argument(
        "--energy-column",
        choices=["auto", "e",],
        default="auto",
        help="Column used for the internal-energy panel.",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for method in METHOD_STYLES:
        plot_method_convergence(
            args.data_root,
            args.output_dir,
            method,
            args.convergence_test,
            args.energy_column,
        )

    for test_name in sorted(TEST_FOLDERS.keys()):
        if test_name == args.convergence_test and not args.include_convergence_overlay:
            continue

        plot_method_overlay(
            args.data_root,
            args.output_dir,
            test_name,
            args.overlay_n,
            args.energy_column,
        )


if __name__ == "__main__":
    main()


# RUN Commands
# cd diffuse_vs_GFM_MMS
# source .venv/bin/activate
# python src/graphing/plot_gfm_dim_1d.py --convergence-test test5 --overlay-n 400 --include-convergence-overlay
