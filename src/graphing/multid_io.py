"""CSV/path helpers for multidimensional solver plots.

The legacy plotter accepts paths relative to `data/csv`. Keeping that convention
centralised here lets the compatibility CLI stay small while the render module
works with already-normalised arrays and labels.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd


def load_solution_csv(filename):
    """Load one 2D/3D solver CSV and return fields in the old plotter shape."""

    input_path = Path(filename)
    data_path = input_path if input_path.is_absolute() else Path("data/csv") / input_path

    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found")

    df = pd.read_csv(data_path)

    dim = 0
    while f"x{dim}" in df.columns:
        dim += 1

    if dim not in (2, 3):
        raise ValueError(f"plot_multid.py only supports 2D or 3D CSV files, found dimension={dim}")

    coords = [df[f"x{d}"].to_numpy() for d in range(dim)]
    velocity = [df[f"u{d}"].to_numpy() for d in range(dim)]

    rho = df["rho"].to_numpy()
    p = df["p"].to_numpy()
    e = df["e"].to_numpy()
    # Interface-aware plots need optional SIM/DIM columns, but regular field plots should not care whether those columns are present.
    extra_fields = {
        column: df[column].to_numpy()
        for column in df.columns
        if (
            column.startswith("phi")
            or column.startswith("alpha")
            or column.startswith("rho_mat")
            or column.startswith("mass")
            or column == "mat"
        )
    }

    return dim, coords, velocity, rho, p, e, extra_fields


def build_label_from_filename(fname):
    name = Path(fname).name.lower()

    if "exact" in name:
        return "Exact"

    n_tokens = []
    for part in name.replace(".csv", "").split("_"):
        if part.startswith("n") and len(part) > 1 and part[1:].isdigit():
            n_tokens.append(part[1:])

    if n_tokens:
        return "N=" + "x".join(n_tokens)

    return Path(fname).stem


def time_from_filename(fname):
    match = re.search(r"_t([0-9peEm+\-]+)_N", Path(fname).name)
    if match is None:
        return None

    text = match.group(1)
    tagged = re.fullmatch(r"([0-9]+)p([0-9]+)e([pm+\-])([0-9]+)", text)

    if tagged is not None:
        integer, fraction, sign_token, exponent = tagged.groups()
        sign = "-" if sign_token in {"m", "-"} else "+"
        return float(f"{integer}.{fraction}e{sign}{exponent}")

    text = text.replace("m", "-").replace("p", ".")
    return float(text)


def format_time_label(time_value):
    exponent = int(np.floor(np.log10(abs(time_value)))) if time_value != 0.0 else 0
    mantissa = time_value / (10.0 ** exponent)
    return rf"$t = {mantissa:.3f}\times 10^{{{exponent}}}$"


# [6] Build output name from folder path
def build_output_name(folder_path: str):
    p = Path(folder_path)
    parts = p.parts[-2:]
    return "_".join(parts)


def time_tagged_csv_files(folder_path):
    """Return `(time, path)` pairs for shock-bubble sequence outputs."""

    files = []

    for csv_path in sorted(folder_path.glob("*.csv")):
        if "_N" not in csv_path.name:
            continue

        time_value = time_from_filename(csv_path.name)
        if time_value is not None:
            files.append((time_value, csv_path))

    return sorted(files, key=lambda item: item[0])


def remove_if_exists(path):
    if path is not None and path.exists():
        path.unlink()
