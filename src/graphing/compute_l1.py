# [0] Import libraries
from pathlib import Path
import re

import numpy as np
import pandas as pd


FIELDS = ("rho", "u", "p", "e")


# [1] Load CSV (dimension-aware)
def load_csv(path):
    df = pd.read_csv(path)

    dim = 0
    while f"x{dim}" in df.columns:
        dim += 1

    coords = [df[f"x{d}"].to_numpy() for d in range(dim)]
    if "e" in df.columns:
        energy_like = df["e"].to_numpy()
    elif "entropy" in df.columns:
        energy_like = df["entropy"].to_numpy()
    else:
        raise KeyError(f"{path} must contain either an e or entropy column")

    fields = {
        "rho": df["rho"].to_numpy(),
        "u": df["u0"].to_numpy(),
        "p": df["p"].to_numpy(),
        "e": energy_like,
    }

    return dim, coords, fields


# [2] Grid helpers
def extract_resolution(filename):
    name = Path(filename).name.lower().replace(".csv", "")
    nums = [int(n) for n in re.findall(r"(?:^|_)n(\d+)", name)]
    return tuple(nums) if nums else None


def representative_resolution(resolution):
    if resolution is None:
        return None
    if len(set(resolution)) == 1:
        return resolution[0]
    return int(round(np.prod(resolution) ** (1.0 / len(resolution))))


def infer_shape(coords):
    return tuple(len(np.unique(c)) for c in coords)


def infer_cell_volume(coords):
    volume = 1.0
    for coord in coords:
        unique = np.unique(coord)
        if len(unique) < 2:
            dx = 1.0
        else:
            dx = float(np.median(np.diff(unique)))
        volume *= abs(dx)
    return volume


# [3] Norms and observed order
def compute_norms(num, ref, cell_volume):
    diff = np.asarray(num) - np.asarray(ref)
    abs_diff = np.abs(diff)

    return {
        "L1": cell_volume * np.sum(abs_diff),
        "L2": np.sqrt(cell_volume * np.sum(diff * diff)),
        "Linf": np.max(abs_diff),
    }


def add_observed_orders(rows):
    rows.sort(key=lambda row: row["N"])

    for row in rows:
        for field in FIELDS:
            row[f"order_{field}_L1"] = np.nan

    for i in range(1, len(rows)):
        n0 = rows[i - 1]["N"]
        n1 = rows[i]["N"]

        if n0 <= 0 or n1 <= n0:
            continue

        for field in FIELDS:
            e0 = rows[i - 1][f"{field}_L1"]
            e1 = rows[i][f"{field}_L1"]

            if e0 > 0.0 and e1 > 0.0:
                rows[i][f"order_{field}_L1"] = np.log(e0 / e1) / np.log(n1 / n0)

    return rows


def add_field_norms(row, num_fields, ref_fields, cell_volume):
    for field in FIELDS:
        norms = compute_norms(num_fields[field], ref_fields[field], cell_volume)
        row[f"{field}_L1"] = norms["L1"]
        row[f"{field}_L2"] = norms["L2"]
        row[f"{field}_Linf"] = norms["Linf"]


# [4] Exact-reference errors
def compute_errors_exact(files):
    exact_file = None
    num_files = []

    for f in files:
        name = f.name.lower()
        if "exact" in name or "reference" in name:
            exact_file = f
        else:
            num_files.append(f)

    if exact_file is None:
        return None

    dim_exact, coords_exact, fields_exact = load_csv(exact_file)

    if dim_exact != 1:
        raise RuntimeError("Exact-reference comparison currently expects 1D exact data")

    rows = []
    x_exact = coords_exact[0]

    for f in num_files:
        resolution = extract_resolution(f.name)
        if resolution is None:
            continue

        dim, coords, fields = load_csv(f)
        if dim != 1:
            raise RuntimeError("Exact-reference comparison currently expects 1D numerical data")

        ref_fields = {
            field: np.interp(coords[0], x_exact, fields_exact[field])
            for field in FIELDS
        }

        row = {
            "N": representative_resolution(resolution),
            "resolution": "x".join(str(n) for n in resolution),
        }
        add_field_norms(row, fields, ref_fields, infer_cell_volume(coords))
        rows.append(row)

    return add_observed_orders(rows)


# [5] Numerical-reference errors
def compute_errors_reference(files):
    valid_files = []
    for f in files:
        resolution = extract_resolution(f.name)
        if resolution is not None:
            valid_files.append((f, resolution))

    valid_files.sort(key=lambda item: representative_resolution(item[1]))

    if len(valid_files) < 2:
        raise RuntimeError("Need at least two numerical resolutions for reference comparison")

    ref_file, ref_resolution = valid_files[-1]
    dim_ref, coords_ref, fields_ref_flat = load_csv(ref_file)
    shape_ref = infer_shape(coords_ref)

    fields_ref = {
        field: values.reshape(shape_ref, order="F")
        for field, values in fields_ref_flat.items()
    }

    rows = []

    for f, resolution in valid_files[:-1]:
        dim, coords, fields_flat = load_csv(f)

        if dim != dim_ref:
            raise RuntimeError(f"Dimension mismatch: {f} vs {ref_file}")

        shape = infer_shape(coords)

        if len(shape) != len(shape_ref):
            raise RuntimeError(f"Shape mismatch: {f} vs {ref_file}")

        ratios = []
        for n_ref, n_coarse in zip(shape_ref, shape):
            if n_ref % n_coarse != 0:
                raise RuntimeError(f"Incompatible grids: {shape_ref} vs {shape}")
            ratios.append(n_ref // n_coarse)

        slicer = tuple(slice(None, None, ratio) for ratio in ratios)

        fields = {
            field: values.reshape(shape, order="F")
            for field, values in fields_flat.items()
        }
        ref_fields = {
            field: values[slicer]
            for field, values in fields_ref.items()
        }

        row = {
            "N": representative_resolution(resolution),
            "resolution": "x".join(str(n) for n in resolution),
        }
        add_field_norms(row, fields, ref_fields, infer_cell_volume(coords))
        rows.append(row)

    return add_observed_orders(rows)


# [6] Dispatcher
def compute_errors(folder):
    folder_path = Path("data/csv") / folder
    files = sorted(
        f for f in folder_path.glob("*.csv")
        if "_N" in f.name and "conservation" not in f.name.lower()
    )

    if not files:
        raise RuntimeError("No CSV files found")

    results = compute_errors_exact(files)

    if results is not None:
        print("Using exact solution for error computation")
        return results

    print("Exact solution not found -> using finest numerical solution as reference")
    return compute_errors_reference(files)


# [7] Save table
def save_error_table(results, folder):
    folder_path = Path("data/csv") / folder
    output_name = "_".join(Path(folder).parts[-2:])
    save_path = folder_path / f"{output_name}_errors.csv"

    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)

    print(f"Saved error table to {save_path}")


# [8] Run
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python compute_l1.py folder_name")
        exit(1)

    folder = sys.argv[1]
    results = compute_errors(folder)

    print("\nL1 Errors and Observed L1 Orders:")
    for row in results:
        order = row["order_rho_L1"]
        order_text = "nan" if np.isnan(order) else f"{order:.3f}"
        print(
            f"N={row['N']} | "
            f"rho={row['rho_L1']:.4e} "
            f"u={row['u_L1']:.4e} "
            f"p={row['p_L1']:.4e} "
            f"e={row['e_L1']:.4e} "
            f"| O_rho={order_text}"
        )

    save_error_table(results, folder)
