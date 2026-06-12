"""Shared Fedkiw/GFM-DIM plotting helpers.

The 1D and 2D Fedkiw comparison plotters use the same naming convention but
used to duplicate path, resolution, and entropy logic. This module is the
single source for those report/validation conventions.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


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


def infer_material_ids(df: pd.DataFrame) -> np.ndarray | pd.Series | None:
    """Infer material labels from either sharp `mat` or diffuse `alpha*` data."""

    if "mat" in df.columns:
        return df["mat"].round().astype(int)

    alpha_columns = sorted([
        column for column in df.columns
        if column.startswith("alpha")
    ])

    if alpha_columns:
        return df[alpha_columns].to_numpy().argmax(axis=1)

    return None


def compute_entropy(df: pd.DataFrame, test_name: str) -> np.ndarray:
    """Compute side-specific ideal-gas entropy using the Fedkiw case gammas."""

    material_ids = infer_material_ids(df)
    gammas = TEST_GAMMAS[test_name]

    if material_ids is None:
        gamma = gammas[0]
    else:
        gamma = np.asarray([gammas[int(mat)] for mat in material_ids])

    return df["p"].to_numpy() / (df["rho"].to_numpy() ** gamma)


def fedkiw_name(test_name: str) -> str:
    return TEST_FOLDERS[test_name]


def solution_folder(
    data_root: Path,
    method: str,
    dimension: int,
    test_name: str,
    name_suffix: str = "",
) -> Path:
    name = fedkiw_name(test_name)
    if dimension == 2:
        name = f"{name}{name_suffix}"

    return data_root / method / f"MM_{dimension}D_validation" / f"{method}_{name}"


def solution_path(
    data_root: Path,
    method: str,
    dimension: int,
    test_name: str,
    resolution: int,
    name_suffix: str = "",
) -> Path:
    name = fedkiw_name(test_name)
    if dimension == 2:
        name = f"{name}{name_suffix}"

    folder = solution_folder(data_root, method, dimension, test_name, name_suffix)

    if dimension == 1:
        return folder / f"{method}_{name}_N{resolution}.csv"

    return folder / f"{method}_{name}_N{resolution}_N{resolution}.csv"


def extract_resolution(csv_path: Path, dimension: int = 1) -> int | None:
    if dimension == 1:
        match = re.search(r"_N(\d+)\.csv$", csv_path.name)
    else:
        match = re.search(r"_N(\d+)_N\1\.csv$", csv_path.name)

    if match is None:
        return None

    return int(match.group(1))


def available_resolutions(
    data_root: Path,
    method: str,
    dimension: int,
    test_name: str,
    name_suffix: str = "",
) -> list[int]:
    folder = solution_folder(data_root, method, dimension, test_name, name_suffix)

    if not folder.exists():
        return []

    resolutions = []

    for csv_path in folder.glob("*.csv"):
        resolution = extract_resolution(csv_path, dimension)

        if resolution is not None:
            resolutions.append(resolution)

    return sorted(set(resolutions))
