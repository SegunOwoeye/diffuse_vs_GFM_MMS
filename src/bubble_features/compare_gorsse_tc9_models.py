#!/usr/bin/env python3
"""Compare TC9 model interface features against the paper-image reference."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bubble_feature_core import extract_main_phi_contour, load_grid  # noqa: E402


def parse_time_seconds(path: Path) -> float | None:
    match = re.search(r"_t([0-9]+p[0-9]+)e([pm+\-])([0-9]+)", path.name)
    if match is None:
        return None
    mantissa = float(match.group(1).replace("p", "."))
    exponent = int(match.group(3))
    if match.group(2) in {"m", "-"}:
        exponent *= -1
    return mantissa * 10.0**exponent


def largest_mask_component(mask: np.ndarray) -> np.ndarray:
    labels = np.zeros(mask.shape, dtype=np.int32)
    best_label = 0
    best_size = 0
    label = 0
    offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))

    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            if not mask[row, col] or labels[row, col] != 0:
                continue
            label += 1
            size = 0
            stack = [(row, col)]
            labels[row, col] = label
            while stack:
                current_row, current_col = stack.pop()
                size += 1
                for row_offset, col_offset in offsets:
                    next_row = current_row + row_offset
                    next_col = current_col + col_offset
                    if next_row < 0 or next_row >= mask.shape[0]:
                        continue
                    if next_col < 0 or next_col >= mask.shape[1]:
                        continue
                    if not mask[next_row, next_col] or labels[next_row, next_col] != 0:
                        continue
                    labels[next_row, next_col] = label
                    stack.append((next_row, next_col))
            if size > best_size:
                best_size = size
                best_label = label

    return labels == best_label


def summarize_alpha_support(
    csv_path: Path,
    method: str,
    *,
    threshold: float,
) -> dict[str, float | str]:
    df = pd.read_csv(csv_path)
    if "alpha1" not in df.columns:
        raise RuntimeError(f"{csv_path} has no alpha1 column for DIM support fallback")

    x0 = np.sort(df["x0"].unique())
    x1 = np.sort(df["x1"].unique())
    alpha = df.pivot(index="x1", columns="x0", values="alpha1").loc[x1, x0].to_numpy()
    mask = largest_mask_component(alpha >= threshold)
    rows, cols = np.where(mask)
    if rows.size == 0:
        raise RuntimeError(f"No alpha1 >= {threshold:g} support found in {csv_path}")

    weights = alpha[rows, cols]
    selected_x0 = x0[cols]
    selected_x1 = x1[rows]
    time_seconds = parse_time_seconds(csv_path)
    time_us = np.nan if time_seconds is None else 1.0e6 * time_seconds
    return {
        "method": method,
        "csv_file": str(csv_path),
        "time_s": np.nan if time_seconds is None else float(time_seconds),
        "time_us": float(time_us),
        "interface_label": f"alpha1>={threshold:g}",
        "extraction_mode": "largest alpha1 support",
        "contour_point_count": int(rows.size),
        "x0_min_m": float(np.nanmin(selected_x0)),
        "x0_max_m": float(np.nanmax(selected_x0)),
        "x1_min_m": float(np.nanmin(selected_x1)),
        "x1_max_m": float(np.nanmax(selected_x1)),
        "centroid_x0_m": float(np.average(selected_x0, weights=weights)),
        "centroid_x1_m": float(np.average(selected_x1, weights=weights)),
    }


def summarize_contour(
    csv_path: Path,
    method: str,
    *,
    dim_support_threshold: float,
) -> dict[str, float | str]:
    grid = load_grid(csv_path)
    try:
        contour = extract_main_phi_contour(grid)
        extraction_mode = "zero contour"
    except RuntimeError:
        if method != "DIM":
            raise
        return summarize_alpha_support(
            csv_path,
            method,
            threshold=dim_support_threshold,
        )
    x0 = contour[:, 0]
    x1 = contour[:, 1]
    time_seconds = parse_time_seconds(csv_path)
    time_us = np.nan if time_seconds is None else 1.0e6 * time_seconds
    return {
        "method": method,
        "csv_file": str(csv_path),
        "time_s": np.nan if time_seconds is None else float(time_seconds),
        "time_us": float(time_us),
        "interface_label": grid.interface_label,
        "extraction_mode": extraction_mode,
        "contour_point_count": int(len(contour)),
        "x0_min_m": float(np.nanmin(x0)),
        "x0_max_m": float(np.nanmax(x0)),
        "x1_min_m": float(np.nanmin(x1)),
        "x1_max_m": float(np.nanmax(x1)),
        "centroid_x0_m": float(np.nanmean(x0)),
        "centroid_x1_m": float(np.nanmean(x1)),
    }


def collect_model_summaries(
    directory: Path,
    glob_pattern: str,
    method: str,
    *,
    dim_support_threshold: float,
) -> pd.DataFrame:
    rows = []
    for csv_path in sorted(directory.glob(glob_pattern)):
        if "runtime" in csv_path.name or "conservation" in csv_path.name:
            continue
        rows.append(
            summarize_contour(
                csv_path,
                method,
                dim_support_threshold=dim_support_threshold,
            )
        )
    if not rows:
        raise FileNotFoundError(f"No solver CSV files matched {directory / glob_pattern}")
    return pd.DataFrame(rows).sort_values(["time_us", "csv_file"]).reset_index(drop=True)


def reference_summary(path: Path, *, x_offset: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["component_id"] == 0].copy()
    for column in ["x0_min_m", "x0_max_m", "centroid_x0_m"]:
        df[column] = df[column] + x_offset
    keep = [
        "time_us",
        "x0_min_m",
        "x0_max_m",
        "x1_min_m",
        "x1_max_m",
        "centroid_x0_m",
        "centroid_x1_m",
    ]
    return df[keep].rename(columns={column: f"reference_{column}" for column in keep if column != "time_us"})


def nearest_reference_rows(model: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, model_row in model.iterrows():
        time_us = float(model_row["time_us"])
        ref_index = (reference["time_us"] - time_us).abs().idxmin()
        ref_row = reference.loc[ref_index]
        row = {**model_row.to_dict(), **ref_row.to_dict()}
        row["reference_time_us"] = float(ref_row["time_us"])
        row["time_delta_us"] = time_us - float(ref_row["time_us"])
        for metric in [
            "x0_min_m",
            "x0_max_m",
            "x1_min_m",
            "x1_max_m",
            "centroid_x0_m",
            "centroid_x1_m",
        ]:
            row[f"{metric}_error_m"] = float(model_row[metric]) - float(ref_row[f"reference_{metric}"])
        rows.append(row)
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-summary",
        type=Path,
        default=Path("data/bubble_collapse_validation/gorsse_2014_tc9/gorsse_tc9_reference_interface_summary.csv"),
    )
    parser.add_argument("--gfm-dir", type=Path, default=Path("data/csv/gfm/gorsse_2014_lowres/gfm_gorsse_tc9_water_air_bubble_2d_lowres"))
    parser.add_argument("--dim-dir", type=Path, default=Path("data/csv/dim/gorsse_2014_lowres/dim_gorsse_tc9_water_air_bubble_2d_lowres"))
    parser.add_argument("--gfm-glob", default="*.csv")
    parser.add_argument("--dim-glob", default="*.csv")
    parser.add_argument("--dim-support-threshold", type=float, default=0.20)
    parser.add_argument(
        "--reference-x-offset",
        type=float,
        default=0.0,
        help="Optional offset applied to reference x0 coordinates before comparison.",
    )
    parser.add_argument("--outdir", type=Path, default=Path("results/bubble_features/gorsse_2014_tc9"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    reference = reference_summary(args.reference_summary, x_offset=args.reference_x_offset)
    gfm = collect_model_summaries(
        args.gfm_dir,
        args.gfm_glob,
        "rGFM",
        dim_support_threshold=args.dim_support_threshold,
    )
    dim = collect_model_summaries(
        args.dim_dir,
        args.dim_glob,
        "DIM",
        dim_support_threshold=args.dim_support_threshold,
    )
    model = pd.concat([gfm, dim], ignore_index=True)

    model_summary_path = args.outdir / "gorsse_tc9_model_interface_summary.csv"
    comparison_path = args.outdir / "gorsse_tc9_model_vs_reference.csv"
    model.to_csv(model_summary_path, index=False)
    nearest_reference_rows(model, reference).to_csv(comparison_path, index=False)

    print(f"Wrote {model_summary_path}")
    print(f"Wrote {comparison_path}")


if __name__ == "__main__":
    main()
