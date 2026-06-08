#!/usr/bin/env python3
"""Fit helium shock-bubble feature velocities from tracked positions.

The position used for velocities is `plot_y_mm_from_bottom`, matching the
feature-tracking plots and the dissertation-style x-t diagrams. The fitted
slope is in mm/code-time and is converted to m/s with the supplied conversion
factor. Pointwise finite differences are deliberately not used as final
velocities because individual frames can accelerate, decelerate, or contain
small detection jitter.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_feature_windows(items: list[str]) -> dict[str, tuple[float, float]]:
    """Parse fixed fitting windows of the form feature:start:end."""

    windows: dict[str, tuple[float, float]] = {}
    for item in items:
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(f"Expected feature:start:end window, got {item!r}")
        feature, start, end = parts
        windows[feature] = (float(start), float(end))
    return windows


def fit_window(times: np.ndarray, positions: np.ndarray) -> dict[str, float]:
    """Least-squares fit of position = a + V*time for one time window."""

    n = len(times)
    if n < 2:
        return {"slope": np.nan, "r_squared": np.nan, "rmse_mm": np.nan}
    t_mean = float(np.mean(times))
    x_mean = float(np.mean(positions))
    denom = float(np.sum((times - t_mean) ** 2))
    if denom == 0.0:
        return {"slope": np.nan, "r_squared": np.nan, "rmse_mm": np.nan}
    slope = float(np.sum((times - t_mean) * (positions - x_mean)) / denom)
    intercept = x_mean - slope * t_mean
    residuals = positions - (intercept + slope * times)
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((positions - x_mean) ** 2))
    r_squared = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    rmse_mm = math.sqrt(ss_res / n)
    return {"slope": slope, "r_squared": r_squared, "rmse_mm": rmse_mm}


def best_sliding_fit(
    group: pd.DataFrame,
    *,
    min_points: int,
    min_displacement_mm: float,
    r2_threshold: float,
) -> dict[str, object]:
    """Select the best contiguous window when no fixed window is supplied.

    The preferred window is the longest interval meeting the requested R2
    threshold; if none pass, the highest-R2 interval is returned with low
    confidence rather than silently fabricating a clean velocity.
    """

    data = group.sort_values("time_code").copy()
    data = data[np.isfinite(data["time_code"]) & np.isfinite(data["plot_y_mm_from_bottom"])]
    data = data[~data["confidence"].isin(["not_detected", "speed_rejected"])]
    if len(data) < min_points:
        return {
            "n_points": len(data),
            "confidence": "insufficient_samples" if len(data) else "not_detected",
        }

    times = data["time_code"].to_numpy(dtype=float)
    positions = data["plot_y_mm_from_bottom"].to_numpy(dtype=float)
    best_any = None
    best_unfiltered = None
    best_passing = None
    for start in range(len(data)):
        for end in range(start + min_points - 1, len(data)):
            fit = fit_window(times[start : end + 1], positions[start : end + 1])
            if not np.isfinite(fit["slope"]):
                continue
            n = end - start + 1
            displacement = float(positions[end] - positions[start])
            candidate = {
                "start": start,
                "end": end,
                "n_points": n,
                "displacement": displacement,
                **fit,
            }
            if (
                best_unfiltered is None
                or candidate["r_squared"] > best_unfiltered["r_squared"]
                or (
                    candidate["r_squared"] == best_unfiltered["r_squared"]
                    and candidate["n_points"] > best_unfiltered["n_points"]
                )
            ):
                best_unfiltered = candidate
            if abs(displacement) < min_displacement_mm:
                continue
            if (
                best_any is None
                or candidate["r_squared"] > best_any["r_squared"]
                or (
                    candidate["r_squared"] == best_any["r_squared"]
                    and candidate["n_points"] > best_any["n_points"]
                )
            ):
                best_any = candidate
            if candidate["r_squared"] >= r2_threshold:
                if (
                    best_passing is None
                    or candidate["n_points"] > best_passing["n_points"]
                    or (
                        candidate["n_points"] == best_passing["n_points"]
                        and candidate["r_squared"] > best_passing["r_squared"]
                    )
                ):
                    best_passing = candidate

    selected = best_passing or best_any or best_unfiltered
    if selected is None:
        return {"n_points": len(data), "confidence": "low"}
    selected["confidence"] = "high" if best_passing is not None else "low"
    selected["data"] = data
    return selected


def fixed_window_fit(group: pd.DataFrame, start_time: float, end_time: float) -> dict[str, object]:
    """Fit a user/report-specified time interval for a feature."""

    data = group.sort_values("time_code").copy()
    data = data[np.isfinite(data["time_code"]) & np.isfinite(data["plot_y_mm_from_bottom"])]
    data = data[~data["confidence"].isin(["not_detected", "speed_rejected"])]
    data = data[(data["time_code"] >= start_time) & (data["time_code"] <= end_time)]
    if len(data) < 2:
        return {"n_points": len(data), "confidence": "insufficient_samples"}
    times = data["time_code"].to_numpy(dtype=float)
    positions = data["plot_y_mm_from_bottom"].to_numpy(dtype=float)
    fit = fit_window(times, positions)
    return {
        "start": 0,
        "end": len(data) - 1,
        "n_points": len(data),
        "displacement": float(positions[-1] - positions[0]),
        "confidence": "fixed_window",
        "data": data,
        **fit,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--positions-csv",
        default="results/bubble_features/bubble_feature_positions_master.csv",
    )
    parser.add_argument(
        "--output-csv",
        default="results/bubble_features/bubble_feature_velocity_fits.csv",
    )
    parser.add_argument("--min-points", type=int, default=6)
    parser.add_argument("--min-displacement-mm", type=float, default=0.0)
    parser.add_argument("--r2-threshold", type=float, default=0.995)
    parser.add_argument(
        "--seconds-per-code-time",
        type=float,
        default=None,
        help="Optional physical conversion, e.g. 427e-6/123.",
    )
    parser.add_argument(
        "--velocity-conversion-mps-per-mm-code-time",
        type=float,
        default=288.0599,
        help="Direct conversion for slope in mm/code-time to m/s.",
    )
    parser.add_argument(
        "--feature-window",
        action="append",
        default=None,
        help="Fit a fixed time window for one feature, formatted feature:start_time_code:end_time_code.",
    )
    args = parser.parse_args()

    if args.feature_window is None:
        args.feature_window = []
    if not args.feature_window:
        args.feature_window = [
            "upstream_helium_interface:42.33:141.1",
            "downstream_helium_interface:42.33:141.1",
            "jet_head:28.22:98.770",
            "transmitted_shock:42.33:141.1",
        ]

    positions = pd.read_csv(args.positions_csv)
    if "feature" in positions.columns:
        positions["feature"] = positions["feature"].replace({"transverse_wave": "transmitted_shock"})
    feature_windows = parse_feature_windows(args.feature_window)
    rows = []
    for feature, group in positions.groupby("feature", sort=True):
        if feature in feature_windows:
            fit = fixed_window_fit(group, *feature_windows[feature])
        else:
            fit = best_sliding_fit(
                group,
                min_points=args.min_points,
                min_displacement_mm=args.min_displacement_mm,
                r2_threshold=args.r2_threshold,
            )
        if "data" not in fit:
            rows.append(
                {
                    "feature": feature,
                    "n_points": fit.get("n_points", 0),
                    "confidence": fit.get("confidence", "not_detected"),
                }
            )
            continue
        data = fit.pop("data")
        start = int(fit["start"])
        end = int(fit["end"])
        slope = float(fit["slope"])
        conversion = args.velocity_conversion_mps_per_mm_code_time
        if args.seconds_per_code_time:
            conversion = 1.0e-3 / args.seconds_per_code_time
        fitted_mps = abs(slope) * conversion
        rows.append(
            {
                "feature": feature,
                "start_time_code": float(data.iloc[start]["time_code"]),
                "end_time_code": float(data.iloc[end]["time_code"]),
                "start_position_mm": float(data.iloc[start]["plot_y_mm_from_bottom"]),
                "end_position_mm": float(data.iloc[end]["plot_y_mm_from_bottom"]),
                "fitted_velocity_mm_per_code_time": abs(slope),
                "fitted_velocity_m_per_s": fitted_mps,
                "r_squared": float(fit["r_squared"]),
                "rmse_mm": float(fit["rmse_mm"]),
                "n_points": int(fit["n_points"]),
                "confidence": fit["confidence"],
            }
        )

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
