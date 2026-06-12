#!/usr/bin/env python3
"""Core helium shock-bubble feature extraction logic.

This module intentionally has no plotting dependencies. It turns solver CSVs
from either rGFM/SIM or DIM into a common geometric representation, then
extracts the feature positions used by the quantitative report tables.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FEATURES = (
    "upstream_helium_interface",
    "downstream_helium_interface",
    "jet_head",
    "transmitted_shock",
)

@dataclass
class GridData:
    """Structured view of one half-domain CSV snapshot.

    The raw solver grid uses `x0` as the axial shock/bubble direction and `x1`
    as the transverse half-domain coordinate. Report plots use a bottom-left
    image convention, so later helpers convert `(raw_x0, raw_x1)` into
    `(plot_x_from_left, plot_y_from_bottom)`.
    """

    x0: np.ndarray
    x1: np.ndarray
    dx0: float
    dx1: float
    x0_min: float
    x0_max: float
    x1_half_width: float
    rho: np.ndarray
    p: np.ndarray
    phi0: np.ndarray
    interface_label: str
    mat: np.ndarray | None


def parse_time_from_name(path: str | Path) -> float | None:
    name = Path(path).name
    match = re.search(r"_t(?P<mant>[0-9]+p[0-9]+)ep(?P<exp>[+-]?[0-9]+)", name)
    if not match:
        return None
    mantissa = float(match.group("mant").replace("p", "."))
    exponent = int(match.group("exp"))
    return mantissa * (10.0**exponent)


def time_token_from_name(path: str | Path) -> str | None:
    match = re.search(r"_(t[0-9]+p[0-9]+ep[+-]?[0-9]+)", Path(path).name)
    return match.group(1) if match else None


def load_grid(csv_path: str | Path) -> GridData:
    """Load a solver CSV and build 2D arrays with a common interface field.

    GFM/SIM snapshots carry a signed level-set `phi0`; DIM/Allaire snapshots
    carry volume fractions. Internally both are converted to a zero contour so
    the feature definitions can share the same geometric code.
    """

    df = pd.read_csv(csv_path)
    required = {"x0", "x1", "rho", "p"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {sorted(missing)}")
    if "phi0" in df.columns:
        interface_column = "phi0"
        interface_label = "phi0=0"
        df["_interface_phi0"] = df["phi0"]
    elif "alpha1" in df.columns:
        interface_column = "_interface_phi0"
        interface_label = "alpha1=0.5"
        df[interface_column] = df["alpha1"] - 0.5
    else:
        raise ValueError("Missing interface column: expected either phi0 or alpha1")

    # Duplicate coordinate rows can appear when runs are concatenated or regenerated. the feature detectors expect one scalar value per cell.
    df = df.drop_duplicates(subset=["x0", "x1"])
    x0 = np.sort(df["x0"].unique())
    x1 = np.sort(df["x1"].unique())
    dx0 = float(np.median(np.diff(x0)))
    dx1 = float(np.median(np.diff(x1)))
    x0_min = float(x0[0] - 0.5 * dx0)
    x0_max = float(x0[-1] + 0.5 * dx0)
    x1_half_width = float(x1[-1] + 0.5 * dx1)

    rho = df.pivot(index="x1", columns="x0", values="rho").loc[x1, x0].to_numpy()
    p = df.pivot(index="x1", columns="x0", values="p").loc[x1, x0].to_numpy()
    phi0 = df.pivot(index="x1", columns="x0", values=interface_column).loc[x1, x0].to_numpy()

    mat = None
    if "mat" in df.columns:
        mat = df.pivot(index="x1", columns="x0", values="mat").loc[x1, x0].to_numpy()

    return GridData(
        x0=x0,
        x1=x1,
        dx0=dx0,
        dx1=dx1,
        x0_min=x0_min,
        x0_max=x0_max,
        x1_half_width=x1_half_width,
        rho=rho,
        p=p,
        phi0=phi0,
        interface_label=interface_label,
        mat=mat,
    )


def raw_to_plot(raw_x0_mm: float, raw_x1_mm: float, grid: GridData) -> tuple[float, float]:
    """Convert solver coordinates to the bottom-left report/overlay frame."""

    plot_x_mm_from_left = raw_x1_mm + grid.x1_half_width
    plot_y_mm_from_bottom = grid.x0_max - raw_x0_mm
    return float(plot_x_mm_from_left), float(plot_y_mm_from_bottom)


def contour_to_physical(contour_rc: np.ndarray, grid: GridData) -> np.ndarray:
    rows = contour_rc[:, 0]
    cols = contour_rc[:, 1]
    raw_x1_mm = grid.x1[0] + rows * grid.dx1
    raw_x0_mm = grid.x0[0] + cols * grid.dx0
    return np.column_stack([raw_x0_mm, raw_x1_mm])


def gaussian_filter(field: np.ndarray, sigma: float) -> np.ndarray:
    """Small dependency-free Gaussian smoother.

    The desktop/project venv does not always include SciPy, so this local
    separable filter keeps the tracker portable while still damping cell-scale
    noise before derivative-based feature extraction.
    """

    if sigma <= 0.0:
        return field.copy()
    radius = max(1, int(math.ceil(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= np.sum(kernel)

    if field.ndim == 1:
        padded = np.pad(field, (radius, radius), mode="edge")
        smooth = np.empty_like(field, dtype=float)
        for i in range(field.shape[0]):
            smooth[i] = np.sum(padded[i : i + 2 * radius + 1] * kernel)
        return smooth

    padded_x = np.pad(field, ((0, 0), (radius, radius)), mode="edge")
    smooth_x = np.empty_like(field, dtype=float)
    for i in range(field.shape[1]):
        smooth_x[:, i] = np.sum(padded_x[:, i : i + 2 * radius + 1] * kernel[None, :], axis=1)

    padded_y = np.pad(smooth_x, ((radius, radius), (0, 0)), mode="edge")
    smooth = np.empty_like(field, dtype=float)
    for j in range(field.shape[0]):
        smooth[j, :] = np.sum(padded_y[j : j + 2 * radius + 1, :] * kernel[:, None], axis=0)
    return smooth


def sort_contour_points(points: np.ndarray) -> np.ndarray:
    if points.shape[0] <= 2:
        return points
    centre = np.nanmean(points, axis=0)
    angles = np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0])
    return points[np.argsort(angles)]


def extract_main_phi_contour(grid: GridData) -> np.ndarray:
    """Extract the main material-interface contour using marching squares.

    This replaces `skimage.measure.find_contours` so the tracker does not need
    scikit-image. The largest connected polyline is treated as the bubble
    interface; small fragments from corrugations or numerical artifacts are not
    used as the primary contour.
    """

    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    phi = grid.phi0

    def crosses(a: float, b: float) -> bool:
        if not np.isfinite(a) or not np.isfinite(b):
            return False
        return a == 0.0 or b == 0.0 or a * b < 0.0

    def weight(a: float, b: float) -> float:
        if b == a:
            return 0.5
        return float(np.clip((0.0 - a) / (b - a), 0.0, 1.0))

    for j in range(phi.shape[0] - 1):
        for i in range(phi.shape[1] - 1):
            corners = [
                (float(phi[j, i]), float(grid.x0[i]), float(grid.x1[j])),
                (float(phi[j, i + 1]), float(grid.x0[i + 1]), float(grid.x1[j])),
                (float(phi[j + 1, i + 1]), float(grid.x0[i + 1]), float(grid.x1[j + 1])),
                (float(phi[j + 1, i]), float(grid.x0[i]), float(grid.x1[j + 1])),
            ]
            edge_points: list[tuple[float, float]] = []
            for a_idx, b_idx in ((0, 1), (1, 2), (2, 3), (3, 0)):
                va, x0a, x1a = corners[a_idx]
                vb, x0b, x1b = corners[b_idx]
                if not crosses(va, vb):
                    continue
                w = weight(va, vb)
                edge_points.append((float(x0a + w * (x0b - x0a)), float(x1a + w * (x1b - x1a))))

            if len(edge_points) == 2:
                segments.append((edge_points[0], edge_points[1]))
            elif len(edge_points) == 4:
                segments.append((edge_points[0], edge_points[1]))
                segments.append((edge_points[2], edge_points[3]))

    if not segments:
        raise RuntimeError("No phi0=0 contour found.")

    # Segment endpoints are rounded to stable integer keys so marching-squares fragments can be stitched without depending on exact floating equality.
    def key(point: tuple[float, float]) -> tuple[int, int]:
        return (int(round(point[0] * 1.0e8)), int(round(point[1] * 1.0e8)))

    points_by_key: dict[tuple[int, int], tuple[float, float]] = {}
    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for a, b in segments:
        ka = key(a)
        kb = key(b)
        points_by_key[ka] = a
        points_by_key[kb] = b
        adjacency.setdefault(ka, []).append(kb)
        adjacency.setdefault(kb, []).append(ka)

    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    polylines: list[list[tuple[float, float]]] = []

    for start in adjacency:
        for first_next in adjacency[start]:
            edge = tuple(sorted((start, first_next)))
            if edge in visited_edges:
                continue
            line = [start]
            prev = start
            current = first_next
            visited_edges.add(edge)
            while True:
                line.append(current)
                candidates = [n for n in adjacency[current] if n != prev]
                next_node = None
                for candidate in candidates:
                    candidate_edge = tuple(sorted((current, candidate)))
                    if candidate_edge not in visited_edges:
                        next_node = candidate
                        break
                if next_node is None:
                    break
                prev, current = current, next_node
                visited_edges.add(tuple(sorted((prev, current))))
                if current == start:
                    line.append(current)
                    break
            polylines.append([points_by_key[item] for item in line])

    main = max(polylines, key=len)
    return np.asarray(main, dtype=float)


def zero_crossings_1d(x: np.ndarray, values: np.ndarray) -> list[float]:
    crossings: list[float] = []
    for idx in range(len(values) - 1):
        v0 = values[idx]
        v1 = values[idx + 1]
        if not np.isfinite(v0) or not np.isfinite(v1):
            continue
        if v0 == 0.0:
            crossings.append(float(x[idx]))
        elif v0 * v1 < 0.0:
            xc = x[idx] + (0.0 - v0) * (x[idx + 1] - x[idx]) / (v1 - v0)
            crossings.append(float(xc))
    return crossings


def centreline_phi_line(grid: GridData, centreline_width_mm: float) -> np.ndarray:
    centre_band = grid.x1 <= centreline_width_mm
    if not np.any(centre_band):
        centre_band = np.zeros_like(grid.x1, dtype=bool)
        centre_band[0] = True
    return np.mean(grid.phi0[centre_band, :], axis=0)


def centreline_rho_line(grid: GridData, centreline_width_mm: float) -> np.ndarray:
    centre_band = grid.x1 <= centreline_width_mm
    if not np.any(centre_band):
        centre_band = np.zeros_like(grid.x1, dtype=bool)
        centre_band[0] = True
    return np.mean(grid.rho[centre_band, :], axis=0)


def centreline_phi_crossings(grid: GridData, centreline_width_mm: float = 1.0) -> tuple[list[float], dict[str, Any]]:
    phi_line = centreline_phi_line(grid, centreline_width_mm)
    crossings = sorted(zero_crossings_1d(grid.x0, phi_line))
    return crossings, {
        "centreline_width_mm": centreline_width_mm,
        "n_crossings": len(crossings),
        "crossings_x0_mm": ";".join(f"{c:.6g}" for c in crossings),
    }


def centreline_density_gradient_ridge(
    grid: GridData,
    x0_low: float,
    x0_high: float,
    centreline_width_mm: float = 1.0,
    margin_mm: float = 4.0,
    min_prominence_ratio: float = 1.5,
) -> tuple[float | None, str]:
    """Pre-jet downstream marker from the strongest centreline density ridge."""

    rho_smooth = gaussian_filter(grid.rho, sigma=1.0)
    drho_dx1, drho_dx0 = np.gradient(rho_smooth, grid.dx1, grid.dx0)
    grad_rho = np.hypot(drho_dx0, drho_dx1)

    band = grid.x1 <= centreline_width_mm
    if not np.any(band):
        band = np.zeros_like(grid.x1, dtype=bool)
        band[0] = True

    line = np.mean(grad_rho[band, :], axis=0)
    lo = min(x0_low, x0_high) + margin_mm
    hi = max(x0_low, x0_high) - margin_mm

    search = (grid.x0 >= lo) & (grid.x0 <= hi)
    if not np.any(search):
        return None, "empty centreline ridge search interval"

    local_x0 = grid.x0[search]
    local_values = line[search]

    if local_values.size == 0 or not np.isfinite(local_values).any():
        return None, "no finite centreline-gradient values"

    idx = int(np.nanargmax(local_values))
    selected_x0 = float(local_x0[idx])

    med = float(np.nanmedian(local_values))
    p90 = float(np.nanpercentile(local_values, 90))
    mx = float(local_values[idx])
    noise_floor = max(med, 0.25 * p90, 1e-12)
    ratio = mx / noise_floor

    if ratio < min_prominence_ratio:
        return None, f"weak centreline ridge: ratio={ratio:.3f}"

    definition = (
        "centreline |grad rho| ridge inside compressed bubble interval; "
        f"search_x0=[{lo:.3f},{hi:.3f}] mm, "
        f"max_grad={mx:.6g}, noise_floor={noise_floor:.6g}, ratio={ratio:.3f}"
    )
    return selected_x0, definition


def detect_upstream_outer_contour(grid: GridData, min_side_x1_mm: float = 2.0) -> tuple[float, float, str]:
    """Post-jet upstream interface from the lower outer material contour.

    Once the jet forms, the centreline is contaminated by the air intrusion and
    is no longer a valid upstream-interface marker. This detector therefore
    excludes the centreline first and falls back gradually only if too few
    contour points remain.
    """

    contour = extract_main_phi_contour(grid)
    x0 = contour[:, 0]
    x1 = contour[:, 1]

    for min_side in [min_side_x1_mm, 6.0, 4.0, 2.0, 0.0]:
        mask = x1 >= min_side
        if np.count_nonzero(mask) >= 5:
            sub = contour[mask]
            idx = int(np.argmax(sub[:, 0]))
            point = sub[idx]
            return float(point[0]), float(point[1]), (
                f"post-jet upstream interface from lower outer {grid.interface_label} contour, "
                f"excluding centreline region x1 < {min_side:.3f} mm"
            )

    raise RuntimeError("Could not detect post-jet upstream outer contour")


def detect_downstream_upper_cap(
    grid: GridData,
    upstream_x0: float,
    cap_half_width_mm: float = 12.0,
    min_vertical_gap_mm: float = 2.0,
    top_k: int = 21,
) -> tuple[float | None, float | None, str]:
    """Post-jet downstream interface from the near-centreline upper helium cap."""

    contour = extract_main_phi_contour(grid)
    x0 = contour[:, 0]
    x1 = contour[:, 1]

    width_candidates = [cap_half_width_mm, 16.0, 20.0, 25.0]
    mask = None
    used_width = None

    for width in width_candidates:
        trial = (np.abs(x1) <= width) & (x0 < upstream_x0 - min_vertical_gap_mm)
        if np.count_nonzero(trial) >= 5:
            mask = trial
            used_width = width
            break

    if mask is None:
        return None, None, "could not identify upper compressed helium cap"

    xc = x0[mask]
    yc = x1[mask]

    order = np.argsort(xc)
    k = min(top_k, len(order))
    top_idx = order[:k]

    downstream_x0 = float(np.median(xc[top_idx]))
    x1_pick = float(yc[top_idx][int(np.argmin(np.abs(yc[top_idx])))])

    definition = (
        f"upper compressed helium cap from main {grid.interface_label} contour in a near-centreline band; "
        f"cap_half_width={used_width:.3f} mm, candidate_count={np.count_nonzero(mask)}, top_k={k}"
    )
    return downstream_x0, x1_pick, definition


def detect_jet_from_centreline_crossing_cluster(
    crossings: list[float],
    cluster_gap_mm: float = 8.0,
) -> tuple[float | None, str]:
    """Detect the jet head as the leading point of the lower centreline cluster."""

    if len(crossings) < 2:
        return None, "not enough centreline crossings for jet-head cluster detector"

    crossings = sorted(float(c) for c in crossings)
    clusters = [[crossings[0]]]
    for value in crossings[1:]:
        if value - clusters[-1][-1] > cluster_gap_mm:
            clusters.append([value])
        else:
            clusters[-1].append(value)

    lower_cluster = max(clusters, key=lambda c: np.mean(c))
    jet_x0 = float(min(lower_cluster))

    definition = (
        f"post-jet centreline interface crossing cluster; cluster_gap={cluster_gap_mm:.3f} mm, "
        f"clusters={';'.join('[' + ','.join(f'{v:.3f}' for v in c) + ']' for c in clusters)}, "
        f"selected_cluster=[{','.join(f'{v:.3f}' for v in lower_cluster)}]"
    )
    return jet_x0, definition




def _empty_candidate_table(extra_columns: list[str] | None = None) -> pd.DataFrame:
    """Return a stable candidate table schema for optional diagnostics."""

    columns = [
        "time_code",
        "raw_x0_mm",
        "raw_x1_mm",
        "plot_x_mm_from_left",
        "plot_y_mm_from_bottom",
        "image_x_px_from_left",
        "image_y_px_from_bottom",
        "row_score",
        "normalized_score",
        "candidate_rank",
        "source",
        "definition",
    ]
    if extra_columns:
        columns.extend(extra_columns)
    return pd.DataFrame(columns=columns)


def _infer_method_hint(candidate_df: pd.DataFrame, method_hint: str | None = None) -> str:
    """Infer whether method-specific branch priors should use GFM or DIM values."""

    if method_hint:
        return method_hint.upper()
    joined = " ".join(str(value).lower() for value in candidate_df.get("csv_file", pd.Series(dtype=str)).tolist())
    joined += " " + " ".join(str(value).lower() for value in candidate_df.get("frame_id", pd.Series(dtype=str)).tolist())
    if "dim" in joined or "allaire" in joined or "five" in joined:
        return "DIM"
    return "GFM"


def _fit_branch_stats(branch: pd.DataFrame) -> pd.DataFrame:
    """Attach linear-fit diagnostics to a selected temporal branch."""

    branch = branch.sort_values("time_code").reset_index(drop=True)
    branch["branch_score"] = np.nan
    branch["selected"] = True
    if len(branch) >= 2:
        t = branch["time_code"].to_numpy(dtype=float)
        y = branch["plot_y_mm_from_bottom"].to_numpy(dtype=float)
        fit = np.polyfit(t, y, deg=1)
        y_hat = fit[0] * t + fit[1]
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        branch["branch_fit_slope_mm_per_code_time"] = float(fit[0])
        branch["branch_fit_r_squared"] = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
        branch["branch_fit_rmse_mm"] = math.sqrt(ss_res / len(branch))
    else:
        branch["branch_fit_slope_mm_per_code_time"] = np.nan
        branch["branch_fit_r_squared"] = np.nan
        branch["branch_fit_rmse_mm"] = np.nan
    return branch


def _dynamic_branch_select(
    candidate_df: pd.DataFrame,
    *,
    feature_name: str,
    min_time_code: float | None,
    slope_bounds: tuple[float, float],
    max_negative_step_mm: float = 1.5,
    local_score_weight: float = 1.0,
    slope_penalty_weight: float = 2.0,
    jump_penalty_weight: float = 0.25,
) -> pd.DataFrame:
    """Choose one candidate per frame with dynamic programming.

    Per-frame maxima are unreliable because several strong ridges can coexist:
    the actual shock branch, interface-adjacent ridges, reflected waves, and
    the outer bow shock. This selector scores entire trajectories instead of
    isolated frames. It rewards local candidate strength, but penalizes
    unphysical backwards motion, slope outside the expected velocity band, and
    sudden frame-to-frame jumps.
    """

    if candidate_df.empty:
        return pd.DataFrame(columns=list(candidate_df.columns) + ["branch_score", "selected"])

    candidates = candidate_df.copy()
    candidates = candidates[np.isfinite(candidates["time_code"]) & np.isfinite(candidates["plot_y_mm_from_bottom"])].copy()
    if min_time_code is not None:
        candidates = candidates[candidates["time_code"] >= min_time_code].copy()
    if candidates.empty:
        candidates["branch_score"] = np.nan
        candidates["selected"] = False
        return candidates

    if "normalized_score" not in candidates.columns:
        if "row_score" in candidates.columns and np.isfinite(candidates["row_score"]).any():
            max_score = max(float(np.nanmax(candidates["row_score"])), 1.0e-12)
            candidates["normalized_score"] = candidates["row_score"] / max_score
        else:
            candidates["normalized_score"] = 1.0

    by_time = [
        group.reset_index(drop=False).rename(columns={"index": "original_index"})
        for _time, group in candidates.sort_values(["time_code", "candidate_rank"]).groupby("time_code", sort=True)
    ]
    if not by_time:
        return candidates

    slope_lo, slope_hi = slope_bounds
    slope_mid = 0.5 * (slope_lo + slope_hi)
    slope_span = max(0.5 * (slope_hi - slope_lo), 1.0e-12)

    scores: list[np.ndarray] = []
    parents: list[np.ndarray] = []

    for ti, group in enumerate(by_time):
        local = local_score_weight * group["normalized_score"].to_numpy(dtype=float)
        if "candidate_rank" in group.columns:
            local -= 0.04 * (group["candidate_rank"].to_numpy(dtype=float) - 1.0)

        if ti == 0:
            scores.append(local)
            parents.append(np.full(len(group), -1, dtype=int))
            continue

        prev_group = by_time[ti - 1]
        prev_scores = scores[-1]
        current_scores = np.full(len(group), -np.inf, dtype=float)
        current_parents = np.full(len(group), -1, dtype=int)
        dt = float(group["time_code"].iloc[0] - prev_group["time_code"].iloc[0])
        if dt <= 0.0:
            dt = 1.0e-12

        for j, row in group.iterrows():
            y = float(row["plot_y_mm_from_bottom"])
            best_score = -np.inf
            best_parent = -1
            for k, prev_row in prev_group.iterrows():
                prev_y = float(prev_row["plot_y_mm_from_bottom"])
                dy = y - prev_y
                slope = dy / dt

                if dy < -max_negative_step_mm:
                    monotone_penalty = 10.0 + abs(dy)
                else:
                    monotone_penalty = 0.0

                if slope < slope_lo:
                    slope_penalty = (slope_lo - slope) / slope_span
                elif slope > slope_hi:
                    slope_penalty = (slope - slope_hi) / slope_span
                else:
                    slope_penalty = 0.10 * abs(slope - slope_mid) / slope_span

                expected_dy = slope_mid * dt
                jump_penalty = abs(dy - expected_dy) / max(abs(expected_dy), 6.0)
                candidate_score = (
                    prev_scores[k]
                    + local[j]
                    - slope_penalty_weight * slope_penalty
                    - jump_penalty_weight * jump_penalty
                    - monotone_penalty
                )
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_parent = k

            current_scores[j] = best_score
            current_parents[j] = best_parent

        scores.append(current_scores)
        parents.append(current_parents)

    last = int(np.nanargmax(scores[-1]))
    selected: list[tuple[int, int]] = []
    for ti in range(len(by_time) - 1, -1, -1):
        selected.append((ti, last))
        last = int(parents[ti][last])
        if last < 0 and ti > 0:
            break
    selected.reverse()

    selected_indices = [int(by_time[ti].iloc[local_idx]["original_index"]) for ti, local_idx in selected]
    branch = candidates.loc[selected_indices].copy()
    branch["feature"] = feature_name
    return _fit_branch_stats(branch)


def transmitted_shock(
    grid: GridData,
    time_code: float | None,
    downstream_y_mm: float | None = None,
    jet_y_mm: float | None = None,
    wave_min_time_code: float = 42.33,
    centreline_width_mm: float = 4.0,
    min_gap_ahead_mm: float = 8.0,
) -> tuple[float | None, float | None, str]:
    """Single-frame transmitted-shock estimate kept for per-frame diagnostics.

    The batch pipeline ultimately replaces these rows with the temporally
    selected branch. Keeping the single-frame estimate is still useful for
    quick frame-level inspection and for bootstrap diagnostics.
    """

    candidates = extract_transmitted_shock_candidates(
        grid,
        time_code=time_code,
        bounds=None,
        downstream_y_mm=downstream_y_mm,
        jet_y_mm=jet_y_mm,
        wave_min_time_code=wave_min_time_code,
        centreline_width_mm=centreline_width_mm,
        min_gap_ahead_mm=min_gap_ahead_mm,
        max_candidates=1,
    )
    if candidates.empty:
        return None, None, f"transmitted shock not detected; time_code={time_code}"
    row = candidates.iloc[0]
    return float(row["raw_x0_mm"]), float(row["raw_x1_mm"]), str(row["definition"])


def extract_transmitted_shock_candidates(
    grid: GridData,
    *,
    time_code: float | None,
    bounds: tuple[int, int, int, int] | None = None,
    downstream_y_mm: float | None = None,
    jet_y_mm: float | None = None,
    wave_min_time_code: float = 42.33,
    centreline_width_mm: float = 4.0,
    min_gap_ahead_mm: float = 8.0,
    max_candidates: int = 10,
    min_prominence_ratio: float = 1.05,
) -> pd.DataFrame:
    """Extract plausible transmitted-shock candidates from one snapshot.

    The signal is the near-centreline axial shock strength
    `|dp/dx0| + 0.5|drho/dx0|`. The search starts ahead of the downstream
    helium cap/jet head so the detector does not snap back to the bubble
    interface. Multiple local maxima are retained because the final decision is
    temporal, not a single-frame maximum.
    """

    extra = [
        "downstream_y_mm",
        "jet_y_mm",
        "lower_bound_y_mm",
        "prominence_ratio",
    ]
    columns = _empty_candidate_table(extra).columns
    if time_code is None or not np.isfinite(time_code) or time_code < wave_min_time_code:
        return pd.DataFrame(columns=columns)

    p_smooth = gaussian_filter(grid.p, sigma=1.0)
    rho_smooth = gaussian_filter(grid.rho, sigma=1.0)
    dp_dx1, dp_dx0 = np.gradient(p_smooth, grid.dx1, grid.dx0)
    drho_dx1, drho_dx0 = np.gradient(rho_smooth, grid.dx1, grid.dx0)

    shock_signal = np.abs(dp_dx0) + 0.50 * np.abs(drho_dx0)
    centre_mask = grid.x1 <= centreline_width_mm
    if not np.any(centre_mask):
        centre_mask = np.zeros_like(grid.x1, dtype=bool)
        centre_mask[0] = True

    line = np.nanpercentile(shock_signal[centre_mask, :], 95.0, axis=0)
    line = gaussian_filter(line, sigma=1.0)
    plot_y_rows = grid.x0_max - grid.x0
    full_height = grid.x0_max - grid.x0_min

    finite_downstream = downstream_y_mm is not None and np.isfinite(downstream_y_mm)
    finite_jet = jet_y_mm is not None and np.isfinite(jet_y_mm)
    lower_bound = 0.0
    if finite_downstream:
        lower_bound = max(lower_bound, float(downstream_y_mm) + min_gap_ahead_mm)
    if finite_jet:
        lower_bound = max(lower_bound, float(jet_y_mm) + min_gap_ahead_mm)
    if lower_bound <= 0.0:
        lower_bound = 0.40 * full_height

    search = (plot_y_rows >= lower_bound) & (plot_y_rows <= full_height - 5.0)
    if not np.any(search):
        return pd.DataFrame(columns=columns)

    valid_values = line[search]
    if valid_values.size == 0 or not np.isfinite(valid_values).any():
        return pd.DataFrame(columns=columns)

    med = float(np.nanmedian(valid_values))
    p90 = float(np.nanpercentile(valid_values, 90.0))
    row_max = float(np.nanmax(valid_values))
    floor = max(med, 0.25 * p90, 1.0e-12)
    ratio = row_max / floor
    if ratio < min_prominence_ratio:
        return pd.DataFrame(columns=columns)

    local_maxima: list[int] = []
    for i in np.where(search)[0]:
        left = line[i - 1] if i > 0 else -np.inf
        right = line[i + 1] if i + 1 < len(line) else -np.inf
        if line[i] >= left and line[i] >= right:
            local_maxima.append(int(i))
    if not local_maxima:
        local_maxima = [int(np.nanargmax(np.where(search, line, -np.inf)))]

    rows: list[dict[str, Any]] = []
    for i in local_maxima:
        raw_x0 = float(grid.x0[i])
        raw_x1 = 0.0
        plot_x, plot_y = raw_to_plot(raw_x0, raw_x1, grid)
        if bounds is None:
            image_x = np.nan
            image_y_bottom = np.nan
        else:
            image_x, _image_y_top, image_y_bottom = physical_to_image_pixel(raw_x0, raw_x1, grid, bounds)

        distance_from_lower = max(plot_y - lower_bound, 0.0)
        score = float(line[i]) / max(row_max, 1.0e-12)
        # Prefer the first strong ridge ahead of the bubble over the far outer bow shock.
        score -= 0.08 * distance_from_lower / max(25.0, min_gap_ahead_mm)

        rows.append({
            "time_code": float(time_code),
            "raw_x0_mm": raw_x0,
            "raw_x1_mm": raw_x1,
            "plot_x_mm_from_left": plot_x,
            "plot_y_mm_from_bottom": plot_y,
            "image_x_px_from_left": image_x,
            "image_y_px_from_bottom": image_y_bottom,
            "row_score": float(line[i]),
            "normalized_score": score,
            "candidate_rank": 0,
            "source": "near-centreline pressure-density shock ridge",
            "definition": (
                "transmitted shock candidate from near-centreline |dp/dx0| + 0.5|drho/dx0| ridge; "
                f"centreline_width={centreline_width_mm:.3f} mm, lower_bound_y={lower_bound:.3f} mm"
            ),
            "downstream_y_mm": np.nan if downstream_y_mm is None else float(downstream_y_mm),
            "jet_y_mm": np.nan if jet_y_mm is None else float(jet_y_mm),
            "lower_bound_y_mm": float(lower_bound),
            "prominence_ratio": float(ratio),
        })

    rows = sorted(rows, key=lambda r: r["normalized_score"], reverse=True)
    for rank, row in enumerate(rows[:max_candidates], start=1):
        row["candidate_rank"] = rank
    return pd.DataFrame(rows[:max_candidates], columns=columns)


def transmitted_shock_slope_bounds(method_hint: str) -> tuple[float, float]:
    hint = method_hint.upper()
    if "DIM" in hint or "ALLAIRE" in hint:
        return 1.35, 1.75
    return 1.10, 1.45


def select_transmitted_shock_branch(
    candidate_df: pd.DataFrame,
    *,
    wave_min_time_code: float = 42.33,
    method_hint: str | None = None,
) -> pd.DataFrame:
    """Select the physically continuous transmitted-shock trajectory."""

    method = _infer_method_hint(candidate_df, method_hint)
    return _dynamic_branch_select(
        candidate_df,
        feature_name="transmitted_shock",
        min_time_code=wave_min_time_code,
        slope_bounds=transmitted_shock_slope_bounds(method),
        max_negative_step_mm=2.0,
        local_score_weight=0.85,
        slope_penalty_weight=3.25,
        jump_penalty_weight=0.65,
    )


def extract_upstream_interface_candidates(
    grid: GridData,
    *,
    time_code: float | None,
    bounds: tuple[int, int, int, int] | None = None,
    jet_y_mm: float | None = None,
    min_side_x1_mm: float = 2.0,
    max_side_fraction: float = 0.70,
    max_candidates: int = 12,
) -> pd.DataFrame:
    """Extract plausible upstream-interface candidates from one snapshot.

    This is used after the first piecewise pass to stabilize the upstream
    marker over time. Candidate scores prefer the rear/lower outer interface,
    avoid the centreline jet cluster, and avoid the extreme side wall.
    """

    extra = [
        "lateral_fraction",
        "jet_y_mm",
        "score_component_lateral",
        "score_component_jet_gap",
    ]
    columns = _empty_candidate_table(extra).columns

    finite_time = time_code is not None and np.isfinite(time_code)
    finite_jet = jet_y_mm is not None and np.isfinite(jet_y_mm)
    if not finite_jet and finite_time and float(time_code) < 28.22:
        crossings, _diagnostics = centreline_phi_crossings(grid, centreline_width_mm=1.0)
        if len(crossings) >= 2:
            raw0 = float(max(crossings))
            raw1 = 0.0
            plot_x, py = raw_to_plot(raw0, raw1, grid)
            if bounds is None:
                image_x = np.nan
                image_y_bottom = np.nan
            else:
                image_x, _image_y_top, image_y_bottom = physical_to_image_pixel(raw0, raw1, grid, bounds)
            return pd.DataFrame([{
                "time_code": float(time_code),
                "raw_x0_mm": raw0,
                "raw_x1_mm": raw1,
                "plot_x_mm_from_left": plot_x,
                "plot_y_mm_from_bottom": py,
                "image_x_px_from_left": image_x,
                "image_y_px_from_bottom": image_y_bottom,
                "row_score": 1.0,
                "normalized_score": 1.0,
                "candidate_rank": 1,
                "source": f"centreline {grid.interface_label} crossing",
                "definition": "pre-jet lower centreline interface crossing treated as upstream interface",
                "lateral_fraction": 0.0,
                "jet_y_mm": np.nan,
                "score_component_lateral": 0.0,
                "score_component_jet_gap": 0.0,
            }], columns=columns)

    contour = extract_main_phi_contour(grid)
    raw_x0 = contour[:, 0]
    raw_x1_abs = np.abs(contour[:, 1])
    plot_y = grid.x0_max - raw_x0

    mask = raw_x1_abs >= min_side_x1_mm
    mask &= raw_x1_abs <= max_side_fraction * grid.x1_half_width

    finite_jet = jet_y_mm is not None and np.isfinite(jet_y_mm)
    if finite_jet:
        # Keep the upstream/rear interface behind the jet head and avoid the centreline jet cluster.
        mask &= plot_y <= float(jet_y_mm) - 1.0

    if np.count_nonzero(mask) < 3:
        mask = raw_x1_abs >= min_side_x1_mm

    idxs = np.where(mask)[0]
    if idxs.size == 0:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for idx in idxs:
        raw0 = float(raw_x0[idx])
        raw1 = float(abs(contour[idx, 1]))
        plot_x, py = raw_to_plot(raw0, raw1, grid)
        if bounds is None:
            image_x = np.nan
            image_y_bottom = np.nan
        else:
            image_x, _image_y_top, image_y_bottom = physical_to_image_pixel(raw0, raw1, grid, bounds)

        lateral_fraction = raw1 / max(grid.x1_half_width, 1.0e-12)
        # The reference upstream marker is not the centreline jet and not the extreme side wall.
        lateral_score = -abs(lateral_fraction - 0.42)
        jet_gap_score = 0.0
        if finite_jet:
            jet_gap_score = -0.04 * abs((float(jet_y_mm) - py) - 10.0)
        score = lateral_score + jet_gap_score

        rows.append({
            "time_code": np.nan if time_code is None else float(time_code),
            "raw_x0_mm": raw0,
            "raw_x1_mm": raw1,
            "plot_x_mm_from_left": plot_x,
            "plot_y_mm_from_bottom": py,
            "image_x_px_from_left": image_x,
            "image_y_px_from_bottom": image_y_bottom,
            "row_score": float(score),
            "normalized_score": float(score),
            "candidate_rank": 0,
            "source": f"{grid.interface_label} upstream contour candidate",
            "definition": (
                "candidate on rear/upstream helium interface from main material contour; "
                "centreline jet cluster excluded by lateral and jet-gap constraints"
            ),
            "lateral_fraction": float(lateral_fraction),
            "jet_y_mm": np.nan if jet_y_mm is None else float(jet_y_mm),
            "score_component_lateral": float(lateral_score),
            "score_component_jet_gap": float(jet_gap_score),
        })

    if not rows:
        return pd.DataFrame(columns=columns)

    raw_scores = np.asarray([row["row_score"] for row in rows], dtype=float)
    score_min = float(np.nanmin(raw_scores))
    score_max = float(np.nanmax(raw_scores))
    for row in rows:
        row["normalized_score"] = (row["row_score"] - score_min) / max(score_max - score_min, 1.0e-12)

    rows = sorted(rows, key=lambda r: r["normalized_score"], reverse=True)
    for rank, row in enumerate(rows[:max_candidates], start=1):
        row["candidate_rank"] = rank
    return pd.DataFrame(rows[:max_candidates], columns=columns)


def select_upstream_interface_branch(
    candidate_df: pd.DataFrame,
    *,
    min_time_code: float = 21.165,
    slope_bounds: tuple[float, float] = (0.55, 0.85),
) -> pd.DataFrame:
    """Select a smooth upstream-interface branch from contour candidates."""

    return _dynamic_branch_select(
        candidate_df,
        feature_name="upstream_helium_interface",
        min_time_code=min_time_code,
        slope_bounds=slope_bounds,
        max_negative_step_mm=2.0,
        local_score_weight=0.90,
        slope_penalty_weight=2.75,
        jump_penalty_weight=0.55,
    )


# Backwards-compatible names for older batch scripts.
def expected_transverse_y(time_code: float | None) -> float | None:
    """Legacy expected-y helper retained for older notebooks/scripts."""

    if time_code is None or not np.isfinite(time_code):
        return None
    return float(np.clip(93.9 + 1.599 * float(time_code), 150.0, 322.0))


def transverse_wave(*args: Any, **kwargs: Any) -> tuple[float | None, float | None, str]:
    return transmitted_shock(*args, **kwargs)


def extract_transverse_wave_candidates(*args: Any, **kwargs: Any) -> pd.DataFrame:
    return extract_transmitted_shock_candidates(*args, **kwargs)


def select_transverse_wave_branch(*args: Any, **kwargs: Any) -> pd.DataFrame:
    return select_transmitted_shock_branch(*args, **kwargs)
def feature_row(
    *,
    time_code: float | None,
    feature: str,
    raw_x0_mm: float | None,
    raw_x1_mm: float | None,
    grid: GridData,
    bounds: tuple[int, int, int, int] | None,
    source: str,
    definition: str,
    confidence: str,
) -> dict[str, Any]:
    """Build one output row using both raw and bottom-left plot coordinates."""

    finite = raw_x0_mm is not None and raw_x1_mm is not None and np.isfinite(raw_x0_mm) and np.isfinite(raw_x1_mm)
    if finite:
        plot_x, plot_y = raw_to_plot(float(raw_x0_mm), float(raw_x1_mm), grid)
        if bounds is not None:
            image_x, _image_y_top, image_y_bottom = physical_to_image_pixel(float(raw_x0_mm), float(raw_x1_mm), grid, bounds)
        else:
            image_x = np.nan
            image_y_bottom = np.nan
    else:
        raw_x0_mm = np.nan
        raw_x1_mm = np.nan
        plot_x = np.nan
        plot_y = np.nan
        image_x = np.nan
        image_y_bottom = np.nan
    return {
        "time_code": time_code,
        "feature": feature,
        "plot_x_mm_from_left": plot_x,
        "plot_y_mm_from_bottom": plot_y,
        "raw_x0_mm": raw_x0_mm,
        "raw_x1_mm": raw_x1_mm,
        "image_x_px_from_left": image_x,
        "image_y_px_from_bottom": image_y_bottom,
        "source": source,
        "definition": definition,
        "confidence": confidence,
    }


def detect_features(
    grid: GridData,
    *,
    time_code: float | None,
    bounds: tuple[int, int, int, int] | None = None,
    jet_min_time_code: float = 28.22,
    wave_min_time_code: float = 42.33,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Run the piecewise per-frame feature detectors.

    This function handles the interface/jet definitions. In batch mode,
    transmitted-shock and upstream-interface rows can be overwritten later by
    temporal branch selection using the candidate tables.
    """

    main_contour = extract_main_phi_contour(grid)
    rows: list[dict[str, Any]] = []
    finite_time = time_code is not None and np.isfinite(time_code)
    jet_enabled = bool(finite_time and time_code >= jet_min_time_code)

    crossings, diagnostics = centreline_phi_crossings(grid, centreline_width_mm=1.0)
    if len(crossings) < 2:
        raise RuntimeError(f"Need at least 2 centreline interface crossings, found {len(crossings)}")

    upper_crossing_x0 = float(min(crossings))
    lower_crossing_x0 = float(max(crossings))
    downstream_x1 = 0.0
    jet_x0 = None

    if jet_enabled:
        upstream_x0, upstream_x1, upstream_def = detect_upstream_outer_contour(
            grid,
            min_side_x1_mm=2.0,
        )
        rows.append(feature_row(
            time_code=time_code,
            feature="upstream_helium_interface",
            raw_x0_mm=upstream_x0,
            raw_x1_mm=upstream_x1,
            grid=grid,
            bounds=bounds,
            source="outer phi0 contour" if grid.interface_label == "phi0=0" else "outer alpha1 contour",
            definition=upstream_def,
            confidence="medium_to_high",
        ))
    else:
        upstream_x0 = lower_crossing_x0
        rows.append(feature_row(
            time_code=time_code,
            feature="upstream_helium_interface",
            raw_x0_mm=upstream_x0,
            raw_x1_mm=0.0,
            grid=grid,
            bounds=bounds,
            source=f"centreline {grid.interface_label} crossing",
            definition=f"pre-jet lower centreline {grid.interface_label} crossing treated as upstream interface",
            confidence="high",
        ))

    if not jet_enabled:
        downstream_x0, downstream_def = centreline_density_gradient_ridge(
            grid,
            upper_crossing_x0,
            lower_crossing_x0,
            centreline_width_mm=1.0,
            margin_mm=4.0,
            min_prominence_ratio=1.5,
        )
        rows.append(feature_row(
            time_code=time_code,
            feature="downstream_helium_interface",
            raw_x0_mm=downstream_x0,
            raw_x1_mm=0.0 if downstream_x0 is not None else None,
            grid=grid,
            bounds=bounds,
            source="centreline |grad rho| ridge",
            definition=(
                "pre-jet downstream interface detected using old jet-position logic; " + downstream_def
                if downstream_x0 is not None
                else downstream_def
            ),
            confidence="medium_to_high" if downstream_x0 is not None else "not_detected",
        ))
        rows.append(feature_row(
            time_code=time_code,
            feature="jet_head",
            raw_x0_mm=None,
            raw_x1_mm=None,
            grid=grid,
            bounds=bounds,
            source=f"centreline {grid.interface_label} crossing cluster",
            definition=f"time gated: jet_min_time_code={jet_min_time_code:g}",
            confidence="not_detected",
        ))
    else:
        downstream_x0, downstream_x1, downstream_def = detect_downstream_upper_cap(
            grid,
            upstream_x0,
            cap_half_width_mm=12.0,
            min_vertical_gap_mm=2.0,
            top_k=21,
        )
        rows.append(feature_row(
            time_code=time_code,
            feature="downstream_helium_interface",
            raw_x0_mm=downstream_x0,
            raw_x1_mm=downstream_x1,
            grid=grid,
            bounds=bounds,
            source="upper compressed helium cap detector",
            definition=downstream_def,
            confidence="medium_to_high" if downstream_x0 is not None else "not_detected",
        ))

        jet_x0, jet_def = detect_jet_from_centreline_crossing_cluster(
            crossings,
            cluster_gap_mm=8.0,
        )
        rows.append(feature_row(
            time_code=time_code,
            feature="jet_head",
            raw_x0_mm=jet_x0,
            raw_x1_mm=0.0 if jet_x0 is not None else None,
            grid=grid,
            bounds=bounds,
            source=f"centreline {grid.interface_label} crossing cluster",
            definition=jet_def,
            confidence="medium_to_high" if jet_x0 is not None else "not_detected",
        ))

    downstream_plot_y = np.nan
    if downstream_x0 is not None and np.isfinite(downstream_x0):
        _px, downstream_plot_y = raw_to_plot(float(downstream_x0), 0.0 if downstream_x1 is None else float(downstream_x1), grid)

    jet_plot_y = np.nan
    if "jet_x0" in locals() and jet_x0 is not None and np.isfinite(jet_x0):
        _px, jet_plot_y = raw_to_plot(float(jet_x0), 0.0, grid)

    shock_x0, shock_x1, shock_def = transmitted_shock(
        grid,
        time_code,
        downstream_y_mm=downstream_plot_y,
        jet_y_mm=jet_plot_y,
        wave_min_time_code=wave_min_time_code,
    )
    rows.append(feature_row(
        time_code=time_code,
        feature="transmitted_shock",
        raw_x0_mm=shock_x0,
        raw_x1_mm=shock_x1,
        grid=grid,
        bounds=bounds,
        source="near-centreline |grad p| + |grad rho| ridge",
        definition=shock_def,
        confidence="low_to_medium" if shock_x0 is not None else "not_detected",
    ))

    feature_df = pd.DataFrame(rows)
    for key, value in diagnostics.items():
        feature_df[key] = value
    return feature_df, main_contour
