#!/usr/bin/env python3
"""Plotting and image-coordinate helpers for bubble feature extraction.

The core tracker works entirely in solver coordinates. This module owns the
messier presentation conversions: reflected half-domain displays, PNG pixel
overlays, and diagnostic plots used to audit feature picks.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from bubble_feature_core import GridData, gaussian_filter, raw_to_plot

def image_domain_bounds(image_path: str | Path) -> tuple[int, int, int, int]:
    """Estimate the plotted-domain bounds inside a generated Schlieren PNG."""

    image = Image.open(image_path).convert("L")
    arr = np.asarray(image)
    # The generated Schlieren images have dark axes/borders. Counting dark rows/columns gives a robust enough crop box for overlaying feature points.
    dark = arr < 50
    row_counts = dark.sum(axis=1)
    candidate_rows = np.where(row_counts > 0.70 * arr.shape[1])[0]
    if len(candidate_rows) >= 2:
        top = int(candidate_rows[0])
        bottom = int(candidate_rows[-1])
    else:
        top = 0
        bottom = arr.shape[0] - 1

    col_counts = dark[top : bottom + 1, :].sum(axis=0)
    right_candidates = np.where(col_counts > 0.70 * (bottom - top + 1))[0]
    right = int(right_candidates[-1]) if len(right_candidates) else arr.shape[1] - 1
    left_candidates = np.where(col_counts > 0.30 * (bottom - top + 1))[0]
    left = int(left_candidates[0]) if len(left_candidates) else 0
    return left, right, top, bottom


def physical_to_image_pixel(
    raw_x0_mm: float,
    raw_x1_mm: float,
    grid: GridData,
    bounds: tuple[int, int, int, int],
) -> tuple[float, float, float]:
    """Map physical solver coordinates onto PNG pixels for visual overlays."""

    left, right, top, bottom = bounds
    plot_x_mm_from_left, plot_y_mm_from_bottom = raw_to_plot(raw_x0_mm, raw_x1_mm, grid)
    full_width_mm = 2.0 * grid.x1_half_width
    full_height_mm = grid.x0_max - grid.x0_min
    col = left + plot_x_mm_from_left / full_width_mm * (right - left)
    row_from_bottom = plot_y_mm_from_bottom / full_height_mm * (bottom - top)
    row_from_top = bottom - row_from_bottom
    return float(col), float(row_from_top), float(row_from_bottom)
def plot_overlay(
    image_path: str | Path,
    grid: GridData,
    feature_df: pd.DataFrame,
    main_contour: np.ndarray,
    output_path: str | Path,
) -> None:
    """Overlay detected feature points on the original Schlieren PNG."""

    image = Image.open(image_path).convert("RGBA")
    bounds = image_domain_bounds(image_path)
    fig, ax = plt.subplots(figsize=(6, 14))
    ax.imshow(np.asarray(image))
    ax.axis("off")

    colours = {
        "downstream_helium_interface": "lime",
        "upstream_helium_interface": "magenta",
        "jet_head": "cyan",
        "transmitted_shock": "red",
    }
    labels = {
        "downstream_helium_interface": "Downstream He",
        "upstream_helium_interface": "Upstream He",
        "jet_head": "Jet head",
        "transmitted_shock": "Transmitted shock",
    }

    for sign in (1.0, -1.0):
        cols: list[float] = []
        rows: list[float] = []
        for raw_x0, raw_x1 in main_contour:
            col, row_top, _row_bottom = physical_to_image_pixel(float(raw_x0), sign * float(raw_x1), grid, bounds)
            cols.append(col)
            rows.append(row_top)
        ax.plot(cols, rows, color="yellow", linewidth=0.8, alpha=0.85)

    for _, row in feature_df.iterrows():
        raw_x0 = row["raw_x0_mm"]
        raw_x1 = row["raw_x1_mm"]
        if not np.isfinite(raw_x0) or not np.isfinite(raw_x1):
            continue
        feature = row["feature"]
        colour = colours.get(feature, "white")
        signs = [1.0] if abs(float(raw_x1)) < 1.0e-10 else [1.0, -1.0]
        for sign in signs:
            col, row_top, _row_bottom = physical_to_image_pixel(float(raw_x0), sign * float(raw_x1), grid, bounds)
            ax.scatter([col], [row_top], s=95, color=colour, edgecolor="black", linewidth=1.0, zorder=10)
        col, row_top, _row_bottom = physical_to_image_pixel(float(raw_x0), float(raw_x1), grid, bounds)
        ax.text(
            col + 6,
            row_top - 6,
            labels.get(feature, feature),
            color=colour,
            fontsize=9,
            weight="bold",
            bbox={"facecolor": "black", "alpha": 0.35, "edgecolor": "none", "pad": 1.0},
        )

    fig.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def make_mirrored_field(field: np.ndarray) -> np.ndarray:
    """Reflect the stored top-half bubble field into the full display domain."""

    return np.vstack([np.flipud(field), field])


def plot_diagnostic(
    grid: GridData,
    feature_df: pd.DataFrame,
    main_contour: np.ndarray,
    output_path: str | Path,
) -> None:
    """Save a CSV-derived diagnostic plot in physical bottom-left coordinates."""

    drho_dx1, drho_dx0 = np.gradient(gaussian_filter(grid.rho, sigma=0.8), grid.dx1, grid.dx0)
    grad = np.hypot(drho_dx0, drho_dx1)
    scale = np.percentile(grad, 99)
    if scale <= 0.0:
        scale = 1.0
    schlieren_like = np.exp(-8.0 * grad / scale)
    full_field = make_mirrored_field(schlieren_like)

    fig, ax = plt.subplots(figsize=(5.5, 12))
    extent = [0.0, 2.0 * grid.x1_half_width, 0.0, grid.x0_max - grid.x0_min]
    ax.imshow(full_field.T, cmap="gray", extent=extent, aspect="auto", origin="lower")

    for sign in (1.0, -1.0):
        plot_x = sign * main_contour[:, 1] + grid.x1_half_width
        plot_y = grid.x0_max - main_contour[:, 0]
        ax.plot(plot_x, plot_y, color="yellow", linewidth=0.8)

    colours = {
        "downstream_helium_interface": "lime",
        "upstream_helium_interface": "magenta",
        "jet_head": "cyan",
        "transmitted_shock": "red",
    }
    for _, row in feature_df.iterrows():
        raw_x0 = row["raw_x0_mm"]
        raw_x1 = row["raw_x1_mm"]
        if not np.isfinite(raw_x0) or not np.isfinite(raw_x1):
            continue
        signs = [1.0] if abs(float(raw_x1)) < 1.0e-10 else [1.0, -1.0]
        for sign in signs:
            plot_x, plot_y = raw_to_plot(float(raw_x0), sign * float(raw_x1), grid)
            ax.scatter(
                plot_x,
                plot_y,
                s=55,
                color=colours.get(row["feature"], "white"),
                edgecolor="black",
                zorder=10,
            )

    ax.set_xlabel("plot x from left [mm]")
    ax.set_ylabel("plot y from bottom [mm]")
    ax.set_title("Detected features in bottom-left physical coordinates")
    ax.set_xlim(0.0, 2.0 * grid.x1_half_width)
    ax.set_ylim(0.0, grid.x0_max - grid.x0_min)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
