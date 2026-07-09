#!/usr/bin/env python3
"""Extract Gorsse et al. TC9 reference interface markers from PDF page renders.

The published TC9 figures contain Schlieren panels with a red iso-zero level-set
contour. This script crops the left-column Schlieren panels from rendered PDF
pages and converts that red contour into a reproducible physical-coordinate
reference table for later comparison against solver CSV outputs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


@dataclass(frozen=True)
class PageSpec:
    page_name: str
    times_us: tuple[int, int, int]


@dataclass(frozen=True)
class PanelBox:
    left: int
    right: int
    top: int
    bottom: int


PAGE_SPECS = (
    PageSpec("page-21.png", (106, 204, 301)),
    PageSpec("page-22.png", (358, 406, 500)),
)


def contiguous_bands(indices: np.ndarray, gap: int = 3) -> list[tuple[int, int]]:
    if indices.size == 0:
        return []
    bands: list[tuple[int, int]] = []
    start = int(indices[0])
    previous = int(indices[0])
    for value in indices[1:]:
        current = int(value)
        if current > previous + gap:
            bands.append((start, previous))
            start = current
        previous = current
    bands.append((start, previous))
    return bands


def detect_left_schlieren_panels(page: np.ndarray, expected_count: int = 3) -> list[PanelBox]:
    """Detect the framed left-column Schlieren panels in a rendered paper page."""

    dark = np.all(page < 70, axis=2)
    search_width = int(0.55 * page.shape[1])
    row_counts = dark[:, :search_width].sum(axis=1)
    long_row_threshold = int(0.20 * page.shape[1])
    row_bands = contiguous_bands(np.where(row_counts > long_row_threshold)[0])

    horizontal_lines: list[int] = []
    for start, end in row_bands:
        if int(row_counts[start : end + 1].max()) < int(0.35 * page.shape[1]):
            continue
        horizontal_lines.append(int(round(0.5 * (start + end))))

    if len(horizontal_lines) < 2 * expected_count:
        raise RuntimeError(
            f"Detected {len(horizontal_lines)} panel border lines, "
            f"expected at least {2 * expected_count}."
        )

    panels: list[PanelBox] = []
    for index in range(expected_count):
        top = horizontal_lines[2 * index]
        bottom = horizontal_lines[2 * index + 1]
        top_pixels = np.where(dark[top, :search_width])[0]
        bottom_pixels = np.where(dark[bottom, :search_width])[0]
        border_pixels = top_pixels if top_pixels.size >= bottom_pixels.size else bottom_pixels
        if border_pixels.size < long_row_threshold:
            raise RuntimeError(f"Could not detect horizontal extent for panel {index + 1}.")
        panels.append(
            PanelBox(
                left=int(border_pixels.min()),
                right=int(border_pixels.max()),
                top=top,
                bottom=bottom,
            )
        )

    return panels


def red_contour_mask(panel: np.ndarray) -> np.ndarray:
    """Return a mask for the anti-aliased red iso-zero contour."""

    red = panel[:, :, 0].astype(int)
    green = panel[:, :, 1].astype(int)
    blue = panel[:, :, 2].astype(int)
    return (
        (red > 65)
        & (red > 1.25 * green + 8)
        & (red > 1.25 * blue + 8)
        & (green < 95)
        & (blue < 95)
    )


def connected_components(mask: np.ndarray) -> np.ndarray:
    """Label 8-connected contour components without requiring SciPy."""

    labels = np.zeros(mask.shape, dtype=np.int32)
    red_rows, red_cols = np.where(mask)
    red_points = set(zip(red_rows.tolist(), red_cols.tolist()))
    label = 0
    offsets = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    )

    while red_points:
        label += 1
        start = red_points.pop()
        labels[start] = label
        stack = [start]
        while stack:
            row, col = stack.pop()
            for drow, dcol in offsets:
                candidate = (row + drow, col + dcol)
                if candidate not in red_points:
                    continue
                red_points.remove(candidate)
                labels[candidate] = label
                stack.append(candidate)

    return labels


def pixel_to_physical(
    cols: np.ndarray,
    rows: np.ndarray,
    *,
    width_px: int,
    height_px: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_values = x_min + cols.astype(float) / max(width_px - 1, 1) * (x_max - x_min)
    y_values = y_max - rows.astype(float) / max(height_px - 1, 1) * (y_max - y_min)
    return x_values, y_values


def save_mask_overlay(panel: np.ndarray, mask: np.ndarray, output_path: Path) -> None:
    overlay = panel.copy()
    overlay[mask] = np.array([255, 0, 0], dtype=np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output_path)


def append_contour_rows(
    rows: list[dict[str, float | int | str]],
    *,
    time_us: int,
    page_name: str,
    panel_name: str,
    mask: np.ndarray,
    labels: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> None:
    red_rows, red_cols = np.where(mask)
    x_values, y_values = pixel_to_physical(
        red_cols,
        red_rows,
        width_px=mask.shape[1],
        height_px=mask.shape[0],
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )
    for row_px, col_px, x0, x1 in zip(red_rows, red_cols, x_values, y_values):
        rows.append(
            {
                "time_us": time_us,
                "source_page": page_name,
                "panel_file": panel_name,
                "component_id": int(labels[row_px, col_px]),
                "pixel_col": int(col_px),
                "pixel_row": int(row_px),
                "x0_m": float(x0),
                "x1_m": float(x1),
            }
        )


def component_summary(
    contour_df: pd.DataFrame,
    *,
    time_us: int,
    source_page: str,
    panel_file: str,
) -> list[dict[str, float | int | str]]:
    frame = contour_df[contour_df["time_us"] == time_us]
    rows: list[dict[str, float | int | str]] = []
    for component_id, group in frame.groupby("component_id"):
        rows.append(
            {
                "time_us": time_us,
                "source_page": source_page,
                "panel_file": panel_file,
                "component_id": int(component_id),
                "red_pixel_count": int(len(group)),
                "x0_min_m": float(group["x0_m"].min()),
                "x0_max_m": float(group["x0_m"].max()),
                "x1_min_m": float(group["x1_m"].min()),
                "x1_max_m": float(group["x1_m"].max()),
                "centroid_x0_m": float(group["x0_m"].mean()),
                "centroid_x1_m": float(group["x1_m"].mean()),
            }
        )
    if not frame.empty:
        rows.append(
            {
                "time_us": time_us,
                "source_page": source_page,
                "panel_file": panel_file,
                "component_id": 0,
                "red_pixel_count": int(len(frame)),
                "x0_min_m": float(frame["x0_m"].min()),
                "x0_max_m": float(frame["x0_m"].max()),
                "x1_min_m": float(frame["x1_m"].min()),
                "x1_max_m": float(frame["x1_m"].max()),
                "centroid_x0_m": float(frame["x0_m"].mean()),
                "centroid_x1_m": float(frame["x1_m"].mean()),
            }
        )
    return rows


def extract_reference(args: argparse.Namespace) -> None:
    contour_rows: list[dict[str, float | int | str]] = []
    panel_metadata: list[dict[str, float | int | str]] = []

    panel_dir = args.output_dir / "reference_schlieren"
    mask_dir = args.output_dir / "reference_contour_masks"
    panel_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for page_spec in PAGE_SPECS:
        page_path = args.page_dir / page_spec.page_name
        if not page_path.exists():
            raise FileNotFoundError(f"Missing rendered PDF page: {page_path}")
        page = np.asarray(Image.open(page_path).convert("RGB"))
        panels = detect_left_schlieren_panels(page)
        for panel_box, time_us in zip(panels, page_spec.times_us):
            domain_aspect = (args.x_max - args.x_min) / (args.y_max - args.y_min)
            panel_height = panel_box.bottom - panel_box.top
            physical_right = panel_box.left + int(round(domain_aspect * panel_height))
            physical_box = PanelBox(
                left=panel_box.left,
                right=min(physical_right, page.shape[1] - 1),
                top=panel_box.top,
                bottom=panel_box.bottom,
            )
            panel = page[
                physical_box.top + args.border_trim : physical_box.bottom - args.border_trim,
                physical_box.left + args.border_trim : physical_box.right - args.border_trim,
            ]
            mask = red_contour_mask(panel)
            labels = connected_components(mask)

            panel_name = f"gorsse_tc9_reference_t{time_us}us_schlieren.png"
            mask_name = f"gorsse_tc9_reference_t{time_us}us_red_contour_overlay.png"
            Image.fromarray(panel).save(panel_dir / panel_name)
            save_mask_overlay(panel, mask, mask_dir / mask_name)

            append_contour_rows(
                contour_rows,
                time_us=time_us,
                page_name=page_spec.page_name,
                panel_name=panel_name,
                mask=mask,
                labels=labels,
                x_min=args.x_min,
                x_max=args.x_max,
                y_min=args.y_min,
                y_max=args.y_max,
            )
            panel_metadata.append(
                {
                    "time_us": time_us,
                    "source_page": page_spec.page_name,
                    "panel_file": panel_name,
                    "mask_overlay_file": mask_name,
                    "left_px": physical_box.left,
                    "right_px": physical_box.right,
                    "top_px": physical_box.top,
                    "bottom_px": physical_box.bottom,
                    "detected_right_px": panel_box.right,
                    "red_pixel_count": int(mask.sum()),
                }
            )

    contour_df = pd.DataFrame(contour_rows)
    contour_csv = args.output_dir / "gorsse_tc9_reference_contour_points.csv"
    contour_df.to_csv(contour_csv, index=False)

    summary_rows: list[dict[str, float | int | str]] = []
    for metadata in panel_metadata:
        summary_rows.extend(
            component_summary(
                contour_df,
                time_us=int(metadata["time_us"]),
                source_page=str(metadata["source_page"]),
                panel_file=str(metadata["panel_file"]),
            )
        )
    summary_csv = args.output_dir / "gorsse_tc9_reference_interface_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    metadata_csv = args.output_dir / "gorsse_tc9_reference_panel_metadata.csv"
    pd.DataFrame(panel_metadata).to_csv(metadata_csv, index=False)

    print(f"Wrote {contour_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {metadata_csv}")
    print(f"Wrote panel crops under {panel_dir}")
    print(f"Wrote contour overlays under {mask_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--page-dir",
        type=Path,
        default=Path("tmp/pdfs/gorsse_2014_tc9"),
        help="Directory containing rendered page-21.png and page-22.png.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/bubble_collapse_validation/gorsse_2014_tc9"),
        help="Directory for cropped reference panels and extracted contour CSVs.",
    )
    parser.add_argument("--x-min", type=float, default=-0.2)
    parser.add_argument("--x-max", type=float, default=1.0)
    parser.add_argument("--y-min", type=float, default=0.0)
    parser.add_argument("--y-max", type=float, default=1.0)
    parser.add_argument(
        "--border-trim",
        type=int,
        default=2,
        help="Pixels removed from detected panel borders before contour extraction.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    extract_reference(parser.parse_args())


if __name__ == "__main__":
    main()
