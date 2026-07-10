#!/usr/bin/env python3
"""Compatibility CLI for helium shock-bubble feature tracking.

The extraction logic lives in `bubble_feature_core.py`; plotting and image
coordinate helpers live in `bubble_feature_plotting.py`. This wrapper keeps the
old import path and command-line interface working.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bubble_feature_core import (  # noqa: F401
    FEATURES,
    GridData,
    centreline_density_gradient_ridge,
    centreline_phi_crossings,
    centreline_phi_line,
    centreline_rho_line,
    contour_to_physical,
    detect_downstream_upper_cap,
    detect_features,
    detect_jet_from_centreline_crossing_cluster,
    detect_upstream_outer_contour,
    expected_transverse_y,
    extract_main_phi_contour,
    extract_transmitted_shock_candidates,
    extract_transverse_wave_candidates,
    extract_upstream_interface_candidates,
    feature_row,
    gaussian_filter,
    load_grid,
    parse_time_from_name,
    raw_to_plot,
    select_transmitted_shock_branch,
    select_transverse_wave_branch,
    select_upstream_interface_branch,
    sort_contour_points,
    time_token_from_name,
    transmitted_shock,
    transmitted_shock_slope_bounds,
    transverse_wave,
    zero_crossings_1d,
)
from bubble_feature_plotting import (  # noqa: F401
    image_domain_bounds,
    make_mirrored_field,
    physical_to_image_pixel,
    plot_diagnostic,
    plot_overlay,
)


def run_tracker(
    csv_path: str | Path,
    image_path: str | Path | None,
    *,
    output_csv: str | Path | None = None,
    overlay_png: str | Path | None = None,
    diagnostic_png: str | Path | None = None,
    jet_min_time_code: float = 28.22,
    wave_min_time_code: float = 42.33,
) -> pd.DataFrame:
    """Track one snapshot and optionally write CSV/overlay/diagnostic artifacts."""

    grid = load_grid(csv_path)
    time_code = parse_time_from_name(csv_path)
    bounds = image_domain_bounds(image_path) if image_path is not None else None
    feature_df, main_contour = detect_features(
        grid,
        time_code=time_code,
        bounds=bounds,
        jet_min_time_code=jet_min_time_code,
        wave_min_time_code=wave_min_time_code,
    )
    feature_df.insert(0, "csv_file", str(csv_path))
    feature_df.insert(1, "image_file", "" if image_path is None else str(image_path))

    if output_csv is not None:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        feature_df.to_csv(output_csv, index=False)
    if image_path is not None and overlay_png is not None:
        Path(overlay_png).parent.mkdir(parents=True, exist_ok=True)
        plot_overlay(image_path, grid, feature_df, main_contour, overlay_png)
    if diagnostic_png is not None:
        Path(diagnostic_png).parent.mkdir(parents=True, exist_ok=True)
        plot_diagnostic(grid, feature_df, main_contour, diagnostic_png)
    return feature_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--image")
    parser.add_argument("--output-csv")
    parser.add_argument("--overlay-png")
    parser.add_argument("--diagnostic-png")
    parser.add_argument("--outdir", default="results/bubble_features/single_frame")
    parser.add_argument("--jet-min-time-code", type=float, default=28.22)
    parser.add_argument("--wave-min-time-code", type=float, default=42.33)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    stem = Path(args.csv).stem
    output_csv = args.output_csv or outdir / f"{stem}_features.csv"
    overlay_png = args.overlay_png or outdir / f"{stem}_overlay.png"
    diagnostic_png = args.diagnostic_png or outdir / f"{stem}_diagnostic.png"
    feature_df = run_tracker(
        args.csv,
        args.image,
        output_csv=output_csv,
        overlay_png=overlay_png if args.image else None,
        diagnostic_png=diagnostic_png,
        jet_min_time_code=args.jet_min_time_code,
        wave_min_time_code=args.wave_min_time_code,
    )
    print(f"Wrote {output_csv}")
    if args.image:
        print(f"Wrote {overlay_png}")
    print(f"Wrote {diagnostic_png}")
    print(feature_df.to_string(index=False))


if __name__ == "__main__":
    main()
