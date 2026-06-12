#!/usr/bin/env python3
"""Batch extract helium shock-bubble features from CSV/Schlieren pairs.

The batch pipeline has two passes:

1. run the deterministic per-frame detectors for every CSV snapshot,
2. collect candidates and overwrite selected features using temporal branch
   tracking, then regenerate per-frame CSVs/overlays from the final master
   table.

This keeps the raw candidate evidence available while making the report-facing
feature positions follow one physically continuous trajectory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bubble_feature_core import (  # noqa: E402
    extract_main_phi_contour,
    extract_transmitted_shock_candidates,
    extract_upstream_interface_candidates,
    load_grid,
    parse_time_from_name,
    raw_to_plot,
    select_transmitted_shock_branch,
    select_upstream_interface_branch,
    time_token_from_name,
)
from bubble_feature_plotting import image_domain_bounds, plot_diagnostic, plot_overlay  # noqa: E402
from bubble_feature_tracker import run_tracker  # noqa: E402


def find_matching_image(csv_path: Path, image_dir: Path, image_prefix: str) -> Path | None:
    """Find the Schlieren PNG with the same time token as a snapshot CSV."""

    token = time_token_from_name(csv_path)
    if token is None:
        return None
    search_dirs = [image_dir]
    if csv_path.parent not in search_dirs:
        search_dirs.append(csv_path.parent)
    for directory in search_dirs:
        candidates = sorted(directory.rglob(f"{image_prefix}_{token}_schlieren.png"))
        if candidates:
            return candidates[0]
        candidates = sorted(directory.rglob(f"*{token}*schlieren.png"))
        if candidates:
            return candidates[0]
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv-dir",
        default="data/csv/gfm/MM_2D_validation/gfm_helium_bubble_2d",
        help="Directory containing snapshot CSV files.",
    )
    parser.add_argument(
        "--csv-glob",
        default="gfm_helium_bubble_2d_t*_N1300_N178.csv",
        help="CSV glob relative to --csv-dir.",
    )
    parser.add_argument(
        "--image-dir",
        default="data/bubble_collapse_validation/GFM",
        help="Directory containing matching Schlieren PNGs.",
    )
    parser.add_argument(
        "--image-prefix",
        default="MM_2D_validation_gfm_helium_bubble_2d",
        help="PNG prefix before the time token.",
    )
    parser.add_argument("--outdir", default="results/bubble_features")
    parser.add_argument("--jet-min-time-code", type=float, default=28.22)
    parser.add_argument("--wave-min-time-code", type=float, default=42.33)
    parser.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="Still extract CSV features when matching PNGs are missing; overlays are skipped.",
    )
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    image_dir = Path(args.image_dir)
    outdir = Path(args.outdir)
    per_frame_dir = outdir / "per_frame"
    overlay_dir = outdir / "overlays"
    diagnostic_dir = outdir / "diagnostics"
    master_csv = outdir / "bubble_feature_positions_master.csv"
    shock_candidate_csv = outdir / "transmitted_shock_candidates.csv"
    shock_selected_branch_csv = outdir / "transmitted_shock_selected_branch.csv"
    upstream_candidate_csv = outdir / "upstream_interface_candidates.csv"
    upstream_selected_branch_csv = outdir / "upstream_interface_selected_branch.csv"

    csv_files = sorted(
        csv_dir.glob(args.csv_glob),
        key=lambda path: (float("inf") if parse_time_from_name(path) is None else parse_time_from_name(path), path.name),
    )
    if not csv_files:
        raise SystemExit(f"No CSV files matched {csv_dir / args.csv_glob}")

    all_rows = []
    all_shock_candidates = []
    all_upstream_candidates = []
    missing_images = []
    for csv_path in csv_files:
        image_path = find_matching_image(csv_path, image_dir, args.image_prefix)
        if image_path is None:
            missing_images.append(str(csv_path))
            if not args.allow_missing_images:
                continue

        stem = csv_path.stem
        frame_csv = per_frame_dir / f"{stem}_features.csv"
        overlay_png = overlay_dir / f"{stem}_overlay.png" if image_path is not None else None
        diagnostic_png = diagnostic_dir / f"{stem}_diagnostic.png"
        rows = run_tracker(
            csv_path,
            image_path,
            output_csv=frame_csv,
            overlay_png=overlay_png,
            diagnostic_png=diagnostic_png,
            jet_min_time_code=args.jet_min_time_code,
            wave_min_time_code=args.wave_min_time_code,
        )
        rows.insert(0, "frame_id", stem)
        all_rows.append(rows)

        # Candidate extraction is intentionally separate from the per-frame
        # detector. The final branch selection needs all frames at once, so the
        # batch driver stores every plausible candidate before it overwrites
        # the report-facing feature rows.
        grid = load_grid(csv_path)
        bounds = image_domain_bounds(image_path) if image_path is not None else None
        time_code = parse_time_from_name(csv_path)

        downstream_row = rows[rows["feature"] == "downstream_helium_interface"]
        jet_row = rows[rows["feature"] == "jet_head"]
        downstream_y = float("nan")
        jet_y = float("nan")
        if not downstream_row.empty and pd.notna(downstream_row.iloc[0]["raw_x0_mm"]):
            _px, downstream_y = raw_to_plot(
                float(downstream_row.iloc[0]["raw_x0_mm"]),
                float(downstream_row.iloc[0]["raw_x1_mm"]),
                grid,
            )
        if not jet_row.empty and pd.notna(jet_row.iloc[0]["raw_x0_mm"]):
            _px, jet_y = raw_to_plot(float(jet_row.iloc[0]["raw_x0_mm"]), 0.0, grid)

        shock_candidates = extract_transmitted_shock_candidates(
            grid,
            time_code=time_code,
            bounds=bounds,
            downstream_y_mm=downstream_y,
            jet_y_mm=jet_y,
            wave_min_time_code=args.wave_min_time_code,
            max_candidates=10,
        )
        if not shock_candidates.empty:
            shock_candidates.insert(0, "frame_id", stem)
            shock_candidates.insert(1, "csv_file", str(csv_path))
            shock_candidates.insert(2, "image_file", "" if image_path is None else str(image_path))
            all_shock_candidates.append(shock_candidates)

        upstream_candidates = extract_upstream_interface_candidates(
            grid,
            time_code=time_code,
            bounds=bounds,
            jet_y_mm=jet_y,
            max_candidates=12,
        )
        if not upstream_candidates.empty:
            upstream_candidates.insert(0, "frame_id", stem)
            upstream_candidates.insert(1, "csv_file", str(csv_path))
            upstream_candidates.insert(2, "image_file", "" if image_path is None else str(image_path))
            all_upstream_candidates.append(upstream_candidates)

        print(f"processed {csv_path.name}")

    if not all_rows:
        message = "No frames were processed."
        if missing_images:
            message += f" Missing images for {len(missing_images)} CSV files."
        raise SystemExit(message)

    outdir.mkdir(parents=True, exist_ok=True)
    master = pd.concat(all_rows, ignore_index=True)
    master = master.sort_values(["time_code", "feature"], na_position="last").reset_index(drop=True)

    if all_shock_candidates:
        shock_candidate_table = pd.concat(all_shock_candidates, ignore_index=True)
        shock_candidate_table = shock_candidate_table.sort_values(
            ["time_code", "candidate_rank"],
            na_position="last",
        ).reset_index(drop=True)
    else:
        shock_candidate_table = pd.DataFrame()
    shock_candidate_table.to_csv(shock_candidate_csv, index=False)

    if all_upstream_candidates:
        upstream_candidate_table = pd.concat(all_upstream_candidates, ignore_index=True)
        upstream_candidate_table = upstream_candidate_table.sort_values(
            ["time_code", "candidate_rank"],
            na_position="last",
        ).reset_index(drop=True)
    else:
        upstream_candidate_table = pd.DataFrame()
    upstream_candidate_table.to_csv(upstream_candidate_csv, index=False)

    shock_selected_branch = select_transmitted_shock_branch(
        shock_candidate_table,
        wave_min_time_code=args.wave_min_time_code,
    )
    shock_selected_branch.to_csv(shock_selected_branch_csv, index=False)

    upstream_selected_branch = select_upstream_interface_branch(upstream_candidate_table)
    upstream_selected_branch.to_csv(upstream_selected_branch_csv, index=False)

    def overwrite_feature_from_branch(feature_name: str, branch: pd.DataFrame, missing_source: str) -> None:
        """Replace per-frame detector rows with temporally selected rows."""

        selected_by_time = {}
        if not branch.empty:
            for _, row in branch.iterrows():
                selected_by_time[round(float(row["time_code"]), 9)] = row

        mask = master["feature"] == feature_name
        for idx in master.index[mask]:
            time_code = master.at[idx, "time_code"]
            if pd.isna(time_code):
                continue
            selected = selected_by_time.get(round(float(time_code), 9))
            if selected is None:
                continue
            for column in [
                "plot_x_mm_from_left",
                "plot_y_mm_from_bottom",
                "raw_x0_mm",
                "raw_x1_mm",
                "image_x_px_from_left",
                "image_y_px_from_bottom",
            ]:
                master.at[idx, column] = selected[column]
            master.at[idx, "source"] = missing_source
            master.at[idx, "definition"] = (
                f"{selected['definition']}; selected by temporal branch tracking"
            )
            master.at[idx, "confidence"] = "medium_to_high"

    overwrite_feature_from_branch(
        "transmitted_shock",
        shock_selected_branch,
        "temporal transmitted-shock pressure-density ridge branch",
    )
    overwrite_feature_from_branch(
        "upstream_helium_interface",
        upstream_selected_branch,
        "temporal upstream-interface contour branch",
    )

    master.to_csv(master_csv, index=False)

    # Regenerate frame-level artifacts after temporal overwrites so every CSV
    # and overlay agrees with the final master table.
    for frame_id, frame_rows in master.groupby("frame_id", sort=False):
        frame_csv = per_frame_dir / f"{frame_id}_features.csv"
        frame_csv.parent.mkdir(parents=True, exist_ok=True)
        frame_rows.drop(columns=["frame_id"]).to_csv(frame_csv, index=False)

        csv_path = Path(str(frame_rows.iloc[0]["csv_file"]))
        image_value = str(frame_rows.iloc[0]["image_file"]) if "image_file" in frame_rows else ""
        image_path = Path(image_value) if image_value else None
        grid = load_grid(csv_path)
        main_contour = extract_main_phi_contour(grid)
        diagnostic_png = diagnostic_dir / f"{frame_id}_diagnostic.png"
        diagnostic_png.parent.mkdir(parents=True, exist_ok=True)
        plot_diagnostic(grid, frame_rows, main_contour, diagnostic_png)
        if image_path is not None and image_path.exists():
            overlay_png = overlay_dir / f"{frame_id}_overlay.png"
            overlay_png.parent.mkdir(parents=True, exist_ok=True)
            plot_overlay(image_path, grid, frame_rows, main_contour, overlay_png)

    print(f"Wrote {master_csv}")
    print(f"Wrote {shock_candidate_csv}")
    print(f"Wrote {shock_selected_branch_csv}")
    print(f"Wrote {upstream_candidate_csv}")
    print(f"Wrote {upstream_selected_branch_csv}")
    print(f"Processed frames: {len(all_rows)}")
    if missing_images:
        print(f"Missing images: {len(missing_images)}")
        for item in missing_images[:10]:
            print(f"  {item}")


if __name__ == "__main__":
    main()
