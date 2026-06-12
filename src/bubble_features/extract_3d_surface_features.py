#!/usr/bin/env python3
"""Publication-facing 3D helium bubble interface plotter.

This replaces the old diagnostic scatter cloud with a connected zero-level
surface reconstructed by marching cubes. For quadrant-domain bubble runs, the
surface can be mirrored across x1=0 and x2=0 so the rendered object represents
the full physical bubble rather than the reduced computational quadrant.

Required columns:
    x0, x1, x2, and either phi0 or alpha1.

Examples:
    python pretty_3d_helium_surface_plot.py \
        --csv quant_gfm_helium_bubble_3d_325x45x45_t1p411000ep01_N325_N45_N45.csv \
        --outdir plots_3d

    python pretty_3d_helium_surface_plot.py --csv frame.csv --outdir plots_3d --no-mirror
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import measure


def load_interface_field(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Load the structured 3D interface field as phi[z, y, x]."""

    df = pd.read_csv(csv_path)
    required = {"x0", "x1", "x2"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

    if "phi0" in df.columns:
        interface = df["phi0"].to_numpy(dtype=float)
        interface_label = "phi0 = 0"
    elif "alpha1" in df.columns:
        interface = df["alpha1"].to_numpy(dtype=float) - 0.5
        interface_label = "alpha1 = 0.5"
    else:
        raise ValueError(f"{csv_path} must contain either phi0 or alpha1")

    df = df.assign(_interface=interface)
    df = df.drop_duplicates(subset=["x0", "x1", "x2"])

    x0 = np.sort(df["x0"].unique())
    x1 = np.sort(df["x1"].unique())
    x2 = np.sort(df["x2"].unique())
    expected = len(x0) * len(x1) * len(x2)
    if len(df) != expected:
        raise ValueError(f"Structured grid is incomplete: got {len(df)} rows, expected {expected}")

    ordered = df.sort_values(["x2", "x1", "x0"], kind="mergesort")
    phi = ordered["_interface"].to_numpy(dtype=float).reshape((len(x2), len(x1), len(x0)))
    return x0, x1, x2, phi, interface_label


def mirror_quadrant_field(
    x1: np.ndarray,
    x2: np.ndarray,
    phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reflect a positive x1/x2 quadrant field into the full physical cross-section."""

    x1_full = np.concatenate([-x1[::-1], x1])
    x2_full = np.concatenate([-x2[::-1], x2])
    phi_y = np.concatenate([phi[:, ::-1, :], phi], axis=1)
    phi_full = np.concatenate([phi_y[::-1, :, :], phi_y], axis=0)
    return x1_full, x2_full, phi_full


def marching_cubes_surface(
    x0: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    phi: np.ndarray,
    level: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return vertices as [x0, x1, x2] and triangular faces."""

    dx0 = float(np.median(np.diff(x0))) if len(x0) > 1 else 1.0
    dx1 = float(np.median(np.diff(x1))) if len(x1) > 1 else 1.0
    dx2 = float(np.median(np.diff(x2))) if len(x2) > 1 else 1.0

    verts, faces, _, _ = measure.marching_cubes(phi, level=level, spacing=(dx2, dx1, dx0))

    coords = np.column_stack([
        verts[:, 2] + float(x0[0]),
        verts[:, 1] + float(x1[0]),
        verts[:, 0] + float(x2[0]),
    ])
    return coords, faces


def axial_feature_points(coords: np.ndarray) -> dict[str, tuple[float, float, float]]:
    """Find clean front/back axial markers for the rendered surface."""

    radius = np.hypot(coords[:, 1], coords[:, 2])
    near_axis_tol = max(1.25, float(np.quantile(radius, 0.015)))
    near_axis = coords[radius <= near_axis_tol]
    if len(near_axis) < 10:
        near_axis = coords

    upstream_x0 = float(np.max(near_axis[:, 0]))
    downstream_x0 = float(np.min(near_axis[:, 0]))
    return {
        "upstream interface": (upstream_x0, 0.0, 0.0),
        "downstream interface": (downstream_x0, 0.0, 0.0),
    }


def set_equal_axes(ax, coords: np.ndarray, pad: float = 4.0) -> None:
    """Use equal 3D scaling around the interface, not the whole simulation box."""

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    centre = 0.5 * (mins + maxs)
    half_range = 0.5 * float(np.max(maxs - mins)) + pad

    ax.set_xlim(centre[0] - half_range, centre[0] + half_range)
    ax.set_ylim(centre[1] - half_range, centre[1] + half_range)
    ax.set_zlim(centre[2] - half_range, centre[2] + half_range)
    ax.set_box_aspect((1, 1, 1))


def save_static_surface(
    coords: np.ndarray,
    faces: np.ndarray,
    output_path: Path,
    title: str,
    *,
    view: tuple[float, float] = (22.0, -58.0),
) -> None:
    """Save a connected triangular surface plot."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9.5, 7.5))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_trisurf(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        triangles=faces,
        linewidth=0.0,
        antialiased=True,
        alpha=0.85,
        shade=True,
    )

    x_min = float(coords[:, 0].min())
    x_max = float(coords[:, 0].max())
    ax.plot([x_min, x_max], [0.0, 0.0], [0.0, 0.0], linewidth=1.1, alpha=0.75)

    for label, point in axial_feature_points(coords).items():
        x0, x1, x2 = point
        ax.scatter([x0], [x1], [x2], s=100, edgecolors="black", linewidths=0.8, label=label)

    ax.set_title(title, pad=16)
    ax.set_xlabel("x0, shock direction [mm]", labelpad=8)
    ax.set_ylabel("x1 [mm]", labelpad=8)
    ax.set_zlabel("x2 [mm]", labelpad=8)
    set_equal_axes(ax, coords)
    ax.view_init(elev=view[0], azim=view[1])
    ax.grid(False)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_interactive_surface(coords: np.ndarray, faces: np.ndarray, output_path: Path, title: str) -> None:
    """Save an interactive Plotly HTML surface. Plotly is optional."""

    try:
        import plotly.graph_objects as go
    except Exception:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = go.Figure()
    fig.add_trace(go.Mesh3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        opacity=0.72,
        name="interface surface",
        flatshading=False,
        showscale=False,
    ))

    for label, point in axial_feature_points(coords).items():
        x0, x1, x2 = point
        fig.add_trace(go.Scatter3d(
            x=[x0],
            y=[x1],
            z=[x2],
            mode="markers+text",
            text=[label],
            textposition="top center",
            name=label,
        ))

    fig.add_trace(go.Scatter3d(
        x=[float(coords[:, 0].min()), float(coords[:, 0].max())],
        y=[0.0, 0.0],
        z=[0.0, 0.0],
        mode="lines",
        name="symmetry axis",
    ))

    fig.update_layout(
        title=title,
        scene={
            "xaxis_title": "x0, shock direction [mm]",
            "yaxis_title": "x1 [mm]",
            "zaxis_title": "x2 [mm]",
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "t": 45, "b": 0},
    )
    fig.write_html(output_path, include_plotlyjs="cdn")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--level", type=float, default=0.0)
    parser.add_argument("--no-mirror", action="store_true", help="Plot only the computational quadrant")
    parser.add_argument("--interactive", action="store_true", help="Also write an interactive Plotly HTML file")
    args = parser.parse_args()

    x0, x1, x2, phi, interface_label = load_interface_field(args.csv)

    if args.no_mirror:
        x1_plot, x2_plot, phi_plot = x1, x2, phi
        suffix = "quadrant_surface"
        title = f"3D helium bubble interface, quadrant-domain surface ({interface_label})"
    else:
        x1_plot, x2_plot, phi_plot = mirror_quadrant_field(x1, x2, phi)
        suffix = "mirrored_full_surface"
        title = f"3D helium bubble interface, mirrored full surface ({interface_label})"

    coords, faces = marching_cubes_surface(x0, x1_plot, x2_plot, phi_plot, level=args.level)

    base = args.csv.stem
    save_static_surface(
        coords,
        faces,
        args.outdir / f"{base}_{suffix}_oblique.png",
        title,
        view=(22.0, -58.0),
    )
    save_static_surface(
        coords,
        faces,
        args.outdir / f"{base}_{suffix}_side.png",
        title,
        view=(12.0, -85.0),
    )

    if args.interactive:
        save_interactive_surface(
            coords,
            faces,
            args.outdir / f"{base}_{suffix}_interactive.html",
            title,
        )

    print(f"vertices={len(coords)} faces={len(faces)}")
    print(f"wrote {args.outdir}")


if __name__ == "__main__":
    main()
