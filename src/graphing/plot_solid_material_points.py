#!/usr/bin/env python3

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib import cm, colors
import numpy as np


def _array_text(piece: ET.Element, parent_name: str, array_name: str | None = None) -> str:
    parent = piece.find(parent_name)
    if parent is None:
        raise ValueError(f"Missing VTP section: {parent_name}")

    for data_array in parent.iter("DataArray"):
        if array_name is None or data_array.attrib.get("Name") == array_name:
            return data_array.text or ""

    raise ValueError(f"Missing VTP DataArray: {array_name}")


def read_vtp_points(path: Path) -> dict[str, np.ndarray]:
    root = ET.parse(path).getroot()
    piece = root.find(".//Piece")
    if piece is None:
        raise ValueError(f"Missing VTP Piece in {path}")

    points = np.fromstring(_array_text(piece, "Points"), sep=" ", dtype=float)
    if points.size % 3 != 0:
        raise ValueError(f"Point array in {path} is not divisible by 3")
    points = points.reshape((-1, 3))

    arrays: dict[str, np.ndarray] = {"points": points}
    point_data = piece.find("PointData")
    if point_data is not None:
        for data_array in point_data.iter("DataArray"):
            name = data_array.attrib.get("Name")
            if name:
                arrays[name] = np.fromstring(data_array.text or "", sep=" ", dtype=float)
    return arrays


def sample_indices(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return np.linspace(0, n - 1, max_points, dtype=int)


def surface_indices(points: np.ndarray, max_points: int) -> np.ndarray:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    tol = np.zeros(3)
    for d in range(3):
        values = np.unique(np.round(points[:, d], decimals=12))
        diffs = np.diff(values)
        positive = diffs[diffs > 0.0]
        tol[d] = 0.51 * positive.min() if positive.size else 1.0e-12

    mask = np.zeros(points.shape[0], dtype=bool)
    for d in range(3):
        mask |= points[:, d] <= mins[d] + tol[d]
        mask |= points[:, d] >= maxs[d] - tol[d]

    indices = np.flatnonzero(mask)
    if indices.size <= max_points:
        return indices
    return indices[np.linspace(0, indices.size - 1, max_points, dtype=int)]


def boundary_tolerance(points: np.ndarray) -> np.ndarray:
    tol = np.zeros(3)
    for d in range(3):
        values = np.unique(np.round(points[:, d], decimals=12))
        diffs = np.diff(values)
        positive = diffs[diffs > 0.0]
        tol[d] = 0.51 * positive.min() if positive.size else 1.0e-12
    return tol


def set_equal_3d(ax, points_cm: np.ndarray) -> None:
    mins = points_cm.min(axis=0)
    maxs = points_cm.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    if radius <= 0.0:
        radius = 1.0

    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def draw_cube_edges(ax, points_cm: np.ndarray, **kwargs) -> None:
    mins = points_cm.min(axis=0)
    maxs = points_cm.max(axis=0)
    corners = np.array(
        [
            [mins[0], mins[1], mins[2]],
            [maxs[0], mins[1], mins[2]],
            [maxs[0], maxs[1], mins[2]],
            [mins[0], maxs[1], mins[2]],
            [mins[0], mins[1], maxs[2]],
            [maxs[0], mins[1], maxs[2]],
            [maxs[0], maxs[1], maxs[2]],
            [mins[0], maxs[1], maxs[2]],
        ]
    )
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for i, j in edges:
        ax.plot(
            [corners[i, 0], corners[j, 0]],
            [corners[i, 1], corners[j, 1]],
            [corners[i, 2], corners[j, 2]],
            **kwargs,
        )


def plot_cube_faces(
    ax,
    reference_points: np.ndarray,
    display_points_cm: np.ndarray,
    values: np.ndarray | None = None,
    norm: colors.Normalize | None = None,
    cmap_name: str = "coolwarm",
):
    mins = reference_points.min(axis=0)
    maxs = reference_points.max(axis=0)
    tol = boundary_tolerance(reference_points)
    cmap = plt.get_cmap(cmap_name)
    collections = []

    for d in range(3):
        axes = [a for a in range(3) if a != d]
        for side in (mins[d], maxs[d]):
            mask = np.abs(reference_points[:, d] - side) <= tol[d]
            if np.count_nonzero(mask) < 3:
                continue

            face_points = display_points_cm[mask]
            triangulation = mtri.Triangulation(
                reference_points[mask, axes[0]],
                reference_points[mask, axes[1]],
            )
            collection = ax.plot_trisurf(
                face_points[:, 0],
                face_points[:, 1],
                face_points[:, 2],
                triangles=triangulation.triangles,
                linewidth=0.08,
                antialiased=True,
                shade=False,
                alpha=0.82,
                color="0.75" if values is None else None,
            )
            if values is not None and norm is not None:
                face_values = values[mask]
                triangle_values = face_values[triangulation.triangles].mean(axis=1)
                collection.set_array(triangle_values)
                collection.set_cmap(cmap)
                collection.set_norm(norm)
            collections.append(collection)

    return collections


def plot_material_points_2d(
    x0: np.ndarray,
    x1: np.ndarray,
    displacement: np.ndarray,
    r0: np.ndarray,
    radial_disp: np.ndarray,
    idx: np.ndarray,
    out_path: Path | None,
    scale: float,
) -> None:
    x0_cm = 100.0 * x0[idx]
    x1_cm = 100.0 * x1[idx]
    displacement_cm = 100.0 * displacement[idx]
    radial_disp_mm = 1000.0 * radial_disp[idx]

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 8.0))
    ax0, ax1, ax2, ax3 = axes.flatten()

    ax0.scatter(x0_cm[:, 0], x0_cm[:, 1], s=2.0, c="0.25", linewidths=0.0)
    ax0.set_title("Initial material points")
    ax0.set_xlabel("x (cm)")
    ax0.set_ylabel("y (cm)")

    sc = ax1.scatter(
        x1_cm[:, 0],
        x1_cm[:, 1],
        s=2.0,
        c=radial_disp_mm,
        cmap="coolwarm",
        linewidths=0.0,
    )
    ax1.set_title("Final material points")
    ax1.set_xlabel("x (cm)")
    ax1.set_ylabel("y (cm)")
    cbar = fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("radial displacement (mm)")

    ax2.scatter(
        x0_cm[:, 0],
        x0_cm[:, 1],
        s=1.0,
        c="0.75",
        linewidths=0.0,
        label="initial",
    )
    ax2.quiver(
        x0_cm[:, 0],
        x0_cm[:, 1],
        scale * displacement_cm[:, 0],
        scale * displacement_cm[:, 1],
        radial_disp_mm,
        cmap="coolwarm",
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.002,
    )
    ax2.set_title(f"Displacement vectors x{scale:g}")
    ax2.set_xlabel("x (cm)")
    ax2.set_ylabel("y (cm)")

    order = np.argsort(r0)
    ax3.plot(100.0 * r0[order], 1000.0 * radial_disp[order], ".", color="black", markersize=1.0)
    ax3.axhline(0.0, color="0.5", linewidth=0.8)
    ax3.set_title("Radial displacement")
    ax3.set_xlabel("initial radius (cm)")
    ax3.set_ylabel("radial displacement (mm)")

    for ax in (ax0, ax1, ax2):
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(direction="in", top=True, right=True)
    ax3.tick_params(direction="in", top=True, right=True)

    save_or_show(fig, out_path)


def plot_material_points_3d(
    x0: np.ndarray,
    x1: np.ndarray,
    displacement: np.ndarray,
    r0: np.ndarray,
    radial_disp: np.ndarray,
    idx: np.ndarray,
    out_path: Path | None,
    scale: float,
    max_vectors: int,
    elev: float,
    azim: float,
) -> None:
    surf_idx = surface_indices(x0, idx.size)
    x0_surface = x0[surf_idx]
    x0_cm = 100.0 * x0_surface
    deformed_cm = 100.0 * (x0_surface + scale * displacement[surf_idx])
    radial_disp_mm = 1000.0 * radial_disp[surf_idx]

    vector_idx = surf_idx
    if vector_idx.size > max_vectors:
        vector_idx = vector_idx[np.linspace(0, vector_idx.size - 1, max_vectors, dtype=int)]
    xv_cm = 100.0 * x0[vector_idx]
    dv_cm = 100.0 * displacement[vector_idx]

    fig = plt.figure(figsize=(11.5, 9.0))
    ax0 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1 = fig.add_subplot(2, 2, 2, projection="3d")
    ax2 = fig.add_subplot(2, 2, 3, projection="3d")
    ax3 = fig.add_subplot(2, 2, 4)

    plot_cube_faces(ax0, x0_surface, x0_cm)
    draw_cube_edges(ax0, x0_cm, color="black", linewidth=0.8)
    ax0.set_title("Initial cube surface")

    norm = colors.Normalize(vmin=radial_disp_mm.min(), vmax=radial_disp_mm.max())
    plot_cube_faces(ax1, x0_surface, deformed_cm, radial_disp_mm, norm=norm)
    draw_cube_edges(ax1, deformed_cm, color="black", linewidth=0.8)
    ax1.set_title(f"Deformed cube surface x{scale:g}")
    mappable = cm.ScalarMappable(norm=norm, cmap="coolwarm")
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax1, fraction=0.046, pad=0.02)
    cbar.set_label("radial displacement (mm)")

    ax2.scatter(xv_cm[:, 0], xv_cm[:, 1], xv_cm[:, 2], s=0.8, c="0.75", linewidths=0.0)
    ax2.quiver(
        xv_cm[:, 0],
        xv_cm[:, 1],
        xv_cm[:, 2],
        scale * dv_cm[:, 0],
        scale * dv_cm[:, 1],
        scale * dv_cm[:, 2],
        length=1.0,
        normalize=False,
        color="black",
        linewidth=0.35,
        arrow_length_ratio=0.15,
    )
    ax2.set_title(f"Displacement vectors x{scale:g}")

    order = np.argsort(r0)
    ax3.plot(100.0 * r0[order], 1000.0 * radial_disp[order], ".", color="black", markersize=1.0)
    ax3.axhline(0.0, color="0.5", linewidth=0.8)
    ax3.set_title("Radial displacement")
    ax3.set_xlabel("initial radius (cm)")
    ax3.set_ylabel("radial displacement (mm)")
    ax3.tick_params(direction="in", top=True, right=True)

    all_points_cm = np.vstack((x0_cm, deformed_cm))
    for ax in (ax0, ax1, ax2):
        set_equal_3d(ax, all_points_cm)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
        ax.set_zlabel("z (cm)")

    save_or_show(fig, out_path)


def save_or_show(fig: plt.Figure, out_path: Path | None) -> None:
    fig.tight_layout(pad=1.2)
    if out_path is None:
        plt.show()
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved figure to {out_path}")


def plot_material_points(
    initial_path: Path,
    final_path: Path,
    out_path: Path | None,
    max_points: int,
    scale: float,
    max_vectors: int,
    elev: float,
    azim: float,
) -> None:
    initial = read_vtp_points(initial_path)
    final = read_vtp_points(final_path)

    x0 = initial["points"]
    x1 = final["points"]
    if x0.shape != x1.shape:
        raise ValueError("Initial and final material-point files have different point counts")

    displacement = x1 - x0
    disp_mag = np.linalg.norm(displacement, axis=1)
    r0 = np.linalg.norm(x0, axis=1)
    r1 = np.linalg.norm(x1, axis=1)
    radial_disp = r1 - r0

    idx = sample_indices(x0.shape[0], max_points)

    if np.ptp(x0[:, 2]) > 0.0 or np.ptp(x1[:, 2]) > 0.0:
        plot_material_points_3d(
            x0,
            x1,
            displacement,
            r0,
            radial_disp,
            idx,
            out_path,
            scale,
            max_vectors,
            elev,
            azim,
        )
    else:
        plot_material_points_2d(x0, x1, displacement, r0, radial_disp, idx, out_path, scale)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Barton material-point deformation output.")
    parser.add_argument("initial", type=Path)
    parser.add_argument("final", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-points", type=int, default=12000)
    parser.add_argument("--max-vectors", type=int, default=1500)
    parser.add_argument("--scale", type=float, default=20.0)
    parser.add_argument("--elev", type=float, default=26.0)
    parser.add_argument("--azim", type=float, default=-54.0)
    args = parser.parse_args()

    plot_material_points(
        args.initial,
        args.final,
        args.out,
        args.max_points,
        args.scale,
        args.max_vectors,
        args.elev,
        args.azim,
    )


if __name__ == "__main__":
    main()
