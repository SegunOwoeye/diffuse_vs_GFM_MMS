"""Plot 2D fluid-solid rGFM paper contours with the shared MM plot style."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import matplotlib.tri as mtri
try:
    from scipy.interpolate import griddata
except ImportError:  # pragma: no cover - optional plotting dependency
    griddata = None

from plot_style import save_figure


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_csv(root: Path) -> Path:
    return (
        root
        / "data"
        / "csv"
        / "fluid_solid"
        / "case4_3_rotated_water_solid_rgfm"
        / "case4_3_rotated_water_solid_rgfm_N251x251.csv"
    )


def load_case(csv_path: Path) -> dict[str, np.ndarray]:
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding=None)
    if data.size == 0:
        raise ValueError(f"No rows found in {csv_path}")
    names = set(data.dtype.names or ())

    x_unique = np.unique(data["x"])
    y_unique = np.unique(data["y"])
    nx = len(x_unique)
    ny = len(y_unique)
    order = np.lexsort((data["x"], data["y"]))
    data = data[order]

    def field(name: str) -> np.ndarray:
        return np.asarray(data[name], dtype=float).reshape(ny, nx)

    material = np.asarray(data["material"]).reshape(ny, nx)
    fluid = material == "fluid"
    solid = material == "solid"

    vn = field("vn")
    sigma_nn = field("sigma_nn")
    sigma_ss = field("sigma_ss")
    sigma_sn = field("sigma_sn")
    phi = field("phi")
    X, Y = np.meshgrid(x_unique, y_unique)
    if {"x_lag", "y_lag"}.issubset(names):
        x_lag = field("x_lag")
        y_lag = field("y_lag")
    else:
        normal = infer_interface_normal(X, Y, phi)
        tangent = np.array([-normal[1], normal[0]])
        x_lag = normal[0] * X + normal[1] * Y
        y_lag = tangent[0] * X + tangent[1] * Y
    y_lag_offset = np.nanmin(y_lag)
    y_lag = y_lag - y_lag_offset

    has_material_coordinates = {"x_mat", "y_mat"}.issubset(names)
    if has_material_coordinates:
        x_mat = field("x_mat")
        y_mat = field("y_mat") - y_lag_offset
    else:
        x_mat = np.where(solid, x_lag, np.nan)
        y_mat = np.where(solid, y_lag, np.nan)

    return {
        "x": x_unique,
        "y": y_unique,
        "X": X,
        "Y": Y,
        "x_lag": x_lag,
        "y_lag": y_lag,
        "x_mat": x_mat,
        "y_mat": y_mat,
        "has_material_coordinates": has_material_coordinates,
        "phi": phi,
        "fluid_mask": fluid,
        "solid_mask": solid,
        "vn_fluid": np.where(fluid, vn, 0.0),
        "sigma_nn_solid": np.where(solid, sigma_nn, 0.0),
        "sigma_ss_solid": np.where(solid, sigma_ss, 0.0),
        "sigma_sn_solid": np.where(solid, sigma_sn, 0.0),
        "vn_fluid_limits": (-20.0, 30.0),
        "sigma_nn_solid_limits": (-50000.0, 0.0),
        "sigma_ss_solid_limits": (-25000.0, 0.0),
        "sigma_sn_solid_limits": (-1500.0, 1000.0),
        "vn_fluid_surface": np.where(fluid, vn, np.nan),
        "sigma_nn_solid_surface": np.where(solid, sigma_nn, np.nan),
        "sigma_ss_solid_surface": np.where(solid, sigma_ss, np.nan),
        "sigma_sn_solid_surface": np.where(solid, sigma_sn, np.nan),
        "vn_fluid_surface_limits": (-20.0, 30.0),
        "sigma_nn_solid_surface_limits": (-50000.0, 0.0),
        "sigma_ss_solid_surface_limits": (-25000.0, 0.0),
        "sigma_sn_solid_surface_limits": (-1500.0, 1000.0),
    }


def infer_interface_normal(
    x: np.ndarray,
    y: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    A = np.column_stack([x.ravel(), y.ravel(), np.ones(x.size)])
    coeffs, *_ = np.linalg.lstsq(A, phi.ravel(), rcond=None)
    normal = np.array([coeffs[0], coeffs[1]], dtype=float)
    mag = np.linalg.norm(normal)
    if not np.isfinite(mag) or mag <= 1.0e-14:
        raise ValueError("Could not infer interface normal from phi")
    return normal / mag


def finite_limits(field: np.ndarray, symmetric: bool = False) -> tuple[float, float]:
    finite = field[np.isfinite(field)]
    if finite.size == 0:
        return (-1.0, 1.0)
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if symmetric:
        mag = max(abs(lo), abs(hi), 1.0e-12)
        return (-mag, mag)
    if abs(hi - lo) < 1.0e-12:
        pad = max(abs(hi), 1.0) * 0.05
        return (lo - pad, hi + pad)
    return (lo, hi)


def interpolate_lag_window(
    case: dict[str, np.ndarray],
    field: str,
    coordinates: str = "lagrangian",
    x_limits: tuple[float, float] = (0.0, 10.0),
    y_limits: tuple[float, float] = (3.0, 7.0),
    nx: int = 251,
    ny: int = 121,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if coordinates == "material":
        x_points = np.asarray(case["x_mat"], dtype=float).ravel()
        y_points = np.asarray(case["y_mat"], dtype=float).ravel()
    else:
        x_points = np.asarray(case["x_lag"], dtype=float).ravel()
        y_points = np.asarray(case["y_lag"], dtype=float).ravel()
    values = np.asarray(case[field], dtype=float).ravel()
    valid = np.isfinite(x_points) & np.isfinite(y_points) & np.isfinite(values)

    xi = np.linspace(x_limits[0], x_limits[1], nx)
    yi = np.linspace(y_limits[0], y_limits[1], ny)
    X, Y = np.meshgrid(xi, yi)
    points = np.column_stack([x_points[valid], y_points[valid]])
    if griddata is not None:
        Z = griddata(points, values[valid], (X, Y), method="linear")
    else:
        triangulation = mtri.Triangulation(points[:, 0], points[:, 1])
        interpolator = mtri.LinearTriInterpolator(triangulation, values[valid])
        Z = interpolator(X, Y)
        if np.ma.isMaskedArray(Z):
            Z = Z.filled(np.nan)
    return X, Y, np.asarray(Z, dtype=float)


def uses_solid_material_coordinates(case: dict[str, np.ndarray], field: str) -> bool:
    return bool(case["has_material_coordinates"]) and field.startswith("sigma_")


def paper_axis_labels(field: str, coordinates: str) -> tuple[str, str]:
    if coordinates == "cartesian":
        return (r"$x$", r"$y$")
    if field.startswith("vn_"):
        return (r"$x$", r"$y$")
    return (r"$x_{\mathrm{Lag}}$", r"$y_{\mathrm{Lag}}$")


def draw_panel(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    phi: np.ndarray,
    values: np.ndarray,
    label: str,
    cmap: str,
    symmetric: bool = False,
) -> None:
    vmin, vmax = finite_limits(values, symmetric=symmetric)
    mesh = ax.pcolormesh(x, y, values, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.contour(x, y, phi, levels=[0.0], colors="black", linewidths=0.9)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.tick_params(direction="in", top=True, right=True)
    cbar = ax.figure.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label)


def render_pair(case: dict[str, np.ndarray], output_path: Path, fields) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.9), constrained_layout=True)
    for ax, (field, label, cmap, symmetric) in zip(axes, fields):
        draw_panel(
            ax,
            case["x"],
            case["y"],
            case["phi"],
            case[field],
            label,
            cmap,
            symmetric=symmetric,
        )
    save_figure(fig, output_path)
    return output_path


def draw_surface_panel(
    ax,
    x_lag: np.ndarray,
    y_lag: np.ndarray,
    values: np.ndarray,
    label: str,
    cmap_name: str,
    symmetric: bool = False,
    x_label: str = r"$x$",
    y_label: str = r"$y$",
    value_limits: tuple[float, float] | None = None,
    x_limits: tuple[float, float] | None = None,
    y_limits: tuple[float, float] | None = None,
) -> None:
    display_values = values.copy()
    if x_limits is not None:
        display_values = np.where(
            (x_lag >= x_limits[0]) & (x_lag <= x_limits[1]),
            display_values,
            np.nan,
        )
    if y_limits is not None:
        display_values = np.where(
            (y_lag >= y_limits[0]) & (y_lag <= y_limits[1]),
            display_values,
            np.nan,
        )

    vmin, vmax = value_limits or finite_limits(display_values, symmetric=symmetric)
    display_values = np.where(
        np.isfinite(display_values),
        np.clip(display_values, vmin, vmax),
        np.nan,
    )
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)
    facecolors = cmap(norm(display_values))

    ax.plot_surface(
        x_lag,
        y_lag,
        display_values,
        facecolors=facecolors,
        rstride=1,
        cstride=1,
        linewidth=0.0,
        antialiased=True,
        shade=False,
    )
    ax.set_zlim(vmin, vmax)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(label, labelpad=8)
    ax.view_init(elev=22.0, azim=-64.0)
    if x_limits is not None:
        ax.set_xlim(*x_limits)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.tick_params(pad=1)
    ax.grid(False)

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(display_values[np.isfinite(display_values)])
    cbar = ax.figure.colorbar(mappable, ax=ax, fraction=0.046, pad=0.08)
    cbar.set_label(label)


def render_surface_pair(
    case: dict[str, np.ndarray],
    output_path: Path,
    fields,
    coordinates: str,
    solid_material_coordinates: bool = False,
) -> Path:
    fig = plt.figure(figsize=(9.4, 4.4), constrained_layout=True)
    x_key = "x_lag" if coordinates == "lagrangian" else "X"
    y_key = "y_lag" if coordinates == "lagrangian" else "Y"
    for i, (field, label, cmap, symmetric) in enumerate(fields, start=1):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        material_coordinates = (
            solid_material_coordinates and uses_solid_material_coordinates(case, field)
        )
        x_label, y_label = paper_axis_labels(field, coordinates)
        if coordinates == "paper":
            x_values, y_values, z_values = interpolate_lag_window(
                case,
                field,
                coordinates="material" if material_coordinates else "lagrangian",
            )
        else:
            x_values = case["x_mat"] if material_coordinates and coordinates == "lagrangian" else case[x_key]
            y_values = case["y_mat"] if material_coordinates and coordinates == "lagrangian" else case[y_key]
            z_values = case[field]
        draw_surface_panel(
            ax,
            x_values,
            y_values,
            z_values,
            label,
            cmap,
            symmetric=symmetric,
            x_label=x_label,
            y_label=y_label,
            value_limits=case.get(f"{field}_limits"),
            x_limits=None,
            y_limits=None,
        )
    save_figure(fig, output_path)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--plot-kind",
        choices=["maps", "surfaces", "all"],
        default="all",
        help="Choose flat maps, paper-style 3D surfaces, or both.",
    )
    parser.add_argument(
        "--surface-coords",
        choices=["paper", "cartesian", "lagrangian"],
        default="paper",
        help="Coordinate frame for 3D surfaces. Paper resamples into the cropped x_Lag/y_Lag window used for Case 4.3.",
    )
    parser.add_argument(
        "--surface-zero-padding",
        action="store_true",
        help="Include the paper-style zero-valued inactive material sheet in 3D surfaces.",
    )
    parser.add_argument(
        "--solid-material-coords",
        action="store_true",
        help="Plot solid stresses against advected material-coordinate labels instead of the paper comparison coordinates.",
    )
    args = parser.parse_args()

    root = repo_root()
    csv_path = args.csv or default_csv(root)
    output_dir = args.output_dir or root / "data" / "plots" / "fluid_solid"
    output_dir.mkdir(parents=True, exist_ok=True)
    case = load_case(csv_path)

    surface_suffix = "" if args.surface_zero_padding or args.surface_coords == "paper" else "_surface"
    normal_fields = [
        (f"vn_fluid{surface_suffix}", r"$u_n$", "viridis", False),
        (f"sigma_nn_solid{surface_suffix}", r"$\sigma_{nn}$", "coolwarm", True),
    ]
    stress_fields = [
        (f"sigma_ss_solid{surface_suffix}", r"$\sigma_{ss}$", "coolwarm", True),
        (f"sigma_sn_solid{surface_suffix}", r"$\sigma_{sn}$", "coolwarm", True),
    ]

    if args.plot_kind in {"maps", "all"}:
        render_pair(
            case,
            output_dir / "case4_3_rotated_water_solid_rgfm_normal_fields.png",
            [
                ("vn_fluid", r"Fluid normal velocity, $u_n$", "viridis", False),
                ("sigma_nn_solid", r"Solid normal stress, $\sigma_{nn}$", "coolwarm", True),
            ],
        )
        render_pair(
            case,
            output_dir / "case4_3_rotated_water_solid_rgfm_stress_fields.png",
            [
                ("sigma_ss_solid", r"Solid tangential stress, $\sigma_{ss}$", "coolwarm", True),
                ("sigma_sn_solid", r"Solid shear stress, $\sigma_{sn}$", "coolwarm", True),
            ],
        )

    if args.plot_kind in {"surfaces", "all"}:
        coord_suffix = "" if args.surface_coords == "paper" else f"_{args.surface_coords}"
        render_surface_pair(
            case,
            output_dir / f"case4_3_rotated_water_solid_rgfm_normal{coord_suffix}_surfaces.png",
            normal_fields,
            args.surface_coords,
            solid_material_coordinates=args.solid_material_coords,
        )
        render_surface_pair(
            case,
            output_dir / f"case4_3_rotated_water_solid_rgfm_stress{coord_suffix}_surfaces.png",
            stress_fields,
            args.surface_coords,
            solid_material_coordinates=args.solid_material_coords,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
