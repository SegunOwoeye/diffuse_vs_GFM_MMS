# [0] Import libraries
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from plot_style import (
    configure_profile_axis,
    field_title,
    field_ylabel,
    plot_profile,
    sort_by_resolution,
)
from exact_reference import (
    load_optional_exact_reference_for_context,
)


# [1] Load solution from CSV written by C++
def load_solution_csv(filename):
    data_path = Path("data/csv") / filename

    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found")

    df = pd.read_csv(data_path)

    dim = 0
    while f"x{dim}" in df.columns:
        dim += 1

    if dim not in (2, 3):
        raise ValueError(f"plot_multid.py only supports 2D or 3D CSV files, found dimension={dim}")

    coords = [df[f"x{d}"].to_numpy() for d in range(dim)]
    velocity = [df[f"u{d}"].to_numpy() for d in range(dim)]

    rho = df["rho"].to_numpy()
    p = df["p"].to_numpy()
    e = df["e"].to_numpy()
    extra_fields = {
        column: df[column].to_numpy()
        for column in df.columns
        if (
            column.startswith("phi")
            or column.startswith("alpha")
            or column.startswith("rho_mat")
            or column.startswith("mass")
            or column == "mat"
        )
    }

    return dim, coords, velocity, rho, p, e, extra_fields


# [2] Build structured 2D grid from scattered cell-centre data
def build_grid(x, y, field):
    df = pd.DataFrame({"x": x, "y": y, "f": field})

    grid = (
        df.pivot(index="y", columns="x", values="f")
        .sort_index()
        .sort_index(axis=1)
        .to_numpy()
    )

    x_unique = np.sort(np.unique(x))
    y_unique = np.sort(np.unique(y))

    X, Y = np.meshgrid(x_unique, y_unique)

    return X, Y, grid


# [3] Compute radial profile in arbitrary dimension
def compute_radial_profile(coords, field, center=None, nbins=100):
    dim = len(coords)

    if center is None:
        center = [0.5 * (coords[d].min() + coords[d].max()) for d in range(dim)]

    r2 = np.zeros_like(field, dtype=float)

    for d in range(dim):
        r2 += (coords[d] - center[d]) ** 2

    r = np.sqrt(r2)

    bins = np.linspace(0.0, r.max(), nbins)
    centers = 0.5 * (bins[:-1] + bins[1:])
    avg = np.full_like(centers, np.nan, dtype=float)

    for i in range(len(centers)):
        mask = (r >= bins[i]) & (r < bins[i + 1])

        if np.any(mask):
            avg[i] = np.mean(field[mask])

    return centers, avg


def compute_radial_velocity(coords, velocity, center=None):
    dim = len(coords)

    if center is None:
        center = [0.5 * (coords[d].min() + coords[d].max()) for d in range(dim)]

    r2 = np.zeros_like(velocity[0], dtype=float)
    offsets = []

    for d in range(dim):
        offset = coords[d] - center[d]
        offsets.append(offset)
        r2 += offset ** 2

    r = np.sqrt(r2)
    radial_velocity = np.zeros_like(velocity[0], dtype=float)

    nonzero = r > 0.0
    for d in range(dim):
        radial_velocity[nonzero] += velocity[d][nonzero] * offsets[d][nonzero] / r[nonzero]

    return radial_velocity


# [4] Extract mid-plane slice from 3D data
def extract_midplane_slice(coords, field, axis=2):
    if len(coords) != 3:
        raise ValueError("extract_midplane_slice expects 3D coordinates")

    slice_coord = coords[axis]
    unique_vals = np.sort(np.unique(slice_coord))
    mid_index = len(unique_vals) // 2
    mid_value = unique_vals[mid_index]

    mask = np.isclose(slice_coord, mid_value)

    slice_coords = [coords[d][mask] for d in range(3) if d != axis]
    slice_field = field[mask]

    return slice_coords, slice_field, mid_value


# [5] Build label from filename
def build_label_from_filename(fname):
    name = Path(fname).name.lower()

    if "exact" in name:
        return "Exact"

    n_tokens = []
    for part in name.replace(".csv", "").split("_"):
        if part.startswith("n") and len(part) > 1 and part[1:].isdigit():
            n_tokens.append(part[1:])

    if n_tokens:
        return "N=" + "x".join(n_tokens)

    return Path(fname).stem


def time_from_filename(fname):
    match = re.search(r"_t([0-9peEm+\-]+)_N", Path(fname).name)
    if match is None:
        return None

    text = match.group(1)
    tagged = re.fullmatch(r"([0-9]+)p([0-9]+)e([pm+\-])([0-9]+)", text)

    if tagged is not None:
        integer, fraction, sign_token, exponent = tagged.groups()
        sign = "-" if sign_token in {"m", "-"} else "+"
        return float(f"{integer}.{fraction}e{sign}{exponent}")

    text = text.replace("m", "-").replace("p", ".")
    return float(text)


def format_time_label(time_value):
    exponent = int(np.floor(np.log10(abs(time_value)))) if time_value != 0.0 else 0
    mantissa = time_value / (10.0 ** exponent)
    return rf"$t = {mantissa:.3f}\times 10^{{{exponent}}}$"


# [6] Build output name from folder path
def build_output_name(folder_path: str):
    p = Path(folder_path)
    parts = p.parts[-2:]
    return "_".join(parts)


def time_tagged_csv_files(folder_path):
    files = []

    for csv_path in sorted(folder_path.glob("*.csv")):
        if "_N" not in csv_path.name:
            continue

        time_value = time_from_filename(csv_path.name)
        if time_value is not None:
            files.append((time_value, csv_path))

    return sorted(files, key=lambda item: item[0])


# [7] Plot 3D wireframe surfaces from 2D slice grids
def remove_if_exists(path):
    if path is not None and path.exists():
        path.unlink()


def should_make_schlieren(save_path, force_schlieren=False):
    if force_schlieren:
        return True

    if save_path is None:
        return False

    path_text = str(save_path).lower()
    return "bubble_collapse" in path_text or "helium_bubble" in path_text


def should_mirror_bubble_half_domain(save_path):
    if save_path is None:
        return False

    path_text = str(save_path).lower()
    return "bubble_collapse" in path_text or "helium_bubble" in path_text


def remove_bubble_collapse_auxiliary_plots(save_path, include_driver_outputs=False):
    if save_path is None:
        return

    suffixes = [
        "_raw_density_gradient",
        "_same_material",
        "_raw_interface",
        "_interface_overlay",
        "_material0_only",
        "_material1_only",
        "_alpha0_only",
        "_alpha1_only",
    ]

    for suffix in suffixes:
        remove_if_exists(save_path.with_name(f"{save_path.stem}{suffix}{save_path.suffix}"))

    if include_driver_outputs:
        for suffix in ["_field_maps", "_maps", "_maps_3d", "_3d"]:
            remove_if_exists(save_path.with_name(f"{save_path.stem}{suffix}{save_path.suffix}"))


def mirror_grid_across_y0(X, Y, field_grid):
    y_unique = Y[:, 0]

    if y_unique.size == 0 or float(np.nanmin(y_unique)) < 0.0:
        return X, Y, field_grid

    x_unique = X[0, :]
    mirror_rows = np.where(y_unique > 0.0)[0][::-1]
    y_full = np.concatenate((-y_unique[mirror_rows], y_unique))
    field_full = np.vstack((field_grid[mirror_rows, :], field_grid))

    X_full, Y_full = np.meshgrid(x_unique, y_full)
    return X_full, Y_full, field_full


def fedkiw_bubble_reference_circle_display(X, Y, center=(175.0, 0.0), radius=25.0):
    x_unique = X[0, :]
    y_unique = Y[:, 0]
    xlim = infer_domain_limits(x_unique)
    ylim = infer_domain_limits(y_unique)

    theta = np.linspace(0.0, 2.0 * np.pi, 720)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)

    x = (x - xlim[0]) / (xlim[1] - xlim[0])
    y = (y - ylim[0]) / (ylim[1] - ylim[0])

    # The Fedkiw/Haas-Sturtevant presentation rotates the reflected half-domain
    # 90 degrees clockwise. These are image-axis fractions after that rotation.
    return y, 1.0 - x


def suppress_current_interface_band(schlieren, phi_grid, dx, dy, band_cells=1.5):
    if phi_grid is None:
        return schlieren

    band_width = band_cells * min(abs(dx), abs(dy))
    band = np.abs(phi_grid) <= band_width

    if not np.any(band):
        return schlieren

    outside = schlieren[~band]
    if outside.size == 0:
        return schlieren

    lifted = schlieren.copy()
    black_cutoff = float(np.percentile(outside, 18.0))
    replacement = float(np.percentile(outside, 72.0))
    bold_interface = band & (schlieren <= black_cutoff)
    lifted[bold_interface] = np.maximum(lifted[bold_interface], replacement)

    return lifted


def same_material_density_gradient(field_grid, material_grid, dx, dy):
    gx = np.zeros_like(field_grid)
    gy = np.zeros_like(field_grid)

    same_left = material_grid[:, 1:-1] == material_grid[:, :-2]
    same_right = material_grid[:, 1:-1] == material_grid[:, 2:]
    both_x = same_left & same_right
    right_x = ~same_left & same_right
    left_x = same_left & ~same_right

    gx_mid = gx[:, 1:-1]
    gx_mid[both_x] = (field_grid[:, 2:][both_x] - field_grid[:, :-2][both_x]) / (2.0 * dx)
    gx_mid[right_x] = (field_grid[:, 2:][right_x] - field_grid[:, 1:-1][right_x]) / dx
    gx_mid[left_x] = (field_grid[:, 1:-1][left_x] - field_grid[:, :-2][left_x]) / dx

    if field_grid.shape[1] > 1:
        same = material_grid[:, 0] == material_grid[:, 1]
        gx_left = gx[:, 0]
        gx_left[same] = (field_grid[:, 1][same] - field_grid[:, 0][same]) / dx
        same = material_grid[:, -1] == material_grid[:, -2]
        gx_right = gx[:, -1]
        gx_right[same] = (field_grid[:, -1][same] - field_grid[:, -2][same]) / dx

    same_down = material_grid[1:-1, :] == material_grid[:-2, :]
    same_up = material_grid[1:-1, :] == material_grid[2:, :]
    both_y = same_down & same_up
    up_y = ~same_down & same_up
    down_y = same_down & ~same_up

    gy_mid = gy[1:-1, :]
    gy_mid[both_y] = (field_grid[2:, :][both_y] - field_grid[:-2, :][both_y]) / (2.0 * dy)
    gy_mid[up_y] = (field_grid[2:, :][up_y] - field_grid[1:-1, :][up_y]) / dy
    gy_mid[down_y] = (field_grid[1:-1, :][down_y] - field_grid[:-2, :][down_y]) / dy

    if field_grid.shape[0] > 1:
        same = material_grid[0, :] == material_grid[1, :]
        gy_bottom = gy[0, :]
        gy_bottom[same] = (field_grid[1, :][same] - field_grid[0, :][same]) / dy
        same = material_grid[-1, :] == material_grid[-2, :]
        gy_top = gy[-1, :]
        gy_top[same] = (field_grid[-1, :][same] - field_grid[-2, :][same]) / dy

    return gx, gy


def fedkiw_mock_schlieren(rho_grid, grad_mag):
    rho_safe = np.maximum(rho_grid, 1.0e-30)
    exponent = -20.0 * grad_mag / (1000.0 * np.sqrt(rho_safe))
    return np.exp(np.clip(exponent, -745.0, 0.0))


def is_dim_plot_path(save_path):
    if save_path is None:
        return False

    path_text = str(save_path).lower()
    return "/dim/" in path_text or "\\dim\\" in path_text or "dim_" in path_text


def enhance_dim_schlieren_contrast(schlieren, save_path):
    if not is_dim_plot_path(save_path):
        return schlieren

    return np.clip(schlieren, 0.0, 1.0) ** 3.0


def rotate_bubble_display(field):
    return np.flipud(np.rot90(field, k=3))


def contour_level_is_present(field, level):
    if field is None:
        return False

    finite = field[np.isfinite(field)]
    if finite.size == 0:
        return False

    return float(np.min(finite)) <= level <= float(np.max(finite))


def save_rotated_bubble_schlieren(
    display,
    X,
    Y,
    save_path,
    draw_reference_circle=True,
    contour_display=None,
    contour_level=None,
    contour_color="tab:red",
    interpolation="bilinear"
):
    height, width = display.shape[:2]
    aspect = height / max(width, 1)

    fig, ax = plt.subplots(figsize=(4.0, 4.0 * aspect), constrained_layout=True)
    ax.imshow(
        display,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        origin="lower",
        interpolation=interpolation,
    )

    if contour_display is not None and contour_level is not None:
        if contour_level_is_present(contour_display, contour_level):
            ax.contour(
                contour_display,
                levels=[contour_level],
                colors=contour_color,
                linewidths=0.75,
                origin="lower",
            )

    if draw_reference_circle:
        cx, cy = fedkiw_bubble_reference_circle_display(X, Y)
        ax.plot(
            cx * (width - 1),
            cy * (height - 1),
            color="black",
            linewidth=0.7,
        )

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("black")

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"Saved Schlieren figure to {save_path}")
        plt.close()
    else:
        plt.show()


def select_highest_resolution_2d(xs, fields_list, velocities_list, labels):
    items = list(zip(xs, fields_list, velocities_list, labels))
    if not items:
        raise ValueError("No 2D solution data available")

    return sort_by_resolution(items)[0]


def select_highest_resolution_3d(xs, fields_list, velocities_list, labels):
    items = list(zip(xs, fields_list, velocities_list, labels))
    if not items:
        raise ValueError("No 3D solution data available")

    return sort_by_resolution(items)[0]


def plot_wireframe_surfaces(X, Y, rho_grid, p_grid, save_path=None):
    ny, nx = rho_grid.shape
    rstride = max(1, ny // 45)
    cstride = max(1, nx // 45)
    elev = 26

    fig = plt.figure(figsize=(10.8, 4.3))
    surface_specs = [
        ("rho", rho_grid, 1, -128),
        ("p", p_grid, 2, -52),
    ]

    for field_key, grid, position, azim in surface_specs:
        ax = fig.add_subplot(1, 2, position, projection="3d")
        ax.plot_wireframe(
            X,
            Y,
            grid,
            rstride=rstride,
            cstride=cstride,
            color="black",
            linewidth=0.35,
        )
        ax.set_title("")
        ax.set_xlabel(r"$x$", labelpad=6)
        ax.set_ylabel(r"$y$", labelpad=6)
        ax.set_zlabel(field_ylabel(field_key), labelpad=4)
        ax.view_init(elev=elev, azim=azim)
        ax.set_proj_type("ortho")
        try:
            ax.set_box_aspect([1.28, 1.0, 0.52], zoom=1.10)
        except TypeError:
            ax.set_box_aspect([1.28, 1.0, 0.52])
        ax.margins(x=0.0, y=0.0, z=0.02)
        ax.tick_params(axis="both", which="major", labelsize=8, pad=1)
        ax.zaxis.set_tick_params(labelsize=8, pad=2)
        ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))

    fig.subplots_adjust(left=0.08, right=0.91, bottom=0.06, top=0.90, wspace=-0.04)

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"Saved 3D figure to {save_path}")
        plt.close()
    else:
        plt.show()


# [8] Plot 2D diagnostics
def choose_common_slice_value_2d(xs, axis):
    coord_index = 1 if axis == "x" else 0
    unique_values = [
        np.sort(np.unique(coords[coord_index]))
        for coords in xs
    ]

    reference_values = min(unique_values, key=len)

    return float(reference_values[len(reference_values) // 2])


def infer_domain_limits(values):
    unique_values = np.sort(np.unique(values))

    if len(unique_values) <= 1:
        return float(unique_values[0]), float(unique_values[0])

    dx_lo = unique_values[1] - unique_values[0]
    dx_hi = unique_values[-1] - unique_values[-2]

    return (
        float(unique_values[0] - 0.5 * dx_lo),
        float(unique_values[-1] + 0.5 * dx_hi),
    )


def sample_axis_slice_2d(x, y, field, axis, slice_value):
    df = pd.DataFrame({"x": x, "y": y, "f": field})
    grid = (
        df.pivot(index="y", columns="x", values="f")
        .sort_index()
        .sort_index(axis=1)
    )

    x_unique = grid.columns.to_numpy(dtype=float)
    y_unique = grid.index.to_numpy(dtype=float)
    values = grid.to_numpy(dtype=float)

    if axis == "x":
        if slice_value < y_unique[0] or slice_value > y_unique[-1]:
            raise ValueError("x-slice y value is outside the grid")

        line = np.array([
            np.interp(slice_value, y_unique, values[:, j])
            for j in range(len(x_unique))
        ])

        return x_unique, line, slice_value

    if axis == "y":
        if slice_value < x_unique[0] or slice_value > x_unique[-1]:
            raise ValueError("y-slice x value is outside the grid")

        line = np.array([
            np.interp(slice_value, x_unique, values[i, :])
            for i in range(len(y_unique))
        ])

        return y_unique, line, slice_value

    raise ValueError("axis must be 'x' or 'y'")


def extract_axis_slice_2d(x, y, field, axis, slice_value=None):
    if axis == "x":
        unique_y = np.sort(np.unique(y))
        if slice_value is None:
            slice_value = unique_y[len(unique_y) // 2]

        mask = np.isclose(y, slice_value)

        if not np.any(mask):
            return sample_axis_slice_2d(x, y, field, axis, slice_value)

        order = np.argsort(x[mask])
        return x[mask][order], field[mask][order], slice_value

    if axis == "y":
        unique_x = np.sort(np.unique(x))
        if slice_value is None:
            slice_value = unique_x[len(unique_x) // 2]

        mask = np.isclose(x, slice_value)

        if not np.any(mask):
            return sample_axis_slice_2d(x, y, field, axis, slice_value)

        order = np.argsort(y[mask])
        return y[mask][order], field[mask][order], slice_value

    raise ValueError("axis must be 'x' or 'y'")


def zero_roundoff_line(values, atol=1.0e-10):
    values = np.asarray(values).copy()

    if values.size == 0:
        return values

    if np.nanmax(np.abs(values)) <= atol:
        values[:] = 0.0

    return values


def transverse_range_by_x(x, y, field):
    df = pd.DataFrame({"x": x, "y": y, "f": field})
    grid = (
        df.pivot(index="y", columns="x", values="f")
        .sort_index()
        .sort_index(axis=1)
    )

    x_unique = grid.columns.to_numpy(dtype=float)
    values = grid.to_numpy()
    transverse_range = np.nanmax(values, axis=0) - np.nanmin(values, axis=0)

    return x_unique, zero_roundoff_line(transverse_range)


def relative_transverse_range_by_x(x, y, field, floor=1.0e-12):
    x_unique, transverse_range = transverse_range_by_x(x, y, field)
    scale = max(float(np.nanmax(np.abs(field))), floor)
    relative_range = transverse_range / scale

    return x_unique, zero_roundoff_line(relative_range, atol=floor)


def transverse_absmax_by_x(x, y, field):
    df = pd.DataFrame({"x": x, "y": y, "f": field})
    grid = (
        df.pivot(index="y", columns="x", values="f")
        .sort_index()
        .sort_index(axis=1)
    )

    x_unique = grid.columns.to_numpy(dtype=float)
    values = grid.to_numpy()
    absmax = np.nanmax(np.abs(values), axis=0)

    return x_unique, zero_roundoff_line(absmax)


def relative_transverse_absmax_by_x(x, y, field, reference, floor=1.0e-12):
    x_unique, absmax = transverse_absmax_by_x(x, y, field)
    scale = max(float(np.nanmax(np.abs(reference))), floor)
    relative_absmax = absmax / scale

    return x_unique, zero_roundoff_line(relative_absmax, atol=floor)


def sorted_prefixed_fields(extras, prefix):
    return sorted(
        [name for name in extras if name.startswith(prefix)],
        key=lambda name: (len(name), name)
    )


def material_grid_from_extras(x, y, extras):
    if "mat" in extras:
        _, _, material_grid = build_grid(x, y, extras["mat"])
        return material_grid

    alpha_names = sorted_prefixed_fields(extras, "alpha")
    if not alpha_names:
        return None

    alpha_grids = []
    for name in alpha_names:
        _, _, alpha_grid = build_grid(x, y, extras[name])
        alpha_grids.append(alpha_grid)

    return np.argmax(np.stack(alpha_grids, axis=0), axis=0)


def bubble_interface_grid_from_extras(x, y, extras):
    phi_names = sorted_prefixed_fields(extras, "phi")
    if phi_names:
        _, _, phi_grid = build_grid(x, y, extras[phi_names[0]])
        return phi_grid, 0.0, phi_names[0]

    alpha_names = sorted_prefixed_fields(extras, "alpha")
    if len(alpha_names) > 1:
        _, _, alpha_grid = build_grid(x, y, extras[alpha_names[1]])
        return alpha_grid, 0.5, alpha_names[1]

    if "mat" in extras:
        _, _, material_grid = build_grid(x, y, extras["mat"])
        return material_grid.astype(float), 0.5, "mat"

    return None, None, None


def material_indicator_grids_from_extras(x, y, extras):
    indicators = []

    if "mat" in extras:
        _, _, material_grid = build_grid(x, y, extras["mat"])
        material_ids = [
            int(value)
            for value in sorted(np.unique(material_grid[np.isfinite(material_grid)]))
        ]
        for material_id in material_ids:
            indicators.append((
                f"material{material_id}",
                (material_grid == material_id).astype(float),
                0.5,
            ))
        return indicators

    alpha_names = sorted_prefixed_fields(extras, "alpha")
    for name in alpha_names:
        _, _, alpha_grid = build_grid(x, y, extras[name])
        indicators.append((name, alpha_grid, 0.5))

    return indicators


def plot_exact_reference_on_axes(axes, field_keys, exact_fields):
    if not exact_fields:
        return

    for ax, field_key in zip(axes, field_keys):
        if field_key in exact_fields:
            exact_x, exact_y = exact_fields[field_key]
            plot_profile(ax, exact_x, exact_y, "Exact", index=0)


def exact_field_for_plot(field_key, exact_fields):
    if not exact_fields:
        return None

    aliases = {
        "u_radial": ("u_radial", "u0", "u"),
        "u0": ("u0", "u"),
        "e": ("e",),
    }

    for exact_key in aliases.get(field_key, (field_key,)):
        if exact_key in exact_fields:
            return exact_fields[exact_key]

    return None


def plot_2d_solution_slice(
    xs,
    fields_list,
    velocities_list,
    labels,
    axis="x",
    slice_value=None,
    save_path=None,
    show_titles=True,
    exact_fields=None,
):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    axis_label = axis

    if slice_value is None:
        slice_value = choose_common_slice_value_2d(xs, axis)

    coord_limits = []

    plotted_items = sort_by_resolution(list(zip(xs, fields_list, velocities_list, labels)))

    for index, (coords, fields, velocity, label) in enumerate(plotted_items):
        x, y = coords
        rho, p, e = fields
        u0, u1 = velocity

        line_coord, rho_line, fixed_value = extract_axis_slice_2d(x, y, rho, axis, slice_value)
        _, u0_line, _ = extract_axis_slice_2d(x, y, u0, axis, slice_value)
        _, u1_line, _ = extract_axis_slice_2d(x, y, u1, axis, slice_value)
        _, p_line, _ = extract_axis_slice_2d(x, y, p, axis, slice_value)
        _, e_line, _ = extract_axis_slice_2d(x, y, e, axis, slice_value)

        u0_line = zero_roundoff_line(u0_line)
        u1_line = zero_roundoff_line(u1_line)
        coord_limits.append(infer_domain_limits(line_coord))

        plot_profile(axes[0, 0], line_coord, rho_line, label, index=index)
        plot_profile(axes[0, 1], line_coord, u0_line, label, index=index)
        plot_profile(axes[0, 2], line_coord, u1_line, label, index=index)
        plot_profile(axes[1, 0], line_coord, p_line, label, index=index)
        plot_profile(axes[1, 1], line_coord, e_line, label, index=index)

    field_keys = ["rho", "u0", "u1", "p", "e"]
    slice_xlabel = r"$x$" if axis == "x" else r"$y$"
    plot_axes = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]

    if axis == "x":
        plot_exact_reference_on_axes(plot_axes, field_keys, exact_fields)

    for ax, field_key in zip(plot_axes, field_keys):
        configure_profile_axis(ax, field_key, x_label=slice_xlabel)
        if not show_titles:
            ax.set_title("")

    axes[1, 2].axis("off")

    for ax in plot_axes:
        if ax.has_data():
            if coord_limits:
                xmin = min(limit[0] for limit in coord_limits)
                xmax = max(limit[1] for limit in coord_limits)
                ax.set_xlim(xmin, xmax)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"Saved {axis}-slice figure to {save_path}")
        plt.close()
    else:
        plt.show()


def relative_deviation_from_mean(values, floor=1.0e-12):
    values = np.asarray(values)
    mean_value = float(np.nanmean(values))
    scale = max(abs(mean_value), floor)

    return zero_roundoff_line((values - mean_value) / scale, atol=floor)


def plot_2d_y_deviation_slice(
    xs,
    fields_list,
    velocities_list,
    labels,
    slice_value=None,
    save_path=None,
    show_titles=True,
):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)

    if slice_value is None:
        slice_value = choose_common_slice_value_2d(xs, "y")

    coord_limits = []

    plotted_items = sort_by_resolution(list(zip(xs, fields_list, velocities_list, labels)))

    for index, (coords, fields, velocity, label) in enumerate(plotted_items):
        x, y = coords
        rho, p, e = fields
        u0, u1 = velocity

        line_coord, rho_line, fixed_value = extract_axis_slice_2d(x, y, rho, "y", slice_value)
        _, u0_line, _ = extract_axis_slice_2d(x, y, u0, "y", slice_value)
        _, u1_line, _ = extract_axis_slice_2d(x, y, u1, "y", slice_value)
        _, p_line, _ = extract_axis_slice_2d(x, y, p, "y", slice_value)
        _, e_line, _ = extract_axis_slice_2d(x, y, e, "y", slice_value)

        coord_limits.append(infer_domain_limits(line_coord))

        plot_profile(axes[0, 0], line_coord, relative_deviation_from_mean(rho_line), label, index=index)
        plot_profile(axes[0, 1], line_coord, relative_deviation_from_mean(u0_line), label, index=index)
        plot_profile(axes[0, 2], line_coord, zero_roundoff_line(u1_line), label, index=index)
        plot_profile(axes[1, 0], line_coord, relative_deviation_from_mean(p_line), label, index=index)
        plot_profile(axes[1, 1], line_coord, relative_deviation_from_mean(e_line), label, index=index)

    titles = [
        "Relative density deviation",
        "Relative velocity deviation",
        "Transverse velocity",
        "Relative pressure deviation",
        "Relative internal-energy deviation",
    ]
    ylabels = [
        r"Relative density deviation",
        r"Relative velocity deviation",
        field_ylabel("u1"),
        r"Relative pressure deviation",
        r"Relative internal-energy deviation",
    ]
    plot_axes = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]

    for ax, title_text, ylabel in zip(plot_axes, titles, ylabels):
        configure_profile_axis(ax, None, x_label=r"$y$")
        ax.set_title(title_text if show_titles else "")
        ax.set_ylabel(ylabel)

    axes[1, 2].axis("off")

    for ax in plot_axes:
        if ax.has_data():
            if coord_limits:
                ymin = min(limit[0] for limit in coord_limits)
                ymax = max(limit[1] for limit in coord_limits)
                ax.set_xlim(ymin, ymax)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"Saved y-deviation slice figure to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_2d_transverse_diagnostics(
    xs,
    fields_list,
    velocities_list,
    labels,
    slice_value=None,
    save_path=None,
    show_titles=True,
    exact_fields=None,
):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex="row")

    if slice_value is None:
        slice_value = choose_common_slice_value_2d(xs, "x")

    plotted_items = sort_by_resolution(list(zip(xs, fields_list, velocities_list, labels)))

    for index, (coords, fields, velocity, label) in enumerate(plotted_items):
        x, y = coords
        rho, p, e = fields
        u0, u1 = velocity

        x_line, rho_x, y_value = extract_axis_slice_2d(x, y, rho, "x", slice_value)
        _, p_x, _ = extract_axis_slice_2d(x, y, p, "x", slice_value)
        _, e_x, _ = extract_axis_slice_2d(x, y, e, "x", slice_value)
        _, u0_x, _ = extract_axis_slice_2d(x, y, u0, "x", slice_value)

        x_tr, rho_tr = relative_transverse_range_by_x(x, y, rho)
        _, p_tr = relative_transverse_range_by_x(x, y, p)
        _, u0_tr = relative_transverse_range_by_x(x, y, u0)
        _, u1_abs = relative_transverse_absmax_by_x(x, y, u1, u0)

        plot_profile(axes[0, 0], x_line, rho_x, label, index=index)
        plot_profile(axes[0, 1], x_line, u0_x, label, index=index)
        plot_profile(axes[0, 2], x_line, p_x, label, index=index)
        plot_profile(axes[0, 3], x_line, e_x, label, index=index)

        plot_profile(axes[1, 0], x_tr, rho_tr, label, index=index)
        plot_profile(axes[1, 1], x_tr, u0_tr, label, index=index)
        plot_profile(axes[1, 2], x_tr, p_tr, label, index=index)
        plot_profile(axes[1, 3], x_tr, u1_abs, label, index=index)

    top_keys = ["rho", "u0", "p", "e"]
    plot_exact_reference_on_axes(axes[0, :], top_keys, exact_fields)

    for ax, field_key in zip(axes[0, :], top_keys):
        configure_profile_axis(ax, field_key, x_label=r"$x$")
        if not show_titles:
            ax.set_title("")

    axes[1, 0].set_title("Relative density transverse range" if show_titles else "")
    axes[1, 1].set_title("Relative u0 transverse range" if show_titles else "")
    axes[1, 2].set_title("Relative pressure transverse range" if show_titles else "")
    axes[1, 3].set_title("max |u1| / max |u0|" if show_titles else "")

    for ax in axes[1, :]:
        configure_profile_axis(ax, None, x_label=r"$x$")

    for ax in axes[1, :]:
        ax.set_yscale("symlog", linthresh=1.0e-12)
        ax.set_ylabel("relative error")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"Saved transverse diagnostic figure to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_2d_diagnostics(xs, fields_list, labels, save_path=None):
    plot_2d_diagnostics_maps(xs, fields_list, labels, save_path=save_path)


def plot_radial_diagnostics(
    xs,
    fields_list,
    velocities_list,
    labels,
    save_path=None,
    exact_fields=None,
):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    axes = axes.flatten()
    fourth_key = "e"

    plotted_items = sort_by_resolution(list(zip(xs, fields_list, velocities_list, labels)))

    for index, (coords, fields, velocity, label) in enumerate(plotted_items):
        rho, p, e = fields
        u_radial = compute_radial_velocity(coords, velocity)

        radial_fields = [
            ("rho", rho),
            ("u_radial", u_radial),
            ("p", p),
            (fourth_key, e),
        ]

        for ax, (_, field) in zip(axes, radial_fields):
            r, radial_avg = compute_radial_profile(coords, field)
            plot_profile(ax, r, radial_avg, label, index=index)

    field_keys = ["rho", "u_radial", "p", fourth_key]

    for ax, field_key in zip(axes, field_keys):
        exact_field = exact_field_for_plot(field_key, exact_fields)
        if exact_field is not None:
            exact_x, exact_y = exact_field
            plot_profile(ax, exact_x, exact_y, "Exact", index=0)

    for ax, field_key in zip(axes, field_keys):
        configure_profile_axis(ax, field_key, x_label=r"$r$", show_title=False)
        ax.set_xlim(0.0, 1.0)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved radial profile figure to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_2d_diagnostics_maps(xs, fields_list, labels, save_path=None):
    coords, fields, _, _ = select_highest_resolution_2d(
        xs,
        fields_list,
        [None] * len(xs),
        labels
    )
    x, y = coords
    rho, p = fields[:2]

    X, Y, rho_grid = build_grid(x, y, rho)
    _, _, p_grid = build_grid(x, y, p)

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    fig = plt.figure(figsize=(12, 5), constrained_layout=True)

    ax1 = fig.add_subplot(1, 2, 1)
    im1 = ax1.pcolormesh(X, Y, rho_grid, shading="auto")
    fig.colorbar(im1, ax=ax1)
    ax1.set_title("Density (2D)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)

    ax2 = fig.add_subplot(1, 2, 2)
    im2 = ax2.pcolormesh(X, Y, p_grid, shading="auto")
    fig.colorbar(im2, ax=ax2)
    ax2.set_title("Pressure (2D)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved map figure to {save_path}")
        plt.close()
    else:
        plt.show()

    wireframe_path = (save_path.parent / f"{save_path.stem}_3d.png") if save_path else None
    plot_wireframe_surfaces(X, Y, rho_grid, p_grid, save_path=wireframe_path)


def plot_2d_field_maps(xs, fields_list, velocities_list, labels, save_path=None):
    coords, fields, velocity, label = select_highest_resolution_2d(
        xs,
        fields_list,
        velocities_list,
        labels
    )
    x, y = coords
    rho, p, e = fields
    u0, u1 = velocity

    X, Y, rho_grid = build_grid(x, y, rho)
    _, _, p_grid = build_grid(x, y, p)
    _, _, e_grid = build_grid(x, y, e)
    _, _, u0_grid = build_grid(x, y, u0)
    _, _, u1_grid = build_grid(x, y, u1)

    maps = [
        ("rho", rho_grid),
        ("p", p_grid),
        ("u0", u0_grid),
        ("u1", u1_grid),
        ("e", e_grid),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    axes = axes.ravel()

    for ax, (field_key, values) in zip(axes, maps):
        im = ax.pcolormesh(X, Y, values, shading="auto")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(field_ylabel(field_key))
        ax.set_title(field_title(field_key))
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_aspect("equal", adjustable="box")

    for ax in axes[len(maps):]:
        ax.axis("off")

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"Saved field-map figure to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_2d_schlieren(
    xs,
    fields_list,
    labels,
    save_path=None,
    extras_list=None,
):
    if extras_list is None:
        extras_list = [{} for _ in xs]

    coords, fields, extras, label = sort_by_resolution(
        list(zip(xs, fields_list, extras_list, labels))
    )[0]
    x, y = coords
    rho = fields[0]

    X, Y, rho_grid = build_grid(x, y, rho)

    is_bubble_collapse = should_mirror_bubble_half_domain(save_path)

    material_grid = material_grid_from_extras(x, y, extras) if is_bubble_collapse else None

    if is_bubble_collapse:
        if material_grid is not None:
            _, _, material_grid = mirror_grid_across_y0(X, Y, material_grid)
        X, Y, rho_grid = mirror_grid_across_y0(X, Y, rho_grid)

    x_unique = X[0, :]
    y_unique = Y[:, 0]

    dx = float(np.mean(np.diff(x_unique))) if len(x_unique) > 1 else 1.0
    dy = float(np.mean(np.diff(y_unique))) if len(y_unique) > 1 else 1.0

    if is_bubble_collapse:
        # Bubble-collapse coordinates are stored in mm. The coursework
        # mock-Schlieren variable uses |grad rho| per metre.
        dx_grad = dx * 1.0e-3
        dy_grad = dy * 1.0e-3
    else:
        dx_grad = dx
        dy_grad = dy

    grad_y, grad_x = np.gradient(rho_grid, dy_grad, dx_grad)
    grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)

    if is_bubble_collapse:
        schlieren = fedkiw_mock_schlieren(rho_grid, grad_mag)
        schlieren = enhance_dim_schlieren_contrast(schlieren, save_path)
    else:
        scale = max(float(np.nanmax(grad_mag)), 1.0e-30)
        schlieren = np.exp(-10.0 * grad_mag / scale)

    if is_bubble_collapse:
        # The reference helium-bubble figure reflects the top-half computation
        # and then rotates the image 90 degrees clockwise for presentation.
        display = rotate_bubble_display(schlieren)
        if save_path is not None:
            remove_bubble_collapse_auxiliary_plots(save_path)

        save_rotated_bubble_schlieren(display, X, Y, save_path, draw_reference_circle=True)
        return

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    im = ax.pcolormesh(X, Y, schlieren, shading="auto", cmap="gray", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax)
    ax.set_title(f"Schlieren-style density gradient ({label})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"Saved Schlieren figure to {save_path}")
        plt.close()
    else:
        plt.show()


def load_pressure_contour_frame(csv_path):
    df = pd.read_csv(csv_path)
    required = {"x0", "x1", "p"}

    if not required.issubset(df.columns):
        raise ValueError(f"{csv_path} must contain x0,x1,p columns")

    X, Y, pressure = build_grid(
        df["x0"].to_numpy(),
        df["x1"].to_numpy(),
        df["p"].to_numpy(),
    )

    phi_grid = None
    if "phi0" in df.columns:
        _, _, phi_grid = build_grid(
            df["x0"].to_numpy(),
            df["x1"].to_numpy(),
            df["phi0"].to_numpy(),
        )

    return X, Y, pressure, phi_grid


def pressure_contour_levels(frames, count=14):
    values = np.concatenate([
        np.asarray(frame[3], dtype=float).ravel()
        for frame in frames
    ])
    values = values[np.isfinite(values) & (values > 0.0)]

    if values.size == 0:
        return None

    low = max(float(np.percentile(values, 5.0)), 1.0e-30)
    high = max(float(np.percentile(values, 99.5)), low * 1.001)

    return np.linspace(np.log10(low), np.log10(high), count)


def make_pressure_contour_axes(fig, nframes):
    if nframes == 3:
        axes = [
            fig.add_subplot(2, 4, (1, 2)),
            fig.add_subplot(2, 4, (3, 4)),
            fig.add_subplot(2, 4, (6, 7)),
        ]
        return axes

    rows = int(np.ceil(nframes / 2))
    return [fig.add_subplot(rows, 2, i + 1) for i in range(nframes)]


def plot_pressure_contours_from_times(folder_path, save_path=None):
    time_files = time_tagged_csv_files(folder_path)

    if not time_files:
        raise FileNotFoundError(f"No time-tagged CSV files found in {folder_path}")

    frames = [
        (time_value, *load_pressure_contour_frame(csv_path))
        for time_value, csv_path in time_files
    ]
    contour_levels = pressure_contour_levels(frames)

    fig = plt.figure(figsize=(8.0, 9.2))
    axes = make_pressure_contour_axes(fig, len(frames))
    letters = "abcdefghijklmnopqrstuvwxyz"

    for index, (ax, frame) in enumerate(zip(axes, frames)):
        time_value, X, Y, pressure, phi_grid = frame
        log_pressure = np.log10(np.maximum(pressure, 1.0e-30))

        ax.contour(
            X,
            Y,
            log_pressure,
            levels=contour_levels,
            colors="0.25",
            linewidths=0.45,
        )

        if phi_grid is not None and np.nanmin(phi_grid) <= 0.0 <= np.nanmax(phi_grid):
            ax.contour(
                X,
                Y,
                phi_grid,
                levels=[0.0],
                colors="black",
                linewidths=1.4,
            )
            ax.text(2.0, 3.15, "Interface", ha="center", va="bottom", fontsize=8, fontweight="bold")
            ax.annotate("", xy=(1.4, 2.3), xytext=(1.9, 3.08), arrowprops={"arrowstyle": "-", "lw": 0.8})
            ax.annotate("", xy=(2.6, 2.3), xytext=(2.1, 3.08), arrowprops={"arrowstyle": "-", "lw": 0.8})

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(float(np.nanmin(X)), float(np.nanmax(X)))
        ax.set_ylim(float(np.nanmin(Y)), float(np.nanmax(Y)))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            rf"({letters[index]}) {format_time_label(time_value)}",
            y=-0.18,
            fontsize=11,
        )

        for spine in ax.spines.values():
            spine.set_linewidth(1.1)

    fig.text(0.5, 0.025, "Problem 6: Pressure contours", ha="center", va="bottom", fontsize=12)
    fig.subplots_adjust(left=0.06, right=0.94, top=0.97, bottom=0.10, hspace=0.38, wspace=0.05)

    if save_path is None:
        save_path = folder_path / f"{folder_path.name}_pressure_contours.png"

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved pressure contour figure to {save_path}")
    plt.close(fig)


def plot_schlieren_sequence_from_times(
    folder_path,
    data_root,
    save_path=None,
):
    time_files = time_tagged_csv_files(folder_path)

    if not time_files:
        raise FileNotFoundError(f"No time-tagged CSV files found in {folder_path}")

    output_stem = save_path.stem if save_path is not None else folder_path.name
    for time_value, csv_path in time_files:
        relative_csv = csv_path.relative_to(data_root)
        time_tag = re.search(r"_(t[0-9peEm+\-]+)_N", csv_path.name)
        if time_tag is not None:
            tag = time_tag.group(1)
        else:
            tag = f"t{time_value:.6g}".replace(".", "p")

        frame_path = folder_path / f"{output_stem}_{tag}.png"
        plot_multiple_cpp_solutions(
            [str(relative_csv)],
            save_path=frame_path,
            force_schlieren=True,
            diagnostics=False,
        )


# [9] Plot 3D diagnostics
def plot_3d_diagnostics(
    xs,
    fields_list,
    velocities_list,
    labels,
    save_path=None,
    exact_fields=None,
):
    plot_radial_diagnostics(
        xs,
        fields_list,
        velocities_list,
        labels,
        save_path=save_path,
        exact_fields=exact_fields,
    )
    if save_path is not None:
        remove_if_exists(save_path.parent / f"{save_path.stem}_midplane_maps.png")
        remove_if_exists(save_path.parent / f"{save_path.stem}_3d.png")

    midplane_surface_path = (save_path.parent / f"{save_path.stem}_midplane_3d.png") if save_path else None
    plot_3d_midplane_wireframe_surfaces(
        xs,
        fields_list,
        velocities_list,
        labels,
        save_path=midplane_surface_path,
    )


def plot_3d_midplane_wireframe_surfaces(
    xs,
    fields_list,
    velocities_list,
    labels,
    save_path=None,
    axis=2,
):
    coords, fields, _, label = select_highest_resolution_3d(
        xs,
        fields_list,
        velocities_list,
        labels
    )
    rho, p = fields[:2]

    slice_coords, rho_slice, mid_value = extract_midplane_slice(
        list(coords),
        rho,
        axis=axis,
    )
    _, p_slice, _ = extract_midplane_slice(
        list(coords),
        p,
        axis=axis,
    )

    X, Y, rho_grid = build_grid(slice_coords[0], slice_coords[1], rho_slice)
    _, _, p_grid = build_grid(slice_coords[0], slice_coords[1], p_slice)

    plot_wireframe_surfaces(X, Y, rho_grid, p_grid, save_path=save_path)


# [10] Plot multiple multidimensional C++ outputs
def plot_multiple_cpp_solutions(
    filenames,
    title="Multidimensional Euler Solution",
    save_path=None,
    force_schlieren=False,
    diagnostics=True,
    exact_root=Path("data/exact"),
):
    coords_list = []
    fields_list = []
    velocities_list = []
    extras_list = []
    labels = []
    dims = []

    for fname in filenames:
        dim, coords, velocity, rho, p, e, extras = load_solution_csv(fname)

        dims.append(dim)
        coords_list.append(coords)
        fields_list.append((rho, p, e))
        velocities_list.append(velocity)
        extras_list.append(extras)
        labels.append(build_label_from_filename(fname))

    if len(set(dims)) != 1:
        raise ValueError("All files passed to plot_multid.py must have the same dimension")

    dim = dims[0]
    exact_context = " ".join([str(save_path) if save_path else "", title, *map(str, filenames)])
    exact_fields = load_optional_exact_reference_for_context(
        exact_root,
        exact_context,
        context=title,
    )

    if exact_fields and "explosion" in exact_context.lower():
        exact_fields = dict(exact_fields)
        if "e" not in exact_fields and "entropy" in exact_fields:
            exact_fields["e"] = exact_fields["entropy"]

    if dim == 2:
        xs = [(coords[0], coords[1]) for coords in coords_list]
        is_bubble_collapse_output = should_mirror_bubble_half_domain(save_path)
        suppress_diagnostic_titles = (
            save_path is not None and
            "explosion" in str(save_path).lower()
        )

        if is_bubble_collapse_output:
            remove_if_exists(save_path)
        else:
            plot_radial_diagnostics(
                xs,
                fields_list,
                velocities_list,
                labels,
                save_path=save_path,
                exact_fields=exact_fields,
            )

        x_slice_path = (save_path.parent / f"{save_path.stem}_x_slices.png") if save_path else None
        y_slice_path = (save_path.parent / f"{save_path.stem}_y_slices.png") if save_path else None
        old_y_deviation_path = (save_path.parent / f"{save_path.stem}_y_deviation_slices.png") if save_path else None
        transverse_path = (save_path.parent / f"{save_path.stem}_transverse.png") if save_path else None
        field_map_path = (save_path.parent / f"{save_path.stem}_field_maps.png") if save_path else None
        map_diagnostic_path = (save_path.parent / f"{save_path.stem}_maps.png") if save_path else None
        map_wireframe_path = (save_path.parent / f"{save_path.stem}_maps_3d.png") if save_path else None
        old_wireframe_path = (save_path.parent / f"{save_path.stem}_3d.png") if save_path else None
        schlieren_path = (save_path.parent / f"{save_path.stem}_schlieren.png") if save_path else None
        legacy_slice_path = (save_path.parent / f"{save_path.stem}_slices.png") if save_path else None

        if diagnostics:
            plot_2d_solution_slice(
                xs,
                fields_list,
                velocities_list,
                labels,
                axis="x",
                save_path=x_slice_path,
                show_titles=not suppress_diagnostic_titles,
                exact_fields=exact_fields,
            )
            plot_2d_y_deviation_slice(
                xs,
                fields_list,
                velocities_list,
                labels,
                save_path=y_slice_path,
                show_titles=not suppress_diagnostic_titles,
            )

        remove_if_exists(old_y_deviation_path)
        remove_if_exists(legacy_slice_path)
        remove_if_exists(field_map_path)
        remove_if_exists(map_diagnostic_path)
        remove_if_exists(map_wireframe_path)
        remove_if_exists(old_wireframe_path)

        if diagnostics:
            plot_2d_transverse_diagnostics(
                xs,
                fields_list,
                velocities_list,
                labels,
                save_path=transverse_path,
                show_titles=not suppress_diagnostic_titles,
                exact_fields=exact_fields,
            )
            plot_2d_diagnostics_maps(
                xs,
                fields_list,
                labels,
                save_path=map_diagnostic_path,
            )

        if should_make_schlieren(save_path, force_schlieren):
            plot_2d_schlieren(
                xs,
                fields_list,
                labels,
                save_path=schlieren_path,
                extras_list=extras_list,
            )
        else:
            remove_if_exists(schlieren_path)
        return

    if dim == 3:
        xs = [(coords[0], coords[1], coords[2]) for coords in coords_list]
        plot_3d_diagnostics(
            xs,
            fields_list,
            velocities_list,
            labels,
            save_path=save_path,
            exact_fields=exact_fields,
        )
        return

    raise ValueError(f"Unsupported dimension: {dim}")


# [11] Script entry
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("python src/graphing/plot_multid.py [--schlieren] [--pressure-contours] file.csv")
        print("python src/graphing/plot_multid.py [--schlieren] [--pressure-contours] directory_name")
        print("python src/graphing/plot_multid.py [--schlieren] file1.csv file2.csv")
        raise SystemExit(1)

    force_schlieren = False
    pressure_contours = False
    args = []

    for arg in sys.argv[1:]:
        if arg == "--schlieren":
            force_schlieren = True
        elif arg == "--paper-schlieren":
            force_schlieren = True
        elif arg == "--pressure-contours":
            pressure_contours = True
        else:
            args.append(arg)

    if not args:
        raise SystemExit("No input files or directories provided")

    data_root = Path("data/csv")

    if len(args) == 1:
        path_arg = data_root / args[0]

        if path_arg.is_dir():
            if pressure_contours:
                save_path = path_arg / f"{path_arg.name}_pressure_contours.png"
                plot_pressure_contours_from_times(path_arg, save_path=save_path)
                raise SystemExit(0)

            csv_files = sorted([f.name for f in path_arg.glob("*.csv") if "_N" in f.name])

            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in {path_arg}")

            filenames = [str(Path(args[0]) / f) for f in csv_files]

            output_name = build_output_name(args[0])
            save_path = path_arg / f"{output_name}.png"

            if force_schlieren and time_tagged_csv_files(path_arg):
                plot_schlieren_sequence_from_times(
                    path_arg,
                    data_root,
                    save_path=save_path,
                )
                raise SystemExit(0)

            plot_multiple_cpp_solutions(
                filenames,
                title=output_name,
                save_path=save_path,
                force_schlieren=force_schlieren,
            )
        else:
            plot_multiple_cpp_solutions(
                [args[0]],
                force_schlieren=force_schlieren,
            )

    else:
        plot_multiple_cpp_solutions(args, force_schlieren=force_schlieren)
