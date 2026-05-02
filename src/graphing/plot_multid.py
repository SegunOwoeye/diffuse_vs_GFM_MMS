# [0] Import libraries
from pathlib import Path
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

    return dim, coords, velocity, rho, p, e


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


# [6] Build output name from folder path
def build_output_name(folder_path: str):
    p = Path(folder_path)
    parts = p.parts[-2:]
    return "_".join(parts)


# [7] Plot 3D wireframe surfaces from 2D slice grids
def remove_if_exists(path):
    if path is not None and path.exists():
        path.unlink()


def should_make_schlieren(save_path, force_schlieren=False):
    if force_schlieren:
        return True

    if save_path is None:
        return False

    return "bubble_collapse" in str(save_path).lower()


def plot_wireframe_surfaces(X, Y, rho_grid, p_grid, save_path=None):
    ny, nx = rho_grid.shape
    rstride = max(1, ny // 45)
    cstride = max(1, nx // 45)
    elev = 24
    azim = -125

    fig = plt.figure(figsize=(14, 5), constrained_layout=True)

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_wireframe(X, Y, rho_grid, rstride=rstride, cstride=cstride, linewidth=0.5)
    ax1.set_title("Density")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("rho")
    ax1.view_init(elev=elev, azim=azim)
    ax1.set_box_aspect([1.3, 1.0, 0.45])
    ax1.margins(x=0.0, y=0.0, z=0.02)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot_wireframe(X, Y, p_grid, rstride=rstride, cstride=cstride, linewidth=0.5)
    ax2.set_title("Pressure")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("p")
    ax2.view_init(elev=elev, azim=azim)
    ax2.set_box_aspect([1.3, 1.0, 0.45])
    ax2.margins(x=0.0, y=0.0, z=0.02)

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


def plot_2d_solution_slice(
    xs,
    fields_list,
    velocities_list,
    labels,
    axis="x",
    slice_value=None,
    save_path=None
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

    for ax, field_key in zip(plot_axes, field_keys):
        configure_profile_axis(ax, field_key, x_label=slice_xlabel)

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
    save_path=None
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
        ax.set_title(title_text)
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


def plot_2d_transverse_diagnostics(xs, fields_list, velocities_list, labels, slice_value=None, save_path=None):
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
    for ax, field_key in zip(axes[0, :], top_keys):
        configure_profile_axis(ax, field_key, x_label=r"$x$")

    axes[1, 0].set_title("Relative density transverse range")
    axes[1, 1].set_title("Relative u0 transverse range")
    axes[1, 2].set_title("Relative pressure transverse range")
    axes[1, 3].set_title("max |u1| / max |u0|")

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
    x, y = xs[-1]
    rho, p = fields_list[-1][:2]

    X, Y, rho_grid = build_grid(x, y, rho)
    _, _, p_grid = build_grid(x, y, p)

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    fig = plt.figure(figsize=(12, 10))

    ax1 = fig.add_subplot(2, 2, 1)
    im1 = ax1.pcolormesh(X, Y, rho_grid, shading="auto")
    fig.colorbar(im1, ax=ax1)
    ax1.set_title("Density (2D)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)

    ax2 = fig.add_subplot(2, 2, 2)
    im2 = ax2.pcolormesh(X, Y, p_grid, shading="auto")
    fig.colorbar(im2, ax=ax2)
    ax2.set_title("Pressure (2D)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)

    ax3 = fig.add_subplot(2, 2, 3)

    for coords, fields, label in zip(xs, fields_list, labels):
        x_i, y_i = coords
        rho_i, p_i = fields[:2]
        r, rho_avg = compute_radial_profile([x_i, y_i], rho_i)
        ax3.plot(r, rho_avg, label=label)

    ax3.set_title("Radial density")
    ax3.set_xlabel("r")
    ax3.set_ylabel("rho")
    ax3.grid(True)
    ax3.legend()

    ax4 = fig.add_subplot(2, 2, 4)

    for coords, fields, label in zip(xs, fields_list, labels):
        x_i, y_i = coords
        rho_i, p_i = fields[:2]
        r, p_avg = compute_radial_profile([x_i, y_i], p_i)
        ax4.plot(r, p_avg, label=label)

    ax4.set_title("Radial pressure")
    ax4.set_xlabel("r")
    ax4.set_ylabel("p")
    ax4.grid(True)
    ax4.legend()

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"Saved figure to {save_path}")
        plt.close()
    else:
        plt.show()

    wireframe_path = (save_path.parent / f"{save_path.stem}_3d.png") if save_path else None
    plot_wireframe_surfaces(X, Y, rho_grid, p_grid, save_path=wireframe_path)


def plot_2d_field_maps(xs, fields_list, velocities_list, labels, save_path=None):
    x, y = xs[-1]
    rho, p, e = fields_list[-1]
    u0, u1 = velocities_list[-1]
    label = labels[-1]

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


def plot_2d_schlieren(xs, fields_list, labels, save_path=None):
    x, y = xs[-1]
    rho = fields_list[-1][0]
    label = labels[-1]

    X, Y, rho_grid = build_grid(x, y, rho)
    x_unique = X[0, :]
    y_unique = Y[:, 0]

    dx = float(np.mean(np.diff(x_unique))) if len(x_unique) > 1 else 1.0
    dy = float(np.mean(np.diff(y_unique))) if len(y_unique) > 1 else 1.0

    grad_y, grad_x = np.gradient(rho_grid, dy, dx)
    grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    scale = max(float(np.nanmax(grad_mag)), 1.0e-30)
    schlieren = np.exp(-10.0 * grad_mag / scale)

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


# [9] Plot 3D diagnostics
def plot_3d_diagnostics(xs, fields_list, labels, save_path=None):
    x, y, z = xs[0]
    rho, p = fields_list[0][:2]

    rho_slice_coords, rho_slice, slice_value = extract_midplane_slice([x, y, z], rho, axis=2)
    p_slice_coords, p_slice, _ = extract_midplane_slice([x, y, z], p, axis=2)

    x2d, y2d = rho_slice_coords

    X, Y, rho_grid = build_grid(x2d, y2d, rho_slice)
    _, _, p_grid = build_grid(x2d, y2d, p_slice)

    xmin, xmax = x2d.min(), x2d.max()
    ymin, ymax = y2d.min(), y2d.max()

    fig = plt.figure(figsize=(12, 10))

    ax1 = fig.add_subplot(2, 2, 1)
    im1 = ax1.pcolormesh(X, Y, rho_grid, shading="auto")
    fig.colorbar(im1, ax=ax1)
    ax1.set_title(f"Density slice (x-y plane, z={slice_value:.3f})")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)

    ax2 = fig.add_subplot(2, 2, 2)
    im2 = ax2.pcolormesh(X, Y, p_grid, shading="auto")
    fig.colorbar(im2, ax=ax2)
    ax2.set_title(f"Pressure slice (x-y plane, z={slice_value:.3f})")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)

    ax3 = fig.add_subplot(2, 2, 3)

    for coords, fields, label in zip(xs, fields_list, labels):
        x_i, y_i, z_i = coords
        rho_i, p_i = fields[:2]
        r, rho_avg = compute_radial_profile([x_i, y_i, z_i], rho_i)
        ax3.plot(r, rho_avg, label=label)

    ax3.set_title("Radial density")
    ax3.set_xlabel("r")
    ax3.set_ylabel("rho")
    ax3.grid(True)
    ax3.legend()

    ax4 = fig.add_subplot(2, 2, 4)

    for coords, fields, label in zip(xs, fields_list, labels):
        x_i, y_i, z_i = coords
        rho_i, p_i = fields[:2]
        r, p_avg = compute_radial_profile([x_i, y_i, z_i], p_i)
        ax4.plot(r, p_avg, label=label)

    ax4.set_title("Radial pressure")
    ax4.set_xlabel("r")
    ax4.set_ylabel("p")
    ax4.grid(True)
    ax4.legend()


    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"Saved figure to {save_path}")
        plt.close()
    else:
        plt.show()

    wireframe_path = (save_path.parent / f"{save_path.stem}_3d.png") if save_path else None
    plot_wireframe_surfaces(X, Y, rho_grid, p_grid, save_path=wireframe_path)


# [10] Plot multiple multidimensional C++ outputs
def plot_multiple_cpp_solutions(
    filenames,
    title="Multidimensional Euler Solution",
    save_path=None,
    force_schlieren=False
):
    coords_list = []
    fields_list = []
    velocities_list = []
    labels = []
    dims = []

    for fname in filenames:
        dim, coords, velocity, rho, p, e = load_solution_csv(fname)

        dims.append(dim)
        coords_list.append(coords)
        fields_list.append((rho, p, e))
        velocities_list.append(velocity)
        labels.append(build_label_from_filename(fname))

    if len(set(dims)) != 1:
        raise ValueError("All files passed to plot_multid.py must have the same dimension")

    dim = dims[0]

    if dim == 2:
        xs = [(coords[0], coords[1]) for coords in coords_list]
        plot_2d_diagnostics(xs, fields_list, labels, save_path=save_path)

        x_slice_path = (save_path.parent / f"{save_path.stem}_x_slices.png") if save_path else None
        y_slice_path = (save_path.parent / f"{save_path.stem}_y_slices.png") if save_path else None
        old_y_deviation_path = (save_path.parent / f"{save_path.stem}_y_deviation_slices.png") if save_path else None
        transverse_path = (save_path.parent / f"{save_path.stem}_transverse.png") if save_path else None
        field_map_path = (save_path.parent / f"{save_path.stem}_field_maps.png") if save_path else None
        schlieren_path = (save_path.parent / f"{save_path.stem}_schlieren.png") if save_path else None
        legacy_slice_path = (save_path.parent / f"{save_path.stem}_slices.png") if save_path else None

        plot_2d_solution_slice(
            xs,
            fields_list,
            velocities_list,
            labels,
            axis="x",
            save_path=x_slice_path
        )
        plot_2d_y_deviation_slice(
            xs,
            fields_list,
            velocities_list,
            labels,
            save_path=y_slice_path
        )
        remove_if_exists(old_y_deviation_path)
        remove_if_exists(legacy_slice_path)

        plot_2d_transverse_diagnostics(
            xs,
            fields_list,
            velocities_list,
            labels,
            save_path=transverse_path
        )
        plot_2d_field_maps(
            xs,
            fields_list,
            velocities_list,
            labels,
            save_path=field_map_path
        )

        if should_make_schlieren(save_path, force_schlieren):
            plot_2d_schlieren(
                xs,
                fields_list,
                labels,
                save_path=schlieren_path
            )
        else:
            remove_if_exists(schlieren_path)
        return

    if dim == 3:
        xs = [(coords[0], coords[1], coords[2]) for coords in coords_list]
        plot_3d_diagnostics(xs, fields_list, labels, save_path=save_path)
        return

    raise ValueError(f"Unsupported dimension: {dim}")


# [11] Script entry
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("python src/graphing/plot_multid.py [--schlieren] file.csv")
        print("python src/graphing/plot_multid.py [--schlieren] directory_name")
        print("python src/graphing/plot_multid.py [--schlieren] file1.csv file2.csv")
        raise SystemExit(1)

    force_schlieren = False
    args = []

    for arg in sys.argv[1:]:
        if arg == "--schlieren":
            force_schlieren = True
        else:
            args.append(arg)

    if not args:
        raise SystemExit("No input files or directories provided")

    data_root = Path("data/csv")

    if len(args) == 1:
        path_arg = data_root / args[0]

        if path_arg.is_dir():
            csv_files = sorted([f.name for f in path_arg.glob("*.csv") if "_N" in f.name])

            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in {path_arg}")

            filenames = [str(Path(args[0]) / f) for f in csv_files]

            output_name = build_output_name(args[0])
            save_path = path_arg / f"{output_name}.png"

            plot_multiple_cpp_solutions(
                filenames,
                title=output_name,
                save_path=save_path,
                force_schlieren=force_schlieren
            )
        else:
            plot_multiple_cpp_solutions(
                [args[0]],
                force_schlieren=force_schlieren
            )

    else:
        plot_multiple_cpp_solutions(args, force_schlieren=force_schlieren)
