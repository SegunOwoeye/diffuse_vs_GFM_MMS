"""Grid, slice, and reduction helpers for multidimensional plots.

These helpers keep interpolation/reduction rules out of the rendering layer:
CSV point clouds become structured grids, 2D slices are sampled consistently
across resolutions, and optional material/interface fields are reshaped for
Schlieren diagnostics.
"""

import numpy as np
import pandas as pd

from plot_style import sort_by_resolution


def build_grid(x, y, field):
    """Pivot flat cell-centred `(x, y, value)` arrays into plotting grids."""

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


def compute_radial_profile(coords, field, center=None, nbins=100):
    """Average a scalar field into radial bins for explosion-style cases."""

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


def extract_midplane_slice(coords, field, axis=2):
    """Extract the nearest available mid-plane from flat 3D cell arrays."""

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


def choose_common_slice_value_2d(xs, axis):
    """Pick a slice coordinate that exists on the coarsest compared grid."""

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
    """Interpolate a slice when the requested coordinate is between rows."""

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
