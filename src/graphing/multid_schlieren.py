"""Schlieren and bubble-display helpers for multidimensional plots.

This module contains display-specific transforms for density-gradient figures,
especially the Haas-Sturtevant/Fedkiw shock-bubble convention where the stored
half-domain is mirrored and rotated for report figures.
"""

import numpy as np
import matplotlib.pyplot as plt

from multid_io import remove_if_exists
from multid_slices import infer_domain_limits


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
    """Reflect top-half shock-bubble data across the symmetry plane `y=0`."""

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
    """Differentiate density without crossing material-interface cells."""

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


def fedkiw_mock_schlieren(rho_grid, grad_mag, coefficient=20.0):
    """Approximate the experimental-style mock Schlieren intensity."""

    rho_safe = np.maximum(rho_grid, 1.0e-30)
    exponent = -coefficient * grad_mag / (1000.0 * np.sqrt(rho_safe))
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


def is_gfm_plot_path(save_path):
    if save_path is None:
        return False

    path_text = str(save_path).lower()
    return "/gfm/" in path_text or "\\gfm\\" in path_text or "gfm_" in path_text


def rotate_bubble_display(field):
    """Rotate the mirrored bubble image into the report/reference orientation."""

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
