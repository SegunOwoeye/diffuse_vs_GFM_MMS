# [0] Import libraries
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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
def plot_wireframe_surfaces(X, Y, rho_grid, p_grid, save_path=None):
    rstride = 4
    cstride = 4
    elev = 30
    azim = 135

    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_wireframe(X, Y, rho_grid, rstride=rstride, cstride=cstride, linewidth=0.5)
    ax1.set_title("Density")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("rho")
    ax1.view_init(elev=elev, azim=azim)
    ax1.set_box_aspect([1, 1, 0.5])

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot_wireframe(X, Y, p_grid, rstride=rstride, cstride=cstride, linewidth=0.5)
    ax2.set_title("Pressure")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("p")
    ax2.view_init(elev=elev, azim=azim)
    ax2.set_box_aspect([1, 1, 0.5])

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"Saved 3D figure to {save_path}")
        plt.close()
    else:
        plt.show()


# [8] Plot 2D diagnostics
def plot_2d_diagnostics(xs, fields_list, labels, title="", save_path=None):
    x, y = xs[-1]
    rho, p = fields_list[-1]

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
        rho_i, p_i = fields
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
        rho_i, p_i = fields
        r, p_avg = compute_radial_profile([x_i, y_i], p_i)
        ax4.plot(r, p_avg, label=label)

    ax4.set_title("Radial pressure")
    ax4.set_xlabel("r")
    ax4.set_ylabel("p")
    ax4.grid(True)
    ax4.legend()

    if title:
        fig.suptitle(title)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
    else:
        plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"Saved figure to {save_path}")
        plt.close()
    else:
        plt.show()

    wireframe_path = (save_path.parent / f"{save_path.stem}_3d.png") if save_path else None
    plot_wireframe_surfaces(X, Y, rho_grid, p_grid, save_path=wireframe_path)


# [9] Plot 3D diagnostics
def plot_3d_diagnostics(xs, fields_list, labels, title="", save_path=None):
    x, y, z = xs[0]
    rho, p = fields_list[0]

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
        rho_i, p_i = fields
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
        rho_i, p_i = fields
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
def plot_multiple_cpp_solutions(filenames, title="Multidimensional Euler Solution", save_path=None):
    coords_list = []
    fields_list = []
    labels = []
    dims = []

    for fname in filenames:
        dim, coords, velocity, rho, p, e = load_solution_csv(fname)

        dims.append(dim)
        coords_list.append(coords)
        fields_list.append((rho, p))
        labels.append(build_label_from_filename(fname))

    if len(set(dims)) != 1:
        raise ValueError("All files passed to plot_multid.py must have the same dimension")

    dim = dims[0]

    if dim == 2:
        xs = [(coords[0], coords[1]) for coords in coords_list]
        plot_2d_diagnostics(xs, fields_list, labels, title=title, save_path=save_path)
        return

    if dim == 3:
        xs = [(coords[0], coords[1], coords[2]) for coords in coords_list]
        plot_3d_diagnostics(xs, fields_list, labels, title=title, save_path=save_path)
        return

    raise ValueError(f"Unsupported dimension: {dim}")


# [11] Script entry
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("python src/graphing/plot_multid.py file.csv")
        print("python src/graphing/plot_multid.py directory_name")
        print("python src/graphing/plot_multid.py file1.csv file2.csv")
        raise SystemExit(1)

    args = sys.argv[1:]
    data_root = Path("data/csv")

    if len(args) == 1:
        path_arg = data_root / args[0]

        if path_arg.is_dir():
            csv_files = sorted([f.name for f in path_arg.glob("*.csv")])

            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in {path_arg}")

            filenames = [str(Path(args[0]) / f) for f in csv_files]

            output_name = build_output_name(args[0])
            save_path = path_arg / f"{output_name}.png"

            plot_multiple_cpp_solutions(
                filenames,
                title=output_name,
                save_path=save_path
            )
        else:
            plot_multiple_cpp_solutions([args[0]])

    else:
        plot_multiple_cpp_solutions(args)



