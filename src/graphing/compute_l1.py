# [0] Import libraries
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# [1] Load CSV (dimension-aware)
def load_csv(path):

    df = pd.read_csv(path)

    dim = 0
    while f"x{dim}" in df.columns:
        dim += 1

    coords = [df[f"x{d}"].to_numpy() for d in range(dim)]

    rho = df["rho"].to_numpy()
    u   = df["u0"].to_numpy()
    p   = df["p"].to_numpy()
    e   = df["e"].to_numpy()

    return dim, coords, rho, u, p, e


# [2] Compute L1
def compute_l1(num, exact):
    return np.mean(np.abs(num - exact))


# [3] Extract resolution
def extract_N(filename):

    name = Path(filename).name.lower()

    parts = [p for p in name.replace(".csv", "").split("_") if p.startswith("n")]

    if not parts:
        return None

    nums = parts[0][1:].split("x")
    return int(nums[0])


# [4] 1D L1 (exact required)
def compute_l1_1d(files):

    exact_file = None
    num_files = []

    for f in files:
        if "exact" in f.name.lower():
            exact_file = f
        else:
            num_files.append(f)

    if exact_file is None:
        raise RuntimeError("1D case requires exact solution file")

    _, coordsE, rhoE, uE, pE, eE = load_csv(exact_file)
    xE = coordsE[0]

    results = []

    for f in num_files:

        N = extract_N(f.name)
        if N is None:
            continue

        _, coords, rho, u, p, e = load_csv(f)
        x = coords[0]

        rhoE_i = np.interp(x, xE, rhoE)
        uE_i   = np.interp(x, xE, uE)
        pE_i   = np.interp(x, xE, pE)
        eE_i   = np.interp(x, xE, eE)

        err_rho = compute_l1(rho, rhoE_i)
        err_u   = compute_l1(u, uE_i)
        err_p   = compute_l1(p, pE_i)
        err_e   = compute_l1(e, eE_i)

        results.append((N, err_rho, err_u, err_p, err_e))

    results.sort(key=lambda x: x[0])

    return results


# [5] Multidimensional L1 (reference solution)
def compute_l1_multid(files):

    files = sorted(files, key=lambda f: extract_N(f.name))

    ref_file = files[-1]

    dim_ref, coords_ref, rho_ref, u_ref, p_ref, e_ref = load_csv(ref_file)

    N_ref = int(round(len(rho_ref) ** (1.0 / dim_ref)))
    shape_ref = tuple([N_ref] * dim_ref)

    rho_ref = rho_ref.reshape(shape_ref)
    u_ref   = u_ref.reshape(shape_ref)
    p_ref   = p_ref.reshape(shape_ref)
    e_ref   = e_ref.reshape(shape_ref)

    results = []

    for f in files[:-1]:

        N = extract_N(f.name)
        if N is None:
            continue

        dim, coords, rho, u, p, e = load_csv(f)

        N_coarse = int(round(len(rho) ** (1.0 / dim)))
        shape = tuple([N_coarse] * dim)

        rho = rho.reshape(shape)
        u   = u.reshape(shape)
        p   = p.reshape(shape)
        e   = e.reshape(shape)

        if N_ref % N_coarse != 0:
            raise RuntimeError(f"Incompatible grids: {N_ref} vs {N_coarse}")

        ratio = N_ref // N_coarse
        slicer = tuple(slice(None, None, ratio) for _ in range(dim))

        rho_r = rho_ref[slicer]
        u_r   = u_ref[slicer]
        p_r   = p_ref[slicer]
        e_r   = e_ref[slicer]

        err_rho = compute_l1(rho, rho_r)
        err_u   = compute_l1(u, u_r)
        err_p   = compute_l1(p, p_r)
        err_e   = compute_l1(e, e_r)

        results.append((N, err_rho, err_u, err_p, err_e))

    results.sort(key=lambda x: x[0])

    return results


# [6] Dispatcher
def compute_l1_errors(folder):

    folder = Path("data/csv") / folder
    files = sorted(folder.glob("*.csv"))

    if not files:
        raise RuntimeError("No CSV files found")

    dim, *_ = load_csv(files[0])

    if dim == 1:
        return compute_l1_1d(files)

    if dim in (2, 3):
        return compute_l1_multid(files)

    raise RuntimeError(f"Unsupported dimension: {dim}")


# [7] Plot convergence
def plot_l1(results, folder):

    folder_path = Path("data/csv") / folder

    N = np.array([r[0] for r in results])

    rho_err = np.array([r[1] for r in results])
    u_err   = np.array([r[2] for r in results])
    p_err   = np.array([r[3] for r in results])
    e_err   = np.array([r[4] for r in results])

    plt.figure(figsize=(8, 6))

    plt.loglog(N, rho_err, 'o-', label="rho")
    plt.loglog(N, u_err, 'o-', label="u")
    plt.loglog(N, p_err, 'o-', label="p")
    plt.loglog(N, e_err, 'o-', label="e")

    ref = rho_err[0] * (N / N[0])**(-1)
    plt.loglog(N, ref, '--', label="O(1)")

    plt.xlabel("N")
    plt.ylabel("L1 Error")
    plt.grid(True, which="both")
    plt.legend()

    output_name = "_".join(Path(folder).parts[-2:])
    save_path = folder_path / f"{output_name}_L1.png"

    plt.savefig(save_path, dpi=300)
    print(f"Saved plot to {save_path}")

    plt.close()


# [8] Save table
def save_l1_table(results, folder):

    folder_path = Path("data/csv") / folder

    df = pd.DataFrame(results, columns=["N", "rho", "u", "p", "e"])

    output_name = "_".join(Path(folder).parts[-2:])
    save_path = folder_path / f"{output_name}_L1.csv"

    df.to_csv(save_path, index=False)

    print(f"Saved L1 table to {save_path}")


# [9] Run
if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:
        print("Usage: python compute_l1.py folder_name")
        exit(1)

    folder = sys.argv[1]

    results = compute_l1_errors(folder)

    print("\nL1 Errors:")
    for r in results:
        print(f"N={r[0]} | rho={r[1]:.4e} u={r[2]:.4e} p={r[3]:.4e} e={r[4]:.4e}")

    save_l1_table(results, folder)
    plot_l1(results, folder)

    