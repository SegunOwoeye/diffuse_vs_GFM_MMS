# [0] Import Libraries
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


# [1] Load solution from CSV written by C++
def load_solution_csv(filename):
    data_path = Path("data/csv") / filename

    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found")

    df = pd.read_csv(data_path)

    x = df["x0"].to_numpy()
    rho = df["rho"].to_numpy()
    u = df["u0"].to_numpy()
    p = df["p"].to_numpy()
    e = df["e"].to_numpy()

    return x, rho, u, p, e


# [2] Plot 1D 
def plot_1d(xs, fields_list, labels, title="", exact=None, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    names = [
        "Density",
        "Velocity",
        "Pressure",
        "Specific internal energy"
    ]

    for x, fields, label in zip(xs, fields_list, labels):
        rho, u, p, e = fields
        field_vars = [rho, u, p, e]

        for ax, field, name in zip(axes, field_vars, names):
            ax.plot(x, field, label=label)
            ax.set_xlabel("Position x")
            ax.set_ylabel(name)
            ax.grid(True)

    if exact is not None:
        for ax, key in zip(axes, ["rho", "u", "p", "e"]):
            ax.plot(exact["x"], exact[key], "k--", linewidth=1.2, label="Exact")

    for ax in axes:
        ax.legend()

    #fig.suptitle(title)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"Saved figure to {save_path}")
        plt.close()
    else:
        plt.show()


# [2.1] Constructing plot names
def build_output_name(folder_path: str) -> str:
    p = Path(folder_path)
    parts = p.parts[-2:]  # last two folder names
    return "_".join(parts)


# [3.1] Plot multiple C++ outputs together
def plot_multiple_cpp_solutions(filenames, title="1D Euler Solution", exact=None, save_path=None):
    xs = []
    fields_list = []
    labels = []

    for fname in filenames:
        x, rho, u, p, e = load_solution_csv(fname)

        xs.append(x)
        fields_list.append((rho, u, p, e))

        name = Path(fname).name.lower()
        if "exact" in name:
            labels.append("N=Exact")

        elif "_n" in name:
            N = name.split("_n")[-1].replace(".csv", "")
            labels.append(f"N={N}")

        else:
            labels.append(name)

    plot_1d(xs, fields_list, labels, title=title, exact=exact, save_path=save_path)



# [4] Script entry
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("python plot_solution_1d.py file.csv")
        print("python plot_solution_1d.py directory_name")
        print("python plot_solution_1d.py file1.csv file2.csv")
        raise SystemExit(1)

    args = sys.argv[1:]
    data_root = Path("data/csv")

    # Case 1: Single argument and it's a directory
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
            # Single file
            plot_multiple_cpp_solutions([args[0]])

    # Case 2: Multiple files explicitly provided
    else:
        plot_multiple_cpp_solutions(args)

# Example Run
