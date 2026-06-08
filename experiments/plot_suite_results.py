from __future__ import annotations

import csv
from pathlib import Path


def _load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _summary_path(result_root: Path, name: str) -> Path:
    nested = result_root / "summaries" / name
    if nested.exists():
        return nested
    return result_root / name


def plot_suite_results(result_root: Path) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warning = result_root / "figures" / "PLOTTING_SKIPPED.txt"
        warning.parent.mkdir(parents=True, exist_ok=True)
        warning.write_text("matplotlib is not installed; summary CSVs were generated without plots.\n", encoding="utf-8")
        return [warning]

    figures = result_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    performance = _load_rows(_summary_path(result_root, "performance_summary.csv"))
    if performance:
        xs = [row["resolution"] for row in performance]
        ys = [float(row.get("wall_time_seconds") or 0.0) for row in performance]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(xs)), ys)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs, rotation=45, ha="right")
        ax.set_ylabel("Wall time (s)")
        ax.set_title("Runtime by resolution")
        fig.tight_layout()
        out = figures / "runtime_by_resolution.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        written.append(out)

    convergence = _load_rows(_summary_path(result_root, "convergence_summary.csv"))
    if convergence:
        fig, ax = plt.subplots(figsize=(7, 4))
        grouped: dict[str, list[tuple[float, float]]] = {}
        for row in convergence:
            if row.get("norm") != "L1":
                continue
            key = f"{row['case']} {row['method']} {row['variable']}"
            grouped.setdefault(key, []).append((float(row["N_fine"]), float(row["error_fine"])))
        for key, values in grouped.items():
            values.sort()
            if values:
                ax.loglog([item[0] for item in values], [item[1] for item in values], marker="o", label=key)
        ax.set_xlabel("N")
        ax.set_ylabel("L1 error")
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        out = figures / "convergence_l1.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        written.append(out)

    bubble = _load_rows(_summary_path(result_root, "bubble_feature_summary.csv"))
    if bubble:
        velocity_keys = [
            "upstream_interface_x_velocity",
            "downstream_interface_x_velocity",
            "air_jet_head_x_velocity",
            "transmitted_shock_x_pressure_gradient_velocity",
            "transverse_interface_y_velocity",
        ]
        labels: list[str] = []
        values: list[float] = []
        for row in bubble:
            prefix = f"{row.get('method', '')} {row.get('resolution', '')}".strip()
            for key in velocity_keys:
                if row.get(key):
                    labels.append(f"{prefix} {key.replace('_velocity', '')}")
                    values.append(float(row[key]))
        if values:
            fig, ax = plt.subplots(figsize=(9, 4.5))
            ax.bar(range(len(values)), values)
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_ylabel("Feature velocity")
            ax.set_title("Shock-bubble feature velocities")
            fig.tight_layout()
            out = figures / "bubble_feature_velocities.png"
            fig.savefig(out, dpi=200)
            plt.close(fig)
            written.append(out)

    return written


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Plot quantitative suite results.")
    parser.add_argument("result_root", type=Path)
    args = parser.parse_args()
    for path in plot_suite_results(args.result_root):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
