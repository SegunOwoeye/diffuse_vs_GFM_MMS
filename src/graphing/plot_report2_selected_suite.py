"""Generate PNG figures for the selected Report 2 quantitative suite."""

from __future__ import annotations

import argparse
import importlib.util
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_LABELS = {
    "SIM": "rGFM",
    "DIM": "DIM",
    "common": "Common",
}

EXACT_REFERENCES = {
    "FedkiwD2 1D": Path("data/exact/generated_multimaterial/test5_exact.csv"),
}

EXACT_FIELD_NAMES = {
    "density": "rho",
    "velocity": "u0",
    "specific internal energy": "e",
    "pressure": "p",
}

PROFILE_FIELDS = [
    ("rho", r"Density, $\rho$"),
    ("u0", r"Velocity, $u$"),
    ("p", r"Pressure, $p$"),
    ("e", r"Specific internal energy, $e$"),
]

HELIUM_BUBBLE_CENTER = (175.0, 0.0)
HELIUM_BUBBLE_RADIUS = 25.0


@dataclass(frozen=True)
class SolutionFile:
    path: Path
    run_id: str
    method: str
    resolution: str
    dimension: int
    time_value: float | None


def log(message: str) -> None:
    print(f"[report2_plots] {message}")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower()


def parse_time(path: Path) -> float | None:
    match = re.search(r"_t([0-9]+p[0-9]+)e([pm+\-])([0-9]+)_N", path.name)
    if match is None:
        return None
    mantissa = float(match.group(1).replace("p", "."))
    exponent = int(match.group(3))
    if match.group(2) in {"m", "-"}:
        exponent *= -1
    return mantissa * 10.0 ** exponent


def parse_resolution(run_id: str, path: Path) -> str:
    for source in (run_id, path.stem):
        matches = re.findall(r"(\d+(?:x\d+){0,2})", source)
        if matches:
            return matches[-1]
    return "unknown"


def resolution_size(resolution: str) -> int:
    values = [int(item) for item in resolution.split("x") if item.isdigit()]
    if not values:
        return 0
    total = 1
    for value in values:
        total *= value
    return total


def infer_method(run_id: str, path: Path) -> str:
    text = f"{run_id} {path.name}".lower()
    if "__sim__" in text or "quant_gfm" in text or "_gfm_" in text:
        return "SIM"
    if "__dim__" in text or "quant_dim" in text or "_dim_" in text:
        return "DIM"
    return "common"


def is_solution_csv(path: Path) -> bool:
    name = path.name.lower()
    if any(token in name for token in ("conservation", "diagnostic", "summary", "features", "exact")):
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            header = set(handle.readline().strip().split(","))
    except OSError:
        return False
    return {"x0", "rho", "p"}.issubset(header)


def collect_solution_files(root: Path) -> list[SolutionFile]:
    files: list[SolutionFile] = []
    raw = root / "raw"
    if not raw.exists():
        return files
    for path in sorted(raw.rglob("*.csv")):
        if not is_solution_csv(path):
            continue
        try:
            df_head = pd.read_csv(path, nrows=1)
        except Exception:
            continue
        dimension = 1 + int("x1" in df_head.columns) + int("x2" in df_head.columns)
        try:
            run_id = path.relative_to(raw).parts[0]
        except ValueError:
            run_id = path.parent.name
        files.append(
            SolutionFile(
                path=path,
                run_id=run_id,
                method=infer_method(run_id, path),
                resolution=parse_resolution(run_id, path),
                dimension=dimension,
                time_value=parse_time(path),
            )
        )
    return files


def latest_files(files: list[SolutionFile]) -> list[SolutionFile]:
    grouped: dict[tuple[str, str], list[SolutionFile]] = {}
    for item in files:
        grouped.setdefault((item.method, item.resolution), []).append(item)
    latest: list[SolutionFile] = []
    for group in grouped.values():
        latest.append(
            max(
                group,
                key=lambda item: -math.inf if item.time_value is None else item.time_value,
            )
        )
    return sorted(latest, key=lambda item: (item.method, resolution_size(item.resolution)))


def latest_files_by_run(files: list[SolutionFile]) -> list[SolutionFile]:
    grouped: dict[str, list[SolutionFile]] = {}
    for item in files:
        grouped.setdefault(item.run_id, []).append(item)
    latest: list[SolutionFile] = []
    for group in grouped.values():
        latest.append(
            max(
                group,
                key=lambda item: -math.inf if item.time_value is None else item.time_value,
            )
        )
    return latest


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def field_values(df: pd.DataFrame, field: str) -> np.ndarray:
    if field == "e" and "e" not in df.columns:
        return np.zeros(len(df))
    return df[field].to_numpy(dtype=float)


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    log(f"saved {path}")


def clear_report_artifacts(path: Path) -> None:
    if not path.exists():
        return
    for pattern in ("*.png", "*.html"):
        for artifact in path.glob(pattern):
            artifact.unlink()


def read_standard_exact_csv(path: Path) -> pd.DataFrame | None:
    try:
        exact = pd.read_csv(path)
    except Exception:
        return None
    required = {"x0", "rho", "u0", "p"}
    if not required.issubset(exact.columns):
        return None
    return exact.dropna(subset=["x0"]).sort_values("x0")


def read_exact_reference(title: str) -> pd.DataFrame | None:
    path = EXACT_REFERENCES.get(title)
    if path is None or not path.exists():
        return None

    raw_path = path.with_name(f"{path.stem}_raw.csv")
    if raw_path.exists():
        raw = pd.read_csv(raw_path)
        required = {"x", "rho", "u", "p", "gamma"}
        if required.issubset(raw.columns):
            exact = pd.DataFrame(
                {
                    "x0": raw["x"],
                    "rho": raw["rho"],
                    "u0": raw["u"],
                    "p": raw["p"],
                }
            )
            exact["e"] = raw["p"] / ((raw["gamma"] - 1.0) * raw["rho"])
            return exact.dropna(subset=["x0"]).sort_values("x0")

    raw = pd.read_csv(path, header=None)
    if raw.shape[0] < 3:
        return None

    labels = raw.iloc[0].fillna("").astype(str).str.strip().tolist()
    data = raw.iloc[2:].apply(pd.to_numeric, errors="coerce")
    exact = pd.DataFrame()
    for index in range(0, len(labels) - 1, 2):
        name = labels[index].strip().lower()
        field = EXACT_FIELD_NAMES.get(name)
        if field is None:
            continue
        x_values = data.iloc[:, index]
        y_values = data.iloc[:, index + 1]
        if "x0" not in exact:
            exact["x0"] = x_values
        exact[field] = y_values

    if exact.empty or "x0" not in exact:
        return None
    return exact.dropna(subset=["x0"]).sort_values("x0")


def read_local_exact_reference(items: list[SolutionFile]) -> pd.DataFrame | None:
    candidates: list[Path] = []
    for item in items:
        candidates.extend(sorted(item.path.parent.glob("*exact*.csv")))
    for path in sorted(set(candidates), key=lambda candidate: candidate.name):
        exact = read_standard_exact_csv(path)
        if exact is not None:
            return exact
    return None


def solution_label(item: SolutionFile) -> str:
    if item.method == "common":
        return f"N={item.resolution}"
    return f"{METHOD_LABELS.get(item.method, item.method)} N={item.resolution}"


def schlieren_1d(x: np.ndarray, rho: np.ndarray) -> np.ndarray:
    if len(x) < 2:
        return np.ones_like(rho)
    gradient = np.abs(np.gradient(rho, x))
    scale = max(float(np.nanpercentile(gradient, 99.5)), 1.0e-30)
    return np.exp(-8.0 * gradient / scale)


def plot_1d_profiles(files: list[SolutionFile], title: str, outdir: Path) -> None:
    items = latest_files([item for item in files if item.dimension == 1])
    if not items:
        log(f"no 1D solution CSVs for {title}")
        return

    methods = sorted({item.method for item in items})

    highest_by_method: list[SolutionFile] = []
    for method in methods:
        method_items = [item for item in items if item.method == method]
        highest_by_method.append(max(method_items, key=lambda item: resolution_size(item.resolution)))

    exact = read_exact_reference(title)
    if exact is None:
        exact = read_local_exact_reference(highest_by_method)
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.2), squeeze=False)
    for item in highest_by_method:
        df = read_csv(item.path).sort_values("x0")
        label = solution_label(item)
        for ax, (field, ylabel) in zip(axes.ravel(), PROFILE_FIELDS):
            ax.plot(df["x0"], field_values(df, field), label=label, linewidth=1.1)
            ax.set_xlabel("x")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)

    if exact is not None:
        for ax, (field, _ylabel) in zip(axes.ravel(), PROFILE_FIELDS):
            if field in exact.columns:
                ax.plot(exact["x0"], exact[field], color="black", label="Exact", linewidth=1.0)

    for ax in axes.ravel():
        ax.legend(frameon=False, fontsize=8)
    if exact is None:
        filename = f"{safe_name(title)}_1d_method_overlay.png"
    else:
        filename = f"{safe_name(title)}_1d_exact_overlay.png"
    save_figure(fig, outdir / filename)

    if title not in {"Toro test 5", "FedkiwD2 1D"}:
        fig, ax = plt.subplots(figsize=(8.2, 3.6))
        for item in highest_by_method:
            df = read_csv(item.path).sort_values("x0")
            x = df["x0"].to_numpy(dtype=float)
            rho = field_values(df, "rho")
            ax.plot(x, schlieren_1d(x, rho), label=solution_label(item), linewidth=1.1)
        if exact is not None and "rho" in exact.columns:
            x = exact["x0"].to_numpy(dtype=float)
            rho = exact["rho"].to_numpy(dtype=float)
            ax.plot(x, schlieren_1d(x, rho), color="black", label="Exact", linewidth=1.0)
        ax.set_xlabel("x")
        ax.set_ylabel(r"$|\nabla \rho|$")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        save_figure(fig, outdir / f"{safe_name(title)}_1d_schlieren.png")


def structured_grid(df: pd.DataFrame, field: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.sort(df["x0"].unique())
    ys = np.sort(df["x1"].unique())
    frame = df.pivot(index="x1", columns="x0", values=field).sort_index().sort_index(axis=1)
    return xs, ys, frame.to_numpy(dtype=float)


def density_gradient_magnitude(xs: np.ndarray, ys: np.ndarray, rho: np.ndarray) -> np.ndarray:
    dx = float(np.mean(np.diff(xs))) if len(xs) > 1 else 1.0
    dy = float(np.mean(np.diff(ys))) if len(ys) > 1 else 1.0
    grad_y, grad_x = np.gradient(rho, dy, dx)
    return np.sqrt(grad_x * grad_x + grad_y * grad_y)


def schlieren_from_gradient(magnitude: np.ndarray, scale: float | None = None) -> np.ndarray:
    if scale is None:
        scale = float(np.nanpercentile(magnitude, 99.5))
    scale = max(float(scale), 1.0e-30)
    return np.exp(-8.0 * magnitude / scale)


def schlieren(xs: np.ndarray, ys: np.ndarray, rho: np.ndarray, scale: float | None = None) -> np.ndarray:
    return schlieren_from_gradient(density_gradient_magnitude(xs, ys, rho), scale)


def schlieren_gradient_scale(items: list[SolutionFile]) -> float | None:
    scales: list[float] = []
    for item in items:
        try:
            df = read_csv(item.path)
            xs, ys, rho = structured_grid(df, "rho")
            magnitude = density_gradient_magnitude(xs, ys, rho)
            scales.append(float(np.nanpercentile(magnitude, 99.5)))
        except Exception as exc:
            log(f"could not estimate Schlieren scale for {item.path}: {exc}")
    if not scales:
        return None
    return max(float(np.nanmedian(scales)), 1.0e-30)


def format_decimal_time(time_value: float) -> str:
    return f"{time_value:.3f}".rstrip("0").rstrip(".")


def format_scaled_time(time_value: float, scale: float, unit: str) -> str:
    scaled = time_value * scale
    if abs(scaled - round(scaled)) < 1.0e-9:
        return f"t = {int(round(scaled))} {unit}"
    return f"t = {scaled:.3g} {unit}"


def mirror_transverse_grid(ys: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(ys) == 0:
        return ys, grid
    if np.isclose(float(ys[0]), 0.0):
        reflected_y = -ys[:0:-1]
        reflected_grid = grid[:0:-1, :]
    else:
        reflected_y = -ys[::-1]
        reflected_grid = grid[::-1, :]
    return np.concatenate([reflected_y, ys]), np.concatenate([reflected_grid, grid], axis=0)


def add_initial_bubble(ax: plt.Axes, center: tuple[float, float] | None, radius: float | None) -> None:
    if center is None or radius is None:
        return
    circle = plt.Circle(center, radius, fill=False, color="black", linestyle=":", linewidth=0.9)
    ax.add_patch(circle)


def add_initial_coated_bubble(
    ax: plt.Axes,
    center: tuple[float, float] | None,
    inner_radius: float | None,
    outer_radius: float | None,
) -> None:
    add_initial_bubble(ax, center, outer_radius)
    if center is None or inner_radius is None:
        return
    circle = plt.Circle(center, inner_radius, fill=False, color="black", linestyle="--", linewidth=0.8)
    ax.add_patch(circle)


def rotate_point_clockwise_90(point: tuple[float, float]) -> tuple[float, float]:
    x, y = point
    return -y, x


def rotate_image_clockwise_90(
    xs: np.ndarray,
    ys: np.ndarray,
    image: np.ndarray,
) -> tuple[np.ndarray, list[float]]:
    rotated = np.rot90(image, k=1)
    extent = [-float(ys[-1]), -float(ys[0]), float(xs[0]), float(xs[-1])]
    return rotated, extent


def plot_2d_schlieren_panel(
    files: list[SolutionFile],
    title: str,
    outdir: Path,
    center: tuple[float, float] | None = None,
    radius: float | None = None,
    inner_radius: float | None = None,
    mirror_transverse: bool = False,
    decimal_time_labels: bool = False,
    rotate_clockwise_90: bool = True,
    time_label_scale: float | None = None,
    time_label_unit: str = "",
    max_frames: int | None = 6,
    ncols_override: int | None = None,
    schlieren_scale: float | None = None,
) -> None:
    items = [item for item in files if item.dimension == 2]
    if not items:
        log(f"no 2D solution CSVs for {title}")
        return
    methods = sorted({item.method for item in items})
    for method in methods:
        method_items = [item for item in items if item.method == method]
        max_resolution = max({item.resolution for item in method_items}, key=resolution_size)
        frames = sorted(
            [item for item in method_items if item.resolution == max_resolution],
            key=lambda item: -math.inf if item.time_value is None else item.time_value,
        )
        if max_frames is not None and len(frames) > max_frames:
            indices = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
            frames = [frames[int(index)] for index in indices]
        method_schlieren_scale = schlieren_scale
        if method_schlieren_scale is None:
            method_schlieren_scale = schlieren_gradient_scale(frames)

        ncols = ncols_override or min(3, len(frames))
        nrows = int(math.ceil(len(frames) / ncols))
        if rotate_clockwise_90:
            figsize = (2.4 * ncols, 6.4 * nrows)
        else:
            figsize = (4.2 * ncols, 3.8 * nrows)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        for ax, item in zip(axes.ravel(), frames):
            df = read_csv(item.path)
            xs, ys, rho = structured_grid(df, "rho")
            image = schlieren(xs, ys, rho, method_schlieren_scale)
            if mirror_transverse:
                ys, image = mirror_transverse_grid(ys, image)
            extent = [float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])]
            plot_center = center
            if rotate_clockwise_90:
                image, extent = rotate_image_clockwise_90(xs, ys, image)
                if center is not None:
                    plot_center = rotate_point_clockwise_90(center)
            ax.imshow(
                image,
                extent=extent,
                origin="lower",
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                interpolation="bilinear",
            )
            if inner_radius is None:
                add_initial_bubble(ax, plot_center, radius)
            else:
                add_initial_coated_bubble(ax, plot_center, inner_radius, radius)
            if item.time_value is not None:
                if time_label_scale is not None:
                    ax.set_title(format_scaled_time(
                        item.time_value,
                        time_label_scale,
                        time_label_unit,
                    ))
                elif decimal_time_labels:
                    ax.set_title(f"t = {format_decimal_time(item.time_value)}")
                else:
                    ax.set_title(f"t={item.time_value:.3e}")
            else:
                ax.set_title(item.resolution)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])
        for ax in axes.ravel()[len(frames):]:
            ax.axis("off")
        save_figure(fig, outdir / f"{safe_name(title)}_{safe_name(method)}_schlieren_panel.png")


def time_slug(time_value: float | None) -> str:
    if time_value is None:
        return "unknown_time"
    return f"t{time_value:.6e}".replace("+", "").replace("-", "m").replace(".", "p")


def plot_2d_schlieren_images(
    files: list[SolutionFile],
    title: str,
    outdir: Path,
    center: tuple[float, float] | None = None,
    radius: float | None = None,
    inner_radius: float | None = None,
    mirror_transverse: bool = False,
    decimal_time_labels: bool = False,
    rotate_clockwise_90: bool = True,
    time_label_scale: float | None = None,
    time_label_unit: str = "",
    max_frames: int | None = None,
    schlieren_scale: float | None = None,
    show_time_label: bool = True,
) -> None:
    items = [item for item in files if item.dimension == 2]
    if not items:
        log(f"no 2D solution CSVs for {title}")
        return

    for method in sorted({item.method for item in items}):
        method_items = [item for item in items if item.method == method]
        max_resolution = max({item.resolution for item in method_items}, key=resolution_size)
        frames = sorted(
            [item for item in method_items if item.resolution == max_resolution],
            key=lambda item: -math.inf if item.time_value is None else item.time_value,
        )
        if max_frames is not None and len(frames) > max_frames:
            indices = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
            frames = [frames[int(index)] for index in indices]
        method_schlieren_scale = schlieren_scale
        if method_schlieren_scale is None:
            method_schlieren_scale = schlieren_gradient_scale(frames)

        for item in frames:
            df = read_csv(item.path)
            xs, ys, rho = structured_grid(df, "rho")
            image = schlieren(xs, ys, rho, method_schlieren_scale)
            if mirror_transverse:
                ys, image = mirror_transverse_grid(ys, image)
            extent = [float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])]
            plot_center = center
            if rotate_clockwise_90:
                image, extent = rotate_image_clockwise_90(xs, ys, image)
                if center is not None:
                    plot_center = rotate_point_clockwise_90(center)

            fig, ax = plt.subplots(figsize=(4.8, 4.2))
            ax.imshow(
                image,
                extent=extent,
                origin="lower",
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                interpolation="bilinear",
            )
            if inner_radius is None:
                add_initial_bubble(ax, plot_center, radius)
            else:
                add_initial_coated_bubble(ax, plot_center, inner_radius, radius)
            if show_time_label and item.time_value is not None:
                if time_label_scale is not None:
                    ax.set_title(format_scaled_time(
                        item.time_value,
                        time_label_scale,
                        time_label_unit,
                    ))
                elif decimal_time_labels:
                    ax.set_title(f"t = {format_decimal_time(item.time_value)}")
                else:
                    ax.set_title(f"t={item.time_value:.3e}")
            elif show_time_label:
                ax.set_title(item.resolution)
            ax.set_aspect("equal", adjustable="box")
            if show_time_label:
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.set_axis_off()
                fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
            save_figure(
                fig,
                outdir / f"{safe_name(title)}_{safe_name(method)}_schlieren_{time_slug(item.time_value)}.png",
            )


def coated_material_indicator(df: pd.DataFrame) -> np.ndarray:
    if {"alpha1", "alpha2"}.issubset(df.columns):
        return np.clip(
            pd.to_numeric(df["alpha1"], errors="coerce").fillna(0.0).to_numpy(dtype=float) +
            pd.to_numeric(df["alpha2"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
            0.0,
            1.0,
        )
    if "mat" in df.columns:
        material = pd.to_numeric(df["mat"], errors="coerce").fillna(-1).to_numpy(dtype=int)
        return np.isin(material, [1, 2]).astype(float)
    return np.zeros(len(df), dtype=float)


def plot_2d_coated_material_panel(
    files: list[SolutionFile],
    title: str,
    outdir: Path,
    center: tuple[float, float],
    inner_radius: float,
    outer_radius: float,
    time_label_scale: float | None = None,
    time_label_unit: str = "",
    max_frames: int | None = 6,
    ncols_override: int | None = None,
) -> None:
    items = [item for item in files if item.dimension == 2]
    if not items:
        return

    for method in sorted({item.method for item in items}):
        method_items = [item for item in items if item.method == method]
        max_resolution = max({item.resolution for item in method_items}, key=resolution_size)
        frames = sorted(
            [item for item in method_items if item.resolution == max_resolution],
            key=lambda item: -math.inf if item.time_value is None else item.time_value,
        )
        if max_frames is not None and len(frames) > max_frames:
            indices = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
            frames = [frames[int(index)] for index in indices]

        ncols = ncols_override or min(3, len(frames))
        nrows = int(math.ceil(len(frames) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 6.4 * nrows), squeeze=False)

        for ax, item in zip(axes.ravel(), frames):
            df = read_csv(item.path)
            values = coated_material_indicator(df)
            plot_df = df.copy()
            plot_df["coated_indicator"] = values
            xs, ys, image = structured_grid(plot_df, "coated_indicator")
            image, extent = rotate_image_clockwise_90(xs, ys, image)
            plot_center = rotate_point_clockwise_90(center)

            ax.imshow(
                image,
                extent=extent,
                origin="lower",
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
            add_initial_coated_bubble(ax, plot_center, inner_radius, outer_radius)
            if item.time_value is not None:
                if time_label_scale is not None:
                    ax.set_title(format_scaled_time(
                        item.time_value,
                        time_label_scale,
                        time_label_unit,
                    ))
                else:
                    ax.set_title(f"t={item.time_value:.3g}")
            else:
                ax.set_title(item.resolution)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])

        for ax in axes.ravel()[len(frames):]:
            ax.axis("off")
        save_figure(fig, outdir / f"{safe_name(title)}_{safe_name(method)}_coated_material_panel.png")


def plot_2d_coated_material_images(
    files: list[SolutionFile],
    title: str,
    outdir: Path,
    center: tuple[float, float],
    inner_radius: float,
    outer_radius: float,
    time_label_scale: float | None = None,
    time_label_unit: str = "",
    max_frames: int | None = None,
    show_time_label: bool = True,
) -> None:
    items = [item for item in files if item.dimension == 2]
    if not items:
        return

    for method in sorted({item.method for item in items}):
        method_items = [item for item in items if item.method == method]
        max_resolution = max({item.resolution for item in method_items}, key=resolution_size)
        frames = sorted(
            [item for item in method_items if item.resolution == max_resolution],
            key=lambda item: -math.inf if item.time_value is None else item.time_value,
        )
        if max_frames is not None and len(frames) > max_frames:
            indices = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
            frames = [frames[int(index)] for index in indices]

        for item in frames:
            df = read_csv(item.path)
            values = coated_material_indicator(df)
            plot_df = df.copy()
            plot_df["coated_indicator"] = values
            xs, ys, image = structured_grid(plot_df, "coated_indicator")
            image, extent = rotate_image_clockwise_90(xs, ys, image)
            plot_center = rotate_point_clockwise_90(center)

            fig, ax = plt.subplots(figsize=(4.8, 4.2))
            ax.imshow(
                image,
                extent=extent,
                origin="lower",
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
            add_initial_coated_bubble(ax, plot_center, inner_radius, outer_radius)
            if show_time_label and item.time_value is not None:
                if time_label_scale is not None:
                    ax.set_title(format_scaled_time(
                        item.time_value,
                        time_label_scale,
                        time_label_unit,
                    ))
                else:
                    ax.set_title(f"t={item.time_value:.3g}")
            elif show_time_label:
                ax.set_title(item.resolution)
            ax.set_aspect("equal", adjustable="box")
            if show_time_label:
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.set_axis_off()
                fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
            save_figure(
                fig,
                outdir / f"{safe_name(title)}_{safe_name(method)}_coated_material_{time_slug(item.time_value)}.png",
            )


def structured_slice(
    df: pd.DataFrame,
    fixed_axis: str,
    fixed_value: float,
    x_axis: str,
    y_axis: str,
    field: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    slice_df = df[np.isclose(df[fixed_axis], fixed_value)]
    xs = np.sort(slice_df[x_axis].unique())
    ys = np.sort(slice_df[y_axis].unique())
    frame = slice_df.pivot(index=y_axis, columns=x_axis, values=field).sort_index().sort_index(axis=1)
    return xs, ys, frame.to_numpy(dtype=float)


def plot_3d_density_slices(files: list[SolutionFile], title: str, outdir: Path) -> None:
    items = [item for item in latest_files(files) if item.dimension == 3]
    if not items:
        log(f"no 3D solution CSVs for {title}")
        return
    for item in items:
        if item.resolution != max([candidate.resolution for candidate in items if candidate.method == item.method], key=resolution_size):
            continue
        df = read_csv(item.path)
        x_values = np.sort(df["x0"].unique())
        y_values = np.sort(df["x1"].unique())
        z_values = np.sort(df["x2"].unique())
        slice_specs = [
            ("z midplane", "x2", z_values[len(z_values) // 2], "x0", "x1"),
            ("y midplane", "x1", y_values[len(y_values) // 2], "x0", "x2"),
            ("x midplane", "x0", x_values[len(x_values) // 2], "x1", "x2"),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.8), squeeze=False)
        for ax, (label, fixed_axis, fixed_value, x_axis, y_axis) in zip(axes.ravel(), slice_specs):
            xs, ys, rho = structured_slice(df, fixed_axis, fixed_value, x_axis, y_axis, "rho")
            image = ax.imshow(
                rho,
                extent=[float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])],
                origin="lower",
                cmap="viridis",
                interpolation="nearest",
            )
            ax.set_title(f"{label}, {fixed_axis}={fixed_value:.3g}")
            ax.set_aspect("equal", adjustable="box")
            fig.colorbar(image, ax=ax, shrink=0.8)
        save_figure(fig, outdir / f"{safe_name(title)}_{safe_name(item.method)}_3d_density_slices.png")


def latest_highest_3d_files(files: list[SolutionFile]) -> list[SolutionFile]:
    items = [item for item in latest_files(files) if item.dimension == 3]
    selected: list[SolutionFile] = []
    for method in sorted({item.method for item in items}):
        method_items = [item for item in items if item.method == method]
        selected.append(max(method_items, key=lambda item: resolution_size(item.resolution)))
    return selected


def write_helium_surface_html(item: SolutionFile, outdir: Path) -> list[Path]:
    try:
        surface_module_path = Path(__file__).resolve().parents[1] / "bubble_features" / "extract_3d_surface_features.py"
        spec = importlib.util.spec_from_file_location("extract_3d_surface_features", surface_module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load {surface_module_path}")
        surface_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(surface_module)
    except Exception as exc:
        log(f"could not import 3D helium surface tools: {exc}")
        return []

    try:
        x0, x1, x2, phi, _interface_label = surface_module.load_interface_field(item.path)
        x1_plot, x2_plot, phi_plot = surface_module.mirror_quadrant_field(x1, x2, phi)
        coords, faces = surface_module.marching_cubes_surface(x0, x1_plot, x2_plot, phi_plot)
    except Exception as exc:
        log(f"could not reconstruct 3D helium surface for {item.path}: {exc}")
        return []

    html_path = outdir / f"{safe_name(item.path.stem)}_mirrored_full_surface_interactive.html"
    surface_module.save_interactive_surface(
        coords,
        faces,
        html_path,
        f"{METHOD_LABELS.get(item.method, item.method)} 3D helium bubble interface",
    )
    if html_path.exists():
        log(f"saved {html_path}")
        return [html_path]
    return []


def write_density_isosurface_html(item: SolutionFile, outdir: Path) -> list[Path]:
    try:
        import plotly.graph_objects as go
    except Exception as exc:
        log(f"could not import plotly for 3D density HTML: {exc}")
        return []

    try:
        df = read_csv(item.path)
    except Exception as exc:
        log(f"could not read 3D density CSV for {item.path}: {exc}")
        return []

    stride = max(1, int(math.ceil(len(df) / 225000)))
    sample = df.iloc[::stride, :]
    rho = sample["rho"].to_numpy(dtype=float)
    fig = go.Figure(
        data=go.Isosurface(
            x=sample["x0"],
            y=sample["x1"],
            z=sample["x2"],
            value=rho,
            isomin=float(np.nanpercentile(rho, 35.0)),
            isomax=float(np.nanpercentile(rho, 96.0)),
            surface_count=4,
            opacity=0.55,
            caps={"x_show": False, "y_show": False, "z_show": False},
        )
    )
    fig.update_layout(
        scene={
            "xaxis_title": "x0",
            "yaxis_title": "x1",
            "zaxis_title": "x2",
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
    )
    html_path = outdir / f"{safe_name(item.path.stem)}_density_isosurface_interactive.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(html_path, include_plotlyjs="cdn")
    log(f"saved {html_path}")
    return [html_path]


def plot_3d_interactive(files: list[SolutionFile], title: str, outdir: Path) -> None:
    for item in latest_highest_3d_files(files):
        if "helium_bubble_3d" in item.run_id.lower() or "helium_bubble_3d" in item.path.name.lower():
            write_helium_surface_html(item, outdir)
        else:
            write_density_isosurface_html(item, outdir)


def sensitivity_label_order_and_slug(run_id: str) -> tuple[float, str, str]:
    text = run_id.lower()
    if "reinit_interval_never" in text:
        return 0.0, "never", "reinit_never"
    reinit = re.search(r"reinit_interval_(\d+)", text)
    if reinit:
        value = int(reinit.group(1))
        order = {1: 1, 2: 2, 5: 3, 10: 4, 20: 5}.get(value, value)
        return float(order), f"{value}", f"reinit_{value}"
    alpha = re.search(r"epsilon_alpha_dx_(\d+)dx", text)
    if alpha:
        value = int(alpha.group(1))
        return float(value), f"{value} dx", f"epsilon_{value}dx"
    tanh_alpha = re.search(r"tanh_alpha_([0-9]+p?[0-9]*)", text)
    if tanh_alpha:
        label = tanh_alpha.group(1).replace("p", ".")
        try:
            value = float(label)
        except ValueError:
            value = 999.0
        return value, label, f"alpha_{tanh_alpha.group(1)}"
    fallback = run_id.split("__")[-1].replace("_", " ")
    return 999.0, fallback, safe_name(fallback)


def plot_sensitivity(
    root: Path,
    title: str,
    global_dir: Path,
    schlieren_scale: float | None = None,
) -> None:
    files = [item for item in collect_solution_files(root) if item.dimension == 2 and "shock_bubble_2d" in item.run_id]
    if any("tanh_alpha" in item.run_id.lower() for item in files):
        files = [item for item in files if "tanh_alpha" in item.run_id.lower()]
    latest = latest_files_by_run(files)
    if not latest:
        log(f"no sensitivity solution CSVs for {title}")
        return
    local_dir = root / "figures" / "report_png"
    clear_report_artifacts(local_dir)
    latest = sorted(latest, key=lambda item: sensitivity_label_order_and_slug(item.run_id)[0])
    title_slug = safe_name(title)

    for item in latest:
        df = read_csv(item.path)
        xs, ys, rho = structured_grid(df, "rho")
        image = schlieren(xs, ys, rho, schlieren_scale)
        ys, image = mirror_transverse_grid(ys, image)
        image, extent = rotate_image_clockwise_90(xs, ys, image)
        fig, ax = plt.subplots(figsize=(4.2, 7.2))
        ax.imshow(
            image,
            extent=extent,
            origin="lower",
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            interpolation="bilinear",
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        _order, _label, slug = sensitivity_label_order_and_slug(item.run_id)
        save_figure(fig, local_dir / f"{title_slug}_{slug}.png")
    mirror_figures(local_dir, global_dir)


def plot_performance(root: Path, outdir: Path) -> None:
    perf = root / "report" / "performance_report_summary.csv"
    if not perf.exists():
        log("no performance report summary found")
        return
    df = pd.read_csv(perf)
    if df.empty:
        return
    value_columns = [column for column in df.columns if column.endswith("speedup") or "speedup" in column]
    if not value_columns:
        value_columns = [column for column in df.columns if "cell_updates_per_second" in column]
    if not value_columns:
        log("performance summary has no plottable speed columns")
        return
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    x = np.arange(len(df))
    width = 0.8 / len(value_columns)
    for index, column in enumerate(value_columns):
        values = pd.to_numeric(df[column], errors="coerce")
        ax.bar(x + index * width, values, width=width, label=column)
    ax.set_xticks(x + width * (len(value_columns) - 1) / 2.0)
    ax.set_xticklabels(df.get("case", pd.Series(range(len(df)))).astype(str), rotation=30, ha="right")
    ax.set_ylabel("Performance metric")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, outdir / "openmp_performance_summary.png")


def mirror_figures(local_dir: Path, global_dir: Path) -> None:
    if not local_dir.exists():
        return
    global_dir.mkdir(parents=True, exist_ok=True)
    for path in list(local_dir.glob("*.png")) + list(local_dir.glob("*.html")):
        target = global_dir / path.name
        shutil.copy2(path, target)


def clear_root_report_artifacts(root: Path) -> None:
    clear_report_artifacts(root / "figures" / "report_png")


def collect_shared_helium_schlieren_scale(base: Path) -> float | None:
    scale_items: list[SolutionFile] = []
    for root in (
        base / "report_selected_helium_bubble_2d",
        base / "report_selected_sim_reinit_sensitivity",
        base / "report_selected_dim_alpha_sensitivity",
    ):
        files = collect_solution_files(root)
        if root.name.endswith("_sensitivity"):
            files = latest_files_by_run([item for item in files if item.dimension == 2])
        scale_items.extend([item for item in files if item.dimension == 2])
    return schlieren_gradient_scale(scale_items)


def plot_root(
    root: Path,
    title: str,
    kind: str,
    global_dir: Path,
    schlieren_scale: float | None = None,
) -> None:
    files = collect_solution_files(root)
    local_dir = root / "figures" / "report_png"
    clear_report_artifacts(local_dir)
    if kind == "1d":
        plot_1d_profiles(files, title, local_dir)
    elif kind == "2d_bubble":
        plot_2d_schlieren_images(
            files,
            title,
            local_dir,
            center=HELIUM_BUBBLE_CENTER,
            radius=HELIUM_BUBBLE_RADIUS,
            mirror_transverse=True,
            decimal_time_labels=True,
            rotate_clockwise_90=True,
            schlieren_scale=schlieren_scale,
            show_time_label=False,
        )
    elif kind == "2d_gorsse":
        plot_2d_schlieren_images(
            files,
            title,
            local_dir,
            center=(0.5, 0.5),
            radius=0.2,
            rotate_clockwise_90=False,
            show_time_label=False,
        )
    elif kind == "2d_coated_bubble":
        plot_2d_schlieren_images(
            files,
            title,
            local_dir,
            center=(0.45, 0.25),
            radius=0.1,
            inner_radius=0.08,
            time_label_scale=10000.0,
            time_label_unit="us",
            max_frames=None,
            show_time_label=False,
        )
        plot_2d_coated_material_images(
            files,
            title,
            local_dir,
            center=(0.45, 0.25),
            inner_radius=0.08,
            outer_radius=0.1,
            time_label_scale=10000.0,
            time_label_unit="us",
            max_frames=None,
            show_time_label=False,
        )
    elif kind == "2d":
        plot_2d_schlieren_images(files, title, local_dir)
    elif kind == "3d":
        plot_3d_density_slices(files, title, local_dir)
        plot_3d_interactive(files, title, local_dir)
    elif kind == "3d_interactive":
        plot_3d_interactive(files, title, local_dir)
    mirror_figures(local_dir, global_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root-base", type=Path, default=Path("results/quantitative"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    base = args.result_root_base
    global_dir = args.output_dir or (base / "report_selected_figures")
    global_dir.mkdir(parents=True, exist_ok=True)
    clear_report_artifacts(global_dir)
    helium_schlieren_scale = collect_shared_helium_schlieren_scale(base)

    plot_root(base / "report_selected_toro_test5", "Toro test 5", "1d", global_dir)
    plot_root(base / "report_selected_fedkiw_d2_1d", "FedkiwD2 1D", "1d", global_dir)
    clear_root_report_artifacts(base / "report_selected_explosion_2d")
    plot_root(base / "report_selected_explosion_3d", "3D explosion", "3d", global_dir)
    plot_root(
        base / "report_selected_helium_bubble_2d",
        "Helium shock-bubble 2D",
        "2d_bubble",
        global_dir,
        schlieren_scale=helium_schlieren_scale,
    )
    plot_root(base / "report_selected_helium_bubble_3d", "Helium shock-bubble 3D", "3d_interactive", global_dir)
    plot_root(base / "report_selected_gorsse_tc9_water_air_2d", "Gorsse TC9 water-air 2D", "2d_gorsse", global_dir)
    plot_root(base / "report_selected_applsci_three_material_2d", "Appl Sci 2021 three-material 2D", "2d_coated_bubble", global_dir)
    plot_root(base / "report_selected_gorsse_tc9_water_air_3d", "Gorsse TC9 water-air 3D", "3d", global_dir)
    plot_sensitivity(
        base / "report_selected_sim_reinit_sensitivity",
        "rGFM reinitialisation",
        global_dir,
        schlieren_scale=helium_schlieren_scale,
    )
    plot_sensitivity(
        base / "report_selected_dim_alpha_sensitivity",
        "DIM tanh alpha",
        global_dir,
        schlieren_scale=helium_schlieren_scale,
    )
    plot_performance(base / "report_selected_openmp_scaling", global_dir)


if __name__ == "__main__":
    main()
