from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np


FIELD_LABELS = {
    "rho": ("Density", r"Density, $\rho$"),
    "u": ("Velocity", r"Velocity, $u$"),
    "u0": ("Velocity", r"Velocity, $u$"),
    "u1": ("Transverse velocity", r"Velocity, $u_1$"),
    "u_radial": ("Velocity", r"Radial velocity, $u_r$"),
    "p": ("Pressure", r"Pressure, $p$"),
    "e": ("Specific internal energy", r"Specific internal energy, $e$"),
    "e_plot": ("Specific internal energy", r"Specific internal energy, $e$"),
    "entropy": ("Entropy", r"Entropy, $s$"),
}

RESOLUTION_STYLES = {
    50: {"color": "tab:purple", "marker": "s"},
    100: {"color": "green", "marker": "o"},
    200: {"color": "blue", "marker": "D"},
    400: {"color": "red", "marker": "+"},
    800: {"color": "tab:orange", "marker": "x"},
    1600: {"color": "tab:brown", "marker": "^"},
}


def apply_plot_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.0,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "mathtext.fontset": "dejavuserif",
    })


def field_title(field_key):
    return FIELD_LABELS.get(field_key, (field_key, field_key))[0]


def field_ylabel(field_key):
    return FIELD_LABELS.get(field_key, (field_key, field_key))[1]


def resolution_from_label(label):
    text = str(label).lower()

    if "exact" in text:
        return None

    nums = [int(n) for n in re.findall(r"n=([0-9]+)", text)]
    if nums:
        return nums[0]

    nums = [int(n) for n in re.findall(r"_n([0-9]+)", text)]
    if nums:
        return nums[0]

    nums = [int(n) for n in re.findall(r"\b([0-9]+)\s*cells?\b", text)]
    if nums:
        return nums[0]

    return None


def legend_label(label):
    text = str(label).lower()

    if "reference" in text:
        return "Reference"

    if "exact" in text:
        return "Exact"

    resolution = resolution_from_label(label)

    if resolution is not None:
        return f"{resolution} cells"

    return str(label)


def marker_step(x, target_markers=46):
    n = len(x)
    if n <= target_markers:
        return 1
    return max(1, n // target_markers)


def numerical_style(label, index=0):
    resolution = resolution_from_label(label)
    style = RESOLUTION_STYLES.get(resolution)

    if style is None:
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["black"])
        markers = ["o", "D", "s", "^", "v", "x", "+"]
        style = {
            "color": cycle[index % len(cycle)],
            "marker": markers[index % len(markers)],
        }

    return style


def plot_profile(ax, x, y, label, index=0):
    display_label = legend_label(label)

    if display_label in {"Exact", "Reference"}:
        ax.plot(
            x,
            y,
            color="black",
            linestyle="-",
            linewidth=1.2,
            label=display_label,
        )
        return

    style = numerical_style(label, index=index)

    ax.plot(
        x,
        y,
        linestyle="None",
        linewidth=0.0,
        marker=style["marker"],
        markevery=marker_step(x),
        markersize=2.8,
        markerfacecolor="none",
        markeredgewidth=0.8,
        color=style["color"],
        label=display_label,
    )


def configure_profile_axis(
    ax,
    field_key=None,
    x_label=r"$x$",
    show_legend=True,
    show_title=True,
):
    ax.set_xlabel(x_label)

    if field_key is not None:
        ax.set_ylabel(field_ylabel(field_key))
        if show_title:
            ax.set_title(field_title(field_key))

    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    for axis in (ax.xaxis, ax.yaxis):
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-3, 4))
        formatter.set_useOffset(False)
        axis.set_major_formatter(formatter)

    ax.grid(False)

    if show_legend:
        ax.legend(frameon=False, loc="best")


def sort_by_resolution(items, label_index=-1):
    def key(item):
        label = item[label_index]
        label_text = str(label).lower()
        if "exact" in label_text or "reference" in label_text:
            return (1, float("inf"))

        resolution = resolution_from_label(label)
        if resolution is None:
            return (0, float("inf"))

        return (0, -resolution)

    return sorted(items, key=key)


def save_figure(fig, save_path):
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
        plt.close(fig)
    else:
        plt.show()


apply_plot_style()
