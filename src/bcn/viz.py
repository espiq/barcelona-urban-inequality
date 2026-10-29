"""Plotting helpers with a single, consistent visual system.

Palette: validated categorical order (blue, orange, aqua), one-hue blue
sequential ramp for magnitudes, and a blue <-> red diverging ramp with a
neutral grey midpoint for signed quantities.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .config import FIGURES

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#a3a29c"
GRID = "#e6e5e0"
NEUTRAL = "#e3e2dd"

CATEGORICAL = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

SEQUENTIAL = LinearSegmentedColormap.from_list("bcn_blue", BLUE_RAMP)
DIVERGING = LinearSegmentedColormap.from_list(
    "bcn_div", ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f2a19d", "#e34948", "#9c2a2a"]
)

LISA_COLORS = {
    "High-High": "#c93c3b",
    "High-Low": "#f2a19d",
    "Low-High": "#9ec5f4",
    "Low-Low": "#184f95",
    "Not significant": NEUTRAL,
}


def set_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.labelcolor": TEXT_2,
            "axes.edgecolor": MUTED,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": TEXT_2,
            "ytick.color": TEXT_2,
            "text.color": TEXT,
            "legend.frameon": False,
            "lines.linewidth": 2,
            "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
        }
    )


def map_axes(ax) -> None:
    ax.set_axis_off()
    ax.set_aspect("equal")


def caption(fig, text: str, y: float = -0.02) -> None:
    fig.text(0.01, y, text, ha="left", va="top", fontsize=8, color=TEXT_2, wrap=True)


def save(fig, name: str, folder: Path = FIGURES) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.png")
    plt.close(fig)


def label_line_ends(ax, labels: dict, min_gap_frac: float = 0.045, **kwargs) -> None:
    """Direct-label line ends, nudging labels apart so they never overlap.

    ``labels`` maps text -> (x, y) of the last point.
    """
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap_frac
    # Place labels 6 points to the right of the last marker.
    shift = ax.transData + mpl.transforms.ScaledTranslation(6 / 72, 0, ax.figure.dpi_scale_trans)
    placed: list[float] = []
    for text, (x, y) in sorted(labels.items(), key=lambda kv: kv[1][1]):
        y_lab = max(y, placed[-1] + gap) if placed else y
        placed.append(y_lab)
        ax.text(x, y_lab, text, transform=shift, va="center", fontsize=9, color=TEXT_2, **kwargs)
