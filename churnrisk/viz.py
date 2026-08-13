"""Shared plot style for the POC. Colors come from a validated palette."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")            # never try to open a window; scripts run headless
import matplotlib.pyplot as plt  # noqa: E402

# Validated categorical palette (light mode), used in fixed order
BLUE = "#2a78d6"     # series 1
ORANGE = "#eb6834"   # series 2
AQUA = "#1baf7a"     # series 3
PLUM = "#8a5cd1"     # series 4

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#d8d7d3"

SERIES = (BLUE, ORANGE, AQUA, PLUM)


def new_axes(width=6.4, height=4.0):
    """One clean axes on the light surface, recessive grid, no top/right spines."""
    fig, ax = plt.subplots(figsize=(width, height))
    _dress(fig, ax)
    return fig, ax


def new_grid(ncols=2, width=11.0, height=4.0):
    """A row of axes sharing the house style."""
    fig, axes = plt.subplots(1, ncols, figsize=(width, height))
    for ax in axes:
        _dress(fig, ax)
    return fig, axes


def _dress(fig, ax):
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_SOFT)
    ax.tick_params(colors=INK_SOFT, labelsize=9)
    ax.grid(True, axis="y", color=INK_SOFT, alpha=0.15, linewidth=0.8)
    ax.set_axisbelow(True)


def label(ax, title, xlabel="", ylabel=""):
    """Title and axis labels without saving, for multi-panel figures."""
    ax.set_title(title, color=INK, fontsize=10, loc="left", pad=8)
    ax.set_xlabel(xlabel, color=INK_SOFT, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_SOFT, fontsize=9)


def save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  plot saved: {path.relative_to(Path.cwd()) if path.is_absolute() else path}")


def finish(fig, ax, title, xlabel, ylabel, path):
    """Label one axes and save. The common single-panel case."""
    label(ax, title, xlabel, ylabel)
    save(fig, path)
