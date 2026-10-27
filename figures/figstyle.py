"""Origin-style figure conventions shared by every data figure in the paper.

The look follows the OriginPro house style used in engineering journals: a closed
four-sided frame, inward ticks on all sides, one consistent heavy stroke for the frame
and for every bar and marker outline, saturated fills, white background and a
boxed legend. Everything is plain Matplotlib, so the figures rebuild from the ledgers
with no extra dependency and no Origin licence.

Usage::

    import figstyle as fs
    fs.apply()
    fig, ax = plt.subplots(figsize=(fs.WIDTH, 2.2), constrained_layout=True)
    fs.bars(ax, x, y, color=fs.BLUE)
    fs.save(fig, "figX.pdf")
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# Okabe-Ito, colour-blind safe. Names are kept from the original generators.
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERM = "#D55E00"
SKY = "#56B4E9"
PINK = "#CC79A7"
YELLOW = "#F0E442"
INK = "0.15"
GRID = "0.88"
REF = "0.35"

FRAME_LW = 1.4       # axes frame
EDGE_LW = 0.9        # outline of every bar, marker and patch
TICK_MAJOR = 4.0
TICK_MINOR = 2.2
FONT = 8.0
ANNOT = 7.4          # in-panel annotations and value labels
TICK = 7.8
LABEL = 8.6
TITLE = 8.6
WIDTH = 6.65         # two-column text width, inches
BODY_WIDTH = 5.5     # body figure width, inches


def apply(serif: bool = False) -> None:
    """Install the Origin-style rcParams. Call once, before creating figures."""
    family = (
        {"font.family": "serif",
         "font.serif": ["Times New Roman", "Liberation Serif", "Times",
                        "DejaVu Serif"],
         "mathtext.fontset": "stix"}
        if serif else
        {"font.family": "sans-serif",
         "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
         "mathtext.fontset": "custom",
         "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
         "mathtext.bf": "Arial:bold", "mathtext.sf": "Arial",
         "mathtext.fallback": "stixsans"}
    )
    plt.rcParams.update({
        **family,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": FONT,
        "axes.labelsize": LABEL,
        "axes.labelweight": "bold",
        "axes.titlesize": TITLE,
        "axes.titleweight": "bold",
        "axes.titlepad": 4.5,
        "axes.labelpad": 2.5,
        "axes.edgecolor": "black",
        "axes.linewidth": FRAME_LW,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.axisbelow": True,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "xtick.major.size": TICK_MAJOR,
        "ytick.major.size": TICK_MAJOR,
        "xtick.minor.size": TICK_MINOR,
        "ytick.minor.size": TICK_MINOR,
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,
        "xtick.labelsize": TICK,
        "ytick.labelsize": TICK,
        "legend.fontsize": ANNOT,
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.edgecolor": "black",
        "legend.framealpha": 1.0,
        "legend.borderpad": 0.35,
        "legend.handletextpad": 0.4,
        "legend.labelspacing": 0.3,
        "lines.linewidth": 1.8,
        "lines.markersize": 5.2,
        "lines.markeredgewidth": EDGE_LW,
        "patch.linewidth": EDGE_LW,
        "patch.edgecolor": "black",
        "errorbar.capsize": 2.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def bar_kw(**overrides) -> dict:
    """Keyword arguments for bar/barh: black outline, solid fill."""
    kw = dict(edgecolor="black", linewidth=EDGE_LW, alpha=1.0,
              error_kw=dict(elinewidth=1.0, ecolor="black", capsize=2.5,
                            capthick=1.0))
    kw.update(overrides)
    return kw


def marker_kw(**overrides) -> dict:
    """Keyword arguments for scatter: black outline around every marker."""
    kw = dict(edgecolor="black", linewidth=EDGE_LW)
    kw.update(overrides)
    return kw


def ref_line(ax, value: float, axis: str = "y", **overrides):
    """Dashed reference line (chance level, zero, beta = 1)."""
    kw = dict(color=REF, lw=1.1, ls="--", zorder=1)
    kw.update(overrides)
    return (ax.axhline if axis == "y" else ax.axvline)(value, **kw)


def minor_ticks(ax, x: bool = True, y: bool = True) -> None:
    """Origin-style minor ticks on linear axes."""
    if x:
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    if y:
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="minor", direction="in", top=x, right=y)


def hgrid(ax, axis: str = "y") -> None:
    """Restrained grid under the data."""
    ax.grid(axis=axis, color=GRID, linewidth=0.5)


def panel_title(ax, text: str, **overrides) -> None:
    """Left-aligned panel title, the (a)/(b)/(c) convention used in the paper."""
    kw = dict(loc="left", fontsize=TITLE, fontweight="bold")
    kw.update(overrides)
    ax.set_title(text, **kw)


def boxed_legend(ax, **overrides):
    """Legend with the thin black frame Origin uses."""
    kw = dict(frameon=True, edgecolor="black", fancybox=False, framealpha=1.0)
    kw.update(overrides)
    leg = ax.legend(**kw)
    leg.get_frame().set_linewidth(1.0)
    return leg


def save(fig, path) -> Path:
    """Write the vector PDF and report the path."""
    path = Path(path)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"wrote {path}")
    return path
