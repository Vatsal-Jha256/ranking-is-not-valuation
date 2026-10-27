#!/usr/bin/env python3
"""Generate the paper's three-RQ main figure from prespecified ledgers."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent / "fig1_submission_story.pdf"


def load(name: str) -> dict:
    with (RESULTS / name).open() as handle:
        return json.load(handle)


headline = load("exp_submission_harmonized_v4.json")
pools = load("exp_candidate_sensitivity_v1.json")
uncertainty = load("exp_uncertainty_policies_v1.json")

import figstyle as fs

OK_BLUE, OK_ORANGE, OK_GREEN = fs.BLUE, fs.ORANGE, fs.GREEN
OK_VERM, OK_SKY, OK_PINK = fs.VERM, fs.SKY, fs.PINK

fs.apply()
# The body figure is three panels in one column width, so its type is a notch
# smaller than the appendix figures while staying well above print minimum.
plt.rcParams.update({"axes.labelsize": 7.8, "axes.titlesize": 7.9,
                     "xtick.labelsize": 7.3, "ytick.labelsize": 7.3,
                     "legend.fontsize": 7.0})
ANNOT = 7.0

fig = plt.figure(figsize=(fs.BODY_WIDTH, 2.75), constrained_layout=True)
outer = fig.add_gridspec(1, 3, width_ratios=[0.86, 1.5, 1.34])

# RQ1: ranking and valuation under the same supplier-disjoint evaluations.
ax = fig.add_subplot(outer[0, 0])
evaluations = headline["evaluations"]
r2 = np.array([row["diagnostic_r2"] for row in evaluations])
pairwise = np.array([row["diagnostic_pairwise_accuracy"] for row in evaluations])
ax.scatter(r2, pairwise, color="0.72", s=26, zorder=3, **fs.marker_kw())
fs.ref_line(ax, 0, "x")
fs.ref_line(ax, .5, "y")
ax.scatter([r2.mean()], [pairwise.mean()], marker="*", s=120, color=OK_VERM,
           zorder=5, **fs.marker_kw())
ax.annotate("mean", (r2.mean(), pairwise.mean()), xytext=(5, 5),
            textcoords="offset points", fontsize=ANNOT + .4, color=OK_VERM,
            fontweight="bold", fontstyle="italic")
ax.set(xlabel=r"Value accuracy ($R^2$)", ylabel="Pairwise accuracy")
fs.panel_title(ax, "(a) Held-out accuracy")
fs.minor_ticks(ax)
ax.set_xlim(-.33, .33)
ax.set_ylim(.48, .67)

# RQ2: the measured depth law, one curve per candidate-pool composition.
ax = fig.add_subplot(outer[0, 1])
styles = {
    "uniform": ("uniform", OK_BLUE, "o", "-"),
    "supplier_equal": ("supplier-eq.", OK_GREEN, "s", "--"),
    "local": ("3 suppliers", OK_ORANGE, "D", "-."),
}
sizes = pools["metadata"]["candidate_sizes"]
# Marker area encodes pool size (stated in the caption), so the sweep that generates the
# depth variation is visible without a second figure. The fitted slope and its R^2 are printed next to
# each rule: within a composition rule the offset C is fixed, so this line IS the
# test of Proposition 1, and it belongs in the body rather than the appendix.
fitted = []
for sampler, (label, color, marker, linestyle) in styles.items():
    depth = np.array([
        pools["cells"][f"{sampler}_n{size}"]["policies"]["argmin"]
        ["selection_depth_log"]["mean"] for size in sizes
    ])
    gap = np.array([
        pools["cells"][f"{sampler}_n{size}"]["policies"]["argmin"]
        ["mean"]["mean"] for size in sizes
    ])
    fit = np.polyfit(depth, gap, 1)
    resid = gap - np.polyval(fit, depth)
    r2 = 1.0 - resid.var() / gap.var()
    grid = np.linspace(depth.min() - .010, depth.max() + .010, 25)
    ax.plot(grid, np.polyval(fit, grid), color=color, lw=1.5, ls=linestyle)
    ax.scatter(depth, gap, s=np.array(sizes) * 1.5 + 14, marker=marker,
               color=color, label=label, zorder=3, **fs.marker_kw())
    fitted.append((label, fit[0], r2, color))
fs.ref_line(ax, 0, "y")
beta = uncertainty["diagnostic_summary"]["calibration_beta"]["mean"]
# One boxed legend carries the fit readout, so the numbers never sit on top of data.
from matplotlib.lines import Line2D
handles = [Line2D([], [], color=color, marker=styles[key][2], ls=styles[key][3],
                  lw=1.5, ms=5.5, mec="black", mew=fs.EDGE_LW)
           for key, (_, _, _, color) in zip(styles, fitted)]
labels = [rf"{name}: $\hat b={slope:.2f}$, $R^2={r2:.2f}$"
          for name, slope, r2, _ in fitted]
legend = ax.legend(handles, labels, loc="upper left", fontsize=ANNOT - .3,
                   title=rf"predicted slope $1-\hat{{\beta}}={1-beta:.2f}$",
                   title_fontproperties={"weight": "bold", "size": ANNOT - .3},
                   handlelength=2.2, borderpad=.45, labelspacing=.4,
                   frameon=True, edgecolor="black", fancybox=False, framealpha=1)
legend.get_frame().set_linewidth(1.0)
ax.set(xlabel=r"Selection depth $d_\pi$", ylabel=r"Claim gap $\Delta_\pi$")
fs.panel_title(ax, "(b) Depth mechanism")
fs.minor_ticks(ax)
ax.set_ylim(-.012, .205)

# RQ3: a direct magnitude comparison avoids hiding the reporting intervention
# behind a broken axis or making the selection methods illegible.
ax = fig.add_subplot(outer[0, 2])
paired = uncertainty["paired_vs_argmin"]
# Labels match the appendix policy names exactly, so a reader moving between
# the body figure and Appendix E does not have to re-map the rules.
shown = [
    ("risk_averse", "risk averse", OK_SKY, "selection"),
    ("conformal_select", "conformal select", OK_BLUE, "selection"),
    ("score_perturbation", "score perturbation", OK_ORANGE, "selection"),
    ("calibrated_report", "calibrated report", OK_PINK, "report"),
    ("conformal_report", "conformal report", OK_GREEN, "report"),
]
ypos = np.arange(len(shown))[::-1]
fs.ref_line(ax, 0, "x")
for y, (name, label, color, family) in zip(ypos, shown):
    row = paired[name]
    value = row["gap_reduction"]["mean"]
    spread = row["gap_reduction"]["sd"]
    cost = row["advantage_cost"]["mean"]
    ax.barh(y, value, height=.58, color=color, zorder=2, **fs.bar_kw())
    ax.errorbar(value, y, xerr=spread, fmt="none", ecolor="black",
                elinewidth=.8, capsize=2.0, capthick=.8, zorder=3)
    if name == "conformal_report":
        ax.text(value, y + .27, f"{value:.3f}", va="bottom",
                ha="center", fontsize=ANNOT + .6, color=color, fontweight="bold")
    else:
        ax.text(.035, y, f"{value:.3f}", va="center", ha="left",
                fontsize=ANNOT, color=color, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", pad=.15, alpha=.88))
    ax.text(.58, y, f"{cost:+.3f}", va="center", ha="right",
            fontsize=ANNOT, color="0.30")
ax.axhline(1.5, color="0.82", lw=.65)
ax.text(.58, 4.58, "adv. cost", fontsize=ANNOT, color="0.30", ha="right",
        fontweight="bold", fontstyle="italic")
ax.set_yticks(ypos)
ax.set_yticklabels([entry[1] for entry in shown])
ax.set_xlim(-.018, .625)
ax.set_ylim(-.55, 4.82)
ax.set_xlabel(r"Claim-gap reduction vs. $\arg\min$")
fs.panel_title(ax, "(c) Selection vs. reporting")
fs.minor_ticks(ax, y=False)

fs.save(fig, OUT)
