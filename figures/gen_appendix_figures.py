#!/usr/bin/env python3
"""Draw supplementary result plots from the released experiment ledgers.
"""
from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import figstyle as fs

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

OK_BLUE, OK_ORANGE, OK_GREEN = fs.BLUE, fs.ORANGE, fs.GREEN
OK_VERM, OK_SKY, OK_PINK, OK_YELLOW = fs.VERM, fs.SKY, fs.PINK, fs.YELLOW

fs.apply()
WIDTH = fs.WIDTH


def load(name: str) -> dict:
    with (RESULTS / name).open() as handle:
        return json.load(handle)


def save(fig, name: str) -> None:
    fs.save(fig, HERE / name)


# ---------------------------------------------------------------------------
# A1  Auditing the conditional calibration premise.
# ---------------------------------------------------------------------------
def figure_premise_audit() -> None:
    ledger = load("exp_calibration_structure_v1.json")
    evaluations = ledger["evaluations"]
    profiles = ledger["profiles"]

    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.7), constrained_layout=True)

    ax = axes[0]
    groupings = [("strength_class", "strength\nclass"),
                 ("spatial_cluster", "spatial\ncluster"),
                 ("manufacturer_size", "supplier\nsize")]
    rng = np.random.default_rng(0)
    for position, (key, label) in enumerate(groupings):
        values = np.array([
            entry["beta"]
            for row in evaluations
            for entry in row["group_calibration"][key]["maps"].values()
        ])
        jitter = rng.uniform(-0.16, 0.16, len(values))
        ax.scatter(position + jitter, values, s=14, color=OK_SKY, zorder=2,
                   **fs.marker_kw(linewidth=.45))
        ax.plot([position - .3, position + .3], [values.mean()] * 2,
                color=OK_VERM, lw=2.0, zorder=4, solid_capstyle="butt")
    global_beta = ledger["summary"]["global_beta"]["mean"]
    fs.ref_line(ax, 1.0)
    ax.axhline(global_beta, color=OK_BLUE, lw=1.0, ls=":", zorder=1)
    ax.text(3.12, 1.0, r"$\beta{=}1$", color="0.35", fontsize=fs.ANNOT,
            va="bottom", ha="right")
    ax.text(3.12, global_beta - .06, f"pooled\n{global_beta:.2f}", color=OK_BLUE,
            fontsize=fs.ANNOT, va="top", ha="right", linespacing=1.1)
    ax.set_xlim(-.5, 3.15)
    ax.set_xticks(range(len(groupings)))
    ax.set_xticklabels([label for _, label in groupings])
    ax.set_ylim(-0.6, 2.1)
    ax.set(ylabel=r"Calibration slope $\beta$")
    fs.panel_title(ax, "(a) One slope does not fit")
    fs.minor_ticks(ax, x=False)

    ax = axes[1]
    mean = np.array(profiles["residual_mean_by_score_decile"]["mean"])
    sd = np.array(profiles["residual_mean_by_score_decile"]["sd"])
    x = np.arange(1, len(mean) + 1)
    fs.ref_line(ax, 0)
    ax.fill_between(x, mean - sd, mean + sd, color=OK_ORANGE, alpha=.25, linewidth=0)
    ax.plot(x, mean, marker="o", ms=5.4, color=OK_ORANGE, mec="black", lw=1.7)
    ax.annotate("flat at zero if\nthe premise holds", xy=(3.6, 0), xytext=(1.2, -0.175),
                fontsize=fs.ANNOT, color="0.35", fontstyle="italic",
                arrowprops=dict(arrowstyle="->", color="0.5", lw=.9))
    ax.set_xticks([1, 3, 5, 7, 10])
    ax.set(xlabel="Predicted-score decile", ylabel="Mean calibration residual")
    fs.panel_title(ax, "(b) The score enters nonlinearly")
    fs.minor_ticks(ax, x=False)

    ax = axes[2]
    raw = np.array(profiles["residual_mean_by_rank"]["mean"])
    raw_sd = np.array(profiles["residual_mean_by_rank"]["sd"])
    adjusted = np.array(profiles["adjusted_residual_mean_by_rank"]["mean"])
    adjusted_sd = np.array(profiles["adjusted_residual_mean_by_rank"]["sd"])
    x = np.arange(1, len(raw) + 1)
    fs.ref_line(ax, 0)
    ax.fill_between(x, raw - raw_sd, raw + raw_sd, color=OK_BLUE, alpha=.20, linewidth=0)
    ax.plot(x, raw, marker="o", ms=5.4, color=OK_BLUE, mec="black", lw=1.7,
            label="raw residual")
    ax.fill_between(x, adjusted - adjusted_sd, adjusted + adjusted_sd,
                    color=OK_GREEN, alpha=.20, linewidth=0)
    ax.plot(x, adjusted, marker="s", ms=5.2, color=OK_GREEN, mec="black", lw=1.7,
            label="net of own score")
    ax.scatter([1], [raw[0]], s=60, facecolor="none", edgecolor=OK_VERM, lw=1.0,
               zorder=6)
    ax.annotate("selected", xy=(1, raw[0]), xytext=(2.0, raw[0] - 0.055),
                fontsize=fs.ANNOT + .4, color=OK_VERM,
                fontweight="bold", fontstyle="italic",
                arrowprops=dict(arrowstyle="->", color=OK_VERM, lw=.9))
    ax.set_xticks([1, 3, 5, 7, 10])
    ax.set(xlabel="Rank within the candidate set",
           ylabel="Mean calibration residual")
    fs.panel_title(ax, "(c) Comparison set matters less")
    fs.minor_ticks(ax, x=False)
    fs.boxed_legend(ax, loc="lower left")

    save(fig, "figA1_premise_audit.pdf")


# ---------------------------------------------------------------------------
# A2  Uncertainty-aware selection against conservative reporting.
# ---------------------------------------------------------------------------
POLICY_STYLE = {
    "argmin": (r"$\arg\min$", "0.35", "o"),
    "score_perturbation": ("score perturbation", OK_SKY, "v"),
    "thompson": ("Thompson", OK_PINK, "P"),
    "risk_averse": (r"risk averse ($+\lambda s$)", OK_GREEN, "s"),
    "conformal_select": ("conformal select", OK_ORANGE, "D"),
    "calibrated_report": ("calibrated report", OK_BLUE, "^"),
    "conformal_report": ("conformal report", OK_VERM, "*"),
}


def figure_selection_vs_reporting() -> None:
    ledger = load("exp_uncertainty_policies_v1.json")
    paired = ledger["paired_vs_argmin"]
    summary = ledger["policy_summary"]

    # Two wide panels rather than four narrow ones. The body figure already shows
    # that the conformal report moves 25x further than any selection rule on a
    # single linear axis, so this figure does not need a broken axis to repeat it.
    # It shows instead what the body cannot: the selection trade-off at full
    # resolution, and what every rule does to the upper tail.
    # Panel (a) gets the width because it carries the trade-off argument. The two
    # tail metrics get one panel each: P(overstate) is a probability and the
    # worst-decile mean is in log GWP, so they must not share an x axis.
    fig, (trade, tail_rate, tail_size) = plt.subplots(
        1, 3, figsize=(WIDTH, 3.1),
        gridspec_kw={"width_ratios": [1.42, 0.80, 0.80]},
        constrained_layout=True)

    # ---- (a) the selection trade-off, at full resolution -------------------
    # Every rule that stays on scale is drawn here; the conformal report is an
    # order of magnitude to the right and is called out instead of compressed.
    # Labels sit clear of their own error bars, which are wide in x for the
    # randomized rules and would otherwise strike through the text.
    OFFSETS = {
        "score_perturbation": (0, 13, "center"),
        "thompson": (-11, -10, "right"),
        "risk_averse": (11, -8, "left"),
        "conformal_select": (11, 5, "left"),
        "calibrated_report": (-7, 13, "right"),
    }
    trade.axhspan(-0.03, 0, color=OK_GREEN, alpha=.055, linewidth=0)
    fs.ref_line(trade, 0, "y")
    fs.ref_line(trade, 0, "x", ls=":", lw=.7)
    for name, entry in paired.items():
        if name == "conformal_report":
            continue
        label, color, marker = POLICY_STYLE[name]
        x, y = entry["gap_reduction"], entry["advantage_cost"]
        trade.errorbar(x["mean"], y["mean"], xerr=x["sd"], yerr=y["sd"],
                       marker=marker, ms=7.6, color=color, ecolor="black", mec="black",
                       mew=fs.EDGE_LW, elinewidth=.7, capsize=2.0, lw=0, zorder=3)
        dx, dy, ha = OFFSETS[name]
        trade.annotate(label, (x["mean"], y["mean"]), xytext=(dx, dy),
                       textcoords="offset points", fontsize=fs.ANNOT, color=color,
                       ha=ha, va="center", zorder=5)
    # argmin is the origin of every paired comparison.
    trade.scatter([0.0], [0.0], marker="o", s=48, color="0.5", zorder=4,
                  **fs.marker_kw())
    trade.annotate(r"$\arg\min$", (0.0, 0.0), xytext=(-8, -12),
                   textcoords="offset points", fontsize=fs.ANNOT, color="0.3",
                   ha="right", va="center")
    trade.set_xlim(-0.068, 0.090)
    trade.set_ylim(-0.030, 0.140)
    trade.set(xlabel=r"Mean claim-gap reduction vs. $\arg\min$ (log GWP)",
              ylabel="Declared-GWP advantage given up")
    fs.panel_title(trade, "(a) Selection rules trade advantage for gap reduction",
                   fontsize=fs.TITLE-.4)
    fs.minor_ticks(trade)
    trade.text(0.086, -0.0235, "costs nothing", fontsize=fs.ANNOT, color=OK_GREEN,
               fontweight="bold", fontstyle="italic",
               ha="right", va="center")
    # The reporting rule is off this scale in x and exactly zero in y. It is
    # called out in the empty upper right rather than compressed into the panel.
    conformal = paired["conformal_report"]["gap_reduction"]["mean"]
    ratio = conformal / paired["conformal_select"]["gap_reduction"]["mean"]
    trade.annotate(
        "conformal report\n"
        f"{conformal:.3f} reduction, {ratio:.0f}$\\times$ further\n"
        "zero advantage given up",
        xy=(0.0885, 0.001), xytext=(0.086, 0.139), fontsize=fs.ANNOT, color=OK_VERM,
        ha="right", va="top", linespacing=1.35, fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=OK_VERM, lw=.8,
                        shrinkA=3, shrinkB=1,
                        connectionstyle="arc3,rad=-0.22"))

    # ---- (b), (c) what each rule does to the upper tail --------------------
    # Two separate axes because the quantities have different units: (b) is a
    # probability, (c) is a mean in log GWP. Putting them on one axis would
    # invite a comparison that is not defined.
    order = ["argmin", "thompson", "score_perturbation", "risk_averse",
             "conformal_select", "calibrated_report", "conformal_report"]
    positions = np.arange(len(order))
    colors = [POLICY_STYLE[name][1] for name in order]
    for ax, metric, title, xlabel, xmax in (
            (tail_rate, "p_over_005", "(b) How often it overstates",
             r"P(overstate $>0.05$)", 0.80),
            (tail_size, "cvar90", "(c) By how much, worst decile",
             "mean gap (log GWP)", 0.76)):
        values = [summary[name][metric]["mean"] for name in order]
        errors = [summary[name][metric]["sd"] for name in order]
        ax.barh(positions, values, height=.66, color=colors, xerr=errors,
                **fs.bar_kw())
        for position, value, error in zip(positions, values, errors):
            ax.text(value + error + 0.03, position, f"{value:.2f}", va="center",
                    ha="left", fontsize=fs.ANNOT, color="0.15", fontweight="bold")
        ax.set_yticks(positions)
        if metric == "p_over_005":
            ax.set_yticklabels([POLICY_STYLE[name][0] for name in order],
                               fontsize=fs.ANNOT)
        else:
            ax.tick_params(labelleft=False)
        ax.invert_yaxis()
        # The one rule that moves the tail is ruled off from the six that do not.
        ax.axhline(5.5, color="0.55", lw=.7)
        ax.set_xlim(0, xmax)
        ax.set_xticks([0, 0.25, 0.50])
        ax.set(xlabel=xlabel)
        fs.panel_title(ax, title, fontsize=fs.TICK)
        fs.minor_ticks(ax, y=False)

    save(fig, "figA2_selection_vs_reporting.pdf")


# ---------------------------------------------------------------------------
# A3  Candidate pool size and composition.
# ---------------------------------------------------------------------------
SAMPLER_STYLE = {
    "uniform": ("uniform (primary)", OK_BLUE, "o"),
    "supplier_equal": ("supplier-equalized", OK_GREEN, "s"),
    "local": ("three local suppliers", OK_ORANGE, "D"),
}


def figure_pool_sensitivity() -> None:
    ledger = load("exp_candidate_sensitivity_v1.json")
    cells = ledger["cells"]
    sizes = ledger["metadata"]["candidate_sizes"]
    # The depth-vs-gap regression that tests Proposition 1 now lives on the body
    # figure, so this figure carries only the two sweeps the body does not show.
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH * 0.8, 2.7),
                             constrained_layout=True)

    def series(sampler: str, metric: str, policy: str = "argmin"):
        mean, sd = [], []
        for size in sizes:
            entry = cells[f"{sampler}_n{size}"]["policies"][policy][metric]
            mean.append(entry["mean"])
            sd.append(entry["sd"])
        return np.array(mean), np.array(sd)

    for index, (metric, title, ylabel) in enumerate((
            ("mean", "(a) Deeper pools, larger claims", "Signed log claim gap"),
            ("realized_quality_fraction", "(b) And larger real advantage",
             "Declared-GWP advantage"))):
        ax = axes[index]
        fs.ref_line(ax, 0)
        for sampler, (label, color, marker) in SAMPLER_STYLE.items():
            mean, sd = series(sampler, metric)
            ax.fill_between(sizes, mean - sd, mean + sd, color=color, alpha=.16,
                            linewidth=0)
            ax.plot(sizes, mean, marker=marker, ms=5.8, color=color, mec="black",
                    lw=1.7, label=label)
        ax.set_xscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(size) for size in sizes])
        ax.minorticks_off()
        ax.set(xlabel="Candidates ranked", ylabel=ylabel)
        fs.panel_title(ax, title)
        fs.minor_ticks(ax, x=False)
        if index == 0:
            fs.boxed_legend(ax, loc="upper left")

    save(fig, "figA3_pool_sensitivity.pdf")


# ---------------------------------------------------------------------------
# A4  A different shift: holding out time instead of suppliers.
# ---------------------------------------------------------------------------
def figure_temporal_holdout() -> None:
    temporal = load("exp_temporal_holdout_v1.json")
    supplier = load("exp_submission_harmonized_v4.json")

    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.8),
                             gridspec_kw={"width_ratios": [1.28, 1.0, 1.0]},
                             constrained_layout=True)

    ax = axes[0]
    r2 = np.array([row["diagnostic_r2"] for row in supplier["evaluations"]])
    accuracy = np.array([row["diagnostic_pairwise_accuracy"]
                         for row in supplier["evaluations"]])
    ax.scatter(r2, accuracy, s=24, color="0.78", label="supplier holdout (24)",
               **fs.marker_kw(linewidth=.45))
    variant_style = {"temporal": (OK_BLUE, "o", "later declarations"),
                     "temporal_disjoint": (OK_VERM, "D", "later + unseen supplier")}
    for variant, (color, marker, label) in variant_style.items():
        rows = [row for row in temporal["evaluations"] if row["variant"] == variant]
        ax.scatter([row["test_r2"] for row in rows],
                   [row["pairwise_accuracy"] for row in rows],
                   s=56, color=color, marker=marker, label=label, zorder=4,
                   **fs.marker_kw())
        for row in rows:
            ax.annotate(str(row["cutoff_year"]),
                        (row["test_r2"], row["pairwise_accuracy"]),
                        textcoords="offset points", xytext=(4.5, -1.5),
                        fontsize=fs.ANNOT, color=color, fontweight="bold")
    fs.ref_line(ax, 0, "x")
    fs.ref_line(ax, .5, "y")
    ax.set(xlabel=r"Value accuracy ($R^2$)", ylabel="Pairwise accuracy")
    fs.panel_title(ax, "(a) The separation is not about suppliers")
    fs.minor_ticks(ax)
    # The panel is narrow in this three-up layout, so the legend has to be small
    # enough to sit inside it rather than running off the axis.
    fs.boxed_legend(ax, loc="lower right", handletextpad=.25, labelspacing=.25,
                    fontsize=fs.ANNOT, borderaxespad=.3)

    # The slope and the claim gap have different units and different zeros, so a
    # shared or twinned axis would invite a false visual comparison. They get one
    # panel each, on their own scale, with the reference line each one needs.
    rows = temporal["evaluations"]
    labels = [f"{int(row['cutoff_year']) % 100:02d}\n{'new' if 'disjoint' in row['variant'] else 'all'}"
              for row in rows]
    positions = np.arange(len(rows))
    bar_colors = [OK_VERM if "disjoint" in row["variant"] else OK_SKY for row in rows]

    ax = axes[1]
    ax.bar(positions, [row["calibration_beta"] for row in rows], .66,
           color=bar_colors, **fs.bar_kw())
    fs.ref_line(ax, 1.0)
    ax.annotate(r"$\beta=1$: no depth effect", (len(rows) - .5, 1.0),
                xytext=(0, 3), textcoords="offset points", fontsize=fs.ANNOT,
                fontstyle="italic",
                color="0.35", ha="right")
    ax.set_ylim(0, 1.22)
    ax.set_ylabel(r"Calibration slope $\beta$")
    fs.panel_title(ax, r"(b) Slope stays below one")
    fs.minor_ticks(ax, x=False)

    ax = axes[2]
    ax.bar(positions, [row["policy"]["raw"]["mean"] for row in rows], .66,
           color=bar_colors, **fs.bar_kw())
    ax.axhline(0, color="black", lw=.8)
    ax.set_ylim(-0.135, 0.055)
    ax.set_ylabel("Signed log claim gap")
    fs.panel_title(ax, "(c) Raw gap turns negative")
    fs.minor_ticks(ax, x=False)

    for ax in axes[1:]:
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=fs.ANNOT)
    fig.legend(handles=[
        Line2D([], [], marker="s", lw=0, ms=7.5, color=OK_SKY, mec="black",
               mew=fs.EDGE_LW, label="later declarations"),
        Line2D([], [], marker="s", lw=0, ms=7.5, color=OK_VERM, mec="black",
               mew=fs.EDGE_LW, label="later + unseen supplier")],
        frameon=True, edgecolor="black", fancybox=False, loc="lower center", ncol=2, fontsize=fs.ANNOT,
        bbox_to_anchor=(0.68, -0.045), handletextpad=.3, columnspacing=1.0)

    save(fig, "figA4_temporal_holdout.pdf")


# ---------------------------------------------------------------------------
# A5  The primary policy comparison, drawn instead of tabulated.
# ---------------------------------------------------------------------------
def figure_accepted_policies() -> None:
    ledger = load("exp_submission_harmonized_v4.json")
    summary = ledger["policy_summary"]
    order = ["argmin", "softmax", "eps_greedy", "topk_uniform", "score_perturbation"]
    labels = [r"$\arg\min$", "softmax", r"$\varepsilon$-greedy", r"top-$k$ uniform",
              "score perturbation"]
    colors = ["0.35", OK_ORANGE, OK_GREEN, OK_SKY, OK_BLUE]

    panels = (("over_claim_log", "(a) Signed log claim gap", "log GWP units"),
              ("selection_depth_log", "(b) Selection depth", "log GWP units"),
              ("realized_quality_fraction", "(c) Declared-GWP advantage",
               "fraction below class median"))

    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.5), constrained_layout=True)
    positions = np.arange(len(order))
    for ax, (metric, title, xlabel) in zip(axes, panels):
        values = [summary[name][metric]["mean"] for name in order]
        errors = [summary[name][metric]["sd"] for name in order]
        ax.barh(positions, values, height=.66, color=colors, xerr=errors,
                **fs.bar_kw())
        for position, value, error in zip(positions, values, errors):
            ax.text(value + error + max(values) * .06, position, f"{value:.3f}",
                    va="center", fontsize=fs.ANNOT, color="0.15", fontweight="bold")
        ax.set_yticks(positions)
        ax.set_yticklabels(labels if metric == "over_claim_log" else [])
        ax.invert_yaxis()
        ax.set(xlabel=xlabel)
        fs.panel_title(ax, title)
        fs.minor_ticks(ax, y=False)
        ax.set_xlim(0, max(v + e for v, e in zip(values, errors)) * 1.42)
    save(fig, "figA5_accepted_policies.pdf")


# ---------------------------------------------------------------------------
# A6  Supplementary checks: slope shrinkage and budgeted allocation.
# ---------------------------------------------------------------------------
def figure_supplementary_checks() -> None:
    shrinkage = load("exp_beta_shrinkage_v2.json")
    knapsack = load("exp_knapsack_v2.json")

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.8), constrained_layout=True)

    ax = axes[0]
    sizes = sorted(int(key) for key in shrinkage["evaluations"][0]["grids"])
    for estimator, label, color, marker in (("raw", r"raw $\hat\beta$", OK_BLUE, "o"),
                                            ("shrunk", "shrunk toward held-out seeds",
                                             OK_VERM, "s")):
        mean, sd = [], []
        for size in sizes:
            values = np.array([
                row["grids"][str(size)][estimator]["p_over_5pp"]
                for row in shrinkage["evaluations"]
            ]) * 100.0
            mean.append(values.mean())
            sd.append(values.std(ddof=1))
        mean = np.array(mean)
        sd = np.array(sd)
        ax.fill_between(sizes, mean - sd, mean + sd, color=color, alpha=.18, linewidth=0)
        ax.plot(sizes, mean, marker=marker, ms=5.8, color=color, mec="black",
                lw=1.7, label=label)
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(size) for size in sizes])
    ax.minorticks_off()
    ax.set(xlabel="Candidates ranked", ylabel="P(over-claim > 5 pp)  (%)")
    fs.panel_title(ax, "(a) Shrinkage helps on average, not everywhere")
    fs.minor_ticks(ax, x=False)
    fs.boxed_legend(ax, loc="lower left")

    ax = axes[1]
    budgets = sorted(int(key) for key in knapsack["budget_summary"])
    rules = (("point", "claimed score", OK_BLUE),
             ("certified", "calibration-corrected", OK_GREEN),
             ("mass", "physical mass", OK_ORANGE))
    width = 0.26
    positions = np.arange(len(budgets))
    for index, (rule, label, color) in enumerate(rules):
        share, error = [], []
        for budget in budgets:
            entry = knapsack["budget_summary"][str(budget)]
            oracle = entry["oracle"]["mean_realized_saving_kgco2e"]["mean"]
            share.append(100.0 * entry[rule]["mean_realized_saving_kgco2e"]["mean"]
                         / oracle)
            error.append(entry[rule]["pooled_over_claim_pct"]["mean"])
        offset = (index - 1) * width
        ax.bar(positions + offset, share, width, color=color, label=label,
               **fs.bar_kw())
        for position, value, over in zip(positions + offset, share, error):
            ax.text(position, value + 1.2, f"{over:.1f}", ha="center", fontsize=fs.ANNOT, fontweight="bold",
                    color="0.3", rotation=90)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"$k={budget}$" for budget in budgets])
    ax.set_ylim(0, 142)
    fs.ref_line(ax, 100)
    ax.text(-0.45, 101.5, "oracle", fontsize=fs.ANNOT, color="0.35", ha="left",
            fontweight="bold", fontstyle="italic")
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set(ylabel="Realized saving (% of oracle)")
    fs.panel_title(ax, "(b) Correcting the score does not reallocate")
    fs.minor_ticks(ax, x=False, y=False)
    fs.boxed_legend(ax, loc="upper left", ncol=3, columnspacing=.8, fontsize=fs.ANNOT)
    ax.text(0.5, 0.845, "rotated numbers: pooled over-claim (%)",
            transform=ax.transAxes, fontsize=fs.ANNOT, color="0.35", ha="center",
            fontstyle="italic")

    save(fig, "figA6_supplementary_checks.pdf")


if __name__ == "__main__":
    figure_premise_audit()
    figure_selection_vs_reporting()
    figure_pool_sensitivity()
    figure_temporal_holdout()
    figure_accepted_policies()
    figure_supplementary_checks()
