"""How a fold is scored: ranking quality, claim gaps, and split/fold summaries.

The project's central claim is that two different things get conflated when a
recommender is evaluated -- how well it *orders* products, and how well it *values*
them -- so the two families of metric are kept visibly apart here.

Ranking quality
    :func:`pairwise_accuracy`, :func:`within_class_spearman`,
    :func:`median_agreement`. All three are computed inside a strength class, because
    ordering products that are not functionally interchangeable is not a decision
    anyone makes.

Valuation quality
    :func:`r2_score_test_mean` against the held-out mean, and
    :func:`gap_distribution`, which reports the claim gap as a distribution rather
    than a signed mean. A signed mean hides offsetting errors: a policy whose claims
    are right on average can still overstate badly on a large minority of items, and
    that minority is what an auditor sees.

Summaries
    :func:`summarize` and :func:`summarize_compact` aggregate a metric over the 24
    seed/fold evaluations. Their percentiles describe sensitivity across splits of one
    corpus; they are not confidence intervals over independent data, and the ledgers
    say so in their own metadata.

Supplies: the headline diagnostic table (RQ1), the claim-gap distribution columns of
Appendix D, and every ``*_summary`` block in ``results/``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

#: Sampled pairs per strength class for the pairwise metrics. Sampling rather than
#: enumerating keeps the cost linear in the number of classes, and 20k draws puts the
#: Monte Carlo error two orders of magnitude below the differences being reported.
N_PAIR_DRAWS = 20_000

#: A class needs this many held-out rows before a within-class metric is computed.
MIN_CLASS_ROWS = 20


def eligible_classes(classes: np.ndarray, minimum: int) -> list:
    """Strength classes with enough rows, in a deterministic order."""
    labels, counts = np.unique(classes.astype(str), return_counts=True)
    return [str(label) for label, count in zip(labels, counts) if count >= minimum]


def r2_score_test_mean(truth: np.ndarray, pred: np.ndarray) -> float:
    """R^2 against the held-out block's own mean.

    Scoring against the test mean rather than the training mean is the strict choice:
    it asks whether the model beats the best constant an evaluator could have used
    after seeing the held-out block, so it can and does go negative.
    """
    denominator = np.sum((truth - truth.mean()) ** 2)
    return float(1.0 - np.sum((truth - pred) ** 2) / denominator)


def within_class_spearman(pred: np.ndarray, truth: np.ndarray,
                          classes: np.ndarray) -> float:
    """Mean within-class Spearman correlation over classes with enough rows."""
    values = []
    for label in pd.unique(classes):
        mask = classes == label
        if mask.sum() < MIN_CLASS_ROWS:
            continue
        rho, _ = spearmanr(pred[mask], truth[mask])
        if np.isfinite(rho):
            values.append(float(rho))
    if not values:
        raise RuntimeError("No eligible class for Spearman correlation")
    return float(np.mean(values))


def pairwise_accuracy(pred: np.ndarray, truth: np.ndarray, classes: np.ndarray,
                      rng: np.random.Generator,
                      n_draws: int = N_PAIR_DRAWS) -> tuple[float, float, int]:
    """Probability that the model orders two same-class products correctly.

    Pairs with tied truth are excluded, since they are not orderable. Pairs where the
    model itself ties are counted as half a win rather than dropped, so a degenerate
    model that predicts one constant scores 0.5 instead of being scored on nothing.

    Returns:
        ``(accuracy, prediction_tie_rate, n_pairs)``.
    """
    wins = 0
    total = 0
    ties = 0
    for label in pd.unique(classes):
        mask = classes == label
        p = pred[mask]
        t = truth[mask]
        if len(p) < MIN_CLASS_ROWS:
            continue
        i = rng.integers(0, len(p), n_draws)
        j = rng.integers(0, len(p), n_draws)
        valid = (i != j) & (t[i] != t[j])
        pred_tie = p[i[valid]] == p[j[valid]]
        correct = ((p[i[valid]] < p[j[valid]]) ==
                   (t[i[valid]] < t[j[valid]]))
        wins += float(np.sum(correct[~pred_tie]) + 0.5 * np.sum(pred_tie))
        total += int(valid.sum())
        ties += int(pred_tie.sum())
    if total == 0:
        raise RuntimeError("No eligible non-tied candidate pairs")
    return float(wins / total), float(ties / total), total


def median_agreement(pred: np.ndarray, truth: np.ndarray, classes: np.ndarray,
                     rng: np.random.Generator,
                     n_draws: int = N_PAIR_DRAWS) -> float:
    """Agreement on the coarsest useful question: is this product below class median?

    A specifier who only wants to know whether a product is better or worse than
    typical for its grade needs far less of the model than a full ordering does. This
    is the metric for that question.
    """
    wins = 0
    total = 0
    for label in pd.unique(classes):
        mask = classes == label
        p = pred[mask]
        t = truth[mask]
        if len(p) < MIN_CLASS_ROWS:
            continue
        idx = rng.integers(0, len(p), n_draws)
        wins += int(np.sum((p[idx] < np.median(p)) ==
                          (t[idx] < np.median(t))))
        total += len(idx)
    if not total:
        raise RuntimeError("No eligible median-agreement draws")
    return float(wins / total)


def gap_distribution(gaps: np.ndarray) -> dict[str, float]:
    """Item-level overstatement distribution, not just its mean.

    Positive gap means the declared outcome was worse than the reported one, i.e. the
    claim overstated the saving.
    """
    g = np.asarray(gaps, dtype=float)
    order = np.sort(g)
    tail_start = int(np.ceil(0.9 * len(order)))
    return {
        "mean": float(g.mean()),
        "sd": float(g.std(ddof=1)),
        "median": float(np.median(g)),
        "p_positive": float(np.mean(g > 0.0)),
        "p_over_005": float(np.mean(g > 0.05)),
        "p_over_010": float(np.mean(g > 0.10)),
        "q90": float(np.quantile(g, 0.90)),
        "q99": float(np.quantile(g, 0.99)),
        "cvar90": float(order[tail_start:].mean()),
        "n_items": int(len(g)),
    }


def summarize(values) -> dict[str, float]:
    """Split/fold summary with sensitivity percentiles.

    The percentiles describe how much a metric moves when the manufacturer partition
    moves. They are not confidence intervals: there is one corpus.
    """
    arr = np.asarray(list(values), dtype=float)
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "sensitivity_p2_5": float(np.percentile(arr, 2.5)),
        "sensitivity_p97_5": float(np.percentile(arr, 97.5)),
        "n": int(len(arr)),
    }


def summarize_compact(values) -> dict[str, float]:
    """Summary without percentiles, for ledgers that report them separately."""
    arr = np.asarray(list(values), dtype=float)
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(len(arr)),
    }
