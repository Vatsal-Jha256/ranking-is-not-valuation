"""Manufacturer-cluster bootstrap.

Rows from one manufacturer are not independent: they share a plant, a mix design and
a reporting practice. Resampling rows would therefore report a standard error far too
small. Resampling *manufacturers* is the right unit, and it has one trap, which this
module exists to avoid.

Drawing manufacturer names with replacement and then selecting rows by set membership
throws away multiplicity: a manufacturer drawn twice contributes exactly as much as
one drawn once, which is not a bootstrap of anything. The implementation below draws
multinomial counts and tiles each cluster's rows by its count, so a manufacturer drawn
twice really does appear twice. This is the defect that kept an earlier design out of
this project's evidence base, and it is recorded in the experiment inclusion audit.

Supplies: every standard error attached to a group-conditional slope or a claim-gap
comparison in Appendix C and Appendix D.
"""
from __future__ import annotations

import numpy as np


def cluster_bootstrap_se(values: np.ndarray, groups: np.ndarray, statistic,
                         n_resamples: int, seed: int) -> float:
    """Cluster bootstrap standard error that preserves resample multiplicity.

    Args:
        values: per-row quantities the statistic reads (kept for call-site clarity;
            the statistic receives row indices, not values).
        groups: manufacturer label per row -- the cluster identity.
        statistic: callable mapping an array of row indices to a scalar.
        n_resamples: number of bootstrap resamples to attempt.
        seed: bootstrap seed.

    Returns:
        The standard deviation of the resampled statistic, or NaN when fewer than 20
        resamples produced a finite value -- reported rather than silently imputed.
    """
    values = np.asarray(values)
    groups = np.asarray(groups)
    unique, inverse = np.unique(groups, return_inverse=True)
    index_by_cluster = [np.flatnonzero(inverse == k) for k in range(len(unique))]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_resamples):
        multiplicity = rng.multinomial(len(unique), np.full(len(unique), 1.0 / len(unique)))
        rows = np.concatenate([
            np.tile(index_by_cluster[k], count)
            for k, count in enumerate(multiplicity) if count and len(index_by_cluster[k])
        ]) if multiplicity.any() else np.zeros(0, dtype=int)
        if len(rows) < 10:
            continue
        stat = statistic(rows)
        if np.isfinite(stat):
            draws.append(float(stat))
    if len(draws) < 20:
        return float("nan")
    return float(np.std(draws, ddof=1))
