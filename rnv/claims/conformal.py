"""Distribution-free quantiles for what a recommendation is allowed to claim.

Split conformal gives a residual quantile that covers an exchangeable draw from the
calibration population. A recommended product is not such a draw. It was chosen for
having the most favourable estimate among several candidates, so its residual comes
from the upper tail of a selection-conditioned distribution, and a marginal quantile
under-covers it by construction.

:func:`selection_conformal_quantile` fixes that by running the same selection over
calibration candidate sets and keeping only the winners' residuals. The quantile then
targets the distribution the deployed claim actually faces. It depends on how many
candidates the winner beat, so pool size is a parameter here, not a constant.

Residuals are normalized by the heteroscedastic scale from
:func:`rnv.evaluation.model.fit_spread_model` before the quantile is taken, so the
resulting interval is wider for products the model finds hard and narrower for
products it finds easy, instead of being one width for everything.

Supplies: RQ3 and Appendix D of the manuscript -- the ``conformal_select`` selection
rule and the ``conformal_report`` reporting rule, including the residual exceedance
that is reported rather than hidden.
"""
from __future__ import annotations

import numpy as np

from rnv.evaluation.metrics import eligible_classes

#: Target miscoverage for the reported floor.
CONFORMAL_ALPHA = 0.10

#: Calibration candidate sets drawn per class when calibrating on the selection event.
N_CONFORMAL_SETS = 500

#: Accepted candidate pool size; the pool sweep overrides it deliberately.
N_CANDIDATES = 10


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample corrected split-conformal quantile.

    The ``ceil((n + 1)(1 - alpha)) / n`` level, not the plain ``1 - alpha`` empirical
    quantile: the correction is what makes the coverage guarantee finite-sample rather
    than asymptotic. Fewer than 50 scores is refused rather than silently reported,
    because the correction saturates at 1.0 and the quantile stops meaning anything.
    """
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 50:
        raise RuntimeError(f"need at least 50 conformal scores, got {len(values)}")
    level = min(1.0, np.ceil((len(values) + 1) * (1 - alpha)) / len(values))
    return float(np.quantile(values, level))


def selection_conformal_quantile(pred: np.ndarray, truth: np.ndarray, spread: np.ndarray,
                                 classes: np.ndarray, alpha: float, beta: float,
                                 level: float, rng: np.random.Generator,
                                 n_candidates: int = N_CANDIDATES) -> float:
    """Calibrate the quantile on the selection event, not on the marginal population.

    Args:
        pred, truth, spread: calibration-block predictions, log declared GWP, and
            per-product scale.
        classes: strength class per calibration row.
        alpha, beta: the fitted calibration map, applied before the residual is taken.
        level: target miscoverage passed to :func:`conformal_quantile`.
        rng: generator for the calibration candidate sets.
        n_candidates: pool size the winner is selected from; must match the pool size
            the deployed rule will face.
    """
    residuals = []
    for label in eligible_classes(classes, 3 * n_candidates):
        mask = classes.astype(str) == label
        p = pred[mask]
        t = truth[mask]
        s = spread[mask]
        for _ in range(N_CONFORMAL_SETS):
            idx = rng.choice(len(p), n_candidates, replace=False)
            winner = int(np.argmin(p[idx]))
            row = idx[winner]
            residuals.append((t[row] - (alpha + beta * p[row])) / s[row])
    return conformal_quantile(np.asarray(residuals), level)
