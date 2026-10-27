"""The decision rules being compared, and the constants that define them.

Keeping the vocabulary here rather than in one experiment's file is what lets the pool
sweep and the policy comparison talk about the same ``risk_averse`` rule with the same
lambda, instead of two rules that happen to share a name.

Two families, and the distinction is the paper's practical point:

Selection rules choose the product. They move selection depth, and therefore they move
both the claim gap and the realized advantage -- you cannot buy honesty from them for
free.

Reporting rules choose the number attached to an unchanged ``argmin`` recommendation.
Their advantage cost is zero by construction, so the only question about them is how
much of the gap, and of its upper tail, they remove.

Supplies: the policy frontier of RQ3 and Figure 1 of the manuscript, and Appendix D.
"""
from __future__ import annotations

import numpy as np

#: Candidate pool size in the accepted protocol.
N_CANDIDATES = 10

#: Weight on the predicted scale in the risk-averse selection rule, ``mu + lambda * s``.
RISK_LAMBDA = 0.5

#: The randomized softenings of argmin compared in the main text.
HEADLINE_POLICIES = (
    "argmin", "softmax", "eps_greedy", "topk_uniform", "score_perturbation",
)

#: Uncertainty-aware rules that change which product is recommended.
SELECTION_POLICIES = ("argmin", "score_perturbation", "risk_averse",
                      "conformal_select", "thompson")

#: Rules that change only the number reported for the argmin recommendation.
REPORT_POLICIES = ("calibrated_report", "conformal_report")


def choose(policy: str, scores: np.ndarray, sigma: float,
           rng: np.random.Generator) -> int:
    """Pick a winner from one candidate set under a headline policy.

    Args:
        policy: one of :data:`HEADLINE_POLICIES`.
        scores: predicted log GWP for each candidate; lower is better.
        sigma: calibration residual scale, used by ``score_perturbation``.
        rng: generator owned by the caller, so a policy's randomness is reproducible
            from the fold identifier.

    Returns:
        Index of the chosen candidate within ``scores``.
    """
    if policy == "argmin":
        return int(np.argmin(scores))
    if policy == "softmax":
        weights = np.exp(-(scores - scores.min()) / 0.05)
        return int(rng.choice(len(scores), p=weights / weights.sum()))
    if policy == "eps_greedy":
        return (int(rng.integers(len(scores))) if rng.random() < 0.2
                else int(np.argmin(scores)))
    if policy == "topk_uniform":
        return int(rng.choice(np.argsort(scores)[:3]))
    if policy == "score_perturbation":
        return int(np.argmin(scores + rng.normal(0.0, sigma, len(scores))))
    raise ValueError(policy)
