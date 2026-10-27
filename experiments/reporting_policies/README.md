# Reporting policies

**Can an uncertainty-aware rule remove the claim gap, and at what cost?**

Protocol: [`../protocols/review_response_suite.md`](../protocols/review_response_suite.md)
(E2) · Runner: `run_policy_comparison.py` · Ledger:
`results/exp_uncertainty_policies_v1.json`

## Design

Five selection rules and two reporting rules, separated on purpose. A selection rule
changes which product is recommended, so it moves both the selection depth and the
realized advantage. A reporting rule changes only the number attached to an unchanged
`argmin` recommendation, so its advantage cost is zero by construction and the runner
asserts that the depth and advantage come out byte-identical to `argmin`.

Every rule is scored on the distribution of the gap, not its signed mean: the probability
of overstating at all, by more than five points, by more than ten, the 90th percentile,
and the mean of the worst decile.

## Result

| Rule | Gap removed | Advantage cost |
|---|---|---|
| score perturbation | +0.0092 ± 0.0443 | 0.0761 |
| risk averse (μ + λs) | +0.0045 ± 0.0111 | −0.0029 |
| conformal selection | +0.0129 ± 0.0319 | 0.0116 |
| Thompson (heteroscedastic) | +0.0027 ± 0.0323 | 0.0529 |
| calibrated report | −0.0029 ± 0.0292 | 0.0000 |
| **selection-conditional conformal report** | **+0.3431 ± 0.0800** | **0.0000** |

The selection rules are all buying the same thing at roughly the same rate, and none of
them buys much. The reporting rule is a different order of magnitude, because it is not
trading against the identity at all. It cuts the probability of overstating by more than
five points from 51.2% to 11.4% and the worst-decile mean from 0.485 to 0.170, while the
product recommended stays exactly the same.

Two results here are negative and are reported as such. Naive calibrated point reporting
is slightly *worse* than the raw score. And the conformal interval under-covers: 15.2% of
claims still exceed, against a 10% target. That shortfall is a measurement of supplier
shift, not a bug to be tuned away.

## What this does not show

Coverage is not validated on unseen suppliers — it is measured, and it falls short. Any
deployment quoting this interval has to quote the shortfall with it.
