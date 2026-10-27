# Pool sensitivity

**How much of the result depends on how the candidate pool was built?**

Protocol: [`../protocols/review_response_suite.md`](../protocols/review_response_suite.md)
(E3) · Runner: `run_pool_sweep.py` · Ledger:
`results/exp_candidate_sensitivity_v1.json`

## Design

Five pool sizes (2, 5, 10, 20, 50) crossed with three composition rules, 15 cells over 24
evaluations. The accepted protocol fixes ten products drawn uniformly within a strength
class, and both halves of that choice are consequential: pool size moves the selection
depth the identity is linear in, and composition decides whose declarations the optimizer
may pick among.

The composition arm carries a caveat that governs how it can be read. This registry
records declarations, not sales, so it contains no market share. Uniform row sampling
already makes a supplier's chance of appearing proportional to how much it publishes,
which is a volume weighting. The sensitivity check brackets the accepted
sampler without assuming a market-share model:
supplier-equalized sampling removes the volume skew entirely, and a three-supplier local
pool tightens it the way a delivery radius would.

## Result

Mean claim gap under `argmin`:

| Sampler | n=2 | n=5 | n=10 | n=20 | n=50 |
|---|---|---|---|---|---|
| uniform | +0.041 | +0.052 | +0.054 | +0.064 | +0.076 |
| supplier-equalized | +0.003 | +0.026 | +0.045 | +0.071 | +0.086 |
| local (3 suppliers) | −0.004 | +0.010 | +0.022 | +0.028 | +0.032 |

The gap rises monotonically with pool size under every rule, which is what the identity
predicts and the reason the prediction is drawn on the figure as a prediction rather than
fitted to the points. The accepted configuration sits between the brackets rather than at
an extreme of them.

## What this does not show

None of these is a market-weighted pool, and this dataset cannot produce one. The sweep
establishes that the finding is not an artifact of one arbitrary pool, not that any
particular pool matches procurement practice.
