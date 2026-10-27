# Model robustness

**Is the ranking-versus-valuation split a property of one estimator, or of the task?**

Protocol: [`../protocols/model_robustness.md`](../protocols/model_robustness.md) plus
addenda v2–v3 · Runner: `run_model_families.py` · Ledger:
`results/exp_model_robustness_v3.json`

## Design

Four model families over the identical partitions used by the headline experiment: a
linear baseline and three tree ensembles, 24 evaluations each, 96 in total. The
evaluation uses one held-out block, no calibration block, and no
policies — because the only question is whether the two capabilities separate.

## Result

| Model | R² | Pairwise accuracy | Above chance | R² < 0 |
|---|---|---|---|---|
| Ridge | +0.005 | 0.507 | 13/24 | 9/24 |
| HistGB | +0.002 | 0.577 | 23/24 | 11/24 |
| ExtraTrees | +0.062 | 0.575 | 24/24 | 7/24 |
| GBDT | +0.099 | 0.594 | 23/24 | 5/24 |

Every nonlinear family shows the same pattern: ordering well above chance in almost every
evaluation, R² near zero and frequently negative. The linear baseline fails at both,
which is the useful control — it rules out the reading that within-class ordering is
trivially easy.

## What this does not show

Four families is not the space of models. A different architecture, or features beyond
what a declaration publishes before a product is chosen, could value better. The claim
is narrower and survives: on this evidence the separation is not an artifact of the
estimator the paper happens to use.
