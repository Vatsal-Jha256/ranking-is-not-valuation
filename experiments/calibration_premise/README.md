# Calibration premise

Test the joint linear-calibration condition used in the depth decomposition.

Protocol: [`../protocols/review_response_suite.md`](../protocols/review_response_suite.md)
(E1) · Runner: `run_premise_audit.py` · Ledger:
`results/exp_calibration_structure_v1.json`

## Design

Proposition 1 needs `E[T_i | X] = α + βX_i` for *every candidate jointly*, not just for a
product considered alone. The audit refits the intercept and slope by least squares on the candidate-set
members. The residual therefore has mean zero and zero covariance with the
candidate's own score. Remaining patterns show departures from this linear fit.

Separate slopes are fitted by strength class, spatial cluster, and supplier-size
tercile to test whether a shared slope describes these groups.

## Result

The audit finds departures from the joint linear-calibration premise.

| | value |
|---|---|
| Slope range across strength classes | 0.762 |
| Slope range across spatial clusters | 1.013 |
| Slope range across size terciles | 0.298 |
| Residual range across predicted-score deciles | 0.217 log units |
| Residual range across ranks, net of own-score effects | 0.084 |
| Residual at the selected rank, adjusted | 0.006 ± 0.043 |

The group slopes differ from the pooled slope of 0.605. Residuals also curve with
the candidate's own score. After adjustment for own-score effects, dependence on
comparison-set rank is smaller and is indistinguishable from zero at the selected rank.

With class-conditional reporting, the mean gap falls from 0.056 to
0.037 and the probability of overstating by more than five points from 0.511 to 0.480.
Global recalibration does not help at all (0.059).

## What this does not show

These results do not validate an exact depth identity with a single global slope.
They motivate checking nonlinear and group-specific calibration.
