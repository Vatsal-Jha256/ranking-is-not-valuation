# Protocol: Submission Metric Harmonization and Policy Sensitivity

Written before the run on 2026-08-09.

## Purpose

Resolve the inconsistent use of "win rate" in the current paper and quantify
whether the policy-frontier conclusions persist under different balanced
manufacturer partitions.

## Hypotheses

1. A gradient-boosted GWP regressor has weak out-of-manufacturer value accuracy
   but above-chance within-strength-class pairwise ordering.
2. Pairwise ordering and median agreement are distinct metrics; reporting both
   from the same predictions removes the current cross-script ambiguity.
3. For candidate sets of size 10, deterministic argmin has the largest
   selection depth. Stochastic policies reduce average over-claim while also
   reducing realized GWP improvement.

## Data and leakage controls

- Source: the locally cached EC3 ready-mix records loaded by `normalize.py`.
- Functional-equivalence gate: only records with a recognized 28-day strength
  class are retained; all comparisons occur within that class.
- Split unit: manufacturer. Each run uses four row-count-balanced manufacturer
  blocks. Test and calibration manufacturers are disjoint from training and
  from each other.
- Split sensitivity: six independently seeded balanced block assignments,
  giving 24 fold-seed evaluations. These are sensitivity replications over the
  same corpus, not independent new datasets; percentile ranges are therefore
  labeled sensitivity intervals, not confidence intervals.

## Model

`GradientBoostingRegressor` with 300 estimators, depth 3, learning rate 0.05,
subsample 0.8. The random state equals the split seed. Features are exactly the
existing `build_features` design used by the paper.

## Metrics

- Standard test-set $R^2$ on log GWP.
- Mean within-class Spearman correlation.
- Pairwise win rate: probability that the prediction correctly orders two
  distinct, uniformly sampled candidates in the same strength class.
- Median agreement: probability that predicted-below-class-median agrees with
  true-below-class-median for a uniformly sampled candidate.
- Policy over-claim: $100(\bar y_W-\bar{\hat y}_W)/\bar{\hat y}_W$ after
  returning selected predictions and truths to natural GWP units.
- Realized quality: mean fractional reduction relative to the true class
  median, computed per selected item before averaging.
- Selection depth: candidate-set mean predicted log GWP minus selected
  predicted log GWP.

## Policies and controls

Candidate-set size is 10. Each fold evaluates argmin, softmax ($\tau=0.05$),
epsilon-greedy ($\epsilon=0.2$), top-3 uniform, and Thompson-style posterior
sampling. Every policy receives the same sampled candidate sets within a fold.
The calibration residual scale used by Thompson sampling is fitted only on the
manufacturer-disjoint calibration block.

## Sampling and reporting

- 20,000 pair draws per strength class and fold for pairwise win rate.
- 1,000 candidate sets per strength class and fold for the policy frontier.
- Save every successful fold-seed result and any failure record. Report means,
  standard deviations, fold-seed minima/maxima, and 2.5/97.5 percentiles across
  the 24 sensitivity evaluations.
- Do not overwrite prior result files. Output:
  `results/exp_submission_harmonized.json`.

## Failure criteria

- Any train/calibration/test manufacturer overlap.
- Any policy evaluated on different candidate sets from the others.
- Non-finite metric, empty eligible class, or output missing a fold-seed entry.
- If argmin does not maximize selection depth in any evaluation, treat that as
  an implementation failure because the bound is pointwise.

