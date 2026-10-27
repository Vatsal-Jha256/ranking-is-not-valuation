# Protocol: review-response experiment suite v1

Date prespecified: 2026-08-10, after the accepted v4 submission and before any of the
four experiments below were executed.

## Scope

Four new experiments answer specific gaps identified in external review of the accepted
manuscript. None of them modifies the two authoritative experiments behind the paper's
headline RQ1-RQ3 numbers, and none of them may retroactively alter an accepted claim. All
four inherit the accepted v4 protocol wherever the accepted protocol already fixes a
choice:

- manufacturer partitions from `stratified_random_group_blocks` with six seeds and
  size-stratified assignment (seed-sensitive by construction);
- fold-local imputation via `build_features_raw` plus a training-fold `SimpleImputer`;
- disjoint train / calibration / test manufacturer roles within every evaluation;
- the accepted gradient-boosting estimator (300 depth-3 trees, learning rate 0.05,
  subsample 0.8) with seed equal to the partition seed;
- natural-log outcomes throughout; candidate comparisons always within strength class;
- every seed/fold evaluation reported, successes and failures alike.

Departures from the accepted protocol are listed per experiment below and nowhere else.

## E1. Calibration structure and the full-vector premise

Question. Proposition 1 assumes `E[T_i | X] = alpha + beta X_i` for every candidate `i`
jointly (full-vector conditional calibration). The accepted paper states this is stronger
than scalar calibration and is not verified. E1 verifies it, and measures whether the
slope varies by group.

Design, per (seed, fold), with the accepted role assignment (two training blocks, one
calibration block, one test block):

1. Fit the global scalar calibration map on the calibration block
   (manufacturer-equal-weighted least squares, the accepted estimator).
2. Fit group-conditional calibration maps on the same block for three prespecified
   groupings: strength class; spatial cluster (k-means with k=6 on training-block
   manufacturer latitude/longitude, cluster centres fit on the training block only and
   applied to the calibration block); and manufacturer-size tercile (terciles of retained
   record count, cut points from the training block). A group is eligible when it holds at
   least 200 calibration rows from at least 2 manufacturers.
3. Full-vector premise test. Draw 1,000 within-class candidate sets of ten from the
   calibration block. For every member of every set compute the global calibration
   residual `e_i = T_i - (alpha + beta X_i)`. Regress `e_i` on three set-level features of
   the *other* nine candidates: their mean score, their minimum score, and the normalized
   within-set rank of `i`. Under the full-vector premise every coefficient is zero.
   Standard errors come from a manufacturer-cluster bootstrap with 200 resamples that
   preserves resample multiplicity. Also report the mean residual at each within-set rank
   position 1..10, which is flat at zero under the premise.
4. Report-rule comparison on the test block. Draw shared within-class candidate sets of
   ten, select by `argmin` of the raw score, and score three report rules against the same
   selection: the raw score, the globally calibrated value, and the class-conditionally
   calibrated value. Record signed log claim gap and the frequency of item-level
   overstatement above 0.05 log units.

Prespecified failure conditions: fewer than 24 completed evaluations; any manufacturer
appearing in two roles; any non-finite reported metric.

## E2. Uncertainty-aware selection and reporting, with tail risk

Question. The accepted paper compares only randomized softenings of `argmin`. Review asked
for uncertainty-aware baselines, and for the distribution rather than the mean of
item-level overstatement.

Design, per (seed, fold), on the accepted role assignment and the same shared candidate
sets used by all policies within an evaluation:

1. Fit a heteroscedastic uncertainty model on the calibration block only: gradient boosting
   (100 depth-3 trees, learning rate 0.05) regressing `log|T - X|` on the same features,
   giving `s(x) = exp(g(x))` clipped to the calibration block's 1st-99th percentile range.
2. Normalized split-conformal quantile `q` at 90% from the calibration block's
   `|T - (alpha + beta X)| / s(x)`, finite-sample corrected.
3. Selection-conditional conformal quantile `q_sel` at 90%, calibrated by running `argmin`
   selection over calibration-block candidate sets and taking the corrected quantile of the
   winners' `(T - (alpha + beta X)) / s(x)` only.
4. Policies on the test block. Selection rules: `argmin`; `risk_averse` (`argmin` of
   `X_i + 0.5 s_i`); `conformal_select` (`argmin` of `X_i + q s_i`); `thompson`
   (`argmin` of `X_i + eps_i`, `eps_i ~ N(0, s_i^2)`). Report rules applied to the
   `argmin` selection: raw score (`argmin` itself), `calibrated_report`
   (`alpha + beta X_W`), and `conformal_report` (`alpha + beta X_W + q_sel s_W`).
5. Metrics per policy: mean signed log claim gap, selection depth, fractional
   declared-GWP advantage relative to the class median, and the item-level overstatement
   distribution: P(gap > 0), P(gap > 0.05), P(gap > 0.10), the 90th percentile of the gap,
   and the mean of the worst decile (CVaR at 90%).

The three report rules share the `argmin` selection exactly, so their selection depth and
declared-GWP advantage are identical to `argmin` by construction; this is checked, not
assumed.

Prespecified failure conditions: as E1, plus failure of the pointwise selection-depth
bound for `argmin` among the four selection rules.

## E3. Candidate-set size and composition sensitivity

Question. The accepted paper fixes candidate sets at ten products sampled uniformly within
strength class. Review asked how conclusions move with set size and with non-uniform,
market-weighted or availability-limited pools.

Design, per (seed, fold): the cross product of candidate-set size `n` in {2, 5, 10, 20, 50}
and three samplers, each drawing 400 sets per eligible strength class:

- `uniform`: the accepted sampler, uniform without replacement within class;
- `share_weighted`: probability proportional to the manufacturer's retained record count in
  the test block, a within-corpus proxy for market share, without replacement;
- `local`: sets restricted to at most three manufacturers drawn uniformly from the class's
  test-block manufacturers, mimicking a limited local supply radius.

Policies scored: `argmin`, `score_perturbation` (the accepted Gaussian softening at the
calibration-fold scale), and `calibrated_report`. Metrics as in E2. A class is eligible for
a given `n` when it holds at least `3n` test rows, and for `local` when it also holds at
least three test-block manufacturers.

The record-count proxy is not market share; it is publication volume in one registry, and
is labelled as such wherever it is reported.

## E4. Temporal holdout

Question. The accepted study is a single-registry, single-snapshot design; review noted the
absence of a temporal holdout as an external-validity gap.

Design. Departure from the accepted protocol: partitions are by declaration date, not by
seeded manufacturer blocks. For each cutoff year in {2022, 2023, 2024}, train on all
records first declared before 1 January of that year and test on records first declared on
or after it. Two test variants per cutoff:

- `temporal`: all later records, so manufacturers may recur across the cutoff;
- `temporal_disjoint`: later records from manufacturers with no pre-cutoff record, which
  compounds temporal drift with supplier shift.

For each cutoff and variant report test R-squared, within-class pairwise accuracy under the
accepted pair-draw procedure, the scalar calibration slope fitted on a held-out calibration
split of the test period, and the `argmin` claim gap and declared-GWP advantage on
candidate sets of ten. A cutoff/variant is eligible when the test period holds at least
2,000 records and at least one eligible strength class.

Prespecified failure conditions: fewer than 6 completed cutoff/variant evaluations; any
non-finite reported metric.

## Reporting rules

Every number that reaches the manuscript comes from the recorded result ledgers, is checked
mechanically before any manuscript edit lands, and is reproduced inside the locked
container by the single-command workflow. Percentiles and standard deviations describe
split/fold sensitivity within one corpus and are never described as independent-data
confidence intervals.
