# Protocol addendum: corrected beta-shrinkage and knapsack reruns

Date fixed: 2026-08-09, after the accepted v4 submission.

## Scope

This addendum governs two experiments that were explicitly excluded from the
accepted paper by `reproducibility/EXPERIMENT_INCLUSION_AUDIT.md`:
`exp_beta_shrinkage.py` and `exp_knapsack.py` / `exp_knapsack_hetero.py`. It
does not modify the accepted submission's two authoritative experiments
(`exp_submission_harmonized_v4.py`, `exp_model_robustness_v3.py`) or any
already-accepted claim in `paper/main.tex`. Results produced under this
addendum are additional robustness material; they do not retroactively alter
the accepted headline numbers.

## Reason for exclusion (recap)

- `exp_beta_shrinkage.py`'s manufacturer bootstrap resampled manufacturer
  names with replacement via `rng.choice(..., replace=True)`, then reduced
  the resample to `np.isin(m, chosen)` set membership before regressing. A
  manufacturer drawn twice contributed the same as one drawn once: the
  resample's multiplicity was discarded, which understates the bootstrap
  variance a cluster bootstrap is supposed to estimate.
- Its population prior (`BETA0, SIGMA_POOL = 0.584, 0.207`) was a hardcoded
  constant taken from a separate, frozen, disconnected script
  (`exp_final_robust.json`), not reproducible or provenance-linked to the
  run being evaluated.
- `exp_beta_shrinkage.py`, `exp_knapsack.py`, and `exp_knapsack_hetero.py` all
  partitioned manufacturers with `splits.balanced_group_blocks`, which uses
  its seed only to break ties among manufacturers with identical row counts.
  Because manufacturer row counts are almost all distinct, different seeds
  produced nearly-invariant blocks (the same defect the v4 addendum fixed for
  the headline experiment).
- All three scripts built features with
  `pilot_optimizers_curse.build_features`, which imputes missing values with
  `X.fillna(X.median(numeric_only=True))` computed over the full dataset
  before any split -- mild cross-fold leakage, inconsistent with the accepted
  paper's fold-local imputation.
- The knapsack experiments' "certified" budget objective used an ad hoc
  class-level band, `(1 - beta) * std_class * a_n`, unconnected to the
  calibration/selection-depth accounting (`calibration_residual_log`) that
  RQ2 of the accepted paper actually validates.

## Corrections applied

Both `research/exp_beta_shrinkage_v2.py` and `research/exp_knapsack_v2.py`:

1. Partition manufacturers with `exp_submission_harmonized_v4.stratified_random_group_blocks`
   (seeded, size-stratified, genuinely seed-sensitive), never
   `splits.balanced_group_blocks`.
2. Build features with `features.build_features_raw` and impute inside each
   training fold only (`exp_submission_harmonized_v4.fit_model`'s
   `SimpleImputer` pipeline), never a globally-fit median.
3. Keep every split role (train / calibration / kappa-tune / eval, or train /
   calibration / test) manufacturer-disjoint, checked programmatically before
   any metric is computed.
4. Run across all six v4 partition seeds, not a single seed, and record every
   partition digest.

`exp_beta_shrinkage_v2.py` additionally:

- Replaces the membership-mask bootstrap with a manufacturer-cluster
  bootstrap that draws per-manufacturer resample COUNTS from a multinomial
  and weights each manufacturer's contribution to the calibration regression
  by `(equal-weight base) * (resample count)`, preserving multiplicity.
- Replaces the hardcoded prior with a leave-one-seed-out prior: the prior
  used to shrink seed `s`'s beta_hat is the mean/SD of beta_hat over the
  OTHER five seeds' calibration blocks only, never seed `s`'s own estimate.
  No independent external corpus exists for this problem (EC3 readymix
  concrete is the only data source), so a same-corpus prior is unavoidable;
  what changes is that it is now reproducible, provenance-hashed, and
  strictly disjoint from the fold it shrinks.
- Adds a fourth disjoint role (`kappa_tune`) so Proposition 1's `kappa` is
  tuned on data untouched by both the calibration-beta estimate and the
  final `eval` measurement.

`exp_knapsack_v2.py` additionally:

- Replaces the ad hoc certified-objective band with the CalibrationMap's own
  `alpha_hat + beta_hat * score`, i.e. exactly the calibrated prediction the
  accepted paper's RQ2 residual accounting already uses, evaluated at the
  chosen candidate rather than as a separate class-level constant.
- Uses the CLF WBLCA portfolio (`exp_slate.clf_concrete`) for line-item mass,
  baseline declared GWP, and project structure ONLY -- never for training or
  for GWP ground truth, consistent with `EXPERIMENT_INCLUSION_AUDIT.md`'s
  existing rule that CLF supplies project structure only.

## Reporting rule

Results are same-corpus partition-sensitivity summaries across the six v4
seeds, not independent-data confidence intervals, exactly as the v4 addendum
already states for the headline experiment. Both scripts report every
attempted (seed, fold/block) unit, including failures, and raise if any
partition role manufacturer sets overlap.
