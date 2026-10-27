# Experiment inclusion audit

This note records which experiments support the active CCAI paper. It keeps research
history out of the manuscript while preventing superseded results from re-entering claims.

## Authoritative paper evidence

### Randomized supplier-partition experiment

- Generator: `experiments/selection_gap/run_selection_gap.py`
- Result: `results/exp_submission_harmonized_v4.json`
- Status: 24/24 successful evaluations; independently reproduced exactly.
- Design: six size-stratified randomized manufacturer partitions, four held-out blocks per
  partition, fold-local imputation, disjoint train/calibration/test manufacturers, common
  pair draws, and shared policy candidate sets.
- Scope: RQ1, scalar-beta diagnostics, RQ2 residual accounting, and RQ3 paired policies.

### Model-family robustness

- Generator: `experiments/model_robustness/run_model_families.py`
- Result: `results/exp_model_robustness_v3.json`
- Status: 96/96 successful model/fold evaluations; independently reproduced to numerical
  precision.
- Scope: Ridge, histogram boosting, ExtraTrees, and GBDT across the same six randomized
  manufacturer partitions.

### Review-response suite (2026-08-10)

Four experiments added after external review of the accepted manuscript, all under
`experiments/protocols/review_response_suite.md`, all using the shared `rnv` library,
which wraps the accepted v4 partition, role assignment, estimator, and fold-local
imputation so the new work cannot drift from the accepted protocol. All four are
manifest-pinned and rerun by the single-command workflow.

- `experiments/calibration_premise/run_premise_audit.py` -> `results/exp_calibration_structure_v1.json`.
  24/24 evaluations. Tests the full-vector conditional calibration premise of Proposition 1
  directly, and measures group-conditional calibration slopes. Finding: the premise fails.
  Refitted slope ranges average 0.762 across strength classes and 1.013 across spatial
  clusters against a pooled 0.605; the residual spans 0.217 log units across predicted-score
  deciles; comparison-set dependence net of own-score effects spans only 0.084 across ranks
  and is 0.006 +/- 0.043 at the selected rank. Supports the main text's RQ2 audit. Also
  finds that class-conditional report calibration helps (0.056 -> 0.037 mean gap) while
  global report calibration does not (0.059).
- `experiments/reporting_policies/run_policy_comparison.py` -> `results/exp_uncertainty_policies_v1.json`.
- `experiments/reporting_policies/run_stratified_quantile.py` -> `results/exp_stratified_quantile_v1.json` (E5, included in the camera-ready container workflow).
  24/24 evaluations. Adds uncertainty-aware selection rules and, separately, reporting
  rules applied to the identical argmin selection. Finding: reporting reduces the claim gap without changing the selected product.
  Selection-conditional conformal reporting removes 0.343 +/- 0.080 of mean gap and cuts
  P(overstatement > 0.05) from 0.512 to 0.114 at exactly zero advantage cost; its residual
  0.152 exceedance against a 0.10 target is reported as a measurement of supplier shift,
  not suppressed.
- `experiments/pool_sensitivity/run_pool_sweep.py` -> `results/exp_candidate_sensitivity_v1.json`.
  360 evaluations (24 folds x 5 pool sizes x 3 composition rules), 0 skipped. Finding: the
  claim gap rises monotonically with pool size under every composition rule, tracking the
  slope 1 - beta that Proposition 1 predicts. This is the strongest empirical support for
  the identity in the paper. The composition arm is explicitly a bracket around the
  accepted uniform sampler, not a market-share simulation; the corpus has no sales data.
- `experiments/temporal_holdout/run_temporal_holdout.py` -> `results/exp_temporal_holdout_v1.json`.
  6/6 evaluations. The one protocol departure in the paper: partitions cut on declaration
  date rather than seeded manufacturer blocks. Finding: R^2 negative in 4 of 6 variants
  while pairwise accuracy exceeds 0.5 in all 6 and beta stays in [0.468, 0.735]. The claim
  gap turns negative under this shift, which the paper reports rather than smoothing over.

## Included beta evidence

The paper includes the v4 scalar calibration-slope distribution and the proposition's
three beta regimes. It does not treat scalar beta as verification of full-vector
conditional calibration. The empirical score-perturbation comparison decomposes exactly
into a +0.03637 depth-derived component and a -0.02738 selected-residual component,
leaving +0.008996 mean signed-log gap reduction. This is the paper-safe beta result.

## Additional appendix-only evidence (corrected v2 reruns, 2026-08-09)

The two designs were corrected and rerun under the v4 partition and pairing protocol.
Both are supplementary results; neither changes the RQ1-RQ3 headline numbers. See `experiments/protocols/supplementary_reruns.md` for the exact
corrections applied.

### Empirical-Bayes calibration-slope shrinkage

- Generator: `experiments/supplementary/run_beta_shrinkage.py`
- Result: `results/exp_beta_shrinkage_v2.json`
- Status: 30/30 successful evaluations (6 seeds x 5 stratified blocks); independently
  reproduced bit-for-bit (ignoring timestamps) inside the locked container twice, content
  hash `1321ad142a35c6b4a8f30055f62cc651a3160e85885726a225080a0d05d86bde`.
- Fixes applied: manufacturer-cluster bootstrap now preserves resample multiplicity
  (multinomial draw counts, not `np.isin` membership); the shrinkage prior is leave-one-
  seed-out over this run's own beta_hat draws, not a hardcoded constant from a disconnected
  old script; partitioning uses `stratified_random_group_blocks`, not
  `balanced_group_blocks`; imputation is fold-local (`build_features_raw`).
- Finding: shrinkage lowers mean P(over-claim > 5pp) at every tested candidate-set size
  (e.g. 55.1%->51.7% at n=2, 33.6%->23.3% at n=50), but the reduction's own split/fold SD
  exceeds the raw estimator's SD at every n -- some individual folds worsen even though the
  mean improves. Reported as a promising, partition-sensitive finding, not a settled one.

### Budgeted portfolio allocation

- Generator: `experiments/supplementary/run_budget_allocation.py`
- Result: `results/exp_knapsack_v2.json`
- Status: 24/24 successful evaluations (6 seeds x 4 folds); independently reproduced
  bit-for-bit (ignoring timestamps) inside the locked container twice, content hash
  `b1ff2602fc0a2e7960599f28c800597ff5db935f41137154d384d13204c374d8`.
- Fixes applied: same partitioning/imputation fixes as above; the "certified" budget
  objective is now the CalibrationMap's own `alpha_hat + beta_hat*score` (the same quantity
  RQ2's residual accounting uses), not the old ad hoc class-level `(1-beta)*std_class*a_n`
  band; CLF WBLCA still supplies portfolio structure (mass, baseline GWP, project grouping)
  only, never GWP ground truth.
- Finding: point and certified budget ranking are statistically indistinguishable at every
  tested budget k (realized saving within 0.3-1.4 pp of oracle-relative %, over-claim within
  0.3 pp). Mass-directed ranking recovers materially less realized saving (59-79% of oracle
  vs. 83-89% for point/certified) with no corresponding over-claim advantage. **This does
  not replicate the previously excluded single-partition version's finding that mass-
  directed allocation reduces over-claim** -- under six different manufacturer
  partitions that advantage does not survive. No public paper claim uses the old
  single-partition result.

## Valid but superseded

- Submission v3 is leakage-safe but its six seeds produced near-invariant partitions; it
  cannot support a six-partition claim.
- Model robustness v2 is a valid fixed-partition comparison, superseded by the 96-run v3
  experiment.
- Full-corpus normalization totals and strength-class definitions remain valid descriptive
  accounting.

## Excluded from paper claims

- The original beta-shrinkage script (`exp_beta_shrinkage.py`, v1) loses resampling
  multiplicity through membership-only indexing and uses a same-corpus prior hardcoded from
  a disconnected old run. Superseded by `exp_beta_shrinkage_v2.py` above; the v1 script and
  result are kept for the historical record, not used.
- Earlier conformal experiments under-cover and do not establish certification or
  targeted abstention.
- Earlier policy-frontier experiments use unmatched candidate samples or natural-scale
  outcomes not governed by the proposition.
- The original slate and knapsack studies (`exp_knapsack.py`, `exp_knapsack_hetero.py`, v1)
  do not evaluate the complete proposed pipeline (near-invariant single-seed partition,
  globally-fit imputation, an ad hoc certified band disconnected from RQ2's calibration
  accounting); one heterogeneity test also uses a row-random internal split. Superseded by
  `exp_knapsack_v2.py` above for the budgeted-allocation question; the v1 scripts and
  results are kept for the historical record, not used. `exp_slate.py`'s own building-level
  aggregation claim is out of scope for this correction and remains
  unaffected.
- VOI studies are acquisition simulations rather than validation of RQ1-RQ3.
- Early pilots, correction sweeps, asymmetric tests, and core-finding artifacts use old
  preprocessing, ineffective partition seeds, incomplete provenance, or superseded data
  snapshots.
- CLF WBLCA supplies project structure only. CLF/ICE baselines, NIST, UK factors, India
  CEA, Climate TRACE, IFC, and IGBC are contextual sources, not independent product-level
  validation sets.

The active paper's main text (five pages) should not add VOI or multi-source-validation
claims without a new prespecified rerun under the v4 partition and pairing rules. The
corrected beta-shrinkage and budgeted-allocation reruns are in under that rule as
appendix-only material, and the review-response suite is in under that rule with main-text
standing for its RQ2 premise audit and its RQ3 reporting-rule result. The conformal claims
now in the paper come from `exp_uncertainty_policies_v1.py` and are stated as
selection-conditional and *under-covering* on unseen manufacturers (0.152 against a 0.10
target); the earlier excluded conformal experiments remain excluded, and nothing here
claims validated coverage.
