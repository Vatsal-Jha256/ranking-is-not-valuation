# Protocol Addendum: Separate Diagnostic and Policy Training Regimes

Written on 2026-08-09 after inspecting the completed iteration-1 output and
before running iteration 2.

Iteration 1 used the policy experiment's disjoint training, calibration, and
test split for every metric. That trains the diagnostic model on 50% of the
corpus, whereas the paper's rank-versus-value diagnostic uses all three
non-test folds (75%). The two questions require different data partitions:

- **Diagnostic model:** test fold held out; all other folds train the model.
  This measures the strongest leakage-free predictor available under the
  stated four-fold evaluation.
- **Policy model:** test and calibration folds held out; the remaining two
  folds train the model. This preserves the disjoint calibration required for
  Thompson scale estimation and policy auditing.

Iteration 2 therefore fits both models explicitly and labels their metrics.
It also stores log-scale over-claim and log-scale realized quality for every
policy, because Proposition 1 is stated on log GWP. Natural-scale percentage
over-claim and fractional quality remain as application-facing metrics but are
not described as theorem-governed quantities.

Iteration-1 output is preserved at
`results/exp_submission_harmonized_v1.json`. Iteration 2 writes
`results/exp_submission_harmonized_v2.json` and does not overwrite iteration 1.

