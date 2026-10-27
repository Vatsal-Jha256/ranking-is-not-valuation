# Protocol Addendum v2: Integrity-audit corrections

Written on 2026-08-09 after independent audit of iteration 1 and before rerun.

Iteration 1 is preserved but is not used for submission claims. Iteration 2 makes
three corrections:

1. Missing numeric covariates are imputed with medians estimated inside each
   training fold through a fitted model pipeline.
2. Every model receives the same deterministic pair indices within a fold.
3. Pairwise accuracy awards half credit to prediction ties, excludes only
   observed-outcome ties, and reports the prediction-tie rate and denominator.

All models, hyperparameters, manufacturer folds, features, and primary
mean-level robustness criterion otherwise remain unchanged. The result records
the split-source hash in addition to the hashes required in iteration 1.
