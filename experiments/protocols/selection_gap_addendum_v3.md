# Protocol Addendum v3: Leakage-safe preprocessing and pair ties

Written on 2026-08-09 after an independent integrity audit of iteration 2 and
before executing iteration 3.

The audit found that the shared feature helper replaced missing covariates with
full-corpus medians before manufacturers were partitioned. This does not use the
outcome, but it lets held-out-manufacturer covariates influence preprocessing.
Iteration 3 moves median imputation inside every fitted training-fold pipeline.

The audit also identified ambiguity in pairwise win rate. Earlier iterations
discarded prediction ties and truth ties. Iteration 3 uses the following primary
pairwise accuracy:

- discard pairs whose observed outcomes are tied;
- award one point for a correct strict ordering;
- award half a point when the predictions tie;
- divide by all attempted pairs with distinct records and distinct outcomes.

The result record also reports prediction-tie rate. Pair indices are generated
from a seed determined only by split and fold, so diagnostic comparisons that
share a test set also share pair draws.

All original model, split, calibration, candidate-set, and policy settings remain
unchanged. Iteration 2 is preserved together with its exact script. Iteration 3
writes a new result and records hashes for the raw input, feature construction,
normalization, split implementation, script, and every protocol document.

