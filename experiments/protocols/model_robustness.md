# Protocol: Model-family robustness under unseen manufacturers

Written before execution on 2026-08-09.

## Research question

Does the separation between numeric value accuracy and within-strength-class
ordering persist across materially different model families when manufacturers,
rather than records, define the held-out unit?

## Data and split

- Source: the frozen local EC3 ready-mix snapshot.
- The existing unit-normalization, physical-plausibility, deduplication, and
  strength-class rules are applied without modification.
- Four row-count-balanced manufacturer blocks are constructed with split seed 0.
  Each block is the test set once; the other three train the model.
- All models receive the same feature matrix and exactly the same folds.

## Models

No model or hyperparameter is selected using test performance.

1. Ridge regression with standardized numeric features.
2. Histogram gradient boosting.
3. Extremely randomized trees.
4. The paper's prespecified gradient-boosted tree model.

The suite spans linear shrinkage, additive binning, randomized tree ensembles,
and the original staged boosting implementation. Hyperparameters are fixed in
the script before the run.

## Outcomes

For every model-fold pair report:

- test-set R-squared on log GWP, using the held-out test mean in the denominator;
- within-strength-class Spearman correlation, averaged over eligible classes;
- within-strength-class pairwise win rate from deterministic seeded pair draws;
- sample sizes and held-out manufacturer counts.

The primary robustness criterion is qualitative, not a significance test:
whether a model has mean pairwise win rate above 0.5 while mean R-squared is
below 0. The four folds are correlated partitions of one corpus and must not be
presented as independent-data confidence intervals.

## Integrity and reporting

- Save all 16 model-fold evaluations and every failure.
- Record hashes for this protocol, the script, the raw EC3 snapshot, and the
  normalization/feature source files.
- Any manufacturer overlap, missing fold, non-finite metric, or failed model is
  a failed experiment. Do not silently omit it.
- This robustness study does not validate transport to another registry,
  country, product category, or project outcome.

