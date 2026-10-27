# Protocol v3 addendum: model families across genuine partitions

Date fixed: 2026-08-09, before the v3 run.

The v2 model-family comparison used one four-fold balanced partition. The paper's final
split-sensitivity experiment now uses six genuinely different, size-stratified seeded
manufacturer partitions. Version 3 evaluates the same four prespecified model families
on all six partitions and all four held-out blocks (96 model/fold evaluations).

The partition rule is exactly the rule fixed in the submission-harmonization v4 addendum:
manufacturers are sorted by record count, grouped into consecutive strata of four, and
assigned one per block by a seeded random permutation within each stratum. Models,
hyperparameters, fold-local median imputation, outcome, pairwise tie handling, and 20,000
draws per eligible strength class remain unchanged. Within a split/fold, every model uses
the same truth-tie-filtered pair draws. Estimator random states remain fixed at zero so
the experiment varies model family and manufacturer partition, not training randomness.

All 96 attempts must be reported. Model summaries are same-corpus partition sensitivity,
not independent-data confidence intervals. No family is declared universally robust from
a mean alone; fold ranges and the number of evaluations above pairwise chance are also
reported.
