# Protocol v4 addendum: genuine partition sensitivity

Date fixed: 2026-08-09, before the v4 run.

## Reason for this addendum

The v3 integrity review found that `balanced_group_blocks` uses its random seed only
to break ties between manufacturers with exactly equal record counts. Because most
manufacturer counts are distinct, the six nominal split seeds produced nearly
invariant manufacturer-block compositions. The 24 v3 evaluations therefore remain a
valid audit of those four blocks, but they do not support a six-partition sensitivity
claim. In particular, a policy-effect sign pattern of 18/24 repeated the same three
positive fold positions across all six seeds.

## Prespecified correction

Version 4 changes only the manufacturer partition constructor. Manufacturers are
sorted by decreasing retained record count and divided into consecutive strata of four.
Within every stratum, a seeded random permutation assigns one manufacturer to each of
the four blocks. The final incomplete stratum is assigned without replacement to a
seeded random subset of blocks. This makes every seed affect the full manufacturer
composition while balancing the manufacturer-size distribution across blocks.

All v3 protections remain unchanged: raw features are imputed inside each training
fold; train, calibration, and test manufacturers are disjoint; prediction ties receive
half credit after truth ties are excluded; diagnostic and policy metrics use common
pair draws; every policy receives the same candidate sets; and all attempted runs are
reported.

The externally reported name of the Gaussian policy is standardized as
`score_perturbation`; the sampling rule itself is unchanged. Version 4 also records a
diagnostic scalar-calibration residual for each policy,
`mean(truth - alpha - beta * score)`. Pairwise policy differences can therefore be
accounted for exactly as a selection-depth component plus a selected-residual component.
This accounting does not test or replace the proposition's stronger full-vector
conditional-calibration premise.

## Reporting rule

The primary report contains all six seeded partitions by four held-out blocks. Means,
standard deviations, and percentiles are same-corpus partition sensitivity summaries,
not independent-data confidence intervals. Sign counts are reported only with their
seed-by-fold structure, never as independent confirmations. The result records block
loads, manufacturer-assignment digests, and pairwise test-set Jaccard similarities so
partition diversity is auditable.
