# Temporal holdout

**Does the separation survive a shift that is not a supplier shift?**

Protocol: [`../protocols/review_response_suite.md`](../protocols/review_response_suite.md)
(E4) · Runner: `run_temporal_holdout.py` · Ledger:
`results/exp_temporal_holdout_v1.json`

## Design

Every other experiment here holds out suppliers from a single snapshot, which leaves one
obvious way the result could be an artifact: the corpus was pulled at one moment, so a
model and its evaluation share whatever was true of declaration practice then. Product
category rules, plant mixes and the set of firms that publish change year to year.

This changes the split used in the accepted protocol, as specified in the
protocol document: partitions cut on declaration date rather than on seeded manufacturer
blocks. Train before a cutoff, test after. The strict variant additionally keeps only
post-cutoff suppliers absent before the cutoff, so drift and supplier shift compound
rather than being confounded.

## Result

| Variant | Cutoff | R² | Pairwise | β | Gap |
|---|---|---|---|---|---|
| temporal | 2022 | +0.027 | 0.575 | 0.468 | −0.048 |
| strict | 2022 | −0.516 | 0.550 | 0.501 | −0.048 |
| temporal | 2023 | +0.223 | 0.633 | 0.729 | −0.018 |
| strict | 2023 | −0.035 | 0.642 | 0.690 | −0.054 |
| temporal | 2024 | −0.069 | 0.585 | 0.735 | −0.086 |
| strict | 2024 | −0.120 | 0.559 | 0.540 | −0.100 |

R² is negative in 4 of 6 variants while pairwise accuracy exceeds chance in all 6 and the
slope stays inside [0.468, 0.735]. The separation survives, and the strict variant is
worse than the plain one in every pair, which is the expected direction.

The claim gap is *negative* in all six. Under a temporal cut the model under-claims. That
is not evidence that selection stopped biasing the estimate; it is evidence that a second
shift is acting on the level of the predictions in the opposite direction, and it is
reported rather than smoothed.

## What this does not show

Six variants on one registry is a check, not external validation. Cross-registry
validation remains impossible with a single source at this product and manufacturer
resolution, and the paper still lists it as the next step.
