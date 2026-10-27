# Selection gap

**Does a model that ranks products well also value them well?**

Protocol: [`../protocols/selection_gap.md`](../protocols/selection_gap.md) plus addenda
v2–v4 · Runner: `run_selection_gap.py` · Ledger:
`results/exp_submission_harmonized_v4.json`

## Design

Six seeded, size-stratified manufacturer partitions, four blocks each, 24 evaluations in
total. Within a partition, one block is held out for testing, the next calibrates, the
rest train, and the three roles never share a manufacturer. Two models are fitted per
evaluation for two different questions: a diagnostic model that sees everything except
the test block, and a policy model trained on less so that a disjoint calibration block
exists.

Candidate sets of ten functionally interchangeable products are then drawn within a
strength class, and five selection rules choose from each: `argmin` and four randomized
softenings of it.

## Result

| | mean | SD | sensitivity range |
|---|---|---|---|
| Held-out R² | 0.082 | 0.144 | [−0.241, 0.273] |
| Within-class pairwise accuracy | 0.591 | 0.043 | [0.522, 0.658] |
| Calibration slope β | 0.605 | 0.203 | [0.219, 0.942] |

R² is negative in 6 of the 24 evaluations while pairwise accuracy exceeds chance in 24 of
24. The two capabilities come apart, and the calibration slope well below one is why:
the model's spread of predictions is wider than the truth warrants.

Under `argmin`, the claim gap is 0.057 ± 0.073 log units at a selection depth of 0.174,
buying a declared-GWP advantage of 0.104. Randomizing the selection reduces both
together: score perturbation gives up 7.7 percentage points of advantage to remove
0.009 ± 0.044 of gap, a trade at roughly the rate `1 − β` predicts.

## What this does not show

It does not show that the model is useless — an ordering that beats chance in every
evaluation is worth having, and specifying against class medians is a real decision.
It does not measure realized emissions; the outcome is declared GWP throughout. And the
sensitivity percentiles describe how much a number moves when the partition moves, on one
corpus. They are not confidence intervals over independent data.
