# Supplementary checks

Two designs that the experiment inclusion audit had excluded, rebuilt under the accepted
protocol so their questions can be answered rather than left open.

Protocol: [`../protocols/supplementary_reruns.md`](../protocols/supplementary_reruns.md)
· Ledgers: `results/exp_beta_shrinkage_v2.json`, `results/exp_knapsack_v2.json`

| Runner | Question |
|---|---|
| `run_beta_shrinkage.py` | Does empirical-Bayes shrinkage of the calibration slope beat the slope itself? |
| `run_budget_allocation.py` | With a budget of k line items, does a certified objective rank them better than a point estimate? |

Both are reported in the appendix as supplementary evidence and are included in
the locked nine-experiment workflow.

Two disciplines separate the reruns from the designs they replace, and both are recorded
in `reproducibility/EXPERIMENT_INCLUSION_AUDIT.md`: the manufacturer bootstrap preserves
resample multiplicity, and the shrinkage prior is built under a strict leave-one-seed-out
rule rather than imported from a disconnected run. There is no external corpus for this
problem, so a prior from outside the data would have to be invented.
