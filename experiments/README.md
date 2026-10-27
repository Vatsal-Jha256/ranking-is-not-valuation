# Experiments

Each experiment directory contains a runner and a description of its design and results.
The smoke checks verify that runner docstrings point to existing protocols and results.

## Reported experiments

| Experiment | Question | Ledger | Where it lands |
|---|---|---|---|
| [`selection_gap/`](selection_gap/) | Does ranking quality imply valuation quality? | `exp_submission_harmonized_v4.json` | RQ1, RQ2 slope, RQ3 frontier, Figure 1 |
| [`model_robustness/`](model_robustness/) | Is the split a property of one estimator? | `exp_model_robustness_v3.json` | RQ1 model-family paragraph, Figure 1 |
| [`calibration_premise/`](calibration_premise/) | Does the premise behind the identity hold? | `exp_calibration_structure_v1.json` | RQ2, Appendix C |
| [`reporting_policies/`](reporting_policies/) | Can an uncertainty-aware rule remove the gap? | `exp_uncertainty_policies_v1.json` | RQ3, Appendix D |
| [`pool_sensitivity/`](pool_sensitivity/) | How much depends on the candidate pool? | `exp_candidate_sensitivity_v1.json` | Appendix E |
| [`temporal_holdout/`](temporal_holdout/) | Does it survive a shift in time, not just suppliers? | `exp_temporal_holdout_v1.json` | Appendix F |
| [`supplementary/`](supplementary/) | Do shrinkage and budgeted allocation add anything? | `exp_beta_shrinkage_v2.json`, `exp_knapsack_v2.json` | Appendix H |

The locked workflow reruns all nine reported experiments, including the per-class
quantile reporter in `reporting_policies/` and both supplementary checks.

## Running one

Runners take no arguments and write exactly one ledger. From the repository root:

```bash
python experiments/selection_gap/run_selection_gap.py
```

Each runner refits models and writes its result file. Use a separate checkout for
modified runs to preserve the released results. The full workflow runs inside Docker
and writes reproduced artifacts to a separate output directory.

## Adding one

1. Write the protocol in `protocols/` first, including the conditions under which the
   result would count as a failure. Do not edit it afterwards.
2. Create `experiments/<concept>/run_<something>.py`. Import folds from
   `rnv.evaluation.corpus` rather than reimplementing them, so the experiment cannot
   drift from the accepted protocol by accident.
3. Give the runner a docstring naming its protocol, its ledger and the part of the paper
   it supplies. The smoke test enforces this.
4. Record `ledger_provenance(__file__, <protocols>)` in the ledger metadata.
5. Add it to `reproducibility/generate_manifest.py`, `reproducibility/run_locked.sh` and
   `.dockerignore` if the paper is going to rely on it.
6. Write the folder README: question, design, result, and the limits of the result.
