# Code map

For the runner behind a result, see the [experiment index](../experiments/README.md).

## Library

| Module | Purpose |
|---|---|
| `rnv/registry/normalize.py` | Parse units, grades, declared GWP, and manufacturer labels |
| `rnv/registry/features.py` | Build features with fold-local preprocessing |
| `rnv/registry/clf_portfolio.py` | Load building portfolio structure for budget allocation |
| `rnv/registry/fetch_*.py` | Download source records; not used during locked reproduction |
| `rnv/evaluation/partitions.py` | Assign manufacturers to seeded, size-stratified blocks |
| `rnv/evaluation/corpus.py` | Load the corpus and construct evaluation folds |
| `rnv/evaluation/model.py` | Fit prediction and scale models |
| `rnv/evaluation/metrics.py` | Calculate ranking accuracy, claim gaps, and summaries |
| `rnv/evaluation/bootstrap.py` | Bootstrap manufacturer clusters with multiplicity |
| `rnv/evaluation/provenance.py` | Record source and protocol hashes in result files |
| `rnv/claims/certification.py` | Fit calibration maps and corrections |
| `rnv/claims/conformal.py` | Calculate conformal quantiles |
| `rnv/claims/policies.py` | Define selection and reporting rules |
| `rnv/paths.py` | Resolve shared repository paths |

`build_features_raw` is the feature constructor used by the reported experiments.
`build_features_global_median` is retained for compatibility with older designs;
it estimates imputation values across the corpus and should not be used for a
held-out evaluation.

Selection rules can change the recommended product and its realized advantage.
Reporting rules attach a different estimate to the same selected product.

## Data preparation

| File | Purpose |
|---|---|
| `scripts/prepare_data.py` | Check API access, download EC3/CLF inputs, and export preprocessing reports |
| `scripts/run_new_data.py` | Run new-data experiments in a separate directory while retaining released references |

## Figures and reproduction

The scripts in `figures/` read result files. They share plotting settings in
`figures/figstyle.py` and do not fit prediction models.

| File | Purpose |
|---|---|
| `reproduce.sh` | Check data availability and launch the pinned Docker workflow |
| `reproducibility/run_locked.sh` | Verify inputs, refit models, compare results, and export artifacts |
| `reproducibility/generate_manifest.py` | Write or verify input hashes |
| `reproducibility/verify_outputs.py` | Compare regenerated results, check provenance, and validate figures |
| `reproducibility/check_results.py` | Check released results without refitting |
| `tests/test_smoke.py` | Check library behavior and links between runners, protocols, and results |
| `tests/test_manifest.py` | Test strict and public-checkout verification |

Result comparisons omit timestamps and compare provenance separately against files
on disk. Numerical fields use an absolute tolerance of 2e-8; other fields must match
exactly.
