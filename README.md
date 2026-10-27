# Ranking Is Not Valuation

Selection-aware carbon claims under supplier shift.
Tackling Climate Change with Machine Learning, NeurIPS 2026.

Vatsal Jha, Anupam Srivastava, Ashok Bragadeshwaran

This repository contains the code and results for our study of concrete product
recommendations. We evaluate ranking accuracy and carbon-saving estimates on
manufacturers excluded from training, then compare changes to the selection rule
with changes to the reported estimate.

## Setup

Use Python 3.12. The reference environment is pinned in
`reproducibility/requirements.lock`.

```bash
git clone https://github.com/Vatsal-Jha256/ranking-is-not-valuation.git
cd ranking-is-not-valuation
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r reproducibility/requirements.lock
python -m pip install --no-deps -e .
```

## Check the released results

These checks need no API keys or raw datasets:

```bash
python tests/test_smoke.py
python -m unittest discover -s tests -p 'test_manifest.py' -v
python reproducibility/check_results.py
```

They check library behavior, experiment links, result completeness, input hashes,
and source provenance. Missing licensed inputs are listed as not checked.
The commands do not refit the models.

## Results

The main experiment uses six manufacturer partitions with four held-out blocks each.

| Quantity | Mean |
|---|---:|
| Held-out log-GWP R² | 0.082 |
| Within-class pairwise accuracy | 0.591 |
| Argmin claim gap | 0.057 log units |

Ranking accuracy exceeds chance in 23 of 24 evaluations, despite weak absolute
prediction. Keeping the selected product fixed and changing its report reduces the
probability of overstatement above 0.05 log units from 51.2% to 11.4%.

The reporting bound has limits: only 16.5% of cases have a positive saving floor,
and 15.2% of unseen-supplier cases exceed the bound against a nominal 10% target.
A simple per-class quantile shift performs similarly. The calibration-premise audit
also finds departures from the linear condition behind the depth identity.

The outcome is declared GWP, not measured project emissions. The study uses one
mostly North American registry; manufacturer holdout does not establish geographic
or cross-registry transfer.

## Start from an API key

[Data setup](docs/DATA_ACCESS.md) covers EC3 account/key creation, a read-access
check, resumable ready-mix downloads, preprocessing, the CLF workbook and concrete
cache, and running the experiments. [Preprocessing](docs/PREPROCESSING.md) lists
the filters, features, strength bins, and fold-local imputation rules.

```bash
# After following the environment and key setup in the guide:
python scripts/prepare_data.py check-key
python scripts/prepare_data.py fetch-ec3
python scripts/prepare_data.py preprocess-ec3
python scripts/prepare_data.py fetch-clf
python scripts/prepare_data.py prepare-clf
```

A current API download can differ from the frozen study snapshot. The guide has
separate instructions for exact reproduction and new-data experiments.

## Reproduce the experiments

Obtain the exact inputs described in [Data access](docs/DATA_ACCESS.md), then run:

```bash
./reproduce.sh
```

This requires Docker and takes several hours. It refits all nine experiments,
compares the outputs with the released results, redraws the figures, and exports
artifacts to a new `reproduction-output/` directory.
See [reproducibility/README.md](reproducibility/README.md) for details.

To redraw the figures without fitting models:

```bash
python figures/gen_submission_figure.py
python figures/gen_appendix_figures.py
```

## Files

| Directory | Contents |
|---|---|
| `rnv/` | Corpus processing, evaluation, and reporting rules |
| `experiments/` | Nine experiment runners and their protocols |
| `results/` | Frozen experiment results |
| `figures/` | Result plots and plotting scripts |
| `reproducibility/` | Environment lock, input hashes, and verification scripts |
| `docs/` | Code map and data access instructions |
| `tests/` | Checks that run without model fitting |

[Code map](docs/CODE_MAP.md) · [Experiment index](experiments/README.md)

## Citation and licence

Citation metadata are in [CITATION.cff](CITATION.cff).

```bibtex
@inproceedings{jha2026ranking,
  title={Ranking Is Not Valuation: Selection-Aware Carbon Claims under Supplier Shift},
  author={Jha, Vatsal and Srivastava, Anupam and Bragadeshwaran, Ashok},
  booktitle={Tackling Climate Change with Machine Learning at NeurIPS},
  year={2026}
}
```

Code is released under the [MIT licence](LICENSE). Raw datasets retain their source
licences and are not redistributed.
