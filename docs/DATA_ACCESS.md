# Data access and setup

These instructions cover account access, downloads, preprocessing, and experiments.
Commands use Bash on Linux or macOS; Windows users can use WSL. Install Git, Python
3.12, and Docker before starting. The data downloads require internet access.

The main corpus comes from the EC3 **materials** API. The CLF workbook is a separate
public download used for portfolio structure in the budget-allocation experiment.
The registry dataset is not included in this repository.

## 1. Install the environment

```bash
git clone https://github.com/Vatsal-Jha256/ranking-is-not-valuation.git
cd ranking-is-not-valuation
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r reproducibility/requirements.lock
python -m pip install --no-deps -e .
python -m pip install 'requests>=2.31,<3' 'openpyxl>=3.1,<4'
```

Requests is used for downloading; openpyxl reads the CLF workbook. Neither is needed
for the frozen model fits inside Docker.

## 2. Create and load an EC3 API key

Create an account or sign in at [EC3](https://buildingtransparency.org/ec3/). Open
Settings, then **API & Integrations → API Keys → Create API key**. Set a name,
read-access scope appropriate to the materials endpoint, and expiry, then create and
copy the key. Direct links: [integrations](https://buildingtransparency.org/ec3/manage-apps)
and [API keys](https://buildingtransparency.org/ec3/manage-apps/keys).
These are the steps documented by
[Building Transparency](https://github.com/Building-Transparency/brex-docs#getting-ec3-api-key).
If your account cannot read materials, request that access from Building Transparency.
Bulk retrieval is subject to the provider's access tier and rate limits; see
[EC3 API access](https://www.buildingtransparency.org/tools/ec3-become-partner/).

Create a local configuration file without overwriting an existing one:

```bash
test -e .env || cp .env.example .env
# Edit .env locally and replace your_key_here with your own key.
set -a
source .env
set +a
python scripts/prepare_data.py check-key
```

The check requests one ready-mix record and reports success or an HTTP error without
printing the key. HTTP 401 generally means the key is invalid or expired; HTTP 403
means access or scope is insufficient. Neither the configuration nor source datasets
are tracked by Git. Repeat the `set -a` / `source .env` / `set +a` commands in a new shell.

## 3. Download ready-mix materials

On a new checkout:

```bash
python scripts/prepare_data.py fetch-ec3
```

The downloader queries `GET /materials` using ready-mix category
`b03dba1dca5b49acb1a5aa4daab546b4`, 250 records per page. It preserves the same field
selection as `rnv/registry/fetch_materials.py`: GWP, declared units, functional
attributes, manufacturer/plant fields, and timestamps.

Each completed page is saved before the next request. An empty page establishes
completion; reaching the page limit or exhausting retries exits with an error.
Final `readymix.jsonl` and `readymix.state.json` files are written only after a
complete download. Resume an interrupted download with:

```bash
python scripts/prepare_data.py fetch-ec3 --resume
```

The default output is `data/raw/ec3_materials/`. It contains:

- `readymix.jsonl`: reduced records, one JSON object per line.
- `readymix.state.json`: final row count and next page.
- `readymix.download.json`: query settings, download status, and SHA-256 digest.
- `.readymix-pages/`: page files used for resuming.

Existing final snapshots are not overwritten. To inspect a new extraction without
replacing one already on disk, add `--output-dir /absolute/path/new-ec3-download`.
Use the matching `--input` path in the preprocessing step below. Experiments expect
the default paths, so use a separate checkout for a separate dataset.

## 4. Run EC3 preprocessing

```bash
python scripts/prepare_data.py preprocess-ec3
```

This runs the existing normalization and feature code and writes:

```text
data/processed/readymix/readymix_clean.csv
data/processed/readymix/readymix_features.csv
data/processed/readymix/readymix_preprocessing.json
```

The report gives the input hash, drop counts, retained rows, manufacturer count,
strength-class counts, feature missingness, and whether the raw file matches the
frozen extraction. Existing preprocessing outputs are not overwritten; use a new
`--output-dir` when rebuilding them.

The frozen extraction has 131,361 raw records and yields 108,326 retained records
with 413 manufacturer labels. Counts from a current API download can differ.
[PREPROCESSING.md](PREPROCESSING.md) lists every filter and explains fold-local
imputation. The CSVs are inspection outputs: experiment runners apply the same
preprocessing directly to `readymix.jsonl`.

## 5. Download the CLF portfolio workbook

No API key is needed for this source:

```bash
python scripts/prepare_data.py fetch-clf
```

This downloads the `full_lca_results.xlsx` file from the
[CLF WBLCA Benchmark Study dataset](https://doi.org/10.6084/m9.figshare.28462145.v2)
to `data/raw/clf_wblca/full_lca_results.xlsx`. The source file is 104,568,115 bytes;
its provider MD5 is `40c83ca205efb3ae44ac0d85d5061c1c`. The downloader checks that
checksum before accepting the file. The dataset's licence is CC BY 4.0; retain its
attribution when using or redistributing it. Source information is also available
in the [provider's repository](https://github.com/Life-Cycle-Lab/wblca-benchmark-v2-data).

## 6. Build the CLF concrete cache

```bash
python scripts/prepare_data.py prepare-clf
```

This reads the workbook, applies the existing `clf_concrete()` filters, and writes
`data/processed/clf_concrete_a1a3.csv` plus a preparation report. Workbook parsing can
take several minutes and needs more memory than reading the resulting CSV. The
frozen cache contains 9,622 rows. The report compares the generated cache hash with
the frozen one; CSV serialization or environment differences can affect that match.

The loader keeps concrete A1-A3 line items with parseable strength, positive mass
and GWP, and plausible GWP per kilogram. The CLF data supply masses, grades, and
project grouping for portfolio allocation. They are not prediction labels or an
independent validation dataset for the EC3 model.

To rebuild a cache without replacing an existing one:

```bash
python scripts/prepare_data.py prepare-clf --output /tmp/clf-rebuilt.csv
```

## 7. Choose the experiment path

### Reproduce the frozen results

The exact files must match `reproducibility/input_manifest.json`:

```text
data/raw/ec3_materials/readymix.jsonl
data/raw/ec3_materials/readymix.state.json
data/processed/clf_concrete_a1a3.csv
```

Check them before launching model fits:

```bash
python reproducibility/generate_manifest.py verify
./reproduce.sh
```

The EC3 extraction was completed on 2026-07-29. An API key gives access to the current
registry; it does not reconstruct that historical snapshot. A fresh download is an
exact reproduction input only if the hashes match. If they differ, obtain the frozen
files from the authors subject to the source terms, or use the new-data path below.
Use the [repository issues](https://github.com/Vatsal-Jha256/ranking-is-not-valuation/issues)
for data-access questions; do not include an API key.
Do not regenerate the release manifest to make different data appear to match.

The Docker workflow checks inputs, refits nine experiments, redraws the figures,
compares fresh results against the frozen results, and writes artifacts to a new,
empty `reproduction-output/` directory. It takes several hours. Set
`REPRO_OUTPUT_DIR=/absolute/new/path` to choose another output location.

### Run on a newly downloaded dataset

After steps 1–6, keep the release manifest unchanged and use:

```bash
python scripts/run_new_data.py --output-dir /absolute/new/path --plan
python scripts/run_new_data.py --output-dir /absolute/new/path
```

The first command checks required files and lists the nine runners without fitting.
The second exports the committed code to the new directory, copies the three data
inputs, saves the released results under `reference-results/`, and runs each experiment
in dependency order. New results go to `results/`, logs to `logs/`, and input hashes,
code revision, environment, and completion status are recorded alongside them.
The driver checks each result for failed or missing evaluations before continuing.
The original checkout is not modified. The new directory must not already exist.

These are new-data experiment results, not a claim to reproduce the published values.
Do not run the frozen-result checker on them or replace the released ledgers with them.
Inspect the logs and per-ledger failure fields when interpreting the outputs.

## 8. Check the release without obtaining data

```bash
python reproducibility/generate_manifest.py verify --allow-missing-raw
python tests/test_smoke.py
python -m unittest discover -s tests -p 'test_*.py' -v
python reproducibility/check_results.py
python figures/gen_submission_figure.py
python figures/gen_appendix_figures.py
```

Absent licensed data are listed as not checked. Changed files that are present still
fail verification. These commands inspect the released evidence and do not refit models.
