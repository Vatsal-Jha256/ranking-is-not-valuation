# Reproducing the workshop evidence

The locked workflow refits nine experiments: supplier-held-out selection, model-family
robustness, calibration-premise audit, selection and reporting policies, candidate-pool
sensitivity, temporal holdout, slope shrinkage, budgeted allocation, and the per-class
quantile reporter. It compares their complete ledgers against the frozen references,
redraws seven data figures, checks the narrative figure assets and source provenance,
and records output hashes.

For account creation, downloads, preprocessing, and new-data runs, start with
[Data setup](../docs/DATA_ACCESS.md). The exact filtering rules are in
[Preprocessing](../docs/PREPROCESSING.md).

## Check the public checkout

With the environment from the root README installed:

```bash
python reproducibility/generate_manifest.py verify --allow-missing-raw
python tests/test_smoke.py
python -m unittest discover -s tests -p 'test_*.py' -v
python reproducibility/check_results.py
```

The first command verifies every available immutable input and explicitly lists missing
licensed inputs as not checked. Missing code, a changed result ledger, or a present raw
file with the wrong hash still fails. These commands require no raw data or credentials.
They verify the released evidence and code connections; they do not establish that new
model fits regenerate the numbers.

## Refit all nine experiments

Requirements: Docker Engine with BuildKit, the three exact data files listed in
[Data access](../docs/DATA_ACCESS.md), several GB of free disk, and time for model fitting.

```bash
./reproduce.sh
# Or choose another new/empty output directory:
REPRO_OUTPUT_DIR=/absolute/empty/path ./reproduce.sh
```

The wrapper checks that the input files are present before building the pinned
Linux/amd64 Python 3.12.2 image. The container verifies all input hashes strictly, installs
the hash-locked environment, and runs the experiments. Only explicitly allowed code,
figure assets, ledgers, and required data enter the image. No network is needed once the
image is built. Raw data are supplied locally and are absent from GitHub.

Success writes nine reproduced ledgers, seven data-figure PDFs, the narrative figure's
specification/SVG/PDF, verification.json, the installed environment, the input manifest,
and SHA256SUMS. The output directory must be empty and is mounted separately, so fresh
fits do not overwrite the host checkout's authoritative results.

The ledger comparison ignores run timestamps and separately verifies provenance digests
against files on disk. All other fields are compared recursively with an absolute
floating-point tolerance of 2e-8. Strings, integers, collection lengths, partition digests,
and success/failure structure remain exact.

[LAST_VERIFIED_RUN.md](LAST_VERIFIED_RUN.md) records the previous complete refit.
A successful fast check is distinct from a new model refit.

## Maintaining the manifest

The manifest pins the raw snapshot, CLF portfolio cache, library and experiment code,
protocols, figure assets, result ledgers, and container workflow.
After an intentional input change, inspect the diff before accepting new hashes:

```bash
python reproducibility/generate_manifest.py write
python reproducibility/generate_manifest.py verify
```

Do not update hashes to bypass a mismatch. Protocols and frozen result ledgers remain
unchanged in this release. See [EXPERIMENT_INCLUSION_AUDIT.md](EXPERIMENT_INCLUSION_AUDIT.md)
for authoritative and excluded experiments.
