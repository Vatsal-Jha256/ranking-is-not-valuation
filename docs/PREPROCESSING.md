# Preprocessing

The implementation is in `rnv/registry/normalize.py`, `rnv/registry/features.py`,
and `rnv/evaluation/corpus.py`. The preparation command calls those functions;
it does not implement a second set of EC3 filters.

## Units and record filtering

1. Parse 28-day compressive strength into MPa. Supported strength units are MPa,
   psi, ksi, and kPa; a unitless value is treated as MPa.
2. Parse the declared volume into cubic metres. Supported volume units include
   m³, yd³, and litres. A cubic yard is 0.764554857984 m³.
3. Parse declared GWP into kgCO₂e and divide by declared volume to obtain
   kgCO₂e/m³. Declared mass is converted to kilograms.
4. Drop records with missing GWP, non-volumetric declared units, or missing
   28-day compressive strength.
5. Keep GWP between 40 and 2,000 kgCO₂e/m³ and strength between 5 and 200 MPa,
   inclusive.
6. Use the numeric density field where available; otherwise calculate density
   as declared mass divided by declared volume. Keep 1,400–3,000 kg/m³, inclusive.
7. Calculate GWP per kilogram and keep values between 0.005 and 1.5, inclusive.
8. Deduplicate on manufacturer name, plant name, product name, and normalized
   GWP per cubic metre, using the first occurrence.
9. Assign a strength class and remove rows that cannot be classified.

The strength-class bins are right-closed, matching `pandas.cut` in the frozen code:

| Stored label | Interval in MPa |
|---|---|
| `C<20` | (0, 20] |
| `C20-25` | (20, 25] |
| `C25-30` | (25, 30] |
| `C30-35` | (30, 35] |
| `C35-40` | (35, 40] |
| `C40-50` | (40, 50] |
| `C50+` | (50, ∞) |

These bins are a functional-comparability proxy. They do not establish structural
interchangeability between products.

## Features and target

The target is the natural logarithm of declared GWP per cubic metre.
`build_features_raw()` constructs strength, density, water/cement ratio when
available, manufacturer latitude/longitude, and keyword indicators from product
name and description. It does not use GWP as a feature.

Missing numeric values remain missing in the exported feature CSV. Median
imputation is fitted inside the model pipeline using training-fold rows only.
Do not impute the whole dataset before splitting; the historical
`build_features_global_median()` helper is not used by the reported experiments.

## Manufacturer splits

Missing manufacturer names are assigned the `unknown` label. Manufacturers are
assigned as whole groups to four size-stratified blocks for each of six seeds.
For fold f, block f is test, block (f + 1) modulo 4 is calibration, and the remaining
two blocks train the model. Manufacturer labels must be disjoint across roles.
The temporal-holdout experiment uses its separately documented date-based split.

## Inspect the preprocessing report

`python scripts/prepare_data.py preprocess-ec3` exports cleaned rows, raw features,
and a JSON report of filtering and missingness. On the frozen EC3 snapshot, expect
131,361 raw records, 108,326 retained records, and 413 manufacturer labels.
The report also records the input hash. Matching row counts alone does not identify
the frozen snapshot; the SHA-256 digest must match too.

## CLF portfolio cache

`prepare-clf` calls the existing loader on the source workbook, with the cache
isolated so that a pre-existing CSV cannot be mistaken for a fresh rebuild. It
keeps concrete A1-A3 rows, extracts strength from `mat_type` in psi, converts it to
MPa, keeps 5–200 MPa, requires positive `inv_mass` and `gwp`, and keeps GWP/mass in
[0.005, 1.5]. It applies the same strength bins as the EC3 corpus.

The output supplies project structure to budget allocation. Prediction labels,
training, and held-out product outcomes still come from EC3.
