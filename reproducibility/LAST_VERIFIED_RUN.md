# Last verified run

Date: 2026-10-13. Command: `./reproduce.sh` (pinned linux/amd64 image, hash-locked environment).

Result: PASS. Nine experiment ledgers regenerated and matched their locked references field
for field, all manuscript figures redrawn, and every quoted manuscript quantity re-derived.
Manuscript body is five pages (TCCML 2026 camera-ready limit). The run covers the final
camera-ready tree, including the corrected bound-violation wording and the style-compliant layout.

`verification.json` SHA-256: `ff094097318682e088d70120e594b733177571b4b93099d2a12c454a3b4e6d99`

The ninth ledger, `results/exp_stratified_quantile_v1.json`, was generated inside the pinned
container; an earlier copy produced outside it differed in the last decimals, which is why
the run environment is part of the evidence.
