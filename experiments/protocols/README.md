# Protocols

Every experiment reported in the paper was specified here **before it ran**, including
the conditions under which its result would count as a failure. These files are not
edited afterwards. When a design changes, an addendum is added and the original stays.

That rule is what makes the numbers in `results/` evidence rather than a search outcome,
and it is enforced mechanically: each protocol is content-addressed in
`reproducibility/input_manifest.json`, and every ledger records the digest of the
protocols it was run under.

| Protocol | Governs |
|---|---|
| `selection_gap.md` + addenda v2–v4 | `../selection_gap/` |
| `model_robustness.md` + addenda v2–v3 | `../model_robustness/` |
| `review_response_suite.md` | `../calibration_premise/` (E1), `../reporting_policies/` (E2), `../pool_sensitivity/` (E3), `../temporal_holdout/` (E4) |
| `supplementary_reruns.md` | `../supplementary/` |

`review_response_suite.md` covers four experiments because they were prespecified
together as one suite, before any of them ran.
