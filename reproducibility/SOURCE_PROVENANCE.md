# Source provenance and scientific scope

This note distinguishes data that directly support the paper's three research
questions from contextual data collected for later lifecycle and deployment work.
These sources have different observation units. They do not provide external
validation of the product-level prediction task.

## Source-to-claim rule

Only a source with product-level declared GWP, concrete strength, and a recorded
manufacturer can support the current unseen-manufacturer prediction, ranking, and
selection-policy outcomes. In the present collection, only EC3 satisfies all three
conditions. Other sources can define reference values, project structure, process
models, transport factors, grid context, or facility context, but they cannot serve
as held-out product truth.

| Source | Verified local content | Role | Excluded claim |
|---|---|---|---|
| EC3/openEPD materials | 131,361 raw ready-mix records; 108,326 retained records; 413 recorded manufacturer labels | Sole product outcome, candidate pool, feature source, and manufacturer split for RQ1-RQ3 | No cross-registry, India, project-emissions, or structural-equivalence claim |
| CLF 2025 North American Material Baselines | 367 populated category references, including 209 ready-mix entries; CC BY-SA 4.0 | Independent category-reference context | Not product observations or supplier holdout |
| ICE Educational v4.1 | 16 material-profile sheets; generic UK strength-class concrete factors; academic-use terms | UK generic factor context | Not product truth; not distributable as extracted data outside its terms |
| CLF WBLCA v2 | 927,950 lifecycle rows across 292 projects; building/material line structure | Future portfolio structure and project clustering | Not an external test of EC3 prediction; lightweight and normal-weight lines cannot be pooled |
| NIST openLCA construction archive | 45 processes, primarily unit inventories | Process and boundary documentation | Not ready-made LCIA/GWP labels without a background database and characterization method |
| UK government conversion factors | Activity-based freight and other reporting factors | Separate A4 transport sensitivity | Not cradle-to-gate product GWP and not an observed concrete route |
| India CEA v21 | Grid-emission database through FY2024-25 | India electricity context | Cannot update concrete GWP without a source-specific energy/process model |
| Climate TRACE | Facility/gas intensity records for cement, lime, steel, aluminium, and glass | Industrial-process and evidence-coverage context | Not concrete A1-A3 labels |
| IFC/EDGE India methodology | India A1-A3 methodology for priority materials | Citation-only methodological context | No numerical ingestion or redistribution without permission |
| IGBC/CII India baseline report | Aggregate case-study results for 20 buildings | Aggregate India building context | Not material-product truth or row-level evaluation data |

## Current paper questions

1. Under recorded-manufacturer holdout, how differently do absolute log-GWP accuracy
   and within-strength-class ordering respond to supplier partition?
2. Under an explicit full-vector calibration premise, how does selection depth enter
   the expected signed log claim gap?
3. On identical EC3 candidate sets, how do alternative selection policies trade
   claim-gap reduction against within-class declared-GWP advantage?

The paper's Docker workflow therefore pins only the EC3 ready-mix snapshot as a raw
experimental input. Contextual sources remain outside the computation, retain their
original licenses, and are not copied into the container or submission artifact.
Retrieval dates, query rules, accession IDs, and content hashes for the EC3 extraction
and International EPD System enumeration are frozen in
`reproducibility/CONTEXTUAL_DATA_SNAPSHOTS.md`.
