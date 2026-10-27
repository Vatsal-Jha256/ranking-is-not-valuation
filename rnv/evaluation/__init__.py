"""The evaluation protocol every headline number is measured under.

One claim is doing all the work in this project: a recommender is scored on suppliers
it has never seen. Everything that enforces that lives here, so that no experiment can
drift from it by accident.

    partitions   seeded, size-stratified manufacturer blocks (the split)
    model        the estimator and the heteroscedastic scale model
    metrics      ranking quality, claim-gap distributions, split/fold summaries
    corpus       Corpus and Fold: the 24 accepted seed/fold evaluations
    bootstrap    manufacturer-cluster bootstrap that preserves resample multiplicity
    provenance   content digests recorded into every result ledger

Supports: the manuscript's evaluation-protocol paragraph, and Appendix B.
"""
