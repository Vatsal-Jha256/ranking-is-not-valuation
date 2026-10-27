"""Selection-aware carbon claims: library code shared by every experiment.

The package is organised by concept rather than by chronology, so that a reader who
knows the paper can find the code for any of its parts without knowing the order in
which they were written.

    rnv.registry     turning the raw EPD registry into a modelling corpus
    rnv.evaluation   the supplier-disjoint protocol, its splits, its metrics
    rnv.claims       calibration, correction, and conformal reporting rules

Nothing in this package writes to ``results/``. Experiments own their outputs; the
library only provides the pieces they share, so that two experiments cannot silently
disagree about what a fold, a slope, or a claim gap is.

Subpackages are imported explicitly rather than re-exported here, so that a script that
only needs a corpus does not pay for scikit-learn imports it will never use.

Layout and the map from code to manuscript sections are documented in ``docs/CODE_MAP.md``.
"""
