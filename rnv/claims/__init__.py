"""Turning a selected product into a defensible number.

Selection and reporting are two different levers, and separating them is the paper's
main practical result. A *selection* rule decides which product is recommended, and so
fixes both the selection depth and the realized advantage. A *reporting* rule decides
what number is attached to that recommendation, and changes nothing about the product.

    certification   the calibration map, shrinkage, adaptive correction, and the
                    selection-conditional conformal floor
    conformal       the quantiles the reporting rules use, including the one
                    calibrated on the selection event rather than the population

Supports: RQ2 and RQ3 of the manuscript, Appendix D, and the reporting recommendation
in the discussion.
"""
