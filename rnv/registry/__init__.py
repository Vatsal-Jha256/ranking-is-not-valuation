"""From published EPDs to a modelling corpus.

An Environmental Product Declaration is a document, not a row in a table: units,
strength grades, densities and declared volumes are all recorded free-form. This
subpackage is everything that stands between that and a numeric corpus.

    fetch_epd, fetch_materials, fetch_climatetrace
        one-off downloaders that populate ``data/raw``; they are not part of any
        reproduction path, because the reproduction pins the snapshot instead
    normalize
        unit and grade parsing, GWP per cubic metre, manufacturer resolution
    features
        decision-time features only, with no imputation statistic estimated here

Supports the manuscript's data section and the corpus counts quoted in the abstract.
"""
