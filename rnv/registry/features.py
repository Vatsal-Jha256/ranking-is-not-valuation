"""Leakage-safe feature construction for ready-mix recommendation experiments.

These are the decision-time attributes a specifier actually has before a product
is chosen: functional requirement, plant location, and what the declaration says
about the binder. No feature is derived from the declared GWP being predicted.

Supplies: the feature vector used by every experiment in ``experiments/``, and the
feature list given in the manuscript's method section.
"""
from __future__ import annotations

import pandas as pd


def build_features_raw(data: pd.DataFrame) -> pd.DataFrame:
    """Build decision-time features without estimating any imputation statistic.

    Missing-value handling belongs inside each training fold. Keeping it out of
    this function prevents held-out manufacturers from influencing preprocessing.
    """
    features = pd.DataFrame(index=data.index)
    features["strength_mpa"] = data.strength_mpa
    features["density"] = data.density_kgm3
    if "concrete_w_c_ratio_n" in data:
        features["wc_ratio"] = pd.to_numeric(
            data["concrete_w_c_ratio_n"], errors="coerce"
        )
    features["lat"] = pd.to_numeric(data.get("mfr_lat"), errors="coerce")
    features["lon"] = pd.to_numeric(data.get("mfr_lon"), errors="coerce")
    text = (
        data.get("description").fillna("")
        + " "
        + data.get("name").fillna("")
    ).str.lower()
    for keyword in [
        "slag", "fly ash", "flyash", "ggbs", "pozz", "type i", "type ii",
        "lightweight", "pump", "psi", "air",
    ]:
        features["kw_" + keyword.replace(" ", "_")] = text.str.contains(
            keyword, regex=False
        ).astype(int)
    return features


def build_features_global_median(d):
    """Same features, but imputed with a median taken over the whole corpus.

    Estimating the imputation statistic before splitting lets held-out manufacturers
    influence preprocessing, which is why nothing the manuscript reports uses this. It
    is kept because the pilot and two superseded designs were run with it, and the
    experiment inclusion audit records that as one reason their numbers are not
    reported. Rewriting them to use :func:`build_features_raw` would change what was
    run without changing what was concluded.
    """
    X = pd.DataFrame(index=d.index)
    X["strength_mpa"] = d.strength_mpa
    X["density"] = d.density_kgm3
    for c, name in [("concrete_w_c_ratio_n", "wc_ratio")]:
        if c in d:
            X[name] = pd.to_numeric(d[c], errors="coerce")
    # provenance / geography
    X["lat"] = pd.to_numeric(d.get("mfr_lat"), errors="coerce")
    X["lon"] = pd.to_numeric(d.get("mfr_lon"), errors="coerce")
    # coarse mix descriptors parsed from free-text where present
    txt = (d.get("description").fillna("") + " " + d.get("name").fillna("")).str.lower()
    for kw in ["slag", "fly ash", "flyash", "ggbs", "pozz", "type i", "type ii",
               "lightweight", "pump", "psi", "air"]:
        X["kw_" + kw.replace(" ", "_")] = txt.str.contains(kw, regex=False).astype(int)
    return X.fillna(X.median(numeric_only=True))

