"""The estimators the project fits, and nothing else.

Two models appear in the manuscript, and they answer different questions.

:func:`fit_model` is the recommender under study: a gradient-boosted regressor on the
log of declared GWP. Its hyperparameters are fixed across every experiment and were
never tuned against a held-out block, so a fold's numbers are not a search result.
The median imputer sits inside the pipeline on purpose -- it is refitted on each
training fold, so a held-out manufacturer never influences how missing values are
filled.

:func:`fit_spread_model` is the scale function used by the normalized conformal rules.
It predicts the log absolute calibration residual, so its exponential is a strictly
positive per-product scale. It is not a posterior variance and the manuscript does not
claim it is one; it only has to be monotone in how uncertain a product is for the
normalized nonconformity score to be useful.

Supplies: the estimator described in the manuscript's method section (RQ1--RQ3) and
the heteroscedastic scale used by Appendix D's conformal rules.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

#: Smallest absolute residual the scale model is allowed to see, so that a residual of
#: exactly zero does not become a negative infinity in log space.
SPREAD_FLOOR = 1e-4


def fit_model(X: pd.DataFrame, y: np.ndarray, seed: int):
    """Fit the accepted GWP model on one training block.

    Args:
        X: decision-time features from :mod:`rnv.registry.features`.
        y: natural log of declared GWP per cubic metre.
        seed: estimator seed; experiments pass the partition seed so a fold is
            reproducible from its identifier alone.
    """
    return make_pipeline(
        SimpleImputer(strategy="median"),
        GradientBoostingRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=seed,
        ),
    ).fit(X, y)


def fit_spread_model(X: pd.DataFrame, residual: np.ndarray, seed: int):
    """Heteroscedastic scale model, fitted on the calibration block only.

    Predicts log absolute calibration residual, so the exponentiated prediction is a
    strictly positive per-product scale. This is the standard normalized-nonconformity
    scale function; it is not a posterior variance and is not claimed to be one.
    """
    target = np.log(np.maximum(np.abs(residual), SPREAD_FLOOR))
    return make_pipeline(
        SimpleImputer(strategy="median"),
        GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05, random_state=seed
        ),
    ).fit(X, target)
