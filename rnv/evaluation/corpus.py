"""The corpus and the 24 accepted evaluations.

This module is the single definition of "one fold" for the whole project. Everything an
experiment needs -- which manufacturers train the model, which calibrate it, which are
held out, the fitted model, and the calibration map -- is produced here and nowhere
else, so two experiments cannot end up meaning different things by the same word.

The protocol, unchanged across every reported experiment:

* Six partition seeds, four blocks each, giving 24 evaluations.
* Blocks are whole manufacturers, size-stratified, and genuinely reseeded.
* Within a seed, block ``f`` is the test block, block ``f + 1`` is the calibration
  block, and the rest train. The three roles are disjoint by manufacturer, and the
  code raises rather than proceeds if that is ever violated.
* Imputation is fitted inside the training fold, so a held-out manufacturer never
  influences preprocessing.

The one experiment that departs from this is the temporal holdout, which cuts on
declaration date instead; it says so in its own docstring and in its protocol.

Supplies: the evaluation protocol of the manuscript (all of RQ1--RQ3), Appendix B, and
the fold machinery behind Appendices C--F.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pandas as pd

from rnv.claims.certification import CalibrationMap
from rnv.evaluation.model import fit_model
from rnv.evaluation.partitions import stratified_random_group_blocks
from rnv.registry.features import build_features_raw
from rnv.registry.normalize import load_jsonl, normalize_concrete

#: Blocks per partition. Four gives a held-out block large enough for within-class
#: pairwise metrics while leaving two blocks to train on.
N_FOLDS = 4

#: Partition seeds. Every reported number is averaged over all six; none is dropped.
SPLIT_SEEDS = list(range(6))

#: A strength class needs this many rows before within-class pairs are drawn from it.
MIN_CLASS_ROWS_FOR_PAIRS = 20


@dataclass
class Corpus:
    """The normalized ready-mix corpus with decision-time features."""

    data: pd.DataFrame
    X: pd.DataFrame
    y: np.ndarray
    manufacturers: np.ndarray
    classes: np.ndarray
    created: np.ndarray

    @property
    def n_records(self) -> int:
        return int(len(self.data))

    @property
    def n_manufacturers(self) -> int:
        return int(pd.Series(self.manufacturers).nunique())


@dataclass
class Fold:
    """One seed/fold evaluation with disjoint manufacturer roles."""

    split_seed: int
    fold: int
    run_id: str
    partition_sha256: str
    train: np.ndarray
    calibration: np.ndarray
    test: np.ndarray
    model: object
    calibration_map: CalibrationMap
    cal_pred: np.ndarray
    test_pred: np.ndarray
    extras: dict = field(default_factory=dict)


def load_corpus() -> Corpus:
    """Load, normalize and featurize the pinned registry snapshot.

    Rows without a resolvable strength class are dropped: products that cannot be
    placed in a functional class cannot take part in a within-class comparison, which
    is the only comparison this project makes.
    """
    data, _ = normalize_concrete(load_jsonl("readymix"), verbose=False)
    data = data[data.strength_class.notna()].copy()
    X = build_features_raw(data).reset_index(drop=True)
    y = np.log(data.gwp_per_m3.to_numpy())
    manufacturers = data.get("mfr_name").fillna("unknown").to_numpy()
    classes = data.strength_class.to_numpy()
    created = pd.to_datetime(data["created_on"], errors="coerce", utc=True).to_numpy()
    return Corpus(data=data, X=X, y=y, manufacturers=manufacturers,
                  classes=classes, created=created)


def iter_folds(corpus: Corpus) -> Iterator[Fold]:
    """Yield the 24 accepted seed/fold evaluations with fitted model and calibration."""
    for split_seed in SPLIT_SEEDS:
        fold_labels, _, partition_digest = stratified_random_group_blocks(
            corpus.manufacturers, n_blocks=N_FOLDS, seed=split_seed
        )
        for fold_index in range(N_FOLDS):
            test = fold_labels == fold_index
            calibration = fold_labels == ((fold_index + 1) % N_FOLDS)
            train = ~(test | calibration)
            train_mfr = set(corpus.manufacturers[train])
            cal_mfr = set(corpus.manufacturers[calibration])
            test_mfr = set(corpus.manufacturers[test])
            if train_mfr & cal_mfr or train_mfr & test_mfr or cal_mfr & test_mfr:
                raise RuntimeError("Manufacturer overlap between split roles")

            model = fit_model(corpus.X[train], corpus.y[train], split_seed)
            cal_pred = model.predict(corpus.X[calibration])
            cmap = CalibrationMap().fit(
                cal_pred, corpus.y[calibration],
                groups=corpus.manufacturers[calibration],
            )
            yield Fold(
                split_seed=split_seed,
                fold=fold_index,
                run_id=f"seed{split_seed}_fold{fold_index}",
                partition_sha256=partition_digest,
                train=train,
                calibration=calibration,
                test=test,
                model=model,
                calibration_map=cmap,
                cal_pred=cal_pred,
                test_pred=model.predict(corpus.X[test]),
                extras={
                    "n_train": int(train.sum()),
                    "n_calibration": int(calibration.sum()),
                    "n_test": int(test.sum()),
                    "n_train_manufacturers": len(train_mfr),
                    "n_calibration_manufacturers": len(cal_mfr),
                    "n_test_manufacturers": len(test_mfr),
                },
            )
