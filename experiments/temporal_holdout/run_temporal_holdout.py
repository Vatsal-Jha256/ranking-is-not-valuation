"""E4: does the ranking-versus-valuation separation survive a shift in time?

Protocol: ``experiments/protocols/review_response_suite.md`` (E4).
Ledger:   ``results/exp_temporal_holdout_v1.json``.
Supplies: Appendix F and the external-validity paragraph of the discussion.

Every other experiment in this project holds out suppliers from a single snapshot. That
leaves one obvious way the result could be an artifact: the corpus was pulled at one
moment, so a model and its evaluation share whatever was true of declaration practice at
that moment. Product category rules, plant mixes, and the set of firms that publish at all
change from year to year.

Departure from the accepted protocol, and the only one: partitions here are cuts in
declaration date, not seeded manufacturer blocks. Train on everything declared before a
cutoff, test on what came after. The strict variant additionally keeps only post-cutoff
suppliers that never appear before it, so temporal drift and supplier shift compound
instead of being confounded.

This script never overwrites earlier result artifacts.
"""
from __future__ import annotations

import json
import pathlib
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from rnv.claims.certification import CalibrationMap  # noqa: E402
from rnv.evaluation.corpus import load_corpus  # noqa: E402
from rnv.evaluation.metrics import (  # noqa: E402
    eligible_classes,
    gap_distribution,
    median_agreement,
    pairwise_accuracy,
    r2_score_test_mean,
    summarize,
    within_class_spearman,
)
from rnv.evaluation.model import fit_model  # noqa: E402
from rnv.evaluation.provenance import ledger_provenance  # noqa: E402
from rnv.paths import RESULTS  # noqa: E402

CUTOFF_YEARS = (2022, 2023, 2024)
VARIANTS = ("temporal", "temporal_disjoint")
MIN_TEST_RECORDS = 2_000
N_POLICY_SETS = 1_000
N_CANDIDATES = 10
CALIBRATION_FRACTION = 0.5


def argmin_policy(pred: np.ndarray, truth: np.ndarray, classes: np.ndarray,
                  alpha: float, beta: float,
                  rng: np.random.Generator) -> dict[str, float] | None:
    """The accepted deterministic policy, with its raw and recalibrated claim gaps."""
    raw, calibrated, depth, advantage = [], [], [], []
    for label in eligible_classes(classes, 3 * N_CANDIDATES):
        mask = classes.astype(str) == label
        p = pred[mask]
        t = truth[mask]
        median_truth = float(np.median(t))
        for _ in range(N_POLICY_SETS):
            idx = rng.choice(len(p), N_CANDIDATES, replace=False)
            winner = int(np.argmin(p[idx]))
            score = float(p[idx][winner])
            declared = float(t[idx][winner])
            raw.append(declared - score)
            calibrated.append(declared - (alpha + beta * score))
            depth.append(float(np.mean(p[idx])) - score)
            advantage.append(float(1.0 - np.exp(declared - median_truth)))
    if not raw:
        return None
    return {
        "raw": gap_distribution(np.asarray(raw)),
        "calibrated_report": gap_distribution(np.asarray(calibrated)),
        "selection_depth_log": float(np.mean(depth)),
        "realized_quality_fraction": float(np.mean(advantage)),
    }


def main() -> None:
    started = datetime.now(timezone.utc).isoformat()
    corpus = load_corpus()
    created = pd.to_datetime(pd.Series(corpus.created), utc=True)
    if created.isna().any():
        raise RuntimeError("Declaration dates are required for the temporal holdout")
    first_seen = created.groupby(pd.Series(corpus.manufacturers)).min()

    evaluations = []
    failures = []
    for cutoff_year in CUTOFF_YEARS:
        cutoff = pd.Timestamp(f"{cutoff_year}-01-01", tz="UTC")
        train = (created < cutoff).to_numpy()
        later = ~train
        established = pd.Series(corpus.manufacturers).map(first_seen) < cutoff
        for variant in VARIANTS:
            run_id = f"cutoff{cutoff_year}_{variant}"
            try:
                test_pool = (later & ~established.to_numpy() if variant == "temporal_disjoint"
                             else later)
                if int(train.sum()) < MIN_TEST_RECORDS:
                    raise RuntimeError("Training period too small")
                if int(test_pool.sum()) < MIN_TEST_RECORDS:
                    raise RuntimeError(
                        f"Test period holds {int(test_pool.sum())} records, "
                        f"fewer than {MIN_TEST_RECORDS}"
                    )

                # Half the post-cutoff suppliers calibrate; the other half is tested. The
                # split is by supplier so calibration never sees a tested manufacturer.
                test_manufacturers = np.unique(corpus.manufacturers[test_pool])
                order = np.random.default_rng(cutoff_year).permutation(
                    len(test_manufacturers)
                )
                cut = max(1, int(CALIBRATION_FRACTION * len(test_manufacturers)))
                calibration_names = set(test_manufacturers[order[:cut]])
                in_calibration = np.asarray([
                    name in calibration_names for name in corpus.manufacturers
                ])
                calibration = test_pool & in_calibration
                test = test_pool & ~in_calibration
                if not calibration.any() or not test.any():
                    raise RuntimeError("Empty calibration or test partition")

                model = fit_model(corpus.X[train], corpus.y[train], cutoff_year)
                cal_pred = model.predict(corpus.X[calibration])
                cmap = CalibrationMap().fit(
                    cal_pred, corpus.y[calibration],
                    groups=corpus.manufacturers[calibration],
                )
                pred = model.predict(corpus.X[test])
                truth = corpus.y[test]
                test_classes = corpus.classes[test]

                pair_rng = np.random.default_rng(600_000 + cutoff_year)
                accuracy, tie_rate, n_pairs = pairwise_accuracy(
                    pred, truth, test_classes, pair_rng
                )
                policy = argmin_policy(
                    pred, truth, test_classes, float(cmap.alpha), float(cmap.beta),
                    np.random.default_rng(610_000 + cutoff_year),
                )
                if policy is None:
                    raise RuntimeError("No eligible strength class in the test period")

                row = {
                    "run_id": run_id,
                    "status": "success",
                    "cutoff_year": cutoff_year,
                    "variant": variant,
                    "n_train": int(train.sum()),
                    "n_calibration": int(calibration.sum()),
                    "n_test": int(test.sum()),
                    "n_train_manufacturers": int(
                        pd.Series(corpus.manufacturers[train]).nunique()
                    ),
                    "n_test_manufacturers": int(
                        pd.Series(corpus.manufacturers[test]).nunique()
                    ),
                    "test_r2": r2_score_test_mean(truth, pred),
                    "within_class_spearman": within_class_spearman(
                        pred, truth, test_classes
                    ),
                    "pairwise_accuracy": accuracy,
                    "prediction_tie_rate": tie_rate,
                    "n_pairwise_comparisons": n_pairs,
                    "median_agreement": median_agreement(
                        pred, truth, test_classes,
                        np.random.default_rng(620_000 + cutoff_year),
                    ),
                    "calibration_alpha": float(cmap.alpha),
                    "calibration_beta": float(cmap.beta),
                    "policy": policy,
                }
                evaluations.append(row)
                print(
                    f"{run_id}: n_test={row['n_test']} R2={row['test_r2']:+.3f} "
                    f"pair={accuracy:.3f} beta={row['calibration_beta']:.3f} "
                    f"gap={policy['raw']['mean']:+.4f}"
                )
            except Exception as exc:  # preserve every failed attempt in the ledger
                failures.append({
                    "run_id": run_id,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                print(f"{run_id}: FAILED: {exc}")

    complete = not failures and len(evaluations) == len(CUTOFF_YEARS) * len(VARIANTS)

    summary = {}
    for variant in VARIANTS:
        subset = [row for row in evaluations if row["variant"] == variant]
        if not subset:
            continue
        summary[variant] = {
            "test_r2": summarize(row["test_r2"] for row in subset),
            "pairwise_accuracy": summarize(
                row["pairwise_accuracy"] for row in subset
            ),
            "calibration_beta": summarize(
                row["calibration_beta"] for row in subset
            ),
            "raw_gap": summarize(row["policy"]["raw"]["mean"] for row in subset),
            "raw_p_over_005": summarize(
                row["policy"]["raw"]["p_over_005"] for row in subset
            ),
            "calibrated_gap": summarize(
                row["policy"]["calibrated_report"]["mean"] for row in subset
            ),
            "realized_quality_fraction": summarize(
                row["policy"]["realized_quality_fraction"] for row in subset
            ),
            "n_negative_r2": int(sum(row["test_r2"] < 0 for row in subset)),
            "n_pairwise_above_half": int(
                sum(row["pairwise_accuracy"] > 0.5 for row in subset)
            ),
            "n_evaluations": len(subset),
        }

    output = {
        "metadata": {
            "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": "E4 temporal holdout",
            **ledger_provenance(
                __file__, "experiments/protocols/review_response_suite.md"
            ),
            "n_records": corpus.n_records,
            "n_manufacturers": corpus.n_manufacturers,
            "cutoff_years": list(CUTOFF_YEARS),
            "variants": list(VARIANTS),
            "n_policy_sets_per_class": N_POLICY_SETS,
            "n_candidates": N_CANDIDATES,
            "date_field": "first declaration timestamp recorded by the registry",
            "protocol_departure": (
                "Partitions cut on declaration date instead of seeded manufacturer "
                "blocks; every other setting follows the accepted protocol."
            ),
            "all_runs_reported": True,
            "complete": complete,
        },
        "summary": summary,
        "evaluations": evaluations,
        "failures": failures,
    }
    out_path = RESULTS / "exp_temporal_holdout_v1.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"wrote {out_path}")
    if not complete:
        raise RuntimeError(f"Incomplete experiment: {failures}")


if __name__ == "__main__":
    main()
