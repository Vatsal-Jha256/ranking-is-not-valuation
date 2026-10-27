"""E5: strength-class empirical-quantile reporters against the learned conformal floor.

Protocol: ``experiments/protocols/stratified_quantile_baseline.md``.
Ledger:   ``results/exp_stratified_quantile_v1.json``.
Supplies: the simple-baseline comparison in RQ3 and Appendix D.

Reuses the E2 folds, candidate sets and argmin selection, so only the attached number
differs. Does not touch the E2 ledger.
"""
from __future__ import annotations

import json
import pathlib
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from rnv.claims.conformal import (  # noqa: E402
    CONFORMAL_ALPHA, N_CONFORMAL_SETS, conformal_quantile,
)
from rnv.claims.policies import N_CANDIDATES, RISK_LAMBDA  # noqa: E402
from rnv.evaluation.corpus import N_FOLDS, SPLIT_SEEDS, iter_folds, load_corpus  # noqa: E402
from rnv.evaluation.metrics import eligible_classes, gap_distribution, summarize  # noqa: E402
from rnv.evaluation.provenance import ledger_provenance  # noqa: E402
from rnv.paths import RESULTS  # noqa: E402

N_POLICY_SETS = 1_000
REPORTS = ("class_shift_report", "class_lookup_report")


def class_quantiles(pred, truth, classes, alpha, beta, rng):
    """Per-class shift and lookup quantiles from the calibration block only."""
    shift, lookup = {}, {}
    for label in eligible_classes(classes, 3 * N_CANDIDATES):
        mask = classes.astype(str) == label
        p, t = pred[mask], truth[mask]
        residuals = []
        for _ in range(N_CONFORMAL_SETS):
            idx = rng.choice(len(p), N_CANDIDATES, replace=False)
            row = idx[int(np.argmin(p[idx]))]
            residuals.append(t[row] - (alpha + beta * p[row]))
        shift[label] = conformal_quantile(np.asarray(residuals), CONFORMAL_ALPHA)
        lookup[label] = conformal_quantile(t, CONFORMAL_ALPHA)
    return shift, lookup


def evaluate(pred, truth, classes, alpha, beta, shift, lookup, rng, sigma):
    records = {r: {"gap": [], "advantage": [], "reported": []} for r in REPORTS}
    for label in eligible_classes(classes, 3 * N_CANDIDATES):
        if label not in shift:
            continue
        mask = classes.astype(str) == label
        p, t = pred[mask], truth[mask]
        median_truth = float(np.median(t))
        for _ in range(N_POLICY_SETS):
            idx = rng.choice(len(p), N_CANDIDATES, replace=False)
            # Consume the same draws as E2 so the sets match its stream.
            rng.normal(0.0, sigma, N_CANDIDATES)
            rng.normal(0.0, 1.0, N_CANDIDATES)
            win = idx[int(np.argmin(p[idx]))]
            reported = {
                "class_shift_report": alpha + beta * float(p[win]) + shift[label],
                "class_lookup_report": lookup[label],
            }
            for name, value in reported.items():
                records[name]["gap"].append(float(t[win]) - value)
                records[name]["advantage"].append(float(1.0 - np.exp(t[win] - median_truth)))
                records[name]["reported"].append(float(1.0 - np.exp(value - median_truth)))
    rows = {}
    for name, v in records.items():
        rows[name] = {
            **gap_distribution(np.asarray(v["gap"])),
            "realized_quality_fraction": float(np.mean(v["advantage"])),
            "reported_saving_fraction": float(np.mean(v["reported"])),
            "positive_saving_fraction": float(np.mean(np.asarray(v["reported"]) > 0.0)),
        }
    return rows


def main() -> None:
    started = datetime.now(timezone.utc).isoformat()
    corpus = load_corpus()
    evaluations, failures = [], []
    for fold in iter_folds(corpus):
        try:
            cmap = fold.calibration_map
            cal_truth = corpus.y[fold.calibration]
            shift, lookup = class_quantiles(
                fold.cal_pred, cal_truth, corpus.classes[fold.calibration],
                float(cmap.alpha), float(cmap.beta),
                np.random.default_rng(330_000 * fold.split_seed + fold.fold),
            )
            rows = evaluate(
                fold.test_pred, corpus.y[fold.test], corpus.classes[fold.test],
                float(cmap.alpha), float(cmap.beta), shift, lookup,
                np.random.default_rng(320_000 * fold.split_seed + fold.fold),
                float(np.sqrt(cmap.sigma2)),
            )
            numeric = [x for r in rows.values() for x in r.values() if isinstance(x, float)]
            if not np.all(np.isfinite(numeric)):
                raise RuntimeError("Non-finite metric")
            evaluations.append({"run_id": fold.run_id, "split_seed": fold.split_seed,
                                "fold": fold.fold, "n_classes_scored": len(shift),
                                "policies": rows})
            print(fold.run_id, {k: round(v["p_over_005"], 3) for k, v in rows.items()})
        except Exception as exc:
            failures.append({"run_id": fold.run_id, "error": f"{type(exc).__name__}: {exc}"})
            print(fold.run_id, "FAILED", exc)
    complete = not failures and len(evaluations) == len(SPLIT_SEEDS) * N_FOLDS
    metrics = ("mean", "p_positive", "p_over_005", "cvar90", "realized_quality_fraction",
               "reported_saving_fraction", "positive_saving_fraction")
    out = {
        "metadata": {
            "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": "E5 stratified empirical-quantile reporters",
            **ledger_provenance(__file__, "experiments/protocols/stratified_quantile_baseline.md"),
            "n_records": corpus.n_records, "n_manufacturers": corpus.n_manufacturers,
            "n_policy_sets_per_class": N_POLICY_SETS,
            "n_conformal_sets_per_class": N_CONFORMAL_SETS,
            "conformal_alpha": CONFORMAL_ALPHA, "risk_lambda": RISK_LAMBDA,
            "complete": complete,
        },
        "policy_summary": {
            r: {m: summarize(e["policies"][r][m] for e in evaluations) for m in metrics}
            for r in REPORTS
        },
        "evaluations": evaluations, "failures": failures,
    }
    path = RESULTS / "exp_stratified_quantile_v1.json"
    path.write_text(json.dumps(out, indent=2))
    print("wrote", path)
    if not complete:
        raise RuntimeError(f"Incomplete: {failures}")


if __name__ == "__main__":
    main()
