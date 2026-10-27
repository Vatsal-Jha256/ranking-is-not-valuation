"""E3: how much of the accepted policy result depends on the candidate pool?

Protocol: ``experiments/protocols/review_response_suite.md`` (E3).
Ledger:   ``results/exp_candidate_sensitivity_v1.json``.
Supplies: Appendix E, including the pool-sensitivity figure and the depth identity
drawn as a prediction rather than a fit.

The accepted manuscript fixes candidate sets at ten products drawn uniformly within a
strength class. Both choices are consequential. Selection depth grows with the number of
candidates ranked, so the size of the pool moves the quantity the manuscript's identity is
linear in. And a pool's supplier composition decides whose declarations the optimizer is
free to pick among.

A caveat governs the composition arm. This corpus records declarations, not sales, so it
contains no market share. Uniform sampling over rows already makes a supplier's chance of
appearing proportional to how many declarations it publishes, which is a volume weighting,
not a share weighting. The honest sensitivity check is therefore to bracket the accepted
sampler from both sides: remove the volume skew entirely by drawing suppliers first, and
tighten the pool to a handful of suppliers the way a limited delivery radius would.

This script never overwrites earlier result artifacts.
"""
from __future__ import annotations

import json
import pathlib
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from rnv.claims.conformal import (  # noqa: E402
    CONFORMAL_ALPHA,
    conformal_quantile,
    selection_conformal_quantile,
)
from rnv.claims.policies import RISK_LAMBDA  # noqa: E402
from rnv.evaluation.corpus import N_FOLDS, SPLIT_SEEDS, iter_folds, load_corpus  # noqa: E402
from rnv.evaluation.metrics import eligible_classes, gap_distribution, summarize  # noqa: E402
from rnv.evaluation.model import fit_spread_model  # noqa: E402
from rnv.evaluation.provenance import ledger_provenance  # noqa: E402
from rnv.paths import RESULTS  # noqa: E402

CANDIDATE_SIZES = (2, 5, 10, 20, 50)
SAMPLERS = ("uniform", "supplier_equal", "local")
N_SETS = 400
LOCAL_SUPPLIERS = 3
POLICIES = ("argmin", "score_perturbation", "risk_averse", "calibrated_report",
            "conformal_report")


def draw_set(sampler: str, size: int, rows: np.ndarray,
             rows_by_manufacturer: list[np.ndarray],
             rng: np.random.Generator) -> np.ndarray | None:
    """Draw one candidate set of distinct products under the requested composition rule."""
    if sampler == "uniform":
        return rng.choice(rows, size, replace=False)
    if sampler == "supplier_equal":
        chosen = rng.choice(len(rows_by_manufacturer), size, replace=False)
        return np.asarray([
            rows_by_manufacturer[index][rng.integers(len(rows_by_manufacturer[index]))]
            for index in chosen
        ])
    if sampler == "local":
        chosen = rng.choice(len(rows_by_manufacturer), LOCAL_SUPPLIERS, replace=False)
        pool = np.concatenate([rows_by_manufacturer[index] for index in chosen])
        if len(pool) < size:
            return None
        return rng.choice(pool, size, replace=False)
    raise ValueError(sampler)


def eligible(sampler: str, size: int, n_rows: int, n_manufacturers: int) -> bool:
    if n_rows < 3 * size:
        return False
    if sampler == "supplier_equal":
        return n_manufacturers >= size
    if sampler == "local":
        return n_manufacturers >= LOCAL_SUPPLIERS
    return True


def evaluate(sampler: str, size: int, pred: np.ndarray, truth: np.ndarray,
             spread: np.ndarray, classes: np.ndarray, manufacturers: np.ndarray,
             alpha: float, beta: float, sigma: float, q_selected: float,
             rng: np.random.Generator) -> dict[str, object] | None:
    records = {policy: {"gap": [], "depth": [], "advantage": []} for policy in POLICIES}
    classes_used = 0
    for label in eligible_classes(classes, 3 * size):
        mask = classes.astype(str) == label
        rows = np.flatnonzero(mask)
        groups = manufacturers[mask]
        unique = np.unique(groups)
        rows_by_manufacturer = [rows[groups == name] for name in unique]
        if not eligible(sampler, size, len(rows), len(unique)):
            continue
        median_truth = float(np.median(truth[rows]))
        classes_used += 1
        for _ in range(N_SETS):
            idx = draw_set(sampler, size, rows, rows_by_manufacturer, rng)
            if idx is None:
                continue
            set_p = pred[idx]
            set_t = truth[idx]
            set_s = spread[idx]
            set_mean = float(np.mean(set_p))
            best = int(np.argmin(set_p))
            winners = {
                "argmin": best,
                "score_perturbation": int(np.argmin(set_p + rng.normal(0.0, sigma, size))),
                "risk_averse": int(np.argmin(set_p + RISK_LAMBDA * set_s)),
            }
            reports = {
                "calibrated_report": alpha + beta * float(set_p[best]),
                "conformal_report": (alpha + beta * float(set_p[best])
                                     + q_selected * float(set_s[best])),
            }
            for policy, winner in winners.items():
                records[policy]["gap"].append(float(set_t[winner] - set_p[winner]))
                records[policy]["depth"].append(set_mean - float(set_p[winner]))
                records[policy]["advantage"].append(
                    float(1.0 - np.exp(set_t[winner] - median_truth))
                )
            for policy, reported in reports.items():
                records[policy]["gap"].append(float(set_t[best]) - reported)
                records[policy]["depth"].append(set_mean - float(set_p[best]))
                records[policy]["advantage"].append(
                    float(1.0 - np.exp(set_t[best] - median_truth))
                )

    if not classes_used or not records["argmin"]["gap"]:
        return None
    rows_out = {
        policy: {
            **gap_distribution(np.asarray(values["gap"])),
            "selection_depth_log": float(np.mean(values["depth"])),
            "realized_quality_fraction": float(np.mean(values["advantage"])),
        }
        for policy, values in records.items()
    }
    return {
        "sampler": sampler,
        "n_candidates": size,
        "n_classes": classes_used,
        "policies": rows_out,
    }


def main() -> None:
    started = datetime.now(timezone.utc).isoformat()
    corpus = load_corpus()

    evaluations = []
    failures = []
    skipped = []
    for fold in iter_folds(corpus):
        try:
            cmap = fold.calibration_map
            cal_truth = corpus.y[fold.calibration]
            cal_residual = cal_truth - cmap.transform(fold.cal_pred)
            spread_model = fit_spread_model(
                corpus.X[fold.calibration], cal_residual, fold.split_seed
            )
            cal_spread = np.exp(spread_model.predict(corpus.X[fold.calibration]))
            low, high = np.percentile(cal_spread, [1.0, 99.0])
            cal_spread = np.clip(cal_spread, low, high)
            test_spread = np.clip(
                np.exp(spread_model.predict(corpus.X[fold.test])), low, high
            )
            conformal_quantile(np.abs(cal_residual) / cal_spread, CONFORMAL_ALPHA)

            for size in CANDIDATE_SIZES:
                # The selection-conditional quantile depends on how many candidates the
                # winner beat, so it is recalibrated at every pool size.
                q_selected = selection_conformal_quantile(
                    fold.cal_pred, cal_truth, cal_spread,
                    corpus.classes[fold.calibration], float(cmap.alpha),
                    float(cmap.beta), CONFORMAL_ALPHA,
                    np.random.default_rng(
                        410_000 * fold.split_seed + 1_000 * fold.fold + size
                    ),
                    n_candidates=size,
                )
                for sampler in SAMPLERS:
                    result = evaluate(
                        sampler, size, fold.test_pred, corpus.y[fold.test], test_spread,
                        corpus.classes[fold.test], corpus.manufacturers[fold.test],
                        float(cmap.alpha), float(cmap.beta),
                        float(np.sqrt(cmap.sigma2)), q_selected,
                        np.random.default_rng(
                            420_000 * fold.split_seed + 1_000 * fold.fold
                            + 10 * size + SAMPLERS.index(sampler)
                        ),
                    )
                    if result is None:
                        skipped.append({
                            "run_id": fold.run_id,
                            "sampler": sampler,
                            "n_candidates": size,
                            "reason": "no eligible strength class",
                        })
                        continue
                    evaluations.append({
                        "run_id": fold.run_id,
                        "status": "success",
                        "split_seed": fold.split_seed,
                        "fold": fold.fold,
                        "partition_sha256": fold.partition_sha256,
                        "selection_conformal_quantile": q_selected,
                        **result,
                    })
            print(
                f"{fold.run_id}: cells={sum(row['run_id'] == fold.run_id for row in evaluations)}"
                f" skipped={sum(row['run_id'] == fold.run_id for row in skipped)}"
            )
        except Exception as exc:  # preserve every failed attempt in the ledger
            failures.append({
                "run_id": fold.run_id,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"{fold.run_id}: FAILED: {exc}")

    complete = not failures and len({row["run_id"] for row in evaluations}) == (
        len(SPLIT_SEEDS) * N_FOLDS
    )

    metrics = ("mean", "p_positive", "p_over_005", "q90", "cvar90",
               "selection_depth_log", "realized_quality_fraction")
    cells = {}
    for sampler in SAMPLERS:
        for size in CANDIDATE_SIZES:
            subset = [row for row in evaluations
                      if row["sampler"] == sampler and row["n_candidates"] == size]
            if not subset:
                continue
            cells[f"{sampler}_n{size}"] = {
                "sampler": sampler,
                "n_candidates": size,
                "n_evaluations": len(subset),
                "policies": {
                    policy: {
                        metric: summarize(
                            row["policies"][policy][metric] for row in subset
                        )
                        for metric in metrics
                    }
                    for policy in POLICIES
                },
            }

    output = {
        "metadata": {
            "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": "E3 candidate-set size and composition sensitivity",
            **ledger_provenance(
                __file__, "experiments/protocols/review_response_suite.md"
            ),
            "n_records": corpus.n_records,
            "n_manufacturers": corpus.n_manufacturers,
            "n_folds": N_FOLDS,
            "split_seeds": SPLIT_SEEDS,
            "candidate_sizes": list(CANDIDATE_SIZES),
            "samplers": list(SAMPLERS),
            "n_sets_per_class": N_SETS,
            "local_suppliers": LOCAL_SUPPLIERS,
            "composition_note": (
                "The corpus records declarations, not sales. Uniform row sampling already "
                "weights a supplier by publication volume; supplier_equal removes that "
                "weighting and local restricts the pool to a few suppliers. None of the "
                "three is a market-share simulation."
            ),
            "all_runs_reported": True,
            "complete": complete,
            "sensitivity_interval_note": (
                "Percentiles summarize split/fold sensitivity on one corpus; "
                "they are not independent-data confidence intervals."
            ),
        },
        "cells": cells,
        "evaluations": evaluations,
        "skipped": skipped,
        "failures": failures,
    }
    out_path = RESULTS / "exp_candidate_sensitivity_v1.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"wrote {out_path}")
    if not complete:
        raise RuntimeError(f"Incomplete experiment: {failures}")


if __name__ == "__main__":
    main()
