"""Headline experiment: ranking quality against valuation quality, and the policy
frontier that follows from it.

Protocol: ``experiments/protocols/selection_gap.md`` and its v2-v4 addenda.
Ledger:   ``results/exp_submission_harmonized_v4.json``.

Per held-out block this measures two things that are usually reported as one. A
*diagnostic* model, trained on everything except the test block, is scored on how well
it orders products within a strength class and on how well it predicts their declared
GWP. A *policy* model, trained on a further-reduced block so that a disjoint
calibration block exists, is then used to run candidate sets through five selection
rules, and each rule's claim is compared with what the winner actually declares.

The pointwise selection-depth bound is asserted, not assumed: if any randomized policy
ever achieves a deeper selection than argmin, the run fails rather than reporting it.

Supplies: the manuscript's RQ1 diagnostic numbers and their sensitivity intervals, the
pooled calibration slope quoted in RQ2, the policy frontier of RQ3, the paired
argmin-versus-perturbation comparison, and the left panels of Figure 1.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from rnv.claims.certification import CalibrationMap  # noqa: E402
from rnv.claims.policies import HEADLINE_POLICIES as POLICIES, N_CANDIDATES, choose  # noqa: E402
from rnv.evaluation.corpus import N_FOLDS, SPLIT_SEEDS  # noqa: E402
from rnv.evaluation.metrics import (  # noqa: E402
    median_agreement,
    pairwise_accuracy,
    r2_score_test_mean,
    summarize,
    within_class_spearman,
)
from rnv.evaluation.model import fit_model  # noqa: E402
from rnv.evaluation.partitions import stratified_random_group_blocks  # noqa: E402
from rnv.evaluation.provenance import ledger_provenance  # noqa: E402
from rnv.paths import RESULTS  # noqa: E402
from rnv.registry.features import build_features_raw  # noqa: E402
from rnv.registry.normalize import load_jsonl, normalize_concrete  # noqa: E402

#: Candidate sets drawn per strength class, per policy, per evaluation.
N_POLICY_SETS = 1_000


def evaluate_policies(pred: np.ndarray, truth: np.ndarray,
                      classes: np.ndarray, alpha: float, beta: float, sigma: float,
                      rng: np.random.Generator) -> dict[str, dict[str, float]]:
    acc = {
        policy: {"pred": [], "truth": [], "quality": [], "depth": []}
        for policy in POLICIES
    }
    for label in pd.unique(classes):
        mask = classes == label
        p = pred[mask]
        t = truth[mask]
        if len(p) < 3 * N_CANDIDATES:
            continue
        median_truth = float(np.median(t))
        for _ in range(N_POLICY_SETS):
            idx = rng.choice(len(p), N_CANDIDATES, replace=False)
            set_p = p[idx]
            set_t = t[idx]
            set_mean = float(np.mean(set_p))
            for policy_index, policy in enumerate(POLICIES):
                policy_rng = np.random.default_rng(
                    int(rng.integers(0, 2**32 - 1)) + policy_index
                )
                winner = choose(policy, set_p, sigma, policy_rng)
                acc[policy]["pred"].append(float(set_p[winner]))
                acc[policy]["truth"].append(float(set_t[winner]))
                acc[policy]["quality"].append(
                    float(1.0 - np.exp(set_t[winner] - median_truth))
                )
                acc[policy]["depth"].append(set_mean - float(set_p[winner]))

    rows = {}
    for policy, values in acc.items():
        p = np.exp(np.asarray(values["pred"]))
        t = np.exp(np.asarray(values["truth"]))
        rows[policy] = {
            "over_claim_log": float(np.mean(np.asarray(values["truth"]) -
                                                 np.asarray(values["pred"]))),
            "calibration_residual_log": float(np.mean(
                np.asarray(values["truth"]) - alpha
                - beta * np.asarray(values["pred"])
            )),
            "over_claim_pct": float(100.0 * (t.mean() - p.mean()) / p.mean()),
            "realized_quality_log": float(
                np.mean(-np.log1p(-np.asarray(values["quality"])))
            ),
            "realized_quality_fraction": float(np.mean(values["quality"])),
            "selection_depth_log": float(np.mean(values["depth"])),
            "n_selected": int(len(p)),
        }
    max_depth = max(row["selection_depth_log"] for row in rows.values())
    if not np.isclose(rows["argmin"]["selection_depth_log"], max_depth,
                      rtol=0.0, atol=1e-12):
        raise RuntimeError("Argmin failed the pointwise selection-depth bound")
    return rows


def main() -> None:
    started = datetime.now(timezone.utc).isoformat()
    data, _ = normalize_concrete(load_jsonl("readymix"), verbose=False)
    data = data[data.strength_class.notna()].copy()
    X = build_features_raw(data).reset_index(drop=True)
    y = np.log(data.gwp_per_m3.to_numpy())
    manufacturers = data.get("mfr_name").fillna("unknown").to_numpy()
    classes = data.strength_class.to_numpy()

    evaluations = []
    failures = []
    test_manufacturer_sets = {}
    for split_seed in SPLIT_SEEDS:
        fold, block_loads, partition_digest = stratified_random_group_blocks(
            manufacturers, n_blocks=N_FOLDS, seed=split_seed
        )
        for fold_index in range(N_FOLDS):
            run_id = f"seed{split_seed}_fold{fold_index}"
            try:
                test = fold == fold_index
                calibration = fold == ((fold_index + 1) % N_FOLDS)
                train = ~(test | calibration)
                train_mfr = set(manufacturers[train])
                cal_mfr = set(manufacturers[calibration])
                test_mfr = set(manufacturers[test])
                test_manufacturer_sets[(split_seed, fold_index)] = test_mfr
                if train_mfr & cal_mfr or train_mfr & test_mfr or cal_mfr & test_mfr:
                    raise RuntimeError("Manufacturer overlap between split roles")

                diagnostic_train = ~test
                diagnostic_model = fit_model(
                    X[diagnostic_train], y[diagnostic_train], split_seed
                )
                diagnostic_pred = diagnostic_model.predict(X[test])

                policy_model = fit_model(X[train], y[train], split_seed)
                cal_pred = policy_model.predict(X[calibration])
                cmap = CalibrationMap().fit(
                    cal_pred, y[calibration], groups=manufacturers[calibration]
                )
                pred = policy_model.predict(X[test])
                truth = y[test]
                test_classes = classes[test]
                rng = np.random.default_rng(100_000 * split_seed + fold_index)
                diagnostic_pair, diagnostic_ties, diagnostic_pairs = pairwise_accuracy(
                    diagnostic_pred, truth, test_classes,
                    np.random.default_rng(500_000 * split_seed + fold_index),
                )
                policy_pair, policy_ties, policy_pairs = pairwise_accuracy(
                    pred, truth, test_classes,
                    np.random.default_rng(500_000 * split_seed + fold_index),
                )
                row = {
                    "run_id": run_id,
                    "status": "success",
                    "split_seed": split_seed,
                    "fold": fold_index,
                    "partition_sha256": partition_digest,
                    "test_manufacturer_sha256": hashlib.sha256(
                        "\\n".join(sorted(map(str, test_mfr))).encode("utf-8")
                    ).hexdigest(),
                    "block_loads": [int(value) for value in block_loads],
                    "n_train": int(train.sum()),
                    "n_diagnostic_train": int(diagnostic_train.sum()),
                    "n_calibration": int(calibration.sum()),
                    "n_test": int(test.sum()),
                    "n_train_manufacturers": len(train_mfr),
                    "n_calibration_manufacturers": len(cal_mfr),
                    "n_test_manufacturers": len(test_mfr),
                    "diagnostic_r2": r2_score_test_mean(truth, diagnostic_pred),
                    "diagnostic_within_class_spearman": within_class_spearman(
                        diagnostic_pred, truth, test_classes
                    ),
                    "diagnostic_pairwise_accuracy": diagnostic_pair,
                    "diagnostic_prediction_tie_rate": diagnostic_ties,
                    "diagnostic_n_pairwise_comparisons": diagnostic_pairs,
                    "diagnostic_median_agreement": median_agreement(
                        diagnostic_pred, truth, test_classes, rng
                    ),
                    "policy_model_r2": r2_score_test_mean(truth, pred),
                    "policy_model_pairwise_accuracy": policy_pair,
                    "policy_model_prediction_tie_rate": policy_ties,
                    "policy_model_n_pairwise_comparisons": policy_pairs,
                    "calibration_alpha": float(cmap.alpha),
                    "calibration_beta": float(cmap.beta),
                    "calibration_sigma": float(np.sqrt(cmap.sigma2)),
                }
                row["policies"] = evaluate_policies(
                    pred, truth, test_classes, row["calibration_alpha"],
                    row["calibration_beta"], row["calibration_sigma"], rng
                )
                numeric_values = [
                    value for key, value in row.items()
                    if isinstance(value, float)
                ] + [
                    value for policy_row in row["policies"].values()
                    for value in policy_row.values() if isinstance(value, float)
                ]
                if not np.all(np.isfinite(numeric_values)):
                    raise RuntimeError("Non-finite metric")
                evaluations.append(row)
                print(
                    f"{run_id}: diagnostic R2={row['diagnostic_r2']:+.3f} "
                    f"pair={row['diagnostic_pairwise_accuracy']:.3f} "
                    f"median={row['diagnostic_median_agreement']:.3f}"
                )
            except Exception as exc:  # preserve all failed attempts in output
                failures.append({
                    "run_id": run_id,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                print(f"{run_id}: FAILED: {exc}")

    complete = not failures and len(evaluations) == len(SPLIT_SEEDS) * N_FOLDS

    metric_summary = {
        metric: summarize([row[metric] for row in evaluations])
        for metric in ("diagnostic_r2", "diagnostic_within_class_spearman",
                       "diagnostic_pairwise_accuracy",
                       "diagnostic_prediction_tie_rate",
                       "diagnostic_median_agreement", "policy_model_r2",
                       "policy_model_pairwise_accuracy",
                       "policy_model_prediction_tie_rate", "calibration_beta")
    }
    policy_summary = {}
    for policy in POLICIES:
        policy_summary[policy] = {
            metric: summarize([
                row["policies"][policy][metric] for row in evaluations
            ])
            for metric in ("over_claim_log", "over_claim_pct",
                           "calibration_residual_log",
                           "realized_quality_log", "realized_quality_fraction",
                           "selection_depth_log")
        }

    same_fold_jaccards = []
    for fold_index in range(N_FOLDS):
        for left_seed in SPLIT_SEEDS:
            for right_seed in SPLIT_SEEDS:
                if right_seed <= left_seed:
                    continue
                left = test_manufacturer_sets[(left_seed, fold_index)]
                right = test_manufacturer_sets[(right_seed, fold_index)]
                same_fold_jaccards.append({
                    "fold": fold_index,
                    "left_seed": left_seed,
                    "right_seed": right_seed,
                    "jaccard": float(len(left & right) / len(left | right)),
                })

    output = {
        "metadata": {
            "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            **ledger_provenance(
                __file__,
                "experiments/protocols/selection_gap.md",
                "experiments/protocols/selection_gap_addendum_v2.md",
                "experiments/protocols/selection_gap_addendum_v3.md",
                "experiments/protocols/selection_gap_addendum_v4.md",
            ),
            "n_records": int(len(data)),
            "n_manufacturers": int(pd.Series(manufacturers).nunique()),
            "n_folds": N_FOLDS,
            "split_seeds": SPLIT_SEEDS,
            "partition_method": "seeded size-stratified manufacturer blocks",
            "n_unique_partition_digests": len({
                row["partition_sha256"] for row in evaluations
            }),
            "all_runs_reported": True,
            "complete": complete,
            "sensitivity_interval_note": (
                "Percentiles summarize split/fold sensitivity on one corpus; "
                "they are not independent-data confidence intervals."
            ),
        },
        "metric_summary": metric_summary,
        "policy_summary": policy_summary,
        "partition_diagnostics": {
            "same_fold_test_manufacturer_jaccards": same_fold_jaccards,
            "jaccard_summary": summarize([
                row["jaccard"] for row in same_fold_jaccards
            ]),
        },
        "evaluations": evaluations,
        "failures": failures,
    }
    out_path = RESULTS / "exp_submission_harmonized_v4.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"wrote {out_path}")
    if not complete:
        raise RuntimeError(f"Incomplete experiment: {failures}")


if __name__ == "__main__":
    main()
