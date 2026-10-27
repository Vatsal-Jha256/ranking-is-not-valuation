"""E2: uncertainty-aware selection and reporting rules, scored on their tail risk.

Protocol: ``experiments/protocols/review_response_suite.md`` (E2).
Ledger:   ``results/exp_uncertainty_policies_v1.json``.
Supplies: RQ3 of the manuscript and Appendix D, including the selection-versus-
reporting figure and the accepted-policy comparison.

The accepted manuscript compares deterministic argmin against four randomized softenings of
it, and summarizes each by a signed mean. Two things are missing. First, none of the
alternatives uses an estimate of its own uncertainty, so none of them can be more careful
exactly where evidence is thin. Second, a signed mean hides offsetting errors: a policy
whose claims are right on average can still overstate badly on a large minority of items,
which is what an auditor actually cares about.

This experiment separates the two levers that the accepted paper's identity conflates.
A SELECTION rule decides which product is recommended and therefore fixes both the
selection depth and the realized declared-GWP advantage. A REPORT rule decides what number
is attached to that recommendation and changes nothing about the product chosen. Report
rules are applied to the identical argmin selection, so their advantage cost is zero by
construction -- the interesting question is only how much of the claim gap, and of its
upper tail, they remove.

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
    N_CONFORMAL_SETS,
    conformal_quantile,
    selection_conformal_quantile,
)
from rnv.claims.policies import (  # noqa: E402
    N_CANDIDATES,
    REPORT_POLICIES,
    RISK_LAMBDA,
    SELECTION_POLICIES,
)
from rnv.evaluation.corpus import N_FOLDS, SPLIT_SEEDS, iter_folds, load_corpus  # noqa: E402
from rnv.evaluation.metrics import eligible_classes, gap_distribution, summarize  # noqa: E402
from rnv.evaluation.model import fit_spread_model  # noqa: E402
from rnv.evaluation.provenance import ledger_provenance  # noqa: E402
from rnv.paths import RESULTS  # noqa: E402

#: Candidate sets drawn per strength class, per evaluation.
N_POLICY_SETS = 1_000

#: Every rule compared here: selection rules first, then reporting rules.
POLICIES = SELECTION_POLICIES + REPORT_POLICIES


def evaluate(pred: np.ndarray, truth: np.ndarray, spread: np.ndarray,
             classes: np.ndarray, alpha: float, beta: float, sigma: float,
             q_norm: float, q_selected: float,
             rng: np.random.Generator) -> dict[str, dict[str, float]]:
    records = {
        policy: {"gap": [], "depth": [], "advantage": [], "reported_advantage": []}
        for policy in POLICIES
    }
    for label in eligible_classes(classes, 3 * N_CANDIDATES):
        mask = classes.astype(str) == label
        p = pred[mask]
        t = truth[mask]
        s = spread[mask]
        median_truth = float(np.median(t))
        for _ in range(N_POLICY_SETS):
            idx = rng.choice(len(p), N_CANDIDATES, replace=False)
            set_p = p[idx]
            set_t = t[idx]
            set_s = s[idx]
            set_mean = float(np.mean(set_p))
            noise = rng.normal(0.0, sigma, N_CANDIDATES)
            heteroscedastic = rng.normal(0.0, 1.0, N_CANDIDATES) * set_s

            winners = {
                "argmin": int(np.argmin(set_p)),
                "score_perturbation": int(np.argmin(set_p + noise)),
                "risk_averse": int(np.argmin(set_p + RISK_LAMBDA * set_s)),
                "conformal_select": int(np.argmin(set_p + q_norm * set_s)),
                "thompson": int(np.argmin(set_p + heteroscedastic)),
            }
            baseline = winners["argmin"]
            reports = {
                "calibrated_report": (baseline, alpha + beta * float(set_p[baseline])),
                "conformal_report": (
                    baseline,
                    alpha + beta * float(set_p[baseline])
                    + q_selected * float(set_s[baseline]),
                ),
            }

            for policy, winner in winners.items():
                records[policy]["gap"].append(float(set_t[winner] - set_p[winner]))
                records[policy]["depth"].append(set_mean - float(set_p[winner]))
                records[policy]["advantage"].append(
                    float(1.0 - np.exp(set_t[winner] - median_truth))
                )
                records[policy]["reported_advantage"].append(
                    float(1.0 - np.exp(set_p[winner] - median_truth))
                )
            for policy, (winner, reported) in reports.items():
                records[policy]["gap"].append(float(set_t[winner]) - reported)
                records[policy]["depth"].append(set_mean - float(set_p[winner]))
                records[policy]["advantage"].append(
                    float(1.0 - np.exp(set_t[winner] - median_truth))
                )
                records[policy]["reported_advantage"].append(
                    float(1.0 - np.exp(reported - median_truth))
                )

    rows = {}
    for policy, values in records.items():
        rows[policy] = {
            **gap_distribution(np.asarray(values["gap"])),
            "selection_depth_log": float(np.mean(values["depth"])),
            "realized_quality_fraction": float(np.mean(values["advantage"])),
            "reported_saving_fraction": float(np.mean(values["reported_advantage"])),
            "positive_saving_fraction": float(
                np.mean(np.asarray(values["reported_advantage"]) > 0.0)
            ),
        }

    deepest = max(rows[policy]["selection_depth_log"] for policy in SELECTION_POLICIES)
    if not np.isclose(rows["argmin"]["selection_depth_log"], deepest,
                      rtol=0.0, atol=1e-12):
        raise RuntimeError("Argmin failed the pointwise selection-depth bound")
    for policy in REPORT_POLICIES:
        same_depth = np.isclose(rows[policy]["selection_depth_log"],
                                rows["argmin"]["selection_depth_log"],
                                rtol=0.0, atol=1e-12)
        same_quality = np.isclose(rows[policy]["realized_quality_fraction"],
                                  rows["argmin"]["realized_quality_fraction"],
                                  rtol=0.0, atol=1e-12)
        if not (same_depth and same_quality):
            raise RuntimeError(f"{policy} changed the selection it was meant to preserve")
    return rows


def main() -> None:
    started = datetime.now(timezone.utc).isoformat()
    corpus = load_corpus()

    evaluations = []
    failures = []
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

            q_norm = conformal_quantile(
                np.abs(cal_residual) / cal_spread, CONFORMAL_ALPHA
            )
            q_selected = selection_conformal_quantile(
                fold.cal_pred, cal_truth, cal_spread,
                corpus.classes[fold.calibration], cmap.alpha, cmap.beta,
                CONFORMAL_ALPHA,
                np.random.default_rng(310_000 * fold.split_seed + fold.fold),
            )

            policies = evaluate(
                fold.test_pred, corpus.y[fold.test], test_spread,
                corpus.classes[fold.test], float(cmap.alpha), float(cmap.beta),
                float(np.sqrt(cmap.sigma2)), q_norm, q_selected,
                np.random.default_rng(320_000 * fold.split_seed + fold.fold),
            )

            row = {
                "run_id": fold.run_id,
                "status": "success",
                "split_seed": fold.split_seed,
                "fold": fold.fold,
                "partition_sha256": fold.partition_sha256,
                **fold.extras,
                "calibration_alpha": float(cmap.alpha),
                "calibration_beta": float(cmap.beta),
                "calibration_sigma": float(np.sqrt(cmap.sigma2)),
                "spread_median_calibration": float(np.median(cal_spread)),
                "spread_ratio_p90_p10": float(
                    np.percentile(cal_spread, 90) / np.percentile(cal_spread, 10)
                ),
                "normalized_conformal_quantile": q_norm,
                "selection_conformal_quantile": q_selected,
                "policies": policies,
            }
            numeric = [value for value in row.values() if isinstance(value, float)] + [
                value for policy in policies.values() for value in policy.values()
                if isinstance(value, float)
            ]
            if not np.all(np.isfinite(numeric)):
                raise RuntimeError("Non-finite metric")
            evaluations.append(row)
            print(
                f"{fold.run_id}: argmin gap={policies['argmin']['mean']:+.4f} "
                f"P>5pp={policies['argmin']['p_over_005']:.3f} | "
                f"calibrated={policies['calibrated_report']['mean']:+.4f} "
                f"P>5pp={policies['calibrated_report']['p_over_005']:.3f} | "
                f"conformal={policies['conformal_report']['mean']:+.4f} "
                f"P>5pp={policies['conformal_report']['p_over_005']:.3f}"
            )
        except Exception as exc:  # preserve every failed attempt in the ledger
            failures.append({
                "run_id": fold.run_id,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"{fold.run_id}: FAILED: {exc}")

    complete = not failures and len(evaluations) == len(SPLIT_SEEDS) * N_FOLDS

    metrics = ("mean", "sd", "p_positive", "p_over_005", "p_over_010", "q90", "q99",
               "cvar90", "selection_depth_log", "realized_quality_fraction",
               "reported_saving_fraction", "positive_saving_fraction")
    policy_summary = {
        policy: {
            metric: summarize(row["policies"][policy][metric]
                                   for row in evaluations)
            for metric in metrics
        }
        for policy in POLICIES
    }
    paired = {}
    for policy in POLICIES:
        if policy == "argmin":
            continue
        paired[policy] = {
            "gap_reduction": summarize(
                row["policies"]["argmin"]["mean"] - row["policies"][policy]["mean"]
                for row in evaluations
            ),
            "p_over_005_reduction": summarize(
                row["policies"]["argmin"]["p_over_005"]
                - row["policies"][policy]["p_over_005"] for row in evaluations
            ),
            "cvar90_reduction": summarize(
                row["policies"]["argmin"]["cvar90"] - row["policies"][policy]["cvar90"]
                for row in evaluations
            ),
            "advantage_cost": summarize(
                row["policies"]["argmin"]["realized_quality_fraction"]
                - row["policies"][policy]["realized_quality_fraction"]
                for row in evaluations
            ),
            "depth_reduction": summarize(
                row["policies"]["argmin"]["selection_depth_log"]
                - row["policies"][policy]["selection_depth_log"]
                for row in evaluations
            ),
        }

    output = {
        "metadata": {
            "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": "E2 uncertainty-aware selection and reporting with tail risk",
            **ledger_provenance(
                __file__, "experiments/protocols/review_response_suite.md"
            ),
            "n_records": corpus.n_records,
            "n_manufacturers": corpus.n_manufacturers,
            "n_folds": N_FOLDS,
            "split_seeds": SPLIT_SEEDS,
            "n_policy_sets_per_class": N_POLICY_SETS,
            "n_conformal_sets_per_class": N_CONFORMAL_SETS,
            "n_candidates": N_CANDIDATES,
            "conformal_alpha": CONFORMAL_ALPHA,
            "risk_lambda": RISK_LAMBDA,
            "selection_policies": list(SELECTION_POLICIES),
            "report_policies": list(REPORT_POLICIES),
            "all_runs_reported": True,
            "complete": complete,
            "sensitivity_interval_note": (
                "Percentiles summarize split/fold sensitivity on one corpus; "
                "they are not independent-data confidence intervals."
            ),
        },
        "diagnostic_summary": {
            name: summarize(row[name] for row in evaluations)
            for name in ("calibration_beta", "spread_ratio_p90_p10",
                         "normalized_conformal_quantile",
                         "selection_conformal_quantile")
        },
        "policy_summary": policy_summary,
        "paired_vs_argmin": paired,
        "evaluations": evaluations,
        "failures": failures,
    }
    out_path = RESULTS / "exp_uncertainty_policies_v1.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"wrote {out_path}")
    if not complete:
        raise RuntimeError(f"Incomplete experiment: {failures}")


if __name__ == "__main__":
    main()
