"""E1: does the full-vector conditional calibration premise hold, and is the slope
group-dependent?

Protocol: ``experiments/protocols/review_response_suite.md`` (E1).
Ledger:   ``results/exp_calibration_structure_v1.json``.
Supplies: RQ2 of the manuscript and Appendix C, including the premise-audit figure.

Proposition 1 in the manuscript assumes E[T_i | X] = alpha + beta X_i for every candidate
i jointly. That is strictly stronger than the scalar calibration the paper measures, and
the accepted manuscript says so without testing it. This experiment tests it directly.

The test is deliberately conservative. Scalar calibration is granted for free: the
intercept and slope are refitted by ordinary least squares on the pooled candidate-set
members themselves, so the residual has mean zero and zero covariance with the candidate's
own score by construction. Whatever structure survives is structure the full-vector premise
forbids and scalar calibration cannot absorb -- namely dependence of a candidate's residual
on the OTHER candidates it is being compared against.

This script never overwrites earlier result artifacts.
"""
from __future__ import annotations

import json
import pathlib
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from rnv.claims.certification import CalibrationMap  # noqa: E402
from rnv.evaluation.bootstrap import cluster_bootstrap_se  # noqa: E402
from rnv.evaluation.corpus import (  # noqa: E402
    Corpus,
    N_FOLDS,
    SPLIT_SEEDS,
    iter_folds,
    load_corpus,
)
from rnv.evaluation.metrics import (  # noqa: E402
    eligible_classes,
    gap_distribution,
    summarize,
)
from rnv.evaluation.provenance import ledger_provenance  # noqa: E402
from rnv.paths import RESULTS  # noqa: E402

N_PREMISE_SETS = 1_000
N_REPORT_SETS = 1_000
N_CANDIDATES = 10
N_BOOTSTRAP = 200
MIN_GROUP_ROWS = 200
MIN_GROUP_MANUFACTURERS = 2
N_SPATIAL_CLUSTERS = 6
N_SCORE_BINS = 10
PREMISE_TERMS = ("own_score", "others_mean", "others_min", "rank_normalized")
COMPARISON_TERMS = ("others_mean", "others_min")


def group_calibration(pred: np.ndarray, truth: np.ndarray, manufacturers: np.ndarray,
                      labels: np.ndarray) -> dict[str, dict[str, float]]:
    """Scalar calibration fitted separately inside each group of the calibration block."""
    out = {}
    for label in sorted(set(map(str, labels))):
        mask = labels.astype(str) == label
        if mask.sum() < MIN_GROUP_ROWS:
            continue
        if pd.Series(manufacturers[mask]).nunique() < MIN_GROUP_MANUFACTURERS:
            continue
        cmap = CalibrationMap().fit(pred[mask], truth[mask], groups=manufacturers[mask])
        out[label] = {
            "alpha": float(cmap.alpha),
            "beta": float(cmap.beta),
            "n_rows": int(mask.sum()),
            "n_manufacturers": int(pd.Series(manufacturers[mask]).nunique()),
        }
    return out


def spread(maps: dict[str, dict[str, float]]) -> dict[str, float]:
    betas = np.asarray([row["beta"] for row in maps.values()], dtype=float)
    if len(betas) < 2:
        return {"n_groups": int(len(betas)), "beta_min": float("nan"),
                "beta_max": float("nan"), "beta_sd": float("nan"),
                "beta_range": float("nan")}
    return {
        "n_groups": int(len(betas)),
        "beta_min": float(betas.min()),
        "beta_max": float(betas.max()),
        "beta_sd": float(betas.std(ddof=1)),
        "beta_range": float(betas.max() - betas.min()),
    }


def spatial_clusters(corpus: Corpus, train: np.ndarray, target: np.ndarray,
                     seed: int) -> np.ndarray:
    """Cluster centres learned on training manufacturers only, applied to the target."""
    coords_train = corpus.X.loc[train, ["lat", "lon"]].to_numpy(dtype=float)
    fill = np.nanmedian(coords_train, axis=0)
    coords_train = np.where(np.isfinite(coords_train), coords_train, fill)
    model = KMeans(n_clusters=N_SPATIAL_CLUSTERS, n_init=10, random_state=seed)
    model.fit(coords_train)
    coords_target = corpus.X.loc[target, ["lat", "lon"]].to_numpy(dtype=float)
    coords_target = np.where(np.isfinite(coords_target), coords_target, fill)
    return model.predict(coords_target).astype(str)


def size_terciles(train_manufacturers: np.ndarray,
                  target_manufacturers: np.ndarray) -> np.ndarray:
    """Manufacturer-size terciles with cut points taken from the training block.

    Manufacturers are partitioned whole, so a manufacturer's record count inside its own
    block is its corpus-wide count; the count is a property of the supplier and carries no
    outcome information. Only the tercile cut points are borrowed from the training block,
    which keeps the grouping rule itself independent of the block being calibrated.
    """
    cuts = np.quantile(
        pd.Series(train_manufacturers).value_counts().to_numpy(dtype=float), [1 / 3, 2 / 3]
    )
    counts = pd.Series(target_manufacturers).value_counts()
    target_counts = pd.Series(target_manufacturers).map(counts).to_numpy(dtype=float)
    return np.where(target_counts <= cuts[0], "small",
                    np.where(target_counts <= cuts[1], "medium", "large"))


def build_premise_design(pred: np.ndarray, truth: np.ndarray, manufacturers: np.ndarray,
                         classes: np.ndarray, rng: np.random.Generator):
    """Pool within-class candidate sets and build the full-vector premise design matrix."""
    own, others_mean, others_min, rank_norm = [], [], [], []
    outcome, cluster, rank_index = [], [], []
    for label in eligible_classes(classes, 3 * N_CANDIDATES):
        mask = classes.astype(str) == label
        p = pred[mask]
        t = truth[mask]
        m = manufacturers[mask]
        for _ in range(N_PREMISE_SETS):
            idx = rng.choice(len(p), N_CANDIDATES, replace=False)
            set_p = p[idx]
            total = float(set_p.sum())
            order = np.argsort(set_p, kind="stable")
            position = np.empty(N_CANDIDATES, dtype=float)
            position[order] = np.arange(N_CANDIDATES, dtype=float)
            for slot in range(N_CANDIDATES):
                rest = np.delete(set_p, slot)
                own.append(float(set_p[slot]))
                others_mean.append((total - float(set_p[slot])) / (N_CANDIDATES - 1))
                others_min.append(float(rest.min()))
                rank_norm.append(position[slot] / (N_CANDIDATES - 1))
                rank_index.append(int(position[slot]))
                outcome.append(float(t[idx[slot]]))
                cluster.append(str(m[idx[slot]]))
    design = {
        "own_score": np.asarray(own),
        "others_mean": np.asarray(others_mean),
        "others_min": np.asarray(others_min),
        "rank_normalized": np.asarray(rank_norm),
    }
    return (design, np.asarray(outcome), np.asarray(cluster),
            np.asarray(rank_index, dtype=int))


def premise_test(design: dict[str, np.ndarray], outcome: np.ndarray,
                 clusters: np.ndarray, ranks: np.ndarray,
                 seed: int) -> dict[str, object]:
    """Test the full-vector premise, separating it from own-score nonlinearity.

    Two distinct ways the premise can fail must not be conflated. The conditional mean can
    be a nonlinear function of the candidate's own score, which a richer scalar map would
    absorb; or it can genuinely depend on which other candidates the product is being
    compared against, which no scalar map of any form can absorb. The first is measured by
    the residual profile across own-score deciles. The second is measured after removing
    those decile means, so that only comparison-set structure is left. Both profiles are
    identically zero under the premise as the manuscript states it.
    """
    own = design["own_score"]
    scalar = np.vstack([np.ones_like(own), own]).T
    coef, *_ = np.linalg.lstsq(scalar, outcome, rcond=None)
    residual = outcome - scalar @ coef

    columns = np.vstack([np.ones_like(own)] + [design[name] for name in PREMISE_TERMS]).T
    comparison = np.vstack([design[name] for name in COMPARISON_TERMS]).T
    quantiles = np.arange(1, N_SCORE_BINS) / N_SCORE_BINS

    def statistics(rows: np.ndarray) -> np.ndarray:
        """Pooled coefficients, decile profile, and comparison-set structure.

        Bin edges are recomputed inside every bootstrap resample. Reusing the full
        sample's edges would let a resample that drops a large supplier empty a decile,
        which makes the statistic undefined for reasons that have nothing to do with the
        hypothesis being tested.
        """
        e = residual[rows]
        pooled, *_ = np.linalg.lstsq(columns[rows], e, rcond=None)

        own_rows = own[rows]
        row_bins = np.digitize(own_rows, np.quantile(own_rows, quantiles))
        adjusted = e.copy()
        row_comparison = comparison[rows].copy()
        score_profile = np.zeros(N_SCORE_BINS)
        for index in range(N_SCORE_BINS):
            mask = row_bins == index
            if not mask.any():
                continue
            score_profile[index] = e[mask].mean()
            adjusted[mask] -= score_profile[index]
            row_comparison[mask] -= row_comparison[mask].mean(axis=0)
        within, *_ = np.linalg.lstsq(row_comparison, adjusted, rcond=None)

        row_ranks = ranks[rows]
        raw_profile = np.asarray([e[row_ranks == p].mean() for p in range(N_CANDIDATES)])
        adjusted_profile = np.asarray([
            adjusted[row_ranks == p].mean() for p in range(N_CANDIDATES)
        ])
        return np.concatenate([pooled, within, score_profile, raw_profile,
                               adjusted_profile])

    point = statistics(np.arange(len(residual)))
    unique, inverse = np.unique(clusters, return_inverse=True)
    index_by_cluster = [np.flatnonzero(inverse == k) for k in range(len(unique))]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(N_BOOTSTRAP):
        multiplicity = rng.multinomial(len(unique), np.full(len(unique), 1.0 / len(unique)))
        rows = np.concatenate([
            np.tile(index_by_cluster[k], count)
            for k, count in enumerate(multiplicity)
            if count and len(index_by_cluster[k])
        ])
        if len(rows) < 100:
            continue
        candidate = statistics(rows)
        if np.all(np.isfinite(candidate)):
            draws.append(candidate)
    draws = np.asarray(draws)
    if len(draws) < 20:
        raise RuntimeError("Cluster bootstrap produced too few usable resamples")
    error = draws.std(axis=0, ddof=1)

    def entry(index: int) -> dict[str, float]:
        estimate = float(point[index])
        standard_error = float(error[index])
        return {
            "estimate": estimate,
            "cluster_bootstrap_se": standard_error,
            "z": float(estimate / standard_error) if standard_error > 0 else float("nan"),
        }

    offset = len(PREMISE_TERMS) + 1
    comparison_offset = offset + len(COMPARISON_TERMS)
    raw_offset = comparison_offset + N_SCORE_BINS
    adjusted_offset = raw_offset + N_CANDIDATES

    raw_profile = [float(value) for value in point[raw_offset:raw_offset + N_CANDIDATES]]
    adjusted_profile = [
        float(value) for value in point[adjusted_offset:adjusted_offset + N_CANDIDATES]
    ]
    score_profile = [
        float(value) for value in point[comparison_offset:comparison_offset + N_SCORE_BINS]
    ]
    return {
        "population_scalar_alpha": float(coef[0]),
        "population_scalar_beta": float(coef[1]),
        "coefficients": {
            name: entry(position + 1) for position, name in enumerate(PREMISE_TERMS)
        },
        "comparison_set_coefficients": {
            name: entry(offset + position)
            for position, name in enumerate(COMPARISON_TERMS)
        },
        "residual_mean_by_score_decile": score_profile,
        "residual_se_by_score_decile": [
            float(value)
            for value in error[comparison_offset:comparison_offset + N_SCORE_BINS]
        ],
        "score_decile_profile_range": float(max(score_profile) - min(score_profile)),
        "residual_mean_by_rank": raw_profile,
        "residual_se_by_rank": [
            float(value) for value in error[raw_offset:raw_offset + N_CANDIDATES]
        ],
        "rank_profile_range": float(max(raw_profile) - min(raw_profile)),
        "adjusted_residual_mean_by_rank": adjusted_profile,
        "adjusted_residual_se_by_rank": [
            float(value)
            for value in error[adjusted_offset:adjusted_offset + N_CANDIDATES]
        ],
        "adjusted_rank_profile_range": float(
            max(adjusted_profile) - min(adjusted_profile)
        ),
        "selected_rank_residual": {
            "estimate": raw_profile[0],
            "cluster_bootstrap_se": float(error[raw_offset]),
            "z": (float(raw_profile[0] / error[raw_offset])
                  if error[raw_offset] > 0 else float("nan")),
        },
        "selected_rank_residual_adjusted": {
            "estimate": adjusted_profile[0],
            "cluster_bootstrap_se": float(error[adjusted_offset]),
            "z": (float(adjusted_profile[0] / error[adjusted_offset])
                  if error[adjusted_offset] > 0 else float("nan")),
        },
        "n_items": int(len(residual)),
        "n_bootstrap_draws": int(len(draws)),
    }


def report_rule_comparison(pred: np.ndarray, truth: np.ndarray, classes: np.ndarray,
                           global_map: CalibrationMap,
                           class_maps: dict[str, dict[str, float]],
                           rng: np.random.Generator) -> dict[str, dict[str, float]]:
    """Fix the argmin selection and vary only the rule used to report its value."""
    gaps = {"raw": [], "global_calibrated": [], "class_calibrated": []}
    class_covered = 0
    class_total = 0
    for label in eligible_classes(classes, 3 * N_CANDIDATES):
        mask = classes.astype(str) == label
        p = pred[mask]
        t = truth[mask]
        local = class_maps.get(label)
        class_total += 1
        class_covered += int(local is not None)
        alpha = global_map.alpha if local is None else local["alpha"]
        beta = global_map.beta if local is None else local["beta"]
        for _ in range(N_REPORT_SETS):
            idx = rng.choice(len(p), N_CANDIDATES, replace=False)
            winner = int(np.argmin(p[idx]))
            score = float(p[idx][winner])
            declared = float(t[idx][winner])
            gaps["raw"].append(declared - score)
            gaps["global_calibrated"].append(
                declared - (global_map.alpha + global_map.beta * score)
            )
            gaps["class_calibrated"].append(declared - (alpha + beta * score))
    out = {rule: gap_distribution(np.asarray(values))
           for rule, values in gaps.items()}
    out["class_map_coverage"] = {
        "classes_with_own_map": int(class_covered),
        "classes_evaluated": int(class_total),
    }
    return out


def main() -> None:
    started = datetime.now(timezone.utc).isoformat()
    corpus = load_corpus()

    evaluations = []
    failures = []
    for fold in iter_folds(corpus):
        try:
            cal_truth = corpus.y[fold.calibration]
            cal_mfr = corpus.manufacturers[fold.calibration]
            cal_classes = corpus.classes[fold.calibration]

            groupings = {
                "strength_class": group_calibration(
                    fold.cal_pred, cal_truth, cal_mfr, cal_classes
                ),
                "spatial_cluster": group_calibration(
                    fold.cal_pred, cal_truth, cal_mfr,
                    spatial_clusters(corpus, fold.train, fold.calibration,
                                     fold.split_seed),
                ),
                "manufacturer_size": group_calibration(
                    fold.cal_pred, cal_truth, cal_mfr,
                    size_terciles(corpus.manufacturers[fold.train], cal_mfr),
                ),
            }

            design, outcome, clusters, ranks = build_premise_design(
                fold.cal_pred, cal_truth, cal_mfr, cal_classes,
                np.random.default_rng(700_000 * fold.split_seed + fold.fold),
            )
            premise = premise_test(
                design, outcome, clusters, ranks,
                seed=800_000 * fold.split_seed + fold.fold,
            )

            reports = report_rule_comparison(
                fold.test_pred, corpus.y[fold.test], corpus.classes[fold.test],
                fold.calibration_map, groupings["strength_class"],
                np.random.default_rng(900_000 * fold.split_seed + fold.fold),
            )

            row = {
                "run_id": fold.run_id,
                "status": "success",
                "split_seed": fold.split_seed,
                "fold": fold.fold,
                "partition_sha256": fold.partition_sha256,
                **fold.extras,
                "global_calibration": {
                    "alpha": float(fold.calibration_map.alpha),
                    "beta": float(fold.calibration_map.beta),
                },
                "group_calibration": {
                    name: {"maps": maps, "spread": spread(maps)}
                    for name, maps in groupings.items()
                },
                "premise_test": premise,
                "report_rules": reports,
            }
            evaluations.append(row)
            print(
                f"{fold.run_id}: beta={row['global_calibration']['beta']:.3f} "
                f"class-beta spread="
                f"{row['group_calibration']['strength_class']['spread']['beta_range']:.3f} "
                f"comparison z="
                f"{premise['comparison_set_coefficients']['others_min']['z']:+.1f} "
                f"raw gap={reports['raw']['mean']:+.4f} "
                f"class-cal gap={reports['class_calibrated']['mean']:+.4f}"
            )
        except Exception as exc:  # preserve every failed attempt in the ledger
            failures.append({
                "run_id": fold.run_id,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"{fold.run_id}: FAILED: {exc}")

    complete = not failures and len(evaluations) == len(SPLIT_SEEDS) * N_FOLDS

    summary = {
        "global_beta": summarize(
            row["global_calibration"]["beta"] for row in evaluations
        ),
    }
    for name in ("strength_class", "spatial_cluster", "manufacturer_size"):
        summary[f"{name}_beta_range"] = summarize(
            row["group_calibration"][name]["spread"]["beta_range"] for row in evaluations
        )
        summary[f"{name}_beta_sd"] = summarize(
            row["group_calibration"][name]["spread"]["beta_sd"] for row in evaluations
        )
    for term in PREMISE_TERMS:
        summary[f"premise_{term}_estimate"] = summarize(
            row["premise_test"]["coefficients"][term]["estimate"] for row in evaluations
        )
        summary[f"premise_{term}_z"] = summarize(
            row["premise_test"]["coefficients"][term]["z"] for row in evaluations
        )
    for term in COMPARISON_TERMS:
        summary[f"comparison_{term}_estimate"] = summarize(
            row["premise_test"]["comparison_set_coefficients"][term]["estimate"]
            for row in evaluations
        )
        summary[f"comparison_{term}_z"] = summarize(
            row["premise_test"]["comparison_set_coefficients"][term]["z"]
            for row in evaluations
        )
        summary[f"comparison_{term}_significant"] = float(np.mean([
            abs(row["premise_test"]["comparison_set_coefficients"][term]["z"]) > 2.0
            for row in evaluations
        ]))
    for name in ("rank_profile_range", "adjusted_rank_profile_range",
                 "score_decile_profile_range"):
        summary[f"premise_{name}"] = summarize(
            row["premise_test"][name] for row in evaluations
        )
    for name in ("selected_rank_residual", "selected_rank_residual_adjusted"):
        summary[f"premise_{name}"] = summarize(
            row["premise_test"][name]["estimate"] for row in evaluations
        )
        summary[f"premise_{name}_significant"] = float(np.mean([
            abs(row["premise_test"][name]["z"]) > 2.0 for row in evaluations
        ]))
    for rule in ("raw", "global_calibrated", "class_calibrated"):
        for metric in ("mean", "p_over_005", "p_positive", "q90", "cvar90"):
            summary[f"report_{rule}_{metric}"] = summarize(
                row["report_rules"][rule][metric] for row in evaluations
            )

    profiles = {
        name: np.asarray([row["premise_test"][name] for row in evaluations])
        for name in ("residual_mean_by_rank", "adjusted_residual_mean_by_rank",
                     "residual_mean_by_score_decile")
    }
    output = {
        "metadata": {
            "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": "E1 calibration structure and full-vector premise",
            **ledger_provenance(
                __file__, "experiments/protocols/review_response_suite.md"
            ),
            "n_records": corpus.n_records,
            "n_manufacturers": corpus.n_manufacturers,
            "n_folds": N_FOLDS,
            "split_seeds": SPLIT_SEEDS,
            "n_premise_sets_per_class": N_PREMISE_SETS,
            "n_report_sets_per_class": N_REPORT_SETS,
            "n_candidates": N_CANDIDATES,
            "n_score_bins": N_SCORE_BINS,
            "n_cluster_bootstrap": N_BOOTSTRAP,
            "all_runs_reported": True,
            "complete": complete,
            "sensitivity_interval_note": (
                "Percentiles summarize split/fold sensitivity on one corpus; "
                "they are not independent-data confidence intervals."
            ),
        },
        "summary": summary,
        "profiles": {
            name: {
                "mean": [float(value) for value in values.mean(axis=0)],
                "sd": [float(value) for value in values.std(axis=0, ddof=1)],
            }
            for name, values in profiles.items()
        },
        "evaluations": evaluations,
        "failures": failures,
    }
    out_path = RESULTS / "exp_calibration_structure_v1.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"wrote {out_path}")
    if not complete:
        raise RuntimeError(f"Incomplete experiment: {failures}")


if __name__ == "__main__":
    main()
