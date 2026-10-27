"""Is the ranking-versus-valuation split a property of one estimator, or of the task?

Protocol: ``experiments/protocols/model_robustness.md`` and its v2-v3 addenda.
Ledger:   ``results/exp_model_robustness_v3.json``.

The headline experiment uses one gradient-boosted model. If ordering survived while
valuation collapsed only for that estimator, the finding would be about the estimator.
This runs four model families -- a linear baseline and three tree ensembles -- over the
identical partitions, and reports every fold of every family.

The evaluation here is deliberately simpler than the headline one: a single held-out
block, no calibration block, no policies. The question is only whether R-squared and
pairwise accuracy separate, and adding machinery would obscure that.

Supplies: the model-family paragraph of RQ1, the Ridge and ensemble numbers quoted in
the main text, and the right panel of Figure 1.
"""
from __future__ import annotations

import json
import pathlib
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
from scipy import __version__ as scipy_version
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from rnv.evaluation.corpus import N_FOLDS, SPLIT_SEEDS  # noqa: E402
from rnv.evaluation.metrics import (  # noqa: E402
    pairwise_accuracy,
    r2_score_test_mean as test_r2,
    summarize_compact as summarize,
    within_class_spearman,
)
from rnv.evaluation.partitions import stratified_random_group_blocks  # noqa: E402
from rnv.evaluation.provenance import ledger_provenance  # noqa: E402
from rnv.paths import RESULTS  # noqa: E402
from rnv.registry.features import build_features_raw  # noqa: E402
from rnv.registry.normalize import load_jsonl, normalize_concrete  # noqa: E402


def models() -> dict[str, object]:
    return {
        "ridge": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)
        ),
        "hist_gradient_boosting": make_pipeline(
            SimpleImputer(strategy="median"), HistGradientBoostingRegressor(
            learning_rate=0.08, max_iter=300, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=0,
        )),
        "extra_trees": make_pipeline(
            SimpleImputer(strategy="median"), ExtraTreesRegressor(
            n_estimators=200, min_samples_leaf=5, max_features=0.8,
            n_jobs=-1, random_state=0,
        )),
        "gradient_boosting": make_pipeline(
            SimpleImputer(strategy="median"), GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=0,
        )),
    }


def main() -> None:
    started = datetime.now(timezone.utc).isoformat()
    data, ledger = normalize_concrete(load_jsonl("readymix"), verbose=True)
    data = data[data.strength_class.notna()].copy()
    features = build_features_raw(data).reset_index(drop=True)
    truth = np.log(data.gwp_per_m3.to_numpy())
    manufacturers = data.get("mfr_name").fillna("unknown").to_numpy()
    classes = data.strength_class.to_numpy()
    partition_configs = []
    for split_seed in SPLIT_SEEDS:
        folds, loads, partition_digest = stratified_random_group_blocks(
            manufacturers, n_blocks=N_FOLDS, seed=split_seed
        )
        for fold in range(N_FOLDS):
            partition_configs.append(
                (split_seed, fold, folds, loads, partition_digest)
            )

    evaluations: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for model_name, model in models().items():
        for split_seed, fold, folds, loads, partition_digest in partition_configs:
            run_id = f"{model_name}_seed{split_seed}_fold{fold}"
            try:
                test = folds == fold
                train = ~test
                train_manufacturers = set(manufacturers[train])
                test_manufacturers = set(manufacturers[test])
                if train_manufacturers & test_manufacturers:
                    raise RuntimeError("Manufacturer overlap")
                model.fit(features[train], truth[train])
                pred = np.asarray(model.predict(features[test]), dtype=float)
                test_truth = truth[test]
                test_classes = classes[test]
                rng = np.random.default_rng(10_000 + 100 * split_seed + fold)
                pair_accuracy, prediction_tie_rate, n_pairs = pairwise_accuracy(
                    pred, test_truth, test_classes, rng
                )
                row = {
                    "run_id": run_id,
                    "status": "success",
                    "model": model_name,
                    "split_seed": split_seed,
                    "fold": fold,
                    "partition_sha256": partition_digest,
                    "block_loads": [int(value) for value in loads],
                    "n_train": int(train.sum()),
                    "n_test": int(test.sum()),
                    "n_train_manufacturers": len(train_manufacturers),
                    "n_test_manufacturers": len(test_manufacturers),
                    "r2": test_r2(test_truth, pred),
                    "within_class_spearman": within_class_spearman(
                        pred, test_truth, test_classes
                    ),
                    "pairwise_accuracy": pair_accuracy,
                    "prediction_tie_rate": prediction_tie_rate,
                    "n_pairwise_comparisons": n_pairs,
                }
                if not all(np.isfinite(row[key]) for key in
                           ("r2", "within_class_spearman", "pairwise_accuracy",
                            "prediction_tie_rate")):
                    raise RuntimeError("Non-finite metric")
                evaluations.append(row)
                print(
                    f"{run_id}: R2={row['r2']:+.3f} "
                    f"rho={row['within_class_spearman']:.3f} "
                    f"pair={row['pairwise_accuracy']:.3f} "
                    f"ties={row['prediction_tie_rate']:.3f}"
                )
            except Exception as exc:
                failures.append({
                    "run_id": run_id,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                print(f"{run_id}: FAILED: {exc}")

    expected = len(models()) * N_FOLDS * len(SPLIT_SEEDS)
    if failures or len(evaluations) != expected:
        raise RuntimeError(
            f"Incomplete experiment: {len(evaluations)}/{expected}; {failures}"
        )

    summary = {}
    for model_name in models():
        rows = [row for row in evaluations if row["model"] == model_name]
        summary[model_name] = {
            metric: summarize([float(row[metric]) for row in rows])
            for metric in ("r2", "within_class_spearman", "pairwise_accuracy",
                           "prediction_tie_rate")
        }
        summary[model_name]["n_pairwise_above_chance"] = int(sum(
            float(row["pairwise_accuracy"]) > 0.5 for row in rows
        ))
        summary[model_name]["n_r2_below_zero"] = int(sum(
            float(row["r2"]) < 0.0 for row in rows
        ))

    output = {
        "metadata": {
            "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            **ledger_provenance(
                __file__,
                "experiments/protocols/model_robustness.md",
                "experiments/protocols/model_robustness_addendum_v2.md",
                "experiments/protocols/model_robustness_addendum_v3.md",
                "experiments/protocols/selection_gap.md",
            ),
            "n_records": int(len(data)),
            "n_manufacturers": int(pd.Series(manufacturers).nunique()),
            "split_seeds": SPLIT_SEEDS,
            "partition_method": "seeded size-stratified manufacturer blocks",
            "n_unique_partition_digests": len({
                config[4] for config in partition_configs
            }),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scipy_version": scipy_version,
            "sklearn_version": sklearn.__version__,
            "normalization_drop_ledger": ledger,
            "all_runs_reported": True,
            "interval_note": (
                "Fold summaries are correlated sensitivity results on one corpus, "
                "not independent-data confidence intervals."
            ),
        },
        "summary": summary,
        "evaluations": evaluations,
        "failures": failures,
    }
    output_path = RESULTS / "exp_model_robustness_v3.json"
    output_path.write_text(json.dumps(output, indent=2))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
