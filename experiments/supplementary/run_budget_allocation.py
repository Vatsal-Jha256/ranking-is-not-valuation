"""Budgeted allocation: which line items are worth re-specifying at all.

Protocol: ``experiments/protocols/supplementary_reruns.md``.
Ledger:   ``results/exp_knapsack_v2.json``.
Supplies: the supplementary checks reported in Appendix H. Not part of the RQ1--RQ3
evidence chain.

The decision problem is the one a project team actually faces: a budget of k line items
can be re-specified, and something has to rank them. Four scoring rules are compared
against an unattainable realized-saving oracle.

Three conditions separate this from the design in ``experiments/archive/`` that the
experiment inclusion audit excluded, and all three are about evaluating the pipeline the
paper proposes rather than a convenient approximation of it: the manufacturer-disjoint
six-seed stratified partition, fold-local imputation, and a certified objective that is
the same calibration map RQ2 fits, evaluated at the chosen candidate rather than as a
separate class-level constant.

The CLF WBLCA portfolio supplies real building line items -- mass, baseline declared GWP,
project structure -- and nothing else. It is never used for training and never as GWP
ground truth, which is the scope the experiment inclusion audit holds it to.

Within each line item, the candidate winner is always chosen by argmin over
the (uncalibrated) predicted score -- the same within-item selection rule the
headline experiment's `policies` evaluate. The budget decision -- which k of
a project's line items are worth re-specifying at all -- is what differs
across the four scoring rules compared here (point / certified / mass /
oracle).
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
from rnv.evaluation.model import fit_model  # noqa: E402
from rnv.evaluation.partitions import stratified_random_group_blocks  # noqa: E402
from rnv.evaluation.provenance import ledger_provenance  # noqa: E402
from rnv.paths import RESULTS  # noqa: E402
from rnv.registry.clf_portfolio import clf_concrete  # noqa: E402
from rnv.registry.features import build_features_raw  # noqa: E402
from rnv.registry.normalize import load_jsonl, normalize_concrete  # noqa: E402

N_FOLDS = 4
SPLIT_SEEDS = list(range(6))
N_CAND = 10
K_GRID = [1, 2, 3, 5, 10, 20]
POLICIES = ("point", "certified", "mass", "oracle")


def main() -> None:
    started = datetime.now(timezone.utc).isoformat()
    data, _ = normalize_concrete(load_jsonl("readymix"), verbose=False)
    data = data[data.strength_class.notna()].copy()
    X = build_features_raw(data).reset_index(drop=True)
    # Per-kg target: CLF line items carry mass (kg) and per-kg GWP, so the
    # model must predict per-kg intensity to combine with that portfolio.
    y = np.log(data.gwp_per_kg_norm.to_numpy())
    manufacturers = data.get("mfr_name").fillna("unknown").to_numpy()
    classes = data.strength_class.to_numpy()

    c_all = clf_concrete()
    print(f"{len(data):,} EC3 records | CLF {len(c_all):,} line items / "
          f"{c_all.project_index.nunique()} buildings\n")

    evaluations = []
    failures = []
    partition_digests = set()
    for split_seed in SPLIT_SEEDS:
        fold, _block_loads, partition_digest = stratified_random_group_blocks(
            manufacturers, n_blocks=N_FOLDS, seed=split_seed
        )
        partition_digests.add(partition_digest)
        for fold_index in range(N_FOLDS):
            run_id = f"seed{split_seed}_fold{fold_index}"
            try:
                test = fold == fold_index
                calibration = fold == (fold_index + 1) % N_FOLDS
                train = ~(test | calibration)
                train_mfr = set(manufacturers[train])
                cal_mfr = set(manufacturers[calibration])
                test_mfr = set(manufacturers[test])
                if train_mfr & cal_mfr or train_mfr & test_mfr or cal_mfr & test_mfr:
                    raise RuntimeError("Manufacturer overlap between split roles")

                model = fit_model(X[train], y[train], split_seed)
                cmap = CalibrationMap().fit(
                    model.predict(X[calibration]), y[calibration],
                    groups=manufacturers[calibration],
                )
                rng = np.random.default_rng(300_000 * split_seed + fold_index)

                pool = data[test].copy()
                pool["pred"] = model.predict(X[test])
                pool["true"] = y[test]
                by_cls = {
                    lab: g for lab, g in pool.groupby("strength_class", observed=True)
                    if len(g) >= N_CAND * 3
                }
                c = c_all[c_all.strength_class.isin(by_cls.keys())]

                acc: dict[tuple[str, int], dict] = {}
                n_projects = 0
                for _, g in c.groupby("project_index"):
                    g = g[g.inv_mass > 0]
                    if len(g) < 5:
                        continue
                    items = []
                    for lab, m, b in zip(g.strength_class.values, g.inv_mass.values,
                                         g.gwp_per_kg.values):
                        cand = by_cls[lab]
                        idx = rng.choice(len(cand), size=N_CAND, replace=False)
                        p = cand.pred.values[idx]
                        t = cand.true.values[idx]
                        winner = int(np.argmin(p))
                        m, b = float(m), float(b)
                        claimed = float(np.exp(p[winner]))
                        certified_log = float(cmap.transform(p[winner]))
                        realized = float(np.exp(t[winner]))
                        items.append(dict(
                            mass=m,
                            save_point=(b - claimed) * m,
                            save_cert=(b - float(np.exp(certified_log))) * m,
                            save_real=(b - realized) * m,
                            claim=claimed * m,
                            real=realized * m,
                        ))
                    it = pd.DataFrame(items)
                    n_projects += 1
                    for k in K_GRID:
                        if k > len(it):
                            continue
                        for pol, key in [("point", "save_point"), ("certified", "save_cert"),
                                         ("mass", "mass"), ("oracle", "save_real")]:
                            S = it.nlargest(k, key)
                            realized_sum = float(S.save_real.sum())
                            claim_sum = float(S.claim.sum())
                            real_sum = float(S.real.sum())
                            slot = acc.setdefault((pol, k), {"real": [], "num": 0.0, "den": 0.0})
                            slot["real"].append(realized_sum)
                            slot["num"] += real_sum - claim_sum
                            slot["den"] += claim_sum

                if not acc:
                    raise RuntimeError("No eligible CLF projects for this fold's candidate pools")

                rows = {}
                for (pol, k), slot in acc.items():
                    rows.setdefault(str(k), {})[pol] = {
                        "mean_realized_saving_kgco2e": float(np.mean(slot["real"])),
                        "pooled_over_claim_pct": (
                            100.0 * slot["num"] / slot["den"] if slot["den"] else float("nan")
                        ),
                        "n_projects": len(slot["real"]),
                    }

                row = {
                    "run_id": run_id,
                    "status": "success",
                    "split_seed": split_seed,
                    "fold": fold_index,
                    "partition_sha256": partition_digest,
                    "n_train": int(train.sum()),
                    "n_calibration": int(calibration.sum()),
                    "n_test": int(test.sum()),
                    "n_train_manufacturers": len(train_mfr),
                    "n_calibration_manufacturers": len(cal_mfr),
                    "n_test_manufacturers": len(test_mfr),
                    "n_eligible_classes": len(by_cls),
                    "n_eligible_projects": n_projects,
                    "calibration_alpha": float(cmap.alpha),
                    "calibration_beta": float(cmap.beta),
                    "by_budget": rows,
                }
                evaluations.append(row)
                p1 = rows.get("1", {})
                point1 = p1.get("point", {}).get("mean_realized_saving_kgco2e")
                cert1 = p1.get("certified", {}).get("mean_realized_saving_kgco2e")
                print(f"{run_id}: beta={cmap.beta:+.3f} n_projects={n_projects} "
                      f"k=1 realized point={point1} certified={cert1}")
            except Exception as exc:
                failures.append({"run_id": run_id, "status": "failed",
                                 "error": f"{type(exc).__name__}: {exc}"})
                print(f"{run_id}: FAILED: {exc}")

    complete = not failures and len(evaluations) == len(SPLIT_SEEDS) * N_FOLDS

    budget_summary = {}
    for k in K_GRID:
        budget_summary[str(k)] = {}
        for pol in POLICIES:
            realized = [
                row["by_budget"].get(str(k), {}).get(pol, {}).get("mean_realized_saving_kgco2e")
                for row in evaluations
            ]
            over_claim = [
                row["by_budget"].get(str(k), {}).get(pol, {}).get("pooled_over_claim_pct")
                for row in evaluations
            ]
            realized = [v for v in realized if v is not None]
            over_claim = [v for v in over_claim if v is not None and np.isfinite(v)]
            if not realized:
                continue
            budget_summary[str(k)][pol] = {
                "mean_realized_saving_kgco2e": {
                    "mean": float(np.mean(realized)), "sd": float(np.std(realized, ddof=1)),
                    "n": len(realized),
                },
                "pooled_over_claim_pct": (
                    {"mean": float(np.mean(over_claim)), "sd": float(np.std(over_claim, ddof=1)),
                     "n": len(over_claim)} if over_claim else None
                ),
            }

    output = {
        "metadata": {
            "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            **ledger_provenance(
                __file__, "experiments/protocols/supplementary_reruns.md"
            ),
            "n_records": int(len(data)),
            "n_manufacturers": int(pd.Series(manufacturers).nunique()),
            "n_folds": N_FOLDS,
            "split_seeds": SPLIT_SEEDS,
            "n_unique_partition_digests": len(partition_digests),
            "clf_role": "portfolio structure only (mass, baseline GWP, project grouping); never used for training or GWP ground truth",
            "complete": complete,
        },
        "budget_summary": budget_summary,
        "evaluations": evaluations,
        "failures": failures,
    }
    out_path = RESULTS / "exp_knapsack_v2.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nwrote {out_path}")
    if not complete:
        raise RuntimeError(f"Incomplete experiment: {failures}")

    print("\n=== budget summary (mean realized saving, kgCO2e/building; pooled over-claim %) ===")
    for k in K_GRID:
        row = budget_summary.get(str(k), {})
        if not row:
            continue
        parts = []
        for pol in POLICIES:
            s = row.get(pol)
            if not s:
                continue
            oc = s["pooled_over_claim_pct"]
            oc_txt = f"{oc['mean']:.2f}%" if oc else "n/a"
            parts.append(f"{pol}={s['mean_realized_saving_kgco2e']['mean']:,.0f} (oc {oc_txt})")
        print(f"k={k:>3}: " + " | ".join(parts))


if __name__ == "__main__":
    main()
