"""Empirical-Bayes shrinkage of the calibration slope.

Protocol: ``experiments/protocols/supplementary_reruns.md``.
Ledger:   ``results/exp_beta_shrinkage_v2.json``.
Supplies: the supplementary checks reported in Appendix H. These are not part of the
RQ1--RQ3 evidence chain; they test whether shrinking the slope buys anything the
identity does not already give.

Two design rules make this measurable rather than decorative, and both are recorded in
``reproducibility/EXPERIMENT_INCLUSION_AUDIT.md`` as conditions an earlier design in
``experiments/archive/`` failed.

  1. The manufacturer bootstrap draws multinomial counts, so a manufacturer drawn twice
     contributes twice. Collapsing a resample to set membership throws away exactly the
     multiplicity a cluster bootstrap needs and understates the sampling variance of
     the slope estimate.
  2. The shrinkage prior is built from this run under a strict leave-one-seed-out rule:
     the prior used to shrink seed s never includes seed s's own draws. There is no
     independent external corpus for this problem -- the ready-mix registry is the only
     data source -- so an out-of-corpus prior would have to be invented, and inventing
     one is not available here.

Design. Each of six partition seeds splits manufacturers into five
size-stratified blocks (stratified_random_group_blocks, n_blocks=5). Per
seed, block i rotates through four disjoint roles:

  calibration  -> fit beta_hat, alpha_hat (CalibrationMap, manufacturer-
                  equal-weighted, same estimator used throughout the paper)
                  and its manufacturer-cluster-bootstrap SE tau_f
  kappa_tune   -> tune the additive correction's kappa (Proposition 1) by
                  minimizing |mean over-claim| on this fold only
  eval         -> held-out measurement of P(over-claim > 5pp): raw beta_hat
                  vs the leave-one-seed-out shrunk beta
  train (x2)   -> fit the GWP model

This never touches the eval fold's manufacturers when fitting beta_hat,
tuning kappa, or building the shrinkage prior for that seed.
"""
from __future__ import annotations

import json
import pathlib
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from rnv.claims.certification import CalibrationMap, adaptive_correction  # noqa: E402
from rnv.evaluation.model import fit_model  # noqa: E402
from rnv.evaluation.partitions import stratified_random_group_blocks  # noqa: E402
from rnv.evaluation.provenance import ledger_provenance  # noqa: E402
from rnv.paths import RESULTS  # noqa: E402
from rnv.registry.features import build_features_raw  # noqa: E402
from rnv.registry.normalize import load_jsonl, normalize_concrete  # noqa: E402

N_BLOCKS = 5
SPLIT_SEEDS = list(range(6))
N_GRID = [2, 3, 5, 10, 50]
TRIALS = 800
KAPPA_TRIALS = 250
KAPPA_GRID = np.arange(0.0, 4.01, 0.4)
N_BOOT = 150
EC3_FACTOR = 1.095


def weighted_beta(p: np.ndarray, t: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    A = np.vstack([np.ones_like(p), p]).T
    sw = np.sqrt(weights)
    coef, *_ = np.linalg.lstsq(A * sw[:, None], t * sw, rcond=None)
    return float(coef[0]), float(coef[1])


def group_equal_weights(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    uniq, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    base_w = 1.0 / counts[inv]
    return uniq, inv, base_w


def cluster_bootstrap_se(p: np.ndarray, t: np.ndarray, groups: np.ndarray,
                         rng: np.random.Generator, n_boot: int = N_BOOT) -> float:
    """Manufacturer-cluster bootstrap SE of beta_hat, preserving resample multiplicity.

    A size-|G| resample-with-replacement of the |G| manufacturers is drawn via
    a multinomial (equivalent to `rng.choice(groups, len(groups), replace=True)`
    but returns per-manufacturer draw COUNTS directly). A manufacturer drawn k
    times gets k times its usual equal-weight contribution to the weighted
    regression. A membership mask would discard k in favour of a 0/1 indicator
    and could not distinguish a manufacturer drawn once from one drawn five
    times.
    """
    uniq, inv, base_w = group_equal_weights(groups)
    n_groups = len(uniq)
    betas = []
    for _ in range(n_boot):
        draw_counts = rng.multinomial(n_groups, np.full(n_groups, 1.0 / n_groups))
        mult = draw_counts[inv]
        w = base_w * mult
        if not np.any(w > 0):
            continue
        w = w / w[w > 0].mean()
        try:
            _, b = weighted_beta(p, t, w)
        except Exception:
            continue
        if np.isfinite(b):
            betas.append(b)
    return float(np.std(betas)) if len(betas) > 5 else float("nan")


def selection_draws(pred, truth, cls, n_cand, rng, trials, transform=None, add=0.0):
    C, T = [], []
    for lab in pd.unique(cls):
        m = cls == lab
        if m.sum() < max(30, n_cand * 3):
            continue
        p, t = pred[m], truth[m]
        for _ in range(trials):
            idx = rng.choice(m.sum(), size=n_cand, replace=False)
            w = int(np.argmin(p[idx]))
            v = p[idx][w]
            if transform is not None:
                v = transform(v)
            C.append(v + add)
            T.append(t[idx][w])
    return np.array(C), np.array(T)


def over_claim_array(C, T):
    return 100 * (np.exp(T) - np.exp(C)) / np.exp(C)


def main() -> None:
    started = datetime.now(timezone.utc).isoformat()
    data, _ = normalize_concrete(load_jsonl("readymix"), verbose=False)
    data = data[data.strength_class.notna()].copy()
    X = build_features_raw(data).reset_index(drop=True)
    y = np.log(data.gwp_per_m3.to_numpy())
    manufacturers = data.get("mfr_name").fillna("unknown").to_numpy()
    classes = data.strength_class.to_numpy()

    # ---- pass 1: fit models, calibration beta_hat/tau_f, and cache eval-fold arrays ----
    units = []
    failures = []
    partition_digests = set()
    for split_seed in SPLIT_SEEDS:
        fold, _block_loads, partition_digest = stratified_random_group_blocks(
            manufacturers, n_blocks=N_BLOCKS, seed=split_seed
        )
        partition_digests.add(partition_digest)
        for i in range(N_BLOCKS):
            run_id = f"seed{split_seed}_block{i}"
            try:
                calibration = fold == i
                kappa_tune = fold == (i + 1) % N_BLOCKS
                ev = fold == (i + 2) % N_BLOCKS
                train = ~(calibration | kappa_tune | ev)

                train_mfr = set(manufacturers[train])
                cal_mfr = set(manufacturers[calibration])
                kap_mfr = set(manufacturers[kappa_tune])
                eval_mfr = set(manufacturers[ev])
                if (train_mfr & cal_mfr or train_mfr & kap_mfr or train_mfr & eval_mfr
                        or cal_mfr & kap_mfr or cal_mfr & eval_mfr or kap_mfr & eval_mfr):
                    raise RuntimeError("Manufacturer overlap between split roles")

                model = fit_model(X[train], y[train], split_seed)
                p_cal = model.predict(X[calibration])
                cmap = CalibrationMap().fit(p_cal, y[calibration], groups=manufacturers[calibration])
                rng = np.random.default_rng(700_000 * split_seed + i)
                tau_f = cluster_bootstrap_se(p_cal, y[calibration], manufacturers[calibration], rng)

                units.append({
                    "run_id": run_id,
                    "split_seed": split_seed,
                    "block": i,
                    "partition_sha256": partition_digest,
                    "n_train": int(train.sum()),
                    "n_calibration": int(calibration.sum()),
                    "n_kappa_tune": int(kappa_tune.sum()),
                    "n_eval": int(ev.sum()),
                    "n_calibration_manufacturers": len(cal_mfr),
                    "alpha_hat": float(cmap.alpha),
                    "beta_hat": float(cmap.beta),
                    "sigma2": float(cmap.sigma2),
                    "tau2": float(cmap.tau2),
                    "tau_f": tau_f,
                    "model": model,
                    "cmap": cmap,
                    "kappa_tune_mask": kappa_tune,
                    "eval_mask": ev,
                })
                print(f"{run_id}: beta_hat={cmap.beta:+.3f} tau_f={tau_f:.3f} "
                      f"n_cal_mfr={len(cal_mfr)}")
            except Exception as exc:
                failures.append({"run_id": run_id, "status": "failed",
                                 "error": f"{type(exc).__name__}: {exc}"})
                print(f"{run_id}: FAILED: {exc}")

    complete = not failures and len(units) == len(SPLIT_SEEDS) * N_BLOCKS

    # ---- pass 2: leave-one-seed-out prior, shrinkage, downstream evaluation ----
    beta_by_seed: dict[int, list[float]] = {s: [] for s in SPLIT_SEEDS}
    for u in units:
        beta_by_seed[u["split_seed"]].append(u["beta_hat"])

    evaluations = []
    for u in units:
        seed = u["split_seed"]
        other_betas = np.concatenate([
            np.asarray(v) for s, v in beta_by_seed.items() if s != seed
        ])
        beta_0 = float(np.mean(other_betas))
        sigma_pool = float(np.std(other_betas, ddof=1))

        tau_f = u["tau_f"]
        if not np.isfinite(tau_f) or tau_f == 0 or sigma_pool == 0:
            w_f = 1.0
        else:
            w_f = sigma_pool ** 2 / (sigma_pool ** 2 + tau_f ** 2)
        beta_shrunk = w_f * u["beta_hat"] + (1 - w_f) * beta_0

        model, cmap = u["model"], u["cmap"]
        rng = np.random.default_rng(900_000 * seed + u["block"])
        p_kap = model.predict(X[u["kappa_tune_mask"]])
        y_kap = y[u["kappa_tune_mask"]]
        c_kap = classes[u["kappa_tune_mask"]]
        p_ev = model.predict(X[u["eval_mask"]])
        y_ev = y[u["eval_mask"]]
        c_ev = classes[u["eval_mask"]]

        row = dict(u)
        for key in ("model", "cmap", "kappa_tune_mask", "eval_mask"):
            row.pop(key)
        row.update({
            "beta_0_prior_mean": beta_0,
            "sigma_pool_prior_sd": sigma_pool,
            "shrinkage_weight_w_f": float(w_f),
            "beta_shrunk": float(beta_shrunk),
            "grids": {},
        })

        for beta_variant, variant_name in [(u["beta_hat"], "raw"), (beta_shrunk, "shrunk")]:
            variant_cmap = CalibrationMap()
            variant_cmap.alpha, variant_cmap.beta = u["alpha_hat"], beta_variant
            adj_kap = variant_cmap.transform(p_kap)
            variant_cmap.sigma2 = float(np.var(y_kap - adj_kap))
            variant_cmap.tau2 = max(float(np.var(adj_kap)), 1e-8)

            best_kappa, best_err = 0.0, float("inf")
            for k in KAPPA_GRID:
                add = adaptive_correction(variant_cmap.sigma2, 10, variant_cmap.tau2, float(k))
                C, T = selection_draws(p_kap, y_kap, c_kap, 10, rng, KAPPA_TRIALS,
                                       transform=variant_cmap.transform, add=add)
                if len(C) == 0:
                    continue
                err = abs(np.mean(over_claim_array(C, T)))
                if err < best_err:
                    best_err, best_kappa = err, float(k)

            for n in N_GRID:
                add = adaptive_correction(variant_cmap.sigma2, n, variant_cmap.tau2, best_kappa)
                C, T = selection_draws(p_ev, y_ev, c_ev, n, rng, TRIALS,
                                       transform=variant_cmap.transform, add=add)
                if len(C) == 0:
                    continue
                over = over_claim_array(C, T)
                row["grids"].setdefault(str(n), {})[variant_name] = {
                    "p_over_5pp": float(np.mean(over > 5)),
                    "mean_over_claim_pct": float(np.mean(over)),
                    "n_draws": int(len(over)),
                    "best_kappa": best_kappa,
                }

        C_raw_ec3, T_raw_ec3 = selection_draws(p_ev, y_ev, c_ev, 10, rng, TRIALS,
                                               transform=lambda v: v, add=float(np.log(EC3_FACTOR)))
        if len(C_raw_ec3):
            ec3_over = over_claim_array(C_raw_ec3, T_raw_ec3)
            row["ec3_band_p_over_5pp_n10"] = float(np.mean(ec3_over > 5))

        evaluations.append(row)

    output = {
        "metadata": {
            "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            **ledger_provenance(
                __file__, "experiments/protocols/supplementary_reruns.md"
            ),
            "n_records": int(len(data)),
            "n_manufacturers": int(pd.Series(manufacturers).nunique()),
            "n_blocks_per_partition": N_BLOCKS,
            "split_seeds": SPLIT_SEEDS,
            "n_unique_partition_digests": len(partition_digests),
            "prior_construction": "leave-one-seed-out over this run's own beta_hat draws",
            "bootstrap_note": (
                "manufacturer-cluster bootstrap via multinomial draw counts, "
                "preserving resample multiplicity (fixes v1's np.isin membership bug)"
            ),
            "complete": complete,
            "corpus_note": (
                "No independent external corpus exists for this problem; the prior "
                "is same-corpus by necessity, made rigorous via leave-one-seed-out "
                "rather than a hardcoded constant from a disconnected old run."
            ),
        },
        "evaluations": evaluations,
        "failures": failures,
    }
    out_path = RESULTS / "exp_beta_shrinkage_v2.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nwrote {out_path}")
    if not complete:
        raise RuntimeError(f"Incomplete experiment: {failures}")

    # console summary
    for n in N_GRID:
        raw_p = [row["grids"].get(str(n), {}).get("raw", {}).get("p_over_5pp")
                for row in evaluations]
        shr_p = [row["grids"].get(str(n), {}).get("shrunk", {}).get("p_over_5pp")
                for row in evaluations]
        raw_p = [v for v in raw_p if v is not None]
        shr_p = [v for v in shr_p if v is not None]
        if raw_p and shr_p:
            print(f"n={n:>3}: mean P(over-claim>5pp) raw={np.mean(raw_p)*100:5.1f}%  "
                  f"shrunk={np.mean(shr_p)*100:5.1f}%")


if __name__ == "__main__":
    main()
