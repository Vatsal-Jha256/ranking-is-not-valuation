#!/usr/bin/env python3
"""Fast checks that the repository is wired correctly, with no model fitting.

The full reproduction refits every model and takes hours. This suite takes seconds. It
does not try to be a substitute: it establishes that the pieces still connect, that the
shared library behaves exactly as the per-experiment code it replaced, and that every
declared path exists.

    python tests/test_smoke.py

Deliberately excluded: loading the corpus (a 200 MB snapshot), fitting any estimator, and
anything that writes into ``results/``. If a check here would need one of those, it
belongs in ``./reproduce.sh`` instead.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FAILURES: list[str] = []


def check(name: str):
    """Decorator that runs a check immediately and records its outcome."""
    def wrap(fn):
        try:
            detail = fn() or ""
            print(f"  ok   {name}{(' - ' + detail) if detail else ''}")
        except Exception as exc:  # noqa: BLE001 - report every failure, do not stop
            FAILURES.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
        return fn
    return wrap


# ---------------------------------------------------------------------------
# 1. The library imports, and every runner imports without executing.
# ---------------------------------------------------------------------------
print("library and runners")

LIBRARY_MODULES = [
    "rnv", "rnv.paths",
    "rnv.registry", "rnv.registry.normalize", "rnv.registry.features",
    "rnv.registry.clf_portfolio",
    "rnv.evaluation", "rnv.evaluation.partitions", "rnv.evaluation.model",
    "rnv.evaluation.metrics", "rnv.evaluation.corpus", "rnv.evaluation.bootstrap",
    "rnv.evaluation.provenance",
    "rnv.claims", "rnv.claims.certification", "rnv.claims.conformal",
    "rnv.claims.policies",
]


@check("every library module imports")
def _library_imports():
    for module in LIBRARY_MODULES:
        importlib.import_module(module)
    return f"{len(LIBRARY_MODULES)} modules"


@check("every experiment runner imports")
def _runner_imports():
    runners = sorted(ROOT.glob("experiments/*/run_*.py"))
    assert len(runners) >= 9, f"expected at least 9 runners, found {len(runners)}"
    for path in runners:
        spec = importlib.util.spec_from_file_location(f"smoke_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return f"{len(runners)} runners"


@check("every library module documents what it supplies")
def _docstrings():
    missing = []
    for module in LIBRARY_MODULES:
        doc = importlib.import_module(module).__doc__ or ""
        if len(doc.strip()) < 40:
            missing.append(module)
    assert not missing, f"thin or absent module docstring: {missing}"
    return f"{len(LIBRARY_MODULES)} documented"


@check("every runner names its protocol, ledger and paper section")
def _runner_docstrings():
    incomplete = []
    for path in sorted(ROOT.glob("experiments/*/run_*.py")):
        doc = path.read_text().split('"""')[1]
        if "Ledger:" not in doc or "Supplies:" not in doc:
            incomplete.append(path.name)
    assert not incomplete, f"runner docstring missing Ledger/Supplies: {incomplete}"


# ---------------------------------------------------------------------------
# 2. The shared library reproduces the per-experiment code it replaced.
#    Reference implementations below are the pre-refactor originals, kept here
#    so the equivalence is asserted rather than asserted-to-have-been-checked.
# ---------------------------------------------------------------------------
print("behavioural equivalence with the code the library replaced")


def _reference_pairwise_accuracy(pred, truth, classes, rng, n_draws=20_000):
    """The headline experiment's original nested-mask formulation."""
    import pandas as pd
    wins = total = ties = 0
    for label in pd.unique(classes):
        mask = classes == label
        p, t = pred[mask], truth[mask]
        if len(p) < 20:
            continue
        i = rng.integers(0, len(p), n_draws)
        j = rng.integers(0, len(p), n_draws)
        valid = i != j
        i, j = i[valid], j[valid]
        pred_order = p[i] < p[j]
        true_order = t[i] < t[j]
        valid = t[i] != t[j]
        pred_ties = p[i[valid]] == p[j[valid]]
        correct = pred_order[valid] == true_order[valid]
        wins += float(np.sum(correct[~pred_ties]) + 0.5 * np.sum(pred_ties))
        total += int(np.sum(valid))
        ties += int(np.sum(pred_ties))
    return float(wins / total), float(ties / total), total


def _synthetic_block(seed=0, n=900):
    rng = np.random.default_rng(seed)
    classes = rng.choice(np.array(["C20-25", "C25-30", "C30-35"]), size=n)
    truth = rng.normal(5.6, 0.3, n)
    pred = 0.6 * truth + rng.normal(0.0, 0.25, n)
    return pred, truth, classes


@check("pairwise_accuracy matches the original implementation")
def _pairwise_equivalence():
    from rnv.evaluation.metrics import pairwise_accuracy
    pred, truth, classes = _synthetic_block()
    shared = pairwise_accuracy(pred, truth, classes, np.random.default_rng(7))
    original = _reference_pairwise_accuracy(
        pred, truth, classes, np.random.default_rng(7)
    )
    assert shared == original, f"{shared} != {original}"
    return f"accuracy {shared[0]:.6f} on {shared[2]} pairs"


@check("r2_score_test_mean scores against the held-out mean")
def _r2_definition():
    from rnv.evaluation.metrics import r2_score_test_mean
    truth = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2_score_test_mean(truth, truth) == 1.0
    # A constant predictor at the test mean scores exactly zero, and anything
    # worse than that constant goes negative. That sign convention is the whole
    # point of RQ1, so it is pinned here.
    assert r2_score_test_mean(truth, np.full(4, truth.mean())) == 0.0
    assert r2_score_test_mean(truth, np.full(4, 10.0)) < 0.0


@check("the partition is seed-sensitive and keeps manufacturers whole")
def _partition_behaviour():
    from rnv.evaluation.partitions import stratified_random_group_blocks
    groups = np.repeat(np.arange(40), np.arange(1, 41))
    a, loads_a, digest_a = stratified_random_group_blocks(groups, n_blocks=4, seed=0)
    b, _, digest_b = stratified_random_group_blocks(groups, n_blocks=4, seed=1)
    again, _, digest_again = stratified_random_group_blocks(groups, n_blocks=4, seed=0)
    assert digest_a == digest_again and np.array_equal(a, again), "not deterministic"
    assert digest_a != digest_b, "different seeds gave the same partition"
    for group in np.unique(groups):
        assert len(set(a[groups == group])) == 1, "a manufacturer was split"
    spread = loads_a.max() / loads_a.min()
    assert spread < 1.5, f"block sizes are unbalanced: {loads_a}"
    return f"largest/smallest block = {spread:.2f}"


@check("the cluster bootstrap preserves resample multiplicity")
def _bootstrap_multiplicity():
    from rnv.evaluation.bootstrap import cluster_bootstrap_se
    # Two clusters with very different means. If multiplicity were collapsed to set
    # membership, almost every resample would contain both clusters and the standard
    # error would be near zero. Preserving counts keeps it far from zero.
    groups = np.repeat(np.array(["a", "b"]), 60)
    values = np.concatenate([np.zeros(60), np.ones(60)])
    se = cluster_bootstrap_se(
        values, groups, lambda rows: float(values[rows].mean()),
        n_resamples=400, seed=0,
    )
    assert se > 0.2, f"standard error {se} is too small to reflect cluster draws"
    return f"se = {se:.3f}"


@check("the conformal quantile applies its finite-sample correction")
def _conformal_correction():
    from rnv.claims.conformal import conformal_quantile
    scores = np.arange(100, dtype=float)
    corrected = conformal_quantile(scores, 0.10)
    plain = float(np.quantile(scores, 0.90))
    assert corrected > plain, f"{corrected} should exceed the plain quantile {plain}"
    try:
        conformal_quantile(np.arange(10, dtype=float), 0.10)
    except RuntimeError:
        pass
    else:
        raise AssertionError("too few scores should be refused, not reported")
    return f"{corrected:.1f} against {plain:.1f}"


@check("argmin achieves the deepest selection of any policy")
def _selection_depth_bound():
    from rnv.claims.policies import HEADLINE_POLICIES, choose
    rng = np.random.default_rng(0)
    scores = rng.normal(0.0, 1.0, 10)
    depths = {
        policy: float(scores.mean() - scores[choose(policy, scores, 0.2, np.random.default_rng(1))])
        for policy in HEADLINE_POLICIES
    }
    assert depths["argmin"] == max(depths.values()), depths
    return f"argmin depth {depths['argmin']:.3f}"


# ---------------------------------------------------------------------------
# 3. Declared paths exist, and the evidence chain resolves.
# ---------------------------------------------------------------------------
print("paths, protocols and the evidence chain")


@check("every public code and evidence path exists")
def _paths_exist():
    from rnv import paths
    # Raw datasets are supplied separately under their source licence.
    data_locations = {"DATA", "RAW_EC3", "PROCESSED", "READYMIX_JSONL"}
    missing = [
        name for name in paths.__all__
        if name not in data_locations and not getattr(paths, name).exists()
    ]
    assert not missing, f"declared but absent: {missing}"
    return f"{len(set(paths.__all__) - data_locations)} public locations"


@check("every shared provenance input exists and hashes")
def _provenance_inputs():
    from rnv.evaluation.provenance import SHARED_INPUTS, source_digests
    manifest = json.loads((ROOT / "reproducibility/input_manifest.json").read_text())
    raw = {item["path"] for item in manifest["files"] if item["role"] == "raw_input"}
    available = [path for path in SHARED_INPUTS if path not in raw or (ROOT / path).is_file()]
    digests = source_digests(*available)
    assert len(digests) == len(available)
    assert all(len(value) == 64 for value in digests.values())
    return f"{len(digests)} available inputs; {len(SHARED_INPUTS)-len(available)} absent licensed inputs not checked"


@check("every runner's declared protocol and ledger exist")
def _evidence_chain():
    for path in sorted(ROOT.glob("experiments/*/run_*.py")):
        doc = path.read_text().split('"""')[1]
        for line in doc.splitlines():
            if line.startswith(("Protocol:", "Ledger:")):
                for token in line.split("``")[1::2]:
                    target = ROOT / token
                    if "and" in token or not token.endswith((".md", ".json", ".csv")):
                        continue
                    assert target.exists(), f"{path.name} names a missing {token}"


@check("the reproduction script runs every reported experiment")
def _run_locked_covers_experiments():
    script = (ROOT / "reproducibility" / "run_locked.sh").read_text()
    manifest = json.loads((ROOT / "reproducibility" / "input_manifest.json").read_text())
    declared = [
        item["path"] for item in manifest["files"]
        if item["role"] == "experiment_code"
    ]
    missing = [path for path in declared if path not in script]
    assert not missing, f"manifest lists experiments the runner never runs: {missing}"
    return f"{len(declared)} experiments"


@check("the input manifest matches the working tree")
def _manifest_current():
    sys.path.insert(0, str(ROOT / "reproducibility"))
    generate_manifest = importlib.import_module("generate_manifest")
    checked, missing = generate_manifest.verify_manifest(allow_missing_raw=True)
    return f"{checked} inputs checked; {len(missing)} absent licensed inputs not checked"


@check("released result figures exist")
def _figures_present():
    names = ["fig1_submission_story.pdf", "fig1_rank_then_report.pdf",
             "figA1_premise_audit.pdf", "figA2_selection_vs_reporting.pdf",
             "figA3_pool_sensitivity.pdf", "figA4_temporal_holdout.pdf",
             "figA5_accepted_policies.pdf", "figA6_supplementary_checks.pdf"]
    for name in names:
        path = ROOT / "figures" / name
        assert path.is_file() and path.read_bytes().startswith(b"%PDF"), name
    return f"{len(names)} figures"


if __name__ == "__main__":
    print()
    if FAILURES:
        print(f"FAIL: {len(FAILURES)} smoke checks failed")
        for failure in FAILURES:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("PASS: all smoke checks")
