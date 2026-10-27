#!/usr/bin/env python3
"""Create or verify the immutable input manifest for the locked reproduction.

Every file listed here is content-addressed before any experiment runs, so a
reproduction that silently used edited code fails at the first step rather than
producing plausible numbers. ``write`` regenerates the manifest after an intentional
change; ``verify`` is what the container runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "reproducibility" / "input_manifest.json"

INPUTS = {
    "raw_input": [
        "data/raw/ec3_materials/readymix.jsonl",
        "data/raw/ec3_materials/readymix.state.json",
        "data/processed/clf_concrete_a1a3.csv",
    ],
    "protocol": [
        "experiments/protocols/selection_gap.md",
        "experiments/protocols/selection_gap_addendum_v2.md",
        "experiments/protocols/selection_gap_addendum_v3.md",
        "experiments/protocols/selection_gap_addendum_v4.md",
        "experiments/protocols/model_robustness.md",
        "experiments/protocols/model_robustness_addendum_v2.md",
        "experiments/protocols/model_robustness_addendum_v3.md",
        "experiments/protocols/review_response_suite.md",
        "experiments/protocols/supplementary_reruns.md",
        "experiments/protocols/stratified_quantile_baseline.md",
    ],
    "library_code": [
        "rnv/__init__.py",
        "rnv/paths.py",
        "rnv/registry/__init__.py",
        "rnv/registry/normalize.py",
        "rnv/registry/features.py",
        "rnv/registry/clf_portfolio.py",
        "rnv/evaluation/__init__.py",
        "rnv/evaluation/partitions.py",
        "rnv/evaluation/model.py",
        "rnv/evaluation/metrics.py",
        "rnv/evaluation/corpus.py",
        "rnv/evaluation/bootstrap.py",
        "rnv/evaluation/provenance.py",
        "rnv/claims/__init__.py",
        "rnv/claims/certification.py",
        "rnv/claims/conformal.py",
        "rnv/claims/policies.py",
    ],
    "experiment_code": [
        "experiments/selection_gap/run_selection_gap.py",
        "experiments/model_robustness/run_model_families.py",
        "experiments/calibration_premise/run_premise_audit.py",
        "experiments/reporting_policies/run_policy_comparison.py",
        "experiments/reporting_policies/run_stratified_quantile.py",
        "experiments/pool_sensitivity/run_pool_sweep.py",
        "experiments/temporal_holdout/run_temporal_holdout.py",
        "experiments/supplementary/run_beta_shrinkage.py",
        "experiments/supplementary/run_budget_allocation.py",
    ],
    "claim_and_figure_code": [
        "figures/gen_submission_figure.py",
        "figures/gen_appendix_figures.py",
        "figures/figstyle.py",
        "figures/specs/fig1_rank_then_report.json",
        "figures/fig1_rank_then_report.svg",
        "figures/fig1_rank_then_report.pdf",
        "reproducibility/verify_outputs.py",
    ],
    "data_preparation": [
        "scripts/prepare_data.py",
        "scripts/run_new_data.py",
    ],
    "locked_reference": [
        "results/exp_submission_harmonized_v4.json",
        "results/exp_model_robustness_v3.json",
        "results/exp_calibration_structure_v1.json",
        "results/exp_uncertainty_policies_v1.json",
        "results/exp_candidate_sensitivity_v1.json",
        "results/exp_temporal_holdout_v1.json",
        "results/exp_beta_shrinkage_v2.json",
        "results/exp_knapsack_v2.json",
        "results/exp_stratified_quantile_v1.json",
    ],
    "container_definition": [
        "Dockerfile",
        ".dockerignore",
        "reproduce.sh",
        "reproducibility/README.md",
        "reproducibility/requirements.in",
        "reproducibility/requirements.lock",
        "reproducibility/EXPERIMENT_INCLUSION_AUDIT.md",
        "reproducibility/CONTEXTUAL_DATA_SNAPSHOTS.md",
        "reproducibility/run_locked.sh",
        "reproducibility/generate_manifest.py",
    ],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest() -> dict:
    files = []
    for role, paths in INPUTS.items():
        for relative in paths:
            path = ROOT / relative
            if not path.is_file():
                raise FileNotFoundError(f"required manifest input is missing: {relative}")
            files.append({
                "path": relative,
                "role": role,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    files.sort(key=lambda item: item["path"])
    return {
        "schema_version": 1,
        "algorithm": "sha256",
        "description": (
            "Immutable inputs for the randomized size-stratified submission v4 "
            "reproduction and the review-response experiment suite."
        ),
        "files": files,
    }


def verify_manifest(manifest: Path = DEFAULT_MANIFEST, *, allow_missing_raw: bool = False) -> tuple[int, list[str]]:
    """Verify recorded inputs, optionally leaving absent licensed data unchecked.

    Present raw files must always match. Missing code or figure inputs always fail.
    The strict default is required before refitting the frozen experiments.
    """
    recorded = json.loads(manifest.read_text())
    if recorded.get("schema_version") != 1 or recorded.get("algorithm") != "sha256":
        raise ValueError("unsupported input manifest schema or hash algorithm")
    expected = {(role, path) for role, paths in INPUTS.items() for path in paths}
    actual = {(item["role"], item["path"]) for item in recorded["files"]}
    if actual != expected or len(actual) != len(recorded["files"]):
        raise ValueError("manifest inventory does not match the declared inputs")
    changed, missing_raw, checked = [], [], 0
    for item in recorded["files"]:
        relative = item["path"]
        path = ROOT / relative
        if not path.is_file():
            if allow_missing_raw and item["role"] == "raw_input":
                missing_raw.append(relative)
                continue
            changed.append(relative + " (missing)")
        elif path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            changed.append(relative + " (changed)")
        else:
            checked += 1
    if changed:
        raise ValueError("input manifest mismatch: " + ", ".join(changed))
    return checked, missing_raw


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("write", "verify"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--allow-missing-raw", action="store_true",
                        help="verify the public checkout without absent licensed raw data")
    args = parser.parse_args()

    if args.mode == "write":
        if args.allow_missing_raw:
            parser.error("--allow-missing-raw applies only to verify")
        current = build_manifest()
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.manifest}")
        return 0

    checked, missing = verify_manifest(args.manifest, allow_missing_raw=args.allow_missing_raw)
    print(f"PASS: verified {checked} immutable inputs")
    for relative in missing:
        print(f"NOT CHECKED: licensed input absent: {relative}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
