#!/usr/bin/env python3
"""Compare regenerated experiment ledgers with their frozen references.

Numerical fields use an absolute floating-point tolerance; run timestamps and
provenance fields are compared separately. Source digests and figure files are
verified from disk. The code release does not require manuscript files.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
#: Metadata that legitimately differs between two correct runs: wall-clock times, and
#: the provenance block, which is verified against disk instead (see the module
#: docstring).
IGNORED_METADATA = {"started_utc", "finished_utc", "script_path"}
# BLAS/platform-level prediction perturbations can alter averaged rank correlations at
# roughly 1e-8 while leaving every discrete decision and other reported metric unchanged.
TOLERANCE = 2e-8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ignored_metadata(key: str) -> bool:
    """True for metadata that two correct runs may legitimately disagree about."""
    return key in IGNORED_METADATA or key.endswith("_sha256")


def verify_provenance(name: str, ledger: dict, *, unavailable_inputs: dict[str, str] | None = None) -> dict[str, Any]:
    """Recompute every digest a ledger records, against the files now on disk."""
    metadata = ledger["metadata"]
    script = metadata["script_path"]
    if metadata["script_sha256"] != sha256(ROOT / script):
        raise AssertionError(f"{name}: ledger script digest does not match {script}")
    recorded = {**metadata["protocol_sha256"], **metadata["source_sha256"]}
    missing = []
    for relative, digest in recorded.items():
        if not (ROOT / relative).is_file() and unavailable_inputs and relative in unavailable_inputs:
            if unavailable_inputs[relative] != digest:
                raise AssertionError(f"{name}: recorded digest for absent {relative} differs from the manifest")
            missing.append(relative)
            continue
        if digest != sha256(ROOT / relative):
            raise AssertionError(f"{name}: recorded digest for {relative} is stale")
    return {"script": script, "verified_digests": 1 + len(recorded) - len(missing), "unavailable_inputs": missing}


def compare(reference: Any, candidate: Any, path: str = "$") -> list[str]:
    mismatches: list[str] = []
    if isinstance(reference, dict) and isinstance(candidate, dict):
        ref_keys = set(reference)
        cand_keys = set(candidate)
        if path == "$.metadata":
            ref_keys = {key for key in ref_keys if not ignored_metadata(key)}
            cand_keys = {key for key in cand_keys if not ignored_metadata(key)}
        for key in sorted(ref_keys | cand_keys):
            if key not in reference or key not in candidate:
                mismatches.append(f"{path}.{key}: missing key")
            else:
                mismatches.extend(compare(reference[key], candidate[key], f"{path}.{key}"))
        return mismatches
    if isinstance(reference, list) and isinstance(candidate, list):
        if len(reference) != len(candidate):
            return [f"{path}: length {len(candidate)} != {len(reference)}"]
        for index, (left, right) in enumerate(zip(reference, candidate)):
            mismatches.extend(compare(left, right, f"{path}[{index}]"))
        return mismatches
    if isinstance(reference, (int, float)) and isinstance(candidate, (int, float)):
        if not math.isclose(float(reference), float(candidate), rel_tol=0.0, abs_tol=TOLERANCE):
            mismatches.append(f"{path}: {candidate!r} != {reference!r}")
        return mismatches
    if reference != candidate:
        mismatches.append(f"{path}: {candidate!r} != {reference!r}")
    return mismatches


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--robustness-reference", type=Path, required=True)
    parser.add_argument("--robustness-candidate", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--appendix-figure", type=Path, action="append", default=[])
    parser.add_argument("--review-pair", nargs=3, action="append", default=[],
                        metavar=("NAME", "REFERENCE", "CANDIDATE"))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    reference = json.loads(args.reference.read_text())
    candidate = json.loads(args.candidate.read_text())
    mismatches = compare(reference, candidate)
    if mismatches:
        preview = "\n".join(mismatches[:20])
        raise AssertionError(f"fresh ledger differs from locked v4 ledger ({len(mismatches)} fields):\n{preview}")

    robustness_reference = json.loads(args.robustness_reference.read_text())
    robustness_candidate = json.loads(args.robustness_candidate.read_text())
    robustness_mismatches = compare(robustness_reference, robustness_candidate)
    if robustness_mismatches:
        preview = "\n".join(robustness_mismatches[:20])
        raise AssertionError(
            "fresh robustness ledger differs from its locked reference "
            f"({len(robustness_mismatches)} fields):\n{preview}"
        )

    if (len(candidate.get("evaluations", [])) != 24 or candidate.get("failures")
            or not candidate.get("metadata", {}).get("complete")):
        raise AssertionError("expected 24 successful evaluations and no failures")
    if (len(robustness_candidate.get("evaluations", [])) != 96
            or robustness_candidate.get("failures")):
        raise AssertionError("expected 96 successful robustness evaluations and no failures")
    if candidate["metadata"].get("n_unique_partition_digests") != 6:
        raise AssertionError("expected six unique submission partition digests")
    jaccards = candidate.get("partition_diagnostics", {}).get(
        "same_fold_test_manufacturer_jaccards", []
    )
    if len(jaccards) != 60 or not all(0.0 <= row["jaccard"] < 1.0 for row in jaccards):
        raise AssertionError("submission partition-Jaccard diagnostics are incomplete")
    if robustness_candidate["metadata"].get("n_unique_partition_digests") != 6:
        raise AssertionError("expected six unique robustness partition digests")
    if not args.figure.is_file() or args.figure.stat().st_size < 10_000:
        raise AssertionError("generated figure is missing or unexpectedly small")
    if args.figure.read_bytes()[:4] != b"%PDF":
        raise AssertionError("generated figure is not a PDF")

    provenance = {
        "selection_gap": verify_provenance("selection_gap", candidate),
        "model_robustness": verify_provenance("model_robustness", robustness_candidate),
    }

    expected_evaluations = {
        "calibration_structure": 24,
        "uncertainty_policies": 24,
        "candidate_sensitivity": 360,
        "temporal_holdout": 6,
        "beta_shrinkage": 30,
        "budget_allocation": 24,
        "stratified_quantile": 24,
    }
    review_ledgers = {}
    review_comparison = {}
    for name, reference_path, candidate_path in args.review_pair:
        if name not in expected_evaluations:
            raise AssertionError(f"unknown review experiment: {name}")
        fresh = json.loads(Path(candidate_path).read_text())
        locked = json.loads(Path(reference_path).read_text())
        differences = compare(locked, fresh)
        if differences:
            preview = "\n".join(differences[:20])
            raise AssertionError(
                f"fresh {name} ledger differs from its locked reference "
                f"({len(differences)} fields):\n{preview}"
            )
        if fresh.get("failures") or not fresh["metadata"].get("complete"):
            raise AssertionError(f"{name} did not complete cleanly")
        if len(fresh["evaluations"]) != expected_evaluations[name]:
            raise AssertionError(
                f"{name} produced {len(fresh['evaluations'])} evaluations, "
                f"expected {expected_evaluations[name]}"
            )
        provenance[name] = verify_provenance(name, fresh)
        review_ledgers[name] = fresh
        review_comparison[name] = {
            "reference": str(reference_path),
            "candidate": str(candidate_path),
            "mismatches": 0,
            "evaluations": len(fresh["evaluations"]),
        }
    if set(review_ledgers) != set(expected_evaluations):
        raise AssertionError("every review-response experiment must be verified")

    for path in args.appendix_figure:
        if not path.is_file() or path.stat().st_size < 5_000:
            raise AssertionError(f"figure missing or too small: {path}")
        if path.read_bytes()[:4] != b"%PDF":
            raise AssertionError(f"figure is not a PDF: {path}")

    packages = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib"):
        packages[name] = importlib.metadata.version(name)

    report = {
        "status": "PASS",
        "comparison": {
            "reference": str(args.reference),
            "candidate": str(args.candidate),
            "ignored_fields": [f"metadata.{key}" for key in sorted(IGNORED_METADATA)],
            "absolute_float_tolerance": TOLERANCE,
            "mismatches": 0,
        },
        "robustness_comparison": {
            "reference": str(args.robustness_reference),
            "candidate": str(args.robustness_candidate),
            "ignored_fields": [f"metadata.{key}" for key in sorted(IGNORED_METADATA)],
            "absolute_float_tolerance": TOLERANCE,
            "mismatches": 0,
        },
        "review_comparison": review_comparison,
        "provenance": provenance,
        "artifacts": {
            "ledger": {"bytes": args.candidate.stat().st_size, "sha256": sha256(args.candidate)},
            "robustness_ledger": {
                "bytes": args.robustness_candidate.stat().st_size,
                "sha256": sha256(args.robustness_candidate),
            },
            "figure": {"bytes": args.figure.stat().st_size, "sha256": sha256(args.figure)},
            "appendix_figures": {
                path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in args.appendix_figure
            },
        },
        "packages": packages,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        "PASS: nine fresh ledgers match their locked references; provenance and figures verified"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
