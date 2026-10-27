#!/usr/bin/env python3
"""Run the experiment suite on new source data in a separate checkout directory."""

from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
INPUTS = (
    "data/raw/ec3_materials/readymix.jsonl",
    "data/raw/ec3_materials/readymix.state.json",
    "data/processed/clf_concrete_a1a3.csv",
)
RUNNERS = (
    "experiments/selection_gap/run_selection_gap.py",
    "experiments/model_robustness/run_model_families.py",
    "experiments/calibration_premise/run_premise_audit.py",
    "experiments/reporting_policies/run_policy_comparison.py",
    "experiments/pool_sensitivity/run_pool_sweep.py",
    "experiments/temporal_holdout/run_temporal_holdout.py",
    "experiments/supplementary/run_beta_shrinkage.py",
    "experiments/supplementary/run_budget_allocation.py",
    "experiments/reporting_policies/run_stratified_quantile.py",
)


EXPECTED_RESULTS = dict(
    zip(
        RUNNERS,
        (
            ("exp_submission_harmonized_v4", 24),
            ("exp_model_robustness_v3", 96),
            ("exp_calibration_structure_v1", 24),
            ("exp_uncertainty_policies_v1", 24),
            ("exp_candidate_sensitivity_v1", 360),
            ("exp_temporal_holdout_v1", 6),
            ("exp_beta_shrinkage_v2", 30),
            ("exp_knapsack_v2", 24),
            ("exp_stratified_quantile_v1", 24),
        ),
    )
)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new directory for the isolated code, inputs, results, and logs",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="validate inputs and list the run order without fitting or writing files",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        parser.error("output directory already exists; choose a new path")
    for relative in INPUTS:
        if not (ROOT / relative).is_file():
            parser.error(f"missing {relative}; follow docs/DATA_ACCESS.md first")
    if args.plan:
        for runner in RUNNERS:
            print(f"{sys.executable} {runner}")
        print(
            f"Outputs would be written under {output}; no files written and no models fitted."
        )
        return 0
    revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    # Export only committed code. Local credentials and working files are not copied.
    with tempfile.TemporaryFile() as archive:
        subprocess.run(
            ["git", "-C", str(ROOT), "archive", "--format=tar", revision],
            stdout=archive,
            check=True,
        )
        archive.seek(0)
        output.mkdir(parents=True)
        with tarfile.open(fileobj=archive) as tar:
            tar.extractall(output, filter="data")
    (output / "results").rename(output / "reference-results")
    (output / "results").mkdir()
    (output / "logs").mkdir()
    for relative in INPUTS:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    metadata = dict(
        status="running",
        kind="new-data experiment; not a frozen-result reproduction",
        code_commit=revision,
        started_utc=datetime.now(timezone.utc).isoformat(),
        python=sys.version,
        inputs={p: sha256(output / p) for p in INPUTS},
        completed_runners=[],
    )
    report = output / "new-data-run.json"

    def save():
        report.write_text(json.dumps(metadata, indent=2) + "\n")

    save()
    env = os.environ.copy()
    for key in ("EC3_API_KEY", "EC3_API_BASE", "OPENEPD_API_BASE"):
        env.pop(key, None)
    env.update(
        {
            k: "1"
            for k in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        }
    )
    env.update(PYTHONHASHSEED="0", MPLBACKEND="Agg")
    with (output / "python-environment.txt").open("w") as stream:
        subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], stdout=stream, check=True
        )
    try:
        for runner in RUNNERS:
            print(
                f"Running {runner}; log: {output / 'logs' / (Path(runner).stem + '.log')}",
                flush=True,
            )
            with (output / "logs" / (Path(runner).stem + ".log")).open("w") as log:
                subprocess.run(
                    [sys.executable, str(output / runner)],
                    cwd=output,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            stem, count = EXPECTED_RESULTS[runner]
            ledger = json.loads((output / "results" / f"{stem}.json").read_text())
            if (
                ledger.get("failures")
                or len(ledger.get("evaluations", [])) != count
                or ledger.get("metadata", {}).get("complete") is False
            ):
                raise RuntimeError(
                    f"{runner}: incomplete or failed evaluations; see its log and ledger"
                )
            metadata["completed_runners"].append(runner)
            save()
        for script in (
            "figures/gen_submission_figure.py",
            "figures/gen_appendix_figures.py",
        ):
            subprocess.run(
                [sys.executable, str(output / script)], cwd=output, env=env, check=True
            )
        metadata["status"] = "complete"
    except (Exception, KeyboardInterrupt):
        metadata["status"] = "failed or interrupted"
        raise
    finally:
        metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    print(
        f"New-data results: {output / 'results'}; released references: {output / 'reference-results'}"
    )
    print("The original checkout and frozen input manifest were not changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
