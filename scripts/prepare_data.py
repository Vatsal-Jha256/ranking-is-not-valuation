#!/usr/bin/env python3
"""Download source data and run the preprocessing used by the experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EC3_BASE = os.environ.get(
    "EC3_API_BASE", "https://buildingtransparency.org/api"
).rstrip("/")
READYMIX_CATEGORY = "b03dba1dca5b49acb1a5aa4daab546b4"
CLF_URL = "https://ndownloader.figshare.com/files/52575182"
CLF_MD5 = "40c83ca205efb3ae44ac0d85d5061c1c"


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def api_session():
    import requests

    key = os.environ.get("EC3_API_KEY", "").strip()
    if not key or key == "your_key_here":
        raise ValueError("Set EC3_API_KEY first; see docs/DATA_ACCESS.md, step 2.")
    session = requests.Session()
    session.headers.update(
        {"Authorization": f"Bearer {key}", "Accept": "application/json"}
    )
    return session


def get_page(session, page, page_size):
    for attempt in range(5):
        response = session.get(
            f"{EC3_BASE}/materials",
            params={
                "category": READYMIX_CATEGORY,
                "page_size": page_size,
                "page_number": page,
            },
            timeout=180,
        )
        if response.status_code in (401, 403):
            raise PermissionError(
                f"EC3 returned HTTP {response.status_code}. Check key expiry, read scope, and account access."
            )
        if response.status_code == 429 or response.status_code >= 500:
            if attempt == 4:
                raise RuntimeError(
                    f"EC3 returned HTTP {response.status_code} after five attempts; rerun with --resume later."
                )
            try:
                delay = float(response.headers.get("Retry-After", 2 ** (attempt + 1)))
            except ValueError:
                delay = 2 ** (attempt + 1)
            time.sleep(max(0, delay))
            continue
        response.raise_for_status()
        batch = response.json()
        if not isinstance(batch, list) or not all(
            isinstance(row, dict) for row in batch
        ):
            raise ValueError(
                "EC3 /materials did not return a list of records; inspect the API schema before continuing."
            )
        return batch, response.headers.get("X-Total-Count")
    raise RuntimeError("No successful response from EC3")


def check_key(_args):
    with api_session() as session:
        batch, total = get_page(session, 1, 1)
    print(
        f"PASS: EC3 materials read access; {len(batch)} sample record; reported total: {total or 'not supplied'}."
    )


def fetch_ec3(args):
    from rnv.registry.fetch_materials import reduce_record

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    dest, state = out / "readymix.jsonl", out / "readymix.state.json"
    pages, info = out / ".readymix-pages", out / "readymix.download.json"
    if dest.exists() or state.exists():
        raise FileExistsError(
            "Final data/state already exist. Choose a new --output-dir; existing snapshots are never overwritten."
        )
    if pages.exists() and not args.resume:
        raise FileExistsError(
            "An unfinished download exists. Use --resume with the same page size."
        )
    pages.mkdir(exist_ok=True)
    if info.exists():
        metadata = json.loads(info.read_text())
        if metadata["page_size"] != args.page_size or metadata["api_base"] != EC3_BASE:
            raise ValueError("Resume settings differ from the interrupted download.")
    else:
        metadata = dict(
            started_utc=utc(),
            api_base=EC3_BASE,
            category=READYMIX_CATEGORY,
            page_size=args.page_size,
            complete=False,
        )
    write_json(info, metadata)
    count, next_page, complete = 0, 1, False
    total = None
    with api_session() as session:
        for page in range(1, args.max_pages + 1):
            page_file = pages / f"{page:06d}.jsonl"
            if page_file.exists():
                batch_lines = page_file.read_text().splitlines()
                count += len(batch_lines)
                next_page = page + 1
                continue
            batch, total_header = get_page(session, page, args.page_size)
            if total_header is not None:
                total = int(total_header)
            if not batch:
                if total is not None and count < total:
                    raise RuntimeError(
                        f"Empty page before reported total: {count} / {total}; download left incomplete."
                    )
                complete = True
                break
            tmp = page_file.with_suffix(".tmp")
            with tmp.open("w") as stream:
                for record in batch:
                    stream.write(json.dumps(reduce_record(record)) + "\n")
            tmp.replace(page_file)
            count += len(batch)
            next_page = page + 1
            metadata.update(rows=count, next_page=next_page, total_reported=total)
            write_json(info, metadata)
            print(f"Page {page}: {count:,} records", flush=True)
            # An empty page, rather than page length, establishes completion.
            time.sleep(args.delay)
    if not complete:
        raise RuntimeError(
            "Page limit reached. Download is incomplete; increase --max-pages and use --resume."
        )
    if count == 0:
        raise ValueError(
            "No ready-mix records returned; check access and category filtering."
        )
    tmp = dest.with_suffix(".tmp")
    with tmp.open("wb") as stream:
        for page_file in sorted(pages.glob("*.jsonl")):
            with page_file.open("rb") as source:
                shutil.copyfileobj(source, stream)
    tmp.replace(dest)
    state.write_text(json.dumps({"next_page": next_page, "n": count}))
    metadata.update(complete=True, finished_utc=utc(), rows=count, sha256=digest(dest))
    write_json(info, metadata)
    print(f"Complete: {count:,} records in {dest}")


def fetch_clf(args):
    import requests

    dest = args.output.resolve()
    if dest.exists():
        if digest(dest, "md5") == CLF_MD5:
            print(f"PASS: existing CLF workbook matches the source checksum: {dest}")
            return
        raise FileExistsError(
            "The output exists with a different checksum; choose a new path."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(CLF_URL, stream=True, timeout=180) as response:
        response.raise_for_status()
        with tmp.open("wb") as stream:
            for chunk in response.iter_content(1024 * 1024):
                stream.write(chunk)
    if digest(tmp, "md5") != CLF_MD5:
        raise ValueError(
            "CLF workbook checksum mismatch; incomplete file retained with .part suffix."
        )
    tmp.replace(dest)
    print(f"PASS: CLF workbook downloaded and source checksum verified: {dest}")


def preprocess_ec3(args):
    import numpy as np
    import pandas as pd
    from rnv.registry.normalize import normalize_concrete
    from rnv.registry.features import build_features_raw

    out = args.output_dir.resolve()
    outputs = [
        out / name
        for name in (
            "readymix_clean.csv",
            "readymix_features.csv",
            "readymix_preprocessing.json",
        )
    ]
    if any(path.exists() for path in outputs):
        raise FileExistsError(
            "Preprocessing outputs already exist; choose a new --output-dir."
        )
    rows = []
    with args.input.open() as stream:
        for lineno, raw in enumerate(stream, 1):
            if raw.strip():
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {lineno}; repair or resume the download."
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Line {lineno} is not an object.")
                rows.append(row)
    if not rows:
        raise ValueError("Input JSONL is empty.")
    frame = pd.DataFrame(rows)
    required = {
        "concrete_compressive_strength_28d",
        "declared_unit",
        "mass_per_declared_unit",
        "gwp",
        "density",
        "description",
        "name",
        "mfr_name",
        "created_on",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("Required API fields absent: " + ", ".join(sorted(missing)))
    clean, drops = normalize_concrete(frame, verbose=False)
    unclassified = int(clean.strength_class.isna().sum())
    clean = clean[clean.strength_class.notna()].reset_index(drop=True)
    if clean.empty:
        raise ValueError("No rows survive preprocessing.")
    features = build_features_raw(clean).reset_index(drop=True)
    clean["target_log_gwp"] = np.log(clean.gwp_per_m3)
    out.mkdir(parents=True, exist_ok=True)
    clean.to_csv(outputs[0], index=False)
    features.to_csv(outputs[1], index=False)
    expected = json.loads((ROOT / "reproducibility/input_manifest.json").read_text())
    expected_hash = next(
        item["sha256"]
        for item in expected["files"]
        if item["path"] == "data/raw/ec3_materials/readymix.jsonl"
    )
    summary = dict(
        input=str(args.input.resolve()),
        input_sha256=digest(args.input),
        raw_rows=len(frame),
        retained_rows=len(clean),
        manufacturer_labels=int(clean.mfr_name.fillna("unknown").nunique()),
        drops=drops,
        unclassified_rows=unclassified,
        feature_columns=list(features.columns),
        feature_missing_counts={k: int(v) for k, v in features.isna().sum().items()},
        strength_class_counts={
            str(k): int(v)
            for k, v in clean.strength_class.value_counts(sort=False).items()
        },
        global_imputation_applied=False,
        matches_frozen_snapshot=digest(args.input) == expected_hash,
    )
    write_json(outputs[2], summary)
    print(
        f"PASS: {len(frame):,} raw -> {len(clean):,} retained; {summary['manufacturer_labels']} manufacturer labels."
    )
    print(
        f"Preprocessing report: {outputs[2]}; frozen snapshot match: {summary['matches_frozen_snapshot']}"
    )


def prepare_clf(args):
    from rnv.registry import clf_portfolio

    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(
            "CLF cache already exists; choose another --output to rebuild for comparison."
        )
    if source.suffix.lower() not in (".xlsx", ".parquet") or not source.is_file():
        raise ValueError(
            "Provide an existing CLF full_lca_results.xlsx or parquet conversion."
        )
    # Isolate the loader so it cannot read an existing cache instead of rebuilding.
    old_root, old_processed = clf_portfolio.ROOT, clf_portfolio.PROCESSED
    with tempfile.TemporaryDirectory() as temp:
        isolated = Path(temp)
        target = isolated / (
            "data/processed/clf_lca.parquet"
            if source.suffix.lower() == ".parquet"
            else "data/raw/clf_wblca/full_lca_results.xlsx"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source)
        try:
            clf_portfolio.ROOT = isolated
            clf_portfolio.PROCESSED = isolated / "data/processed"
            frame = clf_portfolio.clf_concrete()
        finally:
            clf_portfolio.ROOT, clf_portfolio.PROCESSED = old_root, old_processed
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    manifest = json.loads((ROOT / "reproducibility/input_manifest.json").read_text())
    expected = next(
        item["sha256"]
        for item in manifest["files"]
        if item["path"] == "data/processed/clf_concrete_a1a3.csv"
    )
    report = dict(
        source=str(source),
        source_sha256=digest(source),
        rows=len(frame),
        projects=int(frame.project_index.nunique()),
        output_sha256=digest(output),
        matches_frozen_cache=digest(output) == expected,
    )
    write_json(output.with_suffix(".preparation.json"), report)
    print(
        f"PASS: {len(frame):,} concrete A1-A3 rows; {report['projects']} projects; frozen cache match: {report['matches_frozen_cache']}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sub = commands.add_parser(
        "check-key", help="test EC3 materials read access without displaying the key"
    )
    sub.set_defaults(run=check_key)
    sub = commands.add_parser(
        "fetch-ec3", help="download ready-mix materials, with resumable page files"
    )
    sub.add_argument("--output-dir", type=Path, default=ROOT / "data/raw/ec3_materials")
    sub.add_argument("--resume", action="store_true")
    sub.add_argument("--page-size", type=int, default=250)
    sub.add_argument("--max-pages", type=int, default=2000)
    sub.add_argument("--delay", type=float, default=0.25)
    sub.set_defaults(run=fetch_ec3)
    sub = commands.add_parser(
        "fetch-clf", help="download the checksum-pinned public CLF workbook"
    )
    sub.add_argument(
        "--output", type=Path, default=ROOT / "data/raw/clf_wblca/full_lca_results.xlsx"
    )
    sub.set_defaults(run=fetch_clf)
    sub = commands.add_parser(
        "preprocess-ec3",
        help="export cleaned rows, un-imputed features, and a filtering report",
    )
    sub.add_argument(
        "--input", type=Path, default=ROOT / "data/raw/ec3_materials/readymix.jsonl"
    )
    sub.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/processed/readymix"
    )
    sub.set_defaults(run=preprocess_ec3)
    sub = commands.add_parser(
        "prepare-clf", help="rebuild the concrete A1-A3 portfolio cache"
    )
    sub.add_argument(
        "--source", type=Path, default=ROOT / "data/raw/clf_wblca/full_lca_results.xlsx"
    )
    sub.add_argument(
        "--output", type=Path, default=ROOT / "data/processed/clf_concrete_a1a3.csv"
    )
    sub.set_defaults(run=prepare_clf)
    args = parser.parse_args()
    if (
        getattr(args, "page_size", 1) <= 0
        or getattr(args, "max_pages", 1) <= 0
        or getattr(args, "delay", 0) < 0
    ):
        parser.error(
            "page size and page limit must be positive; delay must be nonnegative"
        )
    try:
        args.run(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
