"""Content digests recorded into every result ledger.

A result is only evidence if you can tell which code and which input produced it. Every
experiment writes :func:`ledger_provenance` into its ledger metadata: the digest of its
own script, of each protocol document it was prespecified in, and of the shared library
surface it depends on. The verifier recomputes all of them from disk, so a ledger whose
code has moved on announces itself instead of quietly ageing.

Digests record what a file contained *and where it lived*, because the keys are
repository-relative paths. That is deliberate: a reorganisation should be visible in
the evidence trail rather than invisible.

Supplies: the ``metadata.script_sha256``, ``metadata.protocol_sha256`` and
``metadata.source_sha256`` blocks of every file in ``results/``, and the
reproducibility statement in the appendix.
"""
from __future__ import annotations

import hashlib
import pathlib

from rnv.paths import ROOT


def sha256(path) -> str:
    """Content digest of one file."""
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def source_digests(*relative_paths: str) -> dict[str, str]:
    """Content digests keyed by repository-relative path."""
    return {relative: sha256(ROOT / relative) for relative in relative_paths}


#: The raw input and the shared library surface every experiment's numbers depend on.
SHARED_INPUTS = (
    "data/raw/ec3_materials/readymix.jsonl",
    "rnv/registry/normalize.py",
    "rnv/registry/features.py",
    "rnv/claims/certification.py",
    "rnv/claims/conformal.py",
    "rnv/claims/policies.py",
    "rnv/evaluation/partitions.py",
    "rnv/evaluation/model.py",
    "rnv/evaluation/metrics.py",
    "rnv/evaluation/corpus.py",
    "rnv/evaluation/bootstrap.py",
)


def ledger_provenance(script, *protocols: str) -> dict:
    """The standard provenance block for a result ledger.

    Args:
        script: ``__file__`` of the experiment being run.
        protocols: repository-relative paths of the protocol documents that
            prespecified it, in the order they were written.
    """
    script_path = pathlib.Path(script).resolve()
    return {
        "script_path": str(script_path.relative_to(ROOT)),
        "script_sha256": sha256(script_path),
        "protocol_sha256": source_digests(*protocols),
        "source_sha256": source_digests(*SHARED_INPUTS),
    }
