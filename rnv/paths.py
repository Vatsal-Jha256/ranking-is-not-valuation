"""Repository-relative paths shared by the library and experiment runners.
"""
from __future__ import annotations

import pathlib

#: Repository root: the directory that contains ``rnv/``.
ROOT = pathlib.Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
RAW_EC3 = DATA / "raw" / "ec3_materials"
PROCESSED = DATA / "processed"

RESULTS = ROOT / "results"
EXPERIMENTS = ROOT / "experiments"
PROTOCOLS = EXPERIMENTS / "protocols"

FIGURES = ROOT / "figures"

#: The single raw input every headline number is derived from.
READYMIX_JSONL = RAW_EC3 / "readymix.jsonl"

__all__ = [
    "ROOT", "DATA", "RAW_EC3", "PROCESSED", "RESULTS", "EXPERIMENTS", "PROTOCOLS",
    "FIGURES", "READYMIX_JSONL",
]
