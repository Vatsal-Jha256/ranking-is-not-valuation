"""Real building bills of materials, used for structure only.

The CLF WBLCA portfolio supplies what the EPD registry cannot: how much concrete a real
project actually buys, in how many line items, at which strength grades. The
supplementary experiments use it to ask a portfolio-level question -- whether a claim
about a whole building can be defended when no single substitution can.

Its role is deliberately bounded, and the experiment inclusion audit holds it to that
bound: the portfolio contributes masses, grades and project structure. It is never a
training target and never the ground truth a claim is scored against; declared GWP from
the registry plays that role throughout.

Supplies: the portfolio line items behind the supplementary budgeted-allocation and
building-slate checks reported in the appendix.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from rnv.paths import PROCESSED, ROOT

PSI_TO_MPA = 1.0 / 145.0377

STRENGTH_CLASSES = ["C<20", "C20-25", "C25-30", "C30-35", "C35-40", "C40-50", "C50+"]


def clf_concrete() -> pd.DataFrame:
    """Concrete A1-A3 line items with a parseable strength class.

    ``data/processed/clf_concrete_a1a3.csv`` is a precomputed cache of this function's
    own output, generated once from the parquet/xlsx source and kept so that callers
    needing only this filtered subset -- the locked reproduction container among them,
    which installs no parquet engine -- never require pyarrow or openpyxl. The cache is
    read verbatim when present; the derivation below remains the source of truth for it.
    """
    cache = PROCESSED / "clf_concrete_a1a3.csv"
    if cache.exists():
        c = pd.read_csv(cache)
        c["strength_class"] = pd.Categorical(
            c["strength_class"], categories=STRENGTH_CLASSES, ordered=True,
        )
        return c
    pq = PROCESSED / "clf_lca.parquet"
    df = pd.read_parquet(pq) if pq.exists() else pd.read_excel(
        ROOT / "data/raw/clf_wblca/full_lca_results.xlsx")
    a = df[df.life_cycle_stage.astype(str).str.contains("A1-A3", na=False)]
    c = a[a.mat_group.astype(str).str.lower() == "concrete"].copy()

    def psi(s):
        m = re.search(r"(\d{3,5})", str(s))
        return float(m.group(1)) if m else np.nan

    c["psi"] = c.mat_type.map(psi)
    c["strength_mpa"] = c.psi * PSI_TO_MPA
    c = c[c.strength_mpa.between(5, 200)]
    c = c[(c.inv_mass > 0) & (c.gwp > 0)]
    bins = [0, 20, 25, 30, 35, 40, 50, 1e9]
    c["strength_class"] = pd.cut(c.strength_mpa, bins=bins, labels=STRENGTH_CLASSES)
    c["gwp_per_kg"] = c.gwp / c.inv_mass
    # drop physically implausible intensities
    c = c[c.gwp_per_kg.between(0.005, 1.5)]
    return c
