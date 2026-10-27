"""Unit normalization and quality filtering for EC3 material records.

The raw API mixes units within the same field (m^3 / m3 / yd3, MPa / psi) and
contains physically impossible records. Every downstream result depends on this
being right, so the rules are explicit and each drop is counted and reported.

Supplies: the corpus for every experiment in ``experiments/``, and therefore the
record and manufacturer counts quoted in the manuscript's data description. The
drop ledger returned by :func:`normalize_concrete` is what the appendix's data
filtering table reports.
"""
import json
import pathlib
import re

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]

YD3_TO_M3 = 0.764554857984
PSI_TO_MPA = 1.0 / 145.0377
KSI_TO_MPA = 6.894757
LB_TO_KG = 0.45359237
TON_TO_KG = 907.18474  # US short ton


def _split(v):
    """'564 kgCO2e' -> (564.0, 'kgCO2e');  '1 m^3' -> (1.0, 'm^3')."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan, ""
    if isinstance(v, (int, float)):
        return float(v), ""
    s = str(v).strip()
    m = re.match(r"^\s*(-?[\d.]+(?:[eE]-?\d+)?)\s*(.*)$", s)
    if not m:
        return np.nan, s.lower()
    try:
        return float(m.group(1)), m.group(2).strip().lower()
    except ValueError:
        return np.nan, m.group(2).strip().lower()


def strength_mpa(v):
    """28-day compressive strength -> MPa, handling psi/ksi."""
    q, u = _split(v)
    if np.isnan(q):
        return np.nan
    u = u.replace(" ", "")
    if "ksi" in u:
        return q * KSI_TO_MPA
    if "psi" in u:
        return q * PSI_TO_MPA
    if "mpa" in u or u == "":
        return q
    if "kpa" in u:
        return q / 1000.0
    return np.nan


def declared_volume_m3(v):
    """Declared unit -> volume in m^3 (None if the unit is not a volume)."""
    q, u = _split(v)
    if np.isnan(q):
        return np.nan
    u = u.replace(" ", "").replace("^", "")
    if u in ("m3", "cubicmeter", "cubicmetre"):
        return q
    if u in ("yd3", "cubicyard", "cy"):
        return q * YD3_TO_M3
    if u in ("l", "liter", "litre"):
        return q / 1000.0
    return np.nan


def mass_kg(v):
    q, u = _split(v)
    if np.isnan(q):
        return np.nan
    u = u.replace(" ", "")
    if u.startswith("kg"):
        return q
    if u in ("t", "tonne", "metricton", "mt"):
        return q * 1000.0
    if u in ("ton", "shortton", "uston"):
        return q * TON_TO_KG
    if u in ("lb", "lbs", "pound"):
        return q * LB_TO_KG
    if u == "":
        return q
    return np.nan


def gwp_kgco2e(v):
    q, u = _split(v)
    if np.isnan(q):
        return np.nan
    u = u.replace(" ", "")
    if u.startswith("tco2") or u.startswith("tonneco2"):
        return q * 1000.0
    if u.startswith("gco2"):
        return q / 1000.0
    return q  # kgCO2e (default)


def plain(v):
    return _split(v)[0]


# Physically plausible ranges for concrete, used to drop corrupt records.
CONCRETE_LIMITS = dict(
    gwp_per_m3=(40.0, 2000.0),      # kgCO2e / m3
    density=(1400.0, 3000.0),       # kg / m3
    strength=(5.0, 200.0),          # MPa
    gwp_per_kg=(0.005, 1.5),        # kgCO2e / kg
)


def normalize_concrete(df, verbose=True):
    """Return a cleaned concrete frame plus a drop-reason ledger."""
    n0 = len(df)
    d = df.copy()

    d["strength_mpa"] = d.get("concrete_compressive_strength_28d").map(strength_mpa)
    d["declared_m3"] = d.get("declared_unit").map(declared_volume_m3)
    d["mass_kg"] = d.get("mass_per_declared_unit").map(mass_kg)
    d["gwp_raw"] = d.get("gwp").map(gwp_kgco2e)
    d["density_kgm3"] = d.get("density").map(lambda v: _split(v)[0])

    for c in ["gwp_per_kg", "gwp_z", "uncertainty_factor", "standard_deviation",
              "best_practice", "conservative_estimate", "lowest_plausible_gwp",
              "pct80_gwp_per_category_declared_unit", "concrete_w_c_ratio"]:
        if c in d:
            d[c + "_n"] = d[c].map(plain)

    # Normalize GWP to a per-m3 basis using the declared volume.
    d["gwp_per_m3"] = d.gwp_raw / d.declared_m3

    ledger = {}

    def drop(mask, reason):
        nonlocal d
        k = int(mask.sum())
        if k:
            ledger[reason] = k
            d = d[~mask].copy()

    drop(d.gwp_per_m3.isna(), "missing GWP or non-volumetric declared unit")
    drop(d.strength_mpa.isna(), "missing 28-day strength")
    lo, hi = CONCRETE_LIMITS["gwp_per_m3"]
    drop(~d.gwp_per_m3.between(lo, hi), f"GWP/m3 outside [{lo},{hi}]")
    lo, hi = CONCRETE_LIMITS["strength"]
    drop(~d.strength_mpa.between(lo, hi), f"strength outside [{lo},{hi}] MPa")

    # Density: prefer explicit density, else mass/declared volume.
    dens = d.density_kgm3.where(d.density_kgm3.notna(), d.mass_kg / d.declared_m3)
    d["density_kgm3"] = dens
    lo, hi = CONCRETE_LIMITS["density"]
    drop(~d.density_kgm3.between(lo, hi), f"density outside [{lo},{hi}] kg/m3")

    d["gwp_per_kg_norm"] = d.gwp_per_m3 / d.density_kgm3
    lo, hi = CONCRETE_LIMITS["gwp_per_kg"]
    drop(~d.gwp_per_kg_norm.between(lo, hi), f"GWP/kg outside [{lo},{hi}]")

    # Deduplicate: same plant + name + GWP is the same declaration re-listed.
    before = len(d)
    d = d.drop_duplicates(subset=[c for c in ["mfr_name", "plant_name", "name", "gwp_per_m3"]
                                  if c in d]).copy()
    if before - len(d):
        ledger["duplicate declarations"] = before - len(d)

    # Standard strength classes (functional-equivalence proxy).
    bins = [0, 20, 25, 30, 35, 40, 50, 1e9]
    labels = ["C<20", "C20-25", "C25-30", "C30-35", "C35-40", "C40-50", "C50+"]
    d["strength_class"] = pd.cut(d.strength_mpa, bins=bins, labels=labels)

    if verbose:
        print(f"normalize_concrete: {n0:,} -> {len(d):,} usable "
              f"({100*len(d)/max(n0,1):.1f}% kept)")
        for r, k in sorted(ledger.items(), key=lambda x: -x[1]):
            print(f"   dropped {k:>7,}  {r}")
    return d, ledger


def load_jsonl(slug):
    p = ROOT / "data/raw/ec3_materials" / f"{slug}.jsonl"
    if not p.exists():
        raise FileNotFoundError(p)
    rows = []
    with p.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return pd.DataFrame(rows)
