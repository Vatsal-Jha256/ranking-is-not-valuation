"""Climate TRACE facility-level downloader for heavy-industry subsectors.

Gives named plants with location, capacity, activity, and a per-plant emissions
factor (e.g. t CO2e per t of cement). Used as an independent, globally uniform
anchor for regions where EPD coverage is absent.

SCOPE CAVEAT (matters for any claim we make): Climate TRACE cement factors are
gate-to-gate production emissions for CEMENT, not cradle-to-gate A1-A3 GWP for
CONCRETE. Cement is the dominant but not the only contributor to concrete GWP.
These values are therefore a PREDICTOR / prior for regional clinker intensity,
never a drop-in substitute for an EPD.
"""
import argparse
import json
import pathlib
import time

import requests

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "climatetrace"
API = "https://api.climatetrace.org/v6"

SUBSECTORS = ["cement", "iron-and-steel", "aluminum", "lime", "glass"]

# India plus comparison economies: US/CA anchor the EPD-rich regime,
# CN/VN/ID/BD/BR/TR are other large, EPD-poor producers.
COUNTRIES = ["IND", "USA", "CAN", "CHN", "VNM", "IDN", "BGD", "BRA", "TUR",
             "GBR", "DEU", "SAU", "ARE", "NGA", "KEN", "ZAF", "PAK", "EGY"]


def get(session, path, **params):
    for attempt in range(5):
        try:
            r = session.get(f"{API}/{path}", params=params, timeout=120)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            if attempt == 4:
                print(f"    ! {path} {params} failed: {exc}", flush=True)
                return None
            time.sleep(3 * (attempt + 1))
    return None


def fetch_assets(session, subsector, country, limit=1000):
    """Page through facility records for one (subsector, country)."""
    rows, offset = [], 0
    while True:
        j = get(session, "assets", subsectors=subsector, countries=country,
                limit=limit, offset=offset)
        if not j:
            break
        batch = j.get("assets") if isinstance(j, dict) else j
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < limit:
            break
        offset += limit
        time.sleep(0.2)
    return rows


def coords(a):
    """Assets expose location as Centroid.Geometry [lon, lat], or flat fields."""
    cen = a.get("Centroid")
    if isinstance(cen, dict):
        g = cen.get("Geometry")
        if isinstance(g, (list, tuple)) and len(g) >= 2:
            return g[1], g[0]  # lat, lon
    return a.get("Latitude"), a.get("Longitude")


def flatten(a, subsector, country):
    """One row per (asset, gas) emissions summary entry."""
    lat, lon = coords(a)
    base = dict(
        asset_id=a.get("Id"), name=a.get("Name"), native_id=a.get("NativeId"),
        country=a.get("Country") or country, sector=a.get("Sector") or subsector,
        asset_type=a.get("AssetType"), subsector=subsector,
        lat=lat, lon=lon, confidence=a.get("Confidence"),
    )
    owners = a.get("Owners") or []
    if owners:
        base["owner"] = owners[0].get("CompanyName")
        base["owner_id"] = owners[0].get("CompanyId")

    out = []
    for e in (a.get("EmissionsSummary") or []):
        r = dict(base)
        r.update(
            gas=e.get("Gas"),
            activity=e.get("Activity"), activity_units=e.get("ActivityUnits"),
            capacity=e.get("Capacity"), capacity_units=e.get("CapacityUnits"),
            capacity_factor=e.get("CapacityFactor"),
            emissions_factor=e.get("EmissionsFactor"),
            emissions_factor_units=e.get("EmissionsFactorUnits"),
            emissions_quantity=e.get("EmissionsQuantity"),
            start_time=e.get("StartTime"), end_time=e.get("EndTime"),
        )
        out.append(r)
    return out or [base]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsectors", nargs="*", default=SUBSECTORS)
    ap.add_argument("--countries", nargs="*", default=COUNTRIES)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers.update({"Accept": "application/json",
                      "User-Agent": "Mozilla/5.0 (academic research)"})

    totals = {}
    dest = OUT / "assets.jsonl"
    with dest.open("w") as fh:
        for sub in args.subsectors:
            for c in args.countries:
                rows = fetch_assets(s, sub, c)
                n = 0
                for a in rows:
                    for r in flatten(a, sub, c):
                        fh.write(json.dumps(r, default=str) + "\n")
                        n += 1
                if n:
                    totals[f"{sub}/{c}"] = n
                    print(f"  {sub:>14} {c}  {len(rows):>5} assets -> {n:>5} rows", flush=True)
                time.sleep(0.2)

    # Country-level totals as a cross-check on the facility roll-up.
    ce = []
    for sub in args.subsectors:
        j = get(s, "country/emissions", subsectors=sub,
                countries=",".join(args.countries), since=2021, to=2024)
        if j:
            for row in j:
                row["subsector"] = sub
                ce.append(row)
    (OUT / "country_emissions.json").write_text(json.dumps(ce, indent=1, default=str))
    (OUT / "summary.json").write_text(json.dumps(totals, indent=2))
    print(f"\nwrote {dest} and country_emissions.json")
    print(f"total rows: {sum(totals.values()):,}")


if __name__ == "__main__":
    main()
