"""EC3 /materials downloader — category-filtered, streamed to compact JSONL.

Unlike openEPD /epds (which silently ignores all filter params), EC3 /materials
honours ?category=<uuid>. It also exposes the fields this project needs:

  * GWP + EC3's own heuristic uncertainty treatment
    (gwp_z, standard_deviation, uncertainty_factor, conservative_estimate,
     lowest_plausible_gwp, pct80_gwp_per_category_declared_unit)
  * functional-equivalence attributes (concrete strength/w-c ratio/exposure class,
    steel grade/yield, gypsum thickness, insulation R-value, ...)
  * geography (manufacturer lat/lon/address, plant)

Records are reduced to a whitelist at fetch time so the corpus stays small.
"""
import argparse
import json
import os
import pathlib
import sys
import time

import requests

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "ec3_materials"
BASE = "https://buildingtransparency.org/api"

# Categories worth pulling: high embodied-carbon, structurally substitutable,
# and with enough records to support per-class conformal calibration.
CATEGORIES = {
    "readymix": "b03dba1dca5b49acb1a5aa4daab546b4",
    "aggregates": "f4b20196cf844933b4222809f51386fa",
    "cementitious": "c75fcf155c324377a4fc622744d112ea",
    "precast_concrete": "3d6ccdcd4a83401e96ae385d568dc339",
    "cmu": "4ec837a26a0a493786442296f4cb2730",
    "masonry": "3fc5526e02864303b5a24b05506a308a",
    "steel": "db641aa0dbc34d6096547da02bf73f3a",
    "rebar_steel": "4e82b07fff2e40aeb1b72b695e9ac279",
    "structural_steel": "820e86326d2945fb9322246fd4690fc8",
    "wood": "fd14efc8874c4a55ac30d84a5612feb1",
    "aluminium": "26021028190540229c4994c0ea5ea476",
    "gypsum": "0d1d4544c70f4a82938c79d88f86ec5d",
    "finishes": "9152dd8544d6438fb388b66c2bd272f8",
}

# Scalar fields kept verbatim.
SCALARS = [
    "id", "name", "description", "sku", "epd_id", "open_xpd_uuid",
    "declaration_type", "declared_unit", "mass_per_declared_unit", "density",
    "specific_density", "thickness_per_declared_unit",
    # GWP + EC3's heuristic uncertainty machinery (our baseline)
    "gwp", "gwp_per_kg", "gwp_z", "gwp_per_category_declared_unit",
    "uncertainty_adjusted_gwp", "uncertainty_factor", "standard_deviation",
    "best_practice", "conservative_estimate", "lowest_plausible_gwp",
    "pct80_gwp_per_category_declared_unit",
    "biogenic_embodied_carbon", "stored_carbon",
    # provenance / recency
    "created_on", "updated_on", "reference_service_life", "manufactured",
    "is_failed", "manufacture_type",
    # generic functional attributes
    "compressive_strength", "recycled_content", "post_consumer_recycled_content",
    "total_recycled_content", "thickness", "width", "height", "length",
    "fire_rating", "industry_standards", "applicable_in",
]

# Any key starting with one of these prefixes is kept — these are the
# per-material functional-equivalence specs used to define comparable pairs.
SPEC_PREFIXES = (
    "concrete_", "cement_", "aci_", "csa_", "en_", "astm_", "c1157", "en197_1",
    "steel_", "gypsum_", "insulation_", "aggregate_", "cmu_", "brick_",
    "timber_", "wood_", "precast_element_type_", "alloy", "slag",
)


def session():
    key = os.environ.get("EC3_API_KEY")
    if not key:
        sys.exit("EC3_API_KEY not set (source .env first)")
    s = requests.Session()
    s.headers.update(
        {
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (academic research; embodied-carbon-uncertainty-study)",
        }
    )
    return s


def reduce_record(m):
    """Whitelist-reduce one material record."""
    out = {k: m.get(k) for k in SCALARS if m.get(k) is not None}
    for k, v in m.items():
        if v is not None and k.startswith(SPEC_PREFIXES):
            out[k] = v

    def d(x):
        """Nested API fields are inconsistently dict / list / scalar."""
        if isinstance(x, dict):
            return x
        if isinstance(x, list) and x and isinstance(x[0], dict):
            return x[0]
        return {}

    cat = d(m.get("category"))
    out["category_id"] = cat.get("id")
    out["category_name"] = cat.get("name")
    out["category_display"] = cat.get("display_name")

    mfr = d(m.get("manufacturer"))
    out["mfr_name"] = mfr.get("name")
    out["mfr_address"] = mfr.get("address")
    out["mfr_lat"] = mfr.get("latitude")
    out["mfr_lon"] = mfr.get("longitude")
    loc = d(mfr.get("location"))
    out["mfr_country"] = loc.get("country") or loc.get("countryCode")
    out["mfr_region"] = loc.get("region") or loc.get("state")

    plant = d(m.get("plant_or_group"))
    out["plant_name"] = plant.get("name")
    out["plant_address"] = plant.get("address")
    out["plant_lat"] = plant.get("latitude")
    out["plant_lon"] = plant.get("longitude")
    ploc = d(plant.get("location"))
    out["plant_country"] = ploc.get("country") or ploc.get("countryCode")

    pc = d(m.get("product_classes"))
    out["product_class_ec3"] = pc.get("EC3")
    return out


def fetch(s, slug, cat_id, page_size=250, max_pages=2000, sleep=0.25):
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{slug}.jsonl"
    state = OUT / f"{slug}.state.json"

    start_page = 1
    mode = "w"
    if state.exists() and dest.exists():
        start_page = json.loads(state.read_text()).get("next_page", 1)
        mode = "a"
        print(f"  [{slug}] resuming at page {start_page}", flush=True)

    n = sum(1 for _ in dest.open()) if mode == "a" else 0
    total = None
    with dest.open(mode) as fh:
        for page in range(start_page, max_pages + 1):
            params = {"page_size": page_size, "page_number": page, "category": cat_id}
            r = None
            for attempt in range(5):
                try:
                    r = s.get(f"{BASE}/materials", params=params, timeout=180)
                    if r.status_code == 429:
                        time.sleep(5 * (attempt + 1))
                        r = None
                        continue
                    r.raise_for_status()
                    break
                except Exception as exc:  # noqa: BLE001
                    r = None
                    if attempt == 4:
                        print(f"  [{slug}] page {page} failed: {exc}", flush=True)
                    else:
                        time.sleep(3 * (attempt + 1))
            if r is None:
                break

            if total is None:
                total = r.headers.get("X-Total-Count")
                print(f"  [{slug}] total={total}", flush=True)

            batch = r.json()
            if not batch:
                break
            for m in batch:
                fh.write(json.dumps(reduce_record(m)) + "\n")
            fh.flush()
            n += len(batch)
            state.write_text(json.dumps({"next_page": page + 1, "n": n}))

            if page % 20 == 0 or page == start_page:
                print(f"  [{slug}] page {page} -> {n:,} rows", flush=True)
            if total is not None and n >= int(total):
                break
            time.sleep(sleep)
    print(f"  [{slug}] DONE {n:,} rows", flush=True)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="subset of category slugs")
    ap.add_argument("--page-size", type=int, default=250)
    ap.add_argument("--max-pages", type=int, default=2000)
    args = ap.parse_args()

    cats = CATEGORIES
    if args.only:
        cats = {k: v for k, v in CATEGORIES.items() if k in args.only}

    s = session()
    summary = {}
    for slug, cid in cats.items():
        print(f"=== {slug} ===", flush=True)
        summary[slug] = fetch(s, slug, cid, args.page_size, args.max_pages)
    print("\n=== SUMMARY ===")
    for k, v in sorted(summary.items(), key=lambda x: -x[1]):
        print(f"{v:>9,}  {k}")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
