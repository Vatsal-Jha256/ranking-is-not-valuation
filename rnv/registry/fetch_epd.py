"""Paginated openEPD downloader.

Pulls EPD records for a set of EC3 categories and caches raw JSON to disk.
Resumable: already-downloaded pages are skipped.
"""
import argparse
import json
import os
import pathlib
import sys
import time

import requests

RAW = pathlib.Path(__file__).resolve().parents[2] / "data" / "raw" / "openepd"
BASE = "https://openepd.buildingtransparency.org/api"


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


def fetch_category(s, category, out_dir, page_size=250, max_pages=400, sleep=0.3):
    """Page through /epds filtered to one EC3 category."""
    out_dir.mkdir(parents=True, exist_ok=True)
    total = None
    n = 0
    for page in range(1, max_pages + 1):
        dest = out_dir / f"page_{page:04d}.json"
        if dest.exists():
            n += len(json.loads(dest.read_text()))
            continue

        params = {"page_size": page_size, "page_number": page, "category": category}
        r = None
        for attempt in range(5):
            try:
                r = s.get(f"{BASE}/epds", params=params, timeout=120)
                if r.status_code == 429:
                    time.sleep(5 * (attempt + 1))
                    r = None
                    continue
                r.raise_for_status()
                break
            except Exception as exc:  # noqa: BLE001
                r = None
                if attempt == 4:
                    print(f"  [{category}] page {page} FAILED: {exc}", flush=True)
                    return n
                time.sleep(3 * (attempt + 1))
        if r is None:
            print(f"  [{category}] page {page} gave up", flush=True)
            return n

        if total is None:
            total = r.headers.get("X-Total-Count")
            print(f"  [{category}] x-total-count={total}", flush=True)

        batch = r.json()
        # The API can return slightly fewer rows than page_size on a full page,
        # so only an empty batch is a reliable end-of-results signal.
        if not batch:
            break
        dest.write_text(json.dumps(batch))
        n += len(batch)
        if page % 10 == 0 or page == 1:
            print(f"  [{category}] page {page}: +{len(batch)} (cum {n:,})", flush=True)
        if total is not None and n >= int(total):
            break
        time.sleep(sleep)
    return n


CATEGORIES = [
    "Concrete >> ReadyMix",
    "Masonry",
    "Steel",
    "Wood",
    "Insulation",
    "Gypsum",
    "Aluminium",
    "Glass",
    "Cement",
    "Aggregates",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="*", default=CATEGORIES)
    ap.add_argument("--page-size", type=int, default=250)
    ap.add_argument("--max-pages", type=int, default=400)
    args = ap.parse_args()

    s = session()
    summary = {}
    for cat in args.categories:
        slug = cat.replace(" >> ", "__").replace(" ", "_").lower()
        print(f"=== {cat} ===", flush=True)
        summary[cat] = fetch_category(
            s, cat, RAW / slug, page_size=args.page_size, max_pages=args.max_pages
        )
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{v:>8,}  {k}")
    (RAW / "fetch_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
