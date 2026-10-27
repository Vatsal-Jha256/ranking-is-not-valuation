# Frozen data-query provenance

This record separates author-generated snapshot counts from facts printed on provider
landing pages. Its content hash is included in `input_manifest.json`.

## EC3/openEPD ready-mix snapshot

- Provider/interface: Building Transparency EC3 materials API.
- Retrieval completed: 2026-07-29.
- Local raw artifact: `data/raw/ec3_materials/readymix.jsonl`.
- Page-state artifact: `data/raw/ec3_materials/readymix.state.json` (`next_page=527`,
  `n=131361`).
- Raw rows: 131,361, one JSON object per line.
- Raw SHA-256: `44918250c5c9a3e52866064f3ea2cabe5d0ba7e839f4ec5240c88998685ac296`.
- Analysis category: ready-mix concrete. Filtering is fully itemized in Appendix B of
  the paper and implemented by `rnv/registry/normalize.py`.

The provider documentation supports access through the interface; the row count is an
author-generated property of this frozen extraction.

## International EPD System library enumeration

- Query date: 2026-08-15.
- Filter semantics: `c` is registering-organisation country; `q` is free-text search.
- India, all products: `https://www.environdec.com/library?c=IN` (425 results).
- India, concrete: `https://www.environdec.com/library?c=IN&q=concrete` (6 results).
- India, cement: `https://www.environdec.com/library?c=IN&q=cement` (20 results).
- The cement/concrete union contains 25 unique hits: 17 cement or concrete products
  after manual title screening, including 5 ready-mix declarations.
- Comparison queries, all products: Italy 3,648; Sweden 2,387; United States 969;
  Nigeria 0; Kenya 0.

The five ready-mix accession IDs and titles are:

| Accession | Title |
|---|---|
| `epd6471` | 1 m3 of Ready-Mix Concrete |
| `epd23969` | Ready Mix Concrete (M10, M15, M20 & M25) |
| `epd23970` | Ready Mix Concrete (M30, M35, M40 & M45) |
| `epd23971` | Ready Mix Concrete (M50, M55, M60 & M70) |
| `epd23972` | Ready Mix Concrete (M75 & M80) |

Counts were obtained by walking the public library pagination to its final page and
deduplicating the union by accession ID. They describe one programme operator on one
date, not the complete Indian market, and may change as declarations are added.
