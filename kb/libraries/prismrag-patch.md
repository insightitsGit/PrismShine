# PrismRAG (prismrag-patch)

> Taxonomy-grounded retrieval patch: deterministic Tier-1 category remapping of embeddings before upsert/query against any vector DB. Commercial.

| Field | Value |
|---|---|
| PyPI | `prismrag-patch` (import as `prismrag_patch`) |
| Version | 1.0.0 local (published PyPI 0.2.1 was historically described as OSS; this local tree is the commercial cut) |
| License | Commercial / proprietary ("Commercial — see LICENSE"; no LICENSE/README file actually present locally) |
| Python | >= 3.10 |
| Local path | `C:\code\PrismRagLib` (no git remote configured locally) |
| GitHub | https://github.com/insightitsGit/prismrag |
| Install | `pip install "prismrag-patch[pgvector]"` |

## Purpose

Makes retrieval hallucination-resistant by applying a deterministic category remap to embeddings before they hit the vector DB. You supply a mapping (categories + word→category rules, typically derived from your existing relational schema / taxonomy). Keyword rules assign a category; the vector is blended toward a category direction and L2-normalized. No LLM calls at remap time — fully deterministic, fully auditable (every chunk traces to the rule that placed it).

**When NOT to use:** if you have no pre-existing taxonomy/relational mapping, plain RAG is simpler.

## Architecture

| Module | Role |
|---|---|
| `prismrag_patch.core` | `PrismRAGPatch` — license gate + remap entrypoint |
| `prismrag_patch.mapping` | `Mapping` / `Category` / `Rule` + `remap_vector()` |
| `prismrag_patch.license` | Online license validation + local cache |
| `prismrag_patch.adapters.pgvector` | `PgvectorAdapter` |
| `prismrag_patch.adapters.chroma` | `ChromaAdapter` |
| `prismrag_patch.adapters.pinecone` | `PineconeAdapter` |
| `prismrag_patch.adapters.weaviate` | `WeaviateAdapter` |

## Public API

```python
from prismrag_patch import PrismRAGPatch, Mapping, remap_vector, LicenseError, LicenseExpiredError

PrismRAGPatch(license_key: str, mapping: dict | Mapping, alpha: float = 0.25, adapter: str = "unknown")
patch.remap(text, vector) -> list[float]
patch.remap_with_category(text, vector, category_slug=None) -> list[float]
patch.license_info -> dict
patch.mapping -> Mapping

Mapping.from_dict(d) -> Mapping
Mapping.assign_category(text) -> str | None
remap_vector(vector, category_slug, mapping, alpha=0.25) -> list[float]

# Adapters
PgvectorAdapter(patch, connection=None, dsn=None, table="embeddings", ...)
    .insert(text, vector, ...) / .insert_many / .search(query_text, query_vector, top_k=10, ...)
ChromaAdapter(patch, collection).add / .upsert / .query
PineconeAdapter(patch, index).upsert / .query
WeaviateAdapter(patch, client, class_name, text_property="content").add / .query
```

## Core algorithm

1. **Category assignment**: case-insensitive substring keyword hits; slug with max hit count wins.
2. **Category direction**: tiled one-hot over `n_cats` across embedding dim, L2-normalized.
3. **Remap blend**: `remapped = (1−α)·v + α·cat_dir`, then L2-normalize (default α=0.25). Deterministic — no randomness, no LLM.

## Dependencies

- Required: **none** (`dependencies = []`)
- Extras: `psycopg2-binary`, `chromadb`, `pinecone-client`, `weaviate-client`
- No sibling Insight lib deps (standalone)

## Licensing / config

- Keys must start with `prlib_`; validated online against `https://prismrag.insightits.com/api/v1/lib/validate` (override with `PRISMRAG_LICENSE_SERVER`)
- Cache ~23h in temp dir `.prlib_<hash>.json`; 7-day grace after expiry; stale cache tolerated on network failure after first success
- Known gap: no air-gapped/offline license path yet

## Usage example

```python
from prismrag_patch import PrismRAGPatch
from prismrag_patch.adapters.pgvector import PgvectorAdapter

patch = PrismRAGPatch(
    license_key="prlib_your_key_here",
    mapping={
        "categories": [
            {"slug": "symptoms", "label": "Symptoms & Signs"},
            {"slug": "treatment", "label": "Treatment & Therapy"},
        ],
        "rules": [
            {"word": "fever", "category_slug": "symptoms"},
            {"word": "antibiotics", "category_slug": "treatment"},
        ],
    },
)
adapter = PgvectorAdapter(patch, dsn="postgresql://user:pass@host/db")
```

## Tests / benchmarks

- 8 unit tests in `tests/test_core.py` (mapping/remap only); no license tests, no benchmarks

## Gotchas

- Local tree is missing `README.md` and `LICENSE` despite pyproject references.
- Version/naming drift between published PyPI 0.2.1 and this local 1.0.0 commercial cut (documented in `handoffs/handoff1_licensing.md`).
- Call-home licensing — relevant for offline/air-gap deployments.
- PrismLib and prismlib-plus each contain their own in-tree `PrismRAGPatch`-style bridge under `prism.bridge` — related but separate code.
