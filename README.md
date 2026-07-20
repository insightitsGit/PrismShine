# PrismShine

[![PyPI](https://img.shields.io/pypi/v/prismshine.svg)](https://pypi.org/project/prismshine/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.2.0-informational)](https://pypi.org/project/prismshine/)

**Anti-hallucination verdict engine — cause-side forensics + effect-side grounding in one auditable gate.**

```bash
pip install "prismshine==0.2.0"
prismshine capabilities
```

**Interactive demo:** [insightitsGit.github.io/PrismShine/demo.html](https://insightitsgit.github.io/PrismShine/demo.html) — terminal walkthrough (grounded pass → fabricated number block → empty-retrieval halt). No API key.

> **PASS ≠ world-true.** A `pass` means the answer is grounded in the **preload you provided**, not that the preload is factually correct. See [`docs/LIMITS.md`](docs/LIMITS.md).

---

## What is PrismShine?

PrismShine is a **self-hosted verifier** for agent / RAG answers. It catches hallucinations from both ends:

| Side | When | What it catches |
|------|------|-----------------|
| **Cause (Tier-0)** | Before or after generation | Empty retrieval, swallowed tool errors, truncated context, stale cache reuse, missing ledger hops, memory conflicts |
| **Effect (Tiers 1–4)** | After the answer exists | Fabricated numbers/entities, coverage collapse, unsupported spans, residual gray-zone (optional LLM judge) |

Both fuse into one **`ShineVerdict`**: `decision` + named `resolution_gate` + `evidence_hash` + signatures / spans — replayable and audit-ready.

**PrismShine is not a prompt-injection firewall** (that’s [PrismGuard](https://pypi.org/project/prismguard/)). It is not an agent runtime (that’s [ChorusGraph](https://pypi.org/project/chorusgraph/) or your own graph). It verifies **answers against evidence**.

### When NOT to use PrismShine

- You only need input jailbreak / injection filtering → use PrismGuard.  
- You need mid-stream token verification → not in v0 (buffered answers only).  
- You expect “PASS” to mean world knowledge is correct → it means *grounded in preload*.

---

## Why PrismShine?

| Pain | PrismShine answer |
|------|-------------------|
| Encoders only see the final text | Tier-0 handbook reads **runtime evidence** (trace / ledger / node state) |
| Broken retrieval still burns LLM tokens | `pre_llm_check` / interceptors can **halt before generation** |
| Numbers look fluent but are wrong | Tier-1 copy-check (exact figures / entities) — B2 F1 **1.0 / 0 FP** vs HHEM |
| Judge APIs are expensive | Default path is **0 LLM calls**; Tier-4 is opt-in gray-zone only |
| Stale cache after a fact correction | Consistency hooks + `CACHE_PREDATES_FACT_UPDATE` detection |
| “Why did we allow this?” | Named `resolution_gate` + evidence hash on every verdict |
| Vendor lock-in to one runtime | Works standalone, LangGraph, or ChorusGraph via the same wiring API |

---

## Quick start (30 seconds)

```bash
pip install prismshine
```

```python
from prismshine import EvidenceBundle, PreloadChunk, ShineGate

gate = ShineGate.build(profile="default")
bundle = EvidenceBundle(
    run_id="demo",
    question="What was revenue?",
    answer="Revenue was $1000 in Q1.",
    preload=[
        PreloadChunk(
            chunk_id="c1",
            text="Revenue was $1000 in Q1.",
            source="retrieval",
        )
    ],
)
verdict = gate.verify(bundle)
print(verdict.decision, verdict.resolution_gate, verdict.evidence_hash)
print(gate.capabilities())
```

```bash
prismshine capabilities
prismshine verify path/to/bundle.json --profile default
prismshine bench --suite all --report benchmarks/reports
```

**Runnable demos**

```bash
python examples/enterprise_wiring_demo.py   # pre-gen halt + grounding + consistency
python examples/tier4_judge_demo.py         # optional OpenAI judge (needs key + extra)
```

---

## How to implement (choose your path)

### 1) Standalone dict verify (any app)

Best for scripts, batch eval, or a service that already has `question` / `answer` / `context`.

```python
from prismshine import EvidenceBundle, PreloadChunk, ShineGate, TraceStep

gate = ShineGate.build(profile="finance")
bundle = EvidenceBundle(
    run_id="req-1",
    question="What was Q2 revenue?",
    answer="Q2 revenue was $1,200,000.",
    preload=[PreloadChunk(chunk_id="c0", text="Q2 revenue was $1,200,000.", source="retrieval")],
    trace=[TraceStep(hop="retrieve", kind="retrieval", status="ok", detail={"n_chunks": 1})],
)
verdict = gate.verify(bundle)
if verdict.decision == "block":
    ...  # refuse or regenerate
```

### 2) BYO runtime wiring (LangGraph / custom) — recommended moat path

Same capabilities as ChorusGraph plugins, without requiring ChorusGraph:

```python
from prismshine import ShineGate, wrap_llm, shine_verify_node, require_shine_wiring
from prismshine.wiring import pre_llm_check, record_retrieval, append_trace

gate = ShineGate.build(profile="default")

def retrieve(state: dict) -> dict:
    docs = my_retriever(state["question"])
    state = append_trace(state, record_retrieval("retrieve", n_chunks=len(docs)))
    return {**state, "docs": docs}

# Halt before tokens when preload is broken
decision = pre_llm_check(gate, state)
if decision.should_halt:
    return decision.fallback

# Or wrap the provider boundary
llm = wrap_llm(my_model, gate, state_factory=lambda: current_state)

# Guaranteed post-gen path
graph.add_node("shine", shine_verify_node(gate, answer_key="answer"))
require_shine_wiring(compiled, gate, already_has_shine_node=True)
```

Full matrix: [`docs/INTEGRATION.md`](docs/INTEGRATION.md) §8 · demo: [`examples/enterprise_wiring_demo.py`](examples/enterprise_wiring_demo.py).

### 3) ChorusGraph plugin (richest out of the box)

```bash
pip install "prismshine[chorusgraph]"
```

```python
from prismshine import ShineGate
from prismshine.integrations.chorusgraph import require_shine, shine_node

gate = ShineGate.build(profile="default")
g.add_node("shine", shine_node(gate, answer_key="reply"))
g.add_edge("generate", "shine")
compiled = g.compile(stack=stack)
require_shine(compiled, gate, prefer="both", already_has_shine_node=True)
```

Uses Route Ledger steps, warm chunk vectors when present, and ADR-008 interceptors (`before_llm` / `after_llm`). Details: [`docs/INTEGRATION.md`](docs/INTEGRATION.md) §1.

### 4) LangGraph plugin

```bash
pip install "prismshine[langgraph]"
```

```python
from prismshine.integrations.langgraph import require_shine, shine_langgraph_node

gate = ShineGate.build(profile="default")
graph.add_node("shine", shine_langgraph_node(gate, answer_key="answer"))
require_shine(compiled, gate, already_has_shine_node=True)
```

---

## Features

| Feature | Description |
|---------|-------------|
| **Unified gate** | One `ShineGate.verify` — forensics + grounding fused |
| **Handbook Tier-0** | Versioned YAML signatures (`EMPTY_RETRIEVAL`, cache/tool/LLM failures, …) |
| **Pre-generation halt** | `pre_llm_check` / interceptors stop broken preloads before tokens |
| **Tier-1 copy-check** | Numbers, entities, dates — hard fabricated-figure floor |
| **Tier-2 coverage** | Vector support using runtime embeddings when available; hash fallback |
| **Tier-3 spans** | Optional LettuceDetect-class ONNX (not in the wheel — download once) |
| **Tier-4 judge** | Opt-in OpenAI / Gemini on residual gray zone only |
| **Named audit gate** | Every verdict names `resolution_gate` + `evidence_hash` |
| **Profiles** | `default` / `clinical` / `finance` / `legal` handbook packs |
| **Calibration** | `prismshine calibrate` + feedback JSONL overlays |
| **Consistency contract** | Invalidation hooks + stale-cache detection dual-rail |
| **Zero hard siblings** | Core `pip install prismshine` has no Insight package dependency |

---

## Architecture

```
                    ┌──────────────────────────────────────┐
  question ────────►│  EvidenceBundle                       │
  answer ──────────►│  preload[] · trace[] · node_state     │
  runtime evidence─►└──────────────────┬───────────────────┘
                                       ▼
                              ShineGate.verify()
                                       │
              ┌────────────────────────┼────────────────────────┐
              ▼                        ▼                        ▼
         Tier-0 handbook          Tiers 1–3                 Tier-4 (opt)
         (cause / halt)           copy · cover · spans      LLM judge
              └────────────────────────┼────────────────────────┘
                                       ▼
                         ShineVerdict (pass|flag|block|regenerate)
                         resolution_gate · signatures · spans · hash
```

Design deep-dive: [`docs/DESIGN.md`](docs/DESIGN.md) · diagram: [`docs/prismshine-architecture.png`](docs/prismshine-architecture.png).

---

## Install & extras

```bash
pip install prismshine                 # core — CPU, zero LLM
pip install "prismshine[spans]"        # Tier-3 ONNX runtime + tokenizers
pip install "prismshine[coverage]"     # shared prismlang encoder session
pip install "prismshine[chorusgraph]"  # ChorusGraph plugins
pip install "prismshine[langgraph]"    # LangGraph plugins
pip install "prismshine[guard]"        # PrismGuard symmetry helpers
pip install "prismshine[judge-openai]" # Tier-4 OpenAI
pip install "prismshine[judge-gemini]" # Tier-4 Gemini
pip install "prismshine[dev]"          # pytest + ruff
```

### Production checklist

1. **Tier-3 ONNX** (~1 GB, not in the wheel):

   ```bash
   pip install "prismshine[spans]"
   python -m prismshine.tools.ensure_span_onnx --export
   # then pin:
   #   PRISMSHINE_SPAN_ONNX=.../model.onnx
   #   PRISMSHINE_SPAN_TOKENIZER=.../tokenizer.json
   ```

2. **Domain calibration** (marked row, not the uncalibrated headline):

   ```bash
   python -m prismshine.bench.calibrate_minilm --embedder hash   # CI-safe
   # or --embedder minilm for stronger overlays
   set PRISMSHINE_CALIBRATION=path\to\overlay.json
   ```

3. **Wire the moat** — `examples/enterprise_wiring_demo.py` then [`docs/INTEGRATION.md`](docs/INTEGRATION.md).

4. **Receipts before claims** — `prismshine bench --suite all` and comparative vs HHEM ([`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)).

`gate.capabilities()` reports `span_backend` (`onnx`|`lexical`|`unavailable`), `threshold_status`, and `pass_means`. Without ONNX the gate stays honest (`lexical`) — it never fakes span SotA.

---

## CLI

```bash
prismshine capabilities [--profile default|clinical|finance|legal]
prismshine verify bundle.json --profile default
prismshine feedback bundle.json --label hallucination --out feedback.jsonl
prismshine calibrate ./samples --mode synthetic --profile clinical --out cal.yaml
prismshine bench --suite all|cause|grounding|latency|consistency --report benchmarks/reports
```

---

## Benchmarks (PrismShine only)

Public claims use **PrismShine vs encoder/judge competitors** and Shine-only suites — not sibling package stacks.

### Headline comparative receipt (2026-07-20, Azure ACI, ONNX Tier-3)

Vs Vectara **HHEM-2.1-Open** — HaluEval QA / numbers / summarization. Receipt: [`benchmarks/progress/2026-07-20_run4_onnx/`](benchmarks/progress/2026-07-20_run4_onnx/README.md).

| system | B1 QA F1 | B2 numbers F1 | Bsum F1 | B1 p50 | LLM calls |
|--------|----------|---------------|---------|--------|-----------|
| **prismshine-fast** | **0.831** | **1.000** (0 FP) | **0.600** | **90 ms** | **0** |
| hhem-2.1-open | 0.746 | 0.926 | 0.474 | 216 ms | 0 |

### In-process gates (`prismshine bench`)

| Suite | Proves |
|-------|--------|
| `cause` | Tier-0 catch ≥90% on injected runtime failures; pre-gen 0 model calls |
| `grounding` | Hard synthetic + optional RAGTruth |
| `latency` | p50/p95 + judge escalation |
| `consistency` | Stale-cache dual-rail |

Methodology & fairness: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) · market stance: [`docs/POSITIONING.md`](docs/POSITIONING.md).

---

## Profiles & handbook

```python
gate = ShineGate.build(profile="clinical")  # or finance / legal / default
```

Builtin packs live under `prismshine/handbook/builtin/`. Thresholds stay `proposal` until you run `prismshine calibrate` (or feedback → calibrate). Catalog: [`docs/HANDBOOK.md`](docs/HANDBOOK.md).

---

## Documentation

| Doc | Description |
|-----|-------------|
| [`docs/LIMITS.md`](docs/LIMITS.md) | PASS≠truth, streaming, moat wiring boundaries |
| [`docs/DESIGN.md`](docs/DESIGN.md) | Architecture, scoring, module layout |
| [`docs/INTEGRATION.md`](docs/INTEGRATION.md) | ChorusGraph · LangGraph · Guard · Cortex · BYO |
| [`docs/HANDBOOK.md`](docs/HANDBOOK.md) | Failure-signature taxonomy |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | Receipts before claims |
| [`docs/POSITIONING.md`](docs/POSITIONING.md) | Market comparison & gates |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | ADRs |
| [`docs/UPSTREAM.md`](docs/UPSTREAM.md) | Sibling version floors |
| [`CHANGELOG.md`](CHANGELOG.md) | Release notes |

---

## Examples

| Example | What it shows |
|---------|----------------|
| [`examples/enterprise_wiring_demo.py`](examples/enterprise_wiring_demo.py) | Pre-gen halt, grounding pass/fail, fact-correction cache invalidation |
| [`examples/tier4_judge_demo.py`](examples/tier4_judge_demo.py) | Opt-in Tier-4 OpenAI judge |

---

## Development

```bash
git clone https://github.com/insightitsGit/PrismShine.git
cd PrismShine
pip install -e ".[dev,spans]"
pytest
ruff check prismshine tests
prismshine bench --suite all --report benchmarks/reports
```

---

## Publishing (maintainers)

```bash
pip install build twine
python -m build
twine check dist/*
# twine upload dist/*    # after tag v0.2.0 — requires PyPI credentials
```

Git: commit on `main`, tag `v0.2.0`, push tag when ready. **Do not force-push `main`.**

---

## Status

**0.2.0** — enterprise-ready open source for the self-hosted fast verifier lane (HaluEval vs HHEM receipt + FIX hardening). Category-creator / beats-LLM-judge claims still need production wiring receipts and a fair judge comparator row.

License: Apache-2.0 · Author: Insight IT Solutions LLC · PyPI: [prismshine](https://pypi.org/project/prismshine/) · GitHub: [insightitsGit/PrismShine](https://github.com/insightitsGit/PrismShine)
