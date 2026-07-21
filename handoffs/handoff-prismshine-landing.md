# Handoff — PrismShine product landing page (insightits.com)

**To:** Website / InsightitsAIAgent agent  
**From:** PrismShine maintainers  
**Date:** 2026-07-20  
**Priority:** High — product live on PyPI as [`prismshine==0.2.2`](https://pypi.org/project/prismshine/0.2.2/)  
**Repo to edit:** `C:\code\InsightitsAIAgent` (same pattern as ChorusGraph + PrismGuard product pages)  
**Canonical URL (target):** `https://www.insightits.com/products/prismshine.html`

---

## Goal

Ship a **company-site product landing page** for PrismShine that:

1. Explains clearly **what it is** (anti-hallucination verdict engine — not a firewall, not a runtime).  
2. Uses the same commercial shape as **ChorusGraph**: **Open Source free** + **Enterprise / pilot** upsell.  
3. Matches existing Insight product-page plumbing (SEO shell HTML + React landing + catalog + pricing cards).  
4. Only cites **PrismShine** public receipts (vs HHEM) — do **not** put Guard/ChorusGraph stack-suite S1/P1 numbers on this page.

**Reference pages (copy structure, not claims):**

| Reference | URL / path |
|-----------|------------|
| Live ChorusGraph product page | https://www.insightits.com/products/chorusgraph.html |
| ChorusGraph static marketing (hero / pain / proof / plan) | `C:\code\ChorusGraph\website\index.html` |
| ChorusGraph pricing 3-tier | `C:\code\ChorusGraph\website\pricing.html` |
| PrismGuard product page (OSS + Team/Business/Pilot estimates) | https://www.insightits.com/products/prismguard.html · `InsightitsAIAgent/products/prismguard.html` |
| Catalog + URLs pattern | `InsightitsAIAgent/src/pages/products/productCatalog.js` |
| Pricing cards | `InsightitsAIAgent/src/pages/products/productPricingPlans.js` |

---

## What PrismShine is (copy authority)

**One-liner:** Self-hosted anti-hallucination verdict engine — cause-side forensics + effect-side grounding in one auditable `ShineVerdict`.

**Not:**

- Not a prompt-injection firewall → that’s [PrismGuard](https://www.insightits.com/products/prismguard.html).  
- Not an agent runtime → that’s [ChorusGraph](https://www.insightits.com/products/chorusgraph.html).  
- Not “PASS = world-true” → PASS means grounded in the **provided preload** ([LIMITS](https://github.com/insightitsGit/PrismShine/blob/main/docs/LIMITS.md)).

**Is:**

| Side | When | Value |
|------|------|--------|
| Cause (Tier-0) | Before/after gen | Empty retrieval, tool errors, stale cache, missing ledger hops — can **halt before tokens** |
| Effect (Tiers 1–4) | After answer | Fabricated numbers/entities, coverage collapse, optional LLM judge on gray zone only |
| Audit | Always | Named `resolution_gate` + `evidence_hash` |

**Install:**

```bash
pip install "prismshine==0.2.2"
prismshine capabilities
```

**Links (must appear on page):**

| Resource | URL |
|----------|-----|
| PyPI | https://pypi.org/project/prismshine/0.2.2/ |
| GitHub | https://github.com/insightitsGit/PrismShine |
| README | https://github.com/insightitsGit/PrismShine/blob/main/README.md |
| Benchmarks doc | https://github.com/insightitsGit/PrismShine/blob/main/docs/BENCHMARKS.md |
| Receipt (HHEM) | https://github.com/insightitsGit/PrismShine/tree/main/benchmarks/progress/2026-07-20_run4_onnx |
| Integration | https://github.com/insightitsGit/PrismShine/blob/main/docs/INTEGRATION.md |
| Interactive demo | https://insightitsgit.github.io/PrismShine/demo.html |
| Contact | https://www.insightits.com/#contact · `info@insightits.com` |

---

## Page structure (mirror ChorusGraph)

Follow Hormozi / ChorusGraph sections. **First viewport:** brand **PrismShine** hero-level + one headline + one supporting sentence + CTA group + one dominant visual (product/diagram — not a card collage). Match Insight design system already used by `chorusgraph.html` / `prismguard.html` (existing CSS/JS bundles). Avoid purple-on-white / cream-serif / broadsheet clichés per company frontend rules.

### 1. Hero

- **Brand:** PrismShine (large).  
- **Headline (pick one):**  
  - “Catch hallucinations before they ship — and prove why.”  
  - “Ground every answer. Halt broken preloads before tokens burn.”  
- **Sub:** Cause-side forensics + effect-side grounding in one auditable gate. Apache-2.0. Zero LLM calls on the default path.  
- **Proof strip (verified only):**  
  - B1 QA F1 **0.831** vs HHEM **0.746**  
  - B2 numbers F1 **1.000 / 0 FP**  
  - B1 p50 **90 ms** vs HHEM **216 ms**  
  - `pip install "prismshine==0.2.2"`  
- **CTAs:**  
  1. Primary: Try interactive demo → https://insightitsgit.github.io/PrismShine/demo.html  
  2. Secondary: `pip` / GitHub README  
  3. Secondary: Book 30-min **Verifier Stack Audit** → `#contact`  
  4. Secondary: Benchmarks receipt  

### 2. The pain (3 cards)

| Pain | Copy |
|------|------|
| Fluent lies | RAG answers look correct but invent numbers and entities. |
| Blind checkers | Encoder tools never see empty retrieval, tool failures, or stale cache. |
| Expensive judges | LLM-as-judge burns tokens on every check; you still lack an audit gate. |

### 3. What you get (features)

Use a compact feature grid (from README): unified gate, handbook Tier-0, pre-gen halt, Tier-1 copy-check, Tier-2 coverage, optional Tier-3 ONNX, optional Tier-4 judge, named `resolution_gate`, profiles (clinical/finance/legal), BYO runtime wiring.

### 4. Proof (benchmarks — Shine only)

Table (cite receipt folder; date 2026-07-20, Azure ACI, ONNX Tier-3):

| System | B1 QA F1 | B2 numbers | Bsum F1 | B1 p50 | LLM calls |
|--------|----------|------------|---------|--------|-----------|
| **prismshine-fast** | **0.831** | **1.000** (0 FP) | **0.600** | **90 ms** | **0** |
| HHEM-2.1-Open | 0.746 | 0.926 | 0.474 | 216 ms | 0 |

Footnote: Absolute Bsum is still modest; do not claim “beats all judges.” No RAGAS row yet.

### 5. Plan — three steps

1. **Install** — `pip install "prismshine==0.2.2"`  
2. **Verify** — `ShineGate.verify(EvidenceBundle(...))` or `prismshine verify bundle.json`  
3. **Wire the moat** — `pre_llm_check` / `shine_verify_node` (or ChorusGraph `require_shine`) so Tier-0 can halt before generation  

Link: Integration.md + `examples/enterprise_wiring_demo.py`.

### 6. Pricing (ChorusGraph-shaped)

**In-page section** `#ps-pricing` (same pattern as ChorusGraph `pricing.html` + Guard’s OSS/Team/Business/Pilot). Label commercial numbers **pre-validation estimates** unless finance confirms.

| Tier | Price | Includes |
|------|-------|----------|
| **Open Source** | **Free** (Apache-2.0) | Full library on PyPI; Tiers 0–2; optional Tier-3 ONNX download; CLI; BYO / LangGraph / ChorusGraph wiring; community GitHub issues |
| **Pro / Team** | **Contact** · estimate **~$199/mo** | Email support, calibration guidance, priority triage, help choosing ONNX / overlays |
| **Enterprise** | **Contact** · Production Pilot estimate **~$15k–$25k** one-time | Staging wiring into customer runtime, custom handbook signatures, traffic calibration, decision-log design, 90-day Slack + SLA; optional recurring Business ~**$699/mo** for SLA + multi-tenant audit packaging |

**Secondary offer (ChorusGraph-style):**

- Free **30-min Verifier Stack Audit** (fit call)  
- Then **Production Verifier Pilot** (suggested **$8k–$15k** staging wire-up *or* the fuller **$15k–$25k** package above — pick one primary number for the CTA mailto and keep the other as “custom scope”)

**Recommended CTA mailto subjects:**

```
mailto:info@insightits.com?subject=PrismShine%20Verifier%20Stack%20Audit
mailto:info@insightits.com?subject=PrismShine%20Production%20Pilot
mailto:info@insightits.com?subject=PrismShine%20Team%20(%24199%2Fmo)
mailto:info@insightits.com?subject=PrismShine%20Enterprise
```

**Schema.org `Offer`:** OSS card `price: "0"` (same as ChorusGraph/Guard). Do **not** put estimate monthly prices into JSON-LD as firm offers — describe them as contact / estimate in body copy only.

### 7. FAQ (required for FAQPage JSON-LD)

Suggested Q&As:

1. **Is PrismShine a firewall?** No — use PrismGuard for injection; Shine verifies answers.  
2. **Do I need ChorusGraph?** No — core works standalone; richest wiring is optional `[chorusgraph]`.  
3. **Does PASS mean the answer is true?** No — grounded in preload only.  
4. **Are there LLM costs?** Default path 0 LLM calls; Tier-4 judge is opt-in.  
5. **Is Tier-3 ONNX in the wheel?** No (~1 GB) — `python -m prismshine.tools.ensure_span_onnx --export`.  
6. **What’s free vs paid?** Library is free Apache-2.0; Pro/Enterprise are support, wiring, and compliance packaging.  
7. **How do I reproduce the HHEM numbers?** See `benchmarks/progress/2026-07-20_run4_onnx` and `docs/BENCHMARKS.md`.

### 8. Ecosystem strip (optional, short)

One line: Complements **PrismGuard** (input) + **ChorusGraph** (runtime) + **PrismCortex** (memory). Link those product pages. Do not imply Shine requires them.

---

## Implementation checklist (InsightitsAIAgent)

Mirror how `chorusgraph` / `prismguard` were added:

1. **Catalog** — `src/pages/products/productCatalog.js`  
   - Add `PRISMSHINE_EXTERNAL` URLs (pypi pin 0.2.2, github, benchmarks, audit CTA).  
   - Add `PRODUCT_CATALOG.prismshine` with `isPlatformProduct: true`, shortName, description, installHint, hero image path (create logo assets under `images/NewDesign/` or reuse a Prism-family mark until brand art exists).  

2. **Pricing plans** — `productPricingPlans.js`  
   - Special-case `prismshine` like platform products: **Open Source / Pro / Enterprise** cards (not vertical AI-web deploy cards).  

3. **Marketing copy** — `productMarketingContent.js` (or platform equivalent)  
   - Pain / dream / plan for `prismshine` slug.  

4. **Nav / SEO** — `navProductGroups.js`, `productSeo.js`, sitemap, `llms.txt` / `ai-info.txt` entries.  

5. **Static shell** — `products/prismshine.html`  
   - Same PRODUCT_SEO_START block pattern as `chorusgraph.html` / `prismguard.html` (title, description, OG, JSON-LD Product + SoftwareApplication + FAQPage + BreadcrumbList).  
   - Canonical: `https://www.insightits.com/products/prismshine.html`.  

6. **Home / products index** — list PrismShine under platform products next to ChorusGraph / PrismGuard.  

7. **Deploy** — ship with normal InsightitsAIAgent deploy path (confirm with site owner).

---

## Copy constraints (do / don’t)

**Do**

- Lead with OSS free + pip.  
- Cite only Shine vs HHEM receipt above.  
- Say “pre-validation estimate” on Team/Business/Pilot dollars.  
- Link LIMITS for PASS≠truth.  

**Don’t**

- Claim beats RAGAS / Blue Guardrails / “highest accuracy.”  
- Put stack-suite Guard latency or S1 F1 on this page.  
- Say healthcare/finance “certified” or HIPAA/SOC2 complete.  
- Require license key for basic `pip install prismshine`.  
- Fake a live interactive demo URL until one exists (optional later: GitHub Pages demo like Guard/ChorusGraph).

---

## Suggested hero SEO

- **Title:** `PrismShine — Anti-Hallucination Verdict Engine [Free Apache-2.0]`  
- **Meta description:** `Self-hosted grounding + cause-side forensics. Beats HHEM on HaluEval QA (F1 0.831 vs 0.746) at half the latency. pip install prismshine==0.2.2.`  
- **Keywords:** hallucination detection, RAG grounding, anti-hallucination Python, LettuceDetect alternative, HHEM alternative, agent answer verification, resolution_gate, PrismShine, Insight IT Solutions  

---

## Acceptance criteria

- [ ] Live at `/products/prismshine.html` with working nav from homepage platform products.  
- [ ] Hero shows brand + pip CTA + HHEM proof strip (correct numbers).  
- [ ] Pricing section: **Open Source Free** + Pro/Enterprise contact (ChorusGraph shape).  
- [ ] FAQ + JSON-LD present; OSS Offer price `0`.  
- [ ] No stack-suite / Guard-latency claims.  
- [ ] `llms.txt` / `ai-info.txt` mention PrismShine with PyPI + landing URL.  
- [ ] Mobile + desktop readable in existing Insight layout.  

---

## Out of scope for this handoff

- Building a browser interactive demo (follow-up).  
- Changing PyPI package contents.  
- Publishing firm price list without finance sign-off (estimates only).  

**Update (2026-07-20):** Interactive GitHub Pages demo shipped —  
https://insightitsgit.github.io/PrismShine/demo.html (source: `docs/demo.html`). Wire this URL into the landing hero CTA and catalog `interactiveDemo` field.

---

## Contact for product facts

PrismShine repo README + `docs/POSITIONING.md` + `docs/BENCHMARKS.md`. Company contact: `info@insightits.com` · site https://www.insightits.com.
