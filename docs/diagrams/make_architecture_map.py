"""Render the PrismShine architecture (unified pipeline + plugins + support systems) to PNG."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

STYLE = {
    "tier":    ("#e8f5e9", "#2e7d32"),   # pipeline tiers
    "tier_opt": ("#f1f8e9", "#7cb342"),  # optional tiers
    "data":    ("#e3f2fd", "#1565c0"),   # data objects
    "plugin":  ("#fff3e0", "#ef6c00"),   # runtime plugins
    "support": ("#f3e5f5", "#6a1b9a"),   # support systems
    "verdict": ("#fce4ec", "#c62828"),   # verdict / outcomes
}

# name: (x, y, w, h, category, title, body, fontsize_body)
BOXES = {
    # ---- left column: runtime plugins (evidence in, enforcement out) ----
    "cg":   (0.35, 8.55, 3.30, 1.50, "plugin", "ChorusGraph plugin  (richest)",
             "Route Ledger + node state + cache\ndecisions + warm-index raw vectors\ninterceptor: pre-gen halt"),
    "lg":   (0.35, 6.65, 3.30, 1.30, "plugin", "LangGraph plugin",
             "graph state + retriever results\nnode + conditional-edge router"),
    "gen":  (0.35, 4.95, 3.30, 1.10, "plugin", "Any runtime / standalone",
             "bundle_from_dict — caller supplies\ntrace/vectors; features follow data"),

    # ---- center column: the unified pipeline ----
    "bundle": (4.85, 9.30, 4.90, 1.15, "data", "EvidenceBundle",
               "question · answer? · preload chunks (+reused vectors)\ntrace steps · node state · declared sections"),
    "t0":   (4.85, 7.75, 4.90, 1.00, "tier", "TIER 0 — Forensics   (free, <2 ms)",
             "handbook signatures over trace: retrieval, tools,\ncache, memory conflicts, truncation, encoder drift"),
    "t1":   (4.85, 6.35, 4.90, 0.95, "tier", "TIER 1 — Copy-checks   (free, lexical)",
             "numbers · dates · IDs · entities traceable to preload\n+ arithmetic closure for derived figures"),
    "t2":   (4.85, 4.85, 4.90, 1.05, "tier", "TIER 2 — Vector coverage   (~ms, in-process)",
             "sentence support vs reused chunk vectors\n+ composite support + contradiction-cue screen"),
    "t3":   (4.85, 3.40, 4.90, 0.95, "tier_opt", "TIER 3 — Span classifier   (gray zone only)",
             "local ONNX token classifier -> unsupported spans\nno LLM; CPU ~150 ms"),
    "t4":   (4.85, 2.00, 4.90, 0.90, "tier_opt", "TIER 4 — LLM judge   (opt-in, still-gray only)",
             "claim-level entailment; the ONLY LLM call\nbudget-capped; verdict cached"),
    "fuse": (4.85, 0.60, 4.90, 0.90, "verdict", "FUSION -> ShineVerdict",
             "decision · resolution gate · fused score · confidence\nsignatures · spans · evidence hash · advice"),

    # ---- right column: support systems ----
    "hb":   (10.85, 7.75, 4.60, 1.00, "support", "Handbook (YAML, versioned)",
             "failure-signature catalog: data not code\ncore + clinical/finance/legal packs"),
    "enc":  (10.85, 4.85, 4.60, 1.05, "support", "Shared local encoder",
             "prismlang ONNX session reuse | BYO Embedder |\nlexical fallback — zero embedding-API calls"),
    "prof": (10.85, 3.05, 4.60, 1.20, "support", "Strictness · profiles · calibration",
             "1 knob -> domain profile -> expert overrides\nper-technique thresholds; `prismshine calibrate`"),
    "cache": (10.85, 1.15, 4.60, 1.25, "support", "Verdict cache + consistency contract",
              "content-addressed (self-invalidating keys)\nevery mutation: prevention + detection backstop\n(`on_fact_corrected` eviction · stale-cache signatures)"),

    # ---- bottom left: enforcement outcomes ----
    "out":  (0.35, 0.60, 3.30, 2.30, "verdict", "Enforcement (via plugin)",
             "PASS  -> deliver\nFLAG  -> deliver + audit/review\nBLOCK -> policy fallback answer\nREGEN -> bounded repair loop\n(1 retry w/ span feedback)"),
}

# (src, dst, kind, rad, label, label_pos)
EDGES = [
    ("cg",   "bundle", "flow",  0.12, "", None),
    ("lg",   "bundle", "flow",  0.20, "", None),
    ("gen",  "bundle", "flow",  0.30, "", None),
    ("bundle", "t0",   "main",  0.0, "", None),
    ("t0",   "t1",     "main",  0.0, "", None),
    ("t1",   "t2",     "main",  0.0, "", None),
    ("t2",   "t3",     "gray",  0.0, "only if gray", None),
    ("t3",   "t4",     "gray",  0.0, "only if still gray", None),
    ("t4",   "fuse",   "main",  0.0, "", None),
    ("hb",   "t0",     "supp",  0.0, "", None),
    ("enc",  "t2",     "supp",  0.0, "", None),
    ("prof", "fuse",   "supp",  0.18, "", None),
    ("cache", "fuse",  "supp",  -0.10, "", None),
    ("fuse", "out",    "flow",  0.15, "", None),
]

EDGE_STYLE = {
    "main": dict(color="#1b5e20", lw=2.4, ls="-", alpha=0.95),
    "gray": dict(color="#7cb342", lw=2.0, ls="--", alpha=0.9),
    "flow": dict(color="#1565c0", lw=1.8, ls="-", alpha=0.85),
    "supp": dict(color="#6a1b9a", lw=1.4, ls=":", alpha=0.85),
    "exit": dict(color="#c62828", lw=1.8, ls="-.", alpha=0.9),
}


def center(name):
    x, y, w, h, *_ = BOXES[name]
    return x + w / 2, y + h / 2


def anchor(name, other_center):
    x, y, w, h, *_ = BOXES[name]
    cx, cy = x + w / 2, y + h / 2
    ox, oy = other_center
    if abs(oy - cy) > h / 2 + 0.15:
        return (min(max(ox, x + 0.30), x + w - 0.30), y + h if oy > cy else y)
    return (x + w if ox > cx else x, cy)


fig, ax = plt.subplots(figsize=(17.0, 11.5), dpi=170)
ax.set_xlim(0, 15.8)
ax.set_ylim(-0.35, 11.6)
ax.axis("off")
fig.patch.set_facecolor("white")

ax.text(7.9, 11.30, "PrismShine — Unified Anti-Hallucination Pipeline",
        ha="center", fontsize=19, fontweight="bold", color="#212121")
ax.text(7.9, 10.88,
        "one verify() pass · cause-side forensics + effect-side grounding fused into one auditable verdict\n"
        "deterministic-first (0 LLM calls on the expected majority path) · zero embedding-API calls · runtime-agnostic core",
        ha="center", fontsize=10, color="#616161")

# edges first
for src, dst, kind, rad, label, _ in EDGES:
    sa = anchor(src, center(dst))
    da = anchor(dst, center(src))
    ax.add_patch(FancyArrowPatch(sa, da, arrowstyle="-|>", mutation_scale=14,
                                 connectionstyle=f"arc3,rad={rad}",
                                 zorder=2, **EDGE_STYLE[kind]))
    if label:
        mx, my = (sa[0] + da[0]) / 2, (sa[1] + da[1]) / 2
        ax.text(mx + 0.15, my, label, fontsize=7.6, color="#558b2f",
                style="italic", va="center", zorder=5)

# ---- early-exit rail (right side of pipeline, red dash-dot) ----
ax.add_patch(FancyArrowPatch((9.75, 8.25), (9.75, 1.30),
                             arrowstyle="-|>", mutation_scale=14,
                             connectionstyle="arc3,rad=-0.22",
                             zorder=1, **EDGE_STYLE["exit"]))
ax.text(10.42, 4.80, "EARLY EXITS", fontsize=8.2, color="#c62828",
        fontweight="bold", rotation=90, va="center", zorder=5)

# early-exit rules note (top-right empty area)
ax.add_patch(FancyBboxPatch((10.85, 9.10), 4.60, 1.35,
                            boxstyle="round,pad=0.03,rounding_size=0.10",
                            facecolor="#fff5f5", edgecolor="#c62828",
                            lw=1.6, ls="-.", zorder=3))
ax.text(13.15, 10.22, "Early exits — p50 stays near zero", ha="center", va="center",
        fontsize=9.3, fontweight="bold", color="#b71c1c", zorder=4)
ax.text(13.15, 9.67,
        "T0 fatal signature -> BLOCK/REGENERATE (grounding tiers skipped)\n"
        "T0 clean + T1 zero unmatched + T2 coverage OK -> PASS\n"
        "`CLEAN_FAST_PATH` — the expected majority, < 25 ms, 0 LLM calls\n"
        "T2 coverage collapse (healthy preload) -> `T2_COVERAGE_COLLAPSE`",
        ha="center", va="center", fontsize=7.4, color="#b71c1c", zorder=4)

# ---- pre-generation loop (left of pipeline, arcing back up to the bundle) ----
ax.add_patch(FancyArrowPatch((4.85, 8.25), (4.85, 9.55),
                             arrowstyle="-|>", mutation_scale=12,
                             connectionstyle="arc3,rad=0.55",
                             zorder=1, color="#e65100", lw=1.6, ls="--", alpha=0.9))
ax.add_patch(FancyBboxPatch((0.35, 10.25), 3.30, 0.95,
                            boxstyle="round,pad=0.03,rounding_size=0.10",
                            facecolor="#fff8f0", edgecolor="#e65100",
                            lw=1.4, ls="--", zorder=3))
ax.text(2.00, 10.97, "Pre-generation mode", ha="center", va="center",
        fontsize=9.0, fontweight="bold", color="#e65100", zorder=4)
ax.text(2.00, 10.60,
        "verify(answer=None) runs Tier 0 only —\nhalts/repairs BEFORE the LLM call,\ngeneration tokens never spent (orange loop)",
        ha="center", va="center", fontsize=7.3, color="#bf360c", zorder=4)

# ---- capability note under plugins ----
ax.text(2.00, 4.28,
        "zero hard sibling dependencies (ADR-11)\ncore install: numpy + pydantic + pyyaml\n"
        "missing pieces degrade transparently,\nnever silently (coverage_mode, dormant\nfamilies recorded in every verdict)",
        fontsize=7.6, color="#e65100", ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff8f0", edgecolor="#ffcc80", lw=1.0),
        zorder=4)

# boxes
for name, (x, y, w, h, cat, title, body, *rest) in BOXES.items():
    fill, edge = STYLE[cat]
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.03,rounding_size=0.10",
                                facecolor=fill, edgecolor=edge, lw=2.0, zorder=3))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="center",
            fontsize=9.3, fontweight="bold", color="#212121", zorder=4)
    ax.text(x + w / 2, y + (h - 0.44) / 2, body, ha="center", va="center",
            fontsize=7.5, color="#37474f", zorder=4)

# legend
handles = [
    mpatches.Patch(facecolor=STYLE["data"][0], edgecolor=STYLE["data"][1], label="Data contract"),
    mpatches.Patch(facecolor=STYLE["tier"][0], edgecolor=STYLE["tier"][1], label="Always-on tier (deterministic, free/cheap)"),
    mpatches.Patch(facecolor=STYLE["tier_opt"][0], edgecolor=STYLE["tier_opt"][1], label="Escalation tier (gray zone only)"),
    mpatches.Patch(facecolor=STYLE["plugin"][0], edgecolor=STYLE["plugin"][1], label="Runtime plugin (all optional)"),
    mpatches.Patch(facecolor=STYLE["support"][0], edgecolor=STYLE["support"][1], label="Support system"),
    mpatches.Patch(facecolor=STYLE["verdict"][0], edgecolor=STYLE["verdict"][1], label="Verdict / enforcement"),
    plt.Line2D([0], [0], color="#1b5e20", lw=2.4, label="pipeline flow"),
    plt.Line2D([0], [0], color="#7cb342", lw=2.0, ls="--", label="conditional escalation"),
    plt.Line2D([0], [0], color="#c62828", lw=1.8, ls="-.", label="early exit / short-circuit"),
    plt.Line2D([0], [0], color="#6a1b9a", lw=1.4, ls=":", label="configuration / support"),
]
ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, -0.03),
          fontsize=7.8, framealpha=0.95, ncol=2)

plt.tight_layout()
out = r"c:\code\PrismShine\docs\prismshine-architecture.png"
plt.savefig(out, bbox_inches="tight", facecolor="white")
print("saved", out)
