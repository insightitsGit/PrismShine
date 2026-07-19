"""Render the Insight ecosystem map (12 libraries + identified gaps) to PNG."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# license category -> (fill, edge)
STYLE = {
    "oss": ("#e8f5e9", "#2e7d32"),        # Apache/MIT fully open
    "opencore": ("#fff3e0", "#ef6c00"),   # OSS core + licensed enterprise features
    "commercial": ("#ffebee", "#c62828"),
    "tool": ("#e3f2fd", "#1565c0"),       # standalone tools
    "gap": ("#f3e5f5", "#6a1b9a"),
}

# name: (x, y, w, h, category, subtitle)
BOXES = {
    # dev/ops tools column (right)
    "InsightPlugIn":  (13.45, 7.55, 2.35, 0.95, "tool", "IDE / SMS remote control"),
    "VectorBridge":   (13.45, 6.35, 2.35, 0.95, "tool", "vector DB migration"),
    "ChorusMesh":     (13.45, 5.15, 2.35, 0.95, "commercial", "alerts + Kafka/NATS (paid)"),

    # orchestration layer
    "ChorusGraph":    (1.10, 7.35, 3.60, 1.30, "opencore", "agent graph runtime (BSP)\nthe ecosystem hub"),
    "PrismCortex":    (5.30, 7.35, 3.30, 1.30, "opencore", "bitemporal agent memory\ndeterministic replay"),
    "PrismGuard":     (9.20, 7.35, 3.30, 1.30, "opencore", "prompt-injection firewall\nauditable gates"),

    # middle layer
    "PrismLang":      (1.10, 4.90, 2.60, 1.15, "oss", "64-d vector envelopes\nfor LangGraph"),
    "PrismLib / PrismLib-Plus": (4.10, 4.90, 4.30, 1.15, "oss", "semantic cache + WAL driver\n+ cluster + PrismAPI"),
    "PrismRAG-patch": (8.80, 4.90, 2.70, 1.15, "commercial", "taxonomy remap\nbefore vector DB"),

    # foundation layer
    "CHORUS Fabric":  (2.40, 2.60, 3.10, 1.10, "oss", "encrypted gRPC tensor\ntransport (patented cipher)"),
    "PrismResonance": (6.40, 2.60, 3.10, 1.10, "oss", "wave-memory re-ranking\namplitude + phase"),

    # gaps (proposed PrismShine scope)
    "LLM Gateway\n+ cost ledger":       (0.70, 0.30, 2.70, 1.05, "gap", ""),
    "Output guardrails\n(PII / grounding)": (3.70, 0.30, 2.70, 1.05, "gap", ""),
    "Eval harness\n(agent regression)": (6.70, 0.30, 2.70, 1.05, "gap", ""),
    "Trace console\n(ledger viewer UI)": (9.70, 0.30, 2.70, 1.05, "gap", ""),
    "Meta-package + scaffolding\n(one pinned install)": (12.70, 0.30, 3.10, 1.05, "gap", ""),
}

# (src, dst, style, rad) style: hard / extra / soft / vendored
EDGES = [
    ("ChorusGraph", "PrismLang", "hard", 0.06),
    ("ChorusGraph", "PrismLib / PrismLib-Plus", "hard", 0.06),
    ("ChorusGraph", "PrismResonance", "hard", 0.06),
    ("ChorusGraph", "PrismCortex", "extra", 0.06),
    ("ChorusGraph", "PrismRAG-patch", "extra", 0.06),
    ("ChorusGraph", "PrismGuard", "soft", -0.35),
    ("PrismCortex", "PrismLang", "extra", 0.06),
    ("PrismCortex", "PrismLib / PrismLib-Plus", "extra", 0.06),
    ("PrismCortex", "PrismRAG-patch", "extra", 0.06),
    ("PrismCortex", "PrismResonance", "extra", 0.06),
    ("PrismGuard", "PrismRAG-patch", "extra", 0.06),
    ("PrismGuard", "PrismLib / PrismLib-Plus", "extra", 0.06),
    ("ChorusMesh", "PrismLib / PrismLib-Plus", "hard", -0.30),
    ("PrismLib / PrismLib-Plus", "CHORUS Fabric", "vendored", 0.06),
    ("PrismLib / PrismLib-Plus", "PrismResonance", "vendored", 0.06),
    ("VectorBridge", "CHORUS Fabric", "vendored", -0.32),
    ("PrismResonance", "PrismLang", "extra", 0.06),
]

EDGE_STYLE = {
    "hard":     dict(color="#1b5e20", lw=2.2, ls="-",  alpha=0.95),
    "extra":    dict(color="#546e7a", lw=1.4, ls="--", alpha=0.75),
    "soft":     dict(color="#1565c0", lw=1.4, ls=":",  alpha=0.9),
    "vendored": dict(color="#ad1457", lw=1.6, ls="-.", alpha=0.8),
}


def center(name):
    x, y, w, h, *_ = BOXES[name]
    return x + w / 2, y + h / 2


def edge_anchor(name, other_center):
    """Anchor on box border nearest to the other box center (top/bottom biased)."""
    x, y, w, h, *_ = BOXES[name]
    cx, cy = x + w / 2, y + h / 2
    ox, oy = other_center
    if abs(oy - cy) > h / 2 + 0.1:
        return (min(max(ox, x + 0.25), x + w - 0.25), y + h if oy > cy else y)
    return (x + w if ox > cx else x, cy)


fig, ax = plt.subplots(figsize=(17.5, 10.5), dpi=170)
ax.set_xlim(0, 16.2)
ax.set_ylim(-0.15, 10.35)
ax.axis("off")
fig.patch.set_facecolor("white")

ax.text(8.1, 10.05, "Insight IT Solutions — Multi-Agent Ecosystem Map",
        ha="center", fontsize=19, fontweight="bold", color="#212121")
ax.text(8.1, 9.62, "dependency edges verified from source (Jul 2026) · dashed purple = missing pieces for a complete stack",
        ha="center", fontsize=10.5, color="#616161")

# layer bands
for y0, y1, label in [(7.15, 8.85, "ORCHESTRATION & SAFETY"),
                      (4.70, 6.25, "INTELLIGENCE MIDDLE LAYER"),
                      (2.40, 3.90, "FOUNDATION (transport & wave memory)"),
                      (0.10, 1.55, "MISSING — CANDIDATE PrismShine SCOPE")]:
    ax.add_patch(mpatches.Rectangle((0.45, y0), 12.45 if y0 > 0.2 else 15.6, y1 - y0,
                                    fill=True, facecolor="#fafafa", edgecolor="#e0e0e0",
                                    lw=0.8, zorder=0))
    ax.text(0.50, y1 + 0.07, label, fontsize=8.5, color="#9e9e9e",
            fontweight="bold", zorder=1, va="bottom")

ax.text(14.6, 8.85, "DEV & OPS TOOLS", fontsize=8.5, color="#9e9e9e",
        fontweight="bold", ha="center")

# edges first (under boxes)
for src, dst, kind, rad in EDGES:
    sa = edge_anchor(src, center(dst))
    da = edge_anchor(dst, center(src))
    ax.add_patch(FancyArrowPatch(sa, da, arrowstyle="-|>", mutation_scale=13,
                                 connectionstyle=f"arc3,rad={rad}",
                                 zorder=2, **EDGE_STYLE[kind]))

# boxes
for name, (x, y, w, h, cat, sub) in BOXES.items():
    fill, edge = STYLE[cat]
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.03,rounding_size=0.12",
                                facecolor=fill, edgecolor=edge,
                                lw=2.0 if cat != "gap" else 1.8,
                                linestyle="--" if cat == "gap" else "-",
                                zorder=3))
    if sub:
        ax.text(x + w / 2, y + h - 0.30, name, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color="#212121", zorder=4)
        ax.text(x + w / 2, y + 0.33, sub, ha="center", va="center",
                fontsize=7.8, color="#424242", zorder=4)
    else:
        ax.text(x + w / 2, y + h / 2, name, ha="center", va="center",
                fontsize=9.2, fontweight="bold", color="#4a148c", zorder=4)

# legends
lic_handles = [
    mpatches.Patch(facecolor=STYLE["oss"][0], edgecolor=STYLE["oss"][1], label="Open source (Apache/MIT)"),
    mpatches.Patch(facecolor=STYLE["opencore"][0], edgecolor=STYLE["opencore"][1], label="Open-core (licensed enterprise tier)"),
    mpatches.Patch(facecolor=STYLE["commercial"][0], edgecolor=STYLE["commercial"][1], label="Commercial"),
    mpatches.Patch(facecolor=STYLE["tool"][0], edgecolor=STYLE["tool"][1], label="Standalone tool"),
    mpatches.Patch(facecolor=STYLE["gap"][0], edgecolor=STYLE["gap"][1], linestyle="--", label="Missing / proposed"),
]
edge_handles = [
    plt.Line2D([0], [0], **{k: v for k, v in EDGE_STYLE["hard"].items() if k != "alpha"}, label="hard pip dependency"),
    plt.Line2D([0], [0], **{k: v for k, v in EDGE_STYLE["extra"].items() if k != "alpha"}, label="optional extra / integration"),
    plt.Line2D([0], [0], **{k: v for k, v in EDGE_STYLE["soft"].items() if k != "alpha"}, label="soft integration (guard node)"),
    plt.Line2D([0], [0], **{k: v for k, v in EDGE_STYLE["vendored"].items() if k != "alpha"}, label="vendored math (same brand, own code)"),
]
leg1 = ax.legend(handles=lic_handles, loc="upper left", bbox_to_anchor=(0.795, 0.475),
                 fontsize=8, framealpha=0.95, title="License", title_fontsize=9)
ax.add_artist(leg1)
ax.legend(handles=edge_handles, loc="upper left", bbox_to_anchor=(0.795, 0.30),
          fontsize=8, framealpha=0.95, title="Edges", title_fontsize=9)

plt.tight_layout()
out = r"c:\code\PrismShine\kb\insight-ecosystem-map.png"
plt.savefig(out, bbox_inches="tight", facecolor="white")
print("saved", out)
