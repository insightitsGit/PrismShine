"""Opt-in Tier-4 OpenAI judge example (requires OPENAI_API_KEY + prismshine[judge-openai]).

  python examples/tier4_judge_demo.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prismshine import EvidenceBundle, PreloadChunk, ShineGate, TraceStep


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set — building gate with judge=None (Tier-4 off).")
        print("Set the key and pip install 'prismshine[judge-openai]' for live Tier-4.")
        gate = ShineGate.build(profile="finance")
    else:
        gate = ShineGate.build(profile="finance", judge="openai")

    preload = [
        PreloadChunk(
            chunk_id="c0",
            text="The trial found the drug was effective in adults with mild symptoms.",
            source="retrieval",
        )
    ]
    # Cue-y contradiction that may escalate when judge is present
    bundle = EvidenceBundle(
        run_id="t4-demo",
        question="Was the drug effective?",
        answer="The trial found the drug was ineffective in adults.",
        preload=preload,
        trace=[TraceStep(hop="retrieve", kind="retrieval", status="ok", detail={"n_chunks": 1})],
    )
    v = gate.verify(bundle)
    print(f"decision={v.decision} gate={v.resolution_gate} score={v.fused_score:.3f}")
    print(f"judge configured={gate.capabilities()['judge']} budget_rate={gate.budget.rate:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
