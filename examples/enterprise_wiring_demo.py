"""Enterprise wiring demo — cause-side halt + post-answer verify + consistency hook.

Runnable without ChorusGraph.

  python examples/enterprise_wiring_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prismshine import EvidenceBundle, PreloadChunk, ShineGate, TraceStep
from prismshine.wiring import on_fact_corrected, pre_llm_check, wrap_llm


class _FakeCache:
    def __init__(self) -> None:
        self.entries = {"q1": {"answer": "stale $1000", "tags": ["fact:revenue"]}}
        self.invalidated: list[str] = []

    def invalidate_tags(self, tags: list[str]) -> int:
        n = 0
        for k, v in list(self.entries.items()):
            if any(t in (v.get("tags") or []) for t in tags):
                del self.entries[k]
                self.invalidated.append(k)
                n += 1
        return n


def main() -> int:
    gate = ShineGate.build(profile="default")
    cache = _FakeCache()

    # docs=[] (not preload=[]) so wiring injects the system empty sentinel;
    # n_chunks=0 fires EMPTY_RETRIEVAL before the LLM is called.
    empty_state = {
        "run_id": "demo-empty",
        "question": "What is Q2 revenue?",
        "docs": [],
        "trace": [
            {
                "hop": "retrieve",
                "kind": "retrieval",
                "status": "ok",
                "detail": {"n_chunks": 0},
            }
        ],
    }
    pre = pre_llm_check(gate, empty_state)
    print("1) pre-gen halt on empty retrieval:")
    print(f"   action={pre.action} decision={pre.verdict.decision if pre.verdict else None} "
          f"gate={pre.verdict.resolution_gate if pre.verdict else None}")

    preload = [
        PreloadChunk(chunk_id="c0", text="Q2 revenue was $1,200,000.", source="retrieval")
    ]
    trace = [TraceStep(hop="retrieve", kind="retrieval", status="ok", detail={"n_chunks": 1})]
    good = EvidenceBundle(
        run_id="demo-ok",
        question="What is Q2 revenue?",
        answer="Q2 revenue was $1,200,000.",
        preload=preload,
        trace=trace,
    )
    bad = EvidenceBundle(
        run_id="demo-bad",
        question="What is Q2 revenue?",
        answer="Q2 revenue was $9,999,000.",
        preload=preload,
        trace=trace,
    )
    v_ok = gate.verify(good)
    v_bad = gate.verify(bad)
    print("2) grounding:")
    print(f"   grounded -> {v_ok.decision} ({v_ok.resolution_gate})")
    print(f"   fabricated -> {v_bad.decision} ({v_bad.resolution_gate})")

    on_fact_corrected(cache=cache, subjects=["fact:revenue"])
    print("3) consistency contract:")
    print(f"   invalidated={cache.invalidated} remaining={list(cache.entries)}")

    calls = {"n": 0}

    def llm(system: str, user: str) -> str:
        calls["n"] += 1
        return "should not run"

    guarded = wrap_llm(llm, gate, state_factory=lambda: empty_state)
    out = guarded("sys", "What is Q2 revenue?")
    print("4) wrap_llm pre-halt:")
    print(f"   llm_calls={calls['n']} response={out!r}")

    caps = gate.capabilities()
    print("5) capabilities:", json.dumps(caps, indent=2)[:600])
    print("\nEnterprise demo OK — see docs/INTEGRATION.md for LangGraph/ChorusGraph.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
