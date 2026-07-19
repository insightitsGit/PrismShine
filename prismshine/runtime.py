"""RuntimeAdapter protocol + shared helpers — orchestrator-agnostic (P2)."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from prismshine.actions import actions_for_verdict
from prismshine.gate import ShineGate
from prismshine.models import EvidenceBundle, ShineVerdict
from prismshine.regen import build_repair_feedback, next_route


@runtime_checkable
class RuntimeAdapter(Protocol):
    """Four capabilities every orchestrator plugin must provide.

    Conformance: implement these methods and the same Shine error classes apply
    (see tests/test_runtime_conformance.py and tests/test_byo_runtime.py).
    """

    def extract_bundle(self, run: Any) -> EvidenceBundle:
        """Map runtime run/state/ledger -> EvidenceBundle."""
        ...

    def enforce(self, verdict: ShineVerdict, run: Any) -> Any:
        """Apply pass/flag/block/regenerate to the run (route, interrupt, repair)."""
        ...

    def pre_llm_hook(self, run: Any) -> Any:
        """Optional: Tier-0 halt before provider call (answer=None verify)."""
        ...

    def post_llm_hook(self, run: Any) -> Any:
        """Optional: full verify after generation."""
        ...


REQUIRED_CAPABILITIES = (
    "extract_bundle",
    "enforce",
    "pre_llm_hook",
    "post_llm_hook",
)


def check_adapter(adapter: Any) -> list[str]:
    """Return missing capability names (empty list = conforms)."""
    missing: list[str] = []
    for name in REQUIRED_CAPABILITIES:
        if not callable(getattr(adapter, name, None)):
            missing.append(name)
    return missing


def assert_adapter(adapter: Any) -> None:
    missing = check_adapter(adapter)
    if missing:
        raise TypeError(
            f"RuntimeAdapter incomplete; missing: {', '.join(missing)}. "
            "Implement extract_bundle/enforce/pre_llm_hook/post_llm_hook."
        )


def pull_ledger_steps(run: Any) -> list[Any]:
    """Best-effort ledger/trace extraction from compiled graphs / state / ctx."""
    if run is None:
        return []
    if isinstance(run, dict):
        for key in ("_ledger_steps", "ledger_steps", "ledger", "route_ledger", "trace"):
            if key in run and run[key]:
                return list(run[key])
        for key in ("_compiled", "compiled", "graph"):
            nested = run.get(key)
            if nested is not None:
                steps = pull_ledger_steps(nested)
                if steps:
                    return steps
        return []
    for attr in (
        "ledger_steps",
        "ledger",
        "route_ledger",
        "last_ledger",
        "get_ledger_steps",
        "steps",
    ):
        val = getattr(run, attr, None)
        if callable(val):
            try:
                return list(val())
            except Exception:  # noqa: BLE001
                continue
        if val is None:
            continue
        steps = getattr(val, "steps", None)
        if steps is not None:
            return list(steps)
        if isinstance(val, (list, tuple)):
            return list(val)
    for meth in ("get_ledger", "current_ledger", "ledger_snapshot"):
        fn = getattr(run, meth, None)
        if callable(fn):
            try:
                out = fn()
                if out is None:
                    continue
                steps = getattr(out, "steps", None)
                if steps is not None:
                    return list(steps)
                if isinstance(out, (list, tuple)):
                    return list(out)
            except Exception:  # noqa: BLE001
                continue
    state = getattr(run, "state", None)
    if isinstance(state, dict):
        return pull_ledger_steps(state)
    return []


def enforce_verdict(
    verdict: ShineVerdict,
    state: dict[str, Any],
    *,
    answer_key: str = "reply",
    max_regenerate: int = 1,
) -> dict[str, Any]:
    """Default enforce(): write verdict + route + repair/actions into state."""
    attempts = int(state.get("_shine_regen_attempts") or 0)
    route = next_route(verdict.decision, attempts, max_attempts=max_regenerate)
    out: dict[str, Any] = {
        "shine_verdict": verdict.model_dump(mode="json"),
        "shine_route": route,
        "shine_advice": list(verdict.advice),
        "shine_actions": actions_for_verdict(verdict),
        "_ledger_append": {
            "kind": "shine.verdict",
            "detail": {
                "decision": verdict.decision,
                "resolution_gate": verdict.resolution_gate,
                "fused_score": verdict.fused_score,
                "evidence_hash": verdict.evidence_hash,
                "signatures": [s.id for s in verdict.signatures],
            },
        },
    }
    if route == "block":
        out[answer_key] = state.get("shine_fallback") or (
            "I don't have reliable grounded data for that."
        )
    elif route == "regenerate":
        repair = build_repair_feedback(
            spans=verdict.spans,
            advice=list(verdict.advice),
            signatures=[s.id for s in verdict.signatures],
        )
        out["_shine_regen_attempts"] = attempts + 1
        out["shine_repair_feedback"] = repair
        out["shine_repair_prompt"] = repair["prompt_suffix"]
    elif verdict.decision == "regenerate":
        out["shine_route"] = "flag"
        out["shine_regen_exhausted"] = True
    return out


class GateRuntimeAdapter:
    """Reference RuntimeAdapter wrapping ShineGate + extract_fn."""

    def __init__(
        self,
        gate: ShineGate,
        extract_fn,
        *,
        answer_key: str = "reply",
    ) -> None:
        self.gate = gate
        self._extract_fn = extract_fn
        self.answer_key = answer_key

    def extract_bundle(self, run: Any) -> EvidenceBundle:
        return self._extract_fn(run)

    def enforce(self, verdict: ShineVerdict, run: Any) -> Any:
        state = run if isinstance(run, dict) else getattr(run, "state", {}) or {}
        if not isinstance(state, dict):
            state = dict(state)
        return enforce_verdict(verdict, state, answer_key=self.answer_key)

    def pre_llm_hook(self, run: Any) -> Any:
        bundle = self.extract_bundle(run)
        bundle = bundle.model_copy(update={"answer": None})
        return self.gate.verify(bundle)

    def post_llm_hook(self, run: Any) -> Any:
        bundle = self.extract_bundle(run)
        return self.gate.verify(bundle)
