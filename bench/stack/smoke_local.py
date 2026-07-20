"""Fast in-process Stack-suite smoke test; deliberately never loads HHEM."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench.stack.datasets import build_r1, build_s1  # noqa: E402


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _request(module, sample: dict):
    return module.EvalRequest(**sample)


def _assert_contract(response: dict) -> None:
    required = {
        "id", "track", "decision", "label", "risk", "latency_ms", "llm_calls",
        "cost_usd", "resolution_gate", "components", "saw_evidence",
    }
    missing = required - response.keys()
    if missing:
        raise AssertionError(f"Missing response fields: {sorted(missing)}")


def main() -> int:
    # Make the smoke independent of optional llm-guard model downloads.
    os.environ["STACK_FORCE_REGEX_GUARD"] = "1"
    insight = _load("stack_insight_app", "bench/shims/insight-stack/app.py")
    llmguard = _load("stack_llmguard_app", "bench/shims/oss-llmguard/app.py")
    s1 = build_s1()
    r1 = build_r1()

    checks = [
        ("insight S1 attack", insight.evaluate(_request(insight, s1[0]))),
        ("insight S1 benign", insight.evaluate(_request(insight, s1[-1]))),
        ("insight R1 fail", insight.evaluate(_request(insight, r1[0]))),
        ("insight R1 clean", insight.evaluate(_request(insight, r1[-1]))),
        ("llmguard S1 attack", llmguard.evaluate(_request(llmguard, s1[0]))),
        ("llmguard S1 benign", llmguard.evaluate(_request(llmguard, s1[-1]))),
    ]
    for name, response in checks:
        _assert_contract(response)
        print(f"{name}: decision={response['decision']} label={response['label']}")
    print(f"datasets: S1={len(s1)} R1={len(r1)}; HHEM intentionally skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
