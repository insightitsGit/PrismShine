"""In-process smoke for the runtime suite (no Docker / no HF downloads).

Runs chorus-shine FastAPI app + lightweight competitor stubs that mirror the
container contracts (evidence-blind on R1).
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import uvicorn

from bench.runtime.chorus_shine_loader import load_chorus_shine_app
from bench.runtime.run_runtime_bench import main as run_main


def _start_uvicorn(app: Any, host: str, port: int) -> uvicorn.Server:
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            httpx.get(f"http://{host}:{port}/health", timeout=1.0).raise_for_status()
            return server
        except Exception:
            time.sleep(0.25)
    raise RuntimeError(f"server on {port} failed to become healthy")


def _stub_competitor(system: str):
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI()

    class EvalRequest(BaseModel):
        id: str
        track: str
        question: str
        context: list[str] = []
        answer: str | None = None
        evidence: dict[str, Any] | None = None
        gold: str | None = None

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "system": system, "mode": "smoke_stub"}

    @app.post("/stack_evaluate")
    @app.post("/evaluate")
    def evaluate(req: EvalRequest) -> dict[str, Any]:
        if req.track == "R1":
            return {
                "id": req.id,
                "track": req.track,
                "decision": "pass",
                "label": "runtime_ok",
                "risk": 0.0,
                "latency_ms": 1.0,
                "llm_calls": 0,
                "cost_usd": 0.0,
                "resolution_gate": "EVIDENCE_IGNORED",
                "components": {},
                "saw_evidence": False,
            }
        ctx = " ".join(req.context).lower()
        ans = (req.answer or "").lower()
        tokens = [t for t in ans.replace("$", " ").split() if len(t) > 2]
        missing = sum(1 for t in tokens if t not in ctx)
        halluc = bool(tokens) and missing / len(tokens) >= 0.45
        return {
            "id": req.id,
            "track": req.track,
            "decision": "flag" if halluc else "pass",
            "label": "hallucinated" if halluc else "grounded",
            "risk": 0.8 if halluc else 0.1,
            "latency_ms": 2.0,
            "llm_calls": 0,
            "cost_usd": 0.0,
            "resolution_gate": "SMOKE_LEXICAL",
            "components": {},
            "saw_evidence": False,
        }

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-h1", type=int, default=4)
    parser.add_argument("--out", default="bench/runtime/results/smoke")
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)

    servers = [
        _start_uvicorn(load_chorus_shine_app(), args.host, 18201),
        _start_uvicorn(_stub_competitor("oss-langgraph-hhem"), args.host, 18202),
        _start_uvicorn(_stub_competitor("oss-langgraph-minilm"), args.host, 18203),
        _start_uvicorn(_stub_competitor("oss-langgraph-lettuce"), args.host, 18204),
    ]
    targets = {
        "chorus-shine": f"http://{args.host}:18201",
        "oss-langgraph-hhem": f"http://{args.host}:18202",
        "oss-langgraph-minilm": f"http://{args.host}:18203",
        "oss-langgraph-lettuce": f"http://{args.host}:18204",
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    targets_path = out / "targets.smoke.json"
    targets_path.write_text(json.dumps(targets, indent=2), encoding="utf-8")
    try:
        return run_main(
            [
                "--targets",
                str(targets_path),
                "--n-h1",
                str(args.n_h1),
                "--timeout",
                "120",
                "--out",
                str(out),
            ]
        )
    finally:
        for server in servers:
            server.should_exit = True


if __name__ == "__main__":
    raise SystemExit(main())
