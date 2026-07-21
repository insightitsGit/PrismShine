"""Run the runtime suite against local uvicorn processes (no Docker required).

Starts chorus-shine + HHEM + MiniLM + LettuceDetect shims on ports 18201–18204,
then invokes ``run_runtime_bench``. Heavy models download on first use.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import uvicorn

from bench.runtime.run_runtime_bench import main as run_main

ROOT = Path(__file__).resolve().parents[2]


def _load_app(rel: str) -> Any:
    path = ROOT / rel
    name = path.stem + "_" + path.parent.name.replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    # Ensure module is importable for pydantic forward refs
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    if hasattr(mod, "EvalRequest") and hasattr(mod.EvalRequest, "model_rebuild"):
        mod.EvalRequest.model_rebuild()
    return mod.app


def _start(app: Any, host: str, port: int) -> uvicorn.Server:
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.time() + 600  # model downloads can be slow
    last_err = None
    while time.time() < deadline:
        try:
            r = httpx.get(f"http://{host}:{port}/health", timeout=5.0)
            if r.status_code == 200:
                print(f"ready :{port} -> {r.json()}")
                return server
            last_err = r.text
        except Exception as exc:
            last_err = str(exc)
        time.sleep(1.0)
    raise RuntimeError(f"port {port} not healthy: {last_err}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-h1", type=int, default=40)
    parser.add_argument("--out", default="bench/runtime/results/local")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--skip",
        default="",
        help="Comma-separated systems to skip (e.g. oss-langgraph-lettuce)",
    )
    args = parser.parse_args(argv)
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    specs = [
        ("chorus-shine", "bench/shims/chorus-shine/app.py", 18201),
        ("oss-langgraph-hhem", "bench/shims/oss-langgraph-hhem/app.py", 18202),
        ("oss-langgraph-minilm", "bench/shims/oss-langgraph-minilm/app.py", 18203),
        ("oss-langgraph-lettuce", "bench/shims/oss-langgraph-lettuce/app.py", 18204),
    ]
    servers: list[uvicorn.Server] = []
    targets: dict[str, str] = {}
    try:
        for name, rel, port in specs:
            if name in skip:
                print(f"skip {name}")
                continue
            print(f"starting {name} on {port}…")
            app = _load_app(rel)
            servers.append(_start(app, args.host, port))
            targets[name] = f"http://{args.host}:{port}"

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        targets_path = out / "targets.local.json"
        targets_path.write_text(json.dumps(targets, indent=2), encoding="utf-8")
        return run_main(
            [
                "--targets",
                str(targets_path),
                "--n-h1",
                str(args.n_h1),
                "--timeout",
                "600",
                "--out",
                str(out),
            ]
        )
    finally:
        for server in servers:
            server.should_exit = True


if __name__ == "__main__":
    raise SystemExit(main())
