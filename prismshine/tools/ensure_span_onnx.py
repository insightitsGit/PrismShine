"""Ensure Tier-3 ONNX artifacts are available for enterprise / air-gapped installs.

Does not download a 1GB model into the wheel. Resolves, in order:

1. ``PRISMSHINE_SPAN_ONNX`` (+ tokenizer beside it or ``PRISMSHINE_SPAN_TOKENIZER``)
2. Local export dirs: ``models/lettucedetect``, ``~/.prismshine/models/lettucedetect``,
   ``bench/shims/prismshine/baked``
3. Optional ``--export`` to run ``export_span_onnx`` when torch/transformers are present

Usage::

  python -m prismshine.tools.ensure_span_onnx
  python -m prismshine.tools.ensure_span_onnx --export --out ~/.prismshine/models/lettucedetect
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


CANDIDATE_DIRS = (
    Path("models/lettucedetect"),
    Path.home() / ".prismshine" / "models" / "lettucedetect",
    Path("bench/shims/prismshine/baked"),
)


def _has_pair(onnx: Path) -> Path | None:
    if not onnx.is_file():
        return None
    for tok in (
        onnx.with_name("tokenizer.json"),
        onnx.parent / "tokenizer.json",
    ):
        if tok.is_file():
            return tok
    return None


def resolve_span_artifacts(
    *,
    prefer_env: bool = True,
) -> dict[str, str | None]:
    """Return paths for onnx/tokenizer if found; does not mutate process env."""
    if prefer_env:
        pinned = os.environ.get("PRISMSHINE_SPAN_ONNX")
        if pinned and Path(pinned).is_file():
            tok = os.environ.get("PRISMSHINE_SPAN_TOKENIZER")
            if tok and Path(tok).is_file():
                return {"onnx": pinned, "tokenizer": tok, "source": "env"}
            beside = _has_pair(Path(pinned))
            if beside is not None:
                return {
                    "onnx": pinned,
                    "tokenizer": str(beside),
                    "source": "env",
                }

    for d in CANDIDATE_DIRS:
        onnx = d / "model.onnx"
        tok = _has_pair(onnx)
        if tok is not None:
            return {"onnx": str(onnx.resolve()), "tokenizer": str(tok), "source": str(d)}
    return {"onnx": None, "tokenizer": None, "source": None}


def apply_env(paths: dict[str, str | None]) -> None:
    if paths.get("onnx"):
        os.environ["PRISMSHINE_SPAN_ONNX"] = str(paths["onnx"])
    if paths.get("tokenizer"):
        os.environ["PRISMSHINE_SPAN_TOKENIZER"] = str(paths["tokenizer"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", action="store_true", help="Run export_span_onnx if missing")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path.home() / ".prismshine" / "models" / "lettucedetect",
        help="Export destination when --export",
    )
    ap.add_argument("--apply-env", action="store_true", help="Set PRISMSHINE_SPAN_* in this process")
    ap.add_argument("--json", action="store_true", help="Machine-readable result")
    args = ap.parse_args(argv)

    paths = resolve_span_artifacts()
    if paths["onnx"] is None and args.export:
        from prismshine.tools.export_span_onnx import export

        args.out.mkdir(parents=True, exist_ok=True)
        export(args.out)
        paths = resolve_span_artifacts(prefer_env=False)
        # prefer freshly exported dir
        onnx = args.out / "model.onnx"
        tok = _has_pair(onnx)
        if tok is not None:
            paths = {
                "onnx": str(onnx.resolve()),
                "tokenizer": str(tok),
                "source": "export",
            }

    if args.apply_env:
        apply_env(paths)

    payload = {
        "ok": paths["onnx"] is not None and paths["tokenizer"] is not None,
        **paths,
        "hint": (
            None
            if paths["onnx"]
            else "Run: python -m prismshine.tools.ensure_span_onnx --export"
        ),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        if payload["ok"]:
            print(f"ONNX ready: {paths['onnx']}")
            print(f"tokenizer:  {paths['tokenizer']}")
            print(f"source:     {paths['source']}")
            print("Pin for services:")
            print(f"  set PRISMSHINE_SPAN_ONNX={paths['onnx']}")
            print(f"  set PRISMSHINE_SPAN_TOKENIZER={paths['tokenizer']}")
        else:
            print("ONNX Tier-3 artifacts not found.", file=sys.stderr)
            print(payload["hint"], file=sys.stderr)
            return 1
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
