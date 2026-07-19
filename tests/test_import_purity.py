"""Core modules must not import sibling runtimes or integrations."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "prismshine"

FORBIDDEN_PREFIXES = (
    "chorusgraph",
    "langgraph",
    "prismcortex",
    "prismguard",
    "prismlib",
    "prism.",
    "openai",
    "google.genai",
    "google.generativeai",
)

# Integrations and optional heavy deps may import siblings.
ALLOWED_PARTS = {"integrations", "grounding"}  # spans/judge may import onnx etc.


def _module_files() -> list[Path]:
    files = []
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT)
        if rel.parts and rel.parts[0] == "integrations":
            continue
        # judge/spans may import optional deps — still no sibling Insight libs in core path
        files.append(path)
    return files


def test_core_import_purity():
    violations: list[str] = []
    for path in _module_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                for bad in FORBIDDEN_PREFIXES:
                    if name == bad or name.startswith(bad + "."):
                        # encoder may import prismlang (optional coverage) — allowed
                        if name.startswith("prismlang"):
                            continue
                        # spans/judge optional third-party ok except we listed openai
                        if path.name in {"spans.py", "judge.py"} and name.split(".")[0] in {
                            "onnxruntime",
                            "huggingface_hub",
                            "tokenizers",
                            "openai",
                            "google",
                        }:
                            continue
                        if path.name == "judge.py" and (
                            name.startswith("openai") or name.startswith("google")
                        ):
                            continue
                        violations.append(f"{path.relative_to(ROOT)}: import {name}")
    # prismlang is allowed only in encoder.py
    filtered = []
    for v in violations:
        if "encoder.py" in v and "prismlang" in v:
            continue
        filtered.append(v)
    assert not filtered, "Forbidden imports in core:\n" + "\n".join(filtered)


def test_bare_import_works():
    import prismshine
    from prismshine import EvidenceBundle, ShineGate, ShineVerdict

    assert prismshine.__version__
    assert ShineGate is not None
    assert EvidenceBundle is not None
    assert ShineVerdict is not None
