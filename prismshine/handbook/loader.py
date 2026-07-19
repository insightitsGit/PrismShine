"""YAML handbook load / merge / version pinning."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from prismshine.handbook.schema import Handbook, SignatureDef

BUILTIN_DOMAIN_PACKS = frozenset({"clinical", "finance", "legal", "core"})


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Handbook YAML must be a mapping: {path}")
    return data


def _load_builtin(name: str = "core.yaml") -> dict[str, Any]:
    pkg = resources.files("prismshine.handbook.builtin")
    text = (pkg / name).read_text(encoding="utf-8")
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError("Builtin handbook must be a mapping")
    return data


def _merge_signatures(
    base: list[dict[str, Any]], overlay: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {s["id"]: dict(s) for s in base if "id" in s}
    for sig in overlay:
        sid = sig.get("id")
        if not sid:
            continue
        if sid in by_id:
            merged = dict(by_id[sid])
            merged.update(sig)
            by_id[sid] = merged
        else:
            by_id[sid] = dict(sig)
    return list(by_id.values())


def _load_pack(spec: str | Path) -> dict[str, Any]:
    """Load a pack from filesystem path or builtin name (clinical|finance|legal)."""
    name = str(spec)
    stem = Path(name).stem if name.endswith((".yaml", ".yml")) else name
    # Prefer filesystem when it exists
    p = Path(spec)
    if p.exists() and p.is_file():
        return _load_yaml(p)
    # Builtin name
    if stem in BUILTIN_DOMAIN_PACKS or stem in {"clinical", "finance", "legal", "core"}:
        return _load_builtin(f"{stem}.yaml")
    raise FileNotFoundError(f"Handbook pack not found: {spec}")


def load_handbook(
    *extra_paths: str | Path,
    builtin: str = "core.yaml",
    domain: str | None = None,
) -> Handbook:
    """Merge order: builtin core -> domain pack -> each extra path (tenant overrides)."""
    data = _load_builtin(builtin)
    version = str(data.get("handbook_version") or "0.1.0")
    signatures = list(data.get("signatures") or [])

    packs: list[str | Path] = []
    if domain:
        packs.append(domain)
    packs.extend(extra_paths)

    for path in packs:
        if path is None:
            continue
        overlay = _load_pack(path)
        if overlay.get("handbook_version"):
            version = str(overlay["handbook_version"])
        signatures = _merge_signatures(signatures, list(overlay.get("signatures") or []))

    return Handbook(
        handbook_version=version,
        signatures=[SignatureDef(**s) for s in signatures],
    )


def format_advice(template: str, **fields: Any) -> str:
    try:
        return template.format(**fields)
    except (KeyError, ValueError):
        return template
