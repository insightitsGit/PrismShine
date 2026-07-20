"""Load chorus-shine FastAPI app from the shim path (not an installed package)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


def load_chorus_shine_app() -> Any:
    path = Path(__file__).resolve().parents[1] / "shims" / "chorus-shine" / "app.py"
    spec = importlib.util.spec_from_file_location("chorus_shine_app", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.app
