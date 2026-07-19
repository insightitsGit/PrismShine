"""Environment and programmatic configuration for PrismShine."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from prismshine.models import Strictness


def _env(name: str, default: str | None = None) -> str | None:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    return val


def _env_bool(name: str, default: bool = False) -> bool:
    val = _env(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class ShineConfig:
    profile: str = "default"
    strictness: Strictness = "standard"
    handbook_path: str | None = None
    verdict_db: str | None = None
    disable_tier3: bool = False
    judge_provider: str | None = None
    judge_model: str | None = None
    halt_on_fatal: bool = True
    regenerate_max_attempts: int = 1
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> ShineConfig:
        strictness = _env("PRISMSHINE_STRICTNESS", "standard") or "standard"
        if strictness not in {"lenient", "standard", "strict", "paranoid"}:
            strictness = "standard"
        return cls(
            profile=_env("PRISMSHINE_PROFILE", "default") or "default",
            strictness=strictness,  # type: ignore[arg-type]
            handbook_path=_env("PRISMSHINE_HANDBOOK_PATH"),
            verdict_db=_env("PRISMSHINE_VERDICT_DB"),
            disable_tier3=_env_bool("PRISMSHINE_DISABLE_TIER3", False),
            judge_provider=_env("PRISMSHINE_JUDGE_PROVIDER"),
            judge_model=_env("PRISMSHINE_JUDGE_MODEL"),
            halt_on_fatal=_env_bool("PRISMSHINE_HALT_ON_FATAL", True),
        )

    def merge(self, overrides: dict[str, Any] | None) -> ShineConfig:
        if not overrides:
            return self
        data = {
            "profile": self.profile,
            "strictness": self.strictness,
            "handbook_path": self.handbook_path,
            "verdict_db": self.verdict_db,
            "disable_tier3": self.disable_tier3,
            "judge_provider": self.judge_provider,
            "judge_model": self.judge_model,
            "halt_on_fatal": self.halt_on_fatal,
            "regenerate_max_attempts": self.regenerate_max_attempts,
            "extras": dict(self.extras),
        }
        data.update({k: v for k, v in overrides.items() if v is not None})
        return ShineConfig(**data)


# Programmatic config wins over env when set.
_PROGRAMMATIC: ShineConfig | None = None


def set_config(cfg: ShineConfig | None) -> None:
    global _PROGRAMMATIC
    _PROGRAMMATIC = cfg


def get_config() -> ShineConfig:
    if _PROGRAMMATIC is not None:
        return _PROGRAMMATIC
    return ShineConfig.from_env()
