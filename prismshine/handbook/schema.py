"""Handbook signature schema."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Severity = Literal["fatal", "error", "warning", "info"]
Scope = Literal["preload", "answer", "run"]


class SignatureDef(BaseModel):
    id: str
    title: str = ""
    severity: Severity
    scope: Scope = "preload"
    detector: str
    params: dict[str, Any] = Field(default_factory=dict)
    signal_value: float = 1.0
    advice: str = ""
    references: list[str] = Field(default_factory=list)
    deprecated: bool = False
    replaced_by: str | None = None


class Handbook(BaseModel):
    handbook_version: str
    signatures: list[SignatureDef] = Field(default_factory=list)

    def by_id(self) -> dict[str, SignatureDef]:
        return {s.id: s for s in self.signatures if not s.deprecated}
