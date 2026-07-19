"""Content-addressed verdict cache: memory LRU + SQLite."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Protocol

from prismshine.hashing import content_hash
from prismshine.models import ShineVerdict


class VerdictStore(Protocol):
    def get(self, key: str) -> ShineVerdict | None: ...
    def put(self, key: str, verdict: ShineVerdict) -> None: ...


def verdict_cache_key(
    *,
    preload_ids: list[str],
    preload_hash: str,
    answer_norm: str,
    profile_id: str,
    handbook_version: str,
    calibration_version: str,
    model_artifact_ids: list[str],
) -> str:
    return content_hash(
        {
            "preload_ids": sorted(preload_ids),
            "preload_hash": preload_hash,
            "answer_norm": answer_norm,
            "profile_id": profile_id,
            "handbook_version": handbook_version,
            "calibration_version": calibration_version,
            "model_artifact_ids": sorted(model_artifact_ids),
        }
    )


class MemoryVerdictStore:
    def __init__(self, maxsize: int = 1024) -> None:
        self._maxsize = maxsize
        self._data: OrderedDict[str, ShineVerdict] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> ShineVerdict | None:
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def put(self, key: str, verdict: ShineVerdict) -> None:
        with self._lock:
            self._data[key] = verdict
            self._data.move_to_end(key)
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)


class SqliteVerdictStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS verdicts (
                    key TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.path), check_same_thread=False)

    def get(self, key: str) -> ShineVerdict | None:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT payload FROM verdicts WHERE key = ?", (key,)
                ).fetchone()
        if not row:
            return None
        data = json.loads(row[0])
        return ShineVerdict.model_validate(data)

    def put(self, key: str, verdict: ShineVerdict) -> None:
        payload = json.dumps(verdict.model_dump(mode="json"), sort_keys=True)
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO verdicts(key, payload, created_at)
                    VALUES (?, ?, ?)
                    """,
                    (key, payload, time.time()),
                )
                conn.commit()


class TieredVerdictStore:
    """Memory front + optional SQLite back."""

    def __init__(
        self,
        memory: MemoryVerdictStore | None = None,
        sqlite: SqliteVerdictStore | None = None,
    ) -> None:
        self.memory = memory or MemoryVerdictStore()
        self.sqlite = sqlite

    def get(self, key: str) -> ShineVerdict | None:
        hit = self.memory.get(key)
        if hit is not None:
            return hit
        if self.sqlite is not None:
            hit = self.sqlite.get(key)
            if hit is not None:
                self.memory.put(key, hit)
            return hit
        return None

    def put(self, key: str, verdict: ShineVerdict) -> None:
        self.memory.put(key, verdict)
        if self.sqlite is not None:
            self.sqlite.put(key, verdict)
