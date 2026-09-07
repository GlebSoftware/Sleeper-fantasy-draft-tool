"""Research-note storage: local JSON files, Vercel Blob, or process memory.

The stateless web server persists Claude research notes here so every request
(and every serverless instance) sees the same notes. Selection order in
:func:`make_store`: Vercel Blob when ``BLOB_READ_WRITE_TOKEN`` is set, else the
local data directory when writable, else memory (lost on restart).
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Mapping

import httpx

log = logging.getLogger(__name__)

BLOB_API = "https://blob.vercel-storage.com"
BLOB_API_VERSION = "7"
NOTES_PREFIX = "draftadvisor/notes/"


class NoteStore:
    backend = "memory"

    def get_all(self) -> dict[str, dict]:
        raise NotImplementedError

    def get(self, player_id: str) -> dict | None:
        return self.get_all().get(str(player_id))

    def put(self, player_id: str, note: Mapping[str, Any]) -> None:
        raise NotImplementedError

    def count(self) -> int:
        return len(self.get_all())


class MemoryNoteStore(NoteStore):
    backend = "memory"

    def __init__(self) -> None:
        self._notes: dict[str, dict] = {}

    def get_all(self) -> dict[str, dict]:
        return dict(self._notes)

    def put(self, player_id: str, note: Mapping[str, Any]) -> None:
        self._notes[str(player_id)] = dict(note)


class LocalNoteStore(NoteStore):
    """One JSON file per player under ``<dir>/<player_id>.json`` (same layout as ClaudeResearcher)."""

    backend = "local"

    def __init__(self, directory: Path) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, dict] | None = None
        self._cache_at = 0.0

    def get_all(self) -> dict[str, dict]:
        if self._cache is not None and time.time() - self._cache_at < 5.0:
            return dict(self._cache)
        out: dict[str, dict] = {}
        for p in self.dir.glob("*.json"):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                pid = str(d.get("player_id") or p.stem)
                out[pid] = d
            except Exception as e:  # noqa: BLE001
                log.debug("skip note %s: %s", p, e)
        self._cache, self._cache_at = out, time.time()
        return dict(out)

    def put(self, player_id: str, note: Mapping[str, Any]) -> None:
        p = self.dir / f"{player_id}.json"
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(dict(note)), encoding="utf-8")
        tmp.replace(p)
        if self._cache is not None:
            self._cache[str(player_id)] = dict(note)


class BlobNoteStore(NoteStore):
    """Vercel Blob (REST API) with a short in-memory cache; thread-safe."""

    backend = "blob"

    def __init__(self, token: str, prefix: str = NOTES_PREFIX, ttl: float = 60.0, timeout: float = 15.0) -> None:
        self.token = token
        self.prefix = prefix
        self.ttl = ttl
        self.timeout = timeout
        self._cache: dict[str, dict] = {}
        self._cache_at = 0.0
        self._lock = threading.Lock()

    def _headers(self, **extra: str) -> dict[str, str]:
        h = {"Authorization": f"Bearer {self.token}", "x-api-version": BLOB_API_VERSION}
        h.update(extra)
        return h

    def _refresh(self) -> None:
        notes: dict[str, dict] = {}
        with httpx.Client(timeout=self.timeout) as c:
            cursor = None
            while True:
                params: dict[str, Any] = {"prefix": self.prefix, "limit": 1000}
                if cursor:
                    params["cursor"] = cursor
                r = c.get(BLOB_API + "/", headers=self._headers(), params=params)
                r.raise_for_status()
                body = r.json()
                for b in body.get("blobs", []):
                    url = b.get("url") or b.get("downloadUrl")
                    if not url:
                        continue
                    try:
                        d = c.get(url).json()
                        notes[str(d.get("player_id") or Path(b.get("pathname", "")).stem)] = d
                    except Exception as e:  # noqa: BLE001
                        log.debug("blob note unreadable %s: %s", url, e)
                cursor = body.get("cursor") if body.get("hasMore") else None
                if not cursor:
                    break
        with self._lock:
            self._cache, self._cache_at = notes, time.time()

    def get_all(self) -> dict[str, dict]:
        with self._lock:
            fresh = time.time() - self._cache_at < self.ttl
            snapshot = dict(self._cache)
        if fresh:
            return snapshot
        try:
            self._refresh()
        except Exception as e:  # noqa: BLE001
            log.warning("blob list failed: %s", e)
        with self._lock:
            return dict(self._cache)

    def put(self, player_id: str, note: Mapping[str, Any]) -> None:
        path = f"{self.prefix}{player_id}.json"
        with httpx.Client(timeout=self.timeout) as c:
            r = c.put(f"{BLOB_API}/{path}", content=json.dumps(dict(note)).encode(),
                      headers=self._headers(**{"x-content-type": "application/json", "x-add-random-suffix": "0",
                                               "x-allow-overwrite": "1", "x-cache-control-max-age": "0"}))
            r.raise_for_status()
        with self._lock:
            self._cache[str(player_id)] = dict(note)


def make_store(home: Path | None = None) -> NoteStore:
    token = os.environ.get("BLOB_READ_WRITE_TOKEN")
    if token:
        return BlobNoteStore(token)
    if home is None:
        from ..config import research_dir

        home = research_dir() / "notes"
    try:
        Path(home).mkdir(parents=True, exist_ok=True)
        probe = Path(home) / ".write-test"
        probe.write_text("ok")
        probe.unlink()
        return LocalNoteStore(Path(home))
    except Exception as e:  # noqa: BLE001
        log.warning("local note store unavailable (%s); notes kept in memory only", e)
        return MemoryNoteStore()
