"""Read-only access to the legacy research notes: local JSON files, Vercel Blob, or process memory.

Earlier versions of the app researched players with Claude and wrote one note per player; that loop
is gone (DESIGN.md §3.5) and nothing writes notes any more. The stateless web server still *reads*
whatever notes exist so their red flags and risk numbers keep showing up on the cards and in the
projections. Selection order in :func:`make_store`: Vercel Blob when ``BLOB_READ_WRITE_TOKEN`` is
set, else the local data directory, else memory (always empty).
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

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

    def count(self) -> int:
        return len(self.get_all())


class MemoryNoteStore(NoteStore):
    """No notes at all (the fallback when the data directory is not writable)."""

    backend = "memory"

    def __init__(self) -> None:
        self._notes: dict[str, dict] = {}

    def get_all(self) -> dict[str, dict]:
        return dict(self._notes)


class LocalNoteStore(NoteStore):
    """One JSON file per player under ``<dir>/<player_id>.json`` (the layout the old researcher wrote)."""

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


class BlobNoteStore(NoteStore):
    """Vercel Blob (REST API, list + download only) with a short in-memory cache; thread-safe."""

    backend = "blob"

    def __init__(self, token: str, prefix: str = NOTES_PREFIX, ttl: float = 60.0, timeout: float = 15.0) -> None:
        self.token = token
        self.prefix = prefix
        self.ttl = ttl
        self.timeout = timeout
        self._cache: dict[str, dict] = {}
        self._cache_at = 0.0
        self._lock = threading.Lock()

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}", "x-api-version": BLOB_API_VERSION}

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
