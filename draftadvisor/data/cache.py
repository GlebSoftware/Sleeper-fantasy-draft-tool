"""Tiny disk cache: downloads raw files once, stores derived frames as gzipped CSV/JSON."""
from __future__ import annotations

import gzip
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]
try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]

from ..config import cache_dir, ensure_dirs, raw_dir

log = logging.getLogger(__name__)

_DOWNLOAD_TIMEOUT = 120.0


def download(url: str, dest: Path, max_age_hours: float | None = None, retries: int = 3) -> Path:
    """Download ``url`` to ``dest`` unless a fresh copy exists.

    ``max_age_hours=None`` means "never re-download if present" (immutable files
    such as past-season stats). Raises ``httpx.HTTPError`` after ``retries``.
    """
    ensure_dirs()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        if max_age_hours is None or (time.time() - dest.stat().st_mtime) < max_age_hours * 3600:
            return dest
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=_DOWNLOAD_TIMEOUT) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_bytes():
                        fh.write(chunk)
                tmp.replace(dest)
            log.info("downloaded %s -> %s (%d bytes)", url, dest, dest.stat().st_size)
            return dest
        except (httpx.HTTPError, OSError) as e:  # pragma: no cover - network
            last = e
            log.warning("download failed (%s/%s) %s: %s", attempt + 1, retries, url, e)
            time.sleep(1.5 * (attempt + 1))
    if dest.exists():
        log.warning("using stale copy of %s", dest)
        return dest
    assert last is not None
    raise last


def raw_path(name: str) -> Path:
    return raw_dir() / name


def cached_frame(name: str, builder: Callable[[], pd.DataFrame], refresh: bool = False,
                 max_age_hours: float | None = None, **read_kwargs) -> pd.DataFrame:
    """Return a DataFrame from ``cache/<name>.csv.gz`` or build + store it."""
    ensure_dirs()
    p = cache_dir() / f"{name}.csv.gz"
    if p.exists() and not refresh:
        fresh = max_age_hours is None or (time.time() - p.stat().st_mtime) < max_age_hours * 3600
        if fresh:
            return pd.read_csv(p, low_memory=False, **read_kwargs)
    df = builder()
    df.to_csv(p, index=False, compression="gzip")
    return df


def cached_json(name: str, builder: Callable[[], Any], refresh: bool = False,
                max_age_hours: float | None = None) -> Any:
    ensure_dirs()
    p = cache_dir() / f"{name}.json.gz"
    if p.exists() and not refresh:
        fresh = max_age_hours is None or (time.time() - p.stat().st_mtime) < max_age_hours * 3600
        if fresh:
            with gzip.open(p, "rt", encoding="utf-8") as fh:
                return json.load(fh)
    obj = builder()
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        json.dump(obj, fh)
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)
    tmp.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
