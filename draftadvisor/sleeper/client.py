"""Async Sleeper API client (see DESIGN.md §3.1).

* one keep-alive :class:`httpx.AsyncClient`, 10 s timeout;
* retries with exponential backoff on timeouts / connection errors / 5xx / 429 — never on 404
  (:class:`SleeperNotFound`) or other 4xx;
* ``/players/nfl`` (~5 MB) is cached gzipped under ``cache_dir()`` for 24 h;
* projections / stats are normalised to ``dict[player_id, stats_dict]`` whichever endpoint served them.

An :class:`httpx.AsyncClient` can be injected (``http=``) so tests use ``httpx.MockTransport``.
"""
from __future__ import annotations

import asyncio
import gzip
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Coroutine, Iterable, Sequence, TypeVar

import httpx

from ..config import (
    SKILL_POSITIONS,
    SLEEPER_API_BASE,
    SLEEPER_API_BASE_V2,
    SLEEPER_PLAYERS_TTL_HOURS,
    cache_dir,
)

log = logging.getLogger(__name__)

T = TypeVar("T")

_USER_AGENT = "draftadvisor/0.1 (+https://github.com/draftadvisor)"
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_PLAYERS_CACHE_NAME = "sleeper_players.json.gz"


class SleeperAPIError(Exception):
    """Any failure talking to Sleeper after retries were exhausted."""

    def __init__(self, message: str, status_code: int | None = None, url: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.url = url

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        base = super().__str__()
        extra = " ".join(x for x in (f"status={self.status_code}" if self.status_code else "",
                                     f"url={self.url}" if self.url else "") if x)
        return f"{base} ({extra})" if extra else base


class SleeperNotFound(SleeperAPIError):
    """404 (or JSON ``null`` for an object endpoint): the resource does not exist."""


def _players_cache_path() -> Path:
    return cache_dir() / _PLAYERS_CACHE_NAME


def _normalise_projections(payload: Any) -> dict[str, dict]:
    """Normalise a v1 dict (``{pid: stats}``) or a v2 list (``[{player_id, stats, player}]``)."""
    out: dict[str, dict] = {}
    if isinstance(payload, dict):
        for pid, stats in payload.items():
            if isinstance(stats, dict):
                d = dict(stats)
                d.setdefault("player_id", str(pid))
                out[str(pid)] = d
    elif isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            pid = item.get("player_id")
            if pid is None:
                pl = item.get("player") or {}
                pid = pl.get("player_id")
            if pid is None:
                continue
            stats = item.get("stats")
            if not isinstance(stats, dict) or not stats:
                continue
            d = dict(stats)
            d["player_id"] = str(pid)
            pl = item.get("player")
            if isinstance(pl, dict):
                for k in ("position", "team"):
                    if pl.get(k) is not None and k not in d:
                        d[k] = pl[k]
            out[str(pid)] = d
    return out


class SleeperClient:
    """Thin async wrapper over the public Sleeper HTTP API."""

    def __init__(self, timeout: float = 10.0, retries: int = 3, base_url: str = SLEEPER_API_BASE,
                 base_url_v2: str = SLEEPER_API_BASE_V2, http: httpx.AsyncClient | None = None,
                 backoff_base: float = 0.5, backoff_max: float = 8.0,
                 players_ttl_hours: float = SLEEPER_PLAYERS_TTL_HOURS):
        self.base_url = base_url.rstrip("/")
        self.base_url_v2 = base_url_v2.rstrip("/")
        self.retries = max(0, int(retries))
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.players_ttl_hours = players_ttl_hours
        self._owns_http = http is None
        self._http = http or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers={"User-Agent": _USER_AGENT, "Accept": "application/json"},
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            follow_redirects=True,
        )
        if not self._owns_http and self._http.headers.get("User-Agent", "").startswith("python-httpx"):
            self._http.headers["User-Agent"] = _USER_AGENT
        self.request_count = 0
        self.last_latency_ms = 0.0

    # -- lifecycle ------------------------------------------------------------
    async def aclose(self) -> None:
        """Close the underlying HTTP client (only if we created it)."""
        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self) -> "SleeperClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    # -- core request ---------------------------------------------------------
    def _url(self, path: str, base: str | None) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        return f"{base or self.base_url}/{path.lstrip('/')}"

    def _backoff(self, attempt: int, response: httpx.Response | None = None) -> float:
        delay = min(self.backoff_max, self.backoff_base * (2 ** attempt))
        if response is not None and response.status_code == 429:
            ra = response.headers.get("Retry-After")
            try:
                if ra is not None:
                    delay = max(delay, min(self.backoff_max * 2, float(ra)))
            except ValueError:
                pass
        return delay

    async def get_json(self, path: str, *, base: str | None = None,
                       params: dict | Sequence[tuple[str, Any]] | None = None) -> Any:
        """GET ``path`` (relative to ``base`` or the v1 base) and return the parsed JSON body.

        Retries on timeouts, connection errors, 5xx and 429 with exponential backoff.
        Raises :class:`SleeperNotFound` on 404 and :class:`SleeperAPIError` otherwise.
        """
        url = self._url(path, base)
        last_exc: Exception | None = None
        last_status: int | None = None
        for attempt in range(self.retries + 1):
            t0 = time.perf_counter()
            try:
                resp = await self._http.get(url, params=params)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                self.request_count += 1
                last_exc = e
                log.warning("sleeper %s attempt %d/%d failed: %s", url, attempt + 1, self.retries + 1,
                            type(e).__name__)
                if attempt < self.retries:
                    await asyncio.sleep(self._backoff(attempt))
                continue
            self.request_count += 1
            self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
            status = resp.status_code
            if status == 404:
                raise SleeperNotFound(f"not found: {url}", status_code=404, url=url)
            if status in _RETRY_STATUSES:
                last_status = status
                log.warning("sleeper %s -> HTTP %d (attempt %d/%d)", url, status, attempt + 1, self.retries + 1)
                if attempt < self.retries:
                    await asyncio.sleep(self._backoff(attempt, resp))
                continue
            if status >= 400:
                raise SleeperAPIError(f"HTTP {status} for {url}", status_code=status, url=url)
            try:
                return resp.json()
            except (json.JSONDecodeError, ValueError) as e:
                raise SleeperAPIError(f"invalid JSON from {url}: {e}", status_code=status, url=url) from e
        if last_status is not None:
            raise SleeperAPIError(f"HTTP {last_status} for {url} after {self.retries + 1} attempts",
                                  status_code=last_status, url=url)
        raise SleeperAPIError(f"{type(last_exc).__name__} for {url} after {self.retries + 1} attempts",
                              url=url) from last_exc

    async def _get_object(self, path: str) -> dict:
        """GET an endpoint that returns a JSON object; ``null`` means "not found"."""
        data = await self.get_json(path)
        if data is None:
            raise SleeperNotFound(f"not found: {path}", status_code=404, url=self._url(path, None))
        if not isinstance(data, dict):
            raise SleeperAPIError(f"unexpected payload for {path}: {type(data).__name__}", url=self._url(path, None))
        return data

    async def _get_list(self, path: str) -> list[dict]:
        data = await self.get_json(path)
        if data is None:
            return []
        if not isinstance(data, list):
            raise SleeperAPIError(f"unexpected payload for {path}: {type(data).__name__}", url=self._url(path, None))
        return data

    # -- state / users --------------------------------------------------------
    async def get_state(self, sport: str = "nfl") -> dict:
        """``/state/{sport}``: current season, week, season_type."""
        return await self._get_object(f"/state/{sport}")

    async def get_user(self, username_or_id: str) -> dict:
        """``/user/{username_or_id}``."""
        return await self._get_object(f"/user/{username_or_id}")

    async def get_user_leagues(self, user_id: str, season: int, sport: str = "nfl") -> list[dict]:
        return await self._get_list(f"/user/{user_id}/leagues/{sport}/{season}")

    async def get_user_drafts(self, user_id: str, season: int, sport: str = "nfl") -> list[dict]:
        return await self._get_list(f"/user/{user_id}/drafts/{sport}/{season}")

    # -- league ---------------------------------------------------------------
    async def get_league(self, league_id: str) -> dict:
        return await self._get_object(f"/league/{league_id}")

    async def get_league_users(self, league_id: str) -> list[dict]:
        return await self._get_list(f"/league/{league_id}/users")

    async def get_league_rosters(self, league_id: str) -> list[dict]:
        return await self._get_list(f"/league/{league_id}/rosters")

    async def get_league_drafts(self, league_id: str) -> list[dict]:
        return await self._get_list(f"/league/{league_id}/drafts")

    # -- draft ----------------------------------------------------------------
    async def get_draft(self, draft_id: str) -> dict:
        return await self._get_object(f"/draft/{draft_id}")

    async def get_draft_picks(self, draft_id: str) -> list[dict]:
        return await self._get_list(f"/draft/{draft_id}/picks")

    async def get_traded_picks(self, draft_id: str) -> list[dict]:
        return await self._get_list(f"/draft/{draft_id}/traded_picks")

    # -- players --------------------------------------------------------------
    def _read_players_cache(self, max_age_hours: float | None) -> dict[str, dict] | None:
        p = _players_cache_path()
        try:
            if not p.exists() or p.stat().st_size == 0:
                return None
            if max_age_hours is not None and (time.time() - p.stat().st_mtime) > max_age_hours * 3600:
                return None
            with gzip.open(p, "rt", encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else None
        except (OSError, ValueError) as e:
            log.warning("could not read players cache %s: %s", p, e)
            return None

    def _write_players_cache(self, payload: dict) -> None:
        p = _players_cache_path()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(p.suffix + ".tmp")
            with gzip.open(tmp, "wt", encoding="utf-8") as fh:
                json.dump(payload, fh)
            tmp.replace(p)
        except OSError as e:
            log.warning("could not write players cache %s: %s", p, e)

    async def get_players(self, force_refresh: bool = False) -> dict[str, dict]:
        """``/players/nfl`` (~5 MB), cached gzipped for ``players_ttl_hours``.

        A stale cache is used when the network fails; the result is ``{player_id: player_dict}``.
        """
        if not force_refresh:
            cached = self._read_players_cache(self.players_ttl_hours)
            if cached is not None:
                log.debug("players: using cache (%d entries)", len(cached))
                return cached
        try:
            data = await self.get_json("/players/nfl")
        except SleeperAPIError as e:
            stale = self._read_players_cache(None)
            if stale is not None:
                log.warning("players download failed (%s); using stale cache", e)
                return stale
            raise
        if not isinstance(data, dict):
            raise SleeperAPIError("unexpected players payload", url=self._url("/players/nfl", None))
        payload = {str(k): v for k, v in data.items() if isinstance(v, dict)}
        self._write_players_cache(payload)
        log.info("players: downloaded %d entries", len(payload))
        return payload

    async def get_trending(self, add_drop: str = "add", hours: int = 24, limit: int = 25) -> list[dict]:
        """``/players/nfl/trending/{add|drop}`` -> ``[{player_id, count}]``."""
        return await self._get_list(f"/players/nfl/trending/{add_drop}?lookback_hours={int(hours)}&limit={int(limit)}")

    # -- projections / stats ---------------------------------------------------
    async def _v1_then_v2(self, v1_path: str, v2_path: str, v2_params: list[tuple[str, Any]]) -> dict[str, dict]:
        """Try the v1 dict endpoint, then the v2 list endpoint; normalise both."""
        try:
            data = await self.get_json(v1_path)
            out = _normalise_projections(data)
            if out:
                return out
            log.info("sleeper %s returned no usable data; trying v2", v1_path)
        except SleeperAPIError as e:
            log.info("sleeper %s failed (%s); trying v2", v1_path, e)
        data = await self.get_json(v2_path, base=self.base_url_v2, params=v2_params)
        return _normalise_projections(data)

    @staticmethod
    def _v2_params(season_type: str, positions: Iterable[str], order_by: str | None) -> list[tuple[str, Any]]:
        params: list[tuple[str, Any]] = [("season_type", season_type)]
        params.extend(("position[]", p) for p in positions)
        if order_by:
            params.append(("order_by", order_by))
        return params

    async def get_season_projections(self, season: int, season_type: str = "regular",
                                     positions: tuple[str, ...] = SKILL_POSITIONS) -> dict[str, dict]:
        """Season projections (incl. ``adp_*`` keys) as ``{player_id: stats}``."""
        return await self._v1_then_v2(
            f"/projections/nfl/{season_type}/{season}",
            f"/projections/nfl/{season}",
            self._v2_params(season_type, positions, "adp_ppr"),
        )

    async def get_week_projections(self, season: int, week: int, season_type: str = "regular",
                                   positions: tuple[str, ...] = SKILL_POSITIONS) -> dict[str, dict]:
        return await self._v1_then_v2(
            f"/projections/nfl/{season_type}/{season}/{week}",
            f"/projections/nfl/{season}/{week}",
            self._v2_params(season_type, positions, None),
        )

    async def get_season_stats(self, season: int, season_type: str = "regular",
                               positions: tuple[str, ...] = SKILL_POSITIONS) -> dict[str, dict]:
        return await self._v1_then_v2(
            f"/stats/nfl/{season_type}/{season}",
            f"/stats/nfl/{season}",
            self._v2_params(season_type, positions, None),
        )

    async def get_week_stats(self, season: int, week: int, season_type: str = "regular",
                             positions: tuple[str, ...] = SKILL_POSITIONS) -> dict[str, dict]:
        return await self._v1_then_v2(
            f"/stats/nfl/{season_type}/{season}/{week}",
            f"/stats/nfl/{season}/{week}",
            self._v2_params(season_type, positions, None),
        )


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine from synchronous code (CLI one-offs), even if a loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # A loop is running in this thread (e.g. Jupyter): run in a helper thread.
    result: dict[str, Any] = {}

    def _runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as e:  # noqa: BLE001 - re-raised below
            result["error"] = e

    t = threading.Thread(target=_runner, name="draftadvisor-run-sync")
    t.start()
    t.join()
    if "error" in result:
        raise result["error"]
    return result["value"]
