"""Async ESPN fantasy API client (read-only ``lm-api-reads`` host).

* one keep-alive :class:`httpx.AsyncClient`; cookies ``espn_s2`` + ``SWID`` for private leagues;
* retries with exponential backoff on timeouts / connection errors / 5xx / 429, never on 4xx;
* 401 / 403 on the league endpoint first retries ``/leagueHistory/{id}?seasonId=`` (older seasons and
  some private leagues live there, as ``espn_api`` does), then raises :class:`EspnAccessDenied`;
* 404 raises :class:`EspnNotFound`; a JSON list body is normalised to its first element;
* requests are logged at DEBUG without cookies; cookie values never appear in logs or errors (the
  Fan API URL carries the SWID in its path, so URLs are logged and reported through :func:`redact_url`).

An :class:`httpx.AsyncClient` (``http=``) or a transport (``transport=``, e.g. ``httpx.MockTransport``)
can be injected for tests.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, Sequence

from urllib.parse import unquote
import httpx

from .constants import espn_base_url, fan_api_url

log = logging.getLogger(__name__)

__all__ = [
    "EspnAPIError",
    "EspnNotFound",
    "EspnAccessDenied",
    "EspnClient",
    "format_swid",
    "player_filter",
    "redact_url",
    "LEAGUE_VIEWS",
    "DRAFT_VIEWS",
]

_USER_AGENT = "draftadvisor/0.1 (+https://github.com/draftadvisor)"
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_DENIED_STATUSES = frozenset({401, 403})
_SWID_SEGMENT = re.compile(r"(/fans/)[^/?#]+")


def redact_url(url: str) -> str:
    """``url`` with the SWID segment of a Fan API URL (``/apis/v2/fans/{SWID}``) replaced by ``<swid>``,
    so a logged or reported URL never carries the cookie value."""
    return _SWID_SEGMENT.sub(r"\1<swid>", str(url))

#: Views of the one-off league fetch (settings, teams + owners, rosters).
LEAGUE_VIEWS: tuple[str, ...] = ("mSettings", "mTeam", "mRoster")
#: Views of the per-poll draft fetch (picks + the draft settings the commissioner may change).
DRAFT_VIEWS: tuple[str, ...] = ("mDraftDetail", "mSettings")


class EspnAPIError(Exception):
    """Any failure talking to ESPN after retries were exhausted."""

    def __init__(self, message: str, status_code: int | None = None, url: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.url = url

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        base = super().__str__()
        extra = " ".join(x for x in (f"status={self.status_code}" if self.status_code else "",
                                     f"url={self.url}" if self.url else "") if x)
        return f"{base} ({extra})" if extra else base


class EspnNotFound(EspnAPIError):
    """404: the league (or season) does not exist."""


class EspnAccessDenied(EspnAPIError):
    """401 / 403: a private league without (valid) ``espn_s2`` + ``SWID`` cookies."""


def format_swid(swid: str | None) -> str | None:
    """ESPN expects the SWID cookie with braces: ``{XXXXXXXX-XXXX-...}``.

    Browsers show the cookie either raw (``{...}``) or URL-encoded (``%7B...%7D``); both are accepted,
    as is a bare value without braces.
    """
    if swid is None:
        return None
    s = unquote(str(swid).strip())
    if not s:
        return None
    return s if s.startswith("{") and s.endswith("}") else "{" + s.strip("{}") + "}"


def player_filter(season: int, limit: int = 600, rank_type: str = "PPR") -> dict:
    """The ``x-fantasy-filter`` for ``kona_player_info``: top ``limit`` players by ESPN draft rank
    with actual (``00<season>``) and projected (``10<season>``) season stats."""
    return {
        "players": {
            "limit": int(limit),
            "sortDraftRanks": {"sortPriority": 100, "sortAsc": True, "value": rank_type},
            "filterRanksForRankTypes": {"value": [rank_type]},
            "filterStatsForTopScoringPeriodIds": {"value": 2, "additionalValue": [f"00{season}", f"10{season}"]},
        }
    }


class EspnClient:
    """Thin async wrapper over the ESPN v3 fantasy football API."""

    def __init__(self, espn_s2: str | None = None, swid: str | None = None, timeout: float = 10.0,
                 retries: int = 3, base_url: str | None = None, transport: httpx.AsyncBaseTransport | None = None,
                 *, http: httpx.AsyncClient | None = None, backoff_base: float = 0.5, backoff_max: float = 8.0,
                 fan_base_url: str | None = None):
        self.base_url = (base_url or espn_base_url()).rstrip("/")
        self.fan_base_url = fan_base_url
        self.retries = max(0, int(retries))
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        cookies: dict[str, str] = {}
        if espn_s2:
            cookies["espn_s2"] = str(espn_s2).strip()
        sw = format_swid(swid)
        if sw:
            cookies["SWID"] = sw
        self.swid = sw
        self.has_cookies = "espn_s2" in cookies and "SWID" in cookies
        self._owns_http = http is None
        self._http = http or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers={"User-Agent": _USER_AGENT, "Accept": "application/json"},
            cookies=cookies or None,
            transport=transport,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            follow_redirects=True,
        )
        if not self._owns_http:
            for k, v in cookies.items():
                self._http.cookies.set(k, v)
            if self._http.headers.get("User-Agent", "").startswith("python-httpx"):
                self._http.headers["User-Agent"] = _USER_AGENT
        self.request_count = 0
        self.last_latency_ms = 0.0

    # -- lifecycle ------------------------------------------------------------
    async def aclose(self) -> None:
        """Close the underlying HTTP client (only if we created it)."""
        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self) -> "EspnClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    # -- urls -----------------------------------------------------------------
    def league_url(self, league_id: str | int, season: int) -> str:
        return f"{self.base_url}/seasons/{int(season)}/segments/0/leagues/{league_id}"

    def history_url(self, league_id: str | int) -> str:
        return f"{self.base_url}/leagueHistory/{league_id}"

    # -- core request ---------------------------------------------------------
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

    async def _get(self, url: str, params: Sequence[tuple[str, Any]] | dict | None = None,
                   headers: dict[str, str] | None = None) -> httpx.Response:
        """GET with retries on transport errors / 429 / 5xx; returns the last response (any status).
        Logs and errors show the redacted URL (see :func:`redact_url`)."""
        last_exc: Exception | None = None
        last_resp: httpx.Response | None = None
        shown = redact_url(url)
        log.debug("espn GET %s params=%s", shown, params)
        for attempt in range(self.retries + 1):
            t0 = time.perf_counter()
            try:
                resp = await self._http.get(url, params=params, headers=headers)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                self.request_count += 1
                last_exc = e
                log.warning("espn %s attempt %d/%d failed: %s", shown, attempt + 1, self.retries + 1,
                            type(e).__name__)
                if attempt < self.retries:
                    await asyncio.sleep(self._backoff(attempt))
                continue
            self.request_count += 1
            self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
            last_resp = resp
            if resp.status_code in _RETRY_STATUSES:
                log.warning("espn %s -> HTTP %d (attempt %d/%d)", shown, resp.status_code, attempt + 1,
                            self.retries + 1)
                if attempt < self.retries:
                    await asyncio.sleep(self._backoff(attempt, resp))
                continue
            return resp
        if last_resp is not None:
            raise EspnAPIError(f"HTTP {last_resp.status_code} for {shown} after {self.retries + 1} attempts",
                               status_code=last_resp.status_code, url=shown)
        raise EspnAPIError(f"{type(last_exc).__name__} for {shown} after {self.retries + 1} attempts",
                           url=shown) from last_exc

    def _denied_message(self, league_id: Any) -> str:
        if not self.has_cookies:
            return (f"ESPN league {league_id} is private: add your espn_s2 and SWID cookies "
                    "(log in to fantasy.espn.com, copy both cookies from the browser).")
        return (f"ESPN league {league_id} cannot be accessed with the supplied espn_s2 / SWID cookies "
                "(expired cookies, or the account is not a member of this league).")

    def _payload(self, resp: httpx.Response, url: str, league_id: Any = None) -> Any:
        url = redact_url(url)
        status = resp.status_code
        if status == 404:
            raise EspnNotFound(f"not found: ESPN league {league_id}" if league_id is not None else f"not found: {url}",
                               status_code=404, url=url)
        if status in _DENIED_STATUSES:
            raise EspnAccessDenied(self._denied_message(league_id), status_code=status, url=url)
        if status >= 400:
            raise EspnAPIError(f"HTTP {status} for {url}", status_code=status, url=url)
        try:
            data = resp.json()
        except (json.JSONDecodeError, ValueError) as e:
            raise EspnAPIError(f"invalid JSON from {url}: {e}", status_code=status, url=url) from e
        if isinstance(data, list):
            if not data:
                raise EspnNotFound(f"empty response for {url}", status_code=status, url=url)
            data = data[0]
        return data

    async def get_json(self, url: str, params: Sequence[tuple[str, Any]] | dict | None = None,
                       headers: dict[str, str] | None = None) -> Any:
        """GET any URL and return the parsed JSON (list bodies -> first element)."""
        resp = await self._get(url, params=params, headers=headers)
        return self._payload(resp, url)

    # -- league endpoints -----------------------------------------------------
    async def get_league(self, league_id: str | int, season: int, views: Sequence[str],
                         headers: dict[str, str] | None = None) -> dict:
        """``/seasons/{season}/segments/0/leagues/{league_id}?view=...`` (falls back to ``/leagueHistory/``)."""
        params = [("view", v) for v in views]
        url = self.league_url(league_id, season)
        resp = await self._get(url, params=params, headers=headers)
        if resp.status_code in _DENIED_STATUSES:
            alt = self.history_url(league_id)
            log.info("espn league %s -> HTTP %d; trying %s", league_id, resp.status_code, alt)
            alt_resp = await self._get(alt, params=params + [("seasonId", int(season))], headers=headers)
            if alt_resp.status_code < 400:
                data = self._payload(alt_resp, alt, league_id)
                return data if isinstance(data, dict) else {}
            raise EspnAccessDenied(self._denied_message(league_id), status_code=resp.status_code, url=url)
        data = self._payload(resp, url, league_id)
        if not isinstance(data, dict):
            raise EspnAPIError(f"unexpected payload for league {league_id}: {type(data).__name__}", url=url)
        return data

    async def get_settings_and_teams(self, league_id: str | int, season: int) -> dict:
        """League settings, teams with owners and current rosters (views mSettings, mTeam, mRoster)."""
        return await self.get_league(league_id, season, LEAGUE_VIEWS)

    async def get_draft_detail(self, league_id: str | int, season: int) -> dict:
        """Draft picks + draft settings in one GET (views mDraftDetail, mSettings): the per-poll call."""
        return await self.get_league(league_id, season, DRAFT_VIEWS)

    async def get_players(self, league_id: str | int, season: int, limit: int = 600,
                          rank_type: str = "PPR") -> list[dict]:
        """``kona_player_info`` with the ``x-fantasy-filter`` header: the ``players`` list
        (id, name, position, team, ownership ADP, draft ranks, actual + projected season stats)."""
        headers = {"x-fantasy-filter": json.dumps(player_filter(season, limit, rank_type))}
        data = await self.get_league(league_id, season, ("kona_player_info",), headers=headers)
        players = data.get("players") if isinstance(data, dict) else None
        return [p for p in (players or []) if isinstance(p, dict)]

    # -- optional: my leagues ---------------------------------------------------
    async def get_fan_leagues(self, swid: str | None = None) -> list[dict]:
        """Best-effort list of ``{league_id, name, season, team_name}`` for the user's fantasy football
        leagues from the Fan API (``preferences[].metaData.entry``). ``[]`` when the shape is unknown."""
        sw = format_swid(swid) or self.swid
        if not sw:
            return []
        url = fan_api_url(sw, self.fan_base_url)
        try:
            data = await self.get_json(url)
        except EspnAPIError as e:
            log.info("fan api unavailable: %s", type(e).__name__)
            return []
        return parse_fan_leagues(data)


def parse_fan_leagues(data: Any) -> list[dict]:
    """Extract football league entries from a Fan API payload; unknown shapes yield ``[]``."""
    out: list[dict] = []
    if not isinstance(data, dict):
        return out
    seen: set[tuple[str, str]] = set()
    for pref in data.get("preferences") or []:
        if not isinstance(pref, dict):
            continue
        entry = (pref.get("metaData") or {}).get("entry") if isinstance(pref.get("metaData"), dict) else None
        if not isinstance(entry, dict):
            continue
        game = entry.get("gameId")
        abbrev = str(entry.get("abbrev") or "").upper()
        if game not in (None, 1, "1") and abbrev not in ("", "FFL"):
            continue
        season = entry.get("seasonId")
        meta = entry.get("entryMetadata") if isinstance(entry.get("entryMetadata"), dict) else {}
        team_name = meta.get("teamName") or entry.get("entryName")
        for g in entry.get("groups") or []:
            if not isinstance(g, dict) or g.get("groupId") is None:
                continue
            key = (str(g["groupId"]), str(season))
            if key in seen:
                continue
            seen.add(key)
            out.append({"league_id": str(g["groupId"]), "name": g.get("groupName") or "",
                        "season": int(season) if str(season).isdigit() else season, "team_name": team_name,
                        "team_id": entry.get("entryId")})
    return out
