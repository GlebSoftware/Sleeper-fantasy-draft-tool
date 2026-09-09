"""Stateless FastAPI server for the web app (DESIGN.md §3.9) - runs locally and on Vercel.

No background tasks and no per-process draft state: the browser owns the session
(league / draft ids, identity, mock pick list) and sends it with every request. The
server keeps only warm caches (bundle, Sleeper payloads, ECR, per-league contexts)
and reads the legacy research notes through :mod:`draftadvisor.research.store`
(read-only: nothing researches any more).

The only endpoint that calls the Anthropic API is ``POST /api/chat`` (the user sent a
message); it forwards the token usage and the estimated cost of every answer. Nothing
calls Claude on a poll, on a state change or in a loop.

Platforms: ``session.platform`` is ``"sleeper"`` (default) or ``"espn"``. An ESPN session
carries ``league_id`` + ``season`` (+ ``team_id``); the capture (settings, teams, rosters,
draft, player pool with ADP / projections) is identity-free and cached per league + season
(ESPN: + credential pair) for ``LEAGUE_TTL`` - who "me" is gets resolved per request
(:func:`resolve_me`), so the session the browser sends back after ``/api/session/start``
hits the same cache entry - and every poll is a single ESPN GET (picks + the current draft
settings, so a changed clock / pick order is followed). Private leagues need the ``espn_s2``
/ ``SWID`` cookies: headers ``X-ESPN-S2`` / ``X-ESPN-SWID`` (browser Settings), else the
server's ``ESPN_S2`` / ``ESPN_SWID`` env. One :class:`EspnClient` per credential pair, keyed
by a hash; cookie values never reach logs, cache keys or responses.

ESPN's REST API does **not** publish picks while a draft is running: the board is pre-populated with
one ``playerId: -1`` entry per pick and flushed in one go when the draft completes. The per-poll GET
therefore asks for the team rosters as well (views mDraftDetail + mSettings + mTeam + mRoster in one
request, ``POLL_ROSTERS``), because a drafted player may show up there first: such a player leaves the
pool and is attributed to his team, but never gets a pick number ESPN did not publish - the clock stays
board-derived. Every poll reports what ESPN actually returned in ``status.espn`` (status, redacted URL,
views, board entries / real picks, roster counts, the draft flags, capture age), ``GET
/api/espn/diagnose`` expands that into a readable report, and ``GET /api/state?force=1`` re-captures the
league (settings, teams, rosters, pick order, clock) and re-reads the board, rate-limited server-side.
No pick countdown is claimed (ESPN picks carry no timestamps).
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import threading
import time
import webbrowser
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Mapping, Sequence

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, PrivateAttr, ValidationError

from ..config import DEFAULT_SEASON, Settings, home_dir
from ..models import (DraftSettings, DraftState, LeagueSettings, Player, PlayerValue, Projection, Recommendation,
                      ResearchNote, RosterSpot)
from ..research.store import NoteStore, make_store

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
DEFAULT_PORT = 8787
AVAILABLE_TOP_N = 250
BEST_N = 8
LEAGUE_TTL = 10 * 60          # league / users / rosters / snapshot
TRADED_TTL = 30
CONTEXT_TTL = 10 * 60
VERSION = "2.2"
PLATFORMS: tuple[str, ...] = ("sleeper", "espn")
ESPN_CLIENT_MAX = 16          # distinct credential pairs kept warm per process
ESPN_PRIVATE_HINT = "ESPN says this league is private - add your espn_s2 and SWID cookies under Settings"
#: Ask ESPN for the team rosters on every poll (same request, four views). ``DRAFTADVISOR_ESPN_POLL_ROSTERS=0``
#: drops them without a deploy if ESPN ever chokes on them or the payload gets unreasonable.
POLL_ROSTERS = os.environ.get("DRAFTADVISOR_ESPN_POLL_ROSTERS", "1") != "0"
#: At most one forced re-capture per league per this many seconds; a forced call that arrives sooner
#: falls back to an ordinary (still uncached) poll instead of hammering ESPN.
FORCE_RECAPTURE_MIN_INTERVAL = 5.0
NO_STORE = {"Cache-Control": "no-store, max-age=0", "Vary": "X-ESPN-S2, X-ESPN-SWID, X-Access-Code"}


# ---------------------------------------------------------------------------
# Session model (what the browser sends)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EspnAuth:
    """ESPN private-league cookies for one request: headers first, else the server's env. Never logged."""

    espn_s2: str | None = None
    swid: str | None = None

    @property
    def present(self) -> bool:
        return bool(self.espn_s2 and self.swid)

    @property
    def key(self) -> str:
        """Hash of the credential pair (cache / client key); ``anon`` without cookies."""
        if not (self.espn_s2 or self.swid):
            return "anon"
        return hashlib.sha1(f"{self.espn_s2 or ''}\n{self.swid or ''}".encode()).hexdigest()[:12]

    @classmethod
    def from_request(cls, request: Request | None) -> "EspnAuth":
        h: Mapping[str, str] = request.headers if request is not None else {}
        s2 = (h.get("x-espn-s2") or os.environ.get("ESPN_S2") or "").strip() or None
        swid = (h.get("x-espn-swid") or os.environ.get("ESPN_SWID") or "").strip() or None
        return cls(s2, swid)


class MockConfig(BaseModel):
    teams: int = 12
    rounds: int = 15
    slot: int = 5
    scoring: str = "half_ppr"
    superflex: bool = False
    seed: int | None = None


def _query_int(q: Mapping[str, str], name: str) -> int | None:
    raw = q.get(name)
    if raw in (None, ""):
        return None
    try:
        return int(raw)
    except ValueError:
        raise HTTPException(400, f"{name} must be an integer")


def _mock_from_query(q: Mapping[str, str]) -> MockConfig:
    """Mock settings from GET params: ``mock_<key>`` (what the page sends) wins over a bare ``<key>``."""
    def raw(name: str) -> str | None:
        v = q.get(f"mock_{name}")
        return v if v not in (None, "") else q.get(name)

    def num(name: str, default: int) -> int:
        v = raw(name)
        if v in (None, ""):
            return default
        try:
            return int(v)
        except ValueError:
            raise HTTPException(400, f"{name} must be an integer")

    seed = raw("seed")
    return MockConfig(teams=num("teams", 12), rounds=num("rounds", 15), slot=num("slot", 5),
                      scoring=raw("scoring") or "half_ppr", superflex=(raw("superflex") or "false").lower() in ("1", "true"),
                      seed=num("seed", 0) if seed not in (None, "") else None)


class Session(BaseModel):
    mode: str = "live"                      # live | mock
    platform: str = "sleeper"               # sleeper | espn
    draft_id: str | None = None
    league_id: str | None = None            # espn: the leagueId= number of the league URL
    season: int | None = None               # espn: league season (default DEFAULT_SEASON)
    username: str | None = None
    user_id: str | None = None              # espn: str(team id) once resolved
    slot: int | None = None
    team_id: int | None = None              # espn: my team id
    use_claude: bool = True                 # accepted for compatibility; triggers nothing (chat is user-initiated)
    mock: MockConfig | None = None
    picks: list[str] = []                   # mock only: player ids in pick order
    _espn: EspnAuth | None = PrivateAttr(default=None)

    @classmethod
    def from_query(cls, request: Request) -> "Session":
        """The session of a GET request: the ``session`` JSON parameter when the page sends one, else the
        flat parameters (mock settings as ``mock_<key>`` or bare ``<key>``, picks as a comma list)."""
        q = request.query_params
        blob = q.get("session")
        if blob:
            try:
                data = json.loads(blob)
            except ValueError:
                data = None
            if not isinstance(data, Mapping):
                raise HTTPException(400, "session must be a JSON object")
            try:
                sess = cls.model_validate(data)
            except ValidationError as e:
                raise HTTPException(400, f"invalid session: {e.errors()[0].get('msg') if e.errors() else e}")
            return sess.with_auth(request)
        mode = q.get("mode", "live")
        mock = _mock_from_query(q) if mode == "mock" else None
        picks = [p for p in (q.get("picks") or "").split(",") if p]
        sess = cls(mode=mode, platform=(q.get("platform") or "sleeper").lower(),
                   draft_id=q.get("draft_id") or None, league_id=q.get("league_id") or None,
                   season=_query_int(q, "season"), username=q.get("username") or None, user_id=q.get("user_id") or None,
                   slot=_query_int(q, "slot") if mode != "mock" else None, team_id=_query_int(q, "team_id"),
                   use_claude=q.get("use_claude", "true").lower() not in ("0", "false"), mock=mock, picks=picks)
        return sess.with_auth(request)

    # -- ESPN identity ----------------------------------------------------------
    def with_auth(self, request: Request | None) -> "Session":
        """Attach the request's ESPN cookies (kept out of ``model_dump`` and therefore out of responses)."""
        self._espn = EspnAuth.from_request(request)
        return self

    @property
    def espn_auth(self) -> EspnAuth:
        return self._espn if self._espn is not None else EspnAuth.from_request(None)

    @property
    def espn_season(self) -> int:
        return int(self.season or DEFAULT_SEASON)

    @property
    def espn_team_id(self) -> int | None:
        """``team_id``, else ``user_id`` when it is a team id (a resumed session stores both)."""
        if self.team_id is not None:
            return int(self.team_id)
        return int(self.user_id) if self.user_id and self.user_id.isdigit() else None

    def espn_identity(self) -> dict[str, Any]:
        """Who am I for :func:`draftadvisor.espn.parsing.resolve_my_team`: the team id is the stable identity,
        so ``slot`` only counts when no team is known (the commissioner may reorder the draft)."""
        tid = self.espn_team_id
        return {"swid": self.espn_auth.swid, "team_id": tid, "slot": self.slot if tid is None else None,
                "username": self.username if tid is None else None}


# ---------------------------------------------------------------------------
# Warm caches
# ---------------------------------------------------------------------------


class _Cache:
    def __init__(self) -> None:
        self._d: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str, ttl: float) -> Any | None:
        with self._lock:
            hit = self._d.get(key)
        return hit[1] if hit and time.time() - hit[0] < ttl else None

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._d[key] = (time.time(), value)

    def pop(self, key: str) -> None:
        with self._lock:
            self._d.pop(key, None)


CACHE = _Cache()
_STORE: NoteStore | None = None
_SLEEPER: Any = None
_ESPN_CLIENTS: dict[str, "_PooledClient"] = {}   # EspnAuth.key -> pooled EspnClient, least recently used first
_CLAUDE_CLIENTS: dict[str, Any] = {}
_LOCKS: dict[str, asyncio.Lock] = {}


def store() -> NoteStore:
    global _STORE
    if _STORE is None:
        _STORE = make_store()
    return _STORE


def sleeper() -> Any:
    global _SLEEPER
    if _SLEEPER is None:
        from ..sleeper.client import SleeperClient

        _SLEEPER = SleeperClient()
    return _SLEEPER


@dataclass
class _PooledClient:
    """An :class:`EspnClient` of the pool with its in-flight request count; ``retired`` once evicted."""

    client: Any
    in_flight: int = 0
    retired: bool = False


async def _close_quietly(client: Any) -> None:
    try:
        await client.aclose()
    except Exception:  # noqa: BLE001
        pass


@asynccontextmanager
async def espn_client(auth: EspnAuth) -> AsyncIterator[Any]:
    """One :class:`EspnClient` per credential pair, bounded to ``ESPN_CLIENT_MAX`` (the least recently
    used is evicted first). A client evicted while a request is still using it is closed when that
    request finishes, never underneath it."""
    from ..espn.client import EspnClient

    entry = _ESPN_CLIENTS.pop(auth.key, None)
    if entry is None:
        while len(_ESPN_CLIENTS) >= ESPN_CLIENT_MAX:
            oldest = _ESPN_CLIENTS.pop(next(iter(_ESPN_CLIENTS)))
            oldest.retired = True
            if oldest.in_flight == 0:
                await _close_quietly(oldest.client)
        entry = _PooledClient(EspnClient(espn_s2=auth.espn_s2, swid=auth.swid, retries=2))
    _ESPN_CLIENTS[auth.key] = entry                  # (re)inserted last: most recently used
    entry.in_flight += 1
    try:
        yield entry.client
    finally:
        entry.in_flight -= 1
        if entry.retired and entry.in_flight == 0:
            await _close_quietly(entry.client)


def espn_http_error(e: Exception, league_id: Any, season: int, auth: EspnAuth | None = None) -> HTTPException:
    """ESPN client failures -> HTTP: private league 401, unknown league 404, anything else 502."""
    from ..espn.client import EspnAccessDenied, EspnAPIError, EspnNotFound

    if isinstance(e, EspnAccessDenied):
        if auth is not None and auth.present:
            return HTTPException(401, ESPN_PRIVATE_HINT + " (the cookies you set were rejected: they expire, and the "
                                      "account must be a member of this league)")
        return HTTPException(401, ESPN_PRIVATE_HINT)
    if isinstance(e, EspnNotFound):
        return HTTPException(404, f"ESPN has no league {league_id} for season {season}")
    if isinstance(e, EspnAPIError):
        return HTTPException(502, f"ESPN API error: {e}")
    return HTTPException(502, f"could not reach the ESPN API: {e}")


def lock_for(key: str) -> asyncio.Lock:
    if key not in _LOCKS:
        _LOCKS[key] = asyncio.Lock()
    return _LOCKS[key]


def notes_dict() -> dict[str, ResearchNote]:
    out = {}
    for pid, d in store().get_all().items():
        try:
            out[pid] = ResearchNote.from_dict(d)
        except Exception:  # noqa: BLE001
            continue
    return out


def claude_for(api_key: str | None) -> Any:
    """One :class:`ClaudeChat` per API key (header key, else the server's); disabled without a key."""
    from ..research.claude import ClaudeChat

    key = api_key or os.environ.get("ANTHROPIC_API_KEY") or ""
    h = hashlib.sha1(key.encode()).hexdigest()[:10] if key else "none"
    r = _CLAUDE_CLIENTS.get(h)
    if r is None:
        r = ClaudeChat(api_key=key or None, cache_dir=home_dir() / "research")
        _CLAUDE_CLIENTS[h] = r
    return r


def chat_model() -> str:
    """The default chat model: ``DRAFTADVISOR_CHAT_MODEL`` when it is a priced model, else ``CHAT_MODEL``
    (only models in the price table may be billed, see ``POST /api/chat``)."""
    from ..research.claude import CHAT_MODEL, model_prices

    env = (os.environ.get("DRAFTADVISOR_CHAT_MODEL") or "").strip()
    if env and model_prices(env) is None:
        log.warning("DRAFTADVISOR_CHAT_MODEL=%r is not a priced model; using %s", env, CHAT_MODEL)
        return CHAT_MODEL
    return env or CHAT_MODEL


# ---------------------------------------------------------------------------
# Access control / keys
# ---------------------------------------------------------------------------


def access_code_required() -> bool:
    return bool(os.environ.get("DRAFTADVISOR_ACCESS_CODE"))


async def require_access(request: Request) -> None:
    code = os.environ.get("DRAFTADVISOR_ACCESS_CODE")
    if code and request.headers.get("x-access-code", "") != code:
        raise HTTPException(401, "access code required (set it under Settings)")


def api_key_from(request: Request) -> str | None:
    return request.headers.get("x-anthropic-key") or None


# ---------------------------------------------------------------------------
# Live inputs
# ---------------------------------------------------------------------------


NEGATIVE_TTL = 300     # remember a failed Sleeper/ECR fetch this long instead of retrying every request


async def sleeper_players() -> dict | None:
    hit = CACHE.get("sleeper_players", 6 * 3600)
    if hit is not None:
        return hit
    if CACHE.get("sleeper_players:fail", NEGATIVE_TTL):
        return None
    async with lock_for("sleeper_players"):
        hit = CACHE.get("sleeper_players", 6 * 3600)
        if hit is not None:
            return hit
        try:
            payload = await sleeper().get_players()
            CACHE.set("sleeper_players", payload)
            CACHE.set("sleeper_ok", True)
            return payload
        except Exception as e:  # noqa: BLE001
            log.warning("Sleeper players unavailable: %s", e)
            CACHE.set("sleeper_players:fail", str(e))
            return None


async def sleeper_projections(season: int) -> dict | None:
    hit = CACHE.get("sleeper_proj", 3600)
    if hit is not None:
        return hit
    if CACHE.get("sleeper_proj:fail", NEGATIVE_TTL):
        return None
    async with lock_for("sleeper_proj"):
        hit = CACHE.get("sleeper_proj", 3600)
        if hit is not None:
            return hit
        try:
            proj = await sleeper().get_season_projections(season)
            CACHE.set("sleeper_proj", proj)
            return proj
        except Exception as e:  # noqa: BLE001
            log.warning("Sleeper projections unavailable: %s", e)
            CACHE.set("sleeper_proj:fail", str(e))
            return None


async def ecr_rows() -> list[dict]:
    from ..lean import fetch_ecr

    if CACHE.get("ecr:fail", NEGATIVE_TTL):
        return []
    try:
        return await fetch_ecr()
    except Exception as e:  # noqa: BLE001
        log.warning("ECR unavailable (%s): using the bundle's snapshot", e)
        CACHE.set("ecr:fail", str(e))
        return []


@dataclass
class LeagueBundle:
    """Everything cached for one league capture: league/draft/users/rosters/snapshot + the lean context.

    The capture is identity-free (nobody is "me" in ``snapshot``; :func:`resolve_me` resolves the asking
    manager per request), so one bundle serves every session of the same league and the session the
    browser sends back after ``/api/session/start`` hits the same cache entry. ``key`` is the cache key
    (ESPN: + the credential hash, so a private league's data never serves another cookie pair);
    ``league_key`` keys the shared lean context (``ctx:<league_key>:<fingerprint>``). ESPN bundles also
    carry the id map the capture used, the fetched player pool and its names (``espn_id -> {name, ...}``).
    """

    key: str
    league: LeagueSettings
    draft: DraftSettings | None
    snapshot: Any
    league_raw: dict | None
    users_raw: list
    rosters_raw: list
    traded_raw: list = field(default_factory=list)
    traded_at: float = 0.0
    ctx: Any = None
    ctx_fp: str = ""
    built_at: float = field(default_factory=time.time)
    platform: str = "sleeper"
    season: int = DEFAULT_SEASON
    league_key: str = ""
    id_map: Any = None                                  # espn: EspnIdMap the capture used (the board uses ctx.id_map)
    espn_players: list = field(default_factory=list)    # espn: kona_player_info entries (ADP / projections)
    espn_names: dict = field(default_factory=dict)      # espn: espn_id -> player fields


def _league_key(sess: Session) -> str:
    """Identity-free key of a league: ESPN league + season (the draft id derives from them); Sleeper draft
    id, else league id (the capture is aliased under both once the other one is known)."""
    if sess.platform == "espn":
        return f"espn::{sess.league_id or ''}:{sess.espn_season}"
    return f"sleeper:{sess.draft_id or ''}:{'' if sess.draft_id else (sess.league_id or '')}:"


def _capture_key(sess: Session) -> str:
    """Cache key of the capture: the league key plus, for ESPN, the credential hash (never the cookies)."""
    key = f"league:{_league_key(sess)}"
    return f"{key}:{sess.espn_auth.key}" if sess.platform == "espn" else key


def _bundle_keys(lb: LeagueBundle, key: str) -> list[str]:
    """Every cache key a bundle answers to: the requested one plus, for Sleeper, the resolved draft id and
    league id (a session started by league id polls with the draft id filled in, and vice versa)."""
    keys = [key]
    if lb.platform == "sleeper":
        if lb.draft is not None and lb.draft.draft_id:
            keys.append(f"league:sleeper:{lb.draft.draft_id}::")
        league_id = (lb.league_raw or {}).get("league_id")
        if league_id:
            keys.append(f"league:sleeper::{league_id}:")
    return list(dict.fromkeys(keys))


async def resolve_league(sess: Session) -> LeagueBundle:
    """Capture (cached per league, identity-free) the league behind a session; raises HTTP errors for bad ids."""
    if sess.platform not in PLATFORMS:
        raise HTTPException(400, f"platform must be one of {', '.join(PLATFORMS)}")
    key = _capture_key(sess)
    hit = CACHE.get(key, LEAGUE_TTL)
    if hit is not None:
        return hit
    async with lock_for(key):
        hit = CACHE.get(key, LEAGUE_TTL)
        if hit is not None:
            return hit
        lb = await (_capture_espn(sess, key) if sess.platform == "espn" else _capture_sleeper(sess, key))
        lb.league_key = _league_key(sess)
        if lb.platform == "sleeper" and lb.draft is not None and lb.draft.draft_id:
            lb.league_key = f"sleeper:{lb.draft.draft_id}::"          # one context per draft, however it was asked for
        for k in _bundle_keys(lb, key):
            CACHE.set(k, lb)
        return lb


def drop_capture(sess: Session) -> bool:
    """Forget the cached capture of this session's league so the next :func:`resolve_league` re-reads
    settings, teams, rosters, the pick order and the clock from the platform.

    Rate-limited to one forced re-capture per league per :data:`FORCE_RECAPTURE_MIN_INTERVAL` seconds
    (an ESPN re-capture is 3 GETs, not 1): returns ``False`` when the caller must make do with an
    ordinary - still uncached - poll, which is what holding the Refresh button gets.
    """
    key = _capture_key(sess)
    guard = f"force:{key}"
    if CACHE.get(guard, FORCE_RECAPTURE_MIN_INTERVAL) is not None:
        log.info("force refresh for %s came within %.0f s of the last one; polling without re-capturing",
                 _league_key(sess), FORCE_RECAPTURE_MIN_INTERVAL)
        return False
    CACHE.set(guard, time.time())
    lb = CACHE.get(key, LEAGUE_TTL)
    for k in (_bundle_keys(lb, key) if isinstance(lb, LeagueBundle) else [key]):
        CACHE.pop(k)
    return True


def _espn_me(lb: LeagueBundle, sess: Session) -> tuple[str | None, int | None]:
    """``(team id, slot)`` of the session's manager in an ESPN capture (precedence: slot > team id > SWID >
    username, see :func:`draftadvisor.espn.parsing.resolve_my_team`); ``ValueError`` names the teams when
    the given team id / username matches none of them."""
    from ..espn.parsing import resolve_my_team

    ident = sess.espn_identity()
    managers = lb.snapshot.managers if lb.snapshot is not None else {}
    my_uid, my_slot = resolve_my_team(lb.draft, managers, lb.league_raw, **ident)
    if (ident["username"] or ident["team_id"] is not None) and my_uid is None:
        teams = ", ".join(sorted(f"{m.team_name or m.display_name} (team {m.user_id}, {m.display_name})"
                                 for m in managers.values()))
        raise ValueError(f"could not find {ident['username'] or ident['team_id']!r} among the ESPN league's teams: {teams}")
    return my_uid, my_slot


def resolve_me(lb: LeagueBundle, sess: Session) -> tuple[str | None, int | None]:
    """``(my user / team id, my slot)`` of the session's manager in a shared, identity-free capture.

    400 when the given username / user id / team id matches nobody or the slot lies outside the draft;
    ``(None, None)`` is a spectator. Pure and cheap: called on every request that names a league.
    """
    from ..capture import resolve_identity

    if lb.draft is None:
        return None, None
    if sess.slot is not None and lb.draft.teams and not 1 <= int(sess.slot) <= lb.draft.teams:
        raise HTTPException(400, f"slot must be between 1 and {lb.draft.teams}")
    try:
        if lb.platform == "espn":
            return _espn_me(lb, sess)
        managers = lb.snapshot.managers if lb.snapshot is not None else {}
        return resolve_identity(lb.draft, managers, lb.users_raw, username=sess.username, user_id=sess.user_id,
                                slot=sess.slot)
    except ValueError as e:
        raise HTTPException(400, str(e))


async def _capture_sleeper(sess: Session, key: str) -> LeagueBundle:
    from ..capture import capture_league
    from ..sleeper.client import SleeperAPIError, SleeperNotFound

    try:
        snap = await capture_league(sleeper(), sess.league_id, sess.draft_id, season=DEFAULT_SEASON, save=False)
    except SleeperNotFound as e:
        raise HTTPException(404, f"Sleeper has no league/draft with that id ({e})")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except SleeperAPIError as e:
        raise HTTPException(502, f"Sleeper API error: {e}")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"could not reach the Sleeper API: {e}")
    if snap.draft is None and snap.league is None:
        raise HTTPException(404, "no league or draft found")
    league = snap.league
    if league is None:  # draft without a league (e.g. mock draft on Sleeper): derive scoring from metadata
        from ..lean import default_league

        league = default_league({"ppr": "ppr", "half_ppr": "half_ppr", "std": "std"}.get(snap.draft.scoring_type or "", "half_ppr"),
                                snap.draft.teams)
    raw = snap.raw or {}
    lb = LeagueBundle(key=key, league=league, draft=snap.draft, snapshot=snap, league_raw=raw.get("league"),
                      users_raw=raw.get("users") or [], rosters_raw=raw.get("rosters") or [],
                      traded_raw=raw.get("traded_picks") or [], traded_at=time.time())
    CACHE.set("sleeper_ok", True)
    return lb


async def espn_id_map() -> Any:
    """ESPN id -> canonical id map over the player universe (the Sleeper payload when reachable, else the
    bundle), cached alongside the payload; the capture's snapshot resolves through it. The board and every
    poll use the context's own map instead (:func:`league_context`), which is indexed over the very
    universe the board shows."""
    from ..espn.ids import EspnIdMap
    from ..lean import get_bundle, players_from_sleeper_payload

    payload = await sleeper_players()
    key = "espn_idmap:" + ("sleeper" if payload else "bundle")
    hit = CACHE.get(key, 6 * 3600)
    if hit is not None:
        return hit
    async with lock_for(key):
        hit = CACHE.get(key, 6 * 3600)
        if hit is not None:
            return hit
        bundle = get_bundle()

        def _build() -> Any:
            universe = players_from_sleeper_payload(payload, bundle.players) if payload else bundle.players
            return EspnIdMap.from_players(universe)

        id_map = await asyncio.to_thread(_build)
        CACHE.set(key, id_map)
        return id_map


async def _capture_espn(sess: Session, key: str) -> LeagueBundle:
    """Settings / teams / rosters, draft and player pool of one ESPN league (3 GETs, cached for LEAGUE_TTL)."""
    from ..espn.capture import capture_espn_league, espn_names

    league_id, season, auth = sess.league_id, sess.espn_season, sess.espn_auth
    if not league_id:
        raise HTTPException(400, "league_id required for an ESPN league")
    id_map = await espn_id_map()
    async with espn_client(auth) as client:
        try:
            snap = await capture_espn_league(client, league_id, season, id_map=id_map, save=False)
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception as e:  # noqa: BLE001
            raise espn_http_error(e, league_id, season, auth)
    if snap.league is None or snap.draft is None:
        raise HTTPException(404, f"ESPN has no league {league_id} for season {season}")
    raw = snap.raw or {}
    pool = list(raw.get("players") or [])
    return LeagueBundle(key=key, league=snap.league, draft=snap.draft, snapshot=snap, league_raw=raw.get("league"),
                        users_raw=[], rosters_raw=raw.get("rosters") or [], platform="espn", season=season,
                        id_map=id_map, espn_players=pool, espn_names=espn_names(pool))


def espn_context_inputs(lb: LeagueBundle, id_map: Any) -> dict[str, Any]:
    """The ESPN-specific inputs of :func:`draftadvisor.lean.build_lean_context` for one league.

    ``id_map`` indexes the universe the context is built from (see :func:`_build_context`), so the pool's
    ESPN ADP and projected season lines are keyed by the ids the board uses; players of the pool or of a
    league roster that resolve to a synthetic ``espn:<id>`` become placeholder :class:`Player` objects
    (with their ESPN ADP / projection) so the board never shows a blank name.
    """
    from ..espn.capture import espn_adp, espn_projections
    from ..espn.ids import EspnIdMap, placeholder_player

    entries: list[Mapping[str, Any]] = list(lb.espn_players)
    for team in (lb.league_raw or {}).get("teams") or []:
        if isinstance(team, Mapping):
            entries.extend(e for e in ((team.get("roster") or {}).get("entries") or []) if isinstance(e, Mapping))
    extra: dict[str, Player] = {}
    for entry in entries:
        pid = id_map.resolve_player_json(entry)
        if EspnIdMap.is_synthetic(pid) and pid not in extra:
            extra[pid] = placeholder_player(entry, pid)
    return {"adp_override": espn_adp(lb.espn_players, id_map), "adp_source": "espn",
            "proj_fallback": espn_projections(lb.espn_players, id_map, lb.season), "proj_fallback_source": "espn",
            "extra_players": extra}


def ensure_pick_players(players: dict[str, Player], st: DraftState) -> int:
    """Placeholder :class:`Player` for every pick whose id the universe does not know (an ESPN player
    outside the fetched pool and off every roster); returns how many were added."""
    from ..espn.ids import placeholder_player

    n = 0
    for p in st.picks:
        if p.player_id in players:
            continue
        md = p.metadata or {}
        pl = placeholder_player({"id": md.get("espn_id") or p.player_id, "firstName": md.get("first_name"),
                                 "lastName": md.get("last_name")}, p.player_id)
        pl.position, pl.team = md.get("position") or "UNK", md.get("team") or None
        pl.fantasy_positions = (pl.position,)
        players[p.player_id] = pl
        n += 1
    return n


def ensure_roster_players(players: dict[str, Player], st: DraftState) -> int:
    """Placeholder :class:`Player` for every roster-only player the universe does not know (an ESPN
    player the poll found on a team roster but who is in neither the fetched pool nor the capture);
    returns how many were added. Without it such a player renders as ``espn:<id>``."""
    from ..espn.ids import placeholder_player

    n = 0
    for s in st.roster_spots:
        if s.player_id in players:
            continue
        first, _, last = (s.name or "").partition(" ")
        pl = placeholder_player({"id": s.espn_id or s.player_id, "firstName": first or None, "lastName": last or None},
                                s.player_id)
        if s.name:
            pl.name = s.name
        pl.position = s.position or pl.position or "UNK"
        pl.fantasy_positions = (pl.position,)
        players[s.player_id] = pl
        n += 1
    return n


def _build_context(lb: LeagueBundle, settings: Settings, players_payload: dict | None, proj: dict | None,
                   rows: list[dict], notes: Mapping[str, ResearchNote]) -> Any:
    """Blocking build of the lean context. For ESPN the id map is indexed over the very universe the context
    is built from (and stored on it as ``ctx.id_map``), so picks, rosters and the player pool resolve into
    the ids the board shows - whichever source (Sleeper payload or bundle) the universe came from."""
    from ..lean import build_lean_context, get_bundle, lean_universe

    bundle = get_bundle()
    extra: dict[str, Any] = {}
    if lb.platform == "espn":
        from ..espn.ids import EspnIdMap

        universe, source = lean_universe(bundle, players_payload)
        id_map = EspnIdMap.from_players(universe)
        extra = dict(espn_context_inputs(lb, id_map), players=universe, players_source=source, id_map=id_map)
    return build_lean_context(bundle, lb.league, lb.draft, sleeper_players=players_payload, sleeper_proj=proj,
                              ecr_rows=rows, notes=notes, settings=settings, **extra)


async def league_context(lb: LeagueBundle, sess: Session, *, force: bool = False) -> Any:
    """Lean context for the league (rebuilt when notes / inputs change or after CONTEXT_TTL); identity-free,
    so every session of the same league shares one build. ``force`` rebuilds it from the fresh capture."""
    from ..lean import context_fingerprint

    notes = notes_dict()
    proj = await sleeper_projections(DEFAULT_SEASON)
    players_payload = await sleeper_players()
    fp = context_fingerprint(lb.league, bool(proj), bool(players_payload), len(notes))
    if not force and lb.ctx is not None and lb.ctx_fp == fp and time.time() - lb.ctx.built_at < CONTEXT_TTL:
        return lb.ctx
    ctx_key = f"ctx:{lb.league_key or lb.key}:{fp}"
    async with lock_for(ctx_key):
        ctx = None if force else CACHE.get(ctx_key, CONTEXT_TTL)
        if ctx is None:
            rows = await ecr_rows()
            settings = Settings.from_env(league_id=lb.league.league_id, draft_id=lb.draft.draft_id if lb.draft else None,
                                         username=sess.username, user_id=sess.user_id, slot=sess.slot)
            ctx = await asyncio.to_thread(_build_context, lb, settings, players_payload, proj, rows, notes)
            CACHE.set(ctx_key, ctx)
        lb.ctx, lb.ctx_fp = ctx, fp
        return ctx


def espn_diagnostics(info: Mapping[str, Any], poll_json: Mapping[str, Any] | None, st: DraftState,
                     lb: LeagueBundle, *, forced: bool = False) -> dict:
    """What ESPN actually returned on this poll: counts and flags only, never a payload dump and never
    a cookie (the URL is redacted by the client). Built entirely from this one request - the server is
    stateless, so nothing here compares against a previous poll.
    """
    from ..espn.ids import is_real_player_id
    from ..espn.parsing import DRAFT_ACQUISITIONS, draft_detail_of, draft_settings_of, merge_league_payload

    poll = poll_json if isinstance(poll_json, Mapping) else {}
    detail = draft_detail_of(poll)
    entries = [p for p in (detail.get("picks") or []) if isinstance(p, Mapping)]
    real = sum(1 for p in entries if is_real_player_id(p.get("playerId")))
    merged = merge_league_payload(lb.league_raw or {}, poll)
    teams = [t for t in (merged.get("teams") or []) if isinstance(t, Mapping)]
    with_rosters = sum(1 for t in teams if ((t.get("roster") or {}).get("entries") or []))
    spots = st.roster_spots
    drafted_entries = sum(1 for s in spots if (s.acquisition_type or "").upper() in DRAFT_ACQUISITIONS)
    fresh = sum(1 for s in spots if s.is_fresh)
    roster_only = len(st.roster_only_ids)
    source = ("board+rosters" if (real and roster_only) else "board" if real else "rosters" if roster_only else "none")
    ds = draft_settings_of(poll, lb.league_raw or {})
    return {
        "fetched_at": info.get("read_at"), "http_status": info.get("http_status"), "url": info.get("url"),
        "views": list(info.get("views") or []), "bytes": info.get("bytes"), "latency_ms": info.get("latency_ms"),
        "cache_headers": dict(info.get("cache_headers") or {}), "used_history": bool(info.get("used_history")),
        "season_returned": info.get("season_returned"),
        "has_draft_detail": bool(detail),
        "board_entries": len(entries), "board_picks": real, "board_placeholders": len(entries) - real,
        "roster_drafted": drafted_entries, "rostered_total": len(spots), "roster_fresh": fresh,
        "roster_undated": max(0, drafted_entries - fresh), "roster_only": roster_only,
        "teams_with_rosters": with_rosters, "roster_confidence": st.roster_confidence,
        "drafted": bool(detail.get("drafted")), "in_progress": bool(detail.get("inProgress")),
        "complete_date": detail.get("completeDate"), "draft_status": st.draft.status,
        "draft_date": ds.get("date"), "league_sub_type": ds.get("leagueSubType"),
        "league_id": lb.league.league_id if lb.league is not None else None, "season": lb.season,
        "capture_age_s": _f(time.time() - lb.built_at, 1),
        "source": source, "forced": bool(forced), "poll_rosters": POLL_ROSTERS,
    }


def espn_explanation(diag: Mapping[str, Any]) -> str:
    """One plain-English paragraph saying what the numbers mean (the Diagnostics panel shows it verbatim)."""
    board, entries = int(diag.get("board_picks") or 0), int(diag.get("board_entries") or 0)
    fresh, drafted_on_rosters = int(diag.get("roster_fresh") or 0), int(diag.get("roster_drafted") or 0)
    sub = str(diag.get("league_sub_type") or "")
    if not diag.get("has_draft_detail"):
        return ("ESPN answered without a draftDetail block at all: there is no draft board in the payload. "
                "That is a degraded or throttled answer, not an empty draft - try Refresh.")
    if board == 0 and fresh == 0:
        msg = (f"ESPN returned a {entries}-entry board with 0 real picks and {drafted_on_rosters} drafted players on "
               "team rosters: ESPN is not publishing this draft. ESPN's REST API does not update the draft board "
               "while a draft is in progress - every pick appears at once when the draft finishes - so an empty "
               "board here is what a working poller sees during an ESPN draft.")
    elif board == 0:
        conf = str(diag.get("roster_confidence") or "none")
        where = {"exact": "Their pick numbers were reconstructed from the draft order and the order they were added.",
                 "team": "Each team's pick numbers are known, the order within a team is a guess.",
                 }.get(conf, "They carry no pick number: ESPN's board is the only place pick numbers come from.")
        msg = (f"ESPN's board is still empty ({entries} placeholder entries) but {fresh} freshly drafted players are "
               "already on team rosters, so the draft is running and those players have been removed from the pool. "
               + where)
    else:
        msg = (f"ESPN's board holds {board} of {entries} entries as real picks"
               + (f", and {fresh} drafted players are on team rosters." if fresh else "."))
    if sub in ("MOCKDRAFT_LOBBY", "CUSTOM_MOCK"):
        msg += (f" This room is a {sub}: an ESPN mock / practice draft is a throwaway room whose picks may never be "
                "written back to any league, so they may never appear here at all.")
    if diag.get("used_history"):
        msg += " The answer came from ESPN's leagueHistory endpoint (the live league endpoint refused the request)."
    age = diag.get("cache_headers", {}).get("age")
    if age and str(age) != "0":
        msg += f" A CDN answered this request (Age: {age}s), so it may be up to that old."
    return msg


async def espn_poll(lb: LeagueBundle, sess: Session, id_map: Any = None, *,
                    forced: bool = False) -> tuple[DraftState, dict, dict]:
    """One ESPN GET -> ``(state, diagnostics, the payload)``.

    The single per-poll request asks for the board, the draft settings **and** the team rosters (four
    views, one request): ESPN's board stays empty for the whole of a live draft, so the rosters are the
    only live signal it does publish. ``id_map`` is the context's map (the board's universe); the
    capture's map is only the fallback.
    """
    from ..espn.parsing import state_from_espn

    league_id = sess.league_id or lb.league.league_id or ""
    info: dict[str, Any] = {}
    async with espn_client(sess.espn_auth) as client:
        try:
            poll_json = await client.get_draft_live(league_id, lb.season, rosters=POLL_ROSTERS, no_cache=forced,
                                                    info=info)
        except Exception as e:  # noqa: BLE001
            raise espn_http_error(e, league_id, lb.season, sess.espn_auth)
    st = state_from_espn(lb.league_raw or {}, poll_json, id_map if id_map is not None else lb.id_map,
                         names=lb.espn_names, **sess.espn_identity())
    return st, espn_diagnostics(info, poll_json, st, lb, forced=forced), poll_json


async def espn_live_state(lb: LeagueBundle, sess: Session, id_map: Any = None, *,
                          forced: bool = False) -> tuple[DraftState, dict]:
    """:func:`espn_poll` without the raw payload."""
    st, diag, _ = await espn_poll(lb, sess, id_map, forced=forced)
    return st, diag


async def live_state(lb: LeagueBundle, sess: Session, id_map: Any = None, *,
                     forced: bool = False) -> tuple[DraftState, dict]:
    """Fetch the draft + picks right now and build the DraftState (2 Sleeper GETs, 1 ESPN GET).
    Returns the state and, for ESPN, what that GET returned (``{}`` for Sleeper)."""
    from ..sleeper.client import SleeperAPIError, SleeperNotFound
    from ..sleeper.parsing import state_from_sleeper

    if lb.platform == "espn":
        return await espn_live_state(lb, sess, id_map, forced=forced)
    draft_id = sess.draft_id or (lb.draft.draft_id if lb.draft else None)
    if not draft_id:
        raise HTTPException(400, "no draft id")
    try:
        draft_raw, picks_raw = await asyncio.gather(sleeper().get_draft(draft_id), sleeper().get_draft_picks(draft_id))
        if time.time() - lb.traded_at > TRADED_TTL:
            try:
                lb.traded_raw = await sleeper().get_traded_picks(draft_id)
            except SleeperAPIError:
                pass
            lb.traded_at = time.time()
    except SleeperNotFound as e:
        raise HTTPException(404, f"draft not found: {e}")
    except SleeperAPIError as e:
        raise HTTPException(502, f"Sleeper API error: {e}")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"could not reach the Sleeper API: {e}")
    st = state_from_sleeper(draft_raw, picks_raw, lb.league_raw, lb.users_raw, lb.rosters_raw, lb.traded_raw,
                            username=sess.username, user_id=sess.user_id, slot=sess.slot)
    if st.league is None:
        st.league = lb.league
    return st, {}


def _mock_key(cfg: MockConfig) -> str:
    return f"mock:{cfg.teams}:{cfg.rounds}:{cfg.scoring}:{int(cfg.superflex)}"


async def mock_bundle(cfg: MockConfig, sess: Session) -> tuple[LeagueBundle, Any]:
    """League + lean context for a mock configuration (cached per config)."""
    from ..mock.simulator import make_mock_draft, make_mock_league

    key = _mock_key(cfg)
    lb = CACHE.get(key, LEAGUE_TTL)
    if lb is None:
        league = make_mock_league(cfg.teams, cfg.rounds, cfg.scoring, cfg.superflex)
        league.league_id = key.replace(":", "_")
        draft = make_mock_draft(league, cfg.slot, cfg.teams, cfg.rounds)
        lb = LeagueBundle(key=key, league=league, draft=draft, snapshot=None, league_raw=None, users_raw=[], rosters_raw=[])
        CACHE.set(key, lb)
    if lb.draft is None or lb.draft.teams != cfg.teams or lb.draft.rounds != cfg.rounds:
        lb.draft = make_mock_draft(lb.league, cfg.slot, cfg.teams, cfg.rounds)
    ctx = await league_context(lb, sess)
    return lb, ctx


def build_mock(lb: LeagueBundle, ctx: Any, cfg: MockConfig, picks: Sequence[str]) -> Any:
    """Reconstruct a MockDraft from the browser's pick list (bots are re-seeded from the config)."""
    from ..mock.simulator import MockDraft, make_mock_draft

    draft = make_mock_draft(lb.league, cfg.slot, cfg.teams, cfg.rounds)
    md = MockDraft(ctx.players, lb.league, draft, cfg.slot, seed=cfg.seed if cfg.seed is not None else 7)
    for pid in picks:
        if md.is_complete:
            break
        pl = ctx.players.get(pid)
        if pl is None:
            raise HTTPException(400, f"unknown player in pick list: {pid}")
        md._record(pl, md.on_the_clock_slot)      # replay without consuming bot randomness
    return md


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def _f(v: Any, nd: int = 1) -> float | None:
    try:
        return None if v is None else round(float(v), nd)
    except (TypeError, ValueError):
        return None


def _note_dict(note: ResearchNote | None) -> dict | None:
    if note is None:
        return None
    return {"summary": note.summary, "injury_risk": note.injury_risk, "role_certainty": note.role_certainty,
            "offfield_risk": getattr(note, "offfield_risk", 0.0), "red_flags": list(getattr(note, "red_flags", []) or []),
            "upside": note.upside, "downside": note.downside, "sources": list(note.sources)[:6],
            "generated_at": note.generated_at}


def card_from_player(pl: Player, pr: Projection | None, notes: Mapping[str, ResearchNote] | None = None,
                     vorp: float | None = None) -> dict:
    return {
        "player_id": pl.player_id, "espn_id": pl.espn_id, "name": pl.name, "position": pl.position, "team": pl.team,
        "bye": pl.bye_week,
        "age": _f(pl.age), "years_exp": pl.years_exp, "injury_status": pl.injury_status,
        "depth_chart_order": pl.depth_chart_order,
        "points": _f(pr.points) if pr else None, "floor": _f(pr.floor) if pr else None,
        "ceiling": _f(pr.ceiling) if pr else None, "std": _f(pr.std) if pr else None,
        "ppg": _f(pr.ppg) if pr else None, "games": _f(pr.games) if pr else None,
        "vorp": _f(vorp), "vona": None, "marginal": None, "score": None, "tier": None, "pos_rank": None,
        "overall_rank": None, "adp": _f(pl.adp), "ecr": _f(pl.ecr), "availability_next": None,
        "availability_after_next": None, "reasons": [], "warnings": [], "flags": list(pr.flags) if pr else [],
        "note": _note_dict(notes.get(pl.player_id)) if notes else None, "drafted_by": None,
    }


def card_from_value(v: PlayerValue, notes: Mapping[str, ResearchNote] | None = None) -> dict:
    c = card_from_player(v.player, v.projection, notes, v.vorp)
    c.update({"vona": _f(v.vona), "marginal": _f(v.marginal_value), "score": _f(v.score), "tier": v.tier,
              "pos_rank": v.pos_rank, "overall_rank": v.overall_rank,
              "availability_next": _f(v.availability_next, 3), "availability_after_next": _f(v.availability_after_next, 3),
              "reasons": list(v.reasons), "warnings": list(v.warnings)})
    return c


def snapshot_payload(lb: LeagueBundle, st: DraftState | None, *, my_user_id: str | None = None,
                     my_slot: int | None = None) -> dict | None:
    """The League tab's snapshot; "me" comes from the live state, else from the given ids (the capture
    itself is identity-free)."""
    from ..capture import scoring_diff, strategy_flags

    snap = lb.snapshot
    league = lb.league
    draft = st.draft if st is not None else lb.draft
    my_slot = st.my_slot if st is not None else my_slot
    my_user_id = st.my_user_id if st is not None else my_user_id
    order = []
    if draft is not None:
        for slot in range(1, draft.teams + 1):
            m = st.manager_for_slot(slot) if st is not None else (snap.manager_for_slot(slot) if snap else None)
            order.append({"slot": slot, "display_name": m.display_name if m else f"Slot {slot}",
                          "team_name": m.team_name if m else None, "roster_id": draft.original_roster_for_slot(slot),
                          "picks": draft.picks_for_slot(slot)[:6], "is_me": slot == my_slot})
    diff = snap.diff if (snap and snap.diff) else scoring_diff(league.scoring_settings)
    flags = snap.flags if (snap and snap.flags) else strategy_flags(league, draft)
    unmapped = [{"label": u.get("label"), "points": u.get("points"), "stat_id": u.get("statId")}
                for u in (league.settings.get("unmapped_scoring") or []) if isinstance(u, Mapping)]
    return {
        "platform": lb.platform, "unmapped_scoring": unmapped,
        "flags": list(flags), "diff": diff.to_dict() if diff else None, "draft_order": order,
        "my_picks": draft.picks_for_slot(my_slot) if (draft and my_slot) else [],
        "my_user_id": my_user_id,
        "captured_at": snap.captured_at if snap else lb.built_at,
        "league": {"name": league.name, "league_id": league.league_id, "season": league.season,
                   "teams": league.total_rosters, "total_rosters": league.total_rosters,
                   "roster_positions": list(league.roster_positions),
                   "settings": {k: v for k, v in league.settings.items() if isinstance(v, (int, float, str))},
                   "scoring_settings": dict(league.scoring_settings)},
        "draft": {"draft_id": draft.draft_id, "type": draft.type, "status": draft.status, "teams": draft.teams,
                  "rounds": draft.rounds, "pick_timer": draft.pick_timer, "reversal_round": draft.reversal_round,
                  "start_time": draft.start_time, "player_type": getattr(draft, "player_type", 0)} if draft else None,
    }


def clock_start(st: DraftState, noticed_at: float) -> float | None:
    try:
        from ..ui.dashboard import clock_start as _cs

        return _cs(st, noticed_at)
    except Exception:  # noqa: BLE001
        lp = st.draft.last_picked
        return min(noticed_at, lp / 1000.0) if lp and lp > 1e12 else noticed_at


def build_payload(lb: LeagueBundle, ctx: Any, st: DraftState, rec: Recommendation | None, mode: str,
                  extra_status: Mapping[str, Any] | None = None) -> dict:
    from ..ui.dashboard import assign_roster_slots

    players: Mapping[str, Player] = ctx.players
    projections: Mapping[str, Projection] = ctx.projections
    notes = ctx.notes or {}
    league = ctx.league
    now = time.time()
    otc = st.on_the_clock_slot
    # ESPN picks carry no timestamps: pick_timer is reported (re-read each poll) but no countdown is claimed
    timed = lb.platform != "espn" and st.is_my_turn and st.draft.pick_timer
    start = clock_start(st, now) if timed else None
    seconds_left = None
    if start is not None and st.draft.status != "paused":
        seconds_left = max(0, int(round(st.draft.pick_timer - (now - start))))
    draft = {
        "type": st.draft.type, "status": st.draft.status, "teams": st.teams, "rounds": st.draft.rounds,
        "pick_timer": st.draft.pick_timer or None, "current_round": st.current_round, "next_pick_no": st.next_pick_no,
        "total_picks": st.draft.total_picks, "on_the_clock": {"slot": otc, "label": st.slot_label(otc)} if otc else None,
        "is_my_turn": st.is_my_turn, "my_slot": st.my_slot, "my_next_pick_no": st.my_next_pick_no,
        "my_pick_after_next": st.my_pick_after_next, "picks_until_my_turn": st.picks_until_my_turn,
        "is_complete": st.is_complete, "turn_started_at": start, "seconds_left": seconds_left,
        "last_picked": st.draft.last_picked, "rookie_draft": bool(getattr(st, "is_rookie_draft", False)),
        "rostered_excluded": len(getattr(st, "rostered_ids", set()) or ()),
    }
    # where the picks we know about came from: ESPN's board, the team rosters, or nowhere
    roster_only = sorted(getattr(st, "roster_only_ids", set()) or ())
    draft.update({
        "board_picks": len(st.picks), "rostered_only": len(roster_only), "roster_only_picks": len(roster_only),
        "picks_source": ("board+rosters" if (st.picks and roster_only) else "board" if st.picks
                         else "rosters" if roster_only else "none"),
        "roster_pick_confidence": getattr(st, "roster_confidence", "none"),
    })
    league_d = {"name": league.name, "scoring_type": league.scoring_type, "scoring_description": ctx.engine.describe(),
                "roster_positions": list(league.roster_positions), "teams": league.total_rosters, "season": league.season}
    me = None
    if rec is not None and rec.my_roster is not None:
        rs = rec.my_roster
        rows = assign_roster_slots([(pl, projections[pl.player_id].points if pl.player_id in projections else None)
                                    for pl in rs.players], league.roster_positions)
        me = {"slots": [{"slot": s_, "player": card_from_player(pl, projections.get(pl.player_id), notes) if pl else None}
                        for s_, pl, _ in rows],
              "needs": rs.needs(), "bye_clashes": {str(k): v for k, v in rs.bye_weeks.items() if v >= 2},
              "lineup_points": _f(rs.lineup_points), "bench_points": _f(rs.bench_points),
              "position_counts": dict(rs.position_counts), "open_starters": dict(rs.open_starters)}
    elif st.my_slot is not None:
        mine = [players[p.player_id] for p in st.my_picks() if p.player_id in players]
        rows = assign_roster_slots([(pl, projections[pl.player_id].points if pl.player_id in projections else None) for pl in mine],
                                   league.roster_positions)
        me = {"slots": [{"slot": s_, "player": card_from_player(pl, projections.get(pl.player_id), notes) if pl else None}
                        for s_, pl, _ in rows], "needs": [], "bye_clashes": {}, "lineup_points": None,
              "bench_points": None, "position_counts": {}, "open_starters": {}}
    values = list(rec.best_overall) if rec else []
    by_position = {}
    if rec is not None:
        for pos, adv in rec.by_position.items():
            by_position[pos] = {"action": adv.action, "rationale": adv.rationale,
                                "expected_next_available": _f(adv.expected_next_available), "drop_off": _f(adv.drop_off),
                                "candidates": [card_from_value(v, notes) for v in adv.candidates[:3]]}
    recent = []
    for p in sorted(st.picks, key=lambda x: -x.pick_no)[:12]:
        pl = players.get(p.player_id)
        slot = st.slot_of_pick(p) if hasattr(st, "slot_of_pick") else p.draft_slot
        recent.append({"pick_no": p.pick_no, "round": p.round, "slot": slot, "label": st.slot_label(slot),
                       "player_id": p.player_id, "name": pl.name if pl else p.player_name,
                       "position": pl.position if pl else p.position, "team": pl.team if pl else p.metadata.get("team"),
                       "is_me": slot == st.my_slot, "is_keeper": p.is_keeper})
    # players ESPN shows on a team roster but not on its draft board: they are gone from the pool and
    # their team is known, but ESPN published no pick number for them, so none is invented here
    roster_rows = []
    slot_of_team = {rid: s_ for s_, rid in st.draft.slot_to_roster_id.items()}
    confidence = getattr(st, "roster_confidence", "none")
    pick_numbers: Mapping[str, int] = getattr(st, "roster_pick_numbers", {}) or {}
    roster_only_set = set(roster_only)
    for spot in sorted((s for s in getattr(st, "roster_spots", []) if s.player_id in roster_only_set),
                       key=lambda s: -(s.acquired_at or 0)):
        pl = players.get(spot.player_id)
        slot = slot_of_team.get(spot.roster_id) if spot.roster_id is not None else None
        roster_rows.append({"player_id": spot.player_id, "name": pl.name if pl else (spot.name or spot.player_id),
                            "position": pl.position if pl else spot.position, "team": pl.team if pl else None,
                            "slot": slot, "label": st.slot_label(slot) if slot else None,
                            "roster_id": spot.roster_id, "acquired_at": spot.acquired_at,
                            "acquisition_type": spot.acquisition_type,
                            "pick_no": pick_numbers.get(spot.player_id) if confidence != "none" else None,
                            "confidence": confidence, "source": "roster",
                            "is_me": slot is not None and slot == st.my_slot})
    opponents = []
    if rec is not None:
        taken = getattr(st, "taken_pick_numbers", set())
        for rs in rec.opponent_rosters:
            fut = [n for n in st.draft.picks_for_slot(rs.slot) if n >= st.next_pick_no and n not in taken]
            opponents.append({"slot": rs.slot, "label": rs.label, "needs": rs.needs(), "next_pick": fut[0] if fut else None,
                              "position_counts": dict(rs.position_counts),
                              "players": [{"name": pl.name, "position": pl.position} for pl in rs.players]})
    status = {"compute_ms": _f(rec.compute_ms) if rec else None, "ts": now, "sources": dict(ctx.sources),
              "platform": lb.platform, "espn": None}
    if extra_status:
        status.update(extra_status)
    return {"mode": mode, "version": st.version, "ts": now, "draft": draft, "league": league_d, "me": me,
            "best": [card_from_value(v, notes) for v in values[:BEST_N]], "by_position": by_position,
            "available": [card_from_value(v, notes) for v in values[:AVAILABLE_TOP_N]], "recent": recent,
            "roster_only": roster_rows,
            "opponents": opponents, "pressure": {k: _f(v, 2) for k, v in (rec.position_pressure.items() if rec else [])},
            "notes": list(rec.notes) if rec else [], "snapshot": snapshot_payload(lb, st), "status": status}


def recommend(ctx: Any, st: DraftState) -> Recommendation:
    return ctx.advisor.recommend(st, top_n=AVAILABLE_TOP_N)


# ---------------------------------------------------------------------------
# Session -> (bundle, ctx, state, rec)
# ---------------------------------------------------------------------------


@dataclass
class Resolved:
    lb: LeagueBundle
    ctx: Any
    state: DraftState
    rec: Recommendation | None
    mock: Any = None
    last_picks: list = field(default_factory=list)
    diag: dict = field(default_factory=dict)      # espn: what the poll's ESPN GET returned


async def resolve(sess: Session, *, mock_action: str = "sync", player_id: str | None = None,
                  force: bool = False) -> Resolved:
    if sess.mode == "mock":
        cfg = sess.mock or MockConfig()
        if not (2 <= cfg.teams <= 20 and 3 <= cfg.rounds <= 30 and 1 <= cfg.slot <= cfg.teams):
            raise HTTPException(400, "teams 2-20, rounds 3-30, slot within teams")
        if cfg.scoring not in ("ppr", "half_ppr", "std"):
            raise HTTPException(400, "scoring must be ppr, half_ppr or std")
        lb, ctx = await mock_bundle(cfg, sess)
        md = build_mock(lb, ctx, cfg, sess.picks)
        before = len(sess.picks)
        if mock_action in ("pick", "auto") and not md.is_complete:
            if not md.is_my_turn:
                raise HTTPException(409, "it is not your turn")
            if mock_action == "auto":
                rec0 = recommend(ctx, md.state())
                if not rec0.best_overall:
                    raise HTTPException(409, "no recommendation available")
                player_id = rec0.best_overall[0].player_id
            if not player_id:
                raise HTTPException(400, "player_id required")
            try:
                md.make_pick(player_id)
            except ValueError as e:
                raise HTTPException(400, str(e))
        if mock_action in ("advance", "pick", "auto"):
            md.advance_until_my_turn()
        st = md.state()
        rec = recommend(ctx, st)
        picks_all = [p.player_id for p in sorted(st.picks, key=lambda p: p.pick_no)]
        last = [p for p in sorted(st.picks, key=lambda p: p.pick_no)[before:]]
        r = Resolved(lb, ctx, st, rec, mock=md, last_picks=last)
        r.state.metadata = {"picks": picks_all}  # type: ignore[attr-defined]
        return r
    if sess.platform not in PLATFORMS:
        raise HTTPException(400, f"platform must be one of {', '.join(PLATFORMS)}")
    if sess.platform == "espn":
        if not sess.league_id:
            raise HTTPException(400, "league_id required for an ESPN league (the leagueId= number in the league URL)")
    elif not (sess.draft_id or sess.league_id):
        raise HTTPException(400, "draft_id or league_id required")
    recaptured = drop_capture(sess) if force else False
    lb = await resolve_league(sess)
    resolve_me(lb, sess)                        # 400 for an unknown team / manager / slot, before any poll
    ctx = await league_context(lb, sess, force=recaptured)
    st, diag = await live_state(lb, sess, ctx.id_map, forced=force)
    if lb.platform == "espn":
        ensure_pick_players(ctx.players, st)
        ensure_roster_players(ctx.players, st)
        diag["recaptured"] = recaptured
    rec = recommend(ctx, st)
    return Resolved(lb, ctx, st, rec, diag=diag)


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------


class LookupReq(BaseModel):
    platform: str = "sleeper"
    username: str | None = None             # sleeper
    league_id: str | None = None            # espn
    season: int | None = None               # espn (default DEFAULT_SEASON)


class SessionReq(BaseModel):
    session: Session


class MockStateReq(BaseModel):
    session: Session
    action: str = "sync"
    player_id: str | None = None


class ChatReq(BaseModel):
    session: Session | None = None
    messages: list[dict]
    model: str | None = None


# ---------------------------------------------------------------------------
# Browser extension (POST /api/extension/advice)
# ---------------------------------------------------------------------------
#
# ESPN's REST API does not publish picks while a draft runs (the board stays at ``playerId: -1`` until
# the draft completes), so the owner's own draft-room tab is the only live source. A Chrome/Edge
# extension reads the picks there and posts them here; this endpoint therefore makes **zero** outbound
# calls (no ESPN, no Sleeper, no Anthropic - nothing metered, nothing networked). Everything it needs is
# the bundle plus whatever Sleeper/ECR data happens to be warm in :data:`CACHE` already.


EXT_TOP_N = 5                      # cards per list (overall + per position)
EXT_INDEX_TTL = 6 * 3600
EXT_SCORING = ("ppr", "half_ppr", "std")


class ExtPlayerRef(BaseModel):
    """One player the extension saw on the ESPN board: its numeric ESPN id and/or the name it read."""

    espn_id: int | None = None
    name: str | None = None


class ExtAdviceReq(BaseModel):
    taken: list[ExtPlayerRef]                       # every player off the board, mine included
    mine: list[ExtPlayerRef] = []                   # the subset on my roster
    scoring: str = "ppr"
    teams: int = 12
    rounds: int = 16
    superflex: bool = False
    slot: int | None = None                         # my draft slot (1-based); None -> inferred from the picks made
    made: int | None = None                         # picks on the board, whether or not they were recognised
    espn_scoring_items: list[dict] | None = None    # raw settings.scoringSettings.scoringItems read off the ESPN page
    roster_positions: list[str] | None = None       # Sleeper slot labels; None -> the default lineup


def ext_league(req: ExtAdviceReq) -> LeagueSettings:
    """The synthetic league one extension request describes (no platform call is made for it)."""
    from ..lean import default_league

    league = default_league(req.scoring, req.teams)
    if req.espn_scoring_items:
        # the extension read the league's real rules off the ESPN page: translate them exactly
        # (6-point passing TDs, TE premium, bonuses ...) instead of guessing from a ppr/std label
        from ..espn.scoring import espn_scoring_to_sleeper

        try:
            scoring, _unmapped = espn_scoring_to_sleeper(req.espn_scoring_items)
            if scoring:
                league.scoring_settings = scoring
        except Exception as e:  # noqa: BLE001
            log.warning("extension: could not translate the ESPN scoring items: %s", e)
    if req.roster_positions:
        positions = [str(p).upper() for p in req.roster_positions if str(p).strip()]
    else:
        starters = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"] + (["SUPER_FLEX"] if req.superflex else []) + ["K", "DEF"]
        positions = starters + ["BN"] * max(0, req.rounds - len(starters))
    league.roster_positions = positions
    league.league_id = f"ext_{req.scoring}_{req.teams}_{int(req.superflex)}_{hashlib.sha1(','.join(positions).encode()).hexdigest()[:8]}"
    league.name = f"Extension league ({req.teams} teams, {req.scoring})"
    return league


def ext_context_key(league: LeagueSettings, rounds: int) -> str:
    return f"ctx:ext:{league.league_id}:{rounds}"


async def ext_context(league: LeagueSettings, rounds: int) -> Any:
    """Lean context for one extension league, cached per (scoring, teams, superflex, roster positions).

    Built from the bundle plus only what is **already warm** - Sleeper projections and ECR rows are read
    out of the cache, never fetched: this endpoint must answer with no network at all.
    """
    from ..lean import CACHE as LEAN_CACHE, ECR_TTL, build_lean_context, get_bundle

    key = ext_context_key(league, rounds)
    hit = CACHE.get(key, CONTEXT_TTL)
    if hit is not None:
        return hit
    async with lock_for(key):
        hit = CACHE.get(key, CONTEXT_TTL)
        if hit is not None:
            return hit
        proj = CACHE.get("sleeper_proj", 3600)
        rows = LEAN_CACHE.get("ecr", ECR_TTL) or []
        try:
            notes = notes_dict()
        except Exception:  # noqa: BLE001
            notes = {}
        ctx = await asyncio.to_thread(build_lean_context, get_bundle(), league, None, sleeper_players=None,
                                      sleeper_proj=proj, ecr_rows=rows, notes=notes)
        CACHE.set(key, ctx)
        return ctx


def ext_index() -> dict[str, Any]:
    """``espn_id -> player_id`` and ``normalised name -> [player_id]`` over the bundle, built once."""
    from ..data.names import normalize_name
    from ..lean import get_bundle

    hit = CACHE.get("ext:index", EXT_INDEX_TTL)
    if hit is not None:
        return hit
    by_espn: dict[str, str] = {}
    by_name: dict[str, list[str]] = {}
    for pid, pl in get_bundle().players.items():
        if pl.espn_id:
            by_espn.setdefault(str(pl.espn_id).strip(), pid)
        n = normalize_name(pl.name)
        if n:
            by_name.setdefault(n, []).append(pid)
    idx = {"espn": by_espn, "name": by_name}
    CACHE.set("ext:index", idx)
    return idx


def ext_resolve(refs: Sequence[ExtPlayerRef], idx: Mapping[str, Any], ctx: Any) -> tuple[list[str], list[str]]:
    """``(player ids, labels we could not match)`` for one list of extension references.

    ESPN id first (D/ST ids are negative, ``-16000 - proTeamId``: :func:`draftadvisor.espn.ids.dst_team`),
    then the normalised name. A name several players share is settled by projected points; a name nobody
    has is reported in ``unresolved`` - a request never fails over one unknown player.
    """
    from ..data.names import normalize_name
    from ..espn.ids import dst_team, is_real_player_id

    out: list[str] = []
    unresolved: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        label = ref.name or (f"espn:{ref.espn_id}" if ref.espn_id is not None else "?")
        pid: str | None = None
        if ref.espn_id is not None and is_real_player_id(ref.espn_id):
            pid = idx["espn"].get(str(int(ref.espn_id)))
            if pid is None:
                team = dst_team(ref.espn_id)
                if team and team in ctx.players:
                    pid = team
        if pid is None:
            cands = [p for p in (idx["name"].get(normalize_name(ref.name)) or []) if p in ctx.players]
            if len(cands) == 1:
                pid = cands[0]
            elif cands:
                pid = max(cands, key=lambda p: (ctx.projections[p].points if p in ctx.projections else 0.0))
        if pid is None or pid not in ctx.players:
            unresolved.append(label)
            continue
        if pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out, unresolved


def ext_state(ctx: Any, league: LeagueSettings, rounds: int, taken: Sequence[str], mine: Sequence[str],
              slot: int | None = None, made: int | None = None) -> DraftState:
    """A :class:`DraftState` that holds exactly what the extension reported: every taken player is
    unavailable and my players sit on my slot, so roster needs and lineup value are real.

    An explicit ``slot`` wins (the extension reads it off the ESPN draft banner); otherwise it is the
    slot whose share of the picks made so far matches the size of my roster - a guess that lands on
    slot 1 before any pick exists, which is why the caller should send it.

    ``made`` is how deep the board actually is (the highest pick number the extension saw). The clock
    must not depend on how many players were *recognised*: :attr:`DraftState.next_pick_no` walks up to
    the first empty pick number, so one unreported player used to rewind the draft to an earlier round,
    which opens every starting slot, prices every position at its replacement level and hands RBs the
    36-point head start their steeper curve gives them. Picks 1..``made`` are therefore always filled -
    with a nameless placeholder where the extension could not say who was taken.
    """
    from ..mock.simulator import MY_USER_ID, make_mock_draft

    teams = league.total_rosters
    mine_ids = [p for p in mine]
    mine_set = set(mine_ids)
    others = [p for p in taken if p not in mine_set]
    reported = len(mine_ids) + len(others)
    if slot is not None and 1 <= int(slot) <= teams:
        my_slot = int(slot)
    else:
        order = make_mock_draft(league, 1, teams, rounds)
        my_slot, best = 1, None
        for s in range(1, teams + 1):
            d = abs(sum(1 for n in order.picks_for_slot(s) if n <= reported) - len(mine_ids))
            if best is None or d < best:
                best, my_slot = d, s
    draft = make_mock_draft(league, my_slot, teams, rounds)
    my_numbers = draft.picks_for_slot(my_slot)

    # How deep the board is: what the caller measured, never fewer than the players it listed, and
    # never fewer than my own k-th pick when it says I hold k players.
    board = max(int(made or 0), reported)
    if mine_ids and len(mine_ids) <= len(my_numbers):
        board = max(board, my_numbers[len(mine_ids) - 1])
    board = min(board, draft.total_picks)

    def _pick(no: int, pid: str | None) -> Any:
        from ..models import Pick

        pl = ctx.players.get(pid) if pid else None
        slot = draft.slot_for_pick(no)
        # a placeholder keeps the pick number occupied without claiming a player: the strategy engine
        # counts it as a pick made (recommend.py falls back to the pick's metadata position, here None)
        # and, on my own slot, as a roster body of unknown position rather than an open starting slot
        md = {"position": pl.position if pl else None, "team": pl.team if pl else None,
              "first_name": (pl.name.split(" ")[0] if pl and pl.name else None)}
        if pid is None:
            md = {"position": None, "team": None, "first_name": "Unreported", "last_name": "pick"}
        return Pick(pick_no=no, round=draft.round_of(no), draft_slot=slot,
                    player_id=pid if pid else f"unreported_{no}",
                    roster_id=draft.original_roster_for_slot(slot), metadata=md)

    # my players go on my own pick numbers, in order; every other reported player fills the earliest
    # number that is not one of mine, so nobody else's player is ever attributed to my roster
    taken_by: dict[int, str] = {}
    my_used = [n for n in my_numbers if n <= board][:len(mine_ids)]
    for n, pid in zip(my_used, mine_ids):
        taken_by[n] = pid
    queue = list(others) + mine_ids[len(my_used):]
    reserved = set(my_numbers)
    free = [n for n in range(1, board + 1) if n not in taken_by and n not in reserved]
    for n, pid in zip(free, queue):
        taken_by[n] = pid
    picks = [_pick(n, taken_by.get(n)) for n in range(1, board + 1)]
    # More players than the board is deep: the surplus is a player we know is gone but cannot place -
    # a roster spot, which is exactly the "off the board, no pick number" case the model already has.
    # Inventing pick numbers for them instead would push the clock past the draft's real position.
    spots = []
    for pid in queue[len(free):]:
        pl = ctx.players.get(pid)
        spots.append(RosterSpot(player_id=pid, name=pl.name if pl else None,
                                position=pl.position if pl else None))
    return DraftState(draft=draft, picks=picks, league=league, my_user_id=MY_USER_ID, my_slot=my_slot,
                      rostered_ids=set(taken) | mine_set, roster_spots=spots)


def ext_card(v: PlayerValue) -> dict:
    pl = v.player
    return {"player_id": pl.player_id, "espn_id": pl.espn_id, "name": pl.name, "position": pl.position,
            "team": pl.team, "bye": pl.bye_week, "points": _f(v.projection.points) if v.projection else None,
            "vorp": _f(v.vorp), "adp": _f(pl.adp), "tier": v.tier, "why": "; ".join(list(v.reasons)[:2])}


def ext_roster_warnings(st: DraftState, mine: Sequence[str], unresolved: Sequence[str]) -> list[str]:
    """What the caller should not have to infer from bad advice.

    An advisor that cannot see my roster thinks every starting slot is open and recommends the best
    player alive, which is why the panel says so out loud instead of quietly answering the wrong
    question.
    """
    out: list[str] = []
    if st.my_slot is not None:
        due = [n for n in st.draft.picks_for_slot(st.my_slot) if n < st.next_pick_no]
        if len(due) > len(mine):
            out.append(f"{len(due) - len(mine)} of your {len(due)} picks are not identified: "
                       f"needs and lineup value are incomplete. Set 'my slot' or open your roster panel.")
    if unresolved:
        out.append(f"{len(unresolved)} drafted player(s) could not be matched: " + ", ".join(list(unresolved)[:4]))
    return out


def ext_payload(rec: Recommendation, counts: Mapping[str, int], unresolved: Sequence[str], ms: float,
                warnings: Sequence[str] = (), board: Mapping[str, int] | None = None) -> dict:
    overall = [ext_card(v) for v in rec.best_overall[:EXT_TOP_N]]
    by_position = {pos: [ext_card(v) for v in adv.candidates[:EXT_TOP_N]] for pos, adv in rec.by_position.items()}
    suggestion = None
    if rec.best_overall:
        top = rec.best_overall[0]
        adv = rec.by_position.get(top.player.position or "")
        suggestion = {"player_id": top.player_id, "name": top.player.name, "position": top.player.position,
                      "team": top.player.team, "action": adv.action if adv else "take",
                      "why": (adv.rationale if adv and adv.rationale else "; ".join(list(top.reasons)[:2]))}
    roster = None
    if rec.my_roster is not None:
        roster = {"counts": {k: v for k, v in rec.my_roster.position_counts.items() if v},
                  "players": [{"name": p.name, "position": p.position} for p in rec.my_roster.players]}
    return {"overall": overall, "by_position": by_position, "suggestion": suggestion,
            "needs": rec.my_roster.needs() if rec.my_roster is not None else [],
            "roster": roster, "warnings": list(warnings), "board": dict(board or {}),
            "counts": dict(counts), "unresolved": list(unresolved), "ms": int(round(ms))}


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    app = FastAPI(title="draftadvisor", docs_url="/api/docs", redoc_url=None)
    guarded = [Depends(require_access)]

    # The browser extension calls this API from inside the ESPN draft room (https://*.espn.com). The API
    # holds no secrets and no cookies (credentials are off, so "*" cannot be used to read anyone's data),
    # and the CORS middleware sits above routing, so the OPTIONS preflight succeeds even when
    # DRAFTADVISOR_ACCESS_CODE is set - the access code is still required on the actual request.
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_origin_regex=r"https://([a-z0-9-]+\.)*espn\.com",
                       allow_credentials=False, allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"],
                       max_age=600)

    @app.middleware("http")
    async def no_store_api(request: Request, call_next: Any) -> Any:
        """Never let a CDN or proxy answer an API request.

        Every ``/api/`` URL is byte-identical from one poll to the next and the answer depends on request
        *headers* (the ESPN cookies, the access code), so a shared cache would both freeze the board and
        risk serving one user's league to another. Endpoints that set their own ``Cache-Control`` (the
        chat stream) keep it.
        """
        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", NO_STORE["Cache-Control"])
            response.headers.setdefault("Vary", NO_STORE["Vary"])
        return response

    @app.get("/", include_in_schema=False)
    async def index() -> Any:
        page = STATIC_DIR / "index.html"
        if not page.exists():
            raise HTTPException(404, "index.html missing")
        return FileResponse(str(page), headers={"Cache-Control": "no-cache"})

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Any:
        svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><rect width="64" height="64" rx="12" fill="#1f6f43"/>'
               '<ellipse cx="32" cy="32" rx="22" ry="14" fill="#8b4a1f" stroke="#f3e9d2" stroke-width="3"/>'
               '<path d="M22 32h20M27 27v10M32 26v12M37 27v10" stroke="#f3e9d2" stroke-width="3" stroke-linecap="round"/></svg>')
        return Response(content=svg, media_type="image/svg+xml")

    @app.get("/api/status")
    async def status() -> dict:
        from ..lean import BUNDLE_DIR, get_bundle

        bundle: dict = {"present": False, "dir": str(BUNDLE_DIR)}
        try:
            b = get_bundle()
            bundle = {"present": True, "built_at": b.meta.get("built_at"), "season": b.meta.get("season"),
                      "seasons": b.meta.get("seasons"), "players": len(b.players), "ml_rows": len(b.ml)}
        except Exception as e:  # noqa: BLE001
            bundle["error"] = str(e)
        from ..research.claude import MODEL_PRICES_USD_PER_MTOK

        st_ = store()
        return {"ok": True, "version": VERSION, "season": DEFAULT_SEASON, "bundle": bundle,
                "claude": {"server_key": bool(os.environ.get("ANTHROPIC_API_KEY")), "chat_model": chat_model(),
                           "prices": {k: list(v) for k, v in MODEL_PRICES_USD_PER_MTOK.items()}},
                "notes": {"count": st_.count(), "store": st_.backend},
                "access_code_required": access_code_required(), "sleeper": CACHE.get("sleeper_ok", 3600),
                "platforms": list(PLATFORMS), "espn": {"server_cookies": EspnAuth.from_request(None).present},
                "home": str(home_dir())}

    @app.post("/api/lookup", dependencies=guarded)
    async def lookup(req: LookupReq, request: Request) -> dict:
        from ..sleeper.client import SleeperNotFound

        platform = (req.platform or "sleeper").lower()
        if platform not in PLATFORMS:
            raise HTTPException(400, f"platform must be one of {', '.join(PLATFORMS)}")
        if platform == "espn":
            league_id = (req.league_id or "").strip()
            if not league_id:
                raise HTTPException(400, "league_id required (the leagueId= number in the ESPN league URL)")
            return await espn_lookup(league_id, int(req.season or DEFAULT_SEASON), EspnAuth.from_request(request))
        name = (req.username or "").strip()
        if not name:
            raise HTTPException(400, "username required")
        try:
            user = await sleeper().get_user(name)
            uid = str(user.get("user_id"))
            leagues, drafts = await asyncio.gather(sleeper().get_user_leagues(uid, DEFAULT_SEASON),
                                                   sleeper().get_user_drafts(uid, DEFAULT_SEASON))
        except SleeperNotFound:
            raise HTTPException(404, f"Sleeper user '{name}' not found")
        except Exception as e:  # noqa: BLE001
            CACHE.set("sleeper_ok", False)
            raise HTTPException(502, f"could not reach the Sleeper API: {e}")
        CACHE.set("sleeper_ok", True)

        def stype(lg: dict) -> str:
            rec = float((lg.get("scoring_settings") or {}).get("rec", 0) or 0)
            return "ppr" if rec >= 0.75 else "half_ppr" if rec >= 0.25 else "std"

        return {"user": {"user_id": uid, "display_name": user.get("display_name"), "username": user.get("username")},
                "leagues": [{"league_id": lg.get("league_id"), "name": lg.get("name"), "season": lg.get("season"),
                             "total_rosters": lg.get("total_rosters"), "status": lg.get("status"), "draft_id": lg.get("draft_id"),
                             "scoring_type": stype(lg), "roster_positions": lg.get("roster_positions") or []} for lg in leagues or []],
                "drafts": [{"draft_id": d.get("draft_id"), "league_id": d.get("league_id"), "status": d.get("status"),
                            "type": d.get("type"), "season": d.get("season"), "start_time": d.get("start_time"),
                            "teams": (d.get("settings") or {}).get("teams"), "rounds": (d.get("settings") or {}).get("rounds"),
                            "name": (d.get("metadata") or {}).get("name")} for d in drafts or []]}

    @app.post("/api/session/start", dependencies=guarded)
    async def session_start(req: SessionReq, request: Request) -> dict:
        sess = req.session.with_auth(request)
        if sess.mode == "mock":
            r = await resolve(sess, mock_action="sync")
            cfg = sess.mock or MockConfig()
            out = sess.model_dump()
            out["slot"] = cfg.slot
            return {"session": out, "snapshot": snapshot_payload(r.lb, r.state), "league": {
                "name": r.lb.league.name, "scoring_type": r.lb.league.scoring_type, "teams": r.lb.league.total_rosters}}
        if sess.platform == "espn" and not sess.league_id:
            raise HTTPException(400, "league_id required for an ESPN league (the leagueId= number in the league URL)")
        lb = await resolve_league(sess)
        my_uid, my_slot = resolve_me(lb, sess)
        snap = lb.snapshot
        out = sess.model_dump()
        out.update({"platform": lb.platform, "draft_id": snap.draft_id, "league_id": snap.league_id,
                    "user_id": my_uid or sess.user_id, "slot": my_slot if my_slot is not None else sess.slot})
        if lb.platform == "espn":
            tid = int(my_uid) if my_uid and str(my_uid).isdigit() else sess.espn_team_id
            out.update({"season": lb.season, "team_id": tid, "user_id": str(tid) if tid is not None else None})
            message = (None if my_slot is not None
                       else "draft order not set yet; your slot resolves when the commissioner publishes it" if tid is not None
                       else "no team selected: spectating (pick your team under Setup for on-the-clock advice)")
        else:
            message = "draft order not published yet; your slot resolves when the draft starts" if my_slot is None else None
        await league_context(lb, sess)
        return {"session": out, "snapshot": snapshot_payload(lb, None, my_user_id=my_uid, my_slot=my_slot),
                "league": {"name": lb.league.name, "scoring_type": lb.league.scoring_type, "teams": lb.league.total_rosters},
                "draft_order_known": bool(lb.draft and lb.draft.draft_order), "message": message}

    @app.get("/api/state", dependencies=guarded)
    async def state(request: Request) -> Any:
        """The poll. ``force=1`` re-captures the league (settings, teams, rosters, pick order, clock),
        rebuilds the lean context from it and re-reads the draft, bypassing every warm cache; it is
        rate-limited server-side (see :func:`drop_capture`) so holding the button cannot hammer ESPN.

        The response is ``no-store``: the poll URL is byte-identical every time and varies on the ESPN
        cookie headers, so a CDN or proxy answering it would freeze the board (and could hand one
        user's league to another).
        """
        sess = Session.from_query(request)
        if sess.mode == "mock":
            r = await resolve(sess, mock_action="sync")
            payload = build_payload(r.lb, r.ctx, r.state, r.rec, "mock")
            payload["picks"] = r.state.metadata["picks"]  # type: ignore[attr-defined]
            payload["last_picks"] = []
            return Response(json.dumps(payload), media_type="application/json", headers=NO_STORE)
        t0 = time.perf_counter()
        force = str(request.query_params.get("force", "")).lower() in ("1", "true", "yes")
        r = await resolve(sess, force=force)
        payload = build_payload(r.lb, r.ctx, r.state, r.rec, "live",
                                {"latency_ms": _f((time.perf_counter() - t0) * 1000), "espn": r.diag or None,
                                 "forced": force})
        return Response(json.dumps(payload), media_type="application/json", headers=NO_STORE)

    @app.get("/api/espn/diagnose", dependencies=guarded)
    async def espn_diagnose(request: Request) -> Any:
        """What ESPN returned on one read of this session's league, in numbers the owner can read.

        Costs one ESPN GET (the same request a poll makes), never a re-capture. Cookie values appear
        nowhere: the URL is redacted and only counts and flags are reported.
        """
        sess = Session.from_query(request)
        if sess.platform != "espn" or sess.mode == "mock":
            return Response(json.dumps({
                "platform": sess.platform, "espn": None, "teams": [], "board_sample": [], "identity": None,
                "explanation": "ESPN diagnostics do not apply to this session (it is not an ESPN live draft).",
            }), media_type="application/json", headers=NO_STORE)
        report = await espn_diagnose_report(sess)
        return Response(json.dumps(report), media_type="application/json", headers=NO_STORE)

    @app.post("/api/mock/state", dependencies=guarded)
    async def mock_state(req: MockStateReq, request: Request) -> dict:
        sess = req.session.with_auth(request)
        sess.mode = "mock"
        r = await resolve(sess, mock_action=req.action, player_id=req.player_id)
        payload = build_payload(r.lb, r.ctx, r.state, r.rec, "mock")
        payload["picks"] = r.state.metadata["picks"]  # type: ignore[attr-defined]
        payload["last_picks"] = [{"pick_no": p.pick_no, "round": p.round, "slot": p.draft_slot, "label": r.state.slot_label(p.draft_slot),
                                  "player_id": p.player_id, "name": p.player_name, "position": p.position,
                                  "is_me": p.draft_slot == r.state.my_slot} for p in r.last_picks]
        return payload

    @app.post("/api/extension/advice", dependencies=guarded)
    async def extension_advice(req: ExtAdviceReq) -> Any:
        """Recommendations for a draft the caller describes: it says who is gone and who is mine.

        Zero outbound calls (no ESPN, no Sleeper, nothing metered): the ESPN draft room is the only place
        picks appear while a draft runs, so the extension reads them there and posts them here. Cost of
        one call: 0 tokens, $0.00.
        """
        t0 = time.perf_counter()
        if req.scoring not in EXT_SCORING:
            raise HTTPException(400, f"scoring must be one of {', '.join(EXT_SCORING)}")
        if not 2 <= req.teams <= 20 or not 1 <= req.rounds <= 40:
            raise HTTPException(400, "teams must be 2-20 and rounds 1-40")
        league = ext_league(req)
        ctx = await ext_context(league, req.rounds)
        idx = ext_index()
        taken, unresolved = ext_resolve(req.taken, idx, ctx)
        mine, unresolved_mine = ext_resolve(req.mine, idx, ctx)
        taken_all = list(dict.fromkeys(list(taken) + list(mine)))
        st = ext_state(ctx, league, req.rounds, taken_all, mine, req.slot, req.made)
        rec = await asyncio.to_thread(ctx.advisor.recommend, st, EXT_TOP_N, EXT_TOP_N)
        counts = {"taken": len(req.taken), "resolved": len(taken_all), "mine": len(mine)}
        board = {"made": st.next_pick_no - 1, "on_the_clock": st.next_pick_no,
                 "round": st.draft.round_of(st.next_pick_no), "slot": st.my_slot or 0}
        unmatched = unresolved + [u for u in unresolved_mine if u not in unresolved]
        payload = ext_payload(rec, counts, unmatched, (time.perf_counter() - t0) * 1000.0,
                              ext_roster_warnings(st, mine, unmatched), board)
        return Response(json.dumps(payload), media_type="application/json", headers=NO_STORE)

    @app.post("/api/chat", dependencies=guarded)
    async def chat(req: ChatReq, request: Request) -> StreamingResponse:
        """The one paid endpoint: one Anthropic request per user message. The SSE stream ends with
        ``{"done": true, "model", "usage", "cost_usd"}`` so the page can show what it cost.

        Only a priced model (``/api/status`` -> ``claude.prices``) may be billed: anything else is 400.
        The transcript is trimmed to the newest ``CHAT_MAX_TURNS`` / ``CHAT_MAX_CHARS`` before it goes out
        (a single message over the limit is 413), so one request never carries an unbounded history.
        """
        from ..research.claude import CHAT_MAX_CHARS, MODEL_PRICES_USD_PER_MTOK, build_context_text, model_prices, trim_transcript

        model = req.model or chat_model()
        if model_prices(model) is None:
            raise HTTPException(400, f"unknown chat model {model!r}; choose one of {', '.join(MODEL_PRICES_USD_PER_MTOK)}")
        if req.messages and len(str(req.messages[-1].get("content") or "")) > CHAT_MAX_CHARS:
            raise HTTPException(413, f"message too long (over {CHAT_MAX_CHARS} characters)")
        messages = trim_transcript(req.messages)
        rs = claude_for(api_key_from(request))
        if not rs.enabled:
            raise HTTPException(503, "Claude is off: add an Anthropic API key under Settings")
        context = ""
        if req.session is not None and (req.session.draft_id or req.session.league_id or req.session.mode == "mock"):
            try:
                r = await resolve(req.session.with_auth(request))
                context = build_context_text(r.state, r.rec, r.ctx.players, r.ctx.notes)
            except HTTPException as e:
                context = f"(draft context unavailable: {e.detail})"

        async def gen() -> AsyncIterator[bytes]:
            try:
                async for chunk in rs.chat_stream(messages, context, model=model):
                    if isinstance(chunk, dict):
                        yield f"data: {json.dumps(chunk)}\n\n".encode()
                    else:
                        yield f"data: {json.dumps({'delta': chunk})}\n\n".encode()
            except Exception as e:  # noqa: BLE001
                log.warning("chat failed: %s", e)
                yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n".encode()

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    async def _ctx_for(sess: Session | None) -> tuple[Any, DraftState | None, Recommendation | None]:
        if sess is not None and (sess.draft_id or sess.league_id or sess.mode == "mock"):
            r = await resolve(sess)
            return r.ctx, r.state, r.rec
        return await default_context(), None, None

    @app.get("/api/notes", dependencies=guarded)
    async def notes() -> dict:
        """Legacy research notes (read-only; nothing writes new ones)."""
        return store().get_all()

    @app.get("/api/player/{player_id}", dependencies=guarded)
    async def player(player_id: str, request: Request) -> dict:
        sess = Session.from_query(request)
        ctx, st, rec = await _ctx_for(sess if (sess.draft_id or sess.league_id or sess.mode == "mock") else None)
        pl = ctx.players.get(player_id)
        if pl is None:
            raise HTTPException(404, "unknown player")
        pr = ctx.projections.get(player_id)
        v = None
        if st is not None and player_id not in getattr(st, "unavailable_ids", st.drafted_ids):
            try:
                v = ctx.advisor.value_of(st, player_id)
            except Exception:  # noqa: BLE001
                v = None
        card = card_from_value(v, ctx.notes) if v is not None else card_from_player(pl, pr, ctx.notes)
        card["projection"] = {"components": dict(pr.components), "weights": dict(pr.weights), "flags": list(pr.flags),
                              "stat_line": {k: _f(x) for k, x in pr.stat_line.items() if x}} if pr else None
        try:
            card["explain"] = ctx.advisor.explain_pick(st, player_id) if st is not None else ""
        except Exception:  # noqa: BLE001
            card["explain"] = ""
        stored = store().get(player_id)
        if stored:
            card["note"] = _note_dict(ResearchNote.from_dict(stored))
        return card

    @app.get("/api/projections", dependencies=guarded)
    async def projections(request: Request, position: str | None = None, top: int = 80, q: str | None = None) -> list[dict]:
        from ..data.names import normalize_name
        from ..strategy.replacement import vorp as vorp_fn

        sess = Session.from_query(request)
        if sess.draft_id or sess.league_id or sess.mode == "mock":
            try:
                ctx, st, _ = await _ctx_for(sess)
            except HTTPException:
                ctx, st = await default_context(), None
        else:
            ctx, st = await default_context(), None
        key = f"vorp:{id(ctx)}"
        vorps = CACHE.get(key, CONTEXT_TTL)
        if vorps is None:
            try:
                vorps = vorp_fn(ctx.projections, ctx.players, ctx.league)
            except Exception:  # noqa: BLE001
                vorps = {}
            CACHE.set(key, vorps)
        qn = normalize_name(q) if q else ""
        unavailable = getattr(st, "unavailable_ids", None) if st is not None else None
        rows = []
        for pid, pr in ctx.projections.items():
            pl = ctx.players.get(pid)
            if pl is None or (position and pl.position != position.upper()):
                continue
            if qn and qn not in normalize_name(pl.name):
                continue
            c = card_from_player(pl, pr, ctx.notes, vorps.get(pid))
            c["drafted_by"] = "drafted" if (unavailable and pid in unavailable) else None
            rows.append(c)
        rows.sort(key=lambda c: -(c["points"] or 0.0))
        return rows[: max(1, min(int(top), 600))]

    return app


def _espn_owner_names(team: Mapping[str, Any], members: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Display names of a team's owners (``members[]`` by SWID, else the owner dicts of newer payloads)."""
    from ..espn.parsing import normalize_swid, team_owner_ids

    owner_dicts = {normalize_swid(o.get("id")): o for o in (team.get("owners") or []) if isinstance(o, Mapping)}
    names: list[str] = []
    for sid in team_owner_ids(team):
        m = members.get(sid) or owner_dicts.get(sid) or {}
        name = m.get("displayName") or " ".join(str(x) for x in (m.get("firstName"), m.get("lastName")) if x)
        if name and name not in names:
            names.append(str(name))
    return names


async def espn_diagnose_report(sess: Session) -> dict:
    """``GET /api/espn/diagnose``: one ESPN read of the session's league, reported in numbers.

    The ``espn`` block is the one the poll puts in ``status.espn``; on top of it come the per-team roster
    counts, up to ten board entries as ESPN sent them (pick number, player id, team id, keeper flag - no
    names needed), the resolved league / draft identity, and a plain-English explanation. Nothing here
    can carry a cookie: the URL is redacted by the client and only counts and flags are copied out.
    """
    from ..espn.parsing import DRAFT_ACQUISITIONS, draft_detail_of, merge_league_payload

    if not sess.league_id:
        raise HTTPException(400, "league_id required for an ESPN league (the leagueId= number in the league URL)")
    lb = await resolve_league(sess)
    ctx = await league_context(lb, sess)
    st, diag, poll_json = await espn_poll(lb, sess, ctx.id_map)
    merged = merge_league_payload(lb.league_raw or {}, poll_json)
    spots_by_team: dict[Any, list] = {}
    for s in st.roster_spots:
        spots_by_team.setdefault(s.roster_id, []).append(s)
    teams = []
    for t in (merged.get("teams") or []):
        if not isinstance(t, Mapping) or t.get("id") is None:
            continue
        tid = int(t["id"])
        mine = spots_by_team.get(tid, [])
        m = st.managers.get(str(tid))
        teams.append({"team_id": tid, "label": (m.team_name or m.display_name) if m else f"Team {tid}",
                      "slot": st.draft.draft_order.get(str(tid)),
                      "roster_entries": len(((t.get("roster") or {}).get("entries") or [])),
                      "drafted_entries": sum(1 for s in mine if (s.acquisition_type or "").upper() in DRAFT_ACQUISITIONS),
                      "fresh_entries": sum(1 for s in mine if s.is_fresh),
                      "board_picks": sum(1 for p in st.picks if p.roster_id == tid)})
    detail = draft_detail_of(poll_json)
    sample = []
    for raw in (detail.get("picks") or [])[:10]:
        if isinstance(raw, Mapping):
            sample.append({"overallPickNumber": raw.get("overallPickNumber"), "roundId": raw.get("roundId"),
                           "playerId": raw.get("playerId"), "teamId": raw.get("teamId"),
                           "keeper": bool(raw.get("keeper") or raw.get("reservedForKeeper"))})
    return {
        "platform": "espn", "espn": diag, "teams": teams, "board_sample": sample,
        "identity": {"league_id": lb.league.league_id, "league_name": lb.league.name, "season": lb.season,
                     "teams": st.draft.teams, "rounds": st.draft.rounds, "total_picks": st.draft.total_picks,
                     "draft_type": st.draft.type, "draft_status": st.draft.status,
                     "my_team_id": st.my_user_id, "my_slot": st.my_slot,
                     "pick_order_known": bool(st.draft.draft_order), "next_pick_no": st.next_pick_no},
        "explanation": espn_explanation(diag),
    }


async def espn_lookup(league_id: str, season: int, auth: EspnAuth) -> dict:
    """``POST /api/lookup`` for ESPN: the league, its draft and teams (2 GETs); ``me`` = the team the SWID owns."""
    from ..espn.parsing import normalize_swid, parse_espn_draft, parse_espn_league, parse_espn_managers, resolve_my_team

    async with espn_client(auth) as client:
        try:
            league_json = await client.get_settings_and_teams(league_id, season)
            draft_json = await client.get_draft_detail(league_id, season)
        except Exception as e:  # noqa: BLE001
            raise espn_http_error(e, league_id, season, auth)
    league = parse_espn_league(league_json, draft_json)
    draft = parse_espn_draft(league_json, draft_json)
    managers = parse_espn_managers(league_json, draft)
    my_tid, my_slot = resolve_my_team(draft, managers, league_json, swid=auth.swid)
    members = {normalize_swid(m.get("id")): m for m in (league_json.get("members") or []) if isinstance(m, Mapping)}
    teams = []
    for t in league_json.get("teams") or []:
        if not isinstance(t, Mapping) or t.get("id") is None:
            continue
        m = managers.get(str(t["id"]))
        teams.append({"team_id": int(t["id"]), "name": (m.team_name or m.display_name) if m else f"Team {t['id']}",
                      "abbrev": t.get("abbrev"), "owners": _espn_owner_names(t, members),
                      "slot": m.slot if m else draft.draft_order.get(str(t["id"])), "is_me": str(t["id"]) == my_tid})
    teams.sort(key=lambda d: (d["slot"] is None, d["slot"] or 0, d["team_id"]))
    return {"platform": "espn",
            "league": {"league_id": league.league_id or league_id, "name": league.name, "season": league.season,
                       "teams": league.total_rosters, "scoring_type": league.scoring_type,
                       "is_public": bool(league.settings.get("is_public")),
                       "draft": {"type": draft.type, "status": draft.status, "pick_timer": draft.pick_timer,
                                 "start_time": draft.start_time, "rounds": draft.rounds,
                                 "order_known": bool(draft.draft_order)}},
            "teams": teams,
            "me": {"team_id": int(my_tid), "slot": my_slot} if my_tid and my_tid.isdigit() else None,
            "unmapped_scoring": [{"label": u.get("label"), "points": u.get("points")}
                                 for u in (league.settings.get("unmapped_scoring") or []) if isinstance(u, Mapping)]}


async def default_context() -> Any:
    """Bundle + ECR context for a default half-PPR league (Board tab without a session)."""
    from ..lean import build_lean_context, default_league, get_bundle

    hit = CACHE.get("ctx:default", CONTEXT_TTL)
    if hit is not None:
        return hit
    async with lock_for("ctx:default"):
        hit = CACHE.get("ctx:default", CONTEXT_TTL)
        if hit is not None:
            return hit
        rows = await ecr_rows()
        proj = await sleeper_projections(DEFAULT_SEASON)
        ctx = await asyncio.to_thread(build_lean_context, get_bundle(), default_league(), None, sleeper_players=None,
                                      sleeper_proj=proj, ecr_rows=rows, notes=notes_dict())
        CACHE.set("ctx:default", ctx)
        return ctx


app = create_app()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="draftadvisor web", description="local web app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--home", help="data directory (default ./data or $DRAFTADVISOR_HOME)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.home:
        os.environ["DRAFTADVISOR_HOME"] = args.home
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    url = f"http://{args.host}:{args.port}/"
    print(f"draftadvisor web app: {url}   (Ctrl-C to stop)")
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
