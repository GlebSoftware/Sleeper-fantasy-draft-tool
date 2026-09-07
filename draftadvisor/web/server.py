"""Stateless FastAPI server for the web app (DESIGN.md §3.9) - runs locally and on Vercel.

No background tasks and no per-process draft state: the browser owns the session
(league / draft ids, identity, mock pick list) and sends it with every request. The
server keeps only warm caches (bundle, Sleeper payloads, ECR, per-league contexts)
and persists research notes through :mod:`draftadvisor.research.store`.
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Mapping, Sequence

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

from ..config import DEFAULT_SEASON, Settings, home_dir
from ..models import DraftSettings, DraftState, LeagueSettings, Player, PlayerValue, Projection, Recommendation, ResearchNote
from ..research.store import NoteStore, make_store

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
DEFAULT_PORT = 8787
AVAILABLE_TOP_N = 250
BEST_N = 8
LEAGUE_TTL = 10 * 60          # league / users / rosters / snapshot
TRADED_TTL = 30
CONTEXT_TTL = 10 * 60
ADVICE_TIMEOUT_S = 14.0
RESEARCH_TIMEOUT_S = 100.0
VERSION = "2.0"


# ---------------------------------------------------------------------------
# Session model (what the browser sends)
# ---------------------------------------------------------------------------


class MockConfig(BaseModel):
    teams: int = 12
    rounds: int = 15
    slot: int = 5
    scoring: str = "half_ppr"
    superflex: bool = False
    seed: int | None = None


class Session(BaseModel):
    mode: str = "live"                      # live | mock
    draft_id: str | None = None
    league_id: str | None = None
    username: str | None = None
    user_id: str | None = None
    slot: int | None = None
    use_claude: bool = True
    mock: MockConfig | None = None
    picks: list[str] = []                   # mock only: player ids in pick order

    @classmethod
    def from_query(cls, request: Request) -> "Session":
        q = request.query_params
        mock = None
        if q.get("mode") == "mock":
            mock = MockConfig(teams=int(q.get("teams", 12)), rounds=int(q.get("rounds", 15)), slot=int(q.get("slot", 5)),
                              scoring=q.get("scoring", "half_ppr"), superflex=q.get("superflex", "false").lower() in ("1", "true"),
                              seed=int(q["seed"]) if q.get("seed") else None)
        picks = [p for p in (q.get("picks") or "").split(",") if p]
        return cls(mode=q.get("mode", "live"), draft_id=q.get("draft_id") or None, league_id=q.get("league_id") or None,
                   username=q.get("username") or None, user_id=q.get("user_id") or None,
                   slot=int(q["slot"]) if q.get("slot") and q.get("mode") != "mock" else None,
                   use_claude=q.get("use_claude", "true").lower() not in ("0", "false"), mock=mock, picks=picks)


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
_RESEARCHERS: dict[str, Any] = {}
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


def researcher_for(api_key: str | None) -> Any:
    from ..research.claude import ClaudeResearcher

    key = api_key or os.environ.get("ANTHROPIC_API_KEY") or ""
    h = hashlib.sha1(key.encode()).hexdigest()[:10] if key else "none"
    r = _RESEARCHERS.get(h)
    if r is None:
        r = ClaudeResearcher(api_key=key or None, cache_dir=home_dir() / "research")
        _RESEARCHERS[h] = r
    return r


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
    """Everything cached for one draft: league/draft/users/rosters/snapshot + the lean context."""

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


async def resolve_league(sess: Session) -> LeagueBundle:
    """Capture (cached) the league behind a session; raises HTTP errors for bad ids."""
    from ..capture import capture_league
    from ..sleeper.client import SleeperAPIError, SleeperNotFound

    key = f"league:{sess.draft_id or ''}:{sess.league_id or ''}"
    hit = CACHE.get(key, LEAGUE_TTL)
    if hit is not None:
        return hit
    async with lock_for(key):
        hit = CACHE.get(key, LEAGUE_TTL)
        if hit is not None:
            return hit
        try:
            snap = await capture_league(sleeper(), sess.league_id, sess.draft_id, username=sess.username,
                                        user_id=sess.user_id, slot=sess.slot, season=DEFAULT_SEASON, save=False)
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
        CACHE.set(key, lb)
        CACHE.set("sleeper_ok", True)
        return lb


async def league_context(lb: LeagueBundle, sess: Session) -> Any:
    """Lean context for the league (rebuilt when notes / inputs change or after CONTEXT_TTL)."""
    from ..lean import build_lean_context, context_fingerprint, get_bundle

    notes = notes_dict()
    proj = await sleeper_projections(DEFAULT_SEASON)
    players_payload = await sleeper_players()
    fp = context_fingerprint(lb.league, bool(proj), bool(players_payload), len(notes))
    if lb.ctx is not None and lb.ctx_fp == fp and time.time() - lb.ctx.built_at < CONTEXT_TTL:
        return lb.ctx
    async with lock_for(lb.key + ":ctx"):
        if lb.ctx is not None and lb.ctx_fp == fp and time.time() - lb.ctx.built_at < CONTEXT_TTL:
            return lb.ctx
        rows = await ecr_rows()
        settings = Settings.from_env(league_id=lb.league.league_id, draft_id=lb.draft.draft_id if lb.draft else None,
                                     username=sess.username, user_id=sess.user_id, slot=sess.slot)
        settings.use_claude = sess.use_claude
        ctx = await asyncio.to_thread(build_lean_context, get_bundle(), lb.league, lb.draft,
                                      sleeper_players=players_payload, sleeper_proj=proj, ecr_rows=rows,
                                      notes=notes, settings=settings)
        lb.ctx, lb.ctx_fp = ctx, fp
        return ctx


async def live_state(lb: LeagueBundle, sess: Session) -> DraftState:
    """Fetch the draft + picks right now and build the DraftState (2 Sleeper GETs)."""
    from ..sleeper.client import SleeperAPIError, SleeperNotFound
    from ..sleeper.parsing import state_from_sleeper

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
    return st


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
        "player_id": pl.player_id, "name": pl.name, "position": pl.position, "team": pl.team, "bye": pl.bye_week,
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


def snapshot_payload(lb: LeagueBundle, st: DraftState | None) -> dict | None:
    from ..capture import scoring_diff, strategy_flags

    snap = lb.snapshot
    league = lb.league
    draft = st.draft if st is not None else lb.draft
    my_slot = st.my_slot if st is not None else (snap.my_slot if snap else None)
    order = []
    if draft is not None:
        for slot in range(1, draft.teams + 1):
            m = st.manager_for_slot(slot) if st is not None else (snap.manager_for_slot(slot) if snap else None)
            order.append({"slot": slot, "display_name": m.display_name if m else f"Slot {slot}",
                          "team_name": m.team_name if m else None, "roster_id": draft.original_roster_for_slot(slot),
                          "picks": draft.picks_for_slot(slot)[:6], "is_me": slot == my_slot})
    diff = snap.diff if (snap and snap.diff) else scoring_diff(league.scoring_settings)
    flags = snap.flags if (snap and snap.flags) else strategy_flags(league, draft)
    return {
        "flags": list(flags), "diff": diff.to_dict() if diff else None, "draft_order": order,
        "my_picks": draft.picks_for_slot(my_slot) if (draft and my_slot) else [],
        "my_user_id": st.my_user_id if st is not None else (snap.my_user_id if snap else None),
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
    start = clock_start(st, now) if (st.is_my_turn and st.draft.pick_timer) else None
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
    opponents = []
    if rec is not None:
        taken = getattr(st, "taken_pick_numbers", set())
        for rs in rec.opponent_rosters:
            fut = [n for n in st.draft.picks_for_slot(rs.slot) if n >= st.next_pick_no and n not in taken]
            opponents.append({"slot": rs.slot, "label": rs.label, "needs": rs.needs(), "next_pick": fut[0] if fut else None,
                              "position_counts": dict(rs.position_counts),
                              "players": [{"name": pl.name, "position": pl.position} for pl in rs.players]})
    status = {"compute_ms": _f(rec.compute_ms) if rec else None, "ts": now, "sources": dict(ctx.sources)}
    if extra_status:
        status.update(extra_status)
    return {"mode": mode, "version": st.version, "ts": now, "draft": draft, "league": league_d, "me": me,
            "best": [card_from_value(v, notes) for v in values[:BEST_N]], "by_position": by_position,
            "available": [card_from_value(v, notes) for v in values[:AVAILABLE_TOP_N]], "recent": recent,
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


async def resolve(sess: Session, *, mock_action: str = "sync", player_id: str | None = None) -> Resolved:
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
    if not (sess.draft_id or sess.league_id):
        raise HTTPException(400, "draft_id or league_id required")
    lb = await resolve_league(sess)
    ctx = await league_context(lb, sess)
    st = await live_state(lb, sess)
    rec = recommend(ctx, st)
    return Resolved(lb, ctx, st, rec)


def state_key(sess: Session, st: DraftState) -> str:
    last = max((p.pick_no for p in st.picks), default=0)
    return f"{sess.mode}:{sess.draft_id or _mock_key(sess.mock or MockConfig())}:{len(st.picks)}:{last}:{st.my_slot}"


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------


class LookupReq(BaseModel):
    username: str


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


class ResearchPlayerReq(BaseModel):
    session: Session | None = None
    player_id: str
    force: bool = False


class ResearchNextReq(BaseModel):
    session: Session | None = None
    top: int = 150


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    app = FastAPI(title="draftadvisor", docs_url="/api/docs", redoc_url=None)
    guarded = [Depends(require_access)]

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
        st_ = store()
        return {"ok": True, "version": VERSION, "season": DEFAULT_SEASON, "bundle": bundle,
                "claude": {"server_key": bool(os.environ.get("ANTHROPIC_API_KEY")),
                           "chat_model": os.environ.get("DRAFTADVISOR_CHAT_MODEL", "claude-opus-5"),
                           "research_model": os.environ.get("DRAFTADVISOR_CLAUDE_MODEL", "claude-sonnet-5")},
                "notes": {"count": st_.count(), "store": st_.backend},
                "access_code_required": access_code_required(), "sleeper": CACHE.get("sleeper_ok", 3600),
                "home": str(home_dir())}

    @app.post("/api/lookup", dependencies=guarded)
    async def lookup(req: LookupReq) -> dict:
        from ..sleeper.client import SleeperNotFound

        name = req.username.strip()
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
    async def session_start(req: SessionReq) -> dict:
        sess = req.session
        if sess.mode == "mock":
            r = await resolve(sess, mock_action="sync")
            cfg = sess.mock or MockConfig()
            out = sess.model_dump()
            out["slot"] = cfg.slot
            return {"session": out, "snapshot": snapshot_payload(r.lb, r.state), "league": {
                "name": r.lb.league.name, "scoring_type": r.lb.league.scoring_type, "teams": r.lb.league.total_rosters}}
        lb = await resolve_league(sess)
        snap = lb.snapshot
        out = sess.model_dump()
        out.update({"draft_id": snap.draft_id, "league_id": snap.league_id, "user_id": snap.my_user_id or sess.user_id,
                    "slot": snap.my_slot if snap.my_slot is not None else sess.slot})
        await league_context(lb, sess)
        return {"session": out, "snapshot": snapshot_payload(lb, None),
                "league": {"name": lb.league.name, "scoring_type": lb.league.scoring_type, "teams": lb.league.total_rosters},
                "draft_order_known": bool(lb.draft and lb.draft.draft_order),
                "message": "draft order not published yet; your slot resolves when the draft starts" if snap.my_slot is None else None}

    @app.get("/api/state", dependencies=guarded)
    async def state(request: Request) -> dict:
        sess = Session.from_query(request)
        if sess.mode == "mock":
            r = await resolve(sess, mock_action="sync")
            payload = build_payload(r.lb, r.ctx, r.state, r.rec, "mock")
            payload["picks"] = r.state.metadata["picks"]  # type: ignore[attr-defined]
            payload["last_picks"] = []
            return payload
        t0 = time.perf_counter()
        r = await resolve(sess)
        return build_payload(r.lb, r.ctx, r.state, r.rec, "live", {"latency_ms": _f((time.perf_counter() - t0) * 1000)})

    @app.post("/api/mock/state", dependencies=guarded)
    async def mock_state(req: MockStateReq) -> dict:
        sess = req.session
        sess.mode = "mock"
        r = await resolve(sess, mock_action=req.action, player_id=req.player_id)
        payload = build_payload(r.lb, r.ctx, r.state, r.rec, "mock")
        payload["picks"] = r.state.metadata["picks"]  # type: ignore[attr-defined]
        payload["last_picks"] = [{"pick_no": p.pick_no, "round": p.round, "slot": p.draft_slot, "label": r.state.slot_label(p.draft_slot),
                                  "player_id": p.player_id, "name": p.player_name, "position": p.position,
                                  "is_me": p.draft_slot == r.state.my_slot} for p in r.last_picks]
        return payload

    @app.get("/api/advice", dependencies=guarded)
    async def advice(request: Request) -> dict:
        sess = Session.from_query(request)
        rs = researcher_for(api_key_from(request))
        if not rs.enabled or not sess.use_claude:
            return {"advice": None, "status": "off", "model": None}
        r = await resolve(sess)
        if r.rec is None or r.state.is_complete:
            return {"advice": None, "status": "idle", "model": rs.model}
        key = "advice:" + state_key(sess, r.state)
        hit = CACHE.get(key, 3600)
        if hit is not None:
            return {"advice": hit, "status": "ready", "model": rs.model, "cached": True}
        async with lock_for(key):
            hit = CACHE.get(key, 3600)
            if hit is not None:
                return {"advice": hit, "status": "ready", "model": rs.model, "cached": True}
            try:
                text = await rs.on_the_clock_advice(r.state, r.rec, r.ctx.players, r.ctx.notes, timeout=ADVICE_TIMEOUT_S)
            except Exception as e:  # noqa: BLE001
                log.warning("advice failed: %s", e)
                return {"advice": None, "status": "error", "model": rs.model, "error": str(e)}
        if text:
            CACHE.set(key, text)
            return {"advice": text, "status": "ready", "model": rs.model}
        return {"advice": None, "status": "no answer", "model": rs.model, "error": getattr(rs, "last_error", None)}

    @app.post("/api/chat", dependencies=guarded)
    async def chat(req: ChatReq, request: Request) -> StreamingResponse:
        from ..research.claude import build_context_text

        rs = researcher_for(api_key_from(request))
        if not rs.enabled:
            raise HTTPException(503, "Claude is off: add an Anthropic API key under Settings")
        context = ""
        if req.session is not None and (req.session.draft_id or req.session.league_id or req.session.mode == "mock"):
            try:
                r = await resolve(req.session)
                context = build_context_text(r.state, r.rec, r.ctx.players, r.ctx.notes)
            except HTTPException as e:
                context = f"(draft context unavailable: {e.detail})"
        model = req.model or os.environ.get("DRAFTADVISOR_CHAT_MODEL") or "claude-opus-5"

        async def gen() -> AsyncIterator[bytes]:
            try:
                async for chunk in rs.chat_stream(req.messages, context, model=model):
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

    @app.post("/api/research/player", dependencies=guarded)
    async def research_player(req: ResearchPlayerReq, request: Request) -> dict:
        rs = researcher_for(api_key_from(request))
        if not rs.enabled:
            raise HTTPException(503, "Claude is off: add an Anthropic API key under Settings")
        ctx, _, _ = await _ctx_for(req.session)
        pl = ctx.players.get(req.player_id)
        if pl is None:
            raise HTTPException(404, "unknown player")
        cached = store().get(pl.player_id)
        note = await _research_one(rs, pl, ctx.projections.get(pl.player_id), req.force,
                                   ResearchNote.from_dict(cached) if cached else None)
        return {"note": _note_dict(note), "player_id": pl.player_id, "name": pl.name, "error": getattr(rs, "last_error", None)}

    @app.post("/api/research/next", dependencies=guarded)
    async def research_next(req: ResearchNextReq, request: Request) -> dict:
        rs = researcher_for(api_key_from(request))
        if not rs.enabled:
            raise HTTPException(503, "Claude is off: add an Anthropic API key under Settings")
        ctx, st, rec = await _ctx_for(req.session)
        top = max(1, min(int(req.top), 400))
        targets: list[Player] = []
        seen: set[str] = set()
        if rec is not None:
            for v in rec.best_overall[:BEST_N]:
                targets.append(v.player); seen.add(v.player_id)
        for pl in sorted((p for p in ctx.players.values() if p.adp is not None), key=lambda p: p.adp)[:top]:
            if pl.player_id not in seen:
                targets.append(pl); seen.add(pl.player_id)
        notes = store().get_all()
        fresh_cut = time.time() - 3 * 86400

        def is_fresh(pid: str) -> bool:
            d = notes.get(pid)
            return bool(d) and float(d.get("generated_at", 0)) > fresh_cut and not str(d.get("summary", "")).startswith("(research unavailable)")

        pending = [pl for pl in targets if not is_fresh(pl.player_id)]
        total = len(targets)
        if not pending:
            return {"done": total, "total": total, "remaining": 0, "note": None}
        pl = pending[0]
        note = await _research_one(rs, pl, ctx.projections.get(pl.player_id), False, None)
        remaining = len(pending) - (1 if note is not None else 0)
        return {"done": total - remaining, "total": total, "remaining": remaining,
                "note": _note_dict(note), "player_id": pl.player_id, "name": pl.name,
                "error": getattr(rs, "last_error", None)}

    async def _research_one(rs: Any, pl: Player, proj: Projection | None, force: bool, cached: ResearchNote | None) -> ResearchNote | None:
        from ..research.claude import ResearchAborted

        try:
            note = await asyncio.wait_for(rs.research_player(pl, proj, force=force, cached=cached), timeout=RESEARCH_TIMEOUT_S)
        except asyncio.TimeoutError:
            raise HTTPException(504, f"research for {pl.name} timed out")
        except ResearchAborted as e:
            raise HTTPException(502, f"Claude request rejected: {e}")
        if note is not None:
            store().put(pl.player_id, note.to_dict())
        return note

    @app.get("/api/notes", dependencies=guarded)
    async def notes() -> dict:
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
