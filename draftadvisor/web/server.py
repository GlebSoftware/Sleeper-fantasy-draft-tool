"""FastAPI server for the local web app (DESIGN.md §3.8).

One process, one :class:`Session` (live Sleeper draft or offline mock draft). Every
endpoint answers immediately; long work (data prep, model training, context build,
polling) runs in background tasks or worker threads and reports through
``GET /api/status`` (``log`` lines, ``message``, ``busy``) and ``GET /api/state``.

Run with ``python run.py`` or ``draftadvisor web``.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import threading
import time
import webbrowser
from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import DEFAULT_SEASON, SKILL_POSITIONS, Settings, ensure_dirs, home_dir, models_dir
from ..models import DraftState, Player, PlayerValue, Projection, Recommendation, ResearchNote

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
DEFAULT_PORT = 8787
AVAILABLE_TOP_N = 250
BEST_N = 8


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class LogBuffer(logging.Handler):
    """Keeps the last N log lines from the draftadvisor package for the UI."""

    def __init__(self, maxlen: int = 400):
        super().__init__(level=logging.INFO)
        self.lines: deque[str] = deque(maxlen=maxlen)

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - trivial
        try:
            self.lines.append(f"{time.strftime('%H:%M:%S')} {record.levelname[:1]} {record.getMessage()}")
        except Exception:  # noqa: BLE001
            pass

    def say(self, text: str) -> None:
        self.lines.append(f"{time.strftime('%H:%M:%S')} • {text}")


class _TopN:
    """Advisor proxy that asks for a long ``best_overall`` list (the UI shows the available pool)."""

    def __init__(self, advisor: Any, n: int):
        self._advisor = advisor
        self._n = n

    def recommend(self, state: DraftState, **kw: Any) -> Recommendation:
        kw.setdefault("top_n", self._n)
        return self._advisor.recommend(state, **kw)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._advisor, name)


class Session:
    """All mutable server state (one draft at a time)."""

    def __init__(self) -> None:
        self.logbuf = LogBuffer()
        logging.getLogger("draftadvisor").addHandler(self.logbuf)
        self.mode = "idle"                     # idle | prepping | starting | live | mock
        self.busy = False
        self.message = "ready"
        self.error: str | None = None
        self.season = DEFAULT_SEASON
        self.sleeper_ok: bool | None = None
        self.ctx: Any = None                   # app.AppContext of the running session
        self.board_ctx: Any = None             # default context for the Board tab (no session)
        self.loop: Any = None                  # cli.DraftLoop
        self.poller: Any = None
        self.client: Any = None
        self.mock: Any = None
        self.snapshot: Any = None              # capture.LeagueSnapshot
        self.settings: Settings = Settings.from_env()
        self.status: dict = {}
        self.task: asyncio.Task | None = None
        self.stop_event = asyncio.Event()
        self.lock = asyncio.Lock()
        self.bot_delay = 1.0
        self.autopilot = False
        self.started_at: float | None = None
        self.turn_key: Any = None
        self.turn_started_at: float | None = None
        self._payload_key: Any = None
        self._payload: dict | None = None
        self._vorp_cache: dict[int, dict[str, float]] = {}

    # -- lifecycle ---------------------------------------------------------------
    async def stop(self) -> None:
        self.stop_event.set()
        if self.task is not None and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        if self.client is not None:
            try:
                await self.client.aclose()
            except Exception:  # noqa: BLE001
                pass
        self.task = None
        self.loop = None
        self.poller = None
        self.client = None
        self.mock = None
        self.ctx = None
        self.snapshot = None
        self.mode = "idle"
        self.busy = False
        self.message = "stopped"
        self.turn_key = None
        self.turn_started_at = None
        self._payload = None
        self._payload_key = None
        self.stop_event = asyncio.Event()

    def spawn(self, coro: Any, label: str) -> None:
        """Run ``coro`` in the background; record failures instead of raising."""
        async def runner() -> None:
            try:
                await coro
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                log.exception("%s failed: %s", label, e)
                self.error = f"{label}: {e}"
                self.message = f"{label} failed: {e}"
                self.busy = False
                if self.mode in ("starting", "prepping"):
                    self.mode = "idle"
        self.task = asyncio.ensure_future(runner())

    @property
    def state(self) -> DraftState | None:
        return self.loop.state if self.loop is not None else None

    @property
    def rec(self) -> Recommendation | None:
        return self.loop.rec if self.loop is not None else None

    def summary(self) -> dict | None:
        if self.mode not in ("live", "mock", "starting"):
            return None
        league = self.ctx.league if self.ctx is not None else (self.snapshot.league if self.snapshot else None)
        st = self.state
        return {
            "mode": self.mode,
            "league_name": league.name if league else None,
            "draft_id": st.draft.draft_id if st else (self.snapshot.draft_id if self.snapshot else None),
            "my_slot": st.my_slot if st else (self.snapshot.my_slot if self.snapshot else None),
            "started_at": self.started_at,
        }

    def readiness(self) -> dict:
        from ..data.cache import raw_path

        data = raw_path(f"stats_player_week_{self.season - 1}.csv").exists() or \
            (home_dir() / "cache" / f"canonical_{self.season - 1}.csv.gz").exists()
        return {
            "data": bool(data),
            "model": (models_dir() / "projection_model.pkl").exists(),
            "claude": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "sleeper": self.sleeper_ok,
        }


SESSION = Session()


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _f(v: Any, nd: int = 1) -> float | None:
    try:
        if v is None:
            return None
        return round(float(v), nd)
    except (TypeError, ValueError):
        return None


def _note_dict(note: ResearchNote | None) -> dict | None:
    if note is None:
        return None
    return {"summary": note.summary, "injury_risk": note.injury_risk, "role_certainty": note.role_certainty,
            "upside": note.upside, "downside": note.downside}


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
        "availability_after_next": None, "reasons": [], "warnings": [],
        "flags": list(pr.flags) if pr else [],
        "note": _note_dict(notes.get(pl.player_id)) if notes else None, "drafted_by": None,
    }


def card_from_value(v: PlayerValue, notes: Mapping[str, ResearchNote] | None = None) -> dict:
    c = card_from_player(v.player, v.projection, notes, v.vorp)
    c.update({
        "vona": _f(v.vona), "marginal": _f(v.marginal_value), "score": _f(v.score), "tier": v.tier,
        "pos_rank": v.pos_rank, "overall_rank": v.overall_rank,
        "availability_next": _f(v.availability_next, 3), "availability_after_next": _f(v.availability_after_next, 3),
        "reasons": list(v.reasons), "warnings": list(v.warnings),
    })
    return c


def _snapshot_payload(sess: Session) -> dict | None:
    """Capture snapshot (live) or a synthesised equivalent (mock) for the League tab."""
    from ..capture import scoring_diff, strategy_flags

    snap = sess.snapshot
    st = sess.state
    league = sess.ctx.league if sess.ctx is not None else (snap.league if snap else None)
    draft = st.draft if st is not None else (snap.draft if snap else None)
    if league is None and draft is None:
        return None
    my_slot = st.my_slot if st is not None else (snap.my_slot if snap else None)
    order = []
    if draft is not None:
        for slot in range(1, draft.teams + 1):
            m = st.manager_for_slot(slot) if st is not None else (snap.manager_for_slot(slot) if snap else None)
            order.append({
                "slot": slot,
                "display_name": (m.display_name if m else (f"Slot {slot}")),
                "team_name": (m.team_name if m else None),
                "roster_id": draft.original_roster_for_slot(slot),
                "picks": draft.picks_for_slot(slot)[:6],
                "is_me": slot == my_slot,
            })
    diff = snap.diff if (snap and snap.diff) else (scoring_diff(league.scoring_settings) if league else None)
    flags = snap.flags if (snap and snap.flags) else strategy_flags(league, draft)
    return {
        "flags": list(flags),
        "diff": diff.to_dict() if diff else None,
        "draft_order": order,
        "my_picks": (draft.picks_for_slot(my_slot) if (draft and my_slot) else []),
        "captured_at": snap.captured_at if snap else None,
        "league": {
            "name": league.name if league else None, "league_id": league.league_id if league else None,
            "season": league.season if league else None, "teams": league.total_rosters if league else None,
            "roster_positions": list(league.roster_positions) if league else [],
            "settings": {k: v for k, v in (league.settings.items() if league else []) if isinstance(v, (int, float, str))},
            "scoring_settings": dict(league.scoring_settings) if league else {},
        } if league else None,
        "draft": {
            "draft_id": draft.draft_id, "type": draft.type, "status": draft.status, "teams": draft.teams,
            "rounds": draft.rounds, "pick_timer": draft.pick_timer, "reversal_round": draft.reversal_round,
            "start_time": draft.start_time,
        } if draft else None,
    }


def build_state_payload(sess: Session) -> dict:
    st, rec, ctx = sess.state, sess.rec, sess.ctx
    now = time.time()
    if st is None or ctx is None:
        return {"mode": sess.mode, "version": 0, "ts": now, "draft": None, "league": None, "me": None, "best": [],
                "by_position": {}, "available": [], "recent": [], "opponents": [], "pressure": {}, "notes": [],
                "claude": {"status": sess.status.get("claude", "off"), "advice": None},
                "snapshot": _snapshot_payload(sess), "status": dict(sess.status)}
    # turn tracking (for the countdown)
    key = (st.is_my_turn, st.next_pick_no)
    if st.is_my_turn and key != sess.turn_key:
        sess.turn_key = key
        # Sleeper's clock started at the previous pick (draft.last_picked, epoch ms); if we
        # noticed the turn late (restart, slow bootstrap) use the real start, not "now".
        start = now
        lp = st.draft.last_picked
        if lp and st.picks and lp > 1e12:
            start = min(now, lp / 1000.0)
        sess.turn_started_at = start
    elif not st.is_my_turn:
        sess.turn_key = None
        sess.turn_started_at = None
    cache_key = (st.version, id(rec), rec.claude_advice if rec else None, sess.status.get("claude"), sess.mode)
    if sess._payload is not None and sess._payload_key == cache_key:
        payload = sess._payload
    else:
        payload = _fresh_payload(sess, st, rec, ctx)
        sess._payload_key = cache_key
        sess._payload = payload
    d = payload["draft"]
    d["turn_started_at"] = sess.turn_started_at
    if st.is_my_turn and st.draft.pick_timer and sess.turn_started_at:
        d["seconds_left"] = max(0, int(round(st.draft.pick_timer - (now - sess.turn_started_at))))
    else:
        d["seconds_left"] = None
    payload["ts"] = now
    payload["mode"] = sess.mode
    payload["claude"] = {"status": sess.status.get("claude", "off"), "advice": rec.claude_advice if rec else None}
    payload["status"] = {k: v for k, v in sess.status.items() if isinstance(v, (int, float, str, type(None)))}
    payload["status"]["sources"] = {k: v for k, v in (ctx.sources or {}).items() if isinstance(v, (int, float, str, bool))}
    payload["status"]["message"] = sess.message
    return payload


def _fresh_payload(sess: Session, st: DraftState, rec: Recommendation | None, ctx: Any) -> dict:
    from ..ui.dashboard import assign_roster_slots

    players: Mapping[str, Player] = ctx.players
    projections: Mapping[str, Projection] = ctx.projections
    notes = ctx.notes or {}
    league = ctx.league
    otc = st.on_the_clock_slot
    draft = {
        "type": st.draft.type, "status": st.draft.status, "teams": st.teams, "rounds": st.draft.rounds,
        "pick_timer": st.draft.pick_timer, "current_round": st.current_round, "next_pick_no": st.next_pick_no,
        "total_picks": st.draft.total_picks,
        "on_the_clock": {"slot": otc, "label": st.slot_label(otc)} if otc else None,
        "is_my_turn": st.is_my_turn, "my_slot": st.my_slot, "my_next_pick_no": st.my_next_pick_no,
        "my_pick_after_next": st.my_pick_after_next, "picks_until_my_turn": st.picks_until_my_turn,
        "is_complete": st.is_complete, "turn_started_at": None, "seconds_left": None,
    }
    league_d = {
        "name": league.name, "scoring_type": league.scoring_type, "scoring_description": ctx.engine.describe(),
        "roster_positions": list(league.roster_positions), "teams": league.total_rosters, "season": league.season,
    }
    # my roster
    me = None
    if rec is not None and rec.my_roster is not None:
        rs = rec.my_roster
        rows = assign_roster_slots([(pl, projections[pl.player_id].points if pl.player_id in projections else None)
                                    for pl in rs.players], league.roster_positions)
        me = {
            "slots": [{"slot": slot, "player": (card_from_player(pl, projections.get(pl.player_id), notes) if pl else None)}
                      for slot, pl, _ in rows],
            "needs": rs.needs(), "bye_clashes": {str(k): v for k, v in rs.bye_weeks.items() if v >= 2},
            "lineup_points": _f(rs.lineup_points), "bench_points": _f(rs.bench_points),
            "position_counts": dict(rs.position_counts), "open_starters": dict(rs.open_starters),
        }
    elif st.my_slot is not None:
        mine = [players[p.player_id] for p in st.my_picks() if p.player_id in players]
        rows = assign_roster_slots([(pl, projections[pl.player_id].points if pl.player_id in projections else None) for pl in mine],
                                   league.roster_positions)
        me = {"slots": [{"slot": slot, "player": (card_from_player(pl, projections.get(pl.player_id), notes) if pl else None)}
                        for slot, pl, _ in rows], "needs": [], "bye_clashes": {}, "lineup_points": None,
              "bench_points": None, "position_counts": {}, "open_starters": {}}
    # recommendations
    all_values = list(rec.best_overall) if rec else []
    best = [card_from_value(v, notes) for v in all_values[:BEST_N]]
    available = [card_from_value(v, notes) for v in all_values[:AVAILABLE_TOP_N]]
    by_position = {}
    if rec is not None:
        for pos, adv in rec.by_position.items():
            by_position[pos] = {
                "action": adv.action, "rationale": adv.rationale,
                "expected_next_available": _f(adv.expected_next_available), "drop_off": _f(adv.drop_off),
                "candidates": [card_from_value(v, notes) for v in adv.candidates[:3]],
            }
    # recent picks
    recent = []
    for p in sorted(st.picks, key=lambda x: -x.pick_no)[:12]:
        pl = players.get(p.player_id)
        slot = p.draft_slot
        for s_, rid in st.draft.slot_to_roster_id.items():
            if p.roster_id is not None and rid == p.roster_id:
                slot = s_
                break
        recent.append({"pick_no": p.pick_no, "round": p.round, "slot": slot, "label": st.slot_label(slot),
                       "player_id": p.player_id, "name": pl.name if pl else p.player_name,
                       "position": pl.position if pl else p.position, "team": pl.team if pl else p.metadata.get("team"),
                       "is_me": slot == st.my_slot})
    # opponents
    opponents = []
    if rec is not None:
        for rs in rec.opponent_rosters:
            fut = [n for n in st.draft.picks_for_slot(rs.slot) if n >= st.next_pick_no]
            opponents.append({"slot": rs.slot, "label": rs.label, "needs": rs.needs(), "next_pick": fut[0] if fut else None,
                              "position_counts": dict(rs.position_counts),
                              "players": [{"name": pl.name, "position": pl.position} for pl in rs.players]})
    return {
        "mode": sess.mode, "version": st.version, "ts": time.time(), "draft": draft, "league": league_d, "me": me,
        "best": best, "by_position": by_position, "available": available, "recent": recent, "opponents": opponents,
        "pressure": {k: _f(v, 2) for k, v in (rec.position_pressure.items() if rec else [])},
        "notes": list(rec.notes) if rec else [],
        "claude": {"status": sess.status.get("claude", "off"), "advice": rec.claude_advice if rec else None},
        "snapshot": _snapshot_payload(sess), "status": {},
    }


# ---------------------------------------------------------------------------
# Background flows
# ---------------------------------------------------------------------------


def _make_loop(sess: Session, ctx: Any) -> Any:
    from ..cli import DraftLoop

    sess.status = {"claude": "off"}
    return DraftLoop(_TopN(ctx.advisor, AVAILABLE_TOP_N), ctx.players, None, researcher=ctx.researcher,
                     notes=ctx.notes, status=sess.status, tui=False, out=lambda *_a, **_k: None)


async def _run_prep(sess: Session, refresh: bool, research: bool, top: int) -> None:
    from ..app import prep

    sess.mode, sess.busy, sess.error = "prepping", True, None
    sess.message = "preparing data and model"
    say = sess.logbuf.say
    settings = Settings.from_env()

    def work() -> Any:
        return asyncio.run(prep(settings, research=research, refresh=refresh, top=top, progress=say))

    try:
        ctx = await asyncio.to_thread(work)
        sess.board_ctx = ctx
        metrics = (ctx.sources or {}).get("metrics") or {}
        for pos in ("QB", "RB", "WR", "TE"):
            m = metrics.get(pos) if isinstance(metrics, dict) else None
            if m:
                say(f"backtest {pos}: model MAE {m.get('mae_model', 0):.2f} vs last-season {m.get('mae_last', 0):.2f} ppg")
        sess.message = "data and model ready"
    finally:
        sess.busy = False
        sess.mode = "idle"


async def _start_live(sess: Session, req: "LiveStart") -> None:
    from ..app import build_context
    from ..capture import capture_league
    from ..sleeper.client import SleeperClient
    from ..sleeper.poller import DraftPoller

    sess.mode, sess.busy, sess.error = "starting", True, None
    sess.started_at = time.time()
    say = sess.logbuf.say
    settings = Settings.from_env(league_id=req.league_id, draft_id=req.draft_id, username=req.username,
                                 user_id=req.user_id, slot=req.slot)
    settings.use_claude = bool(req.use_claude)
    client = SleeperClient()
    sess.client = client
    sess.message = "capturing league info"
    say("capturing league, draft order, scoring")
    snap = await capture_league(client, req.league_id, req.draft_id, username=req.username, user_id=req.user_id,
                                slot=req.slot, season=settings.season)
    sess.snapshot = snap
    sess.sleeper_ok = True
    if snap.draft_id is None:
        raise RuntimeError("no draft found for that league")
    settings.league_id, settings.draft_id = snap.league_id, snap.draft_id
    poller = DraftPoller(client, snap.draft_id, league_id=snap.league_id, settings=settings, username=req.username,
                         user_id=req.user_id, slot=req.slot)
    sess.poller = poller
    sess.message = "loading draft state"
    state = await poller.bootstrap()
    sleeper_players = sleeper_proj = None
    sess.message = "fetching players and projections"
    try:
        sleeper_players = await client.get_players()
    except Exception as e:  # noqa: BLE001
        say(f"Sleeper players unavailable: {e}")
    try:
        sleeper_proj = await client.get_season_projections(settings.season)
    except Exception as e:  # noqa: BLE001
        say(f"Sleeper projections unavailable: {e}")
    sess.message = "building projections (first time takes a few seconds)"
    league = snap.league or state.league
    ctx = await asyncio.to_thread(build_context, settings, league, state.draft, sleeper_players=sleeper_players,
                                  sleeper_proj=sleeper_proj, progress=say)
    sess.ctx = ctx
    sess.loop = _make_loop(sess, ctx)
    sess.mode, sess.busy = "live", False
    sess.message = f"live: {league.name if league else snap.draft_id}"
    say("advisor running")
    try:
        await sess.loop.run(poller, sess.stop_event)
        if sess.state is not None and sess.state.is_complete:
            sess.message = "draft complete"
    finally:
        try:
            await client.aclose()
        except Exception:  # noqa: BLE001
            pass


async def _start_mock(sess: Session, req: "MockStart") -> None:
    from ..app import build_context
    from ..mock.simulator import MockDraft, make_mock_draft, make_mock_league

    sess.mode, sess.busy, sess.error = "starting", True, None
    sess.started_at = time.time()
    say = sess.logbuf.say
    league = make_mock_league(req.teams, req.rounds, req.scoring, req.superflex)
    league.league_id = f"mock_{req.scoring}{'_sf' if req.superflex else ''}"      # projection cache name
    draft = make_mock_draft(league, req.slot, req.teams, req.rounds)
    settings = Settings.from_env(league_id=league.league_id)
    settings.use_claude = bool(req.use_claude)
    sess.message = "building projections (first time takes a few seconds)"
    ctx = await asyncio.to_thread(build_context, settings, league, draft, offline=True, use_model=True, progress=say)
    sess.ctx = ctx
    md = MockDraft(ctx.players, league, draft, req.slot, seed=req.seed)
    sess.mock = md
    sess.bot_delay = max(0.0, float(req.bot_delay))
    sess.autopilot = bool(req.autopilot)
    sess.loop = _make_loop(sess, ctx)
    sess.mode, sess.busy = "mock", False
    sess.message = f"mock draft: {req.teams} teams, slot {req.slot}"
    async with sess.lock:
        await sess.loop.handle(md.state())
    while not md.is_complete and not sess.stop_event.is_set():
        if md.is_my_turn:
            if sess.autopilot and sess.rec is not None and sess.rec.best_overall:
                async with sess.lock:
                    if md.is_my_turn:
                        md.make_pick(sess.rec.best_overall[0].player_id)
                        await sess.loop.handle(md.state())
                await asyncio.sleep(min(sess.bot_delay, 0.5))
                continue
            await asyncio.sleep(0.2)
            continue
        async with sess.lock:
            if not md.is_my_turn and not md.is_complete:
                md.bot_pick()
                await sess.loop.handle(md.state())
        await asyncio.sleep(sess.bot_delay)
    sess.message = "mock draft complete" if md.is_complete else "stopped"


async def _run_research(sess: Session, top: int) -> None:
    ctx = sess.ctx or sess.board_ctx
    if ctx is None or not getattr(ctx.researcher, "enabled", False):
        raise RuntimeError("Claude is not enabled (set ANTHROPIC_API_KEY) or no context loaded")
    players = sorted((p for p in ctx.players.values() if p.adp is not None), key=lambda p: p.adp)[:top]
    sess.busy = True
    sess.message = f"researching {len(players)} players with Claude"
    try:
        notes = await ctx.researcher.research_players(players, ctx.projections,
                                                      progress=lambda d, t, n: sess.logbuf.say(f"research {d}/{t}: {n}"))
        ctx.notes.update(notes)
        if sess.loop is not None:
            sess.loop.notes.update(notes)
        err = getattr(ctx.researcher, "last_error", None)
        if err:
            sess.logbuf.say(f"research problem: {err}")
        # re-blend projections so injury_risk / role_certainty change points, VORP and tiers
        try:
            from ..app import rebuild_projections

            await asyncio.to_thread(rebuild_projections, ctx, ctx.notes)
            if sess.loop is not None:
                sess.loop.advisor = _TopN(ctx.advisor, AVAILABLE_TOP_N)
                if sess.state is not None:
                    async with sess.lock:
                        await sess.loop.handle(sess.state)
            sess.logbuf.say("projections re-blended with research notes")
        except ImportError:
            sess.logbuf.say("projections not re-blended (rebuild_projections unavailable)")
        sess.message = f"research done: {len(notes)} notes"
    finally:
        sess.busy = False


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------


class PrepReq(BaseModel):
    refresh: bool = False
    research: bool = False
    top: int = 200


class LookupReq(BaseModel):
    username: str


class LiveStart(BaseModel):
    league_id: str | None = None
    draft_id: str | None = None
    username: str | None = None
    user_id: str | None = None
    slot: int | None = None
    use_claude: bool = True


class MockStart(BaseModel):
    teams: int = 12
    rounds: int = 15
    slot: int = 5
    scoring: str = "half_ppr"
    superflex: bool = False
    seed: int | None = None
    bot_delay: float = 1.0
    autopilot: bool = False
    use_claude: bool = False


class PickReq(BaseModel):
    player_id: str


class AutopilotReq(BaseModel):
    enabled: bool


class AskReq(BaseModel):
    question: str


class ResearchReq(BaseModel):
    top: int = 150


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def create_app(session: Session | None = None) -> FastAPI:
    sess = session or SESSION
    app = FastAPI(title="draftadvisor", docs_url="/api/docs", redoc_url=None)
    app.state.session = sess
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    async def index() -> Any:
        page = STATIC_DIR / "index.html"
        if not page.exists():
            raise HTTPException(404, "index.html missing: the frontend has not been built")
        return FileResponse(str(page))

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Any:
        from fastapi.responses import Response

        svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><rect width="64" height="64" rx="12" fill="#1f6f43"/>'
               '<ellipse cx="32" cy="32" rx="22" ry="14" fill="#8b4a1f" stroke="#f3e9d2" stroke-width="3"/>'
               '<path d="M22 32h20M27 27v10M32 26v12M37 27v10" stroke="#f3e9d2" stroke-width="3" stroke-linecap="round"/></svg>')
        return Response(content=svg, media_type="image/svg+xml")

    @app.get("/api/status")
    async def status() -> dict:
        return {
            "mode": sess.mode, "busy": sess.busy, "message": sess.message, "error": sess.error,
            "ready": sess.readiness(), "log": list(sess.logbuf.lines)[-120:], "season": sess.season,
            "home": str(home_dir()), "session": sess.summary(),
        }

    @app.post("/api/prep")
    async def api_prep(req: PrepReq) -> dict:
        if sess.busy or sess.mode in ("live", "mock", "starting"):
            raise HTTPException(409, f"busy ({sess.mode}); stop the session first")
        sess.spawn(_run_prep(sess, req.refresh, req.research, req.top), "prep")
        return {"ok": True}

    @app.post("/api/lookup")
    async def api_lookup(req: LookupReq) -> dict:
        from ..sleeper.client import SleeperClient, SleeperNotFound

        name = req.username.strip()
        if not name:
            raise HTTPException(400, "username required")
        try:
            async with SleeperClient() as client:
                user = await client.get_user(name)
                uid = str(user.get("user_id"))
                leagues = await client.get_user_leagues(uid, sess.season)
                drafts = await client.get_user_drafts(uid, sess.season)
        except SleeperNotFound:
            raise HTTPException(404, f"Sleeper user '{name}' not found")
        except Exception as e:  # noqa: BLE001
            sess.sleeper_ok = False
            raise HTTPException(502, f"could not reach the Sleeper API: {e}")
        sess.sleeper_ok = True
        return {
            "user": {"user_id": uid, "display_name": user.get("display_name"), "username": user.get("username")},
            "leagues": [{
                "league_id": lg.get("league_id"), "name": lg.get("name"), "season": lg.get("season"),
                "total_rosters": lg.get("total_rosters"), "status": lg.get("status"), "draft_id": lg.get("draft_id"),
                "scoring_type": ("ppr" if float((lg.get("scoring_settings") or {}).get("rec", 0) or 0) >= 0.75
                                 else "half_ppr" if float((lg.get("scoring_settings") or {}).get("rec", 0) or 0) >= 0.25 else "std"),
                "roster_positions": lg.get("roster_positions") or [],
            } for lg in leagues or []],
            "drafts": [{
                "draft_id": d.get("draft_id"), "league_id": d.get("league_id"), "status": d.get("status"),
                "type": d.get("type"), "season": d.get("season"), "start_time": d.get("start_time"),
                "teams": (d.get("settings") or {}).get("teams"), "rounds": (d.get("settings") or {}).get("rounds"),
                "name": (d.get("metadata") or {}).get("name"),
            } for d in drafts or []],
        }

    @app.post("/api/live/start")
    async def api_live_start(req: LiveStart) -> dict:
        if not (req.league_id or req.draft_id):
            raise HTTPException(400, "league_id or draft_id required")
        if sess.mode in ("live", "mock", "starting", "prepping") or sess.busy:
            raise HTTPException(409, f"busy ({sess.mode}); stop the session first")
        sess.spawn(_start_live(sess, req), "live draft")
        return {"ok": True}

    @app.post("/api/mock/start")
    async def api_mock_start(req: MockStart) -> dict:
        if sess.mode in ("live", "mock", "starting", "prepping") or sess.busy:
            raise HTTPException(409, f"busy ({sess.mode}); stop the session first")
        if not (2 <= req.teams <= 20 and 3 <= req.rounds <= 30 and 1 <= req.slot <= req.teams):
            raise HTTPException(400, "teams 2-20, rounds 3-30, slot within teams")
        if req.scoring not in ("ppr", "half_ppr", "std"):
            raise HTTPException(400, "scoring must be ppr, half_ppr or std")
        sess.spawn(_start_mock(sess, req), "mock draft")
        return {"ok": True}

    @app.post("/api/mock/pick")
    async def api_mock_pick(req: PickReq) -> dict:
        if sess.mode != "mock" or sess.mock is None:
            raise HTTPException(409, "no mock draft running")
        async with sess.lock:
            if not sess.mock.is_my_turn:
                raise HTTPException(409, "it is not your turn")
            try:
                pick = sess.mock.make_pick(req.player_id)
            except ValueError as e:
                raise HTTPException(400, str(e))
            await sess.loop.handle(sess.mock.state())
        return {"ok": True, "pick": {"pick_no": pick.pick_no, "player_id": pick.player_id, "name": pick.player_name}}

    @app.post("/api/mock/auto")
    async def api_mock_auto() -> dict:
        if sess.mode != "mock" or sess.mock is None:
            raise HTTPException(409, "no mock draft running")
        async with sess.lock:
            if not sess.mock.is_my_turn:
                raise HTTPException(409, "it is not your turn")
            rec = sess.rec
            if rec is None or not rec.best_overall:
                raise HTTPException(409, "no recommendation yet")
            pick = sess.mock.make_pick(rec.best_overall[0].player_id)
            await sess.loop.handle(sess.mock.state())
        return {"ok": True, "pick": {"pick_no": pick.pick_no, "player_id": pick.player_id, "name": pick.player_name}}

    @app.post("/api/mock/autopilot")
    async def api_mock_autopilot(req: AutopilotReq) -> dict:
        sess.autopilot = bool(req.enabled)
        return {"ok": True, "autopilot": sess.autopilot}

    @app.post("/api/stop")
    async def api_stop() -> dict:
        await sess.stop()
        return {"ok": True}

    @app.get("/api/state")
    async def api_state() -> dict:
        return build_state_payload(sess)

    @app.get("/api/player/{player_id}")
    async def api_player(player_id: str) -> dict:
        ctx = sess.ctx or sess.board_ctx
        if ctx is None or player_id not in ctx.players:
            raise HTTPException(404, "unknown player (no context loaded?)")
        pl = ctx.players[player_id]
        pr = ctx.projections.get(player_id)
        card: dict
        st = sess.state
        v = None
        if st is not None and player_id not in st.drafted_ids:
            try:
                v = ctx.advisor.value_of(st, player_id)
            except Exception:  # noqa: BLE001
                v = None
        card = card_from_value(v, ctx.notes) if v is not None else card_from_player(pl, pr, ctx.notes)
        card["projection"] = {"components": dict(pr.components), "weights": dict(pr.weights), "flags": list(pr.flags),
                              "stat_line": {k: _f(x, 1) for k, x in pr.stat_line.items() if x}} if pr else None
        try:
            card["explain"] = ctx.advisor.explain_pick(st, player_id) if st is not None else ""
        except Exception:  # noqa: BLE001
            card["explain"] = ""
        return card

    @app.post("/api/board/load")
    async def api_board_load() -> dict:
        from ..app import build_context

        if sess.ctx is not None:
            return {"ok": True, "source": "session"}
        if sess.board_ctx is None:
            if sess.busy:
                raise HTTPException(409, "busy")
            sess.busy = True
            sess.message = "building default projections"
            try:
                sess.board_ctx = await asyncio.to_thread(build_context, Settings.from_env(), None, None, offline=True,
                                                         use_model=True, progress=sess.logbuf.say)
                sess.message = "board ready"
            finally:
                sess.busy = False
        return {"ok": True, "source": "board"}

    @app.get("/api/projections")
    async def api_projections(position: str | None = None, top: int = 80, q: str | None = None) -> list[dict]:
        from ..data.crosswalk import normalize_name
        from ..strategy.replacement import vorp as vorp_fn

        ctx = sess.ctx or sess.board_ctx
        if ctx is None:
            raise HTTPException(409, "no projections loaded yet (start a draft or load the board)")
        key = id(ctx)
        if key not in sess._vorp_cache:
            try:
                sess._vorp_cache = {key: vorp_fn(ctx.projections, ctx.players, ctx.league)}
            except Exception:  # noqa: BLE001
                sess._vorp_cache = {key: {}}
        vorps = sess._vorp_cache[key]
        qn = normalize_name(q) if q else ""
        drafted = sess.state.drafted_ids if sess.state is not None else set()
        rows = []
        for pid, pr in ctx.projections.items():
            pl = ctx.players.get(pid)
            if pl is None or (position and pl.position != position.upper()):
                continue
            if qn and qn not in normalize_name(pl.name):
                continue
            c = card_from_player(pl, pr, ctx.notes, vorps.get(pid))
            c["drafted_by"] = "drafted" if pid in drafted else None
            rows.append(c)
        rows.sort(key=lambda c: -(c["points"] or 0.0))
        return rows[: max(1, min(int(top), 600))]

    @app.post("/api/ask")
    async def api_ask(req: AskReq) -> dict:
        from ..research.claude import build_context_text

        ctx = sess.ctx or sess.board_ctx
        if ctx is None or not getattr(ctx.researcher, "enabled", False):
            raise HTTPException(503, "Claude is off: set ANTHROPIC_API_KEY and start a session")
        text = build_context_text(sess.state, sess.rec, ctx.players, ctx.notes) if sess.state is not None else ""
        answer = await ctx.researcher.ask(req.question, text)
        if not answer:
            raise HTTPException(502, f"Claude unavailable: {getattr(ctx.researcher, 'last_error', 'unknown error')}")
        return {"answer": answer}

    @app.post("/api/research")
    async def api_research(req: ResearchReq) -> dict:
        if sess.busy:
            raise HTTPException(409, "busy")
        ctx = sess.ctx or sess.board_ctx
        if ctx is None:
            raise HTTPException(409, "start a session or load the board first")
        sess.spawn(_run_research(sess, req.top), "research")
        return {"ok": True}

    return app


app = create_app()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="draftadvisor web", description="local web app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    parser.add_argument("--home", help="data directory (default ./data or $DRAFTADVISOR_HOME)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.home:
        os.environ["DRAFTADVISOR_HOME"] = args.home
    ensure_dirs()
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
