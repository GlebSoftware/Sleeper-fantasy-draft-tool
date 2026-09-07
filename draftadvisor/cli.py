"""Command line interface (argparse). See DESIGN.md §3.6.

Subcommands: ``prep``, ``train``, ``draft``, ``mock``, ``projections``, ``trade``, ``analyze``,
``research``, ``ask``, ``ids``, ``capture``. Global flags: ``--home DIR``, ``--offline``, ``-v``.

The live draft loop (:class:`DraftLoop` / :func:`run_draft_loop`) is written against small
duck-typed interfaces (a poller-like ``run(on_update, stop)`` or an async iterable of states, an
advisor with ``recommend``, a dashboard with ``update``) so it is testable with fakes.
"""
from __future__ import annotations

import argparse
import asyncio
import difflib
import logging
import os
import sys
import time
from typing import Any, AsyncIterable, Callable, Iterable, Mapping, Sequence

from rich import box
from rich.console import Console
from rich.table import Table

from .config import DEFAULT_SEASON, SKILL_POSITIONS, TRAIN_SEASONS, Settings
from .models import DraftState, LeagueSettings, Player, Projection, Recommendation

log = logging.getLogger(__name__)

__all__ = ["main", "build_parser", "DraftLoop", "run_draft_loop", "resolve_player_input", "parse_seasons"]

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_NETWORK = 3
EXIT_ERROR = 1


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _add_league_args(p: argparse.ArgumentParser, draft: bool = True) -> None:
    p.add_argument("--league", dest="league_id", help="Sleeper league id")
    if draft:
        p.add_argument("--draft", dest="draft_id", help="Sleeper draft id")


def _add_identity_args(p: argparse.ArgumentParser) -> None:
    g = p.add_mutually_exclusive_group()
    g.add_argument("--username", help="your Sleeper username (case-insensitive)")
    g.add_argument("--user-id", dest="user_id", help="your Sleeper user id")
    g.add_argument("--slot", type=int, help="your draft slot (1-based)")


def build_parser() -> argparse.ArgumentParser:
    """The argparse parser for every subcommand."""
    p = argparse.ArgumentParser(prog="draftadvisor", description="Live Sleeper fantasy-football draft advisor.")
    p.add_argument("--home", help="data directory (sets DRAFTADVISOR_HOME; default ./data)")
    p.add_argument("--offline", action="store_true", help="never touch the network")
    p.add_argument("--season", type=int, default=None, help=f"season being drafted (default {DEFAULT_SEASON})")
    p.add_argument("-v", "--verbose", action="count", default=0, help="-v info, -vv debug")
    sub = p.add_subparsers(dest="command", metavar="command")
    sub.required = True

    s = sub.add_parser("prep", help="download data, train the model, cache projections (+ optional Claude research)")
    _add_league_args(s)
    _add_identity_args(s)
    s.add_argument("--refresh", action="store_true", help="re-download data / rebuild caches")
    s.add_argument("--no-train", action="store_true", help="skip model training")
    s.add_argument("--research", action="store_true", help="run Claude research on the top players")
    s.add_argument("--top", type=int, default=200, help="how many players to research (default 200)")
    s.set_defaults(func=cmd_prep)

    s = sub.add_parser("train", help="(re)train the projection model and print the backtest")
    s.add_argument("--seasons", default=None, help="e.g. 2019-2025 or 2020,2021,2022")
    s.add_argument("--refresh", action="store_true")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("draft", help="live draft advisor")
    _add_league_args(s)
    _add_identity_args(s)
    s.add_argument("--poll", type=float, default=None, help="poll interval seconds (default 2)")
    s.add_argument("--no-claude", action="store_true", help="disable Claude advice")
    s.add_argument("--no-tui", action="store_true", help="plain text output instead of the dashboard")
    s.add_argument("--no-model", action="store_true", help="skip the ML model (offline projections)")
    s.add_argument("--refresh", action="store_true", help="rebuild cached projections")
    s.set_defaults(func=cmd_draft)

    s = sub.add_parser("mock", help="offline mock draft against ADP bots")
    s.add_argument("--teams", type=int, default=12)
    s.add_argument("--rounds", type=int, default=15)
    s.add_argument("--slot", type=int, default=None, help="your draft slot (default: middle of the order)")
    s.add_argument("--scoring", choices=("ppr", "half_ppr", "std"), default="half_ppr")
    s.add_argument("--superflex", action="store_true")
    s.add_argument("--auto", action="store_true", help="always take the top recommendation")
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--speed", type=float, default=0.5, help="seconds between bot picks (0 = instant)")
    s.add_argument("--no-tui", action="store_true", help="plain text instead of the dashboard")
    s.add_argument("--no-model", action="store_true", help="skip the ML model (offline projections)")
    s.add_argument("--no-claude", action="store_true")
    s.add_argument("--refresh", action="store_true", help="rebuild cached projections")
    s.set_defaults(func=cmd_mock)

    s = sub.add_parser("projections", help="projection table")
    _add_league_args(s)
    s.add_argument("--position", choices=SKILL_POSITIONS, default=None)
    s.add_argument("--top", type=int, default=40)
    s.add_argument("--no-model", action="store_true")
    s.add_argument("--refresh", action="store_true")
    s.set_defaults(func=cmd_projections)

    s = sub.add_parser("trade", help="evaluate a trade")
    _add_league_args(s, draft=False)
    s.add_argument("--me", required=True, help="my Sleeper username / display name")
    s.add_argument("--them", required=True, help="their Sleeper username / display name")
    s.add_argument("--give", required=True, help='players I give: "Name, Name"')
    s.add_argument("--get", required=True, help='players I get: "Name, Name"')
    s.add_argument("--no-model", action="store_true")
    s.set_defaults(func=cmd_trade)

    s = sub.add_parser("analyze", help="post-draft roster analysis for every team")
    _add_league_args(s, draft=False)
    s.add_argument("--username", default=None)
    s.add_argument("--no-model", action="store_true")
    s.set_defaults(func=cmd_analyze)

    s = sub.add_parser("research", help="run Claude research notes")
    _add_league_args(s, draft=False)
    s.add_argument("--top", type=int, default=200)
    s.add_argument("--no-model", action="store_true")
    s.set_defaults(func=cmd_research)

    s = sub.add_parser("ask", help="free-form Claude question with draft context")
    s.add_argument("question")
    _add_league_args(s)
    _add_identity_args(s)
    s.add_argument("--no-model", action="store_true")
    s.set_defaults(func=cmd_ask)

    s = sub.add_parser("ids", help="list your leagues and drafts with ids")
    s.add_argument("--username", required=True)
    s.set_defaults(func=cmd_ids)

    s = sub.add_parser("web", help="start the local web app (http://127.0.0.1:8787) and open the browser")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8787)
    s.add_argument("--no-browser", action="store_true")
    s.set_defaults(func=cmd_web)

    s = sub.add_parser("capture", help="one-time league info capture (scoring diff, draft order, your picks)")
    _add_league_args(s)
    _add_identity_args(s)
    s.set_defaults(func=cmd_capture)
    return p


def parse_seasons(text: str | None) -> list[int]:
    """"2019-2025" or "2020,2021" -> list of ints (default: TRAIN_SEASONS)."""
    if not text:
        return list(TRAIN_SEASONS)
    if "-" in text:
        lo, hi = text.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in text.split(",") if x.strip()]


def _settings_from_args(args: argparse.Namespace) -> Settings:
    s = Settings.from_env(
        league_id=getattr(args, "league_id", None),
        draft_id=getattr(args, "draft_id", None),
        username=getattr(args, "username", None),
        user_id=getattr(args, "user_id", None),
        slot=getattr(args, "slot", None) if getattr(args, "command", "") != "mock" else None,
        poll_seconds=getattr(args, "poll", None),
        season=getattr(args, "season", None),
    )
    if getattr(args, "no_claude", False):
        s.use_claude = False
    return s


def _configure_logging(verbose: int) -> None:
    level = logging.WARNING if verbose == 0 else (logging.INFO if verbose == 1 else logging.DEBUG)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stderr, force=True)
    logging.getLogger("httpx").setLevel(max(level, logging.WARNING))


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def _is_network_error(e: BaseException) -> bool:
    try:
        import httpx

        if isinstance(e, (httpx.HTTPError, httpx.InvalidURL)):
            return True
    except Exception:  # noqa: BLE001
        pass
    try:
        from .sleeper.client import SleeperAPIError

        if isinstance(e, SleeperAPIError):
            return True
    except Exception:  # noqa: BLE001
        pass
    return isinstance(e, (ConnectionError, TimeoutError, OSError))


def _network_message(e: BaseException) -> str:
    return (f"Could not reach the Sleeper API ({type(e).__name__}: {e}).\n"
            "Check your connection (or that api.sleeper.app is reachable from here); offline commands still work:\n"
            "  draftadvisor mock --auto        draftadvisor projections        draftadvisor train")


def _run(coro):
    """Run a coroutine from the CLI (Ctrl-C friendly)."""
    from .sleeper.client import run_sync

    return run_sync(coro)


# ---------------------------------------------------------------------------
# Draft loop
# ---------------------------------------------------------------------------


class DraftLoop:
    """Glue between a state source, the advisor, the dashboard and the Claude layer.

    ``advisor`` needs ``recommend(state)``; ``dashboard`` (optional) ``start/update/refresh/stop``;
    ``researcher`` (optional) ``enabled`` and ``async on_the_clock_advice(state, rec, players, notes)``.
    Claude advice is requested as a background task whenever it is my turn or I pick within one
    pick, and never blocks rendering.
    """

    def __init__(self, advisor: Any, players: Mapping[str, Player], dashboard: Any = None, *,
                 researcher: Any = None, notes: Mapping | None = None, status: dict | None = None,
                 tui: bool = True, out: Callable[[str], None] | None = None, refresh_seconds: float = 1.0,
                 claude_lookahead: int = 1, claude_grace_s: float = 2.0) -> None:
        self.advisor = advisor
        self.players = players
        self.dashboard = dashboard if tui else None
        self.researcher = researcher
        self.notes = dict(notes or {})
        self.status: dict = dict(status or {})
        self.tui = tui and dashboard is not None
        self.out = out or print
        self.refresh_seconds = refresh_seconds
        self.claude_lookahead = claude_lookahead
        self.claude_grace_s = claude_grace_s
        self.state: DraftState | None = None
        self.rec: Recommendation | None = None
        self.recommend_calls = 0
        self.claude_requests = 0
        self._claude_task: asyncio.Task | None = None
        self._ticker: asyncio.Task | None = None
        self._last_turn_key: tuple | None = None
        self.status.setdefault("claude", "off" if not (researcher is not None and getattr(researcher, "enabled", False)) else "idle")

    # -- rendering ----------------------------------------------------------------------
    def render(self) -> None:
        if self.state is None:
            return
        if self.tui and self.dashboard is not None:
            self.dashboard.update(self.state, self.rec, self.players, self.status)
        else:
            from .ui.dashboard import render_text

            self.out(render_text(self.rec, self.state, self.players))

    def _track_turn(self, state: DraftState) -> None:
        if state.is_my_turn:
            key = (state.next_pick_no,)
            if self._last_turn_key != key:
                self.status["turn_started_at"] = time.time()
                self._last_turn_key = key
        else:
            self.status.pop("turn_started_at", None)
            self._last_turn_key = None

    def _pull_source_stats(self, source: Any) -> None:
        for attr, key in (("last_latency_ms", "latency_ms"), ("last_error", "last_error"), ("poll_count", "poll_count")):
            if hasattr(source, attr):
                self.status[key] = getattr(source, attr)

    # -- state updates -------------------------------------------------------------------
    async def handle(self, state: DraftState, source: Any = None) -> Recommendation | None:
        """Process one state update: recommend, render, maybe ask Claude."""
        self.state = state
        self._track_turn(state)
        if source is not None:
            self._pull_source_stats(source)
        rec: Recommendation | None = None
        t0 = time.perf_counter()
        try:
            rec = self.advisor.recommend(state)
            self.recommend_calls += 1
        except Exception as e:  # noqa: BLE001
            log.exception("recommend failed: %s", e)
            self.status["last_error"] = f"recommend: {e}"
        self.status["compute_ms"] = (time.perf_counter() - t0) * 1000.0
        self.rec = rec
        self.render()
        if rec is not None and self.wants_claude(state):
            self.schedule_claude(state, rec)
        return rec

    def wants_claude(self, state: DraftState) -> bool:
        if self.researcher is None or not getattr(self.researcher, "enabled", False):
            return False
        if state.is_complete:
            return False
        until = state.picks_until_my_turn
        return state.is_my_turn or (until is not None and until <= self.claude_lookahead)

    def schedule_claude(self, state: DraftState, rec: Recommendation) -> asyncio.Task:
        """Kick off ``on_the_clock_advice`` in the background; fill ``rec.claude_advice`` when done."""
        if self._claude_task is not None and not self._claude_task.done():
            self._claude_task.cancel()
        self.claude_requests += 1
        self.status["claude"] = "thinking"

        async def _go() -> None:
            text = None
            try:
                text = await self.researcher.on_the_clock_advice(state, rec, self.players, self.notes)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                log.warning("Claude advice failed: %s", e)
                self.status["claude"] = "error"
            if text:
                rec.claude_advice = text
                self.status["claude"] = "ready"
            elif self.status.get("claude") == "thinking":
                self.status["claude"] = "no answer"
            if self.rec is rec:
                self.render()

        self._claude_task = asyncio.ensure_future(_go())
        return self._claude_task

    # -- ticking ---------------------------------------------------------------------------
    async def _tick(self) -> None:
        while True:
            await asyncio.sleep(self.refresh_seconds)
            if self.tui and self.dashboard is not None and self.state is not None:
                try:
                    self.dashboard.refresh(self.status)
                except Exception as e:  # noqa: BLE001
                    log.debug("dashboard refresh failed: %s", e)

    # -- main -------------------------------------------------------------------------------
    async def run(self, source: Any, stop: asyncio.Event | None = None) -> DraftState | None:
        """Consume ``source`` until the draft completes / ``stop`` is set / Ctrl-C."""
        if self.tui and self.dashboard is not None:
            self.dashboard.start()
        if self.tui and self.refresh_seconds > 0:
            self._ticker = asyncio.ensure_future(self._tick())
        grace = self.claude_grace_s
        try:
            if hasattr(source, "run") and callable(source.run):
                async def on_update(state: DraftState) -> None:
                    await self.handle(state, source)

                await source.run(on_update, stop)
            else:
                async for state in _aiter(source):
                    if stop is not None and stop.is_set():
                        break
                    await self.handle(state, source)
                    if state.is_complete:
                        break
        except asyncio.CancelledError:
            grace = 0.0                      # Ctrl-C: leave immediately
            raise
        finally:
            await self._shutdown(grace)
        return self.state

    async def _shutdown(self, grace: float) -> None:
        """Stop the ticker, let an in-flight Claude request finish for ``grace`` seconds, close the TUI."""
        if self._ticker is not None and not self._ticker.done():
            self._ticker.cancel()
        task = self._claude_task
        if task is not None and not task.done():
            if grace > 0:
                try:
                    await asyncio.wait({task}, timeout=grace)
                except Exception:  # noqa: BLE001
                    pass
            if not task.done():
                task.cancel()
                try:
                    await asyncio.wait({task}, timeout=0.5)
                except Exception:  # noqa: BLE001
                    pass
        if self.tui and self.dashboard is not None:
            self.dashboard.stop()


async def _aiter(source: Any):
    if hasattr(source, "__aiter__"):
        async for s in source:
            yield s
    else:
        for s in source:
            yield s


async def run_draft_loop(source: Any, advisor: Any, dashboard: Any = None, *, players: Mapping[str, Player],
                         researcher: Any = None, notes: Mapping | None = None, status: dict | None = None,
                         tui: bool = True, stop: asyncio.Event | None = None, out: Callable[[str], None] | None = None,
                         refresh_seconds: float = 1.0) -> DraftLoop:
    """Convenience wrapper: build a :class:`DraftLoop`, run it, return it (for inspection)."""
    loop = DraftLoop(advisor, players, dashboard, researcher=researcher, notes=notes, status=status, tui=tui,
                     out=out, refresh_seconds=refresh_seconds)
    await loop.run(source, stop)
    return loop


# ---------------------------------------------------------------------------
# Player name resolution (mock input, trade)
# ---------------------------------------------------------------------------


def resolve_player_input(text: str, candidates: Sequence[Player]) -> Player | None:
    """Exact player_id, exact name (case-insensitive), then fuzzy / substring name match."""
    q = text.strip()
    if not q:
        return None
    by_id = {p.player_id: p for p in candidates}
    if q in by_id:
        return by_id[q]
    ql = q.lower()
    exact = [p for p in candidates if p.name.lower() == ql]
    if exact:
        return exact[0]
    subs = [p for p in candidates if ql in p.name.lower()]
    if len(subs) == 1:
        return subs[0]
    if subs:
        return subs[0]
    names = [p.name for p in candidates]
    hits = difflib.get_close_matches(q, names, n=1, cutoff=0.6)
    if hits:
        return next(p for p in candidates if p.name == hits[0])
    return None


def _split_names(text: str) -> list[str]:
    return [t.strip() for t in text.replace(";", ",").split(",") if t.strip()]


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def _projection_table(ctx, top: int = 30, position: str | None = None, title: str | None = None) -> Table:
    t = Table(title=title or "Projections", box=box.SIMPLE_HEAD)
    for col, kw in (("#", {"justify": "right"}), ("Player", {}), ("Pos", {}), ("Tm", {}), ("Bye", {"justify": "right"}),
                    ("Pts", {"justify": "right"}), ("Floor", {"justify": "right"}), ("Ceil", {"justify": "right"}),
                    ("Std", {"justify": "right"}), ("PPG", {"justify": "right"}), ("G", {"justify": "right"}),
                    ("ADP", {"justify": "right"}), ("ECR", {"justify": "right"}), ("Src", {}), ("Flags", {})):
        t.add_column(col, **kw)
    rows = [(pid, pr) for pid, pr in ctx.projections.items() if pid in ctx.players]
    if position:
        rows = [(pid, pr) for pid, pr in rows if pr.position == position]
    rows.sort(key=lambda r: -r[1].points)
    for i, (pid, pr) in enumerate(rows[:top], start=1):
        pl = ctx.players[pid]
        src = "+".join(k for k, w in sorted(pr.weights.items(), key=lambda kv: -kv[1]) if w > 0) or "-"
        t.add_row(str(i), pl.name, pl.position, pl.team or "FA", str(pl.bye_week or "-"), f"{pr.points:.0f}",
                  f"{pr.floor:.0f}", f"{pr.ceiling:.0f}", f"{pr.std:.0f}", f"{pr.ppg:.1f}", f"{pr.games:.0f}",
                  f"{pl.adp:.0f}" if pl.adp else "-", f"{pl.ecr:.0f}" if pl.ecr else "-", src, ",".join(pr.flags))
    return t


def _metrics_table(metrics: Mapping[str, Any]) -> Table | None:
    rows = [(pos, metrics.get(pos)) for pos in SKILL_POSITIONS if isinstance(metrics.get(pos), Mapping)]
    if not rows:
        return None
    t = Table(title="Backtest (PPR ppg, players with >= 6 games)", box=box.SIMPLE_HEAD)
    for col in ("Pos", "n", "MAE model", "MAE last", "MAE career", "Sp model", "Sp last", "beats last"):
        t.add_column(col, justify="right" if col not in ("Pos", "beats last") else "left")
    for pos, r in rows:
        def f(k: str) -> str:
            v = r.get(k)
            return "-" if v is None else f"{float(v):.3f}"
        t.add_row(pos, str(r.get("n", "-")), f("mae_model"), f("mae_last"), f("mae_career"), f("spearman_model"),
                  f("spearman_last"), "yes" if r.get("beats_last") else "no")
    return t


def _print_sources(console: Console, ctx) -> None:
    s = ctx.sources
    console.print(f"[dim]players: {s.get('players')} ({len(ctx.players)}) • ECR: {'yes' if s.get('ecr') else 'no'} • "
                  f"ADP: {s.get('adp')} • projections: {s.get('projections')} ({len(ctx.projections)}) • "
                  f"notes: {s.get('notes')} • {s.get('total_ms', 0):.0f} ms[/dim]")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _league_for(settings: Settings, args: argparse.Namespace):
    """(league, draft) from Sleeper when ids are given and online, else from a snapshot, else None."""
    from .app import fetch_league, load_snapshot_league

    if not (settings.league_id or settings.draft_id):
        return None, None
    if not getattr(args, "offline", False):
        try:
            from .sleeper.client import SleeperClient

            async def go():
                async with SleeperClient() as client:
                    return await fetch_league(client, settings.league_id, settings.draft_id)

            league, draft = _run(go())
            if league is not None:
                return league, draft
        except Exception as e:  # noqa: BLE001
            log.warning("league lookup failed (%s); trying snapshot", e)
    return load_snapshot_league(settings)


def _build(settings: Settings, args: argparse.Namespace, league=None, draft=None, **kw):
    from .app import build_context

    if league is None:
        league, draft = _league_for(settings, args)
    return build_context(settings, league, draft, offline=getattr(args, "offline", False),
                         use_model=not getattr(args, "no_model", False), refresh=getattr(args, "refresh", False), **kw)


def cmd_prep(args: argparse.Namespace) -> int:
    from .app import prep

    console = Console()
    settings = _settings_from_args(args)
    if args.offline:
        console.print("[dim]--offline: skipping Sleeper[/dim]")
    ctx = _run(prep(settings, research=args.research, refresh=args.refresh, train=not args.no_train, top=args.top,
                    progress=lambda s: console.print(f"[cyan]…[/cyan] {s}")))
    snap = ctx.sources.get("snapshot")
    if snap is not None:
        try:
            from .capture import print_snapshot

            print_snapshot(snap, console)
        except Exception as e:  # noqa: BLE001
            log.warning("could not print snapshot: %s", e)
    mt = _metrics_table(ctx.sources.get("metrics") or {})
    if mt is not None:
        console.print(mt)
    console.print(_projection_table(ctx, 30, title=f"Top 30 — {ctx.league.name}"))
    _print_sources(console, ctx)
    if args.research:
        console.print(f"research notes written: {ctx.sources.get('research', 0)}")
    return EXIT_OK


def cmd_train(args: argparse.Namespace) -> int:
    console = Console()
    seasons = parse_seasons(args.seasons)
    try:
        from .projections.model import train_and_save

        _, metrics = train_and_save(seasons, refresh=args.refresh)
    except NotImplementedError as e:
        console.print(f"[red]training not available: {e}[/red]")
        return EXIT_ERROR
    mt = _metrics_table(metrics)
    if mt is not None:
        console.print(mt)
    console.print(f"model saved to {metrics.get('_path')} • fit {metrics.get('_fit_seconds') or 0:.0f}s • "
                  f"total {metrics.get('_total_seconds') or 0:.0f}s")
    return EXIT_OK


async def _resolve_draft_id(client, settings: Settings, console: Console) -> str | None:
    """Pick the draft id from --draft / --league / --username (interactive list)."""
    if settings.draft_id:
        return settings.draft_id
    from .capture import _pick_draft

    if settings.league_id:
        drafts = await client.get_league_drafts(settings.league_id)
        chosen = _pick_draft(drafts, None, settings.season)
        if chosen is None:
            league = await client.get_league(settings.league_id)
            return league.get("draft_id")
        return str(chosen["draft_id"])
    if settings.username or settings.user_id:
        user = await client.get_user(settings.username or settings.user_id)
        drafts = await client.get_user_drafts(user["user_id"], settings.season)
        if not drafts:
            console.print("no drafts found for this user/season")
            return None
        t = Table(title="Your drafts", box=box.SIMPLE_HEAD)
        for col in ("#", "draft_id", "league_id", "status", "type", "name"):
            t.add_column(col)
        for i, d in enumerate(drafts, start=1):
            t.add_row(str(i), str(d.get("draft_id")), str(d.get("league_id")), str(d.get("status")), str(d.get("type")),
                      str((d.get("metadata") or {}).get("name") or ""))
        console.print(t)
        if len(drafts) == 1:
            return str(drafts[0]["draft_id"])
        raw = input("Which draft? [1] ").strip() or "1"
        try:
            return str(drafts[int(raw) - 1]["draft_id"])
        except (ValueError, IndexError):
            console.print("[red]invalid choice[/red]")
            return None
    console.print("[red]draft: give --draft ID, --league ID or --username U[/red]")
    return None


async def _draft_async(args: argparse.Namespace, settings: Settings, console: Console) -> int:
    from .app import build_context
    from .sleeper.client import SleeperClient
    from .sleeper.poller import DraftPoller
    from .ui.dashboard import Dashboard

    async with SleeperClient() as client:
        draft_id = await _resolve_draft_id(client, settings, console)
        if not draft_id:
            return EXIT_USAGE
        settings.draft_id = draft_id
        poller = DraftPoller(client, draft_id, league_id=settings.league_id, settings=settings)
        console.print(f"[cyan]…[/cyan] connecting to draft {draft_id}")
        state = await poller.bootstrap()
        league = state.league
        if league is not None:
            settings.league_id = settings.league_id or league.league_id
        if state.my_slot is None:
            console.print("[yellow]could not resolve your slot (use --username/--user-id/--slot); observer mode[/yellow]")
        else:
            console.print(f"you are slot {state.my_slot} ({state.slot_label(state.my_slot)}), picks "
                          + ", ".join(f"#{p}" for p in state.my_pick_numbers()[:4]) + " …")
        try:
            from .capture import capture_league, print_snapshot

            snap = await capture_league(client, settings.league_id, draft_id, username=settings.username,
                                        user_id=settings.user_id, slot=settings.slot, season=settings.season)
            print_snapshot(snap, console)
        except Exception as e:  # noqa: BLE001
            log.warning("league capture skipped: %s", e)
        sleeper_players = sleeper_proj = None
        try:
            console.print("[cyan]…[/cyan] loading Sleeper players & projections")
            sleeper_players = await client.get_players()
            sleeper_proj = await client.get_season_projections(settings.season)
        except Exception as e:  # noqa: BLE001
            log.warning("Sleeper players/projections unavailable: %s", e)
        console.print("[cyan]…[/cyan] building projections")
        ctx = build_context(settings, league, state.draft, sleeper_players=sleeper_players, sleeper_proj=sleeper_proj,
                            refresh=args.refresh, use_model=not args.no_model, quiet=True)
        _print_sources(console, ctx)
        dashboard = None if args.no_tui else Dashboard(console, projections=ctx.projections)
        status = {"mode": "live", "projections": ctx.projections,
                  "hint": "Ctrl-C to quit • advice refreshes on every pick"}
        loop = DraftLoop(ctx.advisor, ctx.players, dashboard, researcher=ctx.researcher, notes=ctx.notes,
                         status=status, tui=not args.no_tui, out=console.print)
        final = await loop.run(poller)
        if final is not None and final.is_complete:
            console.print("[green]Draft complete.[/green]")
            _print_final_rosters(console, final, ctx)
    return EXIT_OK


def cmd_draft(args: argparse.Namespace) -> int:
    console = Console()
    settings = _settings_from_args(args)
    if args.offline:
        console.print("[red]draft needs the network (drop --offline); try `draftadvisor mock` offline.[/red]")
        return EXIT_USAGE
    try:
        return _run(_draft_async(args, settings, console))
    except KeyboardInterrupt:
        console.print("\nstopped.")
        return EXIT_OK
    except Exception as e:  # noqa: BLE001
        if _is_network_error(e):
            console.print(f"[red]{_network_message(e)}[/red]")
            return EXIT_NETWORK
        raise


def _print_final_rosters(console: Console, state: DraftState, ctx, my_slot: int | None = None) -> None:
    """Every team's projected starting-lineup points, ranked."""
    try:
        from .strategy.lineup import roster_summary
    except Exception as e:  # noqa: BLE001
        log.warning("lineup module unavailable: %s", e)
        return
    my_slot = my_slot if my_slot is not None else state.my_slot
    rows = []
    for slot in range(1, state.teams + 1):
        s = roster_summary(state, slot, ctx.players, ctx.projections, ctx.league)
        rows.append((s.lineup_points, s.bench_points, slot, s.label, s))
    rows.sort(key=lambda r: -r[0])
    t = Table(title="Projected starting lineups", box=box.SIMPLE_HEAD)
    for col in ("Rank", "Slot", "Team", "Lineup pts", "Bench pts", "Open starters"):
        t.add_column(col, justify="right" if "pts" in col or col in ("Rank", "Slot") else "left")
    my_rank = None
    for rank, (lp, bp, slot, label, s) in enumerate(rows, start=1):
        mine = slot == my_slot
        if mine:
            my_rank = rank
        open_ = " ".join(k for k, v in s.open_starters.items() if v > 0) or "-"
        t.add_row(str(rank), str(slot), label + (" (you)" if mine else ""), f"{lp:.0f}", f"{bp:.0f}", open_,
                  style="bold green" if mine else "")
    console.print(t)
    if my_rank is not None:
        n = len(rows)
        grade = "A" if my_rank <= max(1, n // 6) else "B" if my_rank <= n // 3 else "C" if my_rank <= 2 * n // 3 else "D"
        console.print(f"Your lineup ranks [bold]{my_rank}/{n}[/bold] (grade {grade}).")


def cmd_mock(args: argparse.Namespace) -> int:
    from .app import build_context
    from .mock.simulator import MockDraft, make_mock_draft, make_mock_league
    from .ui.dashboard import Dashboard, render_text

    console = Console()
    settings = _settings_from_args(args)
    if args.slot is None:
        args.slot = (args.teams + 1) // 2
    if not 1 <= args.slot <= args.teams:
        console.print(f"[red]--slot must be in 1..{args.teams}[/red]")
        return EXIT_USAGE
    league = make_mock_league(args.teams, args.rounds, args.scoring, args.superflex)
    league.league_id = f"mock_{args.scoring}{'_sf' if args.superflex else ''}"
    draft = make_mock_draft(league, args.slot, args.teams, args.rounds)
    console.print("[cyan]…[/cyan] building projections (cached after the first run)")
    ctx = build_context(settings, league, draft, offline=True, refresh=args.refresh, use_model=not args.no_model,
                        quiet=True)
    _print_sources(console, ctx)
    mock = MockDraft(ctx.players, league, draft, args.slot, seed=args.seed)
    advisor = ctx.advisor
    dashboard = None if args.no_tui else Dashboard(console, projections=ctx.projections)
    status = {"mode": "mock", "projections": ctx.projections, "claude": "off",
              "hint": "Enter = take the top recommendation • type a name or player_id • q to quit"}
    quit_ = False
    try:
        while not mock.is_complete and not quit_:
            if not mock.is_my_turn:
                pick = mock.bot_pick()
                pl = ctx.players.get(pick.player_id)
                if not args.auto or args.verbose:
                    console.print(f"[dim]#{pick.pick_no:>3} R{pick.round:<2} {mock.state().slot_label(pick.draft_slot):<8} "
                                  f"{pl.name if pl else pick.player_name} ({pick.position})[/dim]")
                if args.speed > 0 and not args.auto:
                    time.sleep(args.speed)
                continue
            state = mock.state()
            rec = advisor.recommend(state)
            top = rec.top_pick
            if args.auto:
                pl = top.player if top else mock.available()[0]
                pick = mock.make_pick(pl.player_id)
                why = "; ".join(top.reasons[:2]) if top else "best available"
                console.print(f"[bold]#{pick.pick_no:>3} R{pick.round:<2} You      {pl.name} ({pl.position}) — {why}[/bold]")
                if args.verbose:
                    console.print(render_text(rec, state, ctx.players))
                continue
            status["turn_started_at"] = time.time()
            if dashboard is not None:
                console.print(dashboard.build(state, rec, ctx.players, status))
            else:
                console.print(render_text(rec, state, ctx.players))
            while True:
                try:
                    raw = input(f"Pick #{state.next_pick_no} (Enter = {top.player.name if top else 'best available'}): ")
                except EOFError:
                    raw = ""
                    quit_ = True
                if raw.strip().lower() in ("q", "quit", "exit"):
                    quit_ = True
                    break
                if not raw.strip():
                    pl = top.player if top else mock.available()[0]
                else:
                    pl = resolve_player_input(raw, mock.available()[:600])
                    if pl is None:
                        console.print("[red]no available player matches; try again[/red]")
                        continue
                try:
                    pick = mock.make_pick(pl.player_id)
                except ValueError as e:
                    console.print(f"[red]{e}[/red]")
                    continue
                console.print(f"[green]You take {pl.name} ({pl.position}, {pl.team or 'FA'}) at #{pick.pick_no}[/green]")
                if top is not None and pl.player_id != top.player_id:
                    try:
                        console.print("[dim]" + advisor.explain_pick(state, pl.player_id) + "[/dim]")
                    except Exception as e:  # noqa: BLE001
                        log.debug("explain_pick failed: %s", e)
                break
    except KeyboardInterrupt:
        console.print("\nstopped.")
        return EXIT_OK
    final = mock.state()
    if final.is_complete:
        console.print("[green]Mock draft complete.[/green]")
    _print_final_rosters(console, final, ctx, my_slot=args.slot)
    my = final.my_picks()
    if my:
        console.print("Your roster: " + ", ".join(
            f"{(ctx.players.get(p.player_id).name if ctx.players.get(p.player_id) else p.player_name)} ({p.position})" for p in my))
    return EXIT_OK


def cmd_projections(args: argparse.Namespace) -> int:
    console = Console()
    settings = _settings_from_args(args)
    ctx = _build(settings, args, quiet=True)
    console.print(_projection_table(ctx, args.top, args.position, title=f"Projections — {ctx.league.name}"))
    _print_sources(console, ctx)
    return EXIT_OK


def _rosters_for_league(settings: Settings, args: argparse.Namespace, console: Console):
    """{roster_id: {"user": Manager-ish dict, "players": [ids]}} from Sleeper or the snapshot."""
    users: list[dict] = []
    rosters: list[dict] = []
    if not args.offline:
        try:
            from .sleeper.client import SleeperClient

            async def go():
                async with SleeperClient() as client:
                    return await client.get_league_users(settings.league_id), await client.get_league_rosters(settings.league_id)

            users, rosters = _run(go())
        except Exception as e:  # noqa: BLE001
            if not _is_network_error(e):
                raise
            log.warning("Sleeper unreachable (%s); trying snapshot", e)
    if not rosters:
        try:
            from .capture import LeagueSnapshot

            snap = LeagueSnapshot.load(str(settings.league_id))
            if snap is not None:
                users = list(snap.raw.get("users") or [])
                rosters = list(snap.raw.get("rosters") or [])
        except Exception as e:  # noqa: BLE001
            log.warning("snapshot unavailable: %s", e)
    if not rosters:
        console.print("[red]no rosters available (Sleeper unreachable and no snapshot; run `draftadvisor capture` online)[/red]")
        return {}
    by_user = {str(u.get("user_id")): u for u in users}
    out: dict[int, dict] = {}
    for r in rosters:
        u = by_user.get(str(r.get("owner_id")), {})
        out[int(r.get("roster_id"))] = {
            "user_id": str(r.get("owner_id")),
            "name": (u.get("metadata") or {}).get("team_name") or u.get("display_name") or f"Roster {r.get('roster_id')}",
            "username": u.get("username") or "",
            "display_name": u.get("display_name") or "",
            "players": [str(p) for p in (r.get("players") or [])],
        }
    return out


def _find_roster(rosters: Mapping[int, dict], who: str) -> int | None:
    w = who.strip().lower()
    for rid, r in rosters.items():
        if w in (r["username"].lower(), r["display_name"].lower(), r["name"].lower(), r["user_id"], str(rid)):
            return rid
    hits = difflib.get_close_matches(w, [r["display_name"].lower() for r in rosters.values()] +
                                     [r["name"].lower() for r in rosters.values()], n=1, cutoff=0.6)
    if hits:
        for rid, r in rosters.items():
            if hits[0] in (r["display_name"].lower(), r["name"].lower()):
                return rid
    return None


def cmd_trade(args: argparse.Namespace) -> int:
    console = Console()
    settings = _settings_from_args(args)
    if not settings.league_id:
        console.print("[red]trade needs --league ID[/red]")
        return EXIT_USAGE
    ctx = _build(settings, args, quiet=True)
    rosters = _rosters_for_league(settings, args, console)
    if not rosters:
        return EXIT_NETWORK
    me, them = _find_roster(rosters, args.me), _find_roster(rosters, args.them)
    if me is None or them is None:
        names = ", ".join(r["display_name"] or r["name"] for r in rosters.values())
        console.print(f"[red]could not find {'--me' if me is None else '--them'}; managers: {names}[/red]")
        return EXIT_USAGE

    def ids_for(text: str, roster_ids: list[str]) -> list[str] | None:
        pool = [ctx.players[pid] for pid in roster_ids if pid in ctx.players]
        out = []
        for name in _split_names(text):
            pl = resolve_player_input(name, pool)
            if pl is None:
                console.print(f"[red]{name!r} is not on that roster ({', '.join(p.name for p in pool)})[/red]")
                return None
            out.append(pl.player_id)
        return out

    give = ids_for(args.give, rosters[me]["players"])
    get = ids_for(args.get, rosters[them]["players"])
    if give is None or get is None:
        return EXIT_USAGE
    from .strategy.trade import evaluate_trade

    ev = evaluate_trade(rosters[me]["players"], rosters[them]["players"], give, get, ctx.players, ctx.projections,
                        ctx.league, settings.bench_discount)
    color = {"ACCEPT": "green", "REJECT": "red"}.get(ev.verdict, "yellow")
    console.print(f"[bold {color}]{ev.verdict}[/bold {color}]  me {ev.my_before:.0f} -> {ev.my_after:.0f} ({ev.my_delta:+.1f})"
                  f" • them {ev.their_before:.0f} -> {ev.their_after:.0f} ({ev.their_delta:+.1f})")
    for line in ev.details:
        console.print(f"  • {line}")
    return EXIT_OK


def cmd_analyze(args: argparse.Namespace) -> int:
    console = Console()
    settings = _settings_from_args(args)
    if not settings.league_id:
        console.print("[red]analyze needs --league ID[/red]")
        return EXIT_USAGE
    ctx = _build(settings, args, quiet=True)
    rosters = _rosters_for_league(settings, args, console)
    if not rosters:
        return EXIT_NETWORK
    from .strategy.lineup import optimal_lineup
    from .strategy.replacement import vorp

    league = ctx.league
    playoff_start = int((league.settings or {}).get("playoff_week_start") or 15)
    slots = league.starting_slots
    t = Table(title=f"Rosters — {league.name}", box=box.SIMPLE_HEAD)
    for col in ("Rank", "Team", "Lineup", "Bench", "Open", "Weakest slot", "Playoff byes", "Counts"):
        t.add_column(col, justify="right" if col in ("Rank", "Lineup", "Bench") else "left")
    rows = []
    rostered: set[str] = set()
    for rid, r in rosters.items():
        rostered.update(r["players"])
        cands = [(ctx.players[p], float(ctx.projections[p].points) if p in ctx.projections else 0.0)
                 for p in r["players"] if p in ctx.players]
        assignment, starters, bench = optimal_lineup(cands, slots)
        pts = {pl.player_id: p for pl, p in cands}
        open_ = [s for i, s in enumerate(slots) if i not in assignment]
        weakest = None
        for i, pid in assignment.items():
            if weakest is None or pts[pid] < weakest[1]:
                weakest = (f"{slots[i]} {ctx.players[pid].name} {pts[pid]:.0f}", pts[pid])
        po_byes = sorted({ctx.players[pid].bye_week for pid in assignment.values()
                          if ctx.players[pid].bye_week and ctx.players[pid].bye_week >= playoff_start})
        counts: dict[str, int] = {}
        for pl, _ in cands:
            counts[pl.position] = counts.get(pl.position, 0) + 1
        rows.append((starters, sum(pts[b] for b in bench), r["name"], " ".join(open_) or "-",
                     weakest[0] if weakest else "-", ",".join(map(str, po_byes)) or "-",
                     " ".join(f"{k}{v}" for k, v in sorted(counts.items()))))
    rows.sort(key=lambda x: -x[0])
    mine = (args.username or settings.username or "").lower()
    for i, row in enumerate(rows, start=1):
        style = "bold green" if mine and mine in row[2].lower() else ""
        t.add_row(str(i), row[2], f"{row[0]:.0f}", f"{row[1]:.0f}", *row[3:], style=style)
    console.print(t)
    avail = [pid for pid in ctx.projections if pid in ctx.players and pid not in rostered]
    try:
        v = vorp(ctx.projections, ctx.players, league, avail)
    except Exception as e:  # noqa: BLE001
        log.warning("vorp failed: %s", e)
        v = {pid: ctx.projections[pid].points for pid in avail}
    best = sorted(avail, key=lambda pid: -v.get(pid, 0.0))[:12]
    w = Table(title="Best waiver targets (by VORP)", box=box.SIMPLE_HEAD)
    for col in ("Player", "Pos", "Tm", "Pts", "VORP"):
        w.add_column(col, justify="right" if col in ("Pts", "VORP") else "left")
    for pid in best:
        pl = ctx.players[pid]
        w.add_row(pl.name, pl.position, pl.team or "FA", f"{ctx.projections[pid].points:.0f}", f"{v.get(pid, 0):+.0f}")
    console.print(w)
    return EXIT_OK


def cmd_research(args: argparse.Namespace) -> int:
    from .app import top_players_by_adp

    console = Console()
    settings = _settings_from_args(args)
    ctx = _build(settings, args, quiet=True)
    if not ctx.researcher.enabled:
        console.print("[yellow]Claude is off — set ANTHROPIC_API_KEY to run research.[/yellow]")
        return EXIT_USAGE
    targets = top_players_by_adp(ctx.players, args.top)
    console.print(f"[cyan]…[/cyan] researching {len(targets)} players (cached notes are reused)")
    notes = _run(ctx.researcher.research_players(targets, ctx.projections))
    t = Table(title="Research notes", box=box.SIMPLE_HEAD)
    for col in ("Player", "Pos", "Inj", "Role", "Summary"):
        t.add_column(col, justify="right" if col in ("Inj", "Role") else "left")
    for pl in targets:
        n = notes.get(pl.player_id) or ctx.notes.get(pl.player_id)
        if n is None:
            continue
        t.add_row(pl.name, pl.position, f"{n.injury_risk:.2f}", f"{n.role_certainty:.2f}", n.summary[:90])
    console.print(t)
    console.print(f"{len(notes)} notes")
    return EXIT_OK


def cmd_ask(args: argparse.Namespace) -> int:
    console = Console()
    settings = _settings_from_args(args)
    ctx = _build(settings, args, quiet=True)
    if not ctx.researcher.enabled:
        console.print("[yellow]Claude is off — set ANTHROPIC_API_KEY to ask questions.[/yellow]")
        return EXIT_USAGE
    context_text = ""
    if settings.draft_id and not args.offline:
        try:
            from .research.claude import build_context_text
            from .sleeper.client import SleeperClient
            from .sleeper.poller import DraftPoller

            async def go():
                async with SleeperClient() as client:
                    poller = DraftPoller(client, settings.draft_id, league_id=settings.league_id, settings=settings)
                    state = await poller.bootstrap()
                    rec = ctx.advisor.recommend(state)
                    return build_context_text(state, rec, ctx.players, ctx.notes)

            context_text = _run(go())
        except Exception as e:  # noqa: BLE001
            if not _is_network_error(e):
                raise
            console.print(f"[yellow]draft context unavailable ({e}); answering from projections only[/yellow]")
    if not context_text:
        top = sorted(ctx.projections.values(), key=lambda p: -p.points)[:40]
        context_text = f"League: {ctx.league.name} ({ctx.league.scoring_type}, slots {' '.join(ctx.league.starting_slots)})\n" + \
            "Top projected players:\n" + "\n".join(
                f"- {ctx.players[p.player_id].name} ({p.position}, {ctx.players[p.player_id].team or 'FA'}) {p.points:.0f} pts, "
                f"ADP {ctx.players[p.player_id].adp or '-'}" for p in top if p.player_id in ctx.players)
    answer = _run(ctx.researcher.ask(args.question, context_text))
    if not answer:
        console.print(f"[red]no answer ({getattr(ctx.researcher, 'last_error', None) or 'unknown error'})[/red]")
        return EXIT_ERROR
    console.print(answer)
    return EXIT_OK


def cmd_ids(args: argparse.Namespace) -> int:
    console = Console()
    settings = _settings_from_args(args)
    if args.offline:
        console.print("[red]ids needs the network[/red]")
        return EXIT_USAGE
    from .sleeper.client import SleeperClient

    async def go():
        async with SleeperClient() as client:
            user = await client.get_user(args.username)
            leagues = await client.get_user_leagues(user["user_id"], settings.season)
            rows = []
            for lg in leagues:
                try:
                    drafts = await client.get_league_drafts(lg["league_id"])
                except Exception as e:  # noqa: BLE001
                    log.info("drafts for %s unavailable: %s", lg.get("league_id"), e)
                    drafts = []
                rows.append((lg, drafts))
            return user, rows

    try:
        user, rows = _run(go())
    except Exception as e:  # noqa: BLE001
        if _is_network_error(e):
            console.print(f"[red]{_network_message(e)}[/red]")
            return EXIT_NETWORK
        raise
    console.print(f"user {user.get('display_name')} (id {user.get('user_id')}) • season {settings.season}")
    t = Table(box=box.SIMPLE_HEAD)
    for col in ("League", "league_id", "Teams", "Status", "draft_id", "Draft status", "Type"):
        t.add_column(col)
    for lg, drafts in rows:
        if not drafts:
            t.add_row(lg.get("name", ""), str(lg.get("league_id")), str(lg.get("total_rosters")), str(lg.get("status")),
                      str(lg.get("draft_id") or "-"), "-", "-")
        for d in drafts:
            t.add_row(lg.get("name", ""), str(lg.get("league_id")), str(lg.get("total_rosters")), str(lg.get("status")),
                      str(d.get("draft_id")), str(d.get("status")), str(d.get("type")))
    console.print(t)
    return EXIT_OK


def cmd_web(args: argparse.Namespace) -> int:
    from .web.server import main as web_main

    argv = ["--host", args.host, "--port", str(args.port)]
    if args.no_browser:
        argv.append("--no-browser")
    return web_main(argv)


def cmd_capture(args: argparse.Namespace) -> int:
    console = Console()
    settings = _settings_from_args(args)
    if not (settings.league_id or settings.draft_id):
        console.print("[red]capture needs --league ID or --draft ID[/red]")
        return EXIT_USAGE
    from .capture import LeagueSnapshot, capture_league, print_snapshot

    if args.offline:
        snap = LeagueSnapshot.load(str(settings.league_id or settings.draft_id))
        if snap is None:
            console.print("[red]no saved snapshot; run online first[/red]")
            return EXIT_USAGE
        print_snapshot(snap, console)
        return EXIT_OK
    from .sleeper.client import SleeperClient

    async def go():
        async with SleeperClient() as client:
            return await capture_league(client, settings.league_id, settings.draft_id, username=settings.username,
                                        user_id=settings.user_id, slot=settings.slot, season=settings.season)

    try:
        snap = _run(go())
    except Exception as e:  # noqa: BLE001
        if _is_network_error(e):
            console.print(f"[red]{_network_message(e)}[/red]")
            return EXIT_NETWORK
        raise
    print_snapshot(snap, console)
    return EXIT_OK


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.home:
        os.environ["DRAFTADVISOR_HOME"] = str(args.home)
    _configure_logging(args.verbose)
    if args.season is None:
        args.season = DEFAULT_SEASON
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\nstopped.", file=sys.stderr)
        return EXIT_OK
    except Exception as e:  # noqa: BLE001
        if _is_network_error(e):
            print(_network_message(e), file=sys.stderr)
            return EXIT_NETWORK
        if args.verbose:
            raise
        print(f"error: {type(e).__name__}: {e}  (run with -v for the traceback)", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
