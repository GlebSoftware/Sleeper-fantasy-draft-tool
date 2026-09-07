"""CLI tests: argparse, the draft loop with fakes, name resolution, offline mock draft."""
from __future__ import annotations

import asyncio
import io
import time
from pathlib import Path

import pytest
from rich.console import Console

from draftadvisor import cli
from draftadvisor.cli import DraftLoop, build_parser, main, parse_seasons, resolve_player_input, run_draft_loop
from draftadvisor.models import Recommendation
from tests.test_dashboard import PLAYERS, PROJ, make_rec, make_state

REPO_DATA = Path(__file__).resolve().parents[1] / "data"


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("argv,command,checks", [
    (["prep", "--league", "1", "--refresh", "--no-train", "--research", "--top", "50"], "prep",
     {"league_id": "1", "refresh": True, "no_train": True, "research": True, "top": 50}),
    (["train", "--seasons", "2019-2025", "--refresh"], "train", {"seasons": "2019-2025", "refresh": True}),
    (["draft", "--draft", "D", "--league", "L", "--username", "bob", "--poll", "3", "--no-claude", "--no-tui"], "draft",
     {"draft_id": "D", "league_id": "L", "username": "bob", "poll": 3.0, "no_claude": True, "no_tui": True}),
    (["draft", "--draft", "D", "--slot", "7"], "draft", {"slot": 7, "username": None}),
    (["mock", "--teams", "10", "--rounds", "14", "--slot", "3", "--scoring", "ppr", "--superflex", "--auto", "--seed", "9",
      "--speed", "0"], "mock",
     {"teams": 10, "rounds": 14, "slot": 3, "scoring": "ppr", "superflex": True, "auto": True, "seed": 9, "speed": 0.0}),
    (["projections", "--position", "RB", "--top", "20", "--league", "L"], "projections",
     {"position": "RB", "top": 20, "league_id": "L"}),
    (["trade", "--league", "L", "--me", "a", "--them", "b", "--give", "X, Y", "--get", "Z"], "trade",
     {"me": "a", "them": "b", "give": "X, Y", "get": "Z"}),
    (["analyze", "--league", "L", "--username", "u"], "analyze", {"league_id": "L", "username": "u"}),
    (["research", "--top", "100", "--league", "L"], "research", {"top": 100}),
    (["ask", "Who should I take?", "--draft", "D"], "ask", {"question": "Who should I take?", "draft_id": "D"}),
    (["ids", "--username", "bob"], "ids", {"username": "bob"}),
    (["capture", "--league", "L", "--draft", "D"], "capture", {"league_id": "L", "draft_id": "D"}),
    (["--home", "/tmp/x", "--offline", "-vv", "--season", "2025", "mock"], "mock",
     {"home": "/tmp/x", "offline": True, "verbose": 2, "season": 2025}),
])
def test_parser(argv, command, checks):
    args = build_parser().parse_args(argv)
    assert args.command == command
    for k, v in checks.items():
        assert getattr(args, k) == v, k
    assert callable(args.func)


def test_parser_rejects_bad_input():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["mock", "--scoring", "weird"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["draft", "--username", "a", "--slot", "2"])   # mutually exclusive
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_parse_seasons():
    assert parse_seasons("2019-2021") == [2019, 2020, 2021]
    assert parse_seasons("2020,2022") == [2020, 2022]
    assert parse_seasons(None)[-1] == 2025


def test_main_sets_home(monkeypatch, tmp_path):
    called = {}

    def fake_cmd(args):
        import os
        called["home"] = os.environ.get("DRAFTADVISOR_HOME")
        return 0

    monkeypatch.setattr(cli, "cmd_train", fake_cmd)
    parser = build_parser()
    monkeypatch.setattr(cli, "build_parser", lambda: _patch_func(parser, "train", fake_cmd))
    assert main(["--home", str(tmp_path), "train"]) == 0
    assert called["home"] == str(tmp_path)


def _patch_func(parser, command, func):
    for action in parser._subparsers._group_actions:
        sub = action.choices.get(command)
        if sub is not None:
            sub.set_defaults(func=func)
    return parser


# ---------------------------------------------------------------------------
# Draft loop with fakes
# ---------------------------------------------------------------------------


class FakeAdvisor:
    def __init__(self):
        self.calls = 0

    def recommend(self, state):
        self.calls += 1
        return make_rec(state)


class FakeResearcher:
    def __init__(self, enabled=True, delay=0.01, text="Take the RB; WR later."):
        self.enabled = enabled
        self.delay = delay
        self.text = text
        self.calls: list[int] = []

    async def on_the_clock_advice(self, state, rec, players, notes, timeout=8.0):
        self.calls.append(state.version)
        await asyncio.sleep(self.delay)
        return self.text


class FakeDashboard:
    def __init__(self):
        self.updates = []
        self.started = self.stopped = False
        self.refreshes = 0

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def update(self, state, rec, players, status):
        self.updates.append((state.version, rec.claude_advice if rec else None, dict(status)))

    def refresh(self, status=None):
        self.refreshes += 1


async def _states(versions):
    for n in versions:
        yield make_state(n)
        await asyncio.sleep(0.005)


async def test_loop_three_updates_three_recommends_claude_only_near_my_turn():
    # my_slot=2 in a 4-team snake: picks #2, #7, #10. States: 3 picks (until=4), 5 picks (until=1), 6 picks (my turn)
    advisor, researcher, dashboard = FakeAdvisor(), FakeResearcher(), FakeDashboard()
    loop = DraftLoop(advisor, PLAYERS, dashboard, researcher=researcher, tui=True, refresh_seconds=0.01)
    await loop.run(_states([3, 5, 6]))
    assert advisor.calls == 3
    assert researcher.calls == [5, 6]                 # picks_until_my_turn <= 1 and my turn only
    assert dashboard.started and dashboard.stopped
    assert loop.rec is not None and loop.rec.claude_advice == "Take the RB; WR later."
    # the advice arrived through a background task and triggered a re-render
    assert any(u[1] == "Take the RB; WR later." for u in dashboard.updates)
    assert dashboard.refreshes >= 1
    assert loop.status["claude"] == "ready"
    assert "turn_started_at" in loop.status         # noticed my turn -> countdown starts


async def test_loop_claude_disabled_never_called():
    advisor, researcher, dashboard = FakeAdvisor(), FakeResearcher(enabled=False), FakeDashboard()
    loop = await run_draft_loop(_states([5, 6]), advisor, dashboard, players=PLAYERS, researcher=researcher,
                                refresh_seconds=0)
    assert advisor.calls == 2 and researcher.calls == [] and loop.status["claude"] == "off"


async def test_loop_no_tui_prints_render_text():
    out = []
    advisor = FakeAdvisor()
    loop = DraftLoop(advisor, PLAYERS, None, researcher=None, tui=False, out=out.append)
    await loop.run(_states([3, 6]))
    assert advisor.calls == 2 and len(out) == 2
    assert "Best picks now" in out[0] and "YOUR PICK" in out[1]


async def test_loop_render_never_blocks_on_slow_claude():
    """A slow Claude call must not delay processing of the next state."""
    advisor, researcher, dashboard = FakeAdvisor(), FakeResearcher(delay=0.5), FakeDashboard()
    loop = DraftLoop(advisor, PLAYERS, dashboard, researcher=researcher, refresh_seconds=0, claude_grace_s=0)
    t0 = asyncio.get_event_loop().time()
    await loop.run(_states([5, 6, 7]))
    elapsed = asyncio.get_event_loop().time() - t0
    assert advisor.calls == 3
    assert elapsed < 0.4, f"loop blocked on Claude ({elapsed:.2f}s)"


async def test_loop_with_poller_like_source_and_stop():
    class FakePoller:
        last_latency_ms = 42.0
        last_error = None
        poll_count = 0

        async def run(self, on_update, stop=None):
            for n in (3, 5, 6):
                self.poll_count += 1
                await on_update(make_state(n))
                if stop is not None and stop.is_set():
                    break
            return make_state(6)

    advisor, dashboard = FakeAdvisor(), FakeDashboard()
    loop = DraftLoop(advisor, PLAYERS, dashboard, refresh_seconds=0)
    stop = asyncio.Event()
    final = await loop.run(FakePoller(), stop)
    assert advisor.calls == 3 and final.version == 6
    assert loop.status["latency_ms"] == 42.0 and loop.status["poll_count"] == 3


async def test_loop_survives_recommend_error():
    class Broken:
        def recommend(self, state):
            raise RuntimeError("boom")

    out = []
    loop = DraftLoop(Broken(), PLAYERS, None, tui=False, out=out.append)
    await loop.run(_states([3]))
    assert "recommend: boom" in loop.status["last_error"]
    assert out and "no recommendation yet" in out[0]


# ---------------------------------------------------------------------------
# name resolution
# ---------------------------------------------------------------------------


def test_resolve_player_input():
    pool = list(PLAYERS.values())
    assert resolve_player_input("rb1", pool).player_id == "rb1"
    assert resolve_player_input("bijan robinson", pool).player_id == "rb1"
    assert resolve_player_input("Jefferson", pool).player_id == "wr2"
    assert resolve_player_input("Jamarr Chase", pool).player_id == "wr1"      # fuzzy
    assert resolve_player_input("Bijon Robison", pool).player_id == "rb1"     # typo
    assert resolve_player_input("", pool) is None
    assert resolve_player_input("Nobody Atall Whatsoever", pool) is None


# ---------------------------------------------------------------------------
# mock command (fake context: no data/raw needed)
# ---------------------------------------------------------------------------


def _fake_context(settings, league, draft, **kw):
    from draftadvisor.app import AppContext, _DisabledResearcher
    from draftadvisor.scoring.engine import ScoringEngine
    from draftadvisor.strategy.recommend import Advisor
    from tests.test_strategy import make_universe

    players, projections = make_universe(seed=1)
    engine = ScoringEngine(league.scoring_settings)
    return AppContext(settings, engine, league, draft, players, projections, Advisor(league, players, projections, settings),
                      _DisabledResearcher(), {}, {}, None, {"players": "fake", "projections": "fake"})


def test_mock_auto_with_fake_context(monkeypatch, capsys):
    import draftadvisor.app as app

    monkeypatch.setattr(app, "build_context", _fake_context)
    rc = main(["mock", "--auto", "--teams", "4", "--rounds", "3", "--speed", "0", "--no-tui", "--seed", "2"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Mock draft complete" in out
    assert "You take" in out or "You " in out
    assert "Projected starting lineups" in out and "Your lineup ranks" in out


def test_mock_interactive_with_fake_context(monkeypatch, capsys):
    import draftadvisor.app as app

    monkeypatch.setattr(app, "build_context", _fake_context)
    answers = iter(["", "WR 0", "nonexistent player zz", "q"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    rc = main(["mock", "--teams", "4", "--rounds", "3", "--slot", "1", "--speed", "0", "--no-tui", "--seed", "2"])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.count("You take") == 2
    assert "no available player matches" in out


def test_draft_offline_flag_refuses(capsys):
    assert main(["--offline", "draft", "--draft", "D"]) == cli.EXIT_USAGE


def test_draft_network_error_is_friendly(monkeypatch, capsys):
    import httpx

    async def boom(*a, **k):
        raise httpx.ConnectError("no route to host")

    monkeypatch.setattr(cli, "_draft_async", boom)
    rc = main(["draft", "--draft", "D"])
    assert rc == cli.EXIT_NETWORK
    assert "Could not reach the Sleeper API" in capsys.readouterr().out


@pytest.mark.skipif(not (REPO_DATA / "raw").exists(), reason="needs data/raw (slow, real offline universe)")
def test_mock_auto_real_offline_universe(capsys):
    rc = main(["--home", str(REPO_DATA), "mock", "--auto", "--teams", "4", "--rounds", "3", "--speed", "0",
               "--no-tui", "--no-model"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Mock draft complete" in out and "Your lineup ranks" in out


# ---------------------------------------------------------------------------
# Review findings: poll errors reach the UI (F1), countdown seed (F3), not-found (F6), --poll (F7),
# plain text output (F9), global flags (F10), research progress (R7), prep --offline (F5)
# ---------------------------------------------------------------------------


class ErrorPoller:
    """Poller-like source: one state, then a run of failures reported through on_error, then a recovery."""

    def __init__(self):
        self.last_latency_ms = 120.0
        self.last_error = None
        self.poll_count = 1
        self.last_poll_at = None

    async def run(self, on_update, stop=None, on_error=None):
        self.last_poll_at = time.time()
        await on_update(make_state(3))
        for _ in range(3):
            self.poll_count += 1
            self.last_error = "SleeperAPIError: HTTP 503"
            if on_error is not None:
                r = on_error(RuntimeError("HTTP 503"))
                if asyncio.iscoroutine(r):
                    await r
            await asyncio.sleep(0.02)
        self.last_error = None
        self.last_latency_ms = 250.0
        self.poll_count += 1
        self.last_poll_at = time.time()
        await asyncio.sleep(0.03)
        await on_update(make_state(5))
        return make_state(5)


class RecordingDashboard(FakeDashboard):
    """Also keeps the status of every refresh (the error path refreshes: there is no new state to update with)."""

    def __init__(self):
        super().__init__()
        self.refresh_status = []

    def refresh(self, status=None):
        super().refresh(status)
        self.refresh_status.append(dict(status or {}))


async def test_loop_reports_poll_errors_and_recovery_to_dashboard():
    advisor, dashboard = FakeAdvisor(), RecordingDashboard()
    status = {}
    loop = DraftLoop(advisor, PLAYERS, dashboard, status=status, refresh_seconds=0.01)
    await loop.run(ErrorPoller())
    assert loop.status is status                       # shared by reference (the web server reads it)
    # the errors reached the dashboard while no new state arrived ...
    seen = dashboard.refresh_status
    assert any(u.get("last_error") and "503" in str(u["last_error"]) for u in seen), seen
    assert any(u.get("stale_since") for u in seen)
    # ... and were cleared again after the recovery (None, not merely absent, so a refresh drops them)
    assert status["last_error"] is None and status["stale_since"] is None
    assert status["latency_ms"] == 250.0 and status["poll_count"] == 5
    assert not dashboard.updates[-1][2].get("last_error")


async def test_loop_tick_pulls_source_stats_between_updates():
    """F1: latency / errors appear on the 1 s tick, not only when a new state arrives."""
    class Quiet:
        last_latency_ms = 10.0
        last_error = None
        poll_count = 1
        last_poll_at = None

        async def run(self, on_update, stop=None, on_error=None):
            await on_update(make_state(3))
            self.last_latency_ms, self.last_error, self.poll_count = 9000.0, "ConnectError: no route", 7
            await asyncio.sleep(0.06)
            return make_state(3)

    dashboard = FakeDashboard()
    loop = DraftLoop(FakeAdvisor(), PLAYERS, dashboard, refresh_seconds=0.01)
    await loop.run(Quiet())
    assert loop.status["latency_ms"] == 9000.0 and loop.status["poll_count"] == 7
    assert "ConnectError" in loop.status["last_error"] and loop.status["stale_since"]


async def test_loop_no_tui_prints_poll_errors():
    out = []
    loop = DraftLoop(FakeAdvisor(), PLAYERS, None, tui=False, out=out.append)
    await loop.run(ErrorPoller())
    assert any("poll error" in s and "503" in s for s in out)


async def test_track_turn_seeds_countdown_from_last_picked():
    loop = DraftLoop(FakeAdvisor(), PLAYERS, None, tui=False, out=lambda s: None)
    state = make_state(6)
    assert state.is_my_turn
    state.draft.last_picked = int((time.time() - 20) * 1000)
    await loop.handle(state)
    assert loop.status["turn_started_at"] == pytest.approx(time.time() - 20, abs=0.5)


def test_not_found_is_not_a_network_error(monkeypatch, capsys):
    from draftadvisor.sleeper.client import SleeperAPIError, SleeperNotFound

    nf = SleeperNotFound("not found", status_code=404, url="https://api.sleeper.app/v1/draft/12345")
    assert cli._is_not_found_error(nf) and not cli._is_network_error(nf)
    assert cli._is_not_found_error(SleeperAPIError("bad request", status_code=400))
    assert cli._is_network_error(SleeperAPIError("HTTP 503", status_code=503))
    assert cli._is_network_error(SleeperAPIError("HTTP 429", status_code=429))

    async def boom(*a, **k):
        raise nf

    monkeypatch.setattr(cli, "_draft_async", boom)
    rc = main(["draft", "--draft", "12345"])
    out = capsys.readouterr().out
    assert rc == cli.EXIT_USAGE
    assert "Sleeper has no draft 12345" in out and "draftadvisor ids" in out
    assert "Could not reach" not in out


def test_poll_is_clamped_to_minimum():
    for raw in ("0", "-1", "0.1"):
        args = build_parser().parse_args(["draft", "--draft", "D", "--poll", raw])
        assert cli._settings_from_args(args).poll_seconds == cli.MIN_POLL_SECONDS
    args = build_parser().parse_args(["draft", "--draft", "D", "--poll", "3"])
    assert cli._settings_from_args(args).poll_seconds == 3.0
    args = build_parser().parse_args(["draft", "--draft", "D"])
    assert cli._settings_from_args(args).poll_seconds == 2.0


def test_plain_printer_ignores_rich_markup():
    console = Console(record=True, file=io.StringIO(), width=120, force_terminal=False, color_system=None)
    out = cli._plain_printer(console)
    out("note [source] here and a stray [/x] tag")     # would drop text / raise MarkupError with markup on
    assert "note [source] here and a stray [/x] tag" in console.export_text()


@pytest.mark.parametrize("argv", [
    ["mock", "--offline"],
    ["--offline", "mock"],
    ["draft", "--draft", "D", "--offline"],
    ["prep", "--offline", "--no-train"],
])
def test_global_flags_accepted_after_subcommand(argv):
    args = build_parser().parse_args(argv)
    assert args.offline is True


def test_global_flags_after_subcommand_values():
    args = build_parser().parse_args(["draft", "--draft", "D", "-vv", "--home", "/tmp/y", "--season", "2025"])
    assert args.verbose == 2 and args.home == "/tmp/y" and args.season == 2025 and args.offline is False
    args = build_parser().parse_args(["-v", "--home", "/tmp/z", "mock"])
    assert args.verbose == 1 and args.home == "/tmp/z" and args.season is None and args.offline is False
    assert main(["draft", "--draft", "D", "--offline"]) == cli.EXIT_USAGE      # refused, not an argparse error


class _RecordingResearcher:
    enabled = True
    last_error = None

    def __init__(self, fail_one: bool = True):
        self.kwargs = None
        self.fail_one = fail_one

    def load_notes(self):
        return {}

    async def research_players(self, players, projections=None, **kw):
        from draftadvisor.models import ResearchNote
        from draftadvisor.research.claude import UNAVAILABLE_SUMMARY

        self.kwargs = kw
        out = {}
        for i, pl in enumerate(players):
            if kw.get("progress"):
                kw["progress"](i + 1, len(players), pl.name)
            if self.fail_one and i == 0:
                self.last_error = "AuthenticationError: invalid x-api-key"
                out[pl.player_id] = ResearchNote(pl.player_id, UNAVAILABLE_SUMMARY, 0.0, 0.5, "", "")
            else:
                out[pl.player_id] = ResearchNote(pl.player_id, f"{pl.name} looks fine", 0.1, 0.8, "up", "down")
        return out


def test_cmd_research_shows_progress_and_counts(monkeypatch, capsys):
    from draftadvisor.config import Settings

    ctx = _fake_context(Settings(), __import__("tests.test_dashboard", fromlist=["x"])._league(), None)
    ctx.researcher = _RecordingResearcher()
    monkeypatch.setattr(cli, "_build", lambda settings, args, **kw: ctx)
    rc = main(["--offline", "research", "--top", "3"])
    out = capsys.readouterr().out
    assert rc == 0
    assert ctx.researcher.kwargs is not None and callable(ctx.researcher.kwargs.get("progress"))
    assert "[1/3]" in out and "[3/3]" in out
    assert "2 researched, 1 failed, 0 cached" in out
    assert "invalid x-api-key" in out


def test_cmd_prep_passes_offline_and_reports_research_error(monkeypatch, capsys):
    import draftadvisor.app as app
    from draftadvisor.config import Settings

    league = __import__("tests.test_dashboard", fromlist=["x"])._league()
    ctx = _fake_context(Settings(), league, None)
    ctx.researcher = _RecordingResearcher()
    ctx.researcher.last_error = "AuthenticationError: invalid x-api-key"
    ctx.sources.update({"research": 3, "snapshot": None, "metrics": {}})
    seen = {}

    async def fake_prep(settings, **kw):
        seen.update(kw)
        return ctx

    monkeypatch.setattr(app, "prep", fake_prep)
    rc = main(["prep", "--offline", "--no-train", "--research"])
    out = capsys.readouterr().out
    assert rc == 0
    assert seen["offline"] is True and seen["train"] is False and seen["research"] is True
    assert "research notes written: 3" in out and "invalid x-api-key" in out
