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
from tests.test_dashboard import PLAYERS, PROJ, make_rec, make_state

REPO_DATA = Path(__file__).resolve().parents[1] / "data"


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("argv,command,checks", [
    (["prep", "--league", "1", "--refresh", "--no-train"], "prep", {"league_id": "1", "refresh": True, "no_train": True}),
    (["train", "--seasons", "2019-2025", "--refresh"], "train", {"seasons": "2019-2025", "refresh": True}),
    (["draft", "--draft", "D", "--league", "L", "--username", "bob", "--poll", "3", "--no-tui"], "draft",
     {"draft_id": "D", "league_id": "L", "username": "bob", "poll": 3.0, "no_tui": True}),
    (["draft", "--draft", "D", "--slot", "7"], "draft", {"slot": 7, "username": None}),
    (["mock", "--teams", "10", "--rounds", "14", "--slot", "3", "--scoring", "ppr", "--superflex", "--auto", "--seed", "9",
      "--speed", "0"], "mock",
     {"teams": 10, "rounds": 14, "slot": 3, "scoring": "ppr", "superflex": True, "auto": True, "seed": 9, "speed": 0.0}),
    (["projections", "--position", "RB", "--top", "20", "--league", "L"], "projections",
     {"position": "RB", "top": 20, "league_id": "L"}),
    (["trade", "--league", "L", "--me", "a", "--them", "b", "--give", "X, Y", "--get", "Z"], "trade",
     {"me": "a", "them": "b", "give": "X, Y", "get": "Z"}),
    (["analyze", "--league", "L", "--username", "u"], "analyze", {"league_id": "L", "username": "u"}),
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
    for argv in (["research", "--top", "5"], ["prep", "--research"], ["prep", "--top", "5"],
                 ["draft", "--draft", "D", "--no-claude"], ["mock", "--no-claude"]):
        with pytest.raises(SystemExit):                                          # the research / Claude flags are gone
            build_parser().parse_args(argv)
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
        self.updates.append((state.version, rec, dict(status)))

    def refresh(self, status=None):
        self.refreshes += 1


async def _states(versions):
    for n in versions:
        yield make_state(n)
        await asyncio.sleep(0.005)


async def test_loop_three_updates_three_recommends_and_never_calls_claude():
    # my_slot=2 in a 4-team snake: picks #2, #7, #10. States: 3 picks (until=4), 5 picks (until=1), 6 picks (my turn)
    advisor, dashboard = FakeAdvisor(), FakeDashboard()
    loop = DraftLoop(advisor, PLAYERS, dashboard, tui=True, refresh_seconds=0.01)
    await loop.run(_states([3, 5, 6]))
    assert advisor.calls == 3
    assert dashboard.started and dashboard.stopped
    assert [u[0] for u in dashboard.updates] == [3, 5, 6] and all(u[1] is not None for u in dashboard.updates)
    assert dashboard.refreshes >= 1
    assert "turn_started_at" in loop.status         # noticed my turn -> countdown starts
    # the loop is recommend + render only: no Claude hooks, no Claude status
    assert "claude" not in loop.status and loop.rec.claude_advice is None
    for name in ("schedule_claude", "wants_claude", "researcher", "notes", "claude_requests", "_claude_task"):
        assert not hasattr(loop, name), name
    with pytest.raises(TypeError):
        DraftLoop(advisor, PLAYERS, dashboard, researcher=object())
    with pytest.raises(TypeError):
        await run_draft_loop(_states([5]), advisor, dashboard, players=PLAYERS, researcher=object())


async def test_loop_no_tui_prints_render_text():
    out = []
    advisor = FakeAdvisor()
    loop = DraftLoop(advisor, PLAYERS, None, tui=False, out=out.append)
    await loop.run(_states([3, 6]))
    assert advisor.calls == 2 and len(out) == 2
    assert "Best picks now" in out[0] and "YOUR PICK" in out[1]
    assert not any("Claude" in s for s in out)


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
    from draftadvisor.app import AppContext, _DisabledClaude
    from draftadvisor.scoring.engine import ScoringEngine
    from draftadvisor.strategy.recommend import Advisor
    from tests.test_strategy import make_universe

    players, projections = make_universe(seed=1)
    engine = ScoringEngine(league.scoring_settings)
    return AppContext(settings, engine, league, draft, players, projections, Advisor(league, players, projections, settings),
                      _DisabledClaude(), {}, {}, None, {"players": "fake", "projections": "fake"})


def test_mock_auto_with_fake_context(monkeypatch, capsys):
    import draftadvisor.app as app

    monkeypatch.setattr(app, "build_context", _fake_context)
    rc = main(["mock", "--auto", "--teams", "4", "--rounds", "3", "--speed", "0", "--no-tui", "--seed", "2", "-v"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Mock draft complete" in out
    assert "You take" in out or "You " in out
    assert "Projected starting lineups" in out and "Your lineup ranks" in out
    assert "== [Mock] Mock League" in out and "[Sleeper]" not in out       # an offline mock is not a Sleeper draft


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
    assert "== [Mock] Mock League" in out and "[Sleeper]" not in out


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
# plain text output (F9), global flags (F10), prep --offline (F5), ask is the only Claude call
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


class _FakeAsk:
    """Chat client double for ``ask``: answers once and reports usage / cost like ClaudeChat."""

    enabled = True
    last_error = None
    model = "claude-sonnet-5"

    def __init__(self, answer="Take the RB [tier 1]; the WR will be there at #7."):
        self.answer = answer
        self.calls: list[tuple[str, str]] = []
        self.last_usage = None
        self.last_cost_usd = None

    def load_notes(self):
        return {}

    async def ask(self, question, context_text):
        self.calls.append((question, context_text))
        self.last_usage = {"input_tokens": 12100, "output_tokens": 400}
        self.last_cost_usd = 0.0705
        return self.answer


def test_cmd_ask_prints_answer_tokens_and_cost(monkeypatch, capsys):
    from draftadvisor.config import Settings

    ctx = _fake_context(Settings(), __import__("tests.test_dashboard", fromlist=["x"])._league(), None)
    ctx.claude = _FakeAsk()
    monkeypatch.setattr(cli, "_build", lambda settings, args, **kw: ctx)
    rc = main(["--offline", "ask", "Who should I take?"])
    out = capsys.readouterr().out
    assert rc == 0
    assert len(ctx.claude.calls) == 1 and ctx.claude.calls[0][0] == "Who should I take?"
    assert "Top projected players" in ctx.claude.calls[0][1]           # offline: projections-only context
    assert "Take the RB [tier 1]" in out                                # printed verbatim (no rich markup parsing)
    assert "12.1k in / 0.4k out - about $0.07 (claude-sonnet-5)" in out
    # a model outside the price table: the line says the cost is unknown instead of dropping it silently,
    # and names the served model when the client records one
    monkeypatch.setenv("COLUMNS", "200")                                  # keep the usage line on one line
    ctx.claude = _FakeAsk()
    ctx.claude.model = "claude-example-9"
    ctx.claude.last_model = "claude-example-9-20990101"

    async def ask_unpriced(question, context_text, _self=ctx.claude):
        _self.calls.append((question, context_text))
        _self.last_usage = {"input_tokens": 12100, "output_tokens": 400}
        _self.last_cost_usd = None
        return "Take the RB."

    ctx.claude.ask = ask_unpriced
    assert main(["--offline", "ask", "Who?"]) == 0
    out = capsys.readouterr().out
    assert "12.1k in / 0.4k out - cost unknown (model not in the price table) (claude-example-9-20990101)" in out
    # without a key: refuse, never call
    ctx.claude = _FakeAsk()
    ctx.claude.enabled = False
    assert main(["--offline", "ask", "Who?"]) == cli.EXIT_USAGE
    assert "Claude is off" in capsys.readouterr().out and ctx.claude.calls == []


def test_cmd_prep_passes_offline_and_never_researches(monkeypatch, capsys):
    import draftadvisor.app as app
    from draftadvisor.config import Settings

    league = __import__("tests.test_dashboard", fromlist=["x"])._league()
    ctx = _fake_context(Settings(), league, None)
    ctx.sources.update({"snapshot": None, "metrics": {}})
    seen = {}

    async def fake_prep(settings, **kw):
        seen.update(kw)
        return ctx

    monkeypatch.setattr(app, "prep", fake_prep)
    rc = main(["prep", "--offline", "--no-train"])
    out = capsys.readouterr().out
    assert rc == 0
    assert seen["offline"] is True and seen["train"] is False
    assert "research" not in seen and "top" not in seen
    assert "Top 30" in out and "research" not in out.lower()


# ---------------------------------------------------------------------------
# ESPN: flags / settings, error messages, capture, ids, the draft loop against the stub, the TUI clock
# ---------------------------------------------------------------------------

import os  # noqa: E402
import re  # noqa: E402

from draftadvisor.config import Settings  # noqa: E402
from tests.espn_stub import LEAGUE_ID, SEASON, SWID_TEAM_1, EspnStub  # noqa: E402


@pytest.mark.parametrize("argv,command,checks", [
    (["capture", "--platform", "espn", "--league", "368876", "--season", "2018", "--team-id", "1", "--espn-s2", "s2",
      "--swid", "{X}"], "capture",
     {"platform": "espn", "league_id": "368876", "season": 2018, "team_id": 1, "espn_s2": "s2", "swid": "{X}"}),
    (["draft", "--platform", "espn", "--league", "L", "--slot", "3", "--no-tui"], "draft",
     {"platform": "espn", "league_id": "L", "slot": 3, "team_id": None, "no_tui": True}),
    (["ids", "--platform", "espn", "--swid", "{X}"], "ids", {"platform": "espn", "swid": "{X}", "username": None}),
    (["ask", "Q", "--platform", "espn", "--league", "L", "--team-id", "2"], "ask", {"platform": "espn", "team_id": 2}),
    (["analyze", "--platform", "espn", "--league", "L", "--username", "Goin' HAM Newton"], "analyze",
     {"platform": "espn", "username": "Goin' HAM Newton"}),
    (["projections", "--platform", "espn", "--league", "L"], "projections", {"platform": "espn", "league_id": "L"}),
    (["trade", "--platform", "espn", "--league", "L", "--me", "a", "--them", "b", "--give", "x", "--get", "y"], "trade",
     {"platform": "espn"}),
    (["prep", "--platform", "espn", "--league", "L", "--team-id", "4", "--no-train"], "prep", {"platform": "espn", "team_id": 4}),
    (["draft", "--draft", "D"], "draft", {"platform": None, "espn_s2": None, "swid": None, "team_id": None}),
])
def test_parser_espn_flags(argv, command, checks):
    args = build_parser().parse_args(argv)
    assert args.command == command
    for k, v in checks.items():
        assert getattr(args, k) == v, k


def test_parser_rejects_bad_espn_input():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["draft", "--platform", "yahoo", "--league", "L"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["draft", "--league", "L", "--team-id", "1", "--slot", "2"])     # mutually exclusive
    with pytest.raises(SystemExit):
        build_parser().parse_args(["capture", "--league", "L", "--team-id", "x"])


def test_settings_from_args_espn_env_and_flags(monkeypatch):
    monkeypatch.setenv("ESPN_S2", "env-s2")
    monkeypatch.setenv("ESPN_SWID", "{ENV}")
    monkeypatch.setenv("SLEEPER_DRAFT_ID", "stale-sleeper-draft")
    args = build_parser().parse_args(["draft", "--platform", "espn", "--league", "368876", "--season", "2018", "--team-id", "4"])
    s = cli._settings_from_args(args)
    assert s.platform == "espn" and s.is_espn and s.league_id == "368876" and s.season == 2018 and s.team_id == 4
    assert s.espn_s2 == "env-s2" and s.swid == "{ENV}" and s.has_espn_cookies
    assert s.draft_id is None                          # the Sleeper env id never leaks into an ESPN session
    assert s.poll_seconds == 3.0                       # ESPN's default cadence
    args = build_parser().parse_args(["draft", "--platform", "espn", "--league", "1", "--espn-s2", "flag", "--swid", "{FLAG}",
                                      "--poll", "1"])
    s = cli._settings_from_args(args)
    assert s.espn_s2 == "flag" and s.swid == "{FLAG}" and s.poll_seconds == 1.0
    # Sleeper is untouched by the ESPN environment
    s = cli._settings_from_args(build_parser().parse_args(["draft", "--draft", "D"]))
    assert s.platform == "sleeper" and not s.is_espn and s.draft_id == "D" and s.poll_seconds == 2.0 and s.team_id is None
    assert cli._settings_from_args(build_parser().parse_args(["draft"])).draft_id == "stale-sleeper-draft"
    # DRAFTADVISOR_PLATFORM / ESPN_LEAGUE_ID env defaults, --platform wins
    monkeypatch.setenv("DRAFTADVISOR_PLATFORM", "espn")
    monkeypatch.setenv("ESPN_LEAGUE_ID", "777")
    s = cli._settings_from_args(build_parser().parse_args(["capture"]))
    assert s.platform == "espn" and s.league_id == "777" and s.draft_id is None
    assert cli._settings_from_args(build_parser().parse_args(["capture", "--platform", "sleeper", "--league", "L"])).platform == "sleeper"
    with pytest.raises(ValueError):
        Settings.from_env(platform="yahoo")


def test_espn_errors_are_classified_and_actionable(monkeypatch, capsys):
    from draftadvisor.espn.client import EspnAccessDenied, EspnAPIError, EspnNotFound

    denied = EspnAccessDenied("private", status_code=401, url="http://espn/league/1")
    assert cli._is_access_denied(denied) and not cli._is_not_found_error(denied) and not cli._is_network_error(denied)
    assert cli._is_not_found_error(EspnNotFound("nf", status_code=404)) and not cli._is_network_error(EspnNotFound("nf"))
    assert cli._is_network_error(EspnAPIError("HTTP 503", status_code=503))
    assert cli._is_network_error(EspnAPIError("HTTP 429", status_code=429))
    assert cli._is_not_found_error(EspnAPIError("bad request", status_code=400))
    assert "ESPN API" in cli._network_message(EspnAPIError("boom"), Settings(platform="espn"))
    assert "Sleeper API" in cli._network_message(ConnectionError("x"), Settings())
    assert "ESPN API" in cli._network_message(ConnectionError("x"), Settings(platform="espn"))
    msg = cli._access_denied_message(denied, Settings(platform="espn", league_id="1"))
    assert "league 1 is private" in msg and "--espn-s2" in msg and "ESPN_S2" in msg and "Cookies" in msg
    msg = cli._access_denied_message(denied, Settings(platform="espn", league_id="1", espn_s2="a", swid="b"))
    assert "rejected" in msg and "a" != msg.split()[0]
    msg = cli._not_found_message(EspnNotFound("nf", status_code=404), Settings(platform="espn", league_id="42", season=2025))
    assert "ESPN has no league 42 for season 2025" in msg and "leagueId=" in msg

    # the draft command maps them to exit codes and prints the advice (never a traceback)
    for exc, rc, needle in ((denied, cli.EXIT_USAGE, "private"),
                            (EspnNotFound("nf", status_code=404), cli.EXIT_USAGE, "ESPN has no league 368876"),
                            (EspnAPIError("HTTP 503", status_code=503), cli.EXIT_NETWORK, "Could not reach the ESPN API")):
        async def boom(*a, **k):
            raise exc

        monkeypatch.setattr(cli, "_draft_async", boom)
        assert main(["draft", "--platform", "espn", "--league", "368876"]) == rc
        assert needle in capsys.readouterr().out
    # main() catches the same errors from any command
    monkeypatch.setattr(cli, "cmd_train", lambda args: (_ for _ in ()).throw(denied))
    parser = build_parser()
    monkeypatch.setattr(cli, "build_parser", lambda: _patch_func(parser, "train", cli.cmd_train))
    assert main(["train"]) == cli.EXIT_USAGE
    assert "private" in capsys.readouterr().err


@pytest.fixture
def espn_stub(monkeypatch):
    """The ESPN stub (mid-draft fixture) as the API of every ESPN client; wide console so names never wrap."""
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.delenv("ESPN_S2", raising=False)
    monkeypatch.delenv("ESPN_SWID", raising=False)
    with EspnStub(draft="in_progress") as s:
        monkeypatch.setenv("DRAFTADVISOR_ESPN_BASE", s.base_url)
        monkeypatch.setenv("DRAFTADVISOR_ESPN_FAN_BASE", s.fan_base_url)
        yield s


def _espn_argv(*extra: str) -> list[str]:
    return ["capture", "--platform", "espn", "--league", LEAGUE_ID, "--season", str(SEASON), *extra]


def test_capture_espn_prints_teams_and_my_slot(espn_stub, capsys):
    rc = main(_espn_argv("--swid", SWID_TEAM_1))
    out = capsys.readouterr().out
    assert rc == 0
    assert "FXBG League" in out and f"espn-{LEAGUE_ID}-{SEASON}" in out and "ESPN league (platform: espn)" in out
    assert "slot 3 (Goin' HAM Newton, roster 1)" in out and "3 ◀ you" in out and "Lutz Get Weird" in out
    assert "ESPN rule not modelled: 1pt Safety" in out and "Pick clock  90 s" in out and "Keepers on board  2" in out
    assert SWID_TEAM_1 not in out
    assert (Path(os.environ["DRAFTADVISOR_HOME"]) / "leagues" / f"{LEAGUE_ID}.json").exists()
    # the saved capture replays offline (no ESPN request is made)
    n = len(espn_stub.requests)
    rc = main(["--offline", *_espn_argv()])
    out = capsys.readouterr().out
    assert rc == 0 and "FXBG League" in out and "slot 3 (Goin' HAM Newton" in out and len(espn_stub.requests) == n
    # --team-id / --username / --slot identify a team too; nothing given -> spectator with a hint
    assert main(_espn_argv("--team-id", "8")) == 0 and "slot 2 (Lutz Get Weird" in capsys.readouterr().out
    assert main(_espn_argv("--username", "lutz")) == 0 and "slot 2 (Lutz Get Weird" in capsys.readouterr().out
    assert main(_espn_argv("--slot", "4")) == 0 and "slot 4 (Misunderstood Mistfits" in capsys.readouterr().out
    assert main(_espn_argv()) == 0
    out = capsys.readouterr().out
    assert "not identified" in out and "--team-id" in out
    # an unknown team lists the league's teams; a slot outside 1..teams is refused (never negative pick numbers)
    assert main(_espn_argv("--team-id", "99")) == cli.EXIT_USAGE
    assert "Goin' HAM Newton (team 1" in capsys.readouterr().out
    for bad in ("99", "0", "-1"):
        assert main(_espn_argv("--slot", bad)) == cli.EXIT_USAGE
        out = capsys.readouterr().out
        assert f"slot {bad} is not a draft slot" in out and "(1..10)" in out and "#-" not in out
    assert main(_espn_argv("--slot", "99", "--swid", SWID_TEAM_1)) == 0          # another hint still identifies the team
    assert "slot 3 (Goin' HAM Newton" in capsys.readouterr().out
    assert main(["capture", "--platform", "espn"]) == cli.EXIT_USAGE
    assert "leagueId=" in capsys.readouterr().out
    assert main(["--offline", "capture", "--platform", "espn", "--league", "424242"]) == cli.EXIT_USAGE


def test_capture_espn_not_found_and_private(monkeypatch, capsys):
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.delenv("ESPN_S2", raising=False)
    monkeypatch.delenv("ESPN_SWID", raising=False)
    with EspnStub(draft="in_progress") as s:
        monkeypatch.setenv("DRAFTADVISOR_ESPN_BASE", s.base_url)
        rc = main(["capture", "--platform", "espn", "--league", "999", "--season", "2018"])
        out = capsys.readouterr().out
        assert rc == cli.EXIT_USAGE and "ESPN has no league 999 for season 2018" in out and "leagueId=" in out
    with EspnStub(private=True, draft="in_progress") as s:
        monkeypatch.setenv("DRAFTADVISOR_ESPN_BASE", s.base_url)
        rc = main(_espn_argv())
        out = capsys.readouterr().out
        assert rc == cli.EXIT_USAGE and "is private" in out and "--espn-s2" in out and "ESPN_S2" in out and "Cookies" in out
        assert "Could not reach" not in out
        # with the cookies (env or flags) the private league answers; the cookie values never show
        monkeypatch.setenv("ESPN_S2", "secret-cookie-value")
        monkeypatch.setenv("ESPN_SWID", SWID_TEAM_1)
        rc = main(_espn_argv())
        out = capsys.readouterr().out
        assert rc == 0 and "slot 3 (Goin' HAM Newton" in out and "secret-cookie-value" not in out
        assert s.requests[-1]["cookies"] == {"espn_s2": "secret-cookie-value", "SWID": SWID_TEAM_1}


def test_ids_espn_lists_fan_leagues_or_explains(espn_stub, capsys, monkeypatch):
    rc = main(["ids", "--platform", "espn", "--swid", SWID_TEAM_1])
    out = capsys.readouterr().out
    assert rc == 0 and "FXBG League" in out and LEAGUE_ID in out and "Goin' HAM Newton" in out
    assert f"draft --platform espn --league {LEAGUE_ID} --season {SEASON} --team-id 1" in out
    # the fan API is best effort: an empty answer explains where the id is
    import draftadvisor.espn.client as espn_client

    async def empty(self, swid=None):
        return []

    monkeypatch.setattr(espn_client.EspnClient, "get_fan_leagues", empty)
    rc = main(["ids", "--platform", "espn", "--swid", SWID_TEAM_1])
    out = capsys.readouterr().out
    assert rc == 0 and "leagueId=" in out and "--espn-s2" in out
    assert main(["ids", "--platform", "espn"]) == cli.EXIT_USAGE
    assert "SWID" in capsys.readouterr().out
    assert main(["ids"]) == cli.EXIT_USAGE                             # Sleeper still needs --username
    assert "--username" in capsys.readouterr().out
    assert main(["--offline", "ids", "--platform", "espn", "--swid", "x"]) == cli.EXIT_USAGE


@pytest.fixture
def espn_context_env(monkeypatch, players_json):
    """build_context runs for real (ESPN overrides included) on the fixture universe: no network, no model."""
    import draftadvisor.app as app
    from draftadvisor.data.crosswalk import Crosswalk
    from draftadvisor.data.universe import players_from_sleeper

    universe = players_from_sleeper(players_json)

    def _raise(exc):
        def go(*a, **k):
            raise exc
        return go

    monkeypatch.setattr(app, "_build_crosswalk", lambda season: Crosswalk())
    monkeypatch.setattr(app, "_load_ecr", lambda superflex: (None, None))
    monkeypatch.setattr(app, "_load_byes", lambda season: {})
    monkeypatch.setattr(app, "_model_predictions", _raise(FileNotFoundError("no model in tests")))
    monkeypatch.setattr(app, "_offline_predictions", _raise(NotImplementedError("no offline projections in tests")))
    monkeypatch.setattr(app, "_build_players", lambda cw, season, sp: (dict(universe), "fixture"))

    async def no_sleeper(settings, console):
        return None, None

    monkeypatch.setattr(cli, "_sleeper_inputs", no_sleeper)
    return universe


def test_draft_espn_no_tui_polls_the_stub_until_told_to_stop(espn_context_env, capsys, monkeypatch):
    monkeypatch.setenv("COLUMNS", "200")
    renders: list[str] = []
    with EspnStub(draft="live", picks_visible=17) as s:
        monkeypatch.setenv("DRAFTADVISOR_ESPN_BASE", s.base_url)
        orig = cli._plain_printer

        def printer(console):
            base = orig(console)

            def out(text):
                renders.append(text)
                base(text)
                if len(renders) == 1:
                    s.picks_visible = 20        # three more picks come in ...
                elif len(renders) == 2:
                    s.picks_visible = None      # ... then the draft completes and the loop exits
            return out

        monkeypatch.setattr(cli, "_plain_printer", printer)
        rc = main(["draft", "--platform", "espn", "--league", LEAGUE_ID, "--season", str(SEASON), "--swid", SWID_TEAM_1,
                   "--no-tui", "--no-model", "--poll", "0.5"])
        requests = list(s.requests)
    out = capsys.readouterr().out
    assert rc == 0 and len(renders) == 3
    first, second, last = renders
    assert first.startswith("== [ESPN] FXBG League") and "Pick 18" in first and "** YOUR PICK **" in first
    assert "ESPN clock: 90 s per pick" in first and "since last pick seen" not in first   # no pick observed yet
    assert "Davante Adams (WR)" in first                                                  # ESPN-only players are named
    assert "Pick 21" in second and "You pick in 2 (#23)" in second
    assert re.search(r"ESPN clock: 90 s per pick • ≈ (8[7-9]|90) s left \(since last pick seen\)", second)
    assert "draft complete (150 picks)" in last and "since last pick seen" not in last
    assert "you are slot 3" in out and "ESPN league (platform: espn)" in out
    assert "ADP: espn" in out and "Draft complete." in out and "Projected starting lineups" in out and "(you)" in out
    assert "ESPN gives no pick timestamps" not in out                                     # the hint is a TUI-only string
    # one ESPN GET per poll after the capture (settings+teams, draft, player pool) and the poller bootstrap
    views = [tuple(r["query"].get("view", [])) for r in requests]
    assert views[:3] == [("mSettings", "mTeam", "mRoster"), ("mDraftDetail", "mSettings"), ("kona_player_info",)]
    assert len(views) >= 6 and all(v == ("mDraftDetail", "mSettings") for v in views[3:])
    assert all(r["cookies"].get("SWID") == SWID_TEAM_1 for r in requests)


def test_prep_espn_reports_private_unknown_league_and_unknown_team(espn_context_env, capsys, monkeypatch):
    """prep is the documented first step: a wrong id / missing cookies / unknown team must not silently
    build a cache for the default league (every other ESPN command already exits 2 with the how-to)."""
    import draftadvisor.data.nflverse as nv

    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setattr(nv, "load_canonical", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no download in tests")))
    home = Path(os.environ["DRAFTADVISOR_HOME"])
    with EspnStub(private=True, draft="in_progress") as s:
        monkeypatch.setenv("DRAFTADVISOR_ESPN_BASE", s.base_url)
        rc = main(["prep", "--platform", "espn", "--league", LEAGUE_ID, "--season", str(SEASON), "--no-train"])
        captured = capsys.readouterr()
        assert rc == cli.EXIT_USAGE and "is private" in captured.err and "--espn-s2" in captured.err
        assert "Top 30" not in captured.out and not list(home.glob("projections*"))
    with EspnStub(draft="in_progress") as s:
        monkeypatch.setenv("DRAFTADVISOR_ESPN_BASE", s.base_url)
        rc = main(["prep", "--platform", "espn", "--league", "999", "--season", str(SEASON), "--no-train"])
        captured = capsys.readouterr()
        assert rc == cli.EXIT_USAGE and "ESPN has no league 999" in captured.err and "Top 30" not in captured.out
        rc = main(["prep", "--platform", "espn", "--league", LEAGUE_ID, "--season", str(SEASON), "--no-train", "--team-id", "99"])
        captured = capsys.readouterr()
        assert rc == cli.EXIT_USAGE and "could not find 99" in captured.out and "Goin' HAM Newton (team 1" in captured.out
        assert "Top 30" not in captured.out
        # a good id still preps: the capture is saved and the league's projections print
        rc = main(["prep", "--platform", "espn", "--league", LEAGUE_ID, "--season", str(SEASON), "--no-train", "--team-id", "1"])
        out = capsys.readouterr().out
        assert rc == 0 and "Top 30 — FXBG League" in out and "slot 3 (Goin' HAM Newton" in out
        assert (home / "leagues" / f"{LEAGUE_ID}.json").exists()


def test_draft_espn_needs_a_league_id(capsys):
    assert main(["draft", "--platform", "espn"]) == cli.EXIT_USAGE
    assert "leagueId=" in capsys.readouterr().out


def test_projections_and_analyze_espn_via_the_stub(espn_stub, espn_context_env, capsys):
    rc = main(["projections", "--platform", "espn", "--league", LEAGUE_ID, "--season", str(SEASON), "--no-model", "--top", "5"])
    out = capsys.readouterr().out
    assert rc == 0 and "Projections — FXBG League" in out and "ADP: espn" in out and "Todd Gurley II" in out
    rc = main(["analyze", "--platform", "espn", "--league", LEAGUE_ID, "--season", str(SEASON), "--no-model",
               "--username", "Goin' HAM Newton"])
    out = capsys.readouterr().out
    assert rc == 0 and "Rosters — FXBG League" in out and "Goin' HAM Newton" in out and "Lutz Get Weird" in out
    assert "Best waiver targets" in out
    # offline with the saved capture: the league and the captured player pool (ESPN ADP) are known
    assert main(["capture", "--platform", "espn", "--league", LEAGUE_ID, "--season", str(SEASON)]) == 0
    capsys.readouterr()
    n = len(espn_stub.requests)
    espn = ["--platform", "espn", "--league", LEAGUE_ID, "--season", str(SEASON), "--no-model"]
    rc = main(["--offline", "projections", *espn, "--top", "3"])
    out = capsys.readouterr().out
    assert rc == 0 and "Projections — FXBG League" in out and "ADP: espn" in out
    # offline analyze / trade see every rostered player (placeholders for ids outside the universe), no request made
    rc = main(["--offline", "analyze", *espn, "--username", "Goin' HAM Newton"])
    out = capsys.readouterr().out
    assert rc == 0 and "Goin' HAM Newton" in out and "QB Cam Newton" in out and "Lutz Get Weird" in out
    assert "(Counts empty)" not in out and "QB RB RB WR WR TE FLEX K DEF" not in out
    rc = main(["--offline", "trade", *espn, "--me", "Goin' HAM Newton", "--them", "Lutz Get Weird",
               "--give", "Antonio Brown", "--get", "Todd Gurley"])
    out = capsys.readouterr().out
    assert rc == 0 and "Antonio Brown" in out and "Todd Gurley" in out and "is not on that roster" not in out
    assert len(espn_stub.requests) == n
    # the same ESPN capture is never replayed through the Sleeper parser when --platform espn is forgotten
    assert main(["--offline", "analyze", "--league", LEAGUE_ID, "--no-model"]) == cli.EXIT_NETWORK
    out = capsys.readouterr().out
    assert "no rosters available" in out and "Roster 4" not in out
    assert main(["--offline", "capture", "--league", LEAGUE_ID]) == cli.EXIT_USAGE
    out = capsys.readouterr().out
    assert "no saved snapshot" in out and "--platform espn" in out and "FXBG" not in out


def test_cmd_ask_espn_uses_the_live_espn_draft_as_context(espn_stub, monkeypatch, capsys):
    ctx = _fake_context(Settings(platform="espn", league_id=LEAGUE_ID, season=SEASON), _tl()._league(), None)
    ctx.claude = _FakeAsk()
    monkeypatch.setattr(cli, "_build", lambda settings, args, **kw: ctx)
    rc = main(["ask", "Who now?", "--platform", "espn", "--league", LEAGUE_ID, "--season", str(SEASON), "--swid", SWID_TEAM_1])
    assert rc == 0 and len(ctx.claude.calls) == 1
    question, context = ctx.claude.calls[0]
    assert question == "Who now?" and context.startswith("FXBG League: round 2 of 15, pick #18 of 150, 10 teams")
    assert "I am slot 3" in context and "IT IS MY PICK NOW" in context
    assert [tuple(r["query"].get("view", [])) for r in espn_stub.requests] == [("mSettings", "mTeam", "mRoster"),
                                                                              ("mDraftDetail", "mSettings")]


def _tl():
    return __import__("tests.test_dashboard", fromlist=["x"])


# -- the TUI: platform pill and the ESPN clock -------------------------------------------------------


def test_dashboard_shows_platform_and_espn_clock():
    from draftadvisor.ui.dashboard import Dashboard, _countdown, espn_clock_text, platform_of, render_text

    td = _tl()
    state = make_state(6)                                # my turn; pick_timer 30; Sleeper by default
    rec = make_rec(state)
    now = time.time()
    console = td._console()
    Dashboard(console, projections=PROJ).update(state, rec, PLAYERS, {"latency_ms": 100, "turn_started_at": now})
    text = console.export_text()
    assert " Sleeper " in text and "s left" in text and "ESPN clock" not in text
    assert platform_of(state) == "sleeper"
    # the same state as an ESPN draft: pill, clock line, no authoritative countdown
    status = {"platform": "espn", "latency_ms": 100, "turn_started_at": now}
    console = td._console()
    Dashboard(console, projections=PROJ).update(state, rec, PLAYERS, status)
    text = console.export_text()
    assert " ESPN " in text and "ESPN clock: 30 s per pick" in text and "s left" not in text
    assert _countdown(state, status) is None and platform_of(state, status) == "espn"
    # once a pick change was observed the clock is approximate and labelled
    status["last_pick_seen_at"] = now - 10
    assert espn_clock_text(state, status, now=now) == "ESPN clock: 30 s per pick • ≈ 20 s left (since last pick seen)"
    assert espn_clock_text(state, status, now=now + 100).endswith("≈ 0 s left (since last pick seen)")
    console = td._console()
    Dashboard(console, projections=PROJ).update(state, rec, PLAYERS, status)
    assert re.search(r"≈ (19|20) s left \(since last pick seen\)", console.export_text())
    # derived from the draft metadata when the status carries no platform
    state.draft.metadata["platform"] = "espn"
    assert platform_of(state) == "espn" and _countdown(state, {"turn_started_at": now}) is None
    no_timer = make_state(6)
    no_timer.draft.pick_timer = 0
    assert espn_clock_text(no_timer, {"last_pick_seen_at": now}) == "ESPN clock: no pick timer"
    done = make_state(6)
    done.draft.status = "complete"
    assert espn_clock_text(done, {"last_pick_seen_at": now}) == "ESPN clock: 30 s per pick"
    # plain text (--no-tui) carries the same information
    plain = render_text(rec, state, PLAYERS, {"platform": "espn", "last_pick_seen_at": now - 5})
    assert plain.startswith("== [ESPN] Test League") and "ESPN clock: 30 s per pick • ≈ 25 s left (since last pick seen)" in plain
    assert render_text(rec, make_state(6), PLAYERS).startswith("== [Sleeper] Test League")
    assert "ESPN clock" not in render_text(rec, make_state(6), PLAYERS, {"platform": "sleeper"})


async def test_loop_records_observed_pick_changes_only():
    loop = DraftLoop(FakeAdvisor(), PLAYERS, None, tui=False, out=lambda s: None, status={"platform": "espn"})
    await loop.handle(make_state(3))
    assert "last_pick_seen_at" not in loop.status              # the bootstrap state is not an observed pick
    await loop.handle(make_state(3))
    assert "last_pick_seen_at" not in loop.status              # same pick number: nothing observed
    await loop.handle(make_state(5))
    assert loop.status["last_pick_seen_at"] == pytest.approx(time.time(), abs=0.5)
    assert loop.status["platform"] == "espn"


def test_trades_espn_via_the_stub(espn_stub, espn_context_env, capsys):
    """The trade finder over the wire: real rosters in, proposals out, no paid call anywhere."""
    espn = ["--platform", "espn", "--league", LEAGUE_ID, "--season", str(SEASON), "--no-model"]
    rc = main(["trades", *espn, "--me", "Goin' HAM Newton", "--limit", "4", "--min-gain", "0"])
    out = capsys.readouterr().out
    assert rc == 0, out
    if "No trade found" in out:
        return                                    # a legitimate answer; the flags below still must work
    assert "send" in out and "they read it as" in out
    assert "Goin' HAM Newton" not in out.split("trade(s)")[-1], "never propose a trade with myself"
    assert "consensus" in out, "the pitch has to be written in their currency"


def test_trades_reports_a_bad_manager_name_instead_of_guessing(espn_stub, espn_context_env, capsys):
    rc = main(["trades", "--platform", "espn", "--league", LEAGUE_ID, "--season", str(SEASON),
               "--no-model", "--me", "Nobody At All"])
    out = capsys.readouterr().out
    assert rc == cli.EXIT_USAGE and "could not find" in out and "Goin' HAM Newton" in out


def test_trades_needs_a_league(capsys):
    assert main(["trades", "--me", "someone"]) == cli.EXIT_USAGE
    assert "needs --league" in capsys.readouterr().out


def test_lineup_espn_via_the_stub(espn_stub, espn_context_env, capsys):
    """Start/sit for one week, the gap against the lineup as set, and the matchup odds."""
    rc = main(["lineup", "--platform", "espn", "--league", LEAGUE_ID, "--season", str(SEASON),
               "--no-model", "--me", "Goin' HAM Newton", "--week", "4"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "Week 4 — Goin' HAM Newton" in out
    assert "regular-season weeks left" in out or "playoffs" in out
    assert "to win" in out and "%" in out, "the matchup odds are the point of the command"
    assert "model estimate" in out, "a win probability has to say it is a model, not a price"
    assert "vs " in out, "it has to name the opponent"
    # a player we cannot project is named, not silently scored zero and benched
    if "No projection for" in out:
        assert "will never be started" in out


def test_lineup_is_espn_only_and_says_so(capsys):
    assert main(["lineup", "--league", "123", "--me", "x"]) == cli.EXIT_USAGE
    assert "ESPN leagues only" in capsys.readouterr().out


def test_lineup_names_the_teams_when_it_cannot_tell_which_is_mine(espn_stub, espn_context_env, capsys):
    rc = main(["lineup", "--platform", "espn", "--league", LEAGUE_ID, "--season", str(SEASON),
               "--no-model", "--me", "Not A Team Here"])
    out = capsys.readouterr().out
    assert rc == cli.EXIT_USAGE and "could not tell which team is yours" in out and "Goin' HAM Newton" in out
