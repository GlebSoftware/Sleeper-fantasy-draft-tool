"""CLI tests: argparse, the draft loop with fakes, name resolution, offline mock draft."""
from __future__ import annotations

import asyncio
import io
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
