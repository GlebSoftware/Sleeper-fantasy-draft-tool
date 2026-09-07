"""Tests for draftadvisor.research (DESIGN.md §3.5). Fully offline: a fake Anthropic client."""
from __future__ import annotations

import asyncio
import json
import time

from draftadvisor.config import research_dir
from draftadvisor.models import (
    DraftSettings,
    DraftState,
    LeagueSettings,
    Pick,
    Player,
    PlayerValue,
    PositionAdvice,
    Projection,
    Recommendation,
    ResearchNote,
    RosterSummary,
)
from draftadvisor.research import ClaudeResearcher, build_context_text, describe_roster_slots, describe_scoring
from draftadvisor.research.claude import UNAVAILABLE_SUMMARY, _parse_json_object


# ---------------------------------------------------------------------------
# Fake Anthropic client
# ---------------------------------------------------------------------------


class Block:
    def __init__(self, type: str, text: str | None = None):
        self.type = type
        self.text = text


class Usage:
    input_tokens = 100
    output_tokens = 50
    cache_read_input_tokens = 0
    cache_creation_input_tokens = 0


class Message:
    def __init__(self, blocks, stop_reason="end_turn", model="claude-sonnet-5"):
        self.content = blocks
        self.stop_reason = stop_reason
        self.usage = Usage()
        self.model = model


def text_message(text: str, stop_reason: str = "end_turn") -> Message:
    return Message([Block("text", text)], stop_reason)


def json_message(**data) -> Message:
    base = dict(summary="Locked-in RB1, healthy after 2025 ankle scare.", injury_risk=0.2,
                role_certainty=0.9, upside="Top-3 RB.", downside="Ankle recurs.", sources=["https://x.y/z"])
    base.update(data)
    # mimic a search-enabled reply: server tool blocks + intro text + final JSON text
    return Message([
        Block("server_tool_use"), Block("web_search_tool_result"), Block("text", "Searching..."),
        Block("text", json.dumps(base)),
    ])


class FakeStream:
    """Stand-in for the object returned by ``client.messages.stream(...)``."""

    def __init__(self, message: Message, delay: float = 0.0):
        self._message = message
        self._delay = delay

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get_final_message(self):
        if self._delay:
            await asyncio.sleep(self._delay)
        return self._message


class FakeMessages:
    def __init__(self, responses=None, stream_message=None, stream_delay=0.0, error=None):
        self.responses = list(responses or [])
        self.stream_message = stream_message or text_message("Take X. Fallback Y.")
        self.stream_delay = stream_delay
        self.error = error
        self.create_calls: list[dict] = []
        self.stream_calls: list[dict] = []

    async def create(self, **kwargs):
        self.create_calls.append(kwargs)
        if self.error is not None:
            raise self.error
        if callable(self.responses[0] if self.responses else None):
            return self.responses.pop(0)(kwargs)
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]

    def stream(self, **kwargs):
        self.stream_calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return FakeStream(self.stream_message, self.stream_delay)


class FakeClient:
    def __init__(self, **kw):
        self.messages = FakeMessages(**kw)


# ---------------------------------------------------------------------------
# Draft fixtures (built directly from models, no other agents' modules)
# ---------------------------------------------------------------------------


def _player(pid, name, pos, team="KC", **kw) -> Player:
    return Player(player_id=pid, name=name, position=pos, team=team, **kw)


PLAYERS: dict[str, Player] = {p.player_id: p for p in [
    _player("1", "Alpha Back", "RB", "SF", bye_week=9, adp=3),
    _player("2", "Bravo Wide", "WR", "MIN", bye_week=6, adp=5),
    _player("3", "Charlie Tight", "TE", "KC", bye_week=10, adp=20),
    _player("4", "Delta Back", "RB", "DET", bye_week=5, adp=12),
    _player("5", "Echo Wide", "WR", "CIN", bye_week=10, adp=15, injury_status="Questionable"),
    _player("6", "Foxtrot QB", "QB", "BUF", bye_week=12, adp=30),
    _player("7", "Golf Back", "RB", "ATL", bye_week=5, adp=25),
    _player("8", "Hotel Wide", "WR", "PHI", bye_week=5, adp=28),
    _player("9", "India Wide", "WR", "DAL", bye_week=7, adp=40),
    _player("SF", "49ers", "DEF", "SF", bye_week=9),
]}


def _league() -> LeagueSettings:
    return LeagueSettings(
        league_id="L1", name="Test League", season=2026, total_rosters=4,
        roster_positions=["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF", "BN", "BN"],
        scoring_settings={"rec": 0.5, "pass_td": 4, "pass_yd": 0.04, "rush_td": 6, "rec_td": 6, "bonus_rec_te": 0.5},
    )


def _state(n_picks: int = 5) -> DraftState:
    d = DraftSettings(draft_id="D1", league_id="L1", type="snake", status="drafting", teams=4, rounds=3,
                      pick_timer=30, draft_order={"me": 2, "b1": 1, "b3": 3, "b4": 4},
                      slot_to_roster_id={1: 1, 2: 2, 3: 3, 4: 4})
    order = ["1", "2", "3", "4", "5", "SF", "7", "8"]  # player ids taken in pick order
    picks = [Pick(pick_no=n, round=d.round_of(n), draft_slot=d.slot_for_pick(n), player_id=order[n - 1],
                  roster_id=d.slot_for_pick(n), metadata={"position": PLAYERS[order[n - 1]].position})
             for n in range(1, n_picks + 1)]
    return DraftState(draft=d, picks=picks, league=_league(), my_user_id="me", my_slot=2, version=n_picks)


def _proj(pl: Player, points: float) -> Projection:
    return Projection(player_id=pl.player_id, position=pl.position, points=points, std=25.0,
                      ppg=points / 17, games=17.0)


def _value(pl: Player, points: float, score: float, avail: float, rank: int) -> PlayerValue:
    return PlayerValue(player=pl, projection=_proj(pl, points), vorp=points - 150, vona=10.0,
                       marginal_value=points - 100, score=score, availability_next=avail,
                       availability_after_next=avail * 0.5, tier=1 + rank // 3, pos_rank=rank, overall_rank=rank,
                       reasons=[f"Proj {points:.0f} pts", "Fills your open slot"], warnings=[])


def _rec(state: DraftState) -> Recommendation:
    # my roster: slot 2 has pick 2 (Bravo Wide, WR) in a 5-pick state
    mine = [PLAYERS[p.player_id] for p in state.my_picks()]
    roster = RosterSummary(
        slot=2, label="Slot 2", players=mine,
        starters_filled={"QB": 0, "RB": 0, "WR": 1, "TE": 0, "FLEX": 0, "K": 0, "DEF": 0},
        open_starters={"QB": 1, "RB": 2, "WR": 1, "TE": 1, "FLEX": 1, "K": 1, "DEF": 1},
        position_counts={"WR": 1}, bye_weeks={6: 1}, lineup_points=280.0,
    )
    available = [PLAYERS[i] for i in ("6", "7", "8", "9")]
    values = [_value(pl, 300 - 20 * i, 90 - 10 * i, 0.3 + 0.15 * i, i + 1) for i, pl in enumerate(available)]
    by_pos = {}
    for pv in values:
        by_pos.setdefault(pv.player.position, []).append(pv)
    advice = {pos: PositionAdvice(position=pos, candidates=c, action="TAKE NOW" if pos == "RB" else "WAIT",
                                  rationale=f"{int(c[0].availability_next*100)}% available at your next pick (#7)",
                                  expected_next_available=200.0, drop_off=20.0) for pos, c in by_pos.items()}
    return Recommendation(state_version=state.version, computed_at=time.time(), compute_ms=3.0,
                          best_overall=values, by_position=advice, my_roster=roster, opponent_rosters=[],
                          position_pressure={"RB": 1.5, "WR": 0.8, "QB": 0.4}, notes=["RB run: 3 of last 5 picks"])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_disabled_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = ClaudeResearcher()
    assert r.enabled is False
    st = _state()
    rec = _rec(st)

    async def run():
        assert await r.research_players(list(PLAYERS.values())) == {}
        assert await r.on_the_clock_advice(st, rec, PLAYERS, {}) is None
        assert await r.ask("who?", "ctx") == ""

    asyncio.run(run())
    assert r.load_notes() == {}
    assert r.request_count == 0


def test_enabled_with_env_key_does_not_build_client_at_init(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    r = ClaudeResearcher()
    assert r.enabled is True
    assert r._client is None           # lazy: SDK client only built on first request


def test_context_text_has_roster_candidates_picks_pressure():
    st = _state(6)                      # 4-team snake: slot 2 owns picks 2, 7, 10, 15 -> pick 7 is mine
    rec = _rec(st)
    notes = {"7": ResearchNote("7", "Committee back but goal-line role.", 0.3, 0.6, "RB1 if injury", "Loses snaps")}
    text = build_context_text(st, rec, PLAYERS, notes)
    assert "round 2 of 3" in text and "pick #7" in text
    assert "I am slot 2" in text and "IT IS MY PICK NOW" in text
    assert "My next picks: #7, #10" in text
    assert "Bravo Wide" in text                                        # my roster
    assert "Open needs" in text and "RB" in text
    for name in ("Foxtrot QB", "Golf Back", "Hotel Wide", "India Wide"):
        assert name in text
    assert "VORP" in text and "avail@next" in text and "tier" in text
    assert "Committee back but goal-line role." in text               # research note attached
    assert "LAST 6 PICKS" in text and "#6" in text and "Echo Wide" in text and "49ers (DEF)" in text
    assert "EXPECTED PICKS BY POSITION" in text and "RB 1.5" in text
    assert "RB: TAKE NOW" in text
    assert "RB run" in text


def test_context_text_without_recommendation():
    st = _state()
    text = build_context_text(st, None, PLAYERS, None)
    assert "MY ROSTER" in text and "Bravo Wide" in text and "LAST 5 PICKS" in text


def test_scoring_and_slot_descriptions():
    lg = _league()
    s = describe_scoring(lg)
    assert "HALF-PPR" in s and "rec 0.5" in s and "TE premium" in s
    slots = describe_roster_slots(lg)
    assert "RB x2" in slots and "FLEX(RB/WR/TE)" in slots and "bench 2" in slots
    assert describe_scoring(None)


def test_research_notes_cached_and_reused():
    client = FakeClient(responses=[json_message()])
    r = ClaudeResearcher(api_key="k", client=client)
    seen = []
    players = [PLAYERS["1"], PLAYERS["2"]]
    projections = {"1": _proj(PLAYERS["1"], 280)}

    notes = asyncio.run(r.research_players(players, projections, progress=lambda d, t, n: seen.append((d, t, n))))
    assert set(notes) == {"1", "2"}
    assert notes["1"].summary.startswith("Locked-in") and notes["1"].injury_risk == 0.2
    assert notes["1"].role_certainty == 0.9 and notes["1"].sources == ["https://x.y/z"]
    assert notes["1"].model == "claude-sonnet-5"
    assert len(client.messages.create_calls) == 2
    assert sorted(seen) == [(1, 2, "Alpha Back"), (2, 2, "Bravo Wide")] or len(seen) == 2

    # request shape
    call = client.messages.create_calls[0]
    assert call["model"] == "claude-sonnet-5"
    assert call["tools"][0]["type"] == "web_search_20260209" and call["tools"][0]["max_uses"] == 3
    assert call["output_config"]["effort"] == "medium"
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "2026" in call["system"][0]["text"]
    user = call["messages"][0]["content"]
    assert "Today is" in user and "Alpha Back" in user and "280 season points" in user
    assert "thinking" not in call

    # on disk
    path = research_dir() / "notes" / "1.json"
    assert path.exists()
    assert json.load(open(path))["player_id"] == "1"
    assert set(r.load_notes()) == {"1", "2"}

    # second run: fresh notes -> no new requests
    notes2 = asyncio.run(r.research_players(players))
    assert len(client.messages.create_calls) == 2 and set(notes2) == {"1", "2"}

    # stale notes are refreshed
    notes3 = asyncio.run(r.research_players(players, max_age_days=0))
    assert len(client.messages.create_calls) == 4 and set(notes3) == {"1", "2"}


def test_research_refusal_and_error_write_minimal_notes():
    client = FakeClient(responses=[text_message("", stop_reason="refusal")])
    r = ClaudeResearcher(api_key="k", client=client)
    notes = asyncio.run(r.research_players([PLAYERS["3"]]))
    n = notes["3"]
    assert n.summary == UNAVAILABLE_SUMMARY and n.injury_risk == 0.0 and n.role_certainty == 0.5
    assert (research_dir() / "notes" / "3.json").exists()

    # errors of any kind -> minimal note, never raised
    bad = FakeClient(error=RuntimeError("boom"))
    r2 = ClaudeResearcher(api_key="k", client=bad)
    notes = asyncio.run(r2.research_players([PLAYERS["4"]]))
    assert notes["4"].summary == UNAVAILABLE_SUMMARY and "boom" in (r2.last_error or "")

    # unavailable notes are retried on the next run (not treated as fresh)
    good = FakeClient(responses=[json_message(summary="Back.")])
    r3 = ClaudeResearcher(api_key="k", client=good)
    notes = asyncio.run(r3.research_players([PLAYERS["4"]]))
    assert notes["4"].summary == "Back." and len(good.messages.create_calls) == 1


def test_research_unparseable_and_clamped_values():
    client = FakeClient(responses=[text_message("not json at all")])
    r = ClaudeResearcher(api_key="k", client=client)
    notes = asyncio.run(r.research_players([PLAYERS["5"]]))
    assert notes["5"].summary == UNAVAILABLE_SUMMARY

    client = FakeClient(responses=[json_message(injury_risk=7, role_certainty=-2, sources="https://a.b")])
    r = ClaudeResearcher(api_key="k", client=client)
    notes = asyncio.run(r.research_players([PLAYERS["5"]], max_age_days=0))
    assert notes["5"].injury_risk == 1.0 and notes["5"].role_certainty == 0.0
    assert notes["5"].sources == ["https://a.b"]


def test_research_pause_turn_resends_with_assistant_content():
    paused = Message([Block("server_tool_use"), Block("text", "still searching")], stop_reason="pause_turn")
    client = FakeClient(responses=[paused, json_message(summary="Done.")])
    r = ClaudeResearcher(api_key="k", client=client)
    notes = asyncio.run(r.research_players([PLAYERS["6"]]))
    assert notes["6"].summary == "Done."
    calls = client.messages.create_calls
    assert len(calls) == 2
    msgs = calls[1]["messages"]
    assert msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant" and msgs[1]["content"] is paused.content


def test_research_concurrency_respects_semaphore():
    active = {"now": 0, "max": 0}

    def make(kwargs):
        return json_message()

    class SlowMessages(FakeMessages):
        async def create(self, **kwargs):
            active["now"] += 1
            active["max"] = max(active["max"], active["now"])
            await asyncio.sleep(0.01)
            active["now"] -= 1
            return json_message()

    client = FakeClient()
    client.messages = SlowMessages()
    r = ClaudeResearcher(api_key="k", client=client)
    notes = asyncio.run(r.research_players(list(PLAYERS.values()), concurrency=2))
    assert len(notes) == len(PLAYERS) and active["max"] == 2


def test_advice_prompt_streaming_and_cache():
    client = FakeClient(stream_message=text_message("Take Golf Back: he fills RB1. Fallback: Hotel Wide."))
    r = ClaudeResearcher(api_key="k", client=client)
    st = _state()
    rec = _rec(st)
    notes = {"7": ResearchNote("7", "Committee back.", 0.3, 0.6, "u", "d")}

    text = asyncio.run(r.on_the_clock_advice(st, rec, PLAYERS, notes))
    assert text and text.startswith("Take Golf Back")
    assert rec.claude_advice == text
    assert len(client.messages.stream_calls) == 1 and client.messages.create_calls == []
    call = client.messages.stream_calls[0]
    assert call["max_tokens"] == 400 and call["output_config"] == {"effort": "low"}
    assert call["model"] == "claude-sonnet-5" and "tools" not in call and "thinking" not in call
    sys_block = call["system"][0]
    assert sys_block["cache_control"] == {"type": "ephemeral"}
    assert "Test League" in sys_block["text"] and "HALF-PPR" in sys_block["text"] and "FLEX" in sys_block["text"]
    user = call["messages"][0]["content"]
    assert "Bravo Wide" in user and "Golf Back" in user and "Committee back." in user and "Open needs" in user

    # cache hit: same state version + same top candidates -> no second request
    again = asyncio.run(r.on_the_clock_advice(st, rec, PLAYERS, notes))
    assert again == text and len(client.messages.stream_calls) == 1
    # system prompt stays byte-identical for the same league (prompt-cache friendly)
    assert r.advice_system_prompt(st.league) == sys_block["text"]

    # new state -> new request
    st2 = st.with_picks(st.picks)
    rec2 = _rec(st2)
    asyncio.run(r.on_the_clock_advice(st2, rec2, PLAYERS, notes))
    assert len(client.messages.stream_calls) == 2


def test_advice_timeout_returns_none():
    client = FakeClient(stream_delay=5.0)
    r = ClaudeResearcher(api_key="k", client=client)
    st = _state()
    rec = _rec(st)
    t0 = time.perf_counter()
    text = asyncio.run(r.on_the_clock_advice(st, rec, PLAYERS, {}, timeout=0.05))
    assert text is None and (time.perf_counter() - t0) < 2.0
    assert rec.claude_advice is None
    assert r._advice_inflight == {}


def test_advice_errors_and_refusal_return_none():
    st = _state()
    rec = _rec(st)
    r = ClaudeResearcher(api_key="k", client=FakeClient(error=ConnectionError("down")))
    assert asyncio.run(r.on_the_clock_advice(st, rec, PLAYERS, {})) is None
    assert "down" in (r.last_error or "")
    r = ClaudeResearcher(api_key="k", client=FakeClient(stream_message=text_message("", "refusal")))
    assert asyncio.run(r.on_the_clock_advice(st, rec, PLAYERS, {})) is None


def test_advice_concurrent_calls_share_one_request():
    client = FakeClient(stream_message=text_message("Take him."), stream_delay=0.02)
    r = ClaudeResearcher(api_key="k", client=client)
    st = _state()
    rec = _rec(st)

    async def run():
        return await asyncio.gather(*(r.on_the_clock_advice(st, rec, PLAYERS, {}) for _ in range(3)))

    assert asyncio.run(run()) == ["Take him."] * 3
    assert len(client.messages.stream_calls) == 1


def test_ask_uses_context_and_medium_effort():
    client = FakeClient(responses=[text_message("Golf Back, because RB is scarce.")])
    r = ClaudeResearcher(api_key="k", client=client)
    st = _state()
    ctx = build_context_text(st, _rec(st), PLAYERS, {})
    answer = asyncio.run(r.ask("Who is the best RB?", ctx))
    assert answer == "Golf Back, because RB is scarce."
    call = client.messages.create_calls[0]
    assert call["max_tokens"] == 1500 and call["output_config"] == {"effort": "medium"}
    content = call["messages"][0]["content"]
    assert "QUESTION: Who is the best RB?" in content and "Golf Back" in content
    # errors -> "" with last_error
    r2 = ClaudeResearcher(api_key="k", client=FakeClient(error=RuntimeError("nope")))
    assert asyncio.run(r2.ask("q", "")) == "" and "nope" in r2.last_error


def test_load_notes_skips_corrupt_files(tmp_path):
    r = ClaudeResearcher(api_key="k", client=FakeClient(), cache_dir=tmp_path / "res")
    r.save_note(ResearchNote("42", "ok", 0.1, 0.8, "u", "d", ["s"]))
    (tmp_path / "res" / "notes" / "bad.json").write_text("{not json")
    notes = r.load_notes()
    assert set(notes) == {"42"} and notes["42"].sources == ["s"]


def test_parse_json_object_variants():
    assert _parse_json_object(['{"a": 1}']) == {"a": 1}
    assert _parse_json_object(["Searching", 'Here: {"a": 2} done'])["a"] == 2
    assert _parse_json_object(["nothing"]) is None
    assert _parse_json_object(['[1, 2]']) is None
