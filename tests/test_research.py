"""Tests for draftadvisor.research (DESIGN.md §3.5). Fully offline: a fake Anthropic client.

The layer has exactly two user-initiated entry points (``chat_stream`` for the web chat, ``ask``
for the CLI); these tests also pin that nothing automatic (research, on-the-clock advice) exists.
"""
from __future__ import annotations

import asyncio
import json
import time

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
from draftadvisor.research import (
    MODEL_PRICES_USD_PER_MTOK,
    ClaudeChat,
    ClaudeResearcher,
    build_context_text,
    describe_roster_slots,
    describe_scoring,
    estimate_cost_usd,
)
from draftadvisor.research.claude import CHAT_SYSTEM_PROMPT, TRUNCATION_MARKER, model_prices, usage_dict


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


def text_message(text: str, stop_reason: str = "end_turn", model: str = "claude-sonnet-5") -> Message:
    return Message([Block("text", text)], stop_reason, model)


class FakeStream:
    """Stand-in for the object returned by ``client.messages.stream(...)``."""

    def __init__(self, message: Message, delay: float = 0.0):
        self._message = message
        self._delay = delay

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    @property
    async def text_stream(self):
        for text in (b.text for b in self._message.content if b.type == "text" and b.text):
            for i in range(0, len(text), 7):        # deliver the answer in small deltas
                yield text[i:i + 7]

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


async def _collect(agen) -> tuple[str, dict]:
    """Drain ``chat_stream``: (joined text, final done-dict)."""
    text, done = "", {}
    async for chunk in agen:
        if isinstance(chunk, dict):
            done = chunk
        else:
            text += chunk
    return text, done


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_disabled_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = ClaudeChat()
    assert r.enabled is False

    async def run():
        assert await r.ask("who?", "ctx") == ""
        try:
            await _collect(r.chat_stream([{"role": "user", "content": "hi"}], ""))
        except RuntimeError as e:
            assert "disabled" in str(e)
        else:  # pragma: no cover
            raise AssertionError("chat_stream must refuse without a key")

    asyncio.run(run())
    assert r.load_notes() == {}
    assert r.request_count == 0 and r.last_usage is None and r.last_cost_usd is None


def test_enabled_with_env_key_does_not_build_client_at_init(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    r = ClaudeChat()
    assert r.enabled is True
    assert r._client is None           # lazy: SDK client only built on first request


def test_no_automatic_claude_paths_exist():
    """The research loop and the on-the-clock advice are gone for good (they ran up the bill)."""
    assert ClaudeResearcher is ClaudeChat                          # backwards-compatible alias only
    for name in ("research_players", "research_player", "on_the_clock_advice", "save_note", "advice_system_prompt"):
        assert not hasattr(ClaudeChat, name), name
    import draftadvisor.research.claude as mod

    for name in ("RESEARCH_SCHEMA", "RESEARCH_SYSTEM_PROMPT", "WEB_SEARCH_TOOL", "ADVICE_MAX_TOKENS", "SERVER_TOOL_BLOCK_TYPES"):
        assert not hasattr(mod, name), name
    assert "web_search" not in open(mod.__file__, encoding="utf-8").read()


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
    assert "Committee back but goal-line role." in text               # legacy note attached
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


# -- cost estimate -------------------------------------------------------------


def test_estimate_cost_usd():
    assert MODEL_PRICES_USD_PER_MTOK == {"claude-opus-5": (5.0, 25.0), "claude-sonnet-5": (2.0, 10.0)}
    # 12.1k in / 0.4k out on Opus: 12100 * 5 + 400 * 25 = 70 500 micro-dollars
    assert estimate_cost_usd("claude-opus-5", {"input_tokens": 12100, "output_tokens": 400}) == 0.0705
    assert estimate_cost_usd("claude-sonnet-5", {"input_tokens": 1000, "output_tokens": 100}) == 0.003
    # cache reads at 10 %, cache writes at 125 % of the input price
    assert estimate_cost_usd("claude-sonnet-5", {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 10000}) == 0.002
    assert estimate_cost_usd("claude-sonnet-5", {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 10000}) == 0.025
    # SDK usage objects work too, and a dated variant of a known id is priced like the base id
    assert estimate_cost_usd("claude-sonnet-5-20260101", Usage()) == round((100 * 2 + 50 * 10) / 1e6, 6)
    assert model_prices("CLAUDE-OPUS-5") == (5.0, 25.0) and model_prices("gpt-x") is None
    assert usage_dict(Usage()) == {"input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    # unknown model / no usage / no token counts -> None, never a guess
    assert estimate_cost_usd("claude-haiku-4-5", {"input_tokens": 10, "output_tokens": 1}) is None
    assert estimate_cost_usd("claude-opus-5", None) is None
    assert estimate_cost_usd("claude-opus-5", {"cache_read_input_tokens": 5}) is None
    assert estimate_cost_usd(None, {"input_tokens": 10}) is None


# -- chat (web) ------------------------------------------------------------------


def test_chat_stream_injects_context_and_reports_usage_and_cost():
    client = FakeClient(stream_message=text_message("Take Golf Back; RB is thin.", model="claude-opus-5"))
    r = ClaudeChat(api_key="k", client=client)
    st = _state()
    ctx = build_context_text(st, _rec(st), PLAYERS, {})
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"},
               {"role": "user", "content": "first"}, {"role": "user", "content": "Who?"}]
    text, done = asyncio.run(_collect(r.chat_stream(history, ctx, model="claude-opus-5")))
    assert text == "Take Golf Back; RB is thin."
    assert done["done"] is True and done["stop_reason"] == "end_turn" and done["model"] == "claude-opus-5"
    assert done["usage"] == {"input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    assert done["cost_usd"] == round((100 * 5 + 50 * 25) / 1e6, 6)
    assert r.request_count == 1 and r.last_cost_usd == done["cost_usd"] and r.last_usage == done["usage"]
    call = client.messages.stream_calls[0]
    assert call["model"] == "claude-opus-5" and call["max_tokens"] == 2000 and call["output_config"] == {"effort": "medium"}
    assert call["system"] == CHAT_SYSTEM_PROMPT and "cache_control" not in call
    msgs = call["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]           # consecutive user turns merged
    assert msgs[0]["content"] == "hi" and msgs[1]["content"] == "hello"
    last = msgs[-1]["content"]
    assert last.startswith("[Live draft context]\n") and "Golf Back" in last and last.endswith("[Question]\nfirst\n\nWho?")


def test_trim_transcript_keeps_the_newest_turns_within_the_caps():
    """The web endpoint bounds what one request carries: newest turns first, user-first, last turn always kept."""
    from draftadvisor.research.claude import CHAT_MAX_CHARS, CHAT_MAX_TURNS, trim_transcript

    turns = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"t{i}"} for i in range(11)]
    assert trim_transcript(turns) == turns and CHAT_MAX_TURNS >= 11 and CHAT_MAX_CHARS >= 60_000
    assert trim_transcript(turns, max_turns=3) == turns[-3:]
    assert trim_transcript(turns, max_turns=4) == turns[-3:]              # the leading assistant turn is dropped
    big = [{"role": "user", "content": "a" * 50_000}, {"role": "assistant", "content": "b" * 20_000},
           {"role": "user", "content": "c" * 5_000}, {"role": "assistant", "content": "d" * 3_000},
           {"role": "user", "content": "e" * 30_000}]
    assert trim_transcript(big) == big[2:]                                 # 30k + 3k + 5k + 20k fit, + 50k would not
    assert trim_transcript(big, max_chars=30_000) == big[-1:]
    huge = [{"role": "user", "content": "z" * 70_000}]
    assert trim_transcript(huge) == huge                                   # never empty: the endpoint rejects it (413)
    assert trim_transcript([]) == []


def test_chat_stream_validates_input_and_prices_unknown_models_as_none():
    client = FakeClient(stream_message=text_message("ok", model="claude-mystery-9"))
    r = ClaudeChat(api_key="k", client=client)
    try:
        asyncio.run(_collect(r.chat_stream([{"role": "assistant", "content": "x"}], "")))
    except ValueError as e:
        assert "last message" in str(e)
    else:  # pragma: no cover
        raise AssertionError("an assistant-last transcript must be rejected")
    assert client.messages.stream_calls == []                       # nothing was sent (and paid for)
    # a served model outside the price table falls back to the requested model's price; none -> None
    _, done = asyncio.run(_collect(r.chat_stream([{"role": "user", "content": "q"}], "", model="claude-mystery-9")))
    assert done["model"] == "claude-mystery-9" and done["cost_usd"] is None
    _, done = asyncio.run(_collect(r.chat_stream([{"role": "user", "content": "q"}], "", model="claude-sonnet-5")))
    assert done["cost_usd"] == round((100 * 2 + 50 * 10) / 1e6, 6)
    assert client.messages.stream_calls[-1]["messages"][-1]["content"] == "q"   # no context block when there is none


# -- ask (CLI) ---------------------------------------------------------------------


def test_ask_uses_context_and_medium_effort():
    client = FakeClient(responses=[text_message("Golf Back, because RB is scarce.")])
    r = ClaudeChat(api_key="k", client=client)
    st = _state()
    ctx = build_context_text(st, _rec(st), PLAYERS, {})
    answer = asyncio.run(r.ask("Who is the best RB?", ctx))
    assert answer == "Golf Back, because RB is scarce."
    call = client.messages.create_calls[0]
    assert call["max_tokens"] == 1500 and call["output_config"] == {"effort": "medium"}
    content = call["messages"][0]["content"]
    assert "QUESTION: Who is the best RB?" in content and "Golf Back" in content
    assert r.request_count == 1 and r.last_usage["input_tokens"] == 100
    assert r.last_cost_usd == round((100 * 2 + 50 * 10) / 1e6, 6)          # the fake answers as claude-sonnet-5
    # errors -> "" with last_error
    r2 = ClaudeChat(api_key="k", client=FakeClient(error=RuntimeError("nope")))
    assert asyncio.run(r2.ask("q", "")) == "" and "nope" in r2.last_error


def test_ask_appends_truncation_marker_on_max_tokens():
    client = FakeClient(responses=[text_message("Golf Back, because RB is scarce and", "max_tokens")])
    r = ClaudeChat(api_key="k", client=client)
    answer = asyncio.run(r.ask("Who?", "ctx"))
    assert answer == "Golf Back, because RB is scarce and" + TRUNCATION_MARKER
    assert "max_tokens" in (r.last_error or "")
    assert client.messages.create_calls[0]["max_tokens"] == 1500
    assert "cache_control" not in client.messages.create_calls[0]["system"][0]
    # no text at all -> "" (caller shows last_error)
    r2 = ClaudeChat(api_key="k", client=FakeClient(responses=[text_message("", "max_tokens")]))
    assert asyncio.run(r2.ask("Who?", "ctx")) == "" and "max_tokens" in (r2.last_error or "")


# -- legacy notes on disk ------------------------------------------------------------


def test_load_notes_reads_legacy_files_and_skips_corrupt_ones(tmp_path):
    r = ClaudeChat(api_key="k", client=FakeClient(), cache_dir=tmp_path / "res")
    notes_dir = tmp_path / "res" / "notes"
    notes_dir.mkdir(parents=True)
    (notes_dir / "42.json").write_text(json.dumps(ResearchNote("42", "ok", 0.1, 0.8, "u", "d", ["s"]).to_dict()), encoding="utf-8")
    (notes_dir / "bad.json").write_text("{not json")
    notes = r.load_notes()
    assert set(notes) == {"42"} and notes["42"].sources == ["s"]
    assert r.request_count == 0                                      # reading notes never calls Claude
