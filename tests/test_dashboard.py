"""Dashboard rendering tests: build models directly, render to a recording console, assert key strings."""
from __future__ import annotations

import io
import time

import pytest
from rich.console import Console

from draftadvisor.models import (
    DraftSettings,
    DraftState,
    LeagueSettings,
    Manager,
    Pick,
    Player,
    PlayerValue,
    PositionAdvice,
    Projection,
    Recommendation,
    RosterSummary,
)
from draftadvisor.ui.dashboard import Dashboard, assign_roster_slots, render_text, scoring_description

TEAMS, ROUNDS = 4, 3
SLOTS = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF", "BN", "BN"]


def _player(pid: str, name: str, pos: str, team: str, bye: int, adp: float) -> Player:
    return Player(player_id=pid, name=name, position=pos, team=team, bye_week=bye, adp=adp, adp_source="test",
                  ecr=adp, ecr_sd=3.0, years_exp=3)


def _proj(pl: Player, pts: float) -> Projection:
    return Projection(pl.player_id, pl.position, pts, 0.15 * pts, pts / 17, 17.0, floor=0.85 * pts, ceiling=1.15 * pts)


POOL = [
    _player("qb1", "Josh Allen", "QB", "BUF", 12, 20), _player("qb2", "Jalen Hurts", "QB", "PHI", 5, 30),
    _player("rb1", "Bijan Robinson", "RB", "ATL", 12, 2), _player("rb2", "Jahmyr Gibbs", "RB", "DET", 6, 3),
    _player("rb3", "Derrick Henry", "RB", "BAL", 13, 9), _player("rb4", "De'Von Achane", "RB", "MIA", 6, 12),
    _player("wr1", "Ja'Marr Chase", "WR", "CIN", 6, 1), _player("wr2", "Justin Jefferson", "WR", "MIN", 6, 4),
    _player("wr3", "CeeDee Lamb", "WR", "DAL", 14, 5), _player("wr4", "Puka Nacua", "WR", "LAR", 11, 7),
    _player("te1", "Brock Bowers", "TE", "LV", 10, 15), _player("te2", "Trey McBride", "TE", "ARI", 11, 22),
    _player("k1", "Brandon Aubrey", "K", "DAL", 14, 140), _player("SF", "SF Defense", "DEF", "SF", 8, 150),
]
PLAYERS = {p.player_id: p for p in POOL}
POINTS = {"qb1": 380, "qb2": 350, "rb1": 300, "rb2": 290, "rb3": 250, "rb4": 240, "wr1": 320, "wr2": 300,
          "wr3": 280, "wr4": 260, "te1": 200, "te2": 170, "k1": 150, "SF": 130}
PROJ = {pid: _proj(PLAYERS[pid], POINTS[pid]) for pid in PLAYERS}


def _league() -> LeagueSettings:
    return LeagueSettings("L1", "Test League", 2026, TEAMS, list(SLOTS), {"rec": 0.5, "pass_td": 4}, draft_id="D1")


def _draft() -> DraftSettings:
    return DraftSettings("D1", "L1", "snake", "drafting", TEAMS, ROUNDS, pick_timer=30,
                         draft_order={"u1": 1, "u2": 2, "u3": 3, "u4": 4},
                         slot_to_roster_id={1: 1, 2: 2, 3: 3, 4: 4}, metadata={"name": "Test League"})


def _managers() -> dict[str, Manager]:
    return {f"u{i}": Manager(f"u{i}", f"User {i}", team_name=f"Team {i}", slot=i, roster_id=i) for i in range(1, 5)}


def _pick(no: int, pid: str) -> Pick:
    d = _draft()
    pl = PLAYERS[pid]
    first, _, last = pl.name.partition(" ")
    return Pick(no, d.round_of(no), d.slot_for_pick(no), pid, roster_id=d.slot_for_pick(no),
                metadata={"first_name": first, "last_name": last, "position": pl.position, "team": pl.team})


def make_state(n_picks: int, my_slot: int = 2) -> DraftState:
    """Snake 4 teams; picks in ADP order."""
    order = sorted(PLAYERS, key=lambda pid: PLAYERS[pid].adp)
    picks = [_pick(i + 1, order[i]) for i in range(n_picks)]
    return DraftState(_draft(), picks, _league(), _managers(), my_user_id=f"u{my_slot}", my_slot=my_slot, version=n_picks)


def _value(pid: str, score: float, avail: float, reasons: list[str] | None = None, tier: int = 1) -> PlayerValue:
    pl = PLAYERS[pid]
    return PlayerValue(pl, PROJ[pid], vorp=score * 0.8, vona=score * 0.3, marginal_value=score, score=score,
                       availability_next=avail, availability_after_next=avail * 0.6, tier=tier, pos_rank=1,
                       overall_rank=1, reasons=reasons or [f"Proj {POINTS[pid]} pts, +{score:.0f} over replacement {pl.position}"],
                       warnings=[])


def make_rec(state: DraftState) -> Recommendation:
    drafted = state.drafted_ids
    avail = [pid for pid in sorted(PLAYERS, key=lambda p: -POINTS[p]) if pid not in drafted]
    best = [_value(pid, POINTS[pid] - 100, 0.3 + 0.1 * i) for i, pid in enumerate(avail[:6])]
    by_pos = {}
    actions = {"QB": "WAIT", "RB": "TAKE NOW", "WR": "SOON", "TE": "WAIT", "K": "SKIP", "DEF": "SKIP"}
    for pos in ("QB", "RB", "WR", "TE", "K", "DEF"):
        cands = [_value(pid, POINTS[pid] - 100, 0.4) for pid in avail if PLAYERS[pid].position == pos][:3]
        by_pos[pos] = PositionAdvice(pos, cands, actions[pos],
                                     f"40% chance {cands[0].player.name if cands else 'nobody'} is there at your next pick (#7)",
                                     expected_next_available=100.0, drop_off=20.0)
    my = RosterSummary(state.my_slot or 1, "Team 2", [PLAYERS[p.player_id] for p in state.my_picks()],
                       starters_filled={"QB": 0, "RB": 1, "WR": 0, "TE": 0, "FLEX": 0, "K": 0, "DEF": 0},
                       open_starters={"QB": 1, "RB": 1, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "DEF": 1},
                       position_counts={"RB": 1}, bye_weeks={6: 2}, lineup_points=290.0, bench_points=0.0)
    opps = [RosterSummary(s, f"Team {s}", [], {}, {"QB": 1, "RB": 2, "WR": 1}, {}, {}) for s in (1, 3, 4)]
    return Recommendation(state.version, time.time(), 4.2, best, by_pos, my, opps, {"RB": 2.5, "WR": 1.0},
                          notes=["RB run: 3 of last 4 picks", "You pick next at #7"])


def _console(width: int = 160, height: int = 45) -> Console:
    return Console(record=True, width=width, height=height, file=io.StringIO(), force_terminal=False, color_system=None)


def _render(width: int = 160, height: int = 45, n_picks: int = 5, status: dict | None = None):
    console = _console(width, height)
    state = make_state(n_picks)
    rec = make_rec(state)
    db = Dashboard(console, projections=PROJ)
    db.update(state, rec, PLAYERS, status or {"latency_ms": 120, "claude": "off"})
    return state, rec, console.export_text()


# ---------------------------------------------------------------------------


def test_state_fixture_sanity():
    st = make_state(5)
    assert st.next_pick_no == 6
    assert st.my_slot == 2 and st.my_next_pick_no == 7 and not st.is_my_turn
    assert make_state(6).is_my_turn


def test_dashboard_key_strings_160x45():
    state, rec, text = _render()
    assert "Best picks" in text
    assert "My roster" in text and "Recent picks" in text and "Opponent needs" in text
    assert "TAKE NOW" in text and "WAIT" in text and "SOON" in text and "SKIP" in text
    # roster slot labels and a drafted player of mine
    for slot in ("QB", "RB", "WR", "TE", "FLX", "K", "DEF", "BN"):
        assert slot in text
    mine = [PLAYERS[p.player_id].name for p in state.my_picks()]
    assert mine and all(name.split()[0] in text for name in mine)
    # candidates
    assert rec.best_overall[0].player.name in text
    assert "You pick in 1 (#7)" in text
    assert "Half PPR" in text and "Test League" in text
    assert "Claude: off" in text and "poll 120 ms" in text
    assert "Need:" in text and "Bye clash" in text
    assert "RB run" in text
    assert "YOUR PICK" not in text


def test_dashboard_your_pick_and_countdown():
    status = {"turn_started_at": time.time() - 5, "claude": "thinking"}
    state, rec, text = _render(n_picks=6, status=status)
    assert state.is_my_turn
    assert "YOUR PICK" in text
    assert "s left" in text          # countdown from pick_timer 30
    assert "Claude: thinking" in text


def test_dashboard_compact_drops_opponent_needs():
    _, _, text = _render(width=120, height=40)
    assert "Opponent needs" not in text
    assert "Recent picks" in text and "Best picks" in text and "My roster" in text
    lines = text.splitlines()
    assert max(len(l) for l in lines) <= 120


def test_dashboard_claude_advice_in_footer():
    console = _console()
    state = make_state(6)
    rec = make_rec(state)
    rec.claude_advice = "Take Bijan Robinson; RBs are flying off the board. Fallback: Gibbs."
    Dashboard(console, projections=PROJ).update(state, rec, PLAYERS, {"claude": "ready"})
    text = console.export_text()
    assert "Take Bijan Robinson" in text


def test_dashboard_handles_missing_rec_and_unknown_players():
    console = _console()
    state = make_state(3)
    state.picks.append(Pick(4, 1, 4, "zzz", metadata={"first_name": "Mystery", "last_name": "Man", "position": "WR"}))
    Dashboard(console).update(state, None, PLAYERS, {})
    text = console.export_text()
    assert "computing" in text and "Mystery Man" in text


def test_dashboard_complete_draft():
    console = _console()
    state = make_state(TEAMS * ROUNDS)
    assert state.is_complete
    Dashboard(console, projections=PROJ).update(state, make_rec(state), PLAYERS, {})
    assert "Draft complete" in console.export_text()


def test_build_is_fast():
    console = _console()
    state = make_state(5)
    rec = make_rec(state)
    db = Dashboard(console, projections=PROJ)
    db.build(state, rec, PLAYERS, {})
    t0 = time.perf_counter()
    for _ in range(20):
        db.build(state, rec, PLAYERS, {})
    per = (time.perf_counter() - t0) / 20 * 1000
    assert per < 40, f"build took {per:.1f} ms"


def test_live_start_stop_and_refresh():
    console = _console()
    db = Dashboard(console, refresh_per_second=4, projections=PROJ)
    db.start()
    assert db.running
    state = make_state(6)
    db.update(state, make_rec(state), PLAYERS, {"turn_started_at": time.time()})
    db.refresh({"claude": "ready"})
    db.stop()
    db.stop()
    assert not db.running


def test_render_text_plain():
    state = make_state(6)
    rec = make_rec(state)
    rec.claude_advice = "Go Bijan."
    text = render_text(rec, state, PLAYERS)
    assert "YOUR PICK" in text
    assert "Best picks now" in text and rec.best_overall[0].player.name in text
    assert "TAKE NOW" in text and "WAIT" in text
    assert "Need:" in text and "Claude: Go Bijan." in text
    assert "RB run" in text
    assert "Last picks" in text
    assert "(no recommendation yet)" in render_text(None, state, PLAYERS)


def test_scoring_description_variants():
    st = make_state(0)
    assert scoring_description(st) == "Half PPR"
    st.league.scoring_settings["rec"] = 1.0
    st.league.roster_positions.append("SUPER_FLEX")
    st.league.scoring_settings["bonus_rec_te"] = 0.5
    assert scoring_description(st) == "PPR • Superflex • TE +0.5"
    st.league = None
    st.draft.scoring_type = "std"
    assert scoring_description(st) == "Standard"


def test_assign_roster_slots_fills_dedicated_then_flex_then_bench():
    roster = [(PLAYERS["rb1"], 300.0), (PLAYERS["rb2"], 290.0), (PLAYERS["rb3"], 250.0), (PLAYERS["wr1"], 320.0),
              (PLAYERS["k1"], 150.0)]
    rows = assign_roster_slots(roster, SLOTS)
    by_slot = {}
    for slot, pl, _ in rows:
        by_slot.setdefault(slot, []).append(pl.player_id if pl else None)
    assert set(by_slot["RB"]) == {"rb1", "rb2"}
    assert by_slot["FLEX"] == ["rb3"]
    assert by_slot["WR"] == ["wr1", None]
    assert by_slot["K"] == ["k1"]
    assert by_slot["QB"] == [None]
    assert "BN" not in by_slot or all(x is None for x in by_slot["BN"])
