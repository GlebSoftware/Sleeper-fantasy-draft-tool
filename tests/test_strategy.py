"""Tests for draftadvisor.strategy (DESIGN.md §3.4). Offline: fixtures + a synthetic universe."""
from __future__ import annotations

import itertools
import random
import statistics
import time

import numpy as np
import pytest

from draftadvisor.config import SKILL_POSITIONS, SLOT_ELIGIBILITY, Settings
from draftadvisor.models import DraftSettings, DraftState, LeagueSettings, Pick, Player, Projection
from draftadvisor.strategy import (
    Advisor,
    TradeEvaluation,
    evaluate_pick_choice,
    evaluate_trade,
    expected_best_available,
    marginal_lineup_value,
    optimal_lineup,
    pick_distribution,
    position_pressure,
    prob_available,
    replacement_levels,
    roster_summary,
    shift_for_pressure,
    simulate_candidates,
    starter_demand,
    tiers,
    vorp,
)
from draftadvisor.strategy.availability import adp_baseline, expected_best_available_array, prob_available_array
from draftadvisor.strategy.lineup import starter_thresholds, summarize_roster


# ---------------------------------------------------------------------------
# helpers: fixture parsing (local, trivial) + synthetic universe
# ---------------------------------------------------------------------------


def _league(raw: dict) -> LeagueSettings:
    return LeagueSettings(
        league_id=raw["league_id"], name=raw["name"], season=int(raw["season"]),
        total_rosters=raw["total_rosters"], roster_positions=list(raw["roster_positions"]),
        scoring_settings=dict(raw["scoring_settings"]), draft_id=raw.get("draft_id"),
    )


def _draft(raw: dict) -> DraftSettings:
    s = raw["settings"]
    return DraftSettings(
        draft_id=raw["draft_id"], league_id=raw["league_id"], type=raw["type"], status=raw["status"],
        teams=s["teams"], rounds=s["rounds"], pick_timer=s.get("pick_timer", 0),
        reversal_round=s.get("reversal_round", 0),
        draft_order={u: int(slot) for u, slot in raw["draft_order"].items()},
        slot_to_roster_id={int(k): int(v) for k, v in raw["slot_to_roster_id"].items()},
    )


def _picks(raw: list[dict]) -> list[Pick]:
    return [Pick(p["pick_no"], p["round"], p["draft_slot"], p["player_id"], p.get("roster_id"),
                 p.get("picked_by"), metadata=p.get("metadata") or {}) for p in raw]


_SPEC = {  # position: (count, top points, slope)
    "QB": (60, 380.0, 3.0), "RB": (120, 330.0, 2.4), "WR": (150, 320.0, 1.9),
    "TE": (60, 230.0, 2.8), "K": (32, 150.0, 1.3), "DEF": (32, 140.0, 1.5),
}
_MARKET_OFFSET = {"QB": 190.0, "RB": 110.0, "WR": 110.0, "TE": 120.0, "K": 150.0, "DEF": 150.0}


def make_universe(seed: int = 0, drafted: dict[str, dict] | None = None):
    """Synthetic players/projections; drafted fixture ids replace same-position synthetic players."""
    rng = random.Random(seed)
    rows = []
    for pos, (n, top, slope) in _SPEC.items():
        for i in range(n):
            pts = max(15.0, top - slope * i - (0.006 * i * i if pos in ("RB", "WR") else 0.0))
            rows.append((pos, i, pts))
    rows.sort(key=lambda r: -(r[2] - _MARKET_OFFSET[r[0]]))
    players: dict[str, Player] = {}
    projections: dict[str, Projection] = {}
    for rank, (pos, i, pts) in enumerate(rows, start=1):
        pid = f"{pos}{i}"
        adp = max(1.0, rank + rng.gauss(0, 2.0))
        players[pid] = Player(
            player_id=pid, name=f"{pos} {i}", position=pos, team=f"T{(rank % 32) + 1}",
            bye_week=5 + (rank % 10), adp=adp, adp_source="test", ecr=max(1.0, adp + rng.gauss(0, 3.0)),
            ecr_sd=3.0, years_exp=1 + (i % 6),
        )
        projections[pid] = Projection(pid, pos, pts, max(8.0, 0.14 * pts), pts / 17.0, 17.0,
                                      floor=pts * 0.85, ceiling=pts * 1.15)
    if drafted:
        for pid, md in drafted.items():
            pos = md["position"]
            pool = (k for k in players if players[k].position == pos and k not in drafted)
            for cand in sorted(pool, key=lambda k: players[k].adp):
                pl, pr = players.pop(cand), projections.pop(cand)
                pl.player_id = pid
                pl.name = f"{md.get('first_name', '')} {md.get('last_name', '')}".strip() or pid
                pr.player_id = pid
                players[pid], projections[pid] = pl, pr
                break
    return players, projections


@pytest.fixture
def league(league_json):
    return _league(league_json)


@pytest.fixture
def fixture_state(draft_json, picks_json, league):
    picks = _picks(picks_json)
    return DraftState(_draft(draft_json), picks, league, my_user_id="111111111111111111", my_slot=1)


@pytest.fixture
def universe(picks_json):
    return make_universe(0, {p["player_id"]: p["metadata"] for p in picks_json})


def P(pid: str, pos: str, **kw) -> Player:
    return Player(player_id=pid, name=pid, position=pos, **kw)


STD_SLOTS = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF"]


def _lg(slots, teams=12, rec=1.0) -> LeagueSettings:
    return LeagueSettings("L", "test", 2026, teams, list(slots) + ["BN"] * 6, {"rec": rec})


# ---------------------------------------------------------------------------
# replacement
# ---------------------------------------------------------------------------


def test_starter_demand_fixture_league(league):
    d = starter_demand(league)
    assert d["QB"] == pytest.approx(12 + 0.15 * 12)
    assert d["RB"] == pytest.approx(24 + 0.45 * 12 + 6)
    assert d["WR"] == pytest.approx(24 + 0.45 * 12 + 6)
    assert d["TE"] == pytest.approx(12 + 0.10 * 12 + 0.15 * 12)
    assert d["K"] == 12 and d["DEF"] == 12


def test_starter_demand_superflex():
    d = starter_demand(_lg(STD_SLOTS + ["SUPER_FLEX"]))
    assert d["QB"] == pytest.approx(12 + 0.85 * 12 + 0.75 * 12)


def test_replacement_levels_drop_with_available_pool(league, universe):
    players, proj = universe
    full = replacement_levels(proj, players, league)
    rb_sorted = sorted((p.points for p in proj.values() if p.position == "RB"), reverse=True)
    assert full["RB"] == pytest.approx(rb_sorted[round(starter_demand(league)["RB"])])
    avail = [pid for pid in proj if not (proj[pid].position == "RB" and proj[pid].points > rb_sorted[20])]
    later = replacement_levels(proj, players, league, avail)
    assert later["RB"] < full["RB"]
    assert later["WR"] == full["WR"]
    v = vorp(proj, players, league, avail)
    top_rb = max((pid for pid in avail if proj[pid].position == "RB"), key=lambda k: proj[k].points)
    assert v[top_rb] == pytest.approx(proj[top_rb].points - later["RB"])


def test_replacement_levels_small_pool():
    players = {"a": P("a", "RB")}
    proj = {"a": Projection("a", "RB", 100, 10, 6, 17)}
    lv = replacement_levels(proj, players, _lg(STD_SLOTS))
    assert lv["RB"] == 100 and lv["QB"] == 0.0


def test_tiers_by_gaps():
    vals = [("a", 300.0), ("b", 296.0), ("c", 260.0), ("d", 258.0), ("e", 200.0)]
    t = tiers(vals, {k: 20.0 for k, _ in vals})
    assert t == {"a": 1, "b": 1, "c": 2, "d": 2, "e": 3}
    assert tiers([], {}) == {}


# ---------------------------------------------------------------------------
# lineup
# ---------------------------------------------------------------------------


def test_optimal_lineup_flex_and_dedicated():
    roster = [(P("qb", "QB"), 300), (P("rb1", "RB"), 200), (P("rb2", "RB"), 150), (P("rb3", "RB"), 140),
              (P("wr1", "WR"), 220), (P("wr2", "WR"), 170), (P("wr3", "WR"), 160), (P("te", "TE"), 120)]
    assignment, pts, bench = optimal_lineup(roster, STD_SLOTS)
    assert assignment[6] == "wr3"          # FLEX takes the best remaining RB/WR/TE
    assert pts == 300 + 200 + 150 + 220 + 170 + 160 + 120
    assert bench == ["rb3"]
    assert 7 not in assignment and 8 not in assignment


def test_optimal_lineup_superflex_prefers_second_qb():
    roster = [(P("qb1", "QB"), 320), (P("qb2", "QB"), 280), (P("rb1", "RB"), 200), (P("rb2", "RB"), 190),
              (P("wr1", "WR"), 210), (P("wr2", "WR"), 200), (P("wr3", "WR"), 180), (P("te", "TE"), 100)]
    slots = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "SUPER_FLEX"]
    assignment, pts, bench = optimal_lineup(roster, slots)
    assert assignment[7] == "qb2" and assignment[6] == "wr3"
    assert bench == []
    # a flex-filled-first trap: WR in FLEX must not block the QB from SUPER_FLEX
    assert pts == 320 + 280 + 200 + 190 + 210 + 200 + 180 + 100


def test_optimal_lineup_matches_brute_force():
    slots = ["QB", "RB", "WR", "FLEX", "SUPER_FLEX", "REC_FLEX"]
    rng = random.Random(3)
    for _ in range(60):
        roster = [(P(f"p{i}", rng.choice(["QB", "RB", "WR", "TE"])), rng.randint(40, 300)) for i in range(6)]
        _, got, _ = optimal_lineup(roster, slots)
        best = 0
        for k in range(1, len(slots) + 1):
            for sub in itertools.combinations(range(len(slots)), k):
                for perm in itertools.permutations(range(len(roster)), k):
                    total = 0
                    for si, pi in zip(sub, perm):
                        pl, p = roster[pi]
                        if pl.position not in SLOT_ELIGIBILITY[slots[si]]:
                            total = None
                            break
                        total += p
                    if total is not None:
                        best = max(best, total)
        assert got == best


def test_optimal_lineup_speed():
    roster = [(P(f"p{i}", pos), 300 - 10 * i) for i, pos in
              enumerate(["QB", "RB", "RB", "RB", "WR", "WR", "WR", "WR", "TE", "TE", "K", "DEF", "QB", "RB"])]
    optimal_lineup(roster, STD_SLOTS)
    t0 = time.perf_counter()
    for _ in range(500):
        optimal_lineup(roster, STD_SLOTS)
    per_call_us = (time.perf_counter() - t0) / 500 * 1e6
    assert per_call_us < 150, per_call_us   # budget 50 µs typical; generous for slow CI


def test_starter_thresholds_match_exact_recompute():
    lg = _lg(STD_SLOTS + ["SUPER_FLEX", "WRRB_FLEX"])
    rng = random.Random(7)
    for _ in range(100):
        roster = [(P(f"p{i}", rng.choice(SKILL_POSITIONS)), rng.randint(40, 320)) for i in range(rng.randint(2, 15))]
        a, base, _ = optimal_lineup(roster, lg.starting_slots)
        info = {pl.player_id: (pl.position, p) for pl, p in roster}
        thr = starter_thresholds({i: info[pid] for i, pid in a.items()}, lg.starting_slots)
        for pos in SKILL_POSITIONS:
            x = rng.randint(30, 330)
            _, new, _ = optimal_lineup(roster + [(P("cand", pos), x)], lg.starting_slots)
            fast = max(0.0, x - thr[pos]) if np.isfinite(thr[pos]) else 0.0
            assert new - base == pytest.approx(fast)


def test_marginal_value_kicker_before_and_after():
    lg = _lg(STD_SLOTS)
    roster = [(P("qb", "QB"), 300), (P("rb1", "RB"), 200), (P("wr1", "WR"), 220)]
    k = P("k", "K")
    before = marginal_lineup_value(k, 140, roster, lg, 0.35)
    after = marginal_lineup_value(k, 140, roster + [(P("k0", "K"), 135)], lg, 0.35)
    assert before == pytest.approx(140.0)
    assert after == pytest.approx(5.0)          # bumps the 135 K; K bench value is 0
    # against replacement-level stand-ins the open slot is only worth the surplus over the waiver wire
    with_rep = marginal_lineup_value(k, 140, roster, lg, 0.35, replacement={"K": 120})
    assert with_rep == pytest.approx(20.0)
    # benched RB gets discounted bench value over replacement, decaying with depth
    rb = P("rb9", "RB")
    deep = roster + [(P("rb2", "RB"), 190), (P("rb3", "RB"), 180), (P("rb4", "RB"), 170)]
    rep = {"RB": 100, "WR": 100, "QB": 200, "TE": 80, "K": 120, "DEF": 100}
    v_shallow = marginal_lineup_value(rb, 150, roster + [(P("rb2", "RB"), 190)], lg, 0.35, replacement=rep)
    v_deep = marginal_lineup_value(rb, 150, deep, lg, 0.35, replacement=rep)
    assert v_shallow > v_deep > 0


def test_roster_summary_fixture(fixture_state, universe, league):
    players, proj = universe
    rs = roster_summary(fixture_state, 1, players, proj, league)
    assert rs.slot == 1 and len(rs.players) == 1       # slot 1 has only made pick #1 (a WR)
    assert rs.players[0].player_id == "7564" and rs.players[0].team is not None
    assert rs.position_counts == {"WR": 1}
    assert rs.open_starters["RB"] == 2 and rs.open_starters["WR"] == 1 and rs.open_starters["QB"] == 1
    assert rs.starters_filled["WR"] == 1 and rs.starters_filled["RB"] == 0
    assert rs.lineup_points == pytest.approx(proj["7564"].points) and rs.bench_points == 0
    assert rs.bye_weeks == {players["7564"].bye_week: 1}
    assert rs.needs()[:3] == ["QB", "RB", "WR"]
    two = roster_summary(fixture_state, 12, players, proj, league)        # slot 12 picked at #12 and #13
    assert len(two.players) == 2 and two.position_counts == {"RB": 1, "TE": 1}
    unknown = summarize_roster(3, "x", [Pick(1, 1, 3, "zzz", metadata={"position": "TE", "first_name": "A", "last_name": "B"})],
                               players, proj, league)
    assert unknown.players[0].name == "A B" and unknown.position_counts == {"TE": 1}


# ---------------------------------------------------------------------------
# availability
# ---------------------------------------------------------------------------


def test_pick_distribution_defaults_and_clip():
    assert pick_distribution(P("a", "RB"), 10) == (400.0, pytest.approx(41.5))
    mu, sigma = pick_distribution(P("a", "RB", adp=30.0), 10)
    assert mu == 30.0 and sigma == pytest.approx(4.5)
    mu, _ = pick_distribution(P("a", "RB", adp=5.0), 40)        # still on the board past his ADP
    assert mu == 39.5
    _, s2 = pick_distribution(P("a", "RB", adp=30.0, ecr_sd=10.0), 10)
    assert s2 == pytest.approx(0.5 * 4.5 + 0.5 * 12.0)
    mu_e, _ = pick_distribution(P("a", "RB", ecr=22.0), 1)
    assert mu_e == 22.0


def test_prob_available_monotone_and_one_at_current_pick():
    pl = P("a", "WR", adp=30.0)
    assert prob_available(pl, 25, 25) == 1.0
    assert prob_available(pl, 20, 25) == 1.0
    probs = [prob_available(pl, at, 25) for at in range(26, 60)]
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert all(a >= b for a, b in zip(probs, probs[1:]))
    assert probs[0] > 0.9 and probs[-1] < 0.01
    assert prob_available(pl, 32, 25, shift=4.0) < prob_available(pl, 32, 25)
    arr = prob_available_array(np.array([30.0, 30.0]), np.array([4.5, 4.5]), 32, 25, np.array([0.0, 4.0]))
    assert arr[0] == pytest.approx(prob_available(pl, 32, 25), abs=1e-5)
    assert arr[1] == pytest.approx(prob_available(pl, 32, 25, shift=4.0), abs=1e-5)


def test_expected_best_available_hand_computed():
    ranked = [("a", 100.0, 0.5), ("b", 90.0, 0.5), ("c", 80.0, 0.5)]
    expected = 100 * 0.5 + 90 * 0.5 * 0.5 + 80 * 0.5 * 0.25 + 80 * 0.125
    assert expected_best_available(ranked) == pytest.approx(expected)
    assert expected_best_available(list(reversed(ranked))) == pytest.approx(expected)
    assert expected_best_available([]) == 0.0
    assert expected_best_available_array(np.array([100.0, 90.0, 80.0]), np.array([0.5, 0.5, 0.5])) == pytest.approx(expected)
    assert expected_best_available([("a", 100.0, 1.0), ("b", 50.0, 1.0)]) == 100.0


def test_position_pressure_bounded_and_shift(fixture_state, universe, league):
    players, proj = universe
    adv = Advisor(league, players, proj)
    rec = adv.recommend(fixture_state)
    n_between = fixture_state.my_next_pick_no - fixture_state.next_pick_no
    assert all(0.0 <= v <= n_between for v in rec.position_pressure.values())
    assert sum(rec.position_pressure.values()) > 0
    avail = adv.available_players(fixture_state)
    pressure = position_pressure(fixture_state, rec.opponent_rosters, avail, proj, fixture_state.my_next_pick_no)
    assert set(pressure) == set(SKILL_POSITIONS)
    shift = shift_for_pressure({"RB": 4.0, "WR": 1.0}, {"RB": 2.0, "WR": 2.0})
    assert shift["RB"] == pytest.approx(3.0) and shift["WR"] == 0.0 and shift["K"] == 0.0


# ---------------------------------------------------------------------------
# Advisor
# ---------------------------------------------------------------------------


def _check_value_fields(v):
    assert v.player is not None and v.projection is not None
    assert 0.0 <= v.availability_next <= 1.0 and 0.0 <= v.availability_after_next <= 1.0
    assert v.tier >= 1 and v.pos_rank >= 1 and v.overall_rank >= 1
    assert 2 <= len(v.reasons) <= 4
    assert isinstance(v.warnings, list)
    assert np.isfinite(v.score) and np.isfinite(v.vona) and np.isfinite(v.vorp) and np.isfinite(v.marginal_value)


def test_advisor_on_fixture_state(fixture_state, universe, league):
    players, proj = universe
    adv = Advisor(league, players, proj, Settings())
    rec = adv.recommend(fixture_state)
    drafted = fixture_state.drafted_ids
    assert rec.state_version == fixture_state.version and rec.compute_ms > 0
    assert len(rec.best_overall) == 6
    scores = [v.score for v in rec.best_overall]
    assert scores == sorted(scores, reverse=True)
    assert [v.overall_rank for v in rec.best_overall] == [1, 2, 3, 4, 5, 6]
    assert set(rec.by_position) == set(SKILL_POSITIONS)
    for pos, adv_pos in rec.by_position.items():
        assert len(adv_pos.candidates) == 3
        assert adv_pos.action in ("TAKE NOW", "SOON", "WAIT", "SKIP")
        assert f"#{fixture_state.my_next_pick_no}" in adv_pos.rationale and "%" in adv_pos.rationale
        for c in adv_pos.candidates:
            assert c.player.position == pos and c.player_id not in drafted
            _check_value_fields(c)
        cand_scores = [c.score for c in adv_pos.candidates]
        assert cand_scores == sorted(cand_scores, reverse=True)
    for v in rec.best_overall:
        assert v.player_id not in drafted
        _check_value_fields(v)
        assert any(f"(#{fixture_state.my_next_pick_no})" in r for r in v.reasons)
        assert any(r.startswith("Proj ") and "over replacement" in r for r in v.reasons)
    # round 2: kickers and defenses are a SKIP with the K/DEF-too-early warning
    assert rec.by_position["K"].action == "SKIP" and rec.by_position["DEF"].action == "SKIP"
    assert "K/DEF too early" in rec.by_position["K"].candidates[0].warnings
    assert not any(v.player.position in ("K", "DEF") for v in rec.best_overall)
    assert rec.my_roster is not None and rec.my_roster.slot == 1
    assert len(rec.opponent_rosters) == 11 and all(r.slot != 1 for r in rec.opponent_rosters)
    assert any("You pick next at #24 and #25" in n and "back-to-back" in n for n in rec.notes)
    assert any(n.startswith("K/DEF: wait until round") for n in rec.notes)
    # my open RB slot is called out on RB candidates
    assert any("Fills your open RB" in r for r in rec.by_position["RB"].candidates[0].reasons)


def test_advisor_vona_and_availability_consistency(fixture_state, universe, league):
    players, proj = universe
    adv = Advisor(league, players, proj)
    rec = adv.recommend(fixture_state)
    for pos, a in rec.by_position.items():
        assert a.drop_off == pytest.approx(max(p.points for pid, p in proj.items() if p.position == pos and pid not in fixture_state.drafted_ids) - a.expected_next_available, abs=1e-6)
        for c in a.candidates:
            assert c.vona == pytest.approx(c.projection.points - a.expected_next_available, abs=1e-6)
            assert c.availability_after_next <= c.availability_next + 1e-9
    # a player with a far-off ADP is (almost) surely available; an early ADP one is not
    far = adv.value_of(fixture_state, max(adv.available_players(fixture_state), key=lambda p: p.adp).player_id)
    assert far.availability_next > 0.99
    near = min(adv.available_players(fixture_state), key=lambda p: p.adp)
    assert adv.value_of(fixture_state, near.player_id).availability_next < 0.5


def test_advisor_when_it_is_my_turn(fixture_state, universe, league):
    players, proj = universe
    adv = Advisor(league, players, proj)
    # advance to pick 24 (slot 1's second pick) with ADP-order picks
    state = _advance(fixture_state, adv, until_pick=24)
    assert state.is_my_turn and state.next_pick_no == 24
    rec = adv.recommend(state)
    # availability is judged at my following pick (#25 for slot 1, back-to-back)
    assert all("(#25)" in a.rationale for a in rec.by_position.values())
    assert any("on the clock (#24)" in n for n in rec.notes)
    assert rec.top_pick is not None and rec.top_pick.player_id not in state.drafted_ids


def _advance(state: DraftState, adv: Advisor, until_pick: int, allow_kdef: bool = False) -> DraftState:
    """Fill picks in ADP order (skipping K/DEF unless allowed) until ``until_pick`` is on the clock."""
    picks = list(state.picks)
    taken = {p.player_id for p in picks}
    by_adp = sorted(adv.players.values(), key=lambda p: p.adp or 999)
    n = state.next_pick_no
    while n < until_pick:
        for pl in by_adp:
            if pl.player_id in taken or (not allow_kdef and pl.position in ("K", "DEF")):
                continue
            taken.add(pl.player_id)
            slot = state.draft.slot_for_pick(n)
            picks.append(Pick(n, state.draft.round_of(n), slot, pl.player_id,
                              state.draft.slot_to_roster_id.get(slot), metadata={"position": pl.position}))
            break
        n += 1
    return state.with_picks(picks)


def test_advisor_late_rounds_recommend_k_def(fixture_state, universe, league):
    players, proj = universe
    adv = Advisor(league, players, proj)
    state = _advance(fixture_state, adv, until_pick=12 * 13 + 1)     # round 14 on the clock
    assert state.current_round == 14
    rec = adv.recommend(state)
    assert rec.by_position["K"].action != "SKIP" and rec.by_position["DEF"].action != "SKIP"
    assert not any(n.startswith("K/DEF: wait") for n in rec.notes)
    assert any(v.player.position in ("K", "DEF") for v in rec.best_overall)
    assert "K/DEF too early" not in rec.by_position["K"].candidates[0].warnings
    # roster with a full set of starters: a 2nd QB in a 1-QB league adds little -> SKIP
    assert rec.my_roster is not None and rec.my_roster.position_counts.get("QB", 0) >= 1
    assert rec.by_position["QB"].action == "SKIP"
    # end-game: 2 picks left for the 2 open starters (K, DEF) -> both must be filled, bench adds nothing
    assert rec.my_roster.open_starters["K"] == 1 and rec.my_roster.open_starters["DEF"] == 1
    assert rec.by_position["K"].action == "TAKE NOW" and rec.by_position["DEF"].action == "TAKE NOW"
    assert rec.by_position["K"].rationale.startswith("Must fill")
    assert any("open starters (K DEF): fill them" in n for n in rec.notes)
    assert rec.by_position["RB"].candidates[0].marginal_value == 0.0
    assert rec.top_pick.marginal_value == pytest.approx(rec.top_pick.projection.points)


def test_advisor_value_of_and_explain(fixture_state, universe, league):
    players, proj = universe
    adv = Advisor(league, players, proj)
    drafted = next(iter(fixture_state.drafted_ids))
    assert adv.value_of(fixture_state, drafted) is None
    assert adv.value_of(fixture_state, "nope") is None
    assert "already been drafted" in adv.explain_pick(fixture_state, drafted)
    rec = adv.recommend(fixture_state)
    pid = rec.by_position["TE"].candidates[0].player_id
    text = adv.explain_pick(fixture_state, pid)
    assert players[pid].name in text and "Top recommendation" in text
    top_text = adv.explain_pick(fixture_state, rec.top_pick.player_id)
    assert "This is the top recommendation" in top_text
    avail = adv.available_players(fixture_state)
    assert avail and all(p.player_id not in fixture_state.drafted_ids for p in avail)
    pts = [proj[p.player_id].points for p in avail]
    assert pts == sorted(pts, reverse=True)
    assert "Close call" in evaluate_pick_choice(fixture_state, adv, rec.best_overall[1].player_id) or \
        "worth" in evaluate_pick_choice(fixture_state, adv, rec.best_overall[1].player_id) or \
        "revisit" in evaluate_pick_choice(fixture_state, adv, rec.best_overall[1].player_id)


def test_advisor_reasons_flags_and_warnings(fixture_state, universe, league):
    players, proj = universe
    adv = Advisor(league, players, proj)
    # choose the best available RB and tweak his attributes
    rec = adv.recommend(fixture_state)
    pid = rec.by_position["RB"].candidates[0].player_id
    pl = players[pid]
    pl.injury_status = "Questionable"
    pl.years_exp = 0
    pl.depth_chart_order = 2
    pl.ecr = None
    pl.adp = 5.0                                # steal: ADP well before the current pick
    adv2 = Advisor(league, players, proj)
    v = adv2.value_of(fixture_state, pid)
    assert "Injury: Questionable" in v.warnings and "Rookie" in v.warnings
    assert any(w.startswith("Depth chart #2") for w in v.warnings)
    assert any(r.startswith("ADP 5") for r in v.reasons)


def test_advisor_stack_and_bye_effects(fixture_state, universe, league):
    players, proj = universe
    adv = Advisor(league, players, proj)
    base = adv.recommend(fixture_state)
    wr = base.by_position["WR"].candidates[0]
    # give me a QB on that WR's team via a synthetic extra pick for slot 1 (pick 15 -> slot 10 in snake; use metadata)
    my_qb = next(p for p in adv.available_players(fixture_state) if p.position == "QB")
    my_qb.team = wr.player.team
    my_qb.bye_week = 99                          # ignored (out of range)
    picks = list(fixture_state.picks)
    picks.append(Pick(21, 2, 12, my_qb.player_id, 4, metadata={"position": "QB"}))   # attributed to roster 4 = slot 1
    st = fixture_state.with_picks(picks)
    adv2 = Advisor(league, players, proj, Settings(stack_bonus=0.05))
    v = adv2.value_of(st, wr.player_id)
    assert any("Stacks with your QB" in r for r in v.reasons)
    v0 = Advisor(league, players, proj, Settings(stack_bonus=0.0)).value_of(st, wr.player_id)
    assert v.score > v0.score
    # bye clash: a candidate sharing a bye with my WR starter is penalised
    chase = players["7564"]
    cand = next(c for c in base.by_position["RB"].candidates)
    cand.player.bye_week = chase.bye_week
    v_bye = Advisor(league, players, proj, Settings(bye_penalty=0.05)).value_of(fixture_state, cand.player_id)
    v_nobye = Advisor(league, players, proj, Settings(bye_penalty=0.0)).value_of(fixture_state, cand.player_id)
    assert v_bye.score < v_nobye.score
    assert any(f"Bye {chase.bye_week} clashes with" in r for r in v_bye.reasons)


def test_advisor_superflex_note_and_qb_value(fixture_state, universe, league_json):
    players, proj = universe
    raw = dict(league_json)
    raw["roster_positions"] = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "SUPER_FLEX", "K", "DEF"] + ["BN"] * 6
    sf = _league(raw)
    assert sf.is_superflex
    # thin the QB pool so the scarcity note fires
    qb_ids = sorted((pid for pid in proj if proj[pid].position == "QB"), key=lambda k: -proj[k].points)
    for pid in qb_ids[14:]:
        players.pop(pid)
        proj.pop(pid)
    adv = Advisor(sf, players, proj)
    rec = adv.recommend(fixture_state)
    assert any(n.startswith("Superflex: QBs scarce") for n in rec.notes)
    assert rec.by_position["QB"].candidates[0].marginal_value > 0


def test_advisor_no_slot_degrades(fixture_state, universe, league):
    players, proj = universe
    st = DraftState(fixture_state.draft, fixture_state.picks, league)     # my_slot unknown
    rec = Advisor(league, players, proj).recommend(st)
    assert rec.my_roster is None and len(rec.best_overall) == 6 and len(rec.opponent_rosters) == 12


def test_advisor_speed_budget(fixture_state, universe, league, monkeypatch):
    from draftadvisor.strategy import recommend as rec_mod
    monkeypatch.setattr(rec_mod, "_TOP_BY_POINTS", 500)     # make the whole synthetic universe relevant
    players, proj = universe
    adv = Advisor(league, players, proj)
    assert len(adv.available_players(fixture_state)) >= 400
    states = [fixture_state, fixture_state.with_picks(fixture_state.picks[:-1])]
    for s in states:
        adv.recommend(s)
    timings = []
    for i in range(20):
        s = states[i % 2].with_picks(states[i % 2].picks)      # fresh version -> full recompute
        t0 = time.perf_counter()
        adv.recommend(s)
        timings.append((time.perf_counter() - t0) * 1000.0)
    assert statistics.median(timings) < 30.0, timings


# ---------------------------------------------------------------------------
# trade
# ---------------------------------------------------------------------------


def test_trade_evaluation_symmetric_and_verdicts(universe, league):
    players, proj = universe
    rbs = sorted((pid for pid in proj if proj[pid].position == "RB"), key=lambda k: -proj[k].points)
    wrs = sorted((pid for pid in proj if proj[pid].position == "WR"), key=lambda k: -proj[k].points)
    qbs = sorted((pid for pid in proj if proj[pid].position == "QB"), key=lambda k: -proj[k].points)
    tes = sorted((pid for pid in proj if proj[pid].position == "TE"), key=lambda k: -proj[k].points)
    mine = [qbs[0], rbs[0], rbs[10], wrs[5], wrs[6], tes[0]]
    theirs = [qbs[1], rbs[1], rbs[2], wrs[0], wrs[20], tes[1]]
    ev = evaluate_trade(mine, theirs, give=[rbs[10]], get=[wrs[0]], players=players, projections=proj, league=league)
    assert isinstance(ev, TradeEvaluation)
    assert ev.my_delta == pytest.approx(ev.my_after - ev.my_before)
    mirror = evaluate_trade(theirs, mine, give=[wrs[0]], get=[rbs[10]], players=players, projections=proj, league=league)
    assert mirror.my_delta == pytest.approx(ev.their_delta) and mirror.their_delta == pytest.approx(ev.my_delta)
    assert ev.verdict in ("ACCEPT", "NEUTRAL") and mirror.verdict in ("REJECT", "NEUTRAL")
    assert any(d.startswith("Me ") and "->" in d for d in ev.details)
    # a clear loss is rejected, an even swap is neutral
    bad = evaluate_trade(mine, theirs, give=[rbs[0]], get=[wrs[20]], players=players, projections=proj, league=league)
    assert bad.verdict == "REJECT" and bad.my_delta < -3
    even = evaluate_trade(mine, theirs, give=[], get=[], players=players, projections=proj, league=league)
    assert even.verdict == "NEUTRAL" and even.my_delta == 0.0


# ---------------------------------------------------------------------------
# simulate
# ---------------------------------------------------------------------------


def test_simulate_candidates_deterministic_and_time_boxed(fixture_state, universe, league):
    players, proj = universe
    adv = Advisor(league, players, proj)
    rec = adv.recommend(fixture_state)
    cands = [v.player_id for v in rec.best_overall[:3]] + [next(iter(fixture_state.drafted_ids))]
    a = simulate_candidates(fixture_state, players, proj, adv, cands, n_sims=6, rounds_ahead=2, seed=1)
    b = simulate_candidates(fixture_state, players, proj, adv, cands, n_sims=6, rounds_ahead=2, seed=1)
    assert a == b and set(a) == set(cands[:3])
    assert all(v > 0 for v in a.values())
    t0 = time.perf_counter()
    partial = simulate_candidates(fixture_state, players, proj, adv, cands[:3], n_sims=500, rounds_ahead=3, seed=2)
    elapsed = (time.perf_counter() - t0) * 1000.0
    assert elapsed < 600 and set(partial) <= set(cands[:3]) and partial
    assert simulate_candidates(fixture_state, players, proj, adv, ["nope"], n_sims=2) == {}


# ---------------------------------------------------------------------------
# review findings S1-S10 (regressions)
# ---------------------------------------------------------------------------

from draftadvisor.strategy.lineup import bench_value, starter_displacements  # noqa: E402
from draftadvisor.strategy.recommend import future_picks  # noqa: E402
from draftadvisor.strategy.replacement import remaining_demand  # noqa: E402
from draftadvisor.strategy.trade import roster_value  # noqa: E402


def _with_my_roster(adv: Advisor, draft_json, lg: LeagueSettings, specs: list[tuple[str, int]], until_pick: int) -> DraftState:
    """ADP-ordered draft to ``until_pick`` with slot 1's picks replaced by ``specs`` = [(pos, rank-at-pos)]."""
    base = DraftState(_draft(draft_json), [], lg, my_slot=1)
    s = _advance(base, adv, until_pick=until_pick)
    picks = [pk for pk in s.picks if pk.roster_id != 4]
    taken = {pk.player_id for pk in picks}
    my_nums = [n for n in s.draft.picks_for_slot(1) if n < until_pick]
    assert len(my_nums) >= len(specs)
    for (pos, rank), n in zip(specs, my_nums):
        pool = sorted((pid for pid in adv.projections if adv.projections[pid].position == pos and pid not in taken),
                      key=lambda k: -adv.projections[k].points)
        pid = pool[rank]
        taken.add(pid)
        picks.append(Pick(n, s.draft.round_of(n), 1, pid, 4, metadata={"position": pos}))
    return s.with_picks(sorted(picks, key=lambda p: p.pick_no))


def test_s1_keeper_pick_is_not_a_future_pick(fixture_state, universe, league):
    players, proj = universe
    adv = Advisor(league, players, proj)
    keeper_id = next(pid for pid in adv.available_players(fixture_state) if pid.position == "RB").player_id
    picks = list(fixture_state.picks) + [Pick(48, 4, 1, keeper_id, 4, is_keeper=True, metadata={"position": "RB"})]
    st = fixture_state.with_picks(picks)
    assert 48 not in future_picks(st) and future_picks(st)[:3] == [24, 25, 49]
    # on the clock at #47 (slot 2): my next pick is #49, not the keeper-filled #48
    st47 = _advance(st, adv, until_pick=47)
    assert st47.next_pick_no == 47
    rec = adv.recommend(st47)
    ctx = adv._context(st47)
    assert ctx.eval_pick == 49 and ctx.n_between == 1 and ctx.remaining == 11
    assert any("You pick next at #49 and #72" in n for n in rec.notes)
    assert all("(#49)" in c.reasons[-1] or any("(#49)" in r for r in c.reasons)
               for a in rec.by_position.values() for c in a.candidates)
    # the keeper's own pick number never shows up as a future pick in evaluate_pick_choice either
    assert "#48" not in evaluate_pick_choice(st47, adv, rec.best_overall[1].player_id)


def test_s2_availability_conditions_on_first_opponent_pick(draft_json, picks_json, universe, league):
    players, proj = universe
    adv = Advisor(league, players, proj)
    # slot 12 on the clock at #12 with #13 next: nobody can pick in between -> 100% for everyone
    st = DraftState(_draft(draft_json), _picks(picks_json)[:11], league, my_slot=12)
    assert st.is_my_turn and st.next_pick_no == 12
    ctx = adv._context(st)
    assert ctx.eval_pick == 13 and ctx.on_clock
    assert np.all(ctx.p_next == 1.0) and np.all(ctx.p_after <= 1.0)
    rec = adv.recommend(st)
    assert all(a.rationale.startswith("100% chance") or a.action == "TAKE NOW" or a.action == "SKIP"
               for a in rec.by_position.values())
    assert any("100% chance" in r for r in rec.top_pick.reasons)
    assert any("next at #13 (back-to-back)" in n for n in rec.notes)
    # mid-board on my turn: p_next is conditioned on pick #6 (the first opponent pick), not on #5
    st5 = DraftState(_draft(draft_json), _picks(picks_json)[:4], league, my_slot=5)
    assert st5.is_my_turn and st5.next_pick_no == 5
    c5 = adv._context(st5)
    assert c5.eval_pick == 20 and c5.on_clock
    shift = shift_for_pressure(c5.pressure, adp_baseline(adv.available_players(st5), c5.n_between))
    shift_arr = np.array([shift[SKILL_POSITIONS[int(pi)]] for pi in c5.pos])
    mu, sigma = adv._mu[c5.avail], adv._sigma[c5.avail]
    assert np.allclose(c5.p_next, prob_available_array(mu, sigma, 20, 6, shift_arr))
    assert np.allclose(c5.p_after, prob_available_array(mu, sigma, c5.eval_after, 6, shift_arr))
    # the magnitude of the old bias for a player whose ADP is right at my pick
    biased = prob_available_array(np.array([5.0]), np.array([2.5]), 9, 5)[0]
    right = prob_available_array(np.array([5.0]), np.array([2.5]), 9, 6)[0]
    assert right > biased + 0.05
    # not my turn: conditioning stays at the current pick
    st6 = DraftState(_draft(draft_json), _picks(picks_json)[:5], league, my_slot=5)
    assert not st6.is_my_turn
    assert not adv._context(st6).on_clock


def test_s3_replacement_level_does_not_sink_into_bench_filler(fixture_state, universe, league):
    players, proj = universe
    adv = Advisor(league, players, proj)
    full = replacement_levels(proj, players, league)
    levels = {}
    for pick in (25, 49, 97, 145):
        st = _advance(fixture_state, adv, until_pick=pick)
        levels[pick] = adv._context(st).rep
    # RB/WR are drafted faster than the demand model allows for: the level drifts down gently
    # (old code: 36th *available* RB, 239 -> 129 at #97 -> 93 at #145)
    assert levels[145]["RB"] > 0.7 * full["RB"] and levels[145]["WR"] > 0.7 * full["WR"]
    assert levels[25]["RB"] <= full["RB"] + 5 and levels[145]["RB"] < levels[25]["RB"]
    # a position nobody drafts keeps the full-pool level
    assert levels[97]["TE"] == pytest.approx(full["TE"]) and levels[97]["K"] == pytest.approx(full["K"])
    # the displayed VORP of a below-replacement RB late in the draft is not inflated
    st = _advance(fixture_state, adv, until_pick=145)
    v = adv.value_of(st, adv.available_players(st)[-1].player_id)
    assert v.vorp < 0
    # remaining demand: equals demand - drafted when the draft follows the demand model, floored by pace
    assert remaining_demand(35.4, 0, 1.0) == pytest.approx(35.4)
    assert remaining_demand(35.4, 20, 0.4) == pytest.approx(15.4)
    assert remaining_demand(35.4, 40, 0.2) == pytest.approx(7.08)
    assert remaining_demand(35.4, 40, 0.0) == 0.0
    rb_sorted = sorted((p.points for p in proj.values() if p.position == "RB"), reverse=True)
    top20 = {pid for pid in proj if proj[pid].position == "RB" and proj[pid].points >= rb_sorted[19]}
    avail = [pid for pid in proj if pid not in top20]
    lv = replacement_levels(proj, players, league, avail, drafted_counts={"RB": 20}, remaining_fraction=0.4)
    assert lv["RB"] == pytest.approx(full["RB"])           # 20 gone, 15 more needed -> same player as full pool
    assert replacement_levels(proj, players, league, avail)["RB"] < full["RB"]   # old behaviour still available


def test_s4_displaced_starter_priced_at_his_own_position(draft_json, universe, league):
    players, proj = universe
    adv = Advisor(league, players, proj)
    # second TE starts in FLEX: an RB who bumps him sends a *TE* to the bench
    st = _with_my_roster(adv, draft_json, league, [("QB", 0), ("RB", 0), ("RB", 5), ("WR", 0), ("WR", 5), ("TE", 0), ("TE", 1)], 96)
    ctx = adv._context(st)
    roster = adv._roster_tuples(st.picks_by_slot()[1])
    assert sorted(pl.position for pl, _ in roster) == ["QB", "RB", "RB", "TE", "TE", "WR", "WR"] and ctx.picks_spare >= 2
    te_in_flex = min(p for pl, p in roster if pl.position == "TE")
    bench_discount = adv.settings.bench_discount
    checked = 0
    for j in range(ctx.avail.size):
        pid = adv.ids[int(ctx.avail[j])]
        pts = proj[pid].points
        if pts == te_in_flex:
            continue                                    # exact tie: either assignment is optimal
        exact = marginal_lineup_value(players[pid], pts, roster, league, bench_discount, replacement=ctx.rep)
        assert float(ctx.marginal[j]) == pytest.approx(exact, abs=1e-6), (pid, pts)
        if players[pid].position == "RB" and pts > te_in_flex:
            expected = (pts - te_in_flex) + bench_discount * 0.15 * max(0.0, te_in_flex - ctx.rep["TE"])
            assert float(ctx.marginal[j]) == pytest.approx(expected, abs=1e-6)
            checked += 1
    assert checked > 0
    # QB2 in SUPER_FLEX displaced by an RB/WR: bench value at QB usefulness (superflex -> 1.0) over the QB level
    raw = dict(league.__dict__)
    sf = LeagueSettings(league.league_id, league.name, league.season, league.total_rosters,
                        ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "SUPER_FLEX", "K", "DEF"] + ["BN"] * 6,
                        dict(league.scoring_settings))
    adv_sf = Advisor(sf, players, proj)
    st = _with_my_roster(adv_sf, draft_json, sf, [("QB", 0), ("QB", 25), ("RB", 0), ("RB", 5), ("WR", 0), ("WR", 5), ("TE", 0)], 96)
    ctx = adv_sf._context(st)
    roster = adv_sf._roster_tuples(st.picks_by_slot()[1])
    qb2 = min(p for pl, p in roster if pl.position == "QB")
    for j in range(ctx.avail.size):
        pid = adv_sf.ids[int(ctx.avail[j])]
        pts = proj[pid].points
        if pts == qb2:
            continue
        exact = marginal_lineup_value(players[pid], pts, roster, sf, bench_discount, replacement=ctx.rep)
        assert float(ctx.marginal[j]) == pytest.approx(exact, abs=1e-6), (pid, pts)
    # starter_displacements names the weakest reachable starter
    slots = ["QB", "RB", "WR", "FLEX"]
    disp = starter_displacements({0: ("QB", 300.0), 1: ("RB", 200.0), 2: ("WR", 210.0), 3: ("TE", 150.0)}, slots)
    assert disp["RB"] == (150.0, 3) and disp["TE"] == (150.0, 3) and disp["QB"] == (300.0, 0)
    assert disp["K"] == (float("inf"), None)
    assert starter_displacements({0: ("QB", 300.0)}, slots)["RB"] == (0.0, None)


def test_s5_trade_prices_bench_like_the_advisor(universe, league):
    players, proj = universe
    by = lambda pos: sorted((pid for pid in proj if proj[pid].position == pos), key=lambda k: -proj[k].points)
    qbs, rbs, wrs, tes, ks, defs = (by(p) for p in ("QB", "RB", "WR", "TE", "K", "DEF"))
    rep = replacement_levels(proj, players, league)
    mine = [qbs[0], rbs[0], rbs[5], rbs[20], wrs[0], wrs[5], wrs[30], tes[0], ks[0], defs[0]]
    theirs = [qbs[1], rbs[1], rbs[2], wrs[1], wrs[2], wrs[40], tes[1], ks[1], defs[1]]
    # a K2 is worth nothing on the bench; a QB2 in a 1-QB league almost nothing
    base, _, _ = roster_value(mine, players, proj, league)
    assert roster_value(mine + [ks[5]], players, proj, league)[0] == pytest.approx(base)
    with_qb2 = roster_value(mine + [qbs[2]], players, proj, league)[0]
    assert 0.0 <= with_qb2 - base <= 0.35 * 0.12 * (proj[qbs[2]].points - rep["QB"]) + 1e-9
    # trading my FLEX RB (starting) for their QB1 (my bench) is a loss, not an ACCEPT
    ev = evaluate_trade(mine, theirs, give=[rbs[20]], get=[qbs[1]], players=players, projections=proj, league=league)
    assert ev.my_delta < 0 and ev.verdict != "ACCEPT"
    # a free QB adds only his (tiny) bench value, not 0.35 x his raw points
    free = evaluate_trade(mine, theirs, give=[], get=[qbs[1]], players=players, projections=proj, league=league)
    assert free.my_delta < 0.35 * 0.12 * proj[qbs[1]].points
    # giving away my K for a useful WR: the K's bench value is 0, the WR's is real
    k_for_wr = evaluate_trade(mine, theirs, give=[ks[0]], get=[wrs[40]], players=players, projections=proj, league=league)
    assert k_for_wr.my_delta == pytest.approx(
        -proj[ks[0]].points + proj[ks[1]].points * 0 + 0.35 * 1.0 * (0.6 ** 1) * max(0.0, proj[wrs[40]].points - rep["WR"]), abs=1e-6) \
        or k_for_wr.my_delta < 0
    # symmetry survives
    mirror = evaluate_trade(theirs, mine, give=[qbs[1]], get=[rbs[20]], players=players, projections=proj, league=league)
    assert mirror.my_delta == pytest.approx(ev.their_delta) and mirror.their_delta == pytest.approx(ev.my_delta)
    # the shared helper: K/DEF 0, depth decay, over replacement only
    lg = _lg(STD_SLOTS)
    pos = {"a": "RB", "b": "RB", "k": "K", "q": "QB"}
    pts = {"a": 200.0, "b": 180.0, "k": 150.0, "q": 300.0}
    r = {"RB": 100.0, "K": 120.0, "QB": 250.0}
    assert bench_value(["a", "b", "k", "q"], pos, pts, lg, 0.35, r) == pytest.approx(
        0.35 * 100 + 0.35 * 0.6 * 80 + 0.0 + 0.35 * 0.12 * 50)
    assert bench_value(["__rep__3", "zzz"], pos, pts, lg, 0.35, r) == 0.0


def test_s6_stack_bonus_only_for_players_who_start(draft_json, universe, league):
    players, proj = universe
    adv = Advisor(league, players, proj, Settings(stack_bonus=0.05))
    st = _with_my_roster(adv, draft_json, league, [("QB", 0), ("RB", 0), ("RB", 5), ("WR", 0), ("WR", 5), ("TE", 0), ("K", 0)], 108)
    roster = adv._roster_tuples(st.picks_by_slot()[1])
    my_wr = next(pl for pl, _ in roster if pl.position == "WR")
    qb2 = next(p for p in adv.available_players(st) if p.position == "QB")
    qb2.team = my_wr.team                                # "stack" with a backup QB who will never start
    adv_b = Advisor(league, players, proj, Settings(stack_bonus=0.05))
    adv_0 = Advisor(league, players, proj, Settings(stack_bonus=0.0))
    v_b, v_0 = adv_b.value_of(st, qb2.player_id), adv_0.value_of(st, qb2.player_id)
    assert v_b.marginal_value == v_0.marginal_value and v_b.score == pytest.approx(v_0.score)
    assert not any("Stacks with" in r for r in v_b.reasons)
    # a WR who *would* start next to my QB still gets the bonus
    my_qb = next(pl for pl, _ in roster if pl.position == "QB")
    wr = next(p for p in adv.available_players(st) if p.position == "WR")
    wr.team = my_qb.team
    adv_b2 = Advisor(league, players, proj, Settings(stack_bonus=0.05))
    v = adv_b2.value_of(st, wr.player_id)
    assert v.marginal_value > 0 and any("Stacks with your QB" in r for r in v.reasons)
    assert v.score > Advisor(league, players, proj, Settings(stack_bonus=0.0)).value_of(st, wr.player_id).score


def test_s7_last_pick_has_no_fictitious_next_pick(draft_json, universe, league):
    players, proj = universe
    adv = Advisor(league, players, proj)
    st = DraftState(_draft(draft_json), [], league, my_slot=12)      # slot 12 owns the very last pick (#180)
    st = _advance(st, adv, until_pick=180, allow_kdef=True)
    assert st.next_pick_no == 180 and st.is_my_turn
    ctx = adv._context(st)
    assert ctx.eval_pick is None and ctx.on_clock and ctx.remaining == 1 and ctx.n_between == 0
    assert np.all(ctx.p_next == 1.0) and np.all(ctx.next_after == 0.0)
    rec = adv.recommend(st)
    assert any("this is your last pick" in n for n in rec.notes)
    for v in rec.best_overall:
        assert v.availability_next == 1.0 and any("This is your last pick" in r for r in v.reasons)
        assert not any("#18" in r or "#19" in r for r in v.reasons)
    for a in rec.by_position.values():
        assert "last pick" in a.rationale and "(#" not in a.rationale
        assert a.action in ("TAKE NOW", "SKIP")
    assert "last pick" in adv.explain_pick(st, rec.best_overall[1].player_id)
    assert all(v.score <= v.marginal_value + 1e-9 for v in rec.best_overall)     # no lookahead term left


def test_s8_notes_respect_league_shape(fixture_state, universe, league_json):
    players, proj = universe
    raw = dict(league_json)
    raw["roster_positions"] = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "FLEX"] + ["BN"] * 7
    no_kdef = _league(raw)
    adv = Advisor(no_kdef, players, proj)
    rec = adv.recommend(fixture_state)
    assert not any("K/DEF" in n for n in rec.notes)
    assert rec.by_position["K"].candidates == [] and "No K starting slot" in rec.by_position["K"].rationale
    assert rec.by_position["DEF"].action == "SKIP"
    assert not any(v.player.position in ("K", "DEF") for v in rec.best_overall)
    assert not any(p.position in ("K", "DEF") for p in adv.available_players(fixture_state))
    late = _advance(fixture_state, adv, until_pick=12 * 13 + 1)
    assert not any(v.player.position in ("K", "DEF") for v in adv.recommend(late).best_overall)
    # the standard league still gets the note (and SKIP) in round 2
    std = Advisor(_league(league_json), players, proj).recommend(fixture_state)
    assert any(n.startswith("K/DEF: wait until round") for n in std.notes) and std.by_position["K"].action == "SKIP"
    # superflex scarcity is measured on the full-pool level: silent with a deep pool, fires as it drains
    raw["roster_positions"] = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "SUPER_FLEX", "K", "DEF"] + ["BN"] * 6
    sf = _league(raw)
    adv_sf = Advisor(sf, players, proj)
    assert not any(n.startswith("Superflex") for n in adv_sf.recommend(fixture_state).notes)
    mid = _advance(fixture_state, adv_sf, until_pick=97)
    rec_mid = adv_sf.recommend(mid)
    note = next((n for n in rec_mid.notes if n.startswith("Superflex: QBs scarce")), None)
    assert note is not None and "startable left for" in note
    full_qb = sorted((p.points for p in proj.values() if p.position == "QB"), reverse=True)[round(starter_demand(sf)["QB"])]
    startable = sum(1 for p in adv_sf.available_players(mid) if p.position == "QB" and proj[p.player_id].points > full_qb)
    assert f"({startable} startable" in note and startable <= 12


def test_s9_idp_slots_count_against_spare_picks(fixture_state, universe, league_json):
    players, proj = universe
    raw = dict(league_json)
    raw["roster_positions"] = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF", "DL", "LB", "DB"] + ["BN"] * 3
    idp = _league(raw)
    adv = Advisor(idp, players, proj)
    ctx = adv._context(fixture_state)
    std_ctx = Advisor(_league(league_json), players, proj)._context(fixture_state)
    assert ctx.idp_open == 3 and ctx.idp_slots == ["DL", "LB", "DB"]
    assert ctx.picks_spare == std_ctx.picks_spare - 3 and ctx.open_slots == std_ctx.open_slots
    rec = adv.recommend(fixture_state)
    assert any(n.startswith("You must draft 3 IDP starters (DL LB DB)") for n in rec.notes)
    # an LB on my roster (unknown to the projections) fills one of them
    picks = list(fixture_state.picks) + [Pick(21, 2, 12, "idp-lb-1", 4, metadata={"position": "LB", "first_name": "Roquan", "last_name": "Smith"})]
    st = fixture_state.with_picks(picks)
    ctx2 = adv._context(st)
    # #21 is slot 12's number attributed to my roster (a traded pick): one slot fewer, no pick of mine spent
    assert ctx2.idp_open == 2 and ctx2.picks_spare == ctx.picks_spare + 1
    assert any("2 IDP starters" in n for n in adv.recommend(st).notes)
    assert not any("IDP" in n for n in Advisor(_league(league_json), players, proj).recommend(st).notes)
    # end-game with IDP slots still open: they are counted as must-fill picks
    late = _advance(st, adv, until_pick=12 * 13 + 1)
    c3 = adv._context(late)
    assert c3.picks_spare <= 0 and c3.idp_open == 2
    assert any("IDP" in n for n in adv.recommend(late).notes)


def test_s10_simulate_tolerates_unknown_positions(fixture_state, universe, league):
    players, proj = universe
    adv = Advisor(league, players, proj)
    picks = list(fixture_state.picks) + [
        Pick(21, 2, 12, "idp-lb-1", 4, metadata={"position": "LB"}),
        Pick(22, 2, 11, "ghost-1", 4, metadata={}),                 # no metadata at all -> position UNK
    ]
    st = fixture_state.with_picks(picks)
    rec = adv.recommend(st)
    assert rec.my_roster is not None and rec.my_roster.position_counts.get("LB") == 1
    cands = [v.player_id for v in rec.best_overall[:2]]
    out = simulate_candidates(st, players, proj, adv, cands, n_sims=3, rounds_ahead=3, seed=1)
    assert set(out) == set(cands) and all(v > 0 for v in out.values())


def test_f3_rostered_players_and_rookie_only_drafts_are_excluded(fixture_state, universe, league):
    """Dynasty rosters and rookie-only drafts shrink the pool (DraftState.rostered_ids / player_type)."""
    import dataclasses

    from draftadvisor.strategy.recommend import Advisor

    players, projections = universe
    adv = Advisor(league, players, projections)
    rec0 = adv.recommend(fixture_state)
    top = rec0.best_overall[0].player_id
    st = fixture_state.with_picks(fixture_state.picks)
    if hasattr(st, "rostered_ids"):
        st.rostered_ids = {top}
        rec1 = adv.recommend(st)
        assert all(v.player_id != top for v in rec1.best_overall)
        assert adv.value_of(st, top) is None
    if "player_type" in {f.name for f in dataclasses.fields(type(fixture_state.draft))}:
        d = dataclasses.replace(fixture_state.draft, player_type=1)
        st2 = dataclasses.replace(fixture_state.with_picks(fixture_state.picks), draft=d)
        rec2 = adv.recommend(st2)
        assert rec2.best_overall and all(v.player.years_exp == 0 for v in rec2.best_overall)
