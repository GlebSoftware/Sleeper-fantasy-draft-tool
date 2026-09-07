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
from draftadvisor.strategy.availability import expected_best_available_array, prob_available_array
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
