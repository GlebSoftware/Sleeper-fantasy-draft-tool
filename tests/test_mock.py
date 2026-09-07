"""Tests for the offline mock-draft simulator."""
from __future__ import annotations

import random
import time
from collections import Counter

import pytest

from draftadvisor.mock import BotStrategy, MockDraft, make_mock_draft, make_mock_league, player_rank
from draftadvisor.models import Player

_COUNTS = {"QB": 32, "RB": 80, "WR": 100, "TE": 30, "K": 20, "DEF": 32}
# rough real-life ADP windows per position (start, end) used to build a plausible board
_ADP_WINDOW = {"QB": (25, 190), "RB": (1, 210), "WR": (1, 210), "TE": (8, 200), "K": (140, 230), "DEF": (135, 230)}


def make_universe(seed: int = 0, unranked: int = 25) -> dict[str, Player]:
    """Synthetic player universe with ADP/ECR populated; the last ``unranked`` have neither."""
    rng = random.Random(seed)
    rows = []
    for pos, n in _COUNTS.items():
        lo, hi = _ADP_WINDOW[pos]
        for i in range(n):
            rows.append((pos, i, lo + (hi - lo) * (i / n) ** 0.9 + rng.uniform(-3, 3)))
    rows.sort(key=lambda r: r[2])
    players: dict[str, Player] = {}
    for rank, (pos, i, _) in enumerate(rows, start=1):
        pid = f"{pos}{i:03d}"
        p = Player(player_id=pid, name=f"{pos} Player{i}", position=pos, team=f"T{i % 32}",
                   fantasy_positions=(pos,), bye_week=5 + (i % 10))
        if rank <= len(rows) - unranked:
            p.adp = float(rank)
            p.adp_source = "test"
            if rank % 3:
                p.ecr = float(rank) + rng.uniform(-2, 2)
        players[pid] = p
    return players


def run_full(teams=12, rounds=15, slot=5, seed=1, superflex=False, scoring="half_ppr") -> MockDraft:
    league = make_mock_league(teams=teams, rounds=rounds, scoring=scoring, superflex=superflex)
    draft = make_mock_draft(league, my_slot=slot, teams=teams, rounds=rounds)
    md = MockDraft(make_universe(), league, draft, my_slot=slot, seed=seed)

    def policy(state):
        return md.available()[0].player_id  # take best by market rank

    md.run_to_completion(policy)
    return md


# ---------------------------------------------------------------------------


def test_make_mock_league_and_draft():
    lg = make_mock_league(teams=10, rounds=16, scoring="ppr", te_premium=0.5)
    assert lg.total_rosters == 10 and lg.scoring_type == "ppr" and lg.te_premium == 0.5
    assert lg.roster_positions[:9] == ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF"]
    assert lg.roster_size == 16 and lg.bench_slots == 7
    assert not lg.is_superflex
    sf = make_mock_league(superflex=True, scoring="std")
    assert sf.is_superflex and "SUPER_FLEX" in sf.starting_slots and sf.rec_points == 0.0
    assert sf.roster_size == 15
    with pytest.raises(ValueError):
        make_mock_league(scoring="weird")

    d = make_mock_draft(lg, my_slot=4, teams=10, rounds=16)
    assert d.teams == 10 and d.rounds == 16 and d.type == "snake" and d.total_picks == 160
    assert d.draft_order["me"] == 4 and d.draft_order["bot-1"] == 1 and len(d.draft_order) == 10
    assert d.slot_to_roster_id == {s: s for s in range(1, 11)}
    assert d.scoring_type == "ppr" and d.pick_timer == 30
    with pytest.raises(ValueError):
        make_mock_draft(lg, my_slot=11, teams=10)


def test_player_rank_fallbacks():
    assert player_rank(Player("1", "a", "RB", adp=12.0, ecr=3.0)) == 12.0
    assert player_rank(Player("1", "a", "RB", ecr=3.0)) == 3.0
    assert player_rank(Player("1", "a", "RB")) == 400.0


def test_initial_state_and_managers():
    league = make_mock_league()
    draft = make_mock_draft(league, my_slot=3)
    md = MockDraft(make_universe(), league, draft, my_slot=3, seed=0)
    st = md.state()
    assert st.my_slot == 3 and st.my_user_id == "me" and st.version == 0
    assert st.slot_label(3) == "You" and st.slot_label(7) == "Bot 7"
    assert len(st.managers) == 12 and st.managers["bot-1"].roster_id == 1
    assert not md.is_my_turn and not md.is_complete
    assert set(md.strategies.values()) <= set(BotStrategy) and len(md.strategies) == 12


def test_full_draft_completes_without_duplicates_and_my_picks_on_my_slot():
    t0 = time.perf_counter()
    md = run_full(slot=5, seed=3)
    elapsed = time.perf_counter() - t0
    st = md.state()
    assert md.is_complete and st.is_complete
    assert len(st.picks) == 180
    assert [p.pick_no for p in st.picks] == list(range(1, 181))
    ids = [p.player_id for p in st.picks]
    assert len(set(ids)) == 180
    mine = [p for p in st.picks if p.picked_by == "me"]
    assert len(mine) == 15
    assert all(p.draft_slot == 5 and p.roster_id == 5 for p in mine)
    assert sorted(p.pick_no for p in mine) == st.draft.picks_for_slot(5)
    assert len(st.my_picks()) == 15
    # pick metadata mirrors Sleeper
    p0 = st.picks[0]
    assert p0.metadata["position"] in ("QB", "RB", "WR", "TE") and p0.metadata["first_name"]
    assert p0.player_name.startswith(p0.metadata["first_name"])
    # every roster has exactly 15 players
    by_slot = st.picks_by_slot()
    assert all(len(v) == 15 for v in by_slot.values())
    assert elapsed < 2.0, f"full draft took {elapsed:.2f}s"


def test_bots_take_k_def_late_and_only_one_each():
    for seed in (1, 2, 3):
        md = run_full(seed=seed)
        rounds = md.draft.rounds
        for p in md.picks:
            if p.picked_by == "me":
                continue
            if p.position in ("K", "DEF"):
                assert p.round >= rounds - 3, f"bot took {p.position} in round {p.round}"
        for slot in range(1, 13):
            if slot == md.my_slot:
                continue
            counts = md.roster_counts(slot)
            assert counts.get("K", 0) == 1 and counts.get("DEF", 0) == 1, counts
            assert counts.get("QB", 0) <= 2
            assert counts.get("TE", 0) <= 3
            assert sum(counts.values()) == md.league.roster_size


def test_bots_fill_starting_lineups():
    md = run_full(seed=7)
    for slot in range(1, 13):
        if slot == md.my_slot:
            continue
        c = md.roster_counts(slot)
        assert c.get("QB", 0) >= 1 and c.get("RB", 0) >= 2 and c.get("WR", 0) >= 2 and c.get("TE", 0) >= 1, c


def test_bots_roughly_follow_adp():
    md = run_full(seed=11)
    first_round = [md.players[p.player_id] for p in md.picks[:12]]
    assert all(player_rank(p) <= 25 for p in first_round)
    # no unranked (ADP 400) player before the last few rounds
    for p in md.picks:
        if player_rank(md.players[p.player_id]) >= 400:
            assert p.round >= md.draft.rounds - 2


def test_superflex_drafts_more_qbs_earlier():
    qb_counts, qb_first = [], []
    for sf in (False, True):
        total, first = 0, 0
        for seed in (1, 2, 3):
            md = run_full(seed=seed, superflex=sf)
            bot_picks = [p for p in md.picks if p.picked_by != "me"]
            total += sum(1 for p in bot_picks if p.position == "QB")
            first += sum(1 for p in bot_picks[:36] if p.position == "QB")
        qb_counts.append(total)
        qb_first.append(first)
    assert qb_counts[1] > qb_counts[0]
    assert qb_first[1] > qb_first[0]
    md = run_full(seed=1, superflex=True)
    assert all(md.roster_counts(s).get("QB", 0) <= 3 for s in range(1, 13) if s != md.my_slot)
    assert sum(md.roster_counts(s).get("QB", 0) >= 2 for s in range(1, 13) if s != md.my_slot) >= 8


def test_strategies_shift_behaviour():
    league = make_mock_league()
    draft = make_mock_draft(league, my_slot=12)
    strategies = {s: BotStrategy.ZERO_RB for s in range(1, 13)}
    strategies.update({1: BotStrategy.EARLY_QB, 2: BotStrategy.HERO_RB, 3: BotStrategy.LATE_QB})
    md = MockDraft(make_universe(), league, draft, my_slot=12, seed=5, strategies=strategies)
    md.advance_until_my_turn()                    # 11 bot picks of round 1
    first_round = md.picks[:11]
    assert first_round[1].position == "RB"                     # hero_rb opens with a RB
    assert all(p.position != "RB" for p in first_round[2:])   # zero_rb bots avoid RB early
    # run 8 rounds: early_qb has a QB among its first 3 picks, late_qb none in its first 7
    while md.current_round <= 8 and not md.is_complete:
        if md.is_my_turn:
            md.make_pick(md.available()[0].player_id)
        else:
            md.bot_pick()
    by_slot = md.state().picks_by_slot()
    assert any(p.position == "QB" for p in by_slot[1][:3])
    assert all(p.position != "QB" for p in by_slot[3][:7])
    # zero_rb bots still end up with RBs by round 8 (need-aware)
    assert all(any(p.position == "RB" for p in by_slot[s]) for s in range(4, 12))


def test_make_pick_validation_and_advance():
    league = make_mock_league()
    draft = make_mock_draft(league, my_slot=4)
    md = MockDraft(make_universe(), league, draft, my_slot=4, seed=0)
    st = md.advance_until_my_turn()
    assert md.is_my_turn and st.is_my_turn and st.next_pick_no == 4 and len(st.picks) == 3
    taken = st.picks[0].player_id
    with pytest.raises(ValueError):
        md.make_pick(taken)
    with pytest.raises(ValueError):
        md.make_pick("does-not-exist")
    pid = md.available()[0].player_id
    pick = md.make_pick(pid)
    assert pick.pick_no == 4 and pick.draft_slot == 4 and pick.picked_by == "me" and pick.player_id == pid
    assert not md.is_my_turn
    st2 = md.state()
    assert st2.version == 4 and len(st2.picks) == 4 and st2 is md.state()  # cached until next pick
    st3 = md.advance_until_my_turn()
    assert st3.next_pick_no == 21 and md.is_my_turn


def test_seed_is_reproducible():
    a = run_full(seed=42)
    b = run_full(seed=42)
    assert [p.player_id for p in a.picks] == [p.player_id for p in b.picks]
    c = run_full(seed=43)
    assert [p.player_id for p in a.picks] != [p.player_id for p in c.picks]


def test_small_draft_and_completion_errors():
    league = make_mock_league(teams=4, rounds=3)
    draft = make_mock_draft(league, my_slot=2, teams=4, rounds=3)
    md = MockDraft(make_universe(), league, draft, my_slot=2, seed=0)
    st = md.run_to_completion(lambda s: md.available()[0].player_id)
    assert st.is_complete and len(st.picks) == 12
    with pytest.raises(ValueError):
        md.make_pick(md.available()[0].player_id)
    with pytest.raises(ValueError):
        md.bot_pick()
    pos = Counter(p.position for p in st.picks)
    assert pos["K"] + pos["DEF"] <= 8
