"""Pick-order math and DraftState derived properties."""
from __future__ import annotations

from draftadvisor.models import DraftSettings, DraftState, LeagueSettings, Pick, normal_cdf, normal_ppf


def _draft(**kw) -> DraftSettings:
    base = dict(draft_id="d", league_id="l", type="snake", status="drafting", teams=12, rounds=15)
    base.update(kw)
    return DraftSettings(**base)


def test_snake_order():
    d = _draft()
    assert [d.slot_for_pick(n) for n in range(1, 14)] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12]
    assert d.slot_for_pick(24) == 1 and d.slot_for_pick(25) == 1 and d.slot_for_pick(36) == 12
    assert d.pick_no_for(2, 5) == 20 and d.slot_for_pick(20) == 5
    for n in range(1, d.total_picks + 1):
        assert d.pick_no_for(d.round_of(n), d.slot_for_pick(n)) == n


def test_third_round_reversal():
    d = _draft(reversal_round=3)
    assert d.slot_for_pick(25) == 12 and d.slot_for_pick(36) == 1     # round 3 reversed
    assert d.slot_for_pick(37) == 1 and d.slot_for_pick(48) == 12     # round 4 forward
    assert d.slot_for_pick(49) == 12                                  # round 5 backward


def test_linear_order():
    d = _draft(type="linear")
    assert d.slot_for_pick(13) == 1 and d.slot_for_pick(24) == 12


def test_traded_picks_change_ownership():
    d = _draft(slot_to_roster_id={s: s + 100 for s in range(1, 13)},
               traded_picks={(3, 102): 101})                            # slot 2's round-3 pick now owned by slot 1
    r3_slot2 = d.pick_no_for(3, 2)
    assert d.owner_roster_for_pick(r3_slot2) == 101
    mine = d.picks_owned_by_roster(101)
    assert r3_slot2 in mine and d.pick_no_for(3, 1) in mine
    assert r3_slot2 not in d.picks_owned_by_roster(102)
    assert d.picks_for_slot(1) == mine


def test_draft_state_turn_logic():
    d = _draft(slot_to_roster_id={s: s for s in range(1, 13)})
    picks = [Pick(pick_no=n, round=d.round_of(n), draft_slot=d.slot_for_pick(n), player_id=str(n), roster_id=d.slot_for_pick(n))
             for n in range(1, 21)]
    st = DraftState(draft=d, picks=picks, my_slot=5)
    assert st.next_pick_no == 21 and st.current_round == 2 and st.on_the_clock_slot == 4
    assert st.my_next_pick_no == 29 and st.my_pick_after_next == 44   # slot 5: picks 5, 20, 29, 44
    assert st.picks_until_my_turn == 8 and not st.is_my_turn
    assert len(st.my_picks()) == 2 and {p.player_id for p in st.my_picks()} == {"5", "20"}
    assert st.drafted_ids == {str(n) for n in range(1, 21)}
    st2 = st.with_picks(picks + [Pick(pick_no=n, round=d.round_of(n), draft_slot=d.slot_for_pick(n), player_id=str(n))
                                  for n in range(21, 29)])
    assert st2.is_my_turn and st2.version == st.version + 1 and st2.picks_until_my_turn == 0


def test_draft_state_complete():
    d = _draft(teams=2, rounds=2)
    picks = [Pick(pick_no=n, round=d.round_of(n), draft_slot=d.slot_for_pick(n), player_id=str(n)) for n in range(1, 5)]
    st = DraftState(draft=d, picks=picks, my_slot=1)
    assert st.is_complete and st.on_the_clock_slot is None and st.my_next_pick_no is None


def test_league_settings_shape():
    lg = LeagueSettings(league_id="l", name="x", season=2026, total_rosters=12,
                        roster_positions=["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF", "BN", "BN", "IR"],
                        scoring_settings={"rec": 0.5, "bonus_rec_te": 0.5})
    assert lg.scoring_type == "half_ppr" and lg.te_premium == 0.5 and not lg.is_superflex
    assert lg.starting_slots == ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF"]
    assert lg.bench_slots == 2 and lg.roster_size == 11
    assert lg.dedicated_starters("RB") == 2 and lg.flex_slots_for("RB") == 1 and lg.flex_slots_for("QB") == 0
    sf = LeagueSettings(league_id="l", name="x", season=2026, total_rosters=12,
                        roster_positions=["QB", "SUPER_FLEX", "BN"], scoring_settings={"rec": 1})
    assert sf.is_superflex and sf.scoring_type == "ppr" and sf.flex_slots_for("QB") == 1


def test_normal_helpers():
    assert abs(normal_cdf(0.0) - 0.5) < 1e-12
    assert abs(normal_ppf(0.8) - 0.8416) < 1e-3
    assert abs(normal_cdf(normal_ppf(0.3)) - 0.3) < 1e-6


# ---------------------------------------------------------------------------
# Review findings: keepers (F2/S1), rostered players (F3), traded picks (F5), rookies (PROJ-5)
# ---------------------------------------------------------------------------


def _picks(d: DraftSettings, numbers) -> list[Pick]:
    return [Pick(pick_no=n, round=d.round_of(n), draft_slot=d.slot_for_pick(n), player_id=f"p{n}",
                 roster_id=d.slot_for_pick(n)) for n in numbers]


def test_keeper_pick_is_not_a_future_pick():
    """F2/S1: a keeper pre-populated at one of my later pick numbers must not count as my next pick."""
    d = _draft(slot_to_roster_id={s: s for s in range(1, 13)})
    # slot 1 picks 1, 24, 25, 48, 49 ...; 46 picks made, keeper sitting at #48 (is_keeper)
    picks = _picks(d, range(1, 47))
    keeper = Pick(pick_no=48, round=4, draft_slot=1, player_id="keeper", roster_id=1, is_keeper=True)
    st = DraftState(draft=d, picks=picks + [keeper], my_slot=1)
    assert st.next_pick_no == 47
    assert 48 in st.taken_pick_numbers and 48 in st.my_pick_numbers()
    assert st.my_future_picks()[:2] == [49, 72]
    assert st.my_next_pick_no == 49 and st.my_pick_after_next == 72
    assert st.picks_until_my_turn == 2 and not st.is_my_turn
    # at #47 (slot 11) and #48 skipped: after pick 47 it is *not* my turn, the board jumps to #49
    st2 = st.with_picks(st.picks + _picks(d, [47]))
    assert st2.next_pick_no == 49 and st2.is_my_turn and st2.picks_until_my_turn == 0
    # my last real pick with a keeper behind it: no phantom future pick
    picks3 = _picks(d, range(1, 168)) + [Pick(169, 15, 1, "keeper2", roster_id=1, is_keeper=True)]
    st3 = DraftState(draft=d, picks=picks3, my_slot=1)
    assert st3.next_pick_no == 168 and st3.is_my_turn
    assert st3.my_future_picks() == [168]
    assert st3.my_pick_after_next is None


def test_rostered_ids_and_unavailable_ids():
    """F3: players already on league rosters are unavailable alongside drafted ones and survive with_picks."""
    d = _draft(slot_to_roster_id={s: s for s in range(1, 13)}, player_type=1)
    st = DraftState(draft=d, picks=_picks(d, [1, 2]), my_slot=1, rostered_ids={"vet1", "vet2"})
    assert st.drafted_ids == {"p1", "p2"}
    assert st.unavailable_ids == {"p1", "p2", "vet1", "vet2"}
    assert st.is_rookie_draft
    st2 = st.with_picks(st.picks + _picks(d, [3]))
    assert st2.rostered_ids == {"vet1", "vet2"} and st2.rostered_ids is not st.rostered_ids
    assert "p3" in st2.unavailable_ids
    assert DraftState(draft=_draft(), picks=[]).unavailable_ids == set()
    assert not DraftState(draft=_draft(), picks=[]).is_rookie_draft


def test_slot_of_pick_uses_roster_mapping_for_traded_picks():
    """F5: a traded pick keeps the original draft_slot; the drafter is identified by roster_id."""
    d = _draft(slot_to_roster_id={s: s + 100 for s in range(1, 13)}, traded_picks={(3, 102): 101})
    traded = Pick(pick_no=d.pick_no_for(3, 2), round=3, draft_slot=2, player_id="x", roster_id=101)
    plain = Pick(pick_no=1, round=1, draft_slot=1, player_id="y", roster_id=101)
    no_roster = Pick(pick_no=2, round=1, draft_slot=2, player_id="z", roster_id=None)
    st = DraftState(draft=d, picks=[plain, no_roster, traded], my_slot=1)
    assert st.slot_of_pick(traded) == 1 and st.slot_of_pick(plain) == 1 and st.slot_of_pick(no_roster) == 2
    assert [p.player_id for p in st.my_picks()] == ["y", "x"]
    assert [p.player_id for p in st.picks_by_slot()[2]] == ["z"]


def test_is_rookie_requires_years_exp_zero():
    """PROJ-5: unknown years_exp (team defenses, some vets) is not a rookie."""
    from draftadvisor.models import Player

    assert Player("1", "X", "WR", years_exp=0).is_rookie
    assert not Player("2", "Y", "WR", years_exp=3).is_rookie
    assert not Player("SF", "SF Defense", "DEF", years_exp=None).is_rookie
