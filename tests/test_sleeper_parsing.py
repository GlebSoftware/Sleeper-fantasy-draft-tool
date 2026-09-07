"""Tests for draftadvisor.sleeper.parsing (raw Sleeper JSON -> models)."""
from __future__ import annotations

import copy

import pytest

from draftadvisor.models import DraftState, Pick
from draftadvisor.sleeper.parsing import (
    adp_key_for,
    adp_map,
    parse_draft,
    parse_league,
    parse_managers,
    parse_pick,
    parse_picks,
    resolve_my_slot,
    state_from_sleeper,
)

GLEB = "111111111111111111"
MIKE = "222222222222222222"


# ---------------------------------------------------------------------------
# league / draft
# ---------------------------------------------------------------------------


def test_parse_league(league_json):
    lg = parse_league(league_json)
    assert lg.league_id == "1180000000000000001"
    assert lg.name == "Fixture League"
    assert lg.season == 2026
    assert lg.total_rosters == 12
    assert lg.roster_positions[:3] == ["QB", "RB", "RB"]
    assert lg.scoring_settings["rec"] == 1.0
    assert lg.scoring_type == "ppr"
    assert lg.te_premium == 0.5
    assert not lg.is_superflex
    assert lg.draft_id == "1180000000000000002"
    assert lg.status == "drafting"
    assert lg.roster_size == 15  # 9 starters + 6 BN (IR excluded)


def test_parse_league_tolerates_missing_fields():
    lg = parse_league({"league_id": 42, "settings": {"num_teams": "10"}, "scoring_settings": {"rec": None, "rec_yd": "0.1"}})
    assert lg.league_id == "42"
    assert lg.total_rosters == 10
    assert lg.scoring_settings == {"rec_yd": 0.1}
    assert lg.roster_positions == []


def test_parse_draft_converts_string_keys(draft_json):
    d = parse_draft(draft_json)
    assert d.teams == 12 and d.rounds == 15 and d.pick_timer == 30 and d.reversal_round == 0
    assert d.type == "snake" and d.status == "drafting"
    assert d.draft_order[GLEB] == 1
    assert all(isinstance(k, str) for k in d.draft_order)
    assert d.slot_to_roster_id == {1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9, 7: 10, 8: 11, 9: 12, 10: 1, 11: 2, 12: 3}
    assert all(isinstance(k, int) for k in d.slot_to_roster_id)
    assert d.season == 2026
    assert d.scoring_type == "ppr"
    assert d.start_time == 1789000000000
    assert d.traded_picks == {}


def test_parse_draft_traded_picks(draft_json, traded_picks_json):
    d = parse_draft(draft_json, traded_picks_json)
    assert d.traded_picks == {(3, 5): 4}
    # round-3 pick of slot 2 (roster 5) now belongs to roster 4 (slot 1)
    assert d.owner_roster_for_pick(d.pick_no_for(3, 2)) == 4
    assert d.owner_roster_for_pick(d.pick_no_for(3, 1)) == 4
    assert d.owner_roster_for_pick(d.pick_no_for(2, 2)) == 5


def test_parse_draft_ignores_other_season_and_self_trades(draft_json):
    traded = [
        {"season": "2025", "round": 1, "roster_id": 5, "owner_id": 4},
        {"season": "2026", "round": 2, "roster_id": 5, "owner_id": 5},
        {"season": "2026", "round": 4, "roster_id": "6", "owner_id": "7"},
    ]
    d = parse_draft(draft_json, traded)
    assert d.traded_picks == {(4, 6): 7}


def test_parse_draft_pre_draft_without_order(draft_json):
    raw = copy.deepcopy(draft_json)
    raw["status"] = "pre_draft"
    raw["draft_order"] = None
    raw["slot_to_roster_id"] = None
    d = parse_draft(raw)
    assert d.status == "pre_draft"
    assert d.draft_order == {} and d.slot_to_roster_id == {}
    assert d.teams == 12  # from settings
    st = DraftState(draft=d, picks=[])
    assert st.next_pick_no == 1 and not st.is_complete
    assert st.my_next_pick_no is None


# ---------------------------------------------------------------------------
# picks
# ---------------------------------------------------------------------------


def test_parse_pick_quirks():
    raw = {
        "pick_no": "7", "round": "1", "draft_slot": "7", "player_id": 4034, "roster_id": "3",
        "picked_by": None, "is_keeper": None,
        "metadata": {"first_name": "Christian", "last_name": "McCaffrey", "position": "RB", "team": "SF",
                     "injury_status": "", "years_exp": "9", "player_id": 4034},
    }
    p = parse_pick(raw)
    assert isinstance(p, Pick)
    assert p.pick_no == 7 and p.round == 1 and p.draft_slot == 7
    assert p.player_id == "4034" and p.roster_id == 3 and p.picked_by is None
    assert p.is_keeper is False
    assert p.metadata["injury_status"] is None
    assert p.metadata["years_exp"] == 9
    assert p.metadata["player_id"] == "4034"
    assert p.player_name == "Christian McCaffrey" and p.position == "RB"

    p2 = parse_pick({"pick_no": 8, "round": 1, "draft_slot": 8, "player_id": "SF", "roster_id": None, "is_keeper": True})
    assert p2.roster_id is None and p2.is_keeper is True and p2.metadata == {}
    assert p2.player_name == "SF"


def test_parse_pick_requires_pick_no():
    with pytest.raises(ValueError):
        parse_pick({"round": 1, "player_id": "1"})


def test_parse_picks_sorted_and_deduplicated(picks_json):
    shuffled = list(reversed(picks_json)) + [copy.deepcopy(picks_json[3])]
    shuffled[-1]["player_id"] = "dup-replaced"
    picks = parse_picks(shuffled)
    assert [p.pick_no for p in picks] == list(range(1, 21))
    assert picks[3].player_id == "dup-replaced"  # last occurrence wins
    assert parse_picks(None) == []
    assert parse_picks([{"round": 1}]) == []  # malformed skipped


# ---------------------------------------------------------------------------
# managers
# ---------------------------------------------------------------------------


def test_parse_managers_from_users_and_rosters(users_json, draft_json, rosters_json):
    d = parse_draft(draft_json)
    mgrs = parse_managers(users_json, d, rosters_json)
    assert len(mgrs) == 12
    me = mgrs[GLEB]
    assert me.display_name == "Gleb" and me.team_name == "Team Gleb"
    assert me.slot == 1 and me.roster_id == 4
    assert mgrs["333333333333333333"].team_name is None
    # roster id from slot_to_roster_id when rosters are missing
    mgrs2 = parse_managers(users_json, d)
    assert mgrs2[MIKE].roster_id == 5 and mgrs2[MIKE].slot == 2


def test_parse_managers_placeholder_for_missing_users(draft_json):
    d = parse_draft(draft_json)
    mgrs = parse_managers(None, d)
    assert len(mgrs) == 12
    assert mgrs[GLEB].display_name == "Team 1" and mgrs[GLEB].slot == 1 and mgrs[GLEB].roster_id == 4


# ---------------------------------------------------------------------------
# ADP
# ---------------------------------------------------------------------------


def test_adp_key_for(league_json, draft_json):
    lg, d = parse_league(league_json), parse_draft(draft_json)
    assert adp_key_for(lg, d) == "adp_ppr"
    lg.scoring_settings["rec"] = 0.5
    assert adp_key_for(lg, d) == "adp_half_ppr"
    lg.scoring_settings["rec"] = 0.0
    assert adp_key_for(lg, d) == "adp_std"
    lg.roster_positions.append("SUPER_FLEX")
    assert adp_key_for(lg, d) == "adp_2qb"
    # draft-only inference
    assert adp_key_for(None, d) == "adp_ppr"
    d.scoring_type = "half_ppr"
    assert adp_key_for(None, d) == "adp_half_ppr"
    d.scoring_type = "dynasty_2qb"
    assert adp_key_for(None, d) == "adp_2qb"
    d.scoring_type = "std"
    d.settings["slots_super_flex"] = 1
    assert adp_key_for(None, d) == "adp_2qb"
    assert adp_key_for(None, None) == "adp_std"


def test_adp_map(projections_json):
    m = adp_map(projections_json, "adp_ppr")
    assert m["7564"] == pytest.approx(0.7)
    assert all(v > 0 for v in m.values())
    assert len(m) == len(projections_json)
    # missing key falls back to another ADP flavour; zero/None ignored
    m2 = adp_map({"a": {"adp_half_ppr": 12.0}, "b": {"adp_ppr": 0}, "c": {"adp_ppr": None}, "d": None}, "adp_ppr")
    assert m2 == {"a": 12.0}
    assert adp_map(None, "adp_ppr") == {}


# ---------------------------------------------------------------------------
# resolve_my_slot / state
# ---------------------------------------------------------------------------


def test_resolve_my_slot_priority(users_json, draft_json, rosters_json):
    d = parse_draft(draft_json)
    mgrs = parse_managers(users_json, d, rosters_json)
    assert resolve_my_slot(d, mgrs, slot=3) == ("333333333333333333", 3)
    assert resolve_my_slot(d, mgrs, username="gleb", slot=3)[1] == 3  # explicit slot wins
    assert resolve_my_slot(d, mgrs, user_id=MIKE) == (MIKE, 2)
    assert resolve_my_slot(d, mgrs, username="GLEB") == (GLEB, 1)             # display_name, case-insensitive
    assert resolve_my_slot(d, mgrs, username="Mike_H", users=users_json) == (MIKE, 2)  # username field
    assert resolve_my_slot(d, mgrs, username="hurts so good") == (MIKE, 2)   # team name
    assert resolve_my_slot(d, mgrs, username="nobody", users=users_json) == (None, None)
    assert resolve_my_slot(d, mgrs) == (None, None)


def test_state_from_sleeper_full(draft_json, picks_json, league_json, users_json, rosters_json, traded_picks_json):
    st = state_from_sleeper(draft_json, picks_json, league_json, users_json, rosters_json, traded_picks_json,
                            username="gleb")
    assert st.my_user_id == GLEB and st.my_slot == 1
    assert st.my_roster_id == 4
    assert len(st.picks) == 20 and st.next_pick_no == 21
    assert st.current_round == 2
    assert st.on_the_clock_slot == 4  # round 2 is reversed: pick 21 -> slot 4
    assert st.league is not None and st.league.name == "Fixture League"
    assert st.version == 0
    # traded pick: slot 1 owns its own round-3 pick (#25) AND slot 2's round-3 pick (#26)
    assert st.my_next_pick_no == 24
    assert st.my_pick_after_next == 25
    assert st.my_future_picks()[:4] == [24, 25, 26, 48]
    assert st.picks_until_my_turn == 3
    assert not st.is_my_turn
    assert st.slot_label(1) == "Team Gleb"
    assert len(st.my_picks()) == 1 and st.my_picks()[0].player_id == "7564"


def test_state_from_sleeper_traded_pick_removed_from_original_owner(draft_json, picks_json, users_json,
                                                                   rosters_json, traded_picks_json):
    st = state_from_sleeper(draft_json, picks_json, None, users_json, rosters_json, traded_picks_json,
                            user_id=MIKE)
    assert st.my_slot == 2 and st.league is None
    fut = st.my_future_picks()
    assert fut[:2] == [23, 47]          # #26 (round 3) was traded away
    assert 26 not in st.my_pick_numbers()
    assert len(st.my_pick_numbers()) == 14


def test_state_from_sleeper_without_traded_picks(draft_json, picks_json, users_json):
    st = state_from_sleeper(draft_json, picks_json, users_raw=users_json, user_id=MIKE)
    assert st.my_future_picks()[:3] == [23, 26, 47]


def test_state_from_sleeper_spectator(draft_json, picks_json):
    st = state_from_sleeper(draft_json, picks_json)
    assert st.my_slot is None and st.my_user_id is None
    assert st.my_next_pick_no is None and not st.is_my_turn
    assert len(st.managers) == 12  # placeholders from draft_order
