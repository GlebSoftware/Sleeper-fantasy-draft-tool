"""ESPN league / draft / team / pick JSON -> models (draftadvisor.espn.parsing + ids)."""
from __future__ import annotations

import copy
import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from draftadvisor.espn.ids import EspnIdMap, espn_player_fields, espn_position, placeholder_player
from draftadvisor.espn.parsing import (
    board_pick_number,
    draft_epoch_ms,
    draft_status,
    fresh_draft_spots,
    league_status,
    merge_league_payload,
    normalize_swid,
    parse_espn_draft,
    parse_espn_league,
    parse_espn_managers,
    parse_espn_picks,
    reconstruct_roster_picks,
    resolve_my_team,
    roster_names,
    roster_spots,
    roster_positions_from_counts,
    rostered_espn_ids,
    rostered_ids,
    rounds_from_counts,
    state_from_espn,
)
from draftadvisor.models import DraftState, Player, RosterSpot

FIXTURES = Path(__file__).parent / "fixtures" / "espn"
SWID_TEAM_1 = "{6863-6934-3455}"        # Goin' HAM Newton, slot 3
SWID_TEAM_2 = "{55791-3368-10456}"      # Rollin' With Mahomies, slot 1
PICK_ORDER = [2, 8, 1, 9, 4, 3, 10, 5, 7, 11]


def load(name: str):
    with open(FIXTURES / name, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture
def league():
    return load("league_settings_teams.json")


@pytest.fixture
def modern():
    return load("league_modern.json")


@pytest.fixture
def complete():
    return load("draft_complete.json")


@pytest.fixture
def in_progress():
    return load("draft_in_progress.json")


@pytest.fixture
def pre_draft():
    return load("draft_pre_draft.json")


@pytest.fixture
def kona():
    return load("players_kona.json")["players"]


def universe() -> dict[str, Player]:
    """A tiny canonical universe: one ESPN-id match, one name match, two defenses."""
    pls = [
        Player("gurley", "Todd Gurley", "RB", "LAR", espn_id="2977644"),
        Player("ab", "Antonio Brown", "WR", "PIT"),
        Player("JAX", "JAX Defense", "DEF", "JAX"),
        Player("WAS", "WAS Defense", "DEF", "WAS"),
        Player("dup1", "Josh Allen", "QB", "BUF"),
        Player("dup2", "Josh Allen", "QB", "MIA"),
    ]
    return {p.player_id: p for p in pls}


# ---------------------------------------------------------------------------
# league
# ---------------------------------------------------------------------------


def test_parse_league(league, complete):
    lg = parse_espn_league(league, complete)
    assert lg.league_id == "368876" and lg.name == "FXBG League" and lg.season == 2018
    assert lg.total_rosters == 10
    assert lg.roster_positions == ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF"] + ["BN"] * 6
    assert lg.scoring_type == "ppr" and lg.te_premium == 0 and not lg.is_superflex
    assert lg.scoring_settings["pass_td"] == 6.0 and lg.roster_size == 15
    assert lg.draft_id == "espn-368876-2018" and lg.status == "in_season"
    s = lg.settings
    assert s["platform"] == "espn" and s["keeper_count"] == 0 and s["max_keepers"] == 0 and s["is_public"] is False
    assert s["draft_type"] == "OFFLINE" and s["scoring_type_espn"] == "H2H_POINTS" and s["num_teams"] == 10
    assert s["unmapped_scoring"] == [{"statId": 209, "label": "1pt Safety", "points": 1.0}]
    assert s["pick_order"] == PICK_ORDER and s["time_per_selection"] == 90
    assert lg.raw is not league and lg.raw["id"] == 368876


def test_parse_league_modern_shape(modern):
    lg = parse_espn_league(modern)
    assert lg.roster_positions == ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "SUPER_FLEX", "K", "DEF"] + ["BN"] * 5
    assert lg.is_superflex and lg.te_premium == 0.5 and lg.scoring_settings["rec"] == 1.0
    assert lg.scoring_settings["pass_yd"] == pytest.approx(0.08)      # 0.04 + "every 5 yards" 0.2
    assert lg.status == "pre_draft" and lg.season == 2026
    assert [u["statId"] for u in lg.settings["unmapped_scoring"]] == [15, 209]
    assert lg.settings["keeper_count"] == 2 and lg.settings["max_keepers"] == 2 and lg.settings["draft_type"] == "SNAKE"


def test_parse_league_tolerates_missing_fields():
    lg = parse_espn_league({})
    assert lg.league_id == "" and lg.total_rosters == 10 and lg.roster_positions == [] and lg.scoring_settings == {}
    assert lg.status is None and lg.settings["platform"] == "espn"
    lg = parse_espn_league({"id": 7, "settings": {"size": "12", "rosterSettings": {"lineupSlotCounts": {"0": "1", "20": 2, "x": 1, "18": 1}}}})
    assert lg.league_id == "7" and lg.total_rosters == 12 and lg.roster_positions == ["QB", "BN", "BN"]
    assert lg.name == "ESPN league 7"


def test_roster_positions_and_rounds():
    counts = {"0": 1, "2": 2, "4": 2, "6": 1, "23": 1, "17": 1, "16": 1, "20": 6, "21": 2, "7": 1, "3": 1, "5": 1,
              "8": 1, "10": 2, "14": 1, "15": 1, "19": 1, "18": 1, "24": 1, "25": 1, "1": 1}
    slots = roster_positions_from_counts(counts)
    assert slots == ["QB", "QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "WRRB_FLEX", "REC_FLEX", "SUPER_FLEX", "K", "DEF",
                     "DL", "LB", "LB", "DB", "IDP_FLEX"] + ["BN"] * 8 + ["IR"] * 2
    assert rounds_from_counts(counts) == sum(int(v) for k, v in counts.items() if k != "21")
    assert rounds_from_counts({}) == 0 and roster_positions_from_counts(None) == []
    assert rounds_from_counts({"0": -1, "2": "x", "21": 5}) == 0


# ---------------------------------------------------------------------------
# draft
# ---------------------------------------------------------------------------


def test_parse_draft_complete(league, complete):
    d = parse_espn_draft(league, complete)
    assert d.draft_id == "espn-368876-2018" and d.league_id == "368876" and d.season == 2018
    assert d.type == "snake" and d.status == "complete" and d.reversal_round == 0 and d.player_type == 0
    assert d.teams == 10 and d.rounds == 15 and d.total_picks == 150 and d.pick_timer == 90
    assert d.draft_order == {str(t): i + 1 for i, t in enumerate(PICK_ORDER)}
    assert d.slot_to_roster_id == {i + 1: t for i, t in enumerate(PICK_ORDER)}
    assert all(isinstance(k, str) for k in d.draft_order) and all(isinstance(k, int) for k in d.slot_to_roster_id)
    assert d.start_time is None and d.traded_picks == {} and d.scoring_type == "ppr"
    assert d.metadata == {"name": "FXBG League", "platform": "espn", "scoring_type": "ppr"}
    assert d.settings["teams"] == 10 and d.settings["rounds"] == 15 and d.settings["pick_timer"] == 90
    assert d.settings["pickOrder"] == PICK_ORDER and "picks" not in d.raw["draftDetail"]
    # snake math: team 11 (slot 10) opens round 2
    assert d.slot_for_pick(11) == 10 and d.pick_no_for(2, 10) == 11 and d.picks_for_slot(3)[:4] == [3, 18, 23, 38]


@pytest.mark.parametrize("name, status", [("draft_complete.json", "complete"), ("draft_in_progress.json", "drafting"),
                                          ("draft_pre_draft.json", "pre_draft")])
def test_status_mapping(league, name, status):
    dj = load(name)
    assert draft_status(dj["draftDetail"]) == status
    assert parse_espn_draft(league, dj).status == status
    assert parse_espn_league(league, dj).status == {"complete": "in_season"}.get(status, status)
    assert league_status(None) is None and league_status({}) is None


def test_parse_draft_prefers_the_per_poll_settings(league, in_progress):
    dj = copy.deepcopy(in_progress)
    dj["settings"]["draftSettings"]["timePerSelection"] = 45
    dj["settings"]["draftSettings"]["pickOrder"] = list(reversed(PICK_ORDER))
    d = parse_espn_draft(league, dj)
    assert d.pick_timer == 45 and d.slot_to_roster_id[1] == 11 and d.type == "snake" and d.status == "drafting"
    assert d.start_time == 1535198400000
    # draft type auction
    dj["settings"]["draftSettings"]["type"] = "AUCTION"
    assert parse_espn_draft(league, dj).type == "auction"


def test_parse_league_prefers_the_per_poll_settings_too(league, in_progress):
    """League and draft objects must agree after a mid-draft commissioner change (bench slot, clock, order)."""
    dj = copy.deepcopy(in_progress)
    dj["settings"]["rosterSettings"]["lineupSlotCounts"]["20"] = 7
    dj["settings"]["draftSettings"]["timePerSelection"] = 30
    dj["settings"]["draftSettings"]["pickOrder"] = list(reversed(PICK_ORDER))
    lg, d = parse_espn_league(league, dj), parse_espn_draft(league, dj)
    assert lg.roster_size == 16 and d.rounds == 16 and lg.roster_positions.count("BN") == 7
    assert lg.settings["time_per_selection"] == 30 and d.pick_timer == 30
    assert lg.settings["pick_order"] == list(reversed(PICK_ORDER)) and lg.settings["lineup_slot_counts"]["20"] == 7
    st = state_from_espn(league, dj, EspnIdMap(), swid=SWID_TEAM_1)
    assert st.league.roster_size == st.draft.rounds == 16 and st.my_slot == 8


def test_team_count_never_shrinks_to_a_partial_pick_order(league, in_progress):
    """ESPN exposes pickOrder while teams are still joining: settings.size wins over a shorter order."""
    lj, dj = copy.deepcopy(league), copy.deepcopy(in_progress)
    for payload in (lj, dj):
        payload["settings"]["draftSettings"]["pickOrder"] = [2, 8, 1]
    d = parse_espn_draft(lj, dj)
    assert d.teams == 10 and d.rounds == 15 and d.total_picks == 150 and parse_espn_league(lj, dj).total_rosters == 10
    assert d.slot_to_roster_id == {1: 2, 2: 8, 3: 1} and d.draft_order == {"2": 1, "8": 2, "1": 3}
    st = state_from_espn(lj, dj, EspnIdMap(), slot=3)
    assert st.my_slot == 3 and st.my_user_id == "1" and st.next_pick_no == 18 and st.on_the_clock_slot == 3
    assert st.my_future_picks()[:3] == [18, 23, 38] and st.is_my_turn
    # a longer order than settings.size wins (the size is the stale value then); no size at all -> the order
    for payload in (lj, dj):
        payload["settings"]["draftSettings"]["pickOrder"] = PICK_ORDER
    lj["settings"]["size"] = 8
    assert parse_espn_draft(lj, dj).teams == 10 and parse_espn_league(lj, dj).total_rosters == 10
    assert parse_espn_draft({"settings": {"draftSettings": {"pickOrder": [1, 2, 3, 4]}}}).teams == 4


def test_traded_picks_from_the_board(league, in_progress):
    """Board entries owned by another team than the snake owner populate ``traded_picks``."""
    dj = copy.deepcopy(in_progress)
    picks = dj["draftDetail"]["picks"]
    # pick 18 (round 2, slot 3 = team 1) belongs to team 2 now; pick 20 stays with its owner; junk owners ignored
    picks.append({"overallPickNumber": 18, "roundId": 2, "roundPickNumber": 3, "teamId": 0, "owningTeamIds": [2], "playerId": 0})
    picks.append({"overallPickNumber": 20, "roundId": 2, "roundPickNumber": 1, "teamId": 2, "owningTeamIds": [2], "playerId": 0})
    picks.append({"overallPickNumber": 21, "roundId": 3, "teamId": 0, "owningTeamIds": [], "playerId": 0})
    picks.append({"overallPickNumber": 22, "roundId": 3, "teamId": 12345, "playerId": 0})
    d = parse_espn_draft(league, dj)
    assert d.traded_picks == {(2, 1): 2} and d.owner_roster_for_pick(18) == 2 and d.owner_roster_for_pick(20) == 2
    assert d.status == "drafting"
    me = state_from_espn(league, dj, EspnIdMap(), swid=SWID_TEAM_1)              # team 1, slot 3
    assert me.next_pick_no == 18 and me.on_the_clock_roster == 2 and not me.is_my_turn
    assert me.my_future_picks()[:2] == [23, 38] and me.picks_until_my_turn == 5
    them = state_from_espn(league, dj, EspnIdMap(), team_id=2)                  # team 2, slot 1
    assert them.is_my_turn and them.my_future_picks()[:3] == [18, 20, 21]
    # a made pick by another team than the snake owner is a trade too (its Pick already carries the acquirer)
    dj2 = copy.deepcopy(in_progress)
    dj2["draftDetail"]["picks"].append({"overallPickNumber": 18, "roundId": 2, "teamId": 2, "owningTeamIds": [2], "playerId": 777})
    d2 = parse_espn_draft(league, dj2)
    assert d2.traded_picks == {(2, 1): 2}
    p18 = next(p for p in parse_espn_picks(dj2, EspnIdMap(), d2) if p.pick_no == 18)
    assert p18.draft_slot == 1 and p18.roster_id == 2
    # with pickOrder withheld the order still comes from the board, so the trade stays visible
    no_order = copy.deepcopy(dj)
    no_order["settings"]["draftSettings"]["pickOrder"] = []
    derived = parse_espn_draft({"settings": {"size": 10, "draftSettings": {"pickOrder": []}}}, no_order)
    assert [derived.slot_to_roster_id[s] for s in range(1, 11)] == PICK_ORDER
    assert derived.traded_picks == {(2, 1): 2}
    # a board that cannot reveal the order (round 1 incomplete) leaves the order and the trades unknown
    blind = copy.deepcopy(no_order)
    blind["draftDetail"]["picks"] = [p for p in blind["draftDetail"]["picks"]
                                     if (p.get("overallPickNumber") or 0) > 6]
    blind_draft = parse_espn_draft({"settings": {"size": 10, "draftSettings": {"pickOrder": []}}}, blind)
    assert blind_draft.slot_to_roster_id == {} and blind_draft.traded_picks == {}
    # nothing for auctions or on the untouched fixtures
    dj["settings"]["draftSettings"]["type"] = "AUCTION"
    assert parse_espn_draft(league, dj).traded_picks == {}
    assert parse_espn_draft(league, in_progress).traded_picks == {} and parse_espn_draft(league, load("draft_complete.json")).traded_picks == {}


def test_draft_status_from_the_board(league, in_progress, pre_draft):
    """Real picks on the board mean drafting even when ESPN never sets inProgress (OFFLINE / commissioner)."""
    dj = copy.deepcopy(in_progress)
    dj["draftDetail"]["inProgress"] = False
    assert draft_status(dj["draftDetail"]) == "drafting" and parse_espn_draft(league, dj).status == "drafting"
    st = state_from_espn(league, dj, EspnIdMap(), swid=SWID_TEAM_1)
    assert st.draft.status == "drafting" and st.is_my_turn and st.league.status == "drafting"
    # keeper-only / empty / pre-populated boards are still pre-draft
    assert draft_status({"picks": [{"overallPickNumber": 3, "playerId": 15825, "keeper": True},
                                   {"overallPickNumber": 14, "playerId": 0, "reservedForKeeper": True}]}) == "pre_draft"
    assert draft_status({"picks": [{"overallPickNumber": n, "playerId": 0, "teamId": 2} for n in range(1, 151)]}) == "pre_draft"
    assert draft_status(pre_draft["draftDetail"]) == "pre_draft" and draft_status({"picks": ["junk"]}) == "pre_draft"
    assert draft_status({"drafted": True, "picks": []}) == "complete"


def test_parse_draft_without_order_or_detail(league):
    lj = copy.deepcopy(league)
    lj["settings"]["draftSettings"]["pickOrder"] = []
    d = parse_espn_draft(lj)
    assert d.status == "pre_draft" and d.teams == 10 and d.rounds == 15 and d.draft_order == {} and d.slot_to_roster_id == {}
    st = DraftState(draft=d, picks=[])
    assert st.next_pick_no == 1 and not st.is_complete and st.my_next_pick_no is None
    d0 = parse_espn_draft({})
    assert d0.teams == 10 and d0.rounds == 15 and d0.status == "pre_draft" and d0.draft_id == "espn--2026"


# ---------------------------------------------------------------------------
# managers
# ---------------------------------------------------------------------------


def test_parse_managers(league, complete):
    d = parse_espn_draft(league, complete)
    m = parse_espn_managers(league, d)
    assert len(m) == 10 and set(m) == {str(t) for t in PICK_ORDER}
    t1 = m["1"]
    assert t1.display_name == "ijgdgvhhj" and t1.team_name == "Goin' HAM Newton" and t1.slot == 3 and t1.roster_id == 1
    assert t1.avatar.startswith("https://")
    assert m["2"].slot == 1 and m["11"].slot == 10 and m["3"].team_name == "Feel the Brees"
    # a team that is in the pick order but not in ``teams`` gets a placeholder
    lj = copy.deepcopy(league)
    lj["teams"] = [t for t in lj["teams"] if t["id"] != 11]
    m2 = parse_espn_managers(lj, d)
    assert m2["11"].display_name == "Team 10" and m2["11"].slot == 10 and m2["11"].roster_id == 11
    # an owner without a members entry falls back to the abbreviation
    lj["members"] = []
    assert parse_espn_managers(lj, d)["1"].display_name == "GHN"


def test_parse_managers_modern_owner_dicts(modern):
    d = parse_espn_draft(modern)
    m = parse_espn_managers(modern, d)
    assert m["1"].display_name == "ijgdgvhhj" and m["1"].team_name == "Goin' HAM Newton"
    mj = copy.deepcopy(modern)
    mj["members"] = []                                        # names come from the owner dicts alone
    assert parse_espn_managers(mj, d)["2"].display_name == "gerhfdhfg"
    for t in mj["teams"]:
        for o in t["owners"]:
            o.pop("displayName")
    assert parse_espn_managers(mj, d)["2"].display_name == "hgfdhf hfgd" or parse_espn_managers(mj, d)["2"].display_name


# ---------------------------------------------------------------------------
# picks / ids
# ---------------------------------------------------------------------------


def test_parse_picks_complete(league, complete, kona):
    from draftadvisor.espn.capture import espn_names

    d = parse_espn_draft(league, complete)
    m = EspnIdMap.from_players(universe())
    picks = parse_espn_picks(complete, m, d, espn_names(kona))
    assert len(picks) == 150 and [p.pick_no for p in picks] == list(range(1, 151))
    p1, p2, p3 = picks[0], picks[1], picks[2]
    assert (p1.round, p1.draft_slot, p1.roster_id, p1.picked_by) == (1, 1, 2, "2")
    assert p2.player_id == "gurley" and p2.player_name == "Todd Gurley II" and p2.position == "RB" and p2.metadata["team"] == "LAR"
    assert p3.player_id == "ab" and m.espn_to_pid["13934"] == "ab"                  # learned from the name match
    assert EspnIdMap.is_synthetic(p1.player_id) and p1.player_id == "espn:15825"    # not in the universe / player pool
    assert p1.metadata == {"espn_id": "15825"} and p1.player_name == "espn:15825"
    # round 2 opens with slot 10 (team 11)
    p11 = picks[10]
    assert p11.round == 2 and p11.draft_slot == 10 and p11.roster_id == 11
    # team defenses: -16030 = JAX (in the universe), -16021 = PHI (not in it: still the abbreviation)
    by_no = {p.pick_no: p for p in picks}
    assert by_no[104].player_id == "JAX" and by_no[104].position == "DEF"
    assert by_no[112].player_id == "PHI" and by_no[112].metadata["espn_id"] == "-16021"
    assert not any(p.is_keeper for p in picks)


def test_parse_picks_keepers_and_fallbacks(league, in_progress):
    d = parse_espn_draft(league, in_progress)
    picks = parse_espn_picks(in_progress, EspnIdMap(), d)
    assert len(picks) == 17
    assert [p.pick_no for p in picks if p.is_keeper] == [3, 14]
    # bare draftDetail dict, duplicates (last wins), empty keeper slots and junk are handled
    detail = {"picks": [
        {"overallPickNumber": 1, "roundId": 1, "roundPickNumber": 1, "teamId": 2, "playerId": 15825},
        {"overallPickNumber": 1, "roundId": 1, "roundPickNumber": 1, "teamId": 2, "playerId": 99},
        {"overallPickNumber": 2, "roundId": 1, "teamId": 8, "playerId": 0, "reservedForKeeper": True},
        {"overallPickNumber": 12, "teamId": 12345, "roundPickNumber": 2, "playerId": 5},
        {"teamId": 2, "playerId": 7}, "junk",
    ]}
    picks = parse_espn_picks(detail, EspnIdMap(), d)
    assert [(p.pick_no, p.player_id, p.draft_slot, p.round) for p in picks] == [(1, "espn:99", 1, 1), (12, "espn:5", 9, 2)]
    # unknown pick order: the snake slot for the pick number
    d_no_order = parse_espn_draft({"settings": {"size": 10, "draftSettings": {"pickOrder": []}}}, in_progress)
    picks = parse_espn_picks(in_progress, EspnIdMap(), d_no_order)
    assert picks[10].draft_slot == 10 and picks[0].draft_slot == 1
    assert parse_espn_picks(None, EspnIdMap(), d) == []


def test_id_map_resolution():
    m = EspnIdMap.from_players(universe())
    assert m.resolve("2977644") == "gurley" and m.resolve(2977644) == "gurley"
    assert m.resolve("-16030") == "JAX" and m.resolve(-16028) == "WAS" and m.resolve("WSH") == "WAS"
    assert m.resolve("-16023") == "PIT"                                  # no PIT DEF in the universe: abbreviation
    assert m.resolve(None, name="Steelers D/ST", position="DEF", pro_team_id=23) == "PIT"
    assert m.resolve("13934", name="Antonio Brown Jr.", position="WR") == "ab" and m.espn_to_pid["13934"] == "ab"
    assert m.resolve("13934") == "ab"
    assert m.resolve("13934", name="Antonio Brown", position="RB") == "ab"     # cached id wins
    assert m.resolve("1", name="Antonio Brown", position="RB") == "espn:1"     # position must match
    assert m.resolve("2", name="Josh Allen", position="QB") == "espn:2"        # ambiguous name without team
    assert m.resolve("3", name="Josh Allen", position="QB", pro_team_id=2) == "dup1"
    assert m.resolve(None) == "espn:unknown" and m.resolve(None, name="No Body", position="TE") == "espn:no-body"
    assert EspnIdMap.is_synthetic("espn:5") and not EspnIdMap.is_synthetic("4034")
    assert len(m) >= 5
    m.register(42, "gurley")
    assert m.resolve(42) == "gurley"
    assert m.resolve_player_json({"id": 2977644, "player": {"id": 2977644, "fullName": "Todd Gurley II"}}) == "gurley"


def test_id_map_keeps_namesakes_apart():
    """A name match never claims a universe player who is already known by a different ESPN id."""
    m = EspnIdMap.from_players({"a": Player("a", "Mike Williams", "WR", "NYJ", espn_id="111"),
                                "sr": Player("sr", "Marvin Harrison", "WR", "IND", espn_id="1"),
                                "noid": Player("noid", "Kyle Williams", "WR", "DEN")})
    assert m.pid_to_espn == {"a": "111", "sr": "1"}
    assert m.resolve("111", name="Mike Williams", position="WR") == "a"
    assert m.resolve("222", name="Mike Williams", position="WR", pro_team_id=20) == "espn:222"
    assert m.resolve("223", name="Mike Williams", position="WR", pro_team_id=24) == "espn:223"
    assert "222" not in m.espn_to_pid and "223" not in m.espn_to_pid and m.espn_to_pid["111"] == "a"
    assert m.resolve("4432708", name="Marvin Harrison Jr.", position="WR", pro_team_id=22) == "espn:4432708"
    # a player without a known ESPN id is learned by name once, and then held by that id
    assert m.resolve("4613202", name="Kyle Williams", position="WR", pro_team_id=17) == "noid"
    assert m.pid_to_espn["noid"] == "4613202" and m.espn_to_pid["4613202"] == "noid"
    assert m.resolve("13488", name="Kyle Williams", position="WR", pro_team_id=7) == "espn:13488"
    assert m.resolve("4613202") == "noid"
    # a map built from a mapping alone gets the reverse index for numeric ids
    m2 = EspnIdMap(espn_to_pid={"111": "a", "JAX": "JAX"})
    assert m2.pid_to_espn == {"a": "111"}


def test_player_fields_and_placeholder(kona):
    entry = next(p for p in kona if p["player"]["fullName"] == "Antonio Brown")
    f = espn_player_fields(entry)
    assert f["espn_id"] == "13934" and f["position"] == "WR" and f["team"] == "PIT" and f["pro_team_id"] == 23
    assert f["first_name"] == "Antonio" and f["last_name"] == "Brown" and f["injury_status"] is None
    pl = placeholder_player(entry)
    assert pl.player_id == "espn:13934" and pl.name == "Antonio Brown" and pl.position == "WR" and pl.team == "PIT"
    assert pl.espn_id == "13934" and pl.fantasy_positions == ("WR",) and pl.metadata["placeholder"] is True
    dst = placeholder_player({"id": -16023, "fullName": "Steelers D/ST", "defaultPositionId": 16, "proTeamId": 23})
    assert dst.player_id == "PIT" and dst.position == "DEF" and dst.team == "PIT" and dst.espn_id == "-16023"
    assert placeholder_player({"id": -16003}).name == "CHI Defense"
    weird = placeholder_player({"id": 5, "firstName": "A", "lastName": "B", "eligibleSlots": [6, 20], "proTeamId": 28,
                                "injuryStatus": "QUESTIONABLE"})
    assert weird.name == "A B" and weird.position == "TE" and weird.team == "WAS" and weird.injury_status == "Questionable"
    assert espn_position({"defaultPositionId": 5}) == "K" and espn_position({"id": -16001}) == "DEF"
    assert placeholder_player(None).player_id == "espn:unknown" and placeholder_player(None).position == "UNK"
    assert espn_position({"defaultPositionId": 99}) is None


def test_idp_punter_and_coach_never_become_skill_players():
    """An LB / DT / CB / P / HC keeps a real label (never WR) - by defaultPositionId or by eligibleSlots."""
    lb = {"id": 4999999001, "player": {"id": 4999999001, "fullName": "Roquan Smith", "defaultPositionId": 11,
                                       "eligibleSlots": [10, 15, 20, 21], "proTeamId": 33}}
    assert espn_position(lb) == "LB"
    pl = placeholder_player(lb)
    assert pl.position == "LB" and pl.fantasy_positions == ("LB",) and pl.team == "BAL" and pl.player_id == "espn:4999999001"
    assert espn_position({"defaultPositionId": 9, "eligibleSlots": [8, 11, 15]}) == "DL"
    assert espn_position({"defaultPositionId": 12}) == "DB" and espn_position({"defaultPositionId": 13}) == "DB"
    assert espn_position({"defaultPositionId": 7, "eligibleSlots": [18, 20, 21]}) == "P"
    assert espn_position({"defaultPositionId": 14}) == "HC"
    assert espn_position({"eligibleSlots": [10, 15, 20]}) == "LB" and espn_position({"eligibleSlots": [18, 20]}) == "P"
    assert placeholder_player({"id": 7, "fullName": "No Position"}).position == "UNK"


# ---------------------------------------------------------------------------
# rosters
# ---------------------------------------------------------------------------


def test_rostered_ids(league, modern):
    ids = rostered_espn_ids(league)
    assert len(ids) == 150 and "13934" in ids
    m = EspnIdMap.from_players(universe())
    res = rostered_ids(league, m)
    assert "ab" in res and "gurley" in res and "JAX" in res and len(res) == 150
    names = roster_names(league)
    assert names["13934"]["name"] == "Antonio Brown" and names["13934"]["position"] == "WR"
    assert len(rostered_espn_ids(modern)) == 20 and rostered_ids({}, m) == set() and rostered_espn_ids(None) == set()


# ---------------------------------------------------------------------------
# who am I
# ---------------------------------------------------------------------------


def test_resolve_my_team(league, complete):
    d = parse_espn_draft(league, complete)
    m = parse_espn_managers(league, d)
    r = lambda **kw: resolve_my_team(d, m, league, **kw)  # noqa: E731
    assert r(swid=SWID_TEAM_1) == ("1", 3)
    assert r(swid="6863-6934-3455") == ("1", 3) and r(swid="{6863-6934-3455}".lower()) == ("1", 3)
    assert r(swid="{4C6935DA-0CBA-4AF9-BB08-06E583265D96}") == ("9", 4) and r(swid="4c6935da-0cba-4af9-bb08-06e583265d96") == ("9", 4)
    assert r(swid="{0000-0000-0000}") == (None, None)
    assert r(username="Rollin' With Mahomies") == ("2", 1) and r(username="rollin' with mahomies") == ("2", 1)
    assert r(username="GHN") == ("1", 3) and r(username="ijgdgvhhj") == ("1", 3) and r(username="hrthr nvbn") == ("5", 8)
    assert r(username="hrthr") == ("5", 8) and r(username="nobody") == (None, None) and r(username="11") == ("11", 10)
    assert r(team_id=11) == ("11", 10) and r(team_id="11") == ("11", 10) and r(team_id=99) == (None, None)
    assert r(slot=3) == ("1", 3) and r(slot="10") == ("11", 10)
    # precedence: slot > team_id > swid > username
    assert r(slot=1, team_id=11, swid=SWID_TEAM_1, username="GHN") == ("2", 1)
    assert r(team_id=11, swid=SWID_TEAM_1) == ("11", 10) and r(swid=SWID_TEAM_1, username="LUTZ") == ("1", 3)
    assert r() == (None, None) and normalize_swid(None) == "" and normalize_swid(" {Ab-c} ") == "AB-C"


def test_resolve_my_team_rejects_slots_outside_the_league(league, complete, caplog):
    """slot 11 of a 10-team league would alias slot 10's pick numbers (false 'my turn' alerts): ignored."""
    d = parse_espn_draft(league, complete)
    m = parse_espn_managers(league, d)
    with caplog.at_level("WARNING", logger="draftadvisor.espn.parsing"):
        assert resolve_my_team(d, m, league, slot=11) == (None, None)
        assert resolve_my_team(d, m, league, slot=0) == (None, None) and resolve_my_team(d, m, league, slot=-1) == (None, None)
    assert "outside 1..10" in caplog.text
    # the other hints still identify the team
    assert resolve_my_team(d, m, league, slot=11, swid=SWID_TEAM_1) == ("1", 3)
    assert resolve_my_team(d, m, league, slot=0, team_id=11) == ("11", 10)
    st = state_from_espn(league, load("draft_in_progress.json"), EspnIdMap(), slot=11)
    assert st.my_slot is None and st.my_user_id is None and not st.is_my_turn and st.my_pick_numbers() == []


def test_resolve_my_team_without_pick_order(league):
    lj = copy.deepcopy(league)
    lj["settings"]["draftSettings"]["pickOrder"] = []
    d = parse_espn_draft(lj)
    m = parse_espn_managers(lj, d)
    assert resolve_my_team(d, m, lj, swid=SWID_TEAM_1) == ("1", None)
    assert resolve_my_team(d, m, lj, team_id=8) == ("8", None) and resolve_my_team(d, m, lj, slot=2) == (None, 2)
    # an explicit slot does not discard the other identity hints while the order is unknown
    assert resolve_my_team(d, m, lj, slot=3, swid=SWID_TEAM_1) == ("1", 3)
    assert resolve_my_team(d, m, lj, slot=3, team_id=8) == ("8", 3)
    assert resolve_my_team(d, m, lj, slot=3, username="LUTZ") == ("8", 3)
    assert resolve_my_team(d, m, lj, slot=3, username="nobody") == (None, 3)
    assert resolve_my_team(d, m, lj, slot=3, team_id=99) == (None, 3)
    assert resolve_my_team(d, m, lj, slot=11, swid=SWID_TEAM_1) == ("1", None)      # out of range: ignored
    # a partial order (teams still joining): the slot outside it is kept, the team learned from the SWID
    lj["settings"]["draftSettings"]["pickOrder"] = [2, 8]
    d2 = parse_espn_draft(lj)
    m2 = parse_espn_managers(lj, d2)
    assert d2.teams == 10 and resolve_my_team(d2, m2, lj, slot=1) == ("2", 1)
    assert resolve_my_team(d2, m2, lj, slot=5, swid=SWID_TEAM_1) == ("1", 5)


def test_resolve_my_team_modern(modern):
    d = parse_espn_draft(modern)
    m = parse_espn_managers(modern, d)
    assert resolve_my_team(d, m, modern, swid="55791-3368-10456") == ("2", 1)
    assert resolve_my_team(d, m, modern, username="gerhfdhfg") == ("2", 1)
    assert resolve_my_team(d, m, modern, username="Rollin' With Mahomies") == ("2", 1)


# ---------------------------------------------------------------------------
# full state
# ---------------------------------------------------------------------------


def test_state_in_progress_is_my_turn(league, in_progress, kona):
    from draftadvisor.espn.capture import espn_names

    m = EspnIdMap.from_players(universe())
    st = state_from_espn(league, in_progress, m, names=espn_names(kona), swid=SWID_TEAM_1)
    assert st.league is not None and st.league.name == "FXBG League" and st.draft.status == "drafting"
    assert st.my_user_id == "1" and st.my_slot == 3 and st.my_roster_id == 1
    assert len(st.picks) == 17 and st.next_pick_no == 18 and st.on_the_clock_slot == 3 and st.on_the_clock_roster == 1
    assert st.is_my_turn and st.picks_until_my_turn == 0
    assert st.my_future_picks()[:3] == [18, 23, 38] and st.my_pick_numbers()[:3] == [3, 18, 23]
    assert len(st.my_picks()) == 1 and st.my_picks()[0].pick_no == 3 and st.my_picks()[0].player_id == "ab"
    assert st.slot_label(3) == "Goin' HAM Newton" and st.slot_label(1) == "Rollin' With Mahomies"
    assert len(st.rostered_ids) == 150 and "ab" in st.rostered_ids and "ab" in st.unavailable_ids
    assert "gurley" in st.drafted_ids and not st.is_complete and st.current_round == 2
    # somebody else: 2 picks away (team 2 / slot 1 picks 20th)
    st2 = state_from_espn(league, in_progress, m, swid=SWID_TEAM_2)
    assert st2.my_slot == 1 and not st2.is_my_turn and st2.picks_until_my_turn == 2
    # spectator
    st3 = state_from_espn(league, in_progress, m)
    assert st3.my_user_id is None and st3.my_slot is None and not st3.is_my_turn and st3.my_future_picks() == []


def test_state_complete_and_pre_draft(league, complete, pre_draft):
    m = EspnIdMap()
    st = state_from_espn(league, complete, m, team_id=11)
    assert st.is_complete and st.draft.status == "complete" and len(st.picks) == 150 and st.my_slot == 10
    assert st.on_the_clock_slot is None and len(st.picks_by_slot()[10]) == 15
    # a redraft league before its draft: the board is empty *and* so are the rosters (the fixture's
    # rosters are that season's finished ones, which are evidence the draft has already been held)
    empty = dict(league, teams=[dict(t, roster={"entries": []}) for t in league["teams"]])
    st0 = state_from_espn(empty, pre_draft, m, username="LUTZ")
    assert st0.draft.status == "pre_draft" and st0.picks == [] and st0.my_slot == 2 and st0.next_pick_no == 1
    assert st0.league.status == "pre_draft" and st0.my_future_picks()[:2] == [2, 19]


def test_state_from_league_json_alone(modern):
    st = state_from_espn(modern, None, EspnIdMap(), username="fergheg")
    assert st.draft.status == "pre_draft" and st.my_user_id == "5" and st.my_slot == 8
    assert len(st.rostered_ids) == 20 and st.picks == [] and st.league.is_superflex


def test_normalize_swid_accepts_url_encoded_cookies():
    """Some browsers display the SWID cookie as %7B...%7D; it must match the raw {…} owner ids."""
    raw = normalize_swid("{6863-6934-3455}")
    assert normalize_swid("%7B6863-6934-3455%7D") == raw
    assert normalize_swid("6863-6934-3455") == raw
    assert normalize_swid(" %7b6863-6934-3455%7d ") == raw


# ---------------------------------------------------------------------------
# Pre-populated boards: ESPN lists every pick before it is made, with playerId -1.
# Counting those as picks filled the board, reported "draft complete" and stopped the poller.
# ---------------------------------------------------------------------------


@pytest.fixture
def prepopulated():
    return load("draft_prepopulated.json")


@pytest.fixture
def prepopulated_live():
    return load("draft_prepopulated_live.json")


def test_placeholder_entries_are_not_picks(league, prepopulated):
    """A board of 150 placeholder entries is an empty board, not a finished draft."""
    detail = prepopulated["draftDetail"]
    assert len(detail["picks"]) == 150 and all(p["playerId"] == -1 for p in detail["picks"])

    draft = parse_espn_draft(league, prepopulated)
    picks = parse_espn_picks(prepopulated, EspnIdMap.from_players({}), draft)
    assert picks == []
    assert draft_status(detail) == "pre_draft"
    assert draft.status == "pre_draft"

    st = state_from_espn(league, prepopulated, EspnIdMap.from_players({}), swid=SWID_TEAM_1)
    assert st.is_complete is False
    assert st.next_pick_no == 1
    assert st.my_slot == 3
    assert st.my_picks() == []


def test_zero_player_id_is_also_a_placeholder(league, prepopulated):
    """Some seasons use 0 instead of -1; both mean "not picked yet"."""
    board = copy.deepcopy(prepopulated)
    for p in board["draftDetail"]["picks"]:
        p["playerId"] = 0
    draft = parse_espn_draft(league, board)
    assert parse_espn_picks(board, EspnIdMap.from_players({}), draft) == []
    assert draft_status(board["draftDetail"]) == "pre_draft"


def test_a_team_defense_is_a_real_pick_despite_its_negative_id(league, prepopulated):
    """D/ST ids are negative (-16000 - proTeamId): they must survive the placeholder filter."""
    board = copy.deepcopy(prepopulated)
    board["draftDetail"]["picks"][0]["playerId"] = -16023          # PIT D/ST
    draft = parse_espn_draft(league, board)
    picks = parse_espn_picks(board, EspnIdMap.from_players({}), draft)
    assert [p.pick_no for p in picks] == [1]
    assert picks[0].player_id == "PIT"
    assert draft_status(board["draftDetail"]) == "drafting"


def test_board_filled_in_place_is_a_live_draft(league, prepopulated_live):
    """17 entries filled, the rest still -1: 17 picks, the draft is running, pick 18 is on the clock."""
    detail = prepopulated_live["draftDetail"]
    assert len(detail["picks"]) == 150
    draft = parse_espn_draft(league, prepopulated_live)
    picks = parse_espn_picks(prepopulated_live, EspnIdMap.from_players({}), draft)
    assert [p.pick_no for p in picks] == list(range(1, 18))
    assert draft.status == "drafting"

    st = state_from_espn(league, prepopulated_live, EspnIdMap.from_players({}), swid=SWID_TEAM_1)
    assert st.is_complete is False
    assert st.next_pick_no == 18
    assert st.my_slot == 3 and st.is_my_turn is True                # slot 3 picks 18th in round 2


def test_pick_order_comes_from_the_board_when_espn_withholds_it(league):
    """ESPN publishes pickOrder late; the pre-populated board already carries each pick's teamId."""
    board = load("draft_prepopulated_no_order.json")
    assert board["settings"]["draftSettings"]["pickOrder"] == []

    draft = parse_espn_draft(league, board)
    assert draft.teams == 10
    assert [draft.slot_to_roster_id[s] for s in range(1, 11)] == PICK_ORDER

    st = state_from_espn(league, board, EspnIdMap.from_players({}), swid=SWID_TEAM_1)
    assert st.my_slot == 3


def test_partial_round_one_never_invents_an_order(league, prepopulated):
    """An incomplete or duplicated round 1 leaves the order unknown rather than guessing it."""
    board = copy.deepcopy(prepopulated)
    board["settings"]["draftSettings"]["pickOrder"] = []
    for p in board["draftDetail"]["picks"][:10]:
        p["teamId"] = 0 if p["overallPickNumber"] > 6 else p["teamId"]
    draft = parse_espn_draft(league, board)
    assert draft.slot_to_roster_id == {}


# ---------------------------------------------------------------------------
# Roster-derived picks: ESPN's REST board stays empty for the whole of a live draft, so a drafted
# player is visible only as a roster entry (see the module docstring of draftadvisor.espn.parsing).
# ---------------------------------------------------------------------------


DRAFT_DATE_MS = 1535198400000       # draftSettings.date of the draft fixtures / league_keepers_rosters


@pytest.fixture
def keepers():
    """A league whose rosters carry every acquisition flavour (see make_fixtures.py)."""
    return load("league_keepers_rosters.json")


def test_roster_spots_carry_team_and_provenance(keepers):
    spots = roster_spots(keepers, EspnIdMap())
    assert len(spots) == 60 and {s.roster_id for s in spots} == {t["id"] for t in keepers["teams"]}
    mine = [s for s in spots if s.roster_id == 1]
    assert [s.acquisition_type for s in mine] == ["DRAFT", "ADD", "TRADE", "DRAFT", "DRAFT", "DRAFT"]
    assert [s.acquired_at for s in mine][:3] == [DRAFT_DATE_MS - 365 * 24 * 3600 * 1000, DRAFT_DATE_MS + 5_000,
                                                 DRAFT_DATE_MS + 6_000]
    assert mine[-1].acquired_at is None and mine[4].lineup_slot_id == 21          # undated, and one on IR
    assert all(s.player_id and s.espn_id for s in mine) and mine[0].name and mine[0].position
    dst = next(s for s in mine if s.espn_id == "-16003")                          # a D/ST id is a real player
    assert dst.position == "DEF" and dst.player_id == "CHI"


def test_fresh_draft_spots_keeps_keepers_and_pickups_out(keepers):
    spots = [s for s in roster_spots(keepers, EspnIdMap()) if s.roster_id == 1]
    threshold = draft_epoch_ms(keepers, None, 2018)
    fresh, undated = fresh_draft_spots(spots, threshold, keeper_count=2)
    # last season's DRAFT (a keeper/dynasty holdover), the ADD, the TRADE and the IR entry are all out;
    # the undated one is counted but not claimed, because this league can have keepers
    assert [s.espn_id for s in fresh] == ["-16003"]
    assert [s.espn_id for s in undated] == [spots[-1].espn_id]
    # with no keepers possible and ESPN saying the draft is running, the undated entry counts too
    fresh2, undated2 = fresh_draft_spots(spots, threshold, keeper_count=0, draft_started=True)
    assert len(fresh2) == 2 and undated2 == []
    # ... but not while ESPN says the draft has not started: nothing then says it is this season's
    assert len(fresh_draft_spots(spots, threshold, keeper_count=0, draft_started=False)[0]) == 1
    # a player the board already flags as a keeper is never a fresh pick
    assert fresh_draft_spots(spots, threshold, keeper_ids=[fresh[0].player_id], keeper_count=2)[0] == []


def test_draft_epoch_ms_falls_back_to_may_of_the_season(keepers):
    assert draft_epoch_ms(keepers, None, 2018) == DRAFT_DATE_MS - 6 * 3600 * 1000
    no_date = copy.deepcopy(keepers)
    no_date["settings"]["draftSettings"]["date"] = 0
    may = draft_epoch_ms(no_date, None, 2018)
    assert may == int(datetime(2018, 5, 1, tzinfo=timezone.utc).timestamp() * 1000)
    assert may < DRAFT_DATE_MS                      # this season's draft is after it ...
    assert may > DRAFT_DATE_MS - 365 * 24 * 3600 * 1000     # ... and last season's is before it


def _spot(pid: str, team: int, at: int | None) -> RosterSpot:
    return RosterSpot(player_id=pid, espn_id=pid, roster_id=team, acquisition_type="DRAFT", acquired_at=at)


def test_reconstruct_roster_picks_confidence_levels(league, pre_draft):
    draft = parse_espn_draft(league, pre_draft)
    assert draft.slot_to_roster_id[1] == PICK_ORDER[0]
    # 'exact': the per-team counts match the first k picks of the snake and the dates order them
    spots = [_spot(f"p{n}", PICK_ORDER[n - 1], DRAFT_DATE_MS + n * 1000) for n in range(1, 6)]
    pairs, conf = reconstruct_roster_picks(spots, draft)
    assert conf == "exact" and [(n, s.player_id) for n, s in pairs] == [(n, f"p{n}") for n in range(1, 6)]
    # ties carry no order: a whole team's picks stamped with the same millisecond is ESPN's OFFLINE
    # (commissioner-entered) shape - the counts can still be right, the order is a guess
    same = [_spot("a", PICK_ORDER[0], DRAFT_DATE_MS), _spot("b", PICK_ORDER[1], DRAFT_DATE_MS),
            _spot("c", PICK_ORDER[2], DRAFT_DATE_MS)]
    pairs2, conf2 = reconstruct_roster_picks(same, draft)
    assert conf2 == "exact" and len(pairs2) == 3          # equal timestamps are permuted to fit the order
    wrong_order = [_spot("a", PICK_ORDER[2], DRAFT_DATE_MS + 1), _spot("b", PICK_ORDER[1], DRAFT_DATE_MS + 2),
                   _spot("c", PICK_ORDER[0], DRAFT_DATE_MS + 3)]
    pairs3, conf3 = reconstruct_roster_picks(wrong_order, draft)
    assert conf3 == "team" and {n for n, _ in pairs3} == {1, 2, 3}
    # 'none': the counts do not match a snake prefix at all (two picks for the team that owns pick 1)
    assert reconstruct_roster_picks([_spot("a", PICK_ORDER[0], 1), _spot("b", PICK_ORDER[0], 2)], draft)[1] == "none"
    # 'none': no pick order published, and nothing at all to reconstruct
    no_order = dataclasses.replace(draft, slot_to_roster_id={}, draft_order={})
    assert reconstruct_roster_picks(spots, no_order) == ([], "none")
    assert reconstruct_roster_picks([], draft) == ([], "none")


def test_roster_spots_never_move_the_clock(league, prepopulated):
    """The regression guard: a roster-derived player is a player, not a pick number. Whatever the
    confidence, the clock, the completeness and my future picks stay board-derived."""
    st = state_from_espn(league, prepopulated, EspnIdMap(), team_id=1)
    assert st.picks == [] and st.next_pick_no == 1 and st.is_complete is False
    assert st.on_the_clock_slot == 1 and st.taken_pick_numbers == set()
    assert len(st.roster_spots) == 150 and len(st.roster_only_ids) == 150      # gone, but not picks
    assert st.roster_only_ids <= st.unavailable_ids and st.my_future_picks()[:2] == [3, 18]
    by_slot = st.roster_spots_by_slot()
    assert sum(len(v) for v in by_slot.values()) == 150 and 0 not in by_slot   # every team is known


def test_state_from_espn_uses_the_polls_rosters_over_the_capture(league, prepopulated):
    """The poll asks for mTeam + mRoster too, and those rosters are this second's: they win over the
    capture, which can be ten minutes old."""
    poll = copy.deepcopy(prepopulated)
    poll["teams"] = [{"id": t["id"], "roster": {"entries": []}} for t in league["teams"]]
    for e in ({"playerId": 13934, "acquisitionType": "DRAFT", "acquisitionDate": DRAFT_DATE_MS + 1000,
               "lineupSlotId": 20},):
        poll["teams"][0]["roster"]["entries"].append(e)
    st = state_from_espn(league, poll, EspnIdMap(), team_id=1)
    assert len(st.roster_spots) == 1 and st.roster_spots[0].espn_id == "13934"
    assert st.draft.status == "drafting" and st.picks == [] and st.next_pick_no == 1   # the board is still empty
    assert st.roster_only_ids == {"espn:13934"} and st.draft.teams == 10
    # without teams in the poll the capture's rosters are used unchanged
    st2 = state_from_espn(league, prepopulated, EspnIdMap(), team_id=1)
    assert len(st2.roster_spots) == 150


def test_merge_league_payload_replaces_teams_wholesale(league, prepopulated):
    poll = dict(prepopulated, teams=[{"id": 1, "roster": {"entries": []}}], members=[{"id": "{x}"}])
    merged = merge_league_payload(league, poll)
    assert [t["id"] for t in merged["teams"]] == [1] and merged["members"] == [{"id": "{x}"}]
    assert merged["settings"] is league["settings"]                     # everything else is the capture's
    assert merge_league_payload(league, prepopulated) is league         # no teams in the poll: unchanged
    assert merge_league_payload(league, {"teams": []}) is league


def test_pick_number_is_derived_when_espn_omits_it(league, in_progress):
    """A board entry with no (or a zero) overallPickNumber still has roundId + roundPickNumber."""
    draft = parse_espn_draft(league, in_progress)
    assert board_pick_number({"roundId": 2, "roundPickNumber": 4}, 10) == 14
    assert board_pick_number({"roundId": 0, "roundPickNumber": 4}, 10) is None
    assert board_pick_number({"roundId": 2, "roundPickNumber": 4}, 0) is None
    payload = copy.deepcopy(in_progress)
    for p in payload["draftDetail"]["picks"]:
        p["overallPickNumber"] = 0                      # ESPN omitted it
    picks = parse_espn_picks(payload, EspnIdMap(), draft, {})
    assert [p.pick_no for p in picks] == list(range(1, 18))


def test_pick_player_can_be_nested_under_player_pool_entry(league, in_progress):
    draft = parse_espn_draft(league, in_progress)
    payload = copy.deepcopy(in_progress)
    for p in payload["draftDetail"]["picks"]:
        p["playerPoolEntry"] = {"player": {"id": p.pop("playerId"), "fullName": "Nested Player",
                                           "defaultPositionId": 2}}
    picks = parse_espn_picks(payload, EspnIdMap(), draft, {})
    assert len(picks) == 17 and picks[0].player_id.startswith("espn:")


def test_keepers_and_holdovers_on_rosters_are_never_new_picks(keepers, prepopulated):
    """A keeper league before its draft: every roster already holds players. They are unavailable (they
    always were), but only an entry that ESPN says was drafted *for this draft* may be reported as one."""
    st = state_from_espn(keepers, prepopulated, EspnIdMap(), team_id=1)
    assert len(st.rostered_ids) == len({s.player_id for s in st.roster_spots})
    fresh = [s for s in st.roster_spots if s.is_fresh]
    # exactly one entry per team is dated after this draft's start; everything else is older or undated
    assert len(fresh) == 10 and {s.acquired_at for s in fresh} == {DRAFT_DATE_MS + 7_000}
    assert {s.roster_id for s in fresh} == {t["id"] for t in keepers["teams"]} and st.picks == []
    # the undated entry, the ADD, the TRADE, the IR entry and last season's draft are all left alone
    assert len(st.roster_spots) == 60 and st.roster_only_ids == set(st.rostered_ids)
    assert st.next_pick_no == 1 and st.is_complete is False    # and none of them touched the clock
