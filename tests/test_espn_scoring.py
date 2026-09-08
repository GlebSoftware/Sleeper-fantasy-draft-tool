"""ESPN scoringItems / projected stats -> Sleeper keys (draftadvisor.espn.scoring)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from draftadvisor.espn.constants import ESPN_STAT_TO_SLEEPER, SCORING_LABELS, label_for_stat
from draftadvisor.espn.scoring import (
    effective_points,
    espn_scoring_to_sleeper,
    espn_stats_to_sleeper,
    projected_points,
    projected_season_stats,
)
from draftadvisor.scoring import SLEEPER_STAT_KEYS, ScoringEngine

FIXTURES = Path(__file__).parent / "fixtures" / "espn"


def load(name: str):
    with open(FIXTURES / name, encoding="utf-8") as fh:
        return json.load(fh)


def item(stat_id: int, points: float, overrides: dict | None = None) -> dict:
    d = {"statId": stat_id, "points": points, "isReverseItem": False}
    if overrides:
        d["pointsOverrides"] = overrides
    return d


# ---------------------------------------------------------------------------
# table sanity
# ---------------------------------------------------------------------------


def test_table_only_uses_known_sleeper_keys():
    known = set(SLEEPER_STAT_KEYS)
    for sid, (keys, divisor) in ESPN_STAT_TO_SLEEPER.items():
        assert keys and divisor > 0, sid
        for k in keys:
            assert k in known, (sid, k)
        assert sid in SCORING_LABELS, sid


def test_labels():
    assert label_for_stat(209) == "1pt Safety"
    assert label_for_stat(53) == "Each reception"
    assert label_for_stat(9999) == "statId 9999" and label_for_stat("x") == "statId x"


# ---------------------------------------------------------------------------
# scoring settings
# ---------------------------------------------------------------------------


def test_fixture_league_scoring_matches_table():
    items = load("league_settings_teams.json")["settings"]["scoringSettings"]["scoringItems"]
    scoring, unmapped = espn_scoring_to_sleeper(items)
    expect = {
        "pass_yd": 0.04, "pass_td": 6.0, "pass_int": -2.0, "pass_2pt": 2.0,
        "rush_yd": 0.1, "rush_td": 6.0, "rush_2pt": 2.0,
        "rec": 1.0, "rec_yd": 0.1, "rec_td": 6.0, "rec_2pt": 2.0,
        "fum_lost": -2.0, "fum_rec_td": 6.0,
        "fgm_0_19": 3.0, "fgm_20_29": 3.0, "fgm_30_39": 3.0, "fgm_40_49": 4.0, "fgm_50p": 5.0,
        "fgmiss": -1.0, "xpm": 1.0, "xpmiss": -1.0,
        "def_td": 6.0, "st_td": 6.0, "def_st_td": 6.0, "def_2pt": 2.0, "int": 2.0, "fum_rec": 2.0, "blk_kick": 2.0,
        "safe": 2.0, "sack": 1.0,
        "pts_allow_0": 10.0, "pts_allow_1_6": 9.0, "pts_allow_7_13": 8.0, "pts_allow_14_20": 5.0,
        "pts_allow_21_27": 2.0, "pts_allow_28_34": 1.0, "pts_allow_35p": -1.0,        # 46+ = -2, 35-45 absent = 0
        "yds_allow_0_100": 5.0, "yds_allow_100_199": 4.0, "yds_allow_200_299": 3.0, "yds_allow_300_349": 2.0,
        "yds_allow_350_399": 1.0, "yds_allow_400_449": -2.0, "yds_allow_450_499": -4.0, "yds_allow_500_549": -6.0,
        "yds_allow_550p": -8.0,
    }
    assert scoring == expect
    assert unmapped == [{"statId": 209, "label": "1pt Safety", "points": 1.0}]
    # no fgm_50_59 / fgm_60p entries were invented (they would double count against split stat lines)
    assert "fgm_50_59" not in scoring and "fgm_60p" not in scoring


def test_every_n_items_add_to_the_per_unit_key():
    scoring, un = espn_scoring_to_sleeper([item(3, 0.04), item(6, 1.0), item(5, 0.2)])
    assert scoring["pass_yd"] == pytest.approx(0.04 + 0.1 + 0.04)
    assert un == []
    scoring, _ = espn_scoring_to_sleeper([item(53, 1.0), item(54, 2.0), item(55, 5.0)])
    assert scoring["rec"] == pytest.approx(1.0 + 0.4 + 0.5)
    scoring, _ = espn_scoring_to_sleeper([item(28, 1.0)])           # every 10 rushing yards, no per-yard item
    assert scoring == {"rush_yd": 0.1}
    scoring, _ = espn_scoring_to_sleeper([item(100, 0.5)])          # 1/2 sack -> per sack
    assert scoring == {"sack": 1.0}


def test_dst_items_take_the_d_st_override():
    scoring, _ = espn_scoring_to_sleeper([item(99, 0.0, {"16": 1.5}), item(95, 2.0), item(89, 0.0, {"16": 10.0})])
    assert scoring == {"sack": 1.5, "int": 2.0, "pts_allow_0": 10.0}
    assert effective_points(item(99, 0.0, {"16": 1.5})) == 1.5
    assert effective_points(item(99, 0.0, {"16": 1.5}), "sack") == 1.5
    assert effective_points(item(4, 4.0, {"16": 1.5}), "pass_td") == 4.0


def test_reception_premium_from_slot_overrides():
    scoring, un = espn_scoring_to_sleeper([item(53, 1.0, {"6": 1.5, "2": 0.5, "4": 1.0})])
    assert scoring == {"rec": 1.0, "bonus_rec_te": 0.5, "bonus_rec_rb": -0.5}
    assert un == []
    scoring, _ = espn_scoring_to_sleeper([item(41, 0.5, {"6": 1.0})])
    assert scoring == {"rec": 0.5, "bonus_rec_te": 0.5}


def test_reception_items_never_both_count():
    scoring, _ = espn_scoring_to_sleeper([item(41, 0.5), item(53, 1.0)])
    assert scoring == {"rec": 1.0}
    scoring, _ = espn_scoring_to_sleeper([item(41, 0.5)])
    assert scoring == {"rec": 0.5}


def test_total_two_point_item_only_fills_missing_specific_items():
    scoring, _ = espn_scoring_to_sleeper([item(62, 2.0)])
    assert scoring == {"pass_2pt": 2.0, "rush_2pt": 2.0, "rec_2pt": 2.0}
    scoring, _ = espn_scoring_to_sleeper([item(62, 2.0), item(19, 3.0)])
    assert scoring == {"pass_2pt": 3.0, "rush_2pt": 2.0, "rec_2pt": 2.0}


def test_multi_source_keys_use_the_mean_not_the_sum():
    scoring, _ = espn_scoring_to_sleeper([item(92, 0.0, {"16": 6.0}), item(121, 0.0, {"16": 4.0})])
    assert scoring == {"pts_allow_14_20": 5.0}
    scoring, _ = espn_scoring_to_sleeper([item(93, 6.0), item(94, 6.0), item(103, 6.0), item(104, 6.0)])
    assert scoring == {"def_td": 6.0}
    scoring, _ = espn_scoring_to_sleeper([item(124, 0.0, {"16": -1.0}), item(125, 0.0, {"16": -3.0})])
    assert scoring == {"pts_allow_35p": -2.0}
    # return TDs: player key from the base points, team-defense key from the D/ST override
    scoring, _ = espn_scoring_to_sleeper([item(101, 6.0, {"16": 4.0}), item(102, 6.0, {"16": 4.0})])
    assert scoring == {"st_td": 6.0, "def_st_td": 4.0}


def test_sub_bracket_partitions_count_an_absent_member_as_zero():
    """ESPN omits 0-point items: 18-21 = 4 with no 14-17 item means 14-17 scores 0, so the merged key is 2."""
    assert espn_scoring_to_sleeper([item(121, 0.0, {"16": 4.0})])[0] == {"pts_allow_14_20": 2.0}
    assert espn_scoring_to_sleeper([item(92, 0.0, {"16": 1.0})])[0] == {"pts_allow_14_20": 0.5}
    assert espn_scoring_to_sleeper([item(125, 0.0, {"16": -2.0})])[0] == {"pts_allow_35p": -1.0}
    assert espn_scoring_to_sleeper([item(196, 0.0, {"16": -2.0})])[0] == {"pts_allow_35p": -1.0}
    assert espn_scoring_to_sleeper([item(191, 0.0, {"16": 6.0}), item(192, 0.0, {"16": 4.0})])[0] == {"pts_allow_14_20": 5.0}
    # the return-TD groups are not partitions: a league scoring 94 alone keeps 6 per defensive TD
    assert espn_scoring_to_sleeper([item(94, 6.0, {"16": 6.0})])[0] == {"def_td": 6.0}
    assert espn_scoring_to_sleeper([item(101, 6.0, {"16": 6.0})])[0] == {"st_td": 6.0, "def_st_td": 6.0}
    # a 35-45 game under the fixture league scores 0 on ESPN and -1 (the partition mean) here, never -2
    scoring, _ = espn_scoring_to_sleeper(load("league_settings_teams.json")["settings"]["scoringSettings"]["scoringItems"])
    assert ScoringEngine(scoring).score({"pts_allow_35p": 1.0}, "DEF") == -1.0


def test_total_return_td_feeds_def_td():
    """statId 105 (= 93 + 94 + 101 + 102) includes INT / fumble return TDs: a 105-only league scores def_td."""
    scoring, un = espn_scoring_to_sleeper([item(105, 6.0, {"16": 6.0}), item(99, 0.0, {"16": 1.0})])
    assert scoring == {"st_td": 6.0, "def_st_td": 6.0, "def_td": 6.0, "sack": 1.0} and un == []
    line = espn_stats_to_sleeper({"93": 0, "94": 2, "101": 0, "102": 0, "103": 2, "104": 0, "105": 2, "99": 40}, "DEF")
    assert line == {"def_td": 2.0, "def_st_td": 0.0, "sack": 40.0}
    assert ScoringEngine(scoring).score(line, "DEF") == 52.0                    # ESPN: 105 x 6 + 99 x 1
    # a league listing 105 next to the individual items keeps their (identical) points
    scoring, _ = espn_scoring_to_sleeper([item(105, 6.0, {"16": 6.0}), item(93, 6.0, {"16": 6.0}), item(94, 6.0, {"16": 6.0}),
                                          item(101, 6.0, {"16": 6.0}), item(102, 6.0, {"16": 6.0})])
    assert scoring == {"st_td": 6.0, "def_st_td": 6.0, "def_td": 6.0}
    # for a player the value lands on st_td only
    assert espn_stats_to_sleeper({"105": 1}, "WR") == {"st_td": 1.0}


def test_kicking_brackets():
    scoring, _ = espn_scoring_to_sleeper([item(80, 3.0), item(77, 4.0), item(74, 5.0), item(82, -1.0), item(76, -0.5)])
    assert scoring == {"fgm_0_19": 3.0, "fgm_20_29": 3.0, "fgm_30_39": 3.0, "fgm_40_49": 4.0, "fgm_50p": 5.0,
                       "fgmiss_0_19": -1.0, "fgmiss_20_29": -1.0, "fgmiss_30_39": -1.0, "fgmiss_50p": -0.5}
    scoring, _ = espn_scoring_to_sleeper([item(198, 5.0), item(201, 6.0)])
    assert scoring == {"fgm_50_59": 5.0, "fgm_60p": 6.0}


def test_unmapped_rules_are_listed_with_labels():
    scoring, un = espn_scoring_to_sleeper([
        item(15, 2.0), item(210, 0.5), item(73, -1.0), item(17, 0.0), item(4, 4.0, {"0": 6.0}),
        item(25, 6.0, {"16": 8.0}), item(175, 1.0),
    ])
    assert scoring == {"pass_td": 4.0, "rush_td": 6.0}
    assert un == [
        {"statId": 4, "label": "TD Pass (QB only)", "points": 6.0},
        {"statId": 15, "label": "40+ yard TD pass bonus", "points": 2.0},
        {"statId": 25, "label": "TD Rush (D/ST only)", "points": 8.0},
        {"statId": 73, "label": "Total Turnovers", "points": -1.0},
        {"statId": 175, "label": "0-9 yd TD pass bonus", "points": 1.0},
        {"statId": 210, "label": "Games Played", "points": 0.5},
    ]


def test_unmapped_rules_expressed_as_slot_overrides_are_listed():
    """ESPN writes position-specific rules as points 0 + a slot override: they must reach the not-modelled list."""
    scoring, un = espn_scoring_to_sleeper([
        item(15, 0.0, {"0": 2.0}), item(209, 0.0, {"16": 1.0}), item(207, 0.0, {"16": 1.0}),
        item(126, 0.0, {"16": -1.0}),                                # D/ST-only id: the override is its value
        item(73, -1.0, {"16": -1.0}),                                # override equal to the base: listed once
        item(16, 0.0, {"0": 0.0}),                                   # all zero: nothing
    ])
    assert scoring == {}
    assert un == [
        {"statId": 15, "label": "40+ yard TD pass bonus (QB only)", "points": 2.0},
        {"statId": 73, "label": "Total Turnovers", "points": -1.0},
        {"statId": 126, "label": "Points Allowed Per Game", "points": -1.0},
        {"statId": 207, "label": "Offensive 1pt Safety (D/ST only)", "points": 1.0},
        {"statId": 209, "label": "1pt Safety (D/ST only)", "points": 1.0},
    ]
    # base points plus a differing override: both listed
    _, un = espn_scoring_to_sleeper([item(15, 2.0, {"0": 3.0, "16": 2.0})])
    assert un == [{"statId": 15, "label": "40+ yard TD pass bonus", "points": 2.0},
                  {"statId": 15, "label": "40+ yard TD pass bonus (QB only)", "points": 3.0}]


def test_zero_and_garbage_items_are_ignored():
    scoring, un = espn_scoring_to_sleeper([
        item(3, 0.0), {"statId": None, "points": 5}, {"points": 5}, "junk", None, {"statId": "24", "points": "0.1"},
        {"statId": 42, "points": None, "pointsOverrides": None}, {"statId": 43, "points": 6, "pointsOverrides": {"6": "x"}},
    ])
    assert scoring == {"rush_yd": 0.1, "rec_td": 6.0}
    assert un == []
    assert espn_scoring_to_sleeper(None) == ({}, [])
    assert espn_scoring_to_sleeper([]) == ({}, [])


def test_float_noise_is_rounded_away():
    scoring, _ = espn_scoring_to_sleeper([item(3, 0.1), item(3, 0.2)])
    assert scoring == {"pass_yd": 0.15}
    scoring, _ = espn_scoring_to_sleeper([item(48, 1.0), item(42, 0.2)])
    assert scoring == {"rec_yd": 0.3}


# ---------------------------------------------------------------------------
# projected stats
# ---------------------------------------------------------------------------


def _kona_by_name() -> dict[str, dict]:
    return {p["player"]["fullName"]: p for p in load("players_kona.json")["players"]}


def test_projected_season_stats_and_points():
    ab = _kona_by_name()["Antonio Brown"]
    st = projected_season_stats(ab, 2018)
    assert st is not None and st["53"] == pytest.approx(110.9) and st["42"] == pytest.approx(1584.9)
    assert projected_points(ab, 2018) == pytest.approx(321.8757, abs=1e-3)
    assert projected_season_stats(ab, 2019) is None and projected_points(ab, 2019) is None
    assert projected_season_stats(ab["player"], 2018) == st          # bare player dict works too
    assert projected_season_stats({"player": {"stats": "x"}}, 2018) is None
    assert projected_season_stats(None, 2018) is None
    # the id-less shape is found through statSourceId / statSplitTypeId / seasonId
    alt = {"stats": [{"seasonId": 2018, "statSourceId": 1, "statSplitTypeId": 0, "scoringPeriodId": 0, "stats": {"3": 100}},
                     {"seasonId": 2018, "statSourceId": 0, "statSplitTypeId": 0, "stats": {"3": 1}}]}
    assert projected_season_stats(alt, 2018) == {"3": 100.0}


def test_stats_conversion_reproduces_espn_points_for_offense_and_kickers():
    league = load("league_settings_teams.json")
    scoring, _ = espn_scoring_to_sleeper(league["settings"]["scoringSettings"]["scoringItems"])
    eng = ScoringEngine(scoring)
    by_name = _kona_by_name()
    for name, pos in (("Antonio Brown", "WR"), ("Todd Gurley II", "RB"), ("Stephen Gostkowski", "K"),
                      ("Greg Zuerlein", "K")):
        entry = by_name[name]
        line = espn_stats_to_sleeper(projected_season_stats(entry, 2018), pos)
        assert eng.score(line, pos) == pytest.approx(projected_points(entry, 2018), abs=0.02), name
        assert not any(k.startswith("pass_yd_") for k in line)
    ab = espn_stats_to_sleeper(projected_season_stats(by_name["Antonio Brown"], 2018), "WR")
    assert ab["rec"] == pytest.approx(110.9) and ab["gp"] == 15.0 and ab["pr_yd"] == pytest.approx(160.763429)
    assert "rec_2pt" in ab and ab["rec_2pt"] == pytest.approx(0.389891)     # 44 present -> 62 skipped
    k = espn_stats_to_sleeper(projected_season_stats(by_name["Greg Zuerlein"], 2018), "K")
    assert k["fgm_50_59"] == k["fgm_50p"] and k["fgmiss_50_59"] == k["fgmiss_50p"]   # filled from the 50+ items
    assert "fgm_60p" not in k
    assert k["fgm_0_19"] > 0 and "fgm_20_29" not in k                       # 80 goes to one bracket key only


def test_stats_conversion_for_a_team_defense():
    dst = _kona_by_name()["Jaguars D/ST"]
    raw = projected_season_stats(dst, 2018)
    line = espn_stats_to_sleeper(raw, "DEF")
    assert line["def_td"] == pytest.approx(raw["93"] + raw["94"])          # 103 / 104 are parts of 94
    assert line["def_st_td"] == pytest.approx(raw["101"] + raw["102"]) and "st_td" not in line
    assert "105" in raw                                                     # aggregate present but skipped
    assert line["sack"] == pytest.approx(raw["99"]) and line["pts_allow"] == pytest.approx(raw["120"])
    assert line["gp"] == raw["210"] and line["yds_allow"] == pytest.approx(raw["127"])
    # ESPN projects every game into one bracket (121: 16, 131: 16): those carry nothing the totals do not,
    # and would score at the merged Sleeper bracket's mean weight, so they are left for the projection layer
    assert raw["121"] == 16.0 and raw["131"] == 16.0
    assert not any(k.startswith(("pts_allow_", "yds_allow_")) for k in line)
    # the same line for a player position lands on the player keys
    as_player = espn_stats_to_sleeper({"101": 1, "102": 2}, "WR")
    assert as_player == {"st_td": 3.0}


def test_degenerate_dst_brackets_are_derived_from_the_totals():
    """Through the real path (blend), every fixture D/ST scores a *derived* bracket distribution: the same
    model for all of them, close to ESPN's total and never the +16 overshoot of the mean-weighted bracket."""
    from draftadvisor.projections.blend import PTS_BRACKETS, YDS_BRACKETS, sleeper_stats_to_projection

    league = load("league_settings_teams.json")
    scoring, _ = espn_scoring_to_sleeper(league["settings"]["scoringSettings"]["scoringItems"])
    eng = ScoringEngine(scoring)
    dsts = [p for p in load("players_kona.json")["players"] if p["player"]["defaultPositionId"] == 16]
    assert len(dsts) == 3
    for entry in dsts:
        raw = projected_season_stats(entry, 2018)
        line = espn_stats_to_sleeper(raw, "DEF")
        pts, games, full = sleeper_stats_to_projection(line, eng, "DEF")
        assert games == 16.0 and sum(full[k] for k in PTS_BRACKETS) == pytest.approx(16.0, abs=0.01)
        assert sum(full[k] for k in YDS_BRACKETS) == pytest.approx(16.0, abs=0.01)
        assert sum(1 for k in PTS_BRACKETS if full[k] > 0.5) >= 4          # a distribution, not one bracket
        espn = projected_points(entry, 2018)
        assert abs(pts - espn) < 20 and pts < espn + 1                       # never the +16 overshoot
    # the brackets stay when the total is missing (nothing to derive from) or when the line is a real spread
    assert espn_stats_to_sleeper({"121": 16.0, "99": 40}, "DEF") == {"pts_allow_14_20": 16.0, "sack": 40.0}
    spread = espn_stats_to_sleeper({"120": 316.0, "92": 3, "121": 4, "124": 1, "125": 2, "127": 5000, "131": 8, "132": 8}, "DEF")
    assert spread == {"pts_allow": 316.0, "pts_allow_14_20": 7.0, "pts_allow_35p": 3.0, "yds_allow": 5000.0,
                      "yds_allow_300_349": 8.0, "yds_allow_350_399": 8.0}
    # a player's line is never touched
    assert espn_stats_to_sleeper({"120": 316.0, "121": 16.0}, "WR") == {"pts_allow": 316.0, "pts_allow_14_20": 16.0}


def test_stats_conversion_edge_cases():
    assert espn_stats_to_sleeper(None) == {} and espn_stats_to_sleeper({}) == {}
    assert espn_stats_to_sleeper({"x": 1, "3": "abc", "4": None}) == {}
    assert espn_stats_to_sleeper({"41": 5, "53": 5}) == {"rec": 5.0}
    assert espn_stats_to_sleeper({"41": 5}) == {"rec": 5.0}
    assert espn_stats_to_sleeper({"62": 2}, "QB") == {"pass_2pt": 2.0}
    assert espn_stats_to_sleeper({"62": 2}, "RB") == {"rush_2pt": 2.0}
    assert espn_stats_to_sleeper({"62": 2}) == {"rec_2pt": 2.0}
    assert espn_stats_to_sleeper({"62": 2, "26": 1}) == {"rush_2pt": 1.0}
    assert espn_stats_to_sleeper({"5": 100, "3": 500}) == {"pass_yd": 500.0}   # every-N ids ignored
    assert espn_stats_to_sleeper({"92": 3, "121": 4, "124": 1, "125": 2}, "DEF") == {"pts_allow_14_20": 7.0, "pts_allow_35p": 3.0}
    assert espn_stats_to_sleeper({"120": 300, "187": 300}, "DEF") == {"pts_allow": 300.0}
    assert espn_stats_to_sleeper({"210": 0}) == {}
    assert espn_stats_to_sleeper({"74": 2, "198": 1, "201": 1}, "K") == {"fgm_50p": 2.0, "fgm_50_59": 1.0, "fgm_60p": 1.0}
