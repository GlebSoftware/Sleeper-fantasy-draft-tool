"""Canonical conversion, crosswalk helpers and player universe (offline, synthetic)."""
from __future__ import annotations

import pandas as pd

from draftadvisor.data.canonical import player_weekly_to_canonical, team_weekly_to_canonical, to_sleeper_team
from draftadvisor.data.crosswalk import Crosswalk, normalize_name
from draftadvisor.data.universe import (assign_adp, enrich_players, filter_relevant, players_from_crosswalk,
                                        players_from_sleeper)
from draftadvisor.models import Player
from draftadvisor.scoring import ScoringEngine
from draftadvisor.scoring.engine import NON_DERIVABLE_STAT_KEYS, SLEEPER_STAT_KEYS
from tests.conftest import load_fixture


def test_team_aliases():
    assert to_sleeper_team("LA") == "LAR" and to_sleeper_team("JAC") == "JAX" and to_sleeper_team("OAK") == "LV"
    assert to_sleeper_team("KC") == "KC" and to_sleeper_team(None) is None and to_sleeper_team(float("nan")) is None
    assert to_sleeper_team("FA") is None
    # DS-6: MFL-era Chargers code and the 'FA*' free-agent marker used by db_playerids
    assert to_sleeper_team("SDC") == "LAC" and to_sleeper_team("FA*") is None


def test_player_weekly_to_canonical_derives_keys():
    raw = pd.DataFrame([
        {"player_id": "00-1", "player_display_name": "A Back", "position": "RB", "season": 2025, "week": 1, "season_type": "REG",
         "team": "LA", "opponent_team": "SF", "carries": 20, "rushing_yards": 105, "rushing_tds": 1, "receptions": 4, "targets": 6,
         "receiving_yards": 30, "receiving_tds": 0, "rushing_fumbles_lost": 1, "fantasy_points_ppr": 27.5},
        {"player_id": "00-2", "player_display_name": "A Kicker", "position": "K", "season": 2025, "week": 1, "season_type": "REG",
         "team": "KC", "opponent_team": "LV", "fg_made": 3, "fg_att": 4, "fg_missed": 1, "fg_made_40_49": 2, "fg_made_50_59": 1,
         "fg_made_list": "42;55;44", "pat_made": 2, "pat_att": 2},
        {"player_id": "00-3", "player_display_name": "Lineman", "position": "OL", "season": 2025, "week": 1, "season_type": "REG", "team": "KC"},
        {"player_id": "00-1", "player_display_name": "A Back", "position": "RB", "season": 2025, "week": 19, "season_type": "POST", "team": "LA"},
    ])
    c = player_weekly_to_canonical(raw)
    assert len(c) == 2 and set(c.position) == {"RB", "K"}
    rb = c[c.position == "RB"].iloc[0]
    assert rb.team == "LAR" and rb.rush_att == 20 and rb.bonus_rush_yd_100 == 1 and rb.bonus_rush_rec_yd_100 == 1 and rb.fum_lost == 1
    k = c[c.position == "K"].iloc[0]
    assert k.fgm == 3 and k.fgmiss == 1 and k.fgm_50p == 1 and k.fgm_yds_over_30 == (12 + 25 + 14) and k.xpm == 2
    e = ScoringEngine({"rush_yd": 0.1, "rush_td": 6, "rec": 1, "rec_yd": 0.1, "fum_lost": -2})
    assert abs(e.score_frame(c[c.position == "RB"]).iloc[0] - (10.5 + 6 + 4 + 3 - 2)) < 1e-9


def test_team_weekly_to_canonical_points_allowed():
    team = pd.DataFrame([
        {"season": 2025, "week": 1, "team": "SF", "season_type": "REG", "opponent_team": "LA", "def_sacks": 3, "def_interceptions": 1,
         "passing_yards": 250, "sack_yards_lost": -10, "rushing_yards": 100, "def_fumbles_forced": 1, "fumble_recovery_opp": 1,
         "def_tds": 1, "def_safeties": 0, "def_punt_blocks": 1, "special_teams_tds": 0},
        {"season": 2025, "week": 1, "team": "LA", "season_type": "REG", "opponent_team": "SF", "def_sacks": 1, "def_interceptions": 0,
         "passing_yards": 300, "sack_yards_lost": -20, "rushing_yards": 80, "def_fumbles_forced": 0, "fumble_recovery_opp": 0,
         "def_tds": 0, "def_safeties": 1, "def_punt_blocks": 0, "special_teams_tds": 1},
    ])
    games = pd.DataFrame([{"season": 2025, "week": 1, "game_type": "REG", "home_team": "SF", "away_team": "LA", "home_score": 24, "away_score": 13}])
    d = team_weekly_to_canonical(team, games).set_index("player_id")
    assert d.loc["SF", "pts_allow"] == 13 and d.loc["LAR", "pts_allow"] == 24
    # DS-1: net yards = passing + rushing - sack yardage (sack_yards_lost is negative in nflverse)
    assert d.loc["SF", "yds_allow"] == 300 - 20 + 80 and d.loc["LAR", "yds_allow"] == 250 - 10 + 100
    assert d.loc["SF", "yds_allow_350_399"] == 1 and d.loc["LAR", "yds_allow_300_349"] == 1
    assert d.loc["SF", "pts_allow_7_13"] == 1 and d.loc["SF", "blk_kick"] == 1 and d.loc["LAR", "def_st_td"] == 1
    assert d.loc["SF", "position"] == "DEF" and d.loc["SF", "opponent"] == "LAR"
    # robust to a positive sack-yardage convention too (same net result)
    d2 = team_weekly_to_canonical(team.assign(sack_yards_lost=team["sack_yards_lost"].abs()), games).set_index("player_id")
    assert d2.loc["SF", "yds_allow"] == 360 and d2.loc["LAR", "yds_allow"] == 340


def test_yds_allow_phi_dal_2025_wk1_example():
    """DS-1 regression on the reviewer's real example: PHI offense 152 pass / -8 sack / 158 rush -> DAL DEF allows 302."""
    team = pd.DataFrame([
        {"season": 2025, "week": 1, "team": "PHI", "season_type": "REG", "opponent_team": "DAL",
         "passing_yards": 152, "sack_yards_lost": -8, "rushing_yards": 158},
        {"season": 2025, "week": 1, "team": "DAL", "season_type": "REG", "opponent_team": "PHI",
         "passing_yards": 188, "sack_yards_lost": -4, "rushing_yards": 92},
    ])
    games = pd.DataFrame([{"season": 2025, "week": 1, "game_type": "REG", "home_team": "PHI", "away_team": "DAL", "home_score": 24, "away_score": 20}])
    d = team_weekly_to_canonical(team, games).set_index("player_id")
    assert d.loc["DAL", "yds_allow"] == 302 and d.loc["DAL", "yds_allow_300_349"] == 1
    assert d.loc["PHI", "yds_allow"] == 276 and d.loc["PHI", "yds_allow_200_299"] == 1


def _player_row(**kw) -> dict:
    base = {"player_id": "00-9", "player_display_name": "X", "position": "WR", "season": 2025, "week": 1,
            "season_type": "REG", "team": "CHI", "opponent_team": "MIN"}
    base.update(kw)
    return base


def test_fumbles_prefer_totals_including_returns():
    """DS-2: nflverse fumbles_total / fumbles_lost_total include kick/punt-return fumbles (Sleeper counts them)."""
    raw = pd.DataFrame([
        _player_row(player_id="00-1", sack_fumbles=0, rushing_fumbles=0, receiving_fumbles=0, sack_fumbles_lost=0,
                    rushing_fumbles_lost=0, receiving_fumbles_lost=0, fumbles_total=1, fumbles_lost_total=1),
        _player_row(player_id="00-2", rushing_fumbles=2, rushing_fumbles_lost=1, fumbles_total=2, fumbles_lost_total=1),
    ])
    c = player_weekly_to_canonical(raw).set_index("player_id")
    assert c.loc["00-1", "fum"] == 1 and c.loc["00-1", "fum_lost"] == 1     # return fumble only
    assert c.loc["00-2", "fum"] == 2 and c.loc["00-2", "fum_lost"] == 1
    # old-format file without the totals -> sum of the split columns
    old = pd.DataFrame([_player_row(rushing_fumbles=1, rushing_fumbles_lost=1, receiving_fumbles=1)])
    c = player_weekly_to_canonical(old).iloc[0]
    assert c.fum == 2 and c.fum_lost == 1


def test_40_plus_play_keys_derived():
    """DS-3: rec_40p / rush_40p / pass_cmp_40p come from receiving_40 / rushing_40 / passing_40."""
    raw = pd.DataFrame([_player_row(position="QB", passing_40=2, rushing_40=1, receiving_40=0),
                        _player_row(player_id="00-3", position="WR", receiving_40=3)])
    c = player_weekly_to_canonical(raw).set_index("player_id")
    assert c.loc["00-9", "pass_cmp_40p"] == 2 and c.loc["00-9", "rush_40p"] == 1 and c.loc["00-9", "rec_40p"] == 0
    assert c.loc["00-3", "rec_40p"] == 3
    e = ScoringEngine({"rec_40p": 1.0, "rush_40p": 1.0, "pass_cmp_40p": 0.5})
    assert e.score_frame(c).tolist() == [2.0, 3.0]
    # every advertised derivable key really is in the canonical frame (players + DEF); non-derivable ones are not advertised
    team = pd.DataFrame([{"season": 2025, "week": 1, "team": "CHI", "season_type": "REG", "opponent_team": "MIN",
                          "passing_yards": 200, "sack_yards_lost": -5, "rushing_yards": 100}])
    games = pd.DataFrame([{"season": 2025, "week": 1, "game_type": "REG", "home_team": "CHI", "away_team": "MIN", "home_score": 20, "away_score": 17}])
    cols = set(c.columns) | set(team_weekly_to_canonical(team, games).columns)
    missing = [k for k in SLEEPER_STAT_KEYS if k not in cols]
    assert missing == [], missing
    assert not set(SLEEPER_STAT_KEYS) & set(NON_DERIVABLE_STAT_KEYS)


def test_blocked_fg_allocated_to_miss_brackets():
    """DS-4: fgmiss counts blocked kicks, so the fgmiss_* distance brackets must too."""
    raw = pd.DataFrame([_player_row(position="K", fg_made=2, fg_att=5, fg_missed=1, fg_blocked=2, fg_made_30_39=2,
                                    fg_missed_50_59=1, fg_made_list="33;38", fg_missed_list="52", fg_blocked_list="47;61")])
    k = player_weekly_to_canonical(raw).iloc[0]
    assert k.fgmiss == 3 and k.fgmiss_40_49 == 1 and k.fgmiss_50_59 == 1 and k.fgmiss_60p == 1 and k.fgmiss_50p == 2
    brackets = ["fgmiss_0_19", "fgmiss_20_29", "fgmiss_30_39", "fgmiss_40_49", "fgmiss_50_59", "fgmiss_60p"]
    assert sum(k[b] for b in brackets) == k.fgmiss
    # no blocked-list column (old files) -> brackets are just the misses
    k2 = player_weekly_to_canonical(raw.drop(columns=["fg_blocked_list"])).iloc[0]
    assert k2.fgmiss == 3 and k2.fgmiss_40_49 == 0 and k2.fgmiss_50_59 == 1


def test_normalize_name_and_crosswalk_lookup():
    assert normalize_name("Amon-Ra St. Brown Jr.") == "amonra st brown"
    assert normalize_name("Ja'Marr Chase") == "jamarr chase"
    cw = Crosswalk()
    cw.add("7564", gsis_id="00-0036900", fp_id="19788", pfr_id="ChasJa00", name="Ja'Marr Chase", position="WR", age=26.5, draft_ovr=5)
    cw.add("7564", gsis_id="00-0036900", name="Ja'Marr Chase", position="WR", team="CIN")   # duplicate rows merge
    assert cw.gsis_for("7564") == "00-0036900" and cw.sleeper_for_gsis("00-0036900") == "7564"
    assert cw.sleeper_for_fp("19788.0") == "7564" and cw.pfr_to_gsis["ChasJa00"] == "00-0036900"
    assert cw.sleeper_for_name("Jamarr Chase", "WR") == "7564" and cw.attrs["7564"]["team"] == "CIN"


def test_players_from_sleeper_and_enrichment():
    payload = load_fixture("players_sample.json")
    players = players_from_sleeper(payload)
    assert "99998" not in players                      # inactive / no team dropped
    assert "SF" in players and players["SF"].position == "DEF" and players["SF"].name == "San Francisco 49ers"
    p = players["8146"]
    assert p.name == "Amon-Ra St. Brown" and p.position == "WR" and p.team == "DET"
    cw = Crosswalk()
    cw.add("8146", gsis_id="00-0036963", fp_id="19799", name="Amon-Ra St. Brown", position="WR", draft_round=4, draft_ovr=112)
    ecr = pd.DataFrame([{"fantasypros_id": "19799", "player": "Amon-Ra St. Brown", "pos": "WR", "team": "DET", "ecr": 5.4, "sd": 1.29, "best": 3, "worst": 9, "bye": 6},
                        {"fantasypros_id": "1", "player": "San Francisco 49ers", "pos": "DEF", "team": "SF", "ecr": 150.0, "sd": 10.0, "best": 1, "worst": 1, "bye": 8}])
    enrich_players(players, cw, ecr, {"DET": 6, "SF": 8})
    assert p.gsis_id == "00-0036963" and p.ecr == 5.4 and p.bye_week == 6 and p.draft_pick_overall == 112
    assert players["SF"].ecr == 150.0 and players["SF"].bye_week == 8
    n = assign_adp(players, {"8146": 4.5}, "sleeper_ppr")
    assert n == 1 and p.adp == 4.5 and p.adp_source == "sleeper_ppr"
    assert players["SF"].adp == 150.0 and players["SF"].adp_source == "ecr"
    rel = filter_relevant(players.values(), {"WR": 2})
    assert sum(1 for x in rel if x.position == "WR") == 2 and rel[0] is not None


def test_ecr_name_fallback_disambiguates_by_team():
    """DS-5: duplicate (name, pos) ECR rows must not silently last-win; team breaks the tie."""
    ecr = pd.DataFrame([
        {"fantasypros_id": "26379", "player": "Isaiah Williams", "pos": "WR", "team": "NYJ", "ecr": 307.3, "sd": 20.0, "bye": 9},
        {"fantasypros_id": "10977", "player": "Isaiah Williams", "pos": "WR", "team": "FA", "ecr": 338.3, "sd": 30.0, "bye": None},
        {"fantasypros_id": "555", "player": "Unique Guy", "pos": "RB", "team": "KC", "ecr": 100.0, "sd": 5.0, "bye": 10},
    ])
    players = {
        "1": Player(player_id="1", name="Isaiah Williams", position="WR", team="NYJ"),   # team match -> NYJ row
        "2": Player(player_id="2", name="Isaiah Williams", position="WR", team="DET"),   # ambiguous, no team match -> none
        "3": Player(player_id="3", name="Unique Guy", position="RB", team="LV"),         # unique name, team mismatch -> still matched
    }
    enrich_players(players, Crosswalk(), ecr)
    assert players["1"].ecr == 307.3 and players["1"].fantasypros_id == "26379"
    assert players["2"].ecr is None and players["2"].fantasypros_id is None
    assert players["3"].ecr == 100.0


def test_players_from_crosswalk_synthetic_ids_for_unlinked_roster_players():
    """DS-7: active roster skill players with no Sleeper id anywhere get 'nfl:<gsis>' ids and a crosswalk entry."""
    cw = Crosswalk()
    cw.add("1234", gsis_id="00-0001", name="Known Back", position="RB", team="LV")
    roster = pd.DataFrame([
        {"gsis_id": "00-0001", "sleeper_id": "1234", "full_name": "Known Back", "position": "RB", "team": "LV", "status": "ACT",
         "years_exp": 3, "entry_year": 2023, "birth_date": "2001-01-01", "draft_number": 40, "pfr_id": "KnowBa00"},
        {"gsis_id": "00-0002", "sleeper_id": None, "full_name": "Trey Smack", "position": "K", "team": "GB", "status": "ACT",
         "years_exp": 0, "entry_year": 2026, "birth_date": "2003-05-05", "draft_number": None, "pfr_id": None},
        {"gsis_id": "00-0003", "sleeper_id": None, "full_name": "Cut Guy", "position": "WR", "team": "MIA", "status": "CUT",
         "years_exp": 1, "entry_year": 2025, "birth_date": "2002-01-01", "draft_number": None, "pfr_id": None},
        {"gsis_id": "00-0004", "sleeper_id": None, "full_name": "Big Tackle", "position": "OL", "team": "MIA", "status": "ACT",
         "years_exp": 1, "entry_year": 2025, "birth_date": "2002-01-01", "draft_number": None, "pfr_id": None},
    ])
    players = players_from_crosswalk(cw, roster, season=2026)
    assert set(players) == {"1234", "nfl:00-0002"}
    k = players["nfl:00-0002"]
    assert k.name == "Trey Smack" and k.position == "K" and k.team == "GB" and k.gsis_id == "00-0002"
    assert k.status == "Active" and k.years_exp == 0 and k.draft_year == 2026 and k.age is not None
    # registered in the crosswalk so history / ML feature lookups by gsis work
    assert cw.gsis_for("nfl:00-0002") == "00-0002" and cw.sleeper_for_gsis("00-0002") == "nfl:00-0002"
    assert cw.attrs["nfl:00-0002"]["position"] == "K"
    # idempotent: a second call with the same crosswalk yields the same universe
    assert set(players_from_crosswalk(cw, roster, season=2026)) == {"1234", "nfl:00-0002"}
