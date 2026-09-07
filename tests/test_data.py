"""Canonical conversion, crosswalk helpers and player universe (offline, synthetic)."""
from __future__ import annotations

import pandas as pd

from draftadvisor.data.canonical import player_weekly_to_canonical, team_weekly_to_canonical, to_sleeper_team
from draftadvisor.data.crosswalk import Crosswalk, normalize_name
from draftadvisor.data.universe import assign_adp, enrich_players, filter_relevant, players_from_sleeper
from draftadvisor.scoring import ScoringEngine
from tests.conftest import load_fixture


def test_team_aliases():
    assert to_sleeper_team("LA") == "LAR" and to_sleeper_team("JAC") == "JAX" and to_sleeper_team("OAK") == "LV"
    assert to_sleeper_team("KC") == "KC" and to_sleeper_team(None) is None and to_sleeper_team(float("nan")) is None
    assert to_sleeper_team("FA") is None


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
         "passing_yards": 250, "sack_yards_lost": 10, "rushing_yards": 100, "def_fumbles_forced": 1, "fumble_recovery_opp": 1,
         "def_tds": 1, "def_safeties": 0, "def_punt_blocks": 1, "special_teams_tds": 0},
        {"season": 2025, "week": 1, "team": "LA", "season_type": "REG", "opponent_team": "SF", "def_sacks": 1, "def_interceptions": 0,
         "passing_yards": 300, "sack_yards_lost": 20, "rushing_yards": 80, "def_fumbles_forced": 0, "fumble_recovery_opp": 0,
         "def_tds": 0, "def_safeties": 1, "def_punt_blocks": 0, "special_teams_tds": 1},
    ])
    games = pd.DataFrame([{"season": 2025, "week": 1, "game_type": "REG", "home_team": "SF", "away_team": "LA", "home_score": 24, "away_score": 13}])
    d = team_weekly_to_canonical(team, games).set_index("player_id")
    assert d.loc["SF", "pts_allow"] == 13 and d.loc["LAR", "pts_allow"] == 24
    assert d.loc["SF", "yds_allow"] == 300 - 20 + 80 and d.loc["LAR", "yds_allow"] == 250 - 10 + 100
    assert d.loc["SF", "pts_allow_7_13"] == 1 and d.loc["SF", "blk_kick"] == 1 and d.loc["LAR", "def_st_td"] == 1
    assert d.loc["SF", "position"] == "DEF" and d.loc["SF", "opponent"] == "LAR"


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
