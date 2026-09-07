"""Scoring engine."""
from __future__ import annotations

import pandas as pd

from draftadvisor.scoring import DEFAULT_SCORING, ScoringEngine
from draftadvisor.scoring.engine import aggregate_season


def test_dot_product_and_position_bonus():
    e = ScoringEngine({"rec": 1, "rec_yd": 0.1, "rec_td": 6, "bonus_rec_te": 0.5, "pass_td": 4, "pass_yd": 0.04})
    line = {"rec": 5, "rec_yd": 70, "rec_td": 1, "bonus_rec_te": 5}
    assert e.score(line, "WR") == 18.0
    assert e.score(line, "TE") == 20.5
    assert e.score({"pass_yd": 300, "pass_td": 2}) == 20.0
    assert e.score({}) == 0.0


def test_zero_weights_dropped_and_missing_columns_ok():
    e = ScoringEngine({"rec": 0, "rec_yd": 0.1, "pass_cmp": None})
    assert e.keys == ["rec_yd"]
    df = pd.DataFrame({"rec_yd": [10, 20], "rec": [1, 2]})
    assert e.score_frame(df).tolist() == [1.0, 2.0]
    assert e.score_frame(df.iloc[0:0]).empty


def test_derived_keys_and_frame_scoring():
    e = ScoringEngine({"rec": 1, "rec_yd": 0.1, "bonus_rec_yd_100": 3, "bonus_rec_te": 0.5, "pts_allow_0": 10, "pts_allow_1_6": 7,
                       "yds_allow_0_100": 5, "fgm_50p": 5})
    df = pd.DataFrame([
        {"position": "TE", "rec": 4, "rec_yd": 101},
        {"position": "WR", "rec": 4, "rec_yd": 99},
        {"position": "DEF", "pts_allow": 0, "yds_allow": 90, "fgm_50_59": 0},
        {"position": "K", "fgm_50_59": 1, "fgm_60p": 1},
    ])
    ScoringEngine.add_derived_keys(df)
    assert df.loc[0, "bonus_rec_yd_100"] == 1 and df.loc[1, "bonus_rec_yd_100"] == 0
    assert df.loc[2, "pts_allow_0"] == 1 and df.loc[2, "yds_allow_0_100"] == 1
    assert df.loc[3, "fgm_50p"] == 2
    pts = e.score_frame(df).tolist()
    assert pts[0] == 4 + 10.1 + 3 + 2.0           # TE bonus applied only to the TE row
    assert pts[1] == 4 + 9.9
    assert pts[2] == 15.0 and pts[3] == 10.0


def test_relevant_keys_and_describe():
    e = ScoringEngine(DEFAULT_SCORING)
    assert "pass_td" in e.relevant_keys("QB") and "sack" not in e.relevant_keys("QB")
    assert "bonus_rec_te" not in e.relevant_keys("WR")
    assert e.describe().startswith("Half PPR")
    assert ScoringEngine({"rec": 1, "pass_td": 6}).describe() == "PPR, 6pt pass TD"


def test_aggregate_season():
    e = ScoringEngine({"rec": 1, "rec_yd": 0.1})
    df = pd.DataFrame([
        {"player_id": "a", "season": 2025, "position": "WR", "rec": 5, "rec_yd": 50},
        {"player_id": "a", "season": 2025, "position": "WR", "rec": 3, "rec_yd": 30},
        {"player_id": "b", "season": 2025, "position": "TE", "rec": 1, "rec_yd": 10},
    ])
    g = aggregate_season(df, e).set_index("player_id")
    assert g.loc["a", "games"] == 2 and g.loc["a", "points"] == 16.0 and g.loc["a", "ppg"] == 8.0
    assert g.loc["b", "position"] == "TE" and g.loc["b", "rec"] == 1


def test_mixed_offense_defense_frame_no_bracket_leak():
    """Offensive rows (NaN pts_allow) must not be scored as 'shutout' defenses."""
    e = ScoringEngine({"pts_allow_0": 10, "yds_allow_0_100": 5, "rec": 1})
    df = pd.DataFrame([{"position": "WR", "rec": 3}, {"position": "DEF", "pts_allow": 0, "yds_allow": 50}])
    ScoringEngine.add_derived_keys(df)
    assert e.score_frame(df).tolist() == [3.0, 15.0]
