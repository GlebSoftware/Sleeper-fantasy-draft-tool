"""In-season tables: measured weekly spread, schedule, defence versus position.

The point of these tables is that a weekly question needs a weekly answer. ``Projection.std`` is the
uncertainty of a season total and ``ppg_std_ppr`` is the error of a season ppg forecast; neither is
how far a player swings from one Sunday to the next, and a win probability built on the wrong one
reports false confidence. These tests pin the distinction and the arithmetic around it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from draftadvisor.data.inseason import (DVP_CLIP, build_inseason_tables, dvp_table, schedule_table,
                                        weekly_ppr_points, weekly_sigma_table)
from draftadvisor.models import Player, Projection
from draftadvisor.projections.inseason import (InSeasonTables, lineup_mean_sigma, week_mean, week_sigma,
                                               win_probability)

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "web_bundle" / "inseason.json"


# --------------------------------------------------------------------------- offline tables

def _weeks(player_id: str, position: str, team: str, season: int, rec_yds: list[float], opp: str = "OPP"):
    """One player-season of receiving weeks (10 yards = 1 PPR point, 1 catch = 1)."""
    return pd.DataFrame([{"player_id": player_id, "position": position, "team": team, "opponent": opp,
                          "season": season, "week": i + 1, "rec": 1.0, "rec_yd": y}
                         for i, y in enumerate(rec_yds)])


def test_weekly_spread_is_measured_within_season_not_across_it():
    """A player who changed role between years is not called volatile for it."""
    steady_a = _weeks("steady", "WR", "AAA", 2023, [100.0] * 8)      # 11 pts every week
    steady_b = _weeks("steady", "WR", "AAA", 2024, [200.0] * 8)      # 21 pts every week, still steady
    swingy = _weeks("swingy", "WR", "BBB", 2024, [0.0, 200.0] * 4)   # same mean, wild
    out = weekly_sigma_table(pd.concat([steady_a, steady_b, swingy], ignore_index=True), min_games=6)
    a, b = out["players"]["steady"], out["players"]["swingy"]
    assert a["weeks"] == 16 and b["weeks"] == 8
    assert a["sd"] < b["sd"], (a, b)
    # the steady player's own spread is zero; only the position prior keeps his sd off the floor
    assert 0 < a["sd"] < 4.0


def test_a_thin_sample_is_pulled_toward_the_position_curve():
    """Six games of luck must not become a confident sigma."""
    many = pd.concat([_weeks(f"p{i}", "WR", "AAA", 2024, [50.0, 150.0, 60.0, 140.0, 40.0, 160.0, 90.0, 110.0])
                      for i in range(40)], ignore_index=True)
    lucky = _weeks("lucky", "WR", "BBB", 2024, [100.0] * 6)          # flat, and only 6 games
    out = weekly_sigma_table(pd.concat([many, lucky], ignore_index=True), min_games=6)
    a, b = out["curve"]["WR"]
    curve_sd = a + b * out["players"]["lucky"]["ppg"]
    assert out["players"]["lucky"]["sd"] > 0.3 * curve_sd, "a flat six games should not read as no risk"
    assert out["players"]["lucky"]["sd"] < curve_sd


def test_ppr_scoring_of_a_week_is_the_ordinary_one():
    df = _weeks("x", "WR", "AAA", 2024, [100.0])
    assert weekly_ppr_points(df).iloc[0] == pytest.approx(11.0)      # 10 yards/pt + 1 reception


def test_schedule_is_two_sided_and_a_bye_is_a_missing_week():
    games = pd.DataFrame([
        {"season": 2026, "game_type": "REG", "week": 1, "home_team": "KC", "away_team": "BUF"},
        {"season": 2026, "game_type": "REG", "week": 2, "home_team": "BUF", "away_team": "KC"},
        {"season": 2026, "game_type": "REG", "week": 3, "home_team": "BUF", "away_team": "NE"},
        {"season": 2025, "game_type": "REG", "week": 1, "home_team": "KC", "away_team": "NE"},
        {"season": 2026, "game_type": "POST", "week": 4, "home_team": "KC", "away_team": "BUF"},
    ])
    sched = schedule_table(games, 2026)
    assert sched["KC"]["1"] == {"opp": "BUF", "home": True}
    assert sched["BUF"]["1"] == {"opp": "KC", "home": False}
    assert set(sched["KC"]) == {"1", "2"}                            # other seasons and the playoffs are out
    tables = InSeasonTables(schedule=sched)
    assert tables.is_bye("KC", 3) and not tables.is_bye("BUF", 3)
    assert not tables.is_bye("ZZZ", 3), "an unknown team is unknown, not on a bye"


def test_defence_multiplier_is_shrunk_and_clipped():
    """A defence is a nudge. One historic season cannot double or halve a projection."""
    rows = []
    for opp, yards in (("SOFT", 300.0), ("HARD", 20.0), ("MID", 100.0)):
        for wk in range(1, 18):
            rows.append({"player_id": f"{opp}{wk}", "position": "WR", "team": "AAA", "opponent": opp,
                         "season": 2025, "week": wk, "rec": 1.0, "rec_yd": yards})
    dvp = dvp_table(pd.DataFrame(rows), 2025)
    assert dvp["SOFT"]["WR"] == DVP_CLIP[1] and dvp["HARD"]["WR"] == DVP_CLIP[0]
    assert DVP_CLIP[0] < dvp["MID"]["WR"] < DVP_CLIP[1]


def test_dvp_falls_back_to_the_last_season_with_games():
    """In September the current season is empty; a table built from it would be all 1.0 and look fine."""
    rows = [{"player_id": "p", "position": "WR", "team": "AAA", "opponent": "OPP",
             "season": 2025, "week": w, "rec": 1.0, "rec_yd": 100.0} for w in range(1, 18)]
    games = pd.DataFrame([{"season": 2026, "game_type": "REG", "week": 1, "home_team": "KC", "away_team": "BUF"}])
    tables = build_inseason_tables(pd.DataFrame(rows), games, 2026)
    assert tables["season"] == 2026 and tables["dvp_season"] == 2025 and tables["dvp"]


# --------------------------------------------------------------------------- runtime

def _fixture() -> InSeasonTables:
    return InSeasonTables.from_dict({
        "season": 2026, "dvp_season": 2025,
        "sigma": {"players": {"steady": {"sd": 3.0, "ppg": 15.0, "weeks": 40},
                              "boom": {"sd": 12.0, "ppg": 15.0, "weeks": 40}},
                  "curve": {"WR": [2.0, 0.45]}},
        "schedule": {"AAA": {"1": {"opp": "SOFT", "home": True}, "2": {"opp": "HARD", "home": False}},
                     "SOFT": {"1": {"opp": "AAA", "home": False}}},
        "dvp": {"SOFT": {"WR": 1.15}, "HARD": {"WR": 0.85}},
    })


def _pair(pid: str, ppg: float, games: float = 17.0, team: str = "AAA"):
    pl = Player(player_id=pid, name=pid, position="WR", team=team)
    pr = Projection(player_id=pid, position="WR", points=ppg * games, std=20.0, ppg=ppg, games=games)
    return pl, pr


def test_the_opponent_moves_the_week_and_a_bye_is_zero():
    t = _fixture()
    pl, pr = _pair("steady", 15.0)
    assert week_mean(pr, pl, t, 1) == pytest.approx(15.0 * 1.15)
    assert week_mean(pr, pl, t, 2) == pytest.approx(15.0 * 0.85)
    assert week_mean(pr, pl, t, 3) == 0.0 and week_sigma(pr, pl, t, 3) == 0.0   # no week-3 game
    assert week_sigma(pr, pl, t, 1) > 0


def test_weekly_spread_is_the_players_own_not_the_positions_when_it_is_known():
    t = _fixture()
    steady = week_sigma(*reversed(_pair("steady", 15.0)), t)          # sd/ppg = 0.20 -> clipped up to 0.25
    boom = week_sigma(*reversed(_pair("boom", 15.0)), t)              # sd/ppg = 0.80
    assert boom > steady * 2.5
    unknown = week_sigma(*reversed(_pair("rookie", 15.0)), t)         # falls back to the WR curve
    assert steady < unknown < boom


def test_spread_scales_with_the_leagues_scoring():
    """Doubling the points a player scores doubles the spread: the ratio is what is stored."""
    t = _fixture()
    a = week_sigma(*reversed(_pair("boom", 15.0)), t)
    b = week_sigma(*reversed(_pair("boom", 30.0)), t)
    assert b == pytest.approx(2 * a)


def test_missed_games_lower_the_weekly_expectation():
    t = _fixture()
    full = week_mean(*reversed(_pair("steady", 15.0, games=17.0)), t, 1)
    half = week_mean(*reversed(_pair("steady", 15.0, games=8.0)), t, 1)
    assert half < full and half == pytest.approx(full * 8.0 / 17.0, rel=1e-6)


def test_win_probability_is_symmetric_and_uncertainty_pulls_it_to_a_coin_flip():
    assert win_probability((100.0, 20.0), (100.0, 20.0)) == pytest.approx(0.5)
    tight = win_probability((110.0, 5.0), (100.0, 5.0))
    loose = win_probability((110.0, 40.0), (100.0, 40.0))
    assert tight > loose > 0.5
    assert win_probability((110.0, 20.0), (100.0, 20.0)) == pytest.approx(
        1 - win_probability((100.0, 20.0), (110.0, 20.0)))
    assert win_probability((110.0, 0.0), (100.0, 0.0)) == 1.0        # no uncertainty, no doubt


def test_a_lineup_adds_means_and_adds_variances():
    t = _fixture()
    one = _pair("boom", 15.0)
    mean, sd = lineup_mean_sigma([one, one, one, one], t, 1)
    m1, s1 = week_mean(one[1], one[0], t, 1), week_sigma(one[1], one[0], t, 1)
    assert mean == pytest.approx(4 * m1)
    assert sd == pytest.approx(2 * s1), "four independent players, not four times the spread"


def test_empty_tables_never_crash_and_never_invent_a_bye():
    t = InSeasonTables()
    assert not t.present
    pl, pr = _pair("x", 15.0)
    assert week_mean(pr, pl, t, 1) == pytest.approx(15.0)             # no schedule: play everyone, no adjustment
    assert week_sigma(pr, pl, t, 1) > 0                               # falls back to the default curve
    assert not t.is_bye("AAA", 1)


# --------------------------------------------------------------------------- the shipped table

@pytest.mark.skipif(not BUNDLE.exists(), reason="needs web_bundle/inseason.json")
def test_the_shipped_table_is_sane():
    t = InSeasonTables.from_dict(json.loads(BUNDLE.read_text()))
    assert t.present and len(t.schedule) == 32 and len(t.sigma) > 500
    assert all(len(v) in (17, 18) for v in t.schedule.values()), "every team plays 17 of 18 weeks"
    assert t.dvp_season < t.season, "defence-vs-position must come from a season that was played"
    for pos, (a, b) in t.curve.items():
        assert 0.0 <= b <= 1.5 and 0.0 < a <= 12.0, (pos, a, b)
    # the measured weekly spread is materially larger than the season-ppg forecast error it replaces
    ml = json.loads((ROOT / "web_bundle" / "ml.json").read_text())
    ratios = []
    for pid, row in ml.items():
        pos, ppg, err = row.get("position"), row.get("pred_ppg_ppr"), row.get("ppg_std_ppr")
        if pos in t.curve and ppg and err and ppg > 8.0:
            a, b = t.curve[pos]
            ratios.append((a + b * ppg) / err)
    assert ratios and sum(ratios) / len(ratios) > 1.15, sum(ratios) / len(ratios)
