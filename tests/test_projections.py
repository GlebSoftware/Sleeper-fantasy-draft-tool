"""Tests for draftadvisor.projections (features, model, blend). Offline unless data/raw exists."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from draftadvisor.config import FANTASY_WEEKS, GAMES_PER_TEAM
from draftadvisor.data.crosswalk import Crosswalk
from draftadvisor.models import Player, ResearchNote
from draftadvisor.projections import (
    FEATURE_COLUMNS,
    TARGETS,
    ProjectionModel,
    Projector,
    build_inference_table,
    build_training_table,
    def_bracket_rates,
    ecr_implied_points,
    project_offline,
    rates_to_season,
    season_aggregates,
    sleeper_stats_to_projection,
)
from draftadvisor.projections.blend import fit_ecr_curve
from draftadvisor.projections.features import ALL_TARGET_KEYS, TARGET_COLUMNS
from draftadvisor.scoring import ScoringEngine

REPO_RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

PPR = {"pass_yd": 0.04, "pass_td": 4, "pass_int": -1, "rush_yd": 0.1, "rush_td": 6, "rec": 1, "rec_yd": 0.1,
       "rec_td": 6, "fum_lost": -2, "bonus_rec_yd_100": 3, "bonus_rec_te": 0.5,
       "fgm_0_19": 3, "fgm_20_29": 3, "fgm_30_39": 3, "fgm_40_49": 4, "fgm_50p": 5, "fgmiss": -1, "xpm": 1, "xpmiss": -1,
       "sack": 1, "int": 2, "ff": 1, "fum_rec": 2, "safe": 2, "def_td": 6, "blk_kick": 2,
       "pts_allow_0": 10, "pts_allow_1_6": 7, "pts_allow_7_13": 4, "pts_allow_14_20": 1, "pts_allow_21_27": 0,
       "pts_allow_28_34": -1, "pts_allow_35p": -4}


# ---------------------------------------------------------------------------
# synthetic canonical data
# ---------------------------------------------------------------------------

def _synthetic_canonical(seasons=(2021, 2022, 2023), n_per_pos=40, seed=0) -> pd.DataFrame:
    """Per-game rows for QB/RB/WR/TE/K/DEF where each player has a stable talent level
    plus a season-specific shift, so season S is only predictable from < S if there is no leakage."""
    rng = np.random.default_rng(seed)
    rows = []
    teams = [f"T{i}" for i in range(8)]
    for pos in ("QB", "RB", "WR", "TE", "K"):
        for i in range(n_per_pos):
            pid = f"{pos}{i:03d}"
            talent = rng.uniform(0.2, 1.0)
            team = teams[i % len(teams)]
            for s in seasons:
                shift = rng.normal(0, 0.15)
                games = int(rng.integers(6, 18))
                for w in range(1, games + 1):
                    lvl = max(0.05, talent + shift)
                    r = {"player_id": pid, "player_name": pid, "position": pos, "team": team, "opponent": "X",
                         "season": s, "week": w}
                    if pos == "QB":
                        r.update(pass_att=35 * lvl, pass_cmp=22 * lvl, pass_yd=rng.normal(260, 40) * lvl,
                                 pass_td=rng.poisson(2 * lvl), pass_int=rng.poisson(0.7), rush_att=4 * lvl,
                                 rush_yd=rng.normal(15, 10) * lvl, rush_td=rng.poisson(0.2 * lvl), fum_lost=rng.poisson(0.05))
                    elif pos == "K":
                        r.update(fgm_0_19=0.1, fgm_20_29=0.5 * lvl, fgm_30_39=0.6 * lvl, fgm_40_49=0.5 * lvl,
                                 fgm_50p=0.2 * lvl, fgmiss=0.3, xpm=2.5 * lvl, xpmiss=0.1)
                    else:
                        r.update(rush_att=(12 if pos == "RB" else 0.5) * lvl, rush_yd=rng.normal(55, 20) * lvl * (1 if pos == "RB" else 0.05),
                                 rush_td=rng.poisson(0.4 * lvl) if pos == "RB" else 0,
                                 rec_tgt=(4 if pos == "RB" else 8) * lvl, rec=(3 if pos == "RB" else 5.5) * lvl,
                                 rec_yd=rng.normal(25 if pos == "RB" else 70, 20) * lvl, rec_td=rng.poisson(0.3 * lvl),
                                 fum_lost=rng.poisson(0.05), f_target_share=0.2 * lvl, f_air_yards_share=0.2 * lvl, f_wopr=0.4 * lvl)
                    rows.append(r)
    for t in teams:
        for s in seasons:
            for w in range(1, 18):
                rows.append({"player_id": t, "player_name": f"{t} Defense", "position": "DEF", "team": t, "opponent": "X",
                             "season": s, "week": w, "sack": rng.poisson(2.3), "int": rng.poisson(0.8), "ff": rng.poisson(0.6),
                             "fum_rec": rng.poisson(0.5), "safe": rng.poisson(0.05), "def_td": rng.poisson(0.15),
                             "blk_kick": rng.poisson(0.05), "pts_allow": max(0, rng.normal(22, 8)), "yds_allow": rng.normal(340, 60)})
    df = pd.DataFrame(rows)
    for k in ("rush_yd", "rec_yd", "pass_yd"):
        if k in df.columns:
            df[k] = df[k].fillna(0.0).clip(lower=0)
    df["f_nfl_fantasy_points_ppr"] = np.nan
    ScoringEngine.add_derived_keys(df)
    return df


def _roster(canonical: pd.DataFrame, season: int) -> pd.DataFrame:
    p = canonical[(canonical["position"] != "DEF")].drop_duplicates("player_id")
    return pd.DataFrame({
        "gsis_id": p["player_id"], "position": p["position"], "team": p["team"], "birth_date": "1998-06-01",
        "years_exp": 3, "entry_year": 2019, "draft_number": 50, "status": "ACT", "week": 1,
    })


def _team_ctx(canonical: pd.DataFrame) -> pd.DataFrame:
    from draftadvisor.data.nflverse import team_context
    return team_context(canonical)


@pytest.fixture(scope="module")
def synth():
    c = _synthetic_canonical()
    agg = season_aggregates(c)
    ctx = _team_ctx(c)
    rosters = {s: _roster(c, s) for s in (2022, 2023)}
    table = build_training_table(agg, ctx, rosters, [2022, 2023])
    return c, agg, ctx, rosters, table


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

def test_season_aggregates_shape_and_rates(synth):
    c, agg, *_ = synth
    assert set(["player_id", "season", "position", "team", "games", "ppg_ppr", "ppg_std"]).issubset(agg.columns)
    assert all(k in agg.columns for k in ALL_TARGET_KEYS)
    # one row per (player, season)
    assert not agg.duplicated(["player_id", "season"]).any()
    one = c[(c.player_id == "WR000") & (c.season == 2022)]
    row = agg[(agg.player_id == "WR000") & (agg.season == 2022)].iloc[0]
    assert row["games"] == len(one)
    assert row["rec_yd"] == pytest.approx(one["rec_yd"].mean())
    assert row["ppg_ppr"] == pytest.approx(ScoringEngine({**PPR}).score_frame(one).mean(), rel=0.2)
    d = agg[agg.position == "DEF"]
    assert (d["games"] == 17).all()


def test_training_table_columns_and_targets(synth):
    _, agg, ctx, rosters, table = synth
    assert all(c in table.columns for c in FEATURE_COLUMNS)
    assert all(c in table.columns for c in TARGET_COLUMNS)
    assert set(table["season"]) == {2022, 2023}
    assert set(table["position"]) == {"QB", "RB", "WR", "TE", "K", "DEF"}
    # target rates are NaN exactly when y_games == 0
    wr = table[table.position == "WR"]
    assert ((wr["y_rec_yd"].isna()) == (wr["y_games"] == 0)).all()
    assert (table["y_games"] <= 18).all()


def test_no_target_leakage(synth):
    """Features for season S must not change when season S (or later) stats change."""
    c, agg, ctx, rosters, table = synth
    # perturb season 2023 wildly and rebuild: 2023 features must be identical
    c2 = c.copy()
    m = c2["season"] == 2023
    c2.loc[m, ["rec_yd", "rush_yd", "pass_yd"]] = c2.loc[m, ["rec_yd", "rush_yd", "pass_yd"]].fillna(0) * 10 + 500
    agg2 = season_aggregates(c2)
    t2 = build_training_table(agg2, _team_ctx(c2), rosters, [2022, 2023])
    a = table[table.season == 2023].set_index("player_id").sort_index()
    b = t2[t2.season == 2023].set_index("player_id").sort_index()
    assert list(a.index) == list(b.index)
    pd.testing.assert_frame_equal(a[FEATURE_COLUMNS], b[FEATURE_COLUMNS], check_dtype=False)
    # ... while the targets did change
    assert not np.allclose(a["y_rec_yd"].fillna(0), b["y_rec_yd"].fillna(0))
    # and the 2022 rows are unaffected too (they only use 2021)
    a22 = table[table.season == 2022].set_index("player_id").sort_index()
    b22 = t2[t2.season == 2022].set_index("player_id").sort_index()
    pd.testing.assert_frame_equal(a22[FEATURE_COLUMNS], b22[FEATURE_COLUMNS], check_dtype=False)
    # sanity: prev_* really is S-1
    wr = table[(table.season == 2023) & (table.player_id == "WR001")].iloc[0]
    prev = agg[(agg.player_id == "WR001") & (agg.season == 2022)].iloc[0]
    assert wr["prev_rec_yd"] == pytest.approx(prev["rec_yd"])
    assert wr["prev_games"] == prev["games"]


def test_rookie_rows_have_nan_history(synth):
    c, agg, ctx, rosters, table = synth
    ro = rosters[2023].copy()
    ro = pd.concat([ro, pd.DataFrame([{"gsis_id": "NEWGUY", "position": "RB", "team": "T0", "birth_date": "2001-01-01",
                                        "years_exp": 0, "entry_year": 2023, "draft_number": 12, "status": "ACT", "week": 1}])])
    t = build_training_table(agg, ctx, {2023: ro}, [2023])
    r = t[t.player_id == "NEWGUY"].iloc[0]
    assert r["rookie"] == 1 and r["undrafted"] == 0 and r["draft_ovr"] == 12
    assert np.isnan(r["prev_ppg_ppr"]) and np.isnan(r["career_ppg_ppr"])
    assert r["y_games"] == 0
    # veterans are not rookies
    assert (t[t.player_id != "NEWGUY"]["rookie"] == 0).all()


def test_inference_table_uses_crosswalk(synth):
    c, agg, ctx, *_ = synth
    cw = Crosswalk()
    cw.add("1001", gsis_id="WR000", name="Wr Zero", position="WR", draft_ovr=20, age=25.0)
    players = {
        "1001": Player("1001", "Wr Zero", "WR", team="T1", years_exp=3),
        "1002": Player("1002", "No History", "RB", team="T2", years_exp=0, draft_pick_overall=5, age=22.0),
        "T3": Player("T3", "T3 Defense", "DEF", team="T3"),
    }
    t = build_inference_table(agg, ctx, players, cw, 2024)
    assert list(t.index) == ["1001", "1002", "T3"]
    prev = agg[(agg.player_id == "WR000") & (agg.season == 2023)].iloc[0]
    assert t.loc["1001", "prev_rec_yd"] == pytest.approx(prev["rec_yd"])
    assert t.loc["1001", "team_changed"] == 1.0  # T0 -> T1
    assert t.loc["1001", "draft_ovr"] == 20
    assert t.loc["1002", "rookie"] == 1.0 and np.isnan(t.loc["1002", "prev_ppg_ppr"])
    assert t.loc["T3", "prev_sack"] == pytest.approx(agg[(agg.player_id == "T3") & (agg.season == 2023)].iloc[0]["sack"])
    assert all(col in t.columns for col in FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# model (synthetic, fast)
# ---------------------------------------------------------------------------

def test_model_fit_predict_backtest_synthetic(synth, tmp_path):
    _, agg, ctx, rosters, table = synth
    params = {"max_iter": 40, "learning_rate": 0.1}
    m = ProjectionModel(seed=1, params=params).fit(table)
    pred = m.predict(table)
    assert len(pred) == len(table)
    assert {"pred_games", "pred_ppg_ppr", "ppg_std_ppr", "rookie_flag", "position"}.issubset(pred.columns)
    for pos, keys in TARGETS.items():
        sub = pred[pred.position == pos]
        for k in keys:
            assert sub[f"pred_{k}"].notna().all() and (sub[f"pred_{k}"] >= 0).all()
    assert pred["pred_games"].between(0, GAMES_PER_TEAM).all()
    assert (pred["ppg_std_ppr"] > 0).all()
    # heteroscedastic: std grows with the prediction
    wr = pred[pred.position == "WR"]
    assert np.corrcoef(wr["pred_ppg_ppr"], wr["ppg_std_ppr"])[0, 1] > 0.99
    # persistence round-trip
    p = m.save(tmp_path / "m.pkl")
    m2 = ProjectionModel.load(p)
    pd.testing.assert_frame_equal(m2.predict(table), pred)
    res = m.backtest(table, holdout_seasons=[2023])
    assert "WR" in res and res["WR"]["n"] > 0 and "mae_last" in res["WR"]


# ---------------------------------------------------------------------------
# blend
# ---------------------------------------------------------------------------

def test_rates_to_season_hand_computed():
    eng = ScoringEngine(PPR)
    rates = {"rec": 5.0, "rec_yd": 70.0, "rec_td": 0.5, "rec_tgt": 7.0, "rush_att": 0.0, "rush_yd": 0.0, "rush_td": 0.0,
             "fum_lost": 0.1, "bonus_rush_yd_100": 0.0, "bonus_rec_yd_100": 0.2}
    pts, ppg, line = rates_to_season(rates, 16, eng, "WR")
    expected_ppg = 5 + 7.0 + 3.0 - 0.2 + 0.6
    assert ppg == pytest.approx(expected_ppg)
    assert pts == pytest.approx(expected_ppg * 16)
    assert line["rec"] == pytest.approx(80) and line["bonus_rec_wr"] == pytest.approx(80)
    # TE premium applies only to TEs
    pts_te, ppg_te, _ = rates_to_season(rates, 16, eng, "TE")
    assert ppg_te == pytest.approx(expected_ppg + 2.5)
    # QB derived pass_inc; K totals; DEF brackets
    _, _, ql = rates_to_season({"pass_att": 30, "pass_cmp": 20}, 10, eng, "QB")
    assert ql["pass_inc"] == pytest.approx(100)
    _, kppg, kl = rates_to_season({"fgm_0_19": 0, "fgm_20_29": 1, "fgm_30_39": 1, "fgm_40_49": 0.5, "fgm_50p": 0.25,
                                   "fgmiss": 0.25, "xpm": 2, "xpmiss": 0.1}, 17, eng, "K")
    assert kl["fgm"] == pytest.approx(2.75 * 17) and kl["fga"] == pytest.approx(3.0 * 17) and kl["xpa"] == pytest.approx(2.1 * 17)
    assert kppg == pytest.approx(3 + 3 + 2 + 1.25 - 0.25 + 2 - 0.1)
    _, dppg, dl = rates_to_season({"sack": 2.5, "int": 1, "ff": 0.5, "fum_rec": 0.5, "safe": 0, "def_td": 0.1, "blk_kick": 0,
                                   "pts_allow": 18.0, "yds_allow": 330}, 17, eng, "DEF")
    br = def_bracket_rates(18.0)
    assert sum(br.values()) == pytest.approx(1.0)
    assert dl["pts_allow_14_20"] == pytest.approx(br["pts_allow_14_20"] * 17)
    assert dppg == pytest.approx(2.5 + 2 + 0.5 + 1 + 0.6 + sum(PPR[k] * v for k, v in br.items()))
    # zero games -> zero points but a valid ppg
    pts0, ppg0, _ = rates_to_season(rates, 0, eng, "WR")
    assert pts0 == 0 and ppg0 == pytest.approx(expected_ppg)


def test_def_bracket_rates_monotone():
    good, bad = def_bracket_rates(16.0), def_bracket_rates(29.0)
    assert good["pts_allow_7_13"] > bad["pts_allow_7_13"]
    assert bad["pts_allow_35p"] > good["pts_allow_35p"]
    assert sum(bad.values()) == pytest.approx(1.0)


def test_sleeper_stats_to_projection(projections_json):
    eng = ScoringEngine(PPR)
    pts, games, line = sleeper_stats_to_projection(projections_json["7564"], eng, "WR")
    assert games == 17.0
    s = projections_json["7564"]
    assert pts == pytest.approx(s["rec"] + 0.1 * s["rec_yd"] + 6 * s["rec_td"] + 0.1 * s["rush_yd"] - 2 * s["fum_lost"])
    assert "adp_ppr" not in line and "pts_ppr" not in line
    pts2, games2, _ = sleeper_stats_to_projection({"rec": 10, "gp": 0}, eng, "WR")
    assert games2 == GAMES_PER_TEAM and pts2 == 10


def _players_with_ecr(n=30, pos="WR"):
    players = {}
    for i in range(n):
        players[str(i)] = Player(str(i), f"P{i}", pos, team="KC", ecr=float(i + 1), ecr_sd=2.0)
    return players


def test_ecr_implied_points_monotone():
    players = _players_with_ecr()
    rng = np.random.default_rng(3)
    provisional = {pid: 300 - 8 * pl.ecr + rng.normal(0, 15) for pid, pl in players.items()}
    imp = ecr_implied_points(players, provisional, "WR")
    assert set(imp) == set(players)
    vals = [imp[str(i)] for i in range(30)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))
    assert vals[0] > vals[-1]
    # players with an ECR but no provisional still get a value
    players["99"] = Player("99", "NoProv", "WR", ecr=5.5)
    imp2 = ecr_implied_points(players, provisional, "WR")
    assert imp2["99"] <= imp2["3"] and imp2["99"] >= imp2["6"]
    # too few points -> skipped
    assert ecr_implied_points({k: v for k, v in list(players.items())[:5]}, provisional, "WR") == {}
    curve = fit_ecr_curve(players, provisional, "WR")
    assert curve.sd_points(10.0, 2.0) > 0


def _ml_pred(pid_rates: dict[str, dict], position: str, games=16.0, std=4.0) -> pd.DataFrame:
    rows = []
    for pid, rates in pid_rates.items():
        r = {f"pred_{k}": np.nan for k in ALL_TARGET_KEYS}
        r.update({f"pred_{k}": v for k, v in rates.items()})
        r.update(pred_games=games, ppg_std_ppr=std, rookie_flag=0.0, position=position)
        rows.append((pid, r))
    return pd.DataFrame([r for _, r in rows], index=[p for p, _ in rows])


WR_RATES = {"rec": 5.0, "rec_yd": 70.0, "rec_td": 0.5, "rec_tgt": 7.0, "rush_att": 0.0, "rush_yd": 0.0, "rush_td": 0.0,
            "fum_lost": 0.0, "bonus_rush_yd_100": 0.0, "bonus_rec_yd_100": 0.0}


def test_blend_weight_renormalisation():
    eng = ScoringEngine(PPR)
    pr = Projector(eng)
    players = {"a": Player("a", "A", "WR", team="KC"), "b": Player("b", "B", "WR", team="KC"), "c": Player("c", "C", "WR", team="KC")}
    ml = _ml_pred({"a": WR_RATES, "b": WR_RATES}, "WR")
    sleeper = {"a": {"rec": 90, "rec_yd": 1200, "rec_td": 8, "gp": 17}, "c": {"rec": 60, "rec_yd": 800, "rec_td": 5, "gp": 15}}
    out = pr.project(players, ml_pred=ml, sleeper_proj=sleeper)
    ml_pts = 15 * 16.0
    sl_a = 90 + 120 + 48
    # a: both sources -> weights 0.45/0.35 renormalised
    assert out["a"].weights == pytest.approx({"sleeper": 0.40 / 0.65, "ml": 0.25 / 0.65})
    assert out["a"].points == pytest.approx((0.40 * sl_a + 0.25 * ml_pts) / 0.65)
    assert out["a"].games == pytest.approx((0.40 * 17 + 0.25 * 16) / 0.65)
    # b: ml only
    assert out["b"].weights == pytest.approx({"ml": 1.0}) and out["b"].points == pytest.approx(ml_pts)
    assert out["b"].components == pytest.approx({"ml": ml_pts})
    # c: sleeper only
    assert out["c"].weights == pytest.approx({"sleeper": 1.0}) and out["c"].games == 15
    assert out["c"].ppg == pytest.approx((60 + 80 + 30) / 15)
    for p in out.values():
        assert p.std >= 0.08 * p.points and p.floor <= p.points <= p.ceiling
        assert len(p.weekly) == FANTASY_WEEKS and sum(p.weekly) == pytest.approx(p.points)


def test_projector_ecr_source_and_ecr_only_players():
    eng = ScoringEngine(PPR)
    pr = Projector(eng)
    players = _players_with_ecr(20)
    ml = _ml_pred({str(i): {**WR_RATES, "rec_yd": 110.0 - 4 * i, "rec": 7.0 - 0.2 * i} for i in range(15)}, "WR")
    players["rk"] = Player("rk", "Rookie", "WR", team="KC", ecr=3.0, ecr_sd=3.0, years_exp=0)
    out = pr.project(players, ml_pred=ml)
    assert "ecr" in out["0"].components and out["0"].weights == pytest.approx({"ml": 0.25 / 0.60, "ecr": 0.35 / 0.60})
    # ECR-only rookie gets points from the curve, default games, rookie flag and bigger std
    rk = out["rk"]
    assert rk.weights == pytest.approx({"ecr": 1.0}) and rk.points > 0 and "ecr_only" in rk.flags and "rookie" in rk.flags
    assert rk.games == 16.0
    # players with an ECR way down the board and no ML still get a (lower) value
    assert out["19"].points < out["0"].points
    # no source at all -> no_data
    players["zz"] = Player("zz", "Nobody", "WR")
    out = pr.project(players, ml_pred=ml)
    assert out["zz"].points == 0 and "no_data" in out["zz"].flags


def test_injury_depth_and_research_adjustments():
    eng = ScoringEngine(PPR)
    pr = Projector(eng)
    base = Player("a", "A", "WR", team="KC")
    ir = Player("ir", "IR guy", "WR", team="KC", injury_status="IR", status="Injured Reserve")
    out_ = Player("o", "Out guy", "WR", team="KC", injury_status="Out")
    sus = Player("s", "Sus guy", "WR", team="KC", injury_status="Sus")
    deep = Player("d", "Deep", "RB", team="KC", depth_chart_order=3)
    deep_ecr = Player("de", "Deep but ranked", "RB", team="KC", depth_chart_order=3, ecr=50.0)
    players = {p.player_id: p for p in (base, ir, out_, sus, deep, deep_ecr)}
    rb_rates = {**WR_RATES, "rush_att": 15.0, "rush_yd": 65.0, "rush_td": 0.5}
    ml = _ml_pred({"a": WR_RATES, "ir": WR_RATES, "o": WR_RATES, "s": WR_RATES}, "WR")
    ml = pd.concat([ml, _ml_pred({"d": rb_rates, "de": rb_rates}, "RB")])
    out = pr.project(players, ml_pred=ml)
    assert out["a"].games == 16.0
    assert out["ir"].games == pytest.approx(10 * 0.3) and "injury:IR" in out["ir"].flags
    assert out["ir"].points == pytest.approx(out["a"].ppg * 3.0)
    assert out["o"].games == 15.0 and out["s"].games == 12.0
    assert out["d"].ppg == pytest.approx(0.8 * out["de"].ppg) and "depth3" in out["d"].flags
    assert out["de"].ppg == pytest.approx(out["a"].ppg + 15 * 0.1 * 0 + 6.5 + 3.0)  # rush 6.5 + 0.5 td
    # research note: injury risk cuts games, role certainty scales std
    note = ResearchNote("a", "x", injury_risk=0.4, role_certainty=1.0, upside="", downside="")
    out2 = pr.project(players, ml_pred=ml, notes={"a": note})
    assert out2["a"].games == pytest.approx(16 * 0.9)
    note0 = ResearchNote("a", "x", injury_risk=0.0, role_certainty=0.0, upside="", downside="")
    out3 = pr.project(players, ml_pred=ml, notes={"a": note0})
    assert out3["a"].std == pytest.approx(out["a"].std * 1.3)


def test_weekly_zero_on_bye():
    eng = ScoringEngine(PPR)
    pr = Projector(eng)
    players = {"a": Player("a", "A", "WR", team="KC", bye_week=7), "b": Player("b", "B", "WR", team="DET")}
    ml = _ml_pred({"a": WR_RATES, "b": WR_RATES}, "WR")
    out = pr.project(players, ml_pred=ml, byes={"DET": 5})
    assert out["a"].weekly[6] == 0.0 and out["a"].weekly[0] > 0
    assert out["b"].weekly[4] == 0.0
    assert sum(out["a"].weekly) == pytest.approx(out["a"].points)
    assert out["a"].points_through(7) == pytest.approx(out["a"].points * 6 / 16)


def test_project_offline_shape_and_shrinkage(synth):
    c, *_ = synth
    eng = ScoringEngine(PPR)
    cw = Crosswalk()
    cw.add("1", gsis_id="WR000", name="A", position="WR")
    cw.add("2", gsis_id="RB000", name="B", position="RB")
    players = {"1": Player("1", "A", "WR", team="T0"), "2": Player("2", "B", "RB", team="T1"),
               "T2": Player("T2", "T2 Defense", "DEF", team="T2"), "9": Player("9", "Rookie", "WR", team="T0")}
    df = project_offline(players, eng, c, cw, 2024)
    assert list(df.index) == ["1", "2", "T2"]
    assert {"pred_games", "pred_ppg_ppr", "ppg_std_ppr", "rookie_flag", "position"}.issubset(df.columns)
    assert all(f"pred_{k}" in df.columns for k in ALL_TARGET_KEYS)
    agg = season_aggregates(c[c.season == 2023])
    last = agg[agg.player_id == "WR000"].iloc[0]
    g = last["games"]
    pos_mean_rows = agg[(agg.position == "WR") & (agg.games >= 6)]
    mean_rec_yd = np.average(pos_mean_rows["rec_yd"], weights=pos_mean_rows["games"])
    assert df.loc["1", "pred_rec_yd"] == pytest.approx((g * last["rec_yd"] + 6 * mean_rec_yd) / (g + 6))
    assert 0 <= df.loc["T2", "pred_games"] <= GAMES_PER_TEAM
    assert np.isnan(df.loc["1", "pred_sack"]) and df.loc["T2", "pred_sack"] > 0
    # Projector consumes it like a model prediction
    out = Projector(eng).project(players, ml_pred=df)
    assert out["1"].points > 0 and out["T2"].points > 0 and "no_data" in out["9"].flags


# ---------------------------------------------------------------------------
# slow, real data
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not REPO_RAW.exists(), reason="needs data/raw")
def test_real_backtest_beats_last_season_baseline(monkeypatch):
    """Train on 2020-2023 (history from 2019), evaluate 2024-2025: beats last-season MAE for RB/WR."""
    monkeypatch.setenv("DRAFTADVISOR_HOME", str(REPO_RAW.parent))
    from draftadvisor.projections.model import load_training_inputs

    agg, ctx, rosters = load_training_inputs(range(2019, 2026))
    table = build_training_table(agg, ctx, rosters, range(2020, 2026))
    res = ProjectionModel().backtest(table, holdout_seasons=[2024, 2025])
    for pos in ("RB", "WR", "TE"):
        assert res[pos]["mae_model"] < res[pos]["mae_last"], res[pos]
        assert res[pos]["spearman_model"] > 0.7


# ---------------------------------------------------------------------------
# review regressions (PROJ-1 .. PROJ-6)
# ---------------------------------------------------------------------------

from draftadvisor.projections.blend import (  # noqa: E402
    MAX_ML_CV,
    MIN_STD_FRACTION,
    PTS_BRACKETS,
    YDS_BRACKETS,
    MarketCurve,
    default_rank_curves,
    yds_bracket_rates,
)


def test_sleeper_def_projection_keeps_points_allowed_keys(projections_json):
    """PROJ-1: the ``pts_`` prefix filter dropped pts_allow and every pts_allow_* bracket."""
    eng = ScoringEngine(PPR)
    sf = projections_json["SF"]
    pts, games, line = sleeper_stats_to_projection(sf, eng, "DEF")
    assert games == 17.0
    assert "pts_ppr" not in line and "adp_ppr" not in line and "gp" not in line
    assert line["pts_allow"] == sf["pts_allow"]
    assert all(line[k] == sf[k] for k in PTS_BRACKETS)
    base = (sf["sack"] + 2 * sf["int"] + sf["ff"] + 2 * sf["fum_rec"] + 2 * sf["safe"] + 6 * sf["def_td"]
            + 2 * sf["blk_kick"] - 2 * sf["fum_lost"])
    assert pts == pytest.approx(base + sum(PPR[k] * sf[k] for k in PTS_BRACKETS))
    # yards-allowed brackets are derived from the season total (Sleeper never ships them)
    assert sum(line[k] for k in YDS_BRACKETS) == pytest.approx(17.0)
    yb = yds_bracket_rates(sf["yds_allow"] / 17.0)
    assert line["yds_allow_300_349"] == pytest.approx(yb["yds_allow_300_349"] * 17.0)
    # points-allowed brackets are derived only when Sleeper omits them
    pts2, _, line2 = sleeper_stats_to_projection({"gp": 17, "pts_allow": 17 * 18.0, "sack": 40}, eng, "DEF")
    br = def_bracket_rates(18.0)
    assert line2["pts_allow_14_20"] == pytest.approx(br["pts_allow_14_20"] * 17)
    assert pts2 == pytest.approx(40 + 17 * sum(PPR[k] * v for k, v in br.items()))
    # a league that scores pts_allow per point sees it too
    eng2 = ScoringEngine({**PPR, "pts_allow": -0.1})
    assert sleeper_stats_to_projection(sf, eng2, "DEF")[0] == pytest.approx(pts - 0.1 * sf["pts_allow"])


def test_ml_games_shrink_reaches_the_season_total():
    """PROJ-2: an injury-shortened pred_games must be shrunk in the ML *points*, not only the label."""
    eng = ScoringEngine(PPR)
    pr = Projector(eng)
    players = {"a": Player("a", "A", "WR", team="KC"), "T1": Player("T1", "T1 Defense", "DEF", team="T1")}
    out = pr.project(players, ml_pred=_ml_pred({"a": WR_RATES}, "WR", games=8.0))
    ppg = 5 + 7.0 + 3.0
    g = 8.0 + 0.6 * (16.0 - 8.0)
    assert out["a"].games == pytest.approx(g)
    assert out["a"].components["ml"] == pytest.approx(ppg * g)
    assert out["a"].points == pytest.approx(ppg * g) and out["a"].ppg == pytest.approx(ppg)
    # a full season is left alone; a DEF is never shrunk
    full = pr.project(players, ml_pred=_ml_pred({"a": WR_RATES}, "WR", games=17.0))
    assert full["a"].games == 17.0 and full["a"].points == pytest.approx(ppg * 17)
    def_rates = {"sack": 2.5, "int": 1, "ff": 0.5, "fum_rec": 0.5, "safe": 0, "def_td": 0.1, "blk_kick": 0,
                 "pts_allow": 18.0, "yds_allow": 330}
    d = pr.project(players, ml_pred=_ml_pred({"T1": def_rates}, "DEF", games=12.0))
    assert d["T1"].games == 12.0


def test_std_adds_ecr_sd_in_quadrature():
    """PROJ-4: a tight expert consensus must not shrink the outcome risk below the ML std."""
    eng = ScoringEngine(PPR)
    pr = Projector(eng, rank_curves=default_rank_curves())
    players = _players_with_ecr(20)
    players["nosd"] = Player("nosd", "No sd", "WR", team="KC", ecr=5.0, ecr_sd=None)
    players["sd"] = Player("sd", "With sd", "WR", team="KC", ecr=5.0, ecr_sd=2.0)
    ml = _ml_pred({"nosd": WR_RATES, "sd": WR_RATES}, "WR", games=16.0, std=4.0)
    out = pr.project(players, ml_pred=ml)
    curve = MarketCurve.from_universe(players, "WR", pr.rank_curves["WR"])
    e = curve.sd_points(5.0, 2.0)
    assert e > 0
    s0, s1 = out["nosd"].std, out["sd"].std
    pts = out["nosd"].points
    assert out["sd"].points == pytest.approx(pts)
    # ML-only std: std_ppg * games rescaled to league ppg (4 * 16 * (pts/16) / 15), under the raised cap
    assert s0 == pytest.approx(4.0 * pts / 15.0) and s0 <= MAX_ML_CV * pts + 1e-9
    assert s1 > s0 and s1 == pytest.approx(np.sqrt(s0 ** 2 + e ** 2))
    assert s1 >= MIN_STD_FRACTION * pts
    # neither player is a rookie (years_exp unknown)
    assert "rookie" not in out["nosd"].flags and "rookie" not in out["sd"].flags


def test_unknown_years_exp_is_not_a_rookie():
    """PROJ-5: only years_exp == 0 is a rookie; a DEF never is (Sleeper ships years_exp null for them)."""
    eng = ScoringEngine(PPR)
    pr = Projector(eng)
    players = {
        "SF": Player("SF", "SF Defense", "DEF", team="SF"),
        "BAL": Player("BAL", "BAL Defense", "DEF", team="BAL", years_exp=0),
        "v": Player("v", "Vet, unknown exp", "WR", team="KC", years_exp=None),
        "r": Player("r", "Rookie", "WR", team="KC", years_exp=0),
    }
    def_rates = {"sack": 2.5, "int": 1, "ff": 0.5, "fum_rec": 0.5, "safe": 0, "def_td": 0.1, "blk_kick": 0,
                 "pts_allow": 18.0, "yds_allow": 330}
    ml = pd.concat([_ml_pred({"v": WR_RATES, "r": WR_RATES}, "WR"), _ml_pred({"SF": def_rates, "BAL": def_rates}, "DEF")])
    out = pr.project(players, ml_pred=ml)
    assert all("rookie" not in out[pid].flags for pid in ("SF", "BAL", "v"))
    assert "rookie" in out["r"].flags
    assert out["r"].std == pytest.approx(out["v"].std * 1.25)
    assert out["SF"].std == pytest.approx(out["BAL"].std)


def test_injury_deduction_spares_the_ecr_component():
    """PROJ-6: IR/Out/Sus cut the stats-based sources only; the market already prices the injury."""
    eng = ScoringEngine(PPR)
    pr = Projector(eng, rank_curves=default_rank_curves())
    players = _players_with_ecr(20)
    players["h"] = Player("h", "Healthy", "WR", team="KC", ecr=5.0, ecr_sd=2.0)
    players["i"] = Player("i", "On IR", "WR", team="KC", ecr=5.0, ecr_sd=2.0, injury_status="IR", status="Injured Reserve")
    players["f"] = Player("f", "ECR-only healthy", "WR", team="KC", ecr=5.0, ecr_sd=2.0)
    players["e"] = Player("e", "ECR-only IR", "WR", team="KC", ecr=5.0, ecr_sd=2.0, injury_status="IR", status="Injured Reserve")
    players["s"] = Player("s", "Suspended", "WR", team="KC", ecr=5.0, ecr_sd=2.0, injury_status="Sus")
    ml = _ml_pred({"h": WR_RATES, "i": WR_RATES, "s": WR_RATES}, "WR")
    out = pr.project(players, ml_pred=ml)
    h, i, s = out["h"], out["i"], out["s"]
    assert i.components["ecr"] == pytest.approx(h.components["ecr"])
    assert i.components["ml"] == pytest.approx(h.components["ml"] * 10.0 / 16.0)
    assert i.games == pytest.approx(10.0) and "injury:IR" in i.flags
    assert i.points == pytest.approx(i.weights["ml"] * i.components["ml"] + i.weights["ecr"] * i.components["ecr"])
    assert h.points > i.points > h.points * 10.0 / 16.0
    assert s.components["ml"] == pytest.approx(h.components["ml"] * 12.0 / 16.0) and s.games == pytest.approx(12.0)
    # ECR-only: points untouched (no double count), games still reflect the stint
    assert out["e"].points == pytest.approx(out["f"].points) and out["e"].games == pytest.approx(10.0)
    # ML-only IR without ECR keeps the full deduction (nothing else prices it)
    lone = {"a": Player("a", "A", "WR", team="KC"), "ir": Player("ir", "IR", "WR", team="KC", injury_status="IR", status="Injured Reserve")}
    o = Projector(eng).project(lone, ml_pred=_ml_pred({"a": WR_RATES, "ir": WR_RATES}, "WR"))
    assert o["ir"].games == pytest.approx(3.0) and o["ir"].points == pytest.approx(o["a"].points * 3.0 / 16.0)
