"""Canonical per-game stat frame: nflverse columns -> Sleeper scoring keys.

The output frame has one row per (player, season, week) regular-season game with
columns:

* identity: ``player_id`` (nflverse gsis id, or team abbreviation for DEF),
  ``player_name``, ``position`` (QB/RB/WR/TE/K/DEF), ``team`` (Sleeper style),
  ``opponent``, ``season``, ``week``
* every Sleeper stat key in :data:`draftadvisor.scoring.SLEEPER_STAT_KEYS` that we can
  derive (missing ones are absent, which the scoring engine treats as zero)
* extra usage/efficiency features prefixed ``f_`` for the ML model
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..scoring.engine import ScoringEngine

#: nflverse team abbreviations -> Sleeper abbreviations.
TEAM_TO_SLEEPER = {"LA": "LAR", "OAK": "LV", "SD": "LAC", "STL": "LAR", "JAC": "JAX", "WSH": "WAS", "FA": None}
SLEEPER_TO_NFLVERSE = {"LAR": "LA"}

FANTASY_POSITIONS = {"QB", "RB", "WR", "TE", "K"}
#: nflverse position aliases folded into fantasy positions.
POSITION_ALIASES = {"FB": "RB", "HB": "RB", "PK": "K"}

#: Feature columns copied from nflverse when present (prefixed with f_).
FEATURE_COLUMNS = (
    "target_share", "air_yards_share", "wopr", "racr", "receiving_epa", "rushing_epa", "passing_epa",
    "receiving_air_yards", "passing_air_yards", "passing_cpoe", "receiving_yards_after_catch",
    "passing_yards_after_catch", "pacr",
)


def to_sleeper_team(team: str | float | None) -> str | None:
    if team is None or (isinstance(team, float) and np.isnan(team)):
        return None
    t = str(team).upper().strip()
    if t in ("", "NAN", "NONE"):
        return None
    return TEAM_TO_SLEEPER.get(t, t)


def _num(df: pd.DataFrame, *names: str) -> pd.Series:
    """First present column among ``names`` as float (missing -> zeros)."""
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n], errors="coerce").fillna(0.0).astype(float)
    return pd.Series(0.0, index=df.index, dtype=float)


def _fg_over_30(list_col: pd.Series) -> pd.Series:
    def one(v) -> float:
        if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() in ("", "nan"):
            return 0.0
        tot = 0.0
        for part in str(v).split(";"):
            try:
                d = float(part)
            except ValueError:
                continue
            if d > 30:
                tot += d - 30
        return tot
    return list_col.map(one).astype(float)


def player_weekly_to_canonical(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert an nflverse ``stats_player_week_YYYY`` frame to the canonical schema."""
    df = raw[raw.get("season_type", "REG") == "REG"].copy() if "season_type" in raw.columns else raw.copy()
    pos = df["position"].map(lambda p: POSITION_ALIASES.get(p, p))
    df = df[pos.isin(FANTASY_POSITIONS)].copy()
    pos = pos[df.index]
    out = pd.DataFrame(index=df.index)
    out["player_id"] = df["player_id"].astype(str)
    out["player_name"] = df.get("player_display_name", df.get("player_name", "")).astype(str)
    out["position"] = pos.astype(str)
    team_col = "team" if "team" in df.columns else "recent_team"
    out["team"] = df[team_col].map(to_sleeper_team)
    out["opponent"] = df["opponent_team"].map(to_sleeper_team) if "opponent_team" in df.columns else None
    out["season"] = pd.to_numeric(df["season"]).astype(int)
    out["week"] = pd.to_numeric(df["week"]).astype(int)

    # passing
    out["pass_att"] = _num(df, "attempts")
    out["pass_cmp"] = _num(df, "completions")
    out["pass_inc"] = out["pass_att"] - out["pass_cmp"]
    out["pass_yd"] = _num(df, "passing_yards")
    out["pass_td"] = _num(df, "passing_tds")
    out["pass_int"] = _num(df, "passing_interceptions", "interceptions")
    out["pass_2pt"] = _num(df, "passing_2pt_conversions")
    out["pass_sack"] = _num(df, "sacks_suffered", "sacks")
    out["pass_fd"] = _num(df, "passing_first_downs")
    # rushing
    out["rush_att"] = _num(df, "carries")
    out["rush_yd"] = _num(df, "rushing_yards")
    out["rush_td"] = _num(df, "rushing_tds")
    out["rush_2pt"] = _num(df, "rushing_2pt_conversions")
    out["rush_fd"] = _num(df, "rushing_first_downs")
    # receiving
    out["rec"] = _num(df, "receptions")
    out["rec_tgt"] = _num(df, "targets")
    out["rec_yd"] = _num(df, "receiving_yards")
    out["rec_td"] = _num(df, "receiving_tds")
    out["rec_2pt"] = _num(df, "receiving_2pt_conversions")
    out["rec_fd"] = _num(df, "receiving_first_downs")
    # misc
    out["fum"] = _num(df, "sack_fumbles") + _num(df, "rushing_fumbles") + _num(df, "receiving_fumbles")
    out["fum_lost"] = _num(df, "sack_fumbles_lost") + _num(df, "rushing_fumbles_lost") + _num(df, "receiving_fumbles_lost")
    out["fum_rec_td"] = _num(df, "fumble_recovery_tds")
    out["st_td"] = _num(df, "special_teams_tds")
    out["kr_yd"] = _num(df, "kickoff_return_yards")
    out["pr_yd"] = _num(df, "punt_return_yards")
    # kicking
    out["fgm"] = _num(df, "fg_made")
    out["fga"] = _num(df, "fg_att")
    out["fgmiss"] = _num(df, "fg_missed") + _num(df, "fg_blocked")
    for lo, hi in ((0, 19), (20, 29), (30, 39), (40, 49), (50, 59)):
        out[f"fgm_{lo}_{hi}"] = _num(df, f"fg_made_{lo}_{hi}")
        out[f"fgmiss_{lo}_{hi}"] = _num(df, f"fg_missed_{lo}_{hi}")
    out["fgm_60p"] = _num(df, "fg_made_60_")
    out["fgmiss_60p"] = _num(df, "fg_missed_60_")
    out["fgm_yds"] = _num(df, "fg_made_distance")
    out["fgm_yds_over_30"] = _fg_over_30(df["fg_made_list"]) if "fg_made_list" in df.columns else 0.0
    out["xpm"] = _num(df, "pat_made")
    out["xpa"] = _num(df, "pat_att")
    out["xpmiss"] = _num(df, "pat_missed") + _num(df, "pat_blocked")
    # reference points from nflverse (sanity checks / fallback targets)
    out["f_nfl_fantasy_points"] = _num(df, "fantasy_points")
    out["f_nfl_fantasy_points_ppr"] = _num(df, "fantasy_points_ppr")
    for c in FEATURE_COLUMNS:
        if c in df.columns:
            out[f"f_{c}"] = pd.to_numeric(df[c], errors="coerce")
    ScoringEngine.add_derived_keys(out)
    return out.reset_index(drop=True)


def team_weekly_to_canonical(team_raw: pd.DataFrame, games: pd.DataFrame | None) -> pd.DataFrame:
    """Convert nflverse ``stats_team_week_YYYY`` into DEF rows (player_id = Sleeper team abbr).

    ``games`` (nflverse schedule with scores) supplies points allowed. Yards allowed
    come from the opponent's offensive row in the same frame.
    """
    df = team_raw[team_raw.get("season_type", "REG") == "REG"].copy() if "season_type" in team_raw.columns else team_raw.copy()
    df["season"] = pd.to_numeric(df["season"]).astype(int)
    df["week"] = pd.to_numeric(df["week"]).astype(int)
    # opponent offensive yards -> yards allowed
    off = pd.DataFrame({
        "season": df["season"], "week": df["week"], "opp": df["team"],
        "_yds": _num(df, "passing_yards") - _num(df, "sack_yards_lost", "sack_yards") + _num(df, "rushing_yards"),
        "_opp_st_td": _num(df, "special_teams_tds"),
    })
    m = df.merge(off, left_on=["season", "week", "opponent_team"], right_on=["season", "week", "opp"], how="left")
    out = pd.DataFrame(index=m.index)
    out["team"] = m["team"].map(to_sleeper_team)
    out["player_id"] = out["team"]
    out["player_name"] = out["team"].map(lambda t: f"{t} Defense")
    out["position"] = "DEF"
    out["opponent"] = m["opponent_team"].map(to_sleeper_team)
    out["season"] = m["season"]
    out["week"] = m["week"]
    out["sack"] = _num(m, "def_sacks")
    out["int"] = _num(m, "def_interceptions")
    out["ff"] = _num(m, "def_fumbles_forced")
    out["fum_rec"] = _num(m, "fumble_recovery_opp")
    out["safe"] = _num(m, "def_safeties")
    out["def_td"] = _num(m, "def_tds")
    out["def_st_td"] = _num(m, "special_teams_tds")
    out["blk_kick"] = _num(m, "def_punt_blocks") + _num(m, "def_pat_blocks") + _num(m, "def_fg_blocks")
    out["def_pass_def"] = _num(m, "def_pass_defended")
    out["def_2pt"] = _num(m, "def_2pt_made")
    out["yds_allow"] = m["_yds"].fillna(0.0)
    # points allowed from schedule
    pts = pd.Series(np.nan, index=m.index)
    if games is not None and len(games):
        g = games.copy()
        g["season"] = pd.to_numeric(g["season"]).astype(int)
        g["week"] = pd.to_numeric(g["week"]).astype(int)
        home = g[["season", "week", "home_team", "away_score"]].rename(columns={"home_team": "t", "away_score": "pa"})
        away = g[["season", "week", "away_team", "home_score"]].rename(columns={"away_team": "t", "home_score": "pa"})
        pa = pd.concat([home, away], ignore_index=True)
        key = pd.DataFrame({"season": m["season"], "week": m["week"], "t": m["team"]})
        pts = key.merge(pa, on=["season", "week", "t"], how="left")["pa"].to_numpy()
        pts = pd.Series(pts, index=m.index)
    out["pts_allow"] = pd.to_numeric(pts, errors="coerce")
    out = out[out["pts_allow"].notna()].copy() if out["pts_allow"].notna().any() else out
    out["pts_allow"] = out["pts_allow"].fillna(0.0)
    ScoringEngine.add_derived_keys(out)
    return out.reset_index(drop=True)
