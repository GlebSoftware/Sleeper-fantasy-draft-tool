"""Feature engineering for the projection model (DESIGN.md §3.3).

Pipeline::

    canonical per-game frame ──► season_aggregates ──► one row per (player, season)
                                                          │
                     roster S / Sleeper players ──► build_training_table / build_inference_table
                                                          │
                                       features from seasons < S  +  targets y_* from season S

Leakage rule: every feature of a row for season ``S`` is computed from seasons
strictly before ``S`` (``prev_*`` = S-1, ``prev2_*`` = S-2, ``career_*`` = all
seasons < S) or from static attributes known before the season starts (age,
draft capital, current team). Targets are per-game rates in season ``S``.
"""
from __future__ import annotations

import logging
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from ..config import SKILL_POSITIONS
from ..data.crosswalk import Crosswalk
from ..models import Player
from ..scoring.engine import DEFAULT_SCORING, ScoringEngine

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

#: Per-game rate targets per position (Sleeper stat keys).
TARGETS: dict[str, list[str]] = {
    "QB": ["pass_att", "pass_cmp", "pass_yd", "pass_td", "pass_int", "rush_att", "rush_yd", "rush_td", "fum_lost",
           "bonus_pass_yd_300", "bonus_rush_yd_100"],
    "RB": ["rush_att", "rush_yd", "rush_td", "rec_tgt", "rec", "rec_yd", "rec_td", "fum_lost", "bonus_rush_yd_100", "bonus_rec_yd_100"],
    "WR": ["rush_att", "rush_yd", "rush_td", "rec_tgt", "rec", "rec_yd", "rec_td", "fum_lost", "bonus_rush_yd_100", "bonus_rec_yd_100"],
    "TE": ["rush_att", "rush_yd", "rush_td", "rec_tgt", "rec", "rec_yd", "rec_td", "fum_lost", "bonus_rush_yd_100", "bonus_rec_yd_100"],
    "K": ["fgm_0_19", "fgm_20_29", "fgm_30_39", "fgm_40_49", "fgm_50p", "fgmiss", "xpm", "xpmiss"],
    "DEF": ["sack", "int", "ff", "fum_rec", "safe", "def_td", "blk_kick", "pts_allow", "yds_allow"],
}


def _ordered_union(lists: Iterable[list[str]]) -> list[str]:
    out: list[str] = []
    for lst in lists:
        for k in lst:
            if k not in out:
                out.append(k)
    return out


#: Every target key used by any position (order stable).
ALL_TARGET_KEYS: list[str] = _ordered_union(TARGETS.values())

#: Scoring presets used for position-agnostic reference points.
PPR_SCORING: dict[str, float] = {**DEFAULT_SCORING, "rec": 1.0}
STD_SCORING: dict[str, float] = {**DEFAULT_SCORING, "rec": 0.0}

USAGE_KEYS: list[str] = ["f_target_share", "f_air_yards_share", "f_wopr"]
EFFICIENCY_KEYS: list[str] = ["ypc", "ypt", "td_per_touch", "catch_rate", "ypa", "td_rate"]
#: Extra per-season aggregates carried as features.
AGG_EXTRA_KEYS: list[str] = ["ppg_ppr", "ppg_std", "ppg_ppr_sd", "pos_rank", "rec_tgt_total", "rush_att_total",
                             "snap_pct", "weeks_out"]

PREV_KEYS: list[str] = ["games"] + ALL_TARGET_KEYS + AGG_EXTRA_KEYS + USAGE_KEYS + EFFICIENCY_KEYS
PREV2_KEYS: list[str] = ["games", "ppg_ppr", "ppg_std", "pos_rank", "snap_pct"] + ALL_TARGET_KEYS
CAREER_KEYS: list[str] = ["ppg_ppr", "ppg_std"] + ALL_TARGET_KEYS
TEAM_CTX_KEYS: list[str] = ["team_pass_att_pg", "team_rush_att_pg", "team_targets_pg", "team_rec_yd_pg", "team_rush_yd_pg"]
STATIC_KEYS: list[str] = ["age", "years_exp", "draft_ovr", "undrafted", "rookie", "team_changed", "no_history"]

#: Features fed to the model (union over positions; NaN where not applicable).
FEATURE_COLUMNS: list[str] = (
    STATIC_KEYS
    + [f"prev_{k}" for k in PREV_KEYS]
    + [f"prev2_{k}" for k in PREV2_KEYS]
    + ["career_games", "career_seasons"] + [f"career_{k}" for k in CAREER_KEYS]
    + [f"prev_{k}" for k in TEAM_CTX_KEYS]
    + [f"cur_{k}" for k in TEAM_CTX_KEYS]
)
#: Features the rookie model uses.
ROOKIE_FEATURE_COLUMNS: list[str] = ["draft_ovr", "undrafted", "age"]

#: Target column names in the training table.
TARGET_COLUMNS: list[str] = [f"y_{k}" for k in ALL_TARGET_KEYS] + ["y_games", "y_ppg_ppr"]

_POSITION_ALIASES = {"FB": "RB", "HB": "RB", "PK": "K", "DST": "DEF"}
_DEF_TEAMS = ("ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB", "HOU", "IND",
              "JAX", "KC", "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA",
              "SF", "TB", "TEN", "WAS")


# ---------------------------------------------------------------------------
# Season aggregates
# ---------------------------------------------------------------------------

def _mode(s: pd.Series):
    m = s.mode(dropna=True)
    return m.iat[0] if len(m) else (s.iloc[0] if len(s) else None)


def _safe_ratio(num: pd.Series, den: pd.Series, min_den: float) -> pd.Series:
    out = num / den.where(den >= min_den)
    return out.replace([np.inf, -np.inf], np.nan)


def _snap_pct(snaps: pd.DataFrame, pfr_to_gsis: Mapping[str, str] | None) -> pd.DataFrame:
    """Mean offensive snap share per (gsis id, season) from a snap-counts frame."""
    if snaps is None or len(snaps) == 0 or pfr_to_gsis is None:
        return pd.DataFrame(columns=["player_id", "season", "snap_pct"])
    s = snaps.copy()
    s["player_id"] = s["pfr_id"].map(lambda p: pfr_to_gsis.get(str(p)) if p is not None else None)
    s = s[s["player_id"].notna()]
    s["offense_pct"] = pd.to_numeric(s["offense_pct"], errors="coerce")
    s = s[s["offense_pct"].notna()]
    if s.empty:
        return pd.DataFrame(columns=["player_id", "season", "snap_pct"])
    s["season"] = pd.to_numeric(s["season"]).astype(int)
    g = s.groupby(["player_id", "season"])["offense_pct"].mean().rename("snap_pct").reset_index()
    return g


def _weeks_out(injuries: pd.DataFrame) -> pd.DataFrame:
    """Number of weeks a player was listed 'Out' per (gsis id, season)."""
    if injuries is None or len(injuries) == 0 or "report_status" not in injuries.columns:
        return pd.DataFrame(columns=["player_id", "season", "weeks_out"])
    i = injuries[injuries["report_status"].astype(str).str.lower() == "out"]
    if i.empty:
        return pd.DataFrame(columns=["player_id", "season", "weeks_out"])
    i = i.assign(season=pd.to_numeric(i["season"]).astype(int), player_id=i["gsis_id"].astype(str))
    g = i.groupby(["player_id", "season"])["week"].nunique().rename("weeks_out").reset_index()
    return g


def season_aggregates(canonical: pd.DataFrame, snaps: pd.DataFrame | None = None,
                      injuries: pd.DataFrame | None = None,
                      pfr_to_gsis: Mapping[str, str] | None = None) -> pd.DataFrame:
    """One row per (player_id, season) with per-game rates, usage and efficiency.

    Columns: ``player_id, season, position, team, games``, per-game means of every
    key in :data:`ALL_TARGET_KEYS`, ``ppg_ppr`` / ``ppg_std`` (engine-scored PPR and
    standard points per game, so K/DEF are on the same scale as offense),
    ``ppg_nfl_ppr`` (nflverse reference), ``ppg_ppr_sd`` (game-to-game std),
    ``pos_rank`` (rank of ``ppg_ppr`` within season+position, players with >= 4
    games), ``rec_tgt_total``, ``rush_att_total``, usage means, ``snap_pct``,
    ``weeks_out`` and efficiency (``ypc, ypt, td_per_touch, catch_rate, ypa, td_rate``).
    """
    if canonical is None or len(canonical) == 0:
        return pd.DataFrame(columns=["player_id", "season", "position", "team", "games"] + ALL_TARGET_KEYS)
    df = canonical.copy()
    df["player_id"] = df["player_id"].astype(str)
    df["season"] = pd.to_numeric(df["season"]).astype(int)
    for k in ALL_TARGET_KEYS + USAGE_KEYS + ["f_nfl_fantasy_points_ppr"]:
        if k not in df.columns:
            df[k] = np.nan
    if "pass_inc" not in df.columns:
        df["pass_inc"] = df.get("pass_att", 0) - df.get("pass_cmp", 0)
    ppr, std = ScoringEngine(PPR_SCORING), ScoringEngine(STD_SCORING)
    df["_pts_ppr"] = ppr.score_frame(df)
    df["_pts_std"] = std.score_frame(df)

    keys = ["player_id", "season"]
    agg_spec: dict[str, tuple[str, str]] = {
        "games": ("week", "size"),
        "position": ("position", _mode),
        "team": ("team", _mode),
        "ppg_ppr": ("_pts_ppr", "mean"),
        "ppg_std": ("_pts_std", "mean"),
        "ppg_ppr_sd": ("_pts_ppr", "std"),
        "ppg_nfl_ppr": ("f_nfl_fantasy_points_ppr", "mean"),
    }
    for k in ALL_TARGET_KEYS:
        agg_spec[k] = (k, "mean")
    for k in USAGE_KEYS:
        agg_spec[k] = (k, "mean")
    agg_spec["rec_tgt_total"] = ("rec_tgt", "sum")
    agg_spec["rush_att_total"] = ("rush_att", "sum")
    agg_spec["_rush_yd_total"] = ("rush_yd", "sum")
    agg_spec["_rec_total"] = ("rec", "sum")
    agg_spec["_rec_yd_total"] = ("rec_yd", "sum")
    agg_spec["_rush_td_total"] = ("rush_td", "sum")
    agg_spec["_rec_td_total"] = ("rec_td", "sum")
    agg_spec["_pass_att_total"] = ("pass_att", "sum")
    agg_spec["_pass_yd_total"] = ("pass_yd", "sum")
    agg_spec["_pass_td_total"] = ("pass_td", "sum")
    g = df.groupby(keys).agg(**agg_spec).reset_index()
    g["position"] = g["position"].map(lambda p: _POSITION_ALIASES.get(p, p))

    # efficiency from season totals (NaN when the denominator is too small)
    g["ypc"] = _safe_ratio(g["_rush_yd_total"], g["rush_att_total"], 10)
    g["ypt"] = _safe_ratio(g["_rec_yd_total"], g["rec_tgt_total"], 10)
    g["td_per_touch"] = _safe_ratio(g["_rush_td_total"] + g["_rec_td_total"], g["rush_att_total"] + g["_rec_total"], 10)
    g["catch_rate"] = _safe_ratio(g["_rec_total"], g["rec_tgt_total"], 10)
    g["ypa"] = _safe_ratio(g["_pass_yd_total"], g["_pass_att_total"], 30)
    g["td_rate"] = _safe_ratio(g["_pass_td_total"], g["_pass_att_total"], 30)
    g = g.drop(columns=[c for c in g.columns if c.startswith("_")])

    # rank within season+position (players with >= 4 games)
    ok = g["games"] >= 4
    g["pos_rank"] = np.nan
    g.loc[ok, "pos_rank"] = g[ok].groupby(["season", "position"])["ppg_ppr"].rank(ascending=False, method="min")

    snap = _snap_pct(snaps, pfr_to_gsis)
    g = g.merge(snap, on=["player_id", "season"], how="left") if len(snap) else g.assign(snap_pct=np.nan)
    wo = _weeks_out(injuries)
    g = g.merge(wo, on=["player_id", "season"], how="left") if len(wo) else g.assign(weeks_out=np.nan)
    g["weeks_out"] = g["weeks_out"].fillna(0.0)
    return g


# ---------------------------------------------------------------------------
# Feature assembly (shared by training and inference)
# ---------------------------------------------------------------------------

def _age_at(birth: pd.Series, season: int) -> pd.Series:
    bd = pd.to_datetime(birth, errors="coerce")
    ref = pd.Timestamp(f"{season}-09-01")
    return (ref - bd).dt.days / 365.25


def _career(agg: pd.DataFrame, season: int) -> pd.DataFrame:
    """Games-weighted career per-game rates over seasons < ``season``."""
    h = agg[agg["season"] < season]
    cols = ["career_games", "career_seasons"] + [f"career_{k}" for k in CAREER_KEYS]
    if h.empty:
        return pd.DataFrame(columns=cols)
    w = h["games"].astype(float)
    data = {"career_games": w, "career_seasons": pd.Series(1.0, index=h.index)}
    for k in CAREER_KEYS:
        data[f"_w_{k}"] = h[k].astype(float) * w
        data[f"_n_{k}"] = w.where(h[k].notna(), 0.0)
    tmp = pd.DataFrame(data, index=h.index)
    tmp["player_id"] = h["player_id"].to_numpy()
    s = tmp.groupby("player_id").sum()
    out = pd.DataFrame(index=s.index)
    out["career_games"] = s["career_games"]
    out["career_seasons"] = s["career_seasons"]
    for k in CAREER_KEYS:
        out[f"career_{k}"] = s[f"_w_{k}"] / s[f"_n_{k}"].where(s[f"_n_{k}"] > 0)
    return out


def _team_ctx_lookup(team_ctx: pd.DataFrame | None, season: int) -> pd.DataFrame:
    if team_ctx is None or len(team_ctx) == 0:
        return pd.DataFrame(columns=TEAM_CTX_KEYS)
    t = team_ctx[pd.to_numeric(team_ctx["season"]) == season]
    t = t.drop_duplicates("team").set_index("team")
    return t.reindex(columns=TEAM_CTX_KEYS)


def _assemble(universe: pd.DataFrame, agg: pd.DataFrame, team_ctx: pd.DataFrame | None, season: int,
              with_targets: bool) -> pd.DataFrame:
    """Build feature (and optionally target) columns for ``universe`` rows for ``season``.

    ``universe`` columns: ``key`` (gsis id / team abbr), ``position``, ``team``,
    ``age``, ``years_exp``, ``draft_ovr``, ``entry_year``; its index is preserved.
    Only seasons < ``season`` of ``agg`` feed features.
    """
    u = universe.copy()
    key = u["key"].astype(str)
    hist = agg[agg["season"] < season]
    prev = hist[hist["season"] == season - 1].drop_duplicates("player_id").set_index("player_id")
    prev2 = hist[hist["season"] == season - 2].drop_duplicates("player_id").set_index("player_id")
    career = _career(agg, season)

    cols: dict[str, pd.Series] = {}
    cols["player_id"] = key
    cols["season"] = pd.Series(season, index=u.index)
    cols["position"] = u["position"]
    cols["team"] = u["team"]

    # static
    cols["age"] = pd.to_numeric(u["age"], errors="coerce")
    cols["years_exp"] = pd.to_numeric(u["years_exp"], errors="coerce")
    draft_ovr = pd.to_numeric(u["draft_ovr"], errors="coerce")
    cols["undrafted"] = draft_ovr.isna().astype(float)
    cols["draft_ovr"] = draft_ovr.fillna(300.0)

    # previous season
    p = prev.reindex(key.to_numpy())
    p.index = u.index
    for k in PREV_KEYS:
        cols[f"prev_{k}"] = pd.to_numeric(p[k], errors="coerce") if k in p.columns else pd.Series(np.nan, index=u.index)
    prev_team = p["team"] if "team" in p.columns else pd.Series(None, index=u.index, dtype=object)
    # two seasons ago
    p2 = prev2.reindex(key.to_numpy())
    p2.index = u.index
    for k in PREV2_KEYS:
        cols[f"prev2_{k}"] = pd.to_numeric(p2[k], errors="coerce") if k in p2.columns else pd.Series(np.nan, index=u.index)
    # career
    c = career.reindex(key.to_numpy())
    c.index = u.index
    for k in ["career_games", "career_seasons"] + [f"career_{k}" for k in CAREER_KEYS]:
        cols[k] = pd.to_numeric(c[k], errors="coerce") if k in c.columns else pd.Series(np.nan, index=u.index)

    has_history = cols["career_games"].fillna(0) > 0
    cols["no_history"] = (~has_history).astype(float)
    yrs = cols["years_exp"]
    entry = pd.to_numeric(u["entry_year"], errors="coerce")
    is_first_year = (yrs.fillna(99) <= 0) | (entry == season)
    cols["rookie"] = ((~has_history) & is_first_year).astype(float)
    # DEF never rookies
    cols["rookie"] = cols["rookie"].where(cols["position"] != "DEF", 0.0)
    changed = (prev_team.notna()) & (u["team"].notna()) & (prev_team.astype(object) != u["team"].astype(object))
    cols["team_changed"] = changed.astype(float).where(prev_team.notna(), np.nan)

    # team context from S-1: the player's previous team and his current team
    ctx = _team_ctx_lookup(team_ctx, season - 1)
    pt = ctx.reindex(prev_team.astype(object).to_numpy())
    pt.index = u.index
    ct = ctx.reindex(u["team"].astype(object).to_numpy())
    ct.index = u.index
    for k in TEAM_CTX_KEYS:
        cols[f"prev_{k}"] = pd.to_numeric(pt[k], errors="coerce") if k in pt.columns else pd.Series(np.nan, index=u.index)
        cols[f"cur_{k}"] = pd.to_numeric(ct[k], errors="coerce") if k in ct.columns else pd.Series(np.nan, index=u.index)

    if with_targets:
        cur = agg[agg["season"] == season].drop_duplicates("player_id").set_index("player_id")
        y = cur.reindex(key.to_numpy())
        y.index = u.index
        for k in ALL_TARGET_KEYS:
            cols[f"y_{k}"] = pd.to_numeric(y[k], errors="coerce") if k in y.columns else pd.Series(np.nan, index=u.index)
        cols["y_games"] = pd.to_numeric(y["games"], errors="coerce").fillna(0.0) if "games" in y.columns else pd.Series(0.0, index=u.index)
        cols["y_ppg_ppr"] = pd.to_numeric(y["ppg_ppr"], errors="coerce") if "ppg_ppr" in y.columns else pd.Series(np.nan, index=u.index)
        # a player with no stat rows has no per-game rates
        for k in ALL_TARGET_KEYS:
            cols[f"y_{k}"] = cols[f"y_{k}"].where(cols["y_games"] > 0)

    out = pd.DataFrame(cols, index=u.index)
    out = out[out["position"].isin(SKILL_POSITIONS)]
    return out


# ---------------------------------------------------------------------------
# Training / inference tables
# ---------------------------------------------------------------------------

def _universe_from_roster(roster: pd.DataFrame, season: int, def_teams: Iterable[str]) -> pd.DataFrame:
    r = roster.copy()
    r = r[r["gsis_id"].notna()]
    r["position"] = r["position"].map(lambda p: _POSITION_ALIASES.get(p, p))
    r = r[r["position"].isin(("QB", "RB", "WR", "TE", "K"))]
    if "week" in r.columns:
        r = r.sort_values("week")
    r = r.drop_duplicates("gsis_id", keep="last")
    entry = pd.to_numeric(r.get("entry_year", pd.Series(np.nan, index=r.index)), errors="coerce")
    yrs = pd.to_numeric(r.get("years_exp", pd.Series(np.nan, index=r.index)), errors="coerce")
    yrs = yrs.fillna((season - entry).where(entry.notna()))
    age = _age_at(r["birth_date"], season) if "birth_date" in r.columns else pd.Series(np.nan, index=r.index)
    age = age.fillna(22.0 + (season - entry).where(entry.notna()))
    u = pd.DataFrame({
        "key": r["gsis_id"].astype(str),
        "position": r["position"].astype(str),
        "team": r["team"].astype(object).where(r["team"].notna(), None),
        "age": age,
        "years_exp": yrs,
        "draft_ovr": pd.to_numeric(r.get("draft_number", pd.Series(np.nan, index=r.index)), errors="coerce"),
        "entry_year": entry,
    })
    d = pd.DataFrame({
        "key": list(def_teams), "position": "DEF", "team": list(def_teams), "age": np.nan, "years_exp": np.nan,
        "draft_ovr": np.nan, "entry_year": np.nan,
    })
    return pd.concat([u, d], ignore_index=True)


def build_training_table(agg: pd.DataFrame, team_ctx: pd.DataFrame | None,
                         rosters_by_season: Mapping[int, pd.DataFrame], seasons: Iterable[int]) -> pd.DataFrame:
    """Rows keyed by (player_id, season S) for every S in ``seasons`` with a roster.

    Features come from seasons < S (see module docstring); targets ``y_<key>``
    are S per-game rates (NaN when 0 games), ``y_games`` and ``y_ppg_ppr``.
    Universe: players on the S roster at skill positions (any status) + 32 DEF.
    """
    def_teams = sorted(set(agg.loc[agg["position"] == "DEF", "player_id"].astype(str)) or set(_DEF_TEAMS))
    frames = []
    for s in sorted(int(x) for x in seasons):
        ro = rosters_by_season.get(s)
        if ro is None or len(ro) == 0:
            log.warning("no roster for %s; skipping training rows", s)
            continue
        u = _universe_from_roster(ro, s, def_teams)
        t = _assemble(u, agg, team_ctx, s, with_targets=True)
        frames.append(t)
        log.info("training rows for %s: %d", s, len(t))
    if not frames:
        return pd.DataFrame(columns=["player_id", "season", "position", "team"] + FEATURE_COLUMNS + TARGET_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    return out


def _player_attr(pl: Player, attrs: Mapping, *names: str):
    for n in names:
        v = attrs.get(n)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            return v
    return None


def build_inference_table(agg: pd.DataFrame, team_ctx: pd.DataFrame | None, players: Mapping[str, Player],
                          cw: Crosswalk, season: int) -> pd.DataFrame:
    """Feature table for the draft-time universe; index = Sleeper player_id.

    History is looked up via ``Player.gsis_id`` / ``cw.gsis_for``; current team
    from ``Player.team``; draft capital / age from the player or the crosswalk.
    """
    rows = []
    for pid, pl in players.items():
        if pl.position not in SKILL_POSITIONS:
            continue
        a = cw.attrs.get(pid, {}) if cw is not None else {}
        if pl.position == "DEF":
            key = pl.team or pid
        else:
            key = pl.gsis_id or (cw.gsis_for(pid) if cw is not None else None) or f"sleeper:{pid}"
        draft_ovr = pl.draft_pick_overall
        if draft_ovr is None:
            draft_ovr = _player_attr(pl, a, "draft_ovr", "draft_number")
        age = pl.age
        if age is None:
            age = _player_attr(pl, a, "age")
        if age is None:
            bd = _player_attr(pl, a, "birthdate", "birth_date") or pl.metadata.get("birth_date")
            if bd:
                age = float(_age_at(pd.Series([bd]), season).iloc[0])
        entry = pl.draft_year or _player_attr(pl, a, "draft_year", "entry_year")
        yrs = pl.years_exp
        if yrs is None:
            yrs = _player_attr(pl, a, "years_exp")
        if yrs is None and entry is not None:
            yrs = max(0, season - int(entry))
        if age is None and entry is not None:
            age = 22.0 + max(0, season - int(entry))
        rows.append({"pid": pid, "key": str(key), "position": pl.position, "team": pl.team, "age": age,
                     "years_exp": yrs, "draft_ovr": draft_ovr, "entry_year": entry})
    if not rows:
        return pd.DataFrame(columns=["player_id", "season", "position", "team"] + FEATURE_COLUMNS)
    u = pd.DataFrame(rows).set_index("pid")
    u.index.name = "player_id"
    t = _assemble(u, agg, team_ctx, season, with_targets=False)
    t = t.rename(columns={"player_id": "gsis_id"})
    return t
