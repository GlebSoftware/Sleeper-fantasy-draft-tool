"""Offline tables that make in-season advice possible.

Three small tables are built here and shipped inside ``web_bundle/`` (see
``scripts/build_bundle.py``), because the deployed runtime has neither pandas nor the raw nflverse
data:

* **weekly sigma** - how much a player's score swings *from week to week*. This is not
  ``Projection.std``, and the difference matters: ``ppg_std_ppr`` (``projections/model.py``) is the
  error of the *season* ppg forecast, fitted as ``a + b * ppg`` on out-of-fold residuals. Week-to-week
  variance is far larger, and a win probability computed on the smaller number would call every
  matchup far more certain than it is.
* **schedule** - who each team plays each week, so a weekly projection knows the opponent.
* **defence versus position** - how many points each defence gives up to each position, relative to
  the league. Shrunk toward 1.0 by sample size, because a full season is only 17 games.

pandas lives here. Nothing in the lean runtime (``lean.py``, ``web/``, ``espn/``, ``sleeper/``) may
import this module - it reads the JSON these functions produce instead.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..config import SKILL_POSITIONS
from ..projections.features import PPR_SCORING
from ..scoring.engine import ScoringEngine
from .canonical import to_sleeper_team

log = logging.getLogger(__name__)

__all__ = ["weekly_ppr_points", "weekly_sigma_table", "schedule_table", "dvp_table"]

_PPR = ScoringEngine(PPR_SCORING)

#: A player-season needs this many games before its own spread is worth anything.
MIN_GAMES = 6
#: Weight of the position curve when shrinking one player's spread toward it, in games.
SIGMA_PRIOR_GAMES = 8.0
#: Weight of "no edge" (1.0) when shrinking a defence-versus-position multiplier, in games.
DVP_PRIOR_GAMES = 12.0
#: A multiplier never moves a projection by more than this, whatever the sample says.
DVP_CLIP = (0.80, 1.20)


def weekly_ppr_points(canonical: pd.DataFrame) -> pd.Series:
    """PPR points for every player-week row of the canonical frame."""
    return _PPR.score_frame(canonical, position_col="position")


def weekly_sigma_table(canonical: pd.DataFrame, min_games: int = MIN_GAMES) -> dict[str, Any]:
    """``{"players": {id: {"sd", "ppg", "weeks"}}, "curve": {position: [a, b]}}``.

    The spread is pooled *within* season and then across seasons: a player who changed role between
    years should not have that change counted as weekly volatility. Players thin on games fall back
    to their position's ``sd = a + b * ppg`` curve, which is also what a rookie gets.
    """
    df = canonical.copy()
    df["ppr"] = weekly_ppr_points(df)
    df = df[df["position"].isin(SKILL_POSITIONS)]
    g = df.groupby(["player_id", "season", "position"])["ppr"]
    per_season = g.agg(n="size", mean="mean", sd=lambda s: float(s.std(ddof=1)) if len(s) > 1 else np.nan)
    per_season = per_season.reset_index()
    per_season = per_season[(per_season["n"] >= min_games) & per_season["sd"].notna()]

    # position curve: sd = a + b * ppg, least squares over player-seasons
    curve: dict[str, list[float]] = {}
    for pos, sub in per_season.groupby("position"):
        ppg = sub["mean"].to_numpy(dtype=float)
        sd = sub["sd"].to_numpy(dtype=float)
        if len(sub) < 30:
            curve[str(pos)] = [float(np.mean(sd)) if len(sd) else 5.0, 0.0]
            continue
        A = np.column_stack([np.ones_like(ppg), ppg])
        coef, *_ = np.linalg.lstsq(A, sd, rcond=None)
        a = float(np.clip(coef[0], 0.5, 12.0))
        b = float(np.clip(coef[1], 0.0, 1.5))
        curve[str(pos)] = [round(a, 4), round(b, 4)]
        log.info("%s weekly sigma: %.2f + %.3f * ppg (n=%d)", pos, a, b, len(sub))

    def _curve_sd(pos: str, ppg: float) -> float:
        a, b = curve.get(pos, [5.0, 0.3])
        return float(max(0.5, a + b * max(0.0, ppg)))

    players: dict[str, dict[str, float]] = {}
    for pid, sub in per_season.groupby("player_id"):
        n = sub["n"].to_numpy(dtype=float)
        sd = sub["sd"].to_numpy(dtype=float)
        pos = str(sub["position"].iloc[-1])
        weeks = float(n.sum())
        ppg = float((sub["mean"].to_numpy(dtype=float) * n).sum() / weeks)
        # pooled within-season variance, then shrink toward the position curve by sample size
        dof = np.maximum(n - 1.0, 0.0)
        pooled = float(np.sqrt((dof * sd ** 2).sum() / max(dof.sum(), 1.0)))
        w = float(dof.sum())
        shrunk = (w * pooled + SIGMA_PRIOR_GAMES * _curve_sd(pos, ppg)) / (w + SIGMA_PRIOR_GAMES)
        players[str(pid)] = {"sd": round(float(shrunk), 3), "ppg": round(ppg, 3), "weeks": int(weeks)}
    return {"players": players, "curve": curve}


def schedule_table(schedule: pd.DataFrame, season: int) -> dict[str, dict[str, Any]]:
    """``{team: {week: {"opp": team, "home": bool}}}`` for one regular season (Sleeper abbrs)."""
    g = schedule[(pd.to_numeric(schedule["season"], errors="coerce") == season) & (schedule["game_type"] == "REG")]
    out: dict[str, dict[str, Any]] = {}
    for r in g.itertuples(index=False):
        week = int(pd.to_numeric(r.week))
        home, away = to_sleeper_team(r.home_team), to_sleeper_team(r.away_team)
        if not home or not away:
            continue
        out.setdefault(home, {})[str(week)] = {"opp": away, "home": True}
        out.setdefault(away, {})[str(week)] = {"opp": home, "home": False}
    return out


def dvp_table(canonical: pd.DataFrame, season: int) -> dict[str, dict[str, float]]:
    """``{defence: {position: multiplier}}`` from one season's actual points allowed.

    The multiplier is that defence's points allowed to the position per game over the league mean,
    shrunk toward 1.0 by how many games it is based on and clipped: a defence is worth a nudge, never
    a bench-or-start decision on its own.
    """
    df = canonical[(pd.to_numeric(canonical["season"], errors="coerce") == season)
                   & canonical["position"].isin(["QB", "RB", "WR", "TE"])].copy()
    if df.empty or "opponent" not in df.columns:
        return {}
    df["ppr"] = weekly_ppr_points(df)
    df = df[df["opponent"].notna()]
    # points a defence allowed to a position in one game, then averaged over its games
    per_game = df.groupby(["opponent", "position", "week"], as_index=False)["ppr"].sum()
    agg = per_game.groupby(["opponent", "position"], as_index=False).agg(mean=("ppr", "mean"), games=("week", "nunique"))
    league = agg.groupby("position")["mean"].mean().to_dict()
    out: dict[str, dict[str, float]] = {}
    for r in agg.itertuples(index=False):
        base = float(league.get(r.position, 0.0))
        if base <= 0:
            continue
        raw = float(r.mean) / base
        n = float(r.games)
        mult = (n * raw + DVP_PRIOR_GAMES * 1.0) / (n + DVP_PRIOR_GAMES)
        out.setdefault(str(r.opponent), {})[str(r.position)] = round(float(np.clip(mult, *DVP_CLIP)), 4)
    return out


def build_inseason_tables(canonical: pd.DataFrame, schedule: pd.DataFrame, season: int,
                          dvp_season: int | None = None) -> dict[str, Any]:
    """Everything the runtime needs, in one JSON-serialisable dict.

    ``dvp_season`` defaults to the last season the canonical frame actually covers - in September the
    current season has no games yet, and a table built from an empty slice would silently be all 1.0.
    """
    seasons = sorted(set(int(s) for s in pd.to_numeric(canonical["season"], errors="coerce").dropna()))
    if dvp_season is None:
        dvp_season = max([s for s in seasons if s <= season], default=season)
    sigma = weekly_sigma_table(canonical)
    return {
        "season": int(season),
        "dvp_season": int(dvp_season),
        "sigma": sigma,
        "schedule": schedule_table(schedule, season),
        "dvp": dvp_table(canonical, dvp_season),
    }
