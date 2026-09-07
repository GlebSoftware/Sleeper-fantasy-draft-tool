"""Blend ML rates, Sleeper projections and FantasyPros ECR into league-scored projections.

Sources per player (each optional):

* ``ml``      - per-game rates + games from :class:`ProjectionModel.predict` (or
  :func:`project_offline`), turned into season totals with :func:`rates_to_season`.
* ``sleeper`` - Sleeper's season projection stat line, scored with the league engine.
* ``ecr``     - market signal: FantasyPros consensus rank mapped onto points with an
  isotonic (monotone decreasing) regression of the provisional points on ECR within
  each position (this also converts the PPR board to the league's scoring).

The blend weights are renormalised over the sources a player actually has.
"""
from __future__ import annotations

import logging
import math
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from ..config import SLOT_ELIGIBILITY, FANTASY_WEEKS, GAMES_PER_TEAM, SKILL_POSITIONS, Settings
from ..data.crosswalk import Crosswalk
from ..models import LeagueSettings, Player, Projection, ResearchNote
from ..scoring.engine import ScoringEngine
from .features import ALL_TARGET_KEYS, PPR_SCORING, TARGETS, season_aggregates

log = logging.getLogger(__name__)

DEFAULT_WEIGHTS: dict[str, float] = {"sleeper": 0.40, "ml": 0.25, "ecr": 0.35}
#: ML games are shrunk this far toward a healthy-starter season (see Projector).
GAMES_SHRINK = 0.6
GAMES_SHRINK_TARGET = 16.0
#: Cap on the ML season std as a fraction of projected points.
MAX_ML_CV = 0.25
#: Preseason expectation = replacement + MARKET_SHRINK * (realized rank curve - replacement).
MARKET_SHRINK = 0.75

#: Minimum players with an ECR at a position before the isotonic mapping is used.
MIN_ECR_POINTS = 8
#: Default games for a player with only an ECR (no ML / Sleeper games estimate).
DEFAULT_GAMES = {"DEF": float(GAMES_PER_TEAM)}
DEFAULT_GAMES_OTHER = 16.0
#: Std as a fraction of points when no source provides an uncertainty.
DEFAULT_CV = 0.22
MIN_STD_FRACTION = 0.08

PTS_BRACKETS = ["pts_allow_0", "pts_allow_1_6", "pts_allow_7_13", "pts_allow_14_20", "pts_allow_21_27",
                "pts_allow_28_34", "pts_allow_35p"]
YDS_BRACKETS = ["yds_allow_0_100", "yds_allow_100_199", "yds_allow_200_299", "yds_allow_300_349",
                "yds_allow_350_399", "yds_allow_400_449", "yds_allow_450_499", "yds_allow_500_549", "yds_allow_550p"]

# Empirical share of games in each points-allowed bracket for team-seasons whose
# mean points allowed falls in a bin (row = [bin centre, shares...]); fitted on the
# 2019-2025 canonical DEF rows. Refit with :func:`fit_def_bracket_table`.
_PTS_TABLE: list[list[float]] = [
    [14.1, 0.125, 0.062, 0.375, 0.188, 0.125, 0.062, 0.062],
    [16.5, 0.045, 0.075, 0.254, 0.358, 0.134, 0.104, 0.030],
    [18.2, 0.016, 0.060, 0.251, 0.313, 0.207, 0.109, 0.044],
    [20.0, 0.021, 0.061, 0.158, 0.308, 0.255, 0.142, 0.055],
    [21.9, 0.009, 0.034, 0.158, 0.261, 0.282, 0.170, 0.087],
    [24.0, 0.003, 0.037, 0.106, 0.241, 0.270, 0.210, 0.133],
    [25.9, 0.002, 0.014, 0.066, 0.221, 0.293, 0.237, 0.167],
    [27.9, 0.000, 0.004, 0.061, 0.182, 0.243, 0.267, 0.243],
    [30.3, 0.000, 0.005, 0.036, 0.102, 0.259, 0.305, 0.294],
]
_YDS_TABLE: list[list[float]] = [
    [305.0, 0.0, 0.059, 0.471, 0.176, 0.235, 0.000, 0.000, 0.059, 0.000],
    [321.0, 0.0, 0.079, 0.317, 0.250, 0.187, 0.115, 0.048, 0.000, 0.004],
    [341.0, 0.0, 0.011, 0.292, 0.263, 0.210, 0.158, 0.050, 0.011, 0.004],
    [360.0, 0.0, 0.013, 0.183, 0.258, 0.259, 0.173, 0.084, 0.022, 0.007],
    [379.0, 0.0, 0.004, 0.151, 0.200, 0.237, 0.242, 0.112, 0.040, 0.014],
    [409.0, 0.0, 0.001, 0.062, 0.158, 0.229, 0.259, 0.178, 0.086, 0.028],
]


def _interp_table(table: list[list[float]], names: list[str], x: float) -> dict[str, float]:
    arr = np.asarray(table, dtype=float)
    centres = arr[:, 0]
    x = float(np.clip(x, centres[0], centres[-1]))
    out = {n: float(np.interp(x, centres, arr[:, i + 1])) for i, n in enumerate(names)}
    tot = sum(out.values())
    if tot > 0:
        out = {k: v / tot for k, v in out.items()}
    return out


def def_bracket_rates(mean_pts_allow: float) -> dict[str, float]:
    """Per-game probability of each ``pts_allow_*`` bracket for a defense that
    allows ``mean_pts_allow`` points per game on average (empirical table)."""
    if mean_pts_allow is None or not np.isfinite(mean_pts_allow):
        mean_pts_allow = 22.9
    return _interp_table(_PTS_TABLE, PTS_BRACKETS, float(mean_pts_allow))


def yds_bracket_rates(mean_yds_allow: float) -> dict[str, float]:
    """Per-game probability of each ``yds_allow_*`` bracket for a mean yards allowed."""
    if mean_yds_allow is None or not np.isfinite(mean_yds_allow):
        mean_yds_allow = 360.0
    return _interp_table(_YDS_TABLE, YDS_BRACKETS, float(mean_yds_allow))


def fit_def_bracket_table(canonical: pd.DataFrame, edges: list[float] | None = None) -> list[list[float]]:
    """Refit the points-allowed bracket table from canonical DEF rows (and install it).

    Returns the table (rows ``[bin centre, share per bracket...]``)."""
    d = canonical[canonical["position"] == "DEF"].copy()
    if d.empty or "pts_allow" not in d.columns:
        return list(_PTS_TABLE)
    d["m"] = d.groupby(["player_id", "season"])["pts_allow"].transform("mean")
    edges = edges or [0, 15, 17, 19, 21, 23, 25, 27, 29, 60]
    d["bin"] = pd.cut(d["m"], edges)
    rows: list[list[float]] = []
    for _, grp in d.groupby("bin", observed=True):
        if len(grp) < 10:
            continue
        rows.append([float(grp["m"].mean())] + [float(grp[b].mean()) if b in grp.columns else 0.0 for b in PTS_BRACKETS])
    if len(rows) >= 3:
        _PTS_TABLE[:] = rows
    return list(_PTS_TABLE)


# ---------------------------------------------------------------------------
# Season totals from per-game rates
# ---------------------------------------------------------------------------

def _finite(v) -> bool:
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def derive_rate_keys(rates: Mapping[str, float], position: str) -> dict[str, float]:
    """Add per-game derived keys (``pass_inc``, K ``fgm/fga/xpa``, DEF brackets,
    position reception bonuses) to a dict of per-game rates."""
    r = {k: float(v) for k, v in rates.items() if _finite(v)}
    if "pass_att" in r and "pass_cmp" in r:
        r.setdefault("pass_inc", max(0.0, r["pass_att"] - r["pass_cmp"]))
    if position == "K" or any(k.startswith("fgm_") for k in r):
        brackets = [r.get(k, 0.0) for k in ("fgm_0_19", "fgm_20_29", "fgm_30_39", "fgm_40_49", "fgm_50p")]
        if "fgm_50p" not in r and ("fgm_50_59" in r or "fgm_60p" in r):
            r["fgm_50p"] = r.get("fgm_50_59", 0.0) + r.get("fgm_60p", 0.0)
            brackets[-1] = r["fgm_50p"]
        r.setdefault("fgm", float(sum(brackets)))
        r.setdefault("fga", r["fgm"] + r.get("fgmiss", 0.0))
        if "xpm" in r:
            r.setdefault("xpa", r["xpm"] + r.get("xpmiss", 0.0))
    if position == "DEF":
        if "pts_allow" in r:
            for k, v in def_bracket_rates(r["pts_allow"]).items():
                r.setdefault(k, v)
        if "yds_allow" in r:
            for k, v in yds_bracket_rates(r["yds_allow"]).items():
                r.setdefault(k, v)
    if "rec" in r:
        r.setdefault(f"bonus_rec_{position.lower()}", r["rec"])
    return r


def rates_to_season(rates: Mapping[str, float], games: float, engine: ScoringEngine,
                    position: str) -> tuple[float, float, dict[str, float]]:
    """Per-game rates + games -> ``(season points, ppg, season stat line)``.

    Derived keys are added with :func:`derive_rate_keys` so bonus thresholds,
    kicking totals and DEF brackets score under any league settings.
    """
    g = float(games) if _finite(games) else 0.0
    g = max(0.0, g)
    per_game = derive_rate_keys(rates, position)
    ppg = engine.score(per_game, position)
    stat_line = {k: v * g for k, v in per_game.items()}
    return ppg * g, ppg, stat_line


def sleeper_stats_to_projection(stats: Mapping[str, float], engine: ScoringEngine,
                                position: str) -> tuple[float, float, dict[str, float]]:
    """Score a Sleeper season projection line -> ``(points, games, stat_line)``.

    Uses ``gp`` when present (else 17). Derived totals that Sleeper may omit are
    filled (``pass_inc``, ``fga``, ``xpa``, ``fgm_50p``, position reception bonus);
    per-game threshold bonuses are *not* derived from season totals.
    """
    line: dict[str, float] = {}
    for k, v in stats.items():
        if k in ("gp",) or k.startswith("adp") or k.startswith("pts_") or k in ("gms_active",):
            continue
        if _finite(v):
            line[k] = float(v)
    if "pass_att" in line and "pass_cmp" in line:
        line.setdefault("pass_inc", max(0.0, line["pass_att"] - line["pass_cmp"]))
    if "fgm_50p" not in line and ("fgm_50_59" in line or "fgm_60p" in line):
        line["fgm_50p"] = line.get("fgm_50_59", 0.0) + line.get("fgm_60p", 0.0)
    if "fgm" in line and "fga" not in line and "fgmiss" in line:
        line["fga"] = line["fgm"] + line["fgmiss"]
    if "xpm" in line and "xpa" not in line and "xpmiss" in line:
        line["xpa"] = line["xpm"] + line["xpmiss"]
    if "rec" in line:
        line.setdefault(f"bonus_rec_{position.lower()}", line["rec"])
    games = stats.get("gp")
    games = float(games) if _finite(games) and float(games) > 0 else float(GAMES_PER_TEAM)
    return engine.score(line, position), games, line


# ---------------------------------------------------------------------------
# ECR -> points
# ---------------------------------------------------------------------------

class EcrCurve:
    """Monotone-decreasing map from ECR overall rank to season points (one position)."""

    def __init__(self, ecr: np.ndarray, points: np.ndarray):
        self.iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
        self.iso.fit(ecr, points)
        self.lo, self.hi = float(np.min(ecr)), float(np.max(ecr))

    def predict(self, ecr: float) -> float:
        return float(self.iso.predict(np.array([float(ecr)]))[0])

    def sd_points(self, ecr: float, sd: float | None) -> float | None:
        """ECR rank std converted to points via the local slope of the curve."""
        if sd is None or not _finite(sd) or sd <= 0:
            return None
        h = max(float(sd), 1.0)
        lo = self.predict(ecr - h)
        hi = self.predict(ecr + h)
        return abs(lo - hi) / 2.0


def fit_ecr_curve(players: Mapping[str, Player], provisional: Mapping[str, float], position: str) -> EcrCurve | None:
    xs, ys = [], []
    for pid, pl in players.items():
        if pl.position != position or pl.ecr is None or not _finite(pl.ecr):
            continue
        v = provisional.get(pid)
        if v is None or not _finite(v):
            continue
        xs.append(float(pl.ecr))
        ys.append(float(v))
    if len(xs) < MIN_ECR_POINTS:
        return None
    return EcrCurve(np.asarray(xs), np.asarray(ys))


def ecr_implied_points(players: Mapping[str, Player], provisional: Mapping[str, float],
                       position: str) -> dict[str, float]:
    """ECR-implied season points for every player at ``position`` with an ECR.

    Isotonic (decreasing) regression of the provisional points on ECR overall rank,
    fitted on players that have both; evaluated for everyone with an ECR. Empty
    when fewer than :data:`MIN_ECR_POINTS` players can be used.
    """
    curve = fit_ecr_curve(players, provisional, position)
    if curve is None:
        return {}
    return {pid: max(0.0, curve.predict(pl.ecr)) for pid, pl in players.items()
            if pl.position == position and pl.ecr is not None and _finite(pl.ecr)}


# ---------------------------------------------------------------------------
# Market curves: consensus positional rank -> expected season points
# ---------------------------------------------------------------------------

#: Realized season points by positional rank (half-PPR, 4-pt pass TD), mean of 2019-2025.
#: Used only when no canonical data is available to fit league-specific curves.
DEFAULT_RANK_TABLE: dict[str, list[tuple[int, float]]] = {
    "QB": [(1, 413), (2, 380), (3, 369), (5, 341), (8, 314), (10, 303), (12, 282), (14, 262), (16, 251), (20, 223),
           (24, 187), (30, 134), (36, 89), (48, 30), (60, 11)],
    "RB": [(1, 355), (2, 310), (3, 297), (5, 259), (8, 230), (10, 218), (12, 208), (14, 203), (16, 191), (20, 178),
           (24, 164), (30, 142), (36, 122), (48, 89), (60, 64)],
    "WR": [(1, 323), (2, 284), (3, 265), (5, 241), (8, 214), (10, 207), (12, 202), (14, 197), (16, 188), (20, 179),
           (24, 172), (30, 157), (36, 144), (48, 118), (60, 97)],
    "TE": [(1, 233), (2, 191), (3, 170), (5, 154), (8, 135), (10, 128), (12, 118), (14, 112), (16, 105), (20, 95),
           (24, 83), (30, 68), (36, 58), (48, 40), (60, 27)],
    "K": [(1, 173), (2, 167), (3, 160), (5, 150), (8, 142), (10, 136), (12, 132), (14, 128), (16, 123), (20, 115),
          (24, 104), (30, 72), (36, 31)],
    "DEF": [(1, 185), (2, 170), (3, 162), (5, 151), (8, 140), (10, 133), (12, 129), (14, 122), (16, 117), (20, 108),
            (24, 99), (30, 72)],
}


class RankCurve:
    """Expected season points as a function of *preseason* positional rank.

    ``realized`` is the average season total of the r-th best finisher in past
    seasons (league scoring). Preseason rankings are noisier than finishes, so the
    expectation for a consensus rank ``r`` above replacement rank ``r0`` is shrunk:
    ``realized(r0) + MARKET_SHRINK * (realized(r) - realized(r0))``.
    """

    def __init__(self, anchors: list[tuple[float, float]], replacement_rank: float, shrink: float = MARKET_SHRINK):
        pts = sorted((float(r), float(p)) for r, p in anchors if _finite(p))
        if len(pts) < 2:
            raise ValueError("need at least two anchors")
        self.ranks = np.array([r for r, _ in pts])
        self.points = np.array([p for _, p in pts])
        self.r0 = max(1.0, float(replacement_rank))
        self.shrink = float(shrink)

    def realized(self, rank: float) -> float:
        r = max(1.0, float(rank))
        if r <= self.ranks[-1]:
            return float(np.interp(r, self.ranks, self.points))
        # exponential tail from the last two anchors, floored at zero
        r1, r2 = self.ranks[-2], self.ranks[-1]
        p1, p2 = max(self.points[-2], 1e-6), max(self.points[-1], 1e-6)
        decay = np.log(p1 / p2) / max(r2 - r1, 1.0)
        return float(p2 * np.exp(-decay * (r - r2)))

    def expected(self, rank: float) -> float:
        base = self.realized(self.r0)
        val = self.realized(rank)
        if rank < self.r0:
            return base + self.shrink * (val - base)
        return val

    def sd_points(self, rank: float, rank_sd: float) -> float:
        h = max(float(rank_sd), 0.75)
        return abs(self.expected(max(1.0, rank - h)) - self.expected(rank + h)) / 2.0


def market_replacement_rank(league: LeagueSettings | None, position: str) -> float:
    """Positional rank of the replacement-level player implied by the roster shape."""
    teams = league.total_rosters if league else 12
    if league is None:
        dedicated = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DEF": 1}[position]
        flex = {"RB": 0.45, "WR": 0.45, "TE": 0.10}.get(position, 0.0)
        superflex = False
    else:
        dedicated = league.dedicated_starters(position)
        flex = 0.0
        for slot in league.starting_slots:
            elig = SLOT_ELIGIBILITY.get(slot, frozenset())
            if len(elig) <= 1 or position not in elig:
                continue
            if slot == "SUPER_FLEX":
                flex += {"QB": 0.85, "RB": 0.05, "WR": 0.05, "TE": 0.05}.get(position, 0.0)
            elif slot == "REC_FLEX":
                flex += {"WR": 0.75, "TE": 0.25}.get(position, 0.0)
            elif slot == "WRRB_FLEX":
                flex += 0.5
            else:
                flex += {"RB": 0.45, "WR": 0.45, "TE": 0.10}.get(position, 0.0)
        superflex = league.is_superflex
    bench = {"RB": 0.5, "WR": 0.5, "TE": 0.15, "QB": 0.75 if superflex else 0.15}.get(position, 0.0)
    return max(1.0, teams * (dedicated + flex + bench))


def fit_rank_curves(canonical: pd.DataFrame, engine: ScoringEngine, league: LeagueSettings | None = None,
                    max_rank: int = 90, min_seasons: int = 2) -> dict[str, RankCurve]:
    """League-scored realized points-by-rank curves per position from canonical per-game rows."""
    from ..scoring.engine import aggregate_season

    df = canonical[canonical["position"].isin(SKILL_POSITIONS)]
    if df.empty:
        return {}
    agg = aggregate_season(df, engine)
    out: dict[str, RankCurve] = {}
    for pos in SKILL_POSITIONS:
        rows = []
        for _, g in agg[agg["position"] == pos].groupby("season"):
            pts = np.sort(g["points"].to_numpy(dtype=float))[::-1][:max_rank]
            rows.append(pts)
        if len(rows) < min_seasons:
            continue
        n = min(len(r) for r in rows)
        if n < 5:
            continue
        mean = np.mean([r[:n] for r in rows], axis=0)
        anchors = [(i + 1, float(mean[i])) for i in range(n)]
        out[pos] = RankCurve(anchors, market_replacement_rank(league, pos))
    return out


def default_rank_curves(league: LeagueSettings | None = None) -> dict[str, RankCurve]:
    return {pos: RankCurve(tab, market_replacement_rank(league, pos)) for pos, tab in DEFAULT_RANK_TABLE.items()}


class MarketCurve:
    """ECR overall rank -> expected points for one position, via positional rank on a RankCurve.

    Same interface as :class:`EcrCurve` (``predict``/``sd_points``) so the Projector can use either.
    """

    def __init__(self, curve: RankCurve, ecrs: np.ndarray):
        self.curve = curve
        self.ecrs = np.sort(np.asarray(ecrs, dtype=float))

    @classmethod
    def from_universe(cls, players: Mapping[str, Player], position: str, curve: RankCurve) -> "MarketCurve | None":
        ecrs = [float(pl.ecr) for pl in players.values()
                if pl.position == position and pl.ecr is not None and _finite(pl.ecr)]
        if len(ecrs) < 3:
            return None
        return cls(curve, np.asarray(ecrs))

    def rank(self, ecr: float) -> float:
        return float(np.searchsorted(self.ecrs, float(ecr), side="left")) + 1.0

    def predict(self, ecr: float) -> float:
        return max(0.0, self.curve.expected(self.rank(ecr)))

    def sd_points(self, ecr: float, sd: float | None) -> float | None:
        if sd is None or not _finite(sd) or sd <= 0:
            return None
        lo = self.rank(max(1.0, float(ecr) - float(sd)))
        hi = self.rank(float(ecr) + float(sd))
        rank_sd = max(0.75, (hi - lo) / 2.0)
        return self.curve.sd_points(self.rank(ecr), rank_sd)


# ---------------------------------------------------------------------------
# Projector
# ---------------------------------------------------------------------------

_PPR_ENGINE = ScoringEngine(PPR_SCORING)


def _blend(values: dict[str, float], weights: Mapping[str, float]) -> tuple[float, dict[str, float]]:
    """Weighted mean over available sources with renormalised weights."""
    avail = {k: float(weights.get(k, 0.0)) for k in values if weights.get(k, 0.0) > 0}
    tot = sum(avail.values())
    if tot <= 0:
        if not values:
            return 0.0, {}
        w = {k: 1.0 / len(values) for k in values}
        return sum(values[k] * w[k] for k in values), w
    w = {k: v / tot for k, v in avail.items()}
    return sum(values[k] * w[k] for k in w), w


class Projector:
    """Turns ML predictions, Sleeper projections and ECR into :class:`Projection` objects."""

    def __init__(self, engine: ScoringEngine, league: LeagueSettings | None = None,
                 settings: Settings | None = None, weights: Mapping[str, float] | None = None,
                 rank_curves: Mapping[str, "RankCurve"] | None = None):
        self.engine = engine
        self.league = league
        self.settings = settings
        self.weights: dict[str, float] = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update({k: float(v) for k, v in weights.items()})
        self.rank_curves: dict[str, RankCurve] = dict(rank_curves) if rank_curves else {}

    def fit_market(self, canonical: pd.DataFrame | None) -> "Projector":
        """Fit league-scored points-by-rank curves (falls back to the built-in table)."""
        curves: dict[str, RankCurve] = {}
        if canonical is not None and len(canonical):
            try:
                curves = fit_rank_curves(canonical, self.engine, self.league)
            except Exception as e:  # noqa: BLE001
                log.warning("rank curve fit failed (%s); using default table", e)
        if not curves:
            curves = default_rank_curves(self.league)
        self.rank_curves = curves
        return self

    # -- per-source ------------------------------------------------------------
    def _ml_source(self, pl: Player, row: pd.Series) -> dict | None:
        keys = TARGETS.get(pl.position, [])
        rates = {k: row.get(f"pred_{k}") for k in keys}
        rates = {k: float(v) for k, v in rates.items() if _finite(v)}
        if not rates:
            return None
        games = row.get("pred_games")
        games = float(games) if _finite(games) else DEFAULT_GAMES.get(pl.position, DEFAULT_GAMES_OTHER)
        games = float(np.clip(games, 0.0, GAMES_PER_TEAM))
        points, ppg, line = rates_to_season(rates, games, self.engine, pl.position)
        ppg_ppr = _PPR_ENGINE.score(derive_rate_keys(rates, pl.position), pl.position)
        std_ppg_ppr = row.get("ppg_std_ppr")
        std_ppg = None
        if _finite(std_ppg_ppr):
            scale = (ppg / ppg_ppr) if ppg_ppr > 0 and ppg > 0 else 1.0
            std_ppg = float(std_ppg_ppr) * scale
        rookie = bool(_finite(row.get("rookie_flag")) and float(row.get("rookie_flag")) > 0)
        return {"points": points, "ppg": ppg, "games": games, "line": line, "std_ppg": std_ppg, "rookie": rookie}

    # -- main ----------------------------------------------------------------------
    def project(self, players: Mapping[str, Player], ml_pred: pd.DataFrame | None = None,
                sleeper_proj: Mapping[str, Mapping] | None = None, notes: Mapping[str, ResearchNote] | None = None,
                byes: Mapping[str, int] | None = None, season_weeks: int = FANTASY_WEEKS) -> dict[str, Projection]:
        """Blend every source per player and apply injury / depth / research adjustments."""
        ml_rows: dict[str, pd.Series] = {}
        if ml_pred is not None and len(ml_pred):
            idx = ml_pred.index.astype(str)
            for pid in players:
                if pid in idx:
                    ml_rows[pid] = ml_pred.loc[pid] if pid in ml_pred.index else ml_pred[idx == pid].iloc[0]
        ml: dict[str, dict] = {}
        sl: dict[str, dict] = {}
        for pid, pl in players.items():
            if pl.position not in SKILL_POSITIONS:
                continue
            row = ml_rows.get(pid)
            if row is not None:
                src = self._ml_source(pl, row)
                if src is not None:
                    ml[pid] = src
            stats = sleeper_proj.get(pid) if sleeper_proj else None
            if stats:
                pts, g, line = sleeper_stats_to_projection(stats, self.engine, pl.position)
                if pts != 0.0 or line:
                    sl[pid] = {"points": pts, "games": g, "line": line}

        # provisional (ml + sleeper) -> ECR curves per position
        provisional: dict[str, float] = {}
        for pid in set(ml) | set(sl):
            vals = {}
            if pid in ml:
                vals["ml"] = ml[pid]["points"]
            if pid in sl:
                vals["sleeper"] = sl[pid]["points"]
            provisional[pid], _ = _blend(vals, self.weights)
        curves: dict[str, EcrCurve | MarketCurve | None] = {}
        for pos in SKILL_POSITIONS:
            mc = MarketCurve.from_universe(players, pos, self.rank_curves[pos]) if pos in self.rank_curves else None
            curves[pos] = mc if mc is not None else fit_ecr_curve(players, provisional, pos)
            if curves[pos] is None:
                log.debug("ECR curve for %s skipped (too few points)", pos)

        out: dict[str, Projection] = {}
        for pid, pl in players.items():
            if pl.position not in SKILL_POSITIONS:
                continue
            out[pid] = self._project_one(pl, ml.get(pid), sl.get(pid), curves.get(pl.position),
                                         notes.get(pid) if notes else None, byes, season_weeks)
        log.info("projected %d players (ml=%d, sleeper=%d, ecr curves=%d)", len(out), len(ml), len(sl),
                 sum(1 for c in curves.values() if c is not None))
        return out

    def _project_one(self, pl: Player, m: dict | None, s: dict | None, curve: "EcrCurve | MarketCurve | None",
                     note: ResearchNote | None, byes: Mapping[str, int] | None, season_weeks: int) -> Projection:
        pos = pl.position
        flags: list[str] = []
        comps: dict[str, float] = {}
        if m is not None:
            comps["ml"] = m["points"]
        if s is not None:
            comps["sleeper"] = s["points"]
        ecr_pts = None
        ecr_sd_pts = None
        if curve is not None and pl.ecr is not None and _finite(pl.ecr):
            ecr_pts = max(0.0, curve.predict(pl.ecr))
            comps["ecr"] = ecr_pts
            ecr_sd_pts = curve.sd_points(pl.ecr, pl.ecr_sd)
        if not comps:
            return Projection(player_id=pl.player_id, position=pos, points=0.0, std=0.0, ppg=0.0, games=0.0,
                              weekly=[0.0] * season_weeks, flags=["no_data"])

        points, weights = _blend(comps, self.weights)
        # games: ML and Sleeper only
        gvals = {}
        if m is not None:
            g_ml = float(m["games"])
            if pos in ("QB", "RB", "WR", "TE", "K") and g_ml < GAMES_SHRINK_TARGET:
                # the games model is trained on realized seasons (injuries, benchings); for a draft
                # we want the expectation for a healthy starter, so shrink toward a full season
                g_ml = g_ml + GAMES_SHRINK * (GAMES_SHRINK_TARGET - g_ml)
            gvals["ml"] = g_ml
        if s is not None:
            gvals["sleeper"] = s["games"]
        if gvals:
            games, _ = _blend(gvals, self.weights)
        else:
            games = DEFAULT_GAMES.get(pos, DEFAULT_GAMES_OTHER)
            flags.append("ecr_only")
        games = float(np.clip(games, 0.0, GAMES_PER_TEAM))
        ppg = points / games if games > 0 else 0.0

        # std from ML blended with ECR sd in points. The model's ppg std is the
        # out-of-fold *season-level* error of points per game, so the season total
        # scales with games (not sqrt(games), which would only cover game-to-game noise).
        svals: dict[str, float] = {}
        if m is not None and m.get("std_ppg") is not None:
            ml_std = m["std_ppg"] * max(games, 1.0) * (ppg / m["ppg"] if m["ppg"] > 0 and ppg > 0 else 1.0)
            svals["ml"] = min(ml_std, MAX_ML_CV * max(points, 1.0))
        if ecr_sd_pts is not None and ecr_sd_pts > 0:
            svals["ecr"] = ecr_sd_pts
        std = _blend(svals, self.weights)[0] if svals else DEFAULT_CV * points
        rookie = pl.is_rookie or bool(m and m.get("rookie"))
        if rookie:
            std *= 1.25
            flags.append("rookie")

        # ---- adjustments (games / points / std) -------------------------------------
        status = (pl.injury_status or "").strip()
        st = status.upper()
        if st in ("IR", "PUP", "NFI"):
            games = max(0.0, games - 6.0)
            if (pl.status or "") == "Injured Reserve" and pl.ecr is None:
                games *= 0.3
            flags.append(f"injury:{status}")
        elif st in ("OUT", "DOUBTFUL"):
            games = max(0.0, games - 1.0)
            flags.append(f"injury:{status}")
        elif st in ("SUS", "SUSPENDED"):
            games = max(0.0, games - 4.0)
            flags.append(f"injury:{status}")
        elif status:
            flags.append(f"injury:{status}")
        if pl.depth_chart_order is not None and pl.depth_chart_order >= 3 and pos in ("RB", "WR"):
            if pl.ecr is None or pl.ecr >= 100:
                ppg *= 0.8
                flags.append(f"depth{pl.depth_chart_order}")
        if note is not None:
            r = float(np.clip(note.injury_risk, 0.0, 1.0))
            c = float(np.clip(note.role_certainty, 0.0, 1.0))
            games *= (1.0 - 0.25 * r)
            std *= (1.3 - 0.3 * c)
        games = float(np.clip(games, 0.0, GAMES_PER_TEAM))
        points = ppg * games
        std = max(std, MIN_STD_FRACTION * points)
        floor = max(0.0, points - 0.84 * std)
        ceiling = points + 0.84 * std

        # weekly: even spread over non-bye weeks
        bye = pl.bye_week
        if bye is None and byes and pl.team:
            bye = byes.get(pl.team)
        weeks = [w for w in range(1, season_weeks + 1) if w != bye]
        per_week = points / len(weeks) if weeks else 0.0
        weekly = [0.0 if w == bye else per_week for w in range(1, season_weeks + 1)]

        # stat line: weighted average of the ML and Sleeper lines
        line: dict[str, float] = {}
        lines = {k: v["line"] for k, v in (("ml", m), ("sleeper", s)) if v is not None}
        if lines:
            lw = {k: weights.get(k, 0.0) for k in lines}
            tot = sum(lw.values()) or float(len(lines))
            for k, ln in lines.items():
                wk = (lw[k] / tot) if sum(lw.values()) > 0 else 1.0 / len(lines)
                for stat, v in ln.items():
                    line[stat] = line.get(stat, 0.0) + wk * v
        if "ml" not in comps and "sleeper" not in comps:
            pass
        elif "ecr" not in comps:
            flags.append("no_ecr")
        return Projection(
            player_id=pl.player_id, position=pos, points=float(points), std=float(std), ppg=float(ppg),
            games=float(games), floor=float(floor), ceiling=float(ceiling), weekly=weekly, stat_line=line,
            components={k: float(v) for k, v in comps.items()}, weights={k: float(v) for k, v in weights.items()},
            flags=flags,
        )


# ---------------------------------------------------------------------------
# Offline fallback (no trained model)
# ---------------------------------------------------------------------------

#: Shrinkage strength (games) toward the position mean for offline projections.
OFFLINE_SHRINK_K = 6.0
_OFFLINE_STD = {"QB": (2.5, 0.30), "RB": (2.0, 0.35), "WR": (2.0, 0.35), "TE": (1.5, 0.35), "K": (1.5, 0.25),
                "DEF": (2.0, 0.35)}


def project_offline(players: Mapping[str, Player], engine: ScoringEngine, canonical: pd.DataFrame, cw: Crosswalk,
                    season: int) -> pd.DataFrame:
    """ML-source stand-in when no trained model exists.

    Last season's (``season - 1``) per-game rates for every target key, shrunk toward
    the position mean with ``k = OFFLINE_SHRINK_K`` games (``rate = (g*r + k*mean) /
    (g + k)``), so the league-scored PPG is the shrunk last-season PPG. Games are
    shrunk the same way toward the position mean games.

    Returns a DataFrame with the **same shape as** :meth:`ProjectionModel.predict`:
    index = Sleeper player_id; columns ``pred_<key>`` for every key in
    :data:`ALL_TARGET_KEYS` (NaN when not a target at the player's position),
    ``pred_games``, ``pred_ppg_ppr``, ``ppg_std_ppr`` (``a + b * ppg``), ``rookie_flag``
    (always 0) and ``position``. Players with no last-season row are omitted (the
    Projector then uses ECR/Sleeper only).
    """
    agg = season_aggregates(canonical[canonical["season"] == season - 1]) if len(canonical) else pd.DataFrame()
    if agg.empty:
        return pd.DataFrame(columns=[f"pred_{k}" for k in ALL_TARGET_KEYS] + ["pred_games", "pred_ppg_ppr", "ppg_std_ppr", "rookie_flag", "position"])
    agg = agg.set_index("player_id")
    # position means among players with >= 6 games, weighted by games
    means: dict[str, dict[str, float]] = {}
    for pos in SKILL_POSITIONS:
        sub = agg[(agg["position"] == pos) & (agg["games"] >= 6)]
        if sub.empty:
            sub = agg[agg["position"] == pos]
        w = sub["games"].to_numpy(dtype=float) if len(sub) else np.array([])
        d = {"games": float(sub["games"].mean()) if len(sub) else 14.0}
        for k in TARGETS[pos]:
            vals = sub[k].to_numpy(dtype=float) if k in sub.columns else np.array([])
            ok = np.isfinite(vals)
            d[k] = float(np.average(vals[ok], weights=w[ok])) if ok.any() and w[ok].sum() > 0 else 0.0
        means[pos] = d
    rows = []
    k_shrink = OFFLINE_SHRINK_K
    for pid, pl in players.items():
        pos = pl.position
        if pos not in SKILL_POSITIONS:
            continue
        key = (pl.team or pid) if pos == "DEF" else (pl.gsis_id or cw.gsis_for(pid))
        if key is None or key not in agg.index:
            continue
        a = agg.loc[key]
        if isinstance(a, pd.DataFrame):
            a = a.iloc[0]
        g = float(a["games"])
        rec = {"pid": pid, "position": pos, "rookie_flag": 0.0}
        for k in TARGETS[pos]:
            r = a.get(k)
            r = float(r) if _finite(r) else means[pos][k]
            rec[f"pred_{k}"] = (g * r + k_shrink * means[pos][k]) / (g + k_shrink)
        rec["pred_games"] = float(np.clip((g * g + k_shrink * means[pos]["games"]) / (g + k_shrink), 0, GAMES_PER_TEAM))
        rows.append(rec)
    if not rows:
        return pd.DataFrame(columns=[f"pred_{k}" for k in ALL_TARGET_KEYS] + ["pred_games", "pred_ppg_ppr", "ppg_std_ppr", "rookie_flag", "position"])
    df = pd.DataFrame(rows).set_index("pid")
    df.index.name = "player_id"
    df = df.reindex(columns=[f"pred_{k}" for k in ALL_TARGET_KEYS] + ["pred_games", "rookie_flag", "position"])
    ppg = np.zeros(len(df))
    for i, (pid, row) in enumerate(df.iterrows()):
        rates = {k: row[f"pred_{k}"] for k in TARGETS[row["position"]] if _finite(row[f"pred_{k}"])}
        ppg[i] = _PPR_ENGINE.score(derive_rate_keys(rates, row["position"]), row["position"])
    df["pred_ppg_ppr"] = ppg
    ab = df["position"].map(lambda p: _OFFLINE_STD.get(p, (2.0, 0.35)))
    df["ppg_std_ppr"] = [a + b * max(0.0, p) for (a, b), p in zip(ab, ppg)]
    return df
