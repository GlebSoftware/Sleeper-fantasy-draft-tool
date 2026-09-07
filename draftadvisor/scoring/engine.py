"""League scoring: Sleeper ``scoring_settings`` applied to stat lines.

Sleeper's convention is that the keys of ``league["scoring_settings"]`` are the
same keys used in its stats/projections payloads, so for a Sleeper-sourced
stat line the score is a plain dot product. For stat lines we derive ourselves
(from nflverse) the per-game derived keys (yardage bonuses, points-allowed
brackets, ...) are produced by :mod:`draftadvisor.data.canonical` so the same
dot product applies.

Everything here is vectorised with numpy/pandas so scoring 100k player-games
takes milliseconds.
"""
from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd

#: Reasonable default (Sleeper "half PPR" preset) used when a league has no
#: scoring settings (e.g. offline mock drafts).
DEFAULT_SCORING: dict[str, float] = {
    "pass_yd": 0.04, "pass_td": 4.0, "pass_int": -1.0, "pass_2pt": 2.0,
    "rush_yd": 0.1, "rush_td": 6.0, "rush_2pt": 2.0,
    "rec": 0.5, "rec_yd": 0.1, "rec_td": 6.0, "rec_2pt": 2.0,
    "fum_lost": -2.0, "fum_rec_td": 6.0, "st_td": 6.0,
    "fgm_0_19": 3.0, "fgm_20_29": 3.0, "fgm_30_39": 3.0, "fgm_40_49": 4.0, "fgm_50p": 5.0,
    "fgmiss": -1.0, "xpm": 1.0, "xpmiss": -1.0,
    "def_td": 6.0, "sack": 1.0, "int": 2.0, "ff": 1.0, "fum_rec": 2.0, "safe": 2.0,
    "blk_kick": 2.0, "def_st_td": 6.0, "def_st_ff": 1.0, "def_st_fum_rec": 1.0,
    "pts_allow_0": 10.0, "pts_allow_1_6": 7.0, "pts_allow_7_13": 4.0, "pts_allow_14_20": 1.0,
    "pts_allow_21_27": 0.0, "pts_allow_28_34": -1.0, "pts_allow_35p": -4.0,
}

#: Stat keys we know how to produce from nflverse data (see data/canonical.py).
#: Anything a league scores that is *not* in this list is scored only when the
#: stat line came from Sleeper (projections/stats), otherwise ignored.
SLEEPER_STAT_KEYS: tuple[str, ...] = (
    # passing
    "pass_att", "pass_cmp", "pass_inc", "pass_yd", "pass_td", "pass_int", "pass_2pt", "pass_sack",
    "pass_fd", "pass_cmp_40p", "pass_td_40p", "pass_td_50p",
    "bonus_pass_yd_300", "bonus_pass_yd_400", "bonus_pass_cmp_25",
    # rushing
    "rush_att", "rush_yd", "rush_td", "rush_2pt", "rush_fd", "rush_40p", "rush_td_40p", "rush_td_50p",
    "bonus_rush_yd_100", "bonus_rush_yd_200", "bonus_rush_att_20",
    # receiving
    "rec", "rec_tgt", "rec_yd", "rec_td", "rec_2pt", "rec_fd", "rec_40p", "rec_td_40p", "rec_td_50p",
    "bonus_rec_yd_100", "bonus_rec_yd_200", "bonus_rec_rb", "bonus_rec_wr", "bonus_rec_te",
    "bonus_rush_rec_yd_100", "bonus_rush_rec_yd_200",
    # misc
    "fum", "fum_lost", "fum_rec_td", "st_td", "st_ff", "st_fum_rec", "kr_yd", "pr_yd",
    # kicking
    "fgm", "fga", "fgmiss", "fgm_yds", "fgm_yds_over_30",
    "fgm_0_19", "fgm_20_29", "fgm_30_39", "fgm_40_49", "fgm_50_59", "fgm_60p", "fgm_50p",
    "fgmiss_0_19", "fgmiss_20_29", "fgmiss_30_39", "fgmiss_40_49", "fgmiss_50p",
    "xpm", "xpa", "xpmiss",
    # team defense
    "def_td", "sack", "int", "ff", "fum_rec", "safe", "blk_kick", "def_2pt", "def_st_td", "def_st_ff",
    "def_st_fum_rec", "def_forced_punts", "def_pass_def", "def_4_and_stop", "def_3_and_out",
    "pts_allow", "pts_allow_0", "pts_allow_1_6", "pts_allow_7_13", "pts_allow_14_20",
    "pts_allow_21_27", "pts_allow_28_34", "pts_allow_35p",
    "yds_allow", "yds_allow_0_100", "yds_allow_100_199", "yds_allow_200_299", "yds_allow_300_349",
    "yds_allow_350_399", "yds_allow_400_449", "yds_allow_450_499", "yds_allow_500_549", "yds_allow_550p",
)

# Position-specific reception bonuses: the bonus key applies only to that position.
_POS_BONUS = {"bonus_rec_rb": "RB", "bonus_rec_wr": "WR", "bonus_rec_te": "TE"}


def _points_allowed_bracket(pts: float) -> str:
    if pts <= 0:
        return "pts_allow_0"
    if pts <= 6:
        return "pts_allow_1_6"
    if pts <= 13:
        return "pts_allow_7_13"
    if pts <= 20:
        return "pts_allow_14_20"
    if pts <= 27:
        return "pts_allow_21_27"
    if pts <= 34:
        return "pts_allow_28_34"
    return "pts_allow_35p"


def _yards_allowed_bracket(yds: float) -> str:
    if yds < 100:
        return "yds_allow_0_100"
    if yds < 200:
        return "yds_allow_100_199"
    if yds < 300:
        return "yds_allow_200_299"
    if yds < 350:
        return "yds_allow_300_349"
    if yds < 400:
        return "yds_allow_350_399"
    if yds < 450:
        return "yds_allow_400_449"
    if yds < 500:
        return "yds_allow_450_499"
    if yds < 550:
        return "yds_allow_500_549"
    return "yds_allow_550p"


class ScoringEngine:
    """Scores stat lines under one league's ``scoring_settings``.

    >>> eng = ScoringEngine({"rec": 1, "rec_yd": 0.1, "rec_td": 6})
    >>> eng.score({"rec": 5, "rec_yd": 70, "rec_td": 1})
    18.0
    """

    def __init__(self, scoring_settings: Mapping[str, float] | None = None):
        src = dict(scoring_settings) if scoring_settings else dict(DEFAULT_SCORING)
        # Drop zero weights so the dot product only touches keys that matter.
        self.weights: dict[str, float] = {k: float(v) for k, v in src.items() if v not in (None, 0, 0.0)}
        self._keys = list(self.weights)
        self._w = np.array([self.weights[k] for k in self._keys], dtype=float)

    # -- introspection ---------------------------------------------------------
    @property
    def keys(self) -> list[str]:
        return list(self._keys)

    def weight(self, key: str) -> float:
        return self.weights.get(key, 0.0)

    @property
    def rec_points(self) -> float:
        return self.weight("rec")

    def relevant_keys(self, position: str | None = None) -> list[str]:
        """Scoring keys that can be non-zero for ``position`` (all keys if None)."""
        if position is None:
            return self.keys
        pref = {
            "QB": ("pass_", "rush_", "rec", "fum", "bonus_pass", "bonus_rush", "bonus_rec", "st_", "kr_", "pr_"),
            "RB": ("rush_", "rec", "pass_", "fum", "bonus_rush", "bonus_rec", "st_", "kr_", "pr_"),
            "WR": ("rec", "rush_", "pass_", "fum", "bonus_rec", "bonus_rush", "st_", "kr_", "pr_"),
            "TE": ("rec", "rush_", "pass_", "fum", "bonus_rec", "bonus_rush", "st_", "kr_", "pr_"),
            "K": ("fg", "xp"),
            "DEF": ("def_", "sack", "int", "ff", "fum_rec", "safe", "blk_kick", "pts_allow", "yds_allow"),
        }.get(position, ())
        out = []
        for k in self._keys:
            if k in _POS_BONUS and _POS_BONUS[k] != position:
                continue
            if any(k.startswith(p) for p in pref) or k in ("fum", "fum_lost", "fum_rec_td"):
                out.append(k)
        return out

    # -- scoring -----------------------------------------------------------------
    def score(self, stats: Mapping[str, float], position: str | None = None) -> float:
        """Score a single stat line (dict of Sleeper keys)."""
        total = 0.0
        for k, w in self.weights.items():
            if k in _POS_BONUS and position is not None and _POS_BONUS[k] != position:
                continue
            v = stats.get(k)
            if v:
                total += w * float(v)
        return total

    def score_frame(self, df: pd.DataFrame, position_col: str | None = "position") -> pd.Series:
        """Vectorised scoring of a DataFrame whose columns are Sleeper stat keys.

        Missing columns count as zero. Position-specific reception bonuses are
        applied only to rows of the matching position when ``position_col`` exists.
        """
        if len(df) == 0:
            return pd.Series(dtype=float, index=df.index)
        total = np.zeros(len(df), dtype=float)
        for k, w in self.weights.items():
            if k not in df.columns:
                continue
            col = pd.to_numeric(df[k], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            if k in _POS_BONUS and position_col and position_col in df.columns:
                col = np.where(df[position_col].to_numpy() == _POS_BONUS[k], col, 0.0)
            total += w * col
        return pd.Series(total, index=df.index)

    # -- derived keys ------------------------------------------------------------
    @staticmethod
    def add_derived_keys(df: pd.DataFrame) -> pd.DataFrame:
        """Add per-game derived keys (bonus thresholds, brackets) to a per-game frame
        that already has the base Sleeper keys. Idempotent; returns the same frame."""
        def col(name: str) -> pd.Series:
            return pd.to_numeric(df[name], errors="coerce").fillna(0.0) if name in df.columns else pd.Series(0.0, index=df.index)

        pass_yd, rush_yd, rec_yd = col("pass_yd"), col("rush_yd"), col("rec_yd")
        df["bonus_pass_yd_300"] = (pass_yd >= 300).astype(float)
        df["bonus_pass_yd_400"] = (pass_yd >= 400).astype(float)
        df["bonus_rush_yd_100"] = (rush_yd >= 100).astype(float)
        df["bonus_rush_yd_200"] = (rush_yd >= 200).astype(float)
        df["bonus_rec_yd_100"] = (rec_yd >= 100).astype(float)
        df["bonus_rec_yd_200"] = (rec_yd >= 200).astype(float)
        df["bonus_rush_rec_yd_100"] = ((rush_yd + rec_yd) >= 100).astype(float)
        df["bonus_rush_rec_yd_200"] = ((rush_yd + rec_yd) >= 200).astype(float)
        df["bonus_pass_cmp_25"] = (col("pass_cmp") >= 25).astype(float)
        df["bonus_rush_att_20"] = (col("rush_att") >= 20).astype(float)
        if "rec" in df.columns:
            rec = col("rec")
            for k in _POS_BONUS:
                df[k] = rec
        # Bracket indicators only where the source value exists: offensive players share the
        # frame with team defenses and must NOT fall into the "0 points allowed" bracket.
        if "pts_allow" in df.columns:
            raw = pd.to_numeric(df["pts_allow"], errors="coerce")
            br = raw.map(lambda v: _points_allowed_bracket(v) if pd.notna(v) else None)
            for k in ("pts_allow_0", "pts_allow_1_6", "pts_allow_7_13", "pts_allow_14_20",
                      "pts_allow_21_27", "pts_allow_28_34", "pts_allow_35p"):
                df[k] = (br == k).astype(float)
        if "yds_allow" in df.columns:
            raw = pd.to_numeric(df["yds_allow"], errors="coerce")
            br = raw.map(lambda v: _yards_allowed_bracket(v) if pd.notna(v) else None)
            for k in ("yds_allow_0_100", "yds_allow_100_199", "yds_allow_200_299", "yds_allow_300_349",
                      "yds_allow_350_399", "yds_allow_400_449", "yds_allow_450_499", "yds_allow_500_549",
                      "yds_allow_550p"):
                df[k] = (br == k).astype(float)
        if "fgm_50_59" in df.columns or "fgm_60p" in df.columns:
            df["fgm_50p"] = col("fgm_50_59") + col("fgm_60p")
        if "fgmiss_50_59" in df.columns or "fgmiss_60p" in df.columns:
            df["fgmiss_50p"] = col("fgmiss_50_59") + col("fgmiss_60p")
        return df

    def describe(self) -> str:
        """Short human description, e.g. 'PPR, 4pt pass TD, TE +0.5'."""
        parts = []
        rec = self.rec_points
        parts.append({1.0: "PPR", 0.5: "Half PPR", 0.0: "Standard"}.get(rec, f"{rec:g}/rec"))
        parts.append(f"{self.weight('pass_td'):g}pt pass TD")
        if self.weight("bonus_rec_te"):
            parts.append(f"TE +{self.weight('bonus_rec_te'):g}")
        if self.weight("pass_yd") and abs(self.weight("pass_yd") - 0.04) > 1e-9:
            parts.append(f"{1/self.weight('pass_yd'):.0f} pass yd/pt")
        return ", ".join(parts)


def aggregate_season(per_game: pd.DataFrame, engine: ScoringEngine, keys: Iterable[str] | None = None,
                     group_cols: tuple[str, ...] = ("player_id", "season")) -> pd.DataFrame:
    """Aggregate a per-game canonical frame into per-player-season totals.

    Returns columns: group cols, ``games``, ``points`` (league scoring), ``ppg``
    and the summed stat keys.
    """
    df = per_game.copy()
    df["points"] = engine.score_frame(df)
    if keys is None:
        keys = [k for k in SLEEPER_STAT_KEYS if k in df.columns]
    agg = {k: "sum" for k in keys}
    agg["points"] = "sum"
    g = df.groupby(list(group_cols), as_index=False).agg(agg)
    games = df.groupby(list(group_cols)).size().rename("games").reset_index()
    g = g.merge(games, on=list(group_cols))
    g["ppg"] = g["points"] / g["games"].clip(lower=1)
    if "position" in df.columns:
        pos = df.groupby(list(group_cols))["position"].agg(lambda s: s.mode().iat[0]).reset_index()
        g = g.merge(pos, on=list(group_cols))
    return g
