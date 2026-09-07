"""Per-position gradient-boosting projection model (DESIGN.md §3.3).

For every position and every per-game rate target in :data:`TARGETS` a
``HistGradientBoostingRegressor`` (NaN-native, so missing history is handled
without imputation). The PPR points-per-game implied by the predicted rates is
calibrated against two cheap, complementary per-position models of PPR ppg -
a direct HGB and a ridge regression on a handful of strong features (last
season / career ppg, games, age, draft capital) - and the rate vector is
rescaled so its score equals the calibrated ppg. On the 2024-2025 backtest this
blend beats each component and the "last season ppg" baseline at RB/WR/TE.

Rookies (no NFL history, first season) use a separate small model per position
driven by draft capital and age. A games model predicts games played (clipped
to [0, 17]). Uncertainty is heteroscedastic: the residual std of PPR ppg is
``a + b * pred`` fitted on 5-fold out-of-fold predictions of the direct model.

Note: ``sample_weight`` is deliberately not used - it makes HGB ~15x slower in
scikit-learn and the training budget is 3 minutes on 4 cores.
"""
from __future__ import annotations

import logging
import pickle
import time
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from ..config import DEFAULT_SEASON, GAMES_PER_TEAM, SKILL_POSITIONS, TRAIN_SEASONS, models_dir
from ..scoring.engine import ScoringEngine
from .features import (
    ALL_TARGET_KEYS,
    FEATURE_COLUMNS,
    PPR_SCORING,
    ROOKIE_FEATURE_COLUMNS,
    TARGETS,
    build_training_table,
    season_aggregates,
)

log = logging.getLogger(__name__)

DEFAULT_PARAMS: dict = {
    "max_iter": 250,
    "learning_rate": 0.03,
    "max_depth": 3,
    "min_samples_leaf": 30,
    "l2_regularization": 5.0,
    "max_leaf_nodes": 8,
    # rows with fewer games than this are too noisy to define a per-game rate
    "min_rate_games": 1,
    # weights of the three PPR-ppg estimates that calibrate the rate vector
    "ppg_blend": {"rates": 1.0 / 3, "direct": 1.0 / 3, "linear": 1.0 / 3},
    # extra shrinkage of predicted rates toward the position mean for rows without history
    "no_history_shrink": 0.35,
    "ridge_lambda": 3.0,
}

ROOKIE_PARAMS: dict = {
    "max_iter": 120, "learning_rate": 0.05, "max_depth": 2, "min_samples_leaf": 15,
    "l2_regularization": 2.0, "max_leaf_nodes": 4,
}

#: Features of the ridge (linear) PPR-ppg model.
RIDGE_FEATURE_COLUMNS: list[str] = [
    "prev_ppg_ppr", "prev2_ppg_ppr", "career_ppg_ppr", "prev_games", "prev2_games", "age", "draft_ovr",
    "prev_pos_rank", "prev_snap_pct", "prev_ppg_ppr_sd",
]

#: Minimum rookie rows per position before a rookie model is fitted (else position rookie means).
MIN_ROOKIE_ROWS = 25
#: Minimum rows per position before HGB models are fitted (else position means).
MIN_POSITION_ROWS = 50

PRED_COLUMNS: list[str] = [f"pred_{k}" for k in ALL_TARGET_KEYS] + ["pred_games", "pred_ppg_ppr", "ppg_std_ppr", "rookie_flag"]

_PPR = ScoringEngine(PPR_SCORING)


def _hgb(params: dict, seed: int) -> HistGradientBoostingRegressor:
    keys = ("max_iter", "learning_rate", "max_depth", "min_samples_leaf", "l2_regularization", "max_leaf_nodes")
    kw = {k: params[k] for k in keys if k in params}
    return HistGradientBoostingRegressor(loss="squared_error", random_state=seed, early_stopping=False, **kw)


def _fit(mdl: HistGradientBoostingRegressor, X: np.ndarray, y: np.ndarray) -> HistGradientBoostingRegressor:
    """Fit with all-NaN feature columns zero-filled (sklearn cannot bin an empty column).

    A constant column is never split on, so NaNs in it at predict time are harmless."""
    allnan = np.isnan(X).all(axis=0)
    if allnan.any():
        X = X.copy()
        X[:, allnan] = 0.0
    return mdl.fit(X, y)


def ppr_ppg_from_rates(rates: pd.DataFrame, positions: pd.Series) -> pd.Series:
    """Score a frame of per-game rate columns (``pred_<key>`` or bare keys) with PPR scoring."""
    df = rates.rename(columns={c: c[5:] for c in rates.columns if c.startswith("pred_")}).copy()
    df["position"] = positions.to_numpy()
    return _PPR.score_frame(df)


class RidgeModel:
    """Ridge regression with mean imputation + missing indicators (pure numpy, picklable)."""

    def __init__(self, lam: float = 3.0):
        self.lam = float(lam)
        self.mu: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.beta: np.ndarray | None = None

    def _design(self, X: np.ndarray) -> np.ndarray:
        miss = np.isnan(X).astype(float)
        Z = np.where(np.isnan(X), self.mu, X)
        return np.column_stack([np.ones(len(X)), Z, miss])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeModel":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
            mu = np.nanmean(X, axis=0)
        self.mu = np.where(np.isnan(mu), 0.0, mu)
        A = self._design(X)
        scale = A.std(axis=0)
        scale[scale == 0] = 1.0
        scale[0] = 1.0
        self.scale = scale
        A = A / scale
        I = np.eye(A.shape[1])
        I[0, 0] = 0.0
        self.beta = np.linalg.solve(A.T @ A + self.lam * I, A.T @ y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.beta is None:
            return np.zeros(len(X))
        return (self._design(X) / self.scale) @ self.beta


class ProjectionModel:
    """Per-position, per-target HGB models with calibration / rookie / games / std sub-models."""

    def __init__(self, seed: int = 7, params: dict | None = None):
        self.seed = int(seed)
        self.params: dict = {**DEFAULT_PARAMS, **(params or {})}
        self.feature_columns: list[str] = list(FEATURE_COLUMNS)
        self.rookie_feature_columns: list[str] = list(ROOKIE_FEATURE_COLUMNS)
        self.ridge_feature_columns: list[str] = list(RIDGE_FEATURE_COLUMNS)
        self.models: dict[str, dict[str, HistGradientBoostingRegressor]] = {}
        self.games_models: dict[str, HistGradientBoostingRegressor] = {}
        self.direct_models: dict[str, HistGradientBoostingRegressor] = {}
        self.ridge_models: dict[str, RidgeModel] = {}
        self.rookie_models: dict[str, dict[str, HistGradientBoostingRegressor]] = {}
        self.rookie_means: dict[str, dict[str, float]] = {}
        self.pos_means: dict[str, dict[str, float]] = {}
        self.std_coef: dict[str, tuple[float, float]] = {}
        self.meta: dict = {}

    # ------------------------------------------------------------------ fit
    @staticmethod
    def _X(sub: pd.DataFrame, cols: list[str]) -> np.ndarray:
        return np.array(sub.reindex(columns=cols).to_numpy(dtype=float), dtype=float, copy=True)

    def _fit_position(self, pos: str, sub: pd.DataFrame) -> None:
        X = self._X(sub, self.feature_columns)
        games = sub["y_games"].fillna(0.0)
        self.models[pos] = {}
        self.pos_means[pos] = {}
        played = games >= float(self.params.get("min_rate_games", 1))
        for key in TARGETS[pos]:
            y = pd.to_numeric(sub[f"y_{key}"], errors="coerce")
            m = played & y.notna()
            if m.sum() < 30:
                log.warning("%s/%s: only %d rows; using mean", pos, key, int(m.sum()))
                self.pos_means[pos][key] = float(y[m].mean()) if m.any() else 0.0
                continue
            self.pos_means[pos][key] = float(y[m].mean())
            self.models[pos][key] = _fit(_hgb(self.params, self.seed), X[m.to_numpy()], y[m].to_numpy(dtype=float))
        # games
        self.games_models[pos] = _fit(_hgb(self.params, self.seed), X, games.clip(0, GAMES_PER_TEAM).to_numpy(dtype=float))
        self.pos_means[pos]["games"] = float(games.clip(0, GAMES_PER_TEAM).mean())
        # calibration models of PPR ppg
        y = pd.to_numeric(sub["y_ppg_ppr"], errors="coerce")
        m = played & y.notna()
        if m.sum() >= 30:
            self.direct_models[pos] = _fit(_hgb(self.params, self.seed), X[m.to_numpy()], y[m].to_numpy(dtype=float))
            Xr = self._X(sub, self.ridge_feature_columns)
            self.ridge_models[pos] = RidgeModel(self.params.get("ridge_lambda", 3.0)).fit(Xr[m.to_numpy()], y[m].to_numpy(dtype=float))
        self.pos_means[pos]["ppg_ppr"] = float(y[m].mean()) if m.any() else 0.0
        # residual std from OOF predictions of the direct ppg model
        self.std_coef[pos] = self._fit_std(pos, X, sub)

    def _fit_means_only(self, pos: str, sub: pd.DataFrame) -> None:
        """Degraded fit for a position with too few rows: predict position means."""
        games = sub["y_games"].fillna(0.0)
        played = games > 0
        self.models[pos] = {}
        self.pos_means[pos] = {}
        for key in TARGETS[pos]:
            y = pd.to_numeric(sub[f"y_{key}"], errors="coerce")
            m = played & y.notna()
            self.pos_means[pos][key] = float(y[m].mean()) if m.any() else 0.0
        self.pos_means[pos]["games"] = float(games.clip(0, GAMES_PER_TEAM).mean())
        y = pd.to_numeric(sub["y_ppg_ppr"], errors="coerce")
        self.pos_means[pos]["ppg_ppr"] = float(y[played].mean()) if played.any() else 0.0
        self.std_coef[pos] = (max(1.0, 0.3 * self.pos_means[pos]["ppg_ppr"]), 0.3)
        self.rookie_models[pos] = {}
        self.rookie_means[pos] = {}

    def _fit_std(self, pos: str, X: np.ndarray, sub: pd.DataFrame) -> tuple[float, float]:
        y = pd.to_numeric(sub["y_ppg_ppr"], errors="coerce")
        # same population as the backtest (>= 6 games): short cameo seasons inflate |resid|
        m = (sub["y_games"].fillna(0) >= 6) & y.notna()
        idx = np.flatnonzero(m.to_numpy())
        if len(idx) < 60:
            mean = float(y[m].mean()) if m.any() else 5.0
            return (max(1.0, 0.3 * mean), 0.3)
        Xm, ym = X[idx], y.to_numpy(dtype=float)[idx]
        oof = np.zeros(len(idx))
        kf = KFold(n_splits=5, shuffle=True, random_state=self.seed)
        fast = {**self.params, "max_iter": min(120, int(self.params["max_iter"])), "learning_rate": 0.06}
        for tr, te in kf.split(Xm):
            mdl = _fit(_hgb(fast, self.seed), Xm[tr], ym[tr])
            oof[te] = mdl.predict(Xm[te])
        resid = np.abs(ym - oof)
        pred = np.clip(oof, 0, None)
        # |resid| ~ a + b * pred  (least squares), clipped to sane values
        A = np.column_stack([np.ones_like(pred), pred])
        coef, *_ = np.linalg.lstsq(A, resid, rcond=None)
        a = float(np.clip(coef[0], 0.5, 4.0))
        b = float(np.clip(coef[1], 0.05, 0.6))
        s = float(np.sqrt(np.pi / 2))  # E|x| = sigma * sqrt(2/pi) for a normal
        log.info("%s std model: std = %.2f + %.3f * pred (n=%d, mean|resid|=%.2f)", pos, a * s, b * s, len(idx), resid.mean())
        return (a * s, b * s)

    def _fit_rookies(self, pos: str, sub: pd.DataFrame) -> None:
        r = sub[(sub["rookie"] == 1) & (sub["y_games"].fillna(0) > 0)]
        self.rookie_models[pos] = {}
        self.rookie_means[pos] = {}
        if r.empty:
            return
        X = self._X(r, self.rookie_feature_columns)
        games = r["y_games"].fillna(0.0)
        for key in TARGETS[pos] + ["games"]:
            y = games.clip(0, GAMES_PER_TEAM) if key == "games" else pd.to_numeric(r[f"y_{key}"], errors="coerce")
            m = y.notna()
            if not m.any():
                continue
            self.rookie_means[pos][key] = float(y[m].mean())
            if m.sum() >= MIN_ROOKIE_ROWS:
                self.rookie_models[pos][key] = _fit(_hgb(ROOKIE_PARAMS, self.seed), X[m.to_numpy()], y[m].to_numpy(dtype=float))

    def fit(self, table: pd.DataFrame) -> "ProjectionModel":
        """Fit every sub-model on a training table from :func:`build_training_table`."""
        t0 = time.perf_counter()
        for pos in SKILL_POSITIONS:
            sub = table[table["position"] == pos]
            if sub.empty:
                continue
            if len(sub) < MIN_POSITION_ROWS:
                log.warning("position %s has only %d training rows; using position means", pos, len(sub))
                self._fit_means_only(pos, sub)
                continue
            self._fit_position(pos, sub)
            self._fit_rookies(pos, sub)
            log.info("fitted %s on %d rows (%.1fs elapsed)", pos, len(sub), time.perf_counter() - t0)
        self.meta.update({
            "fitted_at": time.time(), "rows": int(len(table)),
            "seasons": sorted(int(s) for s in pd.unique(table["season"])) if "season" in table.columns else [],
            "fit_seconds": time.perf_counter() - t0,
        })
        return self

    # -------------------------------------------------------------- predict
    def _calibrated_ppg(self, pos: str, sub: pd.DataFrame, X: np.ndarray, rates_ppg: np.ndarray) -> np.ndarray:
        """Blend of the rates-implied PPR ppg with the direct and ridge estimates."""
        w = dict(self.params.get("ppg_blend", {}))
        est = {"rates": rates_ppg}
        if pos in self.direct_models:
            est["direct"] = np.clip(self.direct_models[pos].predict(X), 0, None)
        if pos in self.ridge_models:
            est["linear"] = np.clip(self.ridge_models[pos].predict(self._X(sub, self.ridge_feature_columns)), 0, None)
        ws = {k: float(w.get(k, 0.0)) for k in est if w.get(k, 0.0) > 0}
        tot = sum(ws.values())
        if tot <= 0:
            return rates_ppg
        return sum(est[k] * (v / tot) for k, v in ws.items())

    def _predict_position(self, pos: str, sub: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=sub.index, columns=PRED_COLUMNS, dtype=float)
        X = self._X(sub, self.feature_columns)
        means = self.pos_means.get(pos, {})
        shrink = float(self.params.get("no_history_shrink", 0.0))
        no_hist = sub["no_history"].fillna(1.0).to_numpy(dtype=float) if "no_history" in sub.columns else np.ones(len(sub))
        rate_cols = [f"pred_{k}" for k in TARGETS[pos]]
        for key in TARGETS[pos]:
            mdl = self.models.get(pos, {}).get(key)
            mean = means.get(key, 0.0)
            pred = np.full(len(sub), mean) if mdl is None else mdl.predict(X)
            if mdl is not None and shrink > 0:
                pred = np.where(no_hist > 0, (1 - shrink) * pred + shrink * mean, pred)
            out[f"pred_{key}"] = np.clip(pred, 0.0, None)
        # calibrate the rate vector so its PPR score matches the blended ppg estimate
        rates_ppg = ppr_ppg_from_rates(out[rate_cols], pd.Series(pos, index=sub.index)).to_numpy(dtype=float)
        target = self._calibrated_ppg(pos, sub, X, rates_ppg)
        safe = np.where(rates_ppg > 0.5, rates_ppg, 1.0)
        scale = np.where(rates_ppg > 0.5, target / safe, 1.0)
        scale = np.clip(scale, 0.5, 2.0)
        out[rate_cols] = out[rate_cols].to_numpy(dtype=float) * scale[:, None]
        gm = self.games_models.get(pos)
        g = gm.predict(X) if gm is not None else np.full(len(sub), means.get("games", 14.0))
        out["pred_games"] = np.clip(g, 0.0, GAMES_PER_TEAM)
        out["rookie_flag"] = sub["rookie"].fillna(0.0).to_numpy(dtype=float) if "rookie" in sub.columns else 0.0
        # rookie override (draft-capital model)
        rk = out["rookie_flag"].to_numpy() > 0
        if rk.any() and pos in self.rookie_means:
            Xr = self._X(sub[rk], self.rookie_feature_columns)
            rm, rmeans = self.rookie_models.get(pos, {}), self.rookie_means[pos]
            for key in TARGETS[pos] + ["games"]:
                col = "pred_games" if key == "games" else f"pred_{key}"
                if key in rm:
                    v = rm[key].predict(Xr)
                elif key in rmeans:
                    v = np.full(int(rk.sum()), rmeans[key])
                else:
                    continue
                out.loc[rk, col] = np.clip(v, 0.0, GAMES_PER_TEAM if key == "games" else None)
        ppg = ppr_ppg_from_rates(out[rate_cols], pd.Series(pos, index=sub.index))
        out["pred_ppg_ppr"] = ppg.to_numpy()
        a, b = self.std_coef.get(pos, (2.0, 0.3))
        out["ppg_std_ppr"] = a + b * np.clip(out["pred_ppg_ppr"].to_numpy(dtype=float), 0, None)
        return out

    def predict(self, table: pd.DataFrame) -> pd.DataFrame:
        """Predict per-game rates for every row of ``table`` (index preserved).

        Columns: ``pred_<key>`` for every key in :data:`ALL_TARGET_KEYS` (NaN for
        keys not modelled at the row's position), ``pred_games``, ``pred_ppg_ppr``
        (PPR points per game implied by the rates), ``ppg_std_ppr`` (1-sigma of
        PPR ppg), ``rookie_flag`` and ``position``.
        """
        frames = []
        for pos in SKILL_POSITIONS:
            sub = table[table["position"] == pos]
            if sub.empty:
                continue
            if pos not in self.pos_means:
                log.warning("no model for %s; skipping %d rows", pos, len(sub))
                continue
            frames.append(self._predict_position(pos, sub))
        if not frames:
            return pd.DataFrame(columns=PRED_COLUMNS + ["position"], index=table.index[:0])
        out = pd.concat(frames)
        out = out.reindex(table.index[table.index.isin(out.index)])
        out["position"] = table.loc[out.index, "position"]
        return out

    # ------------------------------------------------------------- backtest
    @staticmethod
    def _spearman(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) < 3:
            return float("nan")
        ra = pd.Series(a).rank().to_numpy()
        rb = pd.Series(b).rank().to_numpy()
        if ra.std() == 0 or rb.std() == 0:
            return float("nan")
        return float(np.corrcoef(ra, rb)[0, 1])

    def backtest(self, table: pd.DataFrame, holdout_seasons: list[int] | None = None, min_games: int = 6) -> dict:
        """Fit on seasons before the holdout, evaluate PPR ppg on the holdout seasons.

        Returns ``{position: {n, mae_model, spearman_model, n_all, mae_model_all,
        spearman_model_all, mae_last, spearman_last, mae_career, spearman_career,
        beats_last}}``. ``mae_model`` / ``mae_last`` / ``mae_career`` are computed
        on the same rows (players with a previous season); ``*_all`` include
        rookies and players with no S-1 stats.
        """
        holdout = sorted(holdout_seasons or [DEFAULT_SEASON - 2, DEFAULT_SEASON - 1])
        train = table[table["season"] < min(holdout)]
        test = table[table["season"].isin(holdout)]
        if train.empty or test.empty:
            raise ValueError("backtest needs training seasons before the holdout and holdout rows")
        m = ProjectionModel(seed=self.seed, params=self.params).fit(train)
        pred = m.predict(test)
        test = test.loc[pred.index]
        actual = pd.to_numeric(test["y_ppg_ppr"], errors="coerce")
        ok = (test["y_games"].fillna(0) >= min_games) & actual.notna()
        results: dict = {}
        for pos in SKILL_POSITIONS:
            sel = ok & (test["position"] == pos)
            if sel.sum() < 5:
                continue
            y = actual[sel].to_numpy(dtype=float)
            p = pred.loc[sel, "pred_ppg_ppr"].to_numpy(dtype=float)
            last = pd.to_numeric(test.loc[sel, "prev_ppg_ppr"], errors="coerce")
            career = pd.to_numeric(test.loc[sel, "career_ppg_ppr"], errors="coerce")
            has = last.notna().to_numpy()
            r = {
                "n_all": int(sel.sum()),
                "mae_model_all": float(np.mean(np.abs(y - p))),
                "spearman_model_all": self._spearman(y, p),
                "n": int(has.sum()),
            }
            if has.sum() >= 5:
                yl, pl, ll = y[has], p[has], last.to_numpy(dtype=float)[has]
                cc = career.to_numpy(dtype=float)[has]
                cc = np.where(np.isnan(cc), ll, cc)
                r.update({
                    "mae_model": float(np.mean(np.abs(yl - pl))),
                    "spearman_model": self._spearman(yl, pl),
                    "mae_last": float(np.mean(np.abs(yl - ll))),
                    "spearman_last": self._spearman(yl, ll),
                    "mae_career": float(np.mean(np.abs(yl - cc))),
                    "spearman_career": self._spearman(yl, cc),
                })
                r["beats_last"] = bool(r["mae_model"] < r["mae_last"])
            results[pos] = r
        results["_holdout"] = holdout
        results["_train_rows"] = int(len(train))
        results["_backtest_fit_seconds"] = m.meta.get("fit_seconds")
        return results

    # ------------------------------------------------------------ persist
    def save(self, path: Path | None = None) -> Path:
        p = Path(path) if path is not None else models_dir() / "projection_model.pkl"
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(p)
        log.info("saved projection model to %s", p)
        return p

    @classmethod
    def load(cls, path: Path | None = None) -> "ProjectionModel":
        p = Path(path) if path is not None else models_dir() / "projection_model.pkl"
        with open(p, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"{p} does not contain a ProjectionModel")
        return obj


# ---------------------------------------------------------------------------
# End-to-end training
# ---------------------------------------------------------------------------

def load_training_inputs(seasons: Iterable[int], refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, pd.DataFrame]]:
    """Load canonical data, aggregates and rosters for ``seasons`` (network-free when cached).

    Returns ``(season_aggregates, team_context, {season: roster})``."""
    from ..data.crosswalk import build_crosswalk
    from ..data.nflverse import load_canonical, load_injuries, load_roster, load_snap_counts, team_context

    seasons = sorted(int(s) for s in seasons)
    canonical = load_canonical(seasons, refresh=refresh)
    pfr_to_gsis: dict[str, str] = {}
    try:
        pfr_to_gsis = build_crosswalk(DEFAULT_SEASON).pfr_to_gsis
    except Exception as e:  # noqa: BLE001
        log.warning("crosswalk unavailable (snap counts skipped): %s", e)
    snaps, injuries = [], []
    for s in seasons:
        try:
            snaps.append(load_snap_counts(s))
        except Exception as e:  # noqa: BLE001
            log.warning("snap counts %s unavailable: %s", s, e)
        try:
            injuries.append(load_injuries(s))
        except Exception as e:  # noqa: BLE001
            log.warning("injuries %s unavailable: %s", s, e)
    agg = season_aggregates(
        canonical,
        pd.concat(snaps, ignore_index=True) if snaps else None,
        pd.concat(injuries, ignore_index=True) if injuries else None,
        pfr_to_gsis,
    )
    rosters: dict[int, pd.DataFrame] = {}
    for s in seasons:
        try:
            rosters[s] = load_roster(s)
        except Exception as e:  # noqa: BLE001
            log.warning("roster %s unavailable: %s", s, e)
    return agg, team_context(canonical), rosters


def train_and_save(seasons: Iterable[int] = TRAIN_SEASONS, refresh: bool = False,
                   holdout_seasons: list[int] | None = None, path: Path | None = None,
                   backtest: bool = True) -> tuple[ProjectionModel, dict]:
    """Build the training table, backtest, fit on everything and save the model.

    Training rows exist only for seasons whose previous season is loaded (the
    first season is history only). Returns ``(model, metrics)`` where metrics is
    the :meth:`ProjectionModel.backtest` dict plus timing info (``_*`` keys).
    """
    t0 = time.perf_counter()
    seasons = sorted(int(s) for s in seasons)
    agg, team_ctx, rosters = load_training_inputs(seasons, refresh=refresh)
    train_seasons = [s for s in seasons if s - 1 in seasons and s in rosters]
    table = build_training_table(agg, team_ctx, rosters, train_seasons)
    log.info("training table: %d rows, %d features (%.1fs)", len(table), len(FEATURE_COLUMNS), time.perf_counter() - t0)
    model = ProjectionModel()
    holdout = holdout_seasons or [s for s in train_seasons if s >= max(train_seasons) - 1]
    metrics: dict = {}
    if backtest and holdout and min(holdout) > min(train_seasons):
        try:
            metrics = model.backtest(table, holdout)
        except Exception as e:  # noqa: BLE001
            log.warning("backtest failed: %s", e)
    model.fit(table)
    metrics["_train_seasons"] = train_seasons
    metrics["_fit_seconds"] = model.meta.get("fit_seconds")
    metrics["_total_seconds"] = time.perf_counter() - t0
    metrics["_path"] = str(model.save(path))
    return model, metrics
