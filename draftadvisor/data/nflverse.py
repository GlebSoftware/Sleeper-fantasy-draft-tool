"""nflverse loaders (GitHub release assets) with disk caching.

All loaders return pandas frames. Past seasons are immutable and downloaded once;
current-season files (rosters, schedule, players) are refreshed daily.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..config import DEFAULT_SEASON, TRAIN_SEASONS
from .cache import cached_frame, download, raw_path
from .canonical import player_weekly_to_canonical, team_weekly_to_canonical, to_sleeper_team

log = logging.getLogger(__name__)

RELEASE = "https://github.com/nflverse/nflverse-data/releases/download"


def _player_stats_url(season: int) -> list[str]:
    # nflverse moved weekly player stats to the ``stats_player`` release tag in 2025;
    # older seasons exist under both tags. Try new tag first.
    return [
        f"{RELEASE}/stats_player/stats_player_week_{season}.csv",
        f"{RELEASE}/player_stats/stats_player_week_{season}.csv",
    ]


def _team_stats_url(season: int) -> list[str]:
    return [
        f"{RELEASE}/stats_team/stats_team_week_{season}.csv",
        f"{RELEASE}/player_stats/stats_team_week_{season}.csv",
    ]


def _fetch_first(urls: list[str], dest: Path, max_age_hours: float | None) -> Path:
    last = None
    for u in urls:
        try:
            return download(u, dest, max_age_hours=max_age_hours)
        except Exception as e:  # noqa: BLE001
            last = e
            log.info("not available: %s (%s)", u, e)
    if dest.exists():
        return dest
    raise RuntimeError(f"could not download any of {urls}: {last}")


def _age(season: int) -> float | None:
    """Files for finished seasons never change; the current season refreshes daily."""
    return 24.0 if season >= DEFAULT_SEASON - 1 else None


# ---------------------------------------------------------------------------
# Raw frames
# ---------------------------------------------------------------------------

def load_player_weekly_raw(season: int) -> pd.DataFrame:
    p = _fetch_first(_player_stats_url(season), raw_path(f"stats_player_week_{season}.csv"), _age(season))
    return pd.read_csv(p, low_memory=False)


def load_team_weekly_raw(season: int) -> pd.DataFrame:
    p = _fetch_first(_team_stats_url(season), raw_path(f"stats_team_week_{season}.csv"), _age(season))
    return pd.read_csv(p, low_memory=False)


def load_schedule() -> pd.DataFrame:
    """All games (past scores + upcoming schedule) from nflverse ``schedules``."""
    p = download(f"{RELEASE}/schedules/games.csv", raw_path("games.csv"), max_age_hours=24.0)
    return pd.read_csv(p, low_memory=False)


def load_roster(season: int) -> pd.DataFrame:
    """Season roster with id crosswalk columns (gsis_id, sleeper_id, pfr_id, ...)."""
    p = download(f"{RELEASE}/rosters/roster_{season}.csv", raw_path(f"roster_{season}.csv"),
                 max_age_hours=24.0 if season >= DEFAULT_SEASON else None)
    df = pd.read_csv(p, low_memory=False)
    df["team"] = df["team"].map(to_sleeper_team)
    return df


def load_players_master() -> pd.DataFrame:
    p = download(f"{RELEASE}/players/players.csv", raw_path("players.csv"), max_age_hours=24.0 * 7)
    return pd.read_csv(p, low_memory=False)


def load_snap_counts(season: int) -> pd.DataFrame:
    p = download(f"{RELEASE}/snap_counts/snap_counts_{season}.csv", raw_path(f"snap_counts_{season}.csv"), _age(season))
    df = pd.read_csv(p, low_memory=False)
    df = df[df["game_type"] == "REG"] if "game_type" in df.columns else df
    df = df.rename(columns={"pfr_player_id": "pfr_id"})
    df["team"] = df["team"].map(to_sleeper_team)
    return df[["season", "week", "pfr_id", "player", "position", "team", "offense_snaps", "offense_pct"]]


def load_injuries(season: int) -> pd.DataFrame:
    p = download(f"{RELEASE}/injuries/injuries_{season}.csv", raw_path(f"injuries_{season}.csv"), _age(season))
    df = pd.read_csv(p, low_memory=False)
    keep = [c for c in ("season", "week", "gsis_id", "team", "position", "report_status", "practice_status",
                        "report_primary_injury") if c in df.columns]
    return df[keep]


# ---------------------------------------------------------------------------
# Canonical frames (cached)
# ---------------------------------------------------------------------------

def load_canonical_season(season: int, refresh: bool = False) -> pd.DataFrame:
    """Per-game canonical rows (players + team defenses) for one season."""
    def build() -> pd.DataFrame:
        players = player_weekly_to_canonical(load_player_weekly_raw(season))
        try:
            teams = team_weekly_to_canonical(load_team_weekly_raw(season), load_schedule())
        except Exception as e:  # noqa: BLE001
            log.warning("team defense stats unavailable for %s: %s", season, e)
            teams = pd.DataFrame()
        return pd.concat([players, teams], ignore_index=True, sort=False)
    return cached_frame(f"canonical_{season}", build, refresh=refresh, max_age_hours=_age(season))


def load_canonical(seasons: Iterable[int] = TRAIN_SEASONS, refresh: bool = False) -> pd.DataFrame:
    frames = []
    for s in seasons:
        try:
            frames.append(load_canonical_season(s, refresh=refresh))
        except Exception as e:  # noqa: BLE001
            log.warning("season %s unavailable: %s", s, e)
    if not frames:
        raise RuntimeError("no historical seasons could be loaded")
    return pd.concat(frames, ignore_index=True, sort=False)


def bye_weeks(season: int = DEFAULT_SEASON, schedule: pd.DataFrame | None = None) -> dict[str, int]:
    """Team (Sleeper abbr) -> bye week for ``season`` (empty dict if the schedule is unknown)."""
    g = schedule if schedule is not None else load_schedule()
    g = g[(pd.to_numeric(g["season"]) == season) & (g["game_type"] == "REG")]
    if g.empty:
        return {}
    weeks = set(pd.to_numeric(g["week"]).astype(int))
    out: dict[str, int] = {}
    for t in set(g["home_team"]) | set(g["away_team"]):
        played = set(pd.to_numeric(g[(g["home_team"] == t) | (g["away_team"] == t)]["week"]).astype(int))
        missing = sorted(weeks - played)
        if missing:
            out[to_sleeper_team(t)] = missing[0]
    return out


def team_context(canonical: pd.DataFrame) -> pd.DataFrame:
    """Per (team, season) offensive volume: pass attempts / rush attempts / targets per game."""
    off = canonical[canonical["position"].isin(["QB", "RB", "WR", "TE"])]
    g = off.groupby(["team", "season"]).agg(
        team_pass_att=("pass_att", "sum"), team_rush_att=("rush_att", "sum"), team_targets=("rec_tgt", "sum"),
        team_rec_yd=("rec_yd", "sum"), team_rush_yd=("rush_yd", "sum"),
        team_games=("week", "nunique"),
    ).reset_index()
    for c in ("team_pass_att", "team_rush_att", "team_targets", "team_rec_yd", "team_rush_yd"):
        g[c + "_pg"] = g[c] / g["team_games"].clip(lower=1)
    return g
