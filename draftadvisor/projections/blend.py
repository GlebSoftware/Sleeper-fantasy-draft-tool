"""Projection blending. See DESIGN.md §3.3. STUB - "projections" agent."""
from __future__ import annotations

import pandas as pd

from ..config import FANTASY_WEEKS, Settings
from ..data.crosswalk import Crosswalk
from ..models import LeagueSettings, Player, Projection, ResearchNote
from ..scoring import ScoringEngine

DEFAULT_WEIGHTS = {"sleeper": 0.45, "ml": 0.35, "ecr": 0.20}


def rates_to_season(rates: dict[str, float], games: float, engine: ScoringEngine, position: str) -> tuple[float, float, dict]: raise NotImplementedError
def sleeper_stats_to_projection(stats: dict, engine: ScoringEngine, position: str) -> tuple[float, float, dict]: raise NotImplementedError
def ecr_implied_points(players: dict[str, Player], provisional: dict[str, float], position: str) -> dict[str, float]: raise NotImplementedError


class Projector:
    def __init__(self, engine: ScoringEngine, league: LeagueSettings | None = None, settings: Settings | None = None,
                 weights: dict | None = None): raise NotImplementedError
    def project(self, players: dict[str, Player], ml_pred: pd.DataFrame | None = None,
                sleeper_proj: dict[str, dict] | None = None, notes: dict[str, ResearchNote] | None = None,
                byes: dict[str, int] | None = None, season_weeks: int = FANTASY_WEEKS) -> dict[str, Projection]: raise NotImplementedError


def project_offline(players: dict[str, Player], engine: ScoringEngine, canonical: pd.DataFrame, cw: Crosswalk,
                    season: int) -> pd.DataFrame: raise NotImplementedError
