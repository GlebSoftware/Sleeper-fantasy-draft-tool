"""recommend - see DESIGN.md §3.4. STUB - "strategy" agent."""
from __future__ import annotations

from ..config import Settings
from ..models import DraftState, LeagueSettings, Player, PlayerValue, Projection, Recommendation


class Advisor:
    def __init__(self, league: LeagueSettings, players: dict[str, Player], projections: dict[str, Projection],
                 settings: Settings | None = None): raise NotImplementedError
    def recommend(self, state: DraftState, top_n: int = 6, per_position: int = 3) -> Recommendation: raise NotImplementedError
    def explain_pick(self, state: DraftState, player_id: str) -> str: raise NotImplementedError
    def available_players(self, state: DraftState) -> list[Player]: raise NotImplementedError
    def value_of(self, state: DraftState, player_id: str) -> PlayerValue | None: raise NotImplementedError
