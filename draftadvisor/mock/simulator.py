"""Mock draft simulator. See DESIGN.md §3.2. STUB - "sleeper" agent."""
from __future__ import annotations

from typing import Callable

from ..models import DraftSettings, DraftState, LeagueSettings, Pick, Player


def make_mock_league(teams: int = 12, rounds: int = 15, scoring: str = "half_ppr", superflex: bool = False,
                     te_premium: float = 0.0) -> LeagueSettings: raise NotImplementedError
def make_mock_draft(league: LeagueSettings, my_slot: int, teams: int = 12, rounds: int = 15, pick_timer: int = 30,
                    reversal_round: int = 0) -> DraftSettings: raise NotImplementedError


class MockDraft:
    def __init__(self, players: dict[str, Player], league: LeagueSettings, draft: DraftSettings, my_slot: int,
                 seed: int | None = None, bot_noise: float = 0.15):
        raise NotImplementedError

    def state(self) -> DraftState: raise NotImplementedError
    @property
    def is_my_turn(self) -> bool: raise NotImplementedError
    @property
    def is_complete(self) -> bool: raise NotImplementedError
    def bot_pick(self) -> Pick: raise NotImplementedError
    def make_pick(self, player_id: str) -> Pick: raise NotImplementedError
    def advance_until_my_turn(self) -> DraftState: raise NotImplementedError
    def run_to_completion(self, my_policy: Callable[[DraftState], str]) -> DraftState: raise NotImplementedError
