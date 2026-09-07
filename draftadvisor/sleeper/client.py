"""Async Sleeper API client. See DESIGN.md §3.1 for the full contract.

STUB - to be implemented by the "sleeper" agent.
"""
from __future__ import annotations

import asyncio
from typing import Any

import httpx

from ..config import SLEEPER_API_BASE, SLEEPER_API_BASE_V2, SKILL_POSITIONS


class SleeperAPIError(Exception):
    def __init__(self, message: str, status_code: int | None = None, url: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.url = url


class SleeperNotFound(SleeperAPIError):
    pass


class SleeperClient:
    def __init__(self, timeout: float = 10.0, retries: int = 3, base_url: str = SLEEPER_API_BASE,
                 base_url_v2: str = SLEEPER_API_BASE_V2, http: httpx.AsyncClient | None = None):
        raise NotImplementedError

    async def aclose(self) -> None: raise NotImplementedError
    async def get_json(self, path: str, *, base: str | None = None, params: dict | None = None) -> Any: raise NotImplementedError
    async def get_state(self, sport: str = "nfl") -> dict: raise NotImplementedError
    async def get_user(self, username_or_id: str) -> dict: raise NotImplementedError
    async def get_user_leagues(self, user_id: str, season: int, sport: str = "nfl") -> list[dict]: raise NotImplementedError
    async def get_user_drafts(self, user_id: str, season: int, sport: str = "nfl") -> list[dict]: raise NotImplementedError
    async def get_league(self, league_id: str) -> dict: raise NotImplementedError
    async def get_league_users(self, league_id: str) -> list[dict]: raise NotImplementedError
    async def get_league_rosters(self, league_id: str) -> list[dict]: raise NotImplementedError
    async def get_league_drafts(self, league_id: str) -> list[dict]: raise NotImplementedError
    async def get_draft(self, draft_id: str) -> dict: raise NotImplementedError
    async def get_draft_picks(self, draft_id: str) -> list[dict]: raise NotImplementedError
    async def get_traded_picks(self, draft_id: str) -> list[dict]: raise NotImplementedError
    async def get_players(self, force_refresh: bool = False) -> dict[str, dict]: raise NotImplementedError
    async def get_trending(self, add_drop: str = "add", hours: int = 24, limit: int = 25) -> list[dict]: raise NotImplementedError
    async def get_season_projections(self, season: int, season_type: str = "regular",
                                     positions: tuple[str, ...] = SKILL_POSITIONS) -> dict[str, dict]: raise NotImplementedError
    async def get_week_projections(self, season: int, week: int, season_type: str = "regular") -> dict[str, dict]: raise NotImplementedError
    async def get_season_stats(self, season: int, season_type: str = "regular") -> dict[str, dict]: raise NotImplementedError
    async def get_week_stats(self, season: int, week: int, season_type: str = "regular") -> dict[str, dict]: raise NotImplementedError


def run_sync(coro):
    """Run a coroutine from synchronous code (CLI one-offs)."""
    return asyncio.run(coro)
