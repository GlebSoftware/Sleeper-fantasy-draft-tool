"""Sleeper JSON -> models. See DESIGN.md §3.1. STUB - "sleeper" agent."""
from __future__ import annotations

from ..models import DraftSettings, DraftState, LeagueSettings, Manager, Pick


def parse_league(raw: dict) -> LeagueSettings: raise NotImplementedError
def parse_draft(raw: dict, traded_picks: list[dict] | None = None) -> DraftSettings: raise NotImplementedError
def parse_pick(raw: dict) -> Pick: raise NotImplementedError
def parse_picks(raw: list[dict]) -> list[Pick]: raise NotImplementedError
def parse_managers(users: list[dict], draft: DraftSettings, rosters: list[dict] | None = None) -> dict[str, Manager]: raise NotImplementedError
def adp_key_for(league: LeagueSettings | None, draft: DraftSettings | None) -> str: raise NotImplementedError
def adp_map(projections: dict[str, dict], key: str) -> dict[str, float]: raise NotImplementedError
def resolve_my_slot(draft: DraftSettings, managers: dict[str, Manager], *, username: str | None = None,
                    user_id: str | None = None, slot: int | None = None) -> tuple[str | None, int | None]: raise NotImplementedError
def state_from_sleeper(draft_raw: dict, picks_raw: list[dict], league_raw: dict | None = None,
                       users_raw: list[dict] | None = None, rosters_raw: list[dict] | None = None,
                       traded_raw: list[dict] | None = None, *, username: str | None = None,
                       user_id: str | None = None, slot: int | None = None) -> DraftState: raise NotImplementedError
