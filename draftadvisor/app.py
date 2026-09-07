"""Application context / orchestration. See DESIGN.md §3.6. STUB - "ui" agent."""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import Settings


@dataclass
class AppContext:
    settings: Settings
    engine: object
    league: object
    draft: object
    players: dict
    projections: dict
    advisor: object
    researcher: object
    notes: dict
    byes: dict
    crosswalk: object
    sources: dict = field(default_factory=dict)


def build_context(settings: Settings, league=None, draft=None, *, offline: bool = False, sleeper_players=None,
                  sleeper_proj=None, refresh: bool = False, use_model: bool = True, quiet: bool = False) -> AppContext:
    raise NotImplementedError


async def prep(settings: Settings, research: bool = False, refresh: bool = False, train: bool = True) -> AppContext:
    raise NotImplementedError
