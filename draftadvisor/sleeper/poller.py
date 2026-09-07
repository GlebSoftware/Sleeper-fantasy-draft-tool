"""Draft poller. See DESIGN.md §3.1. STUB - "sleeper" agent."""
from __future__ import annotations

import asyncio
from typing import Any, Callable

from ..config import Settings
from ..models import DraftState
from .client import SleeperClient


class DraftPoller:
    def __init__(self, client: SleeperClient, draft_id: str, *, league_id: str | None = None,
                 settings: Settings | None = None, username: str | None = None, user_id: str | None = None,
                 slot: int | None = None):
        raise NotImplementedError

    state: DraftState | None
    last_error: str | None
    poll_count: int
    last_latency_ms: float
    last_poll_at: float | None

    async def bootstrap(self) -> DraftState: raise NotImplementedError
    async def poll_once(self) -> DraftState | None: raise NotImplementedError
    async def run(self, on_update: Callable[[DraftState], Any], stop: asyncio.Event | None = None) -> DraftState: raise NotImplementedError
