"""Claude research + on-the-clock advice. See DESIGN.md §3.5. STUB - "research" agent."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..config import CLAUDE_MODEL, CLAUDE_ON_CLOCK_TIMEOUT_S
from ..models import DraftState, Player, Projection, Recommendation, ResearchNote


class ClaudeResearcher:
    def __init__(self, api_key: str | None = None, model: str = CLAUDE_MODEL, cache_dir: Path | None = None, client=None): raise NotImplementedError
    @property
    def enabled(self) -> bool: raise NotImplementedError
    def load_notes(self) -> dict[str, ResearchNote]: raise NotImplementedError
    async def research_players(self, players: list[Player], projections: dict[str, Projection] | None = None,
                               max_age_days: float = 3, concurrency: int = 4,
                               progress: Callable[[int, int, str], None] | None = None) -> dict[str, ResearchNote]: raise NotImplementedError
    async def on_the_clock_advice(self, state: DraftState, rec: Recommendation, players: dict[str, Player],
                                  notes: dict[str, ResearchNote], timeout: float = CLAUDE_ON_CLOCK_TIMEOUT_S) -> str | None: raise NotImplementedError
    async def ask(self, question: str, context_text: str) -> str: raise NotImplementedError


def build_context_text(state: DraftState, rec: Recommendation | None, players: dict[str, Player],
                       notes: dict[str, ResearchNote] | None) -> str: raise NotImplementedError
