"""rich Live dashboard. See DESIGN.md §3.6. STUB - "ui" agent."""
from __future__ import annotations

from rich.console import Console, RenderableType

from ..models import DraftState, Player, Recommendation


class Dashboard:
    def __init__(self, console: Console | None = None, refresh_per_second: float = 4): raise NotImplementedError
    def build(self, state: DraftState, rec: Recommendation | None, players: dict[str, Player], status: dict) -> RenderableType: raise NotImplementedError
    def start(self) -> None: raise NotImplementedError
    def stop(self) -> None: raise NotImplementedError
    def update(self, state: DraftState, rec: Recommendation | None, players: dict[str, Player], status: dict) -> None: raise NotImplementedError


def render_text(rec: Recommendation | None, state: DraftState, players: dict[str, Player]) -> str: raise NotImplementedError
