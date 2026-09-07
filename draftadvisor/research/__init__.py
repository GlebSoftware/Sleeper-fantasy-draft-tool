"""Optional Claude (Sonnet) research layer. See DESIGN.md §3.5."""
from .claude import (
    UNAVAILABLE_SUMMARY,
    ClaudeResearcher,
    build_context_text,
    describe_roster_slots,
    describe_scoring,
)

__all__ = [
    "ClaudeResearcher",
    "build_context_text",
    "describe_roster_slots",
    "describe_scoring",
    "UNAVAILABLE_SUMMARY",
]
