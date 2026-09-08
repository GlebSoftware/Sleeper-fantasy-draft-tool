"""Optional Claude chat layer (user-initiated only). See DESIGN.md §3.5."""
from .claude import (
    MODEL_PRICES_USD_PER_MTOK,
    UNAVAILABLE_SUMMARY,
    ClaudeChat,
    ClaudeResearcher,
    build_context_text,
    describe_roster_slots,
    describe_scoring,
    estimate_cost_usd,
)

__all__ = [
    "ClaudeChat",
    "ClaudeResearcher",
    "MODEL_PRICES_USD_PER_MTOK",
    "build_context_text",
    "describe_roster_slots",
    "describe_scoring",
    "estimate_cost_usd",
    "UNAVAILABLE_SUMMARY",
]
