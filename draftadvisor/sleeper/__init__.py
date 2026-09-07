"""Sleeper API access: async client, JSON -> model parsing, draft poller. See DESIGN.md §3.1.

Convenience re-exports::

    from draftadvisor.sleeper import SleeperClient, DraftPoller, state_from_sleeper
"""
from __future__ import annotations

from .client import SleeperAPIError, SleeperClient, SleeperNotFound, run_sync
from .parsing import (
    adp_key_for,
    adp_map,
    parse_draft,
    parse_league,
    parse_managers,
    parse_pick,
    parse_picks,
    resolve_my_slot,
    state_from_sleeper,
)
from .poller import DraftPoller

__all__ = [
    "SleeperAPIError",
    "SleeperClient",
    "SleeperNotFound",
    "run_sync",
    "DraftPoller",
    "adp_key_for",
    "adp_map",
    "parse_draft",
    "parse_league",
    "parse_managers",
    "parse_pick",
    "parse_picks",
    "resolve_my_slot",
    "state_from_sleeper",
]
