"""ESPN fantasy football league provider: async client, JSON -> model parsing, id mapping,
scoring translation, league capture and draft poller (pandas-free; mirrors :mod:`draftadvisor.sleeper`).

Convenience re-exports::

    from draftadvisor.espn import EspnClient, EspnDraftPoller, EspnIdMap, capture_espn_league, state_from_espn
"""
from __future__ import annotations

from .capture import capture_espn_league, espn_adp, espn_names, espn_projections, espn_rosters, unmapped_flags
from .client import (
    DRAFT_VIEWS,
    LEAGUE_VIEWS,
    EspnAccessDenied,
    EspnAPIError,
    EspnClient,
    EspnNotFound,
    format_swid,
    parse_fan_leagues,
    player_filter,
    redact_url,
)
from .constants import (
    DEFAULT_POSITION_MAP,
    ESPN_BASE_URL,
    ESPN_STAT_TO_SLEEPER,
    FAN_API_URL,
    POSITION_SLOT_MAP,
    PRO_TEAM_MAP,
    SCORING_LABELS,
    espn_base_url,
    label_for_stat,
    slot_label,
)
from .ids import EspnIdMap, espn_player_fields, espn_position, placeholder_player
from .parsing import (
    draft_status,
    parse_espn_draft,
    parse_espn_league,
    parse_espn_managers,
    parse_espn_picks,
    resolve_my_team,
    roster_names,
    roster_positions_from_counts,
    rostered_espn_ids,
    rostered_ids,
    rounds_from_counts,
    state_from_espn,
    traded_picks_from_detail,
)
from .poller import EspnDraftPoller
from .scoring import espn_scoring_to_sleeper, espn_stats_to_sleeper, projected_points, projected_season_stats

__all__ = [
    "DEFAULT_POSITION_MAP", "DRAFT_VIEWS", "ESPN_BASE_URL", "ESPN_STAT_TO_SLEEPER", "FAN_API_URL", "LEAGUE_VIEWS",
    "POSITION_SLOT_MAP", "PRO_TEAM_MAP", "SCORING_LABELS",
    "EspnAPIError", "EspnAccessDenied", "EspnClient", "EspnDraftPoller", "EspnIdMap", "EspnNotFound",
    "capture_espn_league", "draft_status", "espn_adp", "espn_base_url", "espn_names", "espn_player_fields",
    "espn_position", "espn_projections", "espn_rosters", "espn_scoring_to_sleeper", "espn_stats_to_sleeper",
    "format_swid", "label_for_stat", "parse_espn_draft", "parse_espn_league", "parse_espn_managers",
    "parse_espn_picks", "parse_fan_leagues", "placeholder_player", "player_filter", "projected_points",
    "projected_season_stats", "redact_url", "resolve_my_team", "roster_names", "roster_positions_from_counts",
    "rostered_espn_ids", "rostered_ids", "rounds_from_counts", "slot_label", "state_from_espn",
    "traded_picks_from_detail", "unmapped_flags",
]
