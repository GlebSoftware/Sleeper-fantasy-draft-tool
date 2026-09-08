"""One-time ESPN league capture -> :class:`draftadvisor.capture.LeagueSnapshot` (+ player helpers).

The snapshot has the same shape as the Sleeper one so the CLI / web layers can render either:
``raw`` carries ``{"platform": "espn", "league", "draft", "players", "rosters"}`` where ``rosters``
is ``[{"roster_id": team id, "players": [canonical ids]}]`` so ``LeagueSnapshot.roster_players`` works.

Helpers for the web / CLI layers: :func:`espn_names` (id -> name / position / team),
:func:`espn_adp` (id -> ESPN average draft position) and :func:`espn_projections`
(id -> Sleeper-key projected season stats) from the ``kona_player_info`` list. ADP and projections
cover the skill positions only (IDP, punters and head coaches of an ESPN pool are named but never
ranked or projected).
"""
from __future__ import annotations

import logging
import time
from typing import Any, Iterable, Mapping

from ..capture import LeagueSnapshot, scoring_diff, strategy_flags
from ..config import SKILL_POSITIONS
from .client import EspnAPIError, EspnClient
from .ids import EspnIdMap, espn_player_fields
from .parsing import (
    PLATFORM,
    parse_espn_draft,
    parse_espn_league,
    parse_espn_managers,
    parse_espn_picks,
    resolve_my_team,
    roster_names,
)
from .scoring import espn_stats_to_sleeper, projected_season_stats

log = logging.getLogger(__name__)

__all__ = [
    "capture_espn_league",
    "espn_names",
    "espn_adp",
    "espn_projections",
    "espn_rosters",
    "unmapped_flags",
]


def _float(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


# ---------------------------------------------------------------------------
# kona_player_info helpers
# ---------------------------------------------------------------------------


def espn_names(players_list: Iterable[Mapping[str, Any]] | None) -> dict[str, dict[str, Any]]:
    """``{espn id: {name, first_name, last_name, position, team, pro_team_id, injury_status, ...}}``
    from a ``kona_player_info`` list (or any list of player / roster entries)."""
    out: dict[str, dict[str, Any]] = {}
    for entry in players_list or []:
        f = espn_player_fields(entry)
        if f["espn_id"] is not None and f["name"]:
            out[f["espn_id"]] = f
    return out


def espn_adp(players_list: Iterable[Mapping[str, Any]] | None, id_map: EspnIdMap,
             rank_type: str = "PPR") -> dict[str, float]:
    """``{canonical id: ADP}`` from ``player.ownership.averageDraftPosition`` (> 0), falling back to
    ESPN's ``draftRanksByRankType[rank_type].rank`` (then ``STANDARD``) when a player has no ADP;
    skill positions only."""
    out: dict[str, float] = {}
    for entry in players_list or []:
        if not isinstance(entry, Mapping) or espn_player_fields(entry)["position"] not in SKILL_POSITIONS:
            continue
        pl = entry.get("player") if isinstance(entry.get("player"), Mapping) else entry
        adp = _float((pl.get("ownership") or {}).get("averageDraftPosition")) if isinstance(pl.get("ownership"), Mapping) else None
        if adp is None or adp <= 0:
            ranks = pl.get("draftRanksByRankType") if isinstance(pl.get("draftRanksByRankType"), Mapping) else {}
            for rt in (rank_type, "PPR", "STANDARD"):
                r = ranks.get(rt) if isinstance(ranks, Mapping) else None
                adp = _float(r.get("rank")) if isinstance(r, Mapping) else None
                if adp is not None and adp > 0:
                    break
        if adp is None or adp <= 0:
            continue
        out[id_map.resolve_player_json(entry)] = float(adp)
    return out


def espn_projections(players_list: Iterable[Mapping[str, Any]] | None, id_map: EspnIdMap,
                     season: int) -> dict[str, dict[str, float]]:
    """``{canonical id: Sleeper-key projected season stats}`` (incl. ``gp``) from the ``10<season>``
    projected entry of each skill-position player; players without a projection are skipped."""
    out: dict[str, dict[str, float]] = {}
    for entry in players_list or []:
        if not isinstance(entry, Mapping):
            continue
        f = espn_player_fields(entry)
        if f["position"] not in SKILL_POSITIONS:
            continue
        raw = projected_season_stats(entry, season)
        if not raw:
            continue
        line = espn_stats_to_sleeper(raw, f["position"])
        if line:
            out[id_map.resolve_player_json(entry)] = line
    return out


def espn_rosters(league_json: Mapping[str, Any] | None, id_map: EspnIdMap) -> list[dict]:
    """``[{"roster_id": team id, "players": [canonical ids]}]`` from ``teams[].roster.entries``."""
    out: list[dict] = []
    for team in (league_json or {}).get("teams") or []:
        if not isinstance(team, Mapping) or team.get("id") is None:
            continue
        players: list[str] = []
        for e in ((team.get("roster") or {}).get("entries") or []):
            if isinstance(e, Mapping):
                players.append(id_map.resolve_player_json(e))
        try:
            rid = int(team["id"])
        except (TypeError, ValueError):
            continue
        out.append({"roster_id": rid, "players": players})
    return out


def unmapped_flags(unmapped: Iterable[Mapping[str, Any]] | None) -> list[str]:
    """One strategy flag per ESPN scoring rule the model ignores."""
    out: list[str] = []
    for u in unmapped or []:
        pts = _float(u.get("points"))
        out.append(f"ESPN rule not modelled: {u.get('label')} ({pts:g})" if pts is not None
                   else f"ESPN rule not modelled: {u.get('label')}")
    return out


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


async def capture_espn_league(client: EspnClient, league_id: str, season: int, *, swid: str | None = None,
                              team_id: int | str | None = None, slot: int | None = None,
                              username: str | None = None, id_map: EspnIdMap | None = None,
                              players_limit: int = 600, save: bool = False) -> LeagueSnapshot:
    """Fetch settings / teams / rosters, the draft and the player pool of one ESPN league.

    Raises :class:`ValueError` (listing the league's teams) when ``username`` / ``team_id`` matches no
    team, or when ``slot`` is outside ``1..teams`` and nothing else identifies the team; an unmatched
    ``swid`` only means spectator mode. ``id_map`` resolves ESPN ids to our universe (an empty map
    leaves every id synthetic).
    """
    id_map = id_map or EspnIdMap()
    league_json = await client.get_settings_and_teams(league_id, season)
    draft_json = await client.get_draft_detail(league_id, season)
    players: list[dict] = []
    try:
        players = await client.get_players(league_id, season, limit=players_limit)
    except EspnAPIError as e:
        log.warning("ESPN player pool unavailable (%s); continuing without ADP / projections", e)

    names: dict[str, Mapping[str, Any]] = dict(roster_names(league_json))
    names.update(espn_names(players))
    league = parse_espn_league(league_json, draft_json)
    draft = parse_espn_draft(league_json, draft_json)
    managers = parse_espn_managers(league_json, draft)
    picks = parse_espn_picks(draft_json, id_map, draft, names)
    my_uid, my_slot = resolve_my_team(draft, managers, league_json, swid=swid, team_id=team_id, slot=slot,
                                      username=username)
    if (username or team_id is not None) and my_uid is None:
        teams = ", ".join(sorted(f"{m.team_name or m.display_name} (team {m.user_id}, {m.display_name})"
                                 for m in managers.values()))
        raise ValueError(f"could not find {username or team_id!r} among the ESPN league's teams: {teams}")
    if slot is not None and my_uid is None and my_slot is None:
        raise ValueError(f"slot {slot!r} is not a draft slot of this ESPN league (1..{draft.teams})")
    if my_uid is not None and my_slot is None:
        log.info("draft order not set yet for ESPN league %s; slot for team %s resolves when the commissioner sets it",
                 league_id, my_uid)
    my_roster = draft.original_roster_for_slot(my_slot) if my_slot is not None else None
    if my_roster is None and my_uid and my_uid.isdigit():
        my_roster = int(my_uid)                     # the team is known even while the order is not
    my_picks = draft.picks_for_slot(my_slot) if my_slot is not None else []
    flags = strategy_flags(league, draft) + [f"ESPN league (platform: {PLATFORM})"]
    flags += unmapped_flags(league.settings.get("unmapped_scoring"))
    snap = LeagueSnapshot(
        captured_at=time.time(),
        season=int(league.season or season),
        league=league, draft=draft, managers=managers,
        keepers=[p for p in picks if p.is_keeper], picks_made=len(picks),
        my_user_id=my_uid, my_slot=my_slot, my_roster_id=my_roster, my_picks=my_picks,
        diff=scoring_diff(league.scoring_settings),
        flags=flags,
        nfl_state={},
        raw={"platform": PLATFORM, "league": league_json, "draft": draft_json, "players": players,
             "rosters": espn_rosters(league_json, id_map)},
    )
    if save:
        snap.save()
    return snap
