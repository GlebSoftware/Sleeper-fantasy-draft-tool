"""Replacement levels, VORP and gap-based tiers (DESIGN.md §3.4).

Replacement level is the projected points of the player one past the league's
*demand* at a position.  Demand counts every dedicated starting slot, the share of
each flex slot that usually goes to the position, and a bench allowance.  When the
levels are computed on the *available* pool (what the advisor does every tick)
they drop as the draft goes on, so VORP stays meaningful late in the draft.
"""
from __future__ import annotations

import logging
from typing import Iterable, Mapping, Sequence

from ..config import SKILL_POSITIONS, SLOT_ELIGIBILITY
from ..models import LeagueSettings, Player, Projection

log = logging.getLogger(__name__)

__all__ = ["FLEX_SHARES", "starter_demand", "replacement_levels", "vorp", "tiers"]

#: How each multi-position starting slot is typically filled, by position.
FLEX_SHARES: dict[str, dict[str, float]] = {
    "FLEX": {"RB": 0.45, "WR": 0.45, "TE": 0.10},
    "SUPER_FLEX": {"QB": 0.85, "RB": 0.05, "WR": 0.05, "TE": 0.05},
    "WRRB_FLEX": {"RB": 0.5, "WR": 0.5},
    "REC_FLEX": {"WR": 0.75, "TE": 0.25},
}


def _flex_shares(slot: str) -> dict[str, float]:
    """Shares for ``slot``; unknown multi-position slots split evenly."""
    if slot in FLEX_SHARES:
        return FLEX_SHARES[slot]
    elig = [p for p in SLOT_ELIGIBILITY.get(slot, frozenset()) if p in SKILL_POSITIONS]
    if not elig:
        return {}
    return {p: 1.0 / len(elig) for p in elig}


def starter_demand(league: LeagueSettings) -> dict[str, float]:
    """Expected number of players the league 'needs' at each position.

    Dedicated starters × teams, plus each flex slot's share allocated across its
    eligible positions, plus a bench allowance (RB/WR half a player per team, QB
    0.15 per team — 0.75 in superflex — TE 0.15, K/DEF none).
    """
    teams = float(league.total_rosters or 0)
    demand: dict[str, float] = {p: 0.0 for p in SKILL_POSITIONS}
    for slot in league.starting_slots:
        elig = SLOT_ELIGIBILITY.get(slot, frozenset())
        if len(elig) == 1:
            (pos,) = tuple(elig)
            if pos in demand:
                demand[pos] += teams
        elif len(elig) > 1:
            for pos, share in _flex_shares(slot).items():
                if pos in demand:
                    demand[pos] += share * teams
    demand["RB"] += 0.5 * teams
    demand["WR"] += 0.5 * teams
    demand["QB"] += (0.75 if league.is_superflex else 0.15) * teams
    demand["TE"] += 0.15 * teams
    return demand


def _iter_ids(available: Iterable[str | Player] | None, projections: Mapping[str, Projection]) -> Iterable[str]:
    if available is None:
        return projections.keys()
    return (a.player_id if isinstance(a, Player) else str(a) for a in available)


def _points_by_position(
    projections: Mapping[str, Projection],
    players: Mapping[str, Player],
    available: Iterable[str | Player] | None,
) -> dict[str, list[float]]:
    """Projected points per position (sorted descending) for the chosen pool."""
    by_pos: dict[str, list[float]] = {p: [] for p in SKILL_POSITIONS}
    for pid in _iter_ids(available, projections):
        proj = projections.get(pid)
        if proj is None:
            continue
        pos = proj.position or (players[pid].position if pid in players else None)
        if pos in by_pos:
            by_pos[pos].append(float(proj.points))
    for lst in by_pos.values():
        lst.sort(reverse=True)
    return by_pos


def replacement_levels(
    projections: Mapping[str, Projection],
    players: Mapping[str, Player],
    league: LeagueSettings,
    available: Iterable[str | Player] | None = None,
) -> dict[str, float]:
    """Points of the player at rank ``round(demand) + 1`` per position.

    Computed on ``available`` (ids or Players) when given, else on every projected
    player.  Positions with fewer players than the demand use the last player
    (or 0.0 when the pool is empty).
    """
    demand = starter_demand(league)
    by_pos = _points_by_position(projections, players, available)
    return {pos: _level_at(by_pos[pos], demand[pos]) for pos in SKILL_POSITIONS}


def _level_at(sorted_points: Sequence[float], demand: float) -> float:
    """Value at index ``round(demand)`` (rank demand+1) of a descending list."""
    if not sorted_points:
        return 0.0
    k = int(round(demand))
    if k >= len(sorted_points):
        return float(sorted_points[-1])
    return float(sorted_points[max(k, 0)])


def vorp(
    projections: Mapping[str, Projection],
    players: Mapping[str, Player],
    league: LeagueSettings,
    available: Iterable[str | Player] | None = None,
) -> dict[str, float]:
    """Points over replacement for every player in the pool."""
    levels = replacement_levels(projections, players, league, available)
    out: dict[str, float] = {}
    for pid in _iter_ids(available, projections):
        proj = projections.get(pid)
        if proj is None:
            continue
        out[pid] = float(proj.points) - levels.get(proj.position, 0.0)
    return out


def tiers(values: list[tuple[str, float]], std_by_id: Mapping[str, float] | None = None) -> dict[str, int]:
    """Gap-based tiers for one position.

    ``values`` are ``(player_id, points)`` pairs (any order).  A new tier starts when
    the drop to the next player exceeds ``max(0.06 * top_points, 0.5 * mean std)``.
    Tier 1 is the best.
    """
    if not values:
        return {}
    ordered = sorted(values, key=lambda t: (-t[1], t[0]))
    top = ordered[0][1]
    stds = [float(std_by_id.get(pid, 0.0)) for pid, _ in ordered] if std_by_id else []
    mean_std = sum(stds) / len(stds) if stds else 0.0
    threshold = max(0.06 * top, 0.5 * mean_std)
    out: dict[str, int] = {}
    tier = 1
    prev = top
    for pid, pts in ordered:
        if prev - pts > threshold:
            tier += 1
        out[pid] = tier
        prev = pts
    return out
