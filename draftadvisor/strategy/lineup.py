"""Optimal starting lineups and roster summaries (DESIGN.md §3.4).

:func:`optimal_lineup` is a small deterministic assignment heuristic (dedicated
slots first, then flex slots, then single-swap improvements) that is exact for
every standard Sleeper roster shape and runs in tens of microseconds.

:func:`starter_thresholds` exploits the structure of the problem: adding one
player to an optimal lineup changes it along a single alternating path, so the
lineup gain of a candidate at position ``p`` with ``x`` points is exactly
``max(0, x - t_p)`` where ``t_p`` is the weakest starter reachable from ``p``
through slot eligibility (0 when an eligible slot is empty).  The advisor uses
this to value hundreds of candidates with one numpy expression.
"""
from __future__ import annotations

import logging
from typing import Mapping, Sequence

from ..config import NON_STARTING_SLOTS, SKILL_POSITIONS, SLOT_ELIGIBILITY
from ..models import DraftState, LeagueSettings, Pick, Player, Projection, RosterSummary

log = logging.getLogger(__name__)

__all__ = [
    "optimal_lineup",
    "starter_thresholds",
    "starter_displacements",
    "bench_usefulness",
    "bench_depth_factor",
    "bench_value",
    "marginal_lineup_value",
    "roster_summary",
    "summarize_roster",
    "player_from_pick",
    "phantom_starters",
]

_INF = float("inf")
_PHANTOM_PREFIX = "__rep__"


# ---------------------------------------------------------------------------
# Optimal lineup
# ---------------------------------------------------------------------------


def _slot_order(slots: Sequence[str]) -> list[int]:
    """Fill order: dedicated slots first, then flex slots (most restrictive first)."""
    order = [i for i, s in enumerate(slots) if len(SLOT_ELIGIBILITY.get(s, frozenset())) == 1]
    flex = [i for i, s in enumerate(slots) if len(SLOT_ELIGIBILITY.get(s, frozenset())) > 1]
    flex.sort(key=lambda i: (len(SLOT_ELIGIBILITY[slots[i]]), i))
    return order + flex


def optimal_lineup(
    candidates: Sequence[tuple[Player, float]], slots: Sequence[str]
) -> tuple[dict[int, str], float, list[str]]:
    """Assign players to starting ``slots`` maximising total points.

    Returns ``(assignment, starters_points, bench_ids)`` where ``assignment`` maps a
    slot index (into ``slots``) to a player_id and ``bench_ids`` lists the unassigned
    players by points descending.  Slots with no eligible positions (BN, IDP, ...)
    are never filled.  Deterministic: ties broken by player_id.
    """
    pts: dict[str, float] = {}
    pos: dict[str, str] = {}
    by_pos: dict[str, list[str]] = {}
    for player, p in candidates:
        pid = player.player_id
        if pid in pts:                       # duplicate id: keep the best
            if p <= pts[pid]:
                continue
        pts[pid] = float(p)
        pos[pid] = player.position
    for pid in sorted(pts, key=lambda k: (-pts[k], k)):
        by_pos.setdefault(pos[pid], []).append(pid)

    elig = [SLOT_ELIGIBILITY.get(s, frozenset()) for s in slots]
    assign: list[str | None] = [None] * len(slots)
    used: set[str] = set()

    def best_unused(slot_idx: int) -> str | None:
        best: str | None = None
        best_p = -_INF
        for p in elig[slot_idx]:
            for pid in by_pos.get(p, ()):
                if pid not in used:
                    if pts[pid] > best_p or (pts[pid] == best_p and best is not None and pid < best):
                        best, best_p = pid, pts[pid]
                    break                    # lists are sorted: first unused is the best at p
        return best

    for i in _slot_order(slots):
        pid = best_unused(i)
        if pid is not None:
            assign[i] = pid
            used.add(pid)

    # single-swap improvements: move X from s1 to s2 (bumping Y) and fill s1 with B
    starters = [i for i in range(len(slots)) if elig[i]]
    for _ in range(4):
        improved = False
        for s1 in starters:
            x = assign[s1]
            for s2 in starters:
                if s1 == s2:
                    continue
                y = assign[s2]
                if x is None:
                    # empty s1: pull Y into it and refill s2
                    if y is None or pos[y] not in elig[s1]:
                        continue
                    used.discard(y)
                    b = best_unused(s2)
                    if b is None or b == y:
                        used.add(y)
                        continue
                    assign[s1], assign[s2] = y, b
                    used.add(y)
                    used.add(b)
                    improved = True
                    x = assign[s1]
                    continue
                if pos[x] not in elig[s2]:
                    continue
                b = best_unused(s1)
                if b is None:
                    continue
                gain = pts[b] - (pts[y] if y is not None else 0.0)
                if gain > 1e-9:
                    assign[s2], assign[s1] = x, b
                    used.add(b)
                    if y is not None:
                        used.discard(y)
                    improved = True
                    x = b
        if not improved:
            break

    assignment = {i: pid for i, pid in enumerate(assign) if pid is not None}
    total = sum(pts[pid] for pid in assignment.values())
    bench = [pid for pid in sorted(pts, key=lambda k: (-pts[k], k)) if pid not in assignment.values()]
    return assignment, total, bench


def starter_displacements(
    assignment: Mapping[int, tuple[str, float]], slots: Sequence[str]
) -> dict[str, tuple[float, int | None]]:
    """``(threshold, displaced slot index)`` per position for an optimal lineup.

    ``assignment`` maps slot index -> ``(position, points)`` of its occupant.  For a
    candidate at position ``p`` the lineup gain is ``max(0, x - t_p)``; ``t_p`` is 0
    when a reachable slot is empty and ``inf`` when ``p`` can start nowhere.  The
    slot index is that of the weakest reachable starter — the player who goes to
    the bench when the candidate starts — or ``None`` when the candidate would
    fill an empty slot (or can start nowhere).
    """
    elig = [SLOT_ELIGIBILITY.get(s, frozenset()) for s in slots]
    out: dict[str, tuple[float, int | None]] = {}
    for p in SKILL_POSITIONS:
        reach = {p}
        seen: set[int] = set()
        t = _INF
        weakest: int | None = None
        frontier = True
        while frontier:
            frontier = False
            for i, e in enumerate(elig):
                if i in seen or not (e & reach):
                    continue
                seen.add(i)
                occ = assignment.get(i)
                if occ is None:
                    t, weakest = 0.0, None
                    break
                if occ[1] < t:
                    t, weakest = occ[1], i
                if occ[0] not in reach:
                    reach.add(occ[0])
                    frontier = True
            if t == 0.0:
                break
        out[p] = (t, weakest)
    return out


def starter_thresholds(
    assignment: Mapping[int, tuple[str, float]], slots: Sequence[str]
) -> dict[str, float]:
    """Displacement threshold per position (see :func:`starter_displacements`)."""
    return {p: t for p, (t, _) in starter_displacements(assignment, slots).items()}


# ---------------------------------------------------------------------------
# Bench value helpers
# ---------------------------------------------------------------------------


def bench_usefulness(position: str, league: LeagueSettings) -> float:
    """How useful a benched player at ``position`` is.

    RB/WR depth starts most weeks (byes, injuries, flex): 1.0. A backup QB or TE only
    plays when the starter is out: 0.35 (QB 1.0 in superflex leagues). K/DEF: 0.
    """
    if position in ("RB", "WR"):
        return 1.0
    if position == "QB":
        # a QB2 in a 1-QB league only starts during the QB1's bye / injury (~2 games)
        return 1.0 if league.is_superflex else 0.12
    if position == "TE":
        return 0.15
    return 0.0


def bench_depth_factor(n_same_position_on_bench: int, position: str = "RB") -> float:
    """Each extra bench player at the same position is worth less.

    RB/WR depth keeps real value (injury/bye starts, breakout upside): 0.6 per extra
    body. A third QB or TE almost never plays: 0.25 per extra.
    """
    n = max(0, int(n_same_position_on_bench))
    return (0.25 if position in ("QB", "TE") else 0.6) ** n


def phantom_starters(league: LeagueSettings, replacement: Mapping[str, float]) -> list[tuple[Player, float]]:
    """Replacement-level stand-ins, one per starting slot (assigned to the best eligible position)."""
    out: list[tuple[Player, float]] = []
    for i, slot in enumerate(league.starting_slots):
        elig = [p for p in SLOT_ELIGIBILITY.get(slot, frozenset()) if p in SKILL_POSITIONS]
        if not elig:
            continue
        best = max(elig, key=lambda p: (replacement.get(p, 0.0), p))
        out.append((Player(player_id=f"{_PHANTOM_PREFIX}{i}", name=f"replacement {best}", position=best),
                    float(replacement.get(best, 0.0))))
    return out


def is_phantom(player_id: str) -> bool:
    return player_id.startswith(_PHANTOM_PREFIX)


def bench_value(
    bench_ids: Sequence[str],
    pos_of: Mapping[str, str],
    pts_of: Mapping[str, float],
    league: LeagueSettings,
    bench_discount: float,
    replacement: Mapping[str, float] | None = None,
) -> float:
    """Discounted value of a bench, priced the way the advisor prices bench picks.

    Each benched player is worth ``bench_discount`` × bench usefulness of his
    position × the depth factor for the same-position players ahead of him × his
    points over the position's replacement level (0 when below it).  ``bench_ids``
    should be in points-descending order (what :func:`optimal_lineup` returns);
    unknown positions (IDP, ``UNK``) and K/DEF add nothing.
    """
    rep = replacement or {}
    seen: dict[str, int] = {}
    total = 0.0
    for pid in bench_ids:
        if is_phantom(pid):
            continue
        pos = pos_of.get(pid, "UNK")
        n_before = seen.get(pos, 0)
        seen[pos] = n_before + 1
        over = max(0.0, float(pts_of.get(pid, 0.0)) - float(rep.get(pos, 0.0)))
        total += bench_discount * bench_usefulness(pos, league) * bench_depth_factor(n_before, pos) * over
    return total


def marginal_lineup_value(
    candidate: Player,
    cand_points: float,
    roster: Sequence[tuple[Player, float]],
    league: LeagueSettings,
    bench_discount: float,
    replacement: Mapping[str, float] | None = None,
) -> float:
    """Lineup improvement from adding ``candidate`` plus discounted bench value.

    ``lineup(roster + cand) - lineup(roster)`` plus ``bench_discount`` × bench
    usefulness × depth factor × points of whoever ends up on the bench because of
    the pick (the candidate himself, or the starter he displaces).  When
    ``replacement`` levels are given, open starting slots are first filled with
    replacement-level stand-ins so the value of filling a hole is measured against
    what the waiver wire would provide, and bench value is measured over
    replacement too.
    """
    slots = league.starting_slots
    base = list(roster)
    a_real, p_real, bench_ids0 = optimal_lineup(base, slots)
    if replacement:
        # stand-ins only for the slots the real roster leaves open
        phantoms = [ph for ph in phantom_starters(league, replacement)
                    if int(ph[0].player_id[len(_PHANTOM_PREFIX):]) not in a_real]
        base = base + phantoms
        a0, p0, _ = optimal_lineup(base, slots)
    else:
        a0, p0 = a_real, p_real
    a1, p1, bench1 = optimal_lineup(base + [(candidate, cand_points)], slots)
    gain = p1 - p0

    pos_of = {p.player_id: p.position for p, _ in base}
    pts_of = {p.player_id: pts for p, pts in base}
    pos_of[candidate.player_id] = candidate.position
    pts_of[candidate.player_id] = cand_points
    rep = replacement or {}

    def bench_value(pid: str) -> float:
        if is_phantom(pid):
            return 0.0
        pos = pos_of[pid]
        n_same = sum(1 for b in bench_ids0 if pos_of.get(b) == pos and b != pid)
        over = max(0.0, pts_of[pid] - rep.get(pos, 0.0))
        return bench_discount * bench_usefulness(pos, league) * bench_depth_factor(n_same, pos) * over

    if candidate.player_id not in a1.values():
        return gain + bench_value(candidate.player_id)
    displaced = [pid for pid in bench1 if pid not in bench_ids0 and pid != candidate.player_id and not is_phantom(pid)]
    return gain + sum(bench_value(pid) for pid in displaced)


# ---------------------------------------------------------------------------
# Roster summaries
# ---------------------------------------------------------------------------


def player_from_pick(pick: Pick, players: Mapping[str, Player]) -> Player:
    """The drafted :class:`Player`, synthesised from pick metadata when unknown."""
    p = players.get(pick.player_id)
    if p is not None:
        return p
    md = pick.metadata or {}
    return Player(
        player_id=pick.player_id,
        name=pick.player_name,
        position=str(md.get("position") or "UNK"),
        team=md.get("team") or None,
        injury_status=md.get("injury_status") or None,
        years_exp=int(md["years_exp"]) if str(md.get("years_exp") or "").isdigit() else None,
    )


def summarize_roster(
    slot: int,
    label: str,
    picks: Sequence[Pick],
    players: Mapping[str, Player],
    projections: Mapping[str, Projection],
    league: LeagueSettings,
    extra_ids: Sequence[str] = (),
) -> RosterSummary:
    """Build a :class:`RosterSummary` for one team's picks.

    ``extra_ids`` names players the team holds without a pick of their own - on ESPN a drafted player
    shows up on the team's roster long before (and sometimes instead of) appearing on the draft board.
    A pick always wins over an extra id for the same player.
    """
    roster: list[tuple[Player, float]] = []
    seen: set[str] = set()
    for pk in picks:
        pl = player_from_pick(pk, players)
        proj = projections.get(pl.player_id)
        roster.append((pl, float(proj.points) if proj else 0.0))
        seen.add(pl.player_id)
    for pid in extra_ids:
        pl = players.get(pid)
        if pl is None or pid in seen:
            continue
        seen.add(pid)
        proj = projections.get(pid)
        roster.append((pl, float(proj.points) if proj else 0.0))
    slots = league.starting_slots
    assignment, lineup_pts, bench = optimal_lineup(roster, slots)
    pts = {pl.player_id: p for pl, p in roster}
    by_id = {pl.player_id: pl for pl, _ in roster}

    filled: dict[str, int] = {}
    open_: dict[str, int] = {}
    for i, s in enumerate(slots):
        if s in NON_STARTING_SLOTS:
            continue
        filled.setdefault(s, 0)
        open_.setdefault(s, 0)
        if i in assignment:
            filled[s] += 1
        else:
            open_[s] += 1
    counts: dict[str, int] = {}
    for pl, _ in roster:
        counts[pl.position] = counts.get(pl.position, 0) + 1
    byes: dict[int, int] = {}
    for pid in assignment.values():
        bw = by_id[pid].bye_week
        if bw:
            byes[bw] = byes.get(bw, 0) + 1
    return RosterSummary(
        slot=slot,
        label=label,
        players=[pl for pl, _ in roster],
        starters_filled=filled,
        open_starters=open_,
        position_counts=counts,
        bye_weeks=byes,
        lineup_points=float(lineup_pts),
        bench_points=float(sum(pts[b] for b in bench)),
    )


def roster_summary(
    state: DraftState,
    slot: int,
    players: Mapping[str, Player],
    projections: Mapping[str, Projection],
    league: LeagueSettings,
) -> RosterSummary:
    """Roster summary for the team drafting from ``slot`` in ``state``."""
    picks = state.picks_by_slot().get(slot, [])
    return summarize_roster(slot, state.slot_label(slot), picks, players, projections, league)
