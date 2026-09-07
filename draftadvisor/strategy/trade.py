"""Trade and pick-choice evaluation (DESIGN.md §3.4).

A roster is valued as its optimal starting lineup plus ``bench_discount`` × the
points of its bench.  A trade is judged by the change in that value for both
sides; the details name lineup-slot changes and bye-week effects.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ..config import NON_STARTING_SLOTS
from ..models import DraftState, LeagueSettings, Player, Projection
from .lineup import optimal_lineup

log = logging.getLogger(__name__)

__all__ = ["TradeEvaluation", "evaluate_trade", "roster_value", "evaluate_pick_choice"]

_ACCEPT_MARGIN = 3.0
_LOPSIDED_FLOOR = 15.0


@dataclass
class TradeEvaluation:
    my_before: float
    my_after: float
    their_before: float
    their_after: float
    my_delta: float
    their_delta: float
    verdict: str                    # "ACCEPT" | "REJECT" | "NEUTRAL"
    details: list[str] = field(default_factory=list)


def _tuples(ids: Sequence[str], players: Mapping[str, Player], projections: Mapping[str, Projection]) -> list[tuple[Player, float]]:
    out: list[tuple[Player, float]] = []
    for pid in ids:
        pl = players.get(pid)
        if pl is None:
            log.warning("trade: unknown player id %s ignored", pid)
            continue
        pr = projections.get(pid)
        out.append((pl, float(pr.points) if pr else 0.0))
    return out


def roster_value(
    ids: Sequence[str],
    players: Mapping[str, Player],
    projections: Mapping[str, Projection],
    league: LeagueSettings,
    bench_discount: float = 0.35,
) -> tuple[float, dict[int, str], list[str]]:
    """``(value, lineup assignment, bench ids)``: starters + discounted bench points."""
    roster = _tuples(ids, players, projections)
    assignment, starters, bench = optimal_lineup(roster, league.starting_slots)
    pts = {pl.player_id: p for pl, p in roster}
    value = starters + bench_discount * sum(pts[b] for b in bench)
    return float(value), assignment, bench


def _slot_labels(league: LeagueSettings) -> list[str]:
    labels: list[str] = []
    seen: dict[str, int] = {}
    for s in league.starting_slots:
        if s in NON_STARTING_SLOTS:
            labels.append(s)
            continue
        seen[s] = seen.get(s, 0) + 1
        n = league.starting_slots.count(s)
        labels.append(f"{s}{seen[s]}" if n > 1 else s)
    return labels


def _lineup_changes(
    who: str, before: dict[int, str], after: dict[int, str], players: Mapping[str, Player],
    projections: Mapping[str, Projection], league: LeagueSettings,
) -> list[str]:
    labels = _slot_labels(league)
    out: list[str] = []
    for i, label in enumerate(labels):
        b, a = before.get(i), after.get(i)
        if b == a:
            continue
        def _name(pid: str | None) -> str:
            if pid is None:
                return "(empty)"
            pl = players.get(pid)
            pr = projections.get(pid)
            return f"{pl.name if pl else pid} ({pr.points:.0f})" if pr else (pl.name if pl else pid)
        delta = (projections[a].points if a and a in projections else 0.0) - \
                (projections[b].points if b and b in projections else 0.0)
        out.append(f"{who} {label}: {_name(b)} -> {_name(a)} ({delta:+.0f})")
    return out


def _bye_clashes(assignment: Mapping[int, str], players: Mapping[str, Player]) -> int:
    """Number of starters sharing a bye week with another starter."""
    counts: dict[int, int] = {}
    for pid in assignment.values():
        pl = players.get(pid)
        if pl and pl.bye_week:
            counts[pl.bye_week] = counts.get(pl.bye_week, 0) + 1
    return sum(c for c in counts.values() if c > 1)


def evaluate_trade(
    my_ids: Sequence[str],
    their_ids: Sequence[str],
    give: Sequence[str],
    get: Sequence[str],
    players: Mapping[str, Player],
    projections: Mapping[str, Projection],
    league: LeagueSettings,
    bench_discount: float = 0.35,
) -> TradeEvaluation:
    """Evaluate giving ``give`` (from my roster) for ``get`` (from theirs)."""
    give_s, get_s = set(give), set(get)
    missing = [pid for pid in give if pid not in set(my_ids)] + [pid for pid in get if pid not in set(their_ids)]
    my_after_ids = [pid for pid in my_ids if pid not in give_s] + list(get)
    their_after_ids = [pid for pid in their_ids if pid not in get_s] + list(give)

    my_b, my_lineup_b, _ = roster_value(my_ids, players, projections, league, bench_discount)
    my_a, my_lineup_a, _ = roster_value(my_after_ids, players, projections, league, bench_discount)
    th_b, th_lineup_b, _ = roster_value(their_ids, players, projections, league, bench_discount)
    th_a, th_lineup_a, _ = roster_value(their_after_ids, players, projections, league, bench_discount)
    my_delta, their_delta = my_a - my_b, th_a - th_b

    details: list[str] = []
    if missing:
        details.append("Not on the stated rosters: " + ", ".join(missing))
    details += _lineup_changes("Me", my_lineup_b, my_lineup_a, players, projections, league)
    details += _lineup_changes("Them", th_lineup_b, th_lineup_a, players, projections, league)
    my_bye_b, my_bye_a = _bye_clashes(my_lineup_b, players), _bye_clashes(my_lineup_a, players)
    if my_bye_a != my_bye_b:
        details.append(f"My starter bye clashes: {my_bye_b} -> {my_bye_a}")
    th_bye_b, th_bye_a = _bye_clashes(th_lineup_b, players), _bye_clashes(th_lineup_a, players)
    if th_bye_a != th_bye_b:
        details.append(f"Their starter bye clashes: {th_bye_b} -> {th_bye_a}")
    if len(my_after_ids) != len(my_ids):
        details.append(f"Roster size changes {len(my_ids)} -> {len(my_after_ids)}")

    lopsided = their_delta < -max(_LOPSIDED_FLOOR, 3.0 * my_delta)
    if my_delta > _ACCEPT_MARGIN and not lopsided:
        verdict = "ACCEPT"
    elif my_delta > _ACCEPT_MARGIN:
        verdict = "NEUTRAL"
        details.append(f"Clearly lopsided ({their_delta:+.0f} for them): unlikely to be accepted")
    elif my_delta < -_ACCEPT_MARGIN:
        verdict = "REJECT"
    else:
        verdict = "NEUTRAL"
    details.append(f"Me {my_delta:+.1f} pts, them {their_delta:+.1f} pts")
    return TradeEvaluation(my_b, my_a, th_b, th_a, my_delta, their_delta, verdict, details)


def evaluate_pick_choice(state: DraftState, advisor, player_id: str) -> str:
    """Why (not) take ``player_id`` now versus the advisor's top recommendation."""
    rec = advisor.recommend(state)
    v = advisor.value_of(state, player_id)
    top = rec.top_pick
    if v is None:
        return advisor.explain_pick(state, player_id)
    if top is None or top.player_id == player_id:
        return f"{v.player.display()} is the top recommendation: " + "; ".join(v.reasons) + "."
    gap = top.score - v.score
    lines = [
        f"{v.player.display()} scores {v.score:.1f} (#{v.overall_rank}); top pick {top.player.display()} scores {top.score:.1f}.",
        f"{v.player.name}: " + "; ".join(v.reasons) + ".",
        f"{top.player.name}: " + "; ".join(top.reasons) + ".",
    ]
    if gap <= 5:
        lines.append(f"Close call ({gap:.1f} pts): either pick is fine.")
    elif v.availability_next >= 0.75:
        nxt = [p for p in state.my_future_picks() if p > state.next_pick_no]
        when = f" at #{nxt[0]}" if nxt else " next time"
        lines.append(f"{v.player.name} is {v.availability_next:.0%} likely to be there{when}: "
                     f"take {top.player.name} now and revisit.")
    else:
        lines.append(f"{top.player.name} is worth {gap:.1f} more to your roster.")
    if v.warnings:
        lines.append("Warnings: " + "; ".join(v.warnings) + ".")
    return " ".join(lines)
