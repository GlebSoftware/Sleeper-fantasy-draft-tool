"""Trade and pick-choice evaluation (DESIGN.md §3.4).

A roster is valued as its optimal starting lineup plus its bench priced the way
the advisor prices bench picks: ``bench_discount`` × bench usefulness (RB/WR 1,
TE 0.15, QB 0.12 unless superflex, K/DEF 0) × depth factor × points over the
position's replacement level.  A trade is judged by the change in that value for
both sides; the details name lineup-slot changes and bye-week effects.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from itertools import combinations
from typing import Mapping, Sequence

from ..config import NON_STARTING_SLOTS
from ..models import DraftState, LeagueSettings, Player, Projection
from .lineup import bench_value, optimal_lineup
from .recommend import future_picks
from .replacement import replacement_levels

log = logging.getLogger(__name__)

__all__ = ["TradeEvaluation", "TradeProposal", "TradeSearchResult", "TradeVerdict", "consensus_projections",
           "evaluate_offer", "evaluate_trade", "find_trades", "roster_value", "evaluate_pick_choice",
           "SHAPES_WIDE"]

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
    replacement: Mapping[str, float] | None = None,
) -> tuple[float, dict[int, str], list[str]]:
    """``(value, lineup assignment, bench ids)``: starters + advisor-style bench value.

    ``replacement`` levels default to the full-pool :func:`replacement_levels`; a
    bench player is worth ``bench_discount`` × usefulness × depth factor × his
    points over that level, so a QB2 or K2 adds (almost) nothing.
    """
    roster = _tuples(ids, players, projections)
    assignment, starters, bench = optimal_lineup(roster, league.starting_slots)
    if replacement is None:
        replacement = replacement_levels(projections, players, league)
    pts = {pl.player_id: p for pl, p in roster}
    pos = {pl.player_id: pl.position for pl, _ in roster}
    value = starters + bench_value(bench, pos, pts, league, bench_discount, replacement)
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

    rep = replacement_levels(projections, players, league)
    my_b, my_lineup_b, _ = roster_value(my_ids, players, projections, league, bench_discount, rep)
    my_a, my_lineup_a, _ = roster_value(my_after_ids, players, projections, league, bench_discount, rep)
    th_b, th_lineup_b, _ = roster_value(their_ids, players, projections, league, bench_discount, rep)
    th_a, th_lineup_a, _ = roster_value(their_after_ids, players, projections, league, bench_discount, rep)
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


@dataclass
class TradeVerdict:
    """One specific trade, judged from both sides of the table.

    Same split as the search: my side priced with our projections, theirs with the market's consensus,
    because that is the number they are looking at. Built for a deal that already exists - one offered
    to me, or one I am putting together - so it reports rather than filters.
    """

    my_gain: float                  # our projections: what my roster gains
    their_gain: float               # our projections: what theirs does
    my_view: float                  # consensus: what the market thinks I gained
    their_view: float               # consensus: what they will think they gained
    verdict: str                    # ACCEPT | REJECT | NEUTRAL, by our numbers, for me
    fair: bool                      # does it read as roughly even to them?
    details: list[str] = field(default_factory=list)
    my_before: float = 0.0
    my_after: float = 0.0

    @property
    def edge(self) -> float:
        """How much of my gain is disagreement with the market rather than lineup fit."""
        return self.my_gain - self.my_view

    def read(self) -> str:
        """One line a person can act on."""
        if self.my_gain <= -_ACCEPT_MARGIN:
            return f"Turn it down: {self.my_gain:+.1f} points to your starting lineup."
        if self.my_gain < _ACCEPT_MARGIN:
            return f"Close to nothing either way ({self.my_gain:+.1f}). Only worth it if you want the roster spot."
        if self.their_view < -10.0:
            return (f"Good for you ({self.my_gain:+.1f}), but it reads as {self.their_view:+.0f} to them by "
                    f"consensus - expect a no without sweetening it.")
        return f"Take it: {self.my_gain:+.1f} points to your starting lineup, and it reads as {self.their_view:+.1f} to them."


def evaluate_offer(
    my_ids: Sequence[str],
    their_ids: Sequence[str],
    give: Sequence[str],
    get: Sequence[str],
    players: Mapping[str, Player],
    projections: Mapping[str, Projection],
    league: LeagueSettings,
    *,
    free_agents: Sequence[str] | None = None,
    bench_discount: float = 0.35,
) -> TradeVerdict:
    """Judge one trade that already exists, ours and theirs.

    ``free_agents`` sets the replacement baseline to who is actually available, which is what makes a
    bench player worth what he is really worth rather than what the whole projected universe implies.
    """
    pool = list(free_agents) if free_agents is not None else None
    rep = replacement_levels(projections, players, league, available=pool)
    market = consensus_projections(projections)
    rep_market = replacement_levels(market, players, league, available=pool)

    ours = _Valuer(players, projections, league, rep, bench_discount)
    theirs = _Valuer(players, market, league, rep_market, bench_discount)
    give_s, get_s = set(give), set(get)
    my_after_ids = [p for p in my_ids if p not in give_s] + list(get)
    their_after_ids = [p for p in their_ids if p not in get_s] + list(give)

    my_before, my_after = ours.value(my_ids), ours.value(my_after_ids)
    my_gain = my_after - my_before
    their_gain = ours.value(their_after_ids) - ours.value(their_ids)
    my_view = theirs.value(my_after_ids) - theirs.value(my_ids)
    their_view = theirs.value(their_after_ids) - theirs.value(their_ids)

    ev = evaluate_trade(my_ids, their_ids, give, get, players, projections, league, bench_discount)
    verdict = "ACCEPT" if my_gain > _ACCEPT_MARGIN else "REJECT" if my_gain < -_ACCEPT_MARGIN else "NEUTRAL"
    details = [d for d in ev.details if not d.startswith("Me ")]
    details.append(f"By our projections: you {my_gain:+.1f}, them {their_gain:+.1f}")
    details.append(f"By market consensus: you {my_view:+.1f}, them {their_view:+.1f}")
    unvalued = sorted(players[p].name for p in list(give) + list(get)
                      if p in players and p not in projections)
    if unvalued:
        details.append("No projection for " + ", ".join(unvalued) + ": counted as zero, so treat this with caution")
    return TradeVerdict(my_gain=my_gain, their_gain=their_gain, my_view=my_view, their_view=their_view,
                        verdict=verdict, fair=their_view >= -10.0, details=details,
                        my_before=my_before, my_after=my_after)


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
        nxt = [p for p in future_picks(state) if p > state.next_pick_no]
        when = f" at #{nxt[0]}" if nxt else " next time"
        lines.append(f"{v.player.name} is {v.availability_next:.0%} likely to be there{when}: "
                     f"take {top.player.name} now and revisit.")
    else:
        lines.append(f"{top.player.name} is worth {gap:.1f} more to your roster.")
    if v.warnings:
        lines.append("Warnings: " + "; ".join(v.warnings) + ".")
    return " ".join(lines)


# ---------------------------------------------------------------------------
# Trade search
#
# The evaluator above answers "is this trade good for me". Finding one is the harder half, and the
# thing that makes a proposal *useful* is not that it is good for me - it is that the other manager
# would say yes. Those are different questions, and answering both with our own projections is how a
# trade bot ends up proposing deals nobody accepts.
#
# So the two sides are valued differently on purpose:
#   * my gain      - our blended projection, which is what we actually believe
#   * their gain   - the same swap priced with the *consensus* number (the market's rank-implied
#                    points, already carried on every Projection as its "ecr" component)
# The edge is the gap between the two. We target players our model likes more than the market does,
# and offer players the market likes more than our model does. Both sides read the deal as a win;
# only one of them is using our numbers.
# ---------------------------------------------------------------------------

#: Below this the deal is not worth the message.
MIN_MY_GAIN = 5.0
#: How good the deal has to look to them before we will propose it (their consensus value change).
MIN_THEIR_VIEW = -2.0
#: Candidate players per side per team, best first. Keeps the search inside its time budget.
CANDIDATES_PER_SIDE = 8
#: Every shape worth searching. Two-for-two is where most real trades live but it is also the
#: expensive one (C(n,2) squared), so it comes last and the time budget can cut it off.
SHAPES_WIDE: tuple[tuple[int, int], ...] = ((1, 1), (2, 1), (1, 2), (2, 2))


@dataclass
class TradeSearchResult:
    """What the search found, and what it threw away doing it.

    The counts matter: a well-drafted league offers few clear edges, and "2 proposals" reads as a
    broken search unless it is shown next to "of 1,412 considered, 60 helped me, 58 of those read as
    a loss to them".
    """

    proposals: list["TradeProposal"] = field(default_factory=list)
    considered: int = 0             # swaps actually evaluated
    helped_me: int = 0              # ... of which cleared min_my_gain
    rejected_their_view: int = 0    # ... of those, dropped because they read as a loss to them
    timed_out: bool = False
    teams_searched: int = 0


@dataclass
class TradeProposal:
    """One deal worth sending, with both readings of it."""

    team: str                       # who to ask
    give: list[str]                 # my players
    get: list[str]                  # theirs
    my_gain: float                  # our projections: change in my roster value
    their_gain: float               # our projections: change in theirs (usually negative - that is the edge)
    their_view: float               # consensus projections: change in theirs, i.e. what they see
    my_view: float                  # consensus projections: change in mine, i.e. what they see me getting
    details: list[str] = field(default_factory=list)

    @property
    def edge(self) -> float:
        """How much of my gain comes from disagreeing with the market rather than from the lineup."""
        return self.my_gain - self.my_view

    def describe(self, players: Mapping[str, Player]) -> str:
        def names(ids: Sequence[str]) -> str:
            return " + ".join(players[p].display() if p in players else p for p in ids)
        return f"{names(self.give)} -> {self.team} for {names(self.get)}"

    def pitch(self, players: Mapping[str, Player]) -> str:
        """One line to paste into the league chat - written from *their* side of the table."""
        def names(ids: Sequence[str]) -> str:
            return " + ".join(players[p].name if p in players else p for p in ids)
        verb = "upgrade" if self.their_view > 0 else "even out"
        return (f"Want to {verb} your roster? I'll send {names(self.give)} for {names(self.get)}. "
                f"By consensus value you come out {self.their_view:+.0f} points.")


def consensus_projections(projections: Mapping[str, Projection]) -> dict[str, Projection]:
    """The same players, valued the way the market values them.

    Every :class:`Projection` already carries its rank-implied consensus figure in
    ``components["ecr"]`` - the number the blend mixes with the model. Pulling it back out costs
    nothing and is exactly "what the other manager thinks this player is worth". Players the market
    does not rank keep our number: no opinion is not the same as a low opinion.
    """
    out: dict[str, Projection] = {}
    for pid, pr in projections.items():
        market = pr.components.get("ecr")
        pts = float(market) if market is not None else float(pr.points)
        games = pr.games or 1.0
        out[pid] = replace(pr, points=pts, ppg=(pts / games if games else pr.ppg), components={}, weights={})
    return out


def _tradeable(ids: Sequence[str], players: Mapping[str, Player],
               projections: Mapping[str, Projection]) -> list[str]:
    """Who can be in a proposal at all: not a kicker, not a defence, and not a player we cannot value.

    "I have no number for him" and "he is worth nothing" are different claims, and only the second
    justifies putting a player in a trade. Conflating them is how a bot offers a real starter for
    free and reports a two-hundred-point gain, which is worse than saying nothing.
    """
    out = []
    for pid in ids:
        pl, pr = players.get(pid), projections.get(pid)
        if pl is None or pr is None or pl.position in ("K", "DEF"):
            continue
        if float(pr.points or 0.0) <= 0.0:
            continue
        out.append(pid)
    return out


def _cheapest_to_give(ids: Sequence[str], valuer: "_Valuer", before: float, limit: int) -> list[str]:
    """My players ranked by how little my roster loses without them.

    Counting positions ("I hold four backs, so a back is spare") gets this wrong: it cannot see that
    the fourth back is the one starting at FLEX, and it prices a backup quarterback the same as a
    starting one. Re-optimising the lineup without each player answers the actual question - what
    would this cost me - and the FLEX takes care of itself.
    """
    rows = []
    for pid in ids:
        loss = before - valuer.value([p for p in ids if p != pid])
        rows.append((loss, pid))
    rows.sort()
    return [pid for _, pid in rows[:limit]]


def _most_useful_to_get(their_ids: Sequence[str], my_ids: Sequence[str], valuer: "_Valuer",
                        before: float, limit: int) -> list[str]:
    """Their players ranked by what adding one would do for my starting lineup, best first.

    Only players who would actually improve it are kept: a trade for someone who lands on my bench is
    a trade I do not want, however good he looks in the abstract.
    """
    rows = []
    for pid in their_ids:
        gain = valuer.value(list(my_ids) + [pid]) - before
        if gain > 0:
            rows.append((-gain, pid))
    rows.sort()
    return [pid for _, pid in rows[:limit]]


def _spread(found: Sequence[TradeProposal], limit: int) -> list[TradeProposal]:
    """The best few from one roster, preferring variety over near-duplicates.

    Ranked purely by my gain, a roster returns the same star asked for three different ways. A first
    pass takes only deals that ask for someone not already spoken for, then the remainder fills up.
    """
    out: list[TradeProposal] = []
    asked: set[str] = set()
    for p in found:
        if len(out) >= limit:
            break
        if not (set(p.get) & asked):
            out.append(p)
            asked |= set(p.get)
    for p in found:
        if len(out) >= limit:
            break
        if p not in out:
            out.append(p)
    return out


class _Valuer:
    """Roster values under one set of projections, with the fixed parts computed once."""

    def __init__(self, players: Mapping[str, Player], projections: Mapping[str, Projection],
                 league: LeagueSettings, replacement: Mapping[str, float], bench_discount: float):
        self.players, self.projections, self.league = players, projections, league
        self.replacement, self.bench_discount = replacement, bench_discount

    def value(self, ids: Sequence[str]) -> float:
        v, _, _ = roster_value(ids, self.players, self.projections, self.league,
                               self.bench_discount, self.replacement)
        return v


def find_trades(
    my_ids: Sequence[str],
    their_rosters: Mapping[str, Sequence[str]],
    players: Mapping[str, Player],
    projections: Mapping[str, Projection],
    league: LeagueSettings,
    *,
    free_agents: Sequence[str] | None = None,
    shapes: Sequence[tuple[int, int]] = ((1, 1), (2, 1)),
    limit: int = 10,
    min_my_gain: float = MIN_MY_GAIN,
    min_their_view: float = MIN_THEIR_VIEW,
    bench_discount: float = 0.35,
    candidates_per_side: int = CANDIDATES_PER_SIDE,
    per_team_limit: int = 3,
    time_budget_s: float = 8.0,
) -> TradeSearchResult:
    """Deals that help me and that the other manager has a reason to accept.

    ``shapes`` are ``(give, get)`` counts: ``(2, 1)`` is consolidation, turning depth into a starter.
    ``free_agents``, when given, sets the replacement baseline to players who are actually available -
    the whole projected universe is not, and using it prices every bench player far too generously.
    ``per_team_limit`` keeps one willing manager from filling the whole list with variations of the
    same deal.

    Returns at most ``limit`` proposals, best for me first, with the counts behind them: a league of
    efficient rosters yields few clear edges, and two proposals only reads as a working search next to
    how many swaps were weighed to find them. Partial results on a time budget - a search that ran out
    of time returns what it found, and says so - rather than nothing.

    Players we hold no projection for are never offered and never asked for - see :func:`_tradeable`.
    """
    t0 = time.monotonic()
    mine = [p for p in my_ids if p in players]
    pool = list(free_agents) if free_agents is not None else None
    rep = replacement_levels(projections, players, league, available=pool)
    market = consensus_projections(projections)
    rep_market = replacement_levels(market, players, league, available=pool)

    ours = _Valuer(players, projections, league, rep, bench_discount)
    theirs_view = _Valuer(players, market, league, rep_market, bench_discount)
    my_before, my_before_view = ours.value(mine), theirs_view.value(mine)

    give_pool = _cheapest_to_give(_tradeable(mine, players, projections), ours, my_before, candidates_per_side)
    result = TradeSearchResult()
    out: list[TradeProposal] = []

    for team, roster in their_rosters.items():
        if time.monotonic() - t0 > time_budget_s:
            result.timed_out = True
            log.info("trade search: stopped at the time budget with %d proposals", len(out))
            break
        theirs = [p for p in roster if p in players]
        if not theirs:
            continue
        their_before, their_before_view = ours.value(theirs), theirs_view.value(theirs)
        get_pool = _most_useful_to_get(_tradeable(theirs, players, projections), mine, ours,
                                       my_before, candidates_per_side)
        if not get_pool:
            continue                                  # nothing on this roster would start for me
        result.teams_searched += 1
        found: list[TradeProposal] = []
        for n_give, n_get in shapes:
            for give in combinations(give_pool, n_give):
                for get in combinations(get_pool, n_get):
                    if time.monotonic() - t0 > time_budget_s:
                        result.timed_out = True
                        break
                    result.considered += 1
                    give_s, get_s = set(give), set(get)
                    my_after = [p for p in mine if p not in give_s] + list(get)
                    their_after = [p for p in theirs if p not in get_s] + list(give)
                    my_gain = ours.value(my_after) - my_before
                    if my_gain < min_my_gain:
                        continue
                    result.helped_me += 1
                    their_view = theirs_view.value(their_after) - their_before_view
                    if their_view < min_their_view:
                        result.rejected_their_view += 1
                        continue                      # they would read this as a loss and say no
                    their_gain = ours.value(their_after) - their_before
                    my_view = theirs_view.value(my_after) - my_before_view
                    found.append(TradeProposal(
                        team=str(team), give=list(give), get=list(get),
                        my_gain=my_gain, their_gain=their_gain, their_view=their_view, my_view=my_view,
                        details=[f"My value {my_gain:+.1f} by our projections, {my_view:+.1f} by consensus",
                                 f"Their value {their_gain:+.1f} by ours, {their_view:+.1f} by consensus"],
                    ))
        found.sort(key=lambda p: (-p.my_gain, -p.their_view))
        out.extend(_spread(found, per_team_limit))
    # best first, but never all from one roster: a page of variations on the same deal with the same
    # manager is one option, not eight, and he can only say yes once
    out.sort(key=lambda p: (-p.my_gain, -p.their_view))
    result.proposals = out[:limit]
    return result
