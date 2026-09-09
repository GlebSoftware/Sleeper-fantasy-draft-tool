"""One week of a season already under way: start/sit, the matchup, and who to pick up.

The draft advisor answers "who is the best player". In season the questions change shape - who do I
start *this week*, do I win *this* matchup, who on waivers would actually crack my lineup - and all
three need weekly numbers rather than season totals. Those come from
:mod:`draftadvisor.projections.inseason` (measured weekly spread, schedule, defence versus position)
and from ESPN's own payload (who is not playing, what lineup is set, who I face).

Everything here is pure computation over data already fetched, and it is deliberately platform-shaped
only at the edges: :func:`plan_week` takes a parsed ESPN payload because that is where the set lineup
and the schedule live, while :func:`waiver_targets` needs no platform at all.

No pandas, no paid API.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..models import LeagueSettings, Player, Projection
from ..projections.inseason import InSeasonTables, week_mean, week_sigma, win_probability
from .lineup import optimal_lineup

log = logging.getLogger(__name__)

__all__ = ["WeekSlot", "WeekPlan", "WaiverTarget", "plan_week", "waiver_targets"]

#: Below this a lineup change is not worth making.
SWAP_THRESHOLD = 0.5


@dataclass
class WeekSlot:
    slot: str
    player_id: str | None
    name: str | None
    position: str | None
    team: str | None
    points: float


@dataclass
class WeekPlan:
    week: int
    team_id: int
    team_name: str
    starters: list[WeekSlot] = field(default_factory=list)
    bench: list[WeekSlot] = field(default_factory=list)
    best_points: float = 0.0
    #: Points the lineup currently set on the platform scores, when one could be read.
    set_points: float | None = None
    start: list[str] = field(default_factory=list)          # names to move in
    sit: list[str] = field(default_factory=list)            # names to move out
    #: Roster players with no projection at all. They score 0 and are never started, which is
    #: indistinguishable from "he is bad" unless it is said out loud.
    unprojected: list[str] = field(default_factory=list)
    opponent_id: int | None = None
    opponent_name: str | None = None
    my_points: float = 0.0
    their_points: float = 0.0
    my_sigma: float = 0.0
    their_sigma: float = 0.0
    win_probability: float | None = None
    espn_playoff_pct: float | None = None
    is_playoffs: bool = False
    weeks_remaining: int = 0

    @property
    def gap(self) -> float:
        """Points the lineup as set leaves on the bench (0 when it is already optimal or unknown)."""
        return max(0.0, self.best_points - self.set_points) if self.set_points is not None else 0.0


@dataclass
class WaiverTarget:
    player_id: str
    name: str
    position: str | None
    team: str | None
    week_points: float
    #: What adding him does to my *starting lineup* this week. Scoring a lot on my bench is worth zero.
    lineup_gain: float
    #: Whose place he would take.
    replaces: str | None = None


def _week_values(ids: Sequence[str], players: Mapping[str, Player], projections: Mapping[str, Projection],
                 tables: InSeasonTables, week: int, playing: Any = None) -> dict[str, float]:
    out: dict[str, float] = {}
    for pid in ids:
        pl, pr = players.get(pid), projections.get(pid)
        if pl is None or pr is None:
            out[pid] = 0.0
            continue
        if playing is not None and not playing(pid):
            out[pid] = 0.0
            continue
        out[pid] = week_mean(pr, pl, tables, week)
    return out


def _lineup_stats(ids: Sequence[str], values: Mapping[str, float], players: Mapping[str, Player],
                  projections: Mapping[str, Projection], tables: InSeasonTables, week: int) -> tuple[float, float]:
    """``(mean, sigma)`` from exactly the values the caller is showing, so the odds and the table agree.

    Players are added independently: a real lineup correlates a little (a quarterback and his own
    receiver), so a stacked lineup swings slightly more than this says.
    """
    mean = sum(values.get(p, 0.0) for p in ids)
    var = 0.0
    for pid in ids:
        if values.get(pid, 0.0) > 0 and pid in players and pid in projections:
            var += week_sigma(projections[pid], players[pid], tables, week) ** 2
    return mean, math.sqrt(var)


def _slots(assignment: Mapping[int, str], slot_names: Sequence[str], players: Mapping[str, Player],
           values: Mapping[str, float]) -> list[WeekSlot]:
    out: list[WeekSlot] = []
    for i, slot in enumerate(slot_names):
        pid = assignment.get(i)
        if pid is None:
            continue
        pl = players.get(pid)
        out.append(WeekSlot(slot=slot, player_id=pid, name=pl.name if pl else pid,
                            position=pl.position if pl else None, team=pl.team if pl else None,
                            points=round(float(values.get(pid, 0.0)), 2)))
    return out


def plan_week(
    rosters: Mapping[int, Sequence[str]],
    team_names: Mapping[int, str],
    my_team_id: int,
    players: Mapping[str, Player],
    projections: Mapping[str, Projection],
    league: LeagueSettings,
    tables: InSeasonTables,
    week: int,
    *,
    set_starters: Sequence[str] = (),
    opponent_id: int | None = None,
    not_playing: Sequence[str] = (),
    espn_playoff_pct: float | None = None,
    is_playoffs: bool = False,
    weeks_remaining: int = 0,
) -> WeekPlan:
    """Best lineup for ``week``, what the set lineup leaves behind, and the matchup odds.

    ``not_playing`` is the platform's own answer to "is he active at all" (a bye, an inactive, a
    suspension). Our model supplies the points; the platform is better placed to know who is not on
    the field, so the two are used for what each is good at.
    """
    mine = [p for p in rosters.get(my_team_id, ()) if p in players]
    out_ids = set(not_playing)
    values = _week_values(mine, players, projections, tables, week, playing=lambda p: p not in out_ids)
    assignment, best_points, bench_ids = optimal_lineup([(players[p], values[p]) for p in mine],
                                                        league.starting_slots)
    plan = WeekPlan(week=int(week), team_id=int(my_team_id),
                    team_name=team_names.get(my_team_id, f"Team {my_team_id}"),
                    starters=_slots(assignment, league.starting_slots, players, values),
                    bench=[WeekSlot("BN", p, players[p].name, players[p].position, players[p].team,
                                    round(values.get(p, 0.0), 2)) for p in bench_ids if p in players],
                    best_points=round(float(best_points), 2),
                    unprojected=sorted(players[p].name for p in mine if p not in projections),
                    is_playoffs=bool(is_playoffs), weeks_remaining=int(weeks_remaining),
                    espn_playoff_pct=espn_playoff_pct)

    started = [p for p in set_starters if p in players]
    if started:
        plan.set_points = round(sum(values.get(p, 0.0) for p in started), 2)
        best_ids = set(assignment.values())
        if plan.gap > SWAP_THRESHOLD:
            plan.start = [players[p].name for p in best_ids - set(started) if p in players]
            plan.sit = [players[p].name for p in set(started) - best_ids if p in players]

    if opponent_id is None or opponent_id not in rosters:
        return plan
    theirs = [p for p in rosters[opponent_id] if p in players]
    their_values = _week_values(theirs, players, projections, tables, week, playing=lambda p: p not in out_ids)
    their_assign, their_points, _ = optimal_lineup([(players[p], their_values[p]) for p in theirs],
                                                   league.starting_slots)
    plan.opponent_id = int(opponent_id)
    plan.opponent_name = team_names.get(int(opponent_id), f"Team {opponent_id}")
    plan.my_points, plan.my_sigma = _lineup_stats(list(assignment.values()), values, players, projections, tables, week)
    plan.their_points, plan.their_sigma = _lineup_stats(list(their_assign.values()), their_values, players,
                                                        projections, tables, week)
    plan.my_points, plan.their_points = round(plan.my_points, 2), round(plan.their_points, 2)
    plan.my_sigma, plan.their_sigma = round(plan.my_sigma, 2), round(plan.their_sigma, 2)
    plan.win_probability = round(win_probability((plan.my_points, plan.my_sigma),
                                                 (plan.their_points, plan.their_sigma)), 4)
    return plan


def waiver_targets(
    free_agents: Sequence[str],
    my_ids: Sequence[str],
    players: Mapping[str, Player],
    projections: Mapping[str, Projection],
    league: LeagueSettings,
    tables: InSeasonTables,
    week: int,
    *,
    limit: int = 10,
    not_playing: Sequence[str] = (),
    consider: int = 60,
) -> list[WaiverTarget]:
    """Free agents ranked by what they would do to my *starting lineup* this week.

    Not by who scores most: a 15-point receiver behind two better ones is worth nothing, and the
    waiver wire is full of them. Only the top ``consider`` by weekly points are re-optimised, which
    keeps the work bounded without changing the answer - a player outside that set cannot crack a
    lineup the top of it could not.
    """
    out_ids = set(not_playing)
    mine = [p for p in my_ids if p in players]
    my_values = _week_values(mine, players, projections, tables, week, playing=lambda p: p not in out_ids)
    before_assign, before, _ = optimal_lineup([(players[p], my_values[p]) for p in mine], league.starting_slots)
    starting_now = set(before_assign.values())

    scored: list[tuple[float, str]] = []
    for pid in free_agents:
        pl, pr = players.get(pid), projections.get(pid)
        if pl is None or pr is None or pid in out_ids:
            continue
        scored.append((week_mean(pr, pl, tables, week), pid))
    scored.sort(reverse=True)

    out: list[WaiverTarget] = []
    for pts, pid in scored[:consider]:
        cand = [(players[p], my_values[p]) for p in mine] + [(players[pid], pts)]
        after_assign, after, _ = optimal_lineup(cand, league.starting_slots)
        gain = after - before
        if gain <= 0:
            continue
        # who he displaces: someone who was starting and no longer is once he is added. Includes a
        # player scoring zero - an empty or bye-week slot is exactly the hole a pickup fills.
        dropped = starting_now - set(after_assign.values())
        replaced = next((players[p].name for p in sorted(dropped, key=lambda x: my_values.get(x, 0.0))
                         if p in players), None)
        out.append(WaiverTarget(player_id=pid, name=players[pid].name, position=players[pid].position,
                                team=players[pid].team, week_points=round(float(pts), 2),
                                lineup_gain=round(float(gain), 2), replaces=replaced))
    out.sort(key=lambda t: -t.lineup_gain)
    return out[:limit]
