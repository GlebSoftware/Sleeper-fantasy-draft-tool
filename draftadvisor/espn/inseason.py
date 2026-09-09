"""What ESPN publishes once the season starts: the calendar, the standings and the weekly scores.

These fields sit in payloads the app already fetches and currently throws away. Three of them matter:

* ``settings.scheduleSettings`` says how long the regular season is, how many teams make the playoffs
  and how seeds are broken. The rest of the app has been hardcoding week 15.
* ``teams[].record`` and ``teams[].currentSimulationResults.playoffPct`` are ESPN's own standings and
  its own playoff odds - free, and a check on ours.
* every player's ``stats[]`` carries one entry per week with ``statSourceId`` 0 (actual) or 1
  (projected) and an ``appliedTotal`` **already scored under this league's rules**, so for ESPN
  leagues we never have to re-score a stat line.

A matchup period is not always a week: ``matchupPeriodLength`` can be 2, and then one matchup spans
two scoring periods. The two ids are kept apart here rather than assumed equal.

No pandas: this module runs in the deployed runtime.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

log = logging.getLogger(__name__)

__all__ = ["LeagueCalendar", "TeamRecord", "Matchup", "league_calendar", "team_records", "matchups",
           "weekly_player_points", "opponent_for", "points_to_date", "current_lineups",
           "BENCH_SLOT_ID", "IR_SLOT_ID"]

#: ESPN lineup slots that are not starting a player this week.
BENCH_SLOT_ID, IR_SLOT_ID = 20, 21

#: ESPN's own labels for a player stat block.
ACTUAL, PROJECTED = 0, 1


def _int(v: Any, default: int | None = None) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _float(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None                       # NaN is not a score


def _mapping(v: Any) -> Mapping[str, Any]:
    return v if isinstance(v, Mapping) else {}


@dataclass(frozen=True)
class LeagueCalendar:
    """Where the season is and how it ends."""

    current_week: int = 1
    first_week: int = 1
    final_week: int = 17
    regular_season_weeks: int = 14          # matchupPeriodCount
    playoff_week_start: int = 15            # first playoff matchup period
    playoff_teams: int = 6
    seeding_rule: str | None = None
    matchup_period_length: int = 1
    #: matchup period -> the scoring periods (weeks) it covers, when ESPN publishes the mapping
    periods: dict[int, list[int]] = field(default_factory=dict)

    def is_playoffs(self, week: int | None = None) -> bool:
        return int(week if week is not None else self.current_week) >= self.playoff_week_start

    def weeks_remaining(self, week: int | None = None) -> int:
        """Regular-season weeks left including the current one."""
        w = int(week if week is not None else self.current_week)
        return max(0, self.regular_season_weeks - w + 1)

    def matchup_period_for(self, week: int) -> int:
        for period, weeks in self.periods.items():
            if int(week) in weeks:
                return int(period)
        return int(week)


@dataclass(frozen=True)
class TeamRecord:
    team_id: int
    name: str | None = None
    wins: int = 0
    losses: int = 0
    ties: int = 0
    points_for: float = 0.0
    points_against: float = 0.0
    playoff_seed: int | None = None
    #: ESPN's own playoff probability (percent), when it publishes one. Never ours.
    espn_playoff_pct: float | None = None

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.ties

    @property
    def win_pct(self) -> float:
        return (self.wins + 0.5 * self.ties) / self.games if self.games else 0.0


@dataclass(frozen=True)
class Matchup:
    week: int                                # matchup period
    home_team_id: int | None
    away_team_id: int | None
    home_points: float = 0.0
    away_points: float = 0.0
    winner: str | None = None                # ESPN's own label: HOME / AWAY / TIE / UNDECIDED

    @property
    def played(self) -> bool:
        return bool(self.winner) and self.winner != "UNDECIDED"

    def opponent_of(self, team_id: int) -> int | None:
        if self.home_team_id == team_id:
            return self.away_team_id
        if self.away_team_id == team_id:
            return self.home_team_id
        return None


def league_calendar(league_json: Mapping[str, Any] | None) -> LeagueCalendar:
    """Read the calendar off ``settings.scheduleSettings`` + ``status``, defaults where ESPN is silent."""
    data = _mapping(league_json)
    sched = _mapping(_mapping(data.get("settings")).get("scheduleSettings"))
    status = _mapping(data.get("status"))

    reg = _int(sched.get("matchupPeriodCount")) or 14
    length = max(1, _int(sched.get("matchupPeriodLength"), 1) or 1)
    periods: dict[int, list[int]] = {}
    for k, v in _mapping(sched.get("matchupPeriods")).items():
        p = _int(k)
        if p is not None and isinstance(v, (list, tuple)):
            weeks = [w for w in (_int(x) for x in v) if w is not None]
            if weeks:
                periods[p] = weeks

    # the week ESPN itself considers current; `scoringPeriodId` is the live one, and it is the field
    # that moves on a Tuesday, so it wins over currentMatchupPeriod when both are present
    current = _int(data.get("scoringPeriodId")) or _int(status.get("currentMatchupPeriod")) or 1
    first = _int(status.get("firstScoringPeriod"), 1) or 1
    final = _int(status.get("finalScoringPeriod")) or _int(status.get("latestScoringPeriod")) or (reg + 3)
    return LeagueCalendar(
        current_week=max(first, min(int(current), int(final))),
        first_week=first, final_week=int(final),
        regular_season_weeks=reg, playoff_week_start=reg + 1,
        playoff_teams=_int(sched.get("playoffTeamCount"), 6) or 6,
        seeding_rule=sched.get("playoffSeedingRule") if isinstance(sched.get("playoffSeedingRule"), str) else None,
        matchup_period_length=length, periods=periods,
    )


def _team_name(team: Mapping[str, Any]) -> str | None:
    name = team.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    parts = [str(team.get(k) or "").strip() for k in ("location", "nickname")]
    joined = " ".join(p for p in parts if p)
    return joined or (str(team.get("abbrev")).strip() or None if team.get("abbrev") else None)


def team_records(league_json: Mapping[str, Any] | None) -> list[TeamRecord]:
    """Standings from ``mTeam``. Empty before any game is played, never invented."""
    out: list[TeamRecord] = []
    for team in _mapping(league_json).get("teams") or []:
        if not isinstance(team, Mapping):
            continue
        tid = _int(team.get("id"))
        if tid is None:
            continue
        overall = _mapping(_mapping(team.get("record")).get("overall"))
        sim = _mapping(team.get("currentSimulationResults"))
        out.append(TeamRecord(
            team_id=tid, name=_team_name(team),
            wins=_int(overall.get("wins"), 0) or 0, losses=_int(overall.get("losses"), 0) or 0,
            ties=_int(overall.get("ties"), 0) or 0,
            points_for=_float(overall.get("pointsFor")) or 0.0,
            points_against=_float(overall.get("pointsAgainst")) or 0.0,
            playoff_seed=_int(team.get("playoffSeed")),
            espn_playoff_pct=_float(sim.get("playoffPct")),
        ))
    return out


def matchups(league_json: Mapping[str, Any] | None, week: int | None = None) -> list[Matchup]:
    """The ``schedule`` array (view ``mMatchupScore`` / ``mScoreboard``), optionally one period only."""
    out: list[Matchup] = []
    for game in _mapping(league_json).get("schedule") or []:
        if not isinstance(game, Mapping):
            continue
        period = _int(game.get("matchupPeriodId"))
        if period is None or (week is not None and period != int(week)):
            continue
        home, away = _mapping(game.get("home")), _mapping(game.get("away"))
        out.append(Matchup(
            week=period,
            home_team_id=_int(home.get("teamId")), away_team_id=_int(away.get("teamId")),
            home_points=_float(home.get("totalPoints")) or 0.0,
            away_points=_float(away.get("totalPoints")) or 0.0,
            winner=game.get("winner") if isinstance(game.get("winner"), str) else None,
        ))
    return out


def opponent_for(games: Iterable[Matchup], team_id: int | None, week: int) -> int | None:
    """Team id ``team_id`` faces in matchup period ``week``; ``None`` on a bye or an unknown team."""
    if team_id is None:
        return None
    for g in games:
        if g.week == int(week):
            opp = g.opponent_of(int(team_id))
            if opp is not None:
                return opp
    return None


def weekly_player_points(league_json: Mapping[str, Any] | None) -> dict[str, dict[int, dict[str, float]]]:
    """``{espn player id: {week: {"actual": x, "projected": y}}}`` from the roster payload.

    ``appliedTotal`` is already scored under this league's own rules - custom scoring, TE premium,
    six-point passing touchdowns and all - so it is taken verbatim rather than re-derived. A season
    total block (``statSplitTypeId`` 0, no scoring period) is skipped: only per-week entries are kept.
    """
    out: dict[str, dict[int, dict[str, float]]] = {}
    for team in _mapping(league_json).get("teams") or []:
        entries = _mapping(_mapping(team).get("roster")).get("entries") or []
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            player = _mapping(_mapping(entry.get("playerPoolEntry")).get("player"))
            pid = entry.get("playerId") if entry.get("playerId") is not None else player.get("id")
            if pid is None:
                continue
            for stat in player.get("stats") or []:
                if not isinstance(stat, Mapping):
                    continue
                week = _int(stat.get("scoringPeriodId"))
                total = _float(stat.get("appliedTotal"))
                source = _int(stat.get("statSourceId"))
                if not week or total is None or source not in (ACTUAL, PROJECTED):
                    continue
                slot = out.setdefault(str(pid), {}).setdefault(int(week), {})
                slot["actual" if source == ACTUAL else "projected"] = float(total)
    return out


def points_to_date(weekly: Mapping[str, Mapping[int, Mapping[str, float]]], espn_id: Any,
                   through_week: int) -> tuple[float, int]:
    """``(actual points, weeks with a score)`` for one player up to and including ``through_week``.

    A week with no ``actual`` entry is a week not yet played, not a zero: counting it as a zero is how
    a healthy player in week 2 ends up looking like a bust.
    """
    total, weeks = 0.0, 0
    for week, block in (weekly.get(str(espn_id)) or {}).items():
        if int(week) <= int(through_week) and "actual" in block:
            total += float(block["actual"])
            weeks += 1
    return total, weeks


def current_lineups(league_json: Mapping[str, Any] | None) -> dict[int, dict[str, list[str]]]:
    """``{team id: {"starters": [espn id], "bench": [...], "ir": [...]}}`` as the owner has it set.

    This is what a start/sit recommendation has to be compared against: the gap between the lineup
    that is set and the best one is the only part of the advice that is actionable.
    """
    out: dict[int, dict[str, list[str]]] = {}
    for team in _mapping(league_json).get("teams") or []:
        tid = _int(_mapping(team).get("id"))
        if tid is None:
            continue
        block = out.setdefault(tid, {"starters": [], "bench": [], "ir": []})
        for entry in _mapping(_mapping(team).get("roster")).get("entries") or []:
            if not isinstance(entry, Mapping):
                continue
            pid = entry.get("playerId")
            if pid is None:
                pid = _mapping(_mapping(entry.get("playerPoolEntry")).get("player")).get("id")
            if pid is None:
                continue
            slot = _int(entry.get("lineupSlotId"))
            where = "ir" if slot == IR_SLOT_ID else ("bench" if slot == BENCH_SLOT_ID or slot is None else "starters")
            block[where].append(str(pid))
    return out
