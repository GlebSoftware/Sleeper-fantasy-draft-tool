"""The weekly view of a season projection: who plays whom, and how much one week actually swings.

A season projection answers "how good is this player". Every in-season question - start or sit, who
wins this matchup, is this trade worth it in week 9 - needs the weekly view instead, and the two
differ in a way that is easy to get wrong:

``Projection.std`` is the uncertainty of the **season total**, and the per-game figure behind it
(``ppg_std_ppr``) is the error of the *season ppg forecast*. Neither is how much a player's score
moves week to week. Measured on 2019-2025, the real weekly spread is about 1.3x that figure for a
typical starter and 2.3x for a good receiver. A matchup simulated on the smaller number reports
80% when the truth is nearer 65%, which is exactly the sort of confident wrong answer that gets a
lineup set badly.

So weekly spread comes from :mod:`draftadvisor.data.inseason`, measured within season on actual
weekly scores and shipped in the bundle. This module reads those tables. No pandas: it runs in the
deployed runtime.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from ..config import FANTASY_WEEKS
from ..models import Player, Projection

log = logging.getLogger(__name__)

__all__ = ["InSeasonTables", "week_mean", "week_sigma", "lineup_mean_sigma", "win_probability"]

#: A weekly coefficient of variation outside this range is a data problem, not a player.
CV_RANGE = (0.25, 1.60)
#: Fallback when a position is missing from the fitted curve entirely.
DEFAULT_CURVE = (2.5, 0.45)


@dataclass
class InSeasonTables:
    """The bundle's ``inseason.json``, or empty when the bundle predates it."""

    season: int = 0
    dvp_season: int = 0
    sigma: dict[str, dict[str, float]] = field(default_factory=dict)      # player_id -> {sd, ppg, weeks}
    curve: dict[str, list[float]] = field(default_factory=dict)           # position -> [a, b]
    schedule: dict[str, dict[str, dict]] = field(default_factory=dict)    # team -> week -> {opp, home}
    dvp: dict[str, dict[str, float]] = field(default_factory=dict)        # defence -> position -> multiplier

    @property
    def present(self) -> bool:
        return bool(self.schedule or self.sigma)

    @classmethod
    def from_dict(cls, blob: Mapping[str, Any] | None) -> "InSeasonTables":
        if not blob:
            return cls()
        sig = blob.get("sigma") or {}
        return cls(season=int(blob.get("season") or 0), dvp_season=int(blob.get("dvp_season") or 0),
                   sigma=dict(sig.get("players") or {}), curve=dict(sig.get("curve") or {}),
                   schedule=dict(blob.get("schedule") or {}), dvp=dict(blob.get("dvp") or {}))

    # -- schedule ------------------------------------------------------------------
    def opponent(self, team: str | None, week: int) -> str | None:
        """Defence ``team`` faces in ``week``; ``None`` on a bye or an unknown team."""
        if not team:
            return None
        game = (self.schedule.get(team) or {}).get(str(int(week)))
        return (game or {}).get("opp")

    def is_bye(self, team: str | None, week: int) -> bool:
        """True only when the schedule is known and shows no game: an unknown team is not a bye."""
        return bool(team) and team in self.schedule and str(int(week)) not in self.schedule[team]

    def matchup_multiplier(self, team: str | None, position: str | None, week: int) -> float:
        opp = self.opponent(team, week)
        if not opp or not position:
            return 1.0
        return float((self.dvp.get(opp) or {}).get(position, 1.0))


def _cv(tables: InSeasonTables, player: Player, proj: Projection) -> float:
    """Weekly spread as a fraction of weekly mean - scale-free, so it survives league scoring.

    The tables are measured in PPR points; a league with 6-point passing touchdowns moves a
    quarterback's mean and his spread together, so the ratio carries over where the raw number
    would not.
    """
    row = tables.sigma.get(player.player_id)
    if row and float(row.get("ppg") or 0) > 1.0:
        cv = float(row["sd"]) / float(row["ppg"])
    else:
        a, b = tables.curve.get(player.position or "", DEFAULT_CURVE)
        ppg = max(1.0, float(proj.ppg or 0.0))
        cv = (float(a) + float(b) * ppg) / ppg
    return min(max(cv, CV_RANGE[0]), CV_RANGE[1])


def week_mean(proj: Projection, player: Player, tables: InSeasonTables, week: int,
              season_weeks: int = FANTASY_WEEKS) -> float:
    """Expected points in ``week``: zero on a bye, else ppg adjusted for the defence and for how
    often the projection expects him to be active at all."""
    if proj is None:
        return 0.0
    if tables.is_bye(player.team, week) or (player.bye_week and int(player.bye_week) == int(week)):
        return 0.0
    playable = max(1, season_weeks - (1 if player.bye_week else 0))
    availability = min(1.0, float(proj.games or playable) / playable)
    return float(proj.ppg or 0.0) * availability * tables.matchup_multiplier(player.team, player.position, week)


def week_sigma(proj: Projection, player: Player, tables: InSeasonTables, week: int | None = None,
               season_weeks: int = FANTASY_WEEKS) -> float:
    """1-sigma of a single week's score, in league points. Zero on a bye (a bye is not uncertain)."""
    if proj is None:
        return 0.0
    if week is not None and week_mean(proj, player, tables, week, season_weeks) <= 0.0:
        return 0.0
    return float(proj.ppg or 0.0) * _cv(tables, player, proj)


def lineup_mean_sigma(starters: Sequence[tuple[Player, Projection]], tables: InSeasonTables, week: int,
                      season_weeks: int = FANTASY_WEEKS) -> tuple[float, float]:
    """``(mean, sigma)`` of a starting lineup's weekly total.

    Players are added independently: real scores correlate a little (a quarterback and his receiver,
    a defence and its own offence), so this understates the spread of a stacked lineup slightly. It
    is stated rather than hidden, and stacking is the one case where it matters.
    """
    mean = 0.0
    var = 0.0
    for pl, pr in starters:
        mean += week_mean(pr, pl, tables, week, season_weeks)
        var += week_sigma(pr, pl, tables, week, season_weeks) ** 2
    return mean, math.sqrt(var)


def win_probability(mine: tuple[float, float], theirs: tuple[float, float]) -> float:
    """P(my total > their total) for two independent normal lineups.

    A model estimate, not a market price: fantasy weekly scores are right-skewed, so a heavy
    favourite is slightly less certain than this says.
    """
    dm = mine[0] - theirs[0]
    sd = math.sqrt(mine[1] ** 2 + theirs[1] ** 2)
    if sd <= 0:
        return 1.0 if dm > 0 else (0.0 if dm < 0 else 0.5)
    return 0.5 * (1.0 + math.erf(dm / (sd * math.sqrt(2.0))))
