"""Core data models shared by every layer.

These are plain dataclasses (no I/O, no pandas) so they are cheap to construct
inside the 2-second poll loop and trivial to serialise for fixtures.

Identity: every player is keyed by their **Sleeper player_id** (a numeric string
like "4034", or a team abbreviation like "SF" for team defenses). Historical
data keyed by other ids (nflverse gsis_id, FantasyPros id) is mapped onto the
Sleeper id by ``draftadvisor.data.crosswalk``.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Iterable, Iterator

from .config import (
    FANTASY_WEEKS,
    NON_STARTING_SLOTS,
    SKILL_POSITIONS,
    SLOT_ELIGIBILITY,
)

# ---------------------------------------------------------------------------
# Players
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Player:
    """A draftable player as known at draft time."""

    player_id: str
    name: str
    position: str                               # primary fantasy position (one of SKILL_POSITIONS)
    team: str | None = None                     # Sleeper team abbreviation ("KC", "LAR", ...) or None (FA)
    fantasy_positions: tuple[str, ...] = ()
    age: float | None = None
    years_exp: int | None = None
    injury_status: str | None = None            # "IR" | "Out" | "Doubtful" | "Questionable" | "PUP" | "Sus" | "NA" | None
    status: str | None = None                   # Sleeper status: "Active" | "Inactive" | "Injured Reserve" | ...
    depth_chart_order: int | None = None        # 1 = starter
    depth_chart_position: str | None = None
    bye_week: int | None = None
    search_rank: int | None = None              # Sleeper popularity rank (lower is better); 9999999 = unranked
    gsis_id: str | None = None                  # nflverse id "00-00xxxxx"
    fantasypros_id: str | None = None
    espn_id: str | None = None                  # ESPN fantasy player id (str; team defenses map by pro-team abbreviation)
    draft_year: int | None = None
    draft_round: int | None = None
    draft_pick_overall: int | None = None
    adp: float | None = None                    # overall ADP in the league's scoring format (lower is earlier)
    adp_source: str | None = None               # "sleeper_ppr" | "sleeper_half_ppr" | "ecr" | "search_rank"
    ecr: float | None = None                    # FantasyPros expert consensus overall rank
    ecr_sd: float | None = None
    ecr_pos_rank: float | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def is_rookie(self) -> bool:
        """True only when Sleeper reports ``years_exp == 0``; unknown (``None``, e.g. team defenses) is not a rookie."""
        return self.years_exp == 0

    @property
    def is_defense(self) -> bool:
        return self.position == "DEF"

    def display(self) -> str:
        t = self.team or "FA"
        return f"{self.name} ({self.position}, {t})"


# ---------------------------------------------------------------------------
# League / draft settings
# ---------------------------------------------------------------------------


@dataclass
class LeagueSettings:
    """Subset of a Sleeper league object the advisor needs."""

    league_id: str
    name: str
    season: int
    total_rosters: int
    roster_positions: list[str]                 # e.g. ["QB","RB","RB","WR","WR","TE","FLEX","K","DEF","BN",...]
    scoring_settings: dict[str, float]          # Sleeper scoring keys -> points per unit
    settings: dict = field(default_factory=dict)
    draft_id: str | None = None
    status: str | None = None                   # "pre_draft" | "drafting" | "in_season" | ...
    raw: dict = field(default_factory=dict)

    # -- scoring format -----------------------------------------------------
    @property
    def rec_points(self) -> float:
        return float(self.scoring_settings.get("rec", 0.0) or 0.0)

    @property
    def scoring_type(self) -> str:
        """"ppr" | "half_ppr" | "std" (rounded from the rec setting)."""
        r = self.rec_points
        if r >= 0.75:
            return "ppr"
        if r >= 0.25:
            return "half_ppr"
        return "std"

    @property
    def is_superflex(self) -> bool:
        return "SUPER_FLEX" in self.roster_positions or self.roster_positions.count("QB") >= 2

    @property
    def te_premium(self) -> float:
        return float(self.scoring_settings.get("bonus_rec_te", 0.0) or 0.0)

    # -- roster shape ---------------------------------------------------------
    @property
    def starting_slots(self) -> list[str]:
        return [s for s in self.roster_positions if s not in NON_STARTING_SLOTS]

    @property
    def bench_slots(self) -> int:
        return sum(1 for s in self.roster_positions if s == "BN")

    @property
    def roster_size(self) -> int:
        return sum(1 for s in self.roster_positions if s not in ("IR", "TAXI"))

    def slot_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self.roster_positions:
            out[s] = out.get(s, 0) + 1
        return out

    def dedicated_starters(self, position: str) -> int:
        """Number of starting slots that can *only* hold ``position``."""
        return sum(1 for s in self.starting_slots if SLOT_ELIGIBILITY.get(s) == frozenset({position}))

    def flex_slots_for(self, position: str) -> int:
        """Number of multi-position starting slots this position is eligible for."""
        return sum(
            1
            for s in self.starting_slots
            if position in SLOT_ELIGIBILITY.get(s, frozenset()) and len(SLOT_ELIGIBILITY[s]) > 1
        )


@dataclass
class Manager:
    user_id: str
    display_name: str
    team_name: str | None = None
    slot: int | None = None                     # 1-based draft slot
    roster_id: int | None = None
    avatar: str | None = None


@dataclass
class DraftSettings:
    """Subset of a Sleeper draft object plus pick-order math."""

    draft_id: str
    league_id: str | None
    type: str                                   # "snake" | "linear" | "auction"
    status: str                                 # "pre_draft" | "drafting" | "paused" | "complete"
    teams: int
    rounds: int
    pick_timer: int = 0                         # seconds per pick (0 = no timer)
    reversal_round: int = 0                     # 0 = plain snake; 3 = third-round reversal
    player_type: int = 0                        # Sleeper settings.player_type: 0 all players, 1 rookies only, 2 vets only
    draft_order: dict[str, int] = field(default_factory=dict)      # user_id -> slot (1-based)
    slot_to_roster_id: dict[int, int] = field(default_factory=dict)  # slot -> roster_id
    traded_picks: dict[tuple[int, int], int] = field(default_factory=dict)  # (round, original_roster_id) -> owner roster_id
    season: int | None = None
    scoring_type: str | None = None             # from draft metadata ("ppr", "half_ppr", "std", "dynasty_2qb"...)
    start_time: int | None = None               # epoch ms
    last_picked: int | None = None              # epoch ms
    settings: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    # -- pick-order math ------------------------------------------------------
    @property
    def total_picks(self) -> int:
        return self.teams * self.rounds

    def round_of(self, pick_no: int) -> int:
        return (pick_no - 1) // self.teams + 1

    def _forward(self, rnd: int) -> bool:
        if self.type == "linear":
            return True
        forward = rnd % 2 == 1
        if self.reversal_round and rnd >= self.reversal_round:
            forward = not forward
        return forward

    def slot_for_pick(self, pick_no: int) -> int:
        """1-based draft slot that is on the clock for overall pick ``pick_no``."""
        rnd = self.round_of(pick_no)
        idx = (pick_no - 1) % self.teams
        return idx + 1 if self._forward(rnd) else self.teams - idx

    def pick_no_for(self, rnd: int, slot: int) -> int:
        idx = slot - 1 if self._forward(rnd) else self.teams - slot
        return (rnd - 1) * self.teams + idx + 1

    def original_roster_for_slot(self, slot: int) -> int | None:
        return self.slot_to_roster_id.get(slot)

    def owner_roster_for_pick(self, pick_no: int) -> int | None:
        """Roster that actually makes ``pick_no`` after accounting for traded picks."""
        slot = self.slot_for_pick(pick_no)
        orig = self.original_roster_for_slot(slot)
        if orig is None:
            return None
        return self.traded_picks.get((self.round_of(pick_no), orig), orig)

    def picks_owned_by_roster(self, roster_id: int) -> list[int]:
        return [p for p in range(1, self.total_picks + 1) if self.owner_roster_for_pick(p) == roster_id]

    def picks_for_slot(self, slot: int) -> list[int]:
        """Overall pick numbers for the roster originally in ``slot`` (traded picks respected
        when the slot->roster mapping is known, else plain snake order)."""
        rid = self.original_roster_for_slot(slot)
        if rid is not None and self.slot_to_roster_id:
            return self.picks_owned_by_roster(rid)
        return [self.pick_no_for(r, slot) for r in range(1, self.rounds + 1)]


@dataclass(slots=True)
class Pick:
    pick_no: int
    round: int
    draft_slot: int
    player_id: str
    roster_id: int | None = None
    picked_by: str | None = None                # user_id
    is_keeper: bool = False
    metadata: dict = field(default_factory=dict)

    @property
    def position(self) -> str | None:
        return self.metadata.get("position")

    @property
    def player_name(self) -> str:
        f, l = self.metadata.get("first_name"), self.metadata.get("last_name")
        return f"{f or ''} {l or ''}".strip() or self.player_id


# ---------------------------------------------------------------------------
# Draft state (what the poller produces every tick)
# ---------------------------------------------------------------------------


@dataclass
class DraftState:
    """Everything known about the draft at one instant."""

    draft: DraftSettings
    picks: list[Pick]
    league: LeagueSettings | None = None
    managers: dict[str, Manager] = field(default_factory=dict)   # user_id -> Manager
    my_user_id: str | None = None
    my_slot: int | None = None
    updated_at: float = field(default_factory=time.time)
    version: int = 0                            # increments whenever picks change
    rostered_ids: set[str] = field(default_factory=set)  # players already on league rosters (dynasty/keeper): undraftable

    # -- basic derived values ---------------------------------------------------
    @property
    def teams(self) -> int:
        return self.draft.teams

    @property
    def drafted_ids(self) -> set[str]:
        return {p.player_id for p in self.picks}

    @property
    def unavailable_ids(self) -> set[str]:
        """Players nobody can draft: already picked (incl. keepers on the board) or on a league roster."""
        return self.drafted_ids | self.rostered_ids

    @property
    def is_rookie_draft(self) -> bool:
        return self.draft.player_type == 1

    @property
    def taken_pick_numbers(self) -> set[int]:
        """Pick numbers already on the board (made picks *and* pre-populated keeper picks)."""
        return {p.pick_no for p in self.picks}

    @property
    def next_pick_no(self) -> int:
        """Overall number of the pick currently on the clock (1-based)."""
        taken = self.taken_pick_numbers
        n = 1
        while n in taken:
            n += 1
        return n

    @property
    def is_complete(self) -> bool:
        return self.draft.status == "complete" or self.next_pick_no > self.draft.total_picks

    @property
    def current_round(self) -> int:
        return min(self.draft.round_of(self.next_pick_no), self.draft.rounds)

    @property
    def on_the_clock_slot(self) -> int | None:
        if self.is_complete:
            return None
        return self.draft.slot_for_pick(self.next_pick_no)

    @property
    def on_the_clock_roster(self) -> int | None:
        if self.is_complete:
            return None
        return self.draft.owner_roster_for_pick(self.next_pick_no)

    @property
    def my_roster_id(self) -> int | None:
        if self.my_slot is None:
            return None
        return self.draft.original_roster_for_slot(self.my_slot)

    def my_pick_numbers(self) -> list[int]:
        if self.my_slot is None:
            return []
        return self.draft.picks_for_slot(self.my_slot)

    def my_future_picks(self) -> list[int]:
        """My pick numbers from the current pick on, excluding picks already on the board.

        Sleeper places keepers on the board before the draft as picks (``is_keeper``) at the
        pick number of their keeper round; those picks will never come on the clock.
        """
        n = self.next_pick_no
        taken = self.taken_pick_numbers
        return [p for p in self.my_pick_numbers() if p >= n and p not in taken]

    @property
    def my_next_pick_no(self) -> int | None:
        fut = self.my_future_picks()
        return fut[0] if fut else None

    @property
    def my_pick_after_next(self) -> int | None:
        fut = self.my_future_picks()
        return fut[1] if len(fut) > 1 else None

    @property
    def is_my_turn(self) -> bool:
        return not self.is_complete and self.my_next_pick_no == self.next_pick_no

    @property
    def picks_until_my_turn(self) -> int | None:
        nxt = self.my_next_pick_no
        return None if nxt is None else nxt - self.next_pick_no

    # -- rosters ----------------------------------------------------------------
    def slot_of_pick(self, pick: Pick) -> int:
        """Draft slot of the team that actually made ``pick``.

        Sleeper keeps ``draft_slot`` as the *original* slot of a traded pick while ``roster_id``
        identifies the drafter, so the roster mapping wins when it is known.
        """
        if pick.roster_id is not None:
            for s, rid in self.draft.slot_to_roster_id.items():
                if rid == pick.roster_id:
                    return s
        return pick.draft_slot

    def picks_by_slot(self) -> dict[int, list[Pick]]:
        out: dict[int, list[Pick]] = {s: [] for s in range(1, self.teams + 1)}
        for p in self.picks:
            # attribute the pick to whoever owns the roster that made it, when known
            out.setdefault(self.slot_of_pick(p), []).append(p)
        return out

    def my_picks(self) -> list[Pick]:
        if self.my_slot is None:
            return []
        return self.picks_by_slot().get(self.my_slot, [])

    def manager_for_slot(self, slot: int) -> Manager | None:
        for m in self.managers.values():
            if m.slot == slot:
                return m
        return None

    def slot_label(self, slot: int) -> str:
        m = self.manager_for_slot(slot)
        if m is None:
            return f"Slot {slot}"
        return m.team_name or m.display_name

    def with_picks(self, picks: list[Pick]) -> "DraftState":
        """Return a copy with a different pick list (used by the simulator/poller)."""
        return DraftState(
            draft=self.draft,
            picks=list(picks),
            league=self.league,
            managers=self.managers,
            my_user_id=self.my_user_id,
            my_slot=self.my_slot,
            updated_at=time.time(),
            version=self.version + 1,
            rostered_ids=set(self.rostered_ids),
        )


# ---------------------------------------------------------------------------
# Projections & values
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Projection:
    """Season projection for one player under the league's scoring."""

    player_id: str
    position: str
    points: float                   # expected season total (league scoring, regular season)
    std: float                      # 1-sigma uncertainty on the season total
    ppg: float                      # expected points per game played
    games: float                    # expected games played
    floor: float = 0.0              # ~20th percentile of season total
    ceiling: float = 0.0            # ~80th percentile of season total
    weekly: list[float] | None = None   # optional per-week expected points (index 0 = week 1; 0 on bye)
    stat_line: dict[str, float] = field(default_factory=dict)   # projected season stat totals (Sleeper keys)
    components: dict[str, float] = field(default_factory=dict)  # per-source season totals: {"ml":..,"sleeper":..,"ecr":..}
    weights: dict[str, float] = field(default_factory=dict)     # blend weights actually used
    flags: list[str] = field(default_factory=list)              # e.g. ["rookie", "injury:IR", "depth3"]

    def points_through(self, week: int) -> float:
        if self.weekly:
            return float(sum(self.weekly[:week]))
        return self.points * min(1.0, week / FANTASY_WEEKS)


@dataclass
class RosterSummary:
    """Roster composition for one team at one instant."""

    slot: int
    label: str
    players: list[Player]
    starters_filled: dict[str, int]     # slot label -> count filled (best-lineup assignment)
    open_starters: dict[str, int]       # slot label -> count still open
    position_counts: dict[str, int]     # position -> count on roster
    bye_weeks: dict[int, int]           # bye week -> number of starters on that bye
    lineup_points: float = 0.0          # projected optimal starting-lineup points
    bench_points: float = 0.0

    def needs(self) -> list[str]:
        """Positions with an open dedicated starting slot, most urgent first."""
        out = []
        for slot_label, n in self.open_starters.items():
            if n > 0 and slot_label in SKILL_POSITIONS:
                out.append(slot_label)
        return out


@dataclass
class PlayerValue:
    """A player's value in the context of the current draft state."""

    player: Player
    projection: Projection
    vorp: float                         # points over positional replacement level (static)
    vona: float                         # points over the best player expected to be available at my next pick
    marginal_value: float               # improvement to my optimal lineup (+ discounted bench value)
    score: float                        # final ranking score (need-adjusted, risk-adjusted)
    availability_next: float            # P(still available at my next pick)
    availability_after_next: float      # P(still available at the pick after that)
    tier: int                           # 1 = best tier within position
    pos_rank: int                       # rank among undrafted players at the position (by projection)
    overall_rank: int                   # rank among all undrafted players (by score)
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    sim_delta: float | None = None      # optional Monte-Carlo expected roster delta vs. best alternative

    @property
    def player_id(self) -> str:
        return self.player.player_id


@dataclass
class PositionAdvice:
    position: str
    candidates: list[PlayerValue]       # top-N (3) at this position
    action: str                         # "TAKE NOW" | "SOON" | "WAIT" | "SKIP"
    rationale: str                      # one line, e.g. "84% chance Player X or better is there at pick 41"
    expected_next_available: float      # expected points of best available at my next pick
    drop_off: float                     # best now minus expected next available


@dataclass
class Recommendation:
    """Output of the strategy engine for one draft state."""

    state_version: int
    computed_at: float
    compute_ms: float
    best_overall: list[PlayerValue]                 # top-N across positions, by score
    by_position: dict[str, PositionAdvice]          # position -> advice
    my_roster: RosterSummary | None
    opponent_rosters: list[RosterSummary]
    position_pressure: dict[str, float]             # position -> expected # of picks at that position before my next pick
    notes: list[str] = field(default_factory=list)  # global notes ("RB run in progress", "K/DEF too early")
    claude_advice: str | None = None                # filled asynchronously by research layer

    @property
    def top_pick(self) -> PlayerValue | None:
        return self.best_overall[0] if self.best_overall else None


# ---------------------------------------------------------------------------
# Research notes (Claude)
# ---------------------------------------------------------------------------


@dataclass
class ResearchNote:
    player_id: str
    summary: str
    injury_risk: float                  # 0..1
    role_certainty: float               # 0..1
    upside: str
    downside: str
    sources: list[str] = field(default_factory=list)
    generated_at: float = field(default_factory=time.time)
    model: str | None = None
    offfield_risk: float = 0.0          # 0..1: suspension / legal / holdout risk of missing games
    red_flags: list[str] = field(default_factory=list)   # short, concrete concerns found in the news

    def to_dict(self) -> dict:
        return {
            "player_id": self.player_id,
            "summary": self.summary,
            "injury_risk": self.injury_risk,
            "role_certainty": self.role_certainty,
            "upside": self.upside,
            "downside": self.downside,
            "sources": list(self.sources),
            "generated_at": self.generated_at,
            "model": self.model,
            "offfield_risk": self.offfield_risk,
            "red_flags": list(self.red_flags),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ResearchNote":
        return cls(
            player_id=str(d["player_id"]),
            summary=d.get("summary", ""),
            injury_risk=float(d.get("injury_risk", 0.0)),
            role_certainty=float(d.get("role_certainty", 0.5)),
            upside=d.get("upside", ""),
            downside=d.get("downside", ""),
            sources=list(d.get("sources", [])),
            generated_at=float(d.get("generated_at", 0.0)),
            model=d.get("model"),
            offfield_risk=float(d.get("offfield_risk", 0.0) or 0.0),
            red_flags=[str(x) for x in (d.get("red_flags") or []) if x],
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_ppf(p: float) -> float:
    """Inverse normal CDF (Acklam's approximation, accurate to ~1e-9)."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


def iter_positions(positions: Iterable[str] | None = None) -> Iterator[str]:
    yield from (positions or SKILL_POSITIONS)
