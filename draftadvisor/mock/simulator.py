"""Offline mock-draft simulator (DESIGN.md §3.2).

Bots imitate real Sleeper drafters: they mostly follow ADP (with noise that grows with ADP),
each has a :class:`BotStrategy` (zero-RB, hero-RB, early/late QB, balanced) that shifts its
position preferences, they fill open starting slots as the draft goes on, take exactly one K and
one DEF in the last three rounds, avoid a 3rd QB (2nd unless superflex) and never exceed sensible
depth at a position.

Players come from the offline universe with ``Player.adp`` / ``Player.ecr`` populated; a player
with neither is treated as ADP 400.
"""
from __future__ import annotations

import logging
import random
import time
from enum import Enum
from typing import Callable, Iterable

from ..config import DEFAULT_SEASON, SKILL_POSITIONS
from ..models import DraftSettings, DraftState, LeagueSettings, Manager, Pick, Player

log = logging.getLogger(__name__)

__all__ = ["BotStrategy", "MockDraft", "make_mock_league", "make_mock_draft", "player_rank"]

MY_USER_ID = "me"
_UNRANKED_ADP = 400.0
#: Bots consider the best N available by ADP (plus the top few at every position).
_CANDIDATE_POOL = 30
_PER_POSITION_POOL = 3
#: Bots take K/DEF only in the last this-many rounds.
_KDEF_LAST_ROUNDS = 3
_DEPTH_CAPS = {"QB": 2, "RB": 8, "WR": 8, "TE": 3, "K": 1, "DEF": 1}


class BotStrategy(str, Enum):
    """Position-preference archetypes for bots."""

    BALANCED = "balanced"
    ZERO_RB = "zero_rb"
    HERO_RB = "hero_rb"
    EARLY_QB = "early_qb"
    LATE_QB = "late_qb"


_SCORING_PRESETS: dict[str, dict[str, float]] = {
    "base": {
        "pass_yd": 0.04, "pass_td": 4.0, "pass_int": -1.0, "pass_2pt": 2.0,
        "rush_yd": 0.1, "rush_td": 6.0, "rush_2pt": 2.0,
        "rec_yd": 0.1, "rec_td": 6.0, "rec_2pt": 2.0,
        "fum_lost": -2.0, "fum_rec_td": 6.0, "st_td": 6.0,
        "fgm_0_19": 3.0, "fgm_20_29": 3.0, "fgm_30_39": 3.0, "fgm_40_49": 4.0, "fgm_50p": 5.0,
        "fgmiss": -1.0, "xpm": 1.0, "xpmiss": -1.0,
        "sack": 1.0, "int": 2.0, "ff": 1.0, "fum_rec": 2.0, "safe": 2.0, "blk_kick": 2.0,
        "def_td": 6.0, "def_st_td": 6.0,
        "pts_allow_0": 10.0, "pts_allow_1_6": 7.0, "pts_allow_7_13": 4.0, "pts_allow_14_20": 1.0,
        "pts_allow_21_27": 0.0, "pts_allow_28_34": -1.0, "pts_allow_35p": -4.0,
    },
    "ppr": {"rec": 1.0},
    "half_ppr": {"rec": 0.5},
    "std": {"rec": 0.0},
}


def player_rank(p: Player) -> float:
    """Market rank used by bots: ADP, else ECR, else 400."""
    if p.adp is not None and p.adp > 0:
        return float(p.adp)
    if p.ecr is not None and p.ecr > 0:
        return float(p.ecr)
    return _UNRANKED_ADP


# ---------------------------------------------------------------------------
# League / draft factories
# ---------------------------------------------------------------------------


def make_mock_league(teams: int = 12, rounds: int = 15, scoring: str = "half_ppr", superflex: bool = False,
                     te_premium: float = 0.0) -> LeagueSettings:
    """A synthetic league: QB,RB,RB,WR,WR,TE,FLEX,(SUPER_FLEX,)K,DEF + bench to fill ``rounds``."""
    if scoring not in ("ppr", "half_ppr", "std"):
        raise ValueError(f"scoring must be ppr|half_ppr|std, got {scoring!r}")
    starters = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"]
    if superflex:
        starters.append("SUPER_FLEX")
    starters += ["K", "DEF"]
    bench = max(0, rounds - len(starters))
    scoring_settings = dict(_SCORING_PRESETS["base"])
    scoring_settings.update(_SCORING_PRESETS[scoring])
    if te_premium:
        scoring_settings["bonus_rec_te"] = float(te_premium)
    return LeagueSettings(
        league_id="mock",
        name=f"Mock League ({teams} teams, {scoring}{', superflex' if superflex else ''})",
        season=DEFAULT_SEASON,
        total_rosters=teams,
        roster_positions=starters + ["BN"] * bench,
        scoring_settings=scoring_settings,
        settings={"num_teams": teams, "draft_rounds": rounds, "type": 0},
        draft_id="mock-draft",
        status="drafting",
    )


def make_mock_draft(league: LeagueSettings, my_slot: int, teams: int = 12, rounds: int = 15, pick_timer: int = 30,
                    reversal_round: int = 0) -> DraftSettings:
    """A snake draft whose ``draft_order`` maps ``"me"`` and ``"bot-<slot>"`` to slots (identity roster ids)."""
    if not 1 <= my_slot <= teams:
        raise ValueError(f"my_slot must be in 1..{teams}, got {my_slot}")
    draft_order = {(MY_USER_ID if s == my_slot else f"bot-{s}"): s for s in range(1, teams + 1)}
    return DraftSettings(
        draft_id=league.draft_id or "mock-draft",
        league_id=league.league_id,
        type="snake",
        status="drafting",
        teams=teams,
        rounds=rounds,
        pick_timer=pick_timer,
        reversal_round=reversal_round,
        draft_order=draft_order,
        slot_to_roster_id={s: s for s in range(1, teams + 1)},
        season=league.season,
        scoring_type=league.scoring_type,
        start_time=int(time.time() * 1000),
        settings={"teams": teams, "rounds": rounds, "pick_timer": pick_timer, "reversal_round": reversal_round},
        metadata={"scoring_type": league.scoring_type, "name": league.name},
    )


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


def _split_name(name: str) -> tuple[str, str]:
    parts = name.strip().split(" ", 1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


class MockDraft:
    """Drive a synthetic snake draft: bots pick via :meth:`bot_pick`, you via :meth:`make_pick`."""

    def __init__(self, players: dict[str, Player], league: LeagueSettings, draft: DraftSettings, my_slot: int,
                 seed: int | None = None, bot_noise: float = 0.15,
                 strategies: dict[int, BotStrategy] | None = None):
        self.players = {pid: p for pid, p in players.items() if p.position in SKILL_POSITIONS}
        self.league = league
        self.draft = draft
        self.my_slot = my_slot
        self.rng = random.Random(seed)
        self.bot_noise = bot_noise
        self.picks: list[Pick] = []
        self._drafted: set[str] = set()
        self._counts: dict[int, dict[str, int]] = {s: {} for s in range(1, draft.teams + 1)}
        self._ranked: list[Player] = sorted(self.players.values(), key=lambda p: (player_rank(p), p.player_id))
        self._positions_present: tuple[str, ...] = tuple(pos for pos in SKILL_POSITIONS
                                                         if any(p.position == pos for p in self._ranked))
        self._version = 0
        self._state: DraftState | None = None
        self.strategies: dict[int, BotStrategy] = {
            s: (strategies or {}).get(s) or self.rng.choice(list(BotStrategy))
            for s in range(1, draft.teams + 1)
        }
        self.managers: dict[str, Manager] = {
            uid: Manager(user_id=uid, display_name="You" if uid == MY_USER_ID else f"Bot {slot}",
                         team_name=None, slot=slot, roster_id=draft.slot_to_roster_id.get(slot, slot))
            for uid, slot in draft.draft_order.items()
        }
        self._dedicated = {pos: league.dedicated_starters(pos) for pos in SKILL_POSITIONS}
        self._flex_starters = sum(1 for s in league.starting_slots if s in ("FLEX", "WRRB_FLEX", "REC_FLEX"))
        self._superflex = league.is_superflex
        self._qb_cap = 3 if self._superflex else 2
        log.debug("mock draft: %d teams x %d rounds, %d players, strategies=%s", draft.teams, draft.rounds,
                  len(self.players), {s: v.value for s, v in self.strategies.items()})

    # -- state --------------------------------------------------------------
    def state(self) -> DraftState:
        """Current :class:`DraftState` (cached until the next pick)."""
        if self._state is None or self._state.version != self._version:
            self._state = DraftState(
                draft=self.draft, picks=list(self.picks), league=self.league, managers=self.managers,
                my_user_id=MY_USER_ID, my_slot=self.my_slot, version=self._version,
            )
        return self._state

    @property
    def next_pick_no(self) -> int:
        return len(self.picks) + 1

    @property
    def is_complete(self) -> bool:
        return self.next_pick_no > self.draft.total_picks

    @property
    def on_the_clock_slot(self) -> int | None:
        return None if self.is_complete else self.draft.slot_for_pick(self.next_pick_no)

    @property
    def is_my_turn(self) -> bool:
        return not self.is_complete and self.on_the_clock_slot == self.my_slot

    @property
    def current_round(self) -> int:
        return min(self.draft.round_of(self.next_pick_no), self.draft.rounds)

    def available(self) -> list[Player]:
        """Undrafted players sorted by market rank (best first)."""
        return [p for p in self._ranked if p.player_id not in self._drafted]

    def roster_counts(self, slot: int) -> dict[str, int]:
        return dict(self._counts.get(slot, {}))

    # -- picking ------------------------------------------------------------
    def _user_for_slot(self, slot: int) -> str | None:
        for uid, s in self.draft.draft_order.items():
            if s == slot:
                return uid
        return None

    def _record(self, player: Player, slot: int) -> Pick:
        pick_no = self.next_pick_no
        first, last = _split_name(player.name)
        pick = Pick(
            pick_no=pick_no,
            round=self.draft.round_of(pick_no),
            draft_slot=slot,
            player_id=player.player_id,
            roster_id=self.draft.slot_to_roster_id.get(slot, slot),
            picked_by=self._user_for_slot(slot),
            is_keeper=False,
            metadata={
                "first_name": first, "last_name": last, "position": player.position, "team": player.team,
                "player_id": player.player_id, "injury_status": player.injury_status,
                "years_exp": player.years_exp, "status": player.status, "sport": "nfl",
            },
        )
        self.picks.append(pick)
        self._drafted.add(player.player_id)
        counts = self._counts.setdefault(slot, {})
        counts[player.position] = counts.get(player.position, 0) + 1
        self._version += 1
        return pick

    def make_pick(self, player_id: str) -> Pick:
        """Draft ``player_id`` for whoever is on the clock (normally you). Raises ``ValueError`` if invalid."""
        if self.is_complete:
            raise ValueError("draft is complete")
        pid = str(player_id)
        player = self.players.get(pid)
        if player is None:
            raise ValueError(f"unknown player_id {pid!r}")
        if pid in self._drafted:
            raise ValueError(f"{player.name} ({pid}) is already drafted")
        slot = self.on_the_clock_slot
        assert slot is not None
        return self._record(player, slot)

    def bot_pick(self) -> Pick:
        """Let the bot policy pick for the slot on the clock."""
        if self.is_complete:
            raise ValueError("draft is complete")
        slot = self.on_the_clock_slot
        assert slot is not None
        player = self._choose(slot)
        pick = self._record(player, slot)
        log.debug("bot slot %d (%s) pick #%d: %s", slot, self.strategies[slot].value, pick.pick_no, player.display())
        return pick

    def advance_until_my_turn(self) -> DraftState:
        """Run bots until it is my turn (or the draft completes)."""
        while not self.is_complete and not self.is_my_turn:
            self.bot_pick()
        return self.state()

    def run_to_completion(self, my_policy: Callable[[DraftState], str]) -> DraftState:
        """Run the whole draft; ``my_policy(state)`` returns the player_id to take on my picks."""
        while not self.is_complete:
            if self.is_my_turn:
                self.make_pick(my_policy(self.state()))
            else:
                self.bot_pick()
        return self.state()

    # -- bot policy ---------------------------------------------------------
    def _candidates(self) -> list[Player]:
        """Top of the board by market rank plus the best few at every position."""
        out: list[Player] = []
        per_pos: dict[str, int] = {}
        for p in self._ranked:
            if p.player_id in self._drafted:
                continue
            n = per_pos.get(p.position, 0)
            if len(out) < _CANDIDATE_POOL or n < _PER_POSITION_POOL:
                out.append(p)
                per_pos[p.position] = n + 1
            if len(out) >= _CANDIDATE_POOL and all(per_pos.get(pos, 0) >= _PER_POSITION_POOL
                                                   for pos in self._positions_present):
                break
        return out

    def _choose(self, slot: int) -> Player:
        cands = self._candidates()
        if not cands:
            raise RuntimeError("no undrafted players left")
        rnd = self.current_round
        strategy = self.strategies[slot]
        counts = self._counts.get(slot, {})
        best: Player | None = None
        best_score = -float("inf")
        for p in cands:
            adj = self._position_adjustment(p.position, counts, rnd, strategy)
            if adj is None:
                continue
            mu = player_rank(p)
            noise = self.rng.gauss(0.0, self.bot_noise * max(4.0, 0.1 * mu))
            score = -mu + noise + adj
            if score > best_score:
                best, best_score = p, score
        if best is None:  # every candidate blocked by position rules: take the best-ranked one
            best = cands[0]
        return best

    def _position_adjustment(self, pos: str, counts: dict[str, int], rnd: int,
                             strategy: BotStrategy) -> float | None:
        """Pick-equivalent bonus (+) / penalty (-) for taking ``pos`` now; ``None`` = not allowed."""
        rounds = self.draft.rounds
        have = counts.get(pos, 0)
        rounds_left = rounds - rnd + 1
        if have >= _DEPTH_CAPS.get(pos, 8) and pos != "QB":
            return None
        adj = 0.0

        if pos in ("K", "DEF"):
            if have >= 1 or rnd <= rounds - _KDEF_LAST_ROUNDS:
                return None
            missing_kdef = sum(1 for x in ("K", "DEF") if counts.get(x, 0) == 0 and self._dedicated[x] > 0)
            if missing_kdef >= rounds_left:
                return 1000.0                      # must fill K/DEF with the picks left
            return 10.0 * (rnd - (rounds - _KDEF_LAST_ROUNDS))

        if pos == "QB":
            if have >= self._qb_cap:
                return None
            if self._superflex:
                adj += (25.0, 15.0, -20.0)[min(have, 2)]
            elif have == 1:
                adj -= 40.0 if rnd < rounds - 5 else 15.0
            if have == 0:
                if strategy is BotStrategy.EARLY_QB and rnd <= 4:
                    adj += 8.0 if rnd == 1 else 15.0
                elif strategy is BotStrategy.LATE_QB and rnd <= 7:
                    adj -= 8.0 if self._superflex else 25.0
        elif pos == "RB":
            if strategy is BotStrategy.ZERO_RB:
                adj += -30.0 if rnd <= 5 else (8.0 if rnd <= 10 else 0.0)
            elif strategy is BotStrategy.HERO_RB:
                if rnd == 1 and have == 0:
                    adj += 12.0
                elif 2 <= rnd <= 6:
                    adj -= 15.0
        elif pos == "TE":
            if have == 1:
                adj -= 5.0 if self.league.te_premium > 0 else 20.0
            elif have >= 2:
                adj -= 40.0

        # need-awareness: open dedicated starters get more attractive as the draft goes on
        open_starters = max(0, self._dedicated.get(pos, 0) - have)
        if open_starters and rnd >= 4:
            adj += min(30.0, open_starters * (3.0 + 2.0 * rnd))
        if pos in ("RB", "WR", "TE") and self._flex_starters and rnd >= 6:
            flex_pool = sum(counts.get(x, 0) for x in ("RB", "WR", "TE"))
            flex_need = sum(self._dedicated[x] for x in ("RB", "WR", "TE")) + self._flex_starters
            if flex_pool < flex_need:
                adj += 5.0
        return adj


def strategies_summary(draft: MockDraft) -> dict[int, str]:
    """``{slot: strategy}`` for display."""
    return {s: v.value for s, v in draft.strategies.items()}


def players_by_position(players: Iterable[Player]) -> dict[str, list[Player]]:
    """Group players by position, best market rank first (helper for the CLI/tests)."""
    out: dict[str, list[Player]] = {}
    for p in sorted(players, key=player_rank):
        out.setdefault(p.position, []).append(p)
    return out
