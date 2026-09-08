"""The advisor: value every available player for the current draft state (DESIGN.md §3.4).

Everything heavy is precomputed in :class:`Advisor.__init__` as numpy arrays sorted
by projected points, so :meth:`Advisor.recommend` is a handful of vectorised
passes over the ~400 relevant available players plus a dozen tiny lineup solves.

Scoring of a candidate (higher is better)::

    score = marginal lineup value (over replacement-level stand-ins, + discounted bench value)
          + 0.5 * max(0, VONA)          VONA = points - E[best at the position at my next pick]
          - risk_aversion * std
          + stack bonus - bye penalty - K/DEF-too-early penalty
          - 0.3 * max(0, VONA) if he is very likely (>= 85%) to be there next time
          + 0.25 * max(0, VONA) if I need the position and he probably won't be
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ..config import SKILL_POSITIONS, SLOT_ELIGIBILITY, Settings
from ..models import (
    DraftState,
    LeagueSettings,
    Player,
    PlayerValue,
    PositionAdvice,
    Projection,
    Recommendation,
    RosterSummary,
)
from .availability import (
    adp_baseline,
    expected_best_available_array,
    pick_distribution_arrays,
    pick_mean,
    position_pressure,
    prob_available_array,
    shift_for_pressure,
    slot_for_pick,
)
from .lineup import (
    bench_depth_factor,
    bench_usefulness,
    is_phantom,
    optimal_lineup,
    phantom_starters,
    player_from_pick,
    starter_displacements,
    summarize_roster,
)
from .replacement import remaining_demand, starter_demand

log = logging.getLogger(__name__)

__all__ = ["Advisor", "ACTIONS", "future_picks"]


def future_picks(state: DraftState) -> list[int]:
    """My pick numbers still to be made, in order.

    Like :meth:`DraftState.my_future_picks` but never lists a pick number that is
    already on the board (Sleeper pre-populates keeper picks at their pick number).
    """
    if state.my_slot is None:
        return []
    taken = {p.pick_no for p in state.picks}
    n = state.next_pick_no
    return [p for p in state.my_pick_numbers() if p >= n and p not in taken]

ACTIONS = ("TAKE NOW", "SOON", "WAIT", "SKIP")
_POS_INDEX = {p: i for i, p in enumerate(SKILL_POSITIONS)}
_TOP_BY_POINTS = 350
_ADP_CUTOFF = 250.0
_LIKELY_THERE = 0.85
_TIER_TOP_FRAC = 0.05
_TIER_STD_FRAC = 0.35
_LOOKAHEAD_TOP = 60
_DEFER_EPS = 0.05          # certainty premium: prefer the scarcer player when plans tie
_KDEF_EARLY_PENALTY = 0.25
_RUN_MIN = 4
_MAX_BYE = 24
_SKIP_MIN_VALUE = 8.0


@dataclass
class _Context:
    """Everything computed for one draft state (cached by state key)."""

    key: tuple
    current_pick: int
    current_round: int
    eval_pick: int | None                   # my next pick after the current one (None: this is my last pick)
    eval_after: int | None
    on_clock: bool                          # the current pick is mine
    avail: np.ndarray                       # master indices, points descending
    pos: np.ndarray                         # position index per available player
    points: np.ndarray
    rep: dict[str, float]
    vorp: np.ndarray
    tier: np.ndarray
    n_tiers: dict[str, int]
    pos_rank: np.ndarray
    p_next: np.ndarray
    p_after: np.ndarray
    marginal: np.ndarray
    enters: np.ndarray                      # bool: the candidate would start in my lineup right now
    vona: np.ndarray
    score: np.ndarray
    next_after: np.ndarray                  # expected marginal value of my next pick if I take this player now
    e_next_marg: np.ndarray                 # per position: expected best marginal value available at my next pick
    order: np.ndarray                       # indices into avail, score descending
    overall_rank: np.ndarray
    my_summary: RosterSummary | None
    opp_summaries: list[RosterSummary]
    pressure: dict[str, float]
    e_next: dict[str, float]
    best_now: dict[str, float]
    need_open: dict[str, bool]
    open_label: dict[str, str | None]
    bench_weight: dict[str, float]
    bye_starters: dict[int, list[str]]
    stack_qb: dict[str, str]                # team -> my QB name
    stack_pc: dict[str, str]                # team -> my WR/TE name
    kdef_early: bool
    kdef_round: int
    n_between: int
    open_slots: list[str]
    picks_spare: int
    idp_open: int                           # open IDP/unprojected starting slots I must still fill myself
    idp_slots: list[str]
    remaining: int                          # picks I still have (including the current one when on the clock)
    my_ids: set[str] = field(default_factory=set)


class Advisor:
    """Value players and recommend picks for a league + projection set."""

    def __init__(
        self,
        league: LeagueSettings,
        players: Mapping[str, Player],
        projections: Mapping[str, Projection],
        settings: Settings | None = None,
    ) -> None:
        self.league = league
        self.players = players
        self.projections = projections
        self.settings = settings or Settings()
        self._slots = league.starting_slots
        self._demand = starter_demand(league)
        self._has_kdef = league.dedicated_starters("K") > 0 or league.dedicated_starters("DEF") > 0
        self._useful = np.array([bench_usefulness(p, league) for p in SKILL_POSITIONS])
        self._ctx: _Context | None = None
        self._build_arrays()
        log.debug("Advisor ready: %d projected players, demand=%s", self.n, self._demand)

    # ------------------------------------------------------------------ setup
    def _build_arrays(self) -> None:
        ids = [
            pid for pid, proj in self.projections.items()
            if pid in self.players and (proj.position in _POS_INDEX or self.players[pid].position in _POS_INDEX)
        ]
        ids.sort(key=lambda pid: (-self.projections[pid].points, pid))
        self.ids: list[str] = ids
        self.n = len(ids)
        self._index: dict[str, int] = {pid: i for i, pid in enumerate(ids)}
        pos, pts, std, mu, ecr_sd, bye, team = [], [], [], [], [], [], []
        for pid in ids:
            pl, pr = self.players[pid], self.projections[pid]
            position = pr.position if pr.position in _POS_INDEX else pl.position
            pos.append(_POS_INDEX[position])
            pts.append(float(pr.points))
            std.append(float(pr.std))
            mu.append(pick_mean(pl))
            ecr_sd.append(float(pl.ecr_sd) if pl.ecr_sd else np.nan)
            bye.append(int(pl.bye_week) if pl.bye_week and 0 < pl.bye_week < _MAX_BYE else 0)
            team.append(pl.team or "")
        self._pos = np.array(pos, dtype=np.int64)
        self._points = np.array(pts, dtype=float)
        self._std = np.array(std, dtype=float)
        self._mu = np.array(mu, dtype=float)
        self._sigma = pick_distribution_arrays(self._mu, np.array(ecr_sd, dtype=float))
        self._bye = np.array(bye, dtype=np.int64)
        self._team = np.array(team, dtype=object)
        self._demand_arr = np.array([self._demand[p] for p in SKILL_POSITIONS])
        self._model_rank = self._value_rank()
        self._relevant = (np.arange(self.n) < _TOP_BY_POINTS) | (self._mu < _ADP_CUTOFF)
        # positions the league cannot start (no K or DEF slot) are worthless: keep them off the board
        self._relevant &= self._demand_arr[self._pos] > 0

    def _value_rank(self) -> np.ndarray:
        """1-based rank by points over full-pool replacement (comparable to an overall ECR)."""
        rep = np.zeros(len(SKILL_POSITIONS))
        for pname, pi in _POS_INDEX.items():
            ppts = self._points[self._pos == pi]
            if ppts.size:
                k = int(round(self._demand[pname]))
                rep[pi] = ppts[min(max(k, 0), ppts.size - 1)]
        self._full_rep = {p: float(rep[_POS_INDEX[p]]) for p in SKILL_POSITIONS}
        order = np.argsort(-(self._points - rep[self._pos]), kind="stable")
        rank = np.empty(self.n, dtype=np.int64)
        rank[order] = np.arange(1, self.n + 1)
        return rank

    # ------------------------------------------------------------ public API
    def recommend(self, state: DraftState, top_n: int = 6, per_position: int = 3) -> Recommendation:
        """Full recommendation for ``state`` (< 30 ms for 12 teams / 400 players)."""
        t0 = time.perf_counter()
        ctx = self._context(state)
        best_overall = [self._value(ctx, j) for j in ctx.order[:top_n]]
        by_position: dict[str, PositionAdvice] = {}
        for pos in SKILL_POSITIONS:
            pi = _POS_INDEX[pos]
            js = ctx.order[ctx.pos[ctx.order] == pi][:per_position]
            cands = [self._value(ctx, int(j)) for j in js]
            by_position[pos] = self._advice(ctx, pos, cands)
        rec = Recommendation(
            state_version=state.version,
            computed_at=time.time(),
            compute_ms=0.0,
            best_overall=best_overall,
            by_position=by_position,
            my_roster=ctx.my_summary,
            opponent_rosters=ctx.opp_summaries,
            position_pressure=dict(ctx.pressure),
            notes=self._notes(state, ctx),
        )
        rec.compute_ms = (time.perf_counter() - t0) * 1000.0
        return rec

    def available_players(self, state: DraftState) -> list[Player]:
        """Undrafted, projected players (relevance-filtered) by projected points."""
        ctx = self._context(state)
        return [self.players[self.ids[i]] for i in ctx.avail]

    def value_of(self, state: DraftState, player_id: str) -> PlayerValue | None:
        """Value of one available player in this state (None if drafted/unknown)."""
        ctx = self._context(state)
        i = self._index.get(player_id)
        if i is None:
            return None
        js = np.flatnonzero(ctx.avail == i)
        if js.size == 0:
            return None
        return self._value(ctx, int(js[0]))

    def explain_pick(self, state: DraftState, player_id: str) -> str:
        """Plain-text explanation of a candidate versus the top recommendation."""
        ctx = self._context(state)
        v = self.value_of(state, player_id)
        if v is None:
            pl = self.players.get(player_id)
            name = pl.name if pl else player_id
            if player_id in (getattr(state, "unavailable_ids", None) or state.drafted_ids):
                return f"{name} has already been drafted."
            return f"{name}: no projection available, cannot value."
        top = self._value(ctx, int(ctx.order[0])) if ctx.order.size else None
        parts = [
            f"{v.player.display()}: score {v.score:.1f}, #{v.overall_rank} overall, "
            f"{v.player.position}{v.pos_rank} on the board (tier {v.tier}).",
            "; ".join(v.reasons) + ".",
        ]
        if v.warnings:
            parts.append("Warnings: " + "; ".join(v.warnings) + ".")
        if top is not None and top.player_id != player_id:
            when = (f"{top.availability_next:.0%} available at #{ctx.eval_pick}" if ctx.eval_pick is not None
                    else "this is your last pick")
            parts.append(
                f"Top recommendation is {top.player.display()} (score {top.score:.1f}, {when}): "
                + "; ".join(top.reasons[:2]) + "."
            )
            parts.append(f"Score gap {top.score - v.score:.1f} in favour of {top.player.name}.")
        elif top is not None:
            parts.append("This is the top recommendation.")
        return " ".join(parts)

    # --------------------------------------------------------------- context
    @staticmethod
    def _state_key(state: DraftState) -> tuple:
        last = state.picks[-1].pick_no if state.picks else 0
        return (id(state.draft), state.version, len(state.picks), last, state.my_slot)

    def _context(self, state: DraftState) -> _Context:
        key = self._state_key(state)
        if self._ctx is not None and self._ctx.key == key:
            return self._ctx
        self._ctx = self._build_context(state, key)
        return self._ctx

    def _position_of(self, pk) -> str | None:
        pr = self.projections.get(pk.player_id)
        if pr is not None and pr.position in _POS_INDEX:
            return pr.position
        pl = self.players.get(pk.player_id)
        if pl is not None:
            return pl.position
        return pk.position

    def _build_context(self, state: DraftState, key: tuple) -> _Context:
        s = self.settings
        current = state.next_pick_no
        rounds = max(1, state.draft.rounds)
        current_round = state.current_round
        kdef_round = max(1, rounds - int(s.kicker_def_min_round_from_end) + 1)
        kdef_early = self._has_kdef and current_round < kdef_round
        my_slot = state.my_slot
        fut_all = future_picks(state)
        on_clock = bool(fut_all) and fut_all[0] == current and not state.is_complete
        fut = [p for p in fut_all if p > current]
        if my_slot is None:
            # unknown slot: judge availability a full round ahead
            eval_pick: int | None = current + state.teams
            eval_after: int | None = eval_pick + state.teams
            remaining = rounds
        else:
            eval_pick = fut[0] if fut else None
            eval_after = None if eval_pick is None else (fut[1] if len(fut) > 1 else eval_pick + state.teams)
            remaining = len(fut_all)

        # 1. available pool (+ how many players each position has already lost to the draft)
        mask = self._relevant.copy()
        drafted_counts = {p: 0 for p in SKILL_POSITIONS}
        for pk in state.picks:
            i = self._index.get(pk.player_id)
            if i is not None:
                mask[i] = False
                drafted_counts[SKILL_POSITIONS[int(self._pos[i])]] += 1
            else:
                pname = self._position_of(pk)
                if pname in drafted_counts:
                    drafted_counts[pname] += 1
        # dynasty / keeper leagues - and an ESPN draft whose board is empty while its rosters fill:
        # a player on a league roster cannot be drafted, and one who is not on the board is still gone
        # from his position's pool, so positional scarcity must count him like a pick
        board_ids = state.drafted_ids
        roster_only = {pid for pid in (getattr(state, "roster_only_ids", None) or ()) if pid not in board_ids}
        for pid in getattr(state, "rostered_ids", None) or ():
            i = self._index.get(pid)
            if i is not None:
                mask[i] = False
        for pid in roster_only:
            i = self._index.get(pid)
            pname = SKILL_POSITIONS[int(self._pos[i])] if i is not None else None
            if pname is None:
                pl = self.players.get(pid)
                pname = pl.position if pl is not None else None
            if pname in drafted_counts:
                drafted_counts[pname] += 1
        # rookie-only (1) / veterans-only (2) drafts (Sleeper draft.settings.player_type)
        ptype = int(getattr(state.draft, "player_type", 0) or 0)
        if ptype in (1, 2):
            for i in np.flatnonzero(mask):
                pl = self.players.get(self.ids[i])
                rookie = bool(pl is not None and pl.years_exp == 0)
                if (ptype == 1 and not rookie) or (ptype == 2 and rookie):
                    mask[i] = False
        avail = np.flatnonzero(mask)
        pos = self._pos[avail]
        pts = self._points[avail]
        std = self._std[avail]
        m = avail.size

        # 2. replacement, VORP, tiers, positional rank
        total_picks = max(1, state.draft.total_picks)
        rem_frac = max(0.0, total_picks - current + 1) / total_picks
        rep_arr = np.zeros(len(SKILL_POSITIONS))
        tier = np.ones(m, dtype=np.int64)
        pos_rank = np.ones(m, dtype=np.int64)
        n_tiers: dict[str, int] = {}
        best_now: dict[str, float] = {}
        for pname, pi in _POS_INDEX.items():
            sel = np.flatnonzero(pos == pi)
            n_tiers[pname] = 0
            best_now[pname] = 0.0
            if sel.size == 0:
                continue
            ppts = pts[sel]
            # the (remaining demand + 1)-th best still available: stays at the full-pool level
            # while the draft follows the demand model instead of sinking into bench filler
            k = int(round(remaining_demand(self._demand_arr[pi], drafted_counts[pname], rem_frac)))
            rep_arr[pi] = ppts[min(max(k, 0), sel.size - 1)]
            best_now[pname] = float(ppts[0])
            thr = max(_TIER_TOP_FRAC * ppts[0], _TIER_STD_FRAC * float(std[sel].mean()))
            breaks = (ppts[:-1] - ppts[1:]) > thr
            t = 1 + np.concatenate(([0], np.cumsum(breaks)))
            tier[sel] = t
            n_tiers[pname] = int(t[-1])
            pos_rank[sel] = np.arange(1, sel.size + 1)
        rep = {p: float(rep_arr[_POS_INDEX[p]]) for p in SKILL_POSITIONS}
        vorp = pts - rep_arr[pos]

        # 3. rosters, my lineup structure, pressure, availability
        by_slot = state.picks_by_slot()
        # players a team holds without a pick on the board (ESPN rosters); slot 0 = team unknown, never used
        spots_by_slot = state.roster_spots_by_slot() if hasattr(state, "roster_spots_by_slot") else {}
        summaries: dict[int, RosterSummary] = {}
        for slot in range(1, state.teams + 1):
            extra = [s.player_id for s in spots_by_slot.get(slot, ()) if s.player_id not in board_ids]
            summaries[slot] = summarize_roster(slot, state.slot_label(slot), by_slot.get(slot, []),
                                               self.players, self.projections, self.league, extra_ids=extra)
        my_summary = summaries.get(my_slot) if my_slot is not None else None
        opp_summaries = [sm for slot, sm in summaries.items() if slot != my_slot]
        my_roster = self._roster_tuples(by_slot.get(my_slot, []) if my_slot is not None else [])
        my_ids = {pl.player_id for pl, _ in my_roster}
        for spot in (spots_by_slot.get(my_slot, ()) if my_slot is not None else ()):
            pl = self.players.get(spot.player_id)
            if pl is not None and pl.player_id not in my_ids and pl.player_id not in board_ids:
                my_ids.add(pl.player_id)
                pr = self.projections.get(pl.player_id)
                my_roster.append((pl, float(pr.points) if pr else 0.0))
        lineup_info = self._my_lineup(my_roster, rep, remaining)

        avail_players = [self.players[self.ids[i]] for i in avail]
        mu, sigma = self._mu[avail], self._sigma[avail]
        if eval_pick is None:
            # my last pick: nobody can take anyone from me before a pick I do not have
            pressure = {p: 0.0 for p in SKILL_POSITIONS}
            n_between = 0
            p_next = np.ones(m, dtype=float)
            p_after = np.ones(m, dtype=float)
        else:
            pressure = position_pressure(state, opp_summaries, avail_players, self.projections, eval_pick)
            taken = {p.pick_no for p in state.picks}
            n_between = sum(1 for p in range(current, eval_pick)
                            if p not in taken and (my_slot is None or slot_for_pick(state, p) != my_slot))
            shift = shift_for_pressure(pressure, adp_baseline(avail_players, n_between))
            shift_arr = np.array([shift[p] for p in SKILL_POSITIONS])[pos]
            # condition on the first pick an *opponent* can make: when I am on the clock that is
            # the pick after mine (a back-to-back turn then has nobody in between -> 100%)
            from_pick = current + 1 if on_clock else current
            p_next = prob_available_array(mu, sigma, eval_pick, from_pick, shift_arr)
            p_after = prob_available_array(mu, sigma, eval_after, from_pick, shift_arr)

        # 4. marginal value, VONA, score
        thr_arr = np.array([lineup_info["thresholds"][p] for p in SKILL_POSITIONS])
        thr_arr = np.where(np.isfinite(thr_arr), thr_arr, np.inf)
        bench_w = np.array([lineup_info["bench_weight"][p] for p in SKILL_POSITIONS])
        t_pos = thr_arr[pos]
        gain = np.where(np.isfinite(t_pos), np.maximum(0.0, pts - t_pos), 0.0)
        benched = pts <= t_pos
        enters = ~benched
        own_bench = s.bench_discount * bench_w[pos] * np.maximum(0.0, pts - rep_arr[pos])
        # the player who goes to the bench when the candidate starts is the weakest starter
        # reachable through slot eligibility -- possibly another position (a TE in FLEX, a
        # QB in SUPER_FLEX): price him at *his* usefulness, depth and replacement level
        disp_idx = np.array([_POS_INDEX[d] if d is not None else -1 for d in
                             (lineup_info["displaced"][p] for p in SKILL_POSITIONS)], dtype=np.int64)
        d_pos = disp_idx[pos]
        has_disp = d_pos >= 0
        d_safe = np.where(has_disp, d_pos, 0)
        displaced_bench = np.where(
            has_disp,
            s.bench_discount * bench_w[d_safe] * np.maximum(0.0, np.where(np.isfinite(t_pos), t_pos, 0.0) - rep_arr[d_safe]),
            0.0)
        marginal = gain + np.where(benched, own_bench, displaced_bench)

        e_next_arr = np.zeros(len(SKILL_POSITIONS))
        for pname, pi in _POS_INDEX.items():
            sel = np.flatnonzero(pos == pi)
            if sel.size:
                e_next_arr[pi] = expected_best_available_array(pts[sel], p_next[sel])
        e_next = {p: float(e_next_arr[_POS_INDEX[p]]) for p in SKILL_POSITIONS}
        vona = pts - e_next_arr[pos]

        # ---- two-pick lookahead ---------------------------------------------------
        # plan(X) = marginal(X) now + the best expected marginal value my NEXT pick can
        # still get once X is gone.  Taking X now only beats taking Y now when
        # marginal(X) + next(X) > marginal(Y) + next(Y); a player who will almost surely
        # be there next time contributes nearly his full value to next(Y), which is what
        # makes "you can get him later" a quantitative statement instead of a heuristic.
        npos = len(SKILL_POSITIONS)
        e_next_marg = np.zeros(npos)
        pos_sorted: dict[int, np.ndarray] = {}
        for pi in range(npos):
            sel = np.flatnonzero(pos == pi)
            if sel.size:
                sel = sel[np.argsort(-marginal[sel], kind="stable")]
                pos_sorted[pi] = sel
                e_next_marg[pi] = expected_best_available_array(marginal[sel], p_next[sel])
        best_other = np.array([max([e_next_marg[q] for q in range(npos) if q != pi] or [0.0]) for pi in range(npos)])
        # the same-position option after X: expectation over the rest of the position;
        # when X fills the position's last open starter slot the rest are bench value only
        open_count = {p: 0 for p in SKILL_POSITIONS}
        for sl in lineup_info["open_slots"]:
            for p in SLOT_ELIGIBILITY.get(sl, frozenset()):
                if p in open_count:
                    open_count[p] += 1
        open_arr = np.array([open_count[p] for p in SKILL_POSITIONS])
        same_after = e_next_marg[pos].copy()
        prelim = marginal + np.maximum(best_other[pos], same_after)
        top = np.argsort(-prelim, kind="stable")[:_LOOKAHEAD_TOP]
        for j in top:
            pi = int(pos[j])
            sel = pos_sorted.get(pi)
            if sel is None or sel.size <= 1:
                same_after[j] = 0.0
                continue
            rest = sel[sel != j]
            if open_arr[pi] >= 2 or not np.isfinite(t_pos[j]):
                same_after[j] = expected_best_available_array(marginal[rest], p_next[rest])
            else:
                bench_rest = s.bench_discount * bench_w[pi] * np.maximum(0.0, pts[rest] - rep_arr[pi])
                same_after[j] = expected_best_available_array(bench_rest, p_next[rest])
        next_after = np.maximum(best_other[pos], same_after)
        if eval_pick is None:
            # this is my last pick: there is no next pick to plan for, nothing is left at risk
            next_after = np.zeros_like(next_after)
            score = marginal - float(s.risk_aversion) * std
        else:
            # Two plans with equal expected value are not equal: the one that banks the scarcer
            # player now carries less variance, so charge a small premium on value left at risk.
            score = marginal + next_after - _DEFER_EPS * p_next * marginal - float(s.risk_aversion) * std
        teams = self._team[avail]
        stack_qb: dict[str, str] = lineup_info["stack_qb"]
        stack_pc: dict[str, str] = lineup_info["stack_pc"]
        if stack_qb or stack_pc:
            is_pc = (pos == _POS_INDEX["WR"]) | (pos == _POS_INDEX["TE"])
            is_qb = pos == _POS_INDEX["QB"]
            stack = (is_pc & np.isin(teams, list(stack_qb))) | (is_qb & np.isin(teams, list(stack_pc)))
            # a stack only pays when both players start: no bonus for a backup QB / bench WR
            score = score + np.where(stack & enters, float(s.stack_bonus) * pts, 0.0)
        bye_counts = np.zeros(_MAX_BYE + 1)
        for bw, names in lineup_info["bye_starters"].items():
            bye_counts[bw] = len(names)
        score = score - float(s.bye_penalty) * pts * bye_counts[self._bye[avail]]
        if kdef_early:
            is_kdef = (pos == _POS_INDEX["K"]) | (pos == _POS_INDEX["DEF"])
            score = score - np.where(is_kdef, _KDEF_EARLY_PENALTY * pts, 0.0)

        order = np.argsort(-score, kind="stable")
        overall_rank = np.empty(m, dtype=np.int64)
        overall_rank[order] = np.arange(1, m + 1)

        return _Context(
            key=key, current_pick=current, current_round=current_round,
            eval_pick=eval_pick, eval_after=eval_after, on_clock=on_clock,
            avail=avail, pos=pos, points=pts, rep=rep, vorp=vorp, tier=tier, n_tiers=n_tiers,
            pos_rank=pos_rank, p_next=p_next, p_after=p_after, marginal=marginal, enters=enters, vona=vona,
            score=score, next_after=next_after, e_next_marg=e_next_marg, order=order, overall_rank=overall_rank,
            my_summary=my_summary, opp_summaries=opp_summaries, pressure=pressure,
            e_next=e_next, best_now=best_now, need_open=lineup_info["need_open"],
            open_label=lineup_info["open_label"], bench_weight=lineup_info["bench_weight"],
            bye_starters=lineup_info["bye_starters"], stack_qb=stack_qb, stack_pc=stack_pc,
            kdef_early=kdef_early, kdef_round=kdef_round, n_between=n_between,
            open_slots=lineup_info["open_slots"], picks_spare=lineup_info["picks_spare"],
            idp_open=lineup_info["idp_open"], idp_slots=lineup_info["idp_slots"], remaining=remaining,
            my_ids=my_ids,
        )

    def _roster_tuples(self, picks: Sequence) -> list[tuple[Player, float]]:
        out: list[tuple[Player, float]] = []
        for pk in picks:
            pl = player_from_pick(pk, self.players)
            pr = self.projections.get(pl.player_id)
            out.append((pl, float(pr.points) if pr else 0.0))
        return out

    def _my_lineup(self, roster: list[tuple[Player, float]], rep: Mapping[str, float], remaining_picks: int) -> dict:
        """Lineup-derived structure for my roster: thresholds, needs, bench weights, byes, stacks.

        Open starting slots are valued as replacement-level stand-ins only while I have
        spare picks to fill them later (``remaining_picks`` > open slots); at the end of
        the draft an open slot is simply empty, so filling it is worth the whole player
        and bench picks are worth nothing.
        """
        slots = self._slots
        a_real, _, bench_real = optimal_lineup(roster, slots)
        pos_of = {pl.player_id: pl.position for pl, _ in roster}
        pts_of = {pl.player_id: p for pl, p in roster}
        by_id = {pl.player_id: pl for pl, _ in roster}

        skill = set(SKILL_POSITIONS)
        projected = [i for i, sl in enumerate(slots) if SLOT_ELIGIBILITY.get(sl, frozenset()) & skill]
        open_slots = [slots[i] for i in projected if i not in a_real]
        # IDP (DL/LB/DB/IDP_FLEX) and other unprojected starting slots still cost me picks
        idp_slots = [sl for i, sl in enumerate(slots) if i not in projected]
        idp_have = sum(1 for pl, _ in roster if pl.position not in skill and pl.position != "UNK")
        idp_open = max(0, len(idp_slots) - idp_have)
        picks_spare = int(remaining_picks) - len(open_slots) - idp_open
        bench_scale = 1.0 if picks_spare >= 2 else (0.5 if picks_spare == 1 else 0.0)
        # Replacement-level stand-ins for open starter slots exist only while there are at
        # least two spare picks: with one spare pick the downside of missing the last player
        # at a position is an empty slot, so the slot is valued at the full player.
        phantoms = [ph for ph in phantom_starters(self.league, rep)
                    if int(ph[0].player_id.rsplit("__", 1)[-1]) not in a_real] if picks_spare >= 2 else []
        a_full, _, _ = optimal_lineup(roster + phantoms, slots)
        pos_all = dict(pos_of, **{ph.player_id: ph.position for ph, _ in phantoms})
        pts_all = dict(pts_of, **{ph.player_id: p for ph, p in phantoms})
        disp = starter_displacements({i: (pos_all[pid], pts_all[pid]) for i, pid in a_full.items()}, slots)
        thresholds = {p: t for p, (t, _) in disp.items()}
        # position of the real starter a candidate at p would send to the bench (None: empty
        # slot, replacement-level stand-in, or nowhere to start)
        displaced: dict[str, str | None] = {}
        for p, (_, idx) in disp.items():
            pid = a_full.get(idx) if idx is not None else None
            displaced[p] = pos_all[pid] if pid is not None and not is_phantom(pid) else None

        need_open: dict[str, bool] = {}
        open_label: dict[str, str | None] = {}
        for p in SKILL_POSITIONS:
            label = None
            filled = sum(1 for i, pid in a_real.items() if SLOT_ELIGIBILITY.get(slots[i]) == frozenset({p}))
            dedicated = self.league.dedicated_starters(p)
            if filled < dedicated:
                label = f"{p}{filled + 1}" if dedicated > 1 else p
            else:
                for i, sl in enumerate(slots):
                    elig = SLOT_ELIGIBILITY.get(sl, frozenset())
                    if len(elig) > 1 and p in elig and i not in a_real:
                        label = sl
                        break
            need_open[p] = label is not None
            open_label[p] = label

        bench_counts = {p: 0 for p in SKILL_POSITIONS}
        for pid in bench_real:
            if pos_of[pid] in bench_counts:
                bench_counts[pos_of[pid]] += 1
        bench_weight = {p: bench_scale * bench_usefulness(p, self.league) * bench_depth_factor(bench_counts[p], p)
                        for p in SKILL_POSITIONS}

        bye_starters: dict[int, list[str]] = {}
        for pid in a_real.values():
            bw = by_id[pid].bye_week
            if bw and 0 < bw < _MAX_BYE:
                bye_starters.setdefault(int(bw), []).append(by_id[pid].name)

        # stacks are only worth anything between *starters*: a benched QB2 stacks with nobody
        stack_qb: dict[str, str] = {}
        stack_pc: dict[str, str] = {}
        for pid in a_real.values():
            pl = by_id[pid]
            if not pl.team:
                continue
            if pl.position == "QB":
                stack_qb[pl.team] = pl.name
            elif pl.position in ("WR", "TE"):
                stack_pc.setdefault(pl.team, pl.name)
        return {
            "thresholds": thresholds, "displaced": displaced, "need_open": need_open, "open_label": open_label,
            "bench_weight": bench_weight, "bye_starters": bye_starters,
            "stack_qb": stack_qb, "stack_pc": stack_pc,
            "open_slots": open_slots, "picks_spare": picks_spare,
            "idp_open": idp_open, "idp_slots": idp_slots,
        }

    # ------------------------------------------------------------ PlayerValue
    def _value(self, ctx: _Context, j: int) -> PlayerValue:
        i = int(ctx.avail[j])
        pid = self.ids[i]
        pl, pr = self.players[pid], self.projections[pid]
        pos = SKILL_POSITIONS[int(ctx.pos[j])]
        pts = float(ctx.points[j])
        p_next = float(ctx.p_next[j])
        reasons = self._reasons(ctx, j, pl, pos, pts, p_next)
        warnings = self._warnings(ctx, pl, pr, pos)
        return PlayerValue(
            player=pl,
            projection=pr,
            vorp=float(ctx.vorp[j]),
            vona=float(ctx.vona[j]),
            marginal_value=float(ctx.marginal[j]),
            score=float(ctx.score[j]),
            availability_next=p_next,
            availability_after_next=float(ctx.p_after[j]),
            tier=int(ctx.tier[j]),
            pos_rank=int(ctx.pos_rank[j]),
            overall_rank=int(ctx.overall_rank[j]),
            reasons=reasons,
            warnings=warnings,
        )

    def _reasons(self, ctx: _Context, j: int, pl: Player, pos: str, pts: float, p_next: float) -> list[str]:
        out = [f"Proj {pts:.0f} pts, {float(ctx.vorp[j]):+.0f} over replacement {pos}"]
        label = ctx.open_label.get(pos)
        if label:
            out.append(f"Fills your open {label}")
        if ctx.eval_pick is None:
            out.append("This is your last pick")
        else:
            qualifier = "Only " if p_next < 0.6 else ""
            out.append(f"{qualifier}{p_next:.0%} chance available at your next pick (#{ctx.eval_pick})")
        extras: list[str] = []
        team = pl.team or ""
        if bool(ctx.enters[j]):
            if pos in ("WR", "TE") and team in ctx.stack_qb:
                extras.append(f"Stacks with your QB {ctx.stack_qb[team]}")
            elif pos == "QB" and team in ctx.stack_pc:
                extras.append(f"Stacks with your {ctx.stack_pc[team]}")
        if pl.bye_week and pl.bye_week in ctx.bye_starters:
            names = ctx.bye_starters[pl.bye_week]
            who = names[0] if len(names) == 1 else f"{len(names)} starters"
            extras.append(f"Bye {pl.bye_week} clashes with {who}")
        if pl.adp is not None and pl.adp > 0 and pl.adp <= ctx.current_pick - 5:
            extras.append(f"ADP {pl.adp:.0f} → value at pick {ctx.current_pick}")
        model_rank = int(self._model_rank[int(ctx.avail[j])])
        if pl.ecr is not None and pl.ecr > 0:
            if pl.ecr < model_rank - 10:
                extras.append(f"Market (ECR {pl.ecr:.0f}) higher than model ({model_rank})")
            elif pl.ecr > model_rank + 10:
                extras.append(f"Model (rank {model_rank}) higher than market (ECR {pl.ecr:.0f})")
        out.extend(extras)
        out.append(f"Tier {int(ctx.tier[j])} of {ctx.n_tiers.get(pos, 1)} at {pos}")
        return out[:4]

    def _warnings(self, ctx: _Context, pl: Player, pr: Projection, pos: str) -> list[str]:
        out: list[str] = []
        st = (pl.injury_status or "").strip()
        if st and st.lower() not in ("healthy", "na"):
            out.append(f"Injury: {st}")
        if pos in ("QB", "RB", "WR", "TE") and (pl.depth_chart_order or 0) >= 2 and (pl.ecr is None or pl.ecr > 120):
            out.append(f"Depth chart #{pl.depth_chart_order} with no ECR support")
        if pos in ("K", "DEF") and ctx.kdef_early:
            out.append("K/DEF too early")
        if pl.is_rookie and pl.years_exp is not None:
            out.append("Rookie")
        if "no_data" in (pr.flags or ()):
            out.append("No projection data")
        return out

    # --------------------------------------------------------- PositionAdvice
    def _advice(self, ctx: _Context, pos: str, cands: list[PlayerValue]) -> PositionAdvice:
        e_next = ctx.e_next.get(pos, 0.0)
        drop = ctx.best_now.get(pos, 0.0) - e_next
        nxt = ctx.eval_pick
        if not cands:
            if self._demand.get(pos, 0.0) <= 0:
                return PositionAdvice(pos, [], "SKIP", f"No {pos} starting slot in this league", e_next, drop)
            return PositionAdvice(pos, [], "SKIP", f"No {pos} left on the board", e_next, drop)
        best = cands[0]
        p = best.availability_next
        need = ctx.need_open.get(pos, False)
        low_bench = (ctx.bench_weight.get(pos, 0.0) < 0.3
                     or best.marginal_value < max(_SKIP_MIN_VALUE, 0.05 * best.projection.points))
        if nxt is None:
            base = f"this is your last pick; best {pos} now is {best.player.name} ({best.projection.points:.0f} pts)"
        else:
            base = (f"{p:.0%} chance {best.player.name} is there at your next pick (#{nxt}); "
                    f"expected best {pos} then ~{e_next:.0f} pts (drop-off {drop:.0f})")
        if pos in ("K", "DEF") and ctx.kdef_early:
            action = "SKIP"
            text = f"K/DEF: wait until round {ctx.kdef_round} (now round {ctx.current_round}); {base}"
        elif need and ctx.picks_spare <= 0:
            action = "TAKE NOW"
            must = list(ctx.open_slots) + list(ctx.idp_slots[:ctx.idp_open])
            text = (f"Must fill: {len(must)} open starter(s) ({' '.join(must)}) and only "
                    f"{ctx.remaining} pick(s) left; {base}")
        elif best.overall_rank == 1:
            action = "TAKE NOW"
            text = f"Best value on the board right now; {base}"
        elif nxt is None:
            action = "SKIP"
            text = f"Not the best use of your last pick; {base}"
        elif not need and low_bench:
            action = "SKIP"
            text = f"{pos} starters are set and a bench {pos} adds little; {base}"
        elif p < 0.5 and (drop > 0.6 * best.projection.std or need):
            action = "TAKE NOW"
            text = f"Only {base}"
        elif p < 0.75:
            action = "SOON"
            text = base
        elif need and drop > best.projection.std:
            action = "SOON"
            text = f"{base}; the drop-off is steep for an open {ctx.open_label.get(pos) or pos}"
        else:
            action = "WAIT"
            text = f"{base}; you can wait"
        return PositionAdvice(pos, cands, action, text, float(e_next), float(drop))

    # ------------------------------------------------------------------ notes
    def _notes(self, state: DraftState, ctx: _Context) -> list[str]:
        notes: list[str] = []
        recent = sorted(state.picks, key=lambda p: p.pick_no)[-6:]
        counts: dict[str, int] = {}
        for pk in recent:
            pos = pk.position or (self.projections[pk.player_id].position if pk.player_id in self.projections else None)
            if pos:
                counts[pos] = counts.get(pos, 0) + 1
        for pos, c in sorted(counts.items(), key=lambda kv: -kv[1]):
            if c >= _RUN_MIN and len(recent) >= _RUN_MIN:
                notes.append(f"{pos} run: {c} of last {len(recent)} picks")
        fut = [p for p in future_picks(state) if p > ctx.current_pick]
        if ctx.on_clock:
            head = f"You are on the clock (#{ctx.current_pick})"
            if fut:
                gap = fut[0] - ctx.current_pick
                head += f"; next at #{fut[0]}" + (" (back-to-back)" if gap == 1 else "")
            else:
                head += "; this is your last pick"
            notes.append(head)
        elif len(fut) >= 2:
            gap = fut[1] - fut[0]
            tag = " (back-to-back)" if gap == 1 else (" (at the turn)" if gap <= 3 else "")
            notes.append(f"You pick next at #{fut[0]} and #{fut[1]}{tag}")
        elif fut:
            notes.append(f"You pick next at #{fut[0]}")
        if self.league.is_superflex:
            # startable = above the full-pool replacement level; scarce when the league's open
            # QB/SUPER_FLEX slots could swallow every one of them
            startable = int(np.sum((ctx.pos == _POS_INDEX["QB"]) & (ctx.points > self._full_rep["QB"])))
            everyone = list(ctx.opp_summaries) + ([ctx.my_summary] if ctx.my_summary is not None else [])
            qb_open = sum(sm.open_starters.get("QB", 0) + sm.open_starters.get("SUPER_FLEX", 0) for sm in everyone)
            if qb_open > 0 and (startable <= qb_open or startable <= state.teams):
                notes.append(f"Superflex: QBs scarce ({startable} startable left for {qb_open} open QB/SUPER_FLEX slots)")
        if ctx.kdef_early:
            notes.append(f"K/DEF: wait until round {ctx.kdef_round}")
        remaining = ctx.remaining if state.my_slot is not None else 0
        if ctx.open_slots and remaining and ctx.picks_spare <= 0:
            notes.append(f"{remaining} pick{'s' if remaining != 1 else ''} left for {len(ctx.open_slots)} open "
                         f"starter{'s' if len(ctx.open_slots) != 1 else ''} ({' '.join(ctx.open_slots)}): fill them")
        if ctx.idp_open and remaining:
            labels = sorted(set(ctx.idp_slots), key=ctx.idp_slots.index)
            notes.append(f"You must draft {ctx.idp_open} IDP starter{'s' if ctx.idp_open != 1 else ''} "
                         f"({' '.join(labels)}) yourself: IDP is not projected here")
        hot = [(p, v) for p, v in ctx.pressure.items() if v >= 2.5 and p not in ("K", "DEF")]
        if hot and ctx.n_between > 0:
            p, v = max(hot, key=lambda kv: kv[1])
            notes.append(f"Expect ~{v:.0f} {p}s to go in the {ctx.n_between} picks before #{ctx.eval_pick}")
        return notes
