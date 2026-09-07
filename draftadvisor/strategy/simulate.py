"""Monte-Carlo look-ahead for the top candidates (DESIGN.md §3.4).

For each candidate: take him now, simulate the other teams' picks (sampled from
the availability model: each opponent picks among the best available by ADP with
probability proportional to the hazard of going *right now*, respecting
starting-lineup needs), greedily take my best marginal-value player at each of
my next picks, and value my roster.  Time-boxed: stops after ``time_budget_ms``
and returns whatever candidates completed at least one simulation.
"""
from __future__ import annotations

import logging
import time
from typing import Mapping, Sequence

import numpy as np

from ..config import SKILL_POSITIONS, SLOT_ELIGIBILITY
from ..models import DraftState, Player, Projection
from .availability import slot_for_pick
from .lineup import bench_usefulness, optimal_lineup, phantom_starters, starter_thresholds
from .recommend import Advisor

log = logging.getLogger(__name__)

__all__ = ["simulate_candidates"]

_POS_INDEX = {p: i for i, p in enumerate(SKILL_POSITIONS)}
_OPP_POOL = 24
_MAX_AT_POS = {"QB": 2, "RB": 7, "WR": 7, "TE": 2, "K": 1, "DEF": 1}
_KDEF_LAST_ROUNDS = 3


def _sf_np(z: np.ndarray) -> np.ndarray:
    a = np.abs(z) / np.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * a)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    erf_abs = 1.0 - poly * np.exp(-a * a)
    return 0.5 * (1.0 - np.where(z >= 0, erf_abs, -erf_abs))


class _Sim:
    """Shared arrays for one simulation batch."""

    def __init__(self, state: DraftState, advisor: Advisor, players: Mapping[str, Player],
                 projections: Mapping[str, Projection]) -> None:
        self.state, self.adv, self.players, self.projections = state, advisor, players, projections
        self.league = advisor.league
        self.slots = self.league.starting_slots
        self.rounds = state.draft.rounds
        ctx = advisor._context(state)
        self.rep = ctx.rep
        self.avail0 = np.zeros(advisor.n, dtype=bool)
        self.avail0[ctx.avail] = True
        self.mu = advisor._mu
        self.sigma = advisor._sigma
        self.pos = advisor._pos
        self.points = advisor._points
        self.rep_arr = np.array([self.rep[p] for p in SKILL_POSITIONS])
        self.useful = np.array([bench_usefulness(p, self.league) for p in SKILL_POSITIONS])
        by_slot = state.picks_by_slot()
        self.rosters: dict[int, list[tuple[Player, float]]] = {}
        for slot in range(1, state.teams + 1):
            self.rosters[slot] = advisor._roster_tuples(by_slot.get(slot, []))
        self.my_slot = state.my_slot
        self.my_picks = set(state.my_pick_numbers())
        self.discount = float(advisor.settings.bench_discount)

    # -- opponent pick --------------------------------------------------------
    def opponent_pick(self, rng: np.random.Generator, avail: np.ndarray, pick_no: int,
                      counts: dict[str, int]) -> int | None:
        idx = np.flatnonzero(avail)
        if idx.size == 0:
            return None
        order = idx[np.argsort(self.mu[idx], kind="stable")[:_OPP_POOL]]
        rnd = self.state.draft.round_of(pick_no)
        late = rnd > self.rounds - _KDEF_LAST_ROUNDS
        z = (pick_no - 0.5 - self.mu[order]) / self.sigma[order]
        hazard = np.exp(-0.5 * z * z) / np.maximum(_sf_np(z), 1e-6)
        w = hazard.copy()
        for k, i in enumerate(order):
            p = SKILL_POSITIONS[int(self.pos[i])]
            if counts.get(p, 0) >= _MAX_AT_POS[p] or (p in ("K", "DEF") and not late):
                w[k] = 0.0
        if w.sum() <= 0:
            return int(order[0])
        return int(rng.choice(order, p=w / w.sum()))

    # -- my greedy pick --------------------------------------------------------
    def my_pick(self, avail: np.ndarray, roster: list[tuple[Player, float]], pick_no: int) -> int | None:
        idx = np.flatnonzero(avail)
        if idx.size == 0:
            return None
        a_real, _, bench = optimal_lineup(roster, self.slots)
        open_slots = sum(1 for i, sl in enumerate(self.slots)
                         if i not in a_real and SLOT_ELIGIBILITY.get(sl, frozenset()) & set(SKILL_POSITIONS))
        remaining = sum(1 for p in self.my_picks if p >= pick_no)
        spare = remaining - open_slots
        bench_scale = 1.0 if spare >= 2 else (0.5 if spare == 1 else 0.0)
        phantoms = [ph for ph in phantom_starters(self.league, self.rep)
                    if int(ph[0].player_id.rsplit("__", 1)[-1]) not in a_real] if spare >= 1 else []
        a_full, _, _ = optimal_lineup(roster + phantoms, self.slots)
        info = {pl.player_id: (pl.position, p) for pl, p in roster + phantoms}
        thr = starter_thresholds({i: info[pid] for i, pid in a_full.items()}, self.slots)
        thr_arr = np.array([thr[p] for p in SKILL_POSITIONS])
        bench_n = np.zeros(len(SKILL_POSITIONS))
        for pid in bench:
            bench_n[_POS_INDEX[info[pid][0]]] += 1
        bw = bench_scale * self.useful * (0.7 ** bench_n)
        pos, pts = self.pos[idx], self.points[idx]
        t = thr_arr[pos]
        gain = np.where(np.isfinite(t), np.maximum(0.0, pts - t), 0.0)
        bench_v = self.discount * bw[pos] * np.maximum(0.0, pts - self.rep_arr[pos])
        value = gain + np.where(pts <= t, bench_v, 0.0)
        late = self.state.draft.round_of(pick_no) > self.rounds - _KDEF_LAST_ROUNDS
        if not late:
            value = value - np.where((pos == _POS_INDEX["K"]) | (pos == _POS_INDEX["DEF"]), 0.25 * pts, 0.0)
        return int(idx[int(np.argmax(value))])

    def roster_points(self, roster: list[tuple[Player, float]]) -> float:
        _, starters, bench = optimal_lineup(roster, self.slots)
        pts = {pl.player_id: p for pl, p in roster}
        pos = {pl.player_id: pl.position for pl, _ in roster}
        bench_v = sum(self.discount * bench_usefulness(pos[b], self.league) * max(0.0, pts[b] - self.rep.get(pos[b], 0.0))
                      for b in bench)
        return float(starters + bench_v)

    def run_once(self, rng: np.random.Generator, cand_idx: int, my_picks_ahead: int) -> float:
        avail = self.avail0.copy()
        avail[cand_idx] = False              # reserved: he is my pick at my next turn
        rosters = {s: list(r) for s, r in self.rosters.items()}
        counts: dict[int, dict[str, int]] = {
            s: {p: sum(1 for pl, _ in r if pl.position == p) for p in SKILL_POSITIONS} for s, r in rosters.items()
        }
        my_roster = rosters.setdefault(self.my_slot or 0, [])
        my_done = 0
        pick_no = self.state.next_pick_no
        total = self.state.draft.total_picks
        first = True
        while pick_no <= total and my_done < my_picks_ahead:
            slot = slot_for_pick(self.state, pick_no)
            if slot == self.my_slot:
                i = cand_idx if first else self.my_pick(avail, my_roster, pick_no)
                first = False
                if i is None:
                    break
                avail[i] = False
                pid = self.adv.ids[i]
                my_roster.append((self.players[pid], float(self.points[i])))
                my_done += 1
            else:
                i = self.opponent_pick(rng, avail, pick_no, counts[slot])
                if i is not None:
                    avail[i] = False
                    pid = self.adv.ids[i]
                    rosters[slot].append((self.players[pid], float(self.points[i])))
                    counts[slot][SKILL_POSITIONS[int(self.pos[i])]] += 1
            pick_no += 1
        return self.roster_points(my_roster)


def simulate_candidates(
    state: DraftState,
    players: Mapping[str, Player],
    projections: Mapping[str, Projection],
    advisor: Advisor,
    candidate_ids: Sequence[str],
    n_sims: int = 30,
    rounds_ahead: int = 2,
    seed: int = 0,
    time_budget_ms: float = 250.0,
) -> dict[str, float]:
    """Expected my-roster points after taking each candidate now (deterministic per seed).

    ``rounds_ahead`` counts my picks including the candidate.  Candidates that did
    not complete a single simulation within ``time_budget_ms`` are absent.
    """
    t0 = time.perf_counter()
    if state.my_slot is None:
        log.warning("simulate_candidates: my_slot unknown, nothing to simulate")
        return {}
    ctx = advisor._context(state)
    avail_set = set(int(i) for i in ctx.avail)
    cands = [(pid, advisor._index[pid]) for pid in candidate_ids
             if pid in advisor._index and advisor._index[pid] in avail_set]
    if not cands:
        return {}
    sim = _Sim(state, advisor, players, projections)
    sums = {pid: 0.0 for pid, _ in cands}
    n = {pid: 0 for pid, _ in cands}
    stopped = False
    for k in range(max(1, n_sims)):
        for pid, idx in cands:
            rng = np.random.default_rng(seed * 100_003 + k * 7919 + idx)
            sums[pid] += sim.run_once(rng, idx, max(1, rounds_ahead))
            n[pid] += 1
            if (time.perf_counter() - t0) * 1000.0 > time_budget_ms:
                stopped = True
                break
        if stopped:
            break
    if stopped:
        log.info("simulate_candidates: time budget hit after %d sims", min(n.values()))
    return {pid: sums[pid] / n[pid] for pid in sums if n[pid] > 0}
