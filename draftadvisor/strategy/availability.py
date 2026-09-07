"""Availability model: will a player still be there at my next pick? (DESIGN.md §3.4)

Every player's draft position is modelled as Normal(mu, sigma) in overall-pick
units (mu from ADP, else ECR, else 400).  Because a player still on the board can
not have been taken already, the distribution is truncated at the current pick
and availability at a later pick is the *conditional* survival.  Position
pressure (open starters on the teams picking before me, plus recent runs) shifts
the mean earlier for positions in demand.
"""
from __future__ import annotations

import logging
import math
from typing import Mapping, Sequence

import numpy as np

from ..config import SKILL_POSITIONS, SLOT_ELIGIBILITY
from ..models import DraftState, Player, Projection, RosterSummary, normal_cdf

log = logging.getLogger(__name__)

__all__ = [
    "UNRANKED_PICK",
    "pick_mean",
    "pick_distribution",
    "pick_distribution_arrays",
    "prob_available",
    "prob_available_array",
    "position_pressure",
    "slot_for_pick",
    "adp_baseline",
    "shift_for_pressure",
    "expected_best_available",
    "expected_best_available_array",
]

UNRANKED_PICK = 400.0
_RUN_WINDOW = 6
_RUN_WEIGHT = 0.5
_MARKET_TOP = 12


def pick_mean(player: Player) -> float:
    """ADP if present, else ECR, else :data:`UNRANKED_PICK`."""
    if player.adp is not None and player.adp > 0:
        return float(player.adp)
    if player.ecr is not None and player.ecr > 0:
        return float(player.ecr)
    return UNRANKED_PICK


def _sigma(mu: float, ecr_sd: float | None) -> float:
    s = max(2.5, 0.10 * mu + 1.5)
    if ecr_sd is not None and ecr_sd > 0:
        s = 0.5 * s + 0.5 * (1.2 * float(ecr_sd))
    return s


def pick_distribution(player: Player, current_pick: int) -> tuple[float, float]:
    """``(mu, sigma)`` of the player's draft position, with mu >= current_pick - 0.5."""
    mu = pick_mean(player)
    sigma = _sigma(mu, player.ecr_sd)
    return max(mu, current_pick - 0.5), sigma


def pick_distribution_arrays(mu: np.ndarray, ecr_sd: np.ndarray) -> np.ndarray:
    """Vectorised sigma for raw means ``mu`` (``ecr_sd`` NaN when absent)."""
    base = np.maximum(2.5, 0.10 * mu + 1.5)
    has = np.isfinite(ecr_sd) & (ecr_sd > 0)
    return np.where(has, 0.5 * base + 0.5 * 1.2 * np.nan_to_num(ecr_sd), base)


def _norm_sf_np(z: np.ndarray) -> np.ndarray:
    """Standard normal survival function (Abramowitz-Stegun 7.1.26, |err| < 2e-7)."""
    a = np.abs(z) / math.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * a)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    erf_abs = 1.0 - poly * np.exp(-a * a)
    cdf = 0.5 * (1.0 + np.where(z >= 0, erf_abs, -erf_abs))
    return 1.0 - cdf


def prob_available(player: Player, at_pick: int, current_pick: int, shift: float = 0.0) -> float:
    """P(player still on the board when ``at_pick`` comes up | on the board now).

    ``shift`` (in picks) moves the mean earlier (position pressure).
    """
    if at_pick <= current_pick:
        return 1.0
    mu, sigma = pick_distribution(player, current_pick)
    mu = max(mu - shift, current_pick - 0.5)
    s_now = 1.0 - normal_cdf((current_pick - 0.5 - mu) / sigma)
    s_then = 1.0 - normal_cdf((at_pick - 0.5 - mu) / sigma)
    if s_now <= 1e-12:
        return 0.0
    return float(min(1.0, max(0.0, s_then / s_now)))


def prob_available_array(
    mu: np.ndarray, sigma: np.ndarray, at_pick: int, current_pick: int, shift: np.ndarray | float = 0.0
) -> np.ndarray:
    """Vectorised :func:`prob_available` (``mu`` are raw, unclipped means)."""
    if at_pick <= current_pick:
        return np.ones_like(mu, dtype=float)
    m = np.maximum(mu - shift, current_pick - 0.5)
    s_now = _norm_sf_np((current_pick - 0.5 - m) / sigma)
    s_then = _norm_sf_np((at_pick - 0.5 - m) / sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(s_now > 1e-12, s_then / np.maximum(s_now, 1e-12), 0.0)
    return np.clip(p, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Position pressure
# ---------------------------------------------------------------------------


def slot_for_pick(state: DraftState, pick_no: int) -> int:
    """Draft slot whose roster owns ``pick_no`` (traded picks respected when mapped)."""
    draft = state.draft
    rid = draft.owner_roster_for_pick(pick_no)
    if rid is not None:
        for s, r in draft.slot_to_roster_id.items():
            if r == rid:
                return int(s)
    return draft.slot_for_pick(pick_no)


def _need_weight(summary: RosterSummary | None, pos: str) -> float:
    if summary is None:
        return 0.5
    if summary.open_starters.get(pos, 0) > 0:
        return 1.0
    for slot, n in summary.open_starters.items():
        elig = SLOT_ELIGIBILITY.get(slot, frozenset())
        if n > 0 and len(elig) > 1 and pos in elig:
            return 0.5
    return 0.15


def _market_shares(available: Sequence[Player], top: int = _MARKET_TOP) -> dict[str, float]:
    ranked = sorted(available, key=pick_mean)[:top]
    shares = {p: 0.0 for p in SKILL_POSITIONS}
    if not ranked:
        return shares
    for pl in ranked:
        if pl.position in shares:
            shares[pl.position] += 1.0 / len(ranked)
    return shares


def _recent_position_counts(state: DraftState, projections: Mapping[str, Projection]) -> dict[str, int]:
    counts = {p: 0 for p in SKILL_POSITIONS}
    for pk in sorted(state.picks, key=lambda p: p.pick_no)[-_RUN_WINDOW:]:
        pos = pk.position
        if pos is None and pk.player_id in projections:
            pos = projections[pk.player_id].position
        if pos in counts:
            counts[pos] += 1
    return counts


def position_pressure(
    state: DraftState,
    opponent_summaries: Sequence[RosterSummary],
    available: Sequence[Player],
    projections: Mapping[str, Projection],
    until_pick: int,
) -> dict[str, float]:
    """Expected number of picks at each position before ``until_pick``.

    Each opposing pick between now and ``until_pick`` is spread across positions with
    weight = need (open dedicated starter 1.0, open flex 0.5, else 0.15) × market
    (share of the top-12 available by ADP).  A run term adds 0.5 per pick at the
    position in the last 6 picks league-wide, scaled by how many picks remain (so a
    single intervening pick can't carry three RBs of pressure).
    """
    by_slot = {s.slot: s for s in opponent_summaries}
    market = _market_shares(available)
    pressure = {p: 0.0 for p in SKILL_POSITIONS}
    n_between = 0
    taken = {p.pick_no for p in state.picks}          # keeper picks are already made
    for pick_no in range(state.next_pick_no, max(state.next_pick_no, until_pick)):
        if pick_no in taken:
            continue
        slot = slot_for_pick(state, pick_no)
        if state.my_slot is not None and slot == state.my_slot:
            continue
        n_between += 1
        summary = by_slot.get(slot)
        weights = {p: _need_weight(summary, p) * market[p] for p in SKILL_POSITIONS}
        total = sum(weights.values())
        if total <= 0:
            continue
        for p, w in weights.items():
            pressure[p] += w / total
    if n_between:
        scale = min(1.0, n_between / _RUN_WINDOW)
        for p, c in _recent_position_counts(state, projections).items():
            pressure[p] += _RUN_WEIGHT * c * scale
    # no position can absorb more picks than there are picks
    return {p: min(v, float(n_between)) for p, v in pressure.items()}


def adp_baseline(available: Sequence[Player], n_picks: int) -> dict[str, float]:
    """Picks the market alone would spend per position in the next ``n_picks``."""
    ranked = sorted(available, key=pick_mean)[: max(0, n_picks)]
    out = {p: 0.0 for p in SKILL_POSITIONS}
    for pl in ranked:
        if pl.position in out:
            out[pl.position] += 1.0
    return out


def shift_for_pressure(pressure: Mapping[str, float], baseline: Mapping[str, float]) -> dict[str, float]:
    """Extra ADP shift (picks earlier) per position: 1.5 × max(0, pressure − baseline)."""
    return {p: 1.5 * max(0.0, float(pressure.get(p, 0.0)) - float(baseline.get(p, 0.0))) for p in SKILL_POSITIONS}


# ---------------------------------------------------------------------------
# Expected best available
# ---------------------------------------------------------------------------


def expected_best_available(ranked: Sequence[tuple[str, float, float]]) -> float:
    """E[points of the best player still available] for ``(id, points, p_avail)`` rows.

    Rows are sorted by points descending; ``Σ points_i · p_i · Π_{j<i}(1 − p_j)`` plus
    a tail term giving the last player's points when nobody survives.
    """
    if not ranked:
        return 0.0
    rows = sorted(ranked, key=lambda r: -r[1])
    total = 0.0
    survive_none = 1.0
    for _, pts, p in rows:
        p = min(1.0, max(0.0, float(p)))
        total += pts * p * survive_none
        survive_none *= 1.0 - p
    return total + survive_none * rows[-1][1]


def expected_best_available_array(points_desc: np.ndarray, p_avail: np.ndarray) -> float:
    """Vectorised :func:`expected_best_available` (``points_desc`` already sorted)."""
    if points_desc.size == 0:
        return 0.0
    p = np.clip(p_avail, 0.0, 1.0)
    none_before = np.cumprod(1.0 - p)
    prior = np.concatenate(([1.0], none_before[:-1]))
    return float(np.sum(points_desc * p * prior) + none_before[-1] * points_desc[-1])
