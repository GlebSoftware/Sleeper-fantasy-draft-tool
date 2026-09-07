"""Draft strategy: replacement levels, lineups, availability, recommendations, trades.

See DESIGN.md §3.4.  Public surface:

* :class:`Advisor` — value every available player for a :class:`~draftadvisor.models.DraftState`.
* :func:`replacement_levels`, :func:`vorp`, :func:`starter_demand`, :func:`tiers`.
* :func:`optimal_lineup`, :func:`marginal_lineup_value`, :func:`roster_summary`.
* :func:`prob_available`, :func:`position_pressure`, :func:`expected_best_available`.
* :func:`evaluate_trade` / :class:`TradeEvaluation`, :func:`evaluate_pick_choice`.
* :func:`simulate_candidates` — time-boxed Monte-Carlo look-ahead.
"""
from __future__ import annotations

from .availability import (
    expected_best_available,
    pick_distribution,
    position_pressure,
    prob_available,
    shift_for_pressure,
)
from .lineup import marginal_lineup_value, optimal_lineup, roster_summary
from .recommend import Advisor
from .replacement import replacement_levels, starter_demand, tiers, vorp
from .simulate import simulate_candidates
from .trade import TradeEvaluation, evaluate_pick_choice, evaluate_trade

__all__ = [
    "Advisor",
    "TradeEvaluation",
    "evaluate_trade",
    "evaluate_pick_choice",
    "simulate_candidates",
    "replacement_levels",
    "starter_demand",
    "vorp",
    "tiers",
    "optimal_lineup",
    "marginal_lineup_value",
    "roster_summary",
    "pick_distribution",
    "prob_available",
    "position_pressure",
    "shift_for_pressure",
    "expected_best_available",
]
