"""ML projections and blending (DESIGN.md §3.3).

* :mod:`features` - season aggregates, training / inference tables (no target leakage).
* :mod:`model`    - per-position HistGradientBoosting rate models, rookies, games, std.
* :mod:`blend`    - :class:`Projector` blending ML / Sleeper / ECR into league points,
  plus :func:`project_offline` when no trained model exists.
"""
from .blend import (
    DEFAULT_WEIGHTS,
    Projector,
    def_bracket_rates,
    ecr_implied_points,
    project_offline,
    rates_to_season,
    sleeper_stats_to_projection,
)
from .features import (
    FEATURE_COLUMNS,
    TARGETS,
    build_inference_table,
    build_training_table,
    season_aggregates,
)
from .model import ProjectionModel, train_and_save

__all__ = [
    "DEFAULT_WEIGHTS",
    "FEATURE_COLUMNS",
    "ProjectionModel",
    "Projector",
    "TARGETS",
    "build_inference_table",
    "build_training_table",
    "def_bracket_rates",
    "ecr_implied_points",
    "project_offline",
    "rates_to_season",
    "season_aggregates",
    "sleeper_stats_to_projection",
    "train_and_save",
]
