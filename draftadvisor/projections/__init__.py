"""ML projections and blending. See DESIGN.md §3.3.

The blend (pure Python + numpy) is always importable; the model/feature code needs pandas and
scikit-learn and is exposed only when they are installed (the stateless web app runs without them).
"""
from .blend import (  # noqa: F401
    DEFAULT_WEIGHTS,
    MarketCurve,
    Projector,
    RankCurve,
    def_bracket_rates,
    default_rank_curves,
    ecr_implied_points,
    fit_rank_curves,
    market_replacement_rank,
    rates_to_season,
    sleeper_stats_to_projection,
    yds_bracket_rates,
)
from .features import ALL_TARGET_KEYS, FEATURE_COLUMNS, TARGETS  # noqa: F401  (constants; pandas optional)

try:  # optional heavy dependencies
    from .blend import project_offline  # noqa: F401
    from .features import build_inference_table, build_training_table, season_aggregates  # noqa: F401
    from .model import ProjectionModel, train_and_save  # noqa: F401
except ImportError:  # pragma: no cover - pandas / scikit-learn not installed
    ProjectionModel = None  # type: ignore[assignment]
    train_and_save = None  # type: ignore[assignment]
    project_offline = None  # type: ignore[assignment]
    build_inference_table = build_training_table = season_aggregates = None  # type: ignore[assignment]

__all__ = [
    "ALL_TARGET_KEYS", "FEATURE_COLUMNS", "TARGETS", "def_bracket_rates", "yds_bracket_rates",
    "DEFAULT_WEIGHTS", "MarketCurve", "Projector", "RankCurve", "default_rank_curves", "ecr_implied_points",
    "fit_rank_curves", "market_replacement_rank", "rates_to_season", "sleeper_stats_to_projection",
    "ProjectionModel", "train_and_save", "project_offline", "build_inference_table", "build_training_table",
    "season_aggregates",
]
