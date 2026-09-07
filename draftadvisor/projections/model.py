"""Per-position gradient boosting projection model. See DESIGN.md §3.3. STUB - "projections" agent."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from ..config import TRAIN_SEASONS


class ProjectionModel:
    def __init__(self, seed: int = 7, params: dict | None = None): raise NotImplementedError
    def fit(self, table: pd.DataFrame) -> "ProjectionModel": raise NotImplementedError
    def predict(self, table: pd.DataFrame) -> pd.DataFrame: raise NotImplementedError
    def backtest(self, table: pd.DataFrame, holdout_seasons: list[int]) -> dict: raise NotImplementedError
    def save(self, path: Path | None = None) -> Path: raise NotImplementedError
    @classmethod
    def load(cls, path: Path | None = None) -> "ProjectionModel": raise NotImplementedError


def train_and_save(seasons: Iterable[int] = TRAIN_SEASONS, refresh: bool = False) -> tuple[ProjectionModel, dict]: raise NotImplementedError
