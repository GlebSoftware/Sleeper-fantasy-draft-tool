"""Feature engineering. See DESIGN.md §3.3. STUB - "projections" agent."""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from ..data.crosswalk import Crosswalk
from ..models import Player

TARGETS: dict[str, list[str]] = {
    "QB": ["pass_att", "pass_cmp", "pass_yd", "pass_td", "pass_int", "rush_att", "rush_yd", "rush_td", "fum_lost",
           "bonus_pass_yd_300", "bonus_rush_yd_100"],
    "RB": ["rush_att", "rush_yd", "rush_td", "rec_tgt", "rec", "rec_yd", "rec_td", "fum_lost", "bonus_rush_yd_100", "bonus_rec_yd_100"],
    "WR": ["rush_att", "rush_yd", "rush_td", "rec_tgt", "rec", "rec_yd", "rec_td", "fum_lost", "bonus_rush_yd_100", "bonus_rec_yd_100"],
    "TE": ["rush_att", "rush_yd", "rush_td", "rec_tgt", "rec", "rec_yd", "rec_td", "fum_lost", "bonus_rush_yd_100", "bonus_rec_yd_100"],
    "K": ["fgm_0_19", "fgm_20_29", "fgm_30_39", "fgm_40_49", "fgm_50p", "fgmiss", "xpm", "xpmiss"],
    "DEF": ["sack", "int", "ff", "fum_rec", "safe", "def_td", "blk_kick", "pts_allow", "yds_allow"],
}
FEATURE_COLUMNS: list[str] = []


def season_aggregates(canonical: pd.DataFrame, snaps: pd.DataFrame | None = None,
                      injuries: pd.DataFrame | None = None, pfr_to_gsis: dict[str, str] | None = None) -> pd.DataFrame: raise NotImplementedError
def build_training_table(agg: pd.DataFrame, team_ctx: pd.DataFrame, rosters_by_season: dict[int, pd.DataFrame],
                         seasons: Iterable[int]) -> pd.DataFrame: raise NotImplementedError
def build_inference_table(agg: pd.DataFrame, team_ctx: pd.DataFrame, players: dict[str, Player], cw: Crosswalk,
                          season: int) -> pd.DataFrame: raise NotImplementedError
