"""FantasyPros expert consensus rankings (ECR) via the dynastyprocess/data mirror.

Only PPR boards are published there (overall + superflex + positional). ECR is a
*market* signal: it captures injuries, holdouts and depth-chart news faster than
any stats-only model.
"""
from __future__ import annotations

import logging

import pandas as pd

from .cache import download, raw_path
from .canonical import to_sleeper_team

log = logging.getLogger(__name__)

ECR_URL = "https://raw.githubusercontent.com/dynastyprocess/data/master/files/db_fpecr_latest.csv"
WEEKLY_URL = "https://raw.githubusercontent.com/dynastyprocess/data/master/files/fp_latest_weekly.csv"

OVERALL_PAGE = "redraft-overall"        # PPR overall cheatsheet
SUPERFLEX_PAGE = "redraft-op"            # PPR superflex overall
POSITION_PAGES = {"QB": "redraft-qb", "RB": "redraft-rb", "WR": "redraft-wr", "TE": "redraft-te",
                  "K": "redraft-k", "DEF": "redraft-dst"}


def load_ecr_raw(max_age_hours: float = 12.0) -> pd.DataFrame:
    p = download(ECR_URL, raw_path("db_fpecr_latest.csv"), max_age_hours=max_age_hours)
    df = pd.read_csv(p, low_memory=False)
    df["pos"] = df["pos"].replace({"DST": "DEF"})
    df["team"] = df["team"].map(to_sleeper_team)
    df["fantasypros_id"] = df["id"].astype("Int64").astype(str)
    return df


def overall_ecr(raw: pd.DataFrame | None = None, superflex: bool = False) -> pd.DataFrame:
    """Overall board: columns fantasypros_id, player, pos, team, ecr, sd, best, worst, bye, scrape_date."""
    raw = raw if raw is not None else load_ecr_raw()
    page = SUPERFLEX_PAGE if superflex else OVERALL_PAGE
    df = raw[raw["page_type"] == page]
    if df.empty and superflex:
        df = raw[raw["page_type"] == OVERALL_PAGE]
    cols = ["fantasypros_id", "player", "pos", "team", "ecr", "sd", "best", "worst", "bye", "scrape_date"]
    out = df[[c for c in cols if c in df.columns]].copy().sort_values("ecr").reset_index(drop=True)
    out["ecr_rank"] = range(1, len(out) + 1)
    return out


def positional_ecr(raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """Positional boards stacked: fantasypros_id, pos, pos_ecr, pos_sd."""
    raw = raw if raw is not None else load_ecr_raw()
    frames = []
    for pos, page in POSITION_PAGES.items():
        d = raw[raw["page_type"] == page][["fantasypros_id", "player", "team", "ecr", "sd"]].copy()
        d["pos"] = pos
        frames.append(d.rename(columns={"ecr": "pos_ecr", "sd": "pos_sd"}))
    return pd.concat(frames, ignore_index=True)


def load_weekly_raw(max_age_hours: float = 12.0) -> pd.DataFrame:
    """Weekly (start/sit) rankings - useful in-season, not used for the draft."""
    p = download(WEEKLY_URL, raw_path("fp_latest_weekly.csv"), max_age_hours=max_age_hours)
    return pd.read_csv(p, low_memory=False)
