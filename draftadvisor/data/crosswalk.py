"""Id crosswalk: Sleeper player_id <-> nflverse gsis_id <-> FantasyPros id.

Sources (merged, Sleeper id wins conflicts):
* dynastyprocess ``db_playerids.csv`` (sleeper_id, gsis_id, fantasypros_id, pfr_id, draft info, age)
* nflverse season rosters (gsis_id, sleeper_id, pfr_id, esb_id, ...)
* name+position fuzzy match as last resort (normalised names)
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]

from ..config import DEFAULT_SEASON
from .cache import download, raw_path
from .canonical import to_sleeper_team

log = logging.getLogger(__name__)

PLAYERIDS_URL = "https://raw.githubusercontent.com/dynastyprocess/data/master/files/db_playerids.csv"

from .names import normalize_name  # noqa: E402  (shared with the lean runtime)


def _clean_id(v) -> str | None:
    if v is None or (isinstance(v, float) and v != v):
        return None
    s = str(v).strip()
    if s in ("", "nan", "NA", "<NA>"):
        return None
    if s.endswith(".0"):
        s = s[:-2]
    return s


def load_playerids_raw(max_age_hours: float = 24.0) -> pd.DataFrame:
    p = download(PLAYERIDS_URL, raw_path("db_playerids.csv"), max_age_hours=max_age_hours)
    df = pd.read_csv(p, low_memory=False)
    for c in ("sleeper_id", "gsis_id", "fantasypros_id", "pfr_id", "espn_id", "yahoo_id"):
        if c in df.columns:
            df[c] = df[c].map(_clean_id)
    df["team"] = df["team"].map(to_sleeper_team)
    return df


@dataclass
class Crosswalk:
    """Bidirectional id maps plus per-player reference attributes."""

    sleeper_to_gsis: dict[str, str] = field(default_factory=dict)
    gsis_to_sleeper: dict[str, str] = field(default_factory=dict)
    sleeper_to_fp: dict[str, str] = field(default_factory=dict)
    fp_to_sleeper: dict[str, str] = field(default_factory=dict)
    pfr_to_gsis: dict[str, str] = field(default_factory=dict)
    #: sleeper_id -> {"name","position","team","age","draft_year","draft_round","draft_ovr","gsis_id","fantasypros_id","years_exp","status"}
    attrs: dict[str, dict] = field(default_factory=dict)
    #: (normalized name, position) -> sleeper_id  (fallback matching)
    name_index: dict[tuple[str, str], str] = field(default_factory=dict)

    def gsis_for(self, sleeper_id: str) -> str | None:
        return self.sleeper_to_gsis.get(str(sleeper_id))

    def sleeper_for_gsis(self, gsis_id: str) -> str | None:
        return self.gsis_to_sleeper.get(str(gsis_id))

    def sleeper_for_fp(self, fp_id: str) -> str | None:
        return self.fp_to_sleeper.get(_clean_id(fp_id) or "")

    def sleeper_for_name(self, name: str, position: str | None = None) -> str | None:
        key = normalize_name(name)
        if position:
            hit = self.name_index.get((key, position))
            if hit:
                return hit
        for (n, p), sid in self.name_index.items():
            if n == key:
                return sid
        return None

    def add(self, sleeper_id: str | None, gsis_id: str | None = None, fp_id: str | None = None,
            pfr_id: str | None = None, name: str | None = None, position: str | None = None,
            **attrs) -> None:
        sid = _clean_id(sleeper_id)
        g = _clean_id(gsis_id)
        f = _clean_id(fp_id)
        p = _clean_id(pfr_id)
        if p and g:
            self.pfr_to_gsis.setdefault(p, g)
        if sid is None:
            return
        if g:
            self.sleeper_to_gsis.setdefault(sid, g)
            self.gsis_to_sleeper.setdefault(g, sid)
        if f:
            self.sleeper_to_fp.setdefault(sid, f)
            self.fp_to_sleeper.setdefault(f, sid)
        a = self.attrs.setdefault(sid, {})
        for k, v in attrs.items():
            if v is not None and not (isinstance(v, float) and v != v) and k not in a:
                a[k] = v
        if name and "name" not in a:
            a["name"] = name
        if position and "position" not in a:
            a["position"] = position
        if name and position:
            self.name_index.setdefault((normalize_name(name), position), sid)


def build_crosswalk(season: int = DEFAULT_SEASON, roster_seasons: int = 3) -> Crosswalk:
    from .nflverse import load_roster

    """Merge dynastyprocess ids with the last ``roster_seasons`` nflverse rosters."""
    cw = Crosswalk()
    try:
        ids = load_playerids_raw()
        for r in ids.itertuples(index=False):
            d = r._asdict()
            pos = d.get("position")
            pos = {"DST": "DEF", "PK": "K", "FB": "RB"}.get(pos, pos)
            cw.add(d.get("sleeper_id"), d.get("gsis_id"), d.get("fantasypros_id"), d.get("pfr_id"),
                   name=d.get("name"), position=pos, team=d.get("team"), age=d.get("age"),
                   draft_year=d.get("draft_year"), draft_round=d.get("draft_round"),
                   draft_ovr=d.get("draft_ovr"), birthdate=d.get("birthdate"))
    except Exception as e:  # noqa: BLE001
        log.warning("db_playerids unavailable: %s", e)
    for s in range(season, season - roster_seasons, -1):
        try:
            ro = load_roster(s)
        except Exception as e:  # noqa: BLE001
            log.warning("roster %s unavailable: %s", s, e)
            continue
        for r in ro.itertuples(index=False):
            d = r._asdict()
            pos = {"FB": "RB", "HB": "RB", "PK": "K"}.get(d.get("position"), d.get("position"))
            cw.add(d.get("sleeper_id"), d.get("gsis_id"), None, d.get("pfr_id"),
                   name=d.get("full_name"), position=pos, team=d.get("team") if s == season else None,
                   years_exp=d.get("years_exp") if s == season else None,
                   status=d.get("status") if s == season else None,
                   birth_date=d.get("birth_date"), draft_number=d.get("draft_number"),
                   entry_year=d.get("entry_year"))
    # Team defenses: Sleeper ids are team abbreviations.
    for t in ("ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB", "HOU", "IND",
              "JAX", "KC", "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA",
              "SF", "TB", "TEN", "WAS"):
        cw.add(t, gsis_id=t, name=f"{t} Defense", position="DEF", team=t)
    log.info("crosswalk: %d sleeper ids, %d gsis links, %d fantasypros links",
             len(cw.attrs), len(cw.sleeper_to_gsis), len(cw.sleeper_to_fp))
    return cw
