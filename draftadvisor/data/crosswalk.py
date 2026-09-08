"""Id crosswalk: Sleeper player_id <-> nflverse gsis_id <-> FantasyPros id <-> ESPN id.

Sources (merged, Sleeper id wins conflicts):
* dynastyprocess ``db_playerids.csv`` (sleeper_id, gsis_id, fantasypros_id, espn_id, pfr_id, draft info, age)
* nflverse season rosters (gsis_id, sleeper_id, espn_id, pfr_id, esb_id, ...)
* name+position fuzzy match as last resort (normalised names)

Team defenses use the Sleeper convention (id = team abbreviation) and get the
ESPN D/ST id ``-16000 - proTeamId`` (see :func:`espn_dst_id`).
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

__all__ = ["Crosswalk", "ESPN_PRO_TEAM_IDS", "NFL_TEAMS", "build_crosswalk", "espn_dst_id", "load_playerids_raw",
           "normalize_name"]

#: Sleeper team abbreviation -> ESPN ``proTeamId`` (ESPN's own table, with WSH spelled WAS).
ESPN_PRO_TEAM_IDS: dict[str, int] = {
    "ATL": 1, "BUF": 2, "CHI": 3, "CIN": 4, "CLE": 5, "DAL": 6, "DEN": 7, "DET": 8, "GB": 9, "TEN": 10,
    "IND": 11, "KC": 12, "LV": 13, "LAR": 14, "MIA": 15, "MIN": 16, "NE": 17, "NO": 18, "NYG": 19, "NYJ": 20,
    "PHI": 21, "ARI": 22, "PIT": 23, "LAC": 24, "SF": 25, "SEA": 26, "TB": 27, "WAS": 28, "CAR": 29, "JAX": 30,
    "BAL": 33, "HOU": 34,
}
#: Every team defense the crosswalk knows about (Sleeper DEF ids).
NFL_TEAMS: tuple[str, ...] = tuple(sorted(ESPN_PRO_TEAM_IDS))
_ESPN_DST_BASE = -16000


def espn_dst_id(team: str | None) -> str | None:
    """ESPN player id of a team defense: ``-16000 - proTeamId`` (``"PIT"`` -> ``"-16023"``)."""
    pro = ESPN_PRO_TEAM_IDS.get(to_sleeper_team(team) or "")
    return None if pro is None else str(_ESPN_DST_BASE - pro)


def _clean_id(v) -> str | None:
    if v is None or (isinstance(v, float) and v != v):
        return None
    s = str(v).strip()
    if s in ("", "nan", "NA", "<NA>"):
        return None
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _clean_espn_id(v) -> str | None:
    """ESPN ids are integers (``4837248.0`` in the CSVs -> ``"4837248"``); D/ST ids are negative."""
    s = _clean_id(v)
    if s is None:
        return None
    if s.lstrip("-").isdigit() and s != "0":
        return s
    return None


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
    sleeper_to_espn: dict[str, str] = field(default_factory=dict)
    espn_to_sleeper: dict[str, str] = field(default_factory=dict)
    pfr_to_gsis: dict[str, str] = field(default_factory=dict)
    #: sleeper_id -> {"name","position","team","age","draft_year","draft_round","draft_ovr","espn_id","years_exp","status",...}
    attrs: dict[str, dict] = field(default_factory=dict)
    #: (normalized name, position) -> sleeper_id  (fallback matching)
    name_index: dict[tuple[str, str], str] = field(default_factory=dict)

    def gsis_for(self, sleeper_id: str) -> str | None:
        return self.sleeper_to_gsis.get(str(sleeper_id))

    def sleeper_for_gsis(self, gsis_id: str) -> str | None:
        return self.gsis_to_sleeper.get(str(gsis_id))

    def sleeper_for_fp(self, fp_id: str) -> str | None:
        return self.fp_to_sleeper.get(_clean_id(fp_id) or "")

    def espn_for(self, sleeper_id: str) -> str | None:
        return self.sleeper_to_espn.get(str(sleeper_id))

    def sleeper_for_espn(self, espn_id: str | int | None) -> str | None:
        return self.espn_to_sleeper.get(_clean_espn_id(espn_id) or "")

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
            espn_id: str | int | float | None = None, **attrs) -> None:
        sid = _clean_id(sleeper_id)
        g = _clean_id(gsis_id)
        f = _clean_id(fp_id)
        p = _clean_id(pfr_id)
        e = _clean_espn_id(espn_id)
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
        if e:
            # First source wins; the reverse map only ever mirrors the winning forward value.
            self.sleeper_to_espn.setdefault(sid, e)
            if self.sleeper_to_espn[sid] == e:
                self.espn_to_sleeper.setdefault(e, sid)
            attrs = {"espn_id": self.sleeper_to_espn[sid], **attrs}
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
                   name=d.get("name"), position=pos, espn_id=d.get("espn_id"), team=d.get("team"), age=d.get("age"),
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
                   name=d.get("full_name"), position=pos, espn_id=d.get("espn_id"),
                   team=d.get("team") if s == season else None,
                   years_exp=d.get("years_exp") if s == season else None,
                   status=d.get("status") if s == season else None,
                   birth_date=d.get("birth_date"), draft_number=d.get("draft_number"),
                   entry_year=d.get("entry_year"))
    # Team defenses: Sleeper ids are team abbreviations; ESPN D/ST ids derive from the pro-team id.
    for t in NFL_TEAMS:
        cw.add(t, gsis_id=t, name=f"{t} Defense", position="DEF", espn_id=espn_dst_id(t), team=t)
    log.info("crosswalk: %d sleeper ids, %d gsis links, %d fantasypros links, %d espn links",
             len(cw.attrs), len(cw.sleeper_to_gsis), len(cw.sleeper_to_fp), len(cw.sleeper_to_espn))
    return cw
