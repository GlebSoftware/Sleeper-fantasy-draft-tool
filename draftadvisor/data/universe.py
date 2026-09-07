"""Build the draftable-player universe (dict of Sleeper id -> Player).

Primary source is Sleeper's ``/players/nfl`` payload (live, ~5MB, cached daily).
Offline (or as enrichment) we use the id crosswalk, the current nflverse roster,
FantasyPros ECR and the schedule's bye weeks.
"""
from __future__ import annotations

import logging
from typing import Iterable, Mapping

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]

from ..config import DEFAULT_SEASON, SKILL_POSITIONS
from ..models import Player
from .crosswalk import Crosswalk, _clean_id, normalize_name

log = logging.getLogger(__name__)

_ACTIVE_STATUSES = {"Active", "Injured Reserve", "PUP", "Suspended", "Non Football Injury", "Physically Unable to Perform"}


def _f(v) -> float | None:
    try:
        if v is None or (isinstance(v, float) and v != v):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _i(v) -> int | None:
    f = _f(v)
    return None if f is None else int(f)


def player_from_sleeper(pid: str, p: Mapping) -> Player | None:
    """Convert one entry of Sleeper's players payload. Returns None if not fantasy-relevant."""
    pos = p.get("position")
    fps = tuple(p.get("fantasy_positions") or ())
    if pos not in SKILL_POSITIONS:
        # Some players are listed as e.g. "FB" with fantasy_positions ["RB"].
        pos = next((x for x in fps if x in SKILL_POSITIONS), None)
        if pos is None:
            return None
    name = p.get("full_name") or f"{p.get('first_name', '')} {p.get('last_name', '')}".strip()
    if pos == "DEF":
        name = f"{p.get('first_name', '')} {p.get('last_name', '')}".strip() or f"{pid} Defense"
    return Player(
        player_id=str(pid),
        name=name,
        position=pos,
        team=p.get("team"),
        fantasy_positions=fps or (pos,),
        age=_f(p.get("age")),
        years_exp=_i(p.get("years_exp")),
        injury_status=p.get("injury_status"),
        status=p.get("status"),
        depth_chart_order=_i(p.get("depth_chart_order")),
        depth_chart_position=p.get("depth_chart_position"),
        search_rank=_i(p.get("search_rank")),
        gsis_id=p.get("gsis_id"),
        metadata={k: p.get(k) for k in ("number", "college", "height", "weight", "birth_date", "news_updated",
                                         "injury_notes", "injury_body_part", "practice_participation") if p.get(k) is not None},
    )


def players_from_sleeper(payload: Mapping[str, Mapping], active_only: bool = True) -> dict[str, Player]:
    out: dict[str, Player] = {}
    for pid, p in payload.items():
        pl = player_from_sleeper(str(pid), p)
        if pl is None:
            continue
        if active_only and pl.position != "DEF":
            if pl.team is None:
                continue
            if pl.status and pl.status not in _ACTIVE_STATUSES:
                continue
        out[pl.player_id] = pl
    return out


_ROSTER_STATUS = {"ACT": "Active", "RES": "Injured Reserve", "PUP": "PUP", "SUS": "Suspended", "DEV": "Practice Squad",
                  "CUT": "Inactive", "RET": "Inactive", "EXE": "Inactive"}
_ROSTER_POS = {"FB": "RB", "HB": "RB", "PK": "K"}
#: Prefix of the synthetic player id used for roster players with no Sleeper id in any source.
SYNTHETIC_ID_PREFIX = "nfl:"


def players_from_crosswalk(cw: Crosswalk, roster: pd.DataFrame | None = None, season: int = DEFAULT_SEASON) -> dict[str, Player]:
    """Offline universe: every crosswalk entry with a fantasy position, enriched with the roster.

    Active roster skill players that have no Sleeper id in any source (typically
    rookies the id tables have not caught up with) are included under a synthetic
    id ``nfl:<gsis_id>`` and registered in ``cw`` so history/ML lookups work.
    """
    out: dict[str, Player] = {}
    ro_by_gsis: dict[str, dict] = {}
    if roster is not None and len(roster):
        for r in roster.itertuples(index=False):
            d = r._asdict()
            if _clean_id(d.get("gsis_id")):
                ro_by_gsis[_clean_id(d["gsis_id"])] = d
    # Roster players with no Sleeper link anywhere -> synthetic ids (before the main loop
    # so cw.attrs contains them and they flow through the same enrichment path).
    synthetic: list[str] = []
    for g, d in ro_by_gsis.items():
        if cw.sleeper_for_gsis(g) is not None:
            continue
        pos = _ROSTER_POS.get(d.get("position"), d.get("position"))
        if pos not in SKILL_POSITIONS or pos == "DEF":
            continue
        if _ROSTER_STATUS.get(d.get("status"), d.get("status")) not in _ACTIVE_STATUSES:
            continue
        sid = f"{SYNTHETIC_ID_PREFIX}{g}"
        cw.add(sid, gsis_id=g, pfr_id=d.get("pfr_id"), name=d.get("full_name"), position=pos,
               team=d.get("team"), years_exp=d.get("years_exp"), status=d.get("status"),
               birth_date=d.get("birth_date"), draft_number=d.get("draft_number"), entry_year=d.get("entry_year"))
        synthetic.append(f"{d.get('full_name')} {pos} {d.get('team')}")
    if synthetic:
        log.info("offline universe: %d active roster players have no Sleeper id, using synthetic ids: %s",
                 len(synthetic), ", ".join(synthetic))
    for sid, a in cw.attrs.items():
        pos = a.get("position")
        if pos not in SKILL_POSITIONS:
            continue
        g = cw.gsis_for(sid)
        ro = ro_by_gsis.get(g or "", {})
        team = ro.get("team") or a.get("team")
        if pos != "DEF" and roster is not None and not ro:
            # not on a current roster -> skip (retired / free agent)
            continue
        status = ro.get("status")
        age = _f(a.get("age"))
        if age is None and ro.get("birth_date"):
            try:
                bd = pd.Timestamp(ro["birth_date"])
                age = round((pd.Timestamp(f"{season}-09-01") - bd).days / 365.25, 1)
            except Exception:  # noqa: BLE001
                age = None
        years_exp = _i(ro.get("years_exp"))
        draft_year = _i(a.get("draft_year")) or _i(ro.get("entry_year"))
        if years_exp is None and draft_year is not None:
            years_exp = max(0, season - draft_year)
        out[sid] = Player(
            player_id=sid,
            name=a.get("name") or ro.get("full_name") or sid,
            position=pos,
            team=team if isinstance(team, str) else None,
            fantasy_positions=(pos,),
            age=age,
            years_exp=years_exp,
            injury_status=None,
            status=_ROSTER_STATUS.get(status, status),
            depth_chart_order=None,
            gsis_id=g,
            fantasypros_id=cw.sleeper_to_fp.get(sid),
            draft_year=draft_year,
            draft_round=_i(a.get("draft_round")),
            draft_pick_overall=_i(a.get("draft_ovr")) or _i(ro.get("draft_number")),
        )
    return out


def enrich_players(players: dict[str, Player], cw: Crosswalk, ecr: pd.DataFrame | None = None,
                   byes: Mapping[str, int] | None = None, pos_ecr: pd.DataFrame | None = None) -> dict[str, Player]:
    """Attach crosswalk ids, draft capital, ECR and bye weeks in place. Returns ``players``."""
    for sid, pl in players.items():
        a = cw.attrs.get(sid, {})
        pl.gsis_id = pl.gsis_id or cw.gsis_for(sid)
        pl.fantasypros_id = pl.fantasypros_id or cw.sleeper_to_fp.get(sid)
        pl.draft_year = pl.draft_year or _i(a.get("draft_year")) or _i(a.get("entry_year"))
        pl.draft_round = pl.draft_round or _i(a.get("draft_round"))
        pl.draft_pick_overall = pl.draft_pick_overall or _i(a.get("draft_ovr")) or _i(a.get("draft_number"))
        if pl.age is None:
            pl.age = _f(a.get("age"))
        if byes and pl.team:
            pl.bye_week = byes.get(pl.team)
    if ecr is not None and len(ecr):
        by_fp = {str(r.fantasypros_id): r for r in ecr.itertuples(index=False)}
        # Name fallback: (name, pos, team) first; (name, pos) only when that key is
        # unique on the board (an ambiguous key must not silently match the last row).
        by_name_team: dict[tuple[str, str, str | None], object] = {}
        by_name: dict[tuple[str, str], object] = {}
        ambiguous: set[tuple[str, str]] = set()
        for r in ecr.itertuples(index=False):
            key = (normalize_name(r.player), r.pos)
            by_name_team.setdefault((*key, getattr(r, "team", None)), r)
            if key in by_name:
                ambiguous.add(key)
            else:
                by_name[key] = r
        for sid, pl in players.items():
            r = by_fp.get(pl.fantasypros_id or "")
            if r is None:
                if pl.position == "DEF":
                    r = next((x for x in ecr.itertuples(index=False) if x.pos == "DEF" and x.team == pl.team), None)
                else:
                    key = (normalize_name(pl.name), pl.position)
                    r = by_name_team.get((*key, pl.team))
                    if r is None and key not in ambiguous:
                        r = by_name.get(key)
            if r is not None:
                pl.ecr = _f(r.ecr)
                pl.ecr_sd = _f(r.sd)
                if pl.fantasypros_id is None:
                    pl.fantasypros_id = str(r.fantasypros_id)
                if pl.bye_week is None:
                    pl.bye_week = _i(getattr(r, "bye", None))
    if pos_ecr is not None and len(pos_ecr):
        by_fp = {(str(r.fantasypros_id), r.pos): r for r in pos_ecr.itertuples(index=False)}
        for pl in players.values():
            r = by_fp.get((pl.fantasypros_id or "", pl.position))
            if r is not None:
                pl.ecr_pos_rank = _f(r.pos_ecr)
    return players


def assign_adp(players: dict[str, Player], adp_by_id: Mapping[str, float] | None, source: str) -> int:
    """Set ``adp`` from a Sleeper ADP map; fall back to ECR then search_rank. Returns # with real ADP."""
    n = 0
    for sid, pl in players.items():
        v = adp_by_id.get(sid) if adp_by_id else None
        if v is not None and v > 0:
            pl.adp, pl.adp_source = float(v), source
            n += 1
    # Fallback for players without ADP: ECR (rank ~ ADP for the top of the board).
    for pl in players.values():
        if pl.adp is None and pl.ecr is not None:
            pl.adp, pl.adp_source = float(pl.ecr), "ecr"
    return n


def filter_relevant(players: Iterable[Player], max_per_position: Mapping[str, int] | None = None) -> list[Player]:
    """Keep the most draft-relevant players by ADP/ECR/search_rank per position."""
    caps = {"QB": 45, "RB": 90, "WR": 110, "TE": 45, "K": 34, "DEF": 32}
    if max_per_position:
        caps.update(max_per_position)
    by_pos: dict[str, list[Player]] = {}
    for p in players:
        by_pos.setdefault(p.position, []).append(p)
    out: list[Player] = []
    for pos, lst in by_pos.items():
        def key(p: Player) -> float:
            if p.adp is not None:
                return p.adp
            if p.ecr is not None:
                return 1000 + p.ecr
            return 10000 + (p.search_rank or 9_999_999)
        lst.sort(key=key)
        out.extend(lst[: caps.get(pos, 60)])
    return out
