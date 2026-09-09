"""Lean (pandas-free) runtime used by the stateless web server (local and Vercel).

Everything heavy was precomputed into ``web_bundle/`` by ``draftadvisor bundle``
(model outputs, offline universe, player-season stat totals). At request time
this module fetches the small live inputs (Sleeper players/projections/ADP,
FantasyPros ECR), rebuilds the league-specific projections with the
ScoringEngine + market rank curves, and hands an Advisor back. Only numpy,
httpx and the pure-Python parts of the package are imported.

Other platforms (ESPN) reuse the same universe and blend: :func:`build_lean_context`
takes an ADP override (the platform's own ADP wins, ECR fills the rest), projected
stat lines for players Sleeper has no projection for, and placeholder players for
ids the universe does not know.
"""
from __future__ import annotations

import csv
import dataclasses
import io
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import httpx

from .config import DEFAULT_SEASON, SKILL_POSITIONS, Settings
from .data.names import normalize_name
from .models import DraftSettings, LeagueSettings, Player, Projection, ResearchNote
from .scoring import DEFAULT_SCORING, ScoringEngine

log = logging.getLogger(__name__)

BUNDLE_DIR = Path(os.environ.get("DRAFTADVISOR_BUNDLE", Path(__file__).resolve().parent.parent / "web_bundle"))
ECR_URL = "https://raw.githubusercontent.com/dynastyprocess/data/master/files/db_fpecr_latest.csv"
ECR_TTL = 6 * 3600
SLEEPER_PLAYERS_TTL = 6 * 3600
SLEEPER_PROJ_TTL = 3600
CONTEXT_TTL = 10 * 60

_ACTIVE = {"Active", "Injured Reserve", "PUP", "Suspended", "Non Football Injury", "Physically Unable to Perform"}


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------


def _player_from_dict(d: Mapping[str, Any]) -> Player:
    fps = d.get("fantasy_positions") or [d.get("position")]
    return Player(
        player_id=str(d["player_id"]), name=d.get("name") or "", position=d.get("position") or "",
        team=d.get("team"), fantasy_positions=tuple(fps), age=d.get("age"), years_exp=d.get("years_exp"),
        injury_status=d.get("injury_status"), status=d.get("status"), depth_chart_order=d.get("depth_chart_order"),
        depth_chart_position=d.get("depth_chart_position"), bye_week=d.get("bye_week"), search_rank=d.get("search_rank"),
        gsis_id=d.get("gsis_id"), fantasypros_id=d.get("fantasypros_id"), espn_id=d.get("espn_id"), draft_year=d.get("draft_year"),
        draft_round=d.get("draft_round"), draft_pick_overall=d.get("draft_pick_overall"), adp=d.get("adp"),
        adp_source=d.get("adp_source"), ecr=d.get("ecr"), ecr_sd=d.get("ecr_sd"), ecr_pos_rank=d.get("ecr_pos_rank"),
    )


def player_to_dict(pl: Player) -> dict:
    return {
        "player_id": pl.player_id, "name": pl.name, "position": pl.position, "team": pl.team,
        "fantasy_positions": list(pl.fantasy_positions), "age": pl.age, "years_exp": pl.years_exp,
        "injury_status": pl.injury_status, "status": pl.status, "depth_chart_order": pl.depth_chart_order,
        "depth_chart_position": pl.depth_chart_position, "bye_week": pl.bye_week, "search_rank": pl.search_rank,
        "gsis_id": pl.gsis_id, "fantasypros_id": pl.fantasypros_id, "espn_id": pl.espn_id, "draft_year": pl.draft_year,
        "draft_round": pl.draft_round, "draft_pick_overall": pl.draft_pick_overall,
        # build-time market snapshot: used only when the live ECR fetch fails
        "ecr": pl.ecr, "ecr_sd": pl.ecr_sd, "ecr_pos_rank": pl.ecr_pos_rank,
    }


@dataclass
class Bundle:
    players: dict[str, Player]                 # offline universe (Sleeper ids; DEF = team abbr)
    ml: dict[str, dict]                        # player_id -> {"pred_<key>": v, "pred_games", "ppg_std_ppr", "rookie_flag", "position"}
    season_totals: list[dict]                  # {"season","position","player_id","games","stats":{key: v}}
    byes: dict[str, int]
    #: NFL schedule, defence-vs-position multipliers and measured weekly spread (see
    #: draftadvisor/data/inseason.py). Empty in a bundle built before in-season support.
    inseason: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    gsis_to_pid: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, directory: Path | None = None) -> "Bundle":
        d = Path(directory or BUNDLE_DIR)
        if not (d / "players.json").exists():
            raise FileNotFoundError(f"web bundle not found in {d}; run `draftadvisor bundle`")
        players_raw = json.loads((d / "players.json").read_text(encoding="utf-8"))
        players = {str(p["player_id"]): _player_from_dict(p) for p in players_raw}
        ml = json.loads((d / "ml.json").read_text(encoding="utf-8")) if (d / "ml.json").exists() else {}
        totals = json.loads((d / "season_totals.json").read_text(encoding="utf-8")) if (d / "season_totals.json").exists() else []
        byes = json.loads((d / "byes.json").read_text(encoding="utf-8")) if (d / "byes.json").exists() else {}
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8")) if (d / "meta.json").exists() else {}
        ins = json.loads((d / "inseason.json").read_text(encoding="utf-8")) if (d / "inseason.json").exists() else {}
        b = cls(players=players, ml={str(k): v for k, v in ml.items()}, season_totals=totals,
                byes={k: int(v) for k, v in byes.items()}, inseason=ins, meta=meta)
        b.gsis_to_pid = {pl.gsis_id: pid for pid, pl in players.items() if pl.gsis_id}
        return b


_BUNDLE: Bundle | None = None
_BUNDLE_LOCK = threading.Lock()


def get_bundle(directory: Path | None = None) -> Bundle:
    global _BUNDLE
    with _BUNDLE_LOCK:
        if _BUNDLE is None:
            t0 = time.perf_counter()
            _BUNDLE = Bundle.load(directory)
            log.info("bundle loaded: %d players, %d ml rows, %d season totals in %.0f ms", len(_BUNDLE.players),
                     len(_BUNDLE.ml), len(_BUNDLE.season_totals), (time.perf_counter() - t0) * 1000)
        return _BUNDLE


# ---------------------------------------------------------------------------
# Live inputs (cached per process)
# ---------------------------------------------------------------------------


class _TTLCache:
    def __init__(self) -> None:
        self._items: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str, ttl: float) -> Any | None:
        with self._lock:
            hit = self._items.get(key)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]
        return None

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._items[key] = (time.time(), value)

    def invalidate(self, prefix: str = "") -> None:
        with self._lock:
            for k in [k for k in self._items if k.startswith(prefix)]:
                self._items.pop(k, None)


CACHE = _TTLCache()


async def fetch_ecr(http: httpx.AsyncClient | None = None) -> list[dict]:
    """FantasyPros ECR rows (overall + superflex + positional) parsed with the csv module."""
    hit = CACHE.get("ecr", ECR_TTL)
    if hit is not None:
        return hit
    own = http is None
    client = http or httpx.AsyncClient(timeout=30.0, follow_redirects=True)
    try:
        r = await client.get(ECR_URL)
        r.raise_for_status()
        rows = []
        for row in csv.DictReader(io.StringIO(r.text)):
            if row.get("page_type") not in ("redraft-overall", "redraft-op", "redraft-qb", "redraft-rb", "redraft-wr",
                                            "redraft-te", "redraft-k", "redraft-dst"):
                continue
            try:
                rows.append({
                    "page_type": row["page_type"], "fantasypros_id": str(int(float(row["id"]))) if row.get("id") else None,
                    "player": row.get("player") or "", "pos": {"DST": "DEF"}.get(row.get("pos"), row.get("pos")),
                    "team": {"JAC": "JAX", "LA": "LAR"}.get(row.get("team"), row.get("team")),
                    "ecr": float(row["ecr"]), "sd": float(row["sd"]) if row.get("sd") not in (None, "", "NA") else None,
                    "bye": int(float(row["bye"])) if row.get("bye") not in (None, "", "NA") else None,
                })
            except (KeyError, ValueError):
                continue
        CACHE.set("ecr", rows)
        return rows
    finally:
        if own:
            await client.aclose()


def _player_from_sleeper(pid: str, p: Mapping[str, Any]) -> Player | None:
    pos = p.get("position")
    fps = tuple(p.get("fantasy_positions") or ())
    if pos not in SKILL_POSITIONS:
        pos = next((x for x in fps if x in SKILL_POSITIONS), None)
        if pos is None:
            return None
    name = p.get("full_name") or f"{p.get('first_name', '')} {p.get('last_name', '')}".strip()
    if pos == "DEF":
        name = f"{p.get('first_name', '')} {p.get('last_name', '')}".strip() or f"{pid} Defense"

    def _i(v: Any) -> int | None:
        try:
            return int(float(v)) if v is not None else None
        except (TypeError, ValueError):
            return None

    def _f(v: Any) -> float | None:
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    return Player(player_id=str(pid), name=name, position=pos, team=p.get("team"), fantasy_positions=fps or (pos,),
                  age=_f(p.get("age")), years_exp=_i(p.get("years_exp")), injury_status=p.get("injury_status") or None,
                  status=p.get("status"), depth_chart_order=_i(p.get("depth_chart_order")),
                  depth_chart_position=p.get("depth_chart_position") if isinstance(p.get("depth_chart_position"), str) else None,
                  search_rank=_i(p.get("search_rank")), gsis_id=p.get("gsis_id"),
                  espn_id=str(p["espn_id"]) if p.get("espn_id") not in (None, "") else None)


def players_from_sleeper_payload(payload: Mapping[str, Mapping], base: Mapping[str, Player]) -> dict[str, Player]:
    """Sleeper's live players payload -> universe, enriched with bundle ids (gsis/fantasypros/draft capital)."""
    out: dict[str, Player] = {}
    for pid, p in payload.items():
        pl = _player_from_sleeper(str(pid), p)
        if pl is None:
            continue
        if pl.position != "DEF":
            if pl.team is None or (pl.status and pl.status not in _ACTIVE):
                continue
        b = base.get(pl.player_id)
        if b is not None:
            pl.gsis_id = pl.gsis_id or b.gsis_id
            pl.fantasypros_id = b.fantasypros_id
            pl.espn_id = pl.espn_id or b.espn_id
            pl.draft_year, pl.draft_round, pl.draft_pick_overall = b.draft_year, b.draft_round, b.draft_pick_overall
            if pl.age is None:
                pl.age = b.age
            if pl.years_exp is None:
                pl.years_exp = b.years_exp
        out[pl.player_id] = pl
    return out


def enrich_with_ecr(players: Mapping[str, Player], ecr_rows: Iterable[dict], byes: Mapping[str, int],
                    superflex: bool = False) -> None:
    """Attach overall ECR (PPR or superflex board), positional rank and bye weeks in place."""
    page = "redraft-op" if superflex else "redraft-overall"
    overall = [r for r in ecr_rows if r["page_type"] == page] or [r for r in ecr_rows if r["page_type"] == "redraft-overall"]
    pos_pages = {"QB": "redraft-qb", "RB": "redraft-rb", "WR": "redraft-wr", "TE": "redraft-te", "K": "redraft-k", "DEF": "redraft-dst"}
    by_fp = {r["fantasypros_id"]: r for r in overall if r["fantasypros_id"]}
    by_name: dict[tuple[str, str, str | None], dict] = {}
    by_name_pos: dict[tuple[str, str], list[dict]] = {}
    for r in overall:
        key = (normalize_name(r["player"]), r["pos"])
        by_name[(key[0], key[1], r.get("team"))] = r
        by_name_pos.setdefault(key, []).append(r)
    def_by_team = {r.get("team"): r for r in overall if r["pos"] == "DEF"}
    pos_rank = {(r["fantasypros_id"], pos): r for pos, pg in pos_pages.items() for r in ecr_rows
                if r["page_type"] == pg and r["fantasypros_id"]}
    for pl in players.values():
        if pl.bye_week is None and pl.team:
            pl.bye_week = byes.get(pl.team)
        r = by_fp.get(pl.fantasypros_id or "")
        if r is None:
            if pl.position == "DEF":
                r = def_by_team.get(pl.team)
            else:
                r = by_name.get((normalize_name(pl.name), pl.position, pl.team))
                if r is None:
                    cands = by_name_pos.get((normalize_name(pl.name), pl.position), [])
                    r = cands[0] if len(cands) == 1 else None
        if r is not None:
            pl.ecr, pl.ecr_sd = r["ecr"], r.get("sd")
            pl.fantasypros_id = pl.fantasypros_id or r["fantasypros_id"]
            if pl.bye_week is None:
                pl.bye_week = r.get("bye")
        pr = pos_rank.get((pl.fantasypros_id, pl.position)) if pl.fantasypros_id else None
        if pr is not None:
            pl.ecr_pos_rank = pr["ecr"]


def assign_adp(players: Mapping[str, Player], adp: Mapping[str, float] | None, source: str) -> int:
    n = 0
    for pid, pl in players.items():
        v = adp.get(pid) if adp else None
        if v is not None and v > 0:
            pl.adp, pl.adp_source = float(v), source
            n += 1
        elif pl.ecr is not None:
            pl.adp, pl.adp_source = float(pl.ecr), "ecr"
    return n


# ---------------------------------------------------------------------------
# Rank curves from season totals (pure Python)
# ---------------------------------------------------------------------------


def rank_curves_from_totals(totals: Iterable[dict], engine: ScoringEngine, league: LeagueSettings | None,
                            max_rank: int = 90) -> dict:
    from .projections.blend import RankCurve, market_replacement_rank

    by_key: dict[tuple[int, str], list[float]] = {}
    for row in totals:
        pos = row.get("position")
        if pos not in SKILL_POSITIONS:
            continue
        pts = engine.score(row.get("stats") or {}, pos)
        by_key.setdefault((int(row["season"]), pos), []).append(pts)
    out = {}
    for pos in SKILL_POSITIONS:
        seasons = [sorted(v, reverse=True)[:max_rank] for (s, p), v in by_key.items() if p == pos]
        if len(seasons) < 2:
            continue
        n = min(len(s) for s in seasons)
        if n < 5:
            continue
        anchors = [(i + 1, sum(s[i] for s in seasons) / len(seasons)) for i in range(n)]
        try:
            out[pos] = RankCurve(anchors, market_replacement_rank(league, pos))
        except Exception as e:  # noqa: BLE001
            log.warning("rank curve %s failed: %s", pos, e)
    return out


# ---------------------------------------------------------------------------
# League context
# ---------------------------------------------------------------------------


@dataclass
class LeanContext:
    settings: Settings
    engine: ScoringEngine
    league: LeagueSettings
    draft: DraftSettings | None
    players: dict[str, Player]
    projections: dict[str, Projection]
    advisor: Any
    notes: dict[str, ResearchNote]
    byes: dict[str, int]
    sources: dict = field(default_factory=dict)
    built_at: float = field(default_factory=time.time)
    fingerprint: str = ""
    id_map: Any = None          # platform id map (ESPN) indexed over ``players``: picks resolve into this universe


def lean_universe(bundle: Bundle, sleeper_players: Mapping | None) -> tuple[dict[str, Player], str]:
    """The player universe of a context and its source: Sleeper's live payload when given, else the bundle."""
    if sleeper_players:
        return players_from_sleeper_payload(sleeper_players, bundle.players), "sleeper"
    return {pid: _player_from_dict(player_to_dict(pl)) for pid, pl in bundle.players.items()}, "bundle"


def default_league(scoring: str = "half_ppr", teams: int = 12) -> LeagueSettings:
    rec = {"ppr": 1.0, "half_ppr": 0.5, "std": 0.0}.get(scoring, 0.5)
    return LeagueSettings(league_id=f"default_{scoring}", name=f"Default league ({scoring})", season=DEFAULT_SEASON,
                          total_rosters=teams, roster_positions=["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF"] + ["BN"] * 6,
                          scoring_settings=dict(DEFAULT_SCORING, rec=rec))


def context_fingerprint(league: LeagueSettings, has_sleeper_proj: bool, has_sleeper_players: bool, n_notes: int) -> str:
    import hashlib

    blob = json.dumps({"s": league.scoring_settings, "r": league.roster_positions, "t": league.total_rosters,
                       "p": has_sleeper_proj, "u": has_sleeper_players, "n": n_notes}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def merge_projection_lines(primary: Mapping[str, Mapping] | None, fallback: Mapping[str, Mapping] | None) -> dict[str, Mapping]:
    """``primary`` (Sleeper's season projections) completed with ``fallback`` lines for the players it lacks."""
    out: dict[str, Mapping] = {str(k): v for k, v in (fallback or {}).items() if v}
    out.update({str(k): v for k, v in (primary or {}).items() if v})
    return out


def relabel_projection_source(projections: Mapping[str, Projection], player_ids: Iterable[str], source: str) -> int:
    """Rename the ``sleeper`` blend component to ``source`` (and flag it) for projections whose stat line
    came from a fallback provider; returns how many were relabelled."""
    n = 0
    for pid in player_ids:
        pr = projections.get(pid)
        if pr is None or "sleeper" not in pr.components:
            continue
        pr.components[source] = pr.components.pop("sleeper")
        if "sleeper" in pr.weights:
            pr.weights[source] = pr.weights.pop("sleeper")
        if f"{source}_proj" not in pr.flags:
            pr.flags.append(f"{source}_proj")
        n += 1
    return n


def build_lean_context(bundle: Bundle, league: LeagueSettings | None, draft: DraftSettings | None, *,
                       sleeper_players: Mapping | None, sleeper_proj: Mapping | None, ecr_rows: Iterable[dict],
                       notes: Mapping[str, ResearchNote], settings: Settings | None = None,
                       adp_override: Mapping[str, float] | None = None, adp_source: str | None = None,
                       proj_fallback: Mapping[str, Mapping] | None = None, proj_fallback_source: str = "fallback",
                       extra_players: Mapping[str, Player] | None = None,
                       players: Mapping[str, Player] | None = None, players_source: str | None = None,
                       id_map: Any = None) -> LeanContext:
    """Build players + projections + advisor for one league from the bundle and live inputs.

    ``adp_override`` (``{player_id: adp}``, labelled ``adp_source``) replaces Sleeper's ADP when given -
    players without an entry fall back to ECR. ``proj_fallback`` (``{player_id: Sleeper-key season stats}``)
    supplies stat lines for players missing from ``sleeper_proj``; their projections are labelled
    ``proj_fallback_source``. ``extra_players`` are added to the universe when their id is unknown
    (placeholders for platform-only players, so picks and rosters always resolve to a name).
    ``players`` (with ``players_source``) is a universe already built by :func:`lean_universe` - the caller
    passes it when something else (an ``id_map`` over the same objects) must agree with the context.
    """
    from .projections.blend import Projector
    from .strategy.recommend import Advisor

    t0 = time.perf_counter()
    settings = settings or Settings.from_env()
    league = league or default_league()
    engine = ScoringEngine(league.scoring_settings)
    if players is not None:
        players = dict(players)
        src_players = players_source or ("sleeper" if sleeper_players else "bundle")
    else:
        players, src_players = lean_universe(bundle, sleeper_players)
    n_extra = 0
    for pid, pl in (extra_players or {}).items():
        if pid not in players:
            players[pid] = dataclasses.replace(pl)
            n_extra += 1
    enrich_with_ecr(players, ecr_rows, bundle.byes, superflex=league.is_superflex)
    adp_src = "ecr"
    if adp_override:
        label = adp_source or "override"
        adp_src = label if assign_adp(players, adp_override, label) else "ecr"
    elif sleeper_proj:
        from .sleeper.parsing import adp_key_for, adp_map

        key = adp_key_for(league, draft)
        n = assign_adp(players, adp_map(sleeper_proj, key), f"sleeper_{key[4:]}")
        adp_src = f"sleeper_{key[4:]}" if n else "ecr"
    else:
        assign_adp(players, None, "ecr")
    fallback_ids = [pid for pid in (proj_fallback or {}) if not (sleeper_proj or {}).get(pid)]
    proj_lines = merge_projection_lines(sleeper_proj, proj_fallback) if proj_fallback else sleeper_proj
    # ML rows keyed by Sleeper id (bundle) or via gsis for live payload ids not in the bundle
    ml_pred: dict[str, dict] = {}
    for pid, pl in players.items():
        row = bundle.ml.get(pid)
        if row is None and pl.gsis_id and pl.gsis_id in bundle.gsis_to_pid:
            row = bundle.ml.get(bundle.gsis_to_pid[pl.gsis_id])
        if row:
            ml_pred[pid] = row
    curves = rank_curves_from_totals(bundle.season_totals, engine, league)
    projector = Projector(engine, league, settings, rank_curves=curves)
    projections = projector.project(players, ml_pred=ml_pred, sleeper_proj=proj_lines, notes=notes or None,
                                    byes=bundle.byes)
    n_fallback = relabel_projection_source(projections, fallback_ids, proj_fallback_source) if fallback_ids else 0
    advisor = Advisor(league, players, projections, settings)
    ctx = LeanContext(settings=settings, engine=engine, league=league, draft=draft, players=players,
                      projections=projections, advisor=advisor, notes=dict(notes or {}), byes=dict(bundle.byes),
                      sources={"players": src_players, "adp": adp_src, "sleeper_proj": bool(sleeper_proj),
                               "proj_fallback": n_fallback, "extra_players": n_extra,
                               "ml": len(ml_pred), "ecr": sum(1 for p in players.values() if p.ecr is not None),
                               "rank_curves": sorted(curves), "build_ms": round((time.perf_counter() - t0) * 1000)},
                      fingerprint=context_fingerprint(league, bool(sleeper_proj), bool(sleeper_players), len(notes or {})),
                      id_map=id_map)
    log.info("lean context %s: %d players, %d projections in %d ms", league.league_id, len(players), len(projections),
             ctx.sources["build_ms"])
    return ctx
