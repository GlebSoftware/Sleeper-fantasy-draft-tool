"""Application context and orchestration shared by every CLI command. See DESIGN.md §3.6.

:func:`build_context` turns raw data into an :class:`AppContext`::

    crosswalk -> players (Sleeper payload, else offline roster universe)
              -> enrich (FantasyPros ECR, bye weeks) -> ADP (Sleeper, else ECR)
              -> projections (cached < 12 h; else ML model -> offline shrinkage -> ECR-only)
              -> Advisor (+ optional Claude researcher and its cached notes)

Every step degrades gracefully: a missing data source is logged and skipped so the
tool still runs (with ECR-only projections in the worst case).
"""
from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from .config import (
    DEFAULT_SEASON,
    SKILL_POSITIONS,
    TRAIN_SEASONS,
    Settings,
    cache_dir,
    ensure_dirs,
    models_dir,
)
from .models import DraftSettings, LeagueSettings, Player, Projection, ResearchNote
from .scoring.engine import DEFAULT_SCORING, ScoringEngine

log = logging.getLogger(__name__)

__all__ = [
    "AppContext",
    "build_context",
    "rebuild_projections",
    "prep",
    "default_league",
    "projections_cache_path",
    "projection_fingerprint",
    "load_cached_projections",
    "save_projections",
    "ecr_only_projections",
    "load_snapshot_league",
    "fetch_league",
    "top_players_by_adp",
]

#: Cached projections older than this are rebuilt.
PROJECTIONS_TTL_HOURS = 12.0
#: (top season points, points lost per positional rank) for the ECR-only fallback.
_ECR_ONLY_SPEC: dict[str, tuple[float, float]] = {
    "QB": (380.0, 3.0), "RB": (330.0, 2.4), "WR": (320.0, 1.9), "TE": (230.0, 2.8),
    "K": (150.0, 1.3), "DEF": (140.0, 1.5),
}


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


@dataclass
class AppContext:
    """Everything a command needs, built once per process."""

    settings: Settings
    engine: ScoringEngine
    league: LeagueSettings
    draft: DraftSettings | None
    players: dict[str, Player]
    projections: dict[str, Projection]
    advisor: Any                                    # strategy.recommend.Advisor
    researcher: Any                                 # research.claude.ClaudeResearcher
    notes: dict[str, ResearchNote]
    byes: dict[str, int]
    crosswalk: Any                                  # data.crosswalk.Crosswalk
    sources: dict = field(default_factory=dict)     # what fed the build: {"players": "sleeper", "ml": "model", ...}
    sleeper_proj: Any = None                        # Sleeper season projections payload used (for re-blends)
    use_model: bool = True                          # whether the ML model was allowed for this build

    @property
    def timings(self) -> dict[str, float]:
        return dict(self.sources.get("timings", {}))


class _Timer:
    """Collects step timings (ms) for the log / ``sources["timings"]``."""

    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.last = self.t0
        self.steps: dict[str, float] = {}

    def lap(self, name: str) -> float:
        now = time.perf_counter()
        ms = (now - self.last) * 1000.0
        self.steps[name] = round(ms, 1)
        self.last = now
        log.info("build_context: %s in %.0f ms", name, ms)
        return ms

    @property
    def total_ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000.0


# ---------------------------------------------------------------------------
# Defaults & small helpers
# ---------------------------------------------------------------------------


def default_league(settings: Settings | None = None, teams: int = 12, rounds: int = 15) -> LeagueSettings:
    """A generic half-PPR league used when no Sleeper league is known (projections / mock)."""
    starters = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF"]
    return LeagueSettings(
        league_id="default",
        name="Default league (half PPR)",
        season=settings.season if settings else DEFAULT_SEASON,
        total_rosters=teams,
        roster_positions=starters + ["BN"] * max(0, rounds - len(starters)),
        scoring_settings=dict(DEFAULT_SCORING),
        settings={"num_teams": teams, "draft_rounds": rounds},
        status="pre_draft",
    )


def projections_cache_path(league_id: str | None) -> Path:
    key = str(league_id or "default").replace("/", "_")
    return cache_dir() / f"projections_{key}.json.gz"


def save_projections(path: Path, projections: Mapping[str, Projection], meta: Mapping[str, Any] | None = None) -> Path:
    """Serialise projections (``dataclasses.asdict``) to a gzipped JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": dict(meta or {}), "built_at": time.time(),
               "projections": {pid: dataclasses.asdict(p) for pid, p in projections.items()}}
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)
    tmp.replace(path)
    return path


def load_cached_projections(path: Path, max_age_hours: float | None = PROJECTIONS_TTL_HOURS,
                            fingerprint: str | None = None) -> dict[str, Projection] | None:
    """Projections from :func:`save_projections` if the file exists and is fresh, else None.

    When ``fingerprint`` is given it must equal ``meta["fingerprint"]`` of the cached file
    (see :func:`projection_fingerprint`); a cache built from different inputs (other
    scoring, no Sleeper projections, changed injury statuses, new research notes, ...)
    is reported as missing so the caller rebuilds.
    """
    if not path.exists():
        return None
    age_h = (time.time() - path.stat().st_mtime) / 3600.0
    if max_age_hours is not None and age_h > max_age_hours:
        log.info("projection cache %s is %.1f h old, rebuilding", path, age_h)
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
        if fingerprint is not None and (payload.get("meta") or {}).get("fingerprint") != fingerprint:
            log.info("projection cache %s was built from different inputs, rebuilding", path)
            return None
        fields = {f.name for f in dataclasses.fields(Projection)}
        out = {pid: Projection(**{k: v for k, v in d.items() if k in fields})
               for pid, d in payload.get("projections", {}).items()}
    except Exception as e:  # noqa: BLE001
        log.warning("projection cache %s unreadable (%s), rebuilding", path, e)
        return None
    if not out:
        return None
    log.info("loaded %d cached projections from %s (%.1f h old)", len(out), path, age_h)
    return out


def projection_fingerprint(league: LeagueSettings, players: Mapping[str, Player], *,
                           sleeper_proj: Mapping | None, sleeper_players: Mapping | bool | None, use_model: bool,
                           notes: Mapping[str, ResearchNote] | None) -> str:
    """Digest of everything that changes the projections for ``players`` (the relevant subset).

    League id + scoring settings + roster positions, whether Sleeper payloads were supplied,
    ``use_model``, the research notes and, per player, the inputs the Projector reads
    (injury status, depth chart, team, ECR, bye) plus his Sleeper projection line. A cache
    whose fingerprint differs must not be served: on draft day it would silently override
    fresh Sleeper projections, an IR designation or the notes the user paid for.
    """
    per_player = []
    for pid in sorted(players):
        pl = players[pid]
        note = notes.get(pid) if notes else None
        per_player.append((
            pid, pl.position, pl.team, pl.injury_status, pl.status, pl.depth_chart_order, pl.years_exp,
            pl.bye_week, pl.ecr, pl.ecr_sd,
            sorted((sleeper_proj.get(pid) or {}).items()) if sleeper_proj else None,
            (note.injury_risk, note.role_certainty) if note is not None else None,
        ))
    payload = {
        "league_id": league.league_id,
        "season": league.season,
        "scoring": sorted((league.scoring_settings or {}).items()),
        "roster": list(league.roster_positions or []),
        "sleeper_proj": bool(sleeper_proj),
        "sleeper_players": bool(sleeper_players),
        "use_model": bool(use_model),
        "n_notes": len(notes or {}),
        "players": per_player,
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:20]


def _market_rank(pl: Player) -> float:
    if pl.adp is not None and pl.adp > 0:
        return float(pl.adp)
    if pl.ecr is not None and pl.ecr > 0:
        return float(pl.ecr)
    if pl.search_rank is not None and pl.search_rank < 9_000_000:
        return 400.0 + float(pl.search_rank)
    return 10_000.0


def top_players_by_adp(players: Mapping[str, Player], n: int = 200) -> list[Player]:
    """Best ``n`` players by ADP (ECR / search rank fallback)."""
    return sorted(players.values(), key=_market_rank)[:n]


def ecr_only_projections(players: Mapping[str, Player], league: LeagueSettings | None = None) -> dict[str, Projection]:
    """Last-resort projections: a decreasing function of market rank within each position.

    ``points = max(0.25 * top, top - slope * (rank - 1))`` with per-position ``(top, slope)``
    from :data:`_ECR_ONLY_SPEC`; ``std = 15 %`` of points. Enough to draft sensibly when
    neither the ML model nor historical stats are available.
    """
    out: dict[str, Projection] = {}
    scale_rec = 1.0
    if league is not None and league.rec_points < 0.25:
        scale_rec = 0.92
    for pos in SKILL_POSITIONS:
        top, slope = _ECR_ONLY_SPEC[pos]
        if pos in ("RB", "WR", "TE"):
            top *= scale_rec
        group = sorted((p for p in players.values() if p.position == pos), key=_market_rank)
        for rank, pl in enumerate(group, start=1):
            pts = max(0.25 * top, top - slope * (rank - 1))
            if _market_rank(pl) >= 10_000.0:
                pts = 0.25 * top * 0.5
            std = max(8.0, 0.15 * pts)
            out[pl.player_id] = Projection(
                player_id=pl.player_id, position=pos, points=round(pts, 1), std=round(std, 1),
                ppg=round(pts / 17.0, 2), games=17.0, floor=round(pts - 0.84 * std, 1),
                ceiling=round(pts + 0.84 * std, 1), components={"ecr": round(pts, 1)}, weights={"ecr": 1.0},
                flags=["ecr_only"],
            )
    return out


# ---------------------------------------------------------------------------
# Build steps
# ---------------------------------------------------------------------------


def _build_crosswalk(season: int):
    from .data.crosswalk import Crosswalk, build_crosswalk

    try:
        return build_crosswalk(season)
    except Exception as e:  # noqa: BLE001
        log.warning("crosswalk unavailable: %s", e)
        return Crosswalk()


def _build_players(cw, season: int, sleeper_players: Mapping | None) -> tuple[dict[str, Player], str]:
    from .data.universe import players_from_crosswalk, players_from_sleeper

    if sleeper_players:
        players = players_from_sleeper(sleeper_players)
        if players:
            return players, "sleeper"
        log.warning("Sleeper players payload produced no players, using offline universe")
    roster = None
    try:
        from .data.nflverse import load_roster

        roster = load_roster(season)
    except Exception as e:  # noqa: BLE001
        log.warning("roster %s unavailable: %s", season, e)
    return players_from_crosswalk(cw, roster, season), "offline"


def _load_ecr(superflex: bool):
    """(overall ECR frame, positional ECR frame) or (None, None)."""
    try:
        from .data.fantasypros import load_ecr_raw, overall_ecr, positional_ecr

        raw = load_ecr_raw()
        return overall_ecr(raw, superflex=superflex), positional_ecr(raw)
    except Exception as e:  # noqa: BLE001
        log.warning("FantasyPros ECR unavailable: %s", e)
        return None, None


def _load_byes(season: int) -> dict[str, int]:
    try:
        from .data.nflverse import bye_weeks

        return dict(bye_weeks(season))
    except Exception as e:  # noqa: BLE001
        log.warning("bye weeks unavailable: %s", e)
        return {}


def _assign_adp(players: dict[str, Player], league: LeagueSettings | None, draft: DraftSettings | None,
                sleeper_proj: Mapping | None) -> str:
    from .data.universe import assign_adp

    if sleeper_proj:
        try:
            from .sleeper.parsing import adp_key_for, adp_map

            key = adp_key_for(league, draft)
            n = assign_adp(players, adp_map(sleeper_proj, key), f"sleeper_{key[4:]}")
            log.info("ADP from Sleeper %s for %d players", key, n)
            return f"sleeper_{key[4:]}"
        except Exception as e:  # noqa: BLE001
            log.warning("Sleeper ADP unavailable: %s", e)
    assign_adp(players, None, "ecr")
    return "ecr"


def _relevant_subset(players: dict[str, Player]) -> dict[str, Player]:
    from .data.universe import filter_relevant

    caps = {"QB": 60, "RB": 120, "WR": 150, "TE": 60, "K": 40, "DEF": 32}
    return {p.player_id: p for p in filter_relevant(players.values(), caps)}


def _model_predictions(players: Mapping[str, Player], cw, season: int):
    """ML predictions frame from the saved model (raises when unavailable)."""
    from .projections.features import build_inference_table
    from .projections.model import ProjectionModel, load_training_inputs

    path = models_dir() / "projection_model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"no trained model at {path} (run `draftadvisor prep` or `train`)")
    model = ProjectionModel.load(path)
    agg, team_ctx, _ = load_training_inputs(TRAIN_SEASONS)
    table = build_inference_table(agg, team_ctx, players, cw, season)
    pred = model.predict(table)
    if pred is None or len(pred) == 0:
        raise RuntimeError("model produced no predictions")
    return pred


def _offline_predictions(players: Mapping[str, Player], engine: ScoringEngine, cw, season: int):
    from .data.nflverse import load_canonical
    from .projections.blend import project_offline

    canonical = load_canonical([season - 1])
    pred = project_offline(players, engine, canonical, cw, season)
    if pred is None or len(pred) == 0:
        raise RuntimeError("offline projection produced no rows")
    return pred


def _ml_source(players: Mapping[str, Player], engine: ScoringEngine, cw, season: int, use_model: bool):
    """(predictions frame or None, source label)."""
    if use_model:
        try:
            return _model_predictions(players, cw, season), "model"
        except NotImplementedError as e:
            log.warning("projection model not implemented yet: %s", e)
        except Exception as e:  # noqa: BLE001
            log.warning("projection model unavailable (%s); falling back to offline projections", e)
    try:
        return _offline_predictions(players, engine, cw, season), "offline"
    except NotImplementedError as e:
        log.warning("offline projections not implemented: %s", e)
    except Exception as e:  # noqa: BLE001
        log.warning("offline projections unavailable (%s); ECR-only projections", e)
    return None, "none"


def _blend(players: Mapping[str, Player], engine: ScoringEngine, league: LeagueSettings | None, settings: Settings,
           ml_pred, sleeper_proj: Mapping | None, notes: Mapping[str, ResearchNote] | None,
           byes: Mapping[str, int]) -> dict[str, Projection]:
    from .projections.blend import Projector

    projector = Projector(engine, league, settings)
    season = league.season if league else settings.season
    projector.fit_market(_market_canonical(season))
    return projector.project(players, ml_pred=ml_pred, sleeper_proj=sleeper_proj, notes=notes or None, byes=byes)


def _market_canonical(season: int):
    """Locally available canonical seasons (never downloads) for the market rank curves."""
    from .config import cache_dir
    from .data.cache import raw_path
    from .data.nflverse import load_canonical_season

    frames = []
    for y in range(max(2019, season - 7), season):
        if (cache_dir() / f"canonical_{y}.csv.gz").exists() or raw_path(f"stats_player_week_{y}.csv").exists():
            try:
                frames.append(load_canonical_season(y))
            except Exception as e:  # noqa: BLE001
                log.info("canonical %s unavailable for market curves: %s", y, e)
    if not frames:
        return None
    import pandas as pd

    return pd.concat(frames, ignore_index=True, sort=False)


def build_projections(players: Mapping[str, Player], engine: ScoringEngine, league: LeagueSettings | None,
                      settings: Settings, cw, *, sleeper_proj: Mapping | None = None,
                      notes: Mapping[str, ResearchNote] | None = None, byes: Mapping[str, int] | None = None,
                      use_model: bool = True) -> tuple[dict[str, Projection], str]:
    """Projections for ``players`` and the label of the ML-ish source used
    (``"model"`` | ``"offline"`` | ``"none"`` | ``"ecr_only"``)."""
    season = league.season if league else settings.season
    ml_pred, source = _ml_source(players, engine, cw, season, use_model)
    if ml_pred is None and not sleeper_proj:
        usable = [p for p in players.values() if p.ecr is not None]
        if len(usable) < 8:
            log.warning("no projection source at all: ECR-only fallback")
            return ecr_only_projections(players, league), "ecr_only"
    try:
        proj = _blend(players, engine, league, settings, ml_pred, sleeper_proj, notes, byes or {})
    except NotImplementedError as e:
        log.warning("Projector not implemented (%s): ECR-only projections", e)
        return ecr_only_projections(players, league), "ecr_only"
    except Exception as e:  # noqa: BLE001
        log.exception("Projector failed (%s): ECR-only projections", e)
        return ecr_only_projections(players, league), "ecr_only"
    nonzero = sum(1 for p in proj.values() if p.points > 0)
    if nonzero < 8:
        log.warning("Projector produced only %d non-zero projections: ECR-only fallback", nonzero)
        return ecr_only_projections(players, league), "ecr_only"
    if source == "none":
        source = "ecr+sleeper" if sleeper_proj else "ecr"
    return proj, source


def _make_researcher(settings: Settings):
    try:
        from .research.claude import ClaudeResearcher

        key = settings.anthropic_api_key if settings.use_claude else None
        r = ClaudeResearcher(api_key=key, model=settings.claude_model)
        if not settings.use_claude:
            r._api_key = None  # explicit --no-claude: never enabled even with env key
        return r
    except Exception as e:  # noqa: BLE001
        log.warning("Claude layer unavailable: %s", e)
        return _DisabledResearcher()


class _DisabledResearcher:
    """Stand-in with the researcher interface when the research module is unusable."""

    enabled = False
    last_error: str | None = None

    def load_notes(self) -> dict[str, ResearchNote]:
        return {}

    async def research_players(self, *a: Any, **k: Any) -> dict[str, ResearchNote]:
        return {}

    async def on_the_clock_advice(self, *a: Any, **k: Any) -> str | None:
        return None

    async def ask(self, *a: Any, **k: Any) -> str:
        return ""


# ---------------------------------------------------------------------------
# build_context
# ---------------------------------------------------------------------------


def build_context(settings: Settings, league: LeagueSettings | None = None, draft: DraftSettings | None = None, *,
                  offline: bool = False, sleeper_players: Mapping | None = None, sleeper_proj: Mapping | None = None,
                  refresh: bool = False, use_model: bool = True, quiet: bool = False,
                  progress: Callable[[str], None] | None = None) -> AppContext:
    """Assemble the :class:`AppContext` (see module docstring).

    ``offline`` only documents intent (no network is used here anyway: Sleeper payloads are
    passed in by the caller). ``quiet`` lowers the log level of the timing messages.
    ``progress`` (optional) receives short step descriptions for CLI feedback.
    """
    ensure_dirs()
    tm = _Timer()
    say = progress or (lambda s: None)
    level = logging.DEBUG if quiet else logging.INFO
    league = league or default_league(settings)
    season = league.season or settings.season
    engine = ScoringEngine(league.scoring_settings or None)
    sources: dict[str, Any] = {"offline": offline, "season": season}

    say("building id crosswalk")
    cw = _build_crosswalk(season)
    tm.lap("crosswalk")

    say("building player universe")
    players, sources["players"] = _build_players(cw, season, sleeper_players)
    tm.lap("players")

    say("loading ECR and bye weeks")
    ecr, pos_ecr = _load_ecr(league.is_superflex)
    byes = _load_byes(season)
    from .data.universe import enrich_players

    enrich_players(players, cw, ecr, byes, pos_ecr)
    sources["ecr"] = ecr is not None
    sources["byes"] = len(byes)
    sources["adp"] = _assign_adp(players, league, draft, sleeper_proj)
    tm.lap("enrich")

    researcher = _make_researcher(settings)
    notes: dict[str, ResearchNote] = {}
    try:
        notes = researcher.load_notes()
    except Exception as e:  # noqa: BLE001
        log.warning("research notes unreadable: %s", e)
    sources["notes"] = len(notes)

    relevant = _relevant_subset(players)
    cache_path = projections_cache_path(league.league_id)
    fingerprint = projection_fingerprint(league, relevant, sleeper_proj=sleeper_proj,
                                         sleeper_players=sources["players"] == "sleeper", use_model=use_model, notes=notes)
    # the cache is only served when it was built from exactly these inputs (PROJ-3): the
    # live-draft path (Sleeper payloads supplied) never reuses an offline / stale build
    projections = None if refresh else load_cached_projections(cache_path, fingerprint=fingerprint)
    if projections is not None:
        projections = {pid: p for pid, p in projections.items() if pid in players}
        sources["projections"] = "cache"
    else:
        say("building projections")
        projections, sources["projections"] = _build_and_cache(
            relevant, engine, league, settings, cw, cache_path, fingerprint, sleeper_proj=sleeper_proj,
            notes=notes, byes=byes, use_model=use_model)
    tm.lap("projections")

    say("preparing advisor")
    from .strategy.recommend import Advisor

    advisor = Advisor(league, players, projections, settings)
    tm.lap("advisor")
    sources["timings"] = tm.steps
    sources["total_ms"] = round(tm.total_ms, 1)
    log.log(level, "context ready: %d players, %d projections (%s), %d notes, %.0f ms total",
            len(players), len(projections), sources["projections"], len(notes), tm.total_ms)
    return AppContext(settings=settings, engine=engine, league=league, draft=draft, players=players,
                      projections=projections, advisor=advisor, researcher=researcher, notes=notes, byes=byes,
                      crosswalk=cw, sources=sources, sleeper_proj=sleeper_proj, use_model=use_model)


def _build_and_cache(relevant: Mapping[str, Player], engine: ScoringEngine, league: LeagueSettings, settings: Settings,
                     cw, cache_path: Path, fingerprint: str, *, sleeper_proj: Mapping | None,
                     notes: Mapping[str, ResearchNote] | None, byes: Mapping[str, int],
                     use_model: bool) -> tuple[dict[str, Projection], str]:
    projections, source = build_projections(relevant, engine, league, settings, cw, sleeper_proj=sleeper_proj,
                                            notes=notes, byes=byes, use_model=use_model)
    try:
        save_projections(cache_path, projections, {"league_id": league.league_id, "source": source,
                                                   "season": league.season, "fingerprint": fingerprint,
                                                   "notes": len(notes or {})})
    except Exception as e:  # noqa: BLE001
        log.warning("could not cache projections: %s", e)
    return projections, source


def rebuild_projections(ctx: AppContext, notes: Mapping[str, ResearchNote] | None = None) -> AppContext:
    """Re-blend the projections of ``ctx`` with ``notes`` (default: ``ctx.notes``) and swap
    ``ctx.projections`` / ``ctx.advisor`` in place; the cache file is rewritten too.

    Used after a research run so ``injury_risk`` / ``role_certainty`` reach points, VORP,
    tiers and availability instead of only the Claude prompt (R1). The new dicts are
    assigned (not mutated) so a concurrent reader keeps a consistent snapshot.
    """
    from .strategy.recommend import Advisor

    if notes is not None and notes is not ctx.notes:
        ctx.notes.update(notes)
    relevant = _relevant_subset(ctx.players)
    fingerprint = projection_fingerprint(ctx.league, relevant, sleeper_proj=ctx.sleeper_proj,
                                         sleeper_players=ctx.sources.get("players") == "sleeper",
                                         use_model=ctx.use_model, notes=ctx.notes)
    projections, source = _build_and_cache(
        relevant, ctx.engine, ctx.league, ctx.settings, ctx.crosswalk, projections_cache_path(ctx.league.league_id),
        fingerprint, sleeper_proj=ctx.sleeper_proj, notes=ctx.notes, byes=ctx.byes, use_model=ctx.use_model)
    advisor = Advisor(ctx.league, ctx.players, projections, ctx.settings)
    ctx.projections = projections
    ctx.advisor = advisor
    ctx.sources["projections"] = source
    ctx.sources["notes"] = len(ctx.notes)
    log.info("projections re-blended with %d research notes (%s)", len(ctx.notes), source)
    return ctx


# ---------------------------------------------------------------------------
# League lookup helpers (Sleeper / snapshot)
# ---------------------------------------------------------------------------


def load_snapshot_league(settings: Settings) -> tuple[LeagueSettings | None, DraftSettings | None]:
    """League/draft from a saved capture snapshot (``draftadvisor capture``), if any."""
    try:
        from .capture import LeagueSnapshot
    except Exception:  # noqa: BLE001
        return None, None
    for key in (settings.league_id, settings.draft_id):
        if not key:
            continue
        snap = LeagueSnapshot.load(str(key))
        if snap is not None:
            log.info("using league snapshot %s", key)
            return snap.league, snap.draft
    return None, None


async def fetch_league(client, league_id: str | None, draft_id: str | None = None) -> tuple[LeagueSettings | None, DraftSettings | None]:
    """League and draft settings from Sleeper (``client`` is a :class:`SleeperClient`)."""
    from .sleeper.parsing import parse_draft, parse_league

    league = draft = None
    if draft_id:
        draft_raw = await client.get_draft(draft_id)
        traded = []
        try:
            traded = await client.get_traded_picks(draft_id)
        except Exception as e:  # noqa: BLE001
            log.info("traded picks unavailable: %s", e)
        draft = parse_draft(draft_raw, traded)
        league_id = league_id or draft.league_id
    if league_id:
        league = parse_league(await client.get_league(league_id))
        if draft is None and league.draft_id:
            try:
                draft = parse_draft(await client.get_draft(league.draft_id), [])
            except Exception as e:  # noqa: BLE001
                log.info("draft %s unavailable: %s", league.draft_id, e)
    return league, draft


# ---------------------------------------------------------------------------
# prep
# ---------------------------------------------------------------------------


async def _fetch_sleeper_inputs(settings: Settings) -> tuple:
    """(league, draft, players payload, projections payload, snapshot) from Sleeper; every
    piece is ``None`` when unreachable (network errors are logged, never raised)."""
    league = draft = None
    sleeper_players = sleeper_proj = None
    snapshot = None
    try:
        from .sleeper.client import SleeperClient

        async with SleeperClient() as client:
            league, draft = await fetch_league(client, settings.league_id, settings.draft_id)
            try:
                sleeper_players = await client.get_players()
            except Exception as e:  # noqa: BLE001
                log.warning("Sleeper players unavailable: %s", e)
            try:
                sleeper_proj = await client.get_season_projections(settings.season)
            except Exception as e:  # noqa: BLE001
                log.warning("Sleeper projections unavailable: %s", e)
            if settings.league_id or settings.draft_id:
                try:
                    from .capture import capture_league

                    snapshot = await capture_league(client, settings.league_id, settings.draft_id,
                                                    username=settings.username, user_id=settings.user_id,
                                                    slot=settings.slot, season=settings.season)
                except Exception as e:  # noqa: BLE001
                    log.warning("league capture failed: %s", e)
    except Exception as e:  # noqa: BLE001
        log.warning("Sleeper unreachable (%s); continuing offline", e)
    return league, draft, sleeper_players, sleeper_proj, snapshot


async def prep(settings: Settings, research: bool = False, refresh: bool = False, train: bool = True,
               top: int = 200, progress: Callable[[str], None] | None = None, offline: bool = False) -> AppContext:
    """Download data, train the model, fetch Sleeper payloads if reachable, build + cache the context.

    Network errors are swallowed (logged) so ``prep`` works offline with ``data/raw``; with
    ``offline=True`` Sleeper is not contacted at all (no retries / timeouts to sit through).
    When ``research`` runs, the projections are re-blended with the new notes (and re-cached)
    so the draft command picks them up. ``sources`` of the returned context carries ``metrics``
    (backtest), ``snapshot`` (capture) and ``research`` (number of notes written).
    """
    ensure_dirs()
    say = progress or (lambda s: None)
    metrics: dict = {}
    say("loading historical data")
    try:
        from .data.nflverse import load_canonical

        canonical = load_canonical(TRAIN_SEASONS, refresh=refresh)
        log.info("canonical data: %d rows", len(canonical))
    except Exception as e:  # noqa: BLE001
        log.warning("historical data unavailable: %s", e)
    if train:
        say("training projection model (this takes a minute or two)")
        try:
            from .projections.model import train_and_save

            _, metrics = train_and_save(refresh=refresh)
        except NotImplementedError as e:
            log.warning("training not implemented: %s", e)
        except Exception as e:  # noqa: BLE001
            log.exception("training failed: %s", e)

    league = draft = None
    sleeper_players = sleeper_proj = None
    snapshot = None
    if offline:
        say("Sleeper skipped (--offline)")
    else:
        say("contacting Sleeper")
        league, draft, sleeper_players, sleeper_proj, snapshot = await _fetch_sleeper_inputs(settings)
    if league is None:
        league, draft = load_snapshot_league(settings)

    ctx = build_context(settings, league, draft, offline=sleeper_players is None, sleeper_players=sleeper_players,
                        sleeper_proj=sleeper_proj, refresh=True, use_model=True, progress=progress)
    ctx.sources["metrics"] = metrics
    ctx.sources["snapshot"] = snapshot
    ctx.sources["research"] = 0
    if research:
        if not ctx.researcher.enabled:
            log.warning("research requested but Claude is disabled (no ANTHROPIC_API_KEY)")
        else:
            say(f"running Claude research on the top {top} players")
            targets = top_players_by_adp(ctx.players, top)
            new_notes: dict[str, ResearchNote] = {}
            try:
                new_notes = await ctx.researcher.research_players(
                    targets, ctx.projections, progress=lambda d, t, n: say(f"research {d}/{t}: {n}"))
            except Exception as e:  # noqa: BLE001
                log.warning("research failed: %s", e)
            err = getattr(ctx.researcher, "last_error", None)
            if err:
                log.warning("research problem: %s", err)
            ctx.sources["research"] = len(new_notes)
            if new_notes:
                # the notes must reach the cached projections the draft command loads (R1)
                say("re-blending projections with the research notes")
                try:
                    rebuild_projections(ctx, new_notes)
                except Exception as e:  # noqa: BLE001
                    log.warning("could not re-blend projections with research notes: %s", e)
    return ctx

