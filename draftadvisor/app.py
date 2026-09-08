"""Application context and orchestration shared by every CLI command. See DESIGN.md §3.6.

:func:`build_context` turns raw data into an :class:`AppContext`::

    crosswalk -> players (Sleeper payload, else offline roster universe)
              -> [ESPN league: placeholder players for ESPN-only ids]
              -> enrich (FantasyPros ECR, bye weeks) -> ADP (override e.g. ESPN, else Sleeper, else ECR)
              -> projections (cached < 12 h; else ML model -> offline shrinkage -> ECR-only; a projected
                 stat line from another provider fills in for players Sleeper does not project)
              -> Advisor (+ the Claude chat client for ``ask`` and any legacy research notes on disk)

Every step degrades gracefully: a missing data source is logged and skipped so the
tool still runs (with ECR-only projections in the worst case).

ESPN leagues (``Settings.platform == "espn"``) use the same universe (Sleeper players / offline
roster); :func:`espn_overrides` maps ESPN's ``kona_player_info`` list onto it (ESPN ADP as the ADP
source, ESPN projected season stats as the fallback line, placeholders for unknown players) and
:func:`fetch_espn_league` / :func:`load_snapshot` replace the Sleeper lookups.
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
from typing import Any, Callable, Iterable, Mapping

from .config import (
    DEFAULT_PLATFORM,
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
    "prep",
    "default_league",
    "projections_cache_path",
    "projection_fingerprint",
    "load_cached_projections",
    "save_projections",
    "ecr_only_projections",
    "load_snapshot_league",
    "load_saved_snapshot",
    "load_snapshot",
    "espn_snapshot_from_dict",
    "fetch_league",
    "fetch_espn_league",
    "espn_overrides",
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
    claude: Any                                     # research.claude.ClaudeChat (CLI ``ask`` only; never called by the loop)
    notes: dict[str, ResearchNote]
    byes: dict[str, int]
    crosswalk: Any                                  # data.crosswalk.Crosswalk
    sources: dict = field(default_factory=dict)     # what fed the build: {"players": "sleeper", "ml": "model", ...}
    sleeper_proj: Any = None                        # projection lines used (Sleeper payload + any provider fallback)
    use_model: bool = True                          # whether the ML model was allowed for this build
    platform: str = DEFAULT_PLATFORM                # "sleeper" | "espn": the league provider this context serves

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
                           notes: Mapping[str, ResearchNote] | None, adp_source: str | None = None) -> str:
    """Digest of everything that changes the projections for ``players`` (the relevant subset).

    League id + scoring settings + roster positions, whether Sleeper payloads were supplied,
    ``use_model``, the research notes, the ADP source and, per player, the inputs the Projector reads
    (injury status, depth chart, team, ECR, bye) plus his projection line (Sleeper's, or the
    provider fallback). A cache whose fingerprint differs must not be served: on draft day it would
    silently override fresh Sleeper projections, an IR designation or the legacy notes on disk.
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
        "adp_source": adp_source,
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
                sleeper_proj: Mapping | None, adp_override: Mapping[str, float] | None = None,
                adp_source: str | None = None) -> str:
    """Set ``Player.adp``: ``adp_override`` (e.g. ESPN's ADP for an ESPN league) for every player that has
    one and ECR for the rest; otherwise Sleeper's ADP for the league's scoring, else ECR. Returns the label."""
    from .data.universe import assign_adp

    if adp_override:
        label = adp_source or "override"
        n = assign_adp(players, adp_override, label)
        log.info("ADP from %s for %d players", label, n)
        return label if n else "ecr"
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


def _make_claude(settings: Settings):
    """The Claude chat client for the CLI ``ask`` command (nothing else ever calls it)."""
    try:
        from .research.claude import ClaudeChat

        return ClaudeChat(api_key=settings.anthropic_api_key, model=settings.claude_model)
    except Exception as e:  # noqa: BLE001
        log.warning("Claude layer unavailable: %s", e)
        return _DisabledClaude()


class _DisabledClaude:
    """Stand-in with the chat-client interface when the research module is unusable."""

    enabled = False
    last_error: str | None = None
    last_usage: dict | None = None
    last_cost_usd: float | None = None
    model = "off"

    def load_notes(self) -> dict[str, ResearchNote]:
        return {}

    async def ask(self, *a: Any, **k: Any) -> str:
        return ""


# ---------------------------------------------------------------------------
# build_context
# ---------------------------------------------------------------------------


def build_context(settings: Settings, league: LeagueSettings | None = None, draft: DraftSettings | None = None, *,
                  offline: bool = False, sleeper_players: Mapping | None = None, sleeper_proj: Mapping | None = None,
                  refresh: bool = False, use_model: bool = True, quiet: bool = False,
                  progress: Callable[[str], None] | None = None, espn_players: Iterable[Mapping] | None = None,
                  adp_override: Mapping[str, float] | None = None, adp_source: str | None = None,
                  proj_fallback: Mapping[str, Mapping[str, float]] | None = None,
                  extra_players: Mapping[str, Player] | None = None) -> AppContext:
    """Assemble the :class:`AppContext` (see module docstring).

    ``offline`` only documents intent (no network is used here anyway: Sleeper payloads are
    passed in by the caller). ``quiet`` lowers the log level of the timing messages.
    ``progress`` (optional) receives short step descriptions for CLI feedback.

    Provider overrides (all keyed by canonical player id): ``adp_override`` replaces Sleeper's ADP
    (labelled ``adp_source``; players without one fall back to ECR), ``proj_fallback`` supplies a
    Sleeper-key projected stat line for players Sleeper does not project (never overriding Sleeper's
    own line; the blend labels those components ``"espn"`` for an ESPN league) and ``extra_players``
    joins the universe. ``espn_players`` (an ESPN ``kona_player_info`` list) derives all three through
    :func:`espn_overrides`; explicit overrides win over the derived ones.
    """
    ensure_dirs()
    tm = _Timer()
    say = progress or (lambda s: None)
    level = logging.DEBUG if quiet else logging.INFO
    league = league or default_league(settings)
    season = league.season or settings.season
    engine = ScoringEngine(league.scoring_settings or None)
    platform = "espn" if (settings.is_espn or (league.settings or {}).get("platform") == "espn") else DEFAULT_PLATFORM
    sources: dict[str, Any] = {"offline": offline, "season": season, "platform": platform}

    say("building id crosswalk")
    cw = _build_crosswalk(season)
    tm.lap("crosswalk")

    say("building player universe")
    players, sources["players"] = _build_players(cw, season, sleeper_players)
    tm.lap("players")

    espn_entries = [e for e in (espn_players or []) if isinstance(e, Mapping)]
    if espn_entries:
        say("mapping ESPN players onto the universe")
        placeholders, espn_adp_map, espn_proj = espn_overrides(players, espn_entries, season)
        for pid, pl in placeholders.items():
            players.setdefault(pid, pl)
        adp_override = {**espn_adp_map, **dict(adp_override or {})}
        adp_source = adp_source or "espn"
        proj_fallback = {**espn_proj, **dict(proj_fallback or {})}
        sources["espn"] = {"players": len(espn_entries), "placeholders": len(placeholders),
                           "adp": len(espn_adp_map), "projections": len(espn_proj)}
        tm.lap("espn")
    for pid, pl in (extra_players or {}).items():
        players.setdefault(str(pid), pl)

    say("loading ECR and bye weeks")
    ecr, pos_ecr = _load_ecr(league.is_superflex)
    byes = _load_byes(season)
    from .data.universe import enrich_players

    enrich_players(players, cw, ecr, byes, pos_ecr)
    sources["ecr"] = ecr is not None
    sources["byes"] = len(byes)
    sources["adp"] = _assign_adp(players, league, draft, sleeper_proj, adp_override, adp_source)
    fallback_ids: set[str] = set()
    if proj_fallback:
        merged: dict[str, Any] = dict(sleeper_proj or {})
        for pid, line in proj_fallback.items():
            if line and pid in players and not merged.get(pid):
                merged[pid] = dict(line)
                fallback_ids.add(pid)
        sleeper_proj = merged
    sources["proj_fallback"] = len(fallback_ids)
    tm.lap("enrich")

    claude = _make_claude(settings)
    notes: dict[str, ResearchNote] = {}
    try:
        notes = claude.load_notes()          # legacy research notes on disk (read-only)
    except Exception as e:  # noqa: BLE001
        log.warning("research notes unreadable: %s", e)
    sources["notes"] = len(notes)

    relevant = _relevant_subset(players)
    cache_path = projections_cache_path(league.league_id)
    fingerprint = projection_fingerprint(league, relevant, sleeper_proj=sleeper_proj,
                                         sleeper_players=sources["players"] == "sleeper", use_model=use_model, notes=notes,
                                         adp_source=sources["adp"])
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
            notes=notes, byes=byes, use_model=use_model,
            fallback_ids=fallback_ids, fallback_label="espn" if platform == "espn" else "fallback")
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
                      projections=projections, advisor=advisor, claude=claude, notes=notes, byes=byes,
                      crosswalk=cw, sources=sources, sleeper_proj=sleeper_proj, use_model=use_model,
                      platform=platform)


def _relabel_fallback(projections: Mapping[str, Projection], ids: Iterable[str], label: str) -> int:
    """The Projector calls every supplied stat line ``"sleeper"``; rename the component / weight of the
    projections whose line came from another provider to ``label`` and flag them (``<label>_proj``)."""
    n = 0
    for pid in ids:
        pr = projections.get(pid)
        if pr is None or "sleeper" not in (pr.weights or {}):
            continue
        pr.weights[label] = pr.weights.pop("sleeper")
        if "sleeper" in (pr.components or {}):
            pr.components[label] = pr.components.pop("sleeper")
        flag = f"{label}_proj"
        if flag not in pr.flags:
            pr.flags.append(flag)
        n += 1
    return n


def _build_and_cache(relevant: Mapping[str, Player], engine: ScoringEngine, league: LeagueSettings, settings: Settings,
                     cw, cache_path: Path, fingerprint: str, *, sleeper_proj: Mapping | None,
                     notes: Mapping[str, ResearchNote] | None, byes: Mapping[str, int],
                     use_model: bool, fallback_ids: Iterable[str] = (),
                     fallback_label: str = "fallback") -> tuple[dict[str, Projection], str]:
    projections, source = build_projections(relevant, engine, league, settings, cw, sleeper_proj=sleeper_proj,
                                            notes=notes, byes=byes, use_model=use_model)
    if fallback_ids:
        _relabel_fallback(projections, fallback_ids, fallback_label)
    try:
        save_projections(cache_path, projections, {"league_id": league.league_id, "source": source,
                                                   "season": league.season, "fingerprint": fingerprint,
                                                   "notes": len(notes or {})})
    except Exception as e:  # noqa: BLE001
        log.warning("could not cache projections: %s", e)
    return projections, source


# ---------------------------------------------------------------------------
# ESPN inputs (placeholders, ADP override, projected-stat fallback)
# ---------------------------------------------------------------------------


def espn_overrides(players: Mapping[str, Player], espn_players: Iterable[Mapping] | None, season: int,
                   rank_type: str = "PPR") -> tuple[dict[str, Player], dict[str, float], dict[str, dict[str, float]]]:
    """``(placeholders, adp, projections)`` from an ESPN ``kona_player_info`` list, keyed by canonical id.

    ESPN ids resolve through :class:`draftadvisor.espn.ids.EspnIdMap` built from ``players``
    (``espn_id``, team defenses, then name + position). ESPN players missing from the universe become
    placeholder :class:`Player` objects (``espn:<id>``, see :func:`draftadvisor.espn.ids.placeholder_player`)
    so their picks, ADP and projections still carry a name; ``adp`` is ESPN's ``averageDraftPosition``
    (draft-rank fallback) and ``projections`` the ``10<season>`` projected season stats in Sleeper keys.
    """
    from .espn.capture import espn_adp, espn_projections
    from .espn.ids import EspnIdMap, placeholder_player

    entries = [e for e in (espn_players or []) if isinstance(e, Mapping)]
    if not entries:
        return {}, {}, {}
    id_map = EspnIdMap.from_players(players)
    placeholders: dict[str, Player] = {}
    for entry in entries:
        pid = id_map.resolve_player_json(entry)
        if pid not in players and pid not in placeholders:
            placeholders[pid] = placeholder_player(entry, pid)
    adp = espn_adp(entries, id_map, rank_type)
    proj = espn_projections(entries, id_map, int(season))
    log.info("ESPN player pool: %d entries, %d placeholders, %d with ADP, %d projected", len(entries),
             len(placeholders), len(adp), len(proj))
    return placeholders, adp, proj


# ---------------------------------------------------------------------------
# League lookup helpers (Sleeper / ESPN / snapshot)
# ---------------------------------------------------------------------------


def espn_snapshot_from_dict(d: Mapping[str, Any]):
    """Rebuild a saved ESPN capture (``raw["platform"] == "espn"``) as a
    :class:`draftadvisor.capture.LeagueSnapshot` (its own ``from_dict`` only knows Sleeper payloads).
    Pick ids are synthetic (no universe here); the report, teams, order and "my" slot are complete."""
    from .capture import LeagueSnapshot, ScoringDiff, scoring_diff
    from .espn.capture import espn_names
    from .espn.ids import EspnIdMap
    from .espn.parsing import parse_espn_draft, parse_espn_league, parse_espn_managers, parse_espn_picks, roster_names

    raw = dict(d.get("raw") or {})
    league_json, draft_json = dict(raw.get("league") or {}), dict(raw.get("draft") or {})
    league = parse_espn_league(league_json, draft_json)
    draft = parse_espn_draft(league_json, draft_json)
    managers = parse_espn_managers(league_json, draft)
    names: dict[str, Any] = dict(roster_names(league_json))
    names.update(espn_names(raw.get("players")))
    picks = parse_espn_picks(draft_json, EspnIdMap(), draft, names)
    return LeagueSnapshot(
        captured_at=float(d.get("captured_at") or 0.0),
        season=int(d.get("season") or league.season or DEFAULT_SEASON),
        league=league, draft=draft, managers=managers,
        keepers=[p for p in picks if p.is_keeper],
        picks_made=int(d.get("picks_made") or len(picks)),
        my_user_id=d.get("my_user_id"), my_slot=d.get("my_slot"), my_roster_id=d.get("my_roster_id"),
        my_picks=[int(x) for x in (d.get("my_picks") or [])],
        diff=ScoringDiff.from_dict(d["diff"]) if d.get("diff") else scoring_diff(league.scoring_settings),
        flags=list(d.get("flags") or []), nfl_state=dict(d.get("nfl_state") or {}), raw=raw,
    )


def snapshot_platform(snap: Any) -> str:
    """``"espn"`` or ``"sleeper"`` for a :class:`LeagueSnapshot` (from ``raw["platform"]`` / the league settings)."""
    raw = getattr(snap, "raw", None) or {}
    if raw.get("platform") == "espn":
        return "espn"
    league = getattr(snap, "league", None)
    if league is not None and (league.settings or {}).get("platform") == "espn":
        return "espn"
    return DEFAULT_PLATFORM


def load_snapshot(league_or_draft_id: str, platform: str | None = None):
    """A saved capture (Sleeper or ESPN) by league / draft id, else ``None``.

    ``platform`` (when given) must match the file's provider: numeric Sleeper and ESPN league ids can
    collide in the snapshot directory, and a Sleeper capture must never be served to an ESPN session.
    """
    from .capture import LeagueSnapshot, leagues_dir

    path = leagues_dir() / f"{league_or_draft_id}.json"
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        log.warning("could not read snapshot %s: %s", path, e)
        return None
    saved = "espn" if str((d.get("raw") or {}).get("platform") or "") == "espn" else DEFAULT_PLATFORM
    if platform is not None and saved != platform:
        log.info("snapshot %s is a %s capture, not %s; ignoring", path.name, saved, platform)
        return None
    if saved == "espn":
        try:
            return espn_snapshot_from_dict(d)
        except Exception as e:  # noqa: BLE001
            log.warning("could not load ESPN snapshot %s: %s", path, e)
            return None
    return LeagueSnapshot.load(league_or_draft_id)


def load_saved_snapshot(settings: Settings):
    """The saved capture snapshot of ``settings``' league / draft (``draftadvisor capture``), if any: the
    league id, the draft id, then (ESPN) ``espn-<league>-<season>``; same platform only, else ``None``."""
    keys = [settings.league_id, settings.draft_id]
    if settings.is_espn and settings.league_id:
        keys.append(f"espn-{settings.league_id}-{settings.season}")
    for key in keys:
        if not key:
            continue
        try:
            snap = load_snapshot(str(key), settings.platform)
        except Exception as e:  # noqa: BLE001
            log.warning("snapshot %s unavailable: %s", key, e)
            continue
        if snap is not None:
            log.info("using league snapshot %s", key)
            return snap
    return None


def load_snapshot_league(settings: Settings) -> tuple[LeagueSettings | None, DraftSettings | None]:
    """League/draft from a saved capture snapshot (``draftadvisor capture``), if any (same platform only)."""
    snap = load_saved_snapshot(settings)
    return (snap.league, snap.draft) if snap is not None else (None, None)


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


async def fetch_espn_league(client, league_id: str, season: int, *, players_limit: int = 600
                            ) -> tuple[LeagueSettings, DraftSettings, dict, list[dict]]:
    """``(league, draft, raw league payload, ESPN player pool)`` of one ESPN league.

    ``client`` is a :class:`draftadvisor.espn.client.EspnClient`. Two GETs (settings / teams / rosters and
    the draft) plus the ``kona_player_info`` pool, which is optional (``[]`` when ESPN refuses it: ADP and
    projections then come from ECR / Sleeper). :class:`~draftadvisor.espn.client.EspnNotFound` /
    :class:`~draftadvisor.espn.client.EspnAccessDenied` propagate.
    """
    from .espn.client import EspnAPIError
    from .espn.parsing import parse_espn_draft, parse_espn_league

    league_json = await client.get_settings_and_teams(league_id, season)
    draft_json = await client.get_draft_detail(league_id, season)
    players: list[dict] = []
    try:
        players = await client.get_players(league_id, season, limit=players_limit)
    except EspnAPIError as e:
        log.warning("ESPN player pool unavailable (%s); ADP / projections fall back to ECR", e)
    return (parse_espn_league(league_json, draft_json), parse_espn_draft(league_json, draft_json),
            dict(league_json), players)


# ---------------------------------------------------------------------------
# prep
# ---------------------------------------------------------------------------


async def _fetch_sleeper_inputs(settings: Settings, league_lookup: bool = True) -> tuple:
    """(league, draft, players payload, projections payload, snapshot) from Sleeper; every
    piece is ``None`` when unreachable (network errors are logged, never raised). With
    ``league_lookup=False`` only the player universe and the projections are fetched (an ESPN league
    still draws its players from Sleeper's pool)."""
    league = draft = None
    sleeper_players = sleeper_proj = None
    snapshot = None
    try:
        from .sleeper.client import SleeperClient

        async with SleeperClient() as client:
            if league_lookup:
                league, draft = await fetch_league(client, settings.league_id, settings.draft_id)
            try:
                sleeper_players = await client.get_players()
            except Exception as e:  # noqa: BLE001
                log.warning("Sleeper players unavailable: %s", e)
            try:
                sleeper_proj = await client.get_season_projections(settings.season)
            except Exception as e:  # noqa: BLE001
                log.warning("Sleeper projections unavailable: %s", e)
            if league_lookup and (settings.league_id or settings.draft_id):
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


async def _fetch_espn_inputs(settings: Settings) -> tuple:
    """(league, draft, ESPN player pool, snapshot) for ``settings.league_id`` / ``settings.season``; every
    piece is ``None`` / ``[]`` when ESPN is *unreachable* (transport errors, 5xx: logged, never raised, so
    ``prep`` still finishes offline). A private league (:class:`~draftadvisor.espn.client.EspnAccessDenied`),
    an unknown league id (:class:`~draftadvisor.espn.client.EspnNotFound`) and an unknown ``--team-id`` /
    ``--username`` / ``--slot`` (:class:`ValueError` listing the teams) are re-raised: a cache built for
    the default league would hide the real problem. The capture is saved so ``--offline`` commands can
    replay it."""
    if not settings.league_id:
        return None, None, [], None
    from .espn.capture import capture_espn_league
    from .espn.client import EspnAccessDenied, EspnClient, EspnNotFound

    try:
        async with EspnClient(espn_s2=settings.espn_s2, swid=settings.swid) as client:
            snap = await capture_espn_league(client, settings.league_id, settings.season, swid=settings.swid,
                                             team_id=settings.team_id, slot=settings.slot, username=settings.username,
                                             save=True)
            return snap.league, snap.draft, list(snap.raw.get("players") or []), snap
    except (EspnAccessDenied, EspnNotFound, ValueError):
        raise
    except Exception as e:  # noqa: BLE001
        log.warning("ESPN league %s unavailable (%s); continuing without it", settings.league_id, e)
    return None, None, [], None


async def prep(settings: Settings, refresh: bool = False, train: bool = True,
               progress: Callable[[str], None] | None = None, offline: bool = False) -> AppContext:
    """Download data, train the model, fetch Sleeper payloads if reachable, build + cache the context.

    Network errors are swallowed (logged) so ``prep`` works offline with ``data/raw``; with
    ``offline=True`` Sleeper is not contacted at all (no retries / timeouts to sit through).
    Nothing here calls Claude. ``sources`` of the returned context carries ``metrics``
    (backtest) and ``snapshot`` (capture).
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
    espn_players: list[dict] = []
    snapshot = None
    if offline:
        say("Sleeper skipped (--offline)" if not settings.is_espn else "ESPN and Sleeper skipped (--offline)")
    elif settings.is_espn:
        say(f"contacting ESPN (league {settings.league_id}, season {settings.season})")
        league, draft, espn_players, snapshot = await _fetch_espn_inputs(settings)
        say("contacting Sleeper (player universe and projections)")
        _, _, sleeper_players, sleeper_proj, _ = await _fetch_sleeper_inputs(settings, league_lookup=False)
    else:
        say("contacting Sleeper")
        league, draft, sleeper_players, sleeper_proj, snapshot = await _fetch_sleeper_inputs(settings)
    if league is None:
        saved = load_saved_snapshot(settings)
        if saved is not None:
            league, draft = saved.league, saved.draft
            if settings.is_espn and not espn_players:
                espn_players = list(saved.raw.get("players") or [])      # the captured pool: names / ADP / projections

    ctx = build_context(settings, league, draft, offline=sleeper_players is None, sleeper_players=sleeper_players,
                        sleeper_proj=sleeper_proj, espn_players=espn_players, refresh=True, use_model=True,
                        progress=progress)
    ctx.sources["metrics"] = metrics
    ctx.sources["snapshot"] = snapshot
    return ctx

