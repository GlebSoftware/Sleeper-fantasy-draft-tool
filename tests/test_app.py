"""build_context offline on the fixture players with the data loaders monkeypatched."""
from __future__ import annotations

import time

import pandas as pd
import pytest

import draftadvisor.app as app
from draftadvisor.app import (
    AppContext,
    build_context,
    ecr_only_projections,
    load_cached_projections,
    projections_cache_path,
    save_projections,
)
from draftadvisor.config import Settings
from draftadvisor.data.crosswalk import Crosswalk
from draftadvisor.data.universe import players_from_sleeper
from draftadvisor.models import DraftState, Projection
from draftadvisor.sleeper.parsing import parse_draft, parse_league, parse_picks


@pytest.fixture
def fixture_players(players_json):
    return players_from_sleeper(players_json)


def _ecr_frame(players) -> pd.DataFrame:
    rows = []
    ranked = sorted(players.values(), key=lambda p: (p.search_rank or 9_999_999))
    for i, p in enumerate(ranked, start=1):
        rows.append({"fantasypros_id": str(10_000 + i), "player": p.name, "pos": p.position, "team": p.team,
                     "ecr": float(i), "sd": 2.0 + i / 10, "best": i, "worst": i + 5, "bye": 7, "scrape_date": "2026-09-01"})
    return pd.DataFrame(rows)


@pytest.fixture
def patched_loaders(monkeypatch, fixture_players):
    """Crosswalk / ECR / byes / offline projections replaced with in-memory objects."""
    cw = Crosswalk()
    for p in fixture_players.values():
        cw.add(p.player_id, gsis_id=f"00-{p.player_id}", name=p.name, position=p.position, team=p.team)
    ecr = _ecr_frame(fixture_players)
    byes = {p.team: 5 + (hash(p.team) % 9) for p in fixture_players.values() if p.team}
    calls = {"model": 0, "offline": 0}

    def no_model(*a, **k):
        calls["model"] += 1
        raise FileNotFoundError("no model in tests")

    def no_offline(*a, **k):
        calls["offline"] += 1
        raise NotImplementedError("offline projections unavailable in tests")

    monkeypatch.setattr(app, "_build_crosswalk", lambda season: cw)
    monkeypatch.setattr(app, "_load_ecr", lambda superflex: (ecr, None))
    monkeypatch.setattr(app, "_load_byes", lambda season: byes)
    monkeypatch.setattr(app, "_model_predictions", no_model)
    monkeypatch.setattr(app, "_offline_predictions", no_offline)
    return {"cw": cw, "ecr": ecr, "byes": byes, "calls": calls}


def test_build_context_offline_ecr_only(patched_loaders, players_json, league_json, draft_json, picks_json):
    league = parse_league(league_json)
    draft = parse_draft(draft_json)
    settings = Settings(league_id=league.league_id)
    t0 = time.perf_counter()
    ctx = build_context(settings, league, draft, offline=True, sleeper_players=players_json, use_model=False)
    elapsed = time.perf_counter() - t0
    assert isinstance(ctx, AppContext)
    assert elapsed < 3.0
    assert ctx.sources["players"] == "sleeper" and len(ctx.players) >= 40
    assert ctx.sources["adp"] == "ecr" and ctx.sources["ecr"] is True
    assert patched_loaders["calls"]["model"] == 0            # use_model=False never touches the model
    assert patched_loaders["calls"]["offline"] == 1
    assert ctx.sources["projections"] == "ecr_only"
    assert all(p.points > 0 for p in ctx.projections.values())
    # enrichment happened
    josh = ctx.players["4984"]
    assert josh.ecr is not None and josh.bye_week is not None and josh.adp is not None
    assert ctx.engine.rec_points == league.rec_points
    assert not ctx.researcher.enabled and ctx.notes == {}
    # the advisor works on the fixture draft state
    state = DraftState(draft, parse_picks(picks_json), league, my_user_id="111111111111111111", my_slot=1)
    rec = ctx.advisor.recommend(state)
    assert rec.best_overall and all(v.player_id not in state.drafted_ids for v in rec.best_overall)
    assert set(rec.by_position) == {"QB", "RB", "WR", "TE", "K", "DEF"}
    # cached for next time
    assert projections_cache_path(league.league_id).exists()


def test_build_context_uses_sleeper_projections_and_cache(patched_loaders, players_json, projections_json, league_json):
    league = parse_league(league_json)
    settings = Settings(league_id=league.league_id)
    ctx = build_context(settings, league, None, sleeper_players=players_json, sleeper_proj=projections_json, use_model=False)
    assert ctx.sources["adp"].startswith("sleeper")
    assert ctx.sources["projections"] in ("ecr+sleeper", "ecr")
    chase = ctx.projections["7564"]
    assert chase.points > 150 and "sleeper" in chase.weights
    assert ctx.players["7564"].adp == pytest.approx(0.7)
    # second build hits the cache
    ctx2 = build_context(settings, league, None, sleeper_players=players_json, sleeper_proj=projections_json, use_model=False)
    assert ctx2.sources["projections"] == "cache"
    assert ctx2.projections["7564"].points == pytest.approx(chase.points)
    assert ctx2.projections["7564"].weekly == chase.weekly
    # refresh forces a rebuild
    ctx3 = build_context(settings, league, None, sleeper_players=players_json, sleeper_proj=projections_json,
                         use_model=False, refresh=True)
    assert ctx3.sources["projections"] != "cache"


def test_build_context_model_failure_falls_back(patched_loaders, players_json, league_json):
    league = parse_league(league_json)
    ctx = build_context(Settings(), league, None, sleeper_players=players_json, use_model=True)
    assert patched_loaders["calls"]["model"] == 1 and patched_loaders["calls"]["offline"] == 1
    assert ctx.projections and ctx.sources["projections"] == "ecr_only"


def test_build_context_default_league_and_offline_universe(patched_loaders, fixture_players, monkeypatch):
    monkeypatch.setattr(app, "_build_players", lambda cw, season, sp: (dict(fixture_players), "offline"))
    ctx = build_context(Settings(), None, None, offline=True, use_model=False)
    assert ctx.league.league_id == "default" and ctx.league.scoring_type == "half_ppr"
    assert ctx.sources["players"] == "offline" and len(ctx.projections) > 0
    assert projections_cache_path("default").exists()


def test_projection_cache_roundtrip_and_ttl(tmp_path):
    p = tmp_path / "projections_x.json.gz"
    proj = {"a": Projection("a", "RB", 200.0, 30.0, 12.0, 16.5, 175.0, 225.0, weekly=[12.0] * 17,
                            stat_line={"rush_yd": 1000.0}, components={"ml": 200.0}, weights={"ml": 1.0}, flags=["rookie"])}
    save_projections(p, proj, {"league_id": "x"})
    back = load_cached_projections(p)
    assert back is not None and back["a"] == proj["a"]
    assert load_cached_projections(p, max_age_hours=0) is None
    assert load_cached_projections(tmp_path / "missing.json.gz") is None


def test_ecr_only_projections_monotone(fixture_players):
    for i, p in enumerate(sorted(fixture_players.values(), key=lambda p: p.search_rank or 0), start=1):
        p.ecr = float(i)
    proj = ecr_only_projections(fixture_players)
    assert set(proj) == set(fixture_players)
    for pos in ("QB", "RB", "WR"):
        group = sorted((p for p in fixture_players.values() if p.position == pos), key=lambda p: p.ecr)
        pts = [proj[p.player_id].points for p in group]
        assert pts == sorted(pts, reverse=True)
        assert all(proj[p.player_id].std > 0 and proj[p.player_id].flags == ["ecr_only"] for p in group)
