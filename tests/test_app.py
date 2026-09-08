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
    assert ctx.sources["projections"] in ("ecr", "ecr_only")   # market rank curves make ECR-only universes real projections
    assert all(p.points > 0 for p in ctx.projections.values())
    # enrichment happened
    josh = ctx.players["4984"]
    assert josh.ecr is not None and josh.bye_week is not None and josh.adp is not None
    assert ctx.engine.rec_points == league.rec_points
    assert not ctx.claude.enabled and ctx.notes == {}
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
    assert ctx.projections and ctx.sources["projections"] in ("ecr", "ecr_only")


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


# ---------------------------------------------------------------------------
# review regressions: PROJ-3 (cache fingerprint), R1 (notes reach projections), F5/R7 (prep)
# ---------------------------------------------------------------------------

import asyncio  # noqa: E402
import copy  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402

from draftadvisor.app import prep, projection_fingerprint  # noqa: E402
from draftadvisor.models import ResearchNote  # noqa: E402


def _live(settings, league, players_json, projections_json, **kw):
    return build_context(settings, league, None, sleeper_players=players_json, sleeper_proj=projections_json,
                         use_model=False, **kw)


def test_projection_cache_is_fingerprinted(patched_loaders, players_json, projections_json, league_json):
    league = parse_league(league_json)
    settings = Settings(league_id=league.league_id)
    ctx = _live(settings, league, players_json, projections_json)
    assert ctx.sources["projections"] != "cache"
    chase = ctx.projections["7564"]
    assert _live(settings, league, players_json, projections_json).sources["projections"] == "cache"
    # the offline path must not be served a live build ...
    off = build_context(settings, league, None, offline=True, sleeper_players=players_json, use_model=False)
    assert off.sources["projections"] != "cache" and "sleeper" not in off.projections["7564"].weights
    # ... and the live path must never be served the offline build
    live2 = _live(settings, league, players_json, projections_json)
    assert live2.sources["projections"] != "cache" and live2.projections["7564"].points == pytest.approx(chase.points)
    # edited scoring settings rebuild
    league2 = parse_league(league_json)
    league2.scoring_settings = {**league2.scoring_settings, "rec": 1.5}
    assert _live(settings, league2, players_json, projections_json).sources["projections"] != "cache"
    # an injury designation that appeared during the day rebuilds (and lowers the games)
    pj = copy.deepcopy(players_json)
    pj["7564"]["injury_status"] = "IR"
    hurt = _live(settings, league, pj, projections_json)
    assert hurt.sources["projections"] != "cache" and hurt.projections["7564"].games < chase.games
    # a changed Sleeper projection line for a relevant player rebuilds too
    pr = copy.deepcopy(projections_json)
    pr["7564"] = {**pr["7564"], "gp": 3, "rec": 5, "rec_yd": 40, "rec_td": 0}
    short = _live(settings, league, players_json, pr)
    assert short.sources["projections"] != "cache" and short.projections["7564"].points < chase.points
    # use_model is part of the key
    assert build_context(settings, league, None, sleeper_players=players_json, sleeper_proj=projections_json,
                         use_model=True).sources["projections"] != "cache"


def test_load_cached_projections_checks_fingerprint(tmp_path, players_json, league_json):
    p = tmp_path / "projections_x.json.gz"
    proj = {"a": Projection("a", "RB", 200.0, 30.0, 12.0, 16.5, 175.0, 225.0)}
    save_projections(p, proj, {"league_id": "x", "fingerprint": "abc"})
    assert load_cached_projections(p, fingerprint="abc") is not None
    assert load_cached_projections(p, fingerprint="xyz") is None
    assert load_cached_projections(p) is not None                       # no fingerprint requested
    save_projections(p, proj, {"league_id": "x"})                       # pre-fingerprint cache file
    assert load_cached_projections(p, fingerprint="abc") is None
    # the fingerprint itself reacts to every input it covers
    league = parse_league(league_json)
    players = players_from_sleeper(players_json)
    base = dict(sleeper_proj=None, sleeper_players=players_json, use_model=True, notes=None)
    fp = projection_fingerprint(league, players, **base)
    assert fp == projection_fingerprint(league, players, **base)
    assert fp != projection_fingerprint(league, players, **{**base, "use_model": False})
    assert fp != projection_fingerprint(league, players, **{**base, "sleeper_proj": {"7564": {"rec": 1}}})
    note = ResearchNote("7564", "x", injury_risk=0.5, role_certainty=0.5, upside="", downside="")
    assert fp != projection_fingerprint(league, players, **{**base, "notes": {"7564": note}})
    players["7564"].injury_status = "Out"
    assert fp != projection_fingerprint(league, players, **base)


class _StubClaude(app._DisabledClaude):
    """Offline chat client: ``load_notes`` returns fixed (legacy) notes; ``ask`` records that it was called."""

    def __init__(self, notes=None, enabled=False):
        self.notes = dict(notes or {})
        self.enabled = enabled
        self.ask_calls = 0

    def load_notes(self):
        return dict(self.notes)

    async def ask(self, *a, **k):
        self.ask_calls += 1
        return ""


def test_legacy_notes_on_disk_reach_projections_and_the_cache_key(patched_loaders, players_json, projections_json,
                                                                   league_json, monkeypatch):
    league = parse_league(league_json)
    settings = Settings(league_id=league.league_id)
    ctx = _live(settings, league, players_json, projections_json)                     # no notes on disk
    before = ctx.projections["7564"]
    note = ResearchNote("7564", "torn ACL", injury_risk=1.0, role_certainty=1.0, upside="", downside="")
    stub = _StubClaude({"7564": note})
    monkeypatch.setattr(app, "_make_claude", lambda s: stub)
    ctx2 = _live(settings, league, players_json, projections_json)
    after = ctx2.projections["7564"]
    assert ctx2.sources["projections"] != "cache" and ctx2.notes["7564"] is note and ctx2.sources["notes"] == 1
    assert after.games == pytest.approx(before.games * 0.75) and after.points < before.points   # the note reached points
    assert ctx.projections["7564"].points == pytest.approx(before.points)                     # the old snapshot is untouched
    assert stub.ask_calls == 0                                                                # building never calls Claude
    # the draft command (refresh=False) finds those numbers in the cache while the note is on disk ...
    ctx3 = _live(settings, league, players_json, projections_json)
    assert ctx3.sources["projections"] == "cache" and ctx3.projections["7564"].points == pytest.approx(after.points)
    # ... and never a stale one when the notes differ
    monkeypatch.setattr(app, "_make_claude", lambda s: _StubClaude())
    ctx4 = _live(settings, league, players_json, projections_json)
    assert ctx4.sources["projections"] != "cache" and ctx4.projections["7564"].points == pytest.approx(before.points)


@pytest.fixture
def prep_env(patched_loaders, fixture_players, monkeypatch):
    """prep() offline: no canonical download, fixture universe, and a recording SleeperClient."""
    import draftadvisor.data.nflverse as nv
    import draftadvisor.sleeper.client as sc

    contacts: list[str] = []

    class _RecordingClient:
        def __init__(self, *a, **k):
            contacts.append("SleeperClient")
            raise RuntimeError("no network in tests")

    monkeypatch.setattr(sc, "SleeperClient", _RecordingClient)
    monkeypatch.setattr(nv, "load_canonical", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(app, "_build_players", lambda cw, season, sp: (dict(fixture_players), "offline"))
    return contacts


def test_prep_offline_never_contacts_sleeper(prep_env):
    msgs: list[str] = []
    ctx = asyncio.run(prep(Settings(), train=False, progress=msgs.append, offline=True))
    assert prep_env == [] and isinstance(ctx, AppContext) and ctx.projections
    assert any("Sleeper skipped" in m for m in msgs) and not any("contacting Sleeper" in m for m in msgs)
    # without offline the client is opened (and its failure is swallowed as before)
    ctx2 = asyncio.run(prep(Settings(), train=False, offline=False))
    assert prep_env == ["SleeperClient"] and ctx2.projections


def test_prep_never_calls_claude_and_has_no_research_option(prep_env, monkeypatch):
    stub = _StubClaude(enabled=True)
    monkeypatch.setattr(app, "_make_claude", lambda s: stub)
    msgs: list[str] = []
    ctx = asyncio.run(prep(Settings(), train=False, progress=msgs.append, offline=True))
    assert ctx.claude is stub and stub.ask_calls == 0
    assert "research" not in ctx.sources and not any("research" in m.lower() for m in msgs)
    assert ctx.sources["notes"] == 0 and ctx.projections
    with gzip.open(projections_cache_path("default"), "rt", encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["meta"]["notes"] == 0
    with pytest.raises(TypeError):                   # the research knobs are gone from the signature
        prep(Settings(), research=True, train=False, offline=True)
    with pytest.raises(TypeError):
        prep(Settings(), top=5, train=False, offline=True)


# ---------------------------------------------------------------------------
# ESPN leagues: ESPN ADP override, projected-stat fallback, placeholder players, snapshots, prep
# ---------------------------------------------------------------------------

import os  # noqa: E402
from pathlib import Path  # noqa: E402

from draftadvisor.app import (  # noqa: E402
    espn_overrides,
    espn_snapshot_from_dict,
    fetch_espn_league,
    load_snapshot,
    load_snapshot_league,
)
from tests.espn_stub import LEAGUE_ID, SEASON, EspnStub  # noqa: E402
from tests.espn_stub import load_fixture as load_espn_fixture  # noqa: E402


@pytest.fixture
def kona():
    return load_espn_fixture("players_kona.json")["players"]


@pytest.fixture
def espn_league():
    from draftadvisor.espn.parsing import parse_espn_draft, parse_espn_league

    lj, dj = load_espn_fixture("league_settings_teams.json"), load_espn_fixture("draft_in_progress.json")
    return parse_espn_league(lj, dj), parse_espn_draft(lj, dj)


def test_espn_overrides_map_the_pool_onto_the_universe(fixture_players, kona):
    placeholders, adp, proj = espn_overrides(fixture_players, kona, SEASON)
    matched = [pid for pid in adp if pid in fixture_players]
    # McCaffrey / Evans / Kelce are in the Sleeper fixture (no espn_id): matched by name + position
    assert {"4034", "2216", "1466"} <= set(matched)
    assert len(adp) == 40 and len(proj) == 40 and len(placeholders) == 40 - len(matched)
    assert not any(p.name in ("Christian McCaffrey", "Mike Evans", "Travis Kelce") for p in placeholders.values())
    gurley = next(p for p in placeholders.values() if p.name == "Todd Gurley II")
    assert gurley.player_id == "espn:2977644" and gurley.position == "RB" and gurley.team == "LAR"
    assert gurley.espn_id == "2977644" and gurley.metadata.get("placeholder") is True
    assert adp[gurley.player_id] == pytest.approx(1.4) and proj[gurley.player_id]["rush_yd"] > 1000
    assert proj["1466"]["rec"] > 0 and proj["1466"]["gp"] > 0
    assert espn_overrides(fixture_players, None, SEASON) == ({}, {}, {})
    assert espn_overrides(fixture_players, ["junk", 3], SEASON) == ({}, {}, {})


def test_build_context_espn_league(patched_loaders, players_json, projections_json, espn_league, kona):
    league, draft = espn_league
    settings = Settings(platform="espn", league_id=league.league_id, season=SEASON)
    espn_adp = {str(e["player"]["id"]): e["player"]["ownership"]["averageDraftPosition"] for e in kona}
    ctx = build_context(settings, league, draft, sleeper_players=players_json, sleeper_proj=projections_json,
                        espn_players=kona, use_model=False)
    assert ctx.platform == "espn" and ctx.sources["platform"] == "espn" and ctx.sources["adp"] == "espn"
    espn = ctx.sources["espn"]
    assert espn["players"] == 40 and espn["adp"] == 40 and espn["projections"] == 40 and espn["placeholders"] == 37
    # ESPN ADP for every player ESPN lists (source "espn"), ECR for the rest
    assert ctx.players["4034"].adp == pytest.approx(espn_adp["3117251"]) and ctx.players["4034"].adp_source == "espn"
    assert ctx.players["4984"].adp is not None and ctx.players["4984"].adp_source == "ecr"
    # ESPN-only players joined the universe as placeholders with ESPN ADP and an ESPN projection
    gurley = ctx.players["espn:2977644"]
    assert gurley.name == "Todd Gurley II" and gurley.adp == pytest.approx(1.4) and gurley.metadata["placeholder"]
    pr = ctx.projections["espn:2977644"]
    assert pr.points > 100 and "espn" in pr.weights and "sleeper" not in pr.weights and "espn_proj" in pr.flags
    assert "espn" in pr.components
    # Sleeper's own line wins where it exists (Kelce is in both payloads)
    kelce = ctx.projections["1466"]
    assert "sleeper" in kelce.weights and "espn" not in kelce.weights and "espn_proj" not in kelce.flags
    assert ctx.sources["proj_fallback"] == 37 and "1466" in ctx.sleeper_proj and "espn:2977644" in ctx.sleeper_proj
    # cached (labels included) for the same inputs; a Sleeper build of the same league id is never served it
    ctx2 = build_context(settings, league, draft, sleeper_players=players_json, sleeper_proj=projections_json,
                         espn_players=kona, use_model=False)
    assert ctx2.sources["projections"] == "cache" and "espn" in ctx2.projections["espn:2977644"].weights
    ctx3 = build_context(Settings(league_id=league.league_id, season=SEASON), league, draft,
                         sleeper_players=players_json, sleeper_proj=projections_json, use_model=False)
    assert ctx3.sources["projections"] != "cache" and ctx3.sources["adp"].startswith("sleeper")
    assert ctx3.platform == "espn" and "espn" not in ctx3.sources     # an ESPN league without a player pool
    # explicit overrides win over the derived ones
    ctx4 = build_context(settings, league, draft, sleeper_players=players_json, sleeper_proj=projections_json,
                         espn_players=kona, adp_override={"espn:2977644": 99.0}, adp_source="custom", use_model=False)
    assert ctx4.players["espn:2977644"].adp == 99.0 and ctx4.players["espn:2977644"].adp_source == "custom"
    assert ctx4.players["4034"].adp == pytest.approx(espn_adp["3117251"]) and ctx4.sources["adp"] == "custom"
    # the advisor recommends over the merged universe
    from draftadvisor.espn.ids import EspnIdMap
    from draftadvisor.espn.parsing import state_from_espn

    state = state_from_espn(load_espn_fixture("league_settings_teams.json"), load_espn_fixture("draft_in_progress.json"),
                            EspnIdMap.from_players(ctx.players), team_id=1)
    rec = ctx.advisor.recommend(state)
    assert state.my_slot == 3 and rec.best_overall and all(v.player_id not in state.drafted_ids for v in rec.best_overall)


def test_build_context_generic_fallback_and_extra_players(patched_loaders, players_json, league_json):
    from draftadvisor.models import Player

    league = parse_league(league_json)
    extra = {"x1": Player("x1", "Extra Guy", "RB", "KC")}
    fallback = {"x1": {"gp": 17, "rush_att": 250, "rush_yd": 1200, "rush_td": 10, "rec": 40, "rec_yd": 300},
                "missing": {"rush_yd": 5}}
    ctx = build_context(Settings(league_id=league.league_id), league, None, sleeper_players=players_json,
                        use_model=False, extra_players=extra, proj_fallback=fallback,
                        adp_override={"x1": 12.0}, adp_source="custom")
    assert ctx.platform == "sleeper" and ctx.players["x1"] is extra["x1"] and ctx.players["x1"].adp == 12.0
    assert ctx.sources["adp"] == "custom" and ctx.sources["proj_fallback"] == 1
    pr = ctx.projections["x1"]
    assert pr.points > 100 and "fallback" in pr.weights and "fallback_proj" in pr.flags


def test_load_snapshot_is_platform_aware(espn_league, kona, league_json, draft_json, users_json, rosters_json, picks_json):
    from draftadvisor.capture import LeagueSnapshot, scoring_diff

    league, draft = espn_league
    lj, dj = load_espn_fixture("league_settings_teams.json"), load_espn_fixture("draft_in_progress.json")
    LeagueSnapshot(captured_at=1.0, season=SEASON, league=league, draft=draft, managers={}, my_user_id="1", my_slot=3,
                   my_roster_id=1, my_picks=[3, 18], diff=scoring_diff(league.scoring_settings), flags=["ESPN league"],
                   raw={"platform": "espn", "league": lj, "draft": dj, "players": kona, "rosters": []}).save()
    back = load_snapshot(LEAGUE_ID, "espn")
    assert back is not None and back.league.name == "FXBG League" and back.my_slot == 3 and back.my_user_id == "1"
    assert back.draft.draft_id == f"espn-{LEAGUE_ID}-{SEASON}" and back.draft.status == "drafting" and back.draft.teams == 10
    assert len(back.managers) == 10 and [p.pick_no for p in back.keepers] == [3, 14] and back.picks_made == 17
    assert back.flags == ["ESPN league"] and back.diff.scoring_type == "ppr" and "FXBG League" in back.to_text()
    assert load_snapshot(f"espn-{LEAGUE_ID}-{SEASON}", "espn") is not None       # saved under the draft id too
    assert load_snapshot(LEAGUE_ID, "sleeper") is None                          # never served across platforms
    assert load_snapshot(LEAGUE_ID) is not None and load_snapshot("nope", "espn") is None
    assert espn_snapshot_from_dict({"raw": {"league": lj, "draft": dj}}).league.name == "FXBG League"
    s = Settings(platform="espn", league_id=LEAGUE_ID, season=SEASON)
    lg, dr = load_snapshot_league(s)
    assert lg is not None and lg.league_id == LEAGUE_ID and dr is not None and dr.rounds == 15
    assert load_snapshot_league(Settings(league_id=LEAGUE_ID)) == (None, None)
    # a Sleeper capture keeps loading through LeagueSnapshot.from_dict and is refused to an ESPN session
    sleeper = LeagueSnapshot.from_dict({"captured_at": 2.0, "season": 2026, "raw": {
        "league": league_json, "draft": draft_json, "users": users_json, "rosters": rosters_json, "picks": picks_json}})
    sleeper.save()
    sid = sleeper.league_id
    assert load_snapshot(sid, "sleeper").league.name == sleeper.league.name and load_snapshot(sid, "espn") is None
    assert load_snapshot_league(Settings(league_id=sid))[0].name == sleeper.league.name
    assert load_snapshot_league(Settings(platform="espn", league_id=sid)) == (None, None)


async def test_fetch_espn_league_and_degradation():
    from draftadvisor.espn.client import EspnClient
    from tests.test_espn_capture_poller import FakeEspnClient

    with EspnStub(draft="in_progress") as s:
        async with EspnClient(base_url=s.base_url, retries=0) as c:
            league, draft, league_json, players = await fetch_espn_league(c, LEAGUE_ID, SEASON)
    assert league.name == "FXBG League" and league.settings["platform"] == "espn" and draft.status == "drafting"
    assert league_json["id"] == 368876 and len(players) == 40 and draft.pick_timer == 90
    fc = FakeEspnClient(n=17)
    fc.fail_players = True
    league, draft, league_json, players = await fetch_espn_league(fc, LEAGUE_ID, SEASON)
    assert players == [] and league.name == "FXBG League" and fc.calls == ["settings", "draft", "players"]


def test_prep_espn_captures_and_uses_the_sleeper_universe(prep_env, monkeypatch):
    import draftadvisor.espn.client as espn_client
    from tests.test_espn_capture_poller import FakeEspnClient

    class _Client(FakeEspnClient):
        def __init__(self, *a, **k):
            super().__init__(n=17)
            prep_env.append("EspnClient")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

    monkeypatch.setattr(espn_client, "EspnClient", _Client)
    msgs: list[str] = []
    settings = Settings(platform="espn", league_id=LEAGUE_ID, season=SEASON, team_id=1)
    ctx = asyncio.run(prep(settings, train=False, progress=msgs.append, offline=False))
    assert prep_env == ["EspnClient", "SleeperClient"]            # ESPN league, then Sleeper for the universe
    assert any("contacting ESPN" in m for m in msgs)
    assert ctx.platform == "espn" and ctx.league.name == "FXBG League" and ctx.sources["adp"] == "espn"
    assert ctx.sources["espn"]["players"] == 40 and ctx.sources["snapshot"].my_slot == 3
    assert "espn:2977644" in ctx.players and ctx.projections
    assert (Path(os.environ["DRAFTADVISOR_HOME"]) / "leagues" / f"{LEAGUE_ID}.json").exists()
    # offline: neither API is contacted; the saved capture supplies the league and its player pool
    prep_env.clear()
    ctx2 = asyncio.run(prep(settings, train=False, offline=True))
    assert prep_env == [] and ctx2.league.name == "FXBG League"
    assert ctx2.sources["adp"] == "espn" and "espn:2977644" in ctx2.players


@pytest.mark.parametrize("knob,exc,needle", [
    ("private", "EspnAccessDenied", "private"),
    ("missing", "EspnNotFound", "not found"),
    ("team_id", "ValueError", "could not find 99"),
])
def test_prep_espn_does_not_swallow_access_denied_not_found_or_unknown_team(prep_env, monkeypatch, knob, exc, needle):
    """A private league, an unknown league id or an unknown --team-id must surface (the CLI maps them to
    exit 2 with the how-to) instead of quietly building a cache for the default league; only transport /
    5xx errors keep prep working offline."""
    import draftadvisor.espn.client as espn_client
    from tests.test_espn_capture_poller import FakeEspnClient

    class _Client(FakeEspnClient):
        def __init__(self, *a, **k):
            super().__init__(n=17, private=knob == "private", missing=knob == "missing")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

    monkeypatch.setattr(espn_client, "EspnClient", _Client)
    settings = Settings(platform="espn", league_id=LEAGUE_ID, season=SEASON, team_id=99 if knob == "team_id" else 1)
    with pytest.raises(getattr(espn_client, exc, ValueError), match=needle):
        asyncio.run(prep(settings, train=False, offline=False))
    assert not (Path(os.environ["DRAFTADVISOR_HOME"]) / "leagues" / f"{LEAGUE_ID}.json").exists()
    # a transport failure on the draft payload still degrades: prep finishes on the default league

    class _Flaky(_Client):
        def __init__(self, *a, **k):
            FakeEspnClient.__init__(self, n=17)
            self.fail_draft_times = 5

    monkeypatch.setattr(espn_client, "EspnClient", _Flaky)
    ctx = asyncio.run(prep(Settings(platform="espn", league_id=LEAGUE_ID, season=SEASON), train=False, offline=False))
    assert ctx.league.name != "FXBG League" and ctx.projections
