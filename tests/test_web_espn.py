"""Web API against the ESPN stub (real uvicorn process, ``DRAFTADVISOR_ESPN_BASE`` -> :mod:`tests.espn_stub`).

Sleeper is unreachable here, so the universe is the bundle; the 2018 fixture players that the bundle does not
know become placeholder players carrying ESPN's ADP and projected stat lines.
"""
from __future__ import annotations

import json
import time
from typing import Any

import httpx
import pytest

from tests.espn_stub import _HISTORY_RE, _LEAGUE_RE, LEAGUE_ID, SEASON, SWID_TEAM_1, EspnStub
from tests.test_web import HAS_BUNDLE, _start, _stop

MODERN_ID = "368877"      # the same league with a TE premium (rec override for slot 6), an OP slot and keepers
PRIVATE_ID = "368878"     # 401 without espn_s2 + SWID cookies
FRESH_ID = "368879"       # the classic league again, captured by nothing but the capture-cache test
FORCE_ID = "368880"       # ... and again, so the force-refresh test can drop a capture nobody else uses
COOKIES = {"X-ESPN-S2": "s2-secret", "X-ESPN-SWID": SWID_TEAM_1}
SESSION_KEYS = ("mode", "platform", "draft_id", "league_id", "season", "username", "user_id", "slot", "team_id")


class WebEspnStub(EspnStub):
    """The fixture league four ways: classic at ``LEAGUE_ID`` / ``FRESH_ID``, TE-premium at ``MODERN_ID``,
    private at ``PRIVATE_ID``."""

    def __init__(self, **kw: Any):
        super().__init__(league_ids=(LEAGUE_ID, MODERN_ID, PRIVATE_ID, FRESH_ID, FORCE_ID), **kw)

    def respond(self, path: str, query: dict, headers: dict, cookies: dict) -> tuple[int, Any]:
        m = _LEAGUE_RE.match(path)
        league_id = m.group(2) if m else (_HISTORY_RE.match(path).group(1) if _HISTORY_RE.match(path) else None)
        if league_id == PRIVATE_ID and not (cookies.get("espn_s2") and cookies.get("SWID")):
            with self._lock:
                self.requests.append({"path": path, "query": query, "headers": headers, "cookies": cookies, "at": time.time()})
            return 401, {"messages": ["You are not authorized to view this League."]}
        status, payload = super().respond(path, query, headers, cookies)
        if status != 200 or league_id in (None, LEAGUE_ID):
            return status, payload
        body = payload[0] if isinstance(payload, list) else payload
        views = query.get("view", [])
        if league_id == MODERN_ID and not any(v in views for v in ("kona_player_info", "mDraftDetail")):
            body = self.fixture("league_modern.json")
        body["id"] = int(league_id)                       # ESPN echoes the requested league id
        return status, ([body] if isinstance(payload, list) else body)


@pytest.fixture(scope="module")
def espn():
    """``(server base url, stub)`` with the in-progress draft (17 picks; team 1 / slot 3 is on the clock)."""
    if not HAS_BUNDLE:
        pytest.skip("needs web_bundle/ (run scripts/build_bundle.py)")
    with WebEspnStub(draft="in_progress") as stub:
        proc, base = _start({"DRAFTADVISOR_ESPN_BASE": stub.base_url})
        try:
            yield base, stub
        finally:
            _stop(proc)


def _session(**kw: Any) -> dict:
    s = {"mode": "live", "platform": "espn", "league_id": LEAGUE_ID, "season": SEASON}
    s.update(kw)
    return s


def _draft_gets(stub: EspnStub) -> int:
    return sum(1 for q in stub.requests if "mDraftDetail" in q["query"].get("view", []))


def _query(session: dict) -> dict:
    """What the page's ``sessionQuery()`` flattens out of a stored session (``SESSION_KEYS`` of index.html)."""
    return {k: v for k, v in session.items() if k in SESSION_KEYS and v is not None}


def test_status_lists_platforms(espn):
    base, _ = espn
    body = httpx.get(base + "/api/status", timeout=10).json()
    assert body["platforms"] == ["sleeper", "espn"] and body["espn"] == {"server_cookies": False}


def test_lookup_returns_teams_and_me_by_swid(espn):
    base, stub = espn
    r = httpx.post(base + "/api/lookup", json={"platform": "espn", "league_id": LEAGUE_ID, "season": SEASON},
                   headers={"X-ESPN-SWID": SWID_TEAM_1}, timeout=60)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["platform"] == "espn"
    lg = body["league"]
    assert lg["league_id"] == LEAGUE_ID and lg["name"] == "FXBG League" and lg["season"] == SEASON and lg["teams"] == 10
    assert lg["scoring_type"] == "ppr" and lg["is_public"] is False
    assert lg["draft"] == {"type": "snake", "status": "drafting", "pick_timer": 90, "start_time": 1535198400000,
                           "rounds": 15, "order_known": True}
    assert body["me"] == {"team_id": 1, "slot": 3}
    teams = body["teams"]
    assert len(teams) == 10 and [t["slot"] for t in teams] == list(range(1, 11))
    me = next(t for t in teams if t["team_id"] == 1)
    assert me["is_me"] and me["name"] == "Goin' HAM Newton" and me["abbrev"] == "GHN" and me["owners"] == ["ijgdgvhhj"]
    assert sum(t["is_me"] for t in teams) == 1
    assert {"label": "1pt Safety", "points": 1.0} in body["unmapped_scoring"]
    # the SWID went to ESPN as a cookie (with braces), never as a query parameter
    sent = [q for q in stub.requests if q["cookies"].get("SWID")]
    assert sent and sent[-1]["cookies"]["SWID"] == SWID_TEAM_1 and "swid" not in str(sent[-1]["query"]).lower()
    # without a SWID nobody is "me"; the body still lists the teams
    r = httpx.post(base + "/api/lookup", json={"platform": "espn", "league_id": LEAGUE_ID, "season": SEASON}, timeout=60)
    assert r.status_code == 200 and r.json()["me"] is None and not any(t["is_me"] for t in r.json()["teams"])


def test_lookup_errors(espn):
    base, _ = espn
    r = httpx.post(base + "/api/lookup", json={"platform": "espn", "league_id": "999", "season": SEASON}, timeout=60)
    assert r.status_code == 404 and r.json()["detail"] == f"ESPN has no league 999 for season {SEASON}"
    r = httpx.post(base + "/api/lookup", json={"platform": "espn", "league_id": PRIVATE_ID, "season": SEASON}, timeout=60)
    assert r.status_code == 401 and "Settings" in r.json()["detail"] and "espn_s2" in r.json()["detail"]
    r = httpx.post(base + "/api/lookup", json={"platform": "espn", "league_id": PRIVATE_ID, "season": SEASON},
                   headers=COOKIES, timeout=60)
    assert r.status_code == 200 and r.json()["me"] == {"team_id": 1, "slot": 3}
    assert httpx.post(base + "/api/lookup", json={"platform": "espn"}, timeout=30).status_code == 400
    assert httpx.post(base + "/api/lookup", json={"platform": "yahoo", "username": "x"}, timeout=30).status_code == 400


def test_session_start_resolves_my_slot(espn):
    base, _ = espn
    r = httpx.post(base + "/api/session/start", json={"session": _session(team_id=1)}, timeout=120)
    assert r.status_code == 200, r.text
    body = r.json()
    sess = body["session"]
    assert sess["platform"] == "espn" and sess["league_id"] == LEAGUE_ID and sess["season"] == SEASON
    assert sess["team_id"] == 1 and sess["user_id"] == "1" and sess["slot"] == 3 and sess["draft_id"] == f"espn-{LEAGUE_ID}-{SEASON}"
    assert body["league"] == {"name": "FXBG League", "scoring_type": "ppr", "teams": 10}
    assert body["draft_order_known"] is True and body["message"] is None
    snap = body["snapshot"]
    assert snap["platform"] == "espn" and snap["unmapped_scoring"][0]["label"] == "1pt Safety"
    assert snap["draft_order"][2] == {**snap["draft_order"][2], "slot": 3, "is_me": True, "roster_id": 1, "team_name": "Goin' HAM Newton"}
    assert snap["my_picks"][:2] == [3, 18] and snap["draft"]["pick_timer"] == 90 and snap["draft"]["rounds"] == 15
    assert any(f.startswith("ESPN rule not modelled") for f in snap["flags"])
    # by team / owner name, and a slot only when no team is known
    r = httpx.post(base + "/api/session/start", json={"session": _session(username="Goin' HAM Newton")}, timeout=120)
    assert r.status_code == 200 and r.json()["session"]["team_id"] == 1 and r.json()["session"]["slot"] == 3
    r = httpx.post(base + "/api/session/start", json={"session": _session(slot=4)}, timeout=120)
    assert r.status_code == 200 and r.json()["session"]["team_id"] == 9 and r.json()["session"]["slot"] == 4
    # spectator: no identity at all
    r = httpx.post(base + "/api/session/start", json={"session": _session()}, timeout=120)
    assert r.status_code == 200 and r.json()["session"]["slot"] is None and "spectating" in r.json()["message"]
    # unknown team -> 400 listing the teams; missing league id -> 400
    r = httpx.post(base + "/api/session/start", json={"session": _session(team_id=99)}, timeout=120)
    assert r.status_code == 400 and "Goin' HAM Newton" in r.json()["detail"]
    assert httpx.post(base + "/api/session/start", json={"session": _session(league_id=None)}, timeout=30).status_code == 400


def test_state_renders_the_in_progress_board(espn):
    base, stub = espn
    params = {"mode": "live", "platform": "espn", "league_id": LEAGUE_ID, "season": SEASON, "team_id": 1}
    r = httpx.get(base + "/api/state", params=params, timeout=120)
    assert r.status_code == 200, r.text
    st = r.json()
    d = st["draft"]
    # 17 picks made; pick 18 (round 2, snake) belongs to slot 3 = team 1
    assert d["status"] == "drafting" and d["type"] == "snake" and d["next_pick_no"] == 18 and d["current_round"] == 2
    assert d["is_my_turn"] is True and d["my_slot"] == 3 and d["on_the_clock"]["slot"] == 3 and d["picks_until_my_turn"] == 0
    assert d["teams"] == 10 and d["rounds"] == 15 and d["total_picks"] == 150 and d["is_complete"] is False
    # ESPN gives no pick timestamps: the clock length is reported, no countdown is claimed
    assert d["pick_timer"] == 90 and d["seconds_left"] is None and d["turn_started_at"] is None and d["last_picked"] is None
    assert st["status"]["platform"] == "espn" and st["status"]["latency_ms"] is not None
    assert st["status"]["sources"]["adp"] == "espn" and st["status"]["sources"]["proj_fallback"] > 0
    assert st["league"]["scoring_type"] == "ppr" and st["league"]["teams"] == 10
    assert st["snapshot"]["platform"] == "espn" and st["snapshot"]["unmapped_scoring"]
    assert len(st["best"]) >= 5 and all(b["name"] and b["points"] for b in st["best"])
    assert set(st["by_position"]) == {"QB", "RB", "WR", "TE", "K", "DEF"}
    # every pick renders with a name and its slot; my pick (#3) is flagged
    recent = st["recent"]
    assert len(recent) == 12 and recent[0]["pick_no"] == 17 and all(p["name"] and not p["name"].startswith("espn:") for p in recent)
    assert all(p["label"] and p["slot"] for p in recent)
    assert st["me"] is not None and any(s["player"] for s in st["me"]["slots"])
    assert len(st["opponents"]) == 9
    # a re-poll costs exactly one ESPN GET (draft detail); the capture and the player pool are cached
    before = _draft_gets(stub)
    n_before = len(stub.requests)
    st2 = httpx.get(base + "/api/state", params=params, timeout=120).json()
    assert _draft_gets(stub) == before + 1 and len(stub.requests) == n_before + 1
    assert st2["draft"]["next_pick_no"] == 18 and st2["best"][0]["player_id"] == st["best"][0]["player_id"]
    # a commissioner changing the clock is followed on the next poll
    stub.timer_override = 45
    try:
        assert httpx.get(base + "/api/state", params=params, timeout=120).json()["draft"]["pick_timer"] == 45
    finally:
        stub.timer_override = None


def test_pool_player_unknown_to_the_universe_is_a_placeholder(espn):
    """Todd Gurley (2018, ESPN 2977644) is not in the 2026 bundle: he appears as ``espn:<id>`` with ESPN's ADP
    and a projection scored from ESPN's projected stat line under the league's scoring."""
    base, _ = espn
    params = {"mode": "live", "platform": "espn", "league_id": LEAGUE_ID, "season": SEASON, "team_id": 1}
    r = httpx.get(base + "/api/player/espn:2977644", params=params, timeout=120)
    assert r.status_code == 200, r.text
    card = r.json()
    assert card["name"] == "Todd Gurley II" and card["position"] == "RB" and card["team"] == "LAR" and card["adp"] == 1.4
    assert card["points"] and card["points"] > 200
    assert card["projection"]["components"] == {"espn": pytest.approx(card["points"], abs=0.1)}
    assert "espn_proj" in card["projection"]["flags"] and card["projection"]["stat_line"]["rec"] > 0
    rows = httpx.get(base + "/api/projections", params=dict(params, position="RB", top=400), timeout=120).json()
    row = next(c for c in rows if c["player_id"] == "espn:2977644")
    assert row["drafted_by"] == "drafted" and row["adp"] == 1.4


def test_projections_use_the_league_scoring(espn):
    """TE points: TE-premium league > the same league without it > the default half-PPR board."""
    base, _ = espn

    def board(position: str, **sess: Any) -> dict[str, float]:
        params = dict(sess, position=position, top=400)
        rows = httpx.get(base + "/api/projections", params=params, timeout=120).json()
        return {c["player_id"]: c["points"] for c in rows if c["points"]}

    classic = board("TE", mode="live", platform="espn", league_id=LEAGUE_ID, season=SEASON)
    modern = board("TE", mode="live", platform="espn", league_id=MODERN_ID, season=SEASON)
    default = board("TE")
    common = [pid for pid in sorted(classic, key=lambda p: -classic[p]) if pid in modern and pid in default][:10]
    assert len(common) >= 5
    assert all(modern[pid] > classic[pid] > default[pid] for pid in common), [(pid, modern[pid], classic[pid], default[pid]) for pid in common]
    st = httpx.get(base + "/api/state", params={"mode": "live", "platform": "espn", "league_id": MODERN_ID, "season": SEASON,
                                                "team_id": 1}, timeout=120).json()
    assert "TE +0.5" in st["league"]["scoring_description"] and "SUPER_FLEX" in st["league"]["roster_positions"]
    assert st["snapshot"]["league"]["scoring_settings"]["bonus_rec_te"] == 0.5
    assert {u["label"] for u in st["snapshot"]["unmapped_scoring"]} == {"40+ yard TD pass bonus", "1pt Safety"}


def test_private_league_needs_cookies(espn):
    base, stub = espn
    r = httpx.post(base + "/api/session/start", json={"session": _session(league_id=PRIVATE_ID, team_id=1)}, timeout=120)
    assert r.status_code == 401 and "espn_s2" in r.json()["detail"] and "Settings" in r.json()["detail"]
    r = httpx.post(base + "/api/session/start", json={"session": _session(league_id=PRIVATE_ID, team_id=1)},
                   headers=COOKIES, timeout=120)
    assert r.status_code == 200, r.text
    assert r.json()["session"]["slot"] == 3 and "espn_s2" not in r.text and "s2-secret" not in r.text
    params = {"mode": "live", "platform": "espn", "league_id": PRIVATE_ID, "season": SEASON, "team_id": 1}
    assert httpx.get(base + "/api/state", params=params, timeout=120).status_code == 401
    st = httpx.get(base + "/api/state", params=params, headers=COOKIES, timeout=120)
    assert st.status_code == 200 and st.json()["draft"]["is_my_turn"]
    last = [q for q in stub.requests if q["path"].endswith(PRIVATE_ID)][-1]
    assert last["cookies"]["espn_s2"] == "s2-secret" and last["cookies"]["SWID"] == SWID_TEAM_1


def test_first_poll_after_session_start_reuses_the_capture(espn):
    """The session the browser stores after ``/api/session/start`` (draft id, user id, slot, team filled in)
    must hit the same capture as the start: every first poll costs exactly one ESPN GET (the draft detail),
    never a second capture (3 GETs incl. the 600-player pool). The capture is identity-free, so starting the
    same league as a different manager costs no GET at all; only a new cookie pair captures again."""
    base, stub = espn
    starts = [({"team_id": 1}, 3), ({"username": "Goin' HAM Newton"}, 3), ({"slot": 4}, 4), ({}, None)]
    for i, (start_with, my_slot) in enumerate(starts):
        n0 = len(stub.requests)
        r = httpx.post(base + "/api/session/start", json={"session": _session(league_id=FRESH_ID, **start_with)}, timeout=120)
        assert r.status_code == 200, r.text
        n1 = len(stub.requests)
        assert n1 - n0 == (3 if i == 0 else 0), start_with          # captured once, then shared by every identity
        sess = r.json()["session"]
        assert sess["slot"] == my_slot and sess["draft_id"] == f"espn-{FRESH_ID}-{SEASON}"
        st = httpx.get(base + "/api/state", params=_query(sess), timeout=120)
        assert st.status_code == 200, st.text
        assert len(stub.requests) == n1 + 1 and _draft_gets(stub) >= 1, start_with
        assert st.json()["draft"]["my_slot"] == my_slot and st.json()["draft"]["is_my_turn"] is (my_slot == 3)
    # a SWID identifies the team without any other id; its cookie pair gets its own (still identity-free) capture
    headers = {"X-ESPN-SWID": SWID_TEAM_1}
    n0 = len(stub.requests)
    r = httpx.post(base + "/api/session/start", json={"session": _session(league_id=FRESH_ID)}, headers=headers, timeout=120)
    assert r.status_code == 200 and r.json()["session"]["team_id"] == 1 and r.json()["session"]["slot"] == 3
    assert len(stub.requests) - n0 == 3
    n1 = len(stub.requests)
    assert httpx.get(base + "/api/state", params=_query(r.json()["session"]), headers=headers, timeout=120).json()["draft"]["my_slot"] == 3
    assert len(stub.requests) == n1 + 1
    # an unknown team or an impossible slot is refused from the cached capture without any ESPN request
    n1 = len(stub.requests)
    r = httpx.post(base + "/api/session/start", json={"session": _session(league_id=FRESH_ID, team_id=99)}, timeout=120)
    assert r.status_code == 400 and "Goin' HAM Newton" in r.json()["detail"]
    r = httpx.get(base + "/api/state", params=_query(_session(league_id=FRESH_ID, slot=11)), timeout=120)
    assert r.status_code == 400 and "between 1 and 10" in r.json()["detail"]
    assert len(stub.requests) == n1


def test_espn_client_pool_is_lru_and_never_closes_a_busy_client(monkeypatch):
    """The bounded client pool evicts the least recently used cookie pair; a client that is evicted while one
    of its requests is in flight is closed when that request finishes, never underneath it."""
    import asyncio

    import draftadvisor.web.server as server

    monkeypatch.setattr(server, "ESPN_CLIENT_MAX", 2)
    monkeypatch.setattr(server, "_ESPN_CLIENTS", {})
    a, b, c = (server.EspnAuth(f"s2-{x}", "{" + x + "}") for x in "abc")

    async def run() -> None:
        async with server.espn_client(a) as ca:                 # a is busy for the whole scenario
            async with server.espn_client(b) as cb:
                pass
            async with server.espn_client(a) as again:          # touching a makes b the least recently used
                assert again is ca
            async with server.espn_client(c) as cc:             # full: idle b is evicted and closed, busy a survives
                assert list(server._ESPN_CLIENTS) == [a.key, c.key] and cc is not cb
            assert cb._http.is_closed and not ca._http.is_closed
            async with server.espn_client(b) as cb2:            # full again: a is evicted while in flight ...
                assert list(server._ESPN_CLIENTS) == [c.key, b.key] and cb2 is not cb
                assert not ca._http.is_closed                   # ... and stays open for its running request
        assert ca._http.is_closed                               # closed by that request's own exit
        async with server.espn_client(b) as cb3:
            assert cb3 is cb2 and not cb2._http.is_closed       # a pooled client is reused, not reopened

    asyncio.run(run())


def test_picks_resolve_into_the_universe_the_board_uses(monkeypatch):
    """Sleeper down at capture time, back at the next poll: the board's universe switches to the Sleeper
    payload and the picks must resolve into *it* - a drafted player never stays available under another id."""
    if not HAS_BUNDLE:
        pytest.skip("needs web_bundle/ (run scripts/build_bundle.py)")
    from fastapi.testclient import TestClient

    import draftadvisor.web.server as server
    from tests.conftest import load_fixture

    payload = load_fixture("players_sample.json")
    payload["99999"] = {"player_id": "99999", "first_name": "Tyreek", "last_name": "Hill", "full_name": "Tyreek Hill",
                        "position": "WR", "fantasy_positions": ["WR"], "team": "KC", "status": "Active", "espn_id": 3116406}
    live: dict[str, Any] = {"players": None}

    async def sleeper_players() -> Any:
        return live["players"]

    async def nothing(*args: Any, **kwargs: Any) -> None:
        return None

    async def no_rows() -> list:
        return []

    monkeypatch.setattr(server, "sleeper_players", sleeper_players)
    monkeypatch.setattr(server, "sleeper_projections", nothing)
    monkeypatch.setattr(server, "ecr_rows", no_rows)
    monkeypatch.setattr(server, "CACHE", server._Cache())
    monkeypatch.setattr(server, "_ESPN_CLIENTS", {})
    monkeypatch.setattr(server, "_LOCKS", {})
    params = {"mode": "live", "platform": "espn", "league_id": LEAGUE_ID, "season": SEASON, "team_id": 1}
    with WebEspnStub(draft="in_progress") as stub:
        monkeypatch.setenv("DRAFTADVISOR_ESPN_BASE", stub.base_url)
        with TestClient(server.app) as client:
            r = client.post("/api/session/start", json={"session": _session(team_id=1)})     # bundle universe
            assert r.status_code == 200, r.text
            live["players"] = payload                                                         # Sleeper is back
            st = client.get("/api/state", params=params).json()
            assert st["status"]["sources"]["players"] == "sleeper"
            pick = next(p for p in st["recent"] if p["pick_no"] == 16)                        # Tyreek Hill, ESPN 3116406
            assert pick["player_id"] == "99999" and pick["name"] == "Tyreek Hill"
            ids = {c["player_id"] for c in st["available"]}
            assert "99999" not in ids and "espn:3116406" not in ids
            rows = client.get("/api/projections", params=dict(params, position="WR", top=400)).json()
            mine = [c for c in rows if c["player_id"] == "99999"]
            assert mine and mine[0]["drafted_by"] == "drafted" and not any(c["player_id"] == "espn:3116406" for c in rows)


def test_state_on_a_prepopulated_board_is_not_a_finished_draft():
    """ESPN lists all 150 picks with playerId -1 before the draft starts. Counting those as picks made
    the app report "DRAFT COMPLETE" on an un-started draft and stop following it."""
    if not HAS_BUNDLE:
        pytest.skip("needs web_bundle/ (run scripts/build_bundle.py)")
    # a redraft league before its draft: an all-placeholder board *and* empty rosters (the fixture's
    # rosters are that season's finished ones, which would be evidence the draft has already been held)
    with WebEspnStub(draft="prepopulated", empty_rosters=True) as stub:
        proc, base = _start({"DRAFTADVISOR_ESPN_BASE": stub.base_url})
        try:
            params = {"mode": "live", "platform": "espn", "league_id": LEAGUE_ID, "season": SEASON, "team_id": 1}
            st = httpx.get(base + "/api/state", params=params, timeout=120).json()
            d = st["draft"]
            assert d["is_complete"] is False and d["status"] == "pre_draft"
            assert d["next_pick_no"] == 1 and d["current_round"] == 1
            assert d["my_slot"] == 3 and d["total_picks"] == 150
            assert st["recent"] == []                                  # no picks yet, not 150 phantom ones
            assert not any(s["player"] for s in st["me"]["slots"])      # and my roster is empty
            assert len(st["best"]) >= 5                                 # the board is still fully usable
            assert all(b["name"] and not b["name"].startswith("ESPN player") for b in st["best"])
        finally:
            _stop(proc)


# ---------------------------------------------------------------------------
# A live ESPN draft: the board publishes nothing, the rosters fill
# ---------------------------------------------------------------------------


def test_live_draft_visible_only_through_the_rosters():
    """The reported failure. ESPN's REST board serves one placeholder entry per pick (playerId -1) and
    never updates it while the draft runs; the players that have gone appear on the team rosters. The
    board must therefore report where its picks came from, remove those players from the pool, attribute
    them to their teams - and never invent a pick number ESPN did not publish.

    The league is the reported one: 12 teams, 16 rounds, 192 picks.
    """
    if not HAS_BUNDLE:
        pytest.skip("needs web_bundle/ (run scripts/build_bundle.py)")
    with WebEspnStub(league="twelve", draft="live", roster_picks=5) as stub:
        proc, base = _start({"DRAFTADVISOR_ESPN_BASE": stub.base_url})
        try:
            params = {"mode": "live", "platform": "espn", "league_id": LEAGUE_ID, "season": SEASON, "slot": 3}
            r = httpx.get(base + "/api/state", params=params, timeout=120)
            assert r.status_code == 200, r.text
            assert r.headers["cache-control"] == "no-store, max-age=0" and "X-ESPN-S2" in r.headers["vary"]
            st = r.json()
            d = st["draft"]
            assert d["teams"] == 12 and d["rounds"] == 16 and d["total_picks"] == 192
            assert d["status"] == "drafting" and d["is_complete"] is False
            assert d["board_picks"] == 0 and d["rostered_only"] == 5 and d["picks_source"] == "rosters"
            assert d["next_pick_no"] == 1 and d["current_round"] == 1        # the clock stays board-derived
            assert st["recent"] == []                                        # ESPN published no picks
            rows = st["roster_only"]
            assert len(rows) == 5 and all(row["name"] and row["label"] and row["source"] == "roster" for row in rows)
            assert {row["slot"] for row in rows} == {1, 2, 3, 4, 5} and sum(row["is_me"] for row in rows) == 1
            assert all(row["confidence"] == d["roster_pick_confidence"] for row in rows)
            # gone from the pool and from every recommendation, and counted on their teams
            ids = {row["player_id"] for row in rows}
            assert not (ids & {c["player_id"] for c in st["available"]})
            assert not (ids & {c["player_id"] for c in st["best"]})
            assert sum(sum(o["position_counts"].values()) for o in st["opponents"]) == 4
            assert sum(1 for s_ in st["me"]["slots"] if s_["player"]) == 1    # my own drafted player is mine
            # ... and the page is told exactly what ESPN returned
            espn = st["status"]["espn"]
            assert espn["http_status"] == 200 and espn["board_entries"] == 192 and espn["board_picks"] == 0
            assert espn["roster_drafted"] == 5 and espn["roster_fresh"] == 5 and espn["rostered_total"] == 5
            assert espn["source"] == "rosters" and espn["in_progress"] is True and espn["drafted"] is False
            assert espn["has_draft_detail"] is True and espn["teams_with_rosters"] == 5
            assert espn["views"] == ["mDraftDetail", "mSettings", "mTeam", "mRoster"] and espn["poll_rosters"] is True
            assert espn["league_id"] == LEAGUE_ID and espn["season"] == SEASON and espn["capture_age_s"] >= 0
            assert espn["fetched_at"] > 0 and espn["forced"] is False and espn["bytes"] > 0
            # ... in numbers only: no payload dump, no cookie anywhere
            assert "swid" not in json.dumps(espn).lower() and "espn_s2" not in json.dumps(espn)
        finally:
            _stop(proc)


def test_diagnose_reports_what_espn_returned_and_explains_it():
    """The Diagnostics panel: the same block as ``status.espn`` plus per-team counts, a board sample and
    a sentence the owner can read. Costs one ESPN GET (a poll), never a re-capture."""
    if not HAS_BUNDLE:
        pytest.skip("needs web_bundle/ (run scripts/build_bundle.py)")
    with WebEspnStub(league="twelve", draft="prepopulated", empty_rosters=True) as stub:
        proc, base = _start({"DRAFTADVISOR_ESPN_BASE": stub.base_url})
        try:
            params = {"mode": "live", "platform": "espn", "league_id": LEAGUE_ID, "season": SEASON, "slot": 3}
            httpx.get(base + "/api/state", params=params, timeout=120)      # warm the capture
            before = len(stub.requests)
            r = httpx.get(base + "/api/espn/diagnose", params=params, timeout=120)
            assert r.status_code == 200 and r.headers["cache-control"] == "no-store, max-age=0"
            body = r.json()
            assert len(stub.requests) - before == 1                          # one GET, like a poll
            assert body["platform"] == "espn"
            espn = body["espn"]
            assert espn["board_entries"] == 192 and espn["board_picks"] == 0 and espn["roster_drafted"] == 0
            assert espn["source"] == "none" and espn["league_sub_type"] == "NONE"
            assert len(body["teams"]) == 12 and all(t["roster_entries"] == 0 for t in body["teams"])
            assert sorted(t["slot"] for t in body["teams"]) == list(range(1, 13))
            assert len(body["board_sample"]) == 10 and all(e["playerId"] == -1 for e in body["board_sample"])
            assert body["identity"]["teams"] == 12 and body["identity"]["rounds"] == 16
            assert body["identity"]["my_slot"] == 3 and body["identity"]["pick_order_known"] is True
            assert "0 real picks" in body["explanation"] and "does not update the draft board" in body["explanation"]
            assert "swid" not in r.text.lower() and "espn_s2" not in r.text
            # a Sleeper session gets a report saying ESPN diagnostics do not apply
            sl = httpx.get(base + "/api/espn/diagnose", params={"mode": "live", "platform": "sleeper",
                                                               "draft_id": "1"}, timeout=60).json()
            assert sl["espn"] is None and "do not apply" in sl["explanation"]
        finally:
            _stop(proc)


def test_force_refresh_recaptures_the_league_and_is_rate_limited(espn):
    """The Refresh button: ``force=1`` re-reads settings / teams / rosters / pick order / clock instead of
    using the cached capture, and the server refuses to do that more than once every few seconds."""
    base, stub = espn
    params = {"mode": "live", "platform": "espn", "league_id": FORCE_ID, "season": SEASON, "team_id": 1}
    httpx.get(base + "/api/state", params=params, timeout=120)              # capture + poll
    n0 = len(stub.requests)
    st = httpx.get(base + "/api/state", params=dict(params, force=1), timeout=120).json()
    views = [tuple(q["query"].get("view", [])) for q in stub.requests[n0:]]
    assert ("mSettings", "mTeam", "mRoster") in views and ("kona_player_info",) in views   # the capture again
    assert views[-1] == ("mDraftDetail", "mSettings", "mTeam", "mRoster")                  # ... and a fresh poll
    assert st["status"]["forced"] is True and st["status"]["espn"]["forced"] is True
    assert st["status"]["espn"]["recaptured"] is True and st["status"]["espn"]["capture_age_s"] < 5
    # holding the button: the next forced call within the window polls without re-capturing
    n1 = len(stub.requests)
    st2 = httpx.get(base + "/api/state", params=dict(params, force=1), timeout=120).json()
    assert len(stub.requests) - n1 == 1
    assert st2["status"]["espn"]["forced"] is True and st2["status"]["espn"]["recaptured"] is False
    assert st2["draft"]["next_pick_no"] == st["draft"]["next_pick_no"]
    # an ordinary poll is unchanged: one GET, no re-capture
    n2 = len(stub.requests)
    st3 = httpx.get(base + "/api/state", params=params, timeout=120).json()
    assert len(stub.requests) - n2 == 1 and st3["status"]["forced"] is False
    # this fixture league's capture also carries that season's finished rosters, so the board is not the
    # only source here; what matters is that the 17 board picks are read as board picks
    assert st3["draft"]["board_picks"] == 17 and st3["draft"]["picks_source"].startswith("board")
    assert st3["status"]["espn"]["board_picks"] == 17 and st3["status"]["espn"]["source"].startswith("board")
