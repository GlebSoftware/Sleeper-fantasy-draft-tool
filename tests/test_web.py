"""Stateless web API v2 tests against a real uvicorn process (mock mode, offline)."""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from tests.conftest import RUNTIME_ENV_VARS

ROOT = Path(__file__).resolve().parents[1]
HAS_BUNDLE = (ROOT / "web_bundle" / "players.json").exists()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start(env_extra: dict | None = None):
    """A real uvicorn process with a clean runtime environment (module-scoped fixtures start it before the
    per-test env isolation runs, so the server's env is scrubbed here too)."""
    port = _free_port()
    env = dict(os.environ, DRAFTADVISOR_HOME=str(ROOT / "data"))
    for var in RUNTIME_ENV_VARS:
        env.pop(var, None)
    env.update(env_extra or {})
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "draftadvisor.web.server:app", "--host", "127.0.0.1",
                             "--port", str(port), "--log-level", "warning"], cwd=str(ROOT), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    for _ in range(150):
        try:
            httpx.get(base + "/api/status", timeout=1)
            break
        except Exception:  # noqa: BLE001
            time.sleep(0.2)
    else:
        proc.kill()
        raise RuntimeError("server did not start")
    return proc, base


def _stop(proc) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="module")
def server():
    if not HAS_BUNDLE:
        pytest.skip("needs web_bundle/ (run scripts/build_bundle.py)")
    proc, base = _start()
    try:
        yield base
    finally:
        _stop(proc)


MOCK = {"mode": "mock", "use_claude": False,
        "mock": {"teams": 8, "rounds": 6, "slot": 3, "scoring": "ppr", "superflex": False, "seed": 1}, "picks": []}


def test_status_and_index(server):
    body = httpx.get(server + "/api/status", timeout=10).json()
    assert body["ok"] and body["bundle"]["present"] and body["bundle"]["players"] > 500
    assert body["claude"]["server_key"] is False and body["access_code_required"] is False
    assert isinstance(body["claude"]["chat_model"], str) and body["claude"]["chat_model"]
    assert body["claude"]["prices"]["claude-opus-5"] == [5.0, 25.0] and body["claude"]["prices"]["claude-sonnet-5"] == [2.0, 10.0]
    assert "research_model" not in body["claude"]
    assert body["notes"]["store"] in ("local", "memory", "blob")
    # the in-season tables are reported, so a deployment can be checked for them rather than assumed:
    # a bundle built before they existed answers 0 here instead of just looking healthy
    ins = body["bundle"]["inseason"]
    assert ins["schedule_teams"] == 32 and ins["weekly_sigma_players"] > 500
    assert ins["dvp_season"] and ins["dvp_season"] < body["bundle"]["season"]
    assert httpx.get(server + "/", timeout=5).status_code == 200


def test_mock_flow_is_stateless(server):
    r = httpx.post(server + "/api/session/start", json={"session": MOCK}, timeout=60)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["session"]["mode"] == "mock" and body["snapshot"]["diff"]["scoring_type"] == "ppr"
    assert body["snapshot"]["draft_order"][2]["is_me"]
    sess = dict(MOCK)
    # bots advance until my turn (slot 3 -> pick 3)
    st = httpx.post(server + "/api/mock/state", json={"session": sess, "action": "advance"}, timeout=60).json()
    assert st["mode"] == "mock" and st["draft"]["is_my_turn"] and st["draft"]["next_pick_no"] == 3
    assert len(st["picks"]) == 2 and len(st["last_picks"]) == 2 and st["last_picks"][0]["pick_no"] == 1
    assert len(st["best"]) >= 5 and st["best"][0]["reasons"] and set(st["by_position"]) == {"QB", "RB", "WR", "TE", "K", "DEF"}
    assert st["me"]["slots"][0]["slot"] == "QB" and len(st["available"]) > 50
    assert st["draft"]["pick_timer"] in (None, 30) and "seconds_left" in st["draft"]
    # the same request again (browser re-poll) gives the same board: nothing was stored server-side
    st2 = httpx.get(server + "/api/state", params={"mode": "mock", "teams": 8, "rounds": 6, "slot": 3, "scoring": "ppr",
                                                    "seed": 1, "picks": ",".join(st["picks"])}, timeout=60).json()
    assert st2["draft"]["next_pick_no"] == 3 and st2["best"][0]["player_id"] == st["best"][0]["player_id"]
    # my pick by id, then bots advance to my next turn (pick 14)
    sess["picks"] = st["picks"]
    top = st["best"][0]["player_id"]
    assert httpx.post(server + "/api/mock/state", json={"session": sess, "action": "pick", "player_id": "nope"}, timeout=60).status_code == 400
    st3 = httpx.post(server + "/api/mock/state", json={"session": sess, "action": "pick", "player_id": top}, timeout=60).json()
    assert st3["picks"][2] == top and st3["draft"]["next_pick_no"] == 14 and st3["draft"]["is_my_turn"]
    assert any(s["player"] and s["player"]["player_id"] == top for s in st3["me"]["slots"])
    assert len(st3["last_picks"]) == 11 and st3["last_picks"][0]["is_me"]   # my pick #3 + bots #4-#13
    # not my turn -> 409 when the pick list ends mid-round
    sess["picks"] = st3["picks"][:5]
    assert httpx.post(server + "/api/mock/state", json={"session": sess, "action": "pick", "player_id": st3["available"][0]["player_id"]},
                      timeout=60).status_code == 409
    # auto picks until complete
    sess["picks"] = st3["picks"]
    for _ in range(10):
        stx = httpx.post(server + "/api/mock/state", json={"session": sess, "action": "auto"}, timeout=60).json()
        sess["picks"] = stx["picks"]
        if stx["draft"]["is_complete"]:
            break
    assert stx["draft"]["is_complete"] and len(stx["picks"]) == 48
    # player detail + board
    detail = httpx.get(server + f"/api/player/{top}", params={"mode": "mock", "teams": 8, "rounds": 6, "slot": 3, "scoring": "ppr",
                                                              "seed": 1, "picks": ",".join(stx["picks"])}, timeout=60).json()
    assert detail["name"] and detail["projection"] and "explain" in detail
    rows = httpx.get(server + "/api/projections", params={"position": "RB", "top": 5}, timeout=60).json()
    assert len(rows) == 5 and all(c["position"] == "RB" for c in rows) and rows[0]["points"] >= rows[-1]["points"]


def test_chat_is_the_only_claude_endpoint(server):
    """Without a key /api/chat says so (503); the research / advice endpoints no longer exist (404)."""
    assert httpx.post(server + "/api/chat", json={"messages": [{"role": "user", "content": "hi"}]}, timeout=30).status_code == 503
    assert httpx.post(server + "/api/research/next", json={"top": 5}, timeout=30).status_code == 404
    assert httpx.post(server + "/api/research/player", json={"player_id": "x"}, timeout=30).status_code == 404
    assert httpx.get(server + "/api/advice", params={"mode": "mock", "teams": 8, "rounds": 6, "slot": 3, "scoring": "ppr", "seed": 1},
                     timeout=60).status_code == 404
    assert httpx.get(server + "/api/notes", timeout=10).status_code == 200          # legacy notes stay readable
    # a session that still sends use_claude=true is accepted and triggers nothing
    st = httpx.get(server + "/api/state", params={"mode": "mock", "teams": 8, "rounds": 6, "slot": 3, "scoring": "ppr", "seed": 1,
                                                  "use_claude": "true"}, timeout=60).json()
    assert st["mode"] == "mock" and "claude" not in st


def test_live_requires_ids_and_reports_network(server):
    assert httpx.get(server + "/api/state", timeout=30).status_code == 400
    r = httpx.post(server + "/api/lookup", json={"username": "someone"}, timeout=60)
    assert r.status_code in (502, 404) and "detail" in r.json()
    r = httpx.post(server + "/api/session/start", json={"session": {"mode": "live", "draft_id": "123"}}, timeout=60)
    assert r.status_code in (502, 404)


def test_platform_validation_in_process():
    """Platform / session parsing needs no network: bad platforms and missing ESPN ids are rejected up front."""
    from fastapi.testclient import TestClient

    import draftadvisor.web.server as server

    with TestClient(server.app) as client:
        body = client.get("/api/status").json()
        assert body["platforms"] == ["sleeper", "espn"] and body["espn"]["server_cookies"] is False
        assert client.get("/api/state", params={"mode": "live", "platform": "espn"}).status_code == 400
        assert client.get("/api/state", params={"mode": "live", "platform": "yahoo", "league_id": "1"}).status_code == 400
        assert client.get("/api/state", params={"mode": "live", "platform": "espn", "league_id": "1", "season": "abc"}).status_code == 400
        assert client.post("/api/lookup", json={"platform": "espn"}).status_code == 400
        assert client.post("/api/lookup", json={"platform": "yahoo", "username": "x"}).status_code == 400
        assert client.post("/api/session/start", json={"session": {"mode": "live", "platform": "espn"}}).status_code == 400


def test_session_carries_platform_and_keeps_cookies_out_of_dumps():
    from starlette.requests import Request

    from draftadvisor.web.server import Session

    scope = {"type": "http", "method": "GET", "path": "/api/state", "headers": [(b"x-espn-swid", b"{ABC-1}"), (b"x-espn-s2", b"s2")],
             "query_string": b"mode=live&platform=espn&league_id=42&season=2025&team_id=7&slot=3"}
    sess = Session.from_query(Request(scope))
    assert (sess.platform, sess.league_id, sess.season, sess.team_id, sess.slot) == ("espn", "42", 2025, 7, 3)
    assert sess.espn_auth.present and sess.espn_auth.swid == "{ABC-1}"
    assert sess.espn_identity() == {"swid": "{ABC-1}", "team_id": 7, "slot": None, "username": None}
    dumped = sess.model_dump()
    assert dumped["platform"] == "espn" and dumped["team_id"] == 7 and "s2" not in str(dumped.values()) and "ABC" not in str(dumped)
    other = Session.from_query(Request(dict(scope, headers=[])))
    assert other.espn_auth.key != sess.espn_auth.key and other.espn_auth.key == "anon" and not other.espn_auth.present
    assert Session(mode="live", draft_id="1").platform == "sleeper" and Session(user_id="12").espn_team_id == 12


def test_capture_cache_keys_are_identity_free():
    """The capture key ignores who "me" is (the session the page stores after /api/session/start must hit
    the same entry); a Sleeper capture answers to both its draft id and its league id; ESPN keys carry the
    credential hash and never the cookies."""
    from starlette.requests import Request

    from draftadvisor.models import DraftSettings, LeagueSettings
    from draftadvisor.web.server import LeagueBundle, Session, _bundle_keys, _capture_key, _league_key

    started = Session(mode="live", draft_id="D1", username="bob")
    resumed = Session(mode="live", draft_id="D1", league_id="L1", username="bob", user_id="42", slot=3)
    assert _capture_key(started) == _capture_key(resumed) == "league:sleeper:D1::"
    assert _capture_key(Session(mode="live", league_id="L1")) == "league:sleeper::L1:"
    league = LeagueSettings(league_id="L1", name="x", season=2026, total_rosters=12, roster_positions=["QB"], scoring_settings={})
    draft = DraftSettings(draft_id="D1", league_id="L1", type="snake", status="pre_draft", teams=12, rounds=15)
    lb = LeagueBundle(key="league:sleeper::L1:", league=league, draft=draft, snapshot=None, league_raw={"league_id": "L1"},
                      users_raw=[], rosters_raw=[])
    assert _bundle_keys(lb, lb.key) == ["league:sleeper::L1:", "league:sleeper:D1::"]
    # a league-less Sleeper draft (its league is the synthetic default one) is aliased by draft id only
    orphan = LeagueBundle(key="league:sleeper:D1::", league=league, draft=draft, snapshot=None, league_raw=None,
                          users_raw=[], rosters_raw=[])
    assert _bundle_keys(orphan, orphan.key) == ["league:sleeper:D1::"]
    espn = Session(mode="live", platform="espn", league_id="9", season=2025, team_id=1)
    espn_resumed = Session(mode="live", platform="espn", league_id="9", season=2025, team_id=1, user_id="1", slot=3,
                           draft_id="espn-9-2025")
    assert _league_key(espn) == "espn::9:2025" and _capture_key(espn) == _capture_key(espn_resumed) == "league:espn::9:2025:anon"
    scope = {"type": "http", "method": "GET", "path": "/api/state", "query_string": b"",
             "headers": [(b"x-espn-swid", b"{ABC-1}"), (b"x-espn-s2", b"s2-secret")]}
    with_cookies = Session(mode="live", platform="espn", league_id="9", season=2025).with_auth(Request(scope))
    key = _capture_key(with_cookies)
    assert key != _capture_key(espn) and key.startswith("league:espn::9:2025:") and "s2-secret" not in key and "ABC" not in key


def test_from_query_reads_the_session_blob_and_mock_keys():
    """GET endpoints understand what the page sends: the ``session=`` JSON blob and ``mock_<key>`` params;
    bare ``<key>`` params keep working and bad values are 400, never 500."""
    import json
    from urllib.parse import quote

    from fastapi import HTTPException
    from starlette.requests import Request

    from draftadvisor.web.server import Session

    def parse(qs: str) -> Session:
        return Session.from_query(Request({"type": "http", "method": "GET", "path": "/api/state", "headers": [],
                                           "query_string": qs.encode()}))

    page = parse("mode=mock&slot=3&mock_teams=10&mock_rounds=12&mock_scoring=ppr&mock_superflex=true&mock_seed=7&picks=a,b")
    assert (page.mode, page.slot, page.picks) == ("mock", None, ["a", "b"])
    assert (page.mock.teams, page.mock.rounds, page.mock.slot, page.mock.scoring, page.mock.superflex, page.mock.seed) == (10, 12, 3, "ppr", True, 7)
    assert parse("mode=mock&slot=3&mock_slot=4").mock.slot == 4                     # mock_<key> wins over the bare key
    bare = parse("mode=mock&teams=8&rounds=6&slot=3&scoring=ppr&seed=1")
    assert (bare.mock.teams, bare.mock.rounds, bare.mock.slot, bare.mock.seed, bare.mock.superflex) == (8, 6, 3, 1, False)
    blob = {"mode": "mock", "use_claude": True, "picks": ["x"],
            "mock": {"teams": 10, "rounds": 12, "slot": 3, "scoring": "ppr", "superflex": True, "seed": 7}}
    sess = parse("mode=mock&teams=12&session=" + quote(json.dumps(blob)))           # the blob is authoritative
    assert sess.mock.teams == 10 and sess.mock.superflex and sess.picks == ["x"]
    live = parse("session=" + quote(json.dumps({"mode": "live", "platform": "espn", "league_id": "42", "season": 2025, "team_id": 7})))
    assert (live.platform, live.league_id, live.season, live.team_id, live.mock) == ("espn", "42", 2025, 7, None)
    for bad in ("session=not-json", "session=%5B1%5D", "session=" + quote(json.dumps({"mode": "live", "season": "abc"})),
                "mode=mock&mock_teams=ten", "mode=live&slot=x"):
        with pytest.raises(HTTPException) as e:
            parse(bad)
        assert e.value.status_code == 400


def test_mock_get_honours_the_page_query_shape(server):
    """``/api/state`` (and ``/api/player``) with the page's own query shape use the mock's real settings."""
    q = {"mode": "mock", "slot": 3, "mock_teams": 10, "mock_rounds": 12, "mock_slot": 3, "mock_scoring": "ppr",
         "mock_superflex": "true", "mock_seed": 7, "picks": ""}
    st = httpx.get(server + "/api/state", params=q, timeout=60).json()
    assert st["draft"]["teams"] == 10 and st["draft"]["rounds"] == 12 and st["draft"]["my_slot"] == 3
    assert st["league"]["scoring_type"] == "ppr" and "SUPER_FLEX" in st["league"]["roster_positions"]
    top = st["best"][0]["player_id"]
    card = httpx.get(server + f"/api/player/{top}", params=q, timeout=60).json()
    assert card["player_id"] == top and "explain" in card


def test_access_code():
    if not HAS_BUNDLE:
        pytest.skip("needs web_bundle/")
    proc, base = _start({"DRAFTADVISOR_ACCESS_CODE": "s3cret"})
    try:
        assert httpx.get(base + "/api/status", timeout=10).json()["access_code_required"] is True
        assert httpx.get(base + "/api/notes", timeout=10).status_code == 401
        assert httpx.get(base + "/api/notes", headers={"X-Access-Code": "s3cret"}, timeout=10).status_code == 200
    finally:
        _stop(proc)


def test_chat_forwards_usage_model_and_cost_in_process(monkeypatch):
    """The SSE stream of /api/chat ends with the done event carrying usage, model and cost_usd (fake client, no key)."""
    import json

    from fastapi.testclient import TestClient

    import draftadvisor.web.server as server
    from draftadvisor.research.claude import ClaudeChat
    from tests.test_research import FakeClient, text_message

    fake = FakeClient(stream_message=text_message("Take the RB.", model="claude-opus-5"))
    monkeypatch.setattr(server, "claude_for", lambda api_key: ClaudeChat(api_key="k", client=fake))
    events = []
    with TestClient(server.app) as client:
        with client.stream("POST", "/api/chat", json={"messages": [{"role": "user", "content": "Who?"}], "model": "claude-opus-5"}) as r:
            assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
            for line in r.iter_lines():
                if line.startswith("data:"):
                    events.append(json.loads(line[5:].strip()))
    assert "".join(e["delta"] for e in events if "delta" in e) == "Take the RB."
    done = events[-1]
    assert done["done"] is True and done["model"] == "claude-opus-5"
    assert done["usage"]["input_tokens"] == 100 and done["usage"]["output_tokens"] == 50
    assert done["cost_usd"] == round((100 * 5 + 50 * 25) / 1e6, 6)
    assert len(fake.messages.stream_calls) == 1 and fake.messages.stream_calls[0]["messages"][-1]["content"] == "Who?"


def test_chat_refuses_unpriced_models_and_caps_the_transcript(monkeypatch):
    """Only a model in the price table can be billed (400 otherwise, nothing sent); the transcript that goes
    out is bounded (newest turns within CHAT_MAX_TURNS / CHAT_MAX_CHARS; one oversized message is 413)."""
    import json

    from fastapi.testclient import TestClient

    import draftadvisor.web.server as server
    from draftadvisor.research.claude import CHAT_MAX_CHARS, CHAT_MAX_TURNS, CHAT_MODEL, ClaudeChat
    from tests.test_research import FakeClient, text_message

    fake = FakeClient(stream_message=text_message("ok", model="claude-sonnet-5"))
    monkeypatch.setattr(server, "claude_for", lambda api_key: ClaudeChat(api_key="k", client=fake))
    with TestClient(server.app) as client:
        for model in ("not-a-model", "claude-mystery-9", "gpt-x"):
            r = client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}], "model": model})
            assert r.status_code == 400 and "unknown chat model" in r.json()["detail"], model
        # the server default is validated too: an unpriced DRAFTADVISOR_CHAT_MODEL falls back to the default
        monkeypatch.setenv("DRAFTADVISOR_CHAT_MODEL", "claude-mystery-9")
        assert client.get("/api/status").json()["claude"]["chat_model"] == CHAT_MODEL
        monkeypatch.setenv("DRAFTADVISOR_CHAT_MODEL", "claude-sonnet-5")
        assert client.get("/api/status").json()["claude"]["chat_model"] == "claude-sonnet-5"
        r = client.post("/api/chat", json={"messages": [{"role": "user", "content": "x" * (CHAT_MAX_CHARS + 1)}]})
        assert r.status_code == 413 and fake.messages.stream_calls == []
        big = "y" * 400_000
        history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"{i}:{big}"} for i in range(6)]
        history.append({"role": "user", "content": "Who?"})
        with client.stream("POST", "/api/chat", json={"messages": history, "model": "claude-sonnet-5"}) as r:
            assert r.status_code == 200
            events = [json.loads(line[5:]) for line in r.iter_lines() if line.startswith("data:")]
        assert events[-1]["done"] is True and events[-1]["model"] == "claude-sonnet-5"
        sent = fake.messages.stream_calls[-1]["messages"]
        assert sent[-1]["content"] == "Who?" and sent[0]["role"] == "user" and len(sent) <= CHAT_MAX_TURNS
        assert sum(len(m["content"]) for m in sent) <= CHAT_MAX_CHARS
