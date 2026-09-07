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

ROOT = Path(__file__).resolve().parents[1]
HAS_BUNDLE = (ROOT / "web_bundle" / "players.json").exists()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start(env_extra: dict | None = None):
    port = _free_port()
    env = dict(os.environ, DRAFTADVISOR_HOME=str(ROOT / "data"))
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("DRAFTADVISOR_ACCESS_CODE", None)
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
    assert body["notes"]["store"] in ("local", "memory", "blob")
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


def test_claude_endpoints_without_key(server):
    assert httpx.post(server + "/api/chat", json={"messages": [{"role": "user", "content": "hi"}]}, timeout=30).status_code == 503
    assert httpx.post(server + "/api/research/next", json={"top": 5}, timeout=30).status_code == 503
    assert httpx.post(server + "/api/research/player", json={"player_id": "x"}, timeout=30).status_code == 503
    adv = httpx.get(server + "/api/advice", params={"mode": "mock", "teams": 8, "rounds": 6, "slot": 3, "scoring": "ppr", "seed": 1},
                    timeout=60).json()
    assert adv["status"] == "off" and adv["advice"] is None
    assert httpx.get(server + "/api/notes", timeout=10).status_code == 200


def test_live_requires_ids_and_reports_network(server):
    assert httpx.get(server + "/api/state", timeout=30).status_code == 400
    r = httpx.post(server + "/api/lookup", json={"username": "someone"}, timeout=60)
    assert r.status_code in (502, 404) and "detail" in r.json()
    r = httpx.post(server + "/api/session/start", json={"session": {"mode": "live", "draft_id": "123"}}, timeout=60)
    assert r.status_code in (502, 404)


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
