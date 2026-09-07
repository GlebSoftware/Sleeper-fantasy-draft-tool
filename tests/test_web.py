"""Web server API tests against a real uvicorn process (mock mode, offline)."""
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
HAS_DATA = (ROOT / "data" / "raw" / "stats_player_week_2025.csv").exists()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server():
    if not HAS_DATA:
        pytest.skip("needs data/raw for the offline universe")
    port = _free_port()
    env = dict(os.environ, DRAFTADVISOR_HOME=str(ROOT / "data"))
    env.pop("ANTHROPIC_API_KEY", None)
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "draftadvisor.web.server:app", "--host", "127.0.0.1",
                             "--port", str(port), "--log-level", "warning"], cwd=str(ROOT), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            try:
                httpx.get(base + "/api/status", timeout=1)
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.2)
        else:
            raise RuntimeError("server did not start")
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def _wait_my_turn(base: str, timeout: float = 120) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = httpx.get(base + "/api/state", timeout=5).json()
        if st["mode"] == "mock" and st.get("draft") and (st["draft"]["is_my_turn"] or st["draft"]["is_complete"]):
            return st
        time.sleep(0.3)
    raise AssertionError("mock draft never reached my turn")


def test_status_and_index(server):
    r = httpx.get(server + "/api/status", timeout=5)
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "idle" and set(body["ready"]) == {"data", "model", "claude", "sleeper"}
    assert body["ready"]["claude"] is False
    idx = httpx.get(server + "/", timeout=5)
    assert idx.status_code in (200, 404)          # 404 only while the frontend file is absent


def test_mock_draft_flow(server):
    r = httpx.post(server + "/api/mock/start", json={"teams": 8, "rounds": 6, "slot": 3, "scoring": "ppr", "seed": 1,
                                                     "bot_delay": 0.3}, timeout=5)
    assert r.status_code == 200
    st = _wait_my_turn(server)
    d = st["draft"]
    assert d["teams"] == 8 and d["rounds"] == 6 and d["my_slot"] == 3 and d["is_my_turn"]
    assert st["league"]["scoring_type"] == "ppr" and "PPR" in st["league"]["scoring_description"]
    assert len(st["best"]) >= 5 and st["best"][0]["score"] is not None and st["best"][0]["reasons"]
    assert set(st["by_position"]) == {"QB", "RB", "WR", "TE", "K", "DEF"}
    assert st["by_position"]["K"]["action"] in ("SKIP", "WAIT", "SOON", "TAKE NOW")
    assert len(st["available"]) > 50 and st["me"] is not None and st["me"]["slots"][0]["slot"] == "QB"
    assert st["snapshot"] and st["snapshot"]["diff"]["scoring_type"] == "ppr" and st["snapshot"]["draft_order"][2]["is_me"]
    assert st["claude"]["status"] in ("off", "idle")
    # busy guard
    assert httpx.post(server + "/api/mock/start", json={"teams": 8, "rounds": 6, "slot": 1}, timeout=5).status_code == 409
    # invalid pick, then the recommended pick
    assert httpx.post(server + "/api/mock/pick", json={"player_id": "nope"}, timeout=5).status_code == 400
    top = st["best"][0]["player_id"]
    r = httpx.post(server + "/api/mock/pick", json={"player_id": top}, timeout=5)
    assert r.status_code == 200 and r.json()["pick"]["player_id"] == top
    # bots are picking now (0.3 s each): picking again must be refused as "not your turn"
    assert httpx.post(server + "/api/mock/pick", json={"player_id": st["best"][1]["player_id"]}, timeout=5).status_code == 409
    st2 = httpx.get(server + "/api/state", timeout=5).json()
    assert st2["version"] > st["version"]
    mine = [s["player"]["player_id"] for s in st2["me"]["slots"] if s["player"]]
    assert top in mine
    # auto pick at the next turn + player detail + projections
    st3 = _wait_my_turn(server)
    if not st3["draft"]["is_complete"]:
        r = httpx.post(server + "/api/mock/auto", timeout=5)
        assert r.status_code == 200
    pid = top
    detail = httpx.get(server + f"/api/player/{pid}", timeout=5).json()
    assert detail["name"] and "projection" in detail
    rows = httpx.get(server + "/api/projections", params={"position": "RB", "top": 5}, timeout=10).json()
    assert len(rows) == 5 and all(c["position"] == "RB" for c in rows) and rows[0]["points"] >= rows[-1]["points"]
    assert httpx.post(server + "/api/ask", json={"question": "hi"}, timeout=5).status_code == 503
    # autopilot finishes the draft
    httpx.post(server + "/api/mock/autopilot", json={"enabled": True}, timeout=5)
    t0 = time.time()
    while time.time() - t0 < 120:
        stx = httpx.get(server + "/api/state", timeout=5).json()
        if stx["draft"]["is_complete"]:
            break
        time.sleep(0.3)
    assert stx["draft"]["is_complete"]
    assert httpx.post(server + "/api/stop", timeout=10).json()["ok"]
    assert httpx.get(server + "/api/status", timeout=5).json()["mode"] == "idle"


def test_lookup_without_network_reports_502(server):
    r = httpx.post(server + "/api/lookup", json={"username": "someone"}, timeout=30)
    assert r.status_code in (502, 404)
    assert "detail" in r.json()
