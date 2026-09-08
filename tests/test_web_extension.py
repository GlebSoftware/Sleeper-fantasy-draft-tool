"""POST /api/extension/advice: the endpoint the ESPN draft-room extension calls (offline, hermetic).

The extension reads the picks inside the owner's own ESPN draft room (ESPN's REST API publishes none
while a draft runs) and posts them here, so this endpoint must answer from the bundle alone: no ESPN,
no Sleeper, nothing metered. The server started here has ``DRAFTADVISOR_ESPN_BASE`` pointed at an
unroutable address, so any outbound ESPN call would hang and blow the latency assertions.
"""
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
UNROUTABLE = "http://198.51.100.7:9/apis/v3/games/ffl"     # TEST-NET-2, discard port: nothing can answer


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start(env_extra: dict | None = None):
    port = _free_port()
    env = dict(os.environ, DRAFTADVISOR_HOME=str(ROOT / "data"))
    for var in RUNTIME_ENV_VARS:
        env.pop(var, None)
    env["DRAFTADVISOR_ESPN_BASE"] = UNROUTABLE
    env["DRAFTADVISOR_ESPN_FAN_BASE"] = UNROUTABLE
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


@pytest.fixture(scope="module")
def coded_server():
    """The same server with an access code set (the preflight must still succeed)."""
    if not HAS_BUNDLE:
        pytest.skip("needs web_bundle/ (run scripts/build_bundle.py)")
    proc, base = _start({"DRAFTADVISOR_ACCESS_CODE": "s3cret"})
    try:
        yield base
    finally:
        _stop(proc)


def advice(base: str, body: dict, code: str | None = None) -> dict:
    headers = {"X-Access-Code": code} if code else {}
    r = httpx.post(base + "/api/extension/advice", json=body, headers=headers, timeout=60)
    assert r.status_code == 200, r.text
    return r.json()


def names(payload: dict) -> set[str]:
    out = {c["name"] for c in payload["overall"]}
    for cards in payload["by_position"].values():
        out |= {c["name"] for c in cards}
    return out


def test_espn_ids_remove_exactly_those_players(server):
    """A handful of ESPN ids (a D/ST id among them) leaves the board with those players gone."""
    empty = advice(server, {"taken": []})
    assert len(empty["overall"]) == 5 and set(empty["by_position"]) == {"QB", "RB", "WR", "TE", "K", "DEF"}
    assert all(len(v) == 5 for v in empty["by_position"].values())
    card = empty["overall"][0]
    assert set(card) == {"player_id", "espn_id", "name", "position", "team", "bye", "points", "vorp", "adp",
                         "tier", "why"}
    assert card["espn_id"] and card["points"] and card["why"]

    top = [c for c in empty["overall"]] + empty["by_position"]["QB"][:1] + empty["by_position"]["DEF"][:1]
    taken = [{"espn_id": int(c["espn_id"]), "name": None} for c in top]
    after = advice(server, {"taken": taken})
    assert after["counts"] == {"taken": len(taken), "resolved": len(taken), "mine": 0}
    assert after["unresolved"] == []
    gone = {c["name"] for c in top}
    assert not (gone & names(after)), gone & names(after)
    # only those players moved: the next man up is now on the board
    assert len(after["overall"]) == 5 and after["suggestion"]["name"] not in gone


def test_names_resolve_and_junk_lands_in_unresolved(server):
    base = advice(server, {"taken": []})
    first = base["overall"][0]["name"]
    body = {"taken": [{"name": first}, {"name": "Zzz Notaplayer McNobody"}, {"espn_id": None, "name": None}]}
    out = advice(server, body)
    assert first not in names(out)
    assert out["counts"]["taken"] == 3 and out["counts"]["resolved"] == 1
    assert "Zzz Notaplayer McNobody" in out["unresolved"] and len(out["unresolved"]) == 2
    # an unknown espn id falls back to the name it came with
    out2 = advice(server, {"taken": [{"espn_id": 99999999, "name": first}]})
    assert out2["counts"]["resolved"] == 1 and first not in names(out2)


def test_mine_drives_needs(server):
    """Two running backs on my roster close the RB starting slots."""
    board = advice(server, {"taken": []})
    assert "RB" in board["needs"] and board["counts"]["mine"] == 0
    rbs = [c for c in board["by_position"]["RB"][:2]]
    mine = [{"espn_id": int(c["espn_id"]), "name": c["name"]} for c in rbs]
    out = advice(server, {"taken": list(mine), "mine": list(mine)})
    assert out["counts"] == {"taken": 2, "resolved": 2, "mine": 2}
    assert "RB" not in out["needs"] and "WR" in out["needs"]
    assert not ({c["name"] for c in rbs} & names(out))


def test_scoring_changes_the_ranking(server):
    """PPR and standard order the running backs differently (pass-catching backs move)."""
    ppr = advice(server, {"taken": [], "scoring": "ppr"})
    std = advice(server, {"taken": [], "scoring": "std"})
    rb_ppr = [c["name"] for c in ppr["by_position"]["RB"]]
    rb_std = [c["name"] for c in std["by_position"]["RB"]]
    wr_ppr = [c["name"] for c in ppr["by_position"]["WR"]]
    wr_std = [c["name"] for c in std["by_position"]["WR"]]
    assert rb_ppr != rb_std or wr_ppr != wr_std
    both = [n for n in rb_ppr if n in rb_std]
    assert any(rb_ppr.index(n) != rb_std.index(n) for n in both)
    # the projections themselves move: receptions are worth a point in PPR and nothing in standard
    pts_ppr = {c["name"]: c["points"] for c in ppr["by_position"]["WR"]}
    pts_std = {c["name"]: c["points"] for c in std["by_position"]["WR"]}
    shared = set(pts_ppr) & set(pts_std)
    assert shared and all(pts_ppr[n] > pts_std[n] for n in shared)


def test_superflex_and_roster_positions(server):
    """Superflex raises the quarterbacks; an explicit roster changes what counts as a need."""
    flat = advice(server, {"taken": []})
    sf = advice(server, {"taken": [], "superflex": True})
    qb = sf["by_position"]["QB"][0]["name"]
    assert qb in {c["name"] for c in sf["overall"]} or [c["name"] for c in flat["overall"]] != [c["name"] for c in sf["overall"]]
    two_qb = advice(server, {"taken": [], "roster_positions": ["QB", "QB", "RB", "WR", "TE", "BN", "BN"]})
    assert "K" not in two_qb["needs"] and "DEF" not in two_qb["needs"] and "QB" in two_qb["needs"]


def test_no_outbound_calls_and_warm_latency(server):
    """ESPN is pointed at an unroutable address: a request that touched it could not answer in 2 s."""
    body = {"taken": [{"name": "Bijan Robinson"}, {"espn_id": -16026}], "mine": [{"name": "Bijan Robinson"}],
            "scoring": "ppr", "teams": 12, "rounds": 16}
    advice(server, body)                                     # warm the context
    t0 = time.perf_counter()
    out = advice(server, body)
    elapsed = (time.perf_counter() - t0) * 1000
    assert elapsed < 2000, elapsed
    assert out["ms"] < 2000 and out["counts"]["resolved"] == 2
    # a live ESPN session against the same server does fail (proving the base URL really is unroutable)
    r = httpx.get(server + "/api/state", params={"platform": "espn", "league_id": "1", "season": 2025}, timeout=60)
    assert r.status_code >= 400


def test_cors_preflight_with_access_code(coded_server):
    """The preflight succeeds even behind the access code; the request itself still needs it."""
    r = httpx.request("OPTIONS", coded_server + "/api/extension/advice", timeout=10,
                      headers={"Origin": "https://fantasy.espn.com", "Access-Control-Request-Method": "POST",
                               "Access-Control-Request-Headers": "content-type,x-access-code"})
    assert r.status_code == 200, r.text
    assert r.headers.get("access-control-allow-origin") in ("*", "https://fantasy.espn.com")
    assert "POST" in r.headers.get("access-control-allow-methods", "")
    assert httpx.post(coded_server + "/api/extension/advice", json={"taken": []}, timeout=60).status_code == 401
    ok = advice(coded_server, {"taken": []}, code="s3cret")
    assert ok["overall"] and ok["suggestion"]["player_id"]
    # a plain POST carries the CORS header too
    r2 = httpx.post(coded_server + "/api/extension/advice", json={"taken": []}, timeout=60,
                    headers={"Origin": "https://fantasy.espn.com", "X-Access-Code": "s3cret"})
    assert r2.status_code == 200 and r2.headers.get("access-control-allow-origin") in ("*", "https://fantasy.espn.com")


def test_projection_cards_carry_espn_id(server):
    """The extension downloads /api/projections as its player index: every card needs its ESPN id."""
    rows = httpx.get(server + "/api/projections", params={"position": "RB", "top": 20}, timeout=60).json()
    assert rows and all("espn_id" in c for c in rows)
    assert sum(1 for c in rows if c["espn_id"]) >= len(rows) - 2
