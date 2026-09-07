"""Tests for draftadvisor.sleeper.client using httpx.MockTransport (no network)."""
from __future__ import annotations

import gzip
import json
import os
import time
from pathlib import Path

import httpx
import pytest

from draftadvisor.config import cache_dir
from draftadvisor.sleeper.client import (
    SleeperAPIError,
    SleeperClient,
    SleeperNotFound,
    _normalise_projections,
    run_sync,
)


_FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str):
    with open(_FIXTURES / name, encoding="utf-8") as fh:
        return json.load(fh)


class Router:
    """Tiny request router for MockTransport; records every request."""

    def __init__(self):
        self.routes: dict[str, list] = {}
        self.requests: list[httpx.Request] = []

    def add(self, path_fragment: str, *responses):
        """Queue responses (httpx.Response or Exception); the last one repeats forever."""
        self.routes[path_fragment] = list(responses)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = str(request.url)
        for frag, queue in self.routes.items():
            if frag in url:
                item = queue.pop(0) if len(queue) > 1 else queue[0]
                if isinstance(item, Exception):
                    if isinstance(item, httpx.HTTPError):
                        item.request = request
                    raise item
                return item
        return httpx.Response(404, json=None)

    def count(self, fragment: str) -> int:
        return sum(1 for r in self.requests if fragment in str(r.url))


def make_client(router: Router, **kw) -> SleeperClient:
    http = httpx.AsyncClient(transport=httpx.MockTransport(router))
    return SleeperClient(http=http, backoff_base=0.0, **kw)


def ok(payload) -> httpx.Response:
    return httpx.Response(200, content=json.dumps(payload).encode(), headers={"content-type": "application/json"})


# ---------------------------------------------------------------------------


async def test_basic_endpoints_and_urls():
    r = Router()
    r.add("/state/nfl", ok(load_fixture("state.json")))
    r.add("/user/gleb/leagues/nfl/2026", ok([{"league_id": "1"}]))
    r.add("/user/gleb/drafts/nfl/2026", ok([{"draft_id": "2"}]))
    r.add("/user/gleb", ok({"user_id": "111", "username": "gleb"}))
    r.add("/league/L1/users", ok(load_fixture("users.json")))
    r.add("/league/L1/rosters", ok(load_fixture("rosters.json")))
    r.add("/league/L1/drafts", ok([load_fixture("draft.json")]))
    r.add("/league/L1", ok(load_fixture("league.json")))
    r.add("/draft/D1/picks", ok(load_fixture("picks.json")))
    r.add("/draft/D1/traded_picks", ok(load_fixture("traded_picks.json")))
    r.add("/draft/D1", ok(load_fixture("draft.json")))
    r.add("/players/nfl/trending/add", ok([{"player_id": "1", "count": 5}]))
    c = make_client(r)
    async with c:
        assert (await c.get_state())["season"] == "2026"
        assert (await c.get_user("gleb"))["user_id"] == "111"
        assert await c.get_user_leagues("gleb", 2026) == [{"league_id": "1"}]
        assert await c.get_user_drafts("gleb", 2026) == [{"draft_id": "2"}]
        assert (await c.get_league("L1"))["name"] == "Fixture League"
        assert len(await c.get_league_users("L1")) == 12
        assert len(await c.get_league_rosters("L1")) == 12
        assert (await c.get_league_drafts("L1"))[0]["draft_id"] == "1180000000000000002"
        assert (await c.get_draft("D1"))["type"] == "snake"
        assert len(await c.get_draft_picks("D1")) == 20
        assert (await c.get_traded_picks("D1"))[0]["round"] == 3
        trending = await c.get_trending("add", hours=48, limit=5)
        assert trending[0]["count"] == 5
    assert str(r.requests[0].url) == "https://api.sleeper.app/v1/state/nfl"
    last = str(r.requests[-1].url)
    assert "lookback_hours=48" in last and "limit=5" in last
    assert r.requests[0].headers["User-Agent"].startswith("draftadvisor")


async def test_404_raises_not_found_without_retry():
    r = Router()
    r.add("/league/nope", httpx.Response(404, json=None))
    c = make_client(r)
    with pytest.raises(SleeperNotFound) as ei:
        await c.get_league("nope")
    assert ei.value.status_code == 404 and "league/nope" in (ei.value.url or "")
    assert r.count("league/nope") == 1


async def test_null_object_is_not_found_and_null_list_is_empty():
    r = Router()
    r.add("/user/ghost", ok(None))
    r.add("/draft/D/picks", ok(None))
    c = make_client(r)
    with pytest.raises(SleeperNotFound):
        await c.get_user("ghost")
    assert await c.get_draft_picks("D") == []


async def test_retries_on_5xx_then_succeeds():
    r = Router()
    r.add("/draft/D1/picks", httpx.Response(503), httpx.Response(500), ok([{"pick_no": 1}]))
    c = make_client(r, retries=3)
    assert await c.get_draft_picks("D1") == [{"pick_no": 1}]
    assert r.count("picks") == 3


async def test_retries_exhausted_raises_api_error():
    r = Router()
    r.add("/draft/D1/picks", httpx.Response(502))
    c = make_client(r, retries=2)
    with pytest.raises(SleeperAPIError) as ei:
        await c.get_draft_picks("D1")
    assert ei.value.status_code == 502
    assert r.count("picks") == 3  # 1 + 2 retries


async def test_retries_on_429_and_timeouts():
    r = Router()
    r.add("/draft/D1", httpx.Response(429, headers={"Retry-After": "0"}), httpx.ReadTimeout("slow"),
          httpx.ConnectError("down"), ok({"draft_id": "D1"}))
    c = make_client(r, retries=3)
    assert (await c.get_draft("D1"))["draft_id"] == "D1"
    assert r.count("draft/D1") == 4


async def test_timeout_exhausted_raises_api_error():
    r = Router()
    r.add("/draft/D1", httpx.ReadTimeout("slow"))
    c = make_client(r, retries=1)
    with pytest.raises(SleeperAPIError) as ei:
        await c.get_draft("D1")
    assert ei.value.status_code is None and "ReadTimeout" in str(ei.value)


async def test_other_4xx_not_retried():
    r = Router()
    r.add("/draft/D1", httpx.Response(400))
    c = make_client(r)
    with pytest.raises(SleeperAPIError) as ei:
        await c.get_draft("D1")
    assert ei.value.status_code == 400 and r.count("draft/D1") == 1
    assert not isinstance(ei.value, SleeperNotFound)


async def test_invalid_json_raises():
    r = Router()
    r.add("/state/nfl", httpx.Response(200, content=b"<html>oops</html>"))
    c = make_client(r)
    with pytest.raises(SleeperAPIError):
        await c.get_state()


# ---------------------------------------------------------------------------
# players cache
# ---------------------------------------------------------------------------


async def test_get_players_caches_for_24h():
    players = load_fixture("players_sample.json")
    r = Router()
    r.add("/players/nfl", ok(players))
    c = make_client(r)
    got = await c.get_players()
    assert got.keys() == players.keys()
    path = cache_dir() / "sleeper_players.json.gz"
    assert path.exists() and str(path).startswith(os.environ["DRAFTADVISOR_HOME"])
    with gzip.open(path, "rt") as fh:
        assert json.load(fh)["4984"]["full_name"] == "Josh Allen"
    # second call served from cache
    again = await c.get_players()
    assert again == got and r.count("players/nfl") == 1
    # force refresh hits the network
    await c.get_players(force_refresh=True)
    assert r.count("players/nfl") == 2
    # expired cache -> refetch
    old = time.time() - 25 * 3600
    os.utime(path, (old, old))
    await c.get_players()
    assert r.count("players/nfl") == 3


async def test_get_players_uses_stale_cache_when_network_fails():
    players = load_fixture("players_sample.json")
    path = cache_dir() / "sleeper_players.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as fh:
        json.dump(players, fh)
    old = time.time() - 48 * 3600
    os.utime(path, (old, old))
    r = Router()
    r.add("/players/nfl", httpx.Response(500))
    c = make_client(r, retries=0)
    assert (await c.get_players()).keys() == players.keys()
    # no cache at all -> error propagates
    path.unlink()
    with pytest.raises(SleeperAPIError):
        await c.get_players()


# ---------------------------------------------------------------------------
# projections / stats
# ---------------------------------------------------------------------------


async def test_season_projections_v1_dict():
    proj = load_fixture("projections_sample.json")
    r = Router()
    r.add("/v1/projections/nfl/regular/2026", ok(proj))
    c = make_client(r)
    got = await c.get_season_projections(2026)
    assert set(got) == set(proj)
    assert got["7564"]["adp_ppr"] == pytest.approx(0.7)
    assert got["7564"]["player_id"] == "7564"
    assert r.count("api.sleeper.com") == 0


async def test_season_projections_falls_back_to_v2_list():
    proj = load_fixture("projections_sample.json")
    v2 = [{"player_id": pid, "stats": stats, "player": {"position": "WR", "team": "CIN"}, "season": "2026"}
          for pid, stats in list(proj.items())[:5]]
    v2.append({"player_id": None, "stats": {}})          # junk entry ignored
    v2.append({"player": {"player_id": "X1"}, "stats": {"pts_ppr": 1.0}})
    r = Router()
    r.add("/v1/projections/nfl/regular/2026", httpx.Response(404))
    r.add("api.sleeper.com/projections/nfl/2026", ok(v2))
    c = make_client(r)
    got = await c.get_season_projections(2026)
    assert len(got) == 6 and "X1" in got
    pid0 = list(proj)[0]
    assert got[pid0]["adp_ppr"] == proj[pid0]["adp_ppr"]
    assert got[pid0]["player_id"] == pid0 and got[pid0]["position"] == "WR"
    url = str(r.requests[-1].url)
    assert url.startswith("https://api.sleeper.com/projections/nfl/2026?")
    assert "season_type=regular" in url and "position%5B%5D=QB" in url and "order_by=adp_ppr" in url


async def test_season_projections_v2_when_v1_empty():
    r = Router()
    r.add("/v1/projections/nfl/regular/2026", ok({}))
    r.add("api.sleeper.com/projections/nfl/2026", ok([{"player_id": "1", "stats": {"pts_ppr": 9.0}}]))
    c = make_client(r)
    assert (await c.get_season_projections(2026))["1"]["pts_ppr"] == 9.0


async def test_week_projections_and_stats_paths():
    r = Router()
    r.add("/v1/projections/nfl/regular/2026/3", ok({"1": {"pts_ppr": 12.0}}))
    r.add("/v1/stats/nfl/regular/2025/2", ok({"2": {"pts_ppr": 5.0}}))
    r.add("/v1/stats/nfl/regular/2025", httpx.Response(500))
    r.add("api.sleeper.com/stats/nfl/2025", ok([{"player_id": "3", "stats": {"pts_ppr": 7.0}}]))
    c = make_client(r, retries=0)
    assert (await c.get_week_projections(2026, 3))["1"]["pts_ppr"] == 12.0
    assert (await c.get_week_stats(2025, 2))["2"]["pts_ppr"] == 5.0
    assert (await c.get_season_stats(2025))["3"]["pts_ppr"] == 7.0


def test_normalise_projections_edge_cases():
    assert _normalise_projections(None) == {}
    assert _normalise_projections({"1": None, "2": {"a": 1}}) == {"2": {"a": 1, "player_id": "2"}}
    assert _normalise_projections([{"player_id": 5, "stats": None}, "junk"]) == {}


def test_run_sync():
    async def coro():
        return 42

    assert run_sync(coro()) == 42


async def test_run_sync_inside_running_loop():
    async def coro():
        return "nested"

    assert run_sync(coro()) == "nested"


async def test_owned_http_client_is_created_and_closed():
    c = SleeperClient(timeout=5.0)
    assert isinstance(c._http, httpx.AsyncClient)
    await c.aclose()
    assert c._http.is_closed
