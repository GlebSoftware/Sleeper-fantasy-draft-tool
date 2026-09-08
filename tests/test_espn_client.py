"""EspnClient: cookies, views, x-fantasy-filter, retries, 401 fallback, 404, base URL (MockTransport + stub)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pytest

from draftadvisor.espn.client import (
    EspnAccessDenied,
    EspnAPIError,
    EspnClient,
    EspnNotFound,
    format_swid,
    parse_fan_leagues,
    player_filter,
)
from draftadvisor.espn.constants import DEFAULT_ESPN_BASE_URL, espn_base_url
from tests.espn_stub import LEAGUE_ID, SEASON, EspnStub, load_fixture

FIXTURES = Path(__file__).parent / "fixtures" / "espn"


class Router:
    """Request router for httpx.MockTransport; records every request."""

    def __init__(self):
        self.routes: list[tuple[str, list]] = []
        self.requests: list[httpx.Request] = []

    def add(self, fragment: str, *responses):
        """Queue responses (httpx.Response or Exception) for URLs containing ``fragment``; the last one repeats."""
        self.routes.append((fragment, list(responses)))

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = str(request.url)
        for frag, queue in self.routes:
            if frag in url:
                item = queue.pop(0) if len(queue) > 1 else queue[0]
                if isinstance(item, Exception):
                    if isinstance(item, httpx.HTTPError):
                        item.request = request
                    raise item
                return item
        return httpx.Response(404, json={"messages": ["no route"]})

    def count(self, fragment: str) -> int:
        return sum(1 for r in self.requests if fragment in str(r.url))


def ok(payload) -> httpx.Response:
    return httpx.Response(200, content=json.dumps(payload).encode(), headers={"content-type": "application/json"})


def make_client(router: Router, **kw) -> EspnClient:
    kw.setdefault("backoff_base", 0.0)
    return EspnClient(transport=httpx.MockTransport(router), **kw)


LEAGUE_PATH = f"/seasons/{SEASON}/segments/0/leagues/{LEAGUE_ID}"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def test_format_swid_and_filter():
    assert format_swid("ABC-123") == "{ABC-123}" and format_swid("{ABC-123}") == "{ABC-123}" and format_swid("  ") is None
    assert format_swid(None) is None and format_swid("{abc") == "{abc}"
    # browsers show the cookie URL-encoded; ESPN wants the raw braces
    assert format_swid("%7BABC-123%7D") == "{ABC-123}" and format_swid(" %7Babc%7D ") == "{abc}"
    f = player_filter(2026, 300, "STANDARD")
    assert f["players"]["limit"] == 300 and f["players"]["sortDraftRanks"]["value"] == "STANDARD"
    assert f["players"]["filterStatsForTopScoringPeriodIds"]["additionalValue"] == ["002026", "102026"]


def test_base_url_env_override(monkeypatch):
    monkeypatch.delenv("DRAFTADVISOR_ESPN_BASE", raising=False)
    assert espn_base_url() == DEFAULT_ESPN_BASE_URL
    assert EspnClient().base_url == DEFAULT_ESPN_BASE_URL
    monkeypatch.setenv("DRAFTADVISOR_ESPN_BASE", "http://127.0.0.1:1/apis/v3/games/ffl/")
    assert espn_base_url() == "http://127.0.0.1:1/apis/v3/games/ffl"
    c = EspnClient()
    assert c.base_url == "http://127.0.0.1:1/apis/v3/games/ffl"
    assert c.league_url("5", 2026) == "http://127.0.0.1:1/apis/v3/games/ffl/seasons/2026/segments/0/leagues/5"
    assert EspnClient(base_url="http://x/y/").base_url == "http://x/y"


# ---------------------------------------------------------------------------
# MockTransport
# ---------------------------------------------------------------------------


async def test_views_cookies_and_payloads():
    league = load_fixture("league_settings_teams.json")
    r = Router()
    r.add("view=mDraftDetail", ok(load_fixture("draft_in_progress.json")))
    r.add("view=kona_player_info", ok(load_fixture("players_kona.json")))
    r.add(LEAGUE_PATH, ok(league))
    c = make_client(r, espn_s2="s2%2Bvalue", swid="ABCD-1234")
    assert c.has_cookies and c.swid == "{ABCD-1234}"
    async with c:
        assert (await c.get_settings_and_teams(LEAGUE_ID, SEASON))["settings"]["name"] == "FXBG League"
        assert len((await c.get_draft_detail(LEAGUE_ID, SEASON))["draftDetail"]["picks"]) == 17
        players = await c.get_players(LEAGUE_ID, SEASON, limit=40)
        assert len(players) == 40 and players[0]["player"]["fullName"]
        assert (await c.get_league(LEAGUE_ID, SEASON, ["mSettings"]))["id"] == 368876
    first = r.requests[0]
    assert str(first.url) == f"{DEFAULT_ESPN_BASE_URL}{LEAGUE_PATH}?view=mSettings&view=mTeam&view=mRoster"
    cookie = first.headers["cookie"]
    assert "espn_s2=s2%2Bvalue" in cookie and "SWID={ABCD-1234}" in cookie
    assert first.headers["User-Agent"].startswith("draftadvisor") and first.headers["Accept"] == "application/json"
    assert "view=mDraftDetail&view=mSettings" in str(r.requests[1].url)
    kona = r.requests[2]
    flt = json.loads(kona.headers["x-fantasy-filter"])
    assert flt["players"]["limit"] == 40 and flt["players"]["filterRanksForRankTypes"]["value"] == ["PPR"]
    assert flt["players"]["filterStatsForTopScoringPeriodIds"]["additionalValue"] == [f"00{SEASON}", f"10{SEASON}"]
    assert "view=kona_player_info" in str(kona.url)
    assert c.request_count == 4 and c.last_latency_ms >= 0


async def test_no_cookies_sends_none():
    r = Router()
    r.add(LEAGUE_PATH, ok({"id": 1}))
    c = make_client(r)
    assert not c.has_cookies
    await c.get_league(LEAGUE_ID, SEASON, ["mSettings"])
    assert "cookie" not in r.requests[0].headers


async def test_401_falls_back_to_league_history_then_access_denied():
    r = Router()
    r.add("leagueHistory", httpx.Response(401, json={"messages": ["not authorized"]}))
    r.add(LEAGUE_PATH, httpx.Response(401, json={"messages": ["not authorized"]}))
    c = make_client(r)
    with pytest.raises(EspnAccessDenied) as ei:
        await c.get_settings_and_teams(LEAGUE_ID, SEASON)
    assert ei.value.status_code == 401 and "espn_s2" in str(ei.value) and "SWID" in str(ei.value)
    assert r.count("leagueHistory") == 1 and r.count(LEAGUE_PATH) == 1
    hist = str(r.requests[1].url)
    assert hist.startswith(f"{DEFAULT_ESPN_BASE_URL}/leagueHistory/{LEAGUE_ID}?") and f"seasonId={SEASON}" in hist
    assert "view=mSettings" in hist
    # with cookies the message blames the cookies, not their absence; 403 behaves like 401
    r2 = Router()
    r2.add("leagueHistory", httpx.Response(403))
    r2.add(LEAGUE_PATH, httpx.Response(403))
    c2 = make_client(r2, espn_s2="x", swid="y")
    with pytest.raises(EspnAccessDenied) as ei2:
        await c2.get_draft_detail(LEAGUE_ID, SEASON)
    assert ei2.value.status_code == 403 and "supplied espn_s2" in str(ei2.value) and "x" not in str(ei2.value).split("espn_s2")[0]


async def test_401_then_history_success_normalises_list_body():
    r = Router()
    r.add("leagueHistory", ok([load_fixture("draft_pre_draft.json")]))
    r.add(LEAGUE_PATH, httpx.Response(401))
    c = make_client(r)
    data = await c.get_draft_detail(LEAGUE_ID, SEASON)
    assert isinstance(data, dict) and data["draftDetail"]["picks"] == []
    r2 = Router()
    r2.add("leagueHistory", ok([]))
    r2.add(LEAGUE_PATH, httpx.Response(401))
    with pytest.raises(EspnNotFound):
        await make_client(r2).get_draft_detail(LEAGUE_ID, SEASON)


async def test_404_raises_not_found_without_retry_or_fallback():
    r = Router()
    r.add(LEAGUE_PATH, httpx.Response(404, json={"messages": ["League not found"]}))
    c = make_client(r, retries=3)
    with pytest.raises(EspnNotFound) as ei:
        await c.get_settings_and_teams(LEAGUE_ID, SEASON)
    assert ei.value.status_code == 404 and LEAGUE_ID in str(ei.value) and r.count(LEAGUE_PATH) == 1
    assert r.count("leagueHistory") == 0


async def test_retries_on_5xx_429_and_transport_errors():
    r = Router()
    r.add(LEAGUE_PATH, httpx.Response(503), httpx.Response(429, headers={"Retry-After": "0"}), httpx.ReadTimeout("slow"),
          httpx.ConnectError("down"), ok({"id": 1}))
    c = make_client(r, retries=4)
    assert (await c.get_league(LEAGUE_ID, SEASON, ["mSettings"]))["id"] == 1
    assert r.count(LEAGUE_PATH) == 5


async def test_retries_exhausted_and_other_4xx():
    r = Router()
    r.add(LEAGUE_PATH, httpx.Response(502))
    c = make_client(r, retries=2)
    with pytest.raises(EspnAPIError) as ei:
        await c.get_settings_and_teams(LEAGUE_ID, SEASON)
    assert ei.value.status_code == 502 and r.count(LEAGUE_PATH) == 3 and not isinstance(ei.value, EspnNotFound)
    r2 = Router()
    r2.add(LEAGUE_PATH, httpx.Response(400))
    with pytest.raises(EspnAPIError) as ei2:
        await make_client(r2).get_settings_and_teams(LEAGUE_ID, SEASON)
    assert ei2.value.status_code == 400 and r2.count(LEAGUE_PATH) == 1 and not isinstance(ei2.value, EspnAccessDenied)
    r3 = Router()
    r3.add(LEAGUE_PATH, httpx.ReadTimeout("slow"))
    with pytest.raises(EspnAPIError) as ei3:
        await make_client(r3, retries=1).get_settings_and_teams(LEAGUE_ID, SEASON)
    assert ei3.value.status_code is None and "ReadTimeout" in str(ei3.value)
    r4 = Router()
    r4.add(LEAGUE_PATH, httpx.Response(200, content=b"<html>oops</html>"))
    with pytest.raises(EspnAPIError):
        await make_client(r4).get_settings_and_teams(LEAGUE_ID, SEASON)


async def test_get_players_tolerates_odd_payloads():
    r = Router()
    r.add("kona_player_info", ok({"players": [{"id": 1, "player": {}}, "junk", None]}))
    assert await make_client(r).get_players(LEAGUE_ID, SEASON) == [{"id": 1, "player": {}}]
    r2 = Router()
    r2.add("kona_player_info", ok({"id": 1}))
    assert await make_client(r2).get_players(LEAGUE_ID, SEASON) == []


async def test_fan_leagues():
    payload = {"preferences": [
        {"metaData": {"entry": {"entryId": 7, "gameId": 1, "seasonId": 2026, "entryMetadata": {"teamName": "Me"},
                                "groups": [{"groupId": 111, "groupName": "A"}, {"groupId": 111, "groupName": "A"}]}}},
        {"metaData": {"entry": {"entryId": 8, "gameId": 2, "abbrev": "FBA", "seasonId": 2026, "groups": [{"groupId": 222}]}}},
        {"metaData": {"entry": "junk"}}, {"metaData": None}, "junk",
    ]}
    assert parse_fan_leagues(payload) == [{"league_id": "111", "name": "A", "season": 2026, "team_name": "Me", "team_id": 7}]
    assert parse_fan_leagues({}) == [] and parse_fan_leagues([1]) == [] and parse_fan_leagues({"preferences": "x"}) == []
    r = Router()
    r.add("/apis/v2/fans/", ok(payload))
    c = make_client(r, swid="ABC", fan_base_url="http://fan.local")
    got = await c.get_fan_leagues()
    assert got[0]["league_id"] == "111"
    assert str(r.requests[0].url).startswith("http://fan.local/apis/v2/fans/%7BABC%7D?") or "{ABC}" in str(r.requests[0].url)
    assert await c.get_fan_leagues("") == got                       # falls back to the client's SWID
    assert await make_client(Router()).get_fan_leagues() == []      # no SWID at all
    r2 = Router()
    r2.add("/apis/v2/fans/", httpx.Response(500))
    assert await make_client(r2, swid="ABC", retries=0).get_fan_leagues() == []


async def test_fan_api_swid_never_reaches_logs_or_errors(caplog):
    """The Fan API URL carries the SWID in its path: every log line and error message shows it redacted."""
    from draftadvisor.espn.client import redact_url

    secret = "ABCD-1234-SWID-SECRET"
    assert redact_url(f"http://fan.local/apis/v2/fans/{{{secret}}}?x=1&y=2") == "http://fan.local/apis/v2/fans/<swid>?x=1&y=2"
    assert redact_url("http://fan.local/apis/v2/fans/%7BABC%7D") == "http://fan.local/apis/v2/fans/<swid>"
    assert redact_url(f"{DEFAULT_ESPN_BASE_URL}{LEAGUE_PATH}?view=mSettings") == f"{DEFAULT_ESPN_BASE_URL}{LEAGUE_PATH}?view=mSettings"
    r = Router()
    r.add("/apis/v2/fans/", httpx.Response(503), httpx.ConnectError("down"), httpx.Response(429))
    c = make_client(r, swid=secret, retries=2, fan_base_url="http://fan.local")
    with caplog.at_level("DEBUG", logger="draftadvisor.espn.client"):
        assert await c.get_fan_leagues() == []
        with pytest.raises(EspnAPIError) as ei:
            await c.get_json(f"http://fan.local/apis/v2/fans/{{{secret}}}?context=fantasy")
    assert r.count("/apis/v2/fans/") >= 4 and "attempt" in caplog.text and "HTTP 503" in caplog.text
    assert secret not in caplog.text and "/fans/<swid>" in caplog.text
    assert secret not in str(ei.value) and secret not in (ei.value.url or "") and "/fans/<swid>" in str(ei.value)
    # a 4xx from the payload path is redacted as well
    r2 = Router()
    r2.add("/apis/v2/fans/", httpx.Response(400))
    with pytest.raises(EspnAPIError) as ei2:
        await make_client(r2, swid=secret, fan_base_url="http://fan.local").get_json(f"http://fan.local/apis/v2/fans/{{{secret}}}")
    assert secret not in str(ei2.value) and ei2.value.url == "http://fan.local/apis/v2/fans/<swid>"
    # the request itself still carries the real SWID
    assert secret in str(r.requests[0].url) or "%7B" + secret in str(r.requests[0].url).replace("%2D", "-")


async def test_injected_http_client_and_close():
    r = Router()
    r.add(LEAGUE_PATH, ok({"id": 1}))
    http = httpx.AsyncClient(transport=httpx.MockTransport(r))
    c = EspnClient(http=http, espn_s2="a", swid="b")
    await c.get_league(LEAGUE_ID, SEASON, ["mSettings"])
    assert "SWID={b}" in r.requests[0].headers["cookie"] and r.requests[0].headers["User-Agent"].startswith("draftadvisor")
    await c.aclose()
    assert not http.is_closed                                       # not ours to close
    await http.aclose()
    own = EspnClient(timeout=5.0)
    await own.aclose()
    assert own._http.is_closed


# ---------------------------------------------------------------------------
# the stub server (real HTTP on loopback), as the web e2e will use it
# ---------------------------------------------------------------------------


@pytest.fixture
def stub():
    with EspnStub(draft="live", picks_visible=17) as s:
        yield s


async def test_stub_end_to_end(stub, monkeypatch):
    monkeypatch.setenv("DRAFTADVISOR_ESPN_BASE", stub.base_url)
    async with EspnClient(retries=2, backoff_base=0.0) as c:
        assert c.base_url == stub.base_url
        league = await c.get_settings_and_teams(LEAGUE_ID, SEASON)
        assert league["settings"]["name"] == "FXBG League" and len(league["teams"]) == 10
        draft = await c.get_draft_detail(LEAGUE_ID, SEASON)
        assert len(draft["draftDetail"]["picks"]) == 17 and draft["draftDetail"]["inProgress"] is True
        players = await c.get_players(LEAGUE_ID, SEASON)
        assert len(players) == 40
        with pytest.raises(EspnNotFound):
            await c.get_settings_and_teams("999", SEASON)
        # simulated outage: 2 failures then success
        stub.fail_next = 2
        assert (await c.get_draft_detail(LEAGUE_ID, SEASON))["id"] == 368876
        # the whole draft
        stub.picks_visible = None
        d = await c.get_draft_detail(LEAGUE_ID, SEASON)
        assert d["draftDetail"]["drafted"] is True and len(d["draftDetail"]["picks"]) == 150
        # fan api served by the same stub
        c.fan_base_url = stub.fan_base_url
        c.swid = "{6863-6934-3455}"
        assert (await c.get_fan_leagues())[0] == {"league_id": LEAGUE_ID, "name": "FXBG League", "season": SEASON,
                                                  "team_name": "Goin' HAM Newton", "team_id": 1}
    kona_req = next(q for q in stub.requests if "kona_player_info" in q["query"].get("view", []))
    assert json.loads(kona_req["headers"]["x-fantasy-filter"])["players"]["limit"] == 600
    assert stub.requests[0]["query"]["view"] == ["mSettings", "mTeam", "mRoster"]


async def test_stub_private_league_requires_cookies():
    with EspnStub(private=True) as s:
        async with EspnClient(base_url=s.base_url, retries=0) as c:
            with pytest.raises(EspnAccessDenied) as ei:
                await c.get_settings_and_teams(LEAGUE_ID, SEASON)
            assert "private" in str(ei.value)
        # the 401 triggered the leagueHistory fallback before giving up
        assert [q["path"].split("/ffl")[1] for q in s.requests] == [LEAGUE_PATH, f"/leagueHistory/{LEAGUE_ID}"]
        async with EspnClient(base_url=s.base_url, espn_s2="secret", swid="6863-6934-3455") as c:
            assert (await c.get_settings_and_teams(LEAGUE_ID, SEASON))["id"] == 368876
        assert s.requests[-1]["cookies"] == {"espn_s2": "secret", "SWID": "{6863-6934-3455}"}


def test_stub_can_run_as_a_script_help():
    import subprocess
    import sys

    out = subprocess.run([sys.executable, str(Path(__file__).parent / "espn_stub.py"), "--help"], capture_output=True,
                         text=True, timeout=30, env=dict(os.environ))
    assert out.returncode == 0 and "--draft" in out.stdout


# ---------------------------------------------------------------------------
# The per-poll live GET: board + settings + rosters in one request, and what it reports back
# ---------------------------------------------------------------------------


async def test_get_draft_live_asks_for_the_rosters_in_the_same_request():
    """ESPN's board publishes nothing while a draft runs; the team rosters ride along in the same GET
    (four views, one request) so a drafted player is seen at all."""
    r = Router()
    r.add(LEAGUE_PATH, ok({"id": 368876, "seasonId": SEASON, "draftDetail": {"picks": []}, "teams": []}))
    c = make_client(r)
    info: dict = {}
    await c.get_draft_live(LEAGUE_ID, SEASON, info=info)
    url = str(r.requests[0].url)
    assert url.count("view=") == 4 and "view=mTeam" in url and "view=mRoster" in url and "scoringPeriodId=0" in url
    assert r.requests[0].headers.get("Cache-Control") is None            # an ordinary poll uses the CDN
    assert info["views"] == ["mDraftDetail", "mSettings", "mTeam", "mRoster"]
    assert info["http_status"] == 200 and info["bytes"] > 0 and info["read_at"] > 0 and info["used_history"] is False
    assert info["season_returned"] == SEASON and info["latency_ms"] >= 0
    assert c.last_request_info()["views"] == ["mDraftDetail", "mSettings", "mTeam", "mRoster"]
    # rosters off (DRAFTADVISOR_ESPN_POLL_ROSTERS=0): the old two-view request, no scoringPeriodId
    await c.get_draft_live(LEAGUE_ID, SEASON, rosters=False)
    assert str(r.requests[1].url).endswith("?view=mDraftDetail&view=mSettings")
    # a user-triggered refresh asks intermediaries not to answer from their cache
    await c.get_draft_live(LEAGUE_ID, SEASON, no_cache=True)
    assert r.requests[2].headers["Cache-Control"] == "no-cache" and r.requests[2].headers["Pragma"] == "no-cache"


async def test_last_request_info_reports_the_response_cache_headers():
    """A poll answered by a CDN (Age / X-Cache) looks exactly like a working poll of a frozen board:
    the numbers have to be visible, or it can only be diagnosed by guessing."""
    r = Router()
    r.add(LEAGUE_PATH, httpx.Response(200, json={"id": 1, "seasonId": SEASON},
                                      headers={"Age": "4", "X-Cache": "Hit from cloudfront",
                                               "Cache-Control": "max-age=5"}))
    c = make_client(r)
    info: dict = {}
    await c.get_draft_live(LEAGUE_ID, SEASON, info=info)
    assert info["cache_headers"] == {"age": "4", "x-cache": "Hit from cloudfront", "cache-control": "max-age=5"}
    assert c.last_request_info()["cache_headers"]["age"] == "4"


async def test_league_history_fallback_picks_the_requested_season():
    """The history endpoint answers with one object per season and does not always honour seasonId.
    Serving data[0] blindly renders another year's league as a plausible, frozen board."""
    r = Router()
    other = {"id": 368876, "seasonId": SEASON - 1, "draftDetail": {"drafted": True, "picks": [{"playerId": 1}]}}
    want = {"id": 368876, "seasonId": SEASON, "draftDetail": {"drafted": False, "picks": []}}
    r.add("leagueHistory", ok([other, want]))
    r.add(LEAGUE_PATH, httpx.Response(401))
    info: dict = {}
    data = await make_client(r).get_draft_live(LEAGUE_ID, SEASON, info=info)
    assert data["seasonId"] == SEASON and data["draftDetail"]["picks"] == []
    assert info["used_history"] is True and info["season_returned"] == SEASON
    # a body that carries seasons but not the one asked for is an error, never a silent substitution
    r2 = Router()
    r2.add("leagueHistory", ok([other]))
    r2.add(LEAGUE_PATH, httpx.Response(401))
    with pytest.raises(EspnNotFound) as ei:
        await make_client(r2).get_draft_detail(LEAGUE_ID, SEASON)
    assert str(SEASON) in str(ei.value)
    # ... but a body with no seasonId at all is used as before
    r3 = Router()
    r3.add("leagueHistory", ok([{"id": 368876, "draftDetail": {"picks": []}}]))
    r3.add(LEAGUE_PATH, httpx.Response(401))
    assert (await make_client(r3).get_draft_detail(LEAGUE_ID, SEASON))["id"] == 368876


async def test_stub_serves_every_view_of_one_request():
    """ESPN composes one payload out of every view asked for; so must the stub, or the roster views
    would be silently dropped and every test of them would pass for the wrong reason."""
    with EspnStub(draft="in_progress", roster_picks=5) as s:
        async with EspnClient(base_url=s.base_url, retries=0) as c:
            payload = await c.get_draft_live(LEAGUE_ID, SEASON)
            assert payload["draftDetail"]["picks"] and payload["settings"]["draftSettings"]["pickOrder"]
            entries = [e for t in payload["teams"] for e in t["roster"]["entries"]]
            assert len(entries) == 5 and all(e["acquisitionType"] == "DRAFT" and e["acquisitionDate"] for e in entries)
            assert payload["members"]
            board_only = await c.get_draft_live(LEAGUE_ID, SEASON, rosters=False)
            assert "teams" not in board_only
