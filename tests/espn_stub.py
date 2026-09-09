"""Local stand-in for the ESPN fantasy API: serves ``tests/fixtures/espn`` at the real URL shapes.

Routes (``base_url`` = ``http://127.0.0.1:<port>/apis/v3/games/ffl``):

* ``/seasons/{season}/segments/0/leagues/{id}?view=...`` and ``/leagueHistory/{id}?seasonId=`` (the
  latter answers with a one-element list, like ESPN): the response is *composed* from the ``view``
  params, as ESPN composes one payload out of every view of a request - ``kona_player_info`` ->
  ``players_kona.json`` (exclusive); ``mDraftDetail`` -> the draft fixture selected by
  :attr:`EspnStub.draft` (``complete`` / ``in_progress`` / ``pre_draft`` / ``live``); ``mTeam`` /
  ``mRoster`` -> the ``teams`` and ``members`` of the league fixture; neither -> the whole league
  payload (``league_settings_teams.json``, ``league_modern.json`` when :attr:`EspnStub.league` is
  ``"modern"``, or a synthetic 12-team / 16-round league with a 192-entry placeholder board when it is
  ``"twelve"``);
* unknown league ids -> 404; a private stub without ``espn_s2`` + ``SWID`` cookies -> 401;
* ``/apis/v2/fans/{swid}`` -> a minimal Fan API payload naming the fixture league.

Knobs: ``picks_visible`` truncates the served pick list (``draft="live"`` serves the complete draft's
first N picks with ``inProgress`` until all are visible), ``roster_picks`` moves the first N picks off
the board and onto the owning teams' rosters as ``acquisitionType: DRAFT`` entries - what a real ESPN
draft looks like over REST while it is running: the board stays at ``playerId: -1`` and only the
rosters fill - ``timer_override`` / ``pick_order_override`` rewrite ``draftSettings``, ``fail_next``
makes the next N requests fail with ``fail_status``. Every request is recorded in
:attr:`EspnStub.requests` (path, query, headers, cookies).

The per-poll (``mDraftDetail``) payload carries ``draftSettings`` - the order / clock / type / date the
poller re-reads - but not ``rosterSettings``: the stub serves several league flavours (classic,
TE-premium with an OP slot) off the same draft fixtures, and the parsers read the per-poll settings
first, so the lineup slots stay with the league view to keep the two consistent (real ESPN returns the
same settings under both views; the fixture-driven tests cover lineup changes seen through a poll).

Run stand-alone for a browser / web e2e session::

    python tests/espn_stub.py --port 8765 --draft live --picks 17
    DRAFTADVISOR_ESPN_BASE=http://127.0.0.1:8765/apis/v3/games/ffl ...
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import threading
import time
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "espn"
LEAGUE_ID = "368876"
SEASON = 2018
SWID_TEAM_1 = "{6863-6934-3455}"          # primary owner of team 1 ("Goin' HAM Newton", draft slot 3)
DRAFT_DATE_MS = 1535198400000             # draftSettings.date of the draft fixtures
TWELVE_TEAMS, TWELVE_ROUNDS = 12, 16      # the "twelve" league flavour: 192 picks, like the reported one

_LEAGUE_RE = re.compile(r"^/apis/v3/games/ffl/seasons/(\d+)/segments/0/leagues/([^/]+)/?$")
_HISTORY_RE = re.compile(r"^/apis/v3/games/ffl/leagueHistory/([^/]+)/?$")
_FAN_RE = re.compile(r"^/apis/v2/fans/([^/]+)/?$")


def load_fixture(name: str) -> Any:
    with open(FIXTURES / name, encoding="utf-8") as fh:
        return json.load(fh)


class EspnStub:
    """Threaded HTTP stub; use as a context manager or call :meth:`start` / :meth:`stop`."""

    def __init__(self, *, league_ids: tuple[str, ...] = (LEAGUE_ID,), private: bool = False, draft: str = "complete",
                 league: str = "classic", picks_visible: int | None = None, host: str = "127.0.0.1", port: int = 0,
                 latency: float = 0.0, roster_picks: int | None = None, empty_rosters: bool = False):
        self.league_ids = {str(x) for x in league_ids}
        self.private = private
        self.draft = draft
        self.league = league
        self.picks_visible = picks_visible
        self.roster_picks = roster_picks
        self.empty_rosters = empty_rosters
        self.timer_override: int | None = None
        self.pick_order_override: list[int] | None = None
        self.fail_next = 0
        self.fail_status = 500
        self.latency = latency
        self.requests: list[dict] = []
        self.host, self.port = host, port
        self._cache: dict[str, Any] = {}
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    # -- lifecycle ------------------------------------------------------------
    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/apis/v3/games/ffl"

    @property
    def fan_base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> "EspnStub":
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:  # silence
                pass

            def do_GET(self) -> None:  # noqa: N802
                url = urlparse(self.path)
                query = {k: v for k, v in parse_qs(url.query).items()}
                headers = {k.lower(): v for k, v in self.headers.items()}
                cookies: dict[str, str] = {}
                if headers.get("cookie"):
                    jar = SimpleCookie()
                    jar.load(headers["cookie"])
                    cookies = {k: m.value for k, m in jar.items()}
                status, payload = stub.respond(url.path, query, headers, cookies)
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, name="espn-stub", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> "EspnStub":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # -- payloads -------------------------------------------------------------
    def fixture(self, name: str) -> Any:
        if name not in self._cache:
            self._cache[name] = load_fixture(name)
        return copy.deepcopy(self._cache[name])

    def league_payload(self) -> dict:
        if self.league == "twelve":
            return self.twelve_payload()
        return self.fixture("league_modern.json" if self.league == "modern" else "league_settings_teams.json")

    def twelve_payload(self) -> dict:
        """The classic league grown to 12 teams and 16 rounds - the shape of the league this app was
        reported broken on (12 teams, 16 rounds = 192 picks, PPR). Teams 11 and 12 are clones of the
        first two with fresh ids and empty rosters; one bench slot is added so a full roster is 16."""
        league = self.fixture("league_settings_teams.json")
        teams = league["teams"]
        next_id = max(int(t["id"]) for t in teams) + 1               # the fixture's ids are not 1..10
        for i, src in enumerate(teams[:2]):
            clone = copy.deepcopy(src)
            tid = next_id + i
            clone.update({"id": tid, "abbrev": f"T{tid}", "location": "Expansion", "nickname": f"Team {tid}",
                          "owners": [], "primaryOwner": None, "roster": {"entries": []}})
            teams.append(clone)
        st = league["settings"]
        st["size"] = TWELVE_TEAMS
        st["draftSettings"]["pickOrder"] = [t["id"] for t in teams]
        st["draftSettings"]["type"] = "SNAKE"
        st["draftSettings"]["date"] = DRAFT_DATE_MS
        counts = st["rosterSettings"]["lineupSlotCounts"]
        counts["20"] = int(counts.get("20") or 0) + 1              # one more bench slot: 16 rounds
        return league

    def twelve_board(self) -> dict:
        """A pre-populated 192-entry board for :meth:`twelve_payload`: one entry per pick in snake order,
        every ``playerId`` -1 - exactly what ESPN serves before *and throughout* a live draft."""
        league = self.twelve_payload()
        order = league["settings"]["draftSettings"]["pickOrder"]
        picks = []
        for rnd in range(1, TWELVE_ROUNDS + 1):
            slots = order if rnd % 2 else list(reversed(order))
            for i, team_id in enumerate(slots, start=1):
                picks.append({"overallPickNumber": (rnd - 1) * TWELVE_TEAMS + i, "roundId": rnd, "roundPickNumber": i,
                              "teamId": team_id, "playerId": -1, "keeper": False, "reservedForKeeper": False,
                              "bidAmount": 0, "autoDraftTypeId": 0, "id": (rnd - 1) * TWELVE_TEAMS + i,
                              "lineupSlotId": 0, "nominatingTeamId": 0})
        live = self.draft in ("live", "in_progress")
        return {"gameId": 1, "id": int(LEAGUE_ID), "seasonId": SEASON, "segmentId": 0,
                "settings": {"draftSettings": copy.deepcopy(league["settings"]["draftSettings"])},
                "draftDetail": {"completeDate": 0, "drafted": False, "inProgress": live, "picks": picks}}

    def _apply_overrides(self, payload: dict) -> dict:
        ds = payload.setdefault("settings", {}).setdefault("draftSettings", {})
        if self.timer_override is not None:
            ds["timePerSelection"] = int(self.timer_override)
        if self.pick_order_override is not None:
            ds["pickOrder"] = list(self.pick_order_override)
        return payload

    def draft_payload(self) -> dict:
        if self.league == "twelve":
            return self._apply_overrides(self.twelve_board())
        if self.draft == "live":
            d = self.fixture("draft_complete.json")
            picks = d["draftDetail"]["picks"]
            n = len(picks) if self.picks_visible is None else max(0, int(self.picks_visible))
            drafted = n >= len(picks)
            d["draftDetail"].update({"picks": picks[:n], "drafted": drafted, "inProgress": not drafted and n > 0})
            if not drafted:
                d["draftDetail"]["completeDate"] = 0
                d["settings"]["draftSettings"]["type"] = "SNAKE"
        else:
            d = self.fixture(f"draft_{self.draft}.json")
            if self.picks_visible is not None:
                d["draftDetail"]["picks"] = d["draftDetail"]["picks"][: int(self.picks_visible)]
        d.get("settings", {}).pop("rosterSettings", None)      # lineup slots come from the league view (see module doc)
        return self._apply_overrides(d)

    def inseason_payload(self) -> dict:
        """``league_inseason.json``: the shape views mMatchupScore / mScoreboard add to mTeam + mRoster.

        Its numbers are synthetic (see ``fixtures/espn/make_fixtures.py``); the stub serves it so the
        in-season code paths have something faithful to parse without a live league.
        """
        return self.fixture("league_inseason.json")

    def players_payload(self) -> dict:
        return self.fixture("players_kona.json")

    def teams_payload(self) -> dict:
        """``teams`` + ``members`` as views mTeam + mRoster return them.

        With ``roster_picks`` set, the rosters are rebuilt to hold exactly the first N picks of the
        draft fixture (``acquisitionType`` DRAFT, ``acquisitionDate`` a second apart from the draft's
        scheduled time) and nothing else - the live-draft shape the board never shows. With
        ``empty_rosters`` every roster is emptied (a redraft league before its draft).
        """
        league = self.league_payload()
        teams = league.get("teams") or []
        if self.empty_rosters or self.roster_picks is not None:
            for t in teams:
                t["roster"] = {"entries": []}
        if self.roster_picks:
            # the players of the first N picks of the completed fixture, on the teams that own picks
            # 1..N of the board actually being served, stamped a second apart from the draft's start
            drafted = self.fixture("draft_complete.json")["draftDetail"]["picks"]
            board = sorted((p for p in self.draft_payload()["draftDetail"]["picks"]),
                           key=lambda p: p.get("overallPickNumber") or 0)
            by_id = {t.get("id"): t for t in teams}
            for i, (owner, pick) in enumerate(list(zip(board, drafted))[: int(self.roster_picks)]):
                team = by_id.get(owner.get("teamId"))
                if team is None:
                    continue
                team["roster"]["entries"].append({
                    "playerId": pick["playerId"], "lineupSlotId": 20, "acquisitionType": "DRAFT",
                    "acquisitionDate": DRAFT_DATE_MS + 1000 * i, "injuryStatus": "NORMAL",
                    "playerPoolEntry": {"id": pick["playerId"], "player": {"id": pick["playerId"]}},
                })
        return {"teams": teams, "members": league.get("members") or []}

    def fan_payload(self, swid: str) -> dict:
        return {
            "preferences": [{
                "id": f"{LEAGUE_ID}:1",
                "metaData": {"entry": {
                    "entryId": 1, "gameId": 1, "abbrev": "FFL", "seasonId": SEASON,
                    "entryMetadata": {"teamName": "Goin' HAM Newton"},
                    "groups": [{"groupId": int(LEAGUE_ID), "groupName": "FXBG League"}],
                }},
            }],
        }

    # -- routing --------------------------------------------------------------
    def respond(self, path: str, query: dict[str, list[str]], headers: dict[str, str],
                cookies: dict[str, str]) -> tuple[int, Any]:
        with self._lock:
            self.requests.append({"path": path, "query": query, "headers": headers, "cookies": cookies,
                                  "at": time.time()})
            if self.fail_next > 0:
                self.fail_next -= 1
                return self.fail_status, {"messages": ["stub: simulated failure"]}
        if self.latency:
            time.sleep(self.latency)
        m = _FAN_RE.match(path)
        if m:
            return 200, self.fan_payload(m.group(1))
        history = False
        m = _LEAGUE_RE.match(path)
        if m:
            league_id = m.group(2)
        else:
            m = _HISTORY_RE.match(path)
            if not m:
                return 404, {"messages": ["stub: unknown route"]}
            league_id, history = m.group(1), True
        if league_id not in self.league_ids:
            return 404, {"messages": ["League not found"]}
        if self.private and not (cookies.get("espn_s2") and cookies.get("SWID")):
            return 401, {"messages": ["You are not authorized to view this League."]}
        views = query.get("view", [])
        if "kona_player_info" in views:
            payload: Any = self.players_payload()
        elif any(v in views for v in ("mMatchupScore", "mScoreboard")):
            payload = self.inseason_payload()
        elif "mDraftDetail" in views:
            payload = self.draft_payload()
            if any(v in views for v in ("mTeam", "mRoster")):
                payload.update(self.teams_payload())      # ESPN answers one payload for every view asked
        else:
            payload = self.league_payload()
        return 200, ([payload] if history else payload)


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve the ESPN fixtures at the real URL shapes.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--draft", default="live", choices=("complete", "in_progress", "pre_draft", "live"))
    ap.add_argument("--league", default="classic", choices=("classic", "modern", "twelve"))
    ap.add_argument("--picks", type=int, default=None, help="number of picks visible")
    ap.add_argument("--roster-picks", type=int, default=None,
                    help="serve the first N picks on team rosters only (live-draft shape: empty board)")
    ap.add_argument("--private", action="store_true", help="require espn_s2 + SWID cookies")
    args = ap.parse_args()
    stub = EspnStub(private=args.private, draft=args.draft, league=args.league, picks_visible=args.picks,
                    roster_picks=args.roster_picks, host=args.host, port=args.port).start()
    print(f"ESPN stub serving league {LEAGUE_ID} ({SEASON}) at {stub.base_url}")
    print(f"export DRAFTADVISOR_ESPN_BASE={stub.base_url}")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        stub.stop()


if __name__ == "__main__":
    main()
