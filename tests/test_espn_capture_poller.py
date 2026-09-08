"""ESPN league capture (LeagueSnapshot), ADP / projection helpers and the draft poller."""
from __future__ import annotations

import asyncio
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from draftadvisor.capture import LeagueSnapshot
from draftadvisor.config import Settings
from draftadvisor.espn.capture import capture_espn_league, espn_adp, espn_names, espn_projections, espn_rosters
from draftadvisor.espn.client import EspnAccessDenied, EspnAPIError, EspnClient, EspnNotFound
from draftadvisor.espn.ids import EspnIdMap
from draftadvisor.espn.poller import EspnDraftPoller
from draftadvisor.models import DraftState, Player
from tests.espn_stub import LEAGUE_ID, SEASON, SWID_TEAM_1, EspnStub, load_fixture

ROOT = Path(__file__).resolve().parents[1]
PICK_ORDER = [2, 8, 1, 9, 4, 3, 10, 5, 7, 11]


class FakeEspnClient:
    """Serves the fixtures like a live league: ``n`` picks visible, mutable draft settings, failure knobs."""

    def __init__(self, n: int = 17, private: bool = False, missing: bool = False):
        self.league = load_fixture("league_settings_teams.json")
        self.draft = load_fixture("draft_complete.json")
        self.players = load_fixture("players_kona.json")["players"]
        self.n = n
        self.private, self.missing = private, missing
        self.calls: list[str] = []
        self.fail_draft_times = 0
        self.fail_players = False
        self.drafted: bool | None = None            # None = derived from n

    def _guard(self, name: str) -> None:
        self.calls.append(name)
        if self.missing:
            raise EspnNotFound("not found", status_code=404)
        if self.private:
            raise EspnAccessDenied("private", status_code=401)

    async def get_settings_and_teams(self, league_id, season):
        self._guard("settings")
        return copy.deepcopy(self.league)

    async def get_draft_detail(self, league_id, season):
        self._guard("draft")
        if self.fail_draft_times > 0:
            self.fail_draft_times -= 1
            raise EspnAPIError("boom", status_code=503)
        d = copy.deepcopy(self.draft)
        picks = d["draftDetail"]["picks"][: self.n]
        drafted = (self.n >= 150) if self.drafted is None else self.drafted
        d["draftDetail"].update({"picks": picks, "drafted": drafted, "inProgress": not drafted and self.n > 0})
        return d

    async def get_players(self, league_id, season, limit=600, rank_type="PPR"):
        self.calls.append("players")
        if self.fail_players:
            raise EspnAPIError("players down", status_code=500)
        return copy.deepcopy(self.players)


def universe() -> dict[str, Player]:
    pls = [Player("gurley", "Todd Gurley", "RB", "LAR", espn_id="2977644"), Player("ab", "Antonio Brown", "WR", "PIT"),
           Player("JAX", "JAX Defense", "DEF", "JAX"), Player("tucker", "Justin Tucker", "K", "BAL")]
    return {p.player_id: p for p in pls}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def test_names_adp_projections_helpers():
    kona = load_fixture("players_kona.json")["players"]
    m = EspnIdMap.from_players(universe())
    names = espn_names(kona)
    assert len(names) == 40 and names["2977644"]["name"] == "Todd Gurley II" and names["2977644"]["team"] == "LAR"
    adp = espn_adp(kona, m)
    assert adp["gurley"] == pytest.approx(1.4) and adp["ab"] == pytest.approx(5.5) and len(adp) == 40
    assert all(v > 0 for v in adp.values()) and any(EspnIdMap.is_synthetic(k) for k in adp)
    # ADP missing -> draft rank fallback; nothing at all -> skipped
    entries = [{"id": 1, "player": {"id": 1, "fullName": "Nobody One", "defaultPositionId": 2, "ownership": {"averageDraftPosition": 0},
                                    "draftRanksByRankType": {"STANDARD": {"rank": 12}}}},
               {"id": 2, "player": {"id": 2, "fullName": "Nobody Two", "defaultPositionId": 2}}, "junk"]
    assert espn_adp(entries, m) == {"espn:1": 12.0}
    proj = espn_projections(kona, m, SEASON)
    assert len(proj) == 40 and proj["gurley"]["rush_yd"] > 1000 and proj["gurley"]["gp"] > 0
    assert "JAX" in proj and proj["JAX"]["sack"] > 0 and "def_st_td" in proj["JAX"]
    assert espn_projections(kona, m, 2019) == {} and espn_projections(None, m, SEASON) == {}
    assert espn_projections(entries, m, SEASON) == {}
    rosters = espn_rosters(load_fixture("league_settings_teams.json"), m)
    assert len(rosters) == 10 and rosters[0]["roster_id"] == 1 and "ab" in rosters[0]["players"]
    assert espn_rosters({"teams": [{"id": "x"}, {"roster": {}}]}, m) == []


def test_idp_and_punters_are_named_but_never_ranked_or_projected():
    """An IDP league's pool: the LB keeps its label and gets neither an ADP nor a DEF-key projection as a WR."""
    from draftadvisor.espn.ids import placeholder_player

    m = EspnIdMap.from_players(universe())
    lb = {"id": 4999999001, "player": {"id": 4999999001, "fullName": "Roquan Smith", "defaultPositionId": 11,
                                       "eligibleSlots": [10, 15, 20, 21], "proTeamId": 33,
                                       "ownership": {"averageDraftPosition": 45},
                                       "stats": [{"id": f"10{SEASON}", "seasonId": SEASON, "statSourceId": 1, "statSplitTypeId": 0,
                                                  "appliedTotal": 18.0, "stats": {"99": 3, "95": 2, "106": 2, "96": 1, "210": 17}}]}}
    punter = {"id": 4999999002, "player": {"id": 4999999002, "fullName": "Johnny Hekker", "defaultPositionId": 7,
                                           "eligibleSlots": [18, 20, 21], "proTeamId": 14,
                                           "ownership": {"averageDraftPosition": 170}}}
    rb = {"id": 4999999003, "player": {"id": 4999999003, "fullName": "Some Back", "defaultPositionId": 2, "proTeamId": 14,
                                       "ownership": {"averageDraftPosition": 80}}}
    pool = [lb, punter, rb]
    assert espn_adp(pool, m) == {"espn:4999999003": 80.0}
    assert espn_projections(pool, m, SEASON) == {}
    names = espn_names(pool)
    assert names["4999999001"]["position"] == "LB" and names["4999999002"]["position"] == "P" and names["4999999001"]["team"] == "BAL"
    for entry, pos in ((lb, "LB"), (punter, "P")):
        pl = placeholder_player(entry, m.resolve_player_json(entry))
        assert pl.position == pos and pl.fantasy_positions == (pos,) and pl.player_id.startswith("espn:")
        assert pl.position not in ("QB", "RB", "WR", "TE", "K", "DEF")


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------


async def test_capture_snapshot_fields():
    fc = FakeEspnClient(n=17)
    m = EspnIdMap.from_players(universe())
    snap = await capture_espn_league(fc, LEAGUE_ID, SEASON, swid="6863-6934-3455", id_map=m)
    assert isinstance(snap, LeagueSnapshot)
    assert snap.league.name == "FXBG League" and snap.league_id == LEAGUE_ID and snap.draft_id == "espn-368876-2018"
    assert snap.name == "FXBG League" and snap.season == SEASON
    assert snap.draft.status == "drafting" and snap.draft.teams == 10 and snap.draft.rounds == 15
    assert len(snap.managers) == 10 and snap.manager_for_slot(3).team_name == "Goin' HAM Newton"
    assert snap.my_user_id == "1" and snap.my_slot == 3 and snap.my_roster_id == 1
    assert snap.my_picks == [3, 18, 23, 38, 43, 58, 63, 78, 83, 98, 103, 118, 123, 138, 143]
    assert snap.picks_made == 17 and snap.keepers == []
    assert snap.diff.scoring_type == "ppr" and {d.key: d.league for d in snap.diff.changed}["pass_td"] == 6.0
    assert "ESPN league (platform: espn)" in snap.flags and "ESPN rule not modelled: 1pt Safety (1)" in snap.flags
    assert "Full PPR (1.0 per reception)" in snap.flags and any(f.startswith("6-pt passing TD") for f in snap.flags)
    assert snap.nfl_state == {}
    raw = snap.raw
    assert raw["platform"] == "espn" and raw["league"]["id"] == 368876 and len(raw["draft"]["draftDetail"]["picks"]) == 17
    assert len(raw["players"]) == 40 and len(raw["rosters"]) == 10
    assert len(snap.roster_players(1)) == 15 and "ab" in snap.roster_players(1) and snap.roster_players(99) == []
    assert fc.calls == ["settings", "draft", "players"]
    # the report renders (rich) without touching the network
    assert "FXBG League" in snap.to_text()


async def test_capture_keepers_and_identity_variants():
    fc = FakeEspnClient(n=17)
    fc.draft = load_fixture("draft_in_progress.json")
    snap = await capture_espn_league(fc, LEAGUE_ID, SEASON, username="LUTZ")
    assert [p.pick_no for p in snap.keepers] == [3, 14] and snap.my_slot == 2 and snap.my_user_id == "8"
    snap = await capture_espn_league(fc, LEAGUE_ID, SEASON, team_id=11)
    assert snap.my_slot == 10 and snap.my_roster_id == 11
    snap = await capture_espn_league(fc, LEAGUE_ID, SEASON, slot=4)
    assert snap.my_user_id == "9" and snap.my_slot == 4
    # unmatched swid -> spectator, no error
    snap = await capture_espn_league(fc, LEAGUE_ID, SEASON, swid="{0000-0000}")
    assert snap.my_user_id is None and snap.my_slot is None and snap.my_picks == []
    # unmatched username / team id -> ValueError listing the teams
    with pytest.raises(ValueError) as ei:
        await capture_espn_league(fc, LEAGUE_ID, SEASON, username="nobody")
    assert "Goin' HAM Newton (team 1, ijgdgvhhj)" in str(ei.value)
    with pytest.raises(ValueError):
        await capture_espn_league(fc, LEAGUE_ID, SEASON, team_id=99)


async def test_capture_without_pick_order_keeps_identity():
    fc = FakeEspnClient(n=0)
    fc.league["settings"]["draftSettings"]["pickOrder"] = []
    fc.draft["settings"]["draftSettings"]["pickOrder"] = []
    snap = await capture_espn_league(fc, LEAGUE_ID, SEASON, swid=SWID_TEAM_1)
    assert snap.my_user_id == "1" and snap.my_slot is None and snap.my_roster_id == 1 and snap.my_picks == []
    assert snap.draft.status == "pre_draft" and snap.draft.draft_order == {}
    # an explicit slot next to a name / team id / SWID (CLI: --slot with ESPN_TEAM_ID set) is not an error
    snap = await capture_espn_league(fc, LEAGUE_ID, SEASON, slot=3, username="LUTZ")
    assert snap.my_user_id == "8" and snap.my_slot == 3 and snap.my_roster_id == 8 and snap.my_picks[:2] == [3, 18]
    snap = await capture_espn_league(fc, LEAGUE_ID, SEASON, slot=3, team_id=1)
    assert snap.my_user_id == "1" and snap.my_slot == 3 and snap.my_roster_id == 1
    snap = await capture_espn_league(fc, LEAGUE_ID, SEASON, slot=3, swid=SWID_TEAM_1)
    assert snap.my_user_id == "1" and snap.my_slot == 3
    # a slot alone is still fine before the order exists; an unmatched name still raises
    snap = await capture_espn_league(fc, LEAGUE_ID, SEASON, slot=3)
    assert snap.my_user_id is None and snap.my_slot == 3 and snap.my_roster_id is None
    with pytest.raises(ValueError):
        await capture_espn_league(fc, LEAGUE_ID, SEASON, slot=3, username="nobody")


async def test_capture_rejects_a_slot_outside_the_league():
    fc = FakeEspnClient(n=17)
    with pytest.raises(ValueError) as ei:
        await capture_espn_league(fc, LEAGUE_ID, SEASON, slot=11)
    assert "slot 11" in str(ei.value) and "1..10" in str(ei.value)
    with pytest.raises(ValueError):
        await capture_espn_league(fc, LEAGUE_ID, SEASON, slot=0)
    # with another hint the bad slot is ignored and the team resolved normally
    snap = await capture_espn_league(fc, LEAGUE_ID, SEASON, slot=11, swid=SWID_TEAM_1)
    assert snap.my_user_id == "1" and snap.my_slot == 3


async def test_capture_degrades_without_players_and_can_save(tmp_path):
    fc = FakeEspnClient(n=150)
    fc.fail_players = True
    snap = await capture_espn_league(fc, LEAGUE_ID, SEASON, save=True)
    assert snap.raw["players"] == [] and snap.picks_made == 150 and snap.draft.status == "complete"
    saved = Path(os.environ["DRAFTADVISOR_HOME"]) / "leagues" / f"{LEAGUE_ID}.json"
    assert saved.exists() and json.loads(saved.read_text())["raw"]["platform"] == "espn"


async def test_capture_propagates_hard_errors():
    with pytest.raises(EspnNotFound):
        await capture_espn_league(FakeEspnClient(missing=True), LEAGUE_ID, SEASON)
    with pytest.raises(EspnAccessDenied):
        await capture_espn_league(FakeEspnClient(private=True), LEAGUE_ID, SEASON)


# ---------------------------------------------------------------------------
# poller
# ---------------------------------------------------------------------------


def make_poller(fc: FakeEspnClient, poll_seconds: float = 2.0, **kw) -> EspnDraftPoller:
    kw.setdefault("id_map", EspnIdMap.from_players(universe()))
    kw.setdefault("names", espn_names(fc.players))
    return EspnDraftPoller(fc, LEAGUE_ID, SEASON, poll_seconds=poll_seconds, **kw)


async def test_bootstrap_builds_full_state():
    fc = FakeEspnClient(n=17)
    p = make_poller(fc, swid=SWID_TEAM_1)
    st = await p.bootstrap()
    assert isinstance(st, DraftState) and p.state is st and p.poll_count == 1 and p.last_poll_at is not None
    assert p.draft_id == "espn-368876-2018" and p.settings.poll_seconds == 2.0
    assert st.my_user_id == "1" and st.my_slot == 3 and st.league is not None and len(st.managers) == 10
    assert len(st.picks) == 17 and st.next_pick_no == 18 and st.is_my_turn and st.version == 0
    assert st.picks[1].player_id == "gurley" and st.picks[1].player_name == "Todd Gurley II"
    assert st.picks[0].player_name == "espn:15825"                  # on no roster and not in the player pool
    assert fc.calls == ["settings", "draft"]
    # a pick beyond the player pool still gets its name from the roster payload (names are merged)
    fc.n = 150
    st = await p.poll_once()
    from draftadvisor.espn.parsing import roster_names
    only_roster = set(roster_names(fc.league)) - set(espn_names(fc.players))
    pick = next(pk for pk in st.picks if pk.metadata["espn_id"] in only_roster)
    assert pick.player_name == roster_names(fc.league)[pick.metadata["espn_id"]]["name"] and pick.position


async def test_bootstrap_with_given_league_json_skips_the_settings_call():
    fc = FakeEspnClient(n=17)
    p = make_poller(fc, username="LUTZ", league_json=fc.league)
    st = await p.bootstrap()
    assert fc.calls == ["draft"] and st.my_slot == 2
    st2 = await p.refresh_league()
    assert fc.calls == ["draft", "settings"] and st2 is p.state and st2.version == 1 and st2.my_slot == 2


async def test_poll_once_returns_new_state_only_on_change():
    fc = FakeEspnClient(n=17)
    p = make_poller(fc, swid=SWID_TEAM_1)
    st0 = await p.bootstrap()
    assert await p.poll_once() is None and await p.poll_once() is None and p.poll_count == 3
    fc.n = 19
    st1 = await p.poll_once()
    assert st1 is not None and st1 is not st0 and len(st1.picks) == 19 and st1.version == st0.version + 1
    assert st1.draft is st0.draft and st1.managers is st0.managers and st1.league is st0.league and p.state is st1
    assert st1.rostered_ids is st0.rostered_ids
    assert await p.poll_once() is None
    # status change alone (drafted flag) -> new state, complete
    fc.drafted = True
    st2 = await p.poll_once()
    assert st2 is not None and st2.draft.status == "complete" and st2.is_complete and st2.version == st1.version + 1
    assert st0.draft.status == "drafting" and st2.draft is not st1.draft and st2.my_slot == 3
    assert fc.calls.count("settings") == 1


class PrepopulatedClient(FakeEspnClient):
    """ESPN serving the whole board (150 entries) and filling them in place: unfilled entries have playerId 0."""

    def __init__(self, filled: int = 0):
        super().__init__(n=150)
        self.filled = filled
        self.edits: dict[int, int] = {}                 # overallPickNumber -> playerId overriding the fixture
        self.drop_detail_times = 0

    async def get_draft_detail(self, league_id, season):
        d = await super().get_draft_detail(league_id, season)
        if self.drop_detail_times > 0:
            self.drop_detail_times -= 1
            d.pop("draftDetail")
            return d
        for i, p in enumerate(d["draftDetail"]["picks"]):
            if i >= self.filled:
                p["playerId"] = 0
            if p["overallPickNumber"] in self.edits:
                p["playerId"] = self.edits[p["overallPickNumber"]]
        d["draftDetail"].update({"drafted": self.filled >= 150, "inProgress": 0 < self.filled < 150})
        return d


async def test_in_place_fills_and_corrections_produce_new_states():
    """A pre-populated board keeps its length and its last entry: every fill / edit must still be seen."""
    fc = PrepopulatedClient(filled=0)
    p = make_poller(fc, swid=SWID_TEAM_1)
    st = await p.bootstrap()
    assert len(st.picks) == 0 and st.draft.status == "pre_draft" and st.my_slot == 3
    assert await p.poll_once() is None
    for n in (1, 2, 3, 17):
        fc.filled = n
        st = await p.poll_once()
        assert st is not None and len(st.picks) == n and st.version == p.state.version, n
    assert st.draft.status == "drafting" and st.is_my_turn and st.next_pick_no == 18
    assert await p.poll_once() is None
    # the commissioner corrects pick 6 (same count, same last entry)
    fc.edits[6] = 15825
    st = await p.poll_once()
    assert st is not None and next(pk for pk in st.picks if pk.pick_no == 6).player_id == "espn:15825"
    assert await p.poll_once() is None
    fc.filled = 150
    st = await p.poll_once()
    assert st is not None and st.is_complete and len(st.picks) == 150


async def test_pick_trade_on_the_board_reparses_the_state():
    """A change of owner of a not-yet-made pick is pick-order math: the state is rebuilt with the trade."""
    fc = PrepopulatedClient(filled=17)
    p = make_poller(fc, swid=SWID_TEAM_1)
    st0 = await p.bootstrap()
    assert st0.is_my_turn and st0.draft.traded_picks == {}
    entry = next(e for e in fc.draft["draftDetail"]["picks"] if e["overallPickNumber"] == 18)
    entry["teamId"], entry["owningTeamIds"] = 2, [2]
    st1 = await p.poll_once()
    assert st1 is not None and st1.draft.traded_picks == {(2, 1): 2} and not st1.is_my_turn
    assert st1.on_the_clock_roster == 2 and st1.my_future_picks()[:2] == [23, 38] and st1.version == 1
    assert await p.poll_once() is None


async def test_a_poll_without_draft_detail_is_an_error_not_an_empty_board():
    fc = PrepopulatedClient(filled=17)
    p = make_poller(fc, swid=SWID_TEAM_1)
    st0 = await p.bootstrap()
    fc.drop_detail_times = 1
    with pytest.raises(EspnAPIError) as ei:
        await p.poll_once()
    assert "draftDetail" in str(ei.value) and p.state is st0 and len(p.state.picks) == 17
    assert await p.poll_once() is None                              # the next good poll: nothing changed
    # through run(): counted as an error, backed off, recovered without a spurious update
    fc.drop_detail_times = 2
    p._sleep = lambda s, e: asyncio.sleep(0)  # type: ignore[method-assign]
    updates, errors, stop = [], [], asyncio.Event()

    def on_update(state):
        updates.append(state)
        if len(updates) == 2:
            stop.set()

    def on_error(e):
        errors.append(e)
        if len(errors) == 2:
            fc.filled = 18

    await p.run(on_update, stop, on_error=on_error)
    assert len(errors) == 2 and all("draftDetail" in str(e) for e in errors)
    assert [len(s.picks) for s in updates] == [17, 18] and p.last_error is None


async def test_board_that_empties_is_rendered_but_logged(caplog):
    """Picks vanishing while the draft flags say nothing changed is rendered as returned, but flagged."""
    fc = FakeEspnClient(n=17)
    p = make_poller(fc, swid=SWID_TEAM_1)
    await p.bootstrap()

    async def empty_board(league_id, season):
        d = copy.deepcopy(fc.draft)
        d["draftDetail"].update({"picks": [], "drafted": False, "inProgress": True})
        return d

    fc.get_draft_detail = empty_board  # type: ignore[method-assign]
    with caplog.at_level("WARNING", logger="draftadvisor.espn.poller"):
        st = await p.poll_once()
    assert st is not None and st.picks == [] and st.draft.status == "drafting" and st.next_pick_no == 1
    assert "17 picks to 0" in caplog.text


async def test_timer_and_pick_order_changes_are_followed():
    fc = FakeEspnClient(n=17)
    p = make_poller(fc, swid=SWID_TEAM_1)
    st0 = await p.bootstrap()
    assert st0.draft.pick_timer == 90
    fc.draft["settings"]["draftSettings"]["timePerSelection"] = 30
    st1 = await p.poll_once()
    assert st1 is not None and st1.draft.pick_timer == 30 and st1.my_slot == 3 and st1.version == 1
    assert await p.poll_once() is None
    fc.draft["settings"]["draftSettings"]["pickOrder"] = list(reversed(PICK_ORDER))
    st2 = await p.poll_once()
    assert st2 is not None and st2.my_slot == 8 and st2.draft.slot_to_roster_id[1] == 11 and st2.version == 2
    assert st2.managers["1"].slot == 8 and len(st2.picks) == 17
    # rounds change (bench slot added) -> total picks
    fc.draft["settings"]["rosterSettings"]["lineupSlotCounts"]["20"] = 7
    st3 = await p.poll_once()
    assert st3 is not None and st3.draft.rounds == 16 and st3.draft.total_picks == 160
    assert fc.calls.count("settings") == 1


async def test_order_appearing_after_pre_draft():
    fc = FakeEspnClient(n=0)
    fc.draft["settings"]["draftSettings"]["pickOrder"] = []
    p = make_poller(fc, swid=SWID_TEAM_1)
    st0 = await p.bootstrap()
    assert st0.my_slot is None and st0.draft.status == "pre_draft" and p.interval() == 10.0
    fc.draft["settings"]["draftSettings"]["pickOrder"] = PICK_ORDER
    fc.n = 2
    st1 = await p.poll_once()
    assert st1 is not None and st1.my_slot == 3 and st1.draft.status == "drafting" and st1.is_my_turn


async def test_interval_adapts():
    fc = FakeEspnClient(n=15)
    p = make_poller(fc, swid=SWID_TEAM_1, poll_seconds=2.0)
    await p.bootstrap()
    assert p.state.picks_until_my_turn == 2 and p.interval() == 2.0
    fc.n = 16
    await p.poll_once()
    assert p.state.picks_until_my_turn == 1 and p.interval() == 1.0
    fc.n = 17
    await p.poll_once()
    assert p.state.is_my_turn and p.interval() == 1.0
    fc.n = 18
    await p.poll_once()
    assert p.interval() == 2.0
    p2 = EspnDraftPoller(FakeEspnClient(), LEAGUE_ID, SEASON, settings=Settings(poll_seconds=3.5))
    await p2.bootstrap()
    assert p2.interval() == 3.5 and p2.state.my_slot is None


async def test_run_emits_on_change_with_cadence_and_stops_when_complete():
    fc = FakeEspnClient(n=17)
    p = make_poller(fc, swid=SWID_TEAM_1, poll_seconds=2.0)
    sleeps: list[float] = []
    schedule = iter([19, 19, 20, 21, 22, 23])

    async def fake_sleep(seconds, stop_evt):
        sleeps.append(seconds)
        fc.n = next(schedule, fc.n)
        return bool(stop_evt and stop_evt.is_set())

    p._sleep = fake_sleep  # type: ignore[method-assign]
    seen: list[DraftState] = []
    stop = asyncio.Event()

    def on_update(state):
        seen.append(state)
        if len(seen) >= 4:
            stop.set()

    final = await p.run(on_update, stop)
    assert final is seen[-1] is p.state
    assert [len(s.picks) for s in seen] == [17, 19, 20, 21] and [s.version for s in seen] == [0, 1, 2, 3]
    assert sleeps[0] == 1.0                                          # my turn at bootstrap
    assert p.last_error is None
    # completes on its own
    fc2 = FakeEspnClient(n=17)
    p2 = make_poller(fc2, swid=SWID_TEAM_1)
    p2._sleep = lambda s, e: asyncio.sleep(0)  # type: ignore[method-assign]
    versions = []

    async def on_update2(state):
        versions.append(state.version)
        fc2.drafted = True

    final2 = await p2.run(on_update2)
    assert final2.is_complete and versions == [0, 1]


async def test_run_survives_errors_and_backs_off():
    fc = FakeEspnClient(n=17)
    p = make_poller(fc, swid=SWID_TEAM_1, poll_seconds=2.0)
    sleeps: list[float] = []
    errors: list[Exception] = []
    stop = asyncio.Event()

    async def fake_sleep(seconds, stop_evt):
        sleeps.append(seconds)
        if len(sleeps) == 1:
            fc.fail_draft_times = 3
        return stop_evt.is_set()

    p._sleep = fake_sleep  # type: ignore[method-assign]
    updates = []

    def on_update(state):
        updates.append(state)
        if len(updates) == 2:
            stop.set()

    def on_error(e):
        errors.append(e)
        if len(errors) == 3:
            fc.n = 18

    await p.run(on_update, stop, on_error=on_error)
    assert len(errors) == 3 and all(isinstance(e, EspnAPIError) for e in errors)
    assert len(updates) == 2 and len(updates[1].picks) == 18
    assert p.last_error is None and p.consecutive_errors == 0
    assert sleeps[:4] == [1.0, 4.0, 8.0, 16.0]
    p.consecutive_errors = 100
    assert p._backoff() == 30.0


async def test_run_raises_for_unknown_league_or_missing_cookies():
    for fc in (FakeEspnClient(missing=True), FakeEspnClient(private=True)):
        p = make_poller(fc)
        p._sleep = lambda s, e: asyncio.sleep(0)  # type: ignore[method-assign]
        errors = []
        with pytest.raises((EspnNotFound, EspnAccessDenied)):
            await p.run(lambda st: None, on_error=errors.append)
        assert len(errors) == 1 and fc.calls == ["settings"] and p.last_error


async def test_run_retries_bootstrap_failure_and_survives_hook_errors():
    fc = FakeEspnClient(n=17)
    fc.fail_draft_times = 1
    p = make_poller(fc, swid=SWID_TEAM_1)
    p._sleep = lambda s, e: asyncio.sleep(0)  # type: ignore[method-assign]
    stop = asyncio.Event()
    calls = []

    def on_update(state):
        calls.append(state.version)
        if len(calls) == 1:
            fc.n = 18
            raise RuntimeError("ui bug")
        stop.set()

    await p.run(on_update, stop)
    assert calls == [0, 1] and fc.calls.count("draft") >= 3


async def test_real_sleep_respects_stop_event():
    p = make_poller(FakeEspnClient())
    stop = asyncio.Event()
    asyncio.get_running_loop().call_later(0.01, stop.set)
    assert await p._sleep(5.0, stop) is True
    assert await p._sleep(0.0, None) is False


async def test_poller_against_the_stub_server():
    with EspnStub(draft="live", picks_visible=17) as s:
        async with EspnClient(base_url=s.base_url, retries=1, backoff_base=0.0) as c:
            kona = await c.get_players(LEAGUE_ID, SEASON)
            p = EspnDraftPoller(c, LEAGUE_ID, SEASON, id_map=EspnIdMap.from_players(universe()), names=espn_names(kona),
                                swid=SWID_TEAM_1, poll_seconds=1.0)
            st = await p.bootstrap()
            assert len(st.picks) == 17 and st.is_my_turn and st.draft.status == "drafting" and st.draft.pick_timer == 90
            assert await p.poll_once() is None
            s.picks_visible = 20
            st2 = await p.poll_once()
            assert st2 is not None and len(st2.picks) == 20 and st2.next_pick_no == 21
            s.timer_override = 45
            st3 = await p.poll_once()
            assert st3 is not None and st3.draft.pick_timer == 45 and st3.my_slot == 3
            s.picks_visible = None
            st4 = await p.poll_once()
            assert st4 is not None and st4.is_complete and len(st4.picks) == 150
            assert p.poll_count == 5 and p.last_error is None


def test_package_imports_without_pandas_or_sklearn():
    code = ("import sys; sys.modules['pandas'] = None; sys.modules['sklearn'] = None\n"
            "import draftadvisor.espn, draftadvisor.espn.capture, draftadvisor.espn.poller\n"
            "from draftadvisor.espn import EspnClient, EspnDraftPoller, capture_espn_league, state_from_espn\n"
            "print('ok')")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(ROOT), timeout=120)
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr
