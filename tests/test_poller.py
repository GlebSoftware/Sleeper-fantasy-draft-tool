"""Tests for draftadvisor.sleeper.poller with a fake client whose pick list grows over time."""
from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path

import pytest

from draftadvisor.config import Settings
from draftadvisor.models import DraftState
from draftadvisor.sleeper.client import SleeperAPIError, SleeperNotFound
from draftadvisor.sleeper.poller import DraftPoller


_FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str):
    with open(_FIXTURES / name, encoding="utf-8") as fh:
        return json.load(fh)

GLEB = "111111111111111111"


class FakeClient:
    """Mimics SleeperClient for one draft. ``n`` = number of picks currently visible."""

    def __init__(self, n: int = 20, league: bool = True):
        self.draft = copy.deepcopy(load_fixture("draft.json"))
        self.league = load_fixture("league.json") if league else None
        self.users = load_fixture("users.json")
        self.rosters = load_fixture("rosters.json")
        self.traded = load_fixture("traded_picks.json")
        self.all_picks = load_fixture("picks.json")
        self.n = n
        self.calls: list[str] = []
        self.fail_picks_times = 0
        self.fail_users = False

    def _extra_pick(self, pick_no: int) -> dict:
        p = copy.deepcopy(self.all_picks[-1])
        p["pick_no"] = pick_no
        p["round"] = (pick_no - 1) // 12 + 1
        p["player_id"] = f"synthetic-{pick_no}"
        return p

    async def get_draft(self, draft_id):
        self.calls.append("draft")
        return copy.deepcopy(self.draft)

    async def get_draft_picks(self, draft_id):
        self.calls.append("picks")
        if self.fail_picks_times > 0:
            self.fail_picks_times -= 1
            raise SleeperAPIError("boom", status_code=503)
        picks = list(self.all_picks[: self.n])
        for k in range(len(self.all_picks) + 1, self.n + 1):
            picks.append(self._extra_pick(k))
        return copy.deepcopy(picks)

    async def get_traded_picks(self, draft_id):
        self.calls.append("traded")
        return copy.deepcopy(self.traded)

    async def get_league(self, league_id):
        self.calls.append("league")
        if self.league is None:
            raise SleeperNotFound("no league", status_code=404)
        return copy.deepcopy(self.league)

    async def get_league_users(self, league_id):
        self.calls.append("users")
        if self.fail_users:
            raise SleeperAPIError("users down", status_code=500)
        return copy.deepcopy(self.users)

    async def get_league_rosters(self, league_id):
        self.calls.append("rosters")
        return copy.deepcopy(self.rosters)


def make_poller(client: FakeClient, poll_seconds: float = 2.0, **kw) -> DraftPoller:
    settings = Settings(poll_seconds=poll_seconds)
    return DraftPoller(client, "1180000000000000002", settings=settings, **kw)


# ---------------------------------------------------------------------------


async def test_bootstrap_builds_full_state():
    fc = FakeClient()
    p = make_poller(fc, username="gleb")
    st = await p.bootstrap()
    assert isinstance(st, DraftState)
    assert p.state is st and p.poll_count == 1 and p.last_poll_at is not None
    assert st.my_user_id == GLEB and st.my_slot == 1
    assert st.league is not None and len(st.managers) == 12
    assert len(st.picks) == 20 and st.next_pick_no == 21
    assert st.draft.traded_picks == {(3, 5): 4}
    assert st.my_future_picks()[:3] == [24, 25, 26]
    assert p.league_id == "1180000000000000001"
    assert set(fc.calls) == {"draft", "league", "users", "rosters", "traded", "picks"}


async def test_bootstrap_degrades_without_league_or_users():
    fc = FakeClient(league=False)
    fc.fail_users = True
    p = make_poller(fc, slot=2)
    st = await p.bootstrap()
    assert st.league is None
    assert st.my_slot == 2 and st.my_user_id == "222222222222222222"
    assert len(st.managers) == 12  # placeholders from draft_order
    assert st.managers[GLEB].display_name == "Team 1"


async def test_poll_once_returns_new_state_only_on_change():
    fc = FakeClient()
    p = make_poller(fc, username="gleb")
    st0 = await p.bootstrap()
    assert await p.poll_once() is None
    assert await p.poll_once() is None
    assert p.poll_count == 3
    fc.n = 22
    st1 = await p.poll_once()
    assert st1 is not None and st1 is not st0
    assert len(st1.picks) == 22 and st1.version == st0.version + 1
    assert st1.draft is st0.draft and st1.managers is st0.managers
    assert p.state is st1
    assert await p.poll_once() is None
    # status change alone triggers a new state
    fc.draft["status"] = "paused"
    st2 = await p.poll_once()
    assert st2 is not None and st2.draft.status == "paused" and st2.version == st1.version + 1
    assert st0.draft.status == "drafting"  # old state not mutated


async def test_poll_once_reparses_when_draft_order_appears():
    fc = FakeClient(n=0)
    fc.draft["status"] = "pre_draft"
    order, s2r = fc.draft["draft_order"], fc.draft["slot_to_roster_id"]
    fc.draft["draft_order"] = None
    fc.draft["slot_to_roster_id"] = None
    p = make_poller(fc, username="gleb")
    st0 = await p.bootstrap()
    assert st0.my_slot is None and st0.draft.status == "pre_draft"
    assert p.interval() == 10.0
    fc.draft["draft_order"], fc.draft["slot_to_roster_id"] = order, s2r
    fc.draft["status"] = "drafting"
    st1 = await p.poll_once()
    assert st1 is not None and st1.my_slot == 1 and st1.draft.status == "drafting"
    assert st1.draft.slot_to_roster_id[1] == 4 and st1.version == 1
    assert st1.is_my_turn


async def test_interval_adapts_to_my_turn():
    fc = FakeClient()
    p = make_poller(fc, username="gleb", poll_seconds=2.0)
    await p.bootstrap()
    assert p.state.picks_until_my_turn == 3
    assert p.interval() == 2.0
    fc.n = 22  # next pick #23, mine is #24 -> 1 away
    await p.poll_once()
    assert p.state.picks_until_my_turn == 1 and p.interval() == 1.0
    fc.n = 23  # my turn
    await p.poll_once()
    assert p.state.is_my_turn and p.interval() == 1.0
    fc.n = 26  # all three of my picks done; next mine is #48
    await p.poll_once()
    assert p.interval() == 2.0
    # spectator mode: never "my turn"
    p2 = make_poller(FakeClient(), poll_seconds=3.0)
    await p2.bootstrap()
    assert p2.interval() == 3.0


async def test_run_emits_on_bootstrap_and_change_only_with_cadence():
    fc = FakeClient()
    p = make_poller(fc, username="gleb", poll_seconds=2.0)
    sleeps: list[float] = []
    stop = asyncio.Event()
    # after each (virtual) sleep advance the board: 20 -> 22 -> 22 -> 23 -> 24 -> ...
    schedule = iter([22, 22, 23, 24, 25, 26, 27])

    async def fake_sleep(seconds, stop_evt):
        sleeps.append(seconds)
        fc.n = next(schedule, fc.n)
        return stop_evt.is_set()

    p._sleep = fake_sleep  # type: ignore[method-assign]
    seen: list[DraftState] = []

    def on_update(state):
        seen.append(state)
        if len(seen) >= 5:
            stop.set()

    final = await p.run(on_update, stop)
    assert final is seen[-1] is p.state
    assert [len(s.picks) for s in seen] == [20, 22, 23, 24, 25]
    assert [s.version for s in seen] == [0, 1, 2, 3, 4]
    # cadence: 3 away -> 2 s; after 22 picks 1 away -> 1 s; my turn -> 1 s; after #24 my next is #25 -> 1 s
    assert sleeps[:5] == [2.0, 1.0, 1.0, 1.0, 1.0]
    assert p.last_error is None


async def test_run_supports_async_hook_and_returns_on_complete():
    fc = FakeClient()
    p = make_poller(fc, username="gleb")
    p._sleep = lambda s, e: asyncio.sleep(0)  # type: ignore[method-assign]
    seen = []

    async def on_update(state):
        await asyncio.sleep(0)
        seen.append(state.version)
        fc.draft["status"] = "complete"

    final = await p.run(on_update)
    assert final.is_complete and final.draft.status == "complete"
    assert seen == [0, 1]


async def test_run_survives_errors_and_backs_off():
    fc = FakeClient()
    p = make_poller(fc, username="gleb", poll_seconds=2.0)
    sleeps: list[float] = []
    errors: list[Exception] = []
    stop = asyncio.Event()

    async def fake_sleep(seconds, stop_evt):
        sleeps.append(seconds)
        if len(sleeps) == 1:
            fc.fail_picks_times = 3
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
            fc.n = 21

    await p.run(on_update, stop, on_error=on_error)
    assert len(errors) == 3 and all(isinstance(e, SleeperAPIError) for e in errors)
    assert len(updates) == 2 and len(updates[1].picks) == 21
    assert p.last_error is None and p.consecutive_errors == 0
    # 2 s normal, then 4, 8, 16 s backoff (capped at 30), then back to normal
    assert sleeps[:4] == [2.0, 4.0, 8.0, 16.0]


async def test_backoff_is_capped_at_30s():
    p = make_poller(FakeClient(), poll_seconds=2.0)
    p.consecutive_errors = 10
    assert p._backoff() == 30.0


async def test_run_retries_bootstrap_failure():
    fc = FakeClient()
    fc.fail_picks_times = 1
    p = make_poller(fc, username="gleb")
    p._sleep = lambda s, e: asyncio.sleep(0)  # type: ignore[method-assign]
    stop = asyncio.Event()
    seen = []

    def on_update(state):
        seen.append(state)
        stop.set()

    await p.run(on_update, stop)
    assert len(seen) == 1 and len(seen[0].picks) == 20


async def test_hook_exception_does_not_kill_loop():
    fc = FakeClient()
    p = make_poller(fc, username="gleb")
    p._sleep = lambda s, e: asyncio.sleep(0)  # type: ignore[method-assign]
    stop = asyncio.Event()
    calls = []

    def on_update(state):
        calls.append(state.version)
        if len(calls) == 1:
            fc.n = 21
            raise RuntimeError("ui bug")
        stop.set()

    await p.run(on_update, stop)
    assert calls == [0, 1]


async def test_real_sleep_respects_stop_event():
    p = make_poller(FakeClient())
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop.call_later(0.01, stop.set)
    assert await p._sleep(5.0, stop) is True
    assert await p._sleep(0.0, None) is False


async def test_settings_supply_identity():
    fc = FakeClient()
    settings = Settings(poll_seconds=1.5, username="omar")
    p = DraftPoller(fc, "1180000000000000002", settings=settings)
    st = await p.bootstrap()
    assert st.my_slot == 9 and st.picks_until_my_turn == 12
    assert p.interval() == 1.5


@pytest.mark.parametrize("kw", [{"user_id": "999999999999999999"}, {"slot": 9}])
async def test_identity_by_user_id_or_slot(kw):
    p = make_poller(FakeClient(), **kw)
    st = await p.bootstrap()
    assert st.my_slot == 9 and st.my_user_id == "999999999999999999"


# ---------------------------------------------------------------------------
# Review findings: backoff overflow (F7), settings / traded-pick refresh (F9), unknown draft, rosters (F3)
# ---------------------------------------------------------------------------


async def test_backoff_never_overflows_and_has_a_floor():
    p = make_poller(FakeClient(), poll_seconds=2.0)
    p.consecutive_errors = 1030
    assert p._backoff() == 30.0                     # no OverflowError, capped
    p.consecutive_errors = 1
    assert p._backoff() == 4.0
    p0 = make_poller(FakeClient(), poll_seconds=0.0)
    p0.consecutive_errors = 1
    assert p0._backoff() == 2.0                     # poll 0 still backs off (1 s floor x 2)


async def test_settings_change_reparses_draft():
    """F9: a commissioner changing rounds / pick timer before the start must reach the parsed draft."""
    fc = FakeClient()
    p = make_poller(fc, username="gleb")
    st0 = await p.bootstrap()
    assert st0.draft.rounds == 15 and st0.draft.pick_timer == 30
    fc.draft["settings"]["rounds"] = 16
    fc.draft["settings"]["pick_timer"] = 60
    st1 = await p.poll_once()
    assert st1 is not None and st1.draft.rounds == 16 and st1.draft.pick_timer == 60
    assert st1.draft.total_picks == 192 and st1.version == st0.version + 1
    assert st1.my_slot == 1 and st1.rostered_ids == set()
    assert await p.poll_once() is None


async def test_traded_picks_refreshed_periodically_while_drafting():
    """F9: an in-draft pick trade changes my future picks without any other change on the board."""
    fc = FakeClient()
    p = make_poller(fc, username="gleb")
    p.traded_refresh_seconds = 30.0
    st0 = await p.bootstrap()
    assert fc.calls.count("traded") == 1
    assert st0.my_future_picks()[:3] == [24, 25, 26]
    # not due yet: no traded-picks request, no new state
    assert await p.poll_once() is None and fc.calls.count("traded") == 1
    # a new trade: slot 1 (roster 4) gives its round-5 pick (#49) to roster 5 (slot 2)
    fc.traded.append({"season": "2026", "round": 5, "roster_id": 4, "previous_owner_id": 4, "owner_id": 5})
    p._traded_at -= 31.0                             # make the refresh due
    st1 = await p.poll_once()
    assert fc.calls.count("traded") == 2
    assert st1 is not None and st1.version == st0.version + 1
    assert st1.draft.traded_picks == {(3, 5): 4, (5, 4): 5}
    assert 49 not in st1.my_future_picks() and st1.my_future_picks()[:3] == [24, 25, 26]
    # same payload again -> nothing new
    p._traded_at -= 31.0
    assert await p.poll_once() is None and fc.calls.count("traded") == 3
    # pre_draft: not polled for traded picks on a timer
    fc.draft["status"] = "pre_draft"
    await p.poll_once()
    p._traded_at -= 31.0
    await p.poll_once()
    assert fc.calls.count("traded") == 3


async def test_traded_picks_refetched_when_order_changes():
    fc = FakeClient(n=0)
    fc.draft["status"] = "pre_draft"
    order, s2r = fc.draft["draft_order"], fc.draft["slot_to_roster_id"]
    fc.draft["draft_order"] = fc.draft["slot_to_roster_id"] = None
    p = make_poller(fc, username="gleb")
    await p.bootstrap()
    n_traded = fc.calls.count("traded")
    fc.draft["draft_order"], fc.draft["slot_to_roster_id"], fc.draft["status"] = order, s2r, "drafting"
    st = await p.poll_once()
    assert st is not None and st.my_slot == 1 and fc.calls.count("traded") == n_traded + 1


async def test_run_raises_for_unknown_draft_instead_of_retrying():
    class Missing(FakeClient):
        async def get_draft(self, draft_id):
            self.calls.append("draft")
            raise SleeperNotFound("not found", status_code=404)

    fc = Missing()
    p = make_poller(fc, username="gleb")
    p._sleep = lambda s, e: asyncio.sleep(0)  # type: ignore[method-assign]
    errors = []
    with pytest.raises(SleeperNotFound):
        await p.run(lambda st: None, on_error=errors.append)
    assert len(errors) == 1 and fc.calls.count("draft") == 1
    assert "SleeperNotFound" in (p.last_error or "")


async def test_bootstrap_fills_rostered_ids_from_league_rosters():
    fc = FakeClient()
    fc.rosters[0]["players"] = ["4034"]
    fc.rosters[1]["reserve"] = ["6794"]
    fc.rosters[2]["taxi"] = [9509]
    p = make_poller(fc, username="gleb")
    st = await p.bootstrap()
    assert st.rostered_ids == {"4034", "6794", "9509"}
    fc.n = 21
    st1 = await p.poll_once()
    assert st1.rostered_ids == {"4034", "6794", "9509"}      # carried across incremental states
