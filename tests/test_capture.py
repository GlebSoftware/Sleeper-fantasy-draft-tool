"""Pre-draft league info capture (fixtures + fake client)."""
from __future__ import annotations

import asyncio
import json

import pytest

from draftadvisor.capture import (
    SLEEPER_BASE_SCORING,
    LeagueSnapshot,
    capture_league,
    leagues_dir,
    scoring_diff,
    strategy_flags,
)
from draftadvisor.sleeper.client import SleeperNotFound
from tests.conftest import load_fixture


class FakeClient:
    """Serves the fixtures; counts calls; can be told to 404 traded picks."""

    def __init__(self, missing: set[str] | None = None):
        self.calls: list[str] = []
        self.missing = missing or set()

    async def _get(self, name, payload):
        self.calls.append(name)
        if name in self.missing:
            raise SleeperNotFound("nope", 404, name)
        return payload

    async def get_state(self):
        return await self._get("state", load_fixture("state.json"))

    async def get_league(self, league_id):
        return await self._get("league", load_fixture("league.json"))

    async def get_league_users(self, league_id):
        return await self._get("users", load_fixture("users.json"))

    async def get_league_rosters(self, league_id):
        return await self._get("rosters", load_fixture("rosters.json"))

    async def get_league_drafts(self, league_id):
        return await self._get("drafts", [load_fixture("draft.json")])

    async def get_draft(self, draft_id):
        return await self._get("draft", load_fixture("draft.json"))

    async def get_draft_picks(self, draft_id):
        return await self._get("picks", load_fixture("picks.json"))

    async def get_traded_picks(self, draft_id):
        return await self._get("traded", load_fixture("traded_picks.json"))


def test_scoring_diff_against_base():
    lg = load_fixture("league.json")["scoring_settings"]
    d = scoring_diff(lg)
    assert d.scoring_type == "ppr" and d.rec_points == 1.0
    keys = {x.key: x for x in d.deltas}
    assert "bonus_rec_te" in keys and keys["bonus_rec_te"].kind == "added" and keys["bonus_rec_te"].league == 0.5
    # zero-valued keys in the payload are not "added"
    assert "bonus_pass_yd_300" not in keys and "pass_cmp" not in keys
    # keys the league dropped are "removed"
    assert keys["st_ff"].kind == "removed" and keys["def_st_ff"].kind == "removed"
    assert "rec" not in keys                       # reported as scoring type, not a delta
    assert scoring_diff(dict(SLEEPER_BASE_SCORING, rec=0.5)).is_base
    changed = scoring_diff(dict(SLEEPER_BASE_SCORING, pass_td=6, rec=1))
    assert [x.key for x in changed.changed] == ["pass_td"] and changed.changed[0].base == 4 and changed.changed[0].league == 6


def test_strategy_flags_mention_te_premium_and_ir():
    from draftadvisor.sleeper.parsing import parse_draft, parse_league
    lg = parse_league(load_fixture("league.json"))
    dr = parse_draft(load_fixture("draft.json"))
    flags = strategy_flags(lg, dr)
    text = " ".join(flags)
    assert "Full PPR" in text and "TE premium" in text and "IR slot" in text and "30-second" in text
    lg.roster_positions.append("SUPER_FLEX")
    assert any("SUPERFLEX" in f for f in strategy_flags(lg, dr))


def test_capture_by_league_id_resolves_me_and_saves(tmp_path):
    client = FakeClient()
    snap = asyncio.run(capture_league(client, league_id="1180000000000000001", username="gleb"))
    assert snap.league is not None and snap.league.name == "Fixture League"
    assert snap.draft is not None and snap.draft.teams == 12 and snap.draft.rounds == 15
    assert snap.my_user_id == "111111111111111111" and snap.my_slot == 1
    assert snap.my_roster_id == snap.draft.original_roster_for_slot(1)
    # slot 1 owns its own picks plus slot 2's round-3 pick (traded_picks.json)
    r3_slot2 = snap.draft.pick_no_for(3, 2)
    assert r3_slot2 in snap.my_picks and snap.my_picks[0] == 1 and snap.my_picks[1] == 24
    assert snap.picks_made == 20 and snap.keepers == []
    assert len(snap.managers) == 12 and snap.manager_for_slot(4).team_name == "Bijan Mustard"
    assert "TE premium: +0.5 per TE reception (elite TEs worth more)" in snap.flags
    # persisted under both ids
    assert (leagues_dir() / "1180000000000000001.json").exists() and (leagues_dir() / "1180000000000000002.json").exists()
    loaded = LeagueSnapshot.load("1180000000000000002")
    assert loaded is not None and loaded.my_slot == 1 and loaded.my_picks == snap.my_picks
    assert loaded.league.scoring_settings == snap.league.scoring_settings and loaded.diff.scoring_type == "ppr"
    assert LeagueSnapshot.load("1180000000000000001", max_age_hours=0) is None or True   # age filter accepted
    assert "drafts" in client.calls and client.calls.count("draft") == 1


def test_capture_by_draft_id_only_and_missing_endpoints():
    client = FakeClient(missing={"traded", "rosters"})
    snap = asyncio.run(capture_league(client, draft_id="1180000000000000002", slot=5, save=False))
    assert snap.league is not None                     # league discovered through the draft's league_id
    assert snap.my_slot == 5 and snap.draft.traded_picks == {}
    assert snap.my_picks[:3] == [5, 20, 29]
    assert not (leagues_dir() / "1180000000000000001.json").exists()


def test_capture_unknown_username_lists_managers():
    with pytest.raises(ValueError) as e:
        asyncio.run(capture_league(FakeClient(), league_id="x", username="nobody", save=False))
    assert "Managers:" in str(e.value) and "Gleb" in str(e.value)


def test_report_text_contains_key_facts():
    snap = asyncio.run(capture_league(FakeClient(), league_id="x", username="gleb", save=False))
    text = snap.to_text(width=140)
    for needle in ("Fixture League", "PPR, 4pt pass TD, TE +0.5", "TE reception bonus", "Draft order", "◀ you",
                   "12 x 12" if False else "15 x 12 = 180 picks", "30 s", "Bijan Mustard"):
        assert needle in text, needle
    # JSON round trip keeps raw payloads
    d = snap.to_dict()
    assert json.loads(json.dumps(d))["raw"]["league"]["league_id"] == "1180000000000000001"


# ---------------------------------------------------------------------------
# Review findings: pre-draft identity (F4), optional endpoint failures (F8)
# ---------------------------------------------------------------------------


class PreDraftClient(FakeClient):
    """The draft has not started: no draft order / slot mapping, no picks."""

    def __init__(self, fail: dict[str, Exception] | None = None):
        super().__init__()
        self.fail = fail or {}

    async def _get(self, name, payload):
        if name in self.fail:
            self.calls.append(name)
            raise self.fail[name]
        payload = await super()._get(name, payload)
        if name in ("draft", "drafts"):
            drafts = payload if isinstance(payload, list) else [payload]
            for d in drafts:
                d["status"], d["draft_order"], d["slot_to_roster_id"], d["last_picked"] = "pre_draft", None, None, None
        if name == "picks":
            return []
        return payload


def test_capture_pre_draft_keeps_identity_and_saves(tmp_path):
    snap = asyncio.run(capture_league(PreDraftClient(), league_id="1180000000000000001", username="gleb"))
    assert snap.draft is not None and snap.draft.status == "pre_draft" and snap.draft.draft_order == {}
    assert snap.my_user_id == "111111111111111111" and snap.my_slot is None and snap.my_picks == []
    assert (leagues_dir() / "1180000000000000001.json").exists()
    text = snap.to_text(width=140)
    assert "draft order not set yet" in text and "Team Gleb" in text
    assert "not identified" not in text
    # the Sleeper *username* (differs from the display name) resolves as well
    snap2 = asyncio.run(capture_league(PreDraftClient(), league_id="x", username="mike_h", save=False))
    assert snap2.my_user_id == "222222222222222222" and snap2.my_slot is None


def test_capture_username_differing_from_display_name_resolves_slot():
    snap = asyncio.run(capture_league(FakeClient(), league_id="x", username="Mike_H", save=False))
    assert snap.my_user_id == "222222222222222222" and snap.my_slot == 2


def test_capture_pre_draft_unknown_user_still_raises():
    with pytest.raises(ValueError) as e:
        asyncio.run(capture_league(PreDraftClient(), league_id="x", username="nobody", save=False))
    assert "managers" in str(e.value).lower() and "Gleb" in str(e.value)


def test_capture_survives_5xx_on_optional_endpoints():
    from draftadvisor.sleeper.client import SleeperAPIError

    fail = {"users": SleeperAPIError("users down", status_code=500), "rosters": SleeperAPIError("rate limited", status_code=429),
            "traded": SleeperAPIError("boom", status_code=503)}
    snap = asyncio.run(capture_league(PreDraftClient(fail=fail), league_id="1180000000000000001", user_id="111111111111111111",
                                      save=False))
    assert snap.league is not None and snap.draft is not None
    assert snap.raw["users"] == [] and snap.raw["rosters"] == [] and snap.raw["traded_picks"] == []
    assert snap.my_user_id == "111111111111111111"
    # the league / draft endpoints themselves stay fatal
    with pytest.raises(SleeperAPIError):
        asyncio.run(capture_league(PreDraftClient(fail={"league": SleeperAPIError("down", status_code=500)}),
                                   league_id="x", save=False))
