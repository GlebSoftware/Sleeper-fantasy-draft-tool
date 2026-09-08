"""Shared pytest fixtures. Everything runs offline against tests/fixtures."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"

#: Every environment variable the runtime reads (config.Settings.from_env, the web server, the ESPN
#: client, the note store). Cleared for each test so a developer's real league, cookies, access code or
#: Blob token never leak into the suite - a test that needs one sets it itself (monkeypatch.setenv).
RUNTIME_ENV_VARS: tuple[str, ...] = (
    "ANTHROPIC_API_KEY", "DRAFTADVISOR_ACCESS_CODE", "DRAFTADVISOR_CHAT_MODEL", "DRAFTADVISOR_PLATFORM",
    "DRAFTADVISOR_ESPN_BASE", "DRAFTADVISOR_ESPN_FAN_BASE", "BLOB_READ_WRITE_TOKEN",
    "ESPN_LEAGUE_ID", "ESPN_TEAM_ID", "ESPN_S2", "ESPN_SWID",
    "SLEEPER_LEAGUE_ID", "SLEEPER_DRAFT_ID", "SLEEPER_USERNAME", "SLEEPER_USER_ID",
)


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Point DRAFTADVISOR_HOME at a temp dir and clear every runtime env var so tests never touch real
    caches, leagues or credentials (and never go online because of the developer's shell)."""
    monkeypatch.setenv("DRAFTADVISOR_HOME", str(tmp_path / "home"))
    for var in RUNTIME_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    yield


def load_fixture(name: str):
    with open(FIXTURES / name, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture
def league_json():
    return load_fixture("league.json")


@pytest.fixture
def draft_json():
    return load_fixture("draft.json")


@pytest.fixture
def picks_json():
    return load_fixture("picks.json")


@pytest.fixture
def users_json():
    return load_fixture("users.json")


@pytest.fixture
def rosters_json():
    return load_fixture("rosters.json")


@pytest.fixture
def players_json():
    return load_fixture("players_sample.json")


@pytest.fixture
def projections_json():
    return load_fixture("projections_sample.json")


@pytest.fixture
def traded_picks_json():
    return load_fixture("traded_picks.json")
