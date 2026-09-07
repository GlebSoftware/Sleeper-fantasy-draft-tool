"""Shared pytest fixtures. Everything runs offline against tests/fixtures."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Point DRAFTADVISOR_HOME at a temp dir so tests never touch real caches."""
    monkeypatch.setenv("DRAFTADVISOR_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
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
