"""ESPN's in-season payload: the calendar, the standings, the schedule and per-week scores.

Every field read here is already in payloads the app fetches; it was being discarded. Two of them
replace guesses that were wrong: the playoffs start where ``matchupPeriodCount`` says, not at a
hardcoded week 15, and a week's points come back **already scored under the league's own rules**, so
a 6-point-passing-touchdown league needs no re-scoring on our side.

The fixture is synthetic in its numbers and faithful in its shape (see ``make_fixtures.py``). It
proves the parsing, not that ESPN publishes scores while games are in progress - that is unmeasured,
and nothing here claims otherwise.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from draftadvisor.espn.inseason import (LeagueCalendar, Matchup, league_calendar, matchups, opponent_for,
                                        points_to_date, team_records, weekly_player_points)

FIX = Path(__file__).resolve().parent / "fixtures" / "espn"


@pytest.fixture(scope="module")
def league() -> dict:
    return json.loads((FIX / "league_inseason.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- calendar

def test_the_playoffs_start_where_the_league_says_not_at_a_hardcoded_week(league):
    cal = league_calendar(league)
    assert cal.regular_season_weeks == 13 and cal.playoff_week_start == 14
    assert cal.playoff_teams == 6 and cal.seeding_rule == "TOTAL_POINTS_SCORED"
    assert cal.is_playoffs(14) and not cal.is_playoffs(13)
    assert cal.current_week == 4 and cal.weeks_remaining() == 10


def test_the_current_week_comes_from_the_live_scoring_period(league):
    """``scoringPeriodId`` is the field that turns over; ``currentMatchupPeriod`` can lag it."""
    stale = dict(league, scoringPeriodId=6)
    assert league_calendar(stale).current_week == 6
    # and it is never allowed past the end of the season
    assert league_calendar(dict(league, scoringPeriodId=99)).current_week == 17


def test_a_league_with_two_week_matchups_keeps_the_two_ids_apart():
    cal = league_calendar({"scoringPeriodId": 3, "settings": {"scheduleSettings": {
        "matchupPeriodCount": 7, "matchupPeriodLength": 2,
        "matchupPeriods": {"1": [1, 2], "2": [3, 4], "3": [5, 6]}}}})
    assert cal.matchup_period_length == 2
    assert cal.matchup_period_for(4) == 2 and cal.matchup_period_for(1) == 1
    assert cal.matchup_period_for(99) == 99, "an unknown week is passed through, not guessed"


def test_an_empty_payload_gives_defaults_and_never_raises():
    cal = league_calendar(None)
    assert isinstance(cal, LeagueCalendar) and cal.current_week == 1
    assert league_calendar({}).playoff_week_start > 0


# --------------------------------------------------------------------------- standings

def test_standings_and_espns_own_playoff_odds_are_read(league):
    recs = team_records(league)
    assert len(recs) == 10
    first = recs[0]
    assert first.wins == 3 and first.losses == 0 and first.games == 3
    assert first.win_pct == pytest.approx(1.0)
    assert first.points_for > 0 and first.playoff_seed == 1
    assert first.espn_playoff_pct == pytest.approx(95.0)
    assert all(r.name for r in recs)


def test_a_team_with_no_record_yet_is_zero_and_not_a_guess():
    recs = team_records({"teams": [{"id": 3, "abbrev": "AAA"}]})
    assert len(recs) == 1 and recs[0].games == 0 and recs[0].win_pct == 0.0
    assert recs[0].espn_playoff_pct is None, "no odds published means none, not 50%"


# --------------------------------------------------------------------------- schedule

def test_the_schedule_gives_this_weeks_opponent(league):
    week4 = matchups(league, 4)
    assert week4 and all(m.week == 4 for m in week4)
    assert all(not m.played for m in week4), "week 4 has not been played"
    ids = {m.home_team_id for m in week4} | {m.away_team_id for m in week4}
    assert len(ids) == 10 and None not in ids
    assert opponent_for(week4, 1, 4) is not None
    opp = opponent_for(week4, 1, 4)
    assert opponent_for(week4, opp, 4) == 1, "the pairing has to be symmetric"


def test_played_weeks_carry_scores_and_a_winner(league):
    week1 = matchups(league, 1)
    assert week1 and all(m.played for m in week1)
    assert all(m.home_points > 0 and m.away_points > 0 for m in week1)
    for m in week1:
        expected = "HOME" if m.home_points > m.away_points else ("AWAY" if m.away_points > m.home_points else "TIE")
        assert m.winner == expected


def test_an_unmatched_team_has_no_opponent(league):
    games = [Matchup(week=4, home_team_id=1, away_team_id=2)]
    assert opponent_for(games, 3, 4) is None            # bye
    assert opponent_for(games, 1, 5) is None            # wrong week
    assert opponent_for(games, None, 4) is None
    assert matchups({"schedule": []}) == [] and matchups(None) == []


# --------------------------------------------------------------------------- weekly points

def test_weekly_points_are_taken_as_espn_scored_them(league):
    weekly = weekly_player_points(league)
    assert len(weekly) == 90                            # 10 teams x 9 roster entries
    any_player = next(iter(weekly.values()))
    assert set(any_player) == {1, 2, 3, 4}
    assert set(any_player[1]) == {"actual", "projected"}
    assert "actual" not in any_player[4], "week 4 has not been played; only its projection exists"
    assert any_player[4]["projected"] > 0


def test_a_week_not_played_is_not_a_zero(league):
    """Counting an unplayed week as zero is how a healthy player in week 2 looks like a bust."""
    weekly = weekly_player_points(league)
    pid = next(iter(weekly))
    total, weeks = points_to_date(weekly, pid, through_week=4)
    assert weeks == 3, "three weeks played, not four"
    assert total == pytest.approx(sum(weekly[pid][w]["actual"] for w in (1, 2, 3)))
    early, n = points_to_date(weekly, pid, through_week=2)
    assert n == 2 and early < total
    assert points_to_date(weekly, "no-such-player", 4) == (0.0, 0)


def test_the_season_total_block_is_not_mistaken_for_a_week(league):
    """ESPN also sends a season block (``statSplitTypeId`` 0, no scoring period): it must be skipped."""
    entry = league["teams"][0]["roster"]["entries"][0]
    blocks = entry["playerPoolEntry"]["player"]["stats"]
    assert any(b.get("statSplitTypeId") == 0 and "scoringPeriodId" not in b for b in blocks), "fixture regressed"
    weekly = weekly_player_points(league)
    assert max(weekly[str(entry["playerId"])]) == 4


def test_missing_and_malformed_payloads_are_survived():
    assert weekly_player_points(None) == {} and weekly_player_points({"teams": []}) == {}
    junk = {"teams": [{"roster": {"entries": [
        {"playerId": 7, "playerPoolEntry": {"player": {"stats": [
            {"scoringPeriodId": 1, "statSourceId": 0, "appliedTotal": "n/a"},      # unparseable
            {"scoringPeriodId": None, "statSourceId": 0, "appliedTotal": 9.0},     # no week
            {"scoringPeriodId": 2, "statSourceId": 7, "appliedTotal": 9.0},        # unknown source
            {"scoringPeriodId": 3, "statSourceId": 0, "appliedTotal": 11.5},       # the only good one
        ]}}},
        {"nonsense": True},
    ]}}]}
    assert weekly_player_points(junk) == {"7": {3: {"actual": 11.5}}}


# --------------------------------------------------------------------------- the lineup as set

def test_the_lineup_the_owner_actually_set_is_readable(league):
    """Start/sit advice is only useful next to the lineup it would change."""
    from draftadvisor.espn.inseason import BENCH_SLOT_ID, IR_SLOT_ID, current_lineups

    lineups = current_lineups(league)
    assert len(lineups) == 10
    for tid, block in lineups.items():
        assert set(block) == {"starters", "bench", "ir"}
        assert block["starters"], f"team {tid} starts nobody"
        total = sum(len(v) for v in block.values())
        assert total == 9, f"team {tid} has {total} roster entries"

    mixed = {"teams": [{"id": 1, "roster": {"entries": [
        {"playerId": 1, "lineupSlotId": 2},                 # RB: starting
        {"playerId": 2, "lineupSlotId": BENCH_SLOT_ID},
        {"playerId": 3, "lineupSlotId": IR_SLOT_ID},
        {"playerId": 4},                                    # no slot at all: not a starter
    ]}}]}
    assert current_lineups(mixed) == {1: {"starters": ["1"], "bench": ["2", "4"], "ir": ["3"]}}
    assert current_lineups(None) == {}


# --------------------------------------------------------------------------- over the wire

def test_the_in_season_views_come_back_through_the_stub():
    """One GET carries the calendar, the standings, the rosters and the matchups (no live ESPN)."""
    import asyncio

    from draftadvisor.espn.client import IN_SEASON_VIEWS, EspnClient
    from tests.espn_stub import LEAGUE_ID, SEASON, EspnStub

    async def run(base: str) -> dict:
        async with EspnClient(base_url=base) as client:
            return await client.get_in_season(LEAGUE_ID, SEASON, week=4)

    with EspnStub() as stub:
        data = asyncio.run(run(stub.base_url))
        asked = stub.requests[-1]["query"]
    assert set(asked["view"]) == set(IN_SEASON_VIEWS)
    assert asked["scoringPeriodId"] == ["4"], "the week has to reach ESPN or the stats are the wrong week's"
    cal = league_calendar(data)
    assert cal.current_week == 4 and cal.playoff_week_start == 14
    assert len(team_records(data)) == 10 and matchups(data, 4)
    assert weekly_player_points(data)
