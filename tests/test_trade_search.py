"""Finding trades, and the perception gap that makes them acceptable.

The evaluator answers "is this good for me". The finder has to answer a second question the owner
actually asked for: would the other manager say yes? Those need different numbers. My side is priced
with our blended projection - what we believe. Their side is priced with the market's rank-implied
consensus, which is what they believe. A deal worth sending is one that is good by our numbers and
reads as fine by theirs.

These tests pin that split, and the search behaviour around it: only deals that improve my starting
lineup, never a page of variations on one manager's roster, and partial results rather than none when
the clock runs out.
"""
from __future__ import annotations

import pytest

from draftadvisor.models import LeagueSettings, Player, Projection
from draftadvisor.strategy.trade import consensus_projections, find_trades

SLOTS = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "BN", "BN", "BN"]


def league() -> LeagueSettings:
    return LeagueSettings(league_id="t", name="test", season=2026, total_rosters=4,
                          roster_positions=list(SLOTS), scoring_settings={})


def mk(pid: str, pos: str, points: float, market: float | None = None) -> tuple[Player, Projection]:
    pl = Player(player_id=pid, name=pid, position=pos, team="AAA")
    pr = Projection(player_id=pid, position=pos, points=points, std=20.0, ppg=points / 17.0, games=17.0,
                    components={"ecr": market} if market is not None else {})
    return pl, pr


class World:
    """A tiny league: players, projections and rosters, built from ``mk`` tuples."""

    def __init__(self, *rows: tuple[Player, Projection]):
        self.players = {pl.player_id: pl for pl, _ in rows}
        self.projections = {pr.player_id: pr for _, pr in rows}

    def ids(self, *names: str) -> list[str]:
        return list(names)


def _base_world() -> World:
    return World(
        # mine: three good backs (only two start plus a flex) and thin at receiver
        *[mk("myRB1", "RB", 260.0, 260.0), mk("myRB2", "RB", 240.0, 240.0), mk("myRB3", "RB", 220.0, 220.0),
          mk("myRB4", "RB", 120.0, 120.0), mk("myWR1", "WR", 200.0, 200.0), mk("myWR2", "WR", 120.0, 120.0),
          mk("myQB", "QB", 300.0, 300.0), mk("myTE", "TE", 150.0, 150.0),
          # theirs: a strong receiver, thin at back
          mk("thWR1", "WR", 280.0, 280.0), mk("thWR2", "WR", 250.0, 250.0), mk("thWR3", "WR", 210.0, 210.0),
          mk("thRB1", "RB", 150.0, 150.0), mk("thQB", "QB", 280.0, 280.0), mk("thTE", "TE", 140.0, 140.0)]
    )


MINE = ["myRB1", "myRB2", "myRB3", "myRB4", "myWR1", "myWR2", "myQB", "myTE"]
THEIRS = ["thWR1", "thWR2", "thWR3", "thRB1", "thQB", "thTE"]


# --------------------------------------------------------------------------- the consensus view

def test_consensus_uses_the_markets_number_not_ours():
    w = World(mk("a", "WR", 100.0, 200.0), mk("b", "WR", 100.0))
    market = consensus_projections(w.projections)
    assert market["a"].points == 200.0, "the ecr component is what the other manager sees"
    assert market["b"].points == 100.0, "unranked is not the same as worthless: keep our number"
    assert market["a"].ppg == pytest.approx(200.0 / 17.0), "per-game has to follow the total"
    assert w.projections["a"].points == 100.0, "our own projections must not be mutated"


# --------------------------------------------------------------------------- what gets proposed

def test_depth_i_do_not_start_is_offered_for_the_starter_i_am_thin_at():
    """Backs four deep and receivers two deep: the trade goes one way."""
    w = _base_world()
    out = find_trades(MINE, {"Them": THEIRS}, w.players, w.projections, league(), limit=5)
    assert out, "a spare back for a starting receiver is the obvious trade here"
    best = out[0]
    assert best.my_gain > 0
    assert {w.players[p].position for p in best.get} == {"WR"}, best.get
    assert "myRB1" not in best.give and "myWR1" not in best.give, "never the players I am built around"
    assert set(best.give) & {"myRB3", "myRB4", "myWR2"}, best.give   # what leaves is what I can spare


def test_a_deal_that_reads_as_a_loss_to_them_is_not_proposed():
    """The whole point is a trade they would accept, not one we would like them to."""
    w = _base_world()
    out = find_trades(MINE, {"Them": THEIRS}, w.players, w.projections, league(), limit=20,
                      min_their_view=0.0)
    assert out and all(p.their_view >= 0.0 for p in out), [(p.give, p.get, p.their_view) for p in out]


def test_the_perception_gap_decides_which_player_to_target():
    """Same points to us, but the market rates one of them far lower: that is the one to ask for."""
    w = World(
        *[mk("myRB1", "RB", 260.0, 260.0), mk("myRB2", "RB", 240.0, 240.0), mk("myRB3", "RB", 230.0, 230.0),
          mk("myWR1", "WR", 200.0, 200.0), mk("myWR2", "WR", 120.0, 120.0),
          mk("myQB", "QB", 300.0, 300.0), mk("myTE", "TE", 150.0, 150.0),
          # identical to us; the market loves "hyped" and has soured on "cheap"
          mk("thHyped", "WR", 260.0, 320.0), mk("thCheap", "WR", 260.0, 180.0),
          mk("thWR3", "WR", 150.0, 150.0), mk("thRB1", "RB", 150.0, 150.0),
          mk("thQB", "QB", 280.0, 280.0), mk("thTE", "TE", 140.0, 140.0)]
    )
    mine = ["myRB1", "myRB2", "myRB3", "myWR1", "myWR2", "myQB", "myTE"]
    theirs = ["thHyped", "thCheap", "thWR3", "thRB1", "thQB", "thTE"]
    out = find_trades(mine, {"Them": theirs}, w.players, w.projections, league(), limit=6,
                      min_their_view=0.0)
    assert out, "there is a deal here"
    targets = [p for p in out if "thCheap" in p.get]
    assert targets, "the player the market underrates is the one we can actually get"
    assert all("thHyped" not in p.get for p in out[:1]), "asking for their favourite is how you get a no"


def test_a_player_who_would_sit_on_my_bench_is_never_asked_for():
    w = World(
        *[mk("myRB1", "RB", 260.0, 260.0), mk("myRB2", "RB", 250.0, 250.0), mk("myRB3", "RB", 240.0, 240.0),
          mk("myWR1", "WR", 260.0, 260.0), mk("myWR2", "WR", 250.0, 250.0),
          mk("myQB", "QB", 300.0, 300.0), mk("myTE", "TE", 250.0, 250.0),
          mk("thScrub", "WR", 40.0, 40.0), mk("thScrub2", "RB", 30.0, 30.0)]
    )
    mine = ["myRB1", "myRB2", "myRB3", "myWR1", "myWR2", "myQB", "myTE"]
    out = find_trades(mine, {"Them": ["thScrub", "thScrub2"]}, w.players, w.projections, league())
    assert out == [], "nothing on that roster would start for me"


def test_kickers_and_defences_are_left_out_of_it():
    w = World(
        *[mk("myRB1", "RB", 260.0, 260.0), mk("myRB2", "RB", 240.0, 240.0), mk("myRB3", "RB", 220.0, 220.0),
          mk("myWR1", "WR", 200.0, 200.0), mk("myWR2", "WR", 120.0, 120.0), mk("myQB", "QB", 300.0, 300.0),
          mk("myTE", "TE", 150.0, 150.0), mk("myK", "K", 140.0, 140.0), mk("myDEF", "DEF", 130.0, 130.0),
          mk("thWR1", "WR", 280.0, 280.0), mk("thRB1", "RB", 150.0, 150.0), mk("thK", "K", 150.0, 150.0)]
    )
    mine = ["myRB1", "myRB2", "myRB3", "myWR1", "myWR2", "myQB", "myTE", "myK", "myDEF"]
    out = find_trades(mine, {"Them": ["thWR1", "thRB1", "thK"]}, w.players, w.projections, league(), limit=10)
    moved = {p for prop in out for p in list(prop.give) + list(prop.get)}
    assert not (moved & {"myK", "myDEF", "thK"}), moved


# --------------------------------------------------------------------------- shape of the output

def test_one_willing_manager_cannot_fill_the_whole_list():
    w = _base_world()
    rosters = {f"Team {i}": list(THEIRS) for i in range(1, 5)}      # four identical, equally willing
    out = find_trades(MINE, rosters, w.players, w.projections, league(), limit=8, per_team_limit=2)
    per = {}
    for p in out:
        per[p.team] = per.get(p.team, 0) + 1
    assert max(per.values()) <= 2 and len(per) >= 2, per


def test_results_are_ordered_by_what_i_gain():
    w = _base_world()
    out = find_trades(MINE, {"Them": THEIRS}, w.players, w.projections, league(), limit=10)
    assert out == sorted(out, key=lambda p: (-p.my_gain, -p.their_view))
    assert all(p.my_gain >= 5.0 for p in out), "a deal worth less than the message is not a deal"


def test_the_pitch_speaks_from_their_side_of_the_table():
    w = _base_world()
    out = find_trades(MINE, {"Them": THEIRS}, w.players, w.projections, league(), limit=1)
    pitch = out[0].pitch(w.players)
    assert all(w.players[p].name in pitch for p in out[0].give + out[0].get)
    assert "consensus" in pitch and f"{out[0].their_view:+.0f}" in pitch
    assert "our projection" not in pitch, "never show them our numbers"


def test_running_out_of_time_returns_what_was_found():
    w = _base_world()
    rosters = {f"Team {i}": list(THEIRS) for i in range(1, 40)}
    out = find_trades(MINE, rosters, w.players, w.projections, league(), limit=5, time_budget_s=0.0)
    assert isinstance(out, list), "a budget of zero returns nothing, never raises"


def test_an_empty_or_unknown_roster_is_skipped():
    w = _base_world()
    out = find_trades(MINE, {"Empty": [], "Ghosts": ["nobody", "nothing"]}, w.players, w.projections, league())
    assert out == []
