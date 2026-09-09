"""Derive the small ESPN fixtures in this directory from real ESPN v3 API captures.

Source: the JSON captured by the ``espn_api`` library's tests (``tests/football/unit/data`` in
https://github.com/cwendt94/espn-api): ``league_2018_data.json`` (views mTeam + mRoster + mMatchup +
mSettings + mStandings), ``league_draft_2018.json`` (view mDraftDetail) and ``league_players_2018.json``
(players_wl). Run::

    python tests/fixtures/espn/make_fixtures.py --source /path/to/espn-api/tests/football/unit/data

Outputs (all < 200 KB):

* ``league_settings_teams.json``  settings + status + members + all 10 teams (rosters trimmed to
  ids / names / positions; no stats or rankings)
* ``draft_complete.json``         mDraftDetail + mSettings shape: drafted, all 150 picks
* ``draft_in_progress.json``      same league mid-draft: drafted=false, inProgress=true, 17 picks
  (pick 3 flagged ``keeper``, pick 14 ``reservedForKeeper``)
* ``draft_pre_draft.json``        no picks, pickOrder set
* ``draft_prepopulated.json``    board ESPN filled with placeholders: 150 entries, every playerId -1
* ``draft_prepopulated_no_order.json``  the same board with pickOrder withheld (order is in the board)
* ``draft_prepopulated_live.json``      SYNTHETIC, and not a shape real ESPN serves: 17 entries filled
  in place while ``inProgress`` is true. Every published measurement of a live ESPN draft says the REST
  board stays at ``playerId: -1`` for the whole draft and is flushed in one go at the end, so this
  fixture exercises our in-place-fill parsing only - it is not evidence of live behaviour. The live
  shape ESPN really serves is an empty board with the picks on team rosters: the stub builds that with
  ``EspnStub(roster_picks=N)``.
* ``league_keepers_rosters.json`` rosters with provenance: dynasty holdovers drafted a year before the
  draft date, ADD / TRADE pickups, fresh DRAFT entries after it, one entry on IR (slot 21) and one
  undated DRAFT entry
* ``players_kona.json``           synthetic ``kona_player_info`` shape for ~40 drafted players with
  ownership ADP, draft ranks and the ``10<season>`` projected season stats
* ``league_modern.json``          newer payload shape: owners as dicts, ``name`` instead of
  location + nickname, TE premium (``pointsOverrides {"6": 1.5}`` on statId 53), a SUPER_FLEX (OP) slot,
  SNAKE draft with a pick clock and keepers on every roster
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
#: Directory holding the espn-api capture files (``ESPN_FIXTURE_SOURCE`` or ``--source`` overrides).
DEFAULT_SOURCE = os.environ.get("ESPN_FIXTURE_SOURCE", "espn-api/tests/football/unit/data")

STATUS_KEYS = ("activatedDate", "createdAsLeagueType", "currentLeagueType", "currentMatchupPeriod",
               "finalScoringPeriod", "firstScoringPeriod", "isActive", "isExpired", "isFull", "isViewable",
               "latestScoringPeriod", "previousSeasons", "teamsJoined")
TEAM_KEYS = ("abbrev", "id", "location", "nickname", "logo", "logoType", "owners", "primaryOwner", "divisionId",
             "isActive", "playoffSeed", "waiverRank")
PLAYER_KEYS = ("id", "fullName", "firstName", "lastName", "defaultPositionId", "proTeamId", "eligibleSlots",
               "injuryStatus", "injured", "active", "droppable")
ENTRY_KEYS = ("playerId", "lineupSlotId", "acquisitionType", "acquisitionDate", "injuryStatus", "status")
KONA_PLAYER_KEYS = PLAYER_KEYS + ("ownership", "draftRanksByRankType")


def _dump(path: Path, obj: object) -> None:
    text = json.dumps(obj, indent=1, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")
    size = path.stat().st_size
    assert size < 200_000, f"{path.name} is {size} bytes"
    print(f"{path.name}: {size / 1024:.1f} KB")


def _trim_status(status: dict) -> dict:
    return {k: status[k] for k in STATUS_KEYS if k in status}


def _trim_player(pl: dict) -> dict:
    return {k: pl[k] for k in PLAYER_KEYS if k in pl}


def _trim_team(team: dict) -> dict:
    out = {k: team[k] for k in TEAM_KEYS if k in team}
    entries = []
    for e in (team.get("roster") or {}).get("entries") or []:
        ppe = e.get("playerPoolEntry") or {}
        pl = ppe.get("player") or {}
        entries.append({
            **{k: e[k] for k in ENTRY_KEYS if k in e},
            "playerPoolEntry": {"id": ppe.get("id"), "onTeamId": ppe.get("onTeamId"), "player": _trim_player(pl)},
        })
    out["roster"] = {"entries": entries}
    return out


def build(source: Path, dest: Path) -> None:
    league = json.loads((source / "league_2018_data.json").read_text(encoding="utf-8"))
    draft = json.loads((source / "league_draft_2018.json").read_text(encoding="utf-8"))
    if isinstance(league, list):
        league = league[0]
    if isinstance(draft, list):
        draft = draft[0]
    season = int(league["seasonId"])

    # -- league_settings_teams.json
    base = {
        "gameId": league.get("gameId", 1), "id": league["id"], "scoringPeriodId": league.get("scoringPeriodId"),
        "seasonId": season, "segmentId": league.get("segmentId", 0), "status": _trim_status(league.get("status") or {}),
        "settings": copy.deepcopy(league["settings"]), "members": copy.deepcopy(league.get("members") or []),
        "teams": [_trim_team(t) for t in league["teams"]],
    }
    _dump(dest / "league_settings_teams.json", base)

    # -- draft fixtures (mDraftDetail + mSettings -> full settings block)
    picks = copy.deepcopy(draft["draftDetail"]["picks"])
    common = {"gameId": draft.get("gameId", 1), "id": draft["id"], "scoringPeriodId": draft.get("scoringPeriodId"),
              "seasonId": season, "segmentId": draft.get("segmentId", 0), "status": _trim_status(draft.get("status") or {}),
              "settings": copy.deepcopy(league["settings"])}
    complete = dict(common, draftDetail={"completeDate": draft["draftDetail"].get("completeDate"), "drafted": True,
                                         "inProgress": False, "picks": picks})
    _dump(dest / "draft_complete.json", complete)

    partial = copy.deepcopy(picks[:17])
    partial[2]["keeper"] = True
    partial[13]["reservedForKeeper"] = True
    in_progress = dict(copy.deepcopy(common), draftDetail={"completeDate": 0, "drafted": False, "inProgress": True,
                                                            "picks": partial})
    in_progress["settings"]["draftSettings"]["type"] = "SNAKE"
    in_progress["settings"]["draftSettings"]["date"] = 1535198400000
    _dump(dest / "draft_in_progress.json", in_progress)

    pre = dict(copy.deepcopy(common), draftDetail={"completeDate": 0, "drafted": False, "inProgress": False, "picks": []})
    pre["settings"]["draftSettings"]["type"] = "SNAKE"
    pre["settings"]["draftSettings"]["date"] = 1535198400000
    _dump(dest / "draft_pre_draft.json", pre)

    # A board ESPN has pre-populated: one entry per pick, playerId -1 until the pick is made.
    # This is what a real un-started (and a live) ESPN draft returns; counting those entries as picks
    # fills the board and makes the draft look complete.
    def _placeholder(pick: dict) -> dict:
        out = copy.deepcopy(pick)
        out["playerId"] = -1
        out["keeper"] = False
        out["reservedForKeeper"] = False
        out["autoDraftTypeId"] = 0
        return out

    prepop = dict(copy.deepcopy(common),
                  draftDetail={"completeDate": 0, "drafted": False, "inProgress": False,
                               "picks": [_placeholder(p) for p in picks]})
    prepop["settings"]["draftSettings"]["type"] = "SNAKE"
    prepop["settings"]["draftSettings"]["date"] = 1535198400000
    _dump(dest / "draft_prepopulated.json", prepop)

    # The same board with the order withheld (ESPN publishes pickOrder late) - the board still shows it.
    prepop_no_order = copy.deepcopy(prepop)
    prepop_no_order["settings"]["draftSettings"]["pickOrder"] = []
    _dump(dest / "draft_prepopulated_no_order.json", prepop_no_order)

    # Live: the first 17 entries have been filled in place, the rest are still -1.
    live_picks = [copy.deepcopy(p) for p in picks[:17]] + [_placeholder(p) for p in picks[17:]]
    prepop_live = dict(copy.deepcopy(common),
                       draftDetail={"completeDate": 0, "drafted": False, "inProgress": True, "picks": live_picks})
    prepop_live["settings"]["draftSettings"]["type"] = "SNAKE"
    prepop_live["settings"]["draftSettings"]["date"] = 1535198400000
    _dump(dest / "draft_prepopulated_live.json", prepop_live)

    # -- league_keepers_rosters.json: one league whose rosters carry every acquisition flavour, so the
    # "which roster entries are picks of the draft I am watching?" rules can be tested offline.
    keepers = copy.deepcopy(base)
    draft_date = 1535198400000                                   # the same date the draft fixtures use
    year = 365 * 24 * 3600 * 1000
    keepers["settings"]["draftSettings"]["date"] = draft_date
    keepers["settings"]["draftSettings"]["type"] = "SNAKE"
    keepers["settings"]["draftSettings"]["keeperCount"] = 2
    flavours = [
        {"acquisitionType": "DRAFT", "acquisitionDate": draft_date - year, "lineupSlotId": 20},   # last year's draft
        {"acquisitionType": "ADD", "acquisitionDate": draft_date + 5_000, "lineupSlotId": 20},    # waiver add
        {"acquisitionType": "TRADE", "acquisitionDate": draft_date + 6_000, "lineupSlotId": 20},  # trade
        {"acquisitionType": "DRAFT", "acquisitionDate": draft_date + 7_000, "lineupSlotId": 20},  # this draft
        {"acquisitionType": "DRAFT", "acquisitionDate": draft_date + 8_000, "lineupSlotId": 21},  # this draft, IR
        {"acquisitionType": "DRAFT", "acquisitionDate": None, "lineupSlotId": 20},                # no date at all
    ]
    for t in keepers["teams"]:
        entries = t["roster"]["entries"][: len(flavours)]
        for e, flavour in zip(entries, flavours):
            e.update({k: v for k, v in flavour.items() if v is not None})
            if flavour["acquisitionDate"] is None:
                e.pop("acquisitionDate", None)
        t["roster"]["entries"] = entries
    keepers["draftDetail"] = {"completeDate": 0, "drafted": False, "inProgress": True, "picks": []}
    _dump(dest / "league_keepers_rosters.json", keepers)

    # -- players_kona.json: drafted players that still have full info on a roster
    rostered: dict[int, tuple[dict, int]] = {}
    for t in league["teams"]:
        for e in (t.get("roster") or {}).get("entries") or []:
            pl = (e.get("playerPoolEntry") or {}).get("player") or {}
            if pl.get("id") is not None:
                rostered[pl["id"]] = (pl, t["id"])
    chosen: list[int] = []
    for p in picks:
        pid = p["playerId"]
        if pid in rostered and pid not in chosen and len(chosen) < 34:
            chosen.append(pid)
    for want_pos, n in ((16, 3), (5, 3)):                    # make sure D/ST and K are covered
        have = sum(1 for pid in chosen if rostered[pid][0].get("defaultPositionId") == want_pos)
        for p in picks:
            pid = p["playerId"]
            if have >= n:
                break
            if pid in rostered and pid not in chosen and rostered[pid][0].get("defaultPositionId") == want_pos:
                chosen.append(pid)
                have += 1
    kona = []
    for pid in chosen:
        pl, team_id = rostered[pid]
        proj = [s for s in pl.get("stats") or [] if str(s.get("id")) == f"10{season}"]
        entry_player = {k: copy.deepcopy(pl[k]) for k in KONA_PLAYER_KEYS if k in pl}
        entry_player["stats"] = [{k: s[k] for k in ("id", "seasonId", "statSourceId", "statSplitTypeId", "scoringPeriodId",
                                                     "appliedTotal", "appliedAverage", "stats") if k in s} for s in proj]
        kona.append({"id": pid, "onTeamId": team_id, "status": "ONTEAM", "player": entry_player})
    _dump(dest / "players_kona.json", {"players": kona, "positionAgainstOpponent": {}})

    # -- league_modern.json
    modern = copy.deepcopy(base)
    members = {m["id"]: m for m in modern["members"]}
    for t in modern["teams"]:
        t["name"] = f"{t.pop('location', '')} {t.pop('nickname', '')}".replace("  ", " ").strip()
        owners = []
        for swid in t.get("owners") or []:
            m = members.get(swid) or {}
            owners.append({"id": swid, "displayName": m.get("displayName", ""), "firstName": m.get("firstName", ""),
                           "lastName": m.get("lastName", "")})
        t["owners"] = owners
        t["roster"]["entries"] = t["roster"]["entries"][:2]           # two keepers per team
    st = modern["settings"]
    st["draftSettings"].update({"type": "SNAKE", "timePerSelection": 60, "date": 1756500000000, "keeperCount": 2,
                                "leagueSubType": "KEEPER"})
    counts = st["rosterSettings"]["lineupSlotCounts"]
    counts["7"] = 1                                                # OP = SUPER_FLEX
    counts["20"] = 5                                               # one fewer bench slot: still 15 rounds
    for item in st["scoringSettings"]["scoringItems"]:
        if item["statId"] == 53:
            item["pointsOverrides"] = {"6": 1.5}                   # TE premium
    st["scoringSettings"]["scoringItems"].append({"statId": 15, "points": 2.0, "isReverseItem": False})  # 40+ yd TD pass
    st["scoringSettings"]["scoringItems"].append({"statId": 5, "points": 0.2, "isReverseItem": False})   # every 5 pass yd
    modern["seasonId"] = season + 8
    modern["draftDetail"] = {"drafted": False, "inProgress": False, "picks": []}
    _dump(dest / "league_modern.json", modern)


def build_inseason(dest: Path) -> None:
    """``league_inseason.json``: the in-season shape, derived from ``league_settings_teams.json``.

    SYNTHETIC, and needs saying: nobody here has watched a live ESPN league mid-season, so the numbers
    are made up. What is *not* made up is the shape - ``schedule[]`` with per-side ``totalPoints``,
    ``teams[].record.overall``, ``currentSimulationResults.playoffPct`` and per-week player ``stats``
    blocks with ``statSourceId`` 0/1 and ``appliedTotal`` - which is documented and stable across the
    v3 API. It exercises the parsers; it is not evidence about live behaviour. Whether ESPN publishes
    scores that *move during games* is a separate, unmeasured question.

    Deterministic (a fixed seed), so regenerating it produces the same file.
    """
    base = json.loads((dest / "league_settings_teams.json").read_text(encoding="utf-8"))
    rnd = random.Random(20260909)
    week = 4                                                        # three weeks played, week 4 live
    data = copy.deepcopy(base)
    data["scoringPeriodId"] = week
    data["status"] = dict(data.get("status") or {}, currentMatchupPeriod=week, latestScoringPeriod=week,
                          firstScoringPeriod=1, finalScoringPeriod=17, isActive=True)
    data["settings"]["scheduleSettings"] = dict(data["settings"].get("scheduleSettings") or {},
                                                matchupPeriodCount=13, matchupPeriodLength=1,
                                                playoffTeamCount=6, playoffSeedingRule="TOTAL_POINTS_SCORED")

    teams = data.get("teams") or []
    for i, team in enumerate(teams):
        wins = max(0, 3 - (i % 4))
        team["record"] = {"overall": {"wins": wins, "losses": 3 - wins, "ties": 0,
                                      "pointsFor": round(320.0 + 12.0 * (len(teams) - i), 1),
                                      "pointsAgainst": round(300.0 + 9.0 * i, 1)}}
        team["playoffSeed"] = i + 1
        team["currentSimulationResults"] = {"playoffPct": round(max(1.0, 95.0 - 9.0 * i), 1)}
        roster = team.get("roster") or {}
        roster["entries"] = (roster.get("entries") or [])[:9]       # a full lineup, under the size cap
        for entry in roster["entries"]:
            pool = entry.get("playerPoolEntry") or {}
            full = pool.get("player") or {}
            # only what in-season parsing reads, so the weekly stat blocks fit under the size cap
            player = {k: full[k] for k in ("id", "fullName", "defaultPositionId", "eligibleSlots",
                                           "proTeamId", "injuryStatus") if k in full}
            pool["player"] = player
            base_ppg = 6.0 + (player.get("id", 0) % 13)
            stats = []
            for w in range(1, week + 1):
                proj = round(base_ppg, 1)
                stats.append({"scoringPeriodId": w, "statSourceId": 1, "statSplitTypeId": 1, "appliedTotal": proj})
                if w < week:                                        # week 4 has not been played yet
                    actual = round(max(0.0, rnd.gauss(base_ppg, 0.5 * base_ppg)), 1)
                    stats.append({"scoringPeriodId": w, "statSourceId": 0, "statSplitTypeId": 1,
                                  "appliedTotal": actual})
            stats.append({"statSourceId": 0, "statSplitTypeId": 0, "appliedTotal": round(base_ppg * 3, 1)})
            player["stats"] = stats

    schedule = []
    ids = [t["id"] for t in teams]
    for w in range(1, 14):
        rotation = ids[:1] + ids[1:][(w - 1) % max(1, len(ids) - 1):] + ids[1:][:(w - 1) % max(1, len(ids) - 1)]
        for a, b in zip(rotation[::2], rotation[1::2]):
            game = {"id": len(schedule) + 1, "matchupPeriodId": w,
                    "home": {"teamId": a, "totalPoints": 0.0}, "away": {"teamId": b, "totalPoints": 0.0}}
            if w < week:
                hp, ap = round(rnd.uniform(70, 150), 1), round(rnd.uniform(70, 150), 1)
                game["home"]["totalPoints"], game["away"]["totalPoints"] = hp, ap
                game["winner"] = "HOME" if hp > ap else ("AWAY" if ap > hp else "TIE")
            else:
                game["winner"] = "UNDECIDED"
            schedule.append(game)
    data["schedule"] = schedule
    _dump(dest / "league_inseason.json", data)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=DEFAULT_SOURCE, help="directory with the espn-api test JSON files")
    ap.add_argument("--dest", default=str(HERE))
    ap.add_argument("--inseason-only", action="store_true",
                    help="regenerate only league_inseason.json (needs no external source)")
    args = ap.parse_args()
    if args.inseason_only:
        build_inseason(Path(args.dest))
        return
    build(Path(args.source), Path(args.dest))
    build_inseason(Path(args.dest))


if __name__ == "__main__":
    main()
