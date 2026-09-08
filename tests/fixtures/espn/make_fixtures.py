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
ENTRY_KEYS = ("playerId", "lineupSlotId", "acquisitionType", "injuryStatus", "status")
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=DEFAULT_SOURCE, help="directory with the espn-api test JSON files")
    ap.add_argument("--dest", default=str(HERE))
    args = ap.parse_args()
    build(Path(args.source), Path(args.dest))


if __name__ == "__main__":
    main()
