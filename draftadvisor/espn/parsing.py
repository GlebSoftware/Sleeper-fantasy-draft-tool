"""Pure functions turning ESPN v3 league / draft JSON into :mod:`draftadvisor.models` objects.

Field provenance (ESPN league payload, views mSettings + mTeam + mRoster + mDraftDetail):

* league: ``id``, ``seasonId``, ``settings.name``, ``settings.size``, ``settings.isPublic``,
  ``settings.rosterSettings.lineupSlotCounts`` (slot id -> count), ``settings.scoringSettings.scoringItems``;
* draft: ``settings.draftSettings`` (``type``, ``pickOrder`` = team ids in slot order, ``timePerSelection``,
  ``date``, ``keeperCount``) and ``draftDetail`` (``drafted``, ``inProgress``, ``picks``);
* teams: ``teams[]`` (``id``, ``abbrev``, ``name`` or ``location`` + ``nickname``, ``logo``, ``owners``
  (SWID strings or member dicts), ``primaryOwner``, ``roster.entries[]``) and ``members[]``
  (``id`` = SWID, ``displayName``, ``firstName``, ``lastName``);
* picks: ``draftDetail.picks[]`` (``overallPickNumber``, ``roundId``, ``roundPickNumber``, ``teamId``,
  ``playerId``, ``keeper``, ``reservedForKeeper``).

Everything is side-effect free and tolerant of missing fields: real ESPN payloads vary by season.
Picks are used whatever ``draftDetail.drafted`` says (``espn_api`` ignores them until the draft is
over; a live advisor must not). ESPN has no linear draft order and no "paused" status.
"""
from __future__ import annotations

from urllib.parse import unquote

import dataclasses
import logging
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from ..config import DEFAULT_SEASON
from ..models import DraftSettings, DraftState, LeagueSettings, Manager, Pick, RosterSpot
from .constants import IR_SLOT_ID, ROSTER_SLOT_ORDER, slot_label
from .ids import EspnIdMap, espn_player_fields, is_real_player_id
from .scoring import espn_scoring_to_sleeper

log = logging.getLogger(__name__)

__all__ = [
    "draft_status",
    "league_status",
    "roster_positions_from_counts",
    "rounds_from_counts",
    "draft_settings_of",
    "draft_detail_of",
    "parse_espn_league",
    "parse_espn_draft",
    "parse_espn_managers",
    "parse_espn_picks",
    "board_pick_number",
    "pick_order_from_board",
    "traded_picks_from_detail",
    "rostered_espn_ids",
    "rostered_ids",
    "roster_names",
    "roster_spots",
    "draft_epoch_ms",
    "fresh_draft_spots",
    "reconstruct_roster_picks",
    "merge_league_payload",
    "DRAFT_ACQUISITIONS",
    "normalize_swid",
    "team_owner_ids",
    "resolve_my_team",
    "state_from_espn",
]

PLATFORM = "espn"


# ---------------------------------------------------------------------------
# Small coercion helpers
# ---------------------------------------------------------------------------


def _int(v: Any, default: int | None = None) -> int | None:
    if v is None or v == "":
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return default


def _str(v: Any, default: str | None = None) -> str | None:
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


def _mapping(v: Any) -> Mapping[str, Any]:
    return v if isinstance(v, Mapping) else {}


def _settings(league_json: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return _mapping(_mapping(league_json).get("settings"))


def draft_settings_of(*payloads: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """``settings.draftSettings`` from the first payload that carries it (the per-poll draft payload is
    listed first so a commissioner's change of ``pickOrder`` / ``timePerSelection`` is followed)."""
    for p in payloads:
        ds = _settings(p).get("draftSettings")
        if isinstance(ds, Mapping) and ds:
            return ds
    return {}


def draft_detail_of(*payloads: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """``draftDetail`` from the first payload that has one (a bare ``draftDetail`` dict is accepted too)."""
    for p in payloads:
        p = _mapping(p)
        dd = p.get("draftDetail")
        if isinstance(dd, Mapping):
            return dd
        if "picks" in p or "drafted" in p or "inProgress" in p:
            return p
    return {}


def _lineup_counts(*payloads: Mapping[str, Any] | None) -> Mapping[str, Any]:
    for p in payloads:
        counts = _mapping(_settings(p).get("rosterSettings")).get("lineupSlotCounts")
        if isinstance(counts, Mapping) and counts:
            return counts
    return {}


# ---------------------------------------------------------------------------
# Status / roster shape
# ---------------------------------------------------------------------------


def _is_made_pick(raw: Any) -> bool:
    """A board entry holding a real, non-keeper pick (empty / keeper-reserved slots do not start a draft)."""
    if not isinstance(raw, Mapping) or raw.get("keeper") or raw.get("reservedForKeeper"):
        return False
    return is_real_player_id(raw.get("playerId"))


def draft_status(detail: Mapping[str, Any] | None) -> str:
    """``draftDetail`` -> ``"complete"`` (``drafted``), ``"drafting"`` (``inProgress``, or picks already on
    the board - OFFLINE drafts entered by the commissioner never set ``inProgress``) or ``"pre_draft"``."""
    d = _mapping(detail)
    if d.get("drafted"):
        return "complete"
    if d.get("inProgress") or any(_is_made_pick(p) for p in (d.get("picks") or [])):
        return "drafting"
    return "pre_draft"


def league_status(detail: Mapping[str, Any] | None) -> str | None:
    """Sleeper-style league status: ``"in_season"`` once drafted, ``"drafting"``, ``"pre_draft"``; ``None`` without ``draftDetail``."""
    if not isinstance(detail, Mapping) or not detail:
        return None
    return {"complete": "in_season"}.get(draft_status(detail), draft_status(detail))


def roster_positions_from_counts(counts: Mapping[str, Any] | None) -> list[str]:
    """``lineupSlotCounts`` (``{"0": 1, "2": 2, ...}``) -> ``["QB", "RB", "RB", ...]`` in
    :data:`~draftadvisor.espn.constants.ROSTER_SLOT_ORDER`; punter / head-coach slots are dropped."""
    by_label: dict[str, int] = {}
    for slot_id, n in _mapping(counts).items():
        label = slot_label(slot_id)
        k = _int(n, 0) or 0
        if label is None or k <= 0:
            continue
        by_label[label] = by_label.get(label, 0) + k
    out: list[str] = []
    for label in ROSTER_SLOT_ORDER:
        out.extend([label] * by_label.pop(label, 0))
    for label, k in by_label.items():
        out.extend([label] * k)
    return out


def rounds_from_counts(counts: Mapping[str, Any] | None) -> int:
    """Draft rounds = every lineup slot except IR (slot 21); ESPN drafts a full roster."""
    total = 0
    for slot_id, n in _mapping(counts).items():
        if _int(slot_id) == IR_SLOT_ID:
            continue
        total += max(0, _int(n, 0) or 0)
    return total


def _scoring_type(scoring: Mapping[str, float]) -> str:
    rec = float(scoring.get("rec", 0.0) or 0.0)
    return "ppr" if rec >= 0.75 else "half_ppr" if rec >= 0.25 else "std"


def _team_count(st: Mapping[str, Any], teams_list: list, pick_order: list[int]) -> int:
    """``settings.size`` (else the number of ``teams[]``), never a *shorter* ``pickOrder``: ESPN exposes
    the order while teams are still joining or the commissioner is editing it, and a short order would
    corrupt every snake computation. A longer order wins (the size is the stale value then)."""
    size = _int(st.get("size")) or len(teams_list)
    if not size:
        return len(pick_order) or 10
    if pick_order and len(pick_order) != size:
        log.info("ESPN pickOrder lists %d teams in a %d-team league; using %d", len(pick_order), size,
                 max(size, len(pick_order)))
    return max(size, len(pick_order))


# ---------------------------------------------------------------------------
# League / draft
# ---------------------------------------------------------------------------


def parse_espn_league(league_json: Mapping[str, Any], draft_detail_json: Mapping[str, Any] | None = None) -> LeagueSettings:
    """League payload (view mSettings [+ mTeam]) -> :class:`LeagueSettings`.

    ``settings`` carries ``platform`` ("espn"), ``keeper_count`` / ``max_keepers``, ``is_public``,
    ``draft_type``, ``unmapped_scoring`` (ESPN rules the model ignores), ``scoring_type_espn``
    and ``num_teams``; ``status`` comes from ``draftDetail`` when either payload has one. Like
    :func:`parse_espn_draft`, the draft settings and lineup slot counts are read from the (fresher)
    per-poll payload first, so a commissioner's change is reflected in both objects.
    """
    lj = _mapping(league_json)
    st = _settings(lj)
    teams = lj.get("teams") if isinstance(lj.get("teams"), list) else []
    league_id = _str(lj.get("id"), "") or ""
    season = _int(lj.get("seasonId"), DEFAULT_SEASON) or DEFAULT_SEASON
    counts = _lineup_counts(draft_detail_json, lj)
    scoring, unmapped = espn_scoring_to_sleeper(_mapping(st.get("scoringSettings")).get("scoringItems"))
    ds = draft_settings_of(draft_detail_json, lj)
    pick_order = [t for t in (_int(x) for x in (ds.get("pickOrder") or [])) if t is not None]
    size = _team_count(st, teams, pick_order)
    keeper_count = _int(ds.get("keeperCount"), 0) or 0
    settings: dict[str, Any] = {
        "platform": PLATFORM,
        "keeper_count": keeper_count,
        "max_keepers": keeper_count,
        "is_public": bool(st.get("isPublic", False)),
        "draft_type": _str(ds.get("type")),
        "unmapped_scoring": unmapped,
        "scoring_type_espn": _str(_mapping(st.get("scoringSettings")).get("scoringType")),
        "num_teams": size,
        "auction_budget": _int(ds.get("auctionBudget")),
        "time_per_selection": _int(ds.get("timePerSelection")),
        "pick_order": pick_order,
        "lineup_slot_counts": dict(counts),
    }
    detail = draft_detail_of(draft_detail_json, lj)
    return LeagueSettings(
        league_id=league_id,
        name=_str(st.get("name"), "") or f"ESPN league {league_id}",
        season=season,
        total_rosters=size,
        roster_positions=roster_positions_from_counts(counts),
        scoring_settings=scoring,
        settings=settings,
        draft_id=f"{PLATFORM}-{league_id}-{season}",
        status=league_status(detail),
        raw=dict(lj),
    )


def pick_order_from_board(detail: Mapping[str, Any] | None, teams: int) -> list[int]:
    """Round-1 board entries -> ``[team id in slot 1, slot 2, ...]``.

    ESPN pre-populates the board with one entry per pick (``playerId`` ``-1`` until the pick is made),
    each already carrying its ``teamId``, so the order is visible there before ``draftSettings.pickOrder``
    is published. Returns ``[]`` unless a complete, duplicate-free round 1 is present.
    """
    if teams <= 0:
        return []
    by_pick: dict[int, int] = {}
    for raw in _mapping(detail).get("picks") or []:
        if not isinstance(raw, Mapping):
            continue
        pick_no = _int(raw.get("overallPickNumber"))
        tid = _int(raw.get("teamId"))
        if pick_no is None or not 1 <= pick_no <= teams or tid is None or tid <= 0:
            continue
        by_pick.setdefault(pick_no, tid)
    order = [by_pick[k] for k in sorted(by_pick)]
    if len(order) != teams or len(set(order)) != teams:
        return []
    return order


def traded_picks_from_detail(detail: Mapping[str, Any] | None, draft: DraftSettings) -> dict[tuple[int, int], int]:
    """``{(round, original team id): owning team id}`` for every board entry whose owner is not the snake
    owner of its pick number (``owningTeamIds[0]`` for a pick still to be made, ``teamId`` once made).

    Entries with an unknown owner (``teamId`` 0 on a pre-populated board, a team outside the order) are
    ignored, as is everything when the pick order is unknown or the draft is an auction. Whether ESPN
    lists not-yet-made picks in ``draftDetail.picks`` could not be verified offline: when it does not,
    traded future picks only become visible once they are made.
    """
    d = _mapping(detail)
    if draft.type == "auction" or draft.teams <= 0 or not draft.slot_to_roster_id:
        return {}
    known = set(draft.slot_to_roster_id.values())
    out: dict[tuple[int, int], int] = {}
    for raw in d.get("picks") or []:
        if not isinstance(raw, Mapping):
            continue
        pick_no = _int(raw.get("overallPickNumber"))
        if pick_no is None or pick_no <= 0 or pick_no > draft.total_picks:
            continue
        owners = raw.get("owningTeamIds")
        owner = _int(raw.get("teamId")) if is_real_player_id(raw.get("playerId")) else None
        if owner is None and isinstance(owners, list) and owners:
            owner = _int(owners[0])
        if owner is None:
            owner = _int(raw.get("teamId"))
        if owner not in known:
            continue
        orig = draft.original_roster_for_slot(draft.slot_for_pick(pick_no))
        if orig is not None and orig != owner:
            out[(draft.round_of(pick_no), orig)] = owner
    return out


def parse_espn_draft(league_json: Mapping[str, Any], draft_detail_json: Mapping[str, Any] | None = None) -> DraftSettings:
    """League (+ optional per-poll draft) payload -> :class:`DraftSettings`.

    * ``draft_id`` = ``espn-<league id>-<season>``; ``type`` = ``"auction"`` for ``draftSettings.type``
      AUCTION, ``"snake"`` otherwise (SNAKE / OFFLINE / AUTOPICK); ``status`` per :func:`draft_status`;
    * ``teams`` = ``settings.size`` (else ``teams[]``, else ``len(pickOrder)``; a longer ``pickOrder``
      wins, a shorter one never shrinks the league); ``rounds`` per :func:`rounds_from_counts`;
    * ``pick_timer`` = ``timePerSelection`` (seconds); ``start_time`` = ``draftSettings.date`` (epoch ms);
    * ``draft_order`` = ``{str(teamId): slot}`` and ``slot_to_roster_id`` = ``{slot: teamId}`` from
      ``pickOrder`` (slot 1 = first entry; both empty until the commissioner sets the order);
    * ``traded_picks`` per :func:`traded_picks_from_detail` (``owningTeamIds`` / ``teamId`` of the board);
    * ``settings`` = ``draftSettings`` plus ``teams`` / ``rounds`` / ``pick_timer``;
      ``metadata`` = ``{"name", "platform": "espn", "scoring_type"}``.
    """
    lj = _mapping(league_json)
    st = _settings(lj)
    ds = draft_settings_of(draft_detail_json, lj)
    detail = draft_detail_of(draft_detail_json, lj)
    league_id = _str(lj.get("id")) or _str(_mapping(draft_detail_json).get("id")) or ""
    season = _int(lj.get("seasonId")) or _int(_mapping(draft_detail_json).get("seasonId")) or DEFAULT_SEASON
    pick_order = [t for t in (_int(x) for x in (ds.get("pickOrder") or [])) if t is not None]
    teams_list = lj.get("teams") if isinstance(lj.get("teams"), list) else []
    teams = _team_count(st, teams_list, pick_order)
    if not pick_order:                       # order not published yet: the pre-populated board shows it
        pick_order = pick_order_from_board(detail, teams)
        if pick_order:
            log.info("ESPN league %s: pick order derived from the draft board", league_id or "?")
    counts = _lineup_counts(draft_detail_json, lj)
    rounds = rounds_from_counts(counts) or 15
    pick_timer = _int(ds.get("timePerSelection"), 0) or 0
    dtype = "auction" if (_str(ds.get("type")) or "").upper() == "AUCTION" else "snake"
    scoring, _ = espn_scoring_to_sleeper(_mapping(st.get("scoringSettings")).get("scoringItems"))
    settings = dict(ds)
    settings.update({"teams": teams, "rounds": rounds, "pick_timer": pick_timer})
    draft = DraftSettings(
        draft_id=f"{PLATFORM}-{league_id}-{season}",
        league_id=league_id or None,
        type=dtype,
        status=draft_status(detail),
        teams=teams,
        rounds=rounds,
        pick_timer=pick_timer,
        reversal_round=0,
        player_type=0,
        draft_order={str(tid): i + 1 for i, tid in enumerate(pick_order)},
        slot_to_roster_id={i + 1: tid for i, tid in enumerate(pick_order)},
        traded_picks={},
        season=season,
        scoring_type=_scoring_type(scoring),
        start_time=_int(ds.get("date")) or None,
        last_picked=None,
        settings=settings,
        metadata={"name": _str(st.get("name"), "") or "", "platform": PLATFORM, "scoring_type": _scoring_type(scoring)},
        raw={"draftSettings": dict(ds), "draftDetail": {k: v for k, v in detail.items() if k != "picks"}},
    )
    draft.traded_picks = traded_picks_from_detail(detail, draft)
    return draft


# ---------------------------------------------------------------------------
# Teams / owners
# ---------------------------------------------------------------------------


def normalize_swid(swid: Any) -> str:
    """Compare SWIDs case-insensitively, with or without braces, raw or URL-encoded (``%7B...%7D``)."""
    return unquote(str(swid or "").strip()).strip("{}").upper()


def team_owner_ids(team: Mapping[str, Any]) -> list[str]:
    """Normalised owner SWIDs of a team: ``primaryOwner`` first, then ``owners[]`` (strings or dicts)."""
    out: list[str] = []
    primary = normalize_swid(_mapping(team).get("primaryOwner"))
    if primary:
        out.append(primary)
    for o in _mapping(team).get("owners") or []:
        sid = normalize_swid(o.get("id") if isinstance(o, Mapping) else o)
        if sid and sid not in out:
            out.append(sid)
    return out


def _owner_dicts(team: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {normalize_swid(o.get("id")): o for o in (_mapping(team).get("owners") or []) if isinstance(o, Mapping)}


def _members(league_json: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {normalize_swid(m.get("id")): m for m in (_mapping(league_json).get("members") or []) if isinstance(m, Mapping)}


def _person_name(m: Mapping[str, Any] | None) -> str | None:
    m = _mapping(m)
    return _str(m.get("displayName")) or _str(" ".join(x for x in (_str(m.get("firstName")), _str(m.get("lastName"))) if x))


def _team_name(team: Mapping[str, Any]) -> str | None:
    t = _mapping(team)
    name = _str(t.get("name"))
    if name:
        return name
    return _str(" ".join(x for x in (_str(t.get("location")), _str(t.get("nickname"))) if x))


def parse_espn_managers(league_json: Mapping[str, Any], draft: DraftSettings) -> dict[str, Manager]:
    """``teams[]`` + ``members[]`` -> ``{str(team id): Manager}``.

    ``display_name`` = the primary owner's ``displayName`` (via ``members``, or an owner dict), else
    ``abbrev``; ``team_name`` = ``name`` or ``location`` + ``nickname``; ``slot`` from the draft order;
    ``roster_id`` = team id; ``avatar`` = ``logo``. Teams in the pick order but missing from ``teams``
    get a ``"Team <slot>"`` placeholder.
    """
    lj = _mapping(league_json)
    members = _members(lj)
    out: dict[str, Manager] = {}
    for team in lj.get("teams") or []:
        if not isinstance(team, Mapping):
            continue
        tid = _int(team.get("id"))
        if tid is None:
            continue
        owners = _owner_dicts(team)
        display = None
        for sid in team_owner_ids(team):
            display = _person_name(members.get(sid)) or _person_name(owners.get(sid))
            if display:
                break
        team_name = _team_name(team)
        key = str(tid)
        out[key] = Manager(
            user_id=key,
            display_name=display or _str(team.get("abbrev")) or team_name or f"Team {tid}",
            team_name=team_name,
            slot=draft.draft_order.get(key),
            roster_id=tid,
            avatar=_str(team.get("logo")),
        )
    for key, slot in draft.draft_order.items():
        if key not in out:
            out[key] = Manager(user_id=key, display_name=f"Team {slot}", slot=slot, roster_id=_int(key))
    return out


# ---------------------------------------------------------------------------
# Picks
# ---------------------------------------------------------------------------


def _pick_metadata(info: Mapping[str, Any] | None, espn_id: Any) -> dict:
    meta: dict[str, Any] = {"espn_id": str(espn_id)}
    if not info:
        return meta
    name = _str(info.get("name"))
    first, last = _str(info.get("first_name")), _str(info.get("last_name"))
    if name and not (first or last):
        parts = name.split(" ", 1)
        first, last = parts[0], (parts[1] if len(parts) > 1 else "")
    meta.update({"first_name": first, "last_name": last, "position": _str(info.get("position")),
                 "team": _str(info.get("team"))})
    return meta


def board_pick_number(raw: Mapping[str, Any] | None, teams: int) -> int | None:
    """Overall pick number of a board entry that has no (or a zero) ``overallPickNumber``, derived from
    ``roundId`` + ``roundPickNumber``; ``None`` when neither is usable."""
    rnd = _int(_mapping(raw).get("roundId"), 0) or 0
    in_round = _int(_mapping(raw).get("roundPickNumber"), 0) or 0
    if teams <= 0 or rnd <= 0 or in_round <= 0:
        return None
    return (rnd - 1) * teams + in_round


def parse_espn_picks(draft_detail_json: Mapping[str, Any] | None, id_map: EspnIdMap, draft: DraftSettings,
                     names: Mapping[str, Mapping[str, Any]] | None = None) -> list[Pick]:
    """``draftDetail.picks[]`` -> sorted, de-duplicated :class:`Pick` list.

    ``pick_no`` = ``overallPickNumber``, ``round`` = ``roundId``, ``draft_slot`` = the slot of ``teamId``
    in the pick order (else the snake slot for that pick number, else ``roundPickNumber``),
    ``roster_id`` = ``picked_by`` = ``teamId``, ``is_keeper`` = ``keeper`` or ``reservedForKeeper``,
    ``player_id`` via ``id_map`` (``names`` = ``{espn id: {name, position, team, ...}}`` feeds the name
    fallback and the pick metadata). Entries without a player (empty keeper slots) are skipped.

    An entry whose ``overallPickNumber`` is missing or 0 keeps its pick number from ``roundId`` +
    ``roundPickNumber`` (:func:`board_pick_number`), and an entry that carries its player nested under
    ``playerPoolEntry`` instead of ``playerId`` is read through :func:`espn_player_fields`, so neither
    shape silently empties the board.
    """
    detail = draft_detail_of(draft_detail_json)
    slot_of_team = {tid: slot for slot, tid in draft.slot_to_roster_id.items()}
    names = names or {}
    by_no: dict[int, Pick] = {}
    for raw in detail.get("picks") or []:
        if not isinstance(raw, Mapping):
            continue
        pick_no = _int(raw.get("overallPickNumber")) or board_pick_number(raw, draft.teams)
        espn_id = raw.get("playerId")
        if espn_id is None:
            espn_id = espn_player_fields(raw)["espn_id"]
        if pick_no is None or pick_no <= 0 or not is_real_player_id(espn_id):
            continue                                    # placeholder entry (playerId -1 / 0): not a pick
        tid = _int(raw.get("teamId"))
        slot = slot_of_team.get(tid) if tid is not None else None
        if slot is None:
            if draft.type != "auction" and draft.teams > 0:
                slot = draft.slot_for_pick(pick_no)
            else:
                slot = _int(raw.get("roundPickNumber"), 0) or 0
        info = names.get(str(espn_id)) or {}
        player_id = id_map.resolve(espn_id, name=info.get("name"), position=info.get("position"),
                                   pro_team_id=info.get("pro_team_id"))
        by_no[pick_no] = Pick(
            pick_no=pick_no,
            round=_int(raw.get("roundId")) or (draft.round_of(pick_no) if draft.teams else 0),
            draft_slot=slot,
            player_id=player_id,
            roster_id=tid,
            picked_by=str(tid) if tid is not None else None,
            is_keeper=bool(raw.get("keeper") or raw.get("reservedForKeeper")),
            metadata=_pick_metadata(info, espn_id),
        )
    return [by_no[k] for k in sorted(by_no)]


# ---------------------------------------------------------------------------
# Rosters (keeper / dynasty leagues)
# ---------------------------------------------------------------------------


def _roster_entries(league_json: Mapping[str, Any] | None) -> Iterable[tuple[int | None, Mapping[str, Any]]]:
    for team in _mapping(league_json).get("teams") or []:
        if not isinstance(team, Mapping):
            continue
        tid = _int(team.get("id"))
        for e in _mapping(team.get("roster")).get("entries") or []:
            if isinstance(e, Mapping):
                yield tid, e


def rostered_espn_ids(league_json: Mapping[str, Any] | None) -> set[str]:
    """ESPN ids of every player on a league roster (``teams[].roster.entries[].playerId``)."""
    out: set[str] = set()
    for _, e in _roster_entries(league_json):
        pid = e.get("playerId")
        if pid is None:
            pid = _mapping(_mapping(e.get("playerPoolEntry")).get("player")).get("id")
        if is_real_player_id(pid):
            out.add(str(pid))
    return out


def rostered_ids(league_json: Mapping[str, Any] | None, id_map: EspnIdMap) -> set[str]:
    """Canonical ids of every rostered player (resolved through ``id_map`` with the roster's names)."""
    out: set[str] = set()
    for _, e in _roster_entries(league_json):
        f = espn_player_fields(e)
        if f["espn_id"] is None:
            continue
        out.add(id_map.resolve(f["espn_id"], name=f["name"], position=f["position"], pro_team_id=f["pro_team_id"]))
    return out


#: ``acquisitionType`` values that mean "came out of a draft" (ESPN has used both spellings).
DRAFT_ACQUISITIONS = frozenset({"DRAFT", "DRAFTED"})
#: Nothing older than 1 May of the league's season can be a pick of this season's draft.
_SEASON_FLOOR_MONTH = 5
#: A commissioner can start a draft early; accept picks stamped up to this long before the scheduled time.
_DRAFT_GRACE_MS = 6 * 3600 * 1000


def roster_spots(league_json: Mapping[str, Any] | None, id_map: EspnIdMap) -> list[RosterSpot]:
    """Every ``teams[].roster.entries[]`` entry as a :class:`RosterSpot` (team + acquisition provenance).

    The player half goes through :func:`espn_player_fields` / ``id_map`` exactly like a pick, so a player
    known only from a roster gets the same canonical id the board would give him.
    """
    out: list[RosterSpot] = []
    for tid, e in _roster_entries(league_json):
        f = espn_player_fields(e)
        if f["espn_id"] is None or not is_real_player_id(f["espn_id"]):
            continue
        lineup = _int(e.get("lineupSlotId"))
        position = f["position"] or (slot_label(lineup) if lineup is not None else None)
        out.append(RosterSpot(
            player_id=id_map.resolve(f["espn_id"], name=f["name"], position=f["position"],
                                     pro_team_id=f["pro_team_id"]),
            espn_id=f["espn_id"], roster_id=tid,
            acquisition_type=_str(e.get("acquisitionType")),
            acquired_at=_int(e.get("acquisitionDate")),
            lineup_slot_id=lineup, name=f["name"],
            position=position if position not in ("BN", "IR") else f["position"],
        ))
    return out


def draft_epoch_ms(league_json: Mapping[str, Any] | None, draft_detail_json: Mapping[str, Any] | None,
                   season: int) -> int:
    """Lower bound, in ESPN epoch ms, for "acquired in *this* season's draft".

    ``draftSettings.date`` minus six hours of grace when ESPN publishes one, else 1 May of the league's
    season (no earlier season's draft can be after that). Compared only against ESPN's own
    ``acquisitionDate`` - never against our wall clock.
    """
    date = _int(draft_settings_of(draft_detail_json, league_json).get("date"), 0) or 0
    if date > 0:
        return date - _DRAFT_GRACE_MS
    yr = _int(season) or _int(_mapping(league_json).get("seasonId")) or DEFAULT_SEASON
    return int(datetime(int(yr), _SEASON_FLOOR_MONTH, 1, tzinfo=timezone.utc).timestamp() * 1000)


def fresh_draft_spots(spots: Iterable[RosterSpot], threshold_ms: int, *, keeper_ids: Iterable[str] = (),
                      keeper_count: int = 0, draft_started: bool = True) -> tuple[list[RosterSpot], list[RosterSpot]]:
    """Split roster spots into ``(fresh, undated)``: the ones that look like picks of the draft being
    watched, and the ones that might be but carry no date to prove it.

    In order: an IR entry is never fresh (nobody drafts onto IR); a player the board already flags as a
    keeper is never fresh; an acquisition that is not a draft (ADD / TRADE / WAIVER) is never fresh; an
    entry stamped before ``threshold_ms`` is a previous season's draft (dynasty / keeper holdover).

    ``acquisitionDate`` is not always populated (it is view- and season-dependent), so an entry without
    one is fresh only when nothing contradicts it: the league can have no keepers (``keeper_count`` 0)
    *and* ESPN itself says the draft has started (``draft_started``, i.e. ``drafted`` / ``inProgress``).
    Otherwise it is returned as ``undated`` - counted in the diagnostics, but no pick is claimed for it.
    """
    keepers = {str(k) for k in keeper_ids}
    fresh: list[RosterSpot] = []
    undated: list[RosterSpot] = []
    for s in spots:
        if s.lineup_slot_id == IR_SLOT_ID or s.player_id in keepers:
            continue
        if (s.acquisition_type or "").upper() not in DRAFT_ACQUISITIONS:
            continue
        if s.acquired_at is None:
            (fresh if (keeper_count <= 0 and draft_started) else undated).append(s)
            continue
        if s.acquired_at >= threshold_ms:
            fresh.append(s)
    return fresh, undated


def reconstruct_roster_picks(spots: Sequence[RosterSpot], draft: DraftSettings) -> tuple[list[tuple[int, RosterSpot]], str]:
    """Try to attach pick numbers to fresh roster spots: ``(pairs, confidence)`` with confidence
    ``"exact"`` / ``"team"`` / ``"none"``.

    The count check is what proves anything: in the first ``k`` picks of a snake every team's number of
    picks is forced, so when the spots' per-team counts match that prefix exactly, pick numbers 1..k are
    accounted for. Only then is an order attempted, from ``acquisitionDate`` - and ties carry no
    ordering information, so equal timestamps are permuted within their group before giving up.
    ESPN stamps a whole team's picks with the same millisecond in an OFFLINE (commissioner-entered)
    draft, which lands in ``"team"`` or ``"none"``: exactly right, those timestamps do not order a draft.
    """
    k = len(spots)
    if k == 0 or draft.type == "auction" or draft.teams <= 0 or not draft.slot_to_roster_id:
        return [], "none"
    if draft.total_picks and k > draft.total_picks:      # more rostered players than the draft has picks
        return [], "none"
    expected = [draft.owner_roster_for_pick(n) for n in range(1, k + 1)]
    if any(o is None for o in expected):
        return [], "none"
    if Counter(expected) != Counter(s.roster_id for s in spots):
        return [], "none"
    ordered = sorted(spots, key=lambda s: (s.acquired_at if s.acquired_at is not None else 0, str(s.espn_id)))
    # walk the expected owner sequence, taking the first still-unused spot of that team inside the
    # current group of equal timestamps (ties are unordered, so any member of the group may be next)
    remaining = list(ordered)
    picks: list[tuple[int, RosterSpot]] = []
    exact = True
    for n, owner in enumerate(expected, start=1):
        if not remaining:
            exact = False
            break
        head = remaining[0].acquired_at
        group_end = 0
        while group_end < len(remaining) and remaining[group_end].acquired_at == head:
            group_end += 1
        hit = next((i for i in range(group_end) if remaining[i].roster_id == owner), None)
        if hit is None:
            exact = False
            break
        picks.append((n, remaining.pop(hit)))
    if exact and len(picks) == k:
        return picks, "exact"
    by_team: dict[int | None, list[RosterSpot]] = {}
    for s in ordered:
        by_team.setdefault(s.roster_id, []).append(s)
    numbers: dict[int | None, list[int]] = {}
    for n, owner in enumerate(expected, start=1):
        numbers.setdefault(owner, []).append(n)
    out: list[tuple[int, RosterSpot]] = []
    for owner, team_spots in by_team.items():
        for n, s in zip(numbers.get(owner, []), team_spots):
            out.append((n, s))
    return sorted(out, key=lambda pair: pair[0]), "team"


def merge_league_payload(capture_json: Mapping[str, Any] | None,
                         poll_json: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """The cached league capture with ``teams[]`` (and ``members[]``) replaced by the poll's when the
    poll carries them (views mTeam + mRoster); the capture unchanged when it does not.

    Teams are replaced wholesale, never merged per team: a team missing from the poll means ESPN did
    not send it, and a half-merged roster would resurrect players who are no longer there.
    """
    capture = _mapping(capture_json)
    poll = _mapping(poll_json)
    teams = poll.get("teams")
    if not (isinstance(teams, list) and teams):
        return capture
    merged = dict(capture)
    merged["teams"] = teams
    members = poll.get("members")
    if isinstance(members, list) and members:
        merged["members"] = members
    return merged


def roster_names(league_json: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """``{espn id: {name, first_name, last_name, position, team, pro_team_id}}`` from the roster entries'
    ``playerPoolEntry.player`` (the only names a league payload carries)."""
    out: dict[str, dict[str, Any]] = {}
    for _, e in _roster_entries(league_json):
        f = espn_player_fields(e)
        if f["espn_id"] is not None and f["name"]:
            out[f["espn_id"]] = f
    return out


# ---------------------------------------------------------------------------
# Who am I?
# ---------------------------------------------------------------------------


def _slot_for_team(draft: DraftSettings, managers: Mapping[str, Manager], tid: str) -> int | None:
    slot = draft.draft_order.get(tid)
    if slot is None and tid in managers:
        slot = managers[tid].slot
    return slot


def resolve_my_team(draft: DraftSettings, managers: Mapping[str, Manager], league_json: Mapping[str, Any] | None,
                    *, swid: str | None = None, team_id: int | str | None = None, slot: int | None = None,
                    username: str | None = None) -> tuple[str | None, int | None]:
    """``(my team id as str, my slot)``; precedence ``slot`` > ``team_id`` > ``swid`` > ``username``.

    ``slot`` names the team directly once the pick order is known; until then it is kept as "my" slot
    and the team is learned from the other hints (so a SWID / team id / name still identifies the team
    before the commissioner sets the order). A slot outside ``1..draft.teams`` is ignored (warning).
    ``swid`` matches ``teams[].owners`` / ``primaryOwner`` case-insensitively with or without braces;
    ``username`` matches (case-insensitively) the team name, ``abbrev``, the owners' ``displayName``,
    ``firstName lastName`` or ``firstName``. ``(None, None)`` = spectator.
    """
    my_slot: int | None = None
    if slot is not None:
        s = _int(slot)
        if s is not None:
            if s < 1 or (draft.teams > 0 and s > draft.teams):
                log.warning("slot %s is outside 1..%d for ESPN league %s; ignoring it", s, draft.teams, draft.league_id)
            else:
                tid = draft.slot_to_roster_id.get(s)
                if tid is not None:
                    return str(tid), s
                my_slot = s

    def found(tid: str) -> tuple[str, int | None]:
        return tid, (my_slot if my_slot is not None else _slot_for_team(draft, managers, tid))

    if team_id is not None and str(team_id).strip() != "":
        tid = _str(_int(team_id)) or str(team_id).strip()
        if tid in managers or tid in draft.draft_order:
            return found(tid)
        return None, my_slot
    lj = _mapping(league_json)
    teams = [t for t in (lj.get("teams") or []) if isinstance(t, Mapping) and _int(t.get("id")) is not None]
    if swid:
        needle = normalize_swid(swid)
        if needle:
            for team in teams:
                if needle in team_owner_ids(team):
                    return found(str(_int(team.get("id"))))
            log.info("SWID does not own a team in ESPN league %s", lj.get("id"))
    if username:
        needle = username.strip().lower()
        if needle:
            members = _members(lj)
            for team in teams:
                tid = str(_int(team.get("id")))
                cands: list[str] = [x for x in (_team_name(team), _str(team.get("abbrev")), _str(team.get("location")),
                                                _str(team.get("nickname"))) if x]
                owners = _owner_dicts(team)
                for sid in team_owner_ids(team):
                    m = members.get(sid) or owners.get(sid) or {}
                    full = " ".join(x for x in (_str(m.get("firstName")), _str(m.get("lastName"))) if x)
                    cands.extend(x for x in (_str(m.get("displayName")), full, _str(m.get("firstName"))) if x)
                if any(c.lower() == needle for c in cands):
                    return found(tid)
            if needle in managers:                          # a team id was passed as the "username"
                return found(needle)
            log.warning("username %r not found among the ESPN league's teams", username)
    return None, my_slot


# ---------------------------------------------------------------------------
# Full state
# ---------------------------------------------------------------------------


def state_from_espn(league_json: Mapping[str, Any], draft_detail_json: Mapping[str, Any] | None, id_map: EspnIdMap, *,
                    names: Mapping[str, Mapping[str, Any]] | None = None, swid: str | None = None,
                    team_id: int | str | None = None, slot: int | None = None,
                    username: str | None = None) -> DraftState:
    """Assemble a :class:`DraftState` from the league payload (settings / teams / rosters) and the
    draft payload (``draftDetail`` + fresh ``draftSettings``); rostered players resolve through ``id_map``.

    When the draft payload carries teams of its own (the per-poll GET asks for mTeam + mRoster too),
    those rosters win over the league payload's: they are this second's rosters, the league capture is
    up to ten minutes old. Players who are on a roster but not on the board become
    :class:`~draftadvisor.models.RosterSpot` records - they leave the draftable pool and are attributed
    to their team, but they never become picks and never move the clock. A board that is still empty
    while fresh draft acquisitions sit on the rosters reports the draft as in progress.
    """
    merged = merge_league_payload(league_json, draft_detail_json)
    league = parse_espn_league(merged, draft_detail_json)
    draft = parse_espn_draft(merged, draft_detail_json)
    managers = parse_espn_managers(merged, draft)
    all_names: dict[str, Mapping[str, Any]] = dict(roster_names(merged))
    all_names.update(names or {})
    picks = parse_espn_picks(draft_detail_json if draft_detail_json is not None else merged, id_map, draft, all_names)
    my_uid, my_slot = resolve_my_team(draft, managers, merged, swid=swid, team_id=team_id, slot=slot,
                                      username=username)
    detail = draft_detail_of(draft_detail_json, merged)
    spots = roster_spots(merged, id_map)
    threshold = draft_epoch_ms(merged, draft_detail_json, league.season)
    fresh, undated = fresh_draft_spots(spots, threshold, keeper_ids=[p.player_id for p in picks if p.is_keeper],
                                       keeper_count=_int(league.settings.get("keeper_count"), 0) or 0,
                                       draft_started=bool(detail.get("drafted") or detail.get("inProgress")))
    fresh_ids = {s.player_id for s in fresh}
    spots = [dataclasses.replace(s, is_fresh=s.player_id in fresh_ids) for s in spots]
    board_ids = {p.player_id for p in picks}
    roster_only = [s for s in fresh if s.player_id not in board_ids]
    numbered, confidence = reconstruct_roster_picks(roster_only, draft)
    if draft.status == "pre_draft" and roster_only:
        log.info("ESPN board is empty but %d fresh DRAFT roster entries exist; treating the draft as in progress",
                 len(roster_only))
        draft = dataclasses.replace(draft, status="drafting")
    if undated:
        log.info("%d roster entries have no acquisitionDate in a keeper league; counted, not claimed as picks",
                 len(undated))
    return DraftState(
        draft=draft,
        picks=picks,
        league=league,
        managers=managers,
        my_user_id=my_uid,
        my_slot=my_slot,
        rostered_ids=rostered_ids(merged, id_map),
        roster_spots=spots,
        roster_confidence=confidence,
        roster_pick_numbers={s.player_id: n for n, s in numbered},
    )
