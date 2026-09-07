"""Pure functions turning raw Sleeper JSON into :mod:`draftadvisor.models` objects.

Everything here is side-effect free and tolerant of Sleeper's JSON quirks:

* numeric ids arrive as strings (``"4034"``) but sometimes as ints;
* ``draft_order`` / ``slot_to_roster_id`` keys are strings;
* a pick's ``roster_id`` may be an int, a string or ``None``; ``is_keeper`` may be ``null``;
* ``injury_status`` ``""`` means "healthy" (``None``);
* ``draft_order`` and ``slot_to_roster_id`` are ``null`` before the draft starts.
"""
from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping

from ..config import DEFAULT_SEASON
from ..models import DraftSettings, DraftState, LeagueSettings, Manager, Pick

log = logging.getLogger(__name__)

__all__ = [
    "parse_league",
    "parse_draft",
    "parse_pick",
    "parse_picks",
    "parse_managers",
    "adp_key_for",
    "adp_map",
    "resolve_my_slot",
    "rostered_player_ids",
    "state_from_sleeper",
]

# Preference order when the requested ADP key is missing for a player. The
# scales are close enough that mixing them is far better than "no ADP".
_ADP_FALLBACK_KEYS: tuple[str, ...] = ("adp_half_ppr", "adp_ppr", "adp_std", "adp_2qb")


# ---------------------------------------------------------------------------
# Small coercion helpers
# ---------------------------------------------------------------------------


def _int(v: Any, default: int | None = None) -> int | None:
    """Coerce ``v`` (int / numeric str / float) to int; ``default`` when impossible."""
    if v is None or v == "":
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return default


def _float(v: Any, default: float | None = None) -> float | None:
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _str(v: Any, default: str | None = None) -> str | None:
    if v is None:
        return default
    s = str(v)
    return s if s != "" else default


def _clean_metadata(meta: Mapping[str, Any] | None) -> dict:
    """Copy pick metadata, normalising ``""`` to ``None`` for status-like keys."""
    out: dict = dict(meta or {})
    for k in ("injury_status", "status", "team"):
        if k in out and out[k] == "":
            out[k] = None
    if "years_exp" in out:
        out["years_exp"] = _int(out.get("years_exp"))
    if "player_id" in out and out["player_id"] is not None:
        out["player_id"] = str(out["player_id"])
    return out


# ---------------------------------------------------------------------------
# League / draft
# ---------------------------------------------------------------------------


def parse_league(raw: Mapping[str, Any]) -> LeagueSettings:
    """Convert a ``/league/{id}`` payload into :class:`LeagueSettings`."""
    settings = dict(raw.get("settings") or {})
    roster_positions = [str(s) for s in (raw.get("roster_positions") or [])]
    scoring: dict[str, float] = {}
    for k, v in (raw.get("scoring_settings") or {}).items():
        f = _float(v)
        if f is not None:
            scoring[str(k)] = f
    total = _int(raw.get("total_rosters")) or _int(settings.get("num_teams")) or 12
    return LeagueSettings(
        league_id=_str(raw.get("league_id"), "") or "",
        name=_str(raw.get("name"), "") or "",
        season=_int(raw.get("season"), DEFAULT_SEASON) or DEFAULT_SEASON,
        total_rosters=total,
        roster_positions=roster_positions,
        scoring_settings=scoring,
        settings=settings,
        draft_id=_str(raw.get("draft_id")),
        status=_str(raw.get("status")),
        raw=dict(raw),
    )


def _parse_traded_picks(traded: Iterable[Mapping[str, Any]] | None,
                        season: int | None) -> dict[tuple[int, int], int]:
    """``[{round, roster_id (original), owner_id (current)}]`` -> ``{(round, orig): owner}``."""
    out: dict[tuple[int, int], int] = {}
    for tp in traded or []:
        rnd = _int(tp.get("round"))
        orig = _int(tp.get("roster_id"))
        owner = _int(tp.get("owner_id"))
        if rnd is None or orig is None or owner is None:
            continue
        tp_season = _int(tp.get("season"))
        if season is not None and tp_season is not None and tp_season != season:
            continue
        if owner == orig:
            continue
        out[(rnd, orig)] = owner
    return out


def parse_draft(raw: Mapping[str, Any], traded_picks: list[dict] | None = None) -> DraftSettings:
    """Convert a ``/draft/{id}`` payload (+ optional traded picks) into :class:`DraftSettings`."""
    settings = dict(raw.get("settings") or {})
    metadata = dict(raw.get("metadata") or {})
    draft_order: dict[str, int] = {}
    for uid, slot in (raw.get("draft_order") or {}).items():
        s = _int(slot)
        if s is not None:
            draft_order[str(uid)] = s
    slot_to_roster: dict[int, int] = {}
    for slot, rid in (raw.get("slot_to_roster_id") or {}).items():
        s, r = _int(slot), _int(rid)
        if s is not None and r is not None:
            slot_to_roster[s] = r
    teams = _int(settings.get("teams")) or len(draft_order) or len(slot_to_roster) or 12
    rounds = _int(settings.get("rounds")) or 15
    season = _int(raw.get("season"))
    return DraftSettings(
        draft_id=_str(raw.get("draft_id"), "") or "",
        league_id=_str(raw.get("league_id")),
        type=_str(raw.get("type"), "snake") or "snake",
        status=_str(raw.get("status"), "pre_draft") or "pre_draft",
        teams=teams,
        rounds=rounds,
        pick_timer=_int(settings.get("pick_timer"), 0) or 0,
        reversal_round=_int(settings.get("reversal_round"), 0) or 0,
        player_type=_int(settings.get("player_type"), 0) or 0,
        draft_order=draft_order,
        slot_to_roster_id=slot_to_roster,
        traded_picks=_parse_traded_picks(traded_picks, season),
        season=season,
        scoring_type=_str(metadata.get("scoring_type")),
        start_time=_int(raw.get("start_time")),
        last_picked=_int(raw.get("last_picked")),
        settings=settings,
        metadata=metadata,
        raw=dict(raw),
    )


# ---------------------------------------------------------------------------
# Picks
# ---------------------------------------------------------------------------


def parse_pick(raw: Mapping[str, Any]) -> Pick:
    """Convert one entry of ``/draft/{id}/picks``. ``pick_no``/``round``/``draft_slot`` are required."""
    pick_no = _int(raw.get("pick_no"))
    if pick_no is None:
        raise ValueError(f"pick without pick_no: {raw!r}")
    return Pick(
        pick_no=pick_no,
        round=_int(raw.get("round"), 0) or 0,
        draft_slot=_int(raw.get("draft_slot"), 0) or 0,
        player_id=str(raw.get("player_id") or ""),
        roster_id=_int(raw.get("roster_id")),
        picked_by=_str(raw.get("picked_by")),
        is_keeper=bool(raw.get("is_keeper") or False),
        metadata=_clean_metadata(raw.get("metadata")),
    )


def parse_picks(raw: Iterable[Mapping[str, Any]] | None) -> list[Pick]:
    """Parse, sort by ``pick_no`` and de-duplicate (the last occurrence of a pick number wins)."""
    by_no: dict[int, Pick] = {}
    for item in raw or []:
        try:
            p = parse_pick(item)
        except (ValueError, TypeError) as e:
            log.warning("skipping malformed pick: %s", e)
            continue
        if not p.player_id:
            log.warning("skipping pick %s without player_id", p.pick_no)
            continue
        by_no[p.pick_no] = p
    return [by_no[k] for k in sorted(by_no)]


# ---------------------------------------------------------------------------
# Managers
# ---------------------------------------------------------------------------


def parse_managers(users: list[dict] | None, draft: DraftSettings,
                   rosters: list[dict] | None = None) -> dict[str, Manager]:
    """Build ``user_id -> Manager`` from league users, the draft order and (optionally) rosters.

    Users that appear in ``draft.draft_order`` but not in ``users`` (e.g. when the users
    endpoint is unavailable) get a placeholder ``Manager`` named ``"Team <slot>"``.
    """
    rosters_by_owner: dict[str, int] = {}
    for r in rosters or []:
        rid = _int(r.get("roster_id"))
        if rid is None:
            continue
        owner = r.get("owner_id")
        if owner is not None:
            rosters_by_owner.setdefault(str(owner), rid)
        for co in r.get("co_owners") or []:
            rosters_by_owner.setdefault(str(co), rid)

    out: dict[str, Manager] = {}
    for u in users or []:
        uid = _str(u.get("user_id"))
        if uid is None:
            continue
        slot = draft.draft_order.get(uid)
        roster_id = rosters_by_owner.get(uid)
        if roster_id is None and slot is not None:
            roster_id = draft.slot_to_roster_id.get(slot)
        meta = u.get("metadata") or {}
        out[uid] = Manager(
            user_id=uid,
            display_name=_str(u.get("display_name")) or _str(u.get("username")) or uid,
            team_name=_str(meta.get("team_name")),
            slot=slot,
            roster_id=roster_id,
            avatar=_str(u.get("avatar")),
        )
    for uid, slot in draft.draft_order.items():
        if uid not in out:
            out[uid] = Manager(
                user_id=uid,
                display_name=f"Team {slot}",
                slot=slot,
                roster_id=rosters_by_owner.get(uid) or draft.slot_to_roster_id.get(slot),
            )
    return out


def rostered_player_ids(rosters: Iterable[Mapping[str, Any]] | None) -> set[str]:
    """Every player id on any league roster (``players`` + ``reserve`` + ``taxi``), str-coerced.

    In dynasty / keeper leagues these players cannot be drafted (Sleeper's board excludes them);
    in a redraft league the rosters are empty before the draft, so the set is empty.
    """
    out: set[str] = set()
    for r in rosters or []:
        if not isinstance(r, Mapping):
            continue
        for key in ("players", "reserve", "taxi"):
            for pid in r.get(key) or []:
                if pid is not None and pid != "":
                    out.add(str(pid))
    return out


# ---------------------------------------------------------------------------
# ADP
# ---------------------------------------------------------------------------


def _draft_is_superflex(draft: DraftSettings | None) -> bool:
    if draft is None:
        return False
    s = draft.settings or {}
    if (_int(s.get("slots_super_flex"), 0) or 0) > 0 or (_int(s.get("slots_qb"), 0) or 0) >= 2:
        return True
    st = (draft.scoring_type or "").lower()
    return "2qb" in st or "superflex" in st


def adp_key_for(league: LeagueSettings | None, draft: DraftSettings | None) -> str:
    """Which Sleeper ADP key matches this league: ``adp_ppr`` / ``adp_half_ppr`` / ``adp_std`` / ``adp_2qb``.

    Superflex (or 2-QB) leagues use ``adp_2qb``. Dynasty ADP keys are never used for redraft.
    """
    if (league is not None and league.is_superflex) or _draft_is_superflex(draft):
        return "adp_2qb"
    if league is not None:
        rec = league.rec_points
    else:
        st = (draft.scoring_type or "").lower() if draft is not None else ""
        if "half" in st:
            rec = 0.5
        elif "ppr" in st:
            rec = 1.0
        else:
            rec = 0.0
    if rec >= 0.75:
        return "adp_ppr"
    if rec >= 0.25:
        return "adp_half_ppr"
    return "adp_std"


def adp_map(projections: Mapping[str, Mapping[str, Any]] | None, key: str) -> dict[str, float]:
    """``{player_id: adp}`` from a normalised projections dict, using ``key`` (with sane fallbacks)."""
    out: dict[str, float] = {}
    for pid, stats in (projections or {}).items():
        if not isinstance(stats, Mapping):
            continue
        v = _float(stats.get(key))
        if v is None or v <= 0:
            for fk in _ADP_FALLBACK_KEYS:
                v = _float(stats.get(fk))
                if v is not None and v > 0:
                    break
        if v is not None and v > 0:
            out[str(pid)] = v
    return out


# ---------------------------------------------------------------------------
# Who am I?
# ---------------------------------------------------------------------------


def _slot_for_user(draft: DraftSettings, managers: Mapping[str, Manager], uid: str) -> int | None:
    slot = draft.draft_order.get(uid)
    if slot is not None:
        return slot
    m = managers.get(uid)
    if m is None:
        return None
    if m.slot is not None:
        return m.slot
    if m.roster_id is not None:
        for s, rid in draft.slot_to_roster_id.items():
            if rid == m.roster_id:
                return s
    return None


def _user_for_slot(draft: DraftSettings, managers: Mapping[str, Manager], slot: int) -> str | None:
    for uid, s in draft.draft_order.items():
        if s == slot:
            return uid
    for m in managers.values():
        if m.slot == slot:
            return m.user_id
    return None


def resolve_my_slot(draft: DraftSettings, managers: dict[str, Manager], *, username: str | None = None,
                    user_id: str | None = None, slot: int | None = None,
                    users: list[dict] | None = None) -> tuple[str | None, int | None]:
    """Work out ``(my_user_id, my_slot)``. An explicit ``slot`` wins, then ``user_id``, then ``username``.

    ``username`` matching is case-insensitive against ``Manager.display_name`` / ``team_name`` and, when
    the raw ``users`` payload is given, the users' ``username`` too. Returns ``(None, None)`` when nothing
    matches (the advisor then runs in spectator mode).
    """
    if slot is not None:
        s = _int(slot)
        if s is not None:
            return _user_for_slot(draft, managers, s), s
    if user_id:
        uid = str(user_id)
        return uid, _slot_for_user(draft, managers, uid)
    if username:
        needle = username.strip().lower()
        for m in managers.values():
            if m.display_name.lower() == needle or (m.team_name or "").lower() == needle:
                return m.user_id, _slot_for_user(draft, managers, m.user_id)
        for u in users or []:
            uname = (u.get("username") or "").lower()
            dname = (u.get("display_name") or "").lower()
            if needle in (uname, dname) and u.get("user_id") is not None:
                uid = str(u["user_id"])
                return uid, _slot_for_user(draft, managers, uid)
        if needle in managers or needle in draft.draft_order:  # it was actually a user id
            return needle, _slot_for_user(draft, managers, needle)
        log.warning("username %r not found among the draft's managers", username)
    return None, None


# ---------------------------------------------------------------------------
# Full state
# ---------------------------------------------------------------------------


def state_from_sleeper(draft_raw: dict, picks_raw: list[dict], league_raw: dict | None = None,
                       users_raw: list[dict] | None = None, rosters_raw: list[dict] | None = None,
                       traded_raw: list[dict] | None = None, *, username: str | None = None,
                       user_id: str | None = None, slot: int | None = None) -> DraftState:
    """Assemble a :class:`DraftState` from the raw payloads of the relevant Sleeper endpoints."""
    draft = parse_draft(draft_raw, traded_raw)
    league = parse_league(league_raw) if league_raw else None
    managers = parse_managers(users_raw, draft, rosters_raw)
    my_user_id, my_slot = resolve_my_slot(draft, managers, username=username, user_id=user_id,
                                          slot=slot, users=users_raw)
    return DraftState(
        draft=draft,
        picks=parse_picks(picks_raw),
        league=league,
        managers=managers,
        my_user_id=my_user_id,
        my_slot=my_slot,
        rostered_ids=rostered_player_ids(rosters_raw),
    )
