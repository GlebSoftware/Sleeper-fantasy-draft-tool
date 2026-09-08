"""ESPN player ids -> our canonical player ids.

Our universe is keyed by Sleeper ids (team defenses by team abbreviation, see
:mod:`draftadvisor.models`). ESPN picks and rosters only carry ESPN's numeric ``playerId``
(team defenses: ``-16000 - proTeamId``), so every ESPN id is resolved through
:class:`EspnIdMap`:

1. ``Player.espn_id`` from the bundle / Sleeper payload;
2. team defenses by pro team (``-16023`` -> ``PIT``);
3. normalised ``(name, position)`` (plus team when the name is ambiguous);
4. a synthetic ``espn:<id>`` id (see :func:`placeholder_player`) so the pick is still on the board.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping

from ..data.names import normalize_name
from ..models import Player
from .constants import (
    DEFAULT_POSITION_MAP,
    DST_ID_BASE,
    ELIGIBLE_SLOT_POSITION,
    INJURY_STATUS_MAP,
    PRO_TEAM_MAP,
    TEAM_TO_PRO_ID,
)

log = logging.getLogger(__name__)

__all__ = [
    "SYNTHETIC_PREFIX",
    "EspnIdMap",
    "espn_player_fields",
    "espn_position",
    "espn_team",
    "dst_team",
    "placeholder_player",
]

SYNTHETIC_PREFIX = "espn:"


def _int(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return None


def _str(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def dst_team(espn_id: Any) -> str | None:
    """Team abbreviation when ``espn_id`` is a D/ST id (``-16000 - proTeamId``), else ``None``."""
    i = _int(espn_id)
    if i is None or i >= 0:
        return None
    return PRO_TEAM_MAP.get(DST_ID_BASE - i)


def espn_team(pro_team_id: Any) -> str | None:
    """``player.proTeamId`` -> Sleeper abbreviation (``None`` for 0 / free agents / unknown)."""
    i = _int(pro_team_id)
    return PRO_TEAM_MAP.get(i) if i is not None else None


def _player_dict(obj: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """The ``player`` dict inside a kona entry / roster entry, or ``obj`` itself when it already is one."""
    if not isinstance(obj, Mapping):
        return {}
    ppe = obj.get("playerPoolEntry")
    if isinstance(ppe, Mapping):
        return _player_dict(ppe)
    inner = obj.get("player")
    if isinstance(inner, Mapping):
        return inner
    return obj


def espn_position(player: Mapping[str, Any] | None) -> str | None:
    """Our position for an ESPN player dict: ``defaultPositionId`` (1 QB, 2 RB, 3 WR, 4 TE, 5 K, 16 D/ST;
    IDP / punter / head coach keep their own label - DL, LB, DB, P, HC - so they never pass for a skill
    player), else the first single-position entry of ``eligibleSlots``, else ``DEF`` for a negative id."""
    pl = _player_dict(player)
    pos = DEFAULT_POSITION_MAP.get(_int(pl.get("defaultPositionId")) or -1)
    if pos:
        return pos
    for slot in pl.get("eligibleSlots") or []:
        p = ELIGIBLE_SLOT_POSITION.get(_int(slot) or -1)
        if p:
            return p
    if dst_team(pl.get("id")):
        return "DEF"
    return None


def espn_player_fields(obj: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalise a kona entry (``{"id", "player": {...}}``), a roster entry or a bare player dict into
    ``{espn_id, name, first_name, last_name, position, team, pro_team_id, injury_status, injured, on_team_id}``."""
    pl = _player_dict(obj)
    outer = obj if isinstance(obj, Mapping) else {}
    espn_id = pl.get("id")
    if espn_id is None:
        espn_id = outer.get("playerId", outer.get("id"))
    first, last = _str(pl.get("firstName")), _str(pl.get("lastName"))
    name = _str(pl.get("fullName")) or " ".join(x for x in (first, last) if x) or None
    team = espn_team(pl.get("proTeamId")) or dst_team(espn_id)
    status = _str(pl.get("injuryStatus")) or _str(outer.get("injuryStatus"))
    return {
        "espn_id": str(espn_id) if espn_id is not None else None,
        "name": name,
        "first_name": first,
        "last_name": last,
        "position": espn_position(pl),
        "team": team,
        "pro_team_id": _int(pl.get("proTeamId")),
        "injury_status": INJURY_STATUS_MAP.get(status.upper(), status) if status else None,
        "injured": bool(pl.get("injured")) if pl.get("injured") is not None else None,
        "on_team_id": _int(pl.get("onTeamId", outer.get("onTeamId"))),
    }


class EspnIdMap:
    """Resolve ESPN player ids to canonical ids (see the module docstring for the precedence)."""

    def __init__(self, espn_to_pid: Mapping[str, str] | None = None,
                 name_index: Mapping[tuple[str, str], str] | None = None) -> None:
        self.espn_to_pid: dict[str, str] = {str(k): str(v) for k, v in (espn_to_pid or {}).items()}
        #: canonical id -> the ESPN id it is known by (a name match must not steal a player whose own
        #: ESPN id is different: namesakes such as "Kyle Williams" 2010 / 2025 or "Marvin Harrison [Jr.]")
        self.pid_to_espn: dict[str, str] = {v: k for k, v in self.espn_to_pid.items() if _int(k) is not None}
        self.name_index: dict[tuple[str, str], str] = dict(name_index or {})
        self.name_team_index: dict[tuple[str, str, str | None], str] = {}
        self.ambiguous: set[tuple[str, str]] = set()

    # -- construction -----------------------------------------------------------
    @classmethod
    def from_players(cls, players: Mapping[str, Player]) -> "EspnIdMap":
        """Index a universe: ``Player.espn_id``, normalised names and D/ST ids for every ``DEF``."""
        m = cls()
        for pid, pl in players.items():
            m.add_player(str(pid), pl)
        return m

    def add_player(self, pid: str, pl: Player) -> None:
        if pl.espn_id:
            self.espn_to_pid[str(pl.espn_id)] = pid
            self.pid_to_espn[pid] = str(pl.espn_id)
        key = (normalize_name(pl.name), pl.position)
        if key[0]:
            if key in self.name_index and self.name_index[key] != pid:
                self.ambiguous.add(key)
            else:
                self.name_index[key] = pid
            self.name_team_index[(key[0], key[1], pl.team)] = pid
        if pl.position == "DEF":
            team = pl.team or pid
            pro = TEAM_TO_PRO_ID.get(team)
            if pro is not None:
                self.espn_to_pid[str(DST_ID_BASE - pro)] = pid
            self.espn_to_pid.setdefault(team, pid)
            if team == "WAS":
                self.espn_to_pid.setdefault("WSH", pid)

    def register(self, espn_id: Any, pid: str) -> None:
        """Remember ``espn_id -> pid`` (e.g. after a name match) so later lookups are direct and a
        namesake with another ESPN id can no longer claim ``pid`` by name."""
        if espn_id is not None and pid:
            self.espn_to_pid[str(espn_id)] = str(pid)
            self.pid_to_espn.setdefault(str(pid), str(espn_id))

    # -- lookup ----------------------------------------------------------------
    @staticmethod
    def is_synthetic(pid: Any) -> bool:
        """True for ids minted for unknown ESPN players (``espn:<id>``)."""
        return str(pid).startswith(SYNTHETIC_PREFIX)

    @staticmethod
    def synthetic_id(espn_id: Any) -> str:
        return f"{SYNTHETIC_PREFIX}{espn_id}"

    def lookup_name(self, name: str | None, position: str | None, team: str | None = None) -> str | None:
        """Canonical id by normalised ``(name, position)``; ``team`` breaks ties for duplicate names."""
        n = normalize_name(name)
        if not n or not position:
            return None
        key = (n, position)
        if key in self.ambiguous:
            return self.name_team_index.get((n, position, team)) if team else None
        return self.name_index.get(key)

    def resolve(self, espn_id: Any, *, name: str | None = None, position: str | None = None,
                pro_team_id: Any = None) -> str:
        """Canonical id for ``espn_id`` (D/ST ids -> team abbreviation), falling back to ``name`` +
        ``position`` (learned for next time) and finally the synthetic ``espn:<id>``. A name match is
        refused when the candidate is already known by a different ESPN id (a namesake)."""
        key = str(espn_id).strip() if espn_id is not None else None
        if key:
            hit = self.espn_to_pid.get(key)
            if hit:
                return hit
            team = dst_team(key)
            if team:
                return self.espn_to_pid.get(team, team)
        if position == "DEF":
            team = espn_team(pro_team_id)
            if team:
                return self.espn_to_pid.get(team, team)
        team = espn_team(pro_team_id)
        hit = self.lookup_name(name, position, team)
        if hit and key and self.pid_to_espn.get(hit, key) != key:
            log.debug("ESPN id %s (%s) shares its name with %s (ESPN id %s); keeping them apart", key, name, hit,
                      self.pid_to_espn[hit])
            hit = None
        if hit:
            if key:
                self.register(key, hit)
            return hit
        if key:
            return self.synthetic_id(key)
        return self.synthetic_id(normalize_name(name).replace(" ", "-") or "unknown")

    def resolve_player_json(self, obj: Mapping[str, Any] | None) -> str:
        """:meth:`resolve` for a kona entry / roster entry / player dict."""
        f = espn_player_fields(obj)
        return self.resolve(f["espn_id"], name=f["name"], position=f["position"], pro_team_id=f["pro_team_id"])

    def __len__(self) -> int:
        return len(self.espn_to_pid)


def placeholder_player(espn_player_json: Mapping[str, Any] | None, player_id: str | None = None) -> Player:
    """A :class:`Player` for an ESPN player that is not in our universe.

    ``player_id`` defaults to the team abbreviation for a D/ST and ``espn:<id>`` otherwise; name from
    ``fullName`` (``firstName`` + ``lastName``), position from ``defaultPositionId`` / ``eligibleSlots``
    (``"UNK"`` when ESPN gives neither - never a guessed skill position), team from ``proTeamId``,
    ``injury_status`` from ``injuryStatus`` and ``espn_id`` set.
    """
    f = espn_player_fields(espn_player_json)
    pos = f["position"] or "UNK"
    espn_id = f["espn_id"]
    if player_id is None:
        team = dst_team(espn_id) if espn_id else None
        player_id = team if (pos == "DEF" and team) else EspnIdMap.synthetic_id(espn_id or "unknown")
    name = f["name"] or (f"{f['team']} Defense" if pos == "DEF" and f["team"] else f"ESPN player {espn_id}")
    return Player(
        player_id=str(player_id), name=name, position=pos, team=f["team"], fantasy_positions=(pos,),
        injury_status=f["injury_status"], espn_id=espn_id,
        metadata={"platform": "espn", "placeholder": True},
    )
