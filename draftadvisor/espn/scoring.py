"""ESPN scoring settings / projected stat lines -> Sleeper keys.

* :func:`espn_scoring_to_sleeper` turns ``settings.scoringSettings.scoringItems`` into a Sleeper
  ``scoring_settings`` dict (points per unit) plus the list of ESPN rules the advisor cannot model.
* :func:`espn_stats_to_sleeper` turns a projected ``stats`` dict (keyed by statId strings) into a
  Sleeper stat line that :class:`draftadvisor.scoring.ScoringEngine` can score.
* :func:`projected_season_stats` finds ESPN's projected season line for one player.

Conversion rules (see :data:`draftadvisor.espn.constants.ESPN_STAT_TO_SLEEPER`):

* "every N yards/receptions" items (5-14, 27-34, 47-52, 54-55, 116-119, 217-222) become
  ``points / N`` per unit and ADD to the per-unit key (ESPN applies both);
* team-defense keys take ``pointsOverrides["16"]`` (the D/ST slot) when present, else ``points``;
* a ``pointsOverrides`` entry for slot 2 / 4 / 6 on the reception item is a position premium:
  ``bonus_rec_rb`` / ``bonus_rec_wr`` / ``bonus_rec_te`` = override - base;
* reception items 41 and 53 never both count (53 wins);
* statId 62 (total 2-pt conversions) feeds ``pass_2pt`` / ``rush_2pt`` / ``rec_2pt`` only for the
  ones without their own item (19 / 26 / 44);
* when several ESPN items feed one Sleeper key (92 + 121 -> ``pts_allow_14_20``, 93 / 94 / 103 / 104
  -> ``def_td``, 101 / 102 -> ``st_td``) the scoring takes their mean, never their sum; for the
  sub-bracket pairs of :data:`~draftadvisor.espn.constants.BRACKET_PARTITIONS` an absent member (ESPN
  omits 0-point items) counts 0.0 in that mean;
* items with points that have no Sleeper equivalent (40+/50+ yard TD bonuses, per-distance TD
  bonuses, games played, turnovers, ...) are returned as ``unmapped`` with ESPN's label - base points
  and every per-slot override alike, so a rule expressed as ``points 0`` + a slot override is listed too;
* a projected D/ST line whose points- / yards-allowed brackets are degenerate (every game in one
  bracket, the way ESPN projects) loses that bracket family when its season total is present: the
  projection layer derives per-bracket rates from the total instead, as it does for Sleeper's lines.
"""
from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping

from .constants import (
    BRACKET_PARTITIONS,
    DEF_BRACKET_FAMILIES,
    DEF_KEYS,
    DST_DUPLICATE_STAT_IDS,
    DST_STAT_IDS,
    ESPN_SLOT_NAMES,
    ESPN_STAT_TO_SLEEPER,
    GAMES_PLAYED_STAT_ID,
    RECEPTION_PREMIUM_SLOTS,
    RECEPTION_STAT_IDS,
    STAT_AGGREGATES,
    STAT_COMPONENTS,
    TWO_POINT_FALLBACK,
    label_for_stat,
)

log = logging.getLogger(__name__)

__all__ = [
    "espn_scoring_to_sleeper",
    "espn_stats_to_sleeper",
    "projected_season_stats",
    "projected_points",
    "effective_points",
]

_EPS = 1e-12
_DST_SLOT = "16"


def _int(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return None


def _float(v: Any, default: float | None = None) -> float | None:
    if v is None or v == "":
        return default
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if f == f and f not in (float("inf"), float("-inf")) else default


def _overrides(item: Mapping[str, Any]) -> dict[str, float]:
    """``pointsOverrides`` as ``{slot id str: points}`` (non-numeric entries dropped)."""
    out: dict[str, float] = {}
    raw = item.get("pointsOverrides")
    if isinstance(raw, Mapping):
        for k, v in raw.items():
            f = _float(v)
            if f is not None:
                out[str(k)] = f
    return out


def effective_points(item: Mapping[str, Any], key: str | None = None) -> float:
    """Points of one scoring item for Sleeper key ``key``: ``pointsOverrides["16"]`` for team-defense
    keys (and for D/ST-only items when no key is given), else ``points``."""
    base = _float(item.get("points"), 0.0) or 0.0
    ov = _overrides(item)
    sid = _int(item.get("statId"))
    use_dst = key in DEF_KEYS if key is not None else (sid in DST_STAT_IDS)
    if use_dst and _DST_SLOT in ov:
        return ov[_DST_SLOT]
    return base


def _clean(scoring: Mapping[str, float]) -> dict[str, float]:
    """Round away float noise (``0.30000000000000004``) and drop zero weights (= absent)."""
    out: dict[str, float] = {}
    for k, v in scoring.items():
        r = round(float(v), 10)
        if abs(r) > _EPS:
            out[k] = r
    return out


def _slot_name(slot: str) -> str:
    i = _int(slot)
    return (ESPN_SLOT_NAMES.get(i) if i is not None else None) or f"slot {slot}"


def _override_entries(sid: int, base: float, ov: Mapping[str, float], skip: Iterable[str] = ()) -> list[dict]:
    """One ``unmapped`` entry per slot override that differs from ``base`` (the model has no per-position
    points), labelled ``"<label> (<ESPN slot name> only)"``; ``skip`` names the slots handled elsewhere."""
    skip = set(skip)
    return [{"statId": sid, "label": f"{label_for_stat(sid)} ({_slot_name(slot)} only)", "points": pts}
            for slot, pts in ov.items() if slot not in skip and abs(pts - base) > _EPS]


def _unmapped_entries(sid: int, item: Mapping[str, Any]) -> list[dict]:
    """``unmapped`` entries of an item without a Sleeper key: its effective points (the D/ST override for
    D/ST-only ids, else the base) when non-zero, plus every other slot override that differs from the base."""
    base = _float(item.get("points"), 0.0) or 0.0
    eff = effective_points(item)
    out = [{"statId": sid, "label": label_for_stat(sid), "points": eff}] if abs(eff) > _EPS else []
    counted = (_DST_SLOT,) if sid in DST_STAT_IDS else ()
    return out + _override_entries(sid, base, _overrides(item), counted)


def espn_scoring_to_sleeper(scoring_items: Iterable[Mapping[str, Any]] | None) -> tuple[dict[str, float], list[dict]]:
    """``settings.scoringSettings.scoringItems`` -> ``(sleeper scoring_settings, unmapped rules)``.

    ``unmapped`` entries are ``{"statId", "label", "points"}`` for every item with non-zero points that
    has no Sleeper key (base points and per-slot overrides alike), plus per-slot overrides of mapped
    items the model ignores (``label`` then ends with the ESPN slot name in parentheses). Missing items
    are simply absent (= 0.0).
    """
    items: list[tuple[int, Mapping[str, Any]]] = []
    for it in scoring_items or []:
        if not isinstance(it, Mapping):
            continue
        sid = _int(it.get("statId"))
        if sid is not None:
            items.append((sid, it))
    present = {sid for sid, _ in items}
    values: dict[str, list[float]] = {}      # key -> per-item points (averaged)
    additive: dict[str, float] = {}          # key -> summed "every N" per-unit points
    unmapped: list[dict] = []
    winning_rec = next((sid for sid in RECEPTION_STAT_IDS if sid in present), None)

    for sid, it in items:
        base = _float(it.get("points"), 0.0) or 0.0
        ov = _overrides(it)
        if sid in RECEPTION_STAT_IDS and sid != winning_rec:
            continue                                        # never count 41 and 53 both
        if sid == 62:
            for key, specific in TWO_POINT_FALLBACK.items():
                if specific not in present:
                    values.setdefault(key, []).append(base)
            continue
        spec = ESPN_STAT_TO_SLEEPER.get(sid)
        if spec is None:
            unmapped.extend(_unmapped_entries(sid, it))
            continue
        keys, divisor = spec
        for key in keys:
            eff = effective_points(it, key)
            if divisor != 1.0:
                additive[key] = additive.get(key, 0.0) + eff / divisor
            else:
                values.setdefault(key, []).append(eff)
        if sid in RECEPTION_STAT_IDS:
            for slot, key in RECEPTION_PREMIUM_SLOTS.items():
                if slot in ov and abs(ov[slot] - base) > _EPS:
                    values[key] = [ov[slot] - base]
        # per-position overrides the model does not represent (the D/ST override on defense keys and
        # the reception premium slots are handled above)
        handled: list[str] = [_DST_SLOT] if any(k in DEF_KEYS for k in keys) else []
        if sid in RECEPTION_STAT_IDS:
            handled.extend(RECEPTION_PREMIUM_SLOTS)
        unmapped.extend(_override_entries(sid, base, ov, handled))

    # sub-brackets of one Sleeper bracket: the members ESPN left out score 0 and belong in the mean
    for part in BRACKET_PARTITIONS:
        have = sum(1 for sid in part if sid in present)
        if 0 < have < len(part):
            key = ESPN_STAT_TO_SLEEPER[part[0]][0][0]
            values.setdefault(key, []).extend([0.0] * (len(part) - have))

    scoring: dict[str, float] = {k: sum(v) / len(v) for k, v in values.items() if v}
    for k, v in additive.items():
        scoring[k] = scoring.get(k, 0.0) + v
    unmapped.sort(key=lambda d: (d["statId"], d["label"]))
    return _clean(scoring), unmapped


def _pick_key(keys: tuple[str, ...], position: str | None) -> str:
    """Which of a multi-key entry receives a stat *value*: the team-defense key for a D/ST, the
    player key otherwise; same-value bracket entries (80 / 82) use their first key."""
    want_def = position == "DEF"
    for k in keys:
        if (k in DEF_KEYS) == want_def:
            return k
    return keys[0]


def _drop_degenerate_brackets(line: dict[str, float]) -> None:
    """Remove a D/ST bracket family that has every game in one bracket while its season total is present.

    ESPN projects a defense's whole season into the single bracket that holds its mean (``121: 16``), which
    carries no information beyond ``pts_allow`` / ``gp`` and would be scored at the merged Sleeper
    bracket's mean weight; without the keys, the projection layer derives per-bracket rates from the total.
    """
    for total, family in DEF_BRACKET_FAMILIES.items():
        if total not in line:
            continue
        counts = [line[k] for k in family if k in line]
        if counts and sum(1 for c in counts if abs(c) > _EPS) <= 1:
            for k in family:
                line.pop(k, None)


def espn_stats_to_sleeper(stats: Mapping[str, Any] | None, position: str | None = None) -> dict[str, float]:
    """A projected/actual ``stats`` dict keyed by statId strings -> Sleeper stat line.

    Values of several ESPN ids that feed one Sleeper key are summed (92 + 121 -> ``pts_allow_14_20``),
    ESPN's own aggregates are not double counted (94 = 103 + 104; 105 = all return TDs; 62 = all 2-pt
    conversions; 53 = 41), "every N" ids and unmapped ids are ignored, ``210`` becomes ``gp`` and,
    when ESPN only reports 50+ yard kicks (74 / 76), the 50-59 keys are filled from them so the line
    scores correctly under either family of Sleeper kicking keys. A team-defense bracket family with
    every game in one bracket is dropped when its season total is present (see
    :func:`_drop_degenerate_brackets`).
    """
    vals: dict[int, float] = {}
    for k, v in (stats or {}).items():
        sid, f = _int(k), _float(v)
        if sid is not None and f is not None:
            vals[sid] = f
    if not vals:
        return {}
    skip: set[int] = set()
    for agg, comps in STAT_AGGREGATES.items():
        if agg in vals:
            skip.update(comps)
    for agg, comps in STAT_COMPONENTS.items():
        if agg in vals and any(c in vals for c in comps):
            skip.add(agg)
    for dup, primary in DST_DUPLICATE_STAT_IDS.items():
        if dup in vals and primary in vals:
            skip.add(dup)
    out: dict[str, float] = {}
    for sid, v in vals.items():
        if sid in skip:
            continue
        if sid == GAMES_PLAYED_STAT_ID:
            if v > 0:
                out["gp"] = v
            continue
        if sid == 62:                                       # no specific 2-pt ids at all
            key = {"QB": "pass_2pt", "RB": "rush_2pt"}.get(position or "", "rec_2pt")
            out[key] = out.get(key, 0.0) + v
            continue
        spec = ESPN_STAT_TO_SLEEPER.get(sid)
        if spec is None:
            continue
        keys, divisor = spec
        if divisor != 1.0:
            continue
        key = _pick_key(keys, position)
        out[key] = out.get(key, 0.0) + v
    if 198 not in vals and 201 not in vals and "fgm_50p" in out:
        out.setdefault("fgm_50_59", out["fgm_50p"])
    if 200 not in vals and 203 not in vals and "fgmiss_50p" in out:
        out.setdefault("fgmiss_50_59", out["fgmiss_50p"])
    if position == "DEF":
        _drop_degenerate_brackets(out)
    return {k: round(v, 6) for k, v in out.items()}


def _player_dict(obj: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """The ``player`` dict of a kona / roster entry (``{"player": {...}}`` or ``{"playerPoolEntry": {"player": ...}}``),
    or ``obj`` itself when it already is one."""
    if not isinstance(obj, Mapping):
        return {}
    for key in ("playerPoolEntry", "player"):
        inner = obj.get(key)
        if isinstance(inner, Mapping):
            return _player_dict(inner) if key == "playerPoolEntry" else inner
    return obj


def _season_entry(player_json: Mapping[str, Any] | None, season: int, source_id: int) -> Mapping[str, Any] | None:
    """The ``stats[]`` entry for the whole ``season`` from ``statSourceId`` (0 actual, 1 projected)."""
    pl = _player_dict(player_json)
    want_id = f"{source_id}0{season}"
    fallback = None
    for st in pl.get("stats") or []:
        if not isinstance(st, Mapping):
            continue
        if str(st.get("id")) == want_id:
            return st
        if (_int(st.get("statSourceId")) == source_id and _int(st.get("statSplitTypeId")) == 0
                and _int(st.get("seasonId")) == int(season) and _int(st.get("scoringPeriodId") or 0) == 0):
            fallback = fallback or st
    return fallback


def projected_season_stats(player_json: Mapping[str, Any] | None, season: int) -> dict[str, float] | None:
    """ESPN's projected season stat line (``stats[]`` entry ``id == "10<season>"``: ``statSourceId`` 1,
    ``statSplitTypeId`` 0) keyed by statId strings, or ``None`` when the player has none."""
    st = _season_entry(player_json, season, 1)
    if st is None:
        return None
    raw = st.get("stats")
    if not isinstance(raw, Mapping):
        return None
    out: dict[str, float] = {}
    for k, v in raw.items():
        f = _float(v)
        if f is not None:
            out[str(k)] = f
    return out


def projected_points(player_json: Mapping[str, Any] | None, season: int) -> float | None:
    """ESPN's own projected season total (``appliedTotal`` of the projected entry), if any."""
    st = _season_entry(player_json, season, 1)
    return _float(st.get("appliedTotal")) if st is not None else None
