"""ESPN fantasy football id tables and the statId -> Sleeper scoring-key map.

Everything the rest of :mod:`draftadvisor.espn` needs to translate ESPN's numeric ids
(lineup slots, positions, pro teams, scoring stat ids) into the Sleeper-flavoured labels
and keys used by :mod:`draftadvisor.models` and :mod:`draftadvisor.scoring`.

Sources: the ESPN v3 fantasy API payloads and the ``espn_api`` library's
``football/constant.py`` (``POSITION_MAP``, ``PRO_TEAM_MAP``, ``SETTINGS_SCORING_FORMAT_MAP``).
"""
from __future__ import annotations

import os

__all__ = [
    "DEFAULT_ESPN_BASE_URL",
    "ESPN_BASE_URL",
    "DEFAULT_FAN_API_BASE",
    "FAN_API_URL",
    "espn_base_url",
    "fan_api_url",
    "POSITION_SLOT_MAP",
    "ESPN_SLOT_NAMES",
    "ROSTER_SLOT_ORDER",
    "IR_SLOT_ID",
    "slot_label",
    "PRO_TEAM_MAP",
    "TEAM_TO_PRO_ID",
    "DST_ID_BASE",
    "DEFAULT_POSITION_MAP",
    "ELIGIBLE_SLOT_POSITION",
    "INJURY_STATUS_MAP",
    "ESPN_STAT_TO_SLEEPER",
    "RECEPTION_STAT_IDS",
    "RECEPTION_PREMIUM_SLOTS",
    "TWO_POINT_FALLBACK",
    "DST_STAT_IDS",
    "DEF_KEYS",
    "BRACKET_PARTITIONS",
    "DEF_BRACKET_FAMILIES",
    "STAT_AGGREGATES",
    "STAT_COMPONENTS",
    "DST_DUPLICATE_STAT_IDS",
    "GAMES_PLAYED_STAT_ID",
    "SCORING_LABELS",
    "label_for_stat",
]

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

#: The read-only fantasy API host. The league endpoint is
#: ``{base}/seasons/{season}/segments/0/leagues/{league_id}?view=...``.
DEFAULT_ESPN_BASE_URL = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl"
#: Value at import time; :func:`espn_base_url` re-reads the environment at call time.
ESPN_BASE_URL = os.environ.get("DRAFTADVISOR_ESPN_BASE", DEFAULT_ESPN_BASE_URL).rstrip("/")

#: Optional "my leagues" lookup (needs the user's SWID). Parsed defensively; the shape is undocumented.
DEFAULT_FAN_API_BASE = "https://fan.api.espn.com"
FAN_API_URL = (
    "{base}/apis/v2/fans/{swid}?displayEvents=true&displayNow=true&displayRecs=true&recLimit=5"
    "&context=fantasy&useCookies=true&source=fantasyapp-ios&lang=en&section=espn&region=us"
)


def espn_base_url() -> str:
    """League API base URL; ``DRAFTADVISOR_ESPN_BASE`` overrides it (tests point it at a local stub)."""
    return os.environ.get("DRAFTADVISOR_ESPN_BASE", DEFAULT_ESPN_BASE_URL).rstrip("/")


def fan_api_url(swid: str, base: str | None = None) -> str:
    """Fan API URL for ``swid`` (braces added when missing); ``DRAFTADVISOR_ESPN_FAN_BASE`` overrides the host."""
    s = str(swid).strip()
    if not s.startswith("{"):
        s = "{" + s.strip("{}") + "}"
    b = (base or os.environ.get("DRAFTADVISOR_ESPN_FAN_BASE", DEFAULT_FAN_API_BASE)).rstrip("/")
    return FAN_API_URL.format(base=b, swid=s)


# ---------------------------------------------------------------------------
# Lineup slots (``settings.rosterSettings.lineupSlotCounts`` keys, ``pick.lineupSlotId``,
# ``player.eligibleSlots``) -> our Sleeper-style roster labels. ``None`` = not a draftable slot.
# ---------------------------------------------------------------------------

POSITION_SLOT_MAP: dict[int, str | None] = {
    0: "QB",
    1: "QB",            # TQB (team QB)
    2: "RB",
    3: "WRRB_FLEX",     # RB/WR
    4: "WR",
    5: "REC_FLEX",      # WR/TE
    6: "TE",
    7: "SUPER_FLEX",    # OP (offensive player)
    8: "DL",            # DT
    9: "DL",            # DE
    10: "LB",
    11: "DL",
    12: "DB",           # CB
    13: "DB",           # S
    14: "DB",
    15: "IDP_FLEX",     # DP
    16: "DEF",          # D/ST
    17: "K",
    18: None,           # P
    19: None,           # HC
    20: "BN",           # BE
    21: "IR",
    22: None,
    23: "FLEX",         # RB/WR/TE
    24: "BN",           # ER
    25: "BN",           # Rookie
}

#: ESPN's own slot names (for messages / unmapped-rule labels).
ESPN_SLOT_NAMES: dict[int, str] = {
    0: "QB", 1: "TQB", 2: "RB", 3: "RB/WR", 4: "WR", 5: "WR/TE", 6: "TE", 7: "OP", 8: "DT", 9: "DE", 10: "LB",
    11: "DL", 12: "CB", 13: "S", 14: "DB", 15: "DP", 16: "D/ST", 17: "K", 18: "P", 19: "HC", 20: "BE", 21: "IR",
    22: "", 23: "RB/WR/TE", 24: "ER", 25: "Rookie",
}

#: Order in which :func:`draftadvisor.espn.parsing.roster_positions_from_counts` lists the slots.
ROSTER_SLOT_ORDER: tuple[str, ...] = (
    "QB", "RB", "WR", "TE", "FLEX", "WRRB_FLEX", "REC_FLEX", "SUPER_FLEX", "K", "DEF",
    "DL", "LB", "DB", "IDP_FLEX", "BN", "IR",
)

#: The IR slot does not add draft rounds.
IR_SLOT_ID = 21


def slot_label(slot_id: int | str | None) -> str | None:
    """Our label for an ESPN lineup slot id (``None`` = skip: punter, head coach, unknown)."""
    try:
        return POSITION_SLOT_MAP.get(int(slot_id))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Pro teams (``player.proTeamId``) -> Sleeper abbreviations (ESPN's WSH is Sleeper's WAS).
# ---------------------------------------------------------------------------

PRO_TEAM_MAP: dict[int, str] = {
    1: "ATL", 2: "BUF", 3: "CHI", 4: "CIN", 5: "CLE", 6: "DAL", 7: "DEN", 8: "DET", 9: "GB", 10: "TEN",
    11: "IND", 12: "KC", 13: "LV", 14: "LAR", 15: "MIA", 16: "MIN", 17: "NE", 18: "NO", 19: "NYG", 20: "NYJ",
    21: "PHI", 22: "ARI", 23: "PIT", 24: "LAC", 25: "SF", 26: "SEA", 27: "TB", 28: "WAS", 29: "CAR", 30: "JAX",
    33: "BAL", 34: "HOU",
}

#: Reverse map (both spellings of Washington accepted).
TEAM_TO_PRO_ID: dict[str, int] = {abbr: pid for pid, abbr in PRO_TEAM_MAP.items()}
TEAM_TO_PRO_ID["WSH"] = 28

#: Team defenses have negative player ids: ``DST_ID_BASE - proTeamId`` (``-16023`` = PIT D/ST).
DST_ID_BASE = -16000

#: ``player.defaultPositionId`` -> our position. The skill positions are the ones the model projects; the
#: IDP / punter / head-coach ids keep a real label so they never masquerade as a skill player.
DEFAULT_POSITION_MAP: dict[int, str] = {
    1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "DEF",
    7: "P", 9: "DL", 10: "DL", 11: "LB", 12: "DB", 13: "DB", 14: "HC",
}

#: Position implied by an entry of ``player.eligibleSlots`` (single-position slots only), used when
#: ``defaultPositionId`` is missing or unknown, the way ``espn_api`` derives a player's position.
ELIGIBLE_SLOT_POSITION: dict[int, str] = {
    0: "QB", 2: "RB", 4: "WR", 6: "TE", 16: "DEF", 17: "K",
    8: "DL", 9: "DL", 10: "LB", 11: "DL", 12: "DB", 13: "DB", 14: "DB", 18: "P", 19: "HC",
}

#: ``player.injuryStatus`` -> Sleeper-style ``Player.injury_status`` (``None`` = healthy).
INJURY_STATUS_MAP: dict[str, str | None] = {
    "ACTIVE": None, "NORMAL": None, "PROBABLE": None, "QUESTIONABLE": "Questionable", "DOUBTFUL": "Doubtful",
    "OUT": "Out", "INJURY_RESERVE": "IR", "IR": "IR", "SUSPENSION": "Sus", "SUSPENDED": "Sus", "PUP": "PUP",
    "PHYSICALLY_UNABLE_TO_PERFORM": "PUP", "DAY_TO_DAY": "Questionable", "FIFTEEN_DAY_DL": "Out",
    "SIXTY_DAY_DL": "Out", "SEVEN_DAY_DL": "Out", "NON_FOOTBALL_INJURY": "NA",
}

# ---------------------------------------------------------------------------
# Scoring: ESPN ``scoringItems[].statId`` (and the keys of a projected ``stats`` dict) -> Sleeper keys.
#
# Value = ``(sleeper_keys, divisor)``. ``divisor`` > 1 (or < 1) marks an "every N units" item whose
# points convert to per-unit points as ``points / divisor`` and ADD to the per-unit key (these items
# are ignored when converting stat *values*, where the per-unit stat is already present). Entries
# with several keys either mean "the same value for every key" (80: FG 0-39 covers Sleeper's three
# sub-40 brackets) or "player key + team-defense key" (101/102: ``st_td`` for a returner,
# ``def_st_td`` for a D/ST, the latter taking ``pointsOverrides["16"]``).
# ---------------------------------------------------------------------------

_ONE = 1.0

ESPN_STAT_TO_SLEEPER: dict[int, tuple[tuple[str, ...], float]] = {
    # -- passing
    0: (("pass_att",), _ONE), 1: (("pass_cmp",), _ONE), 2: (("pass_inc",), _ONE), 3: (("pass_yd",), _ONE),
    4: (("pass_td",), _ONE),
    5: (("pass_yd",), 5.0), 6: (("pass_yd",), 10.0), 7: (("pass_yd",), 20.0), 8: (("pass_yd",), 25.0),
    9: (("pass_yd",), 50.0), 10: (("pass_yd",), 100.0),
    11: (("pass_cmp",), 5.0), 12: (("pass_cmp",), 10.0), 13: (("pass_inc",), 5.0), 14: (("pass_inc",), 10.0),
    17: (("bonus_pass_yd_300",), _ONE), 18: (("bonus_pass_yd_400",), _ONE), 19: (("pass_2pt",), _ONE),
    20: (("pass_int",), _ONE), 64: (("pass_sack",), _ONE), 211: (("pass_fd",), _ONE),
    # -- rushing
    23: (("rush_att",), _ONE), 24: (("rush_yd",), _ONE), 25: (("rush_td",), _ONE), 26: (("rush_2pt",), _ONE),
    27: (("rush_yd",), 5.0), 28: (("rush_yd",), 10.0), 29: (("rush_yd",), 20.0), 30: (("rush_yd",), 25.0),
    31: (("rush_yd",), 50.0), 32: (("rush_yd",), 100.0), 33: (("rush_att",), 5.0), 34: (("rush_att",), 10.0),
    37: (("bonus_rush_yd_100",), _ONE), 38: (("bonus_rush_yd_200",), _ONE), 212: (("rush_fd",), _ONE),
    # -- receiving (41 and 53 are both "receptions": 53 wins when both are present, never both)
    41: (("rec",), _ONE), 42: (("rec_yd",), _ONE), 43: (("rec_td",), _ONE), 44: (("rec_2pt",), _ONE),
    47: (("rec_yd",), 5.0), 48: (("rec_yd",), 10.0), 49: (("rec_yd",), 20.0), 50: (("rec_yd",), 25.0),
    51: (("rec_yd",), 50.0), 52: (("rec_yd",), 100.0), 53: (("rec",), _ONE), 54: (("rec",), 5.0),
    55: (("rec",), 10.0), 56: (("bonus_rec_yd_100",), _ONE), 57: (("bonus_rec_yd_200",), _ONE),
    58: (("rec_tgt",), _ONE), 213: (("rec_fd",), _ONE),
    # -- misc / returns
    63: (("fum_rec_td",), _ONE), 68: (("fum",), _ONE), 72: (("fum_lost",), _ONE),
    114: (("kr_yd",), _ONE), 115: (("pr_yd",), _ONE), 116: (("kr_yd",), 10.0), 117: (("kr_yd",), 25.0),
    118: (("pr_yd",), 10.0), 119: (("pr_yd",), 25.0),
    # -- kicking
    74: (("fgm_50p",), _ONE), 76: (("fgmiss_50p",), _ONE), 77: (("fgm_40_49",), _ONE), 79: (("fgmiss_40_49",), _ONE),
    80: (("fgm_0_19", "fgm_20_29", "fgm_30_39"), _ONE), 82: (("fgmiss_0_19", "fgmiss_20_29", "fgmiss_30_39"), _ONE),
    83: (("fgm",), _ONE), 84: (("fga",), _ONE), 85: (("fgmiss",), _ONE), 86: (("xpm",), _ONE), 87: (("xpa",), _ONE),
    88: (("xpmiss",), _ONE), 198: (("fgm_50_59",), _ONE), 200: (("fgmiss_50_59",), _ONE), 201: (("fgm_60p",), _ONE),
    203: (("fgmiss_60p",), _ONE), 214: (("fgm_yds",), _ONE),
    217: (("fgm_yds",), 5.0), 218: (("fgm_yds",), 10.0), 219: (("fgm_yds",), 20.0), 220: (("fgm_yds",), 25.0),
    221: (("fgm_yds",), 50.0), 222: (("fgm_yds",), 100.0),
    # -- team defense / special teams
    89: (("pts_allow_0",), _ONE), 90: (("pts_allow_1_6",), _ONE), 91: (("pts_allow_7_13",), _ONE),
    92: (("pts_allow_14_20",), _ONE),                       # 14-17 (121 covers 18-21)
    93: (("def_td",), _ONE),                                # blocked punt / FG returned for TD
    94: (("def_td",), _ONE),                                # fumble or INT returned for TD (= 103 + 104)
    95: (("int",), _ONE), 96: (("fum_rec",), _ONE), 97: (("blk_kick",), _ONE), 98: (("safe",), _ONE),
    99: (("sack",), _ONE), 100: (("sack",), 0.5),           # "1/2 sack" items: points per half sack
    101: (("st_td", "def_st_td"), _ONE), 102: (("st_td", "def_st_td"), _ONE),
    103: (("def_td",), _ONE), 104: (("def_td",), _ONE),
    105: (("st_td", "def_st_td", "def_td"), _ONE),          # total return TD (= 93 + 94 + 101 + 102), so a
                                                            # 105-only league scores INT / fumble return TDs too
    106: (("ff",), _ONE), 113: (("def_pass_def",), _ONE),
    120: (("pts_allow",), _ONE), 121: (("pts_allow_14_20",), _ONE), 122: (("pts_allow_21_27",), _ONE),
    123: (("pts_allow_28_34",), _ONE), 124: (("pts_allow_35p",), _ONE), 125: (("pts_allow_35p",), _ONE),
    127: (("yds_allow",), _ONE), 128: (("yds_allow_0_100",), _ONE), 129: (("yds_allow_100_199",), _ONE),
    130: (("yds_allow_200_299",), _ONE), 131: (("yds_allow_300_349",), _ONE), 132: (("yds_allow_350_399",), _ONE),
    133: (("yds_allow_400_449",), _ONE), 134: (("yds_allow_450_499",), _ONE), 135: (("yds_allow_500_549",), _ONE),
    136: (("yds_allow_550p",), _ONE),
    187: (("pts_allow",), _ONE), 188: (("pts_allow_0",), _ONE), 189: (("pts_allow_1_6",), _ONE),
    190: (("pts_allow_7_13",), _ONE), 191: (("pts_allow_14_20",), _ONE), 192: (("pts_allow_14_20",), _ONE),
    193: (("pts_allow_21_27",), _ONE), 194: (("pts_allow_28_34",), _ONE), 195: (("pts_allow_35p",), _ONE),
    196: (("pts_allow_35p",), _ONE),
    205: (("def_2pt",), _ONE), 206: (("def_2pt",), _ONE),
}

#: Reception items in preference order (only one of them counts).
RECEPTION_STAT_IDS: tuple[int, ...] = (53, 41)

#: ``pointsOverrides`` slot -> Sleeper per-position reception bonus key (override minus base points).
RECEPTION_PREMIUM_SLOTS: dict[str, str] = {"2": "bonus_rec_rb", "4": "bonus_rec_wr", "6": "bonus_rec_te"}

#: statId 62 "Total 2pt Conversions" feeds each 2-pt key only when its specific item is absent.
TWO_POINT_FALLBACK: dict[str, int] = {"pass_2pt": 19, "rush_2pt": 26, "rec_2pt": 44}

#: D/ST-only scoring items: the effective points are ``pointsOverrides["16"]`` when present.
DST_STAT_IDS: frozenset[int] = frozenset(range(89, 137)) | frozenset(range(187, 198))

#: Sleeper keys that only ever apply to a team defense (they take the D/ST slot override).
DEF_KEYS: frozenset[str] = frozenset(
    {"sack", "int", "ff", "fum_rec", "safe", "blk_kick"}
    | {k for keys, _ in ESPN_STAT_TO_SLEEPER.values() for k in keys
       if k.startswith(("def_", "pts_allow", "yds_allow"))}
)

#: ESPN sub-brackets of one Sleeper points-allowed bracket (14-17 + 18-21 -> ``pts_allow_14_20``, 35-45 + 46+
#: -> ``pts_allow_35p``, and their D/ST-prefixed twins). ESPN omits items scored 0, so the Sleeper weight is the
#: mean over the whole partition (an absent member counts 0.0), not over the members present. This is only
#: right for per-game brackets that partition a Sleeper bracket; the return-TD groups (93 / 94 / 103 / 104,
#: 101 / 102) are not partitions with comparable counts, so absent members stay out of their mean.
BRACKET_PARTITIONS: tuple[tuple[int, ...], ...] = ((92, 121), (191, 192), (124, 125), (195, 196))

#: The two bracket families of a team-defense stat line and the season total each one is derived from.
#: ESPN's projected D/ST lines are degenerate - every game sits in the one bracket holding the season mean
#: - so such a family is dropped from the converted line (when its total is present) and
#: :func:`draftadvisor.projections.blend.sleeper_stats_to_projection` derives per-bracket rates from the
#: total, exactly as it does for Sleeper's own D/ST lines.
DEF_BRACKET_FAMILIES: dict[str, tuple[str, ...]] = {
    "pts_allow": ("pts_allow_0", "pts_allow_1_6", "pts_allow_7_13", "pts_allow_14_20", "pts_allow_21_27",
                  "pts_allow_28_34", "pts_allow_35p"),
    "yds_allow": ("yds_allow_0_100", "yds_allow_100_199", "yds_allow_200_299", "yds_allow_300_349",
                  "yds_allow_350_399", "yds_allow_400_449", "yds_allow_450_499", "yds_allow_500_549",
                  "yds_allow_550p"),
}

#: Projected-stat aggregates whose components must be skipped when the aggregate is present.
STAT_AGGREGATES: dict[int, tuple[int, ...]] = {94: (103, 104), 53: (41,)}
#: Projected-stat aggregates that are skipped when any of their components is present
#: (they lump stats that map to different Sleeper keys).
STAT_COMPONENTS: dict[int, tuple[int, ...]] = {105: (93, 94, 101, 102, 103, 104), 62: (19, 26, 44)}
#: D/ST duplicates of the primary points-allowed ids (skipped when the primary id is present).
DST_DUPLICATE_STAT_IDS: dict[int, int] = {
    187: 120, 188: 89, 189: 90, 190: 91, 191: 92, 192: 121, 193: 122, 194: 123, 195: 124, 196: 125, 206: 205,
}
#: statId 210: games played (projected), exposed as Sleeper's ``gp``.
GAMES_PLAYED_STAT_ID = 210

# ---------------------------------------------------------------------------
# statId -> ESPN's label (``espn_api`` SETTINGS_SCORING_FORMAT_MAP), for "rule not modelled" lists
# ---------------------------------------------------------------------------

SCORING_LABELS: dict[int, str] = {
    0: "Each Pass Attempted",
    1: "Each Pass Completed",
    2: "Each Incomplete Pass",
    3: "Passing Yards",
    4: "TD Pass",
    5: "Every 5 passing yards",
    6: "Every 10 passing yards",
    7: "Every 20 passing yards",
    8: "Every 25 passing yards",
    9: "Every 50 passing yards",
    10: "Every 100 passing yards",
    11: "Every 5 pass completions",
    12: "Every 10 pass completions",
    13: "Every 5 pass incompletions",
    14: "Every 10 pass incompletions",
    15: "40+ yard TD pass bonus",
    16: "50+ yard TD pass bonus",
    17: "300-399 yard passing game",
    18: "400+ yard passing game",
    19: "2pt Passing Conversion",
    20: "Interceptions Thrown",
    21: "Passing Completion Pct",
    22: "Passing Yards Per Game",
    23: "Rushing Attempts",
    24: "Rushing Yards",
    25: "TD Rush",
    26: "2pt Rushing Conversion",
    27: "Every 5 rushing yards",
    28: "Every 10 rushing yards",
    29: "Every 20 rushing yards",
    30: "Every 25 rushing yards",
    31: "Every 50 rushing yards",
    32: "Every 100 rushing yards",
    33: "Every 5 rush attempts",
    34: "Every 10 rush attempts",
    35: "40+ yard TD rush bonus",
    36: "50+ yard TD rush bonus",
    37: "100-199 yard rushing game",
    38: "200+ yard rushing game",
    39: "Rushing Yards Per Attempt",
    40: "Rushing Yards Per Game",
    41: "Receptions",
    42: "Receiving Yards",
    43: "TD Reception",
    44: "2pt Receiving Conversion",
    45: "40+ yard TD rec bonus",
    46: "50+ yard TD rec bonus",
    47: "Every 5 receiving yards",
    48: "Every 10 receiving yards",
    49: "Every 20 receiving yards",
    50: "Every 25 receiving yards",
    51: "Every 50 receiving yards",
    52: "Every 100 receiving yards",
    53: "Each reception",
    54: "Every 5 receptions",
    55: "Every 10 receptions",
    56: "100-199 yard receiving game",
    57: "200+ yard receiving game",
    58: "Receiving Target",
    59: "Receiving Yards After Catch",
    60: "Receiving Yards Per Catch",
    61: "Receiving Yards Per Game",
    62: "Total 2pt Conversions",
    63: "Fumble Recovered for TD",
    64: "Sacked",
    65: "Passing Fumbles",
    66: "Rushing Fumbles",
    67: "Receiving Fumbles",
    68: "Total Fumbles",
    69: "Passing Fumbles Lost",
    70: "Rushing Fumbles Lost",
    71: "Receiving Fumbles Lost",
    72: "Total Fumbles Lost",
    73: "Total Turnovers",
    74: "FG Made (50+ yards)",
    75: "FG Attempted (50+ yards)",
    76: "FG Missed (50+ yards)",
    77: "FG Made (40-49 yards)",
    78: "FG Attempted (40-49 yards)",
    79: "FG Missed (40-49 yards)",
    80: "FG Made (0-39 yards)",
    81: "FG Attempted (0-39 yards)",
    82: "FG Missed (0-39 yards)",
    83: "Total FG Made",
    84: "Total FG Attempted",
    85: "Total FG Missed",
    86: "Each PAT Made",
    87: "Each PAT Attempted",
    88: "Each PAT Missed",
    89: "0 points allowed",
    90: "1-6 points allowed",
    91: "7-13 points allowed",
    92: "14-17 points allowed",
    93: "Blocked Punt or FG return for TD",
    94: "Fumble or INT Return for TD",
    95: "Each Interception",
    96: "Each Fumble Recovered",
    97: "Blocked Punt, PAT or FG",
    98: "Each Safety",
    99: "Each Sack",
    100: "1/2 Sack",
    101: "Kickoff Return TD",
    102: "Punt Return TD",
    103: "Interception Return TD",
    104: "Fumble Return TD",
    105: "Total Return TD",
    106: "Each Fumble Forced",
    107: "Assisted Tackles",
    108: "Solo Tackles",
    109: "Total Tackles",
    110: "Every 3 Total Tackles",
    111: "Every 5 Total Tackles",
    112: "Stuffs",
    113: "Passes Defensed",
    114: "Kickoff Return Yards",
    115: "Punt Return Yards",
    116: "Every 10 kickoff return yards",
    117: "Every 25 kickoff return yards",
    118: "Every 10 punt return yards",
    119: "Every 25 punt return yards",
    120: "Points Allowed",
    121: "18-21 points allowed",
    122: "22-27 points allowed",
    123: "28-34 points allowed",
    124: "35-45 points allowed",
    125: "46+ points allowed",
    126: "Points Allowed Per Game",
    127: "Yards Allowed",
    128: "Less than 100 total yards allowed",
    129: "100-199 total yards allowed",
    130: "200-299 total yards allowed",
    131: "300-349 total yards allowed",
    132: "350-399 total yards allowed",
    133: "400-449 total yards allowed",
    134: "450-499 total yards allowed",
    135: "500-549 total yards allowed",
    136: "550+ total yards allowed",
    137: "Yards Allowed Per Game",
    138: "Net Punts",
    139: "Punt Yards",
    140: "Punts Inside the 10",
    141: "Punts Inside the 20",
    142: "Blocked Punts",
    143: "Punts Returned",
    144: "Punt Return Yards",
    145: "Touchbacks",
    146: "Fair Catches",
    147: "Punt Average",
    148: "Punt Average 44.0+",
    149: "Punt Average 42.0-43.9",
    150: "Punt Average 40.0-41.9",
    151: "Punt Average 38.0-39.9",
    152: "Punt Average 36.0-37.9",
    153: "Punt Average 34.0-35.9",
    154: "Punt Average 33.9 or less",
    155: "Team Win",
    156: "Team Loss",
    157: "Team Tie",
    158: "Points Scored",
    159: "Points Scored Per Game",
    160: "Margin of Victory",
    161: "25+ point Win Margin",
    162: "20-24 point Win Margin",
    163: "15-19 point Win Margin",
    164: "10-14 point Win Margin",
    165: "5-9 point Win Margin",
    166: "1-4 point Win Margin",
    167: "1-4 point Loss Margin",
    168: "5-9 point Loss Margin",
    169: "10-14 point Loss Margin",
    170: "15-19 point Loss Margin",
    171: "20-24 point Loss Margin",
    172: "25+ point Loss Margin",
    173: "Margin of Victory Per Game",
    174: "Winning Pct",
    175: "0-9 yd TD pass bonus",
    176: "10-19 yd TD pass bonus",
    177: "20-29 yd TD pass bonus",
    178: "30-39 yd TD pass bonus",
    179: "0-9 yd TD rush bonus",
    180: "10-19 yd TD rush bonus",
    181: "20-29 yd TD rush bonus",
    182: "30-39 yd TD rush bonus",
    183: "0-9 yd TD rec bonus",
    184: "10-19 yd TD rec bonus",
    185: "20-29 yd TD rec bonus",
    186: "30-39 yd TD rec bonus",
    187: "D/ST Points Allowed",
    188: "D/ST 0 points allowed",
    189: "D/ST 1-6 points allowed",
    190: "D/ST 7-13 points allowed",
    191: "D/ST 14-17 points allowed",
    192: "D/ST 18-21 points allowed",
    193: "D/ST 22-27 points allowed",
    194: "D/ST 28-34 points allowed",
    195: "D/ST 35-45 points allowed",
    196: "D/ST 46+ points allowed",
    197: "D/ST Points Allowed Per Game",
    198: "FG Made (50-59 yards)",
    199: "FG Attempted (50-59 yards)",
    200: "FG Missed (50-59 yards)",
    201: "FG Made (60+ yards)",
    202: "FG Attempted (60+ yards)",
    203: "FG Missed (60+ yards)",
    204: "Offensive 2pt Return",
    205: "Defensive 2pt Return",
    206: "2pt Return",
    207: "Offensive 1pt Safety",
    208: "Defensive 1pt Safety",
    209: "1pt Safety",
    210: "Games Played",
    211: "Passing First Down",
    212: "Rushing First Down",
    213: "Receiving First Down",
    214: "FG Made Yards",
    215: "FG Missed Yards",
    216: "FG Attempt Yards",
    217: "Every 5 FG Made yards",
    218: "Every 10 FG Made yards",
    219: "Every 20 FG Made yards",
    220: "Every 25 FG Made yards",
    221: "Every 50 FG Made yards",
    222: "Every 100 FG Made yards",
    223: "Every 5 FG Missed yards",
    224: "Every 10 FG Missed yards",
    225: "Every 20 FG Missed yards",
    226: "Every 25 FG Missed yards",
    227: "Every 50 FG Missed yards",
    228: "Every 100 FG Missed yards",
    229: "Every 5 FG Attempt yards",
    230: "Every 10 FG Attempt yards",
    231: "Every 20 FG Attempt yards",
    232: "Every 25 FG Attempt yards",
    233: "Every 50 FG Attempt yards",
    234: "Every 100 FG Attempt yards",
}


def label_for_stat(stat_id: int | str | None) -> str:
    """ESPN's label for ``stat_id`` (``"statId <n>"`` when unknown)."""
    try:
        sid = int(stat_id)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return f"statId {stat_id}"
    return SCORING_LABELS.get(sid, f"statId {sid}")
