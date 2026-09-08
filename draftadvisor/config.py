"""Configuration and constants shared by every layer.

Nothing in here does I/O at import time. Paths are resolved lazily so tests can
point DRAFTADVISOR_HOME at a temporary directory.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Season / positions
# ---------------------------------------------------------------------------

#: The season being drafted for. Override with DRAFTADVISOR_SEASON.
DEFAULT_SEASON: int = int(os.environ.get("DRAFTADVISOR_SEASON", "2026"))

#: Regular-season length used for season-total projections.
REG_SEASON_WEEKS: int = 18
#: Games each team plays (18 weeks, 1 bye).
GAMES_PER_TEAM: int = 17
#: Fantasy regular season usually ends week 14-15; playoffs to 17. Draft value
#: is measured over these weeks (bye-week penalties look at this window too).
FANTASY_WEEKS: int = 17

#: Fantasy-relevant positions, in display order.
SKILL_POSITIONS: tuple[str, ...] = ("QB", "RB", "WR", "TE", "K", "DEF")
OFFENSE_POSITIONS: tuple[str, ...] = ("QB", "RB", "WR", "TE")

#: Sleeper roster slot label -> set of eligible positions.
SLOT_ELIGIBILITY: dict[str, frozenset[str]] = {
    "QB": frozenset({"QB"}),
    "RB": frozenset({"RB"}),
    "WR": frozenset({"WR"}),
    "TE": frozenset({"TE"}),
    "K": frozenset({"K"}),
    "DEF": frozenset({"DEF"}),
    "FLEX": frozenset({"RB", "WR", "TE"}),
    "WRRB_FLEX": frozenset({"RB", "WR"}),
    "REC_FLEX": frozenset({"WR", "TE"}),
    "SUPER_FLEX": frozenset({"QB", "RB", "WR", "TE"}),
    # IDP slots exist in some leagues; we do not project IDP so they are ignored.
    "DL": frozenset(), "LB": frozenset(), "DB": frozenset(), "IDP_FLEX": frozenset(),
    "BN": frozenset(), "IR": frozenset(), "TAXI": frozenset(),
}
NON_STARTING_SLOTS: frozenset[str] = frozenset({"BN", "IR", "TAXI"})

#: Historical seasons available from nflverse for training (inclusive range).
TRAIN_SEASONS: tuple[int, ...] = tuple(range(2019, DEFAULT_SEASON))

#: Sleeper API
SLEEPER_API_BASE = "https://api.sleeper.app/v1"
SLEEPER_API_BASE_V2 = "https://api.sleeper.com"
SLEEPER_POLL_SECONDS = 2.0
SLEEPER_PLAYERS_TTL_HOURS = 24

#: League providers (``Settings.platform``). ESPN drafts are addressed by league id + season (no draft id);
#: the per-poll draft GET is heavier than Sleeper's, hence the slower default cadence.
PLATFORMS: tuple[str, ...] = ("sleeper", "espn")
DEFAULT_PLATFORM = "sleeper"
ESPN_POLL_SECONDS = 3.0

#: Claude (user-initiated chat / ask only: nothing calls Claude automatically). This is the model
#: of the CLI ``ask`` command; the web chat model is ``DRAFTADVISOR_CHAT_MODEL`` (research/claude.py).
CLAUDE_MODEL = os.environ.get("DRAFTADVISOR_CLAUDE_MODEL", "claude-sonnet-5")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def home_dir() -> Path:
    """Root for caches, models and legacy research notes (DRAFTADVISOR_HOME or ./data)."""
    return Path(os.environ.get("DRAFTADVISOR_HOME", Path.cwd() / "data")).expanduser()


def raw_dir() -> Path:
    return home_dir() / "raw"


def cache_dir() -> Path:
    return home_dir() / "cache"


def models_dir() -> Path:
    return home_dir() / "models"


def research_dir() -> Path:
    return home_dir() / "research"


def ensure_dirs() -> None:
    for d in (raw_dir(), cache_dir(), models_dir(), research_dir()):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Runtime settings (CLI flags / env / toml)
# ---------------------------------------------------------------------------

def normalize_platform(value: str | None) -> str:
    """``"sleeper"`` / ``"espn"`` (case-insensitive, default Sleeper); anything else raises ``ValueError``."""
    p = (str(value).strip().lower() if value is not None else "") or DEFAULT_PLATFORM
    if p not in PLATFORMS:
        raise ValueError(f"unknown platform {value!r} (expected one of {', '.join(PLATFORMS)})")
    return p


@dataclass
class Settings:
    """Everything the draft loop needs to know that is not league data.

    ``platform`` selects the league provider: ``"sleeper"`` (``league_id`` / ``draft_id`` / ``username`` /
    ``user_id`` are Sleeper's) or ``"espn"`` (``league_id`` is the ``leagueId=`` number of the ESPN league
    URL, the draft is addressed by ``season``, and "me" is ``team_id`` / ``slot`` / ``username`` = team or
    owner name, else the ``swid`` cookie's team). ``espn_s2`` + ``swid`` are the ESPN cookies of a private
    league (``ESPN_S2`` / ``ESPN_SWID`` in the environment); they are never logged.
    """

    season: int = DEFAULT_SEASON
    platform: str = DEFAULT_PLATFORM     # "sleeper" | "espn"
    league_id: str | None = None
    draft_id: str | None = None          # Sleeper only (ESPN: league + season)
    username: str | None = None          # Sleeper username of the person we advise (ESPN: team / owner name)
    user_id: str | None = None           # or their user_id
    slot: int | None = None              # or their draft slot (1-based)
    team_id: int | None = None           # or (ESPN) their team id
    espn_s2: str | None = None           # ESPN cookies for private leagues (env ESPN_S2 / ESPN_SWID)
    swid: str | None = None
    poll_seconds: float = SLEEPER_POLL_SECONDS
    anthropic_api_key: str | None = None
    claude_model: str = CLAUDE_MODEL     # model of the CLI ``ask`` command (silently disabled if no key)
    # Strategy knobs (see strategy/recommend.py for semantics)
    bench_discount: float = 0.35
    kicker_def_min_round_from_end: int = 3
    stack_bonus: float = 0.02
    bye_penalty: float = 0.015
    risk_aversion: float = 0.0           # >0 penalises std, <0 rewards upside
    extra: dict = field(default_factory=dict)

    @property
    def is_espn(self) -> bool:
        return self.platform == "espn"

    @property
    def has_espn_cookies(self) -> bool:
        return bool(self.espn_s2 and self.swid)

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        """Settings from the environment, then ``overrides`` (``None`` values are ignored).

        Env: ``DRAFTADVISOR_PLATFORM``, ``SLEEPER_LEAGUE_ID`` / ``SLEEPER_DRAFT_ID`` / ``SLEEPER_USERNAME`` /
        ``SLEEPER_USER_ID`` (Sleeper), ``ESPN_LEAGUE_ID`` / ``ESPN_TEAM_ID`` / ``ESPN_S2`` / ``ESPN_SWID``
        (ESPN), ``ANTHROPIC_API_KEY``. Only the platform's own environment ids are read, so a stale
        ``SLEEPER_DRAFT_ID`` cannot leak into an ESPN session (explicit overrides are always kept);
        when the platform is ESPN the poll cadence defaults to :data:`ESPN_POLL_SECONDS`.
        """
        platform = normalize_platform(overrides.get("platform") or os.environ.get("DRAFTADVISOR_PLATFORM"))
        s = cls(
            platform=platform,
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            espn_s2=os.environ.get("ESPN_S2") or None,
            swid=os.environ.get("ESPN_SWID") or None,
        )
        if platform == "espn":
            s.league_id = os.environ.get("ESPN_LEAGUE_ID") or None
            s.team_id = _int_or_none(os.environ.get("ESPN_TEAM_ID"))
            s.poll_seconds = ESPN_POLL_SECONDS
        else:
            s.league_id = os.environ.get("SLEEPER_LEAGUE_ID") or None
            s.draft_id = os.environ.get("SLEEPER_DRAFT_ID") or None
            s.username = os.environ.get("SLEEPER_USERNAME") or None
            s.user_id = os.environ.get("SLEEPER_USER_ID") or None
        for k, v in overrides.items():
            if v is not None and k != "platform":
                setattr(s, k, v)
        if s.slot is not None:
            s.slot = int(s.slot)
        if s.team_id is not None:
            s.team_id = int(s.team_id)
        if s.league_id is not None:
            s.league_id = str(s.league_id).strip() or None
        return s


def _int_or_none(value: str | None) -> int | None:
    if value is None or not str(value).strip():
        return None
    try:
        return int(str(value).strip())
    except ValueError:
        return None
