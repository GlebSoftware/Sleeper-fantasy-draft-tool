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

#: Claude research
CLAUDE_MODEL = os.environ.get("DRAFTADVISOR_CLAUDE_MODEL", "claude-sonnet-5")
CLAUDE_ON_CLOCK_TIMEOUT_S = 8.0
CLAUDE_RESEARCH_CONCURRENCY = 4


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def home_dir() -> Path:
    """Root for caches, models and research notes (DRAFTADVISOR_HOME or ./data)."""
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

@dataclass
class Settings:
    """Everything the draft loop needs to know that is not league data."""

    season: int = DEFAULT_SEASON
    league_id: str | None = None
    draft_id: str | None = None
    username: str | None = None          # Sleeper username of the person we advise
    user_id: str | None = None           # or their user_id
    slot: int | None = None              # or their draft slot (1-based)
    poll_seconds: float = SLEEPER_POLL_SECONDS
    anthropic_api_key: str | None = None
    claude_model: str = CLAUDE_MODEL
    use_claude: bool = True              # silently disabled if no key
    # Strategy knobs (see strategy/recommend.py for semantics)
    bench_discount: float = 0.35
    kicker_def_min_round_from_end: int = 3
    stack_bonus: float = 0.02
    bye_penalty: float = 0.015
    risk_aversion: float = 0.0           # >0 penalises std, <0 rewards upside
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        s = cls(
            league_id=os.environ.get("SLEEPER_LEAGUE_ID"),
            draft_id=os.environ.get("SLEEPER_DRAFT_ID"),
            username=os.environ.get("SLEEPER_USERNAME"),
            user_id=os.environ.get("SLEEPER_USER_ID"),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )
        for k, v in overrides.items():
            if v is not None:
                setattr(s, k, v)
        if s.slot is not None:
            s.slot = int(s.slot)
        return s
