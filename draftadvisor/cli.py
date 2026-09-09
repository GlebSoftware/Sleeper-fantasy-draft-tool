"""Command line interface (argparse). See DESIGN.md §3.6.

Subcommands: ``prep``, ``train``, ``draft``, ``mock``, ``projections``, ``trade``, ``analyze``,
``ask``, ``ids``, ``web``, ``capture``. Global flags: ``--home DIR``, ``--offline``, ``--season``, ``-v``.

Every command that involves a league takes ``--platform sleeper|espn`` (default Sleeper). For ESPN,
``--league`` is the ``leagueId=`` number of the league URL, the draft is addressed by ``--season``,
"me" is ``--team-id`` / ``--slot`` / ``--username`` (team or owner name) or the team owned by the
``--swid`` cookie, and a private league needs ``--espn-s2`` + ``--swid`` (env ``ESPN_S2`` /
``ESPN_SWID``). The player universe, ECR and the ML projections are shared; ESPN supplies the ADP,
a projected stat line for players Sleeper does not project, and the live draft through
:class:`draftadvisor.espn.poller.EspnDraftPoller` (same interface as the Sleeper poller). ESPN
publishes no pick timestamps: the TUI shows "ESPN clock: N s per pick" and only an approximate,
labelled countdown anchored to the last pick change this process observed.

Only ``ask`` talks to Claude (one request per invocation); the draft loop never does.

The live draft loop (:class:`DraftLoop` / :func:`run_draft_loop`) is written against small
duck-typed interfaces (a poller-like ``run(on_update, stop)`` or an async iterable of states, an
advisor with ``recommend``, a dashboard with ``update``) so it is testable with fakes.
"""
from __future__ import annotations

import argparse
import asyncio
import difflib
import logging
import math
import os
import sys
import time
from typing import Any, Callable, Mapping, Sequence

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from .config import DEFAULT_SEASON, PLATFORMS, SKILL_POSITIONS, TRAIN_SEASONS, Settings
from .models import DraftState, Player, Recommendation

log = logging.getLogger(__name__)

__all__ = ["main", "build_parser", "DraftLoop", "run_draft_loop", "resolve_player_input", "parse_seasons",
           "MIN_POLL_SECONDS", "ESPN_LEAGUE_ID_HELP", "ESPN_COOKIE_HELP"]

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_NETWORK = 3
EXIT_ERROR = 1


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


#: Where to find an ESPN league id / team id when the fan API cannot list them.
ESPN_LEAGUE_ID_HELP = (
    "Find the ESPN league id in the league's URL on fantasy.espn.com: ...?leagueId=NNNNNNN (your team's page adds "
    "teamId=N). Then: draftadvisor capture --platform espn --league NNNNNNN --season YYYY [--team-id N]"
)
#: How to get the cookies of a private ESPN league.
ESPN_COOKIE_HELP = (
    "Private league: pass --espn-s2 VALUE --swid VALUE (or export ESPN_S2 / ESPN_SWID). Find them logged in at "
    "espn.com -> browser dev tools -> Application/Storage -> Cookies -> espn_s2 and SWID (SWID looks like "
    "{XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX}, braces included). They are never logged."
)


def _add_platform_args(p: argparse.ArgumentParser) -> None:
    """``--platform`` + the ESPN cookie flags (shared by every command that names a league)."""
    p.add_argument("--platform", choices=PLATFORMS, default=None,
                   help="league provider: sleeper (default) or espn (env DRAFTADVISOR_PLATFORM)")
    p.add_argument("--espn-s2", dest="espn_s2", default=None, metavar="COOKIE",
                   help="ESPN espn_s2 cookie of a private league (env ESPN_S2)")
    p.add_argument("--swid", default=None, metavar="COOKIE",
                   help="ESPN SWID cookie, {XXXXXXXX-...} (env ESPN_SWID); also identifies your team")


def _add_league_args(p: argparse.ArgumentParser, draft: bool = True) -> None:
    p.add_argument("--league", dest="league_id",
                   help="Sleeper league id; with --platform espn the ESPN league id (leagueId= in the league URL)")
    if draft:
        p.add_argument("--draft", dest="draft_id", help="Sleeper draft id (ESPN drafts are addressed by --league + --season)")
    _add_platform_args(p)


def _add_identity_args(p: argparse.ArgumentParser) -> None:
    g = p.add_mutually_exclusive_group()
    g.add_argument("--username", help="your Sleeper username (ESPN: your team or owner name), case-insensitive")
    g.add_argument("--user-id", dest="user_id", help="your Sleeper user id")
    g.add_argument("--slot", type=int, help="your draft slot (1-based)")
    g.add_argument("--team-id", dest="team_id", type=int, help="your ESPN team id (--platform espn)")


#: Smallest accepted ``--poll`` interval (seconds); anything lower would hammer Sleeper and kill the error backoff.
MIN_POLL_SECONDS = 0.5


def _add_global_args(p: argparse.ArgumentParser, suppress: bool = False) -> None:
    """``--home/--offline/--season/-v``. With ``suppress`` the defaults are omitted so a subparser only
    sets an attribute when the flag is actually given (otherwise its defaults would overwrite the values
    the root parser already parsed)."""
    d = argparse.SUPPRESS if suppress else None
    p.add_argument("--home", default=d, help="data directory (sets DRAFTADVISOR_HOME; default ./data)")
    p.add_argument("--offline", action="store_true", default=argparse.SUPPRESS if suppress else False,
                   help="never touch the network")
    p.add_argument("--season", type=int, default=d, help=f"season being drafted (default {DEFAULT_SEASON})")
    p.add_argument("-v", "--verbose", action="count", default=argparse.SUPPRESS if suppress else 0,
                   help="-v info, -vv debug")


def build_parser() -> argparse.ArgumentParser:
    """The argparse parser for every subcommand.

    The global flags (``--home``, ``--offline``, ``--season``, ``-v``) are accepted both before and
    after the subcommand (``draftadvisor --offline mock`` and ``draftadvisor mock --offline``).
    """
    p = argparse.ArgumentParser(prog="draftadvisor", description="Live fantasy-football draft advisor (Sleeper / ESPN).")
    _add_global_args(p)
    common = argparse.ArgumentParser(add_help=False)
    _add_global_args(common, suppress=True)
    root_sub = p.add_subparsers(dest="command", metavar="command")
    root_sub.required = True

    class _Sub:
        """``add_parser`` that attaches the shared global flags to every subcommand."""

        def add_parser(self, name: str, **kw: Any) -> argparse.ArgumentParser:
            return root_sub.add_parser(name, parents=[common], **kw)

    sub = _Sub()

    s = sub.add_parser("prep", help="download data, train the model, cache projections")
    _add_league_args(s)
    _add_identity_args(s)
    s.add_argument("--refresh", action="store_true", help="re-download data / rebuild caches")
    s.add_argument("--no-train", action="store_true", help="skip model training")
    s.set_defaults(func=cmd_prep)

    s = sub.add_parser("train", help="(re)train the projection model and print the backtest")
    s.add_argument("--seasons", default=None, help="e.g. 2019-2025 or 2020,2021,2022")
    s.add_argument("--refresh", action="store_true")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("draft", help="live draft advisor (Sleeper or ESPN)")
    _add_league_args(s)
    _add_identity_args(s)
    s.add_argument("--poll", type=float, default=None,
                   help=f"poll interval seconds (default 2 Sleeper / 3 ESPN, min {MIN_POLL_SECONDS})")
    s.add_argument("--no-tui", action="store_true", help="plain text output instead of the dashboard")
    s.add_argument("--no-model", action="store_true", help="skip the ML model (offline projections)")
    s.add_argument("--refresh", action="store_true", help="rebuild cached projections")
    s.set_defaults(func=cmd_draft)

    s = sub.add_parser("mock", help="offline mock draft against ADP bots")
    s.add_argument("--teams", type=int, default=12)
    s.add_argument("--rounds", type=int, default=15)
    s.add_argument("--slot", type=int, default=None, help="your draft slot (default: middle of the order)")
    s.add_argument("--scoring", choices=("ppr", "half_ppr", "std"), default="half_ppr")
    s.add_argument("--superflex", action="store_true")
    s.add_argument("--auto", action="store_true", help="always take the top recommendation")
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--speed", type=float, default=0.5, help="seconds between bot picks (0 = instant)")
    s.add_argument("--no-tui", action="store_true", help="plain text instead of the dashboard")
    s.add_argument("--no-model", action="store_true", help="skip the ML model (offline projections)")
    s.add_argument("--refresh", action="store_true", help="rebuild cached projections")
    s.set_defaults(func=cmd_mock)

    s = sub.add_parser("projections", help="projection table")
    _add_league_args(s)
    s.add_argument("--position", choices=SKILL_POSITIONS, default=None)
    s.add_argument("--top", type=int, default=40)
    s.add_argument("--no-model", action="store_true")
    s.add_argument("--refresh", action="store_true")
    s.set_defaults(func=cmd_projections)

    s = sub.add_parser("trade", help="evaluate a trade")
    _add_league_args(s, draft=False)
    s.add_argument("--me", required=True, help="my Sleeper username / display name")
    s.add_argument("--them", required=True, help="their Sleeper username / display name")
    s.add_argument("--give", required=True, help='players I give: "Name, Name"')
    s.add_argument("--get", required=True, help='players I get: "Name, Name"')
    s.add_argument("--no-model", action="store_true")
    s.set_defaults(func=cmd_trade)

    s = sub.add_parser("trades", help="find trades that help me and that the other manager would accept")
    _add_league_args(s, draft=False)
    s.add_argument("--me", required=True, help="my manager / team name")
    s.add_argument("--limit", type=int, default=10, help="how many proposals to print (default 10)")
    s.add_argument("--per-team", type=int, default=2, help="most proposals from any one manager (default 2)")
    s.add_argument("--min-gain", type=float, default=5.0, help="ignore deals worth less than this to me")
    s.add_argument("--min-their-view", type=float, default=-2.0,
                   help="how good the deal must look to them, by consensus value (default -2)")
    s.add_argument("--two-for-one", dest="two_for_one", action="store_true", default=True,
                   help="also search 2-for-1 consolidations (default on)")
    s.add_argument("--one-for-one-only", dest="two_for_one", action="store_false",
                   help="only straight swaps")
    s.add_argument("--no-model", action="store_true")
    s.set_defaults(func=cmd_trades)

    s = sub.add_parser("lineup", help="best starting lineup for a week, and this week's matchup odds (ESPN)")
    _add_league_args(s, draft=False)
    s.add_argument("--me", default=None, help="my team / manager name (default: the --team-id team)")
    s.add_argument("--week", type=int, default=None, help="scoring period (default: the league's current week)")
    s.add_argument("--no-model", action="store_true")
    s.set_defaults(func=cmd_lineup)

    s = sub.add_parser("analyze", help="post-draft roster analysis for every team")
    _add_league_args(s, draft=False)
    s.add_argument("--username", default=None)
    s.add_argument("--no-model", action="store_true")
    s.set_defaults(func=cmd_analyze)

    s = sub.add_parser("ask", help="free-form Claude question with draft context (one paid request)")
    s.add_argument("question")
    _add_league_args(s)
    _add_identity_args(s)
    s.add_argument("--no-model", action="store_true")
    s.set_defaults(func=cmd_ask)

    s = sub.add_parser("ids", help="list your leagues and drafts with ids (Sleeper: --username; ESPN: --swid)")
    s.add_argument("--username", default=None, help="your Sleeper username")
    _add_platform_args(s)
    s.set_defaults(func=cmd_ids)

    s = sub.add_parser("web", help="start the local web app (http://127.0.0.1:8787) and open the browser")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8787)
    s.add_argument("--no-browser", action="store_true")
    s.set_defaults(func=cmd_web)

    s = sub.add_parser("capture", help="one-time league info capture (scoring diff, draft order, your picks; Sleeper or ESPN)")
    _add_league_args(s)
    _add_identity_args(s)
    s.set_defaults(func=cmd_capture)
    return p


def parse_seasons(text: str | None) -> list[int]:
    """"2019-2025" or "2020,2021" -> list of ints (default: TRAIN_SEASONS)."""
    if not text:
        return list(TRAIN_SEASONS)
    if "-" in text:
        lo, hi = text.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in text.split(",") if x.strip()]


def _poll_seconds(value: float | None) -> float | None:
    """Clamp ``--poll`` to :data:`MIN_POLL_SECONDS` (0 / negative would mean a tight loop and no backoff)."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v or v < MIN_POLL_SECONDS:  # NaN or too small
        log.warning("--poll %s is below the minimum; using %.1f s", value, MIN_POLL_SECONDS)
        return MIN_POLL_SECONDS
    return v


def _settings_from_args(args: argparse.Namespace) -> Settings:
    s = Settings.from_env(
        platform=getattr(args, "platform", None),
        league_id=getattr(args, "league_id", None),
        draft_id=getattr(args, "draft_id", None),
        username=getattr(args, "username", None),
        user_id=getattr(args, "user_id", None),
        slot=getattr(args, "slot", None) if getattr(args, "command", "") != "mock" else None,
        team_id=getattr(args, "team_id", None),
        espn_s2=getattr(args, "espn_s2", None),
        swid=getattr(args, "swid", None),
        poll_seconds=_poll_seconds(getattr(args, "poll", None)),
        season=getattr(args, "season", None),
    )
    if s.is_espn and getattr(args, "draft_id", None):
        log.warning("--draft is ignored with --platform espn (an ESPN draft is addressed by --league and --season)")
        s.draft_id = None
    if not s.is_espn:
        given = [f for f, a in (("--team-id", "team_id"), ("--espn-s2", "espn_s2"), ("--swid", "swid"))
                 if getattr(args, a, None) is not None]
        if given:
            log.warning("%s only apply with --platform espn", " / ".join(given))
    return s


class _StderrHandler(logging.StreamHandler):
    """Logs to whatever ``sys.stderr`` is *now*.

    ``logging.basicConfig(stream=sys.stderr)`` would bind the stream object at configuration time;
    the dashboard's ``Live(redirect_stderr=True)`` later swaps ``sys.stderr`` for a proxy that prints
    above the live display, and only a late-bound handler follows it (otherwise poller / client retry
    warnings paint over the TUI).
    """

    def __init__(self) -> None:
        super().__init__(sys.stderr)

    @property
    def stream(self):  # type: ignore[override]
        return sys.stderr

    @stream.setter
    def stream(self, value) -> None:  # the base class assigns it; ignore
        pass


def _configure_logging(verbose: int) -> None:
    level = logging.WARNING if verbose == 0 else (logging.INFO if verbose == 1 else logging.DEBUG)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[_StderrHandler()], force=True)
    logging.getLogger("httpx").setLevel(max(level, logging.WARNING))


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def _espn_error_types() -> tuple:
    """``(EspnAPIError, EspnAccessDenied, EspnNotFound)`` or ``()`` when the ESPN package is unavailable."""
    try:
        from .espn.client import EspnAccessDenied, EspnAPIError, EspnNotFound

        return EspnAPIError, EspnAccessDenied, EspnNotFound
    except Exception:  # noqa: BLE001
        return ()


def _is_access_denied(e: BaseException) -> bool:
    """ESPN 401 / 403: a private league without (valid) ``espn_s2`` + ``SWID`` cookies."""
    types = _espn_error_types()
    return bool(types) and isinstance(e, types[1])


def _is_not_found_error(e: BaseException) -> bool:
    """A Sleeper / ESPN 404 (unknown draft or league id) or any other 4xx client error (not 401 / 403 / 429)."""
    if _is_access_denied(e):
        return False
    try:
        from .sleeper.client import SleeperAPIError, SleeperNotFound

        if isinstance(e, SleeperNotFound):
            return True
        if isinstance(e, SleeperAPIError):
            code = getattr(e, "status_code", None)
            return code is not None and 400 <= int(code) < 500 and int(code) != 429
    except Exception:  # noqa: BLE001
        pass
    types = _espn_error_types()
    if types:
        if isinstance(e, types[2]):
            return True
        if isinstance(e, types[0]):
            code = getattr(e, "status_code", None)
            return code is not None and 400 <= int(code) < 500 and int(code) not in (401, 403, 429)
    return False


def _is_network_error(e: BaseException) -> bool:
    """Transport errors, timeouts, 5xx / 429 from Sleeper or ESPN: things a connection check can fix."""
    if _is_not_found_error(e) or _is_access_denied(e):
        return False
    try:
        import httpx

        if isinstance(e, (httpx.HTTPError, httpx.InvalidURL)):
            return True
    except Exception:  # noqa: BLE001
        pass
    try:
        from .sleeper.client import SleeperAPIError

        if isinstance(e, SleeperAPIError):
            return True
    except Exception:  # noqa: BLE001
        pass
    types = _espn_error_types()
    if types and isinstance(e, types[0]):
        return True
    return isinstance(e, (ConnectionError, TimeoutError, OSError))


def _platform_of(e: BaseException, settings: Settings | None) -> str:
    """Which API an error came from: the exception type first, then the settings."""
    types = _espn_error_types()
    if types and isinstance(e, types[0]):
        return "espn"
    try:
        from .sleeper.client import SleeperAPIError

        if isinstance(e, SleeperAPIError):
            return "sleeper"
    except Exception:  # noqa: BLE001
        pass
    return settings.platform if settings is not None else "sleeper"


def _network_message(e: BaseException, settings: Settings | None = None) -> str:
    if _platform_of(e, settings) == "espn":
        api, host = "ESPN", "lm-api-reads.fantasy.espn.com"
    else:
        api, host = "Sleeper", "api.sleeper.app"
    return (f"Could not reach the {api} API ({type(e).__name__}: {e}).\n"
            f"Check your connection (or that {host} is reachable from here); offline commands still work:\n"
            "  draftadvisor mock --auto        draftadvisor projections        draftadvisor train")


def _not_found_message(e: BaseException, settings: Settings | None = None) -> str:
    url = getattr(e, "url", None)
    url_line = f"  ({url})\n" if url and str(url) not in str(e) else ""
    if _platform_of(e, settings) == "espn":
        league = settings.league_id if settings is not None and settings.league_id else "that id"
        season = settings.season if settings is not None else DEFAULT_SEASON
        return (f"ESPN has no league {league} for season {season} ({type(e).__name__}: {e}).\n" + url_line
                + "Check the leagueId= number in the league URL and --season (a private league answers 401, not 404).\n"
                + ESPN_LEAGUE_ID_HELP)
    ids = []
    if settings is not None:
        if settings.draft_id:
            ids.append(f"draft {settings.draft_id}")
        if settings.league_id:
            ids.append(f"league {settings.league_id}")
    what = " / ".join(ids) if ids else "that draft/league id"
    return (f"Sleeper has no {what} ({type(e).__name__}: {e}).\n" + url_line
            + "Check the id (draft ids and league ids differ): `draftadvisor ids --username U` lists yours.")


def _access_denied_message(e: BaseException, settings: Settings | None = None) -> str:
    league = settings.league_id if settings is not None and settings.league_id else "this"
    lines = [f"ESPN says league {league} is private ({type(e).__name__}: {e})."]
    if settings is not None and settings.has_espn_cookies:
        lines.append("The espn_s2 / SWID cookies given were rejected: they expire after a while (copy fresh ones from "
                     "the browser) and the ESPN account must be a member of the league.")
    else:
        lines.append("Add your espn_s2 and SWID cookies under --espn-s2 / --swid (or export ESPN_S2 / ESPN_SWID).")
    lines.append(ESPN_COOKIE_HELP)
    return "\n".join(lines)


def _report_api_error(e: BaseException, console: Console, settings: Settings | None = None) -> int | None:
    """Print a friendly message for access-denied / not-found / network errors; return the exit code, or None to re-raise."""
    if _is_access_denied(e):
        console.print(_access_denied_message(e, settings), markup=False, highlight=False, style="red")
        return EXIT_USAGE
    if _is_not_found_error(e):
        console.print(_not_found_message(e, settings), markup=False, highlight=False)
        return EXIT_USAGE
    if _is_network_error(e):
        console.print(_network_message(e, settings), markup=False, highlight=False, style="red")
        return EXIT_NETWORK
    return None


def _run(coro):
    """Run a coroutine from the CLI (Ctrl-C friendly)."""
    from .sleeper.client import run_sync

    return run_sync(coro)


# ---------------------------------------------------------------------------
# Draft loop
# ---------------------------------------------------------------------------


class DraftLoop:
    """Glue between a state source, the advisor and the dashboard.

    ``advisor`` needs ``recommend(state)``; ``dashboard`` (optional) ``start/update/refresh/stop``.
    The loop never calls Claude: every update is recommend + render and nothing else (ask
    Claude yourself with ``draftadvisor ask``).
    """

    def __init__(self, advisor: Any, players: Mapping[str, Player], dashboard: Any = None, *,
                 status: dict | None = None, tui: bool = True, out: Callable[[str], None] | None = None,
                 refresh_seconds: float = 1.0) -> None:
        self.advisor = advisor
        self.players = players
        self.dashboard = dashboard if tui else None
        # shared by reference on purpose: the caller (cmd_draft) reads poll health and staleness
        # out of the very dict it passed in
        self.status: dict = status if status is not None else {}
        self.tui = tui and dashboard is not None
        self.out = out or print
        self.refresh_seconds = refresh_seconds
        self.state: DraftState | None = None
        self.rec: Recommendation | None = None
        self.recommend_calls = 0
        self._ticker: asyncio.Task | None = None
        self._last_turn_key: tuple | None = None
        self.source: Any = None

    # -- rendering ----------------------------------------------------------------------
    def render(self) -> None:
        if self.state is None:
            return
        if self.tui and self.dashboard is not None:
            self.dashboard.update(self.state, self.rec, self.players, self.status)
        else:
            from .ui.dashboard import render_text

            self.out(render_text(self.rec, self.state, self.players, self.status))

    def _track_pick_change(self, state: DraftState) -> None:
        """Remember when *this process* saw the pick number advance (a previous state must exist: the
        bootstrap state is not an observed pick). The ESPN clock, which has no timestamps of its own,
        counts approximately from here; Sleeper's real clock (``draft.last_picked``) ignores it."""
        prev = self.state
        if prev is not None and prev.next_pick_no != state.next_pick_no:
            self.status["last_pick_seen_at"] = time.time()

    def _track_turn(self, state: DraftState) -> None:
        if state.is_my_turn:
            key = (state.next_pick_no,)
            if self._last_turn_key != key:
                # Sleeper's clock started at the previous pick; if we noticed the turn late
                # ((re)start mid-turn, slow bootstrap) seed the countdown from that, not from "now"
                from .ui.dashboard import clock_start

                self.status["turn_started_at"] = clock_start(state, time.time())
                self._last_turn_key = key
        else:
            self.status.pop("turn_started_at", None)
            self._last_turn_key = None

    def _pull_source_stats(self, source: Any) -> None:
        """Copy the poller's health into ``status`` (called on every update *and* every tick)."""
        if source is None:
            return
        for attr, key in (("last_latency_ms", "latency_ms"), ("poll_count", "poll_count"), ("last_poll_at", "last_poll_at")):
            if hasattr(source, attr):
                self.status[key] = getattr(source, attr)
        if hasattr(source, "last_error"):
            err = getattr(source, "last_error")
            if err:
                self.status["last_error"] = str(err)
                if self.status.get("stale_since") is None:
                    self.status["stale_since"] = getattr(source, "last_poll_at", None) or time.time()
            elif self.status.get("stale_since") is not None:
                # the poller recovered: clear the outage indicators (set to None rather than popped so a
                # dashboard refresh, which merges status keys, drops them from the screen too)
                self.status["stale_since"] = None
                self.status["last_error"] = None

    def on_source_error(self, e: BaseException) -> None:
        """Poller ``on_error`` hook: surface the outage immediately instead of at the next successful poll."""
        self.status["last_error"] = f"{type(e).__name__}: {e}"
        if self.status.get("stale_since") is None:
            self.status["stale_since"] = getattr(self.source, "last_poll_at", None) or time.time()
        if self.tui and self.dashboard is not None:
            if self.state is not None:
                try:
                    self.dashboard.refresh(self.status)
                except Exception as ex:  # noqa: BLE001
                    log.debug("dashboard refresh failed: %s", ex)
        else:
            self.out(f"poll error: {self.status['last_error']} (board may be stale)")

    # -- state updates -------------------------------------------------------------------
    async def handle(self, state: DraftState, source: Any = None) -> Recommendation | None:
        """Process one state update: recommend and render."""
        self._track_pick_change(state)
        self.state = state
        self._track_turn(state)
        if source is not None:
            self._pull_source_stats(source)
        rec: Recommendation | None = None
        t0 = time.perf_counter()
        try:
            rec = self.advisor.recommend(state)
            self.recommend_calls += 1
            if str(self.status.get("last_error") or "").startswith("recommend:"):
                self.status.pop("last_error", None)
        except Exception as e:  # noqa: BLE001
            log.exception("recommend failed: %s", e)
            self.status["last_error"] = f"recommend: {e}"
        self.status["compute_ms"] = (time.perf_counter() - t0) * 1000.0
        self.rec = rec
        self.render()
        return rec

    # -- ticking ---------------------------------------------------------------------------
    async def _tick(self) -> None:
        while True:
            await asyncio.sleep(self.refresh_seconds)
            if self.tui and self.dashboard is not None and self.state is not None:
                try:
                    self._pull_source_stats(self.source)   # latency / errors / staleness between picks too
                    self.dashboard.refresh(self.status)
                except Exception as e:  # noqa: BLE001
                    log.debug("dashboard refresh failed: %s", e)

    # -- main -------------------------------------------------------------------------------
    async def run(self, source: Any, stop: asyncio.Event | None = None) -> DraftState | None:
        """Consume ``source`` until the draft completes / ``stop`` is set / Ctrl-C."""
        self.source = source
        if self.tui and self.dashboard is not None:
            self.dashboard.start()
        if self.tui and self.refresh_seconds > 0:
            self._ticker = asyncio.ensure_future(self._tick())
        try:
            if hasattr(source, "run") and callable(source.run):
                async def on_update(state: DraftState) -> None:
                    await self.handle(state, source)

                if _accepts_kwarg(source.run, "on_error"):
                    await source.run(on_update, stop, on_error=self.on_source_error)
                else:                       # a source without the poller's ``on_error`` hook
                    await source.run(on_update, stop)
            else:
                async for state in _aiter(source):
                    if stop is not None and stop.is_set():
                        break
                    await self.handle(state, source)
                    if state.is_complete:
                        break
        finally:
            self._shutdown()
        return self.state

    def _shutdown(self) -> None:
        """Stop the ticker and close the TUI."""
        if self._ticker is not None and not self._ticker.done():
            self._ticker.cancel()
        if self.tui and self.dashboard is not None:
            self.dashboard.stop()


def _accepts_kwarg(fn: Any, name: str) -> bool:
    try:
        import inspect

        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


async def _aiter(source: Any):
    if hasattr(source, "__aiter__"):
        async for s in source:
            yield s
    else:
        for s in source:
            yield s


async def run_draft_loop(source: Any, advisor: Any, dashboard: Any = None, *, players: Mapping[str, Player],
                         status: dict | None = None, tui: bool = True, stop: asyncio.Event | None = None,
                         out: Callable[[str], None] | None = None, refresh_seconds: float = 1.0) -> DraftLoop:
    """Convenience wrapper: build a :class:`DraftLoop`, run it, return it (for inspection)."""
    loop = DraftLoop(advisor, players, dashboard, status=status, tui=tui, out=out, refresh_seconds=refresh_seconds)
    await loop.run(source, stop)
    return loop


# ---------------------------------------------------------------------------
# Player name resolution (mock input, trade)
# ---------------------------------------------------------------------------


def resolve_player_input(text: str, candidates: Sequence[Player]) -> Player | None:
    """Exact player_id, exact name (case-insensitive), then fuzzy / substring name match."""
    q = text.strip()
    if not q:
        return None
    by_id = {p.player_id: p for p in candidates}
    if q in by_id:
        return by_id[q]
    ql = q.lower()
    exact = [p for p in candidates if p.name.lower() == ql]
    if exact:
        return exact[0]
    subs = [p for p in candidates if ql in p.name.lower()]
    if len(subs) == 1:
        return subs[0]
    if subs:
        return subs[0]
    names = [p.name for p in candidates]
    hits = difflib.get_close_matches(q, names, n=1, cutoff=0.6)
    if hits:
        return next(p for p in candidates if p.name == hits[0])
    return None


def _split_names(text: str) -> list[str]:
    return [t.strip() for t in text.replace(";", ",").split(",") if t.strip()]


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def _projection_table(ctx, top: int = 30, position: str | None = None, title: str | None = None) -> Table:
    t = Table(title=title or "Projections", box=box.SIMPLE_HEAD)
    for col, kw in (("#", {"justify": "right"}), ("Player", {}), ("Pos", {}), ("Tm", {}), ("Bye", {"justify": "right"}),
                    ("Pts", {"justify": "right"}), ("Floor", {"justify": "right"}), ("Ceil", {"justify": "right"}),
                    ("Std", {"justify": "right"}), ("PPG", {"justify": "right"}), ("G", {"justify": "right"}),
                    ("ADP", {"justify": "right"}), ("ECR", {"justify": "right"}), ("Src", {}), ("Flags", {})):
        t.add_column(col, **kw)
    rows = [(pid, pr) for pid, pr in ctx.projections.items() if pid in ctx.players]
    if position:
        rows = [(pid, pr) for pid, pr in rows if pr.position == position]
    rows.sort(key=lambda r: -r[1].points)
    for i, (pid, pr) in enumerate(rows[:top], start=1):
        pl = ctx.players[pid]
        src = "+".join(k for k, w in sorted(pr.weights.items(), key=lambda kv: -kv[1]) if w > 0) or "-"
        t.add_row(str(i), pl.name, pl.position, pl.team or "FA", str(pl.bye_week or "-"), f"{pr.points:.0f}",
                  f"{pr.floor:.0f}", f"{pr.ceiling:.0f}", f"{pr.std:.0f}", f"{pr.ppg:.1f}", f"{pr.games:.0f}",
                  f"{pl.adp:.0f}" if pl.adp else "-", f"{pl.ecr:.0f}" if pl.ecr else "-", src, ",".join(pr.flags))
    return t


def _metrics_table(metrics: Mapping[str, Any]) -> Table | None:
    rows = [(pos, metrics.get(pos)) for pos in SKILL_POSITIONS if isinstance(metrics.get(pos), Mapping)]
    if not rows:
        return None
    t = Table(title="Backtest (PPR ppg, players with >= 6 games)", box=box.SIMPLE_HEAD)
    for col in ("Pos", "n", "MAE model", "MAE last", "MAE career", "Sp model", "Sp last", "beats last"):
        t.add_column(col, justify="right" if col not in ("Pos", "beats last") else "left")
    for pos, r in rows:
        def f(k: str) -> str:
            v = r.get(k)
            return "-" if v is None else f"{float(v):.3f}"
        t.add_row(pos, str(r.get("n", "-")), f("mae_model"), f("mae_last"), f("mae_career"), f("spearman_model"),
                  f("spearman_last"), "yes" if r.get("beats_last") else "no")
    return t


def _print_sources(console: Console, ctx) -> None:
    s = ctx.sources
    console.print(f"[dim]players: {s.get('players')} ({len(ctx.players)}) • ECR: {'yes' if s.get('ecr') else 'no'} • "
                  f"ADP: {s.get('adp')} • projections: {s.get('projections')} ({len(ctx.projections)}) • "
                  f"notes: {s.get('notes')} • {s.get('total_ms', 0):.0f} ms[/dim]")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _espn_client(settings: Settings):
    """An :class:`draftadvisor.espn.client.EspnClient` with the session's cookies (the base URL honours
    ``DRAFTADVISOR_ESPN_BASE``, which the tests point at a local stub)."""
    from .espn.client import EspnClient

    return EspnClient(espn_s2=settings.espn_s2, swid=settings.swid)


def _league_for(settings: Settings, args: argparse.Namespace):
    """``(league, draft, extra)`` from the platform's API when ids are given and online, else from a
    snapshot, else ``(None, None, {})``. ``extra`` carries ESPN inputs for :func:`_build`
    (``espn_players`` for :func:`draftadvisor.app.build_context`, ``espn_league_json`` for the rosters);
    a saved ESPN capture supplies both too, so offline rosters keep their names / ADP / projections."""
    from .app import fetch_espn_league, fetch_league, load_saved_snapshot

    if not (settings.league_id or settings.draft_id):
        return None, None, {}
    if not getattr(args, "offline", False):
        try:
            if settings.is_espn:
                async def go_espn():
                    async with _espn_client(settings) as client:
                        return await fetch_espn_league(client, settings.league_id, settings.season)

                league, draft, league_json, kona = _run(go_espn())
                return league, draft, {"espn_players": kona, "espn_league_json": league_json}
            from .sleeper.client import SleeperClient

            async def go():
                async with SleeperClient() as client:
                    return await fetch_league(client, settings.league_id, settings.draft_id)

            league, draft = _run(go())
            if league is not None:
                return league, draft, {}
        except Exception as e:  # noqa: BLE001
            if settings.is_espn and (_is_access_denied(e) or _is_not_found_error(e)):
                raise                       # wrong id / missing cookies: a snapshot would hide the real problem
            log.warning("league lookup failed (%s); trying snapshot", e)
    snap = load_saved_snapshot(settings)
    if snap is None:
        return None, None, {}
    extra: dict = {}
    if settings.is_espn:
        raw = snap.raw or {}
        extra = {"espn_players": list(raw.get("players") or []), "espn_league_json": dict(raw.get("league") or {})}
    return snap.league, snap.draft, extra


def _build(settings: Settings, args: argparse.Namespace, league=None, draft=None, **kw):
    from .app import build_context

    extra: dict = {}
    if league is None:
        league, draft, extra = _league_for(settings, args)
    league_json = extra.pop("espn_league_json", None)
    ctx = build_context(settings, league, draft, offline=getattr(args, "offline", False),
                        use_model=not getattr(args, "no_model", False), refresh=getattr(args, "refresh", False),
                        **extra, **kw)
    if settings.is_espn:
        ctx.sources["espn_inputs"] = {"league_json": league_json, "players": list(extra.get("espn_players") or [])}
    return ctx


def cmd_prep(args: argparse.Namespace) -> int:
    from .app import prep

    console = Console()
    settings = _settings_from_args(args)
    if args.offline:
        console.print("[dim]--offline: Sleeper will not be contacted[/dim]")
    progress = lambda s: console.print(Text("… ", style="cyan") + Text(str(s)))  # noqa: E731
    kw = dict(refresh=args.refresh, train=not args.no_train, progress=progress)
    try:
        try:
            ctx = _run(prep(settings, offline=args.offline, **kw))
        except TypeError as e:
            if "offline" not in str(e):
                raise
            ctx = _run(prep(settings, **kw))    # older app.prep without the offline parameter
    except ValueError as e:                     # ESPN: unknown --username / --team-id / --slot (the message lists the teams)
        if not settings.is_espn:
            raise
        console.print(str(e), style="red", markup=False, highlight=False)
        return EXIT_USAGE
    snap = ctx.sources.get("snapshot")
    if snap is not None:
        try:
            from .capture import print_snapshot

            print_snapshot(snap, console)
        except Exception as e:  # noqa: BLE001
            log.warning("could not print snapshot: %s", e)
    mt = _metrics_table(ctx.sources.get("metrics") or {})
    if mt is not None:
        console.print(mt)
    console.print(_projection_table(ctx, 30, title=f"Top 30 — {ctx.league.name}"))
    _print_sources(console, ctx)
    return EXIT_OK


def cmd_train(args: argparse.Namespace) -> int:
    console = Console()
    seasons = parse_seasons(args.seasons)
    try:
        from .projections.model import train_and_save

        _, metrics = train_and_save(seasons, refresh=args.refresh)
    except NotImplementedError as e:
        console.print(f"[red]training not available: {e}[/red]")
        return EXIT_ERROR
    mt = _metrics_table(metrics)
    if mt is not None:
        console.print(mt)
    console.print(f"model saved to {metrics.get('_path')} • fit {metrics.get('_fit_seconds') or 0:.0f}s • "
                  f"total {metrics.get('_total_seconds') or 0:.0f}s")
    return EXIT_OK


async def _resolve_draft_id(client, settings: Settings, console: Console) -> str | None:
    """Pick the draft id from --draft / --league / --username (interactive list)."""
    if settings.draft_id:
        return settings.draft_id
    from .capture import _pick_draft

    if settings.league_id:
        drafts = await client.get_league_drafts(settings.league_id)
        chosen = _pick_draft(drafts, None, settings.season)
        if chosen is None:
            league = await client.get_league(settings.league_id)
            return league.get("draft_id")
        return str(chosen["draft_id"])
    if settings.username or settings.user_id:
        user = await client.get_user(settings.username or settings.user_id)
        drafts = await client.get_user_drafts(user["user_id"], settings.season)
        if not drafts:
            console.print("no drafts found for this user/season")
            return None
        t = Table(title="Your drafts", box=box.SIMPLE_HEAD)
        for col in ("#", "draft_id", "league_id", "status", "type", "name"):
            t.add_column(col)
        for i, d in enumerate(drafts, start=1):
            t.add_row(str(i), str(d.get("draft_id")), str(d.get("league_id")), str(d.get("status")), str(d.get("type")),
                      str((d.get("metadata") or {}).get("name") or ""))
        console.print(t)
        if len(drafts) == 1:
            return str(drafts[0]["draft_id"])
        raw = input("Which draft? [1] ").strip() or "1"
        try:
            return str(drafts[int(raw) - 1]["draft_id"])
        except (ValueError, IndexError):
            console.print("[red]invalid choice[/red]")
            return None
    console.print("[red]draft: give --draft ID, --league ID or --username U[/red]")
    return None


def _announce_slot(console: Console, state: DraftState, hint: str) -> None:
    if state.my_slot is None:
        console.print(f"[yellow]could not resolve your slot ({hint}); observer mode[/yellow]")
    else:
        console.print(f"you are slot {state.my_slot} ({state.slot_label(state.my_slot)}), picks "
                      + ", ".join(f"#{p}" for p in state.my_pick_numbers()[:4]) + " …")


async def _sleeper_inputs(settings: Settings, console: Console) -> tuple[Any, Any]:
    """``(players payload, season projections payload)`` from Sleeper: the player universe of every
    platform. ``(None, None)`` when Sleeper is unreachable (the offline roster universe is used then)."""
    try:
        from .sleeper.client import SleeperClient

        console.print("[cyan]…[/cyan] loading Sleeper players & projections (the player universe)")
        async with SleeperClient() as client:
            players = await client.get_players()
            proj = await client.get_season_projections(settings.season)
            return players, proj
    except Exception as e:  # noqa: BLE001
        log.warning("Sleeper players/projections unavailable (%s); using the offline universe", e)
        return None, None


async def _draft_espn_async(args: argparse.Namespace, settings: Settings, console: Console) -> int:
    """The ESPN live draft: capture (settings / teams / draft / player pool), context, then
    :class:`~draftadvisor.espn.poller.EspnDraftPoller` through the same :class:`DraftLoop`."""
    from .app import build_context
    from .capture import print_snapshot
    from .espn.capture import capture_espn_league, espn_names
    from .espn.ids import EspnIdMap
    from .espn.parsing import roster_names
    from .espn.poller import EspnDraftPoller
    from .ui.dashboard import Dashboard

    if not settings.league_id:
        console.print("[red]draft --platform espn needs --league ID (the leagueId= number in the ESPN league URL)[/red]")
        return EXIT_USAGE
    async with _espn_client(settings) as client:
        console.print(f"[cyan]…[/cyan] connecting to ESPN league {settings.league_id} (season {settings.season})")
        try:
            snap = await capture_espn_league(client, settings.league_id, settings.season, swid=settings.swid,
                                             team_id=settings.team_id, slot=settings.slot, username=settings.username,
                                             save=True)
        except ValueError as e:                     # unknown --username / --team-id: the message lists the teams
            console.print(str(e), style="red", markup=False, highlight=False)
            return EXIT_USAGE
        print_snapshot(snap, console)
        league_json = dict(snap.raw.get("league") or {})
        kona = list(snap.raw.get("players") or [])
        sleeper_players, sleeper_proj = await _sleeper_inputs(settings, console)
        console.print("[cyan]…[/cyan] building projections")
        ctx = build_context(settings, snap.league, snap.draft, sleeper_players=sleeper_players, sleeper_proj=sleeper_proj,
                            espn_players=kona, refresh=args.refresh, use_model=not args.no_model, quiet=True)
        _print_sources(console, ctx)
        names: dict[str, Any] = dict(roster_names(league_json))
        names.update(espn_names(kona))
        poller = EspnDraftPoller(client, settings.league_id, settings.season, id_map=EspnIdMap.from_players(ctx.players),
                                 names=names, swid=settings.swid, team_id=settings.team_id, slot=settings.slot,
                                 username=settings.username, settings=settings, league_json=league_json)
        state = await poller.bootstrap()
        _announce_slot(console, state, "use --team-id/--slot/--username, or --swid so your team is recognised")
        if state.draft.status == "pre_draft" and not state.draft.draft_order:
            console.print("[yellow]draft order not set yet; your slot and pick numbers appear once the commissioner "
                          "sets the order / the draft starts[/yellow]")
        dashboard = None if args.no_tui else Dashboard(console, projections=ctx.projections)
        clock = f"ESPN clock: {state.draft.pick_timer} s per pick" if state.draft.pick_timer else "ESPN clock: no pick timer"
        status = {"mode": "live", "platform": "espn", "projections": ctx.projections,
                  "hint": f"Ctrl-C to quit • {clock} (ESPN gives no pick timestamps: any countdown is approximate) "
                          "• the board refreshes on every poll"}
        loop = DraftLoop(ctx.advisor, ctx.players, dashboard, status=status, tui=not args.no_tui,
                         out=_plain_printer(console))
        final = await loop.run(poller)
        if final is not None and final.is_complete:
            console.print("[green]Draft complete.[/green]")
            _print_final_rosters(console, final, ctx)
    return EXIT_OK


async def _draft_async(args: argparse.Namespace, settings: Settings, console: Console) -> int:
    if settings.is_espn:
        return await _draft_espn_async(args, settings, console)
    from .app import build_context
    from .sleeper.client import SleeperClient
    from .sleeper.poller import DraftPoller
    from .ui.dashboard import Dashboard

    async with SleeperClient() as client:
        draft_id = await _resolve_draft_id(client, settings, console)
        if not draft_id:
            return EXIT_USAGE
        settings.draft_id = draft_id
        poller = DraftPoller(client, draft_id, league_id=settings.league_id, settings=settings)
        console.print(f"[cyan]…[/cyan] connecting to draft {draft_id}")
        state = await poller.bootstrap()
        league = state.league
        if league is not None:
            settings.league_id = settings.league_id or league.league_id
        _announce_slot(console, state, "use --username/--user-id/--slot")
        try:
            from .capture import capture_league, print_snapshot

            snap = await capture_league(client, settings.league_id, draft_id, username=settings.username,
                                        user_id=settings.user_id, slot=settings.slot, season=settings.season)
            print_snapshot(snap, console)
        except Exception as e:  # noqa: BLE001
            log.warning("league capture skipped: %s", e)
        sleeper_players = sleeper_proj = None
        try:
            console.print("[cyan]…[/cyan] loading Sleeper players & projections")
            sleeper_players = await client.get_players()
            sleeper_proj = await client.get_season_projections(settings.season)
        except Exception as e:  # noqa: BLE001
            log.warning("Sleeper players/projections unavailable: %s", e)
        console.print("[cyan]…[/cyan] building projections")
        ctx = build_context(settings, league, state.draft, sleeper_players=sleeper_players, sleeper_proj=sleeper_proj,
                            refresh=args.refresh, use_model=not args.no_model, quiet=True)
        _print_sources(console, ctx)
        dashboard = None if args.no_tui else Dashboard(console, projections=ctx.projections)
        status = {"mode": "live", "platform": "sleeper", "projections": ctx.projections,
                  "hint": "Ctrl-C to quit • the board refreshes on every pick"}
        loop = DraftLoop(ctx.advisor, ctx.players, dashboard, status=status, tui=not args.no_tui,
                         out=_plain_printer(console))
        final = await loop.run(poller)
        if final is not None and final.is_complete:
            console.print("[green]Draft complete.[/green]")
            _print_final_rosters(console, final, ctx)
    return EXIT_OK


def _plain_printer(console: Console) -> Callable[[str], None]:
    """Print plain text: no rich markup / highlighting (names and notes may contain ``[brackets]``)."""
    return lambda s: console.print(str(s), markup=False, highlight=False)


def cmd_draft(args: argparse.Namespace) -> int:
    console = Console()
    settings = _settings_from_args(args)
    if args.offline:
        console.print("[red]draft needs the network (drop --offline); try `draftadvisor mock` offline.[/red]")
        return EXIT_USAGE
    try:
        return _run(_draft_async(args, settings, console))
    except KeyboardInterrupt:
        console.print("\nstopped.")
        return EXIT_OK
    except Exception as e:  # noqa: BLE001
        rc = _report_api_error(e, console, settings)
        if rc is not None:
            return rc
        raise


def _print_final_rosters(console: Console, state: DraftState, ctx, my_slot: int | None = None) -> None:
    """Every team's projected starting-lineup points, ranked."""
    try:
        from .strategy.lineup import roster_summary
    except Exception as e:  # noqa: BLE001
        log.warning("lineup module unavailable: %s", e)
        return
    my_slot = my_slot if my_slot is not None else state.my_slot
    rows = []
    for slot in range(1, state.teams + 1):
        s = roster_summary(state, slot, ctx.players, ctx.projections, ctx.league)
        rows.append((s.lineup_points, s.bench_points, slot, s.label, s))
    rows.sort(key=lambda r: -r[0])
    t = Table(title="Projected starting lineups", box=box.SIMPLE_HEAD)
    for col in ("Rank", "Slot", "Team", "Lineup pts", "Bench pts", "Open starters"):
        t.add_column(col, justify="right" if "pts" in col or col in ("Rank", "Slot") else "left")
    my_rank = None
    for rank, (lp, bp, slot, label, s) in enumerate(rows, start=1):
        mine = slot == my_slot
        if mine:
            my_rank = rank
        open_ = " ".join(k for k, v in s.open_starters.items() if v > 0) or "-"
        t.add_row(str(rank), str(slot), label + (" (you)" if mine else ""), f"{lp:.0f}", f"{bp:.0f}", open_,
                  style="bold green" if mine else "")
    console.print(t)
    if my_rank is not None:
        n = len(rows)
        grade = "A" if my_rank <= max(1, n // 6) else "B" if my_rank <= n // 3 else "C" if my_rank <= 2 * n // 3 else "D"
        console.print(f"Your lineup ranks [bold]{my_rank}/{n}[/bold] (grade {grade}).")


def cmd_mock(args: argparse.Namespace) -> int:
    from .app import build_context
    from .mock.simulator import MockDraft, make_mock_draft, make_mock_league
    from .ui.dashboard import Dashboard, render_text

    console = Console()
    settings = _settings_from_args(args)
    if args.slot is None:
        args.slot = (args.teams + 1) // 2
    if not 1 <= args.slot <= args.teams:
        console.print(f"[red]--slot must be in 1..{args.teams}[/red]")
        return EXIT_USAGE
    league = make_mock_league(args.teams, args.rounds, args.scoring, args.superflex)
    league.league_id = f"mock_{args.scoring}{'_sf' if args.superflex else ''}"
    draft = make_mock_draft(league, args.slot, args.teams, args.rounds)
    console.print("[cyan]…[/cyan] building projections (cached after the first run)")
    ctx = build_context(settings, league, draft, offline=True, refresh=args.refresh, use_model=not args.no_model,
                        quiet=True)
    _print_sources(console, ctx)
    mock = MockDraft(ctx.players, league, draft, args.slot, seed=args.seed)
    advisor = ctx.advisor
    dashboard = None if args.no_tui else Dashboard(console, projections=ctx.projections)
    status = {"mode": "mock", "platform": "mock", "projections": ctx.projections,
              "hint": "Enter = take the top recommendation • type a name or player_id • q to quit"}
    quit_ = False
    try:
        while not mock.is_complete and not quit_:
            if not mock.is_my_turn:
                pick = mock.bot_pick()
                pl = ctx.players.get(pick.player_id)
                if not args.auto or args.verbose:
                    console.print(Text(f"#{pick.pick_no:>3} R{pick.round:<2} {mock.state().slot_label(pick.draft_slot):<8} "
                                       f"{pl.name if pl else pick.player_name} ({pick.position})", style="dim"))
                if args.speed > 0 and not args.auto:
                    time.sleep(args.speed)
                continue
            state = mock.state()
            rec = advisor.recommend(state)
            top = rec.top_pick
            if args.auto:
                pl = top.player if top else mock.available()[0]
                pick = mock.make_pick(pl.player_id)
                why = "; ".join(top.reasons[:2]) if top else "best available"
                console.print(Text(f"#{pick.pick_no:>3} R{pick.round:<2} You      {pl.name} ({pl.position}) — {why}", style="bold"))
                if args.verbose:
                    console.print(render_text(rec, state, ctx.players, status), markup=False, highlight=False)
                continue
            status["turn_started_at"] = time.time()
            if dashboard is not None:
                console.print(dashboard.build(state, rec, ctx.players, status))
            else:
                console.print(render_text(rec, state, ctx.players, status), markup=False, highlight=False)
            while True:
                try:
                    raw = input(f"Pick #{state.next_pick_no} (Enter = {top.player.name if top else 'best available'}): ")
                except EOFError:
                    raw = ""
                    quit_ = True
                if raw.strip().lower() in ("q", "quit", "exit"):
                    quit_ = True
                    break
                if not raw.strip():
                    pl = top.player if top else mock.available()[0]
                else:
                    pl = resolve_player_input(raw, mock.available()[:600])
                    if pl is None:
                        console.print("[red]no available player matches; try again[/red]")
                        continue
                try:
                    pick = mock.make_pick(pl.player_id)
                except ValueError as e:
                    console.print(f"[red]{e}[/red]")
                    continue
                console.print(Text(f"You take {pl.name} ({pl.position}, {pl.team or 'FA'}) at #{pick.pick_no}", style="green"))
                if top is not None and pl.player_id != top.player_id:
                    try:
                        console.print(Text(str(advisor.explain_pick(state, pl.player_id)), style="dim"))
                    except Exception as e:  # noqa: BLE001
                        log.debug("explain_pick failed: %s", e)
                break
    except KeyboardInterrupt:
        console.print("\nstopped.")
        return EXIT_OK
    final = mock.state()
    if final.is_complete:
        console.print("[green]Mock draft complete.[/green]")
    _print_final_rosters(console, final, ctx, my_slot=args.slot)
    my = final.my_picks()
    if my:
        console.print("Your roster: " + ", ".join(
            f"{(ctx.players.get(p.player_id).name if ctx.players.get(p.player_id) else p.player_name)} ({p.position})" for p in my))
    return EXIT_OK


def cmd_projections(args: argparse.Namespace) -> int:
    console = Console()
    settings = _settings_from_args(args)
    ctx = _build(settings, args, quiet=True)
    console.print(_projection_table(ctx, args.top, args.position, title=f"Projections — {ctx.league.name}"))
    _print_sources(console, ctx)
    return EXIT_OK


def _espn_rosters_for_league(settings: Settings, args: argparse.Namespace, console: Console, ctx=None) -> dict[int, dict]:
    """``{team id: {user_id, name, username, display_name, players}}`` from the ESPN league payload: the one
    the context was built from, else a fresh fetch, else the saved capture. Player ids resolve through the
    context's universe (placeholders included) so they match ``ctx.players``."""
    from .app import load_snapshot
    from .espn.capture import espn_rosters
    from .espn.ids import EspnIdMap
    from .espn.parsing import parse_espn_draft, parse_espn_managers

    league_json = ((ctx.sources.get("espn_inputs") or {}).get("league_json")) if ctx is not None else None
    if not league_json and not args.offline:
        try:
            async def go():
                async with _espn_client(settings) as client:
                    return await client.get_settings_and_teams(settings.league_id, settings.season)

            league_json = _run(go())
        except Exception as e:  # noqa: BLE001
            if _is_access_denied(e) or _is_not_found_error(e):
                raise
            log.warning("ESPN unreachable (%s); trying snapshot", e)
    if not league_json:
        snap = load_snapshot(str(settings.league_id), "espn")
        league_json = dict(snap.raw.get("league") or {}) if snap is not None else None
    if not league_json:
        console.print("[red]no rosters available (ESPN unreachable and no snapshot; run "
                      "`draftadvisor capture --platform espn` online)[/red]")
        return {}
    draft = parse_espn_draft(league_json)
    managers = parse_espn_managers(league_json, draft)
    id_map = EspnIdMap.from_players(ctx.players) if ctx is not None else EspnIdMap()
    if ctx is not None:
        _add_roster_placeholders(league_json, id_map, ctx.players)
    out: dict[int, dict] = {}
    dropped: list[str] = []
    for r in espn_rosters(league_json, id_map):
        rid = int(r["roster_id"])
        m = managers.get(str(rid))
        out[rid] = {
            "user_id": str(rid),
            "name": ((m.team_name or m.display_name) if m else None) or f"Team {rid}",
            "username": (m.display_name if m else "") or "",
            "display_name": (m.display_name if m else "") or "",
            "players": [str(p) for p in r["players"]],
        }
        if ctx is not None:
            dropped += [str(p) for p in r["players"] if str(p) not in ctx.players]
    if dropped:
        log.warning("%d rostered ESPN player(s) are not in the player universe and are left out of the rosters: %s",
                    len(dropped), ", ".join(sorted(set(dropped))))
    return out


def _add_roster_placeholders(league_json: Mapping[str, Any], id_map: Any, players: dict[str, Player]) -> int:
    """Add a placeholder :class:`~draftadvisor.models.Player` to ``players`` for every roster entry of the
    ESPN league payload (``teams[].roster.entries[]``) whose id is not in the universe, so ``analyze`` /
    ``trade`` name every rostered player (offline too, when no kona pool was fetched); returns how many."""
    from .espn.ids import placeholder_player

    added = 0
    for team in league_json.get("teams") or []:
        if not isinstance(team, Mapping):
            continue
        for entry in ((team.get("roster") or {}).get("entries") or []):
            if not isinstance(entry, Mapping):
                continue
            pid = id_map.resolve_player_json(entry)
            if pid not in players:
                players[pid] = placeholder_player(entry, pid)
                added += 1
    if added:
        log.info("%d rostered ESPN player(s) outside the universe added as placeholders", added)
    return added


def _rosters_for_league(settings: Settings, args: argparse.Namespace, console: Console, ctx=None):
    """{roster_id: {"user": Manager-ish dict, "players": [ids]}} from Sleeper / ESPN or the snapshot."""
    if settings.is_espn:
        return _espn_rosters_for_league(settings, args, console, ctx)
    users: list[dict] = []
    rosters: list[dict] = []
    if not args.offline:
        try:
            from .sleeper.client import SleeperClient

            async def go():
                async with SleeperClient() as client:
                    return await client.get_league_users(settings.league_id), await client.get_league_rosters(settings.league_id)

            users, rosters = _run(go())
        except Exception as e:  # noqa: BLE001
            if not _is_network_error(e):
                raise
            log.warning("Sleeper unreachable (%s); trying snapshot", e)
    if not rosters:
        try:
            from .app import load_snapshot

            snap = load_snapshot(str(settings.league_id), settings.platform)      # never an ESPN capture
            if snap is not None:
                users = list(snap.raw.get("users") or [])
                rosters = list(snap.raw.get("rosters") or [])
        except Exception as e:  # noqa: BLE001
            log.warning("snapshot unavailable: %s", e)
    if not rosters:
        console.print("[red]no rosters available (Sleeper unreachable and no snapshot; run `draftadvisor capture` online)[/red]")
        return {}
    by_user = {str(u.get("user_id")): u for u in users}
    out: dict[int, dict] = {}
    for r in rosters:
        u = by_user.get(str(r.get("owner_id")), {})
        out[int(r.get("roster_id"))] = {
            "user_id": str(r.get("owner_id")),
            "name": (u.get("metadata") or {}).get("team_name") or u.get("display_name") or f"Roster {r.get('roster_id')}",
            "username": u.get("username") or "",
            "display_name": u.get("display_name") or "",
            "players": [str(p) for p in (r.get("players") or [])],
        }
    return out


def _find_roster(rosters: Mapping[int, dict], who: str) -> int | None:
    w = who.strip().lower()
    for rid, r in rosters.items():
        if w in (r["username"].lower(), r["display_name"].lower(), r["name"].lower(), r["user_id"], str(rid)):
            return rid
    hits = difflib.get_close_matches(w, [r["display_name"].lower() for r in rosters.values()] +
                                     [r["name"].lower() for r in rosters.values()], n=1, cutoff=0.6)
    if hits:
        for rid, r in rosters.items():
            if hits[0] in (r["display_name"].lower(), r["name"].lower()):
                return rid
    return None


def cmd_trade(args: argparse.Namespace) -> int:
    console = Console()
    settings = _settings_from_args(args)
    if not settings.league_id:
        console.print("[red]trade needs --league ID[/red]")
        return EXIT_USAGE
    ctx = _build(settings, args, quiet=True)
    rosters = _rosters_for_league(settings, args, console, ctx)
    if not rosters:
        return EXIT_NETWORK
    me, them = _find_roster(rosters, args.me), _find_roster(rosters, args.them)
    if me is None or them is None:
        names = ", ".join(r["display_name"] or r["name"] for r in rosters.values())
        console.print(f"[red]could not find {'--me' if me is None else '--them'}; managers: {names}[/red]")
        return EXIT_USAGE

    def ids_for(text: str, roster_ids: list[str]) -> list[str] | None:
        pool = [ctx.players[pid] for pid in roster_ids if pid in ctx.players]
        out = []
        for name in _split_names(text):
            pl = resolve_player_input(name, pool)
            if pl is None:
                console.print(f"[red]{name!r} is not on that roster ({', '.join(p.name for p in pool)})[/red]")
                return None
            out.append(pl.player_id)
        return out

    give = ids_for(args.give, rosters[me]["players"])
    get = ids_for(args.get, rosters[them]["players"])
    if give is None or get is None:
        return EXIT_USAGE
    from .strategy.trade import evaluate_trade

    ev = evaluate_trade(rosters[me]["players"], rosters[them]["players"], give, get, ctx.players, ctx.projections,
                        ctx.league, settings.bench_discount)
    color = {"ACCEPT": "green", "REJECT": "red"}.get(ev.verdict, "yellow")
    console.print(f"[bold {color}]{ev.verdict}[/bold {color}]  me {ev.my_before:.0f} -> {ev.my_after:.0f} ({ev.my_delta:+.1f})"
                  f" • them {ev.their_before:.0f} -> {ev.their_after:.0f} ({ev.their_delta:+.1f})")
    for line in ev.details:
        console.print(f"  • {line}")
    return EXIT_OK


def cmd_trades(args: argparse.Namespace) -> int:
    """Search every other roster for a deal that helps me and reads as fair to them.

    Costs nothing: this is arithmetic over projections already in the bundle. No paid API is called.
    """
    console = Console()
    settings = _settings_from_args(args)
    if not settings.league_id:
        console.print("[red]trades needs --league ID[/red]")
        return EXIT_USAGE
    ctx = _build(settings, args, quiet=True)
    rosters = _rosters_for_league(settings, args, console, ctx)
    if not rosters:
        return EXIT_NETWORK
    me = _find_roster(rosters, args.me)
    if me is None:
        # both names, because ESPN's member display name is often nothing a human would type
        names = ", ".join(sorted({n for r in rosters.values() for n in (r["name"], r["display_name"]) if n}))
        console.print(f"[red]could not find --me {args.me!r}[/red]")
        console.print(f"  known teams and managers: {names}")
        return EXIT_USAGE

    from .strategy.trade import SHAPES_WIDE, find_trades

    mine = list(rosters[me]["players"])
    others = {(r["display_name"] or r["name"] or str(rid)): list(r["players"])
              for rid, r in rosters.items() if rid != me}
    rostered = {pid for r in rosters.values() for pid in r["players"]}
    free_agents = [pid for pid in ctx.projections if pid not in rostered]
    shapes = SHAPES_WIDE if args.two_for_one else ((1, 1),)
    search = find_trades(mine, others, ctx.players, ctx.projections, ctx.league,
                         free_agents=free_agents, shapes=shapes, limit=args.limit,
                         min_my_gain=args.min_gain, min_their_view=args.min_their_view,
                         per_team_limit=args.per_team, bench_discount=settings.bench_discount)
    proposals = search.proposals
    weighed = (f"{search.considered} swaps weighed across {search.teams_searched} rosters; "
               f"{search.helped_me} helped you, {search.rejected_their_view} of those read as a loss to them"
               + (" (search hit its time budget)" if search.timed_out else ""))
    if not proposals:
        console.print("[yellow]No trade found that helps you and that the other manager would plausibly take.[/yellow]")
        console.print(f"  [dim]{weighed}[/dim]")
        console.print("  Try --min-gain 0 to see marginal deals, or --min-their-view -10 to include harder sells.")
        return EXIT_OK

    console.print(f"[bold]{len(proposals)} trade(s)[/bold]  "
                  f"[dim]my gain is our projection; their view is the market's consensus[/dim]")
    console.print(f"[dim]{weighed}[/dim]")
    for i, p in enumerate(proposals, 1):
        gives = ", ".join(ctx.players[x].display() for x in p.give if x in ctx.players)
        gets = ", ".join(ctx.players[x].display() for x in p.get if x in ctx.players)
        console.print(f"\n[bold cyan]{i}. {p.team}[/bold cyan]  send [red]{gives}[/red] for [green]{gets}[/green]")
        console.print(f"   me [bold]{p.my_gain:+.1f}[/bold] pts • they read it as [bold]{p.their_view:+.1f}[/bold] "
                      f"• {p.edge:+.1f} of my gain is the market disagreeing with us")
        console.print(f"   [dim]{p.pitch(ctx.players)}[/dim]")
    return EXIT_OK


def _inseason_tables(console: Console):
    """The bundle's in-season tables, or an empty set with a warning saying what is degraded."""
    from .projections.inseason import InSeasonTables

    try:
        from .lean import get_bundle

        tables = InSeasonTables.from_dict(get_bundle().inseason)
    except Exception as e:  # noqa: BLE001
        log.warning("no web bundle: %s", e)
        tables = InSeasonTables()
    if not tables.present:
        console.print("[yellow]No in-season tables in the bundle (run scripts/build_bundle.py).[/yellow]")
        console.print("  Weekly spread falls back to a position curve and no opponent adjustment is applied;"
                      " byes come from ESPN's own weekly projection instead of the schedule.")
    return tables


def espn_week_inputs(league_json, ctx, console: Console):
    """``(calendar, rosters, team_names, weekly points)`` from one ESPN in-season payload."""
    from .espn.capture import espn_rosters
    from .espn.ids import EspnIdMap
    from .espn.inseason import league_calendar, weekly_player_points
    from .espn.parsing import parse_espn_draft, parse_espn_managers

    id_map = EspnIdMap.from_players(ctx.players)
    _add_roster_placeholders(league_json, id_map, ctx.players)
    draft = parse_espn_draft(league_json)
    managers = parse_espn_managers(league_json, draft)
    rosters = {int(r["roster_id"]): [str(p) for p in r["players"]] for r in espn_rosters(league_json, id_map)}
    names = {}
    for k, m in managers.items():
        if str(k).isdigit():
            names[int(k)] = (m.team_name or m.display_name) or f"Team {k}"
    return league_calendar(league_json), rosters, names, weekly_player_points(league_json)


def _not_playing(ctx, weekly: Mapping[str, Any], week: int) -> list[str]:
    """Players ESPN itself projects at zero this week: a bye, an inactive, a suspension.

    Our model supplies the points; ESPN is better placed to know who is not on the field at all, so
    each is used for what it is good at.
    """
    out = []
    for pid, pl in ctx.players.items():
        if not pl.espn_id:
            continue
        block = (weekly.get(str(pl.espn_id)) or {}).get(week) or {}
        proj = block.get("projected")
        if proj is not None and proj <= 0.0:
            out.append(pid)
    return out


def cmd_lineup(args: argparse.Namespace) -> int:
    """Optimal start/sit for one week, the gap against the lineup that is set, and the matchup odds.

    ESPN only for now: the lineup that is *set* and the week's opponent both come from ESPN's payload.
    Costs nothing - one ESPN request, no paid API.
    """
    console = Console()
    settings = _settings_from_args(args)
    if settings.platform != "espn":
        console.print("[red]lineup currently supports ESPN leagues only (--platform espn)[/red]")
        return EXIT_USAGE
    if not settings.league_id:
        console.print("[red]lineup needs --league ID[/red]")
        return EXIT_USAGE
    ctx = _build(settings, args, quiet=True)

    from .espn.inseason import current_lineups, matchups, opponent_for, team_records
    from .strategy.week import plan_week, waiver_targets

    async def go():
        async with _espn_client(settings) as client:
            return await client.get_in_season(settings.league_id, settings.season, week=args.week)

    try:
        league_json = _run(go())
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]ESPN unreachable: {e}[/red]")
        return EXIT_NETWORK

    tables = _inseason_tables(console)
    cal, rosters, names, weekly = espn_week_inputs(league_json, ctx, console)
    week = int(args.week or cal.current_week)

    want = (args.me or "").strip().lower()
    my_team = next((tid for tid, name in names.items() if want and want == name.lower()), None)
    if my_team is None and settings.team_id:
        my_team = int(settings.team_id)
    if my_team is None or my_team not in rosters:
        console.print(f"[red]could not tell which team is yours[/red] (--me or --team-id); "
                      f"teams: {', '.join(sorted(names.values()))}")
        return EXIT_USAGE

    period = cal.matchup_period_for(week)
    opp = opponent_for(matchups(league_json, period), my_team, period)
    espn_of = {pid: pl.espn_id for pid, pl in ctx.players.items() if pl.espn_id}
    started_espn = set(current_lineups(league_json).get(my_team, {}).get("starters", []))
    set_starters = [p for p in rosters[my_team] if str(espn_of.get(p) or "") in started_espn]
    pct = {r.team_id: r.espn_playoff_pct for r in team_records(league_json)}

    plan = plan_week(rosters, names, my_team, ctx.players, ctx.projections, ctx.league, tables, week,
                     set_starters=set_starters, opponent_id=opp,
                     not_playing=_not_playing(ctx, weekly, week), espn_playoff_pct=pct.get(my_team),
                     is_playoffs=cal.is_playoffs(week), weeks_remaining=cal.weeks_remaining(week))

    if plan.unprojected:
        console.print(f"[yellow]No projection for {len(plan.unprojected)} player(s) on your roster: "
                      f"{', '.join(plan.unprojected)}[/yellow]")
        console.print("  They score 0 here and will never be started. Rebuild the bundle if they are real.")
    tail = "playoffs" if plan.is_playoffs else f"{plan.weeks_remaining} regular-season weeks left"
    console.print(f"[bold]Week {plan.week} — {plan.team_name}[/bold]  [dim]{tail}[/dim]")
    for slot in plan.starters:
        console.print(f"  {slot.slot:<10} {(slot.name or ''):<24} {(slot.team or ''):<4} {slot.points:6.1f}")
    console.print(f"  [bold]{'total':<10} {'':<24} {'':<4} {plan.best_points:6.1f}[/bold]")

    if plan.set_points is not None:
        if plan.gap > 0:
            console.print(f"\n[yellow]Your lineup as set scores {plan.set_points:.1f}: "
                          f"{plan.gap:+.1f} left on the bench.[/yellow]")
            if plan.start and plan.sit:
                console.print(f"  start {', '.join(plan.start)}; sit {', '.join(plan.sit)}")
        else:
            console.print("\n[green]Your lineup is already the best one.[/green]")

    if plan.win_probability is None:
        console.print("\n[dim]No opponent this week (bye or unpublished schedule).[/dim]")
    else:
        console.print(f"\n[bold]vs {plan.opponent_name}[/bold]  {plan.my_points:.1f} to {plan.their_points:.1f}  "
                      f"→ [bold]{plan.win_probability:.0%}[/bold] to win")
        console.print(f"  [dim]model estimate: weekly spread ±{plan.my_sigma:.0f} for you, "
                      f"±{plan.their_sigma:.0f} for them, measured on 2019-2025 weekly scores; players added "
                      f"independently, so a stacked lineup swings a little more than this says[/dim]")
        if plan.espn_playoff_pct is not None:
            console.print(f"  [dim]ESPN's own playoff odds for you: {plan.espn_playoff_pct:.0f}%[/dim]")

    rostered = {p for ids in rosters.values() for p in ids}
    fa = [p for p in ctx.projections if p not in rostered]
    targets = waiver_targets(fa, rosters[my_team], ctx.players, ctx.projections, ctx.league, tables, week,
                             limit=5, not_playing=_not_playing(ctx, weekly, week))
    if targets:
        console.print("\n[bold]Waiver targets[/bold] [dim]ranked by what they do to your starting lineup,"
                      " not by points[/dim]")
        for t in targets:
            over = f" over {t.replaces}" if t.replaces else ""
            console.print(f"  {(t.position or '?'):<4} {t.name:<24} {t.week_points:5.1f} pts  "
                          f"[green]{t.lineup_gain:+.1f}[/green] to your lineup{over}")
    return EXIT_OK


def cmd_analyze(args: argparse.Namespace) -> int:
    console = Console()
    settings = _settings_from_args(args)
    if not settings.league_id:
        console.print("[red]analyze needs --league ID[/red]")
        return EXIT_USAGE
    ctx = _build(settings, args, quiet=True)
    rosters = _rosters_for_league(settings, args, console, ctx)
    if not rosters:
        return EXIT_NETWORK
    from .strategy.lineup import optimal_lineup
    from .strategy.replacement import vorp

    league = ctx.league
    playoff_start = int((league.settings or {}).get("playoff_week_start") or 15)
    slots = league.starting_slots
    t = Table(title=f"Rosters — {league.name}", box=box.SIMPLE_HEAD)
    for col in ("Rank", "Team", "Lineup", "Bench", "Open", "Weakest slot", "Playoff byes", "Counts"):
        t.add_column(col, justify="right" if col in ("Rank", "Lineup", "Bench") else "left")
    rows = []
    rostered: set[str] = set()
    for rid, r in rosters.items():
        rostered.update(r["players"])
        cands = [(ctx.players[p], float(ctx.projections[p].points) if p in ctx.projections else 0.0)
                 for p in r["players"] if p in ctx.players]
        assignment, starters, bench = optimal_lineup(cands, slots)
        pts = {pl.player_id: p for pl, p in cands}
        open_ = [s for i, s in enumerate(slots) if i not in assignment]
        weakest = None
        for i, pid in assignment.items():
            if weakest is None or pts[pid] < weakest[1]:
                weakest = (f"{slots[i]} {ctx.players[pid].name} {pts[pid]:.0f}", pts[pid])
        po_byes = sorted({ctx.players[pid].bye_week for pid in assignment.values()
                          if ctx.players[pid].bye_week and ctx.players[pid].bye_week >= playoff_start})
        counts: dict[str, int] = {}
        for pl, _ in cands:
            counts[pl.position] = counts.get(pl.position, 0) + 1
        rows.append((starters, sum(pts[b] for b in bench), r["name"], " ".join(open_) or "-",
                     weakest[0] if weakest else "-", ",".join(map(str, po_byes)) or "-",
                     " ".join(f"{k}{v}" for k, v in sorted(counts.items()))))
    rows.sort(key=lambda x: -x[0])
    mine = (args.username or settings.username or "").lower()
    if settings.is_espn and settings.team_id is not None and settings.team_id in rosters:
        mine = rosters[settings.team_id]["name"].lower()
    for i, row in enumerate(rows, start=1):
        style = "bold green" if mine and mine in row[2].lower() else ""
        t.add_row(str(i), row[2], f"{row[0]:.0f}", f"{row[1]:.0f}", *row[3:], style=style)
    console.print(t)
    avail = [pid for pid in ctx.projections if pid in ctx.players and pid not in rostered]
    try:
        v = vorp(ctx.projections, ctx.players, league, avail)
    except Exception as e:  # noqa: BLE001
        log.warning("vorp failed: %s", e)
        v = {pid: ctx.projections[pid].points for pid in avail}
    best = sorted(avail, key=lambda pid: -v.get(pid, 0.0))[:12]
    w = Table(title="Best waiver targets (by VORP)", box=box.SIMPLE_HEAD)
    for col in ("Player", "Pos", "Tm", "Pts", "VORP"):
        w.add_column(col, justify="right" if col in ("Pts", "VORP") else "left")
    for pid in best:
        pl = ctx.players[pid]
        w.add_row(pl.name, pl.position, pl.team or "FA", f"{ctx.projections[pid].points:.0f}", f"{v.get(pid, 0):+.0f}")
    console.print(w)
    return EXIT_OK


def _usage_line(claude: Any) -> str:
    """'12.1k in / 0.4k out - about $0.07 (claude-sonnet-5)' for the last request, or ''.

    When the model is not in the price table (``last_cost_usd`` is ``None``) the line says so instead of
    silently dropping the estimate. The model shown is the served id when the chat client records one
    (``last_model``), else the configured ``model``."""
    usage = getattr(claude, "last_usage", None) or {}
    if not usage:
        return ""
    k = lambda n: f"{(n or 0) / 1000:.1f}k"  # noqa: E731
    text = f"{k(usage.get('input_tokens'))} in / {k(usage.get('output_tokens'))} out"
    cost = getattr(claude, "last_cost_usd", None)
    if cost is not None:
        text += f" - about ${cost:.2f}" if cost >= 0.005 else " - under $0.01"
    else:
        text += " - cost unknown (model not in the price table)"
    model = getattr(claude, "last_model", None) or getattr(claude, "model", None) or "?"
    return f"{text} ({model})"


async def _live_context_text(settings: Settings, ctx) -> str:
    """The current draft state (one bootstrap of the platform's poller) rendered for Claude."""
    from .research.claude import build_context_text

    if settings.is_espn:
        from .espn.capture import espn_names
        from .espn.ids import EspnIdMap
        from .espn.poller import EspnDraftPoller

        inputs = ctx.sources.get("espn_inputs") or {}
        async with _espn_client(settings) as client:
            poller = EspnDraftPoller(client, settings.league_id, settings.season,
                                     id_map=EspnIdMap.from_players(ctx.players), names=espn_names(inputs.get("players")),
                                     swid=settings.swid, team_id=settings.team_id, slot=settings.slot,
                                     username=settings.username, settings=settings, league_json=inputs.get("league_json"))
            state = await poller.bootstrap()
    else:
        from .sleeper.client import SleeperClient
        from .sleeper.poller import DraftPoller

        async with SleeperClient() as client:
            poller = DraftPoller(client, settings.draft_id, league_id=settings.league_id, settings=settings)
            state = await poller.bootstrap()
    rec = ctx.advisor.recommend(state)
    return build_context_text(state, rec, ctx.players, ctx.notes)


def cmd_ask(args: argparse.Namespace) -> int:
    """One Claude request with the draft context; the only CLI path that costs money."""
    console = Console()
    settings = _settings_from_args(args)
    ctx = _build(settings, args, quiet=True)
    if not ctx.claude.enabled:
        console.print("[yellow]Claude is off — set ANTHROPIC_API_KEY to ask questions.[/yellow]")
        return EXIT_USAGE
    context_text = ""
    live = (settings.is_espn and settings.league_id) or (not settings.is_espn and settings.draft_id)
    if live and not args.offline:
        try:
            context_text = _run(_live_context_text(settings, ctx))
        except Exception as e:  # noqa: BLE001
            if not _is_network_error(e):
                raise
            console.print(f"[yellow]draft context unavailable ({e}); answering from projections only[/yellow]")
    if not context_text:
        top = sorted(ctx.projections.values(), key=lambda p: -p.points)[:40]
        context_text = f"League: {ctx.league.name} ({ctx.league.scoring_type}, slots {' '.join(ctx.league.starting_slots)})\n" + \
            "Top projected players:\n" + "\n".join(
                f"- {ctx.players[p.player_id].name} ({p.position}, {ctx.players[p.player_id].team or 'FA'}) {p.points:.0f} pts, "
                f"ADP {ctx.players[p.player_id].adp or '-'}" for p in top if p.player_id in ctx.players)
    answer = _run(ctx.claude.ask(args.question, context_text))
    if not answer:
        console.print(f"[red]no answer ({getattr(ctx.claude, 'last_error', None) or 'unknown error'})[/red]")
        return EXIT_ERROR
    console.print(answer, markup=False, highlight=False)
    usage = _usage_line(ctx.claude)
    if usage:
        console.print(Text(usage, style="dim"))
    return EXIT_OK


def _ids_espn(args: argparse.Namespace, settings: Settings, console: Console) -> int:
    """ESPN leagues of the SWID's account via the fan API (best effort), else how to find the id."""
    if not settings.swid:
        console.print("[red]ids --platform espn needs your SWID cookie: --swid {...} or ESPN_SWID[/red]")
        console.print(ESPN_LEAGUE_ID_HELP, markup=False, highlight=False)
        console.print(ESPN_COOKIE_HELP, markup=False, highlight=False)
        return EXIT_USAGE

    async def go():
        async with _espn_client(settings) as client:
            return await client.get_fan_leagues(settings.swid)

    try:
        leagues = _run(go())
    except Exception as e:  # noqa: BLE001
        rc = _report_api_error(e, console, settings)
        if rc is not None:
            return rc
        raise
    if not leagues:
        console.print("ESPN's fan API listed no fantasy football leagues for that SWID (it is best effort and "
                      "sometimes empty).", markup=False, highlight=False)
        console.print(ESPN_LEAGUE_ID_HELP, markup=False, highlight=False)
        console.print(ESPN_COOKIE_HELP, markup=False, highlight=False)
        return EXIT_OK
    t = Table(title="Your ESPN leagues", box=box.SIMPLE_HEAD)
    for col in ("League", "league_id", "Season", "Your team", "team_id"):
        t.add_column(col)
    for lg in leagues:
        t.add_row(str(lg.get("name") or ""), str(lg.get("league_id")), str(lg.get("season") or settings.season),
                  str(lg.get("team_name") or ""), str(lg.get("team_id") if lg.get("team_id") is not None else "-"))
    console.print(t)
    first = leagues[0]
    console.print(f"next: draftadvisor draft --platform espn --league {first.get('league_id')} "
                  f"--season {first.get('season') or settings.season}"
                  + (f" --team-id {first.get('team_id')}" if first.get("team_id") is not None else " --swid ...")
                  + "  (private league: add --espn-s2 / --swid)", markup=False, highlight=False)
    return EXIT_OK


def cmd_ids(args: argparse.Namespace) -> int:
    console = Console()
    settings = _settings_from_args(args)
    if args.offline:
        console.print("[red]ids needs the network[/red]")
        return EXIT_USAGE
    if settings.is_espn:
        return _ids_espn(args, settings, console)
    if not args.username:
        console.print("[red]ids needs --username U (Sleeper) or --platform espn --swid {...} (ESPN)[/red]")
        return EXIT_USAGE
    from .sleeper.client import SleeperClient

    async def go():
        async with SleeperClient() as client:
            user = await client.get_user(args.username)
            leagues = await client.get_user_leagues(user["user_id"], settings.season)
            rows = []
            for lg in leagues:
                try:
                    drafts = await client.get_league_drafts(lg["league_id"])
                except Exception as e:  # noqa: BLE001
                    log.info("drafts for %s unavailable: %s", lg.get("league_id"), e)
                    drafts = []
                rows.append((lg, drafts))
            return user, rows

    try:
        user, rows = _run(go())
    except Exception as e:  # noqa: BLE001
        if _is_not_found_error(e):
            console.print(Text(f"Sleeper has no user {args.username!r} ({e})", style="red"))
            return EXIT_USAGE
        rc = _report_api_error(e, console, settings)
        if rc is not None:
            return rc
        raise
    console.print(f"user {user.get('display_name')} (id {user.get('user_id')}) • season {settings.season}")
    t = Table(box=box.SIMPLE_HEAD)
    for col in ("League", "league_id", "Teams", "Status", "draft_id", "Draft status", "Type"):
        t.add_column(col)
    for lg, drafts in rows:
        if not drafts:
            t.add_row(lg.get("name", ""), str(lg.get("league_id")), str(lg.get("total_rosters")), str(lg.get("status")),
                      str(lg.get("draft_id") or "-"), "-", "-")
        for d in drafts:
            t.add_row(lg.get("name", ""), str(lg.get("league_id")), str(lg.get("total_rosters")), str(lg.get("status")),
                      str(d.get("draft_id")), str(d.get("status")), str(d.get("type")))
    console.print(t)
    return EXIT_OK


def cmd_web(args: argparse.Namespace) -> int:
    from .web.server import main as web_main

    argv = ["--host", args.host, "--port", str(args.port)]
    if args.no_browser:
        argv.append("--no-browser")
    return web_main(argv)


def _capture_espn(args: argparse.Namespace, settings: Settings, console: Console) -> int:
    from .capture import print_snapshot

    if not settings.league_id:
        console.print("[red]capture --platform espn needs --league ID (the leagueId= number in the ESPN league URL)[/red]")
        return EXIT_USAGE
    if args.offline:
        from .app import load_snapshot

        snap = load_snapshot(str(settings.league_id), "espn") or load_snapshot(f"espn-{settings.league_id}-{settings.season}", "espn")
        if snap is None:
            console.print("[red]no saved ESPN snapshot for this league; run online first[/red]")
            return EXIT_USAGE
        print_snapshot(snap, console)
        return EXIT_OK
    from .espn.capture import capture_espn_league

    async def go():
        async with _espn_client(settings) as client:
            return await capture_espn_league(client, settings.league_id, settings.season, swid=settings.swid,
                                             team_id=settings.team_id, slot=settings.slot, username=settings.username,
                                             save=True)

    try:
        snap = _run(go())
    except ValueError as e:                         # unknown --username / --team-id: the message lists the teams
        console.print(str(e), style="red", markup=False, highlight=False)
        return EXIT_USAGE
    except Exception as e:  # noqa: BLE001
        rc = _report_api_error(e, console, settings)
        if rc is not None:
            return rc
        raise
    print_snapshot(snap, console)
    if snap.my_user_id is None:
        console.print("not identified as a team: pass --team-id N, --slot N or --username \"team or owner name\" "
                      "(or --swid: your SWID cookie identifies your team)", style="yellow", markup=False, highlight=False)
    return EXIT_OK


def cmd_capture(args: argparse.Namespace) -> int:
    console = Console()
    settings = _settings_from_args(args)
    if settings.is_espn:
        return _capture_espn(args, settings, console)
    if not (settings.league_id or settings.draft_id):
        console.print("[red]capture needs --league ID or --draft ID[/red]")
        return EXIT_USAGE
    from .capture import capture_league, print_snapshot

    if args.offline:
        from .app import load_snapshot

        snap = load_snapshot(str(settings.league_id or settings.draft_id), settings.platform)   # never an ESPN capture
        if snap is None:
            console.print("[red]no saved snapshot for this league on Sleeper; run online first "
                          "(an ESPN capture needs --platform espn)[/red]")
            return EXIT_USAGE
        print_snapshot(snap, console)
        return EXIT_OK
    from .sleeper.client import SleeperClient

    async def go():
        async with SleeperClient() as client:
            return await capture_league(client, settings.league_id, settings.draft_id, username=settings.username,
                                        user_id=settings.user_id, slot=settings.slot, season=settings.season)

    try:
        snap = _run(go())
    except Exception as e:  # noqa: BLE001
        rc = _report_api_error(e, console, settings)
        if rc is not None:
            return rc
        raise
    print_snapshot(snap, console)
    return EXIT_OK


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.home:
        os.environ["DRAFTADVISOR_HOME"] = str(args.home)
    _configure_logging(args.verbose)
    if args.season is None:
        args.season = DEFAULT_SEASON
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\nstopped.", file=sys.stderr)
        return EXIT_OK
    except Exception as e:  # noqa: BLE001
        if _is_access_denied(e):
            print(_access_denied_message(e, _settings_from_args(args)), file=sys.stderr)
            return EXIT_USAGE
        if _is_not_found_error(e):
            print(_not_found_message(e, _settings_from_args(args)), file=sys.stderr)
            return EXIT_USAGE
        if _is_network_error(e):
            print(_network_message(e, _settings_from_args(args)), file=sys.stderr)
            return EXIT_NETWORK
        if args.verbose:
            raise
        print(f"error: {type(e).__name__}: {e}  (run with -v for the traceback)", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
