"""Terminal dashboard for the live draft (rich ``Live`` layout). See DESIGN.md §3.6.

The dashboard is a pure function of ``(state, rec, players, status)`` -> renderable
(:meth:`Dashboard.build`) wrapped in a ``rich.live.Live`` so the draft loop can push
updates without blocking. Nothing here does I/O other than writing to the console.

Layout at 160x45 (columns are proportional; the opponent-needs panel is dropped
below 130 columns):

    ┌ header: league • scoring • round/pick • on the clock • your next picks • latency • Claude ┐
    │ My roster (left) │ Best picks now (center top)             │ Recent picks (right) │
    │                  │ By position: 6 mini tables (center bot) │ Opponent needs       │
    └ footer: Claude advice / notes / key hint ─────────────────────────────────────────────┘

``status`` is a free-form dict the caller fills in; recognised keys:
``latency_ms``, ``compute_ms``, ``claude`` ("off" | "thinking" | "ready" | "error" | text),
``turn_started_at`` (epoch seconds when we noticed it became my turn), ``poll_count``,
``last_error``, ``mode`` ("live" | "mock"), ``hint`` (key hint text), ``message``,
``projections`` (``dict[player_id, Projection]`` used to show my roster's projected points).
"""
from __future__ import annotations

import logging
import time
from typing import Iterable, Mapping, Sequence

from rich import box
from rich.console import Console, Group, RenderableType
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..config import NON_STARTING_SLOTS, SKILL_POSITIONS, SLOT_ELIGIBILITY
from ..models import DraftState, Pick, Player, PlayerValue, Projection, Recommendation, RosterSummary

log = logging.getLogger(__name__)

__all__ = ["Dashboard", "render_text", "scoring_description", "action_style"]

#: Below this many columns the opponent-needs panel is dropped.
COMPACT_WIDTH = 150
#: Below this many rows the per-position tables show fewer candidates.
COMPACT_HEIGHT = 42

_ACTION_STYLE = {"TAKE NOW": "bold white on red", "SOON": "bold black on yellow", "WAIT": "bold black on green",
                 "SKIP": "dim"}
_SLOT_ORDER = ("QB", "RB", "WR", "TE", "FLEX", "WRRB_FLEX", "REC_FLEX", "SUPER_FLEX", "K", "DEF")


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------

def action_style(action: str) -> str:
    """rich style for a :class:`PositionAdvice` action badge."""
    return _ACTION_STYLE.get(action, "")


def scoring_description(state: DraftState) -> str:
    """"PPR" / "Half PPR" / "Standard" (+ superflex / TE premium) from the league settings."""
    lg = state.league
    if lg is None:
        st = (state.draft.scoring_type or "").lower()
        return {"ppr": "PPR", "half_ppr": "Half PPR", "std": "Standard"}.get(st, st or "?")
    label = {"ppr": "PPR", "half_ppr": "Half PPR", "std": "Standard"}[lg.scoring_type]
    if lg.is_superflex:
        label += " • Superflex"
    if lg.te_premium:
        label += f" • TE +{lg.te_premium:g}"
    if float(lg.scoring_settings.get("pass_td", 4) or 4) >= 6:
        label += " • 6pt pass TD"
    return label


def _fmt_pts(v: float | None) -> str:
    return "-" if v is None else f"{v:.0f}"


def _fmt_pct(v: float | None) -> str:
    return "-" if v is None else f"{100 * v:.0f}%"


def _fmt_adp(v: float | None) -> str:
    return "-" if v is None else f"{v:.0f}"


def _trunc(s: str, n: int) -> str:
    return s if len(s) <= n else s[: max(1, n - 1)] + "…"


def _player_name(pid: str, players: Mapping[str, Player], pick: Pick | None = None) -> str:
    pl = players.get(pid)
    if pl is not None:
        return pl.name
    if pick is not None:
        return pick.player_name
    return pid


def _player_pos(pid: str, players: Mapping[str, Player], pick: Pick | None = None) -> str:
    pl = players.get(pid)
    if pl is not None:
        return pl.position
    if pick is not None and pick.position:
        return str(pick.position)
    return "?"


def _points_of(pid: str, projections: Mapping[str, Projection] | None) -> float | None:
    if not projections:
        return None
    pr = projections.get(pid)
    return None if pr is None else float(pr.points)


def _countdown(state: DraftState, status: Mapping) -> int | None:
    """Seconds left on my pick clock (None when no timer / not my turn)."""
    timer = int(state.draft.pick_timer or 0)
    started = status.get("turn_started_at")
    if timer <= 0 or started is None:
        return None
    return max(0, int(round(timer - (time.time() - float(started)))))


# ---------------------------------------------------------------------------
# Roster slot assignment (display only)
# ---------------------------------------------------------------------------

def assign_roster_slots(players_pts: Sequence[tuple[Player, float | None]], slots: Sequence[str]) -> list[tuple[str, Player | None, float | None]]:
    """Fill ``slots`` (starting slot labels) with the roster for display.

    Dedicated slots first (best points per position), then flex slots greedily,
    everything else goes to ``BN`` rows. Uses :func:`strategy.lineup.optimal_lineup`
    when available (same assignment as the strategy engine), else a greedy fill.
    """
    starters = [s for s in slots if s not in NON_STARTING_SLOTS]
    known = [(pl, float(p) if p is not None else 0.0) for pl, p in players_pts]
    assignment: dict[int, str] = {}
    try:
        from ..strategy.lineup import optimal_lineup

        assignment, _, _ = optimal_lineup(known, starters)
    except Exception as e:  # noqa: BLE001 - display must never fail
        log.debug("optimal_lineup unavailable, greedy fill (%s)", e)
        assignment = _greedy_assign(known, starters)
    by_id = {pl.player_id: (pl, p) for pl, p in players_pts}
    rows: list[tuple[str, Player | None, float | None]] = []
    used: set[str] = set()
    for i, s in enumerate(starters):
        pid = assignment.get(i)
        if pid is not None and pid in by_id:
            pl, p = by_id[pid]
            rows.append((s, pl, p))
            used.add(pid)
        else:
            rows.append((s, None, None))
    bench = [(pl, p) for pl, p in players_pts if pl.player_id not in used]
    bench.sort(key=lambda t: -(t[1] or 0.0))
    for pl, p in bench:
        rows.append(("BN", pl, p))
    return rows


def _greedy_assign(roster: list[tuple[Player, float]], starters: Sequence[str]) -> dict[int, str]:
    order = sorted(range(len(starters)), key=lambda i: (len(SLOT_ELIGIBILITY.get(starters[i], frozenset())), i))
    pool = sorted(roster, key=lambda t: -t[1])
    used: set[str] = set()
    out: dict[int, str] = {}
    for i in order:
        elig = SLOT_ELIGIBILITY.get(starters[i], frozenset())
        for pl, _ in pool:
            if pl.player_id not in used and pl.position in elig:
                out[i] = pl.player_id
                used.add(pl.player_id)
                break
    return out


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

class Dashboard:
    """rich ``Live`` dashboard. Construct once, :meth:`start`, then :meth:`update` on every change."""

    def __init__(self, console: Console | None = None, refresh_per_second: float = 4,
                 projections: Mapping[str, Projection] | None = None):
        self.console = console or Console()
        self.refresh_per_second = refresh_per_second
        self.projections = projections
        self._live: Live | None = None
        self._last: tuple | None = None
        self.last_build_ms: float = 0.0

    # -- lifecycle ----------------------------------------------------------------
    def start(self) -> None:
        """Enter the live display (idempotent)."""
        if self._live is None:
            self._live = Live(Text("starting…"), console=self.console, refresh_per_second=self.refresh_per_second,
                              screen=False, transient=False, redirect_stderr=False, redirect_stdout=False)
            self._live.start()

    def stop(self) -> None:
        """Leave the live display (idempotent)."""
        if self._live is not None:
            try:
                self._live.stop()
            finally:
                self._live = None

    @property
    def running(self) -> bool:
        return self._live is not None

    def update(self, state: DraftState, rec: Recommendation | None, players: dict[str, Player], status: dict) -> None:
        """Rebuild the layout and push it to the live display (or print once when not live)."""
        self._last = (state, rec, players, dict(status))
        renderable = self.build(state, rec, players, status)
        if self._live is not None:
            self._live.update(renderable)
        else:
            self.console.print(renderable)

    def refresh(self, status_updates: Mapping | None = None) -> None:
        """Re-render the last update (countdown / Claude status ticks)."""
        if self._last is None:
            return
        state, rec, players, status = self._last
        if status_updates:
            status.update(status_updates)
        self.update(state, rec, players, status)

    # -- building -------------------------------------------------------------------
    def build(self, state: DraftState, rec: Recommendation | None, players: dict[str, Player], status: dict) -> RenderableType:
        """The whole screen as one renderable (also usable with ``console.print``)."""
        t0 = time.perf_counter()
        width = self.console.size.width or 160
        height = self.console.size.height or 45
        compact = width < COMPACT_WIDTH
        short = height < COMPACT_HEIGHT
        projections = status.get("projections") or self.projections

        root = Layout(name="root")
        root.split_column(
            Layout(self.header(state, status), name="header", size=4),
            Layout(name="body"),
            Layout(self.footer(rec, status), name="footer", size=4 if short else 5),
        )
        body = root["body"]
        left = Layout(self.roster_panel(state, rec, players, projections, compact), name="left", ratio=9 if compact else 11)
        center = Layout(name="center", ratio=21 if compact else 22)
        center.split_column(
            Layout(self.best_picks_panel(state, rec, players, compact), name="best", size=11),
            Layout(self.positions_panel(rec, state, short, compact), name="positions"),
        )
        right = Layout(name="right", ratio=7 if compact else 8)
        if compact:
            right.update(self.recent_picks_panel(state, players, compact=True))
        else:
            right.split_column(
                Layout(self.recent_picks_panel(state, players), name="recent", ratio=1),
                Layout(self.opponents_panel(state, rec), name="opponents", ratio=1),
            )
        body.split_row(left, center, right)
        self.last_build_ms = (time.perf_counter() - t0) * 1000.0
        return root

    # -- header / footer --------------------------------------------------------------
    def header(self, state: DraftState, status: Mapping) -> RenderableType:
        """League • scoring • round/pick • on the clock • my next picks • latency • Claude."""
        d = state.draft
        name = state.league.name if state.league else (d.metadata.get("name") or f"Draft {d.draft_id}")
        parts = Text()
        parts.append(_trunc(name, 28), style="bold")
        parts.append(" • ")
        parts.append(scoring_description(state))
        parts.append(" • ")
        if state.is_complete:
            parts.append("Draft complete", style="bold green")
        else:
            n = state.next_pick_no
            parts.append(f"Round {state.current_round} • Pick {n}")
            slot = state.on_the_clock_slot
            if slot is not None:
                parts.append(" • On the clock: ")
                parts.append(_trunc(state.slot_label(slot), 18), style="bold")
        parts.append("\n")
        if state.is_my_turn:
            flash = int(time.time() * 2) % 2 == 0
            parts.append(" YOUR PICK ", style="bold white on red" if flash else "bold red on white")
            left = _countdown(state, status)
            if left is not None:
                parts.append(f"  {left:>3d}s left", style="bold red" if left <= 10 else "bold yellow")
            fut = state.my_future_picks()
            if len(fut) > 1:
                parts.append(f"  then #{fut[1]}")
        elif state.my_slot is None:
            parts.append("Observer mode (no slot resolved)", style="dim")
        elif not state.is_complete:
            until = state.picks_until_my_turn
            nxt = state.my_next_pick_no
            if nxt is None:
                parts.append("No picks left", style="dim")
            else:
                parts.append(f"You pick in {until} (#{nxt})", style="bold cyan")
                after = state.my_pick_after_next
                if after is not None:
                    parts.append(f", then #{after}")
        lat = status.get("latency_ms")
        if lat is not None:
            parts.append(f"  • poll {float(lat):.0f} ms", style="dim")
        cm = status.get("compute_ms")
        if cm is not None:
            parts.append(f" • calc {float(cm):.0f} ms", style="dim")
        claude = status.get("claude")
        if claude:
            style = {"thinking": "yellow", "ready": "green", "error": "red", "off": "dim"}.get(str(claude), "")
            parts.append(f" • Claude: {claude}", style=style)
        err = status.get("last_error")
        if err:
            parts.append(f" • {_trunc(str(err), 40)}", style="red")
        return Panel(parts, box=box.ROUNDED, padding=(0, 1), style="on red" if (state.is_my_turn and int(time.time() * 2) % 2 == 0 and False) else "")

    def footer(self, rec: Recommendation | None, status: Mapping) -> RenderableType:
        lines = Text()
        advice = rec.claude_advice if rec is not None else None
        claude = status.get("claude")
        if advice:
            lines.append("Claude: ", style="bold magenta")
            lines.append(advice.strip())
        elif claude == "thinking":
            lines.append("Claude: thinking…", style="yellow")
        elif claude in (None, "off"):
            lines.append("Claude: off — set ANTHROPIC_API_KEY for on-the-clock advice", style="dim")
        else:
            lines.append(f"Claude: {claude}", style="dim")
        if rec is not None and rec.notes:
            lines.append("\n")
            lines.append(" • ".join(rec.notes), style="cyan")
        msg = status.get("message")
        if msg:
            lines.append("\n")
            lines.append(str(msg), style="bold")
        hint = status.get("hint") or "Ctrl-C to quit"
        lines.append("\n")
        lines.append(hint, style="dim")
        return Panel(lines, box=box.ROUNDED, padding=(0, 1))

    # -- left column -------------------------------------------------------------------
    def roster_panel(self, state: DraftState, rec: Recommendation | None, players: Mapping[str, Player],
                     projections: Mapping[str, Projection] | None, compact: bool = False) -> RenderableType:
        """My roster by slot label (starters, then bench) + needs + bye clashes."""
        t = Table(box=box.SIMPLE_HEAD, expand=True, padding=(0, 1), collapse_padding=True, show_edge=False)
        t.add_column("Slot", style="bold", width=4 if compact else 5, no_wrap=True)
        t.add_column("Player", ratio=1, no_wrap=True, overflow="ellipsis", min_width=8)
        if compact:
            t.add_column("Pos", width=3, no_wrap=True)
        else:
            t.add_column("Pos/Tm", width=6, no_wrap=True)
            t.add_column("Bye", width=3, justify="right", no_wrap=True)
            t.add_column("Pts", width=3, justify="right", no_wrap=True)
        picks = state.my_picks()
        roster: list[tuple[Player, float | None]] = []
        for pk in picks:
            pl = players.get(pk.player_id)
            if pl is None:
                pl = Player(player_id=pk.player_id, name=pk.player_name, position=str(pk.position or "?"),
                            team=pk.metadata.get("team"))
            roster.append((pl, _points_of(pk.player_id, projections)))
        slots = state.league.roster_positions if state.league else _default_slots(state)
        rows = assign_roster_slots(roster, slots)
        bench_total = sum(1 for s in slots if s == "BN")
        bench_shown = 0
        for slot, pl, pts in rows:
            if slot == "BN":
                bench_shown += 1
            if pl is None:
                cells = [_slot_abbr(slot), Text("—", style="dim"), "", "", ""]
            elif compact:
                cells = [_slot_abbr(slot), pl.name, pl.position]
            else:
                cells = [_slot_abbr(slot), pl.name, f"{pl.position} {pl.team or 'FA'}",
                         str(pl.bye_week) if pl.bye_week else "-", _fmt_pts(pts)]
            t.add_row(*(cells[:3] if compact else cells))
        for _ in range(max(0, bench_total - bench_shown)):
            t.add_row("BN", Text("—", style="dim"), *([""] * (1 if compact else 3)))
        extras: list[RenderableType] = [t]
        summary = rec.my_roster if rec is not None else None
        needs = summary.needs() if summary is not None else _needs_from_rows(rows)
        need_text = Text("Need: ", style="bold")
        need_text.append(", ".join(_label_needs(rows, needs)) if needs else "starters filled", style="yellow" if needs else "green")
        extras.append(need_text)
        if summary is not None:
            clash = [f"wk {w} x{n}" for w, n in sorted(summary.bye_weeks.items()) if n > 1]
            if clash:
                extras.append(Text("Bye clash: " + ", ".join(clash), style="red"))
            if summary.lineup_points:
                extras.append(Text(f"Lineup {summary.lineup_points:.0f} pts • bench {summary.bench_points:.0f}", style="dim"))
        title = f"My roster ({len(picks)} picks)"
        return Panel(Group(*extras), title=title, box=box.ROUNDED, padding=(0, 1))

    # -- center ------------------------------------------------------------------------
    def best_picks_panel(self, state: DraftState, rec: Recommendation | None, players: Mapping[str, Player],
                         compact: bool = False) -> RenderableType:
        """Top-N table; ``compact`` drops the bye / tier / score / reason columns."""
        t = Table(box=box.SIMPLE_HEAD, expand=True, padding=(0, 1), collapse_padding=True, show_edge=False)
        cols = [("#", dict(width=2, justify="right")), ("Player", dict(ratio=2, overflow="ellipsis", min_width=10)),
                ("Pos/Tm", dict(width=6)), ("Bye", dict(width=3, justify="right")), ("Proj", dict(width=4, justify="right")),
                ("VORP", dict(width=4, justify="right")), ("T", dict(width=2, justify="right")),
                ("ADP", dict(width=3, justify="right")), ("Avail", dict(width=5, justify="right")),
                ("Score", dict(width=5, justify="right")), ("Why", dict(ratio=3, overflow="ellipsis", min_width=10))]
        keep = [i for i, (name, _) in enumerate(cols) if not (compact and name in ("Bye", "T", "Score"))]
        for i in keep:
            name, kw = cols[i]
            t.add_column(name, no_wrap=True, **kw)
        if rec is None or not rec.best_overall:
            t.add_row("", Text("computing…", style="dim"), *[""] * (len(keep) - 2))
        for i, v in enumerate(rec.best_overall if rec else [], start=1):
            pl = v.player
            style = ""
            if i == 1:
                style = "bold green" if state.is_my_turn else "bold"
            name = Text(pl.name, style=style, no_wrap=True, overflow="ellipsis")
            if v.warnings:
                name.append(" !", style="red")
            cells = [str(i), name, f"{pl.position} {pl.team or 'FA'}", str(pl.bye_week or "-"),
                     _fmt_pts(v.projection.points), f"{v.vorp:+.0f}", str(v.tier), _fmt_adp(pl.adp),
                     _fmt_pct(v.availability_next), f"{v.score:.0f}",
                     Text(v.reasons[0] if v.reasons else "", style=style, no_wrap=True, overflow="ellipsis")]
            t.add_row(*[cells[j] for j in keep], style=style)
        title = "Best picks now"
        if state.is_my_turn:
            title += " — YOUR PICK"
        elif state.my_next_pick_no is not None and not state.is_complete:
            title += f" (for your pick #{state.my_next_pick_no})"
        return Panel(t, title=title, box=box.ROUNDED, padding=(0, 0),
                     border_style="green" if state.is_my_turn else "")

    def positions_panel(self, rec: Recommendation | None, state: DraftState, short: bool = False,
                        compact: bool = False) -> RenderableType:
        grid = Table.grid(expand=True, padding=(0, 1))
        for _ in range(3):
            grid.add_column(ratio=1)
        cells = [self.position_table(pos, rec.by_position.get(pos) if rec else None, short, compact)
                 for pos in SKILL_POSITIONS]
        grid.add_row(*cells[:3])
        grid.add_row(*cells[3:])
        return Panel(grid, title="By position", box=box.ROUNDED, padding=(0, 0))

    def position_table(self, pos: str, advice, short: bool = False, compact: bool = False) -> RenderableType:
        """One mini table: action badge + rationale + candidates (name, proj, avail, tier)."""
        head = Text()
        head.append(f" {pos} ", style="bold reverse")
        if advice is None:
            head.append(" …", style="dim")
            return Group(head)
        head.append(" ")
        head.append(f" {advice.action} ", style=action_style(advice.action))
        t = Table(box=None, expand=True, padding=(0, 1), collapse_padding=True, show_header=False, show_edge=False)
        t.add_column("Player", ratio=1, no_wrap=True, overflow="ellipsis", min_width=8)
        t.add_column("Proj", width=3, justify="right", no_wrap=True)
        t.add_column("Av", width=4, justify="right", no_wrap=True)
        if not compact:
            t.add_column("T", width=1, justify="right", no_wrap=True)
        for v in advice.candidates[: 2 if short else 3]:
            cells = [v.player.name, _fmt_pts(v.projection.points), _fmt_pct(v.availability_next)]
            t.add_row(*(cells if compact else cells + [str(v.tier)]))
        if not advice.candidates:
            t.add_row(Text("none available", style="dim"), *([""] * (2 if compact else 3)))
        rationale = Text(_short_rationale(advice.rationale, 72), style="dim")
        return Group(head, rationale, t)

    # -- right column ---------------------------------------------------------------------
    def recent_picks_panel(self, state: DraftState, players: Mapping[str, Player], n: int = 10,
                           compact: bool = False) -> RenderableType:
        """Last ``n`` picks, newest first (``compact`` drops the team column)."""
        t = Table(box=None, expand=True, padding=(0, 1), collapse_padding=True, show_header=True, show_edge=False)
        t.add_column("#", width=3, justify="right", no_wrap=True)
        if not compact:
            t.add_column("Team", width=7, no_wrap=True, overflow="ellipsis")
        t.add_column("Player", ratio=1, no_wrap=True, overflow="ellipsis", min_width=8)
        t.add_column("Pos", width=3, no_wrap=True)
        recent = sorted(state.picks, key=lambda p: p.pick_no)[-n:]
        my_slot = state.my_slot
        for pk in reversed(recent):
            slot = pk.draft_slot
            mine = my_slot is not None and slot == my_slot
            style = "bold cyan" if mine else ""
            cells = [str(pk.pick_no), state.slot_label(slot), _player_name(pk.player_id, players, pk),
                     _player_pos(pk.player_id, players, pk)]
            t.add_row(*(cells[:1] + cells[2:] if compact else cells), style=style)
        if not recent:
            t.add_row("", Text("no picks yet", style="dim"), *([""] if compact else ["", ""]))
        return Panel(t, title="Recent picks", box=box.ROUNDED, padding=(0, 0))

    def opponents_panel(self, state: DraftState, rec: Recommendation | None) -> RenderableType:
        t = Table(box=None, expand=True, padding=(0, 1), collapse_padding=True, show_header=True, show_edge=False)
        t.add_column("Team", width=7, no_wrap=True, overflow="ellipsis")
        t.add_column("Needs", ratio=1, no_wrap=True, overflow="ellipsis", min_width=8)
        t.add_column("Next", width=4, justify="right", no_wrap=True)
        summaries: Iterable[RosterSummary] = rec.opponent_rosters if rec is not None else []
        n = state.next_pick_no
        rows = []
        for s in summaries:
            picks = [p for p in state.draft.picks_for_slot(s.slot) if p >= n]
            nxt = picks[0] if picks else None
            rows.append((nxt if nxt is not None else 10 ** 6, s))
        rows.sort(key=lambda r: r[0])
        for nxt, s in rows:
            open_ = " ".join(_slot_abbr(k) if v == 1 else f"{_slot_abbr(k)}{v}" for k, v in s.open_starters.items() if v > 0)
            t.add_row(s.label, open_ or Text("full", style="green"), str(nxt) if nxt < 10 ** 6 else "-")
        if not rows:
            t.add_row(Text("…", style="dim"), "", "")
        pressure = rec.position_pressure if rec is not None else {}
        extra: list[RenderableType] = [t]
        if pressure:
            hot = sorted(pressure.items(), key=lambda kv: -kv[1])[:4]
            extra.append(Text("Pressure: " + ", ".join(f"{k} {v:.1f}" for k, v in hot if v > 0), style="dim"))
        return Panel(Group(*extra), title="Opponent needs", box=box.ROUNDED, padding=(0, 0))


_SLOT_ABBR = {"FLEX": "FLX", "SUPER_FLEX": "SFLX", "WRRB_FLEX": "W/R", "REC_FLEX": "W/T", "IDP_FLEX": "IDP"}


def _slot_abbr(slot: str) -> str:
    return _SLOT_ABBR.get(slot, slot)


def _short_rationale(text: str, n: int) -> str:
    """First clause(s) of the rationale that fit in ``n`` characters."""
    if len(text) <= n:
        return text
    parts = [p.strip() for p in text.split(";")]
    out = parts[0]
    for p in parts[1:]:
        if len(out) + 2 + len(p) > n:
            break
        out += "; " + p
    return _trunc(out, n)


def _default_slots(state: DraftState) -> list[str]:
    s = state.draft.settings or {}
    out: list[str] = []
    for key, label in (("slots_qb", "QB"), ("slots_rb", "RB"), ("slots_wr", "WR"), ("slots_te", "TE"),
                       ("slots_flex", "FLEX"), ("slots_super_flex", "SUPER_FLEX"), ("slots_k", "K"),
                       ("slots_def", "DEF"), ("slots_bn", "BN")):
        out.extend([label] * int(s.get(key, 0) or 0))
    return out or ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF"] + ["BN"] * max(0, state.draft.rounds - 9)


def _needs_from_rows(rows: Sequence[tuple[str, Player | None, float | None]]) -> list[str]:
    return [slot for slot, pl, _ in rows if pl is None and slot in SKILL_POSITIONS]


def _label_needs(rows: Sequence[tuple[str, Player | None, float | None]], needs: Sequence[str]) -> list[str]:
    """"RB2" style labels: the ordinal of the first open slot at each needed position."""
    out: list[str] = []
    for pos in needs:
        idx = 0
        label = pos
        for slot, pl, _ in rows:
            if slot != pos:
                continue
            idx += 1
            if pl is None:
                label = f"{pos}{idx}" if sum(1 for s, _, _ in rows if s == pos) > 1 else pos
                break
        if label not in out:
            out.append(label)
    return out


# ---------------------------------------------------------------------------
# Plain-text rendering
# ---------------------------------------------------------------------------

def render_text(rec: Recommendation | None, state: DraftState, players: dict[str, Player]) -> str:
    """Plain-text summary used by ``--no-tui`` and mock logs."""
    lines: list[str] = []
    d = state.draft
    name = state.league.name if state.league else (d.metadata.get("name") or f"Draft {d.draft_id}")
    if state.is_complete:
        lines.append(f"== {name} • draft complete ({len(state.picks)} picks)")
    else:
        slot = state.on_the_clock_slot
        who = state.slot_label(slot) if slot is not None else "?"
        lines.append(f"== {name} • {scoring_description(state)} • Round {state.current_round} • Pick {state.next_pick_no} • On the clock: {who}")
        if state.is_my_turn:
            fut = state.my_future_picks()
            then = f" (then #{fut[1]})" if len(fut) > 1 else ""
            lines.append(f"** YOUR PICK **{then}")
        elif state.my_next_pick_no is not None:
            after = state.my_pick_after_next
            lines.append(f"You pick in {state.picks_until_my_turn} (#{state.my_next_pick_no})" + (f", then #{after}" if after else ""))
    if state.picks:
        last = sorted(state.picks, key=lambda p: p.pick_no)[-3:]
        lines.append("Last picks: " + "; ".join(
            f"#{p.pick_no} {state.slot_label(p.draft_slot)}: {_player_name(p.player_id, players, p)} ({_player_pos(p.player_id, players, p)})"
            for p in last))
    if rec is None:
        lines.append("(no recommendation yet)")
        return "\n".join(lines)
    lines.append("Best picks now:")
    for i, v in enumerate(rec.best_overall, start=1):
        pl = v.player
        why = "; ".join(v.reasons[:2])
        warn = f" [!] {'; '.join(v.warnings)}" if v.warnings else ""
        lines.append(f" {i}. {pl.name} ({pl.position}, {pl.team or 'FA'}) proj {v.projection.points:.0f} "
                     f"VORP {v.vorp:+.0f} tier {v.tier} ADP {_fmt_adp(pl.adp)} avail {_fmt_pct(v.availability_next)} "
                     f"score {v.score:.0f} — {why}{warn}")
    lines.append("By position:")
    for pos in SKILL_POSITIONS:
        adv = rec.by_position.get(pos)
        if adv is None:
            continue
        cands = ", ".join(f"{c.player.name} {c.projection.points:.0f} ({_fmt_pct(c.availability_next)}, T{c.tier})"
                          for c in adv.candidates)
        lines.append(f" {pos:<3} {adv.action:<8} {adv.rationale} | {cands}")
    if rec.my_roster is not None:
        needs = rec.my_roster.needs()
        lines.append("Need: " + (", ".join(needs) if needs else "starters filled")
                     + f" • lineup {rec.my_roster.lineup_points:.0f} pts")
    if rec.notes:
        lines.append("Notes: " + " • ".join(rec.notes))
    if rec.claude_advice:
        lines.append("Claude: " + rec.claude_advice.strip())
    return "\n".join(lines)
