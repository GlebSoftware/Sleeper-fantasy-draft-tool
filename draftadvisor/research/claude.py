"""Optional Claude (Sonnet) layer: per-player research notes, on-the-clock advice, free-form Q&A.

See DESIGN.md §3.5. Everything in here is optional: when no API key is available
:class:`ClaudeResearcher` reports ``enabled == False`` and every method returns
``None`` / ``{}`` / ``""`` immediately. The ``anthropic`` SDK is imported lazily
(first real request), so importing this module never touches the network stack.

Design notes
------------
* Research notes are cached as JSON under ``research_dir()/notes/<player_id>.json``
  (``ResearchNote.to_dict``) and reused while younger than ``max_age_days``.
  Notes marked "(research unavailable)" (refusal / error) are retried on the next
  run instead of being treated as fresh.
* ``on_the_clock_advice`` is the only call on the draft loop's critical path. It
  streams a short answer with thinking **disabled** and ``effort="low"`` (on
  ``claude-sonnet-5`` an omitted ``thinking`` runs adaptive thinking, whose
  tokens count against ``max_tokens`` and could leave a truncated or empty
  answer on the 30-second clock), enforces ``asyncio.wait_for``, de-duplicates
  concurrent requests for the same draft state and caches results by
  ``(state.version, top candidate ids)``. The result is stored from a task
  done-callback, so a lookahead request whose awaiter was cancelled by the draft
  loop still lands in the cache instead of being paid for twice. It never raises.
  ``stop_reason == "max_tokens"`` is treated as a failure (``None`` +
  ``last_error``) rather than showing a half sentence as the pick.
* Research requests fail fast: a 400/401/403/404 (bad key, no model access,
  rejected schema, wrong model id) stops the whole run with ``last_error`` set
  and returns the notes gathered so far; a 429 pauses every worker for the
  server's ``retry-after`` and retries that player once. ``last_run_stats``
  (``ok`` / ``failed`` / ``cached`` / ``skipped``) describes the last run.
* Prompt caching: no ``cache_control`` markers are sent. The advice system prompt
  is ~300 tokens and the research/ask prompts are shorter, all below the 1024-token
  minimum cacheable prefix on Sonnet 5, so a marker would be silently ignored;
  padding the prompt past 1024 tokens would cost more per cache miss (write
  premium + extra input tokens) than it saves, and in a 12-team draft my picks are
  ~6 minutes apart, longer than the 5-minute cache TTL. The advice system prompt is
  still built once per league and kept byte-identical because it is cheap and
  deterministic, not because it is cached.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterable

from ..config import (
    CLAUDE_MODEL,
    CLAUDE_ON_CLOCK_TIMEOUT_S,
    CLAUDE_RESEARCH_CONCURRENCY,
    DEFAULT_SEASON,
    SKILL_POSITIONS,
    SLOT_ELIGIBILITY,
    research_dir,
)
from ..models import (
    DraftState,
    LeagueSettings,
    Pick,
    Player,
    Projection,
    Recommendation,
    ResearchNote,
)

log = logging.getLogger(__name__)

__all__ = [
    "ClaudeResearcher",
    "build_context_text",
    "describe_scoring",
    "describe_roster_slots",
    "UNAVAILABLE_SUMMARY",
    "RESEARCH_SCHEMA",
    "ResearchAborted",
]

#: Summary text stored in a minimal note when research failed / was refused.
UNAVAILABLE_SUMMARY = "(research unavailable)"

#: Number of candidates shown to Claude on the clock (and used in the cache key).
ADVICE_TOP_N = 8
#: Number of most recent picks shown in the context.
RECENT_PICKS_N = 6

WEB_SEARCH_TOOL: dict[str, Any] = {"type": "web_search_20260209", "name": "web_search", "max_uses": 3}

#: Output budget for on-the-clock advice (thinking disabled, 2-4 plain sentences).
ADVICE_MAX_TOKENS = 600
#: Output budget for one research request (adaptive thinking + up to 3 searches share it) and
#: the larger budget used for the single retry after ``stop_reason == "max_tokens"``.
RESEARCH_MAX_TOKENS = 4096
RESEARCH_RETRY_MAX_TOKENS = 12288
#: Output budget for :meth:`ClaudeResearcher.ask`.
ASK_MAX_TOKENS = 1500
#: Appended to an ``ask`` answer that hit ``max_tokens``.
TRUNCATION_MARKER = "\n\n[answer truncated: max_tokens reached]"

#: HTTP statuses that mean the run cannot succeed (bad key, no access, rejected request, wrong
#: model id). One of these aborts a research run instead of burning every remaining request.
FATAL_STATUS_CODES = frozenset({400, 401, 403, 404})
FATAL_ERROR_NAMES = frozenset({"AuthenticationError", "PermissionDeniedError", "BadRequestError", "NotFoundError"})
#: Pause applied to every research worker after a 429 when the server sends no ``retry-after``.
RATE_LIMIT_DEFAULT_PAUSE_S = 15.0
RATE_LIMIT_MAX_PAUSE_S = 120.0
#: Content-block types of the server-side web search tool (answer text follows the last one).
SERVER_TOOL_BLOCK_TYPES = frozenset({"server_tool_use", "web_search_tool_result"})


class ResearchAborted(Exception):
    """Raised inside a research run when a fatal API error makes further requests pointless."""


#: Structured-output schema for a research note.
RESEARCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "<= 60 words: current situation, role, health, outlook."},
        "injury_risk": {"type": "number", "description": "0 (healthy, durable) .. 1 (currently out / very likely to miss games)."},
        "role_certainty": {"type": "number", "description": "0 (role unknown / committee / may lose job) .. 1 (locked-in starter)."},
        "upside": {"type": "string", "description": "<= 25 words: best case."},
        "downside": {"type": "string", "description": "<= 25 words: worst case."},
        "sources": {"type": "array", "items": {"type": "string"}, "description": "URLs consulted."},
    },
    "required": ["summary", "injury_risk", "role_certainty", "upside", "downside", "sources"],
    "additionalProperties": False,
}

RESEARCH_SYSTEM_PROMPT = (
    "You are a sharp, concise fantasy-football analyst preparing notes for the {season} NFL season "
    "redraft market. For the player you are given, use web search (at most 3 searches) to find the "
    "latest news: injury status and history, depth-chart role, offseason changes (coaching, scheme, "
    "competition, contract), training-camp reports and preseason usage. Then fill the JSON schema. "
    "Be factual and current; if you cannot find reliable recent information, say so in the summary and "
    "use moderate values (injury_risk 0.1-0.2, role_certainty 0.5). Never invent sources."
)

ADVICE_STRATEGY_GUIDANCE = (
    "Advice principles: the numbers you receive are league-scored season projections; VORP is points over "
    "positional replacement; 'avail' is the probability the player is still there at the manager's next pick. "
    "Prefer the best value that fills a starting need; take a player before a tier break at a scarce position; "
    "if the top option is very likely (>=85%) to be available at the next pick, consider taking a scarcer player "
    "now and him later; do not draft K/DEF before the last three rounds; treat bye-week overlap as a minor tie-breaker; "
    "weigh injury/role notes but do not over-react to a single report. "
    "Respond in 2-4 short sentences of plain text (no markdown, no lists, no preamble): name the pick, give the key "
    "reason, and name the fallback if that player is gone."
)

ASK_SYSTEM_PROMPT = (
    "You are an expert fantasy-football draft analyst helping a manager during or before their Sleeper draft. "
    "Use the draft context provided (roster, candidates, projections, notes) as the primary source of truth and "
    "answer the question directly and concisely, with concrete player names and numbers when relevant."
)


# ---------------------------------------------------------------------------
# Small helpers (pure)
# ---------------------------------------------------------------------------


def _clamp01(x: Any, default: float) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if v != v:  # NaN
        return default
    return max(0.0, min(1.0, v))


def _describe_error(exc: BaseException) -> str:
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    text = str(exc).replace("\n", " ")[:200]
    return f"{name}({status}): {text}" if status else f"{name}: {text}"


def _error_kind(exc: BaseException) -> str:
    """Classify an API error without importing the SDK: ``"fatal"`` (400/401/403/404 - the run
    cannot succeed), ``"rate_limit"`` (429) or ``"other"`` (retryable / unknown)."""
    status = getattr(exc, "status_code", None)
    name = type(exc).__name__
    if status in FATAL_STATUS_CODES or name in FATAL_ERROR_NAMES:
        return "fatal"
    if status == 429 or name == "RateLimitError":
        return "rate_limit"
    return "other"


def _retry_after_seconds(exc: BaseException) -> float:
    """``retry-after`` from a 429 response (seconds; HTTP-date form falls back to the default), clamped."""
    delay = RATE_LIMIT_DEFAULT_PAUSE_S
    headers = getattr(getattr(exc, "response", None), "headers", None)
    raw = None
    if headers is not None:
        try:
            raw = headers.get("retry-after")
        except Exception:  # unusual header container
            raw = None
    if raw is not None:
        try:
            delay = float(raw)
        except (TypeError, ValueError):
            delay = RATE_LIMIT_DEFAULT_PAUSE_S
    return max(0.0, min(RATE_LIMIT_MAX_PAUSE_S, delay))


def _supports_thinking_disabled(model: str) -> bool:
    """``thinking={"type": "disabled"}`` is accepted on Sonnet/Opus/Haiku ids but returns 400 on the
    Fable / Mythos line (thinking is always on there); those must omit the parameter."""
    m = (model or "").lower()
    return not ("fable" in m or "mythos" in m)


def _text_blocks(message: Any) -> list[str]:
    """All non-empty text blocks of a response, in order (tolerates fake objects)."""
    out: list[str] = []
    for block in getattr(message, "content", None) or []:
        if getattr(block, "type", None) == "text":
            text = getattr(block, "text", None)
            if text:
                out.append(str(text))
    return out


def _answer_text_blocks(message: Any) -> list[str]:
    """Text blocks after the last server-tool block (the structured answer once searching is
    done); the whole text list when there were no tool blocks or nothing follows them."""
    blocks = list(getattr(message, "content", None) or [])
    last_tool = -1
    for i, block in enumerate(blocks):
        if getattr(block, "type", None) in SERVER_TOOL_BLOCK_TYPES:
            last_tool = i
    tail: list[str] = []
    for block in blocks[last_tool + 1:]:
        if getattr(block, "type", None) == "text":
            text = getattr(block, "text", None)
            if text:
                tail.append(str(text))
    return tail or _text_blocks(message)


def _response_text(message: Any) -> str:
    return "\n".join(_text_blocks(message)).strip()


def _parse_json_object(texts: Iterable[str]) -> dict[str, Any] | None:
    """Parse the JSON object Claude produced: each block alone (last first), then the blocks
    concatenated with ``''`` (a structured answer may be split across blocks by citations; a
    newline inside a string literal would make it invalid) sliced at the outer braces."""
    texts = [t for t in texts if t]
    for candidate in reversed(texts):
        try:
            obj = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    joined = "".join(texts)
    start, end = joined.find("{"), joined.rfind("}")
    if 0 <= start < end:
        try:
            obj = json.loads(joined[start:end + 1])
        except ValueError:
            return None
        if isinstance(obj, dict):
            return obj
    return None


def _parse_research_response(response: Any) -> dict[str, Any] | None:
    """Structured note from a research response; prefers the text after the last search block."""
    answer = _answer_text_blocks(response)
    data = _parse_json_object(answer)
    if data is None:
        everything = _text_blocks(response)
        if everything != answer:
            data = _parse_json_object(everything)
    if data is None and log.isEnabledFor(logging.DEBUG):
        log.debug("research response not parseable as JSON; raw text blocks: %r", _text_blocks(response))
    return data


def _player_line(pl: Player, points: float | None = None) -> str:
    bits = [f"{pl.name} ({pl.position}, {pl.team or 'FA'}"]
    if pl.bye_week:
        bits.append(f", bye {pl.bye_week}")
    bits.append(")")
    if points is not None:
        bits.append(f" {points:.0f} pts")
    if pl.injury_status and pl.injury_status not in ("NA", "Healthy"):
        bits.append(f" [{pl.injury_status}]")
    return "".join(bits)


def describe_scoring(league: LeagueSettings | None) -> str:
    """One-line human description of the league scoring (stable text for the cached system prompt)."""
    if league is None:
        return "scoring unknown (assume half-PPR)"
    s = league.scoring_settings or {}

    def g(key: str, default: float = 0.0) -> float:
        try:
            return float(s.get(key, default) or 0.0)
        except (TypeError, ValueError):
            return default

    parts = [f"{league.scoring_type.replace('_', '-').upper()} (rec {g('rec'):g})"]
    parts.append(f"pass TD {g('pass_td', 4):g}, pass yd {g('pass_yd', 0.04):g}/yd, INT {g('pass_int', -1):g}")
    parts.append(f"rush/rec TD {g('rush_td', 6):g}/{g('rec_td', 6):g}")
    if league.te_premium:
        parts.append(f"TE premium +{league.te_premium:g}/rec")
    for key, label in (("bonus_rec_wr", "WR bonus"), ("bonus_rec_rb", "RB bonus"), ("pass_2pt", "2pt pass")):
        if g(key):
            parts.append(f"{label} {g(key):g}")
    if league.is_superflex:
        parts.append("SUPERFLEX")
    return "; ".join(parts)


def describe_roster_slots(league: LeagueSettings | None) -> str:
    """Compact roster-shape description, e.g. 'QB, RB x2, WR x2, TE, FLEX(RB/WR/TE), K, DEF; bench 6'."""
    if league is None:
        return "starting slots unknown"
    counts: dict[str, int] = {}
    for slot in league.starting_slots:
        counts[slot] = counts.get(slot, 0) + 1
    items = []
    for slot, n in counts.items():
        elig = SLOT_ELIGIBILITY.get(slot)
        label = slot
        if elig and len(elig) > 1:
            label = f"{slot}({'/'.join(sorted(elig, key=SKILL_POSITIONS.index))})"
        items.append(f"{label} x{n}" if n > 1 else label)
    extra = f"; bench {league.bench_slots}"
    ir = league.roster_positions.count("IR")
    if ir:
        extra += f", IR {ir}"
    return ", ".join(items) + extra


# ---------------------------------------------------------------------------
# Context text (shared by advice and ask)
# ---------------------------------------------------------------------------


def _pick_name(pk: Pick, players: dict[str, Player]) -> str:
    pl = players.get(pk.player_id)
    if pl is not None:
        return f"{pl.name} ({pl.position})"
    pos = pk.position
    return f"{pk.player_name} ({pos})" if pos else pk.player_name


def _roster_section(state: DraftState, rec: Recommendation | None, players: dict[str, Player]) -> list[str]:
    lines: list[str] = ["MY ROSTER:"]
    summary = rec.my_roster if rec is not None else None
    if summary is not None:
        by_pos: dict[str, list[Player]] = {}
        for pl in summary.players:
            by_pos.setdefault(pl.position, []).append(pl)
        proj_by_id = {pv.player_id: pv.projection for pv in (rec.best_overall if rec else [])}
        if not summary.players:
            lines.append("  (empty)")
        for pos in SKILL_POSITIONS:
            group = by_pos.get(pos)
            if not group:
                continue
            names = ", ".join(_player_line(pl, proj_by_id[pl.player_id].points if pl.player_id in proj_by_id else None)
                              for pl in group)
            lines.append(f"  {pos}: {names}")
        for pos, group in by_pos.items():
            if pos not in SKILL_POSITIONS:
                lines.append(f"  {pos}: {', '.join(pl.name for pl in group)}")
        slots = ", ".join(
            f"{slot} {summary.starters_filled.get(slot, 0)}/{summary.starters_filled.get(slot, 0) + n}"
            for slot, n in summary.open_starters.items()
        )
        if slots:
            lines.append(f"  Starting slots filled: {slots}")
        needs = [slot for slot, n in summary.open_starters.items() if n > 0]
        lines.append(f"  Open needs: {', '.join(needs) if needs else 'none (bench depth only)'}")
        clashes = [f"week {w} x{n}" for w, n in sorted(summary.bye_weeks.items()) if n >= 2]
        if clashes:
            lines.append(f"  Bye clashes among starters: {', '.join(clashes)}")
        if summary.lineup_points:
            lines.append(f"  Projected lineup points so far: {summary.lineup_points:.0f}")
        return lines
    mine = state.my_picks()
    if not mine:
        lines.append("  (empty)")
    for pk in mine:
        lines.append(f"  R{pk.round}: {_pick_name(pk, players)}")
    return lines


def _candidate_lines(rec: Recommendation, notes: dict[str, ResearchNote] | None) -> list[str]:
    lines = [f"TOP {min(ADVICE_TOP_N, len(rec.best_overall))} CANDIDATES (by model score):"]
    for i, pv in enumerate(rec.best_overall[:ADVICE_TOP_N], start=1):
        pl, pr = pv.player, pv.projection
        head = (
            f"  {i}. {_player_line(pl)} - {pr.points:.0f} pts, VORP {pv.vorp:+.0f}, "
            f"avail@next {pv.availability_next * 100:.0f}%, tier {pv.tier}, "
            f"pos rank {pv.pos_rank}, score {pv.score:.0f}"
        )
        if pl.adp:
            head += f", ADP {pl.adp:.0f}"
        lines.append(head)
        if pv.reasons:
            lines.append(f"     why: {'; '.join(pv.reasons[:4])}")
        if pv.warnings:
            lines.append(f"     warn: {'; '.join(pv.warnings[:3])}")
        note = (notes or {}).get(pl.player_id)
        if note is not None and note.summary and note.summary != UNAVAILABLE_SUMMARY:
            lines.append(
                f"     note: {note.summary} (injury risk {note.injury_risk:.1f}, role certainty {note.role_certainty:.1f})"
            )
    return lines


def _position_lines(rec: Recommendation) -> list[str]:
    lines = ["POSITION OUTLOOK:"]
    for pos in SKILL_POSITIONS:
        adv = rec.by_position.get(pos)
        if adv is None:
            continue
        best = adv.candidates[0] if adv.candidates else None
        best_txt = f" best: {best.player.name} {best.projection.points:.0f} pts" if best else ""
        lines.append(f"  {pos}: {adv.action} - {adv.rationale}{best_txt}")
    return lines


def _recent_picks_lines(state: DraftState, players: dict[str, Player]) -> list[str]:
    recent = sorted(state.picks, key=lambda p: p.pick_no)[-RECENT_PICKS_N:]
    lines = [f"LAST {len(recent)} PICKS:"]
    if not recent:
        lines.append("  (none yet)")
    for pk in recent:
        lines.append(f"  #{pk.pick_no} {state.slot_label(pk.draft_slot)}: {_pick_name(pk, players)}")
    return lines


def build_context_text(
    state: DraftState,
    rec: Recommendation | None,
    players: dict[str, Player],
    notes: dict[str, ResearchNote] | None,
) -> str:
    """Compact plain-text picture of the draft for Claude (roster, needs, candidates, picks, pressure)."""
    draft = state.draft
    league = state.league
    lines: list[str] = []
    league_name = league.name if league is not None else "League"
    on_clock = state.on_the_clock_slot
    header = (
        f"{league_name}: round {state.current_round} of {draft.rounds}, pick #{state.next_pick_no} "
        f"of {draft.total_picks}, {draft.teams} teams"
    )
    if on_clock is not None:
        header += f"; on the clock: {state.slot_label(on_clock)}"
    lines.append(header)
    if state.my_slot is not None:
        fut = state.my_future_picks()
        mine = f"I am slot {state.my_slot} ({state.slot_label(state.my_slot)})."
        if state.is_my_turn:
            mine += " IT IS MY PICK NOW."
        if fut:
            nxt = ", ".join(f"#{p}" for p in fut[:3])
            until = state.picks_until_my_turn
            mine += f" My next picks: {nxt}" + (f" ({until} picks away)." if until else ".")
        lines.append(mine)
    lines.append("")
    lines.extend(_roster_section(state, rec, players))
    lines.append("")
    if rec is not None:
        lines.extend(_candidate_lines(rec, notes))
        lines.append("")
        lines.extend(_position_lines(rec))
        lines.append("")
    lines.extend(_recent_picks_lines(state, players))
    if rec is not None and rec.position_pressure:
        pressure = ", ".join(
            f"{pos} {rec.position_pressure[pos]:.1f}" for pos in SKILL_POSITIONS if pos in rec.position_pressure
        )
        lines.append("")
        lines.append(f"EXPECTED PICKS BY POSITION BEFORE MY NEXT TURN: {pressure}")
    if rec is not None and rec.notes:
        lines.append("NOTES: " + " | ".join(rec.notes[:5]))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Researcher
# ---------------------------------------------------------------------------


class ClaudeResearcher:
    """Optional Claude layer. Safe to construct anywhere; does nothing when disabled."""

    #: Upper bound for one research request (web search included).
    research_timeout_s: float = 120.0
    #: Upper bound for :meth:`ask`.
    ask_timeout_s: float = 120.0
    #: Max cached advice entries kept in memory.
    _ADVICE_CACHE_MAX = 64

    def __init__(
        self,
        api_key: str | None = None,
        model: str = CLAUDE_MODEL,
        cache_dir: Path | None = None,
        client: Any = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY") or None
        self.model = model or CLAUDE_MODEL
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._client = client
        self._sdk_unavailable = False
        self.last_error: str | None = None
        #: Outcome of the last :meth:`research_players` run: ``ok`` (parsed notes), ``failed``
        #: (minimal notes / fatal error), ``cached`` (fresh notes reused), ``skipped`` (not
        #: attempted because the run was aborted).
        self.last_run_stats: dict[str, int] = {"ok": 0, "failed": 0, "cached": 0, "skipped": 0}
        self.season: int = DEFAULT_SEASON
        self._advice_cache: dict[tuple, str] = {}
        self._advice_inflight: dict[tuple, "asyncio.Task[str | None]"] = {}
        self._system_cache: dict[str, str] = {}
        self._rate_limited_until: float = 0.0  # loop.time() deadline shared by research workers
        self.request_count = 0

    # -- state ----------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        if self._sdk_unavailable:
            return False
        return self._client is not None or bool(self._api_key)

    @property
    def notes_dir(self) -> Path:
        base = self._cache_dir if self._cache_dir is not None else research_dir()
        return base / "notes"

    def _get_client(self) -> Any | None:
        """Lazily build the AsyncAnthropic client (imports the SDK on first use)."""
        if self._client is not None:
            return self._client
        if not self._api_key or self._sdk_unavailable:
            return None
        try:
            import anthropic  # noqa: WPS433 (lazy import by design)
        except Exception as exc:  # pragma: no cover - SDK missing/broken
            log.warning("anthropic SDK unavailable, Claude layer disabled: %s", _describe_error(exc))
            self._sdk_unavailable = True
            return None
        self._client = anthropic.AsyncAnthropic(api_key=self._api_key, max_retries=2)
        return self._client

    # -- notes on disk --------------------------------------------------------
    def _note_path(self, player_id: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(player_id))
        return self.notes_dir / f"{safe}.json"

    def load_notes(self) -> dict[str, ResearchNote]:
        """Read every cached note from ``notes_dir`` (corrupt files are skipped)."""
        out: dict[str, ResearchNote] = {}
        d = self.notes_dir
        if not d.is_dir():
            return out
        for path in sorted(d.glob("*.json")):
            try:
                with open(path, encoding="utf-8") as fh:
                    note = ResearchNote.from_dict(json.load(fh))
            except Exception as exc:
                log.warning("skipping corrupt research note %s: %s", path.name, _describe_error(exc))
                continue
            out[note.player_id] = note
        return out

    def save_note(self, note: ResearchNote) -> None:
        """Atomically write one note to disk (errors are logged, never raised)."""
        path = self._note_path(note.player_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(note.to_dict(), fh, indent=1)
            os.replace(tmp, path)
        except OSError as exc:
            log.warning("could not write research note %s: %s", path, _describe_error(exc))

    @staticmethod
    def _is_fresh(note: ResearchNote | None, max_age_days: float) -> bool:
        if note is None or note.summary == UNAVAILABLE_SUMMARY:
            return False
        return (time.time() - note.generated_at) <= max_age_days * 86400.0

    def _minimal_note(self, player_id: str, reason: str) -> ResearchNote:
        log.info("research unavailable for %s: %s", player_id, reason)
        return ResearchNote(
            player_id=player_id, summary=UNAVAILABLE_SUMMARY, injury_risk=0.0, role_certainty=0.5,
            upside="", downside="", sources=[], generated_at=time.time(), model=None,
        )

    # -- research -------------------------------------------------------------
    async def research_players(
        self,
        players: list[Player],
        projections: dict[str, Projection] | None = None,
        max_age_days: float = 3,
        concurrency: int = CLAUDE_RESEARCH_CONCURRENCY,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> dict[str, ResearchNote]:
        """Research each player (one web-search request each), caching notes on disk.

        Returns notes for every requested player that was attempted (fresh cached notes are
        reused without a request; per-player failures yield a minimal "(research unavailable)"
        note so callers never block). A fatal API error (bad key, no model access, rejected
        request, wrong model id) aborts the run: ``last_error`` is set, the remaining players are
        left un-researched (no note written, so they are retried next run) and whatever was
        gathered is returned. ``last_run_stats`` summarises the run.
        """
        stats = {"ok": 0, "failed": 0, "cached": 0, "skipped": 0}
        self.last_run_stats = stats
        if not players:
            return {}
        cached = self.load_notes()
        result: dict[str, ResearchNote] = {}
        todo: list[Player] = []
        seen: set[str] = set()
        for pl in players:
            if pl.player_id in seen:
                continue
            seen.add(pl.player_id)
            note = cached.get(pl.player_id)
            if self._is_fresh(note, max_age_days):
                result[pl.player_id] = note  # type: ignore[assignment]
                stats["cached"] += 1
            else:
                todo.append(pl)
        if not todo:
            return result
        if not self.enabled or self._get_client() is None:
            log.info("Claude disabled: %d players left un-researched", len(todo))
            stats["skipped"] = len(todo)
            return result
        log.info("researching %d players (%d cached)", len(todo), len(result))
        sem = asyncio.Semaphore(max(1, int(concurrency)))
        total, done = len(todo), 0
        aborted = False

        async def one(pl: Player) -> None:
            nonlocal done, aborted
            if aborted:
                stats["skipped"] += 1
                return
            async with sem:
                if aborted:  # a sibling hit a fatal error while we waited for the semaphore
                    stats["skipped"] += 1
                    return
                proj = (projections or {}).get(pl.player_id)
                try:
                    note = await self._research_one(pl, proj)
                except ResearchAborted as exc:
                    if not aborted:
                        log.error("research aborted: %s", exc)
                    aborted = True
                    stats["failed"] += 1
                    return
            self.save_note(note)
            result[pl.player_id] = note
            stats["ok" if note.summary != UNAVAILABLE_SUMMARY else "failed"] += 1
            done += 1
            if progress is not None:
                try:
                    progress(done, total, pl.name)
                except Exception as exc:  # never let a UI callback break research
                    log.debug("progress callback failed: %s", _describe_error(exc))

        await asyncio.gather(*(one(pl) for pl in todo))
        log.info("research run: %d ok, %d failed, %d cached, %d skipped",
                 stats["ok"], stats["failed"], stats["cached"], stats["skipped"])
        return result

    async def _wait_for_rate_limit(self) -> None:
        """Block until the shared 429 pause (set by any worker) has elapsed."""
        while True:
            remaining = self._rate_limited_until - asyncio.get_running_loop().time()
            if remaining <= 0:
                return
            await asyncio.sleep(remaining)

    def _pause_for_rate_limit(self, exc: BaseException) -> float:
        """Record a 429: every research worker waits ``retry-after`` before its next request."""
        delay = _retry_after_seconds(exc)
        loop = asyncio.get_running_loop()
        self._rate_limited_until = max(self._rate_limited_until, loop.time() + delay)
        log.warning("Claude rate limited (%s); pausing research for %.0fs", _describe_error(exc), delay)
        return delay

    def _research_user_message(self, pl: Player, proj: Projection | None) -> str:
        facts = [
            f"Today is {date.today().isoformat()}. Research this player for the {self.season} fantasy season.",
            f"Player: {pl.name}, {pl.position}, team {pl.team or 'free agent'}.",
        ]
        if pl.age:
            facts.append(f"Age {pl.age:.0f}.")
        if pl.years_exp is not None:
            facts.append("Rookie." if pl.years_exp == 0 else f"{pl.years_exp} NFL seasons.")
        if pl.injury_status:
            facts.append(f"Sleeper injury status: {pl.injury_status}.")
        if pl.depth_chart_order:
            facts.append(f"Depth chart: {pl.depth_chart_position or pl.position}{pl.depth_chart_order}.")
        if pl.adp:
            facts.append(f"ADP {pl.adp:.0f}.")
        if pl.ecr:
            facts.append(f"Expert consensus rank {pl.ecr:.0f}.")
        if proj is not None:
            facts.append(f"Our projection: {proj.points:.0f} season points ({proj.ppg:.1f}/game over {proj.games:.0f} games).")
        facts.append("Fill in the schema. Keep summary <= 60 words, upside/downside <= 25 words each.")
        return " ".join(facts)

    async def _research_call(self, client: Any, user_text: str, max_tokens: int) -> Any:
        """One research request (plus the ``pause_turn`` continuation). Raises SDK errors."""
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_text}]
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": RESEARCH_SYSTEM_PROMPT.format(season=self.season)}],
            tools=[dict(WEB_SEARCH_TOOL)],
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": RESEARCH_SCHEMA}},
        )
        response = await asyncio.wait_for(
            client.messages.create(messages=messages, **kwargs), self.research_timeout_s,
        )
        self.request_count += 1
        if getattr(response, "stop_reason", None) == "pause_turn":
            messages = messages + [{"role": "assistant", "content": response.content}]
            response = await asyncio.wait_for(
                client.messages.create(messages=messages, **kwargs), self.research_timeout_s,
            )
            self.request_count += 1
        return response

    async def _research_one(self, pl: Player, proj: Projection | None) -> ResearchNote:
        """Research one player. Per-player problems (timeout, refusal, unparseable, 5xx, network)
        become a minimal note; a fatal API error raises :class:`ResearchAborted`; a 429 pauses
        all workers for ``retry-after`` and retries once; ``stop_reason == "max_tokens"`` retries
        once with a larger output budget."""
        client = self._get_client()
        if client is None:
            return self._minimal_note(pl.player_id, "disabled")
        user_text = self._research_user_message(pl, proj)
        max_tokens = RESEARCH_MAX_TOKENS
        rate_limit_retried = False
        while True:
            await self._wait_for_rate_limit()
            try:
                response = await self._research_call(client, user_text, max_tokens)
            except asyncio.TimeoutError:
                return self._minimal_note(pl.player_id, "timeout")
            except Exception as exc:
                self.last_error = _describe_error(exc)
                kind = _error_kind(exc)
                if kind == "fatal":
                    raise ResearchAborted(self.last_error) from exc
                if kind == "rate_limit" and not rate_limit_retried:
                    self._pause_for_rate_limit(exc)
                    rate_limit_retried = True
                    continue
                return self._minimal_note(pl.player_id, self.last_error)
            stop = getattr(response, "stop_reason", None)
            if stop == "max_tokens" and max_tokens < RESEARCH_RETRY_MAX_TOKENS:
                log.info("research for %s hit max_tokens=%d; retrying with %d",
                         pl.name, max_tokens, RESEARCH_RETRY_MAX_TOKENS)
                max_tokens = RESEARCH_RETRY_MAX_TOKENS
                continue
            break
        if stop == "refusal":
            return self._minimal_note(pl.player_id, "refusal")
        if stop == "max_tokens":
            self.last_error = f"max_tokens ({RESEARCH_RETRY_MAX_TOKENS}) exceeded twice for {pl.name}"
            return self._minimal_note(pl.player_id, self.last_error)
        data = _parse_research_response(response)
        if data is None:
            return self._minimal_note(pl.player_id, f"unparseable response (stop_reason={stop})")
        return self._note_from_data(pl.player_id, data)

    def _note_from_data(self, player_id: str, data: dict[str, Any]) -> ResearchNote:
        sources = data.get("sources") or []
        if not isinstance(sources, list):
            sources = [str(sources)]
        return ResearchNote(
            player_id=player_id,
            summary=str(data.get("summary") or "").strip() or UNAVAILABLE_SUMMARY,
            injury_risk=_clamp01(data.get("injury_risk"), 0.0),
            role_certainty=_clamp01(data.get("role_certainty"), 0.5),
            upside=str(data.get("upside") or "").strip(),
            downside=str(data.get("downside") or "").strip(),
            sources=[str(s) for s in sources if s][:10],
            generated_at=time.time(),
            model=self.model,
        )

    # -- on the clock ---------------------------------------------------------
    def advice_system_prompt(self, league: LeagueSettings | None) -> str:
        """Stable (byte-identical per league) system prompt so prompt caching hits."""
        key = league.league_id if league is not None else "-"
        cached = self._system_cache.get(key)
        if cached is not None:
            return cached
        teams = league.total_rosters if league is not None else "?"
        name = league.name if league is not None else "the league"
        season = league.season if league is not None else self.season
        text = (
            f"You are an expert fantasy-football draft analyst advising one manager live during a {season} "
            f"Sleeper draft ({name}, {teams} teams) with a 30-second pick clock.\n"
            f"Scoring: {describe_scoring(league)}.\n"
            f"Starting lineup: {describe_roster_slots(league)}.\n"
            f"{ADVICE_STRATEGY_GUIDANCE}"
        )
        self._system_cache[key] = text
        return text

    @staticmethod
    def _advice_key(state: DraftState, rec: Recommendation) -> tuple:
        return (state.version, tuple(pv.player_id for pv in rec.best_overall[:ADVICE_TOP_N]))

    async def on_the_clock_advice(
        self,
        state: DraftState,
        rec: Recommendation,
        players: dict[str, Player],
        notes: dict[str, ResearchNote],
        timeout: float = CLAUDE_ON_CLOCK_TIMEOUT_S,
    ) -> str | None:
        """2-4 sentences of advice for the current pick, or None (disabled / timeout / error).

        Cached by ``(state.version, top candidate ids)``; concurrent calls for the same key share
        one request. On success the text is also stored in ``rec.claude_advice``. The request task
        stores its own result via a done-callback, so a caller that is cancelled while waiting
        (the draft loop cancels the lookahead request when my pick comes up) does not waste the
        answer: the next call for the same state is a cache hit.
        """
        if not self.enabled or rec is None or not rec.best_overall:
            return None
        key = self._advice_key(state, rec)
        hit = self._advice_cache.get(key)
        if hit is not None:
            rec.claude_advice = hit
            return hit
        task = self._advice_inflight.get(key)
        owner = task is None
        if task is None:
            task = asyncio.ensure_future(self._advice_request(state, rec, players, notes))
            self._advice_inflight[key] = task
            task.add_done_callback(lambda t, key=key: self._advice_done(key, t))
        try:
            text = await asyncio.wait_for(asyncio.shield(task), timeout)
        except asyncio.TimeoutError:
            log.warning("Claude advice timed out after %.1fs", timeout)
            if owner:
                task.cancel()
            text = None
        except asyncio.CancelledError:
            if not task.cancelled():
                raise  # the caller itself was cancelled; the request finishes and caches itself
            text = None
        except Exception as exc:
            self.last_error = _describe_error(exc)
            log.warning("Claude advice failed: %s", self.last_error)
            text = None
        if text:
            self._remember_advice(key, text)
            rec.claude_advice = text
        return text

    def _advice_done(self, key: tuple, task: "asyncio.Task[str | None]") -> None:
        """Done-callback of an advice request: drop it from the in-flight table and cache a
        successful answer even when nobody is awaiting it any more."""
        if self._advice_inflight.get(key) is task:
            self._advice_inflight.pop(key, None)
        if task.cancelled():
            return
        exc = task.exception()  # also marks the exception as retrieved when no awaiter is left
        if exc is not None:
            self.last_error = _describe_error(exc)
            return
        text = task.result()
        if text:
            self._remember_advice(key, text)

    def _remember_advice(self, key: tuple, text: str) -> None:
        if len(self._advice_cache) >= self._ADVICE_CACHE_MAX:
            oldest = next(iter(self._advice_cache))
            self._advice_cache.pop(oldest, None)
        self._advice_cache[key] = text

    async def _advice_request(
        self,
        state: DraftState,
        rec: Recommendation,
        players: dict[str, Player],
        notes: dict[str, ResearchNote],
    ) -> str | None:
        client = self._get_client()
        if client is None:
            return None
        context = build_context_text(state, rec, players, notes)
        user_text = f"{context}\n\nWho should I pick now? Answer in 2-4 sentences."
        t0 = time.perf_counter()
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=ADVICE_MAX_TOKENS,
            system=[{"type": "text", "text": self.advice_system_prompt(state.league)}],
            messages=[{"role": "user", "content": user_text}],
            output_config={"effort": "low"},
        )
        if _supports_thinking_disabled(self.model):
            # On Sonnet 5 an omitted ``thinking`` runs adaptive thinking whose tokens count against
            # max_tokens; on the 30-second clock we want the whole budget to be the answer.
            kwargs["thinking"] = {"type": "disabled"}
        async with client.messages.stream(**kwargs) as stream:
            message = await stream.get_final_message()
        self.request_count += 1
        stop = getattr(message, "stop_reason", None)
        if stop == "refusal":
            log.info("Claude refused the advice request")
            return None
        if stop == "max_tokens":
            # A cut-off sentence could name the wrong pick or fallback; better no advice than that.
            self.last_error = f"advice truncated (stop_reason=max_tokens at {ADVICE_MAX_TOKENS} tokens)"
            log.warning("Claude advice hit max_tokens; discarding partial answer")
            return None
        text = _response_text(message)
        log.info("Claude advice in %.0f ms (%d chars)", (time.perf_counter() - t0) * 1000, len(text))
        return text or None

    # -- free-form ------------------------------------------------------------
    async def ask(self, question: str, context_text: str) -> str:
        """Free-form question with draft context. Returns "" when disabled or on error
        (``last_error`` then holds the reason)."""
        if not self.enabled:
            return ""
        client = self._get_client()
        if client is None:
            return ""
        content = f"{context_text.strip()}\n\nQUESTION: {question.strip()}" if context_text else question.strip()
        try:
            response = await asyncio.wait_for(
                client.messages.create(
                    model=self.model,
                    max_tokens=ASK_MAX_TOKENS,
                    system=[{"type": "text", "text": ASK_SYSTEM_PROMPT}],
                    messages=[{"role": "user", "content": content}],
                    output_config={"effort": "medium"},
                ),
                self.ask_timeout_s,
            )
            self.request_count += 1
        except asyncio.TimeoutError:
            self.last_error = "timeout"
            log.warning("Claude ask timed out")
            return ""
        except Exception as exc:
            self.last_error = _describe_error(exc)
            log.warning("Claude ask failed: %s", self.last_error)
            return ""
        stop = getattr(response, "stop_reason", None)
        if stop == "refusal":
            self.last_error = "refusal"
            return ""
        text = _response_text(response)
        if stop == "max_tokens":
            self.last_error = f"answer truncated (stop_reason=max_tokens at {ASK_MAX_TOKENS} tokens)"
            log.warning("Claude ask hit max_tokens; answer truncated")
            if not text:
                return ""
            text += TRUNCATION_MARKER
        return text
