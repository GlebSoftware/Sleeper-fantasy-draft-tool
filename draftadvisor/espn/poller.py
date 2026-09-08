"""ESPN draft poller: keeps a :class:`DraftState` in sync with ``draftDetail`` (same contract as
:class:`draftadvisor.sleeper.poller.DraftPoller`, so ``cli.DraftLoop`` can consume either).

Each poll is one GET (views mDraftDetail + mSettings). Besides the picks and the drafted / inProgress
flags, the poll re-reads ``draftSettings.pickOrder``, ``timePerSelection``, ``type``, the lineup
slot counts and the owners of not-yet-made picks, so a commissioner changing the order / clock /
rounds (or a pick trade) before or during the draft is followed (the state is rebuilt, including "my"
slot). Whether ESPN appends to ``draftDetail.picks`` while a live draft is in progress or serves the
whole board and fills entries in place could not be verified offline: the change signature covers
every entry, so either shape (and a commissioner correcting an earlier pick) produces a new state.
A poll whose payload has no ``draftDetail`` at all while the previous one had is treated as an
error (degraded response), never as an empty board.

Cadence (``interval()``): ``poll_seconds`` normally; :attr:`EspnDraftPoller.my_turn_interval` when
``picks_until_my_turn <= 1``; :attr:`EspnDraftPoller.pre_draft_interval` before the draft starts.
Errors never escape :meth:`EspnDraftPoller.run` (logged, ``last_error``, ``on_error`` hook,
exponential backoff up to 30 s) except an unknown league (:class:`EspnNotFound`) or missing cookies
(:class:`EspnAccessDenied`) while bootstrapping, which cannot fix themselves.
"""
from __future__ import annotations

import asyncio
import dataclasses
import inspect
import logging
import time
from typing import Any, Callable, Mapping

from ..config import Settings
from ..models import DraftState
from .client import EspnAccessDenied, EspnAPIError, EspnClient, EspnNotFound
from .ids import EspnIdMap
from .parsing import (
    PLATFORM,
    draft_detail_of,
    draft_settings_of,
    draft_status,
    parse_espn_picks,
    roster_names,
    rounds_from_counts,
    state_from_espn,
)

log = logging.getLogger(__name__)

UpdateHook = Callable[[DraftState], Any]
ErrorHook = Callable[[Exception], Any]

#: Cap on the backoff exponent (``poll_seconds * 2**n`` must never overflow a float).
_MAX_BACKOFF_EXP = 8


class EspnDraftPoller:
    """Poll one ESPN league's draft and produce a new :class:`DraftState` whenever something changed."""

    #: Poll interval when it is (almost) my turn.
    my_turn_interval: float = 1.0
    #: Poll interval while the draft is still ``pre_draft``.
    pre_draft_interval: float = 10.0
    #: Upper bound of the exponential backoff after errors.
    max_backoff: float = 30.0

    def __init__(self, client: EspnClient, league_id: str, season: int, *, id_map: EspnIdMap | None = None,
                 names: Mapping[str, Mapping[str, Any]] | None = None, swid: str | None = None,
                 team_id: int | str | None = None, slot: int | None = None, username: str | None = None,
                 poll_seconds: float = 3.0, league_json: Mapping[str, Any] | None = None,
                 settings: Settings | None = None):
        self.client = client
        self.league_id = str(league_id)
        self.season = int(season)
        self.draft_id = f"{PLATFORM}-{self.league_id}-{self.season}"
        self.id_map = id_map or EspnIdMap()
        self.names: dict[str, Mapping[str, Any]] = dict(names or {})
        self.swid, self.team_id, self.slot, self.username = swid, team_id, slot, username
        self.settings = settings or Settings(poll_seconds=float(poll_seconds))

        self.state: DraftState | None = None
        self.last_error: str | None = None
        self.poll_count: int = 0
        self.last_latency_ms: float = 0.0
        self.last_poll_at: float | None = None
        self.consecutive_errors: int = 0

        self._league_json: dict | None = dict(league_json) if league_json else None
        self._draft_json: dict = {}
        self._names: dict[str, Mapping[str, Any]] = {}
        self._signature: tuple | None = None

    # -- helpers ----------------------------------------------------------------
    @staticmethod
    def _board_entries(draft_json: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        return [p for p in (draft_detail_of(draft_json).get("picks") or []) if isinstance(p, Mapping)]

    @classmethod
    def _order_signature(cls, draft_json: Mapping[str, Any]) -> tuple:
        """The parts of the per-poll payload that change pick-order math / the clock (a change re-parses):
        the draft settings, the rounds and the owners of picks still to be made (pick trades)."""
        ds = draft_settings_of(draft_json)
        counts = ((draft_json.get("settings") or {}).get("rosterSettings") or {}).get("lineupSlotCounts") or {}
        future_owners = tuple(sorted(
            (str(p.get("overallPickNumber")), str(p.get("owningTeamIds") or p.get("teamId")))
            for p in cls._board_entries(draft_json) if p.get("playerId") in (None, "", 0, "0")))
        return (
            tuple(str(t) for t in (ds.get("pickOrder") or [])),
            str(ds.get("timePerSelection")),
            str(ds.get("type")),
            str(ds.get("date")),
            rounds_from_counts(counts) if counts else None,
            future_owners,
        )

    @classmethod
    def _make_signature(cls, draft_json: Mapping[str, Any]) -> tuple:
        """Every board entry (number, player, team, keeper flag) plus the draft flags and the order signature,
        so a board ESPN fills in place, or a pick corrected mid-list, changes it as surely as an appended pick."""
        detail = draft_detail_of(draft_json)
        entries = tuple(sorted(
            (str(p.get("overallPickNumber")), str(p.get("playerId")), str(p.get("teamId")),
             bool(p.get("keeper") or p.get("reservedForKeeper")))
            for p in cls._board_entries(draft_json)))
        return (entries, bool(detail.get("drafted")), bool(detail.get("inProgress")), cls._order_signature(draft_json))

    def _build_state(self, draft_json: Mapping[str, Any], version: int = 0) -> DraftState:
        st = state_from_espn(self._league_json or {}, draft_json, self.id_map, names=self._names, swid=self.swid,
                             team_id=self.team_id, slot=self.slot, username=self.username)
        st.version = version
        return st

    def _refresh_names(self) -> None:
        merged: dict[str, Mapping[str, Any]] = dict(roster_names(self._league_json))
        merged.update(self.names)
        self._names = merged

    # -- public API -------------------------------------------------------------
    async def bootstrap(self) -> DraftState:
        """Fetch settings / teams (unless given) and the draft; build the initial state."""
        t0 = time.perf_counter()
        if self._league_json is None:
            self._league_json = await self.client.get_settings_and_teams(self.league_id, self.season)
        draft_json = await self.client.get_draft_detail(self.league_id, self.season)
        self._refresh_names()
        prev_version = self.state.version if self.state is not None else -1
        self.state = self._build_state(draft_json, version=prev_version + 1)
        self._draft_json = dict(draft_json)
        self._signature = self._make_signature(draft_json)
        self.poll_count += 1
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        self.last_poll_at = time.time()
        self.last_error = None
        self.consecutive_errors = 0
        log.info("bootstrap: ESPN league %s status=%s picks=%d my_slot=%s (%.0f ms)", self.league_id,
                 self.state.draft.status, len(self.state.picks), self.state.my_slot, self.last_latency_ms)
        return self.state

    async def refresh_league(self) -> DraftState | None:
        """Re-fetch settings / teams / rosters (owners joining, keepers set) and rebuild the state."""
        self._league_json = await self.client.get_settings_and_teams(self.league_id, self.season)
        self._refresh_names()
        if self.state is None:
            return None
        self.state = self._build_state(self._draft_json, version=self.state.version + 1)
        return self.state

    def _check_payload(self, draft_json: Mapping[str, Any]) -> None:
        """Refuse a poll payload without ``draftDetail`` once a previous poll had one (a throttled / degraded
        200 must not wipe the board); a board that emptied with the draft flags unchanged is only logged."""
        new, old = draft_detail_of(draft_json), draft_detail_of(self._draft_json)
        if old and not new:
            raise EspnAPIError("ESPN payload has no draftDetail block (degraded response?)")
        n_old = len([p for p in (old.get("picks") or []) if isinstance(p, Mapping)])
        n_new = len(self._board_entries(draft_json))
        if n_old > 0 and n_new == 0 and (bool(old.get("drafted")), bool(old.get("inProgress"))) == (
                bool(new.get("drafted")), bool(new.get("inProgress"))):
            log.warning("ESPN board went from %d picks to 0 with the draft flags unchanged (draft reset?); "
                        "rendering it as returned", n_old)

    async def poll_once(self) -> DraftState | None:
        """One draft GET; a new state only when the picks, the draft flags or the draft settings changed."""
        if self.state is None:
            return await self.bootstrap()
        t0 = time.perf_counter()
        draft_json = await self.client.get_draft_detail(self.league_id, self.season)
        self.poll_count += 1
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        self.last_poll_at = time.time()
        self._check_payload(draft_json)
        sig = self._make_signature(draft_json)
        if sig == self._signature:
            return None
        old = self.state
        if self._order_signature(draft_json) != self._order_signature(self._draft_json):
            log.info("draft order / settings changed; re-parsing ESPN league %s", self.league_id)
            new_state = self._build_state(draft_json, version=old.version + 1)
        else:
            draft = old.draft
            status = draft_status(draft_detail_of(draft_json))
            if status != draft.status:
                detail = draft_detail_of(draft_json)
                draft = dataclasses.replace(draft, status=status,
                                            raw=dict(draft.raw, draftDetail={k: v for k, v in detail.items() if k != "picks"}))
            new_state = DraftState(
                draft=draft, picks=parse_espn_picks(draft_json, self.id_map, draft, self._names), league=old.league,
                managers=old.managers, my_user_id=old.my_user_id, my_slot=old.my_slot, version=old.version + 1,
                rostered_ids=old.rostered_ids,
            )
        self._draft_json = dict(draft_json)
        self._signature = sig
        self.state = new_state
        log.debug("poll %d: picks=%d status=%s next=#%d (%.0f ms)", self.poll_count, len(new_state.picks),
                  new_state.draft.status, new_state.next_pick_no, self.last_latency_ms)
        return new_state

    def interval(self, state: DraftState | None = None) -> float:
        """Adaptive poll interval for ``state`` (defaults to the current one)."""
        st = state or self.state
        base = float(self.settings.poll_seconds)
        if st is None:
            return base
        if st.draft.status == "pre_draft":
            return max(base, self.pre_draft_interval)
        until = st.picks_until_my_turn
        if until is not None and until <= 1:
            return min(base, self.my_turn_interval)
        return base

    async def _sleep(self, seconds: float, stop: asyncio.Event | None) -> bool:
        """Sleep ``seconds`` or until ``stop`` is set. Returns True if stopped."""
        if seconds <= 0:
            await asyncio.sleep(0)
            return bool(stop and stop.is_set())
        if stop is None:
            await asyncio.sleep(seconds)
            return False
        try:
            await asyncio.wait_for(stop.wait(), timeout=seconds)
            return True
        except asyncio.TimeoutError:
            return False

    async def _emit(self, hook: Callable[..., Any] | None, *args: Any) -> None:
        if hook is None:
            return
        try:
            res = hook(*args)
            if inspect.isawaitable(res):
                await res
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - a UI bug must not kill the poll loop
            log.exception("update hook failed")

    def _record_error(self, e: Exception) -> None:
        self.consecutive_errors += 1
        self.last_error = f"{type(e).__name__}: {e}"
        log.warning("poll error (%d in a row): %s", self.consecutive_errors, self.last_error)

    def _backoff(self) -> float:
        """Exponential backoff after errors: ``max(1, poll_seconds) * 2**n`` capped at :attr:`max_backoff`."""
        base = max(1.0, float(self.settings.poll_seconds or 0.0))
        return min(self.max_backoff, base * (2 ** min(self.consecutive_errors, _MAX_BACKOFF_EXP)))

    async def run(self, on_update: UpdateHook, stop: asyncio.Event | None = None,
                  on_error: ErrorHook | None = None) -> DraftState:
        """Bootstrap, then poll until the draft completes or ``stop`` is set.

        ``on_update(state)`` is called after bootstrap and on every change (awaited if it returns an
        awaitable). Errors are logged, stored in ``last_error``, passed to ``on_error`` and never raised,
        except :class:`EspnNotFound` / :class:`EspnAccessDenied` during bootstrap, which are re-raised.
        """
        while self.state is None:
            if stop is not None and stop.is_set():
                raise RuntimeError("poller stopped before bootstrap completed")
            try:
                await self.bootstrap()
            except asyncio.CancelledError:
                raise
            except (EspnNotFound, EspnAccessDenied) as e:
                self._record_error(e)
                await self._emit(on_error, e)
                raise
            except Exception as e:  # noqa: BLE001
                self._record_error(e)
                await self._emit(on_error, e)
                if await self._sleep(self._backoff(), stop):
                    raise RuntimeError(f"poller stopped before bootstrap completed: {self.last_error}") from e
        assert self.state is not None
        await self._emit(on_update, self.state)
        while True:
            if self.state.is_complete:
                log.info("ESPN draft %s complete after %d polls", self.draft_id, self.poll_count)
                return self.state
            if stop is not None and stop.is_set():
                return self.state
            wait = self._backoff() if self.consecutive_errors else self.interval(self.state)
            if await self._sleep(wait, stop):
                return self.state
            try:
                new_state = await self.poll_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                self._record_error(e)
                await self._emit(on_error, e)
                continue
            if self.consecutive_errors:
                log.info("poll recovered after %d errors", self.consecutive_errors)
            self.consecutive_errors = 0
            self.last_error = None
            if new_state is not None:
                await self._emit(on_update, new_state)
