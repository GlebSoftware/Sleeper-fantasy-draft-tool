"""Draft poller: keeps a :class:`DraftState` in sync with Sleeper (see DESIGN.md §3.1).

Cadence (``interval()``):

* ``settings.poll_seconds`` normally (2 s);
* :attr:`DraftPoller.my_turn_interval` (1 s) when ``picks_until_my_turn <= 1``;
* :attr:`DraftPoller.pre_draft_interval` (10 s) while the draft has not started.

Exceptions never escape :meth:`DraftPoller.run`: they are logged, recorded in ``last_error``, reported
to the optional ``on_error`` hook and the poller backs off exponentially (up to 30 s) before trying again.
The one exception is a draft that does not exist (:class:`SleeperNotFound` while bootstrapping): that
cannot fix itself, so ``run`` raises it instead of retrying forever.

Besides picks and status, :meth:`DraftPoller.poll_once` notices commissioner changes to the draft
settings (teams / rounds / pick timer / reversal round / player type) and re-parses the draft, and it
re-reads ``/traded_picks`` whenever the order changes and every :attr:`DraftPoller.traded_refresh_seconds`
while the draft is running (picks can be traded mid-draft).
"""
from __future__ import annotations

import asyncio
import dataclasses
import inspect
import logging
import time
from typing import Any, Awaitable, Callable

from ..config import Settings
from ..models import DraftState, Pick
from .client import SleeperClient, SleeperNotFound
from .parsing import parse_picks, state_from_sleeper

log = logging.getLogger(__name__)

UpdateHook = Callable[[DraftState], Any]
ErrorHook = Callable[[Exception], Any]

_ORDER_KEYS = ("draft_order", "slot_to_roster_id", "status", "type")
#: ``draft.settings`` keys that change pick-order math / the clock; a change re-parses the draft.
_SETTINGS_KEYS = ("teams", "rounds", "pick_timer", "reversal_round", "player_type")
#: Cap on the backoff exponent (``poll_seconds * 2**n`` must never overflow a float).
_MAX_BACKOFF_EXP = 8


class DraftPoller:
    """Poll one Sleeper draft and produce a new :class:`DraftState` whenever something changed."""

    #: Poll interval when it is (almost) my turn.
    my_turn_interval: float = 1.0
    #: Poll interval while the draft is still ``pre_draft``.
    pre_draft_interval: float = 10.0
    #: Upper bound of the exponential backoff after errors.
    max_backoff: float = 30.0
    #: How often ``/traded_picks`` is re-read while the draft is drafting / paused.
    traded_refresh_seconds: float = 30.0

    def __init__(self, client: SleeperClient, draft_id: str, *, league_id: str | None = None,
                 settings: Settings | None = None, username: str | None = None, user_id: str | None = None,
                 slot: int | None = None):
        self.client = client
        self.draft_id = str(draft_id)
        self.league_id = str(league_id) if league_id else None
        self.settings = settings or Settings()
        self.username = username if username is not None else self.settings.username
        self.user_id = user_id if user_id is not None else self.settings.user_id
        self.slot = slot if slot is not None else self.settings.slot

        self.state: DraftState | None = None
        self.last_error: str | None = None
        self.poll_count: int = 0
        self.last_latency_ms: float = 0.0
        self.last_poll_at: float | None = None
        self.consecutive_errors: int = 0

        # raw payloads kept so a draft re-parse (e.g. draft order appearing) needs no extra requests
        self._draft_raw: dict = {}
        self._league_raw: dict | None = None
        self._users_raw: list[dict] | None = None
        self._rosters_raw: list[dict] | None = None
        self._traded_raw: list[dict] = []
        self._traded_at: float | None = None
        self._signature: tuple | None = None

    # -- helpers ----------------------------------------------------------------
    @staticmethod
    def _order_signature(draft_raw: dict) -> tuple:
        """Everything in the draft payload (other than status / last_picked) that changes the parsed draft."""
        order = draft_raw.get("draft_order") or {}
        s2r = draft_raw.get("slot_to_roster_id") or {}
        settings = draft_raw.get("settings") or {}
        return (
            tuple(sorted((str(k), str(v)) for k, v in order.items())),
            tuple(sorted((str(k), str(v)) for k, v in s2r.items())),
            draft_raw.get("type"),
            tuple(str(settings.get(k)) for k in _SETTINGS_KEYS),
        )

    @staticmethod
    def _traded_signature(traded: list[dict] | None) -> tuple:
        return tuple(sorted(
            (str(tp.get("season")), str(tp.get("round")), str(tp.get("roster_id")), str(tp.get("owner_id")))
            for tp in (traded or []) if isinstance(tp, dict)
        ))

    def _traded_due(self, now: float) -> bool:
        st = self.state
        if st is None or st.draft.status not in ("drafting", "paused"):
            return False
        return self._traded_at is None or (now - self._traded_at) >= self.traded_refresh_seconds

    def _make_signature(self, picks: list[Pick], draft_raw: dict) -> tuple:
        return (len(picks), picks[-1].pick_no if picks else 0, draft_raw.get("status"),
                self._order_signature(draft_raw))

    async def _fetch_optional(self, name: str, coro: Awaitable[Any], default: Any) -> Any:
        try:
            return await coro
        except SleeperNotFound:
            log.info("%s: not found", name)
        except Exception as e:  # noqa: BLE001 - degrade gracefully
            log.warning("%s: %s", name, e)
        return default

    def _build_state(self, picks_raw: list[dict], draft_raw: dict, version: int = 0) -> DraftState:
        st = state_from_sleeper(draft_raw, picks_raw, self._league_raw, self._users_raw, self._rosters_raw,
                                self._traded_raw, username=self.username, user_id=self.user_id, slot=self.slot)
        st.version = version
        return st

    # -- public API -------------------------------------------------------------
    async def bootstrap(self) -> DraftState:
        """Fetch draft, league (users, rosters), traded picks and picks; build the initial state."""
        t0 = time.perf_counter()
        draft_raw = await self.client.get_draft(self.draft_id)
        league_id = self.league_id or (str(draft_raw.get("league_id")) if draft_raw.get("league_id") else None)
        if league_id:
            self.league_id = league_id
            league, users, rosters = await asyncio.gather(
                self._fetch_optional("league", self.client.get_league(league_id), None),
                self._fetch_optional("league users", self.client.get_league_users(league_id), None),
                self._fetch_optional("league rosters", self.client.get_league_rosters(league_id), None),
            )
            self._league_raw, self._users_raw, self._rosters_raw = league, users, rosters
        else:
            log.info("draft %s has no league (mock draft?)", self.draft_id)
        traded, picks_raw = await asyncio.gather(
            self._fetch_optional("traded picks", self.client.get_traded_picks(self.draft_id), []),
            self.client.get_draft_picks(self.draft_id),
        )
        self._traded_raw = list(traded or [])
        self._traded_at = time.time()
        self._draft_raw = draft_raw
        prev_version = self.state.version if self.state is not None else -1
        self.state = self._build_state(picks_raw, draft_raw, version=prev_version + 1)
        self._signature = self._make_signature(self.state.picks, draft_raw)
        self.poll_count += 1
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        self.last_poll_at = time.time()
        self.last_error = None
        self.consecutive_errors = 0
        log.info("bootstrap: draft %s status=%s picks=%d my_slot=%s (%.0f ms)", self.draft_id,
                 self.state.draft.status, len(self.state.picks), self.state.my_slot, self.last_latency_ms)
        return self.state

    async def poll_once(self) -> DraftState | None:
        """Fetch picks + draft (+ traded picks when due); return a new state only when something changed.

        "Something" = picks, status, last_picked, the draft order / slot mapping, the settings that
        drive pick-order math, or the traded-pick list.
        """
        if self.state is None:
            return await self.bootstrap()
        t0 = time.perf_counter()
        want_traded = self._traded_due(time.time())
        coros: list[Awaitable[Any]] = [self.client.get_draft_picks(self.draft_id), self.client.get_draft(self.draft_id)]
        if want_traded:
            coros.append(self._fetch_optional("traded picks", self.client.get_traded_picks(self.draft_id), None))
        results = await asyncio.gather(*coros)
        picks_raw, draft_raw = results[0], results[1]
        traded_new = results[2] if want_traded else None
        self.poll_count += 1
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        self.last_poll_at = time.time()
        picks = parse_picks(picks_raw)
        sig = self._make_signature(picks, draft_raw)
        order_changed = self._order_signature(draft_raw) != self._order_signature(self._draft_raw)
        if order_changed and not want_traded:
            # a new order / slot mapping can come with a new set of traded picks (e.g. draft (re)started)
            traded_new = await self._fetch_optional("traded picks", self.client.get_traded_picks(self.draft_id), None)
            want_traded = True
        traded_changed = False
        if want_traded:
            self._traded_at = time.time()
            if traded_new is not None:
                traded_new = list(traded_new or [])
                if self._traded_signature(traded_new) != self._traded_signature(self._traded_raw):
                    log.info("traded picks changed (%d entries); re-parsing draft %s", len(traded_new), self.draft_id)
                    self._traded_raw = traded_new
                    traded_changed = True
        if sig == self._signature and not traded_changed:
            return None
        old = self.state
        if order_changed or traded_changed:
            # draft order / slot mapping / settings / traded picks appeared or changed: re-parse
            if order_changed:
                log.info("draft order or settings changed; re-parsing draft %s", self.draft_id)
            new_state = self._build_state(picks_raw, draft_raw, version=old.version + 1)
        else:
            draft = old.draft
            if draft_raw.get("status") != draft.status or draft_raw.get("last_picked") != draft.last_picked:
                draft = dataclasses.replace(
                    draft,
                    status=str(draft_raw.get("status") or draft.status),
                    last_picked=_int_or(draft_raw.get("last_picked"), draft.last_picked),
                    raw=dict(draft_raw),
                )
            new_state = DraftState(
                draft=draft, picks=picks, league=old.league, managers=old.managers,
                my_user_id=old.my_user_id, my_slot=old.my_slot, version=old.version + 1,
                rostered_ids=old.rostered_ids,
            )
        self._draft_raw = draft_raw
        self._signature = sig
        self.state = new_state
        log.debug("poll %d: picks=%d status=%s next=#%d (%.0f ms)", self.poll_count, len(picks),
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
        """Exponential backoff after errors: ``max(1, poll_seconds) * 2**n`` capped at :attr:`max_backoff`.

        The exponent is capped so that a long outage (or a draft that keeps failing) can never
        overflow the float and crash the loop.
        """
        base = max(1.0, float(self.settings.poll_seconds or 0.0))
        return min(self.max_backoff, base * (2 ** min(self.consecutive_errors, _MAX_BACKOFF_EXP)))

    async def run(self, on_update: UpdateHook, stop: asyncio.Event | None = None,
                  on_error: ErrorHook | None = None) -> DraftState:
        """Bootstrap, then poll until the draft completes or ``stop`` is set.

        ``on_update(state)`` is called after bootstrap and on every change (awaited if it returns an
        awaitable). Errors are logged, stored in ``last_error``, passed to ``on_error`` and never raised,
        except a :class:`SleeperNotFound` during bootstrap (unknown draft id), which is re-raised.
        """
        while self.state is None:
            if stop is not None and stop.is_set():
                raise RuntimeError("poller stopped before bootstrap completed")
            try:
                await self.bootstrap()
            except asyncio.CancelledError:
                raise
            except SleeperNotFound as e:
                # the draft id does not exist: retrying cannot help (and would loop forever)
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
                log.info("draft %s complete after %d polls", self.draft_id, self.poll_count)
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


def _int_or(v: Any, default: int | None) -> int | None:
    try:
        return int(v) if v is not None else default
    except (TypeError, ValueError):
        return default
