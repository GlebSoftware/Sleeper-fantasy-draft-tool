# Frontend notes — `index.html`

One file, no build step, no external dependencies: HTML + `<style>` + `<script>` (vanilla JS, ES2018).
Served by `draftadvisor/web/server.py` at `/` (locally `http://127.0.0.1:8787/`, on Vercel the project URL) and
talks to the **stateless** JSON API in `DESIGN.md §3.9` on the same origin (ESPN specifics in §3.10). The browser
owns the session: `localStorage["da.session"]` holds mode, platform, ids and (for mocks) every pick, so a reload
resumes where you were and the server keeps nothing but caches. Two platforms share one page: Sleeper and ESPN
differ only in the Setup form, the Settings cookies, the platform pill / League-tab extras and the clock rule.

## Layout of the file

| part | what to edit there |
|---|---|
| `<style>` `:root` / `html[data-theme="light"]` | colours, fonts, radii. Everything uses these tokens; the ◐ button (or Settings → theme) toggles light/dark. |
| `<style>` layout | `.cols` (draft: roster / best picks / recent), `.pos-grid`, `.setup-grid`, `.dhead` (sticky draft header, grid areas `l c r`), `.drawer` (chat), `.pop` (player card), `.modal` (Settings: `max-height` + `overflow-y: auto`, so Save stays reachable on short / landscape phones). |
| `<style>` phone block (bottom) | `@media (max-width: 700px / 640px / 480px)`: compact top bar, one-column grids, bottom-sheet player card, full-screen chat, and **column hiding by `nth-child`** per table (bye / VORP / ADP go first, then tier / why). Adding a column to a table shifts these indexes — update them. |
| `.dhead` right side / `#picks-notice` / `#diag-panel` | the freshness + Refresh controls, the "no picks" notice (top of the middle column, so it is the first thing on a phone) and the collapsed Diagnostics panel (in `#draft-foot`). See **Refresh, freshness and diagnostics** below. |
| `<section id="tab-…">` | static markup for the four tabs (Setup, Draft, Board, League). Inputs live here and are never re-rendered, so they keep focus and values across polls. The Live draft panel holds a Sleeper \| ESPN segmented switch (`#platform-seg`) and one form per platform (`#live-form`, `#espn-form`). |
| `#chat-drawer` / `#chat-fab` | the chat: a sidebar next to the board at ≥ 1200 px (`body.chat-open` shifts `main`), an overlay below that, full screen on phones. Opened from the 💬 buttons, the floating button, or the `c` key; Esc closes. |
| `<script>` §1 helpers | `esc`, `get(obj, 'a.b', def)`, formatters, `setHTML`/`setText` (write only when changed), `lsGet`/`lsSet`, `mdHTML` (markdown-lite for Claude's answers). |
| §2 api | `api(path, {method, body, timeout})` with `X-Anthropic-Key` / `X-Access-Code` / `X-ESPN-S2` / `X-ESPN-SWID` headers from Settings (`authHeaders()`), `sse()` for `/api/chat`, `toast()`. A 401 opens Settings: at the access code by default, at the ESPN cookie fields when the detail names `espn_s2` / `SWID` (`isEspnAuthError`; opened once per attempt, then toasts only; the thrown `ApiError` carries `espnAuth = true`). The access-code marker (`ui.needAccess`, the `· 401` pill and the Setup chip) is set by any 401 and cleared only by a 2xx from a *guarded* endpoint (`accessAccepted`; `/api/status` answers without the code, so it never counts); `verifyAccess()` fires the cheapest guarded call (`GET /api/notes`) after boot and after Settings → Save to settle the chip. |
| §3 store | `S` — status (incl. the Claude price table, `season`, `platforms`, `espn.server_cookies`), session, last render payload, chat transcript (per draft; assistant messages carry `usage` / `cost_usd` / `model` from the done event), legacy notes, board rows, UI state (`ui.platform`, `ui.espn` = the last ESPN look-up). `sessionPayload()` / `sessionQuery()` serialise the session for POST / GET: `mode` plus `SESSION_KEYS` (`platform`, `draft_id`, `league_id`, `season`, `username`, `user_id`, `slot`, `team_id`; `use_claude` is always `true`, a compatibility field the server ignores). `platform()` / `isEspn()` read the running session's platform. |
| §4 renderers | `renderTop`, `renderSetup` (+ `renderPlatform`, `renderEspnLookup`), `renderDraft` (→ `renderPlatformPill`, roster, best, positions, recent, opponents, footer, available, `renderChatPill`, `renderUpdated`, `renderEspnStrip`, `renderPicksNotice`, `renderDiag`), `renderBoard`, `renderLeague` (facts, flags, scoring diff, "ESPN rules not modelled", draft order), `renderChat` (per-answer tokens + cost, running total in the drawer head, cost hint), `renderPop`. Each builds an HTML string and hands it to `setHTML`. |
| §5 session | `forceRefresh()` (the ⟳ button / `r`), `loadDiagnostics()` (`GET /api/espn/diagnose`, only on demand), `startSession` (`POST /api/session/start`; the server's `message` — spectating, order not set — is toasted), `resumeSession`, `tick()` polling (`GET /api/state` every 2 s in live mode; `POST /api/mock/state` drives mocks), `mockAction`, autopilot. `applyState` → `observePick` remembers when `draft.next_pick_no` moved (the ESPN clock anchor). `onFail` counts consecutive failures (two → the red "offline" pill) — except an ESPN 401 (`err.espnAuth`), which the server *did* answer: that sets `ui.espnDenied`, turns the same pill into "ESPN 401" (title: paste fresh cookies), writes the detail into `#last-error` and slows the poll to 5 s until a poll succeeds or the cookies are saved again. Polling never touches Claude. |
| §6 chat | `sendChat` streams `/api/chat` — the only call that costs money, made only when you press Send (or a quick prompt). The body carries `model` only when the user picked one in Settings (`settings.model`; `''` = "server default", the first option of the select), so the server's `DRAFTADVISOR_CHAT_MODEL` applies to every browser that never chose; `chatModel()` (the pick, else `/api/status` → `claude.chat_model`) is what every label and cost estimate shows. The final `done` event's `usage` / `cost_usd` / `model` are stored on the assistant message and shown under the bubble. Nothing calls Claude automatically. |
| §7 events | tabs, forms (Sleeper look-up / start, ESPN `espnLoad` / start, mock), keyboard (`Enter` = take the recommendation in a mock, `c` = chat, `Esc` = close settings / card / chat, `1`–`4` = tabs), delegated row hover/click, settings modal. |

## Refresh, freshness and diagnostics

ESPN's REST API does not publish picks while a draft is running (it serves a full board of `playerId: -1`
placeholders and flushes every pick when the draft completes), so a live ESPN board can legitimately stay
empty for an hour. The page must therefore never *look* idle, and must be able to say what ESPN actually
answered instead of leaving it to be guessed.

* **Freshness** — `#dh-updated` ("updated 3 s ago", `0 s old` on a phone) is re-rendered by the 1 s
  interval next to `renderCountdown`, from `S.stateAt` (when the last poll was *applied*). It turns amber
  after 15 s. `renderUpdated()` also drives `renderRefresh()` (the busy spinner / disabled state of the
  three Refresh buttons).
* **Force refresh** — `forceRefresh()` is wired to `#refresh-btn` (draft header), `#refresh-btn2` (Setup,
  enabled while a session runs), `#notice-refresh` and the `r` key (same guard as `c`: not while typing,
  not while Settings is open). It calls `GET /api/state` with `force=1` **and** a `_=<nonce>` parameter,
  on top of the `cache: "no-store"` every `api()` call already sends, so no cache anywhere can answer it.
  `force=1` makes the server re-capture the league (settings, teams, rosters, pick order, clock) instead
  of using its 10-minute capture; the server rate-limits it to one re-capture per league per 5 s. The
  answer is toasted with what came back, `S.ui.forcedAt` is stamped, `S.board.loaded` is cleared (the
  Board tab re-reads under the fresh capture) and the normal poll is re-scheduled. Mocks sync instead;
  an idle page just re-reads `/api/status`. Nothing here costs money — Claude is only ever the chat.
* **The "no picks" notice** (`#picks-notice`, `renderPicksNotice`) — shown for a live session in two
  states: **nothing known** (board and rosters both empty) and **board empty, picks from the rosters**
  (`draft.rostered_only > 0` with `board_picks === 0`). It always says when the page last checked, what
  ESPN answered (`status.espn`: HTTP status, filled board slots of total, drafted players on rosters,
  `drafted` / `inProgress`), and — for ESPN — that its API does not publish picks mid-draft. After 90 s
  of an unchanged answer (`S.espnSig`, tracked here because the server is stateless) it adds what a
  practice / mock room means. The Refresh and Diagnostics buttons sit inside it.
* **Roster-derived players never get an invented pick number.** `st.roster_only` (players ESPN shows on a
  team roster but not on its board) is rendered as its own block under the recent picks, with the pick
  cell `—` when `confidence === 'none'`, `~N` for `'team'` and the plain number for `'exact'`, always with
  a `roster` badge and a title saying where it came from. While the confidence is `'none'` the header's
  "You pick in N (#M)" line is replaced by "your next pick cannot be counted while ESPN's board is
  behind" — the board-derived counter is knowably wrong then, and a wrong countdown is worse than none.
* **Diagnostics** (`#diag-panel`, collapsed, in the draft footer) — `renderDiag()` shows what the last
  poll reported (`status.espn`) without asking for anything; **Run diagnostics** (or opening the panel,
  or the notice's button) calls `GET /api/espn/diagnose` once. `diagReportHTML()` renders *whatever*
  shape the server sends: `explanation` first in a highlighted box, then scalars as a `.kv` list and every
  nested object / array-of-objects as a labelled sub-list or table (`DIAG_LABELS` supplies the English
  names, `diagValue()` the formatting — times, ages, booleans, the redacted URL). Copy puts the JSON on
  the clipboard (with an `execCommand` fallback). A Sleeper session gets a "does not apply" line plus the
  last-poll block; a mock gets one sentence; a 404 from the endpoint is reported and the last-poll block
  is shown anyway. The header strip `#dh-espn` carries the same numbers in one line under the round.

Nothing in this section polls ESPN by itself: the diagnostics call happens only when you press the button,
and the refresh only when you press the button (or `r`).

## The clock

The rule differs by platform and nothing in the advice depends on either.

**Sleeper**: `renderCountdown` shows a countdown **only** from what the last poll reported: `draft.pick_timer` (0 or null →
nothing is shown) and `draft.seconds_left` (null → nothing is shown; the page derives nothing from `turn_started_at`),
ticking down locally until the next poll re-bases it. While `draft.status` is `paused` the clock is hidden and the
PAUSED banner wins over YOUR PICK — the server sends no `seconds_left` then, and nothing must tick. Nothing about
the clock length is assumed by the page; if the commissioner changes it mid-draft the next poll picks it up.

**ESPN** publishes no pick timestamps, so `seconds_left` is always null there. `renderEspnClock` shows
"ESPN clock: *N* s per pick" (re-read every poll) and, only after this page has *seen* `draft.next_pick_no`
change (`S.pickSeen`, set by `observePick`), an approximate countdown anchored to that moment, labelled
"≈ since last pick seen". It is an estimate and is presented as one; before any pick has been observed nothing
is counted down. Whether ESPN updates its pick list while a draft is in progress could not be verified — the page
renders whatever each poll returns.

## ESPN sessions

* Setup → Live draft → **ESPN**: League ID (the `leagueId=` number in the league URL; a pasted URL is reduced to
  the number), Season (defaults to `/api/status` → `season`), **Load league** (also `Enter` in either field — never
  the form's Start) → `POST /api/lookup`
  `{platform: "espn", league_id, season}` → league facts, the not-modelled rules and a team `<select>` that
  preselects `is_me` ("(you)", found through the SWID). **Start live draft** posts
  `{mode: "live", platform: "espn", league_id, season, team_id}` (a slot override is sent only when no team is
  chosen; no team = spectating).
* Settings → **ESPN (private leagues)**: `espn_s2` and `SWID` (braces added when missing), kept in
  `localStorage["da.settings"]` and sent as `X-ESPN-S2` / `X-ESPN-SWID` on every request. The how-to under the
  fields says where to copy them from (espn.com → dev tools → Application / Storage → Cookies).
* The draft header shows a platform pill ("Sleeper" / "ESPN", hidden for mocks); the League tab adds a
  "Platform" fact, an "ESPN rules not modelled" panel from `snapshot.unmapped_scoring`, and a platform-aware clock
  fact. `localStorage["da.session"]` carries `platform` / `season` / `team_id`, so a reload resumes; the Board
  tab sends the whole live session (`sessionQuery()`) so ESPN leagues get their own scoring.

## Common changes

* **Add a column to "Best picks"**: extend `#best-tbl`'s `<thead>` and `cardRow()` (shared with Available; `opts.why` tells them apart), then fix the `nth-child` indexes in the phone CSS block.
* **Show another field in the player card**: `cardHTML()` — every card field from the API is on `c`, the legacy research note (if any) on `noteFor(c)`.
* **Cost display**: `DEFAULT_PRICES` (overridden by `/api/status` → `claude.prices`), `EST_TOKENS_PER_MESSAGE` (the assumption behind the "about $x per message" hint), `usageText()` (under each answer), `chatTotals()` + `totalCostLabel()` (drawer head: answers without a price are counted as "+ N answers of unknown cost", never folded into the sum). `costFromUsage()` is the fallback when an old message has usage but no `cost_usd`.
* **Change the action badge colours**: `.badge.take/.soon/.wait/.skip`; the mapping from `TAKE NOW | SOON | WAIT | SKIP` is `actionClass()`.
* **Quick chat prompts**: `QUICK_PROMPTS`.
* **Poll rates**: `schedule()`.
* **Change what the diagnostics show**: `DIAG_LABELS` (English names), `diagValue()` (formatting per key),
  `diagReportHTML()` (section order); the block itself comes from the server, so a new field appears
  automatically with a humanised label.
* **Change the honest wording for an empty board**: `renderPicksNotice()` (both states) and the empty-row
  text in `renderRecent()`.
* **Add a session field**: extend `SESSION_KEYS` (both `sessionPayload()` and `sessionQuery()` read it) and the
  form handler that sets it; the server echoes the resolved session from `/api/session/start` and it is stored whole.
* **Another platform**: a button in `#platform-seg`, a form, a `renderPlatform` branch, `platformLabel()`, and a
  decision on the clock: use `renderCountdown` when the server can give `seconds_left`, the observed-pick
  estimate (`observePick` / `renderEspnClock`) when it cannot.
