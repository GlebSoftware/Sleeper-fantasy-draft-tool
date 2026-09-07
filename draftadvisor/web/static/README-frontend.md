# Frontend notes — `index.html`

One file, no build step, no external dependencies: HTML + `<style>` + `<script>` (vanilla JS, ES2018).
Served by `draftadvisor/web/server.py` at `/` (locally `http://127.0.0.1:8787/`, on Vercel the project URL) and
talks to the **stateless** JSON API in `DESIGN.md §3.9` on the same origin. The browser owns the session:
`localStorage["da.session"]` holds mode, ids and (for mocks) every pick, so a reload resumes where you were and
the server keeps nothing but caches.

## Layout of the file

| part | what to edit there |
|---|---|
| `<style>` `:root` / `html[data-theme="light"]` | colours, fonts, radii. Everything uses these tokens; the ◐ button (or Settings → theme) toggles light/dark. |
| `<style>` layout | `.cols` (draft: roster / best picks / recent), `.pos-grid`, `.setup-grid`, `.dhead` (sticky draft header, grid areas `l c r`), `.drawer` (chat), `.pop` (player card). |
| `<style>` phone block (bottom) | `@media (max-width: 700px / 640px / 480px)`: compact top bar, one-column grids, bottom-sheet player card, full-screen chat, and **column hiding by `nth-child`** per table (bye / VORP / ADP go first, then tier / why). Adding a column to a table shifts these indexes — update them. |
| `<section id="tab-…">` | static markup for the four tabs (Setup, Draft, Board, League). Inputs live here and are never re-rendered, so they keep focus and values across polls. |
| `#chat-drawer` / `#chat-fab` | the chat: a sidebar next to the board at ≥ 1200 px (`body.chat-open` shifts `main`), an overlay below that, full screen on phones. Opened from the 💬 buttons, the floating button, or the `c` key; Esc closes. |
| `<script>` §1 helpers | `esc`, `get(obj, 'a.b', def)`, formatters, `setHTML`/`setText` (write only when changed), `lsGet`/`lsSet`, `mdHTML` (markdown-lite for Claude's answers). |
| §2 api | `api(path, {method, body, timeout})` with `X-Anthropic-Key` / `X-Access-Code` headers from Settings, `sse()` for `/api/chat`, `toast()`. A 401 opens Settings. |
| §3 store | `S` — status, session, last render payload, advice, chat transcript (per draft), research progress, notes, board rows, UI state. `sessionPayload()` / `sessionQuery()` serialise the session for POST / GET. |
| §4 renderers | `renderTop`, `renderSetup`, `renderDraft` (→ roster, best, positions, recent, opponents, footer, available), `renderBoard`, `renderLeague`, `renderChat`, `renderPop`, `renderResearch`. Each builds an HTML string and hands it to `setHTML`. |
| §5 session | `startSession` (`POST /api/session/start`), `resumeSession`, `tick()` polling (`GET /api/state` every 2 s in live mode; `POST /api/mock/state` drives mocks), `mockAction`, autopilot, `maybeFetchAdvice` (`GET /api/advice` on your turn, never blocking the poll). |
| §6 chat / research | `sendChat` streams `/api/chat`; `researchLoop` calls `/api/research/next` one player at a time (resumable), `researchPlayer` forces one. |
| §7 events | tabs, forms, keyboard (`Enter` = take the recommendation in a mock, `c` = chat, `Esc` = close settings / card / chat, `1`–`4` = tabs), delegated row hover/click, settings modal. |

## The clock

`renderCountdown` shows a countdown **only** from what the last poll reported: `draft.pick_timer` (0 or null →
nothing is shown) and `draft.seconds_left`, ticking down locally until the next poll re-bases it. Nothing about
the clock length is assumed by the page; if the commissioner changes it mid-draft the next poll picks it up.

## Common changes

* **Add a column to "Best picks"**: extend `#best-tbl`'s `<thead>` and `cardRow()` (shared with Available; `opts.why` tells them apart), then fix the `nth-child` indexes in the phone CSS block.
* **Show another field in the player card**: `cardHTML()` — every card field from the API is on `c`, the research note on `noteFor(c)`.
* **Change the action badge colours**: `.badge.take/.soon/.wait/.skip`; the mapping from `TAKE NOW | SOON | WAIT | SKIP` is `actionClass()`.
* **Quick chat prompts**: `QUICK_PROMPTS`.
* **Poll rates**: `schedule()`.
