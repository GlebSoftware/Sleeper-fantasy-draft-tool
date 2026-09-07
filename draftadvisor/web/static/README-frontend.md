# Frontend notes — `index.html`

One file, no build step, no external dependencies: HTML + `<style>` + `<script>` (vanilla JS, ES2018).
Served by `draftadvisor/web/server.py` at `http://127.0.0.1:8787/`; talks to the JSON API described in
`DESIGN.md §3.8` on the same origin.

## Layout of the file

| part | what to edit there |
|---|---|
| `<style>` `:root` / `html[data-theme="light"]` | colours, fonts, radii. Everything uses these variables, so retheming is a matter of changing the tokens. The ◐ button in the top bar toggles light/dark (saved in `localStorage`). |
| `<style>` rest | grid layout (`.cols`, `.pos-grid`, `.setup-grid`), tables, badges, popover, toasts. Breakpoints: 1300 px (position grid 3→2 columns), 1000 px (three columns stack). |
| `<section id="tab-…">` | static markup for the five tabs. Inputs/selects/textarea live here and are **never re-rendered**, so they keep focus and values across polls. |
| `<script>` §1 helpers | `esc` (HTML escaping), `get(obj, 'a.b.c', default)` (safe accessor — use it for anything that may be missing), number formatters, `setHTML`/`setText` (write only when changed). |
| §2 api | `api(path, {method, body, timeout})` — fetch with an `AbortController` timeout; non-2xx → `ApiError(detail)`. `post()` wraps it and shows a toast on failure. `toast(msg, kind)` de-duplicates identical messages. |
| §3 state store | `S` holds the last `/api/status`, the last `/api/state`, board rows and UI state (tab, hover/pinned card, available-list filter, autopilot). `findCard(id)` looks a player up in every list of the state. |
| §4 renderers | one function per region: `renderTop`, `renderSetup`, `renderDraft` (→ `renderRoster`, `renderBest`, `renderPositions`, `renderRecent`, `renderOpponents`, `renderFooter`, `renderAvailable`), `renderBoard`, `renderLeague`, `renderAsk`, `renderPop` (the hover card). Each builds an HTML string and hands it to `setHTML`, which skips the DOM write when nothing changed. |
| §5 polling | `tick()` → `/api/state` every 2 s in live/mock (1 s when `draft.is_my_turn`), `/api/status` every 2 s otherwise (and every ~5 s during a draft to keep the log fresh). `pokeNow()` re-polls right after a POST. The countdown is recomputed client-side every second from `turn_started_at + pick_timer` (server `ts` is used to correct clock skew; `seconds_left` is the fallback). |
| §6 events | tabs, forms, keyboard (`Enter` = take recommendation in a mock when it is your turn, `Esc` = close card, `1`–`5` = tabs), row hover/click (delegated, so it survives re-rendering). |

## Common changes

* **Add a column to "Best picks"**: extend the `<thead>` of `#best-tbl` and `cardRow()` (used by both Best picks and Available; `opts.why` tells them apart). Exactly one `td` per table carries class `fill` — it absorbs the remaining width and truncates with an ellipsis.
* **Show another field in the hover card**: edit `cardHTML()`; every card field from the API (`points`, `floor`, `ceiling`, `vorp`, `vona`, `marginal`, `score`, `tier`, `adp`, `ecr`, `availability_next/after_next`, `reasons`, `warnings`, `note`) is already available on `c`.
* **Change the action badge colours**: `.badge.take/.soon/.wait/.skip` in the CSS; the mapping from the API's `"TAKE NOW" | "SOON" | "WAIT" | "SKIP"` is `actionClass()`.
* **Change poll rates**: `schedule()`.
* **New API endpoint**: call `api('/api/…')` from an event handler or `tick()`, store the result on `S`, and render it in the matching `render…` function. Never throw on a missing field — use `get()` / `Array.isArray` guards as the existing code does.

## Testing without the Python backend

Any small server that serves this file at `/` and answers the `/api/...` routes with the JSON shapes from
`DESIGN.md §3.8` works (a `python -m http.server`-style stub is enough). The page tolerates `null` for
`snapshot`, `me`, `best`, `by_position`, `available`, `recent`, `opponents`, `claude`, `status`, `log` and
shows "computing…" placeholders; API errors surface as toasts and the top-bar "offline" pill appears after
two consecutive failed polls.
