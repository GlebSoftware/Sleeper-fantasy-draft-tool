# draftadvisor - ESPN draft-room overlay (Edge / Chrome, MV3)

A panel that sits inside your own ESPN draft-room tab and shows, live:

* **the next suggested pick** (big, with a one-line reason),
* **top 5 available overall**,
* **top 5 per position** (QB / RB / WR / TE / K / DEF).

No AI, no chat. It watches the picks in *your* tab and asks the already-deployed advice API
(`https://sleeper-draft-advisor.vercel.app`) for the board. Nothing here costs money to run.

## Load it in Edge (under a minute)

1. Open `edge://extensions` (Chrome: `chrome://extensions`).
2. Turn on **Developer mode** (toggle, bottom-left in Edge / top-right in Chrome).
3. Click **Load unpacked** and pick this `extension` folder (the one containing `manifest.json`).
4. Open your ESPN draft room (`https://fantasy.espn.com/...`). **Reload the tab if it was already
   open** - the websocket hook must be installed before ESPN opens its socket.
5. The panel appears top-right. Drag it by its header, resize from the bottom-right corner,
   collapse it with the `-` button. Position, size and the whole draft are remembered across reloads.

## Reading the header

The coloured dot and the word next to "draftadvisor" say **which layer last produced a pick**:

| layer | meaning |
| --- | --- |
| `websocket` (green) | frames from ESPN's draft socket are being parsed - the good case |
| `dom` (amber) | picks are being read off the page text |
| `manual` (blue) | nothing automatic has fired yet; you are typing picks in |

Next to it: `N taken / M mine / pick #P (rd R)`, a **Refresh** button (re-asks the API), and the
collapse toggle. `pick #P` is the pick the *server* thinks is on the clock, so a wrong round is
visible immediately instead of showing up as strange advice.

Under the suggestion, the panel prints **the roster the advice was computed against**. Check it. An
advisor that cannot see your four running backs believes every starting slot is open and recommends
the best player alive - which is another running back. When rows read `UNK` the extension knows a
pick of yours was made but could not name the player, and the panel says so in a warning; set
**my slot** in Settings, or open your roster panel on the ESPN page, and it will pick it up.

Board depth is tracked separately from what the extension could name: a pick whose player is
unrecognised still advances the clock, and the round it produces is what drives replacement levels
and positional scarcity. Use **Clear this draft** in Settings when you move from a mock to the real
one, or the old depth carries over.

## The three detection layers

1. **WebSocket hook** (`hook.js`, runs in the page's own JS world). Wraps `window.WebSocket` and
   reads every frame. A pick frame is expected to be plain text: `SELECTED <teamId> <playerId>
   <overallPick> <memberId>`; `PICK`, `SELECT`, `DRAFTED`, `PICKED`, `AUTOPICK` are accepted as
   aliases. JSON frames are also scanned for a `playerId`. Any frame with three or more integers
   where one is a known ESPN player id becomes a **candidate** in the Debug section with a click
   to accept, so an unexpected verb is never silently dropped.
2. **DOM scan** (fallback). Once the player index is loaded, a MutationObserver plus a 1 s sweep
   look for known player names on the page, and only count one as taken when it (or an ancestor
   within 3 levels) also looks like a pick: `R1.03`, "Round", "Pick", "selected by". Toggle it off
   in Settings if it misfires. There is a second, **off by default** mode ("treat disappearing
   names as picks") - leave it off unless the pick-context mode is finding nothing, because ESPN's
   available-players list is virtualised and scrolling alone removes rows.
3. **Manual entry** (always works). Type a name in the search box and click the result to mark them
   taken; click a taken player again (or the `x` in the Taken list) to undo. Tick **my pick** before
   clicking to also add them to *your* roster. If ESPN changed everything, this alone runs the draft.

All three feed one de-duplicated set (by ESPN id and by normalised name), persisted to
`chrome.storage.local` on every change.

### Switching to manual mode

There is no mode switch to hunt for: manual entry is always live and always visible. If the dot
never turns green and no picks appear, just type them in - and open **Settings** to untick
**DOM scan** if it is adding players that were not actually picked.

## Settings row

`scoring` (ppr / half_ppr / std), `teams`, `rounds`, `superflex`, `my ESPN team id` (optional - when
set, picks by that team id are also added to *your* roster), the **API** base URL, and an **access
code** (only needed if the deployment has `DRAFTADVISOR_ACCESS_CODE` set; it is sent as
`X-Access-Code`, and a 401 in the panel tells you to fill it in).

### Pointing at a local server

Set **API** to `http://127.0.0.1:8787` (any host/port; `host_permissions` already covers
`127.0.0.1` and `localhost`). The index is re-fetched immediately and the next advice call goes to
your local server. The default lives at the top of `overlay.js` (`API_BASE_DEFAULT`).

## If something goes wrong

* **Panel says "Advice API error"** - it shows the error text and a **Retry** button, and it falls
  back to a local ranking computed in the browser from the projections it already downloaded, so
  the panel is never blank. Fallback rows are the raw projection order (no positional-need logic).
* **"Player index unavailable"** - the `/api/projections` fetch failed. WebSocket ids still work
  (the server resolves them); manual search has no names to offer until it succeeds.
* **Nothing is being detected** - open **Debug - raw frames & verbs**. It lists every frame verb
  seen and the last 40 frames. Copy a couple of lines from there; that is exactly what is needed to
  fix the parser in one edit.

## Files

| file | what it is |
| --- | --- |
| `manifest.json` | MV3 manifest: `storage` permission, host permissions for the API + espn.com |
| `hook.js` | MAIN-world `WebSocket` wrapper, `document_start`; forwards raw frames, parses nothing |
| `styles.js` | the overlay stylesheet (a JS string, injected into the Shadow DOM) |
| `overlay.js` | panel, three detection layers, storage, API calls, local fallback |
| `selftest.html` | standalone page: stubbed `fetch`, faked `WebSocket`; open it in a browser to exercise everything without ESPN |

Plain JavaScript, no build step, no dependencies. `selftest.html` is checked headlessly with
Playwright (28 assertions: rendering, a `SELECTED` frame marking a pick, DOM-context detection,
de-duplication, manual entry, undo, persistence, the API-error fallback).

## Assumptions that are NOT verified against a live draft room

* The pick frame is `SELECTED <teamId> <playerId> <overallPick> <memberId>` in plain text. Taken
  from two public captures, not from this league. If the verb differs, the frame still shows up in
  Debug and (if it carries a known player id) as a one-click candidate.
* ESPN's draft room opens its socket with `window.WebSocket` after our `document_start` script
  runs. If it uses a worker, a pre-bundled socket, or long-polling instead, layer 1 sees nothing.
* The DOM scan's pick-context patterns are guesses about ESPN's markup.
* `/api/projections` rows are assumed to carry `espn_id` (also accepted as `espn` or `ids.espn`).
  Without it, only name matching works.
