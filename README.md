# draftadvisor — live fantasy draft advisor for Sleeper or ESPN leagues

A laptop-friendly draft assistant for [Sleeper](https://sleeper.com) and [ESPN](https://fantasy.espn.com)
fantasy football leagues. It polls your draft every couple of seconds, recomputes value in milliseconds, and shows:

* the **best 3 players at every position** (and the best 6 overall) with plain-English reasons,
* whether to **take a position now or wait** ("84% chance Jonathan Taylor or better is there at your pick #41"),
* your roster needs, bye-week clashes, every opponent's open starting slots and the position runs in progress,
* projections from an **ML model trained on 2019–2025 NFL data** (nflverse), blended with the platform's own
  projections/ADP (Sleeper or ESPN) and FantasyPros expert consensus, all scored with **your league's exact
  scoring settings**,
* an optional **Claude chat** to argue with the recommendation — used only when you send a message, with the
  tokens and estimated cost shown under every answer (nothing calls Claude automatically),
* **mock drafts** against ADP-driven bots to practise, plus **trade / pick evaluation** and post-draft analysis.

Everything runs locally (Python 3.10+, no GPU). The only network calls are to the public Sleeper API, ESPN's
fantasy API (for ESPN leagues), GitHub (nflverse data) and — only if you set a key — the Anthropic API.

## Quick start

### Hosted on Vercel (no install)

The app is deployed from this branch to Vercel (project `sleeper-draft-advisor`):

**https://sleeper-draft-advisor.vercel.app**

Every push to the branch redeploys. The production URL is public (Vercel Authentication is kept on for preview
deployments only) so it opens on any device without a Vercel login; protect it with the access code below or
re-enable Vercel Authentication for production in Project → Settings → Deployment Protection. Open the URL,
go to **Settings** on the page and paste your Anthropic API key (it is stored only in your browser and sent
as a request header) — or set `ANTHROPIC_API_KEY` as a Vercel environment variable so nobody has to.
Recommended Vercel environment variables (Project → Settings → Environment Variables, then redeploy):

| variable | purpose |
|---|---|
| `DRAFTADVISOR_ACCESS_CODE` | a shared secret the page asks for once; without it anyone with the URL can use your Claude key |
| `ANTHROPIC_API_KEY` | server-side Claude key (optional if you paste one in the page) |
| `ESPN_S2`, `ESPN_SWID` | server-side ESPN cookies for a private ESPN league (optional: the page can hold them per browser instead, see *ESPN leagues*) |
| `BLOB_READ_WRITE_TOKEN` | optional: a Vercel Blob store holding research notes written by earlier versions (read-only now; without it those legacy notes are simply absent) |
| `DRAFTADVISOR_CHAT_MODEL` | default chat model, `claude-opus-5` ($5 / $25 per 1M input / output tokens); `claude-sonnet-5` ($2 / $10) can be picked in the page's Settings |

The server is stateless (the browser keeps the session), so the hosted app polls Sleeper or ESPN directly on
every refresh; there is nothing to "prepare" — model outputs ship in `web_bundle/` (rebuild with
`python scripts/build_bundle.py` after retraining and push).

### Run it locally

```bash
git clone <this repo> && cd Sleeper-fantasy-draft-tool
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[full]"                             # web app + data + model + CLI
export ANTHROPIC_API_KEY=sk-ant-...                      # optional
python run.py                                            # opens http://127.0.0.1:8787
```

### Using the app

1. **Setup tab** → *Live draft* → pick the platform. **Sleeper**: type your username → pick your league/draft.
   **ESPN**: paste the league id and season → *Load league* → pick your team (see *ESPN leagues* below).
   *Start live draft* captures the league (scoring diff vs Sleeper's base scoring — also for ESPN, whose rules are
   translated first — draft order, your pick numbers → **League tab**), then the **Draft tab** refreshes every 2 seconds: best picks with reasons,
   TAKE NOW / SOON / WAIT / SKIP per position, your roster and needs, opponents' needs, recent picks, a
   searchable list of everyone still available, and a **chat sidebar** (💬 button, floating button, or the
   `c` key; a sidebar next to the board on a wide screen, full screen on a phone) to argue with the
   recommendation ("why not the WR?", "compare #1 and #2", "any red flags on my top 3?"). The page works on
   a phone: the tables drop secondary columns, player cards open as a bottom sheet.
2. **Claude chat (optional, paid per message)**: Claude is called only when you press Send. Every answer
   shows its tokens and estimated cost ("12.1k in / 0.4k out — about $0.07"), the drawer head keeps a running
   total for the transcript, and a hint states roughly what one message costs with the selected model. There
   is no automatic advice and no research loop any more (see *Claude usage and cost* below).
3. **Mock draft**: choose teams / rounds / slot / scoring → bots draft by ADP; on your turn click a player,
   *Take recommended*, or press Enter; autopilot lets the advisor draft for you. The pick list lives in your
   browser, so a reload resumes the draft.
4. **Board tab**: the projection table (points, floor/ceiling, ADP, consensus rank) with search and filters.

The pick clock shown for a Sleeper draft is whatever Sleeper reports at that moment (`pick_timer` and the time
of the last pick, re-read on every refresh). Nothing in the advice depends on it, so a commissioner changing the
clock mid-draft changes only the display. ESPN's clock is different — see the next section.

## ESPN leagues

* **League id and season.** Open the league on fantasy.espn.com: the id is the `leagueId=` number in the URL
  (your team's page adds `teamId=`). ESPN drafts have no separate draft id — a draft is addressed by league id
  + season, so the Setup form asks for both (the season defaults to the one the server is built for).
* **Public vs private.** A public league needs nothing else. A private league answers 401 until the request
  carries your two ESPN cookies, `espn_s2` and `SWID`: on espn.com, logged in, open the browser dev tools →
  *Application* (Chrome / Edge) or *Storage* (Firefox / Safari) → *Cookies* → `espn.com` and copy both values
  (the SWID looks like `{XXXXXXXX-XXXX-...}`, braces included; they are added if you paste it without). Put
  them under **Settings → ESPN (private leagues)** on the page: they are stored only in that browser's
  localStorage and sent as the `X-ESPN-S2` / `X-ESPN-SWID` headers of every request, never logged, never
  written by the server. Alternatively set `ESPN_S2` / `ESPN_SWID` on the server (Vercel env or your shell) and
  every visitor of that deployment shares them. Cookies expire after a while — when ESPN starts answering 401
  again, paste fresh ones. `/api/status` reports `espn.server_cookies` so the page knows whether the server
  already has a pair.
* **Team picker.** *Load league* shows the league name, scoring type, draft type / clock / rounds and the
  scoring rules the advisor cannot model, then a team `<select>` (sorted by draft slot once the commissioner
  has set the order). When your SWID is known, the team it owns is preselected and marked "(you)"; the same
  works from the CLI with `--team-id`, `--slot` or `--username "team or owner name"`. Starting without a team
  spectates: the board still works but there is no "your pick" advice. The team id is the stable identity; the
  slot is re-derived from `pickOrder` on every poll, so a re-ordered draft before kickoff is followed.
* **Clock semantics.** ESPN publishes `timePerSelection` (re-read every poll, shown as "ESPN clock: *N* s per
  pick") but **no pick timestamps**, so there is no authoritative countdown. The page and the TUI show only an
  approximate countdown anchored to the moment *they* saw the pick number advance, labelled "≈ since last pick
  seen", and show nothing until a pick has been observed. Treat it as a hint.
* **Live pick updates.** The advisor renders whatever ESPN's `draftDetail` returns on each poll (one GET
  every ~2–3 s, plus a cached league capture). It uses the picks regardless of ESPN's `drafted` flag.
  ESPN publishes the **whole board up front**: once the order is set there is one entry per pick, and every
  pick that has not been made yet carries `playerId` `-1` (`0` in some seasons). Those entries are
  placeholders, and ESPN fills them **in place** as picks happen, so the advisor ignores any entry without a
  real player id and re-reads every entry on each poll (a pick made, corrected, or traded mid-list is picked
  up either way). A pre-populated board also carries each pick's `teamId`, which is used as the draft order
  when ESPN has not published `pickOrder` yet.
* **Scoring.** ESPN's `scoringItems` are translated into the same per-unit keys the engine uses for Sleeper
  ("every 10 yards" items become points per yard, TE/RB/WR reception overrides become position premiums,
  D/ST items take the D/ST override). Rules with no equivalent — 40+/50+ yard TD bonuses, per-distance TD
  bonuses, games played, turnovers, and similar — are listed as **"ESPN rules not modelled"** on the League
  tab (and in the capture report / `strategy flags`) with ESPN's own label and point value, so you know what
  the projections ignore.
* **Players, ADP, projections.** The player universe is the same one used for Sleeper leagues; ESPN ids are
  mapped onto it (by stored ESPN id, team defense id, or name + position). ESPN's `averageDraftPosition` is the
  ADP source (ECR fills the gaps), ESPN's projected season line is used for players Sleeper does not project,
  and an ESPN player the universe does not know appears as a placeholder with ESPN's name / position / team so
  a pick never shows up blank. Auction drafts are recognised but not advised.

## Claude usage and cost

Only two things in the whole project call the Anthropic API, and both need a person to act:

* the web chat (`POST /api/chat`) — one request per message you send;
* the CLI `draftadvisor ask "…"` — one request per invocation.

Nothing calls Claude on a timer, on every poll, when you come on the clock or in a loop. Earlier versions
researched every player with web search and asked for advice on every turn; that ran up a large bill, so
both were removed on purpose and there is no switch to bring them back. Each request re-sends the compact
draft context plus the chat history, so long conversations cost more per message. The page shows the tokens
and the estimate under every answer and a running total in the drawer; the CLI prints one line after the
answer. Prices used for the estimate (USD per 1M tokens):

| model | input | output | notes |
|---|---|---|---|
| `claude-opus-5` (web default) | 5 | 25 | cache reads billed at 10 %, cache writes at 125 % of the input price |
| `claude-sonnet-5` (CLI default, selectable in the page) | 2 | 10 | same cache rules |

Notes written by earlier research runs are still read (red ⚠ flags on cards, expected games / uncertainty in
the projections) but nothing writes new ones.

## Command line (optional)

The same engine is available as a CLI (`draftadvisor --help`): a terminal dashboard for the live draft,
mock drafts, the projection table, trade evaluation and post-draft analysis. Every command that names a
league takes `--platform sleeper|espn` (default Sleeper, env `DRAFTADVISOR_PLATFORM`).

```bash
# Sleeper
draftadvisor ids --username YOUR_SLEEPER_USERNAME        # your league and draft ids
draftadvisor prep --league <LEAGUE_ID>                    # data, model, league capture
draftadvisor draft --draft <DRAFT_ID> --username YOU      # terminal dashboard
draftadvisor mock --teams 12 --rounds 15 --slot 5         # terminal mock draft (--auto to watch)

# ESPN (the draft is addressed by league id + season; identify your team one of three ways)
draftadvisor capture --platform espn --league 1234567 --season 2026 --team-id 4
draftadvisor draft   --platform espn --league 1234567 --season 2026 --username "Gridiron Gang"
draftadvisor draft   --platform espn --league 1234567 --slot 7 --espn-s2 "$ESPN_S2" --swid "{...}"   # private league
draftadvisor ids     --platform espn --swid "{...}"       # best-effort list of your ESPN leagues (fan API)
```

`--espn-s2` / `--swid` fall back to `ESPN_S2` / `ESPN_SWID`; the SWID alone also identifies your team when no
`--team-id` / `--slot` / `--username` is given. `ids --platform espn` asks ESPN's fan API for the leagues of that
SWID; the API is undocumented and sometimes empty, in which case the command prints where to find the id in
the league URL instead. `--draft` is ignored with `--platform espn`. The ESPN poll interval defaults to 3 s.

## Pre-draft info capture

The first time you point the tool at a league it captures everything that shapes strategy and saves it under
`data/leagues/<league_id>.json` (`prep` and `draft` do this automatically; `draftadvisor capture --league ID`
does it on its own; an ESPN snapshot is also written as `espn-<league id>-<season>.json`):

* teams, managers, draft order, your slot and **every one of your pick numbers** (traded picks included),
* roster slots (starters / bench / IR / taxi) and league settings (keepers, playoffs, waivers, best ball),
* the full scoring rules and a **diff against Sleeper's base scoring** ("TE reception bonus: base — → 0.5",
  "Interception thrown: -1 → -2"), plus plain-English strategy flags derived from them
  (superflex, TE premium, 6-pt pass TD, no kicker slot, deep bench, ...; for ESPN also one flag per rule that
  could not be translated),
* draft type, rounds, pick clock, start time, keepers already on the board.

The report prints once; the snapshot lets trade/analyze commands work offline afterwards.

## Other CLI commands

| command | what it does |
|---|---|
| `draftadvisor projections --position RB --top 40 --league ID` | projection table (points, floor/ceiling, ADP, ECR) in your league's scoring |
| `draftadvisor trade --league ID --me me --them rival --give "Player A" --get "Player B, Player C"` | lineup-aware evaluation of one trade |
| `draftadvisor trades --league ID --me me` | **search** every other roster for trades worth sending (see below) |
| `draftadvisor lineup --platform espn --league ID --me me [--week N]` | best start/sit for the week, what your set lineup leaves on the bench, and the matchup win probability |
| `draftadvisor analyze --league ID` | post-draft strengths/weaknesses for every team, waiver targets |
| `draftadvisor ask "Should I take a TE in round 3?" --draft ID` | free-form question with full draft context — the only CLI command that calls Claude (one request); prints the answer, then tokens and estimated cost (or "cost unknown" for a model outside the price table) |
| `draftadvisor train --seasons 2019-2025` | retrain the projection model and print the backtest |

All of them accept `--platform espn --league ID [--season YYYY]` in place of the Sleeper ids.

## Finding trades

`draftadvisor trades --league ID --me "My Team"` searches the other rosters for deals and prints, for
each: who to ask, what to send, what you gain, how the deal reads to *them*, and a line to paste into
the league chat.

The two sides are priced differently, on purpose. **Your** side uses our blended projection — what the
model actually believes. **Their** side uses the market's consensus value (the FantasyPros
rank-implied number every projection already carries), because that is what the other manager
believes. A proposal has to clear both bars: it improves your starting lineup by our numbers, and it
does not read as a loss by theirs. Deals we would love but that look bad to them are filtered out —
they are not trades, they are messages that get ignored.

What the search will not do: offer a player you start, ask for one who would sit on your bench, trade
kickers or defences, or fill the list with eight variations of the same deal with one manager
(`--per-team`, default 2). `--min-gain` and `--min-their-view` loosen or tighten both bars;
`--one-for-one-only` turns off 2-for-1 consolidation.

Costs nothing: it is arithmetic over projections already in the bundle. No paid API is involved.

## In-season data

The bundle ships three small tables (~84 KB) that the draft never needed:

* **Weekly spread**, measured within season on 2019–2025 actuals. This is *not* `Projection.std`
  (season-total uncertainty) or `ppg_std_ppr` (the error of the season ppg forecast). A starter's real
  week-to-week swing is about 1.3× the latter, and 2.3× for a good receiver; a matchup win probability
  built on the smaller number reports confidence it has not earned.
* **NFL schedule** by team and week, so a weekly projection knows the opponent and a bye is a zero.
* **Defence versus position**, shrunk by sample size and clipped to ±20%: a nudge, never a start/sit
  decision on its own.

For an ESPN league the app also reads the league's own calendar (`matchupPeriodCount` decides when the
playoffs start — it was assuming week 15), standings, matchups and per-week player scores. Those
scores come back **already scored under your league's rules**, so custom scoring needs no re-derivation.

## How the recommendations work

1. **Projections.** For every player: per-game stat rates from gradient-boosted models (one per position and
   stat, trained on prior-season usage, efficiency, age, draft capital, snaps, injuries and team context),
   the platform's projected stat line (Sleeper's, with ESPN's as the fallback for an ESPN league), and a
   FantasyPros-consensus-implied value. The three are converted to points with your league's scoring and
   blended (default 45/35/20 with a platform line, 60/40 without), then adjusted for injury status, depth
   chart and any legacy research notes on disk. Each projection carries an uncertainty, floor and ceiling.
2. **Value.** VORP against replacement level computed on the *remaining* pool, the marginal improvement to your
   optimal starting lineup (FLEX/SUPER_FLEX aware) plus discounted bench value, bye-week and stacking effects,
   and an early-K/DEF penalty.
3. **Timing.** Each player's expected draft position (ADP/ECR with uncertainty) gives the probability they are
   still there at your next pick, shifted by the positional needs of the teams picking before you and by runs.
   VONA (value over next available) tells you what you lose by waiting.
4. **Advice.** Per position: TAKE NOW / SOON / WAIT / SKIP with the probability and pick number that justify it.

See `DESIGN.md` for the module layout and contracts (§3.10 for the ESPN provider).

## Configuration

Environment variables (CLI flags override them):

| variable | purpose |
|---|---|
| `DRAFTADVISOR_PLATFORM` | default `--platform` (`sleeper` or `espn`) |
| `SLEEPER_LEAGUE_ID`, `SLEEPER_DRAFT_ID`, `SLEEPER_USERNAME`, `SLEEPER_USER_ID` | Sleeper ids (read only when the platform is Sleeper) |
| `ESPN_LEAGUE_ID`, `ESPN_TEAM_ID` | ESPN ids (read only when the platform is ESPN) |
| `ESPN_S2`, `ESPN_SWID` | ESPN cookies for private leagues (CLI, and the web server's fallback when the browser sends none) |
| `DRAFTADVISOR_ESPN_BASE` | ESPN API base URL, default `https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl` (the tests point it at a local stub); `DRAFTADVISOR_ESPN_FAN_BASE` does the same for the fan API used by `ids` |
| `DRAFTADVISOR_SEASON` | season being drafted (default 2026); also the default ESPN season |
| `DRAFTADVISOR_HOME` | data / cache directory (default `./data`) |
| `ANTHROPIC_API_KEY` | Claude key (or paste one under Settings in the page) |
| `DRAFTADVISOR_CLAUDE_MODEL` | model of the CLI `ask` command, default `claude-sonnet-5` |
| `DRAFTADVISOR_CHAT_MODEL` | web chat model, default `claude-opus-5` |
| `DRAFTADVISOR_ACCESS_CODE` | when set, every `/api/*` call needs the `X-Access-Code` header (the page asks once) |

## Limitations

* Snake, third-round-reversal and linear drafts are supported; auction drafts are not (ESPN auction leagues are
  detected but not advised).
* IDP positions are ignored (they are not projected).
* ESPN: no authoritative pick clock (see *ESPN leagues*), scoring rules without a Sleeper equivalent are
  listed rather than modelled, and live pick updates depend on what ESPN's `draftDetail` returns during a draft.
* Claude is optional and only ever called when you send a chat message or run `ask`; without a key everything
  else works unchanged.
