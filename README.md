# draftadvisor — live Sleeper fantasy draft advisor

A laptop-friendly draft assistant for [Sleeper](https://sleeper.com) leagues. It polls your draft every
couple of seconds, recomputes value in milliseconds, and shows:

* the **best 3 players at every position** (and the best 6 overall) with plain-English reasons,
* whether to **take a position now or wait** ("84% chance Jonathan Taylor or better is there at your pick #41"),
* your roster needs, bye-week clashes, every opponent's open starting slots and the position runs in progress,
* projections from an **ML model trained on 2019–2025 NFL data** (nflverse), blended with Sleeper's own
  projections/ADP and FantasyPros expert consensus, all scored with **your league's exact scoring settings**,
* optional **Claude (Sonnet) research notes** per player and a short on-the-clock recommendation,
* **mock drafts** against ADP-driven bots to practise, plus **trade / pick evaluation** and post-draft analysis.

Everything runs locally (Python 3.10+, no GPU). The only network calls are to the public Sleeper API,
GitHub (nflverse data) and — only if you set a key — the Anthropic API.

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
| `BLOB_READ_WRITE_TOKEN` | attach a Vercel Blob store (Storage tab) so research notes persist across serverless instances; otherwise notes live only for the life of one instance |
| `DRAFTADVISOR_CHAT_MODEL` | chat model, default `claude-opus-5` (research always uses `claude-sonnet-5`) |

The server is stateless (the browser keeps the session), so the hosted app polls Sleeper directly on every
refresh; there is nothing to "prepare" — model outputs ship in `web_bundle/` (rebuild with
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

1. **Setup tab** → type your Sleeper username → pick your league/draft → *Start live draft*. The app captures
   the league (scoring diff vs Sleeper base, draft order, your pick numbers → **League tab**), then the
   **Draft tab** refreshes every 2 seconds: best picks with reasons, TAKE NOW / SOON / WAIT / SKIP per position,
   your roster and needs, opponents' needs, recent picks, a searchable list of everyone still available,
   Claude's take when you are on the clock, and a **chat** panel to argue with the recommendation
   ("why not the WR?", "compare #1 and #2", "any red flags on my top 3?").
2. **Research**: *Research top N* runs Claude (Sonnet) with live web search per player — injuries, off-field
   issues (arrests, lawsuits, suspensions), holdouts, depth-chart changes, bust/breakout commentary — and
   stores `red_flags`, `injury_risk`, `role_certainty`, `offfield_risk` on each note. Flags show as a red ⚠ on
   every card and feed the projections (expected games and uncertainty). Any player card has *Research now*.
3. **Mock draft**: choose teams / rounds / slot / scoring → bots draft by ADP; on your turn click a player,
   *Take recommended*, or press Enter; autopilot lets the advisor draft for you. The pick list lives in your
   browser, so a reload resumes the draft.
4. **Board tab**: the projection table (points, floor/ceiling, ADP, consensus rank) with search and filters.

The pick clock shown is whatever Sleeper reports at that moment (`pick_timer` and the time of the last pick,
re-read on every refresh). Nothing in the advice depends on it, so a commissioner changing the clock mid-draft
changes only the display.

## Command line (optional)

The same engine is available as a CLI (`draftadvisor --help`): a terminal dashboard for the live draft,
mock drafts, the projection table, trade evaluation and post-draft analysis.

```bash
draftadvisor ids --username YOUR_SLEEPER_USERNAME        # your league and draft ids
draftadvisor prep --league <LEAGUE_ID> --research         # data, model, capture, Claude research
draftadvisor draft --draft <DRAFT_ID> --username YOU      # terminal dashboard
draftadvisor mock --teams 12 --rounds 15 --slot 5         # terminal mock draft (--auto to watch)
```

## Pre-draft info capture

The first time you point the tool at a league it captures everything that shapes strategy and saves it under
`data/leagues/<league_id>.json` (`prep` and `draft` do this automatically; `draftadvisor capture --league ID`
does it on its own):

* teams, managers, draft order, your slot and **every one of your pick numbers** (traded picks included),
* roster slots (starters / bench / IR / taxi) and league settings (keepers, playoffs, waivers, best ball),
* the full scoring rules and a **diff against Sleeper's base scoring** ("TE reception bonus: base — → 0.5",
  "Interception thrown: -1 → -2"), plus plain-English strategy flags derived from them
  (superflex, TE premium, 6-pt pass TD, no kicker slot, deep bench, ...),
* draft type, rounds, pick clock, start time, keepers already on the board.

The report prints once; the snapshot lets trade/analyze commands work offline afterwards.

## Other CLI commands

| command | what it does |
|---|---|
| `draftadvisor projections --position RB --top 40 --league ID` | projection table (points, floor/ceiling, ADP, ECR) in your league's scoring |
| `draftadvisor trade --league ID --me me --them rival --give "Player A" --get "Player B, Player C"` | lineup-aware trade evaluation |
| `draftadvisor analyze --league ID` | post-draft strengths/weaknesses for every team, waiver targets |
| `draftadvisor research --top 200 --league ID` | (re)run Claude research notes |
| `draftadvisor ask "Should I take a TE in round 3?" --draft ID` | free-form question with full draft context |
| `draftadvisor train --seasons 2019-2025` | retrain the projection model and print the backtest |

## How the recommendations work

1. **Projections.** For every player: per-game stat rates from gradient-boosted models (one per position and
   stat, trained on prior-season usage, efficiency, age, draft capital, snaps, injuries and team context),
   Sleeper's projected stat line, and a FantasyPros-consensus-implied value. The three are converted to points
   with your league's `scoring_settings` and blended (default 45/35/20 with Sleeper, 60/40 without),
   then adjusted for injury status, depth chart and Claude's injury/role notes. Each projection carries an
   uncertainty, floor and ceiling.
2. **Value.** VORP against replacement level computed on the *remaining* pool, the marginal improvement to your
   optimal starting lineup (FLEX/SUPER_FLEX aware) plus discounted bench value, bye-week and stacking effects,
   and an early-K/DEF penalty.
3. **Timing.** Each player's expected draft position (ADP/ECR with uncertainty) gives the probability they are
   still there at your next pick, shifted by the positional needs of the teams picking before you and by runs.
   VONA (value over next available) tells you what you lose by waiting.
4. **Advice.** Per position: TAKE NOW / SOON / WAIT / SKIP with the probability and pick number that justify it.

See `DESIGN.md` for the module layout and contracts.

## Configuration

Environment variables (or CLI flags): `SLEEPER_LEAGUE_ID`, `SLEEPER_DRAFT_ID`, `SLEEPER_USERNAME`,
`ANTHROPIC_API_KEY`, `DRAFTADVISOR_HOME` (data/cache directory, default `./data`),
`DRAFTADVISOR_SEASON` (default 2026), `DRAFTADVISOR_CLAUDE_MODEL` (default `claude-sonnet-5`).

## Limitations

* Snake, third-round-reversal and linear drafts are supported; auction drafts are not.
* IDP positions are ignored (they are not projected).
* Claude research is optional and rate-limited; without a key everything else works unchanged.
