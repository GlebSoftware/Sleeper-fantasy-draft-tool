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

## Quick start (web app)

```bash
git clone <this repo> && cd Sleeper-fantasy-draft-tool
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...                      # optional: enables Claude research + advice
python run.py                                            # opens http://127.0.0.1:8787 in your browser
```

Everything happens in the browser tab:

1. **Setup tab** → *Prepare data & train model* (once, ~2 minutes: downloads ~70 MB of NFL history, trains
   the projection model, caches projections). The log streams on the page.
2. **Live draft**: type your Sleeper username → pick your league/draft from the list → *Start*. The app captures
   the league (scoring diff vs Sleeper base, draft order, your pick numbers → **League tab**), then the
   **Draft tab** updates every 2 seconds: best picks with reasons, TAKE NOW / SOON / WAIT / SKIP per position,
   your roster and needs, opponents' needs, recent picks, a searchable list of everyone still available, and
   Claude's take when you are on the clock.
3. **Mock draft**: choose teams / rounds / your slot / scoring → *Start mock*. Bots draft by ADP; when it is
   your turn click a player (or *Take recommended*, or press Enter). Autopilot lets the advisor draft for you.
4. **Board tab**: the full projection table (points, floor/ceiling, ADP, consensus rank) with search and filters.
5. **Ask tab**: free-form questions to Claude with the live draft as context (needs the API key).

Everything runs locally; the only network calls are to the public Sleeper API, GitHub (nflverse data) and,
only if you set a key, the Anthropic API. The page is one file — `draftadvisor/web/static/index.html`
(vanilla HTML/CSS/JS, no build step) — and the server is `draftadvisor/web/server.py` (FastAPI), so both are
easy to modify. `python run.py --port 9000` changes the port; `--no-browser` skips opening a tab.

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
