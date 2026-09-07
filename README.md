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

## Install

```bash
git clone <this repo> && cd Sleeper-fantasy-draft-tool
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Before the draft (once, ~2 minutes)

```bash
export ANTHROPIC_API_KEY=sk-ant-...        # optional: enables Claude research + on-the-clock advice
draftadvisor ids --username YOUR_SLEEPER_USERNAME   # prints your league and draft ids for 2026
draftadvisor prep --league <LEAGUE_ID> --research    # downloads data, trains the model, caches projections,
                                                     # runs Claude research on the top 200 players (optional)
```

`prep` downloads ~70 MB of historical stats the first time, trains the projection model (< 3 min), pulls
your league's scoring/roster settings and Sleeper's ADP, and prints a backtest report plus the top-30 board.

## Draft day

```bash
draftadvisor draft --draft <DRAFT_ID> --username YOUR_SLEEPER_USERNAME
```

The dashboard updates as picks come in. When you are on the clock the header flashes **YOUR PICK** with the
timer, the best pick is highlighted, and (if enabled) Claude's 2–3 sentence take appears in the footer within
a few seconds. Ctrl-C exits. Use `--no-tui` for a plain-text stream, `--poll 3` to poll every 3 seconds.

## Practise

```bash
draftadvisor mock --teams 12 --rounds 15 --slot 5 --scoring half_ppr        # interactive: Enter = take the recommendation
draftadvisor mock --auto --seed 3                                           # let the advisor draft for you
```

## Other commands

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
