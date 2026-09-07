# draftadvisor — design & module contracts

Live Sleeper draft advisor. Polls the draft every ~2 s, recomputes recommendations in
milliseconds, shows the best 3 players per position (and overall) with reasons, tells you
when you can wait on a position, models opponents' needs, and optionally asks Claude
(Sonnet) for research and on-the-clock advice. Also: mock drafts, trade/pick evaluation.

Target: runs on a laptop (4 cores, no GPU), Python 3.10+, install with `pip install -e .`.

## 1. Data flow

```
                 ┌──────────── one-time / daily (`draftadvisor prep`) ────────────┐
 nflverse CSVs ──► data/nflverse.py ──► canonical per-game frame ──► projections/features.py
 (2019-2025)        (cached gz)          (Sleeper stat keys)             │
 dynastyprocess ──► data/crosswalk.py  (sleeper_id <-> gsis_id <-> fantasypros_id)
 FantasyPros ECR ─► data/fantasypros.py                                  ▼
                                                             projections/model.py (HistGB per position)
                                                                          │  per-game stat rates + games
 Sleeper API ─────► sleeper/client.py  players, season projections (+ADP)  │
                    league (scoring_settings, roster_positions)            ▼
                                            scoring/engine.py ◄── projections/blend.py ──► dict[player_id, Projection]
                                                                                            (league points, std, weekly)
                 ┌──────────── every 2 s during the draft (`draftadvisor draft`) ────────────┐
 Sleeper picks ──► sleeper/poller.py ──► DraftState ──► strategy/recommend.py (Advisor) ──► Recommendation
                                                          │  VORP, lineup marginal value,          │
                                                          │  availability @ my next pick,           ▼
                                                          │  opponent needs / position runs    ui/dashboard.py (rich Live)
                                                          └─► research/claude.py (async, optional) ─┘
```

Identity everywhere: **Sleeper player_id** (string). DEF ids are team abbreviations ("SF").

## 2. Contracts (already implemented — do not change signatures without updating DESIGN.md)

* `draftadvisor/config.py` — constants, paths (`home_dir()` = `$DRAFTADVISOR_HOME` or `./data`), `Settings`.
* `draftadvisor/models.py` — `Player`, `LeagueSettings`, `DraftSettings` (snake / 3RR / linear pick math,
  traded picks), `Pick`, `DraftState` (derived: `next_pick_no`, `my_next_pick_no`, `is_my_turn`, ...),
  `Manager`, `Projection`, `RosterSummary`, `PlayerValue`, `PositionAdvice`, `Recommendation`,
  `ResearchNote`, `normal_cdf`, `normal_ppf`.
* `draftadvisor/scoring/engine.py` — `ScoringEngine(scoring_settings)`: `.score(dict, position)`,
  `.score_frame(df)`, `ScoringEngine.add_derived_keys(df)`, `aggregate_season(...)`, `DEFAULT_SCORING`.
* `draftadvisor/data/canonical.py` — nflverse -> Sleeper keys per game (`player_weekly_to_canonical`,
  `team_weekly_to_canonical`, `to_sleeper_team`).
* `draftadvisor/data/nflverse.py` — `load_canonical(seasons)`, `load_canonical_season`, `load_roster(season)`,
  `load_schedule()`, `bye_weeks(season)`, `load_snap_counts(season)`, `load_injuries(season)`,
  `team_context(canonical)`.
* `draftadvisor/data/fantasypros.py` — `load_ecr_raw()`, `overall_ecr(raw, superflex)`, `positional_ecr(raw)`.
* `draftadvisor/data/crosswalk.py` — `build_crosswalk(season)` -> `Crosswalk` (`gsis_for`, `sleeper_for_gsis`,
  `sleeper_for_fp`, `sleeper_for_name`, `attrs`, `pfr_to_gsis`), `normalize_name`.
* `draftadvisor/data/universe.py` — `players_from_sleeper(payload)`, `players_from_crosswalk(cw, roster)`,
  `enrich_players(players, cw, ecr, byes, pos_ecr)`, `assign_adp(players, adp_by_id, source)`, `filter_relevant`.
* `draftadvisor/data/cache.py` — `download`, `cached_frame`, `cached_json`, `read_json`, `write_json`.

Raw data for the sandbox is already in `data/raw/` (2019-2025 weekly stats, team stats, rosters 2019-2026,
snap counts, injuries, schedule, `db_playerids.csv`, `db_fpecr_latest.csv`). Loaders use it without
downloading. **The Sleeper API and Anthropic API are NOT reachable from the sandbox** — everything that
touches them must be testable with fixtures / mocks (see `tests/fixtures/*.json`, faithful to the
Sleeper API docs shapes) and must degrade gracefully offline.

Canonical per-game frame columns: `player_id` (gsis id, or team abbr for DEF), `player_name`, `position`
(QB/RB/WR/TE/K/DEF), `team`, `opponent`, `season`, `week`, every derivable Sleeper stat key
(`pass_att, pass_cmp, pass_inc, pass_yd, pass_td, pass_int, pass_2pt, pass_sack, pass_fd, rush_att, rush_yd,
rush_td, rush_2pt, rush_fd, rec, rec_tgt, rec_yd, rec_td, rec_2pt, rec_fd, fum, fum_lost, fum_rec_td, st_td,
kr_yd, pr_yd, fgm, fga, fgmiss, fgm_0_19..fgm_50_59, fgm_60p, fgm_50p, fgmiss_*, fgm_yds, fgm_yds_over_30,
xpm, xpa, xpmiss, sack, int, ff, fum_rec, safe, def_td, def_st_td, blk_kick, def_pass_def, def_2pt, pts_allow,
yds_allow, pts_allow_* brackets, yds_allow_* brackets, bonus_* thresholds, bonus_rec_rb/wr/te`) and features
`f_target_share, f_air_yards_share, f_wopr, f_racr, f_receiving_epa, f_rushing_epa, f_passing_epa,
f_receiving_air_yards, f_passing_air_yards, f_passing_cpoe, f_receiving_yards_after_catch,
f_passing_yards_after_catch, f_pacr, f_nfl_fantasy_points, f_nfl_fantasy_points_ppr`.

## 3. Module specs (to implement)

### 3.1 `sleeper/` — client, parsing, poller  (owner: agent "sleeper")

`client.py`
* `class SleeperAPIError(Exception)` with `.status_code`, `.url`; `class SleeperNotFound(SleeperAPIError)`.
* `class SleeperClient` (httpx.AsyncClient, keep-alive, 10 s timeout, 3 retries with backoff on
  timeouts/5xx/429, never on 404). Methods (all `async`, return parsed JSON):
  `get_state(sport="nfl")`, `get_user(username_or_id)`, `get_user_leagues(user_id, season)`,
  `get_user_drafts(user_id, season)`, `get_league(id)`, `get_league_users(id)`, `get_league_rosters(id)`,
  `get_league_drafts(id)`, `get_draft(id)`, `get_draft_picks(id)`, `get_traded_picks(draft_id)`,
  `get_players(force_refresh=False)` (5 MB; cache to `cache_dir()/sleeper_players.json.gz` for 24 h),
  `get_trending(add_drop, hours, limit)`, `get_season_projections(season, season_type="regular")`,
  `get_week_projections(season, week)`, `get_season_stats(season)`, `get_week_stats(season, week)`.
  Projections/stats are **normalised to `dict[player_id, stats_dict]`**. Try the v1 path
  `/v1/projections/nfl/{season_type}/{season}` (dict keyed by player_id, includes `adp_ppr`, `adp_half_ppr`,
  `adp_std`, `adp_2qb`, `adp_dynasty*`, `pts_ppr`, `gp`, stat keys) and fall back to the v2 list endpoint
  `https://api.sleeper.com/projections/nfl/{season}?season_type=regular&position[]=QB&...&order_by=adp_ppr`
  (list of `{player_id, stats:{...}, player:{...}}`) — merge `stats` and keep `player_id`.
  Accept an injected `httpx.AsyncClient` (for tests with `httpx.MockTransport`).
* `run_sync(coro)` helper for CLI one-offs.

`parsing.py` — pure functions raw JSON -> models:
`parse_league(raw) -> LeagueSettings`, `parse_draft(raw, traded_picks=None) -> DraftSettings`
(`settings.teams/rounds/pick_timer/reversal_round`, `draft_order`, `slot_to_roster_id` keys are strings in
JSON -> ints; traded picks `[{round, roster_id (original), owner_id (current)}]` -> `traded_picks[(round, orig)] = owner`),
`parse_pick(raw) -> Pick` (roster_id may be int or str or None), `parse_picks(list)` (sorted, de-duplicated by
pick_no), `parse_managers(users, draft, rosters=None) -> dict[user_id, Manager]` (slot from draft_order, roster_id
from rosters owner_id or `slot_to_roster_id`), `adp_key_for(league, draft) -> str` (`adp_ppr` if rec>=0.75,
`adp_half_ppr` if >=0.25 else `adp_std`; `adp_2qb` when superflex; `adp_dynasty*` never for redraft),
`adp_map(projections, key) -> dict[player_id, float]`,
`resolve_my_slot(draft, managers, username=None, user_id=None, slot=None) -> (user_id, slot)` (username
matching is case-insensitive against `Manager.display_name` and the users' `username`; explicit slot wins),
`state_from_sleeper(draft_raw, picks_raw, league_raw=None, users_raw=None, rosters_raw=None, traded_raw=None,
 username=None, user_id=None, slot=None) -> DraftState`.

`poller.py`
* `class DraftPoller(client, draft_id, *, league_id=None, settings: Settings, username=None, user_id=None, slot=None)`
  `await bootstrap() -> DraftState` (draft, league via draft.league_id or league_id, users, rosters, traded picks, picks);
  `await poll_once() -> DraftState | None` (fetch picks + draft status; return a *new* `DraftState` (via
  `state.with_picks`) only when the pick list or draft status changed — compare `(len, last pick_no, status)`);
  `await run(on_update, stop: asyncio.Event | None = None) -> DraftState` (calls `on_update(state)` after
  bootstrap and on every change; awaits it if it returns an awaitable; adaptive interval: `settings.poll_seconds`
  normally, 1.0 s when `picks_until_my_turn <= 1`, 10 s while `status == "pre_draft"`; on exceptions log,
  set `last_error`, back off exponentially to 30 s, keep going; return when `state.is_complete` or `stop` is set).
  Attributes: `state`, `last_error`, `poll_count`, `last_latency_ms`, `last_poll_at`.

Tests: `tests/test_sleeper_parsing.py`, `tests/test_sleeper_client.py` (MockTransport), `tests/test_poller.py`
(fake client returning fixture picks growing over time; assert new states only on change, my-turn cadence).

### 3.2 `mock/simulator.py` — offline draft simulator  (owner: agent "sleeper")

* `make_mock_league(teams=12, rounds=15, scoring="half_ppr"|"ppr"|"std", superflex=False, te_premium=0.0) -> LeagueSettings`
  (roster_positions: QB,RB,RB,WR,WR,TE,FLEX,K,DEF + BN to fill rounds; superflex adds SUPER_FLEX).
* `make_mock_draft(league, my_slot, teams=12, rounds=15, pick_timer=30, reversal_round=0) -> DraftSettings`
  (draft_order maps synthetic user ids `"bot-<slot>"` and `"me"` to slots; slot_to_roster_id identity).
* `class MockDraft(players: dict[str, Player], league, draft, my_slot, seed=None, bot_noise=0.15)`:
  `state() -> DraftState` (managers named "Bot 3" etc, mine "You"), `is_my_turn`, `is_complete`,
  `bot_pick() -> Pick` (for the slot on the clock), `make_pick(player_id) -> Pick` (validates undrafted,
  raises `ValueError`), `advance_until_my_turn() -> DraftState`, `run_to_completion(my_policy) -> DraftState`.
  Bot policy: score = -(adp or ecr or 400) + Normal(0, bot_noise * max(4, 0.1*adp)); each bot has a random
  `BotStrategy` in {"balanced","zero_rb","hero_rb","early_qb","late_qb"} that shifts position preferences;
  bots never take a 2nd K/DEF, take K/DEF only in the last 2-3 rounds, avoid a 3rd QB (2nd unless superflex),
  respect starting-lineup needs in later rounds (prefer open starter slots), and never exceed roster size.
  Pick metadata mirrors Sleeper (`first_name`, `last_name`, `position`, `team`).

Tests: `tests/test_mock.py` (full 12x15 draft completes, no duplicate players, my picks land on my slots,
K/DEF not before round rounds-3 for bots, superflex variant drafts more QBs).

### 3.3 `projections/` — ML model + blending  (owner: agent "projections")

`features.py`
* `TARGETS: dict[str, list[str]]` per-game rate targets:
  QB: `pass_att, pass_cmp, pass_yd, pass_td, pass_int, rush_att, rush_yd, rush_td, fum_lost, bonus_pass_yd_300, bonus_rush_yd_100`
  RB/WR/TE: `rush_att, rush_yd, rush_td, rec_tgt, rec, rec_yd, rec_td, fum_lost, bonus_rush_yd_100, bonus_rec_yd_100`
  K: `fgm_0_19, fgm_20_29, fgm_30_39, fgm_40_49, fgm_50p, fgmiss, xpm, xpmiss`
  DEF: `sack, int, ff, fum_rec, safe, def_td, blk_kick, pts_allow, yds_allow`  (DEF also needs the bracket
  distribution — approximate bracket indicators from predicted `pts_allow` per game using the empirical
  distribution of per-game points allowed around that mean; see blend.py).
  Every position also gets target `games` (REG games with a stat row; team defenses 17).
* `season_aggregates(canonical, snaps=None, injuries=None) -> DataFrame` one row per (player_id, season, position):
  `games`, per-game means of every TARGET key for that position (+ `ppg_ppr`, `ppg_std` from `f_nfl_fantasy_points*`),
  usage: mean `f_target_share`, `f_air_yards_share`, `f_wopr`, sum targets/carries, `snap_pct` (from snap counts
  joined via `Crosswalk.pfr_to_gsis`), `weeks_out` (injury report "Out"), `team` (most frequent), efficiency
  (`ypc`, `ypt`, `td_per_touch`, `catch_rate`, `ypa`, `td_rate`).
* `build_training_table(agg, team_ctx, rosters_by_season, seasons) -> DataFrame` rows keyed by
  (player_id, season S): features from S-1 (`prev_*`), S-2 (`prev2_*`), `career_*` per-game to date, `age`
  (birth_date from roster, or draft_year heuristic), `years_exp`, `draft_ovr` (missing -> 300, `undrafted=1`),
  `rookie` flag (no prior NFL season), `team_changed`, prior team context (`prev_team_pass_att_pg`, ...),
  `prev_weeks_out`; targets `y_<key>` = S per-game rates (NaN when 0 games), `y_games`. Only players on a
  roster in S (any status) at skill positions. Include S-1 rows for rookies with all prev_* = NaN.
* `build_inference_table(agg, team_ctx, players: dict[str, Player], cw, season) -> DataFrame` index = Sleeper
  player_id, same feature columns (history via `cw.gsis_for`), current team from `Player.team`.
* `FEATURE_COLUMNS: list[str]` (the union used by the model).

`model.py`
* `class ProjectionModel(seed=7, params=None)`: per position, per target a
  `sklearn.ensemble.HistGradientBoostingRegressor` (loss "squared_error" for rates, `max_iter`≈300,
  `learning_rate`≈0.05, `max_depth`≈4, `min_samples_leaf`≈20, `l2_regularization`≈1.0, NaN-native).
  Rookies: separate small model per position using only (`draft_ovr`, `undrafted`, `age`, position) trained on
  historical rookie seasons; used when `rookie == 1`. Also a `games` model (HGB, clipped to [0, 17]) and
  residual std per position estimated from 5-fold out-of-fold predictions of `ppg_ppr` (heteroscedastic:
  `std = a + b * pred`, fit by simple regression of |resid| on pred).
  `fit(table)`, `predict(table) -> DataFrame` (`pred_<key>` per target, `pred_games`, `ppg_std_ppr`,
  `rookie_flag`), `backtest(table, holdout_seasons=[2024, 2025]) -> dict` (per position: MAE and Spearman of
  predicted PPR ppg vs actual for players with >= 6 games, versus baselines "last season ppg" and "career ppg";
  the model must beat "last season ppg" MAE for RB/WR/TE — report numbers), `save(path=models_dir()/"projection_model.pkl")`,
  `load(path)`.
* `train_and_save(seasons=TRAIN_SEASONS, refresh=False) -> (model, metrics)`; run it in the sandbox with the
  data in `data/raw` and put the metrics in the final report. Training must finish in < 3 minutes on 4 cores.

`blend.py`
* `rates_to_season(rates: dict, games: float, engine, position) -> (points, ppg, stat_line)` where derived keys
  (bonus thresholds) come from the rate targets, `pass_inc = pass_att - pass_cmp`, K `fgm = sum brackets`,
  `fga = fgm + fgmiss`, `xpa = xpm + xpmiss`, DEF bracket rates from `pts_allow` mean via
  `def_bracket_rates(mean_pts_allow)` (empirical table: fit on the canonical DEF rows: for bins of team-season
  mean points allowed, the share of games in each bracket).
* `sleeper_stats_to_projection(stats, engine, position) -> (points, games, stat_line)`: score Sleeper's
  projected stat line with the league engine (`gp` or 17 games).
* `ecr_implied_points(players, provisional_points, position) -> dict[id, float]`: isotonic (monotone
  decreasing) regression of provisional points on ECR overall rank within position; a player's ECR-implied
  points = fitted value at their ECR. Needs >= 8 points else skip.
* `class Projector(engine, league=None, settings=None, weights=None)` with default weights
  `{"sleeper": 0.45, "ml": 0.35, "ecr": 0.20}` re-normalised over available sources per player;
  `project(players, ml_pred=None, sleeper_proj=None, notes=None, byes=None) -> dict[id, Projection]`:
  1. per-source season points (ML via `rates_to_season`, Sleeper via `sleeper_stats_to_projection`, ECR-implied);
  2. blend -> `points`; `games` = blend of ML games and Sleeper `gp`, then adjustments:
     `injury_status` IR/PUP/NFI -> games -= 6 (min 0; if `status` == "Injured Reserve" and ECR missing -> games *= 0.3),
     Out/Doubtful -> games -= 1, Sus -> games -= 4 (all clipped), depth_chart_order >= 3 for RB/WR -> points *= 0.8
     unless ECR rank < 100 (market disagrees), research note `injury_risk` r -> games *= (1 - 0.25*r),
     `role_certainty` c -> std *= (1.3 - 0.3*c);
  3. `std`: from ML `ppg_std` * sqrt(games) blended with ECR sd converted to points (slope of the isotonic
     curve * sd), rookies * 1.25, min 8% of points; `floor`/`ceiling` = points ∓ 0.84*std;
  4. `weekly`: points spread evenly over the 17 fantasy weeks (0 on bye; if games < 17, scale);
  5. `components`, `weights`, `flags` filled. Players with no source at all get `points=0` and flag `"no_data"`.
* Provide `project_offline(players, engine, canonical, cw, season)` convenience: last-season league-scored
  PPG regressed to the position mean by games (shrinkage k=6) as the "ml" source when no model exists.

Tests: `tests/test_projections.py` — synthetic canonical frame -> features (shapes, no leakage: features for
season S only use < S), `rates_to_season` on a hand-computed line, isotonic ECR mapping monotone, blend
weight renormalisation when sources are missing, injury/depth adjustments, `Projection.weekly` zero on bye.
Plus a slow, opt-in test (`@pytest.mark.skipif(not Path("data/raw").exists())`) that trains on real data
2019-2023, backtests 2024-2025, and asserts the model beats the last-season baseline for RB/WR.

### 3.4 `strategy/` — value & recommendations  (owner: agent "strategy")

`replacement.py`
* `starter_demand(league) -> dict[pos, float]`: dedicated starters * teams plus each flex slot's share
  allocated across eligible positions (FLEX: RB 0.45 / WR 0.45 / TE 0.10; SUPER_FLEX: QB 0.85 / rest split;
  WRRB_FLEX 0.5/0.5; REC_FLEX WR 0.75/TE 0.25). Bench allowance: RB +1.0 * teams * 0.5, WR +1.0 * teams * 0.5,
  QB +0.15*teams (0.75*teams if superflex), TE +0.15*teams, K/DEF 0.
* `replacement_levels(projections, players, league, available=None) -> dict[pos, float]`: the points of the
  player at rank `round(demand)+1` at each position among `available` (or all) players, computed on
  `Projection.points`. Cache-friendly (called every tick with the current available set).
* `vorp(projections, players, league, available=None) -> dict[id, float]`.
* `tiers(values: list[tuple[id, points]], std_by_id) -> dict[id, int]`: gap-based tiering — new tier when the
  drop to the next player exceeds `max(0.06 * top_points, 0.5 * mean std)`; tier 1 = best.

`lineup.py`
* `optimal_lineup(candidates: list[tuple[Player, float]], slots: list[str]) -> (assignment: dict[int, str],
  starters_points: float, bench: list[str])` — fill dedicated slots first (by points desc per position), then
  multi-position slots greedily by best remaining eligible player, then try single-swap improvements
  (a player in a dedicated slot moving to a flex to let a better player in). Deterministic. < 50 µs per call.
* `roster_summary(state, slot, players, projections, league) -> RosterSummary` (starters filled/open per slot
  label, position counts, bye overlap among starters, lineup/bench points).
* `marginal_lineup_value(candidate: Player, cand_points: float, roster: list[tuple[Player, float]], league,
  bench_discount: float) -> float` = lineup(roster+cand) − lineup(roster) + bench_discount * (cand points if
  benched, scaled by "bench usefulness": RB/WR 1.0, TE/QB 0.5 (QB 1.0 in superflex), K/DEF 0.0).

`availability.py`
* `pick_distribution(player, current_pick) -> (mu, sigma)`: mu = ADP if present (else ECR, else 400);
  sigma = `max(2.5, 0.10 * mu + 1.5)` blended with `ecr_sd * 1.2` when present. Values are in overall-pick
  units; clip mu >= current_pick - 0.5 (a player still on the board cannot have been taken).
* `prob_available(player, at_pick, current_pick, shift=0.0) -> float` = S(at_pick - 0.5) / S(current_pick - 0.5)
  with S the survival of Normal(mu - shift, sigma), truncated at current_pick; return 1.0 when at_pick <=
  current_pick.
* `position_pressure(state, opponent_summaries, available: list[Player], projections, until_pick) -> dict[pos, float]`:
  expected number of picks at each position by the teams picking between now and `until_pick`: for each such
  team, distribute one pick across positions with weights = (need weight: open dedicated starter 1.0, flex
  need 0.5, otherwise 0.15) × (market weight: share of the top-12 available players by ADP at that position);
  add a "run" term: +0.5 per pick at that position in the last 6 picks league-wide. Return per position the
  sum. `shift_for_pressure(pressure, baseline)`: extra ADP shift = 1.5 * max(0, pressure - baseline) where
  baseline is the position's share of ADP in that pick range.
* `expected_best_available(ranked: list[tuple[id, points, p_avail]]) -> float`:
  Σ points_i · p_i · Π_{j<i}(1 − p_j), plus tail term with the last player.

`recommend.py`
* `class Advisor(league, players, projections, settings)`; precomputes numpy arrays (ids, positions, points,
  std) so `recommend` is O(P log P):
  `recommend(state, top_n=6, per_position=3) -> Recommendation`:
  1. available = undrafted players with projections (filter to top ~350 by points + all with ADP < 250).
  2. replacement levels & VORP on the *available* set; tiers per position.
  3. my roster summary + opponent summaries; pressure per position until my next pick; `p_next`, `p_after`.
  4. per candidate: marginal lineup value (with bench discount), VONA = points − expected_best_available at
     position at my next pick, score = marginal_value + 0.5 * VONA_positive − risk_aversion * std + stack
     bonus (my QB's WR/TE or vice versa, `settings.stack_bonus * points`) − bye penalty (each starter sharing
     the bye, `settings.bye_penalty * points`) − K/DEF early penalty (before `rounds - kicker_def_min_round_from_end`:
     score −= 25 % of points), then round-context: if `p_next` >= 0.85 for a candidate the score is reduced by
     `0.3 * VONA_positive` ("you can get him later"), and if nothing at a position will be there next time,
     scarcity bonus.
  5. reasons: 2-4 short strings, e.g. "Proj 268 pts, +41 over replacement RB", "Tier 1 of 3 at TE",
     "Only 38% chance available at your next pick (#41)", "Fills your open RB2", "Bye 7 clashes with QB",
     "ADP 22 → value at pick 31", "Market (ECR 18) higher than model (28)".
     warnings: injury status, depth chart >= 2 with no ECR support, "K/DEF too early", rookie.
  6. `PositionAdvice.action`: "TAKE NOW" if best candidate `p_next` < 0.5 and drop_off > 0.6*std or need is
     open; "SOON" if `p_next` < 0.75; "WAIT" if `p_next` >= 0.75 and drop_off small; "SKIP" for K/DEF before
     the late rounds or when the position's starters are already full and bench value is low.
     rationale text includes the probability and my next pick number.
  7. `notes`: "RB run: 5 of last 6 picks", "You pick next at #41 and #44 (back-to-back)", "Superflex: QBs
     scarce", "K/DEF: wait until round 14".
  Must run in < 30 ms for 12 teams / 400 available players (assert in a test with `time.perf_counter`).
  `explain_pick(state, player_id) -> str`, `available_players(state) -> list[Player]`, `value_of(state, player_id) -> PlayerValue`.

`trade.py`
* `@dataclass TradeEvaluation(my_before, my_after, their_before, their_after, my_delta, their_delta, verdict,
  details: list[str])`; `evaluate_trade(my_ids, their_ids, give: list[id], get: list[id], players, projections,
  league, bench_discount=0.35) -> TradeEvaluation` using optimal lineups (starting points + discounted bench),
  verdict "ACCEPT" if my_delta > 3 and not clearly lopsided against them beyond +/-; "REJECT" if my_delta < -3;
  else "NEUTRAL". Details name the lineup slot changes and bye effects.
* `evaluate_pick_choice(state, advisor, player_id) -> str` (why/why not vs the top recommendation).

`simulate.py` (optional but wanted): `simulate_candidates(state, players, projections, advisor, candidate_ids,
n_sims=30, rounds_ahead=2, seed=0) -> dict[id, float]` Monte-Carlo: for each candidate, take him now, then
simulate the other teams' picks (sample by availability model) until my pick after next and greedily pick
my best marginal value there; return expected my-lineup points. Time-box: stop when > 250 ms elapsed and
return partial results (mark missing ids absent).

Tests: `tests/test_strategy.py` — replacement demand for the fixture league; optimal lineup with FLEX and
SUPER_FLEX; marginal value of a K before/after having one; availability monotone in pick distance and equals
1.0 at current pick; expected_best_available hand-computed on 3 players; Advisor on the fixture draft state
(20 picks made) returns 3 per position, best_overall sorted by score, no drafted players, and runs < 30 ms
on a synthetic 400-player universe; trade evaluation symmetric sanity.

### 3.5 `research/claude.py` — optional Claude layer  (owner: agent "research")

Model `claude-sonnet-5` (constant `CLAUDE_MODEL`, user asked for Sonnet). Use the official `anthropic` SDK
(1.x): `anthropic.AsyncAnthropic()`; never raw HTTP. Fully optional: if no API key (`Settings.anthropic_api_key`
or `ANTHROPIC_API_KEY`) the class reports `enabled == False` and every method returns `None`/`{}` instantly.
* `class ClaudeResearcher(api_key=None, model=CLAUDE_MODEL, cache_dir=None, client=None)`
  * `load_notes() -> dict[id, ResearchNote]` from `research_dir()/notes/<player_id>.json`.
  * `await research_players(players: list[Player], projections=None, max_age_days=3, concurrency=4,
    progress=None) -> dict[id, ResearchNote]`: one request per player (skip fresh cached notes), using the
    `web_search_20260209` server tool (`max_uses` 3) plus structured output (`output_config.format` JSON schema:
    summary (<= 60 words), injury_risk 0-1, role_certainty 0-1, upside (<= 25 words), downside (<= 25 words),
    sources (urls)). System prompt: fantasy-football analyst for the 2026 season; today's date passed in the
    user turn. `output_config={"effort": "medium"}`. Handle `stop_reason == "refusal"` and errors by caching a
    minimal note with `summary="(research unavailable)"`, `injury_risk=0`, `role_certainty=0.5` so the draft
    loop never blocks; semaphore for concurrency; write each note to disk as it completes.
  * `await on_the_clock_advice(state, rec, players, notes, timeout=CLAUDE_ON_CLOCK_TIMEOUT_S) -> str | None`:
    a single streaming request (no tools), `output_config={"effort": "low"}`, `max_tokens=400`, with a
    **stable cached system prompt** (`cache_control` ephemeral: league scoring description, roster slots,
    strategy guidance) and a compact user message (round/pick, my roster by slot, open needs, next picks,
    top 8 candidates with points/VORP/availability/tier/reasons, research notes for them, last 6 picks,
    position pressure). Return 2-4 sentences: the pick, why, the fallback. Enforce `asyncio.wait_for(timeout)`;
    return None on timeout/error. Cache by `(state.version, top candidate ids)` so repeated ticks reuse.
  * `await ask(question, context_text) -> str` free-form Q&A (effort medium, max_tokens 1500).
  * `build_context_text(state, rec, players, notes) -> str` helper used by both.
* Provide `FakeAnthropic` in tests via monkeypatching `client.messages.create/stream`; assert prompt content
  includes roster + candidates, timeout returns None, cache hit avoids second call, disabled mode.

### 3.6 `ui/dashboard.py`, `cli.py`, `app.py`  (owner: agent "ui")

`app.py` — orchestration shared by commands:
* `@dataclass AppContext(settings, engine, league, draft, players, projections, advisor, researcher, notes, byes, crosswalk, sources: dict)`
* `build_context(settings, league=None, draft=None, *, offline=False, sleeper_players=None, sleeper_proj=None,
  refresh=False, use_model=True, quiet=False) -> AppContext`:
  crosswalk -> players (Sleeper payload if given else offline roster universe) -> enrich (ECR, byes) -> ADP
  (Sleeper `adp_map` if projections given else ECR) -> ML inference table + model predict (if a saved model
  exists and `use_model`; else `project_offline`) -> `Projector.project` -> `Advisor`. Log timings. Cache the
  projections table to `cache_dir()/projections_<league_id or 'default'>.json.gz` so the draft command starts
  in < 3 s.
* `async prep(settings, research=False, refresh=False, train=True)`: download data (`load_canonical`),
  build crosswalk, train model (`projections.model.train_and_save`), fetch Sleeper players/projections if
  reachable (swallow network errors), build context, optionally run research for the top 200 by ADP.

`ui/dashboard.py` — `rich` `Live` layout (refresh 4/s):
* header: league name • scoring description • "Round 3 • Pick 31 • On the clock: Team X" • "You pick in 4
  (#35), then #46" • poll latency • Claude status.
* left column: **My roster** table by slot label (starter slots then bench) with player, pos, team, bye,
  projected points; needs line "Need: RB2, TE, K, DEF"; bye clash warning.
* center top: **Best picks now** (top 6): #, player, pos/team, bye, proj, VORP, tier, ADP, avail@next %, score,
  reason (first reason). Highlight row 1. Row for the recommended pick in bold green when it's my turn.
* center bottom: **By position** — six mini-tables (QB/RB/WR/TE/K/DEF), each with the action badge
  (TAKE NOW = red, SOON = yellow, WAIT = green, SKIP = dim) + rationale + 3 candidates (name, proj, avail %, tier).
* right column: **Recent picks** (last 10: pick#, team label, player, pos) and **Opponent needs** (slot label,
  open starters string like "RB WR TE", next pick #).
* footer: Claude advice text (or "Claude: off — set ANTHROPIC_API_KEY"), notes, key hint.
* When it is my turn the header flashes "YOUR PICK" and the pick timer countdown (from `draft.pick_timer` and
  the time we noticed the turn).
* Also `render_text(rec, state, players) -> str` plain text (used by `--no-tui` and `mock --auto` logs).

`cli.py` (argparse, entry `main(argv=None)`): subcommands
* `prep [--league ID | --draft ID] [--refresh] [--no-train] [--research] [--top N]` — downloads, trains, caches;
  prints backtest metrics and a top-30 projection table.
* `train [--seasons 2019-2025] [--refresh]`.
* `draft --draft ID [--league ID] [--username U | --user-id ID | --slot N] [--poll 2] [--no-claude] [--no-tui]`
  — live advisor; if `--draft` missing but `--league` given, resolve via `get_league_drafts`; if neither given
  but `--username` and season given, list the user's drafts and ask.
* `mock [--teams 12] [--rounds 15] [--slot 5] [--scoring half_ppr] [--superflex] [--auto] [--seed 1] [--speed 0.5]`
  — offline simulator with the same dashboard; in interactive mode when it's my turn read a line from stdin:
  empty = take the top recommendation, otherwise player name (fuzzy) or player_id; `--auto` always takes
  the top recommendation and prints the final roster + total projected points + a grade vs the other teams.
* `projections [--position RB] [--top 40] [--league ID]` — table of projections (points, std, ppg, games, ADP, ECR).
* `trade --league ID --me U --them V --give "Name, Name" --get "Name"` — evaluate.
* `analyze --league ID [--username U]` — post-draft roster strengths/weaknesses for every team, playoff-week
  byes, best waiver targets (best available by VORP).
* `research --top 200 [--league ID]` — run Claude research and print a summary.
* `ask "question" [--draft ID]` — free-form Claude question with context.
* `ids --username U [--season 2026]` — list the user's leagues and drafts with ids.
* Global: `--home DIR` (sets DRAFTADVISOR_HOME), `--offline`, `-v`.
Every command must work offline for the sandbox where sensible (`mock`, `projections`, `train`, `prep --offline`).

Tests: `tests/test_cli.py` (argparse parsing, `mock --auto --rounds 3 --teams 4` completes offline using the
fixture-free offline universe — mark slow if it needs `data/raw`), `tests/test_dashboard.py` (render to a
`Console(record=True)` and assert key strings appear), `tests/test_app.py` (`build_context(offline=True)` on
the fixture players with `use_model=False`).

## 4. Rules for implementers

* Own only the files listed for your module (+ your test files). Do not edit `models.py`, `config.py`,
  `scoring/`, `data/` — if you need a change there, describe it in your final report instead.
* Python 3.10 compatible (`from __future__ import annotations`, no `match` statements needed, no 3.12-only APIs).
* No new third-party dependencies beyond `pyproject.toml` (httpx, numpy, pandas, scikit-learn, rich, anthropic).
* Performance budgets: `Advisor.recommend` < 30 ms; dashboard render < 20 ms; poller tick overhead negligible;
  `build_context` from caches < 3 s; model training < 3 min.
* Never block the event loop on network I/O in the draft loop; Claude calls run as background tasks.
* Log with `logging.getLogger(__name__)`; no prints inside library code (CLI prints are fine).
* Tests must pass offline: `python -m pytest tests -q`. Use the fixtures in `tests/fixtures/`.
* Type hints everywhere; docstrings on public functions; keep functions small.

### 3.7 `capture.py` — one-time pre-draft league info capture  (owner: integrator)

When a league id (and/or draft id) is supplied, the tool performs a one-time **info capture** before the
draft and persists it under `home_dir()/leagues/<league_id>.json` (raw payloads + derived summary):

* league: name, season, status, number of teams, roster slots (starters / bench / IR / taxi), settings that
  affect strategy (best ball, max keepers, taxi slots, reserve slots, playoff teams & start week, trade
  deadline, waiver type, superflex / TE premium / 6-pt pass TD flags);
* scoring: full `scoring_settings` and a **diff against Sleeper's base scoring** (added keys, changed values,
  removed keys) with human-readable labels — so you know exactly how this league deviates from defaults;
* draft: type, rounds, pick timer, reversal round, start time, status, draft order (slot -> manager /
  team name / roster id), traded picks, keepers already on the board, and for the advised user: slot,
  roster id and the full list of their pick numbers;
* managers/rosters: user ids, display names, team names, roster ids, current rosters (dynasty/keeper).

API: `capture_league(client, league_id=None, draft_id=None, *, username=None, user_id=None, slot=None)
-> LeagueSnapshot` (async), `LeagueSnapshot.save()/load(league_id_or_draft_id)`, `LeagueSnapshot.report()`
(rich renderable) and `.to_text()`. `scoring_diff(scoring_settings) -> ScoringDiff`.
CLI: `draftadvisor capture --league ID [--draft ID] [--username U]`; `prep` and `draft` run the capture
automatically (draft refreshes it at bootstrap because the draft order can change until the draft starts)
and print the report once.

### 3.8 `web/` — local web app  (owner: integrator + "frontend" agent)

`python run.py` (or `draftadvisor web`) starts a FastAPI/uvicorn server on http://127.0.0.1:8787 and opens
the browser. The page is a single file `draftadvisor/web/static/index.html` (vanilla HTML/CSS/JS, no build
step) that polls `GET /api/state` every 2 s (1 s when it is my turn) and renders everything. The server
(`draftadvisor/web/server.py`) holds one `Session` (live or mock) and wraps the existing engine; all heavy
work runs in background tasks so the API always answers immediately.

Tabs: **Setup** (readiness, prepare data/model, live-draft form, mock-draft form), **Draft** (the advisor
dashboard), **Board** (projection table with search/filter/sort), **League** (capture report: scoring diff,
flags, draft order), **Ask** (Claude Q&A, only when a key is set).

API (all JSON; errors are `{"detail": "..."}` with 4xx/5xx):

* `GET /api/status` → `{"mode": "idle|prepping|starting|live|mock", "busy": bool, "message": str,
  "ready": {"data": bool, "model": bool, "claude": bool, "sleeper": bool|null}, "log": [str], "season": int,
  "home": str, "session": null | {"mode", "league_name", "draft_id", "my_slot", "started_at"}}`
* `POST /api/prep` `{"refresh": bool, "research": bool, "top": int}` → `{"ok": true}`; progress in `status.log`.
* `POST /api/lookup` `{"username": str}` → `{"user": {"user_id","display_name","username"},
  "leagues": [{"league_id","name","season","total_rosters","status","draft_id","scoring_type","roster_positions"}],
  "drafts": [{"draft_id","league_id","status","type","season","start_time","teams","rounds","name"}]}`
* `POST /api/live/start` `{"league_id"?, "draft_id"?, "username"?, "user_id"?, "slot"?, "use_claude"?: bool}`
  → `{"ok": true}` (background: capture → build_context → poller; `mode` becomes `live`).
* `POST /api/mock/start` `{"teams": 12, "rounds": 15, "slot": 5, "scoring": "half_ppr|ppr|std",
  "superflex": false, "seed": int|null, "bot_delay": 1.0, "autopilot": false}` → `{"ok": true}` (`mode` → `mock`).
  Bots pick automatically every `bot_delay` seconds until it is my turn; `POST /api/mock/pick {"player_id"}`
  makes my pick, `POST /api/mock/auto` takes the top recommendation, `POST /api/mock/autopilot {"enabled"}`.
* `POST /api/stop` → `{"ok": true}` (mode → `idle`).
* `GET /api/state` → the render payload:
  ```
  {"mode", "version", "ts",
   "draft": {"type","status","teams","rounds","pick_timer","current_round","next_pick_no","total_picks",
             "on_the_clock": {"slot", "label"}|null, "is_my_turn", "my_slot", "my_next_pick_no",
             "my_pick_after_next", "picks_until_my_turn", "is_complete", "turn_started_at", "seconds_left"},
   "league": {"name","scoring_type","scoring_description","roster_positions","teams","season"},
   "me": {"slots": [{"slot": "RB", "player": card|null}], "needs": [str], "bye_clashes": {"11": 2},
          "lineup_points", "bench_points", "position_counts": {pos: n}},
   "best": [card],                       // top 8 by score
   "by_position": {pos: {"action","rationale","expected_next_available","drop_off","candidates": [card x3]}},
   "available": [card],                  // top 250 undrafted by score (client filters/sorts)
   "recent": [{"pick_no","round","slot","label","player_id","name","position","team","is_me"}],  // newest first, 12
   "opponents": [{"slot","label","needs": [str],"next_pick","position_counts": {}, "players": [{"name","position"}]}],
   "pressure": {pos: float}, "notes": [str],
   "claude": {"status": "off|idle|thinking|ready|error|no answer", "advice": str|null},
   "snapshot": null | {"flags": [str], "diff": {"scoring_type","rec_points","deltas": [{"key","label","base","league","kind"}]},
                       "draft_order": [{"slot","display_name","team_name","roster_id","picks": [int],"is_me"}],
                       "my_picks": [int], "captured_at": float, "league": {...}, "draft": {...}},
   "status": {"latency_ms","compute_ms","poll_count","last_error","sources": {}}}
  ```
  A **card** is `{"player_id","name","position","team","bye","age","years_exp","injury_status","depth_chart_order",
  "points","floor","ceiling","std","ppg","games","vorp","vona","marginal","score","tier","pos_rank","overall_rank",
  "adp","ecr","availability_next","availability_after_next","reasons": [str],"warnings": [str],
  "note": null|{"summary","injury_risk","role_certainty","upside","downside"}, "drafted_by": null|label}`.
* `GET /api/player/{player_id}` → card + `{"projection": {"components","weights","flags","stat_line"}, "explain": str}`.
* `GET /api/projections?position=RB&top=80&q=text` → `[card]` from the session's (or a default offline)
  context sorted by points; `POST /api/board/load` builds the default context when there is no session.
* `POST /api/ask` `{"question"}` → `{"answer"}` (503 when Claude is disabled).
* `POST /api/research` `{"top": 150}` → `{"ok": true}` (background, progress in `status.log`).
