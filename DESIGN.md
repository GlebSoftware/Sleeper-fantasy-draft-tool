# draftadvisor — design & module contracts

Live draft advisor for Sleeper or ESPN leagues. Polls the draft every ~2 s, recomputes recommendations in
milliseconds, shows the best 3 players per position (and overall) with reasons, tells you
when you can wait on a position, models opponents' needs, and has an optional Claude chat
(user-initiated only: nothing calls Claude automatically). Also: mock drafts, trade/pick evaluation.

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
 Sleeper picks ──► sleeper/poller.py ──┐
 ESPN draftDetail ► espn/poller.py ─────┴► DraftState ──► strategy/recommend.py (Advisor) ──► Recommendation
 (espn/: ids -> universe, scoringItems -> scoring_settings, kona ADP / projections -> overrides; §3.10)
                                                          │  VORP, lineup marginal value,          │
                                                          │  availability @ my next pick,           ▼
                                                          │  opponent needs / position runs    ui/dashboard.py (rich Live)
                                                          └─► research/claude.py (chat / ask: user-initiated only) ─┘
```

Identity everywhere: **Sleeper player_id** (string). DEF ids are team abbreviations ("SF"). ESPN ids are
mapped onto these (`espn/ids.py`); an ESPN player the universe does not know gets a synthetic `espn:<id>`.

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
downloading. **The Sleeper API, the ESPN API and the Anthropic API are NOT reachable from the sandbox** —
everything that touches them must be testable with fixtures / mocks (see `tests/fixtures/*.json`, faithful to
the Sleeper API docs shapes, and `tests/fixtures/espn/*.json` + `tests/espn_stub.py`, a local HTTP stub of the
ESPN endpoints) and must degrade gracefully offline.

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

### 3.5 `research/claude.py` — Claude chat layer, user-initiated only  (owner: agent "research")

The only code that calls the Anthropic API, through the official `anthropic` SDK (1.x, `AsyncAnthropic`;
never raw HTTP). Exactly two entry points, both triggered by a person: the web chat (`POST /api/chat`) and
the CLI `ask` command. **Nothing calls Claude on a timer, on a poll, on a state change or in a loop** — the
former per-player research loop (web search + structured notes) and the automatic on-the-clock advice were
removed because they ran up a large bill. Fully optional: without a key (`Settings.anthropic_api_key`,
`ANTHROPIC_API_KEY` or the `X-Anthropic-Key` header) the class reports `enabled == False`, `chat_stream`
raises `RuntimeError` and `ask` returns `""`.
* `class ClaudeChat(api_key=None, model=CLAUDE_MODEL, cache_dir=None, client=None)` (`ClaudeResearcher` is a
  backwards-compatible alias of the same class).
  * `async chat_stream(messages, context_text, *, model=None, max_tokens=2000, system=None)`: one streaming
    request (`output_config={"effort": "medium"}`, no tools) with the browser-held transcript and the live
    draft context (`build_context_text`) injected into the last user turn; yields text deltas, then a final
    dict `{"done": True, "stop_reason", "model", "usage": {input_tokens, output_tokens,
    cache_read_input_tokens, cache_creation_input_tokens}, "cost_usd": float|None}`. Model:
    `DRAFTADVISOR_CHAT_MODEL`, default `CHAT_MODEL = "claude-opus-5"`.
  * `trim_transcript(messages, *, max_turns=CHAT_MAX_TURNS, max_chars=CHAT_MAX_CHARS) -> list` (40 turns /
    60 000 characters): the newest turns that fit, user-first, the last turn always kept; `POST /api/chat`
    applies it before every request so the history a client sends is never forwarded unbounded.
  * `async ask(question, context_text) -> str`: one request (effort medium, `max_tokens` 1500); `""` on
    error / refusal with `last_error` set; `last_usage` / `last_cost_usd` describe the request.
  * `load_notes() -> dict[id, ResearchNote]`: reads the notes an earlier version wrote under
    `research_dir()/notes/<player_id>.json`. Read-only legacy data: nothing writes notes any more and
    nothing depends on them existing.
* Cost: `MODEL_PRICES_USD_PER_MTOK = {"claude-opus-5": (5.0, 25.0), "claude-sonnet-5": (2.0, 10.0)}` (input,
  output per million tokens); `estimate_cost_usd(model, usage) -> float | None` bills `input_tokens` and
  `output_tokens` at list price, cache reads at 10 % and cache writes at 125 % of the input price, and
  returns `None` for a model outside the table. The web page shows the tokens and the estimate under every
  answer plus a running total per transcript; the CLI prints one line after `ask`.
* `build_context_text(state, rec, players, notes) -> str` (roster, needs, next picks, top-8 candidates with
  points/VORP/availability/tier/reasons and any legacy note, position outlook, last 6 picks, position
  pressure) is shared by both entry points; `describe_scoring` / `describe_roster_slots` are helpers.
* Prompt caching: no `cache_control` markers (both system prompts are far below the minimum cacheable prefix).
* Tests (`tests/test_research.py`, fake client): context text; `chat_stream` (context injection, merged
  turns, usage + cost in the done event, an assistant-last transcript is rejected before anything is sent);
  `ask`; `estimate_cost_usd`; disabled mode never sends; a guard that no research / advice code path exists.

### 3.6 `ui/dashboard.py`, `cli.py`, `app.py`  (owner: agent "ui")

`app.py` — orchestration shared by commands:
* `@dataclass AppContext(settings, engine, league, draft, players, projections, advisor, claude, notes, byes, crosswalk, sources: dict)`
  (`claude`: the `ClaudeChat` used by `ask` only; `notes`: legacy research notes read from disk, never written).
* `build_context(settings, league=None, draft=None, *, offline=False, sleeper_players=None, sleeper_proj=None,
  refresh=False, use_model=True, quiet=False) -> AppContext`:
  crosswalk -> players (Sleeper payload if given else offline roster universe) -> enrich (ECR, byes) -> ADP
  (Sleeper `adp_map` if projections given else ECR) -> ML inference table + model predict (if a saved model
  exists and `use_model`; else `project_offline`) -> `Projector.project` -> `Advisor`. Log timings. Cache the
  projections table to `cache_dir()/projections_<league_id or 'default'>.json.gz` so the draft command starts
  in < 3 s.
* `async prep(settings, refresh=False, train=True, progress=None, offline=False)`: download data
  (`load_canonical`), build crosswalk, train model (`projections.model.train_and_save`), fetch Sleeper
  players/projections if reachable (swallow network errors), build context. Never calls Claude.

`ui/dashboard.py` — `rich` `Live` layout (refresh 4/s):
* header: league name • scoring description • "Round 3 • Pick 31 • On the clock: Team X" • "You pick in 4
  (#35), then #46" • poll latency.
* left column: **My roster** table by slot label (starter slots then bench) with player, pos, team, bye,
  projected points; needs line "Need: RB2, TE, K, DEF"; bye clash warning.
* center top: **Best picks now** (top 6): #, player, pos/team, bye, proj, VORP, tier, ADP, avail@next %, score,
  reason (first reason). Highlight row 1. Row for the recommended pick in bold green when it's my turn.
* center bottom: **By position** — six mini-tables (QB/RB/WR/TE/K/DEF), each with the action badge
  (TAKE NOW = red, SOON = yellow, WAIT = green, SKIP = dim) + rationale + 3 candidates (name, proj, avail %, tier).
* right column: **Recent picks** (last 10: pick#, team label, player, pos) and **Opponent needs** (slot label,
  open starters string like "RB WR TE", next pick #).
* footer: engine notes (position runs, next pick), optional message, key hint. The dashboard never shows or
  requests Claude output; the draft loop (`cli.DraftLoop`) is recommend + render only.
* When it is my turn the header flashes "YOUR PICK" and the pick timer countdown (from `draft.pick_timer` and
  the time we noticed the turn).
* Also `render_text(rec, state, players) -> str` plain text (used by `--no-tui` and `mock --auto` logs).

`cli.py` (argparse, entry `main(argv=None)`): subcommands
* `prep [--league ID | --draft ID] [--refresh] [--no-train]` — downloads, trains, caches;
  prints backtest metrics and a top-30 projection table.
* `train [--seasons 2019-2025] [--refresh]`.
* `draft --draft ID [--league ID] [--username U | --user-id ID | --slot N] [--poll 2] [--no-tui]`
  — live advisor; if `--draft` missing but `--league` given, resolve via `get_league_drafts`; if neither given
  but `--username` and season given, list the user's drafts and ask. With `--platform espn --league ID
  [--season YYYY] [--team-id N | --slot N | --username "team or owner"] [--espn-s2 C --swid C]` the same loop
  (`DraftLoop`) consumes an `EspnDraftPoller` (§3.10); `--draft` is ignored, poll default 3 s.
* `mock [--teams 12] [--rounds 15] [--slot 5] [--scoring half_ppr] [--superflex] [--auto] [--seed 1] [--speed 0.5]`
  — offline simulator with the same dashboard; in interactive mode when it's my turn read a line from stdin:
  empty = take the top recommendation, otherwise player name (fuzzy) or player_id; `--auto` always takes
  the top recommendation and prints the final roster + total projected points + a grade vs the other teams.
* `projections [--position RB] [--top 40] [--league ID]` — table of projections (points, std, ppg, games, ADP, ECR).
* `trade --league ID --me U --them V --give "Name, Name" --get "Name"` — evaluate.
* `analyze --league ID [--username U]` — post-draft roster strengths/weaknesses for every team, playoff-week
  byes, best waiver targets (best available by VORP).
* `ask "question" [--draft ID]` — free-form Claude question with context: the only CLI command that calls
  Claude (one request); prints the answer, then one line with tokens and estimated cost.
* `ids --username U [--season 2026]` — list the user's leagues and drafts with ids; `ids --platform espn --swid
  {...}` asks ESPN's fan API (best effort) and otherwise prints where the id sits in the league URL.
* Every command naming a league takes `--platform sleeper|espn` (env `DRAFTADVISOR_PLATFORM`) plus
  `--espn-s2` / `--swid` (env `ESPN_S2` / `ESPN_SWID`); `app.build_context` / `capture` are platform-aware.
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
* Never block the event loop on network I/O in the draft loop. Nothing may call Claude automatically: the
  only Anthropic calls are `POST /api/chat` and `draftadvisor ask`, both user-initiated.
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
`resolve_identity(draft, managers, users_raw, *, username, user_id, slot) -> (my_user_id, my_slot)` is the pure
"who am I" step of the capture (raises `ValueError` naming the managers when nobody matches); the web server
calls it per request on its identity-free capture.
CLI: `draftadvisor capture --league ID [--draft ID] [--username U]`; `prep` and `draft` run the capture
automatically (draft refreshes it at bootstrap because the draft order can change until the draft starts)
and print the report once.

### 3.8 Render payload  (owner: integrator + "frontend" agent)

> The v1 stateful server this section once described (`/api/prep`, `/api/live/start`, `/api/ask`,
> `/api/research`, the "Ask" tab, a `claude` advice field) is gone; §3.9 is the API. Only the payload shape
> below is still shared.

The **render payload** returned by `GET /api/state` / `POST /api/mock/state` (§3.9):
  ```
  {"mode", "version", "ts",
   "draft": {"type","status","teams","rounds","pick_timer","current_round","next_pick_no","total_picks",
             "on_the_clock": {"slot", "label"}|null, "is_my_turn", "my_slot", "my_next_pick_no",
             "my_pick_after_next", "picks_until_my_turn", "is_complete", "turn_started_at", "seconds_left",
             "last_picked"},        // seconds_left is null for ESPN (no pick timestamps)
   "league": {"name","scoring_type","scoring_description","roster_positions","teams","season"},
   "me": {"slots": [{"slot": "RB", "player": card|null}], "needs": [str], "bye_clashes": {"11": 2},
          "lineup_points", "bench_points", "position_counts": {pos: n}},
   "best": [card],                       // top 8 by score
   "by_position": {pos: {"action","rationale","expected_next_available","drop_off","candidates": [card x3]}},
   "available": [card],                  // top 250 undrafted by score (client filters/sorts)
   "recent": [{"pick_no","round","slot","label","player_id","name","position","team","is_me"}],  // newest first, 12
   "opponents": [{"slot","label","needs": [str],"next_pick","position_counts": {}, "players": [{"name","position"}]}],
   "pressure": {pos: float}, "notes": [str],
   "snapshot": null | {"platform": "sleeper"|"espn", "flags": [str],
                       "diff": {"scoring_type","rec_points","deltas": [{"key","label","base","league","kind"}]},
                       "unmapped_scoring": [{"label","points","stat_id"}],   // ESPN rules not modelled; [] for Sleeper
                       "draft_order": [{"slot","display_name","team_name","roster_id","picks": [int],"is_me"}],
                       "my_picks": [int], "captured_at": float, "league": {...}, "draft": {...}},
   "status": {"latency_ms","compute_ms","ts","sources": {},"platform": "sleeper"|"espn"}}
  ```
  A **card** is `{"player_id","name","position","team","bye","age","years_exp","injury_status","depth_chart_order",
  "points","floor","ceiling","std","ppg","games","vorp","vona","marginal","score","tier","pos_rank","overall_rank",
  "adp","ecr","availability_next","availability_after_next","reasons": [str],"warnings": [str],
  "note": null|{"summary","injury_risk","role_certainty","upside","downside"}, "drafted_by": null|label}`.
* `GET /api/player/{player_id}` → card + `{"projection": {"components","weights","flags","stat_line"}, "explain": str}`.

### 3.9 Stateless web API v2 (local + Vercel)  (owner: integrator + "frontend-v2" agent)

The web app runs on Vercel (serverless: no background tasks, no persistent disk, no pandas/scikit-learn in the
function bundle) and locally with the same code. Therefore **the server is stateless**: the browser owns the
session config and, for mock drafts, the pick list; every request carries what the server needs; the server
keeps only warm-memory caches (bundle, Sleeper players/projections, ECR, per-league capture and context, one
`EspnClient` per cookie pair) and reads the legacy research notes (read-only) from Vercel Blob
(`BLOB_READ_WRITE_TOKEN`) or the local data dir.

Model outputs are precomputed into `web_bundle/` (`draftadvisor bundle`, committed): `players.json` (offline
universe), `ml.json` (per player predicted per-game rates, games, std, rookie flag), `season_totals.json`
(player-season stat totals 2019-2025 for the rank curves under any scoring), `byes.json`, `meta.json`.

Headers (all optional, all stored only in the browser's localStorage, never logged): `X-Access-Code` (required
for every `/api/*` call when env `DRAFTADVISOR_ACCESS_CODE` is set), `X-Anthropic-Key` (else env
`ANTHROPIC_API_KEY`), `X-ESPN-S2` / `X-ESPN-SWID` (ESPN cookies for private leagues; else env `ESPN_S2` /
`ESPN_SWID`). Claude is called by `POST /api/chat` only (chat model `claude-opus-5`, env
`DRAFTADVISOR_CHAT_MODEL`; `claude-sonnet-5` selectable in the page); nothing polls or researches.

**Session object** (sent by the browser as JSON body field `session`, or as query params for GET):
`{"mode": "live"|"mock", "platform": "sleeper"|"espn" (default sleeper),
  "draft_id", "league_id" (ESPN: the leagueId= number as a string), "season": int|null (ESPN; default
  DEFAULT_SEASON), "username", "user_id" (ESPN: str(team id) once resolved), "slot", "team_id": int|null (ESPN),
  "use_claude": bool,   // accepted for compatibility, ignored (chat is user-initiated)
  "mock": {"teams","rounds","slot","scoring","superflex","seed"}, "picks": [player_id,...]  // mock only }`
`Session.from_query` reads the same fields from GET query params (`/api/state`, `/api/player/{id}`,
`/api/projections`): the `session` JSON parameter when the page sends one (authoritative), else the flat
fields with the mock settings as `mock_<key>` (what the page sends) or bare `<key>`; malformed values are 400.
For ESPN the team id is the identity; `slot` / `username` only count when no team id is known
(`Session.espn_identity`).

Endpoints (JSON; errors `{"detail"}`):
* `GET /api/status` → `{"ok", "version", "season", "bundle": {"built_at","players","seasons"},
  "claude": {"server_key": bool, "chat_model": str, "prices": {model: [usd_per_M_input, usd_per_M_output]}},
  "notes": {"count", "store": "blob"|"local"|"memory"}, "access_code_required": bool,
  "platforms": ["sleeper", "espn"], "espn": {"server_cookies": bool}}`
* `POST /api/lookup {"username"}` (Sleeper) → `{"user", "leagues", "drafts"}`.
  `POST /api/lookup {"platform": "espn", "league_id", "season"}` → `{"platform": "espn", "league": {"league_id",
  "name", "season", "teams", "scoring_type", "is_public", "draft": {"type", "status", "pick_timer", "start_time",
  "rounds", "order_known"}}, "teams": [{"team_id", "name", "abbrev", "owners": [names], "slot": int|null,
  "is_me"}], "me": {"team_id", "slot"}|null (the team the SWID owns), "unmapped_scoring": [{"label", "points"}]}`.
  ESPN errors: 401/403 → HTTP 401 "ESPN says this league is private - add your espn_s2 and SWID cookies under
  Settings", 404 → 404 "ESPN has no league <id> for season <season>", anything else → 502.
* `POST /api/session/start {session}` → validates, captures the league (identity-free and shared by every
  session of the league: scoring diff, flags, draft order; "me" — user / team id, slot, picks — is resolved per
  request from that capture by `resolve_me`, 400 for an unknown team / manager or a slot outside the draft),
  warms the league context; returns `{"session": {...resolved ids/slot; for ESPN also
  platform, season, team_id, user_id = str(team id)...}, "snapshot": {...}, "league": {"name", "scoring_type",
  "teams"}, "draft_order_known": bool, "message": str|null}` (the message says "spectating" or "draft order not
  set yet"). For mock: builds the league/draft and returns the same shape (snapshot synthesised).
* `GET /api/state?...session params...` (live) / `POST /api/mock/state {session, action, player_id?}` (mock)
  → the render payload of §3.8 (draft, league, me, best, by_position, available, recent, opponents, pressure,
  notes, snapshot, status); there is no Claude field (nothing is generated automatically). For ESPN
  `draft.pick_timer` is `timePerSelection` re-read every poll, `draft.seconds_left` is null and
  `status.platform` is `"espn"`. Mock `action` ∈ `"sync"` (just
  render), `"advance"` (bots pick until my turn or the end), `"pick"` (my pick then advance), `"auto"` (take
  the top recommendation then advance); the response includes `"picks"` (the full list the browser must keep)
  and `"last_picks"` (picks made in this call, for the feed).
* `POST /api/chat {session, messages: [{"role","content"}], model?}` → **SSE stream** (`text/event-stream`,
  events `data: {"delta": "..."}` … `data: {"done": true, "stop_reason", "model", "usage": {"input_tokens",
  "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"}, "cost_usd": float|null}`) — the
  **only endpoint that calls the Anthropic API**: one request per user message, with the live context (the
  current recommendation, roster, needs, candidates, notes) injected server-side; the browser keeps the
  transcript (localStorage), sends the whole history each time and shows the tokens / estimated cost of every
  answer plus a running total. 503 without a key. There is no advice endpoint and no research endpoint.
  `model` must be a priced model (`claude.prices` of `/api/status`): anything else is 400 before a request goes
  out, and an unpriced `DRAFTADVISOR_CHAT_MODEL` falls back to the default. The transcript is trimmed
  server-side to the newest `CHAT_MAX_TURNS` = 40 turns / `CHAT_MAX_CHARS` = 60 000 characters
  (`research.claude.trim_transcript`; a single message above the cap is 413), so one request never carries an
  unbounded history.
* `GET /api/notes` → `{player_id: note}` (legacy research notes, read-only); `GET /api/player/{id}?...` → card +
  projection detail + note.
* `GET /api/projections?...session params...&position=&top=&q=` → board (the league's scoring when a session is
  given — Sleeper or ESPN — else half-PPR).

Legacy research notes (written by earlier versions; read-only, nothing writes new ones): `summary,
injury_risk (0-1), role_certainty (0-1), offfield_risk (0-1), red_flags [str], upside, downside, sources [url],
generated_at, model`. Where a note exists the Projector still uses `injury_risk` (games), `role_certainty`
(std) and `offfield_risk` (games and std), and the cards still show `red_flags`.

Timer: for Sleeper the countdown shows only what Sleeper reports **right now** (`draft.pick_timer` re-read every
poll and `last_picked`); if the commissioner changes the clock mid-draft the display follows. For ESPN the server
reports `pick_timer` and no `seconds_left`; the page derives an approximate countdown from the moment it saw
`next_pick_no` change and labels it as such. Nothing in the advice logic depends on the timer.

Caching (`web/server.py`): `CACHE` entries `league:sleeper:<draft_id>::` / `league:sleeper::<league_id>:` (a
capture is stored under both once it knows both ids) and `league:espn::<league_id>:<season>:<credential hash>`
hold the identity-free `LeagueBundle` (capture) for `LEAGUE_TTL` = 600 s — nobody is "me" in a capture;
`resolve_me` resolves the asking manager per request, so the resolved session the browser sends back after
`/api/session/start` hits the same entry and the first poll costs one GET, never a second capture. `ctx:...`
holds the `LeanContext` for `CONTEXT_TTL` = 600 s or until its fingerprint changes; for ESPN it carries
`ctx.id_map`, the ESPN id map indexed over the very universe the context was built from (Sleeper payload or
bundle), which every poll resolves picks / rosters through — so a drafted player can never stay "available"
under another id when the universe source changes between capture and poll. `espn_idmap:sleeper|bundle` is
only the map the capture's snapshot used. `_ESPN_CLIENTS` keeps one `EspnClient` per sha1 of the cookie pair
(max 16; the least recently used is evicted first and closed only once its in-flight requests finish, never
underneath one). Cookie values never appear in keys, logs or responses.

### 3.10 ESPN provider  (owner: agent "espn")

`draftadvisor/espn/` mirrors `sleeper/` and is pandas-free (it runs in the Vercel function). Modules:

| module | contract |
|---|---|
| `constants.py` | `espn_base_url()` (`DRAFTADVISOR_ESPN_BASE`, default `https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl`), `fan_api_url()` (`DRAFTADVISOR_ESPN_FAN_BASE`), `POSITION_SLOT_MAP` (lineup slot id → our roster label), `PRO_TEAM_MAP` / `TEAM_TO_PRO_ID`, `DST_ID_BASE` (−16000), `DEFAULT_POSITION_MAP`, `ELIGIBLE_SLOT_POSITION`, `INJURY_STATUS_MAP`, `ESPN_STAT_TO_SLEEPER` (statId → Sleeper keys + divisor), `SCORING_LABELS` / `label_for_stat` (espn-api's labels for the "not modelled" list). |
| `client.py` | `EspnClient(espn_s2, swid, timeout, retries, base_url, transport, http)`: `get_league(id, season, views)` (`view=` repeated; on 401/403 retries `/leagueHistory/{id}?seasonId=`), `get_settings_and_teams` (mSettings + mTeam + mRoster), `get_draft_detail` (mDraftDetail + mSettings: the per-poll call), `get_players` (kona_player_info with the `x-fantasy-filter` header from `player_filter(season, limit=600, "PPR")`), `get_fan_leagues(swid)` (best effort, `[]` on any failure). Retries with backoff on timeouts / 5xx / 429, never on 4xx. `EspnAPIError(status_code, url)`, `EspnNotFound` (404), `EspnAccessDenied` (401/403; the message never contains cookie values). `format_swid` adds the braces. |
| `ids.py` | `EspnIdMap.from_players(universe)`: `resolve(espn_id, name=, position=, pro_team_id=)` → canonical id by (1) `Player.espn_id`, (2) D/ST id `-16000 - proTeamId` → team abbreviation (WSH → WAS), (3) normalised name + position (+ team for ambiguous names; learned for next time), (4) synthetic `espn:<id>`. `espn_player_fields(json)` normalises a kona / roster / player dict; `placeholder_player(json)` builds a `Player` (metadata `platform: espn, placeholder: true`) so an unknown pick is never blank. |
| `scoring.py` | `espn_scoring_to_sleeper(scoringItems) -> (scoring_settings, unmapped)`, `espn_stats_to_sleeper(stats, position)` (projected line keyed by statId strings → Sleeper keys, incl. `gp`), `projected_season_stats(player_json, season)` (the `10<season>` / `statSourceId 1` / `statSplitTypeId 0` entry), `projected_points`. |
| `parsing.py` | `parse_espn_league(league_json, draft_json) -> LeagueSettings` (`settings`: `platform`, `keeper_count`, `is_public`, `draft_type`, `unmapped_scoring`, `scoring_type_espn`, `num_teams`), `parse_espn_draft -> DraftSettings`, `parse_espn_managers -> dict[team id str, Manager]`, `parse_espn_picks(draft_json, id_map, draft, names) -> list[Pick]`, `resolve_my_team(draft, managers, league_json, swid=, team_id=, slot=, username=) -> (team id str|None, slot|None)` (precedence slot > team_id > swid > username; `(None, None)` = spectator), `roster_names`, `rostered_ids`, `rounds_from_counts`, `roster_positions_from_counts`, `draft_status`, `state_from_espn(league_json, draft_json, id_map, *, names, swid, team_id, slot, username) -> DraftState`. |
| `capture.py` | `capture_espn_league(client, league_id, season, *, swid, team_id, slot, username, id_map, players_limit, save) -> LeagueSnapshot` (3 GETs; the player pool is optional — a failure only drops ADP / projections); `espn_names`, `espn_adp` (canonical id → `ownership.averageDraftPosition`, else the PPR / STANDARD draft rank), `espn_projections` (canonical id → Sleeper-key season line), `espn_rosters` (`[{"roster_id": team id, "players": [ids]}]` so `LeagueSnapshot.roster_players` works), `unmapped_flags`. |
| `poller.py` | `EspnDraftPoller(client, league_id, season, *, id_map, names, swid, team_id, slot, username, poll_seconds=3, league_json, settings)` with the `DraftPoller` interface (`bootstrap`, `poll_once`, `run(on_update, stop, on_error)`, `interval`, `state`, `last_error`, `poll_count`, `last_latency_ms`, `last_poll_at`). One GET per poll; a new state only when the pick signature or the order signature (`pickOrder`, `timePerSelection`, `type`, `date`, rounds) changed — an order change re-parses the whole state including "my" slot. Cadence 3 s / 1 s near my turn / 10 s pre-draft; exponential backoff to 30 s on errors; `EspnNotFound` / `EspnAccessDenied` during bootstrap are re-raised. |

**Sources per field** (ESPN v3 league payload; views in brackets):

| ours | ESPN |
|---|---|
| `LeagueSettings.league_id / season / name / total_rosters` | `id`, `seasonId`, `settings.name`, `settings.size` [mSettings] |
| `roster_positions` | `settings.rosterSettings.lineupSlotCounts` via `POSITION_SLOT_MAP` (0 QB, 1 TQB → QB, 2 RB, 3 → WRRB_FLEX, 4 WR, 5 → REC_FLEX, 6 TE, 7 OP → SUPER_FLEX, 8/9/11 → DL, 10 LB, 12/13/14 → DB, 15 → IDP_FLEX, 16 → DEF, 17 K, 20 → BN, 21 → IR, 23 → FLEX, 24/25 → BN; 18 P and 19 HC skipped) |
| `scoring_settings` + `settings.unmapped_scoring` | `settings.scoringSettings.scoringItems[]` (`statId`, `points`, `pointsOverrides`) |
| `DraftSettings.type / status / teams / rounds` | `draftSettings.type` (AUCTION → auction, else snake; `reversal_round` 0), `draftDetail.drafted` → complete / `inProgress` → drafting / else pre_draft, `len(pickOrder)` or `settings.size`, sum of `lineupSlotCounts` except slot 21 |
| `pick_timer / start_time / draft_order / slot_to_roster_id` | `draftSettings.timePerSelection` (s), `draftSettings.date` (ms), `pickOrder` (team ids; slot 1 = first; empty until set) |
| `Manager` | `teams[]` (`id`, `abbrev`, `name` or `location`+`nickname`, `owners`, `primaryOwner`) + `members[]` (`id` = SWID, `displayName`, first/last name); `user_id` = team id as str, `roster_id` = team id |
| `Pick` | `draftDetail.picks[]` (`overallPickNumber`, `roundId`, `teamId`, `playerId`, `keeper`); names from the rosters / player pool; used whatever `drafted` says |
| players / ADP / projections | `kona_player_info` `players[]` (`player.id`, `fullName`, `defaultPositionId`, `eligibleSlots`, `proTeamId`, `injuryStatus`, `ownership.averageDraftPosition`, `draftRanksByRankType`, `stats[]` with `id == "10<season>"`) |

**Scoring conversion** (`ESPN_STAT_TO_SLEEPER`; points per unit keep their semantics): 3 pass_yd, 4 pass_td,
20 pass_int, 19 pass_2pt, 17/18 bonus_pass_yd_300/400, 64 pass_sack, 211 pass_fd; 23 rush_att, 24 rush_yd,
25 rush_td, 26 rush_2pt, 37/38 bonus_rush_yd_100/200, 212 rush_fd; 53 (else 41) rec, 42 rec_yd, 43 rec_td,
44 rec_2pt, 58 rec_tgt, 56/57 bonus_rec_yd_100/200, 213 rec_fd; 68 fum, 72 fum_lost, 63 fum_rec_td; kicking
83 fgm, 84 fga, 85 fgmiss, 80 → fgm_0_19/20_29/30_39, 77 fgm_40_49, 74 fgm_50p (+ 50_59 / 60p unless 198 / 201
present), 82/79/76 fgmiss brackets, 86 xpm, 87 xpa, 88 xpmiss, 214 fgm_yds; defense 94 (+93/103/104) def_td,
101/102 st_td, 99 sack, 95 int, 96 fum_rec, 106 ff, 98 safe, 97 blk_kick, 113 def_pass_def, 205/206 def_2pt,
120 pts_allow, 89/90/91 pts_allow_0/1_6/7_13, 92+121 pts_allow_14_20, 122 pts_allow_21_27, 123 pts_allow_28_34,
124/125 pts_allow_35p, 127 yds_allow and 128–136 the yds_allow brackets. Rules: "every N yards / receptions"
items (5–14, 27–34, 47–52, 54–55, 116–119, 217–222) add `points / N` to the per-unit key; D/ST items (89–136,
187–197) take `pointsOverrides["16"]` when present; `pointsOverrides` for slot 2 / 4 / 6 on the reception item
become `bonus_rec_rb / wr / te` = override − base; statId 62 feeds `*_2pt` only where 19 / 26 / 44 are absent;
several items feeding one key take their mean; anything else with non-zero points (15/16 long-TD bonuses,
175–186 per-distance TD bonuses, 210 games played, 73 turnovers, ...) lands in `unmapped` as `{statId, label,
points}`, is surfaced as `settings.unmapped_scoring`, one strategy flag each ("ESPN rule not modelled: ..."),
`snapshot.unmapped_scoring` in the API and the League tab's list. `scoring_diff` still compares against
Sleeper's base scoring after the translation.

**Web integration** (`web/server.py`): `Session.platform == "espn"` → `resolve_league` runs `_capture_espn`
(identity-free; cache key `league:espn::<league_id>:<season>:<credential hash>`) → `LeagueBundle(platform="espn",
season, id_map, espn_players, espn_names, snapshot)`; `resolve_me` → `resolve_my_team` gives "me" per request;
`league_context` builds the universe (`lean.lean_universe`) and `EspnIdMap.from_players` over it, passes
`espn_context_inputs(lb, id_map)` = `adp_override` (source `"espn"`), `proj_fallback` (source `"espn"`) and
`extra_players` (placeholders for pool / roster players that resolved to `espn:<id>`) plus that universe and map
to `lean.build_lean_context` (stored as `ctx.id_map`); `live_state` → `espn_live_state` = one `get_draft_detail`
+ `state_from_espn` with the cached league JSON and `ctx.id_map`; `build_payload` adds placeholder players for
unknown picked ids and reports no countdown. The CLI path is `app.fetch_espn_league` / `app.espn_overrides` /
`app.build_context(espn_players=...)` and `cli.DraftLoop` over `EspnDraftPoller`; the TUI shows "ESPN clock:
N s per pick" plus an approximate countdown from the moment the process saw the pick number move.

**Caveats.** ESPN picks carry no timestamps (no authoritative clock). Whether `draftDetail.picks` updates while
a draft is in progress could not be verified from the sandbox: the code renders whatever ESPN returns each poll
and the docs say so. ESPN has no linear order and no "paused" status; auction drafts parse as `type
"auction"` and are not advised. The fan API used by `ids` is undocumented (parsed defensively, `[]` when the
shape is unknown). Cookie values never reach logs, cache keys, error messages or responses.

Tests: `tests/test_espn_client.py` (`httpx.MockTransport` + the stub: views, cookies, `x-fantasy-filter`, retries,
401 → history fallback → `EspnAccessDenied`, 404, base URL override), `tests/test_espn_parsing.py` (fixtures in `tests/fixtures/espn/`: league / draft in
each status, managers, picks, `resolve_my_team` precedence, id mapping), `tests/test_espn_scoring.py`
(conversion table, overrides, premiums, unmapped list, projected line scoring), `tests/test_espn_capture_poller.py`
(capture shape, poller change detection and cadence), `tests/test_web_espn.py` (the server against
`tests/espn_stub.py`, a local HTTP stub selected through `DRAFTADVISOR_ESPN_BASE`: lookup, session start, state,
header / env cookies, error mapping).
