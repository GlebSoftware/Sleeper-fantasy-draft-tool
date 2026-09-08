# Rules for working in this repository

These rules exist because a previous version of this app ran a per-player Claude web-search loop with
no spend cap, on Vercel, with its "already done" cache on the function's throwaway `/tmp` disk. It
re-researched the same players again and again and cost the owner about $180 in minutes. Nothing
like that may ship again. Read these before changing anything.

## 1. Anything metered is treated as money, not as a feature

A metered call is any call billed per use: LLM APIs (Anthropic, OpenAI, ...), paid search, SMS,
email, maps, paid data feeds, serverless compute you do not control.

* Never add a metered call that runs on a timer, on a poll, on a state change, in a loop, or in a
  background task. Only a direct user action (a button press, a CLI command) may trigger one, and
  one action triggers one call.
* Before shipping any metered call: measure one real call (tokens, seconds, dollars) and write the
  number in the PR description and the README. Estimates from reading the code do not count.
* Any batch of metered calls (N players, N documents, N rows) needs, in this order: a durable
  "already done" record (see section 2), a hard ceiling on count and dollars that the server
  enforces, a visible estimate ("150 players, about $45") the user must confirm, and a stop button.
  Missing any one of these means the batch does not ship.
* Show the tokens and the estimated cost of every metered call in the UI or CLI output, and keep a
  running total. The price table lives in one place (`draftadvisor/research/claude.py`,
  `MODEL_PRICES_USD_PER_MTOK`); update it when models change.
* Server-side tools that inject content into the context (web search, web fetch, file search)
  multiply input tokens by 10x to 50x per call. Treat them as a separate cost line, never as a flag.
* Timeouts do not refund. A request that is cancelled after 90 s has already spent 90 s of tokens.
  Never build a retry-on-timeout loop around a metered call.
* Chat: models are allowlisted, transcript length is capped, and the user-facing cost is shown per
  message. Keep all three when touching `/api/chat`.

## 2. Persistence: the expensive thing is the durable thing

* Vercel functions (and every serverless runtime) have no shared or persistent disk. `/tmp` is per
  instance and wiped on cold start. Anything written there is gone. Design as if it never existed.
* If a feature pays to produce data (research notes, model outputs, scraped pages), that data goes
  to durable storage (Vercel Blob, a database, the git repo, the user's browser) before the feature
  is considered working. "Optional persistence" is forbidden: if the durable store is not
  configured, the feature refuses to run and says why. It never degrades silently to ephemeral
  storage.
* Idempotency keys ("this player was researched on this date") live in the same durable store as
  the data, never in process memory or local disk.
* The user's browser (`localStorage`) is a legitimate durable store for per-user state (session,
  picks, chat transcript, cookies, API key). It is not a store for anything that cost money to make
  and that another device or a reload of a different browser should see.

## 3. Verification that counts

* Tests that mock a paid API prove control flow, not cost. Every feature that calls a paid API also
  gets one documented real run with the measured cost before it is merged.
* Tests must be hermetic: no live network, no env vars leaking from the shell
  (`tests/conftest.py` clears them). Anything that needs a remote service uses a fixture or a local
  stub (`tests/espn_stub.py`, `tests/fixtures/`).
* Reproduce the target platform's constraints in tests: for Vercel that means the lean import path
  (no pandas / scikit-learn at module top level in `lean.py`, `web/`, `espn/`, `sleeper/`,
  `capture.py`, `models.py`, `research/`), bounded per-request work, no background tasks, and no
  reliance on local disk between requests.
* Before pushing: `python3 -m pytest -q` (all green), `node --check` on the extracted page script,
  and the Playwright scripts for the mock flow and the ESPN stub flow. After pushing: confirm the
  Vercel build is READY and probe `/api/status` on the production URL.
* Every review must include a "spend and persistence" dimension: grep for every call site of the
  paid client, confirm each one is behind a user action, and confirm every durable-looking cache is
  actually durable on the deployment target.

## 4. Scope discipline

* Build what was asked at the size that was asked. "Maybe do some research" is not a 150-player
  default sweep with a resume loop. When a feature multiplies cost, ship the smallest version
  (one player on demand) and let the owner ask for more.
* When a design has a cost or data-loss consequence, say it in one sentence at the top of the
  summary, not in an open-issues list at the bottom.

## 5. Deployment facts for this repo

* Production: https://sleeper-draft-advisor.vercel.app (FastAPI on Vercel, Python 3.12, entry
  `api/index.py`). Preview deployments require Vercel SSO; production is public unless
  `DRAFTADVISOR_ACCESS_CODE` is set. Do not put a server-side `ANTHROPIC_API_KEY` on a public
  deployment without an access code.
* The only code paths that call the Anthropic API are `POST /api/chat` (page Send button) and the
  CLI `draftadvisor ask`. `tests/test_research.py` asserts the research and advice paths are gone;
  keep that test.
* The Anthropic Console monthly spend limit is the owner's backstop. It is not a reason to skip any
  rule above.
* Sleeper and ESPN APIs are unreachable from the Claude Code sandbox; all platform tests use
  fixtures under `tests/fixtures/` and the stub server.
