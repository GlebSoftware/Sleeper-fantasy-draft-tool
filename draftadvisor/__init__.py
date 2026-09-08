"""draftadvisor - live fantasy draft advisor for Sleeper or ESPN leagues.

Layers (see DESIGN.md):
  sleeper/      - Sleeper API client, draft poller, JSON -> models parsing
  espn/         - ESPN API client, id mapping, scoring translation, capture, draft poller (DESIGN.md 3.10)
  data/         - historical data (nflverse), consensus rankings (FantasyPros via dynastyprocess),
                  id crosswalks, disk cache, offline player universe
  scoring/      - league scoring_settings -> fantasy points for any stat line
  projections/  - ML model (features/train/predict), projection blending
  strategy/     - replacement levels, roster needs, availability, recommendations, trades
  research/     - optional Claude chat / ask (user-initiated only) + read-only legacy notes
  web/          - stateless FastAPI server + single-file browser front end
  ui/           - rich terminal dashboard
  mock/         - offline mock-draft simulator
  cli.py        - command line entry point
"""

__version__ = "0.1.0"
