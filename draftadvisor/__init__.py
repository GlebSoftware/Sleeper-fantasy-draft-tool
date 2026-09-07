"""draftadvisor - live Sleeper fantasy draft advisor.

Layers (see DESIGN.md):
  sleeper/      - Sleeper API client, draft poller, JSON -> models parsing
  data/         - historical data (nflverse), consensus rankings (FantasyPros via dynastyprocess),
                  id crosswalks, disk cache, offline player universe
  scoring/      - league scoring_settings -> fantasy points for any stat line
  projections/  - ML model (features/train/predict), projection blending
  strategy/     - replacement levels, roster needs, availability, recommendations, trades
  research/     - optional Claude (Sonnet) research + on-the-clock advice
  ui/           - rich terminal dashboard
  mock/         - offline mock-draft simulator
  cli.py        - command line entry point
"""

__version__ = "0.1.0"
