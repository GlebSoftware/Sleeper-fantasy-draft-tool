#!/usr/bin/env python3
"""Build web_bundle/ for the stateless web app (local + Vercel).

Needs the full dependency set (pandas, scikit-learn) and data/raw (run `draftadvisor prep` first).
Outputs (JSON): players.json, ml.json, season_totals.json, byes.json, inseason.json, meta.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "web_bundle"))
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--seasons", default="2019-2025", help="historical seasons for the rank curves")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    import numpy as np
    import pandas as pd

    from draftadvisor.config import DEFAULT_SEASON
    from draftadvisor.data import fantasypros as fp
    from draftadvisor.data import nflverse as nv
    from draftadvisor.data.crosswalk import build_crosswalk
    from draftadvisor.data.universe import enrich_players, players_from_crosswalk
    from draftadvisor.lean import player_to_dict
    from draftadvisor.projections import ProjectionModel, build_inference_table
    from draftadvisor.projections.features import ALL_TARGET_KEYS
    from draftadvisor.projections.model import load_training_inputs
    from draftadvisor.scoring.engine import SLEEPER_STAT_KEYS

    season = args.season or DEFAULT_SEASON
    lo, hi = (int(x) for x in args.seasons.split("-"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("• universe (crosswalk + roster + ECR)")
    cw = build_crosswalk(season)
    roster = nv.load_roster(season)
    players = players_from_crosswalk(cw, roster, season)
    raw = fp.load_ecr_raw()
    enrich_players(players, cw, fp.overall_ecr(raw), nv.bye_weeks(season), fp.positional_ecr(raw))
    byes = nv.bye_weeks(season)
    (out / "players.json").write_text(json.dumps([player_to_dict(p) for p in players.values()]), encoding="utf-8")
    (out / "byes.json").write_text(json.dumps(byes), encoding="utf-8")
    print(f"  {len(players)} players, {len(byes)} byes")
    espn_n = sum(1 for p in players.values() if p.espn_id)
    espn_pos = {pos: (sum(1 for p in players.values() if p.position == pos and p.espn_id),
                      sum(1 for p in players.values() if p.position == pos))
                for pos in sorted({p.position for p in players.values()})}
    print(f"  espn ids: {espn_n} of {len(players)} players "
          f"({', '.join(f'{pos} {n}/{m}' for pos, (n, m) in espn_pos.items())})")

    print("• model predictions")
    agg, team_ctx, rosters = load_training_inputs(range(lo, hi + 1))
    table = build_inference_table(agg, team_ctx, players, cw, season)
    model = ProjectionModel.load()
    pred = model.predict(table)
    ml: dict[str, dict] = {}
    keep = [c for c in pred.columns if c.startswith("pred_")] + ["ppg_std_ppr", "rookie_flag", "position"]
    for pid, row in pred.iterrows():
        d = {}
        for c in keep:
            v = row.get(c)
            if isinstance(v, str):
                d[c] = v
            elif v is not None and np.isfinite(v):
                d[c] = round(float(v), 4)
        ml[str(pid)] = d
    (out / "ml.json").write_text(json.dumps(ml), encoding="utf-8")
    print(f"  {len(ml)} players with ML rows ({len(ALL_TARGET_KEYS)} target keys)")

    print("• season stat totals for the rank curves")
    can = nv.load_canonical(range(lo, hi + 1))
    keys = [k for k in SLEEPER_STAT_KEYS if k in can.columns]
    grp = can.groupby(["player_id", "season", "position"], as_index=False)
    totals = grp[keys].sum()
    games = grp.size().rename(columns={"size": "games"})
    totals = totals.merge(games, on=["player_id", "season", "position"])
    rows = []
    for r in totals.itertuples(index=False):
        d = r._asdict()
        stats = {k: round(float(d[k]), 2) for k in keys if d.get(k) and float(d[k]) != 0.0}
        rows.append({"player_id": d["player_id"], "season": int(d["season"]), "position": d["position"],
                     "games": int(d["games"]), "stats": stats})
    (out / "season_totals.json").write_text(json.dumps(rows), encoding="utf-8")
    print(f"  {len(rows)} player-seasons")

    print("• in-season tables (weekly spread, schedule, defence vs position)")
    from draftadvisor.data.inseason import build_inseason_tables

    tables = build_inseason_tables(can, nv.load_schedule(), season)
    (out / "inseason.json").write_text(json.dumps(tables), encoding="utf-8")
    print(f"  weekly spread for {len(tables['sigma']['players'])} players, "
          f"{len(tables['schedule'])} team schedules, defence-vs-position from {tables['dvp_season']}")

    meta = {"built_at": time.time(), "season": season, "seasons": f"{lo}-{hi}", "players": len(players),
            "espn_ids": espn_n, "ml_rows": len(ml), "season_totals": len(rows),
            "inseason": {"sigma_players": len(tables["sigma"]["players"]), "dvp_season": tables["dvp_season"],
                         "schedule_teams": len(tables["schedule"])}}
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    sizes = {p.name: round(p.stat().st_size / 1e6, 2) for p in out.glob("*.json")}
    print(f"done in {time.time() - t0:.0f}s: {sizes} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
