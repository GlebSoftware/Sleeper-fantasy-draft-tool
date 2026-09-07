#!/usr/bin/env python
"""Train the projection model on the local nflverse data and print backtest metrics.

Usage::

    python scripts/train_model.py [--seasons 2019-2025] [--refresh] [--no-backtest] [--out PATH]

Writes ``data/models/projection_model.pkl`` (or ``$DRAFTADVISOR_HOME/models/...``).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from draftadvisor.config import DEFAULT_SEASON, SKILL_POSITIONS, TRAIN_SEASONS
from draftadvisor.projections.model import train_and_save


def _parse_seasons(text: str | None) -> list[int]:
    if not text:
        return list(TRAIN_SEASONS)
    if "-" in text:
        lo, hi = text.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in text.split(",") if x.strip()]


def format_metrics(metrics: dict) -> str:
    """Human-readable per-position table of the backtest metrics."""
    lines = [f"{'pos':<4}{'n':>5}{'MAE model':>11}{'MAE last':>10}{'MAE career':>12}{'Sp model':>10}{'Sp last':>9}{'Sp career':>11}  beats last"]
    for pos in SKILL_POSITIONS:
        r = metrics.get(pos)
        if not r or "mae_model" not in r:
            continue
        lines.append(
            f"{pos:<4}{r['n']:>5}{r['mae_model']:>11.3f}{r['mae_last']:>10.3f}{r['mae_career']:>12.3f}"
            f"{r['spearman_model']:>10.3f}{r['spearman_last']:>9.3f}{r['spearman_career']:>11.3f}  {'yes' if r.get('beats_last') else 'NO'}"
        )
    extra = {k: v for k, v in metrics.items() if k.startswith("_")}
    if extra:
        lines.append("")
        for k, v in extra.items():
            lines.append(f"{k[1:]}: {v:.1f}" if isinstance(v, float) else f"{k[1:]}: {v}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seasons", default=f"{TRAIN_SEASONS[0]}-{DEFAULT_SEASON - 1}", help="e.g. 2019-2025")
    ap.add_argument("--refresh", action="store_true", help="re-download / rebuild cached data")
    ap.add_argument("--no-backtest", action="store_true", help="skip the holdout backtest (faster)")
    ap.add_argument("--out", type=Path, default=None, help="model path (default data/models/projection_model.pkl)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    _, metrics = train_and_save(_parse_seasons(args.seasons), refresh=args.refresh, path=args.out,
                                backtest=not args.no_backtest)
    print(format_metrics(metrics))
    return 0


if __name__ == "__main__":
    sys.exit(main())
