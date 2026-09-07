"""Command line interface. See DESIGN.md §3.6. STUB - "ui" agent."""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    print("draftadvisor CLI not implemented yet", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
