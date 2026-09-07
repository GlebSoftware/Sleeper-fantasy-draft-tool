#!/usr/bin/env python3
"""Start the draftadvisor web app on localhost and open it in your browser.

    python run.py                # http://127.0.0.1:8787
    python run.py --port 9000    # different port
    python run.py --no-browser   # do not open a browser tab

Everything else (data download, model training, league capture, live draft, mock draft)
is driven from the web page. Set ANTHROPIC_API_KEY in the environment to enable Claude.
"""
from draftadvisor.web.server import main

if __name__ == "__main__":
    raise SystemExit(main())
