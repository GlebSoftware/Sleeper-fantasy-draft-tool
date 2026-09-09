"""The deployed runtime has no pandas and no scikit-learn (see requirements.txt).

Locally they are installed, so an accidental top-level import of either is invisible until Vercel
fails to build or a request 500s in production. This test reproduces the deployment target's
constraint instead: it blocks the heavy packages at the import system and then imports every module
the serverless function actually loads.

The usual way to break this is not an obvious ``import pandas`` - it is adding a module under a
package whose ``__init__`` pulls in the heavy path (``draftadvisor/projections/__init__.py`` imports
the scikit-learn model behind a try/except for exactly this reason). Importing a submodule always
runs its package ``__init__``, so a new file in the wrong package is enough.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

HEAVY = ("pandas", "sklearn", "scipy", "matplotlib")

#: Everything the Vercel entry point (api/index.py -> draftadvisor.web.server) can reach.
LEAN_MODULES = [
    "draftadvisor.lean",
    "draftadvisor.web.server",
    "draftadvisor.models",
    "draftadvisor.capture",
    "draftadvisor.espn.parsing",
    "draftadvisor.espn.client",
    "draftadvisor.sleeper.parsing",
    "draftadvisor.research.store",
    "draftadvisor.projections.blend",
    "draftadvisor.projections.inseason",
    "draftadvisor.strategy.recommend",
    "draftadvisor.strategy.trade",
]

_SCRIPT = textwrap.dedent("""
    import importlib, importlib.abc, sys
    BLOCKED = {blocked!r}

    class Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in BLOCKED:
                raise ImportError(fullname + " is not installed on the deployment target")
            return None

    sys.meta_path.insert(0, Blocker())
    for name in {modules!r}:
        importlib.import_module(name)
    # and the runtime has to actually work, not merely import
    from draftadvisor.projections.inseason import InSeasonTables, win_probability
    assert not InSeasonTables().present
    assert 0.6 < win_probability((110.0, 20.0), (100.0, 20.0)) < 0.7
    print("ok")
""")


def test_the_deployed_modules_import_without_pandas_or_scikit_learn():
    r = subprocess.run([sys.executable, "-c", _SCRIPT.format(blocked=HEAVY, modules=LEAN_MODULES)],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"the lean import path pulled in a heavy dependency:\n{r.stderr[-3000:]}"
    assert r.stdout.strip().endswith("ok")


def test_the_blocker_would_actually_catch_a_heavy_import():
    """A guard that cannot fail proves nothing: the same harness must reject a real pandas import."""
    script = _SCRIPT.format(blocked=HEAVY, modules=["pandas"])
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert r.returncode != 0 and "not installed on the deployment target" in r.stderr
