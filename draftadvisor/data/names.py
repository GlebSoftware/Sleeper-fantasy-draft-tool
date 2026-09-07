"""Name normalisation shared by the crosswalk and the lean (pandas-free) runtime."""
from __future__ import annotations

import re
import unicodedata

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize_name(name: str | None) -> str:
    """'Amon-Ra St. Brown Jr.' -> 'amonra st brown'."""
    if not name or not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = s.lower().replace("-", "").replace("'", "").replace(".", "")
    parts = [p for p in re.split(r"\s+", s.strip()) if p and p not in _SUFFIXES]
    return " ".join(parts)
