from __future__ import annotations

import gzip
import json
from importlib import resources
from typing import Any


def read_json_resource(*parts: str) -> Any:
    """Read a bundled JSON resource, preferring a gzip-compressed sibling."""

    path = resources.files(__package__).joinpath(*parts)
    gz_path = resources.files(__package__).joinpath(*parts[:-1], parts[-1] + ".gz")
    if gz_path.is_file():
        return json.loads(gzip.decompress(gz_path.read_bytes()).decode("utf-8"))
    return json.loads(path.read_text(encoding="utf-8"))
