"""Private, per-batch oracle cache keyed by exact input, code and environment."""

from __future__ import annotations

import hashlib
import os
import pickle
import platform
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def oracle_cache_path(
    kg: str, data_root: Path, directory: Path, property_fraction: float
) -> Path:
    """Invalidate the cache when data, preparation code, or dependencies change."""
    source = data_root / kg / "data.quintuplet"
    if not source.exists():
        source = data_root / kg / "train_cst_knowledge.quintuplet"
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        digest.update(hashlib.file_digest(stream, "sha256").digest())
    package = Path(__file__).resolve().parent
    for name in (
        "qcn_generator2.py",
        "generator_common.py",
        "allen_list.py",
        "time_package.py",
        "rule_discovery.py",
        "oracle_cache.py",
    ):
        digest.update((package / name).read_bytes())
    lockfile = package.parents[1] / "uv.lock"
    if lockfile.exists():
        digest.update(lockfile.read_bytes())
    digest.update(
        repr(
            (kg, property_fraction, platform.python_version(), np.__version__)
        ).encode()
    )
    return directory / f"{kg}-{digest.hexdigest()}.pickle"


def read_oracle_cache(path: Path) -> dict[str, Any] | None:
    """Read only cache files generated inside the caller's own batch directory."""
    if not path.exists():
        return None
    with path.open("rb") as stream:
        value = pickle.load(stream)
    required = {"initial", "after_oracle", "properties", "kept_facts_count"}
    if not isinstance(value, dict) or not required.issubset(value):
        raise ValueError(f"Invalid oracle cache: {path}")
    return value


def write_oracle_cache(path: Path, preparation: dict[str, Any]) -> None:
    """Publish the complete cache atomically; interrupted runs leave no partial cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            pickle.dump(preparation, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
