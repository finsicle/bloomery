# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Where bloomery keeps things.

One override, ``BLOOMERY_HOME``, so a user with a small system drive can point
everything at a bigger disk in a single move. Checkpoints and tokenized shards
run to hundreds of gigabytes; the default must be easy to relocate.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_HOME = "BLOOMERY_HOME"


def home() -> Path:
    """Root directory for all bloomery state. Not created by this call."""
    override = os.environ.get(ENV_HOME)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".bloomery"


def datasets_dir() -> Path:
    return home() / "datasets"


def blooms_dir() -> Path:
    """From-scratch base checkpoints, before any refinement."""
    return home() / "blooms"


def runs_dir() -> Path:
    return home() / "runs"


def exports_dir() -> Path:
    return home() / "exports"


def cache_dir() -> Path:
    return home() / "cache"


def db_path() -> Path:
    return home() / "bloomery.db"


def ensure_home() -> Path:
    """Create the directory tree, returning the root."""
    root = home()
    for path in (
        root,
        datasets_dir(),
        blooms_dir(),
        runs_dir(),
        exports_dir(),
        cache_dir(),
    ):
        path.mkdir(parents=True, exist_ok=True)
    return root
