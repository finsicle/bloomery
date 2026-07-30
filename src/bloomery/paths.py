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


def mixtures_dir() -> Path:
    """Weighted dataset blends, one directory per name, one file per version."""
    return home() / "mixtures"


def exports_dir() -> Path:
    return home() / "exports"


def cache_dir() -> Path:
    return home() / "cache"


def dataset_dir(name: str) -> Path:
    """Everything produced by `prepare` for one named corpus."""
    return datasets_dir() / slug(name)


def tokenizer_dir(name: str) -> Path:
    return dataset_dir(name) / "tokenizer"


def tokens_dir(name: str) -> Path:
    return dataset_dir(name) / "tokens"


def run_dir(name: str) -> Path:
    return runs_dir() / slug(name)


def slug(name: str) -> str:
    """Make a user-supplied name safe to use as a directory.

    Names come from the CLI and will later come from a web form, so this must
    never allow a path to escape the bloomery home.
    """
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in name.strip())
    cleaned = cleaned.strip(".-") or "unnamed"
    return cleaned[:64]


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
        mixtures_dir(),
        exports_dir(),
        cache_dir(),
    ):
        path.mkdir(parents=True, exist_ok=True)
    return root
