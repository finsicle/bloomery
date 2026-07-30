# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reading text in, and a synthetic corpus for smoke-testing.

A document here is just a string. Bloomery treats each one as an independent
sequence and separates them with an end-of-text token when packing, so the model
does not learn to run one document into the next.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from pathlib import Path

TEXT_SUFFIXES = frozenset({".txt", ".md", ".rst", ".text"})
JSONL_SUFFIXES = frozenset({".jsonl", ".ndjson"})

# Read in chunks so a single enormous file does not have to fit in memory.
_CHUNK_BYTES = 1 << 20


def iter_documents(source: Path, *, jsonl_field: str = "text") -> Iterator[str]:
    """Yield documents from a file or a directory tree.

    Plain text files are one document each. JSONL files are one document per
    line, read from ``jsonl_field``. Unreadable files are skipped rather than
    aborting a long ingest — one bad file in a corpus of thousands should not
    cost you the whole run.
    """
    if source.is_dir():
        for path in sorted(source.rglob("*")):
            if path.is_file():
                yield from iter_documents(path, jsonl_field=jsonl_field)
        return

    suffix = source.suffix.lower()
    if suffix in JSONL_SUFFIXES:
        yield from _iter_jsonl(source, jsonl_field)
    elif suffix in TEXT_SUFFIXES:
        text = _read_text(source)
        if text and text.strip():
            yield text


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _iter_jsonl(path: Path, field: str) -> Iterator[str]:
    import json  # noqa: PLC0415 - only needed on this branch

    try:
        handle = path.open(encoding="utf-8", errors="replace")
    except OSError:
        return
    with handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(record, dict):
                value = record.get(field)
                if isinstance(value, str) and value.strip():
                    yield value
            elif isinstance(record, str) and record.strip():
                yield record


def count_bytes(source: Path) -> int:
    """Total size of the ingestible files under ``source``."""
    if source.is_file():
        return (
            source.stat().st_size
            if source.suffix.lower() in (TEXT_SUFFIXES | JSONL_SUFFIXES)
            else 0
        )
    total = 0
    for path in source.rglob("*"):
        if path.is_file() and path.suffix.lower() in (TEXT_SUFFIXES | JSONL_SUFFIXES):
            total += path.stat().st_size
    return total


# --------------------------------------------------------------------------- #
# Synthetic corpus
# --------------------------------------------------------------------------- #

_NAMES = ("Ana", "Ben", "Cleo", "Dev", "Esme", "Finn", "Gia", "Hugo", "Ivy", "Jonas")
_ANIMALS = ("cat", "dog", "fox", "owl", "bear", "mouse", "frog", "goat")
_ADJECTIVES = ("small", "happy", "sleepy", "brave", "quiet", "clever", "hungry", "shy")
_PLACES = ("garden", "river", "forest", "kitchen", "hill", "beach", "market", "attic")
_OBJECTS = ("ball", "hat", "spoon", "kite", "book", "lamp", "key", "basket")
_FEELINGS = ("happy", "proud", "calm", "excited", "sleepy", "glad")

_TEMPLATES = (
    "{name} found a {adj} {obj} in the {place}. "
    "The {animal} wanted it too. They shared it and felt {feeling}.",
    "One morning {name} walked to the {place}. "
    "A {adj} {animal} was waiting there. {name} felt {feeling}.",
    "The {adj} {animal} lost its {obj}. "
    "{name} looked in the {place} and found it. The {animal} was {feeling}.",
    "{name} and the {animal} played near the {place}. "
    "They found a {adj} {obj}. Everyone was {feeling}.",
    "There was a {adj} {animal} who lived by the {place}. "
    "Each day it carried a {obj}. {name} thought that was {feeling}.",
)


def synthetic_documents(count: int, *, seed: int = 0) -> list[str]:
    """A tiny deterministic corpus for proving the pipeline works.

    Not a substitute for real data. It exists so that ``bloomery demo`` can run
    end to end in seconds with no download, and so tests can assert that loss
    actually falls — this grammar has low enough entropy that a very small model
    should fit it quickly, which makes "training is broken" easy to detect.
    """
    rng = random.Random(seed)
    documents = []
    for _ in range(count):
        template = rng.choice(_TEMPLATES)
        documents.append(
            template.format(
                name=rng.choice(_NAMES),
                animal=rng.choice(_ANIMALS),
                adj=rng.choice(_ADJECTIVES),
                place=rng.choice(_PLACES),
                obj=rng.choice(_OBJECTS),
                feeling=rng.choice(_FEELINGS),
            )
        )
    return documents
