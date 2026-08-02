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
from dataclasses import dataclass
from pathlib import Path

TEXT_SUFFIXES = frozenset({".txt", ".md", ".rst", ".text"})
JSONL_SUFFIXES = frozenset({".jsonl", ".ndjson"})

# Read in chunks so a single enormous file does not have to fit in memory.
_CHUNK_BYTES = 1 << 20

# Roles a conversation record may use. Anything else is a record we do not know
# how to mask, and guessing would train on the wrong half of it.
_ROLES = frozenset({"system", "user", "assistant"})


@dataclass(frozen=True, slots=True)
class Example:
    """One conversation, as turns.

    Kept separate from the plain-string document type that pretraining uses.
    That type flows into tokenizer training as well as packing, and widening it
    would mean every consumer learning about a shape only one of them needs.
    """

    messages: tuple[dict[str, str], ...]

    @property
    def has_response(self) -> bool:
        """Whether anything here is worth computing a loss on."""
        return any(turn["role"] == "assistant" for turn in self.messages)


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


def iter_examples(source: Path) -> Iterator[Example]:
    """Yield conversations from JSONL, for supervised fine-tuning.

    Two shapes are accepted, because both are what conversation corpora are
    published as:

    - ``{"messages": [{"role": "user", "content": ...}, ...]}``
    - ``{"prompt": ..., "completion": ...}``, which is the same thing with the
      roles implied.

    A record with no assistant turn is skipped: with completion-only masking
    there would be nothing in it to compute a loss on, so packing it would add
    tokens that are all masked and quietly dilute the corpus.
    """
    for path in _jsonl_files(source):
        yield from _iter_conversations(path)


def looks_like_conversations(source: Path) -> bool:
    """Whether this source is chat data that was handed to the plain text path.

    Used to turn "no documents found" — which names nothing — into a message
    that says what the file actually is and which flag reads it.
    """
    import json

    for path in _jsonl_files(source):
        try:
            handle = path.open(encoding="utf-8", errors="replace")
        except OSError:
            continue
        with handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    return False
                return isinstance(record, dict) and (
                    "messages" in record or ("prompt" in record and "completion" in record)
                )
    return False


def _jsonl_files(source: Path) -> Iterator[Path]:
    if source.is_dir():
        for path in sorted(source.rglob("*")):
            if path.is_file() and path.suffix.lower() in JSONL_SUFFIXES:
                yield path
    elif source.suffix.lower() in JSONL_SUFFIXES:
        yield source


def _iter_conversations(path: Path) -> Iterator[Example]:
    import json

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
            if not isinstance(record, dict):
                continue
            example = _as_example(record)
            if example is not None and example.has_response:
                yield example


def _as_example(record: dict[str, object]) -> Example | None:
    raw = record.get("messages")
    if isinstance(raw, list):
        turns: list[dict[str, str]] = []
        for turn in raw:
            if not isinstance(turn, dict):
                return None
            role, content = turn.get("role"), turn.get("content")
            # An unknown role is refused rather than coerced: masking is decided
            # by role, so guessing means training on the wrong half.
            if not isinstance(role, str) or role not in _ROLES:
                return None
            if not isinstance(content, str) or not content.strip():
                return None
            turns.append({"role": role, "content": content})
        return Example(tuple(turns)) if turns else None

    prompt, completion = record.get("prompt"), record.get("completion")
    if isinstance(prompt, str) and isinstance(completion, str) and completion.strip():
        return Example(
            (
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            )
        )
    return None


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
