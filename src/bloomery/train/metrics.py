# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The metrics protocol: one JSON object per line, appended and flushed.

Every engine bloomery grows will write this same format, and the UI will read
only this. Keeping the contract narrow is what stops the web layer sprouting a
special case per trainer.

Written to disk and flushed on every event rather than buffered in memory. A
twenty-hour run must survive the UI being refreshed, the server being restarted,
and the trainer being killed — in the last case the file is the only record of
what happened, so losing the final buffered events would lose exactly the ones
that explain the failure.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any


@dataclass
class MetricsWriter:
    """Append-only JSONL event log for a single run."""

    path: Path
    _handle: Any = field(default=None, init=False, repr=False)
    _started: float = field(default=0.0, init=False, repr=False)

    def __enter__(self) -> MetricsWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")
        self._started = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc is not None:
            self.emit("error", kind=exc_type.__name__ if exc_type else "Unknown", message=str(exc))
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def emit(self, event: str, **fields: Any) -> dict[str, Any]:
        """Write one event. Returns it, so callers can also render it."""
        record = {
            "event": event,
            "t": round(time.perf_counter() - self._started, 3),
            **fields,
        }
        if self._handle is not None:
            self._handle.write(json.dumps(record, default=str) + "\n")
            self._handle.flush()
        return record


def read_events(path: Path) -> list[dict[str, Any]]:
    """Read an event log, skipping any truncated final line.

    A partial last line is expected, not exceptional: it is what a killed
    process leaves behind, and that is precisely when you want to read the log.
    """
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(record, dict):
            events.append(record)
    return events


@dataclass(slots=True)
class Throughput:
    """Exponentially smoothed tokens per second.

    Smoothed because raw per-step numbers on a shared machine swing enough to be
    unreadable, and an unreadable number invites people to ignore the one signal
    that tells them a run has gone wrong.
    """

    smoothing: float = 0.9
    value: float | None = None

    def update(self, tokens: int, seconds: float) -> float:
        if seconds <= 0:
            return self.value or 0.0
        instant = tokens / seconds
        self.value = (
            instant
            if self.value is None
            else self.smoothing * self.value + (1 - self.smoothing) * instant
        )
        return self.value
