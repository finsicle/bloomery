# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Job records.

A job is one unit of work that runs in its own OS process: preparing a corpus,
training a model, benchmarking the machine. Separate processes rather than
threads for three reasons that all matter here — ``CUDA_VISIBLE_DEVICES`` is read
when torch is imported and cannot be changed afterwards, CUDA contexts do not
reliably tear down in-process, and a run that has to be cancellable has to be
killable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class JobKind(StrEnum):
    # The value is the CLI subcommand: build_command runs
    # `python -m bloomery.cli <value>`, so these must not drift apart.
    PREPARE = "prepare"
    TRAIN = "train"
    ADAPT = "adapt"
    BENCH = "bench"
    EXPORT = "export"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    # The server died while this job was running and could not confirm what
    # happened to it. Distinct from FAILED, which means the work itself failed.
    INTERRUPTED = "interrupted"

    @property
    def terminal(self) -> bool:
        return self in (
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.INTERRUPTED,
        )


# Jobs that put a model on an accelerator. These are serialised against each
# other because VRAM cannot be partitioned on consumer hardware: two training
# processes sharing one card do not each get half, they contend for all of it
# and the second one dies.
# EXPORT is deliberately absent: it reads a checkpoint and writes a file, so
# it can run beside a training job rather than queueing behind one.
#
# That concurrency used to be unsafe. checkpoint.save deleted `latest` before
# renaming the new one into place, so an export starting in that window found
# nothing there. The save now moves the old checkpoint aside instead, leaving
# no moment when the directory is absent — which is what makes running the two
# together sound, rather than a note saying to be careful.
EXCLUSIVE_KINDS = frozenset({JobKind.TRAIN, JobKind.ADAPT, JobKind.BENCH})


def utc_now() -> str:
    """Timestamp with microsecond precision.

    Second resolution is not enough: several jobs are routinely submitted inside
    one second, and a timestamp that cannot tell them apart cannot order them.
    """
    return datetime.now(UTC).isoformat(timespec="microseconds")


@dataclass(slots=True)
class Job:
    """One queued or completed unit of work."""

    id: str
    kind: JobKind
    status: JobStatus
    params: dict[str, Any]
    created_at: str
    name: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    exit_code: int | None = None
    error: str | None = None

    # Process identity. The PID alone is not enough to decide whether a job is
    # still alive: PIDs are reused, and a server that restarts hours later could
    # otherwise mistake an unrelated process for its own worker. The creation
    # time pins it.
    pid: int | None = None
    pid_created_at: float | None = None

    # Where the worker's stdout and stderr are written, and where the trainer
    # writes its own JSONL metrics.
    log_path: str | None = None
    run_dir: str | None = None

    # Populated when a memory cap could not be applied, so the UI can say so
    # rather than implying a limit is in force.
    limits_note: str | None = None

    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def exclusive(self) -> bool:
        """Whether this job needs sole use of the accelerator."""
        return self.kind in EXCLUSIVE_KINDS

    @property
    def active(self) -> bool:
        return not self.status.terminal

    def duration_seconds(self) -> float | None:
        if not self.started_at:
            return None
        end = self.finished_at or utc_now()
        try:
            started = datetime.fromisoformat(self.started_at)
            finished = datetime.fromisoformat(end)
        except ValueError:
            return None
        return (finished - started).total_seconds()

    def to_row(self) -> dict[str, Any]:
        """Flatten for SQLite. Dicts are stored as JSON text."""
        return {
            "id": self.id,
            "kind": self.kind.value,
            "status": self.status.value,
            "name": self.name,
            "params": json.dumps(self.params),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "error": self.error,
            "pid": self.pid,
            "pid_created_at": self.pid_created_at,
            "log_path": self.log_path,
            "run_dir": self.run_dir,
            "limits_note": self.limits_note,
            "extra": json.dumps(self.extra),
        }

    @classmethod
    def from_row(cls, row: Any) -> Job:
        return cls(
            id=row["id"],
            kind=JobKind(row["kind"]),
            status=JobStatus(row["status"]),
            name=row["name"] or "",
            params=json.loads(row["params"]) if row["params"] else {},
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            exit_code=row["exit_code"],
            error=row["error"],
            pid=row["pid"],
            pid_created_at=row["pid_created_at"],
            log_path=row["log_path"],
            run_dir=row["run_dir"],
            limits_note=row["limits_note"],
            extra=json.loads(row["extra"]) if row["extra"] else {},
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view, as served by the API."""
        return {
            "id": self.id,
            "kind": self.kind.value,
            "status": self.status.value,
            "name": self.name,
            "params": self.params,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds(),
            "exit_code": self.exit_code,
            "error": self.error,
            "log_path": self.log_path,
            "run_dir": self.run_dir,
            "limits_note": self.limits_note,
            "extra": self.extra,
        }


@dataclass(frozen=True, slots=True)
class ResourceRequest:
    """What a job asked to be given.

    ``gpus`` are indices into the report from :mod:`bloomery.probe`, translated
    into ``CUDA_VISIBLE_DEVICES`` or ``HIP_VISIBLE_DEVICES`` for the worker.
    """

    cores: int | None = None
    memory_bytes: int | None = None
    gpus: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "cores": self.cores,
            "memory_bytes": self.memory_bytes,
            "gpus": list(self.gpus),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> ResourceRequest:
        payload = payload or {}
        return cls(
            cores=payload.get("cores"),
            memory_bytes=payload.get("memory_bytes"),
            gpus=tuple(payload.get("gpus") or ()),
        )


def log_path_for(root: Path, job_id: str) -> Path:
    return root / "jobs" / f"{job_id}.log"
