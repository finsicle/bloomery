# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Job persistence.

Plain ``sqlite3``, no ORM. The whole point of dropping Docker was that bloomery
should need no external services, and an ORM would be a dependency carried by
every install for a schema of one table.

WAL mode, because the API process reads while the supervisor writes. Without it
a reader blocks a writer and the UI stalls whenever a job changes state.
"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from bloomery import paths
from bloomery.jobs.types import Job, JobKind, JobStatus, utc_now

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id             TEXT PRIMARY KEY,
    kind           TEXT NOT NULL,
    status         TEXT NOT NULL,
    name           TEXT,
    params         TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    started_at     TEXT,
    finished_at    TEXT,
    exit_code      INTEGER,
    error          TEXT,
    pid            INTEGER,
    pid_created_at REAL,
    log_path       TEXT,
    run_dir        TEXT,
    limits_note    TEXT,
    extra          TEXT
);
CREATE INDEX IF NOT EXISTS jobs_status_created ON jobs (status, created_at);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def new_job_id() -> str:
    """Short, sortable-enough, collision-free identifier."""
    return uuid.uuid4().hex[:12]


class JobStore:
    """Every job that has been queued, running or finished.

    One connection per thread. sqlite3 connections are not safe to share across
    threads, and the supervisor, the API and the test suite all touch this.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or paths.db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        # Outside connect(): executescript() issues an implicit COMMIT before it
        # runs, which would leave the transaction helper with nothing to commit.
        conn = self._conn()
        conn.executescript(_SCHEMA)
        conn.execute(
            "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )

    # ------------------------------------------------------------------ plumbing

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=10.0, isolation_level=None)
            conn.row_factory = sqlite3.Row
            # WAL lets the API read while the supervisor writes. NORMAL sync is
            # the documented companion: durable across process death, which is
            # what matters here, without an fsync per transaction.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Run a block inside one immediate transaction.

        Both exits check ``in_transaction`` first. Some statements — DDL, and
        anything run through ``executescript`` — commit implicitly, and issuing a
        second COMMIT after that raises rather than being a no-op.
        """
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        else:
            if conn.in_transaction:
                conn.execute("COMMIT")

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # -------------------------------------------------------------------- writes

    def create(
        self,
        kind: JobKind,
        params: dict[str, Any],
        *,
        name: str = "",
        job_id: str | None = None,
    ) -> Job:
        job = Job(
            id=job_id or new_job_id(),
            kind=kind,
            status=JobStatus.QUEUED,
            name=name,
            params=params,
            created_at=utc_now(),
        )
        row = job.to_row()
        columns = ", ".join(row)
        placeholders = ", ".join(f":{k}" for k in row)
        with self.connect() as conn:
            conn.execute(f"INSERT INTO jobs ({columns}) VALUES ({placeholders})", row)
        return job

    def update(self, job: Job) -> Job:
        row = job.to_row()
        assignments = ", ".join(f"{k} = :{k}" for k in row if k != "id")
        with self.connect() as conn:
            conn.execute(f"UPDATE jobs SET {assignments} WHERE id = :id", row)
        return job

    def mark_started(self, job_id: str, *, pid: int, pid_created_at: float) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, started_at = ?, pid = ?, pid_created_at = ? "
                "WHERE id = ?",
                (JobStatus.RUNNING.value, utc_now(), pid, pid_created_at, job_id),
            )

    def mark_finished(
        self,
        job_id: str,
        status: JobStatus,
        *,
        exit_code: int | None = None,
        error: str | None = None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, finished_at = ?, exit_code = ?, error = ? "
                "WHERE id = ?",
                (status.value, utc_now(), exit_code, error, job_id),
            )

    def delete(self, job_id: str) -> bool:
        with self.connect() as conn:
            cursor = conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        return cursor.rowcount > 0

    # --------------------------------------------------------------------- reads

    def get(self, job_id: str) -> Job | None:
        row = self._conn().execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return Job.from_row(row) if row else None

    def find(
        self,
        *,
        status: JobStatus | None = None,
        kind: JobKind | None = None,
        limit: int = 100,
    ) -> list[Job]:
        clauses: list[str] = []
        values: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            values.append(status.value)
        if kind is not None:
            clauses.append("kind = ?")
            values.append(kind.value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        values.append(limit)
        rows = self._conn().execute(
            f"SELECT * FROM jobs {where} ORDER BY rowid DESC LIMIT ?",
            values,
        )
        return [Job.from_row(r) for r in rows]

    def next_queued(self) -> Job | None:
        """Oldest queued job, or None.

        Ordered by rowid, which is the true insertion order. Ordering by
        timestamp would be wrong twice over: several jobs are commonly submitted
        within one clock tick, and a clock that steps backwards — every NTP
        correction — would reorder the queue underneath a running system.
        """
        row = (
            self._conn()
            .execute(
                "SELECT * FROM jobs WHERE status = ? ORDER BY rowid ASC LIMIT 1",
                (JobStatus.QUEUED.value,),
            )
            .fetchone()
        )
        return Job.from_row(row) if row else None

    def running(self) -> list[Job]:
        return self.find(status=JobStatus.RUNNING, limit=1000)

    def counts(self) -> dict[str, int]:
        rows = self._conn().execute("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status")
        return {r["status"]: r["n"] for r in rows}
