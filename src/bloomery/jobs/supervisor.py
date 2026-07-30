# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The queue: what runs, when, and what happens when the server dies.

A background thread rather than an asyncio task. The work being supervised is
child processes, and polling those is a blocking, thread-shaped problem; running
it beside the event loop keeps a slow ``wait`` from stalling the API.

Two rules govern scheduling:

* **At most one accelerator job at a time.** VRAM cannot be partitioned on
  consumer hardware, so two training runs sharing a card do not get half each —
  they contend, and the second one dies. Queuing is the honest scheduler.
* **Everything else may overlap it.** Preparing a corpus is CPU and IO bound, so
  making it wait behind a twelve-hour training run would be pointless.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from bloomery import paths
from bloomery.jobs import runner
from bloomery.jobs.store import JobStore
from bloomery.jobs.types import Job, JobKind, JobStatus, ResourceRequest, log_path_for

log = logging.getLogger(__name__)

POLL_SECONDS = 0.5

# Non-accelerator jobs that may run alongside a training run.
DEFAULT_MAX_CONCURRENT = 2


@dataclass(slots=True)
class _Running:
    job_id: str
    launched: runner.Launched


class Supervisor:
    """Owns the queue and the processes it has started."""

    def __init__(
        self,
        store: JobStore | None = None,
        *,
        home: Path | None = None,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        poll_seconds: float = POLL_SECONDS,
        on_change: Callable[[Job], None] | None = None,
    ) -> None:
        self.store = store or JobStore()
        self.home = home or paths.home()
        self.max_concurrent = max(1, max_concurrent)
        self.poll_seconds = poll_seconds
        self.on_change = on_change

        self._running: dict[str, _Running] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ----------------------------------------------------------------- lifecycle

    def start(self) -> None:
        """Reconcile anything left over, then begin scheduling."""
        self.reconcile()
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="bloomery-supervisor", daemon=True)
        self._thread.start()

    def stop(self, *, cancel_running: bool = False, timeout: float = 5.0) -> None:
        """Stop scheduling. Running jobs are left alone unless asked otherwise.

        Leaving them is the deliberate default: a training run that survives a
        server restart is a feature, and reconcile() will adopt or account for it
        next time.
        """
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        if cancel_running:
            for job_id in list(self._running):
                self.cancel(job_id)

    def reconcile(self) -> list[Job]:
        """Account for jobs the store still calls running.

        A job is only genuinely alive if its recorded PID exists *and* that
        process started when we recorded it starting. Checking the PID alone
        would let a reused number resurrect a job that died with the machine.
        """
        recovered: list[Job] = []
        for job in self.store.running():
            if job.id in self._running:
                continue
            if runner.is_alive(job.pid, job.pid_created_at):
                # Still going, but this supervisor did not start it and has no
                # handle to wait on. Leave it be rather than killing a live run.
                log.info("job %s still running under pid %s, not adopted", job.id, job.pid)
                continue
            self.store.mark_finished(
                job.id,
                JobStatus.INTERRUPTED,
                error="the server stopped while this job was running",
            )
            refreshed = self.store.get(job.id)
            if refreshed is not None:
                recovered.append(refreshed)
                self._notify(refreshed)
        return recovered

    # -------------------------------------------------------------------- queue

    def submit(
        self,
        kind: JobKind,
        params: dict[str, object],
        *,
        name: str = "",
    ) -> Job:
        job = self.store.create(kind, dict(params), name=name)
        self._notify(job)
        return job

    def cancel(self, job_id: str) -> bool:
        """Cancel a queued or running job. Returns whether anything changed."""
        with self._lock:
            job = self.store.get(job_id)
            if job is None or job.status.terminal:
                return False

            if job.status is JobStatus.QUEUED:
                self.store.mark_finished(job_id, JobStatus.CANCELLED)
                self._after_change(job_id)
                return True

            entry = self._running.pop(job_id, None)
            pid = entry.launched.pid if entry else job.pid
            if pid is not None:
                runner.terminate(pid)
            self.store.mark_finished(job_id, JobStatus.CANCELLED)
            self._after_change(job_id)
            return True

    @property
    def running_ids(self) -> list[str]:
        with self._lock:
            return list(self._running)

    # --------------------------------------------------------------- the loop

    def _loop(self) -> None:
        while not self._stop.wait(self.poll_seconds):
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - a supervisor that dies is worse
                log.exception("supervisor tick failed")

    def tick(self) -> None:
        """One scheduling pass. Public so tests can drive it deterministically."""
        self._collect_finished()
        self._start_eligible()

    def _collect_finished(self) -> None:
        with self._lock:
            for job_id, entry in list(self._running.items()):
                code = entry.launched.process.poll()
                if code is None:
                    continue
                del self._running[job_id]

                current = self.store.get(job_id)
                # A cancel may have already recorded the outcome; do not
                # overwrite it with the exit code the kill produced.
                if current is not None and current.status.terminal:
                    continue

                status = JobStatus.SUCCEEDED if code == 0 else JobStatus.FAILED
                error = None if code == 0 else _describe_exit(code, entry.launched)
                self.store.mark_finished(job_id, status, exit_code=code, error=error)
                self._after_change(job_id)

    def _start_eligible(self) -> None:
        with self._lock:
            while len(self._running) < self.max_concurrent:
                job = self.store.next_queued()
                if job is None:
                    return
                if job.exclusive and self._exclusive_running():
                    # Head-of-line blocking is intended: the queue exists
                    # precisely so two runs do not fight over one card.
                    return
                if not self._launch(job):
                    return

    def _exclusive_running(self) -> bool:
        for entry in self._running.values():
            job = self.store.get(entry.job_id)
            if job is not None and job.exclusive:
                return True
        return False

    def _launch(self, job: Job) -> bool:
        request = ResourceRequest.from_dict(job.params.get("resources"))
        log_path = log_path_for(self.home, job.id)
        try:
            launched = runner.launch(job, request, log_path=log_path)
        except runner.JobLaunchError as exc:
            self.store.mark_finished(job.id, JobStatus.FAILED, error=str(exc))
            self._after_change(job.id)
            return True  # handled; keep draining the queue

        job.log_path = str(log_path)
        job.limits_note = launched.limits_note
        self.store.update(job)
        self.store.mark_started(job.id, pid=launched.pid, pid_created_at=launched.created_at())
        self._running[job.id] = _Running(job_id=job.id, launched=launched)
        self._after_change(job.id)
        return True

    # ------------------------------------------------------------------ notify

    def _after_change(self, job_id: str) -> None:
        job = self.store.get(job_id)
        if job is not None:
            self._notify(job)

    def _notify(self, job: Job) -> None:
        if self.on_change is None:
            return
        try:
            self.on_change(job)
        except Exception:  # noqa: BLE001 - a listener must not break scheduling
            log.exception("job listener failed for %s", job.id)


def _describe_exit(code: int, launched: runner.Launched) -> str:
    """Turn an exit code into something a person can act on."""
    if code == 137:
        return (
            "killed with SIGKILL (exit 137). Usually the out-of-memory killer: "
            "the process asked for more memory than it was allowed."
        )
    if code == 139:
        return "segmentation fault (exit 139), a crash inside a native library"
    if code == 132:
        return "illegal instruction (exit 132), a crash inside a native library"
    if code < 0:
        return f"killed by signal {-code}"
    return f"exited with status {code}"
