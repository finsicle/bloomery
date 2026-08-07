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
    # Kept beside the pid so shutdown can prove identity before signalling.
    pid_created_at: float | None
    # Fixed once the job starts, so it is recorded here rather than re-read from
    # the store on every scheduling pass while the lock is held.
    exclusive: bool
    log_path: Path


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
        if not cancel_running:
            return

        # Reap first. The loop exits the moment the stop event is set, without a
        # final pass, so a process that finished during the last poll interval is
        # still sitting in _running with its real outcome unrecorded. Marking it
        # cancelled here would overwrite a job that had actually succeeded.
        self._collect_finished()

        # Signal every tree at once. Cancelling serially would pay the full
        # grace period per job, so shutting down four jobs would take a minute.
        with self._lock:
            entries = list(self._running.items())
            self._running.clear()
            targets = [(entry.launched.pid, entry.pid_created_at) for _, entry in entries]
            for job_id, _ in entries:
                job = self.store.get(job_id)
                if job is not None and not job.status.terminal:
                    self.store.mark_finished(job_id, JobStatus.CANCELLED)

        runner.terminate_all(targets)
        for job_id, _ in entries:
            self._after_change(job_id)

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
            # Nothing bounded this log while the supervisor was gone, and once
            # the job is marked finished nothing else will ever look at it. This
            # is the one chance to reclaim what a crashed run left behind.
            if job.log_path:
                runner.compact_log(Path(job.log_path))

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
        """Cancel a queued or running job. Returns whether anything changed.

        The bookkeeping happens under the lock; the kill does not. Terminating a
        process tree waits out the full grace period, and holding the scheduler
        lock for those seconds would stall every other job — no new work started,
        no finished work collected — for the duration. That matters most once
        cancels arrive from HTTP handlers rather than one at a time.
        """
        with self._lock:
            job = self.store.get(job_id)
            if job is None or job.status.terminal:
                return False

            if job.status is JobStatus.QUEUED:
                self.store.mark_finished(job_id, JobStatus.CANCELLED)
                self._after_change(job_id)
                return True

            entry = self._running.pop(job_id, None)
            pid: int | None
            pid_created_at: float | None
            if entry is not None:
                # This supervisor holds the handle, so the pid is its own by
                # construction and needs no further proof.
                pid, pid_created_at = entry.launched.pid, entry.pid_created_at
            elif job.pid_created_at is None:
                # A record with no identity, from a launch where the creation
                # time could not be read. There is no way to tell the original
                # process from whatever now holds the number, so it is left
                # alone. Recording the cancellation without killing anything is
                # the safe half of the job; killing the wrong process is not
                # recoverable.
                log.warning(
                    "job %s has no recorded process identity; cancelling without signalling",
                    job_id,
                )
                pid, pid_created_at = None, None
            else:
                # No handle, so this record may be stale — reconcile() leaves a
                # live-but-unadopted job running, and once it exits nothing
                # collects it. The creation time is what stops the kill landing
                # on whatever the machine has since given that number to.
                pid, pid_created_at = job.pid, job.pid_created_at
            # Recorded as cancelled before the process actually dies, so a
            # concurrent tick() cannot reclassify it from its exit code.
            self.store.mark_finished(job_id, JobStatus.CANCELLED)

        if pid is not None:
            runner.terminate(pid, pid_created_at)
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
        self._compact_logs()
        self._start_eligible()

    def _compact_logs(self) -> None:
        """Keep running jobs' logs bounded.

        Checked here rather than by the writer because nothing sits between the
        job and its log file: the runner hands the descriptor to the child and
        closes its own copy, so the job keeps writing even if this process dies.
        Pumping the output through the supervisor instead would bound the file
        but would also mean a dead supervisor blocks a running job on a full
        pipe, which is a worse failure than a large log.

        A stat per running job per pass is one cheap syscall, and it does no
        work at all until a log is actually over the cap. On a platform that
        cannot trim underneath a writer this does nothing at all, and the log is
        bounded when the job exits instead.
        """
        with self._lock:
            entries: list[tuple[str, Path, bool | None]] = [
                (entry.job_id, entry.log_path, entry.launched.true_append)
                for entry in self._running.values()
            ]
        adopted = {job_id for job_id, _, _ in entries}

        # Jobs the store calls running that this supervisor did not start. A run
        # survives a restart, reconcile() deliberately leaves it alone rather
        # than killing it, and it is then in no supervisor's _running — so
        # nothing here would ever look at its log again. Those are precisely the
        # longest-lived runs, which is to say the ones whose logs get large.
        for job in self.store.running():
            if job.id not in adopted and job.log_path:
                # Started by a supervisor that is gone, so nothing here knows how
                # its handle was opened. None means "unknown", which compact_log
                # resolves conservatively.
                entries.append((job.id, Path(job.log_path), None))

        # Deliberately outside the lock: this touches the filesystem, and
        # scheduling should not wait behind a multi-megabyte rewrite.
        for job_id, path, appends in entries:
            dropped = runner.compact_log(path, still_writing=True, true_append=appends)
            if dropped:
                log.info("dropped %d bytes from the log for job %s", dropped, job_id)

    def _collect_finished(self) -> None:
        finished: list[Path] = []
        with self._lock:
            for job_id, entry in list(self._running.items()):
                code = entry.launched.process.poll()
                if code is None:
                    continue
                # Taken before the entry goes: a job removed from _running is
                # never looked at again by _compact_logs, so whatever it wrote
                # between the last pass and exiting would stay on disk forever.
                finished.append(entry.log_path)
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

        # Outside the lock, and after the loop: a final pass over each log that
        # just stopped growing. Nothing else will ever look at these again.
        for path in finished:
            runner.compact_log(path)

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
        return any(entry.exclusive for entry in self._running.values())

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
        pid_created_at = launched.created_at()
        self.store.mark_started(job.id, pid=launched.pid, pid_created_at=pid_created_at)
        self._running[job.id] = _Running(
            job_id=job.id,
            launched=launched,
            pid_created_at=pid_created_at,
            exclusive=job.exclusive,
            log_path=log_path,
        )
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
