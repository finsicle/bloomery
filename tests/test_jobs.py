# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the job store, runner and queue.

Scheduling is exercised against a stubbed command so the tests are fast and
deterministic; one integration test at the end runs a real job end to end, since
a queue that has never actually started a process proves very little.
"""

from __future__ import annotations

import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import pytest

from bloomery.jobs import runner
from bloomery.jobs.store import JobStore, new_job_id
from bloomery.jobs.supervisor import Supervisor
from bloomery.jobs.types import (
    Job,
    JobKind,
    JobStatus,
    ResourceRequest,
    log_path_for,
)


@pytest.fixture
def store(tmp_path: Path) -> JobStore:
    return JobStore(tmp_path / "jobs.db")


def sleeper(seconds: float = 30) -> list[str]:
    """A command that lives until it is killed."""
    return [sys.executable, "-c", f"import time; time.sleep({seconds})"]


def quick(exit_code: int = 0) -> list[str]:
    return [sys.executable, "-c", f"raise SystemExit({exit_code})"]


@pytest.fixture
def stub_command(monkeypatch: pytest.MonkeyPatch):
    """Replace the real CLI invocation with a controllable stand-in."""

    def use(command_for):  # noqa: ANN001 - local helper
        monkeypatch.setattr(runner, "build_command", lambda job: command_for(job))

    return use


# --------------------------------------------------------------------------- #
# types
# --------------------------------------------------------------------------- #


class TestJobTypes:
    @pytest.mark.parametrize(
        ("status", "terminal"),
        [
            (JobStatus.QUEUED, False),
            (JobStatus.RUNNING, False),
            (JobStatus.SUCCEEDED, True),
            (JobStatus.FAILED, True),
            (JobStatus.CANCELLED, True),
            (JobStatus.INTERRUPTED, True),
        ],
    )
    def test_terminal_states(self, status: JobStatus, terminal: bool) -> None:
        assert status.terminal is terminal

    def test_accelerator_jobs_are_exclusive(self, store: JobStore) -> None:
        """Only these contend for VRAM, so only these need serialising."""
        assert store.create(JobKind.TRAIN, {}).exclusive
        assert store.create(JobKind.BENCH, {}).exclusive
        assert not store.create(JobKind.PREPARE, {}).exclusive

    def test_duration_is_none_before_start(self, store: JobStore) -> None:
        assert store.create(JobKind.PREPARE, {}).duration_seconds() is None

    def test_resource_request_round_trip(self) -> None:
        request = ResourceRequest(cores=4, memory_bytes=1024, gpus=(0, 2))
        assert ResourceRequest.from_dict(request.to_dict()) == request

    def test_resource_request_from_nothing(self) -> None:
        empty = ResourceRequest.from_dict(None)
        assert empty.cores is None and empty.gpus == ()


# --------------------------------------------------------------------------- #
# store
# --------------------------------------------------------------------------- #


class TestJobStore:
    def test_create_and_get(self, store: JobStore) -> None:
        job = store.create(JobKind.PREPARE, {"name": "x"}, name="prep")
        loaded = store.get(job.id)
        assert loaded is not None
        assert loaded.kind is JobKind.PREPARE
        assert loaded.status is JobStatus.QUEUED
        assert loaded.params == {"name": "x"}
        assert loaded.name == "prep"

    def test_get_unknown(self, store: JobStore) -> None:
        assert store.get("nope") is None

    def test_params_survive_a_round_trip(self, store: JobStore) -> None:
        params = {"a": 1, "b": [1, 2], "c": {"d": True}, "e": None}
        job = store.create(JobKind.TRAIN, params)
        assert store.get(job.id).params == params

    def test_mark_started_records_process_identity(self, store: JobStore) -> None:
        job = store.create(JobKind.TRAIN, {})
        store.mark_started(job.id, pid=4321, pid_created_at=99.5)
        loaded = store.get(job.id)
        assert loaded.status is JobStatus.RUNNING
        assert (loaded.pid, loaded.pid_created_at) == (4321, 99.5)
        assert loaded.started_at is not None

    def test_mark_finished(self, store: JobStore) -> None:
        job = store.create(JobKind.TRAIN, {})
        store.mark_finished(job.id, JobStatus.FAILED, exit_code=2, error="boom")
        loaded = store.get(job.id)
        assert loaded.status is JobStatus.FAILED
        assert loaded.exit_code == 2
        assert loaded.error == "boom"
        assert loaded.finished_at is not None

    def test_list_filters(self, store: JobStore) -> None:
        store.create(JobKind.PREPARE, {})
        train = store.create(JobKind.TRAIN, {})
        store.mark_finished(train.id, JobStatus.SUCCEEDED, exit_code=0)
        assert len(store.find()) == 2
        assert len(store.find(kind=JobKind.TRAIN)) == 1
        assert len(store.find(status=JobStatus.SUCCEEDED)) == 1
        assert len(store.find(status=JobStatus.QUEUED)) == 1

    def test_next_queued_is_oldest_first(self, store: JobStore) -> None:
        first = store.create(JobKind.PREPARE, {}, job_id="aaa")
        store.create(JobKind.PREPARE, {}, job_id="bbb")
        assert store.next_queued().id == first.id

    def test_queue_is_fifo_for_a_burst_of_submissions(self, store: JobStore) -> None:
        """Jobs submitted inside one clock tick must still drain in order.

        Ordering originally used the timestamp, which had second resolution, so
        a burst tied and fell back to the random job id — the queue ran in
        arbitrary order whenever more than one job was submitted per second.
        """
        submitted = [store.create(JobKind.TRAIN, {}).id for _ in range(8)]

        drained = []
        while (job := store.next_queued()) is not None:
            drained.append(job.id)
            store.mark_finished(job.id, JobStatus.SUCCEEDED, exit_code=0)

        assert drained == submitted

    def test_ordering_survives_a_clock_that_steps_backwards(
        self, store: JobStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An NTP correction must not reorder a queue that is already running."""
        first = store.create(JobKind.TRAIN, {})
        # The next job records an earlier timestamp than the one before it.
        monkeypatch.setattr(
            "bloomery.jobs.store.utc_now", lambda: "2000-01-01T00:00:00.000000+00:00"
        )
        second = store.create(JobKind.TRAIN, {})

        assert store.next_queued().id == first.id
        assert [j.id for j in store.find()] == [second.id, first.id]

    def test_next_queued_skips_finished(self, store: JobStore) -> None:
        job = store.create(JobKind.PREPARE, {})
        store.mark_finished(job.id, JobStatus.SUCCEEDED, exit_code=0)
        assert store.next_queued() is None

    def test_delete(self, store: JobStore) -> None:
        job = store.create(JobKind.PREPARE, {})
        assert store.delete(job.id) is True
        assert store.get(job.id) is None
        assert store.delete(job.id) is False

    def test_counts(self, store: JobStore) -> None:
        store.create(JobKind.PREPARE, {})
        done = store.create(JobKind.TRAIN, {})
        store.mark_finished(done.id, JobStatus.SUCCEEDED, exit_code=0)
        assert store.counts() == {"queued": 1, "succeeded": 1}

    def test_survives_reopening(self, tmp_path: Path) -> None:
        path = tmp_path / "jobs.db"
        job = JobStore(path).create(JobKind.PREPARE, {"k": "v"})
        assert JobStore(path).get(job.id).params == {"k": "v"}

    def test_usable_from_several_threads(self, store: JobStore) -> None:
        """The supervisor writes while the API reads; connections are per-thread."""
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for _ in range(10):
                    job = store.create(JobKind.PREPARE, {})
                    store.get(job.id)
                    store.find(limit=5)
            except Exception as exc:  # noqa: BLE001 - recorded and re-raised below
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, errors
        assert len(store.find(limit=100)) == 40

    def test_ids_are_unique(self) -> None:
        assert len({new_job_id() for _ in range(500)}) == 500


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #


class TestBuildCommand:
    def make(self, kind: JobKind, params: dict) -> Job:
        return Job(id="x", kind=kind, status=JobStatus.QUEUED, params=params, created_at="now")

    def test_train_command(self) -> None:
        command = runner.build_command(
            self.make(JobKind.TRAIN, {"data": "corpus", "depth": 8, "steps": 100})
        )
        assert command[:4] == [sys.executable, "-m", "bloomery.cli", "train"]
        assert "--data" in command and "corpus" in command
        assert "--depth" in command and "8" in command

    def test_switches_are_bare_flags(self) -> None:
        command = runner.build_command(
            self.make(JobKind.TRAIN, {"data": "c", "grad_checkpoint": True, "resume": False})
        )
        assert "--grad-checkpoint" in command
        assert "--resume" not in command

    def test_unknown_parameters_are_dropped(self) -> None:
        """A request from the API must not be able to reach an arbitrary flag."""
        command = runner.build_command(
            self.make(JobKind.TRAIN, {"data": "c", "--rm": "-rf", "evil": "; rm -rf /"})
        )
        assert "--rm" not in command
        assert "; rm -rf /" not in command
        assert "evil" not in command

    def test_none_and_empty_are_omitted(self) -> None:
        command = runner.build_command(
            self.make(JobKind.TRAIN, {"data": "c", "mix": None, "size": ""})
        )
        assert "--mix" not in command
        assert "--size" not in command

    def test_each_kind_maps(self) -> None:
        for kind in (JobKind.PREPARE, JobKind.TRAIN, JobKind.BENCH):
            assert runner.build_command(self.make(kind, {}))[3] == kind.value


class TestBuildEnvironment:
    def test_gpu_selection_sets_both_vendors(self) -> None:
        """Whether torch is a CUDA or ROCm build is not known until it imports."""
        env = runner.build_environment(ResourceRequest(gpus=(0, 2)))
        assert env["CUDA_VISIBLE_DEVICES"] == "0,2"
        assert env["HIP_VISIBLE_DEVICES"] == "0,2"

    def test_no_selection_leaves_devices_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Setting the variable to empty would hide every GPU, not select all."""
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        env = runner.build_environment(ResourceRequest())
        assert "CUDA_VISIBLE_DEVICES" not in env

    def test_cores_cap_thread_pools(self) -> None:
        env = runner.build_environment(ResourceRequest(cores=3))
        assert env["OMP_NUM_THREADS"] == "3"
        assert env["MKL_NUM_THREADS"] == "3"

    def test_colour_is_disabled_for_captured_logs(self) -> None:
        assert runner.build_environment(ResourceRequest())["NO_COLOR"] == "1"


class TestProcessIdentity:
    def test_is_alive_for_this_process(self) -> None:
        me = psutil.Process()
        assert runner.is_alive(me.pid, me.create_time()) is True

    def test_a_reused_pid_is_not_the_same_process(self) -> None:
        """The whole reason creation time is stored alongside the PID."""
        me = psutil.Process()
        assert runner.is_alive(me.pid, me.create_time() - 5000) is False

    def test_missing_pid(self) -> None:
        assert runner.is_alive(None, None) is False

    def test_nonexistent_pid(self) -> None:
        assert runner.is_alive(999_999, 1.0) is False

    def test_terminate_reports_nothing_to_kill(self) -> None:
        assert runner.terminate(999_999) is False

    def test_terminate_kills_a_real_process(self, tmp_path: Path) -> None:
        import subprocess

        process = subprocess.Popen(sleeper(60))
        try:
            assert runner.terminate(process.pid, grace=5) is True
            process.wait(timeout=10)
            assert process.poll() is not None
        finally:
            if process.poll() is None:  # pragma: no cover - cleanup
                process.kill()


# --------------------------------------------------------------------------- #
# supervisor
# --------------------------------------------------------------------------- #


class TestSupervisor:
    def test_runs_a_job_to_completion(self, store: JobStore, tmp_path: Path, stub_command) -> None:
        stub_command(lambda job: quick(0))
        sup = Supervisor(store=store, home=tmp_path, poll_seconds=0.05)
        job = sup.submit(JobKind.PREPARE, {})

        for _ in range(100):
            sup.tick()
            if store.get(job.id).status.terminal:
                break
            time.sleep(0.05)

        done = store.get(job.id)
        assert done.status is JobStatus.SUCCEEDED
        assert done.exit_code == 0
        assert done.started_at and done.finished_at

    def test_a_failing_job_records_its_exit_code(
        self, store: JobStore, tmp_path: Path, stub_command
    ) -> None:
        stub_command(lambda job: quick(3))
        sup = Supervisor(store=store, home=tmp_path, poll_seconds=0.05)
        job = sup.submit(JobKind.PREPARE, {})
        for _ in range(100):
            sup.tick()
            if store.get(job.id).status.terminal:
                break
            time.sleep(0.05)
        done = store.get(job.id)
        assert done.status is JobStatus.FAILED
        assert done.exit_code == 3
        assert "status 3" in (done.error or "")

    def test_output_is_captured_to_the_log(
        self, store: JobStore, tmp_path: Path, stub_command
    ) -> None:
        stub_command(lambda job: [sys.executable, "-c", "print('hello from the job')"])
        sup = Supervisor(store=store, home=tmp_path, poll_seconds=0.05)
        job = sup.submit(JobKind.PREPARE, {})
        for _ in range(100):
            sup.tick()
            if store.get(job.id).status.terminal:
                break
            time.sleep(0.05)
        assert "hello from the job" in log_path_for(tmp_path, job.id).read_text()

    def test_accelerator_jobs_do_not_overlap(
        self, store: JobStore, tmp_path: Path, stub_command
    ) -> None:
        """VRAM cannot be partitioned, so the queue is the scheduler."""
        stub_command(lambda job: sleeper(30))
        sup = Supervisor(store=store, home=tmp_path, max_concurrent=4, poll_seconds=0.05)
        first = sup.submit(JobKind.TRAIN, {})
        second = sup.submit(JobKind.TRAIN, {})
        try:
            for _ in range(10):
                sup.tick()
                time.sleep(0.05)
            assert store.get(first.id).status is JobStatus.RUNNING
            assert store.get(second.id).status is JobStatus.QUEUED
        finally:
            sup.cancel(first.id)
            sup.cancel(second.id)

    def test_other_work_runs_alongside_training(
        self, store: JobStore, tmp_path: Path, stub_command
    ) -> None:
        """Preparing a corpus should not wait behind a twelve-hour run."""
        stub_command(lambda job: sleeper(30))
        sup = Supervisor(store=store, home=tmp_path, max_concurrent=4, poll_seconds=0.05)
        train = sup.submit(JobKind.TRAIN, {})
        prep = sup.submit(JobKind.PREPARE, {})
        try:
            for _ in range(10):
                sup.tick()
                time.sleep(0.05)
            assert store.get(train.id).status is JobStatus.RUNNING
            assert store.get(prep.id).status is JobStatus.RUNNING
        finally:
            sup.cancel(train.id)
            sup.cancel(prep.id)

    def test_concurrency_limit_is_respected(
        self, store: JobStore, tmp_path: Path, stub_command
    ) -> None:
        stub_command(lambda job: sleeper(30))
        sup = Supervisor(store=store, home=tmp_path, max_concurrent=2, poll_seconds=0.05)
        jobs = [sup.submit(JobKind.PREPARE, {}) for _ in range(4)]
        try:
            for _ in range(10):
                sup.tick()
                time.sleep(0.05)
            running = [j for j in jobs if store.get(j.id).status is JobStatus.RUNNING]
            assert len(running) == 2
        finally:
            for job in jobs:
                sup.cancel(job.id)

    def test_cancelling_a_queued_job(self, store: JobStore, tmp_path: Path) -> None:
        sup = Supervisor(store=store, home=tmp_path, poll_seconds=0.05)
        job = sup.submit(JobKind.TRAIN, {})
        assert sup.cancel(job.id) is True
        assert store.get(job.id).status is JobStatus.CANCELLED

    def test_cancelling_a_running_job_kills_it(
        self, store: JobStore, tmp_path: Path, stub_command
    ) -> None:
        stub_command(lambda job: sleeper(60))
        sup = Supervisor(store=store, home=tmp_path, poll_seconds=0.05)
        job = sup.submit(JobKind.TRAIN, {})
        for _ in range(50):
            sup.tick()
            if store.get(job.id).status is JobStatus.RUNNING:
                break
            time.sleep(0.05)

        pid = store.get(job.id).pid
        assert pid is not None
        assert sup.cancel(job.id) is True
        assert store.get(job.id).status is JobStatus.CANCELLED

        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if not runner.is_alive(pid, store.get(job.id).pid_created_at):
                break
            time.sleep(0.2)
        assert not runner.is_alive(pid, store.get(job.id).pid_created_at)

    def test_cancel_does_not_stall_the_scheduler(
        self, store: JobStore, tmp_path: Path, stub_command, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Killing a tree waits out the grace period; the lock must not be held.

        Otherwise a single cancel freezes all scheduling for up to fifteen
        seconds — nothing starts, nothing is collected — which matters as soon as
        cancels arrive from HTTP handlers rather than one at a time.
        """
        stub_command(lambda job: sleeper(60))
        sup = Supervisor(store=store, home=tmp_path, max_concurrent=4, poll_seconds=0.05)
        victim = sup.submit(JobKind.PREPARE, {})
        for _ in range(50):
            sup.tick()
            if store.get(victim.id).status is JobStatus.RUNNING:
                break
            time.sleep(0.05)

        # Captured now, because the point of the fix under test is that cancel()
        # records the job as cancelled *before* killing it — by the time the
        # cleanup below runs it is no longer visible as a running job, and the
        # kill that would normally reap it is stubbed out.
        strays = [store.get(victim.id).pid]

        # A termination slow enough that holding the lock would be obvious.
        started = threading.Event()
        release = threading.Event()

        def slow_terminate(pid, created_at=None, **kwargs):  # noqa: ANN001, ANN003
            started.set()
            release.wait(timeout=10)
            return True

        monkeypatch.setattr(runner, "terminate", slow_terminate)

        canceller = threading.Thread(target=sup.cancel, args=(victim.id,))
        canceller.start()
        assert started.wait(timeout=5), "cancel never reached the kill"

        try:
            # The kill is in flight. Scheduling must still work.
            other = sup.submit(JobKind.PREPARE, {})
            acquired = threading.Event()

            def try_tick() -> None:
                sup.tick()
                acquired.set()

            ticker = threading.Thread(target=try_tick)
            ticker.start()
            assert acquired.wait(timeout=5), "tick() blocked while a cancel was killing"
            ticker.join(timeout=5)
            assert store.get(other.id).status is JobStatus.RUNNING
        finally:
            release.set()
            canceller.join(timeout=10)
            strays.extend(j.pid for j in store.find(status=JobStatus.RUNNING, limit=10) if j.pid)
            runner.terminate_all([(pid, None) for pid in strays if pid], grace=1)

    def test_terminate_all_pays_the_grace_period_once(self) -> None:
        """Shutting down several jobs must not cost one grace period each."""
        import subprocess

        processes = [subprocess.Popen(sleeper(60)) for _ in range(3)]
        try:
            started = time.monotonic()
            signalled = runner.terminate_all([(p.pid, None) for p in processes], grace=5)
            elapsed = time.monotonic() - started
            assert signalled >= 3
            # Serial termination of three trees would exceed one grace period.
            assert elapsed < 5, f"took {elapsed:.1f}s, suggesting serial waits"
            for process in processes:
                process.wait(timeout=10)
        finally:
            for process in processes:
                if process.poll() is None:  # pragma: no cover - cleanup
                    process.kill()

    def test_cancelling_a_finished_job_changes_nothing(
        self, store: JobStore, tmp_path: Path
    ) -> None:
        sup = Supervisor(store=store, home=tmp_path)
        job = sup.submit(JobKind.PREPARE, {})
        store.mark_finished(job.id, JobStatus.SUCCEEDED, exit_code=0)
        assert sup.cancel(job.id) is False
        assert store.get(job.id).status is JobStatus.SUCCEEDED

    def test_cancel_wins_over_the_exit_code(
        self, store: JobStore, tmp_path: Path, stub_command
    ) -> None:
        """Killing a process produces a non-zero exit; that is not a failure."""
        stub_command(lambda job: sleeper(60))
        sup = Supervisor(store=store, home=tmp_path, poll_seconds=0.05)
        job = sup.submit(JobKind.TRAIN, {})
        for _ in range(50):
            sup.tick()
            if store.get(job.id).status is JobStatus.RUNNING:
                break
            time.sleep(0.05)
        sup.cancel(job.id)
        for _ in range(20):
            sup.tick()
            time.sleep(0.05)
        assert store.get(job.id).status is JobStatus.CANCELLED

    def test_a_job_that_cannot_start_is_recorded_as_failed(
        self, store: JobStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            runner, "build_command", lambda job: ["definitely-not-a-real-binary-xyz"]
        )
        sup = Supervisor(store=store, home=tmp_path, poll_seconds=0.05)
        job = sup.submit(JobKind.PREPARE, {})
        sup.tick()
        done = store.get(job.id)
        assert done.status is JobStatus.FAILED
        assert "could not start" in (done.error or "")

    def test_listeners_are_notified_of_every_transition(
        self, store: JobStore, tmp_path: Path, stub_command
    ) -> None:
        stub_command(lambda job: quick(0))
        seen: list[str] = []
        sup = Supervisor(
            store=store,
            home=tmp_path,
            poll_seconds=0.05,
            on_change=lambda j: seen.append(j.status.value),
        )
        job = sup.submit(JobKind.PREPARE, {})
        for _ in range(100):
            sup.tick()
            if store.get(job.id).status.terminal:
                break
            time.sleep(0.05)
        assert seen[0] == "queued"
        assert "running" in seen
        assert seen[-1] == "succeeded"

    def test_a_broken_listener_does_not_stop_scheduling(
        self, store: JobStore, tmp_path: Path, stub_command
    ) -> None:
        stub_command(lambda job: quick(0))

        def explode(job: Job) -> None:
            raise RuntimeError("listener is broken")

        sup = Supervisor(store=store, home=tmp_path, poll_seconds=0.05, on_change=explode)
        job = sup.submit(JobKind.PREPARE, {})
        for _ in range(100):
            sup.tick()
            if store.get(job.id).status.terminal:
                break
            time.sleep(0.05)
        assert store.get(job.id).status is JobStatus.SUCCEEDED


class TestReconcile:
    def test_a_job_whose_process_is_gone_is_interrupted(
        self, store: JobStore, tmp_path: Path
    ) -> None:
        """What a killed server leaves behind, and why it is not FAILED."""
        job = store.create(JobKind.TRAIN, {})
        store.mark_started(job.id, pid=999_999, pid_created_at=1.0)

        recovered = Supervisor(store=store, home=tmp_path).reconcile()

        assert [j.id for j in recovered] == [job.id]
        assert store.get(job.id).status is JobStatus.INTERRUPTED
        assert "server stopped" in store.get(job.id).error

    def test_a_still_running_job_is_left_alone(self, store: JobStore, tmp_path: Path) -> None:
        """Killing a live training run because we lost its handle would be worse."""
        me = psutil.Process()
        job = store.create(JobKind.TRAIN, {})
        store.mark_started(job.id, pid=me.pid, pid_created_at=me.create_time())

        recovered = Supervisor(store=store, home=tmp_path).reconcile()

        assert recovered == []
        assert store.get(job.id).status is JobStatus.RUNNING

    def test_a_reused_pid_does_not_resurrect_a_job(self, store: JobStore, tmp_path: Path) -> None:
        """This process exists, but it is not the worker that was recorded."""
        me = psutil.Process()
        job = store.create(JobKind.TRAIN, {})
        store.mark_started(job.id, pid=me.pid, pid_created_at=me.create_time() - 10_000)

        Supervisor(store=store, home=tmp_path).reconcile()

        assert store.get(job.id).status is JobStatus.INTERRUPTED

    def test_cancelling_a_stale_record_cannot_kill_a_bystander(
        self, store: JobStore, tmp_path: Path
    ) -> None:
        """The reachable version of pid reuse, and the reason kills are verified.

        reconcile() deliberately leaves a live-but-unadopted job as RUNNING, and
        nothing then collects it when the process finally exits — so the store
        can hold a pid long after it stopped belonging to that job. Cancelling
        such a record used to send SIGTERM then SIGKILL to whatever the machine
        had since given that number to.
        """
        import subprocess

        bystander = subprocess.Popen(sleeper(45))
        try:
            recorded_at = psutil.Process(bystander.pid).create_time()
            job = store.create(JobKind.TRAIN, {})
            # Same pid, but recorded as having started long before this process.
            store.mark_started(job.id, pid=bystander.pid, pid_created_at=recorded_at - 10_000)

            sup = Supervisor(store=store, home=tmp_path)
            assert sup.cancel(job.id) is True
            assert store.get(job.id).status is JobStatus.CANCELLED

            time.sleep(1.0)
            assert bystander.poll() is None, "cancel killed an unrelated process"
        finally:
            bystander.kill()
            bystander.wait(timeout=10)

    def test_terminate_skips_a_recycled_pid(self) -> None:
        """The guard at the level it is enforced."""
        import subprocess

        process = subprocess.Popen(sleeper(45))
        try:
            wrong_time = psutil.Process(process.pid).create_time() - 10_000
            assert runner.terminate(process.pid, wrong_time, grace=1) is False
            time.sleep(0.5)
            assert process.poll() is None

            # With the right identity it is stopped as usual.
            correct_time = psutil.Process(process.pid).create_time()
            assert runner.terminate(process.pid, correct_time, grace=5) is True
            process.wait(timeout=10)
        finally:
            if process.poll() is None:  # pragma: no cover - cleanup
                process.kill()

    def test_shutdown_keeps_a_real_outcome(
        self, store: JobStore, tmp_path: Path, stub_command
    ) -> None:
        """A job that finished just before shutdown must not be recorded cancelled.

        The loop exits the moment the stop event is set, with no final pass, so a
        process that completed during the last poll interval is still in
        _running with its real outcome unrecorded.
        """
        stub_command(lambda job: quick(0))
        sup = Supervisor(store=store, home=tmp_path, poll_seconds=30)
        job = sup.submit(JobKind.PREPARE, {})
        sup.tick()  # launches it
        assert store.get(job.id).status is JobStatus.RUNNING

        # Let it finish, without giving the supervisor a chance to notice.
        entry = sup._running[job.id]
        entry.launched.process.wait(timeout=30)

        sup.stop(cancel_running=True)

        assert store.get(job.id).status is JobStatus.SUCCEEDED
        assert store.get(job.id).exit_code == 0

    def test_shutdown_still_cancels_what_is_really_running(
        self, store: JobStore, tmp_path: Path, stub_command
    ) -> None:
        stub_command(lambda job: sleeper(60))
        sup = Supervisor(store=store, home=tmp_path, poll_seconds=30)
        job = sup.submit(JobKind.PREPARE, {})
        sup.tick()
        pid = store.get(job.id).pid

        sup.stop(cancel_running=True)

        assert store.get(job.id).status is JobStatus.CANCELLED
        assert not runner.is_alive(pid, store.get(job.id).pid_created_at)

    def test_a_job_with_no_recorded_identity_is_not_signalled(
        self, store: JobStore, tmp_path: Path
    ) -> None:
        """An unreadable creation time must not become a licence to kill by pid.

        Storing 0.0 for an unavailable identity was worse than storing nothing:
        it looked like data, matched no real process, and left the job both
        unkillable and permanently misreported.
        """
        import subprocess

        bystander = subprocess.Popen(sleeper(45))
        try:
            job = store.create(JobKind.TRAIN, {})
            store.mark_started(job.id, pid=bystander.pid, pid_created_at=None)

            sup = Supervisor(store=store, home=tmp_path)
            assert sup.cancel(job.id) is True
            assert store.get(job.id).status is JobStatus.CANCELLED

            time.sleep(1.0)
            assert bystander.poll() is None, "signalled a pid it could not verify"
        finally:
            bystander.kill()
            bystander.wait(timeout=10)

    def test_exclusivity_is_read_from_the_entry_not_the_store(
        self, store: JobStore, tmp_path: Path, stub_command, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Scheduling must not hit the database once per running job per tick."""
        stub_command(lambda job: sleeper(30))
        sup = Supervisor(store=store, home=tmp_path, max_concurrent=4, poll_seconds=0.05)
        first = sup.submit(JobKind.TRAIN, {})
        sup.tick()

        reads = 0
        original = store.get

        def counting_get(job_id: str):  # noqa: ANN202
            nonlocal reads
            reads += 1
            return original(job_id)

        monkeypatch.setattr(store, "get", counting_get)
        try:
            before = reads
            sup._exclusive_running()
            assert reads == before, "exclusivity check still queried the store"
        finally:
            monkeypatch.undo()
            sup.cancel(first.id)

    def test_queued_jobs_are_untouched(self, store: JobStore, tmp_path: Path) -> None:
        job = store.create(JobKind.TRAIN, {})
        Supervisor(store=store, home=tmp_path).reconcile()
        assert store.get(job.id).status is JobStatus.QUEUED


class TestRealJob:
    """One genuine end-to-end run, because a stub proves nothing about the CLI."""

    def test_a_real_prepare_job_succeeds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        monkeypatch.setenv("BLOOMERY_HOME", str(home))
        store = JobStore(home / "jobs.db")
        sup = Supervisor(store=store, home=home, poll_seconds=0.1)

        job = sup.submit(JobKind.PREPARE, {"name": "queued-corpus", "synthetic": 300, "vocab": 300})

        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            sup.tick()
            if store.get(job.id).status.terminal:
                break
            time.sleep(0.2)

        done = store.get(job.id)
        log = log_path_for(home, job.id)
        assert done.status is JobStatus.SUCCEEDED, (
            done.error,
            log.read_text(errors="replace") if log.exists() else "no log",
        )
        assert done.exit_code == 0
        # The job really did the work, not just exit zero.
        assert (home / "datasets" / "queued-corpus" / "tokens" / "meta.json").is_file()


class TestLogCompaction:
    """Job logs are append-only and nothing rotated them.

    A long training run wrote to one file for hours, so a single job could fill
    the disk and take the rest of the machine's work down with it.
    """

    def test_a_log_under_the_cap_is_left_exactly_as_it_was(self, tmp_path: Path) -> None:
        path = tmp_path / "job.log"
        path.write_bytes(b"line\n" * 100)
        before = path.read_bytes()

        assert runner.compact_log(path, cap=1_000_000, keep=1000) == 0
        assert path.read_bytes() == before

    def test_an_oversized_log_keeps_its_most_recent_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "job.log"
        path.write_bytes(b"".join(f"line {i}\n".encode() for i in range(20_000)))
        size = path.stat().st_size

        dropped = runner.compact_log(path, cap=4096, keep=2048)

        assert dropped > 0
        text = path.read_text(encoding="utf-8")
        assert path.stat().st_size < size
        # The end of a run is what a person reads: the error, or the last loss.
        assert text.rstrip().endswith("line 19999")
        assert "line 0\n" not in text

    def test_the_dropped_bytes_are_declared_in_the_log_itself(self, tmp_path: Path) -> None:
        """Silently shortening a log would make a run look like it never started."""
        path = tmp_path / "job.log"
        path.write_bytes(b"".join(f"line {i}\n".encode() for i in range(20_000)))

        runner.compact_log(path, cap=4096, keep=2048)

        first = path.read_text(encoding="utf-8").splitlines()[0]
        assert first.startswith("[bloomery]")
        assert "dropped" in first

    def test_compaction_never_leaves_a_partial_line(self, tmp_path: Path) -> None:
        """The kept window opens mid-line, and that fragment reads as corruption."""
        path = tmp_path / "job.log"
        path.write_bytes(b"".join(f"line {i:06d}\n".encode() for i in range(20_000)))

        runner.compact_log(path, cap=4096, keep=2048)

        body = path.read_text(encoding="utf-8").splitlines()[1:]
        assert all(re.fullmatch(r"line \d{6}", line) for line in body), body[:3]

    def test_compacting_twice_does_not_stack_markers(self, tmp_path: Path) -> None:
        path = tmp_path / "job.log"
        path.write_bytes(b"".join(f"line {i}\n".encode() for i in range(20_000)))
        runner.compact_log(path, cap=4096, keep=2048)
        path.open("ab").write(b"".join(f"more {i}\n".encode() for i in range(20_000)))

        runner.compact_log(path, cap=4096, keep=2048)

        text = path.read_text(encoding="utf-8")
        assert text.count("[bloomery]") == 1

    def test_a_missing_log_is_not_an_error(self, tmp_path: Path) -> None:
        assert runner.compact_log(tmp_path / "nothing.log") == 0

    def test_the_marker_reads_sensibly_at_any_scale(self, tmp_path: Path) -> None:
        """A fixed MiB reading said a trimmed log had dropped "0.0 MiB"."""
        path = tmp_path / "job.log"
        path.write_bytes(b"".join(f"line {i}\n".encode() for i in range(500)))

        runner.compact_log(path, cap=512, keep=256)

        first = path.read_text(encoding="utf-8").splitlines()[0]
        assert "0.0 MiB" not in first
        assert "KiB" in first, first

    def test_a_tail_that_ends_mid_line_does_not_glue_onto_the_next(self, tmp_path: Path) -> None:
        """A job's `print` is not one write syscall.

        The text and its newline can arrive separately, so the size this
        measures can fall between them. Ending the rewrite mid-line glues the
        job's next line onto the truncated one — "line 0085line 0086" in a log
        that never contained such a line. Caught on the macOS matrix, and
        reproducible directly by cutting the file at a byte offset.
        """
        path = tmp_path / "job.log"
        body = b"".join(f"line {i:04d}\n".encode() for i in range(500))
        # Chop off the final newline, exactly as a half-written line looks.
        path.write_bytes(body[:-1])

        runner.compact_log(path, cap=512, keep=256)
        # Whatever the job writes next must start its own line.
        with path.open("ab", buffering=0) as handle:
            handle.write(b"line 9999\n")

        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines[-1] == "line 9999"
        assert all(re.fullmatch(r"line \d{4}", line) for line in lines[1:]), lines[-3:]

    def test_a_window_with_no_line_break_keeps_what_it_can(self, tmp_path: Path) -> None:
        """A progress display that only ever redrew with bare carriage returns.

        There is no newline anywhere in the window, so there is no boundary to
        trim back to. Discarding on that basis would throw away the only output
        the job produced; the fragment is terminated instead, and the marker
        above it already says the beginning went.
        """
        path = tmp_path / "job.log"
        path.write_bytes(b"\r".join(f"progress {i}%".encode() for i in range(5_000)))

        runner.compact_log(path, cap=4096, keep=2048)

        # Bytes, not read_text: universal newlines would translate every bare
        # carriage return into a newline and hide what is actually on disk.
        raw = path.read_bytes()
        body = raw.split(b"\n")[1]
        assert body, "the whole log was discarded"
        assert b"progress 4999%" in body, "the most recent redraw went missing"
        assert raw.endswith(b"\n"), "a fragment must still be terminated"
        assert raw.count(b"\n") == 2, "no newline should have been invented mid-fragment"

    def test_the_rewrite_never_makes_the_file_longer(self, tmp_path: Path) -> None:
        """The invariant that keeps a concurrent append from being spliced.

        If marker + tail were longer than what is on disk — which a small cap
        makes easy, since the marker is a couple of hundred bytes on its own —
        the write would extend the file, and a line the job appended meanwhile
        would land inside the region still being written. Shrinking means such a
        line always lands past the new end, where truncate discards it whole.
        """
        for cap, keep in ((150, 120), (200, 100), (512, 400), (4096, 2048)):
            path = tmp_path / f"job-{cap}.log"
            path.write_bytes(b"".join(f"line {i:04d}\n".encode() for i in range(500)))
            size = path.stat().st_size

            runner.compact_log(path, cap=cap, keep=keep)

            assert path.stat().st_size <= size, f"cap={cap} grew the file"

    def test_a_live_writer_is_left_alone_where_that_is_unsafe(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Corrupting a log to enforce a limit is worse than exceeding it.

        Windows has no O_APPEND for an inherited handle, so a truncation there
        leaves the child writing at its old offset behind a run of NULs, and the
        file is immediately back to its previous size anyway.
        """
        path = tmp_path / "job.log"
        path.write_bytes(b"".join(f"line {i}\n".encode() for i in range(20_000)))
        before = path.read_bytes()

        monkeypatch.setattr(runner, "CAN_TRIM_WHILE_WRITING", False)
        assert runner.compact_log(path, cap=4096, keep=2048, still_writing=True) == 0
        assert path.read_bytes() == before

        # Once the job has exited nothing holds the file, so it is trimmed
        # everywhere — this is what bounds a Windows log.
        assert runner.compact_log(path, cap=4096, keep=2048) > 0
        assert path.stat().st_size < len(before)

    def test_a_job_keeps_writing_where_the_log_was_cut(self, tmp_path: Path) -> None:
        """The reason this truncates in place instead of replacing the file.

        The job holds an open descriptor. Renaming a replacement over the top
        would leave it writing to an unlinked inode and everything it logged
        afterwards would go nowhere. Truncating the file it already has works
        because every write lands at the file's current end, so a shorter file
        just means the next line arrives at the new end.

        Opened through `_open_log` rather than with "ab", which is the whole
        point on Windows: the C runtime's emulated append would put the child's
        next write at a stale offset. This skips only when that fallback is what
        was returned — a property of the handle, not of the platform, so the
        case is exercised wherever the kernel will do a real append.
        """
        path = tmp_path / "job.log"
        handle, true_append = runner._open_log(path)
        if not true_append:
            pytest.skip("this platform gave an emulated append, so a cut is unsafe")
        child = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-c",
                "import sys, time\n"
                "for i in range(200):\n"
                "    print(f'line {i:04d}')\n"
                "    sys.stdout.flush()\n"
                "    time.sleep(0.01)\n",
            ],
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        handle.close()
        try:
            deadline = time.monotonic() + 10
            while path.stat().st_size < 400 and time.monotonic() < deadline:
                time.sleep(0.02)

            dropped = runner.compact_log(path, cap=200, keep=100)
            assert dropped > 0

            assert child.wait(timeout=30) == 0
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)

        content = path.read_bytes()
        # A hole would mean the child wrote at a stale offset past the new end.
        assert b"\0" not in content
        text = content.decode("utf-8")
        assert text.startswith("[bloomery]")
        # Output from after the cut is present and intact.
        assert "line 0199" in text
        assert all(
            re.fullmatch(r"line \d{4}", line) for line in text.splitlines()[1:] if line.strip()
        )

    def test_a_log_left_by_a_crashed_supervisor_is_reclaimed(
        self, store: JobStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nothing bounded this log while the supervisor was gone.

        Once reconcile marks the job interrupted, no later pass looks at it
        again, so this is the only chance to reclaim the space.
        """
        monkeypatch.setattr(runner, "LOG_CAP_BYTES", 4096)
        monkeypatch.setattr(runner, "LOG_KEEP_BYTES", 1024)

        job = store.create(JobKind.PREPARE, {})
        path = log_path_for(tmp_path, job.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"".join(f"line {i}\n".encode() for i in range(20_000)))
        job.log_path = str(path)
        store.update(job)
        # A pid that is not alive, which is what a crash leaves behind.
        store.mark_started(job.id, pid=2**22, pid_created_at=1.0)

        oversized = path.stat().st_size
        sup = Supervisor(store=store, home=tmp_path, poll_seconds=0.05)
        recovered = sup.reconcile()

        assert [j.id for j in recovered] == [job.id]
        assert store.get(job.id).status is JobStatus.INTERRUPTED
        assert path.stat().st_size < oversized
        assert path.read_text(encoding="utf-8").startswith("[bloomery]")

    def test_the_capability_is_per_job_not_per_process(self, tmp_path: Path) -> None:
        """One job falling back must not disable trimming for every other job.

        The first version of this set a module-level flag from `launch`, so a
        single log that could not get a kernel append turned trimming off for
        the whole process — permanently, and depending on which job happened to
        start first.
        """
        path = tmp_path / "job.log"
        path.write_bytes(b"line\n" * 4000)
        before = path.stat().st_size

        # Proven kernel append: safe to cut underneath the writer.
        assert runner.compact_log(path, cap=200, keep=100, still_writing=True, true_append=True) > 0
        assert path.stat().st_size < before

        path.write_bytes(b"line\n" * 4000)
        # Emulated append on this job, whatever any other job managed.
        assert (
            runner.compact_log(path, cap=200, keep=100, still_writing=True, true_append=False) == 0
        )
        assert path.stat().st_size == before

    def test_an_unknown_handle_falls_back_to_the_platform(self, tmp_path: Path) -> None:
        """A job adopted from a supervisor that is gone: nobody knows how it opened.

        Unproven counts as unsafe, so this resolves to what the platform can be
        relied on to do — which keeps the long-lived adopted runs bounded on
        POSIX without risking a corrupted log on Windows.
        """
        path = tmp_path / "job.log"
        path.write_bytes(b"line\n" * 4000)

        dropped = runner.compact_log(path, cap=200, keep=100, still_writing=True, true_append=None)
        assert (dropped > 0) is runner.CAN_TRIM_WHILE_WRITING

    def test_a_finished_job_is_trimmed_whatever_its_handle_was(self, tmp_path: Path) -> None:
        """Nothing holds the file, so how it was opened stopped mattering."""
        path = tmp_path / "job.log"
        path.write_bytes(b"line\n" * 4000)
        assert runner.compact_log(path, cap=200, keep=100, true_append=False) > 0

    @pytest.mark.skipif(
        not runner.CAN_TRIM_WHILE_WRITING,
        reason="an unadopted job is still writing, so its log waits for the next restart",
    )
    def test_a_job_that_outlived_a_restart_still_has_its_log_bounded(
        self, store: JobStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The gap that matters most, because it is the longest-running jobs.

        A run survives a supervisor restart. reconcile() finds it alive and
        deliberately leaves it be rather than killing it, so it is in the
        store's running set but in no supervisor's _running. Scanning only
        _running would mean nothing ever bounded its log again.

        Where a live writer cannot be trimmed this necessarily waits: the job
        holds no handle this supervisor can wait on, so its log is bounded by a
        reconcile() after it has finally exited, at the next start.
        """
        monkeypatch.setattr(runner, "LOG_CAP_BYTES", 4096)
        monkeypatch.setattr(runner, "LOG_KEEP_BYTES", 1024)

        # A real live process this supervisor knows nothing about.
        child = subprocess.Popen(sleeper(30))
        try:
            job = store.create(JobKind.PREPARE, {})
            path = log_path_for(tmp_path, job.id)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"".join(f"line {i}\n".encode() for i in range(20_000)))
            job.log_path = str(path)
            store.update(job)
            store.mark_started(
                job.id, pid=child.pid, pid_created_at=psutil.Process(child.pid).create_time()
            )

            sup = Supervisor(store=store, home=tmp_path, poll_seconds=0.05)
            assert sup.reconcile() == [], "a live job must not be marked interrupted"
            assert job.id not in sup.running_ids, "it must not have been adopted"

            oversized = path.stat().st_size
            sup.tick()

            assert path.stat().st_size < oversized, "an unadopted job's log was never bounded"
            assert path.read_text(encoding="utf-8").startswith("[bloomery]")
        finally:
            child.kill()
            child.wait(timeout=10)

    def test_no_running_job_falls_off_the_end_of_the_query(
        self, store: JobStore, tmp_path: Path
    ) -> None:
        """running() used to cap at 1000, and results come back newest first.

        So past the cap it dropped the *oldest* running rows — the long-lived
        jobs, which are both the ones with the largest logs and the ones
        reconcile() most needs to see.
        """
        first = store.create(JobKind.PREPARE, {})
        store.mark_started(first.id, pid=1, pid_created_at=1.0)
        for _ in range(1200):
            job = store.create(JobKind.PREPARE, {})
            store.mark_started(job.id, pid=2, pid_created_at=1.0)

        running = store.running()

        assert len(running) == 1201
        assert first.id in {job.id for job in running}, "the oldest running job was dropped"

    def test_the_supervisor_bounds_a_log_when_the_job_exits(
        self, store: JobStore, tmp_path: Path, stub_command, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Runs on every platform, and is the whole of the bound on Windows.

        A job removed from _running is never looked at again, so if its final
        output is not trimmed here it stays on disk forever.
        """
        monkeypatch.setattr(runner, "LOG_CAP_BYTES", 4096)
        monkeypatch.setattr(runner, "LOG_KEEP_BYTES", 1024)
        # Writes well past the cap and exits immediately, so nothing is trimmed
        # while it runs even where that is allowed.
        stub_command(
            lambda job: [
                sys.executable,
                "-c",
                "print('x' * 79 + '\\n', end='')\n" * 1
                + "print(('y' * 79 + '\\n') * 2000, end='')",
            ]
        )
        sup = Supervisor(store=store, home=tmp_path, poll_seconds=0.05)
        job = sup.submit(JobKind.PREPARE, {})

        # A wall-clock deadline rather than a fixed iteration count: a loaded CI
        # runner takes far longer per pass than this machine does, and a count
        # tuned here fails there for no reason to do with the code.
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            sup.tick()
            if store.get(job.id).status.terminal:
                break
            time.sleep(0.02)

        assert store.get(job.id).status.terminal, "job never finished"
        path = log_path_for(tmp_path, job.id)
        assert path.stat().st_size < 10_000, "the log was not trimmed when the job exited"
        assert path.read_text(encoding="utf-8").startswith("[bloomery]")

    @pytest.mark.skipif(
        not runner.CAN_TRIM_WHILE_WRITING,
        reason="logs are bounded at exit rather than during the run on this platform",
    )
    def test_the_supervisor_bounds_a_running_job(
        self, store: JobStore, tmp_path: Path, stub_command, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End to end: the growth path that actually fills a disk."""
        monkeypatch.setattr(runner, "LOG_CAP_BYTES", 4096)
        monkeypatch.setattr(runner, "LOG_KEEP_BYTES", 1024)
        stub_command(
            lambda job: [
                sys.executable,
                "-u",
                "-c",
                "import sys, time\n"
                "for i in range(2000):\n"
                "    print('x' * 80)\n"
                "    sys.stdout.flush()\n"
                "    time.sleep(0.001)\n",
            ]
        )
        sup = Supervisor(store=store, home=tmp_path, poll_seconds=0.05)
        job = sup.submit(JobKind.PREPARE, {})

        path = log_path_for(tmp_path, job.id)
        peak = 0
        # A wall-clock deadline rather than a fixed iteration count, which was
        # tuned on a fast machine and timed out on a loaded CI runner.
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            sup.tick()
            if path.exists():
                peak = max(peak, path.stat().st_size)
            if store.get(job.id).status.terminal:
                break
            time.sleep(0.02)

        assert store.get(job.id).status.terminal, "job never finished"
        # Uncompacted this run writes ~160 KiB. The bound is the cap plus
        # whatever accumulates between passes, not the cap exactly.
        assert peak < 100_000, f"log reached {peak} bytes despite a 4 KiB cap"
        assert path.stat().st_size < 100_000


class TestOutputEncoding:
    """Redirected output must not depend on the platform's locale codec.

    Windows uses cp1252 whenever stdout is not a console, which is exactly the
    case when output is redirected to a file or captured by the job runner. The
    reports contain "→" and "×", none of which cp1252 can encode, so commands
    died with UnicodeEncodeError partway through printing instead of doing their
    work. Reproducible anywhere by forcing the codec.
    """

    def test_reports_survive_a_legacy_codec(self, tmp_path: Path) -> None:
        import subprocess

        env = dict(**__import__("os").environ)
        env["PYTHONIOENCODING"] = "cp1252"
        env["BLOOMERY_HOME"] = str(tmp_path / "home")
        env["NO_COLOR"] = "1"
        log = tmp_path / "out.log"

        with log.open("wb") as handle:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "bloomery.cli",
                    "prepare",
                    "--name",
                    "enc",
                    "--synthetic",
                    "200",
                    "--vocab",
                    "300",
                ],
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=env,
                check=False,
                timeout=300,
            )

        output = log.read_text(encoding="utf-8", errors="replace")
        assert result.returncode == 0, output
        assert "UnicodeEncodeError" not in output

    def test_the_runner_pins_the_child_encoding(self) -> None:
        env = runner.build_environment(ResourceRequest())
        assert env["PYTHONIOENCODING"] == "utf-8"
