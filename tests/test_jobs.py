# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the job store, runner and queue.

Scheduling is exercised against a stubbed command so the tests are fast and
deterministic; one integration test at the end runs a real job end to end, since
a queue that has never actually started a process proves very little.
"""

from __future__ import annotations

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
