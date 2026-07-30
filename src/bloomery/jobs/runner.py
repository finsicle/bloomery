# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Launching, limiting and killing job processes.

Each job runs the ordinary command line in a child process — ``python -m
bloomery.cli train ...`` — rather than a second, parallel execution path. The
CLI is already covered by tests, and a job that behaves differently from the
command a user would type is a job whose failures nobody can reproduce.

The separate process is not incidental. ``CUDA_VISIBLE_DEVICES`` is read when
torch is imported and cannot be changed afterwards, so per-job GPU selection is
only possible by setting it before the interpreter starts.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

from bloomery.jobs.types import Job, JobKind, ResourceRequest

# How long a cancelled process is given to exit on its own before being killed.
# Long enough for the trainer to finish writing the checkpoint it is midway
# through, short enough that a cancel feels like a cancel.
TERMINATE_GRACE_SECONDS = 10.0


class JobLaunchError(RuntimeError):
    """The job could not be started at all."""


@dataclass(frozen=True, slots=True)
class Launched:
    process: subprocess.Popen[bytes]
    command: list[str]
    limits_note: str | None

    @property
    def pid(self) -> int:
        return self.process.pid

    def created_at(self) -> float:
        """Process creation time, used with the PID to prove identity later.

        A PID on its own is not evidence: they are reused, and a supervisor that
        restarts an hour later must not mistake an unrelated process for the
        worker it lost.
        """
        try:
            return psutil.Process(self.process.pid).create_time()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0


# Parameter name -> command-line flag, per job kind. Declarative so that a
# request coming from the API cannot introduce an argument the CLI does not
# already accept.
_FLAGS: dict[JobKind, dict[str, str]] = {
    JobKind.PREPARE: {
        "name": "--name",
        "source": "--source",
        "synthetic": "--synthetic",
        "vocab": "--vocab",
        "val_fraction": "--val-fraction",
    },
    JobKind.TRAIN: {
        "data": "--data",
        "mix": "--mix",
        "mix_version": "--mix-version",
        "name": "--name",
        "depth": "--depth",
        "size": "--size",
        "steps": "--steps",
        "batch": "--batch",
        "seq": "--seq",
        "grad_accum": "--grad-accum",
        "lr": "--lr",
        "eval_every": "--eval-every",
        "save_every": "--save-every",
        "cores": "--cores",
        "device": "--device",
        "seed": "--seed",
    },
    JobKind.BENCH: {
        "size": "--size",
        "depth": "--depth",
        "vocab": "--vocab",
        "batch": "--batch",
        "seq": "--seq",
        "steps": "--steps",
        "cores": "--cores",
        "device": "--device",
    },
}

# Boolean parameters, emitted as a bare flag when true.
_SWITCHES: dict[JobKind, dict[str, str]] = {
    JobKind.PREPARE: {},
    JobKind.TRAIN: {
        "grad_checkpoint": "--grad-checkpoint",
        "resume": "--resume",
        "force": "--force",
    },
    JobKind.BENCH: {"grad_checkpoint": "--grad-checkpoint"},
}


def build_command(job: Job) -> list[str]:
    """Turn a job's parameters into an argv for the CLI.

    Only parameters named in the tables above are passed through. An unknown key
    is ignored rather than forwarded, so a malformed or hostile request cannot
    reach the command line — and because this is argv rather than a shell string,
    there is no quoting to get wrong either.
    """
    flags = _FLAGS.get(job.kind)
    if flags is None:
        raise JobLaunchError(f"no command mapping for job kind {job.kind!r}")

    command = [sys.executable, "-m", "bloomery.cli", job.kind.value]
    for key, flag in flags.items():
        value = job.params.get(key)
        if value is None or value == "":
            continue
        command.extend([flag, str(value)])
    for key, flag in _SWITCHES[job.kind].items():
        if job.params.get(key):
            command.append(flag)
    return command


def build_environment(request: ResourceRequest) -> dict[str, str]:
    """Environment for the worker, with device and thread limits applied.

    Both visible-device variables are set. Which one matters depends on whether
    torch turns out to be a CUDA or a ROCm build, and that is not known until it
    imports — which is after this point.
    """
    env = dict(os.environ)

    # An empty selection means "whatever the machine has", which is the
    # inherited default. Setting the variable to an empty string would instead
    # hide every GPU, so the no-selection case must leave it alone.
    if request.gpus:
        visible = ",".join(str(i) for i in request.gpus)
        env["CUDA_VISIBLE_DEVICES"] = visible
        env["HIP_VISIBLE_DEVICES"] = visible

    if request.cores and request.cores > 0:
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[name] = str(request.cores)

    # Child output is a pipe, not a terminal, so rich would strip colour anyway;
    # being explicit keeps the captured log free of escape sequences.
    env["NO_COLOR"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    # Belt and braces with the CLI's own stream reconfiguration. Redirected
    # output on Windows otherwise defaults to the locale codec, which cannot
    # encode the arrows the reports use.
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _systemd_run_prefix(memory_bytes: int, cores: int | None) -> list[str] | None:
    """A ``systemd-run`` wrapper that applies a real memory cap, if available.

    cgroups are the only mechanism here that caps memory without breaking
    accelerated training. The obvious alternative, ``RLIMIT_AS``, limits virtual
    address space rather than resident memory, and CUDA reserves enormous
    amounts of address space it never touches — capping it kills the process for
    doing something entirely normal.

    Returns None when systemd is not usable, in which case the caller records
    that no cap is in force rather than pretending otherwise.
    """
    if sys.platform != "linux" or shutil.which("systemd-run") is None:
        return None
    prefix = [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        "-p",
        f"MemoryMax={memory_bytes}",
        # Without this the cap is soft: the kernel swaps instead of failing, and
        # a training run that starts swapping is slower than one that stopped.
        "-p",
        "MemorySwapMax=0",
    ]
    if cores and cores > 0:
        prefix.extend(["-p", f"CPUQuota={cores * 100}%"])
    return prefix


def launch(
    job: Job,
    request: ResourceRequest,
    *,
    log_path: Path,
    cwd: Path | None = None,
) -> Launched:
    """Start a job's process and return a handle to it."""
    command = build_command(job)
    env = build_environment(request)

    limits_note: str | None = None
    if request.memory_bytes:
        prefix = _systemd_run_prefix(request.memory_bytes, request.cores)
        if prefix is None:
            limits_note = (
                "memory cap not applied: cgroups via systemd-run are only available "
                "on Linux. CPU threads are still limited."
            )
        else:
            command = [*prefix, *command]

    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("ab", buffering=0)

    # A new session (POSIX) or process group (Windows) means the whole tree can
    # be signalled as one. Without it, killing the worker leaves any dataloader
    # or systemd-run child behind.
    creation: dict[str, Any] = {}
    if os.name == "nt":
        # Only defined on Windows, so it is fetched dynamically rather than
        # referenced directly — a bare attribute access fails type checking on
        # every other platform.
        creation["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        creation["start_new_session"] = True

    try:
        process = subprocess.Popen(  # noqa: S603 - argv built from a fixed table
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            cwd=str(cwd) if cwd else None,
            **creation,
        )
    except OSError as exc:
        handle.close()
        raise JobLaunchError(f"could not start {command[0]}: {exc}") from exc
    finally:
        # Popen dups the descriptor; the parent's copy is not needed and would
        # otherwise keep the log file open for the lifetime of the supervisor.
        handle.close()

    return Launched(process=process, command=command, limits_note=limits_note)


def _family(pid: int) -> list[psutil.Process]:
    """A process and its descendants, children first.

    Children first so a worker cannot outlive a dataloader it spawned.
    """
    try:
        parent = psutil.Process(pid)
        return [*parent.children(recursive=True), parent]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return []


def terminate_all(pids: list[int], *, grace: float = TERMINATE_GRACE_SECONDS) -> int:
    """Stop several process trees, paying the grace period once rather than per tree.

    Everything is signalled first and waited on together. Cancelling four jobs
    one at a time would otherwise cost four full grace periods in sequence.
    Returns how many processes were signalled.
    """
    family: list[psutil.Process] = []
    for pid in pids:
        family.extend(_family(pid))
    if not family:
        return 0

    for member in family:
        try:
            member.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    _, alive = psutil.wait_procs(family, timeout=grace)
    for member in alive:
        try:
            member.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    psutil.wait_procs(alive, timeout=5)
    return len(family)


def terminate(pid: int, *, grace: float = TERMINATE_GRACE_SECONDS) -> bool:
    """Stop one job's process tree. Returns whether anything was signalled.

    Everything is asked politely before being killed, which gives the trainer a
    chance to finish writing a checkpoint it is midway through.
    """
    return terminate_all([pid], grace=grace) > 0


def is_alive(pid: int | None, created_at: float | None) -> bool:
    """Whether the recorded process is still running *and* still the same one.

    Both halves matter. PIDs are reused, so a supervisor restarting after a
    reboot would otherwise adopt whatever now happens to hold that number.
    """
    if pid is None:
        return False
    try:
        process = psutil.Process(pid)
        if created_at is not None and abs(process.create_time() - created_at) > 1.0:
            return False
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False
