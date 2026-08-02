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

import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

from bloomery.jobs.types import Job, JobKind, ResourceRequest

log = logging.getLogger(__name__)

# How far a process's reported creation time may drift from the recorded one and
# still be considered the same process. Filesystem and clock granularity vary by
# platform, so an exact match is not reliable.
PID_IDENTITY_TOLERANCE = 1.0

# How long a cancelled process is given to exit on its own before being killed.
# Long enough for the trainer to finish writing the checkpoint it is midway
# through, short enough that a cancel feels like a cancel.
TERMINATE_GRACE_SECONDS = 10.0

# Bounds on a job's log. A training run writes to one file for hours and nothing
# ever rotated it, so a single job could fill the disk — and when it did, it took
# the rest of the machine's work with it rather than just its own.
#
# The cap is generous because the log is the only account of what a run did, and
# the kept tail is what a person actually reads: 8 MiB is on the order of a
# hundred thousand lines. What gets dropped is the middle of a long run, which is
# repetitive step output. The run's configuration is not lost with it — that
# lives in the job record, not only in the log.
LOG_CAP_BYTES = 32 * 1024 * 1024
LOG_KEEP_BYTES = 8 * 1024 * 1024

# Whether a log can be trimmed while the job is still writing to it.
#
# It can wherever O_APPEND means what POSIX says: append mode belongs to the
# open file description, so every write goes to the file's current end no matter
# what happened to it in between. Truncating underneath a running job is then
# safe — its next line simply arrives at the new end.
#
# Windows does not give that for an inherited handle. There, append is emulated
# by the C runtime seeking to the end before each write, inside the process that
# opened the file. The child receives the handle as its stdout and writes through
# it directly, carrying its own file pointer. Truncate underneath it and the next
# write lands at the old offset: the gap between is filled with NULs and the file
# is immediately back to the size it was, so the trim achieves nothing and
# corrupts the log on the way. Confirmed on the Windows matrix, which is why the
# check is here rather than in a comment.
#
# So on Windows a log is only trimmed once its job has exited and nothing holds
# the file. That leaves a run unbounded while it is going, which is no worse than
# before this existed, and native Windows is best-effort here anyway — under WSL2
# this is Linux and gets the full behaviour.
CAN_TRIM_WHILE_WRITING = os.name != "nt"


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

    def created_at(self) -> float | None:
        """Process creation time, used with the PID to prove identity later.

        A PID on its own is not evidence: they are reused, and a supervisor that
        restarts an hour later must not mistake an unrelated process for the
        worker it lost.

        None when the identity could not be read — AccessDenied can happen while
        the process is perfectly alive. Returning 0.0 for that case was worse
        than useless: it stored a plausible-looking number that no real creation
        time can match, so the job became both unkillable and permanently
        misreported as interrupted.
        """
        try:
            return psutil.Process(self.process.pid).create_time()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None


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
    JobKind.ADAPT: {
        "from": "--from",
        "data": "--data",
        "mix": "--mix",
        "mix_version": "--mix-version",
        "name": "--name",
        "method": "--method",
        "lora_r": "--lora-r",
        "lora_alpha": "--lora-alpha",
        "lora_dropout": "--lora-dropout",
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
    JobKind.ADAPT: {
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


def _readable(count: int) -> str:
    """A byte count a person can read, at whatever scale it happens to be.

    Fixed MiB read as "0 MiB" for anything small, which made the marker in a
    trimmed log say it had dropped nothing.
    """
    size = float(count)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GiB"  # pragma: no cover - the loop returns first


def compact_log(
    path: Path,
    *,
    cap: int | None = None,
    keep: int | None = None,
    still_writing: bool = False,
) -> int:
    """Trim an oversized log in place, keeping its most recent lines.

    Returns the number of bytes dropped, or 0 if nothing needed doing.

    Pass ``still_writing`` when the job that owns this log is alive. On a
    platform where truncating underneath a writer is unsafe the call then does
    nothing, rather than corrupting the log to enforce a limit — see
    ``CAN_TRIM_WHILE_WRITING``.

    The bounds are read at call time rather than bound as default arguments, so
    that setting ``runner.LOG_CAP_BYTES`` actually takes effect. A default
    argument would capture the value at import and quietly ignore every later
    change to it.

    Rewritten in place rather than replaced. A running job holds an open
    descriptor to this file: swapping in a new one by rename would leave the job
    writing to an unlinked inode, and everything it logged from then on would go
    nowhere. Truncating the file the job already has is what keeps its output
    arriving.

    A line the job writes in the instant between measuring and truncating is
    lost — discarded whole, never spliced, because the rewrite only ever shrinks
    the file. The window is microseconds against a cap measured in tens of
    megabytes, and the alternative is stopping the job to take a lock on its own
    diagnostic output.
    """
    if still_writing and not CAN_TRIM_WHILE_WRITING:
        return 0

    cap = LOG_CAP_BYTES if cap is None else cap
    keep = LOG_KEEP_BYTES if keep is None else keep

    try:
        size = path.stat().st_size
    except OSError:
        return 0
    if size <= cap:
        return 0

    try:
        with path.open("r+b") as handle:
            handle.seek(size - keep)
            tail = handle.read(keep)
            # The window opens mid-line; that fragment belongs to a line whose
            # beginning is being dropped, so it would read as corruption.
            newline = tail.find(b"\n")
            if newline != -1:
                tail = tail[newline + 1 :]

            # It has to end on a line boundary too. A job's `print` is not one
            # write syscall — the text and its newline can go separately — so
            # the size measured above may fall between them. Ending mid-line
            # glues the next line the job writes onto this one, which is how
            # "line 0085line 0086" appears in a log that never contained it.
            if tail and not tail.endswith(b"\n"):
                end = tail.rfind(b"\n")
                if end != -1:
                    tail = tail[: end + 1]
                else:
                    # No boundary anywhere in the window: one line longer than
                    # it. A progress display that only ever redrew with bare
                    # carriage returns looks exactly like this. Terminate the
                    # fragment rather than discard the only output there is —
                    # the marker above it already says the beginning went.
                    tail += b"\n"

            marker = (
                f"[bloomery] {_readable(size - len(tail))} of earlier output was dropped "
                f"to keep this log under {_readable(cap)}. What follows is the most "
                f"recent output; anything the job writes from here on is appended "
                f"below.\n"
            ).encode()

            # The rewrite must only ever shrink the file. If marker + tail were
            # longer than what is on disk — which a small cap makes easy — the
            # write would extend it, and a line the job appends meanwhile would
            # land inside the region still being written and come back spliced.
            # Shrinking means a concurrent append always lands past the end of
            # the new content, where truncate discards it whole.
            overflow = len(marker) + len(tail) - size
            if overflow > 0:
                cut = tail.find(b"\n", overflow)
                tail = tail[cut + 1 :] if cut != -1 else b""

            dropped = size - len(tail)
            handle.seek(0)
            handle.write(marker + tail)
            handle.truncate()
    except OSError:
        # Losing the log is not worth losing the job over.
        log.warning("could not compact %s", path, exc_info=True)
        return 0
    return dropped


def _same_process(process: psutil.Process, created_at: float | None) -> bool:
    """Whether this is really the process that was recorded, not a reused pid.

    ``created_at`` of None means the caller is not asking for a check, which is
    only safe when it holds the process handle itself. The supervisor never
    passes None for a pid recovered from the store — see Supervisor.cancel.
    """
    if created_at is None:
        return True
    try:
        return abs(process.create_time() - created_at) <= PID_IDENTITY_TOLERANCE
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def _family(pid: int, created_at: float | None) -> list[psutil.Process]:
    """A process and its descendants, children first, or nothing if the pid was reused.

    Children first so a worker cannot outlive a dataloader it spawned. The
    identity check is the important part: pids are recycled, and a stale record
    pointed at a recycled number would otherwise send SIGKILL to whatever the
    machine has since put there.
    """
    try:
        parent = psutil.Process(pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return []
    if not _same_process(parent, created_at):
        return []
    try:
        return [*parent.children(recursive=True), parent]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return []


def terminate_all(
    targets: Sequence[tuple[int, float | None]],
    *,
    grace: float = TERMINATE_GRACE_SECONDS,
) -> int:
    """Stop several process trees, paying the grace period once rather than per tree.

    Each target is the pid together with the creation time recorded when it was
    started, so a recycled pid is skipped rather than killed. Everything is
    signalled first and waited on together: cancelling four jobs one at a time
    would otherwise cost four full grace periods in sequence.

    Returns how many processes were signalled.
    """
    family: list[psutil.Process] = []
    for pid, created_at in targets:
        family.extend(_family(pid, created_at))
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


def terminate(
    pid: int,
    created_at: float | None = None,
    *,
    grace: float = TERMINATE_GRACE_SECONDS,
) -> bool:
    """Stop one job's process tree. Returns whether anything was signalled.

    Everything is asked politely before being killed, which gives the trainer a
    chance to finish writing a checkpoint it is midway through.
    """
    return terminate_all([(pid, created_at)], grace=grace) > 0


def is_alive(pid: int | None, created_at: float | None) -> bool:
    """Whether the recorded process is still running *and* still the same one.

    Both halves matter. PIDs are reused, so a supervisor restarting after a
    reboot would otherwise adopt whatever now happens to hold that number.
    """
    if pid is None:
        return False
    try:
        process = psutil.Process(pid)
        if not _same_process(process, created_at):
            return False
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False
