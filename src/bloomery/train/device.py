# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Choosing a device and a compute dtype.

The dtype decision is made by *measuring* rather than by consulting a table of
what each backend claims to support. Backend coverage — especially Metal's —
changes between PyTorch releases, so a hardcoded assumption either crashes on a
machine that would have worked or falls back to fp32 on one that did not need to.

The probe has two parts, and the second is the one that matters. Asking whether
bf16 *works* is not enough: on hardware without native bf16 the dtype is
emulated, so every result is correct and every matmul is far slower. On an Apple
M1 that penalty is 46x on CPU (3182 ms/step against 68 ms) and 1.4x on the GPU
(94 ms against 67 ms) — an entirely silent regression that only shows up on a run
long enough to notice. So bf16 has to be both correct *and* measurably faster
before it gets used.

Set ``BLOOMERY_PRECISION=fp32|bf16|fp16`` to bypass the probe.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from bloomery.probe.backend import HSA_OVERRIDE, ROCM_SUPPORTED_ARCHS, override_hint

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

log = logging.getLogger(__name__)

ENV_PRECISION = "BLOOMERY_PRECISION"
# Set to "0" to skip the out-of-process bf16 timing probe entirely.
ENV_BF16_PROBE = "BLOOMERY_BF16_PROBE"

# Memoised because the timing probe costs up to a second on Metal — negligible
# once per training run, not once per call. See _cache_key for the identity used.
_PRECISION_CACHE: dict[str, Any] = {}
# Same reasoning, for the "can this device execute at all" probe.
_EXECUTES_CACHE: dict[str, bool] = {}


def clear_precision_cache() -> None:
    """Forget memoised precision decisions. For tests."""
    _PRECISION_CACHE.clear()
    _EXECUTES_CACHE.clear()


def _cache_key(device: torch.device) -> str:
    """Cache identity for a device.

    CUDA is keyed by concrete index rather than by ``device.type``. A box can
    hold cards of different generations — an A100 alongside a T4, say — where one
    supports bf16 natively and the other does not. Caching both under "cuda"
    would apply the first card's answer to every subsequent one.
    """
    if device.type != "cuda":
        return device.type
    try:
        import torch

        index = device.index if device.index is not None else torch.cuda.current_device()
    except Exception:  # noqa: BLE001 - fall back to the family key
        return device.type
    return f"cuda:{index}"


@dataclass(frozen=True, slots=True)
class DeviceChoice:
    device: torch.device
    dtype: torch.dtype
    # True when the dtype should be applied via autocast rather than by casting
    # the parameters themselves. Master weights stay fp32 either way.
    autocast: bool
    reason: str
    # What to do about it, when this choice was forced by something the user can
    # fix. Kept apart from ``reason`` because the reason prints inline beside the
    # device and is truncated at the terminal width — which on the first run of
    # this swallowed the environment variable, the only part anybody needed.
    remedy: str | None = None

    @property
    def type(self) -> str:
        return self.device.type

    def label(self) -> str:
        name = str(self.device)
        precision = str(self.dtype).removeprefix("torch.")
        return f"{name} ({precision})"


def select_device(prefer: str | None = None) -> torch.device:
    """Pick a device.

    ``CUDA_VISIBLE_DEVICES`` and ``HIP_VISIBLE_DEVICES`` are honoured by PyTorch
    itself, since they are read at import time — which is why bloomery launches
    training in a separate process with those already set rather than trying to
    change them in-process.
    """
    import torch

    if prefer:
        return torch.device(prefer)

    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    except AttributeError:  # pragma: no cover - very old torch
        pass
    return torch.device("cpu")


def _cpu_bf16_is_native() -> bool:
    """Whether this CPU implements bf16 in hardware — asked, never executed.

    Everything else in this module decides by running the operation and looking
    at the result. That approach cannot be used here, because the failure mode
    is not a wrong answer or an exception.

    A CPU whose bf16 kernels use instructions it does not implement does not
    return a bad number and does not raise: it takes an illegal-instruction trap
    and the process dies where it stands. On Windows that surfaces as exception
    ``0xc000001d`` and exit code 132. ``except Exception`` is no defence — a
    hardware trap is not a Python exception, and there is nothing left running
    to catch it.

    So capability is read from torch rather than discovered by executing a
    matmul, and anything that cannot be positively confirmed counts as absent.
    Being wrong in this direction costs some speed on an unrecognised CPU;
    being wrong in the other direction kills a training run at startup.
    """
    import torch

    cpu = getattr(torch, "cpu", None)
    if cpu is None:  # pragma: no cover - very old torch
        return False

    # x86: the bf16 matmul path needs AVX512-BF16 (or AMX, which implies it).
    probe = getattr(cpu, "_is_avx512_bf16_supported", None)
    if probe is not None:
        try:
            if bool(probe()):
                return True
        except Exception:  # noqa: BLE001 - absence of an answer is a "no"
            pass

    # aarch64 and anything else torch can describe. Apple silicon reports
    # bf16=False here, which is why the probe must never have run there.
    capabilities = getattr(cpu, "get_capabilities", None)
    if capabilities is not None:
        try:
            reported = capabilities()
            return bool(reported.get("bf16") or reported.get("sve_bf16"))
        except Exception:  # noqa: BLE001
            return False

    return False


def _bf16_works(device: torch.device) -> bool:
    """Run a real bf16 matmul and see whether it produces finite numbers.

    Only safe once the device is known to support bf16: on a CPU that does not,
    this matmul is fatal rather than wrong. See :func:`_cpu_bf16_is_native`.
    """
    import torch

    try:
        a = torch.ones((8, 8), device=device, dtype=torch.bfloat16)
        result = (a @ a).float().sum().item()
    except Exception:  # noqa: BLE001 - any failure means "not usable"
        return False
    return bool(result == 512.0)


def _synchronize(device: torch.device) -> None:
    """Block until queued work finishes, so a timing measurement means something."""
    import torch

    try:
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
    except Exception:  # noqa: BLE001
        pass


def _bf16_is_faster(device: torch.device, *, size: int = 384, reps: int = 5) -> bool:
    """Whether bf16 matmuls actually beat fp32 on this device.

    Correctness is not the same as usefulness. On a CPU without hardware bf16
    support the dtype is emulated: every result is right and every matmul is
    dramatically slower. Measured on an Apple M1 CPU, bf16 autocast ran a
    training step in 3182 ms against 68 ms for fp32 — a 46x penalty that a short
    test run would never reveal.

    So the decision is made by timing both, not by asking whether bf16 exists.
    Both paths are warmed first, because a cold Metal shader compile or a cold
    allocator would otherwise dominate whichever ran first.
    """
    import time

    import torch

    try:
        fp32 = torch.randn((size, size), device=device, dtype=torch.float32)
        bf16 = fp32.to(torch.bfloat16)

        for _ in range(2):
            _ = fp32 @ fp32
            _ = bf16 @ bf16
        _synchronize(device)

        started = time.perf_counter()
        for _ in range(reps):
            _ = fp32 @ fp32
        _synchronize(device)
        fp32_seconds = time.perf_counter() - started

        started = time.perf_counter()
        for _ in range(reps):
            _ = bf16 @ bf16
        _synchronize(device)
        bf16_seconds = time.perf_counter() - started
    except Exception:  # noqa: BLE001 - treat any failure as "do not use bf16"
        return False

    # A 10% margin: bf16 halves memory traffic, so parity on a microbenchmark
    # still tends to win on a real model. Anything slower than that is emulation.
    return bf16_seconds <= fp32_seconds * 1.1


def _bf16_probe_says_faster(device: torch.device, *, timeout: float = 180.0) -> bool:
    """Run the timing probe in a child process and read its verdict.

    The child may die from a hardware trap; that is expected, and is exactly why
    it is a child. Any outcome other than a clean "bf16" is treated as "do not
    use bf16", which is the answer a crash was implying anyway.

    ``BLOOMERY_BF16_PROBE=0`` skips it and answers no, for anyone who would
    rather not spend the subprocess at all.
    """
    if os.environ.get(ENV_BF16_PROBE, "").strip() == "0":
        return False

    command = [sys.executable, "-m", "bloomery.train._bf16_probe", str(device)]
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        log.debug("bf16 probe could not run; assuming bf16 is not usable")
        return False

    if completed.returncode != 0:
        # Includes the illegal-instruction case. Negative on Unix (killed by a
        # signal), 132 and friends on Windows.
        log.info("bf16 probe exited with %s on %s; using fp32", completed.returncode, device)
        return False
    return completed.stdout.strip().splitlines()[-1:] == ["bf16"]


def select_precision(device: torch.device) -> tuple[torch.dtype, bool, str]:
    """Choose a compute dtype for this device.

    Returns ``(dtype, use_autocast, reason)``. Memoised per device identity: the
    timing probe costs up to a second on Metal, which is negligible once per
    training run but not once per call.

    ``BLOOMERY_PRECISION=fp32|bf16|fp16`` overrides the probe entirely, for when
    it gets the answer wrong on hardware we have not seen.
    """
    import torch

    key = _cache_key(device)
    cached = _PRECISION_CACHE.get(key)
    if cached is not None:
        return cached

    override = os.environ.get(ENV_PRECISION, "").strip().lower()
    if override:
        forced = {
            "fp32": (torch.float32, False),
            "float32": (torch.float32, False),
            "bf16": (torch.bfloat16, True),
            "bfloat16": (torch.bfloat16, True),
            "fp16": (torch.float16, True),
            "float16": (torch.float16, True),
        }.get(override)
        if forced is not None:
            result = (forced[0], forced[1], f"forced by {ENV_PRECISION}={override}")
            _PRECISION_CACHE[key] = result
            return result
        log.warning("ignoring unrecognised %s=%r", ENV_PRECISION, override)

    result = _probe_precision(device)
    _PRECISION_CACHE[key] = result
    return result


def _probe_precision(device: torch.device) -> tuple[torch.dtype, bool, str]:
    import torch

    if device.type == "cuda":
        # including_emulation defaults to True, which would accept a card that
        # merely emulates bf16 — the exact trap this module exists to avoid on
        # CPU and Metal. Probed inside the device's own context so a multi-GPU
        # box answers for the right card.
        with torch.cuda.device(device if device.index is not None else torch.cuda.current_device()):
            try:
                native = torch.cuda.is_bf16_supported(including_emulation=False)
            except TypeError:
                # Older torch has no such parameter; its answer already excluded
                # emulation.
                native = torch.cuda.is_bf16_supported()
        if native:
            return torch.bfloat16, True, "cuda with native bf16"
        return torch.float16, True, "cuda without native bf16; using fp16 autocast"

    if device.type == "mps":
        # Metal's bf16 coverage is uneven and moves between releases, so this is
        # decided by experiment rather than by a version check.
        if _bf16_works(device) and _bf16_is_faster(device):
            return torch.bfloat16, True, "metal with working bf16"
        return torch.float32, False, "metal without usable bf16; using fp32"

    # Two gates, and both are needed.
    #
    # The cheap one first: a CPU that does not claim bf16 gets fp32 without
    # spawning anything. That covers Apple silicon and most consumer x86, so the
    # common case pays nothing.
    if not _cpu_bf16_is_native():
        return torch.float32, False, "cpu without hardware bf16; using fp32"

    # Then the timing probe, in a process we can afford to lose. A CPU that
    # claims bf16 can still trap on the matmul — a Windows runner reporting
    # AVX512-BF16 survived an 8x8 bf16 matmul and died on a 384x384 one, because
    # the larger shape dispatches to a kernel needing instructions the chip did
    # not have. There is no way to ask about that in advance, and no way to
    # catch it in process, so it is asked somewhere dying is survivable.
    if _bf16_probe_says_faster(device):
        return torch.bfloat16, True, "cpu with accelerated bf16"
    return torch.float32, False, "cpu bf16 not usably faster; using fp32"


class DeviceUnusableError(RuntimeError):
    """A device was asked for by name and cannot execute anything."""


def unsupported_rocm_arch(device: torch.device) -> str | None:
    """This device's gfx name, when there is reason to doubt it can be used.

    ``None`` for anything else, including every NVIDIA card — ``gcnArchName`` is
    a ROCm-only property, so this costs one attribute read on a CUDA box and
    nothing at all on CPU or Metal.

    A cheap gate in front of an expensive probe, the same shape as the CPU path:
    ask the free question first, and spend a subprocess only where the answer
    gives reason to doubt.

    Two such reasons. The first is the obvious one: an architecture outside
    ROCm's supported set. The second is that ``HSA_OVERRIDE_GFX_VERSION`` is set
    at all — because making the card claim a different architecture is precisely
    what that variable does, so the name torch reports is no longer evidence
    about the hardware. An RX 6700 XT with the override at 11.0.0 reports itself
    as gfx1100, which is on the supported list, and then hangs on its first real
    work. Asking torch cannot catch that; running something can.
    """
    import torch

    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    arch = str(getattr(torch.cuda.get_device_properties(index), "gcnArchName", "") or "")
    arch = arch.split(":")[0].strip().lower()
    if not arch:
        return None
    if arch in ROCM_SUPPORTED_ARCHS and not os.environ.get(HSA_OVERRIDE, "").strip():
        return None
    return arch


def _device_executes(device: torch.device, *, timeout: float = 60.0) -> bool:
    """Run one operation on this device in a child process, and see if it lives.

    Only called for hardware :func:`unsupported_rocm_arch` has flagged, so a
    supported card never pays for it.

    Sixty seconds is generous by a wide margin. Measured on an RX 6700 XT: 2.5
    seconds when the card works, nearly all of it creating the device context.
    The timeout is what bounds the other case, where a misconfigured card hangs
    forever — that path refuses after about 65 seconds, which is slow for an
    error message and finite, where before it was neither.

    Erring long is deliberate. Refusing a working GPU because a cold context took
    longer than expected is a worse failure than a slow refusal of a broken one.
    """
    key = _cache_key(device)
    cached = _EXECUTES_CACHE.get(key)
    if cached is not None:
        return cached

    command = [sys.executable, "-m", "bloomery.train._device_probe", str(device)]
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
        alive = completed.returncode == 0
        if not alive:
            log.info("device probe exited with %s on %s", completed.returncode, device)
    except subprocess.TimeoutExpired:
        # A timeout is an answer, not a failure to get one. An RX 6700 XT given
        # an override naming the wrong architecture does not crash — it hangs,
        # and so does everything downstream of it. This branch used to share the
        # OSError case below and wave the device through, so the training run
        # inherited the hang and sat there indefinitely.
        log.info("device probe timed out on %s; treating it as unusable", device)
        alive = False
    except OSError:
        # Genuinely could not ask: no process could be spawned. Unlike the case
        # above, that says nothing about the device, and calling it dead would
        # refuse a working GPU because the machine was briefly out of handles.
        log.debug("device probe could not run on %s; assuming it works", device)
        alive = True

    _EXECUTES_CACHE[key] = alive
    return alive


def choose(prefer: str | None = None) -> DeviceChoice:
    """Select a device and precision together.

    An accelerator that cannot execute anything is caught here rather than at
    the first training step. It is a real state and not a hypothetical one: an
    unsupported AMD card without ``HSA_OVERRIDE_GFX_VERSION`` answers yes to
    every question torch can be asked and then segfaults on its first matmul.
    Asked for by name, that is refused; chosen on the user's behalf, it falls
    back to the CPU and says why.
    """
    device = select_device(prefer)

    arch = unsupported_rocm_arch(device)
    if arch is not None and not _device_executes(device):
        override = override_hint(arch)
        setting = os.environ.get(HSA_OVERRIDE, "").strip()
        if setting:
            # Already set, so "set it" is no help — and the arch above came from
            # torch, which the override has made unreliable. doctor reads the
            # real one out of sysfs, so send them there rather than guessing.
            remedy = f"{HSA_OVERRIDE}={setting} may be wrong for this card; run bloomery doctor"
        elif override:
            remedy = f"set {HSA_OVERRIDE}={override} to use this GPU"
        else:
            remedy = f"check ROCm's compatibility matrix for {arch}"
        if prefer:
            raise DeviceUnusableError(
                f"{prefer} was requested, but this {arch} GPU cannot execute anything.\n"
                "Every operation on it takes the process down, so the run would "
                "die on its first step.\n"
                f"  {remedy}"
            )
        import torch

        device = torch.device("cpu")
        dtype, autocast, _ = select_precision(device)
        return DeviceChoice(
            device=device,
            dtype=dtype,
            autocast=autocast,
            # Short enough to survive being printed beside the device name.
            reason=f"{arch} GPU cannot execute; using cpu",
            remedy=remedy,
        )

    dtype, autocast, reason = select_precision(device)
    return DeviceChoice(device=device, dtype=dtype, autocast=autocast, reason=reason)


def thread_limit(cores: int | None) -> None:
    """Cap CPU threads for this process.

    Applied before the first heavy op. This is what "allocate N cores" means on
    every platform — unlike a memory cap, it needs no cgroups or job objects, so
    it is the one resource control that behaves identically everywhere.
    """
    import torch

    if not cores or cores < 1:
        return
    os.environ.setdefault("OMP_NUM_THREADS", str(cores))
    os.environ.setdefault("MKL_NUM_THREADS", str(cores))
    torch.set_num_threads(cores)
