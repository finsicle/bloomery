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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

log = logging.getLogger(__name__)

ENV_PRECISION = "BLOOMERY_PRECISION"

# Memoised because the timing probe costs up to a second on Metal — negligible
# once per training run, not once per call. See _cache_key for the identity used.
_PRECISION_CACHE: dict[str, Any] = {}


def clear_precision_cache() -> None:
    """Forget memoised precision decisions. For tests."""
    _PRECISION_CACHE.clear()


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


def _bf16_works(device: torch.device) -> bool:
    """Run a real bf16 matmul and see whether it produces finite numbers."""
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

    # CPU bf16 is where the correct-but-slow trap lives. See _bf16_is_faster.
    if _bf16_works(device) and _bf16_is_faster(device):
        return torch.bfloat16, True, "cpu with accelerated bf16"
    return torch.float32, False, "cpu without accelerated bf16; using fp32"


def choose(prefer: str | None = None) -> DeviceChoice:
    """Select a device and precision together."""
    device = select_device(prefer)
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
