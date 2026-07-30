# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Choosing a device and a compute dtype.

The dtype decision is made by *trying* the operation rather than by consulting a
table of what each backend supports. Backend coverage — especially Metal's —
changes between PyTorch releases, and a hardcoded assumption produces either a
crash on a machine that would have worked or fp32 on a machine that did not need
it. A one-off matmul costs microseconds and is always right.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch


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


def select_precision(device: torch.device) -> tuple[torch.dtype, bool, str]:
    """Choose a compute dtype for this device.

    Returns ``(dtype, use_autocast, reason)``.
    """
    import torch

    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, True, "cuda with bf16 support"
        return torch.float16, True, "cuda without bf16; using fp16 autocast"

    if device.type == "mps":
        # Metal's bf16 coverage is uneven and moves between releases, so this is
        # decided by experiment. fp32 is slower but always correct, and at the
        # sizes that fit on a Mac the difference is not what limits you.
        if _bf16_works(device):
            return torch.bfloat16, True, "metal with working bf16"
        return torch.float32, False, "metal without usable bf16; using fp32"

    if _bf16_works(device):
        return torch.bfloat16, True, "cpu with bf16 support"
    return torch.float32, False, "cpu; using fp32"


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
