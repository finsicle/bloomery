# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Apple Silicon GPU detection.

Apple GPUs have no dedicated VRAM — the GPU addresses the same unified memory as
the CPU. What limits a training run is therefore the recommended working-set
size the driver advertises, not a separate pool.

``iogpu.wired_limit_mb``, when set, caps how much of that memory can be wired
for GPU use. Left at 0 the system picks roughly 75% of physical RAM. We report
the effective figure rather than total RAM, because reporting 128 GB on a 128 GB
Mac would promise memory a training run cannot actually have.
"""

from __future__ import annotations

import sys

from bloomery.probe.types import GpuInfo, Vendor
from bloomery.probe.util import parse_int, run

# Fraction of physical memory macOS will wire for the GPU by default.
_DEFAULT_WIRED_FRACTION = 0.75


def _sysctl(key: str) -> str | None:
    result = run(["sysctl", "-n", key], timeout=3)
    if result.ok and result.stdout.strip():
        return result.stdout.strip()
    return None


def chip_name() -> str | None:
    """The marketing name, e.g. ``Apple M3 Max``."""
    return _sysctl("machdep.cpu.brand_string")


def probe() -> list[GpuInfo]:
    """Detect an Apple Silicon GPU. Empty on Intel Macs and everything else."""
    if sys.platform != "darwin":
        return []

    # Non-zero only on Apple Silicon.
    if parse_int(_sysctl("hw.optional.arm64")) != 1:
        return []

    total_ram = parse_int(_sysctl("hw.memsize"))
    usable: int | None = None
    if total_ram:
        wired_mb = parse_int(_sysctl("iogpu.wired_limit_mb")) or 0
        usable = (
            wired_mb * 1024 * 1024 if wired_mb > 0 else int(total_ram * _DEFAULT_WIRED_FRACTION)
        )

    name = chip_name() or "Apple Silicon GPU"
    cores = parse_int(_sysctl("hw.perflevel0.physicalcpu"))

    return [
        GpuInfo(
            index=0,
            vendor=Vendor.APPLE,
            name=name,
            vram_total=usable,
            vram_free=None,
            arch=f"apple-silicon-p{cores}" if cores else "apple-silicon",
            source="sysctl",
        )
    ]
