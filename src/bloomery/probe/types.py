# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Data types for the host probe.

Deliberately plain dataclasses rather than pydantic models. The probe has to
work on a bare install with no ML or web dependencies present, because it is
the tool you reach for when nothing else works yet.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import Any

MIB = 1024 * 1024
GIB = 1024 * 1024 * 1024


class Vendor(StrEnum):
    """GPU vendor, as detected."""

    NVIDIA = "nvidia"
    AMD = "amd"
    APPLE = "apple"
    INTEL = "intel"


class Platform(StrEnum):
    """Host platform, at the granularity that changes our behaviour."""

    LINUX = "linux"
    WSL2 = "wsl2"
    WINDOWS = "windows"
    MACOS = "macos"
    UNKNOWN = "unknown"


class Backend(StrEnum):
    """A PyTorch build target.

    Values match ``uv pip install --torch-backend=<value>`` where one exists,
    so the recommendation can be handed to the installer verbatim. ``mps`` is
    the exception: Apple builds ship on plain PyPI, so uv needs ``cpu`` there
    and Metal support comes along for the ride.
    """

    CU130 = "cu130"
    CU128 = "cu128"
    CU126 = "cu126"
    CU118 = "cu118"
    ROCM = "rocm7.2"
    XPU = "xpu"
    MPS = "mps"
    CPU = "cpu"

    @property
    def uv_torch_backend(self) -> str:
        """The value to pass to ``--torch-backend``."""
        return "cpu" if self is Backend.MPS else self.value

    @property
    def accelerated(self) -> bool:
        return self is not Backend.CPU


@dataclass(slots=True)
class GpuInfo:
    """A single GPU.

    ``vram_total`` and ``vram_free`` are bytes, or None when the source we read
    from did not report them. Never guess a VRAM figure: downstream memory
    estimates are only honest if this is measured.
    """

    index: int
    vendor: Vendor
    name: str
    vram_total: int | None = None
    vram_free: int | None = None
    # NVIDIA: "8.9". AMD: "gfx1100". Apple: None.
    arch: str | None = None
    pci_bus_id: str | None = None
    uuid: str | None = None
    # Where this record came from, so surprising output can be traced.
    source: str = "unknown"

    @property
    def vram_gib(self) -> float | None:
        return None if self.vram_total is None else self.vram_total / GIB

    def label(self) -> str:
        if self.vram_gib is None:
            return self.name
        return f"{self.name} ({self.vram_gib:.0f} GiB)"


@dataclass(slots=True)
class CpuInfo:
    model: str | None = None
    physical_cores: int | None = None
    logical_cores: int | None = None
    arch: str = "unknown"


@dataclass(slots=True)
class MemoryInfo:
    total: int | None = None
    available: int | None = None

    @property
    def total_gib(self) -> float | None:
        return None if self.total is None else self.total / GIB

    @property
    def available_gib(self) -> float | None:
        return None if self.available is None else self.available / GIB


@dataclass(slots=True)
class DiskInfo:
    path: str
    total: int | None = None
    free: int | None = None

    @property
    def free_gib(self) -> float | None:
        return None if self.free is None else self.free / GIB


@dataclass(slots=True)
class HostInfo:
    platform: Platform
    system: str
    release: str
    machine: str
    python_version: str
    distro: str | None = None
    # Set on WSL2, where the Windows-side driver is what actually matters.
    wsl_kernel: str | None = None


@dataclass(slots=True)
class TorchInfo:
    """What an already-installed PyTorch reports.

    Absent on a fresh install, which is fine — this section is confirmation,
    not the basis for any recommendation.
    """

    version: str
    build: str | None = None
    cuda_available: bool = False
    cuda_version: str | None = None
    hip_version: str | None = None
    mps_available: bool = False
    xpu_available: bool = False
    device_count: int = 0


@dataclass(slots=True)
class Issue:
    """Something the user should know about, surfaced in the report."""

    level: str  # "error" | "warn" | "info"
    message: str
    hint: str | None = None


@dataclass(slots=True)
class HostReport:
    host: HostInfo
    cpu: CpuInfo
    memory: MemoryInfo
    disk: DiskInfo
    gpus: list[GpuInfo] = field(default_factory=list)
    backend: Backend = Backend.CPU
    backend_reason: str = ""
    nvidia_driver: str | None = None
    nvidia_cuda_version: str | None = None
    rocm_version: str | None = None
    torch: TorchInfo | None = None
    issues: list[Issue] = field(default_factory=list)

    @property
    def total_vram(self) -> int | None:
        known = [g.vram_total for g in self.gpus if g.vram_total is not None]
        return sum(known) if known else None

    @property
    def largest_vram(self) -> int | None:
        known = [g.vram_total for g in self.gpus if g.vram_total is not None]
        return max(known) if known else None

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict. Enums become their string values."""
        return _asdict(self)


def _asdict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _asdict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (list, tuple)):
        return [_asdict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _asdict(v) for k, v in obj.items()}
    return obj
