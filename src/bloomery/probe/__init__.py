# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Host probe.

:func:`probe_host_report` gathers everything bloomery needs to know about the
machine it is running on. It has no ML dependencies, never raises on a hostile
environment, and every external call is bounded by a timeout — it has to work on
the machine where nothing else does.
"""

from __future__ import annotations

from bloomery.probe import amd, apple, backend, intel, nvidia, pci, system, torchinfo
from bloomery.probe.types import (
    GIB,
    Backend,
    GpuInfo,
    HostReport,
    Issue,
    Platform,
    Vendor,
)

__all__ = [
    "Backend",
    "GpuInfo",
    "HostReport",
    "Issue",
    "Platform",
    "Vendor",
    "probe_host_report",
]

# A single bf16 7B model is ~14 GiB. Add a base model, several checkpoints and a
# GGUF intermediate and 50 GiB is a floor, not a comfortable margin.
_DISK_WARN = 50 * GIB
_DISK_ERROR = 10 * GIB


def probe_host_report(*, include_torch: bool = True) -> HostReport:
    """Build a full report of the current host."""
    host = system.probe_host()

    nvidia_gpus, nvidia_driver, cuda_version = nvidia.probe()
    amd_gpus, rocm_version = amd.probe()
    intel_gpus = intel.probe()
    apple_gpus = apple.probe()

    gpus = _renumber(nvidia_gpus + amd_gpus + intel_gpus + apple_gpus)

    chosen, reason = backend.resolve(gpus, cuda_version=cuda_version, driver_version=nvidia_driver)

    report = HostReport(
        host=host,
        cpu=system.probe_cpu(),
        memory=system.probe_memory(),
        disk=system.probe_disk(),
        gpus=gpus,
        backend=chosen,
        backend_reason=reason,
        nvidia_driver=nvidia_driver,
        nvidia_cuda_version=cuda_version,
        rocm_version=rocm_version,
    )

    if include_torch:
        report.torch = torchinfo.probe()

    report.issues = _collect_issues(report)
    return report


def _renumber(gpus: list[GpuInfo]) -> list[GpuInfo]:
    """Give every GPU a unique report-wide index.

    Each vendor numbers from zero, so a mixed machine would otherwise have two
    GPUs both calling themselves 0. The vendor-local index stays available via
    the ``source`` field's tooling if it is ever needed; what callers need here
    is a stable identifier for the UI.
    """
    for position, gpu in enumerate(gpus):
        gpu.index = position
    return gpus


def _collect_issues(report: HostReport) -> list[Issue]:
    issues: list[Issue] = []
    vendors = {gpu.vendor for gpu in report.gpus}
    plat = report.host.platform

    issues.extend(_hardware_visibility_issues(report, vendors))

    if plat is Platform.WSL2:
        if Vendor.AMD in vendors or Vendor.AMD in _bus_vendors():
            issues.append(
                Issue(
                    level="warn",
                    message="AMD GPU under WSL2. ROCm here goes through /dev/dxg, not /dev/kfd.",
                    hint=(
                        "Needs Windows 11, a recent Adrenalin driver and AMD's "
                        "ROCm-on-WSL packages. Narrower hardware support than native Linux."
                    ),
                )
            )
        issues.append(
            Issue(
                level="info",
                message="Running under WSL2, so RAM and CPU are capped by the WSL VM.",
                hint="Raise the ceiling in %UserProfile%\\.wslconfig if a run is memory-starved.",
            )
        )

    if plat is Platform.MACOS and Vendor.APPLE in vendors:
        issues.append(
            Issue(
                level="info",
                message="Apple Silicon shares memory between CPU and GPU.",
                hint=(
                    "Metal has no flash-attention and patchy bf16 coverage. "
                    "Good for small from-scratch runs; not for multi-billion-parameter training."
                ),
            )
        )

    if plat is Platform.WINDOWS:
        issues.append(
            Issue(
                level="warn",
                message="Native Windows is best-effort. WSL2 is the supported path.",
                hint="Distributed training and memory limits are both more reliable under WSL2.",
            )
        )

    issues.extend(backend.arch_issues(report.gpus))
    issues.extend(_disk_issues(report))

    if not report.gpus:
        issues.append(
            Issue(
                level="info",
                message="No GPU detected. Training will run on CPU.",
                hint="Workable for models up to a few million parameters. Slow beyond that.",
            )
        )

    issues.extend(torchinfo.consistency_issues(report.torch, report.gpus, report.backend))
    return issues


def _bus_vendors() -> set[Vendor]:
    try:
        return pci.vendors_present()
    except Exception:  # noqa: BLE001 - sysfs shape varies; never fail the probe
        return set()


def _hardware_visibility_issues(report: HostReport, vendors: set[Vendor]) -> list[Issue]:
    """Flag GPUs that are on the PCI bus but invisible to their tooling.

    This is the "I bought an AMD card and nothing works" diagnosis, and it is
    worth a precise message because the alternative — an empty GPU list — sends
    people looking in entirely the wrong place.
    """
    if report.host.platform not in (Platform.LINUX, Platform.WSL2):
        return []

    try:
        devices = pci.list_display_devices()
    except Exception:  # noqa: BLE001
        return []

    issues: list[Issue] = []

    for vendor, tool, hint in (
        (
            Vendor.NVIDIA,
            "nvidia-smi",
            "Install or repair the NVIDIA driver, then check `nvidia-smi` runs.",
        ),
        (
            Vendor.AMD,
            "amd-smi / rocm-smi",
            "Install ROCm and add your user to the `render` and `video` groups.",
        ),
    ):
        on_bus = [d for d in devices if d.vendor is vendor]
        if not on_bus or vendor in vendors:
            continue

        names = ", ".join(sorted({d.slot for d in on_bus}))
        unbound = [d for d in on_bus if not d.driver]
        detail = (
            " No kernel driver is bound to it."
            if unbound
            else f" Kernel driver: {on_bus[0].driver}."
        )
        issues.append(
            Issue(
                level="error",
                message=(
                    f"{len(on_bus)} {vendor.value.upper()} display device(s) on the PCI "
                    f"bus ({names}) but {tool} reports nothing.{detail}"
                ),
                hint=hint,
            )
        )

    return issues


def _disk_issues(report: HostReport) -> list[Issue]:
    free = report.disk.free
    if free is None:
        return []
    if free < _DISK_ERROR:
        return [
            Issue(
                level="error",
                message=f"Only {free / GIB:.1f} GiB free at {report.disk.path}.",
                hint="Not enough for a single base model. Set BLOOMERY_HOME to a larger disk.",
            )
        ]
    if free < _DISK_WARN:
        return [
            Issue(
                level="warn",
                message=f"{free / GIB:.0f} GiB free at {report.disk.path}.",
                hint=(
                    "Checkpoints, tokenized shards and GGUF intermediates add up fast. "
                    "Set BLOOMERY_HOME to point somewhere roomier."
                ),
            )
        ]
    return []
