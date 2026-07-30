# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Host, CPU, memory and disk detection."""

from __future__ import annotations

import os
import platform
import shutil
import sys
from pathlib import Path

import psutil

from bloomery import paths
from bloomery.probe.types import CpuInfo, DiskInfo, HostInfo, MemoryInfo, Platform
from bloomery.probe.util import read_text, run


def detect_platform() -> Platform:
    """Classify the host.

    WSL2 is split out from Linux deliberately. It looks like Linux to Python but
    the GPU story is entirely different: the driver lives on the Windows side,
    ROCm needs ``/dev/dxg`` rather than ``/dev/kfd``, and the amount of RAM a
    process can claim is capped by ``.wslconfig`` rather than by the host's
    physical memory.
    """
    system = sys.platform
    if system.startswith("linux"):
        return Platform.WSL2 if _is_wsl() else Platform.LINUX
    if system == "darwin":
        return Platform.MACOS
    if system in ("win32", "cygwin"):
        return Platform.WINDOWS
    return Platform.UNKNOWN


def _is_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    for candidate in ("/proc/sys/kernel/osrelease", "/proc/version"):
        text = read_text(candidate)
        if text and "microsoft" in text.lower():
            return True
    return False


def _linux_distro() -> str | None:
    text = read_text("/etc/os-release")
    if not text:
        return None
    for line in text.splitlines():
        if line.startswith("PRETTY_NAME="):
            return line.partition("=")[2].strip().strip('"')
    return None


def probe_host() -> HostInfo:
    plat = detect_platform()
    distro: str | None = None
    wsl_kernel: str | None = None

    if plat in (Platform.LINUX, Platform.WSL2):
        distro = _linux_distro()
        if plat is Platform.WSL2:
            wsl_kernel = read_text("/proc/sys/kernel/osrelease")
    elif plat is Platform.MACOS:
        mac_version = platform.mac_ver()[0]
        distro = f"macOS {mac_version}" if mac_version else "macOS"
    elif plat is Platform.WINDOWS:
        distro = f"Windows {platform.release()}"

    return HostInfo(
        platform=plat,
        system=platform.system(),
        release=platform.release(),
        machine=platform.machine(),
        python_version=platform.python_version(),
        distro=distro,
        wsl_kernel=wsl_kernel,
    )


def _cpu_model() -> str | None:
    if sys.platform == "darwin":
        result = run(["sysctl", "-n", "machdep.cpu.brand_string"], timeout=3)
        if result.ok and result.stdout.strip():
            return result.stdout.strip()
    elif sys.platform.startswith("linux"):
        text = read_text("/proc/cpuinfo")
        if text:
            for line in text.splitlines():
                # x86 uses "model name"; arm64 kernels often only have
                # "Model" or nothing useful at all.
                if line.lower().startswith(("model name", "hardware")):
                    value = line.partition(":")[2].strip()
                    if value:
                        return value
    # platform.processor() is empty on many Linux builds and returns the bare
    # arch on others, so it is the fallback rather than the first choice.
    return platform.processor() or None


def probe_cpu() -> CpuInfo:
    return CpuInfo(
        model=_cpu_model(),
        physical_cores=psutil.cpu_count(logical=False),
        logical_cores=psutil.cpu_count(logical=True),
        arch=platform.machine(),
    )


def probe_memory() -> MemoryInfo:
    virtual = psutil.virtual_memory()
    return MemoryInfo(total=virtual.total, available=virtual.available)


def probe_disk(path: Path | None = None) -> DiskInfo:
    """Free space where bloomery will write.

    Falls back up the tree if the configured home does not exist yet, so this
    reports something useful before first run.
    """
    target = path or paths.home()
    probe_target = target
    while not probe_target.exists() and probe_target != probe_target.parent:
        probe_target = probe_target.parent

    try:
        usage = shutil.disk_usage(probe_target)
    except OSError:
        return DiskInfo(path=str(target))
    return DiskInfo(path=str(target), total=usage.total, free=usage.free)
