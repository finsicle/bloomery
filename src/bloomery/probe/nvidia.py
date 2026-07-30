# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""NVIDIA GPU detection via ``nvidia-smi``.

Deliberately shells out instead of importing ``pynvml``. The probe must run
before any heavy dependency is installed — it is the tool you use when the
install is what's broken — and ``nvidia-smi`` ships with the driver.
"""

from __future__ import annotations

import re

from bloomery.probe.types import GpuInfo, Vendor
from bloomery.probe.util import parse_int, run

# Ordered to match the parser below. compute_cap is absent on older drivers, so
# it lives in the optional tail rather than the core set.
_CORE_FIELDS = ("index", "name", "memory.total", "memory.free", "driver_version")
_EXTRA_FIELDS = ("compute_cap", "uuid", "pci.bus_id")

_CUDA_RE = re.compile(r"CUDA\s*Version\s*:?\s*([0-9]+\.[0-9]+)", re.IGNORECASE)


def parse_query_csv(text: str, *, with_extras: bool) -> list[GpuInfo]:
    """Parse ``nvidia-smi --query-gpu`` CSV output.

    Expects ``--format=csv,noheader,nounits``, which puts memory in MiB with no
    unit suffix.
    """
    gpus: list[GpuInfo] = []
    expected = len(_CORE_FIELDS) + (len(_EXTRA_FIELDS) if with_extras else 0)

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < len(_CORE_FIELDS):
            continue
        # Tolerate a driver reporting more or fewer optional columns than asked.
        if len(parts) < expected:
            parts += [""] * (expected - len(parts))

        index = parse_int(parts[0])
        if index is None:
            continue

        total_mib = parse_int(parts[2])
        free_mib = parse_int(parts[3])

        arch = uuid = bus_id = None
        if with_extras:
            base = len(_CORE_FIELDS)
            arch = _clean(parts[base])
            uuid = _clean(parts[base + 1])
            bus_id = _clean(parts[base + 2])

        gpus.append(
            GpuInfo(
                index=index,
                vendor=Vendor.NVIDIA,
                name=parts[1] or "NVIDIA GPU",
                vram_total=None if total_mib is None else total_mib * 1024 * 1024,
                vram_free=None if free_mib is None else free_mib * 1024 * 1024,
                arch=arch,
                pci_bus_id=bus_id,
                uuid=uuid,
                source="nvidia-smi",
            )
        )
    return gpus


def _clean(value: str) -> str | None:
    value = value.strip()
    if not value or value.startswith("[") or value in ("N/A", "Not Supported"):
        return None
    return value


def parse_cuda_version(text: str) -> str | None:
    """Pull the CUDA version out of ``nvidia-smi`` output.

    Works against both ``nvidia-smi --version`` (a key/value block) and the
    banner at the top of plain ``nvidia-smi`` output.
    """
    match = _CUDA_RE.search(text)
    return match.group(1) if match else None


def probe() -> tuple[list[GpuInfo], str | None, str | None]:
    """Detect NVIDIA GPUs.

    Returns ``(gpus, driver_version, cuda_version)``. An empty list means no
    usable NVIDIA GPU, whether that is because there is none or because the
    driver is not responding — both cases mean the same thing to a caller
    deciding what to train on.
    """
    query = ",".join(_CORE_FIELDS + _EXTRA_FIELDS)
    result = run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"])
    with_extras = True

    if not result.ok:
        # Retry without the optional fields; an unsupported field makes the
        # whole query fail rather than blanking that one column.
        query = ",".join(_CORE_FIELDS)
        result = run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"])
        with_extras = False

    if not result.ok:
        return [], None, None

    gpus = parse_query_csv(result.stdout, with_extras=with_extras)
    if not gpus:
        return [], None, None

    driver = _driver_version(result.stdout)
    return gpus, driver, _cuda_version()


def _driver_version(csv_text: str) -> str | None:
    for line in csv_text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= len(_CORE_FIELDS):
            value = _clean(parts[4])
            if value:
                return value
    return None


def _cuda_version() -> str | None:
    # --version is the structured form on recent drivers.
    result = run(["nvidia-smi", "--version"])
    if result.ok:
        version = parse_cuda_version(result.stdout)
        if version:
            return version
    # Older drivers only print it in the banner.
    result = run(["nvidia-smi"])
    if result.ok:
        return parse_cuda_version(result.stdout)
    return None
