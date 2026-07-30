# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Intel GPU detection.

Minimal on purpose. Intel Arc training through PyTorch XPU works but is a small
slice of the user base, so this reports what ``xpu-smi`` says and nothing more.
No sysfs fallback: guessing that an integrated display adapter is a training
device would produce worse advice than reporting nothing.
"""

from __future__ import annotations

import json

from bloomery.probe.types import GpuInfo, Vendor
from bloomery.probe.util import find_key, parse_int, run


def parse_discovery(payload: object) -> list[GpuInfo]:
    """Parse ``xpu-smi discovery -j``."""
    if isinstance(payload, dict):
        found = find_key(payload, ("device_list", "devices"))
        entries = found if isinstance(found, list) else []
    elif isinstance(payload, list):
        entries = payload
    else:
        return []

    gpus: list[GpuInfo] = []
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        index = parse_int(find_key(entry, ("device_id", "id", "device_index")))
        name = find_key(entry, ("device_name", "name", "model"))
        memory = parse_int(find_key(entry, ("memory_physical_size_byte", "memory_size")))
        gpus.append(
            GpuInfo(
                index=index if index is not None else position,
                vendor=Vendor.INTEL,
                name=str(name) if name else "Intel GPU",
                vram_total=memory,
                pci_bus_id=_as_str(find_key(entry, ("pci_bdf_address", "pci_bdf"))),
                source="xpu-smi",
            )
        )
    return gpus


def _as_str(value: object) -> str | None:
    return None if value is None else str(value)


def probe() -> list[GpuInfo]:
    result = run(["xpu-smi", "discovery", "-j"])
    if not result.ok:
        return []
    try:
        payload = json.loads(result.stdout)
    except (json.JSONDecodeError, ValueError):
        return []
    try:
        return parse_discovery(payload)
    except (TypeError, ValueError, AttributeError):
        return []
