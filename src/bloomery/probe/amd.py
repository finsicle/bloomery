# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""AMD GPU detection.

Three sources, tried in order, because AMD's tooling has changed names and JSON
shapes repeatedly across ROCm 5, 6 and 7:

1. ``amd-smi`` — current, but field names have churned, so it is parsed by
   searching for plausible keys rather than by pinning to one schema.
2. ``rocm-smi`` — the older tool, still present on many installs.
3. ``/sys/class/kfd`` — the kernel driver's own topology. No tooling required,
   stable across releases, and the only source that works when ROCm userspace
   is not installed at all.

The sysfs path matters more than it looks: "I have an AMD card but ROCm isn't
set up" is the single most common state a new AMD user is in, and a probe that
reports nothing there is useless exactly when it is needed most.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from bloomery.probe.types import GpuInfo, Vendor
from bloomery.probe.util import find_key, gfx_name, parse_int, read_text, run

KFD_TOPOLOGY = Path("/sys/class/kfd/kfd/topology/nodes")

# HSA heap types that live on the card rather than in system memory.
_VRAM_HEAP_TYPES = {1, 2}

# Binary multiples throughout. AMD tools label VRAM "MB" but mean MiB — a 24 GiB
# RX 7900 XTX reports 24560, which is only correct read as MiB.
_UNIT_SCALE = {
    "B": 1,
    "KB": 1024,
    "KIB": 1024,
    "MB": 1024**2,
    "MIB": 1024**2,
    "GB": 1024**3,
    "GIB": 1024**3,
}


def _scaled_bytes(value: object, default_unit: str = "MB") -> int | None:
    """Normalise a vendor size field to bytes.

    Accepts a bare number, a ``"24 GB"`` string, or a
    ``{"value": 24, "unit": "GB"}`` object. AMD tools have used all three.
    """
    unit = default_unit
    raw: object = value

    if isinstance(value, dict):
        raw = find_key(value, ("value", "size", "total"))
        found_unit = find_key(value, ("unit", "units"))
        if isinstance(found_unit, str):
            unit = found_unit
    elif isinstance(value, str):
        match = re.search(r"([KMG]i?B|B)\s*$", value.strip(), re.IGNORECASE)
        if match:
            unit = match.group(1)

    number = parse_int(raw)
    if number is None:
        return None
    return number * _UNIT_SCALE.get(unit.strip().upper(), _UNIT_SCALE[default_unit])


def parse_amd_smi(payload: object) -> list[GpuInfo]:
    """Parse ``amd-smi static --json``.

    The top level is a list of per-GPU objects. Field names inside are looked up
    by candidate rather than by path, so a schema change in a point release
    degrades to a missing field instead of a crash.
    """
    if isinstance(payload, dict):
        # Some builds wrap the list, e.g. {"gpus": [...]}.
        found = find_key(payload, ("gpus", "gpu_data", "data"))
        entries = found if isinstance(found, list) else [payload]
    elif isinstance(payload, list):
        entries = payload
    else:
        return []

    gpus: list[GpuInfo] = []
    for position, entry in enumerate(entries):
        if not isinstance(entry, (dict, list)):
            continue

        index = parse_int(find_key(entry, ("gpu", "gpu_id", "device_id", "id")))
        name = find_key(entry, ("market_name", "product_name", "card_series", "name"))

        vram_node = find_key(entry, ("vram", "vram_total", "total_vram", "memory"))
        vram = _scaled_bytes(vram_node) if vram_node is not None else None
        if vram is None:
            vram = _scaled_bytes(find_key(entry, ("size", "total")))

        arch = find_key(
            entry,
            ("target_graphics_version", "gfx_version", "graphics_version", "asic_family"),
        )
        arch_text = _normalise_arch(arch)

        bdf = find_key(entry, ("bdf", "pci_bus_id", "bus_id", "pcie_bdf"))
        uuid = find_key(entry, ("uuid", "gpu_uuid", "serial_number"))

        gpus.append(
            GpuInfo(
                index=index if index is not None else position,
                vendor=Vendor.AMD,
                name=str(name) if name else "AMD GPU",
                vram_total=vram,
                arch=arch_text,
                pci_bus_id=str(bdf) if bdf else None,
                uuid=str(uuid) if uuid else None,
                source="amd-smi",
            )
        )
    return gpus


def _normalise_arch(value: object) -> str | None:
    """Coerce whatever the tool called the architecture into ``gfxNNNN``."""
    if value is None:
        return None
    if isinstance(value, int):
        return gfx_name(value)
    text = str(value).strip()
    if not text:
        return None
    if text.lower().startswith("gfx"):
        return text.lower()
    # Sometimes reported as the bare KFD integer in a string.
    number = parse_int(text)
    if number is not None and number > 1000:
        return gfx_name(number)
    return text


def parse_rocm_smi(payload: object) -> list[GpuInfo]:
    """Parse ``rocm-smi --json``, whose top level is ``{"card0": {...}}``."""
    if not isinstance(payload, dict):
        return []

    gpus: list[GpuInfo] = []
    for key, entry in payload.items():
        match = re.fullmatch(r"card(\d+)", str(key), re.IGNORECASE)
        if not match or not isinstance(entry, dict):
            continue

        name = find_key(entry, ("card_series", "card_model", "market_name", "device_name"))
        # rocm-smi reports this one in bytes, and says so in the key.
        vram = parse_int(find_key(entry, ("vram_total_memory_b", "vramtotalmemoryb")))
        if vram is None:
            vram = _scaled_bytes(find_key(entry, ("vram_total_memory", "vram_total")))

        gpus.append(
            GpuInfo(
                index=int(match.group(1)),
                vendor=Vendor.AMD,
                name=str(name) if name else "AMD GPU",
                vram_total=vram,
                arch=_normalise_arch(find_key(entry, ("gfx_version", "target_graphics_version"))),
                pci_bus_id=_str_or_none(find_key(entry, ("pci_bus", "pcibus"))),
                source="rocm-smi",
            )
        )
    return sorted(gpus, key=lambda g: g.index)


def _str_or_none(value: object) -> str | None:
    return None if value is None else str(value)


def probe_kfd(root: Path = KFD_TOPOLOGY) -> list[GpuInfo]:
    """Read GPU topology straight from the amdgpu kernel driver.

    Node 0 is normally the CPU. A node is a GPU when it reports SIMDs and no CPU
    cores, which is how the ROCm runtime itself distinguishes them.
    """
    if not root.is_dir():
        return []

    gpus: list[GpuInfo] = []
    try:
        nodes = sorted(
            (p for p in root.iterdir() if p.name.isdigit()),
            key=lambda p: int(p.name),
        )
    except OSError:
        return []

    for node in nodes:
        props = _parse_kfd_properties(read_text(node / "properties"))
        if not props:
            continue
        if props.get("simd_count", 0) <= 0 or props.get("cpu_cores_count", 0) > 0:
            continue

        vram = _kfd_vram(node)
        target = props.get("gfx_target_version", 0)
        name = read_text(node / "name")

        gpus.append(
            GpuInfo(
                index=len(gpus),
                vendor=Vendor.AMD,
                name=name or "AMD GPU",
                vram_total=vram,
                arch=gfx_name(target) if target else None,
                pci_bus_id=_kfd_bdf(props.get("location_id")),
                source="sysfs/kfd",
            )
        )
    return gpus


def _parse_kfd_properties(text: str | None) -> dict[str, int]:
    """KFD properties are ``key value`` lines, one per line, all integers."""
    if not text:
        return {}
    props: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            props[parts[0]] = int(parts[1])
        except ValueError:
            continue
    return props


def _kfd_vram(node: Path) -> int | None:
    """Sum the card-local memory banks for a KFD node."""
    banks = node / "mem_banks"
    if not banks.is_dir():
        return None
    total = 0
    found = False
    try:
        bank_dirs = sorted(p for p in banks.iterdir() if p.is_dir())
    except OSError:
        return None
    for bank in bank_dirs:
        props = _parse_kfd_properties(read_text(bank / "properties"))
        if props.get("heap_type") in _VRAM_HEAP_TYPES:
            size = props.get("size_in_bytes")
            if size:
                total += size
                found = True
    return total if found else None


def _kfd_bdf(location_id: int | None) -> str | None:
    """Render a KFD ``location_id`` as a PCI BDF string.

    The field packs bus in bits 8-15 and device/function in the low byte.
    """
    if not location_id:
        return None
    bus = (location_id >> 8) & 0xFF
    device = (location_id >> 3) & 0x1F
    function = location_id & 0x07
    return f"{bus:02x}:{device:02x}.{function}"


def detect_rocm_version() -> str | None:
    """Find the installed ROCm version, if any."""
    for path in ("/opt/rocm/.info/version", "/opt/rocm/.info/version-dev"):
        text = read_text(path)
        if text:
            return text.splitlines()[0].strip()

    result = run(["hipconfig", "--version"])
    if result.ok and result.stdout.strip():
        return result.stdout.strip().splitlines()[0]

    # Fall back to whatever /opt/rocm points at, e.g. /opt/rocm-7.2.1.
    try:
        target = Path("/opt/rocm").resolve()
    except OSError:
        return None
    match = re.search(r"rocm-?([0-9][0-9.]*)", target.name)
    return match.group(1) if match else None


def probe() -> tuple[list[GpuInfo], str | None]:
    """Detect AMD GPUs, trying each source until one yields something."""
    result = run(["amd-smi", "static", "--json"])
    if result.ok:
        gpus = _try_json(result.stdout, parse_amd_smi)
        if gpus:
            return _fill_missing_vram(gpus), detect_rocm_version()

    result = run(["rocm-smi", "--showallinfo", "--json"])
    if not result.ok:
        result = run(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"])
    if result.ok:
        gpus = _try_json(result.stdout, parse_rocm_smi)
        if gpus:
            return _fill_missing_vram(gpus), detect_rocm_version()

    return probe_kfd(), detect_rocm_version()


def _try_json(text: str, parser) -> list[GpuInfo]:  # noqa: ANN001 - local helper
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return []
    try:
        return parser(payload)
    except (TypeError, ValueError, AttributeError):
        return []


def _fill_missing_vram(gpus: list[GpuInfo]) -> list[GpuInfo]:
    """Backfill VRAM from sysfs when the CLI tool did not report it.

    Happens on partial ROCm installs and inside some containers. Matched by
    index because both sources enumerate in the same order.
    """
    if all(g.vram_total is not None for g in gpus):
        return gpus
    fallback = probe_kfd()
    if not fallback:
        return gpus
    by_index = {g.index: g for g in fallback}
    for position, gpu in enumerate(gpus):
        if gpu.vram_total is None:
            source = by_index.get(position)
            if source and source.vram_total:
                gpu.vram_total = source.vram_total
                gpu.source = f"{gpu.source}+sysfs"
            if source and not gpu.arch:
                gpu.arch = source.arch
    return gpus
