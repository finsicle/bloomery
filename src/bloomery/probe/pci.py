# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Raw PCI enumeration of display-class devices.

This exists to tell two very different situations apart:

* there is no GPU in this machine, and
* there is a GPU, but its driver or userspace tooling is not working.

The vendor tools cannot distinguish them — ``nvidia-smi`` failing to run looks
identical to having no NVIDIA card. Reading the PCI bus directly can, and the
second case is by far the more common one for a new user, so it deserves a
specific message rather than a shrug.

Pure sysfs, no ``lspci`` binary required. Linux and WSL2 only; sysfs is not
available on macOS or native Windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bloomery.probe.types import Vendor
from bloomery.probe.util import read_text

PCI_DEVICES = Path("/sys/bus/pci/devices")

_VENDOR_IDS: dict[int, Vendor] = {
    0x10DE: Vendor.NVIDIA,
    0x1002: Vendor.AMD,
    0x1022: Vendor.AMD,  # older AMD/ATI allocations
    0x8086: Vendor.INTEL,
}

# PCI base class 0x03 is "display controller", covering both VGA-compatible
# adapters and the headless compute cards that report subclass 0x02.
_DISPLAY_BASE_CLASS = 0x03


@dataclass(frozen=True, slots=True)
class PciDevice:
    slot: str
    vendor: Vendor
    device_id: int
    driver: str | None


def _hex(path: Path) -> int | None:
    raw = read_text(path)
    if not raw:
        return None
    try:
        return int(raw, 16)
    except ValueError:
        return None


def list_display_devices(root: Path = PCI_DEVICES) -> list[PciDevice]:
    """All display-class PCI devices, whatever their driver state."""
    if not root.is_dir():
        return []

    devices: list[PciDevice] = []
    try:
        slots = sorted(root.iterdir())
    except OSError:
        return []

    for slot in slots:
        class_id = _hex(slot / "class")
        if class_id is None or (class_id >> 16) != _DISPLAY_BASE_CLASS:
            continue
        vendor_id = _hex(slot / "vendor")
        vendor = _VENDOR_IDS.get(vendor_id) if vendor_id is not None else None
        if vendor is None:
            continue

        driver: str | None = None
        driver_link = slot / "driver"
        if driver_link.exists():
            try:
                driver = driver_link.resolve().name
            except OSError:
                driver = None

        devices.append(
            PciDevice(
                slot=slot.name,
                vendor=vendor,
                device_id=_hex(slot / "device") or 0,
                driver=driver,
            )
        )
    return devices


def vendors_present(root: Path = PCI_DEVICES) -> set[Vendor]:
    """Vendors with a display-class device on the bus."""
    return {device.vendor for device in list_display_devices(root)}
