# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for raw PCI enumeration, against a synthetic sysfs tree."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from bloomery.probe.pci import list_display_devices, vendors_present
from bloomery.probe.types import Vendor


def _symlinks_available() -> bool:
    """Whether this platform lets an unprivileged process create a symlink."""
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "target"
        target.mkdir()
        try:
            (Path(tmp) / "link").symlink_to(target)
        except (OSError, NotImplementedError):
            return False
    return True


# sysfs models a bound driver as a symlink, so the driver-name tests need real
# symlinks. Creating one on Windows requires developer mode or elevation. The
# code under test only ever reads /sys, which exists on Linux alone, so there is
# genuinely nothing to cover on Windows — a fixture that failed there would say
# nothing about the Linux behaviour.
needs_symlinks = pytest.mark.skipif(
    not _symlinks_available(), reason="symlink creation unavailable on this platform"
)


def add_device(
    root: Path,
    slot: str,
    *,
    vendor: str,
    device: str,
    pci_class: str,
    driver: str | None = None,
) -> None:
    path = root / slot
    path.mkdir(parents=True)
    (path / "vendor").write_text(f"{vendor}\n")
    (path / "device").write_text(f"{device}\n")
    (path / "class").write_text(f"{pci_class}\n")
    if driver is not None:
        # sysfs models the bound driver as a symlink whose basename is its name.
        target = root.parent / "drivers" / driver
        target.mkdir(parents=True, exist_ok=True)
        (path / "driver").symlink_to(target)


class TestListDisplayDevices:
    @needs_symlinks
    def test_finds_vga_class_nvidia(self, tmp_path: Path) -> None:
        devices = tmp_path / "devices"
        add_device(
            devices,
            "0000:01:00.0",
            vendor="0x10de",
            device="0x2684",
            pci_class="0x030000",
            driver="nvidia",
        )
        found = list_display_devices(devices)
        assert len(found) == 1
        assert found[0].vendor is Vendor.NVIDIA
        assert found[0].driver == "nvidia"
        assert found[0].device_id == 0x2684

    @needs_symlinks
    def test_finds_headless_compute_class(self, tmp_path: Path) -> None:
        # Datacentre cards report subclass 0x02, not the VGA-compatible 0x00.
        devices = tmp_path / "devices"
        add_device(
            devices,
            "0000:03:00.0",
            vendor="0x1002",
            device="0x74a1",
            pci_class="0x038000",
            driver="amdgpu",
        )
        found = list_display_devices(devices)
        assert len(found) == 1
        assert found[0].vendor is Vendor.AMD

    def test_unbound_driver_is_none(self, tmp_path: Path) -> None:
        """The signal for 'card present, driver not loaded'."""
        devices = tmp_path / "devices"
        add_device(devices, "0000:03:00.0", vendor="0x1002", device="0x744c", pci_class="0x030000")
        found = list_display_devices(devices)
        assert len(found) == 1
        assert found[0].driver is None

    def test_ignores_non_display_devices(self, tmp_path: Path) -> None:
        devices = tmp_path / "devices"
        # A network card from a vendor we do recognise.
        add_device(devices, "0000:05:00.0", vendor="0x8086", device="0x1533", pci_class="0x020000")
        assert list_display_devices(devices) == []

    def test_ignores_unknown_vendors(self, tmp_path: Path) -> None:
        devices = tmp_path / "devices"
        add_device(devices, "0000:06:00.0", vendor="0x1234", device="0x0001", pci_class="0x030000")
        assert list_display_devices(devices) == []

    def test_multiple_vendors(self, tmp_path: Path) -> None:
        devices = tmp_path / "devices"
        add_device(devices, "0000:01:00.0", vendor="0x10de", device="0x2684", pci_class="0x030000")
        add_device(devices, "0000:03:00.0", vendor="0x1002", device="0x744c", pci_class="0x030000")
        add_device(devices, "0000:00:02.0", vendor="0x8086", device="0xa780", pci_class="0x030000")
        assert vendors_present(devices) == {Vendor.NVIDIA, Vendor.AMD, Vendor.INTEL}

    def test_missing_root_is_empty(self, tmp_path: Path) -> None:
        assert list_display_devices(tmp_path / "nope") == []
        assert vendors_present(tmp_path / "nope") == set()

    def test_malformed_files_are_skipped(self, tmp_path: Path) -> None:
        devices = tmp_path / "devices"
        path = devices / "0000:07:00.0"
        path.mkdir(parents=True)
        (path / "class").write_text("not-hex\n")
        (path / "vendor").write_text("0x10de\n")
        assert list_display_devices(devices) == []
