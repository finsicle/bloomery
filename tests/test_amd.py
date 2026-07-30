# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for AMD detection across all three sources.

AMD's tooling has renamed fields and reshaped its JSON repeatedly across ROCm 5,
6 and 7, so the parsers are written to search for plausible keys rather than
follow a fixed path. These tests pin that behaviour against several plausible
shapes, including ones we do not expect to see today.
"""

from __future__ import annotations

from pathlib import Path

from bloomery.probe.amd import (
    _kfd_bdf,
    _scaled_bytes,
    parse_amd_smi,
    parse_rocm_smi,
    probe_kfd,
)
from bloomery.probe.types import Vendor

GIB = 1024**3

# amd-smi static --json, ROCm 7-era shape.
AMD_SMI_DUAL = [
    {
        "gpu": 0,
        "asic": {
            "market_name": "Radeon RX 7900 XTX",
            "vendor_id": "0x1002",
            "target_graphics_version": "gfx1100",
        },
        "bus": {"bdf": "0000:03:00.0"},
        "vram": {"size": {"value": 24560, "unit": "MB"}},
        "driver": {"version": "6.10.5"},
    },
    {
        "gpu": 1,
        "asic": {
            "market_name": "Radeon RX 7900 XTX",
            "target_graphics_version": "gfx1100",
        },
        "bus": {"bdf": "0000:04:00.0"},
        "vram": {"size": {"value": 24560, "unit": "MB"}},
    },
]

# An older/alternative shape: flat fields, bare integer gfx version, wrapped list.
AMD_SMI_FLAT = {
    "gpus": [
        {
            "gpu_id": 0,
            "product_name": "Instinct MI300X",
            "gfx_version": 90402,
            "vram_total": "196608 MB",
        }
    ]
}

# rocm-smi --json
ROCM_SMI = {
    "card0": {
        "Card Series": "Radeon RX 7900 XTX",
        "Card Model": "0x744c",
        "VRAM Total Memory (B)": "25753026560",
        "GFX Version": "gfx1100",
        "PCI Bus": "0000:03:00.0",
    },
    "card1": {
        "Card Series": "Radeon RX 7900 XTX",
        "VRAM Total Memory (B)": "25753026560",
        "GFX Version": "gfx1100",
    },
    "system": {"Driver version": "6.10.5"},
}


class TestScaledBytes:
    def test_dict_with_unit(self) -> None:
        assert _scaled_bytes({"value": 24, "unit": "GB"}) == 24 * GIB

    def test_string_with_unit_suffix(self) -> None:
        assert _scaled_bytes("196608 MB") == 196608 * 1024**2

    def test_bare_number_defaults_to_mib(self) -> None:
        # AMD tools label VRAM "MB" but mean MiB.
        assert _scaled_bytes(24560) == 24560 * 1024**2

    def test_explicit_default_unit(self) -> None:
        assert _scaled_bytes(1024, default_unit="B") == 1024

    def test_unparseable(self) -> None:
        assert _scaled_bytes("N/A") is None
        assert _scaled_bytes(None) is None


class TestParseAmdSmi:
    def test_dual_gpu_current_shape(self) -> None:
        gpus = parse_amd_smi(AMD_SMI_DUAL)
        assert len(gpus) == 2

        first = gpus[0]
        assert first.vendor is Vendor.AMD
        assert first.index == 0
        assert first.name == "Radeon RX 7900 XTX"
        assert first.arch == "gfx1100"
        assert first.pci_bus_id == "0000:03:00.0"
        assert first.source == "amd-smi"
        # 24560 MiB is 24 GiB once read correctly.
        assert first.vram_gib is not None
        assert 23.5 < first.vram_gib < 24.5

        assert gpus[1].index == 1

    def test_flat_shape_with_wrapper_and_integer_gfx(self) -> None:
        gpus = parse_amd_smi(AMD_SMI_FLAT)
        assert len(gpus) == 1
        assert gpus[0].name == "Instinct MI300X"
        # Bare KFD integer must be decoded into a gfx name.
        assert gpus[0].arch == "gfx942"
        assert gpus[0].vram_total == 196608 * 1024**2

    def test_index_falls_back_to_position(self) -> None:
        gpus = parse_amd_smi([{"asic": {"market_name": "Some GPU"}}])
        assert gpus[0].index == 0

    def test_unnamed_gpu_gets_placeholder(self) -> None:
        gpus = parse_amd_smi([{"gpu": 0}])
        assert gpus[0].name == "AMD GPU"

    def test_garbage_input(self) -> None:
        assert parse_amd_smi("not json") == []
        assert parse_amd_smi(None) == []
        assert parse_amd_smi([]) == []


class TestParseRocmSmi:
    def test_two_cards(self) -> None:
        gpus = parse_rocm_smi(ROCM_SMI)
        assert len(gpus) == 2
        assert [g.index for g in gpus] == [0, 1]

        first = gpus[0]
        assert first.name == "Radeon RX 7900 XTX"
        # Key says (B), so this is already bytes and must not be rescaled.
        assert first.vram_total == 25753026560
        assert first.arch == "gfx1100"
        assert first.source == "rocm-smi"

    def test_ignores_non_card_keys(self) -> None:
        gpus = parse_rocm_smi(ROCM_SMI)
        assert all("system" not in g.name for g in gpus)

    def test_garbage_input(self) -> None:
        assert parse_rocm_smi([]) == []
        assert parse_rocm_smi(None) == []


class TestKfdBdf:
    def test_decodes_location_id(self) -> None:
        # bus 0x03, device 0, function 0
        assert _kfd_bdf(3 << 8) == "03:00.0"

    def test_with_device_and_function(self) -> None:
        location = (0x41 << 8) | (0x02 << 3) | 1
        assert _kfd_bdf(location) == "41:02.1"

    def test_zero_and_none(self) -> None:
        assert _kfd_bdf(0) is None
        assert _kfd_bdf(None) is None


def build_kfd_tree(root: Path) -> None:
    """Write a synthetic /sys/class/kfd topology: one CPU node, two GPUs."""
    cpu = root / "0"
    cpu.mkdir(parents=True)
    (cpu / "properties").write_text("cpu_cores_count 16\nsimd_count 0\n")
    (cpu / "name").write_text("CPU\n")

    for node_id, location in ((1, (3 << 8)), (2, (4 << 8))):
        node = root / str(node_id)
        node.mkdir(parents=True)
        (node / "properties").write_text(
            "cpu_cores_count 0\n"
            "simd_count 192\n"
            "gfx_target_version 110000\n"
            f"location_id {location}\n"
        )
        (node / "name").write_text("navi31\n")

        # Two banks: one system-visible aperture, one card-local.
        system_bank = node / "mem_banks" / "0"
        system_bank.mkdir(parents=True)
        system_bank.joinpath("properties").write_text("heap_type 0\nsize_in_bytes 8589934592\n")
        vram_bank = node / "mem_banks" / "1"
        vram_bank.mkdir(parents=True)
        vram_bank.joinpath("properties").write_text("heap_type 1\nsize_in_bytes 25753026560\n")


class TestProbeKfd:
    def test_reads_gpu_nodes_only(self, tmp_path: Path) -> None:
        build_kfd_tree(tmp_path)
        gpus = probe_kfd(tmp_path)

        # The CPU node must not be reported as a GPU.
        assert len(gpus) == 2
        assert [g.index for g in gpus] == [0, 1]

        first = gpus[0]
        assert first.vendor is Vendor.AMD
        assert first.name == "navi31"
        assert first.arch == "gfx1100"
        assert first.source == "sysfs/kfd"
        assert first.pci_bus_id == "03:00.0"

    def test_sums_only_card_local_memory(self, tmp_path: Path) -> None:
        build_kfd_tree(tmp_path)
        gpu = probe_kfd(tmp_path)[0]
        # heap_type 0 is system memory and must be excluded.
        assert gpu.vram_total == 25753026560

    def test_missing_root(self, tmp_path: Path) -> None:
        assert probe_kfd(tmp_path / "nope") == []

    def test_node_without_mem_banks(self, tmp_path: Path) -> None:
        node = tmp_path / "1"
        node.mkdir(parents=True)
        (node / "properties").write_text(
            "cpu_cores_count 0\nsimd_count 64\ngfx_target_version 90402\n"
        )
        gpus = probe_kfd(tmp_path)
        assert len(gpus) == 1
        assert gpus[0].vram_total is None
        assert gpus[0].arch == "gfx942"

    def test_ignores_non_numeric_entries(self, tmp_path: Path) -> None:
        build_kfd_tree(tmp_path)
        (tmp_path / "properties").write_text("junk\n")
        assert len(probe_kfd(tmp_path)) == 2
