# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for NVIDIA parsing, against captured nvidia-smi output.

Fixture-driven so the NVIDIA path is testable on machines without an NVIDIA GPU
— which includes every Mac, and therefore most of the development happening on
this project.
"""

from __future__ import annotations

from bloomery.probe.nvidia import parse_cuda_version, parse_query_csv
from bloomery.probe.types import Vendor

# nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version,
#            compute_cap,uuid,pci.bus_id --format=csv,noheader,nounits
DUAL_4090 = """\
0, NVIDIA GeForce RTX 4090, 24564, 24136, 570.86.15, 8.9, GPU-1a2b3c4d-0000-0000-0000-000000000001, 00000000:01:00.0
1, NVIDIA GeForce RTX 4090, 24564, 23990, 570.86.15, 8.9, GPU-1a2b3c4d-0000-0000-0000-000000000002, 00000000:02:00.0
"""

# Older driver: compute_cap unsupported, so the query was retried without extras.
OLD_DRIVER_CORE_ONLY = """\
0, Tesla T4, 15360, 15109, 450.80.02
"""

# A driver that answers the query but blanks the optional columns.
NOT_SUPPORTED_EXTRAS = """\
0, NVIDIA GeForce GTX 1080 Ti, 11264, 11000, 470.199.02, [Not Supported], [N/A], 00000000:03:00.0
"""

VERSION_BLOCK = """\
NVIDIA-SMI version  : 570.86.15
NVML version        : 570.86
DRIVER version      : 570.86.15
CUDA Version        : 12.8
"""

BANNER = """\
Fri Jul 30 09:12:00 2026
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.05   Driver Version: 535.104.05   CUDA Version: 12.2     |
|-------------------------------+----------------------+----------------------+
"""


class TestParseQueryCsv:
    def test_parses_two_gpus_with_extras(self) -> None:
        gpus = parse_query_csv(DUAL_4090, with_extras=True)
        assert len(gpus) == 2

        first = gpus[0]
        assert first.index == 0
        assert first.vendor is Vendor.NVIDIA
        assert first.name == "NVIDIA GeForce RTX 4090"
        # 24564 MiB, reported without units
        assert first.vram_total == 24564 * 1024 * 1024
        assert first.vram_free == 24136 * 1024 * 1024
        assert first.arch == "8.9"
        assert first.pci_bus_id == "00000000:01:00.0"
        assert first.uuid is not None
        assert first.source == "nvidia-smi"

        assert gpus[1].index == 1
        assert gpus[1].pci_bus_id == "00000000:02:00.0"

    def test_vram_reported_in_gib(self) -> None:
        gpu = parse_query_csv(DUAL_4090, with_extras=True)[0]
        assert gpu.vram_gib is not None
        assert 23.5 < gpu.vram_gib < 24.5

    def test_core_only_query(self) -> None:
        gpus = parse_query_csv(OLD_DRIVER_CORE_ONLY, with_extras=False)
        assert len(gpus) == 1
        assert gpus[0].name == "Tesla T4"
        assert gpus[0].vram_total == 15360 * 1024 * 1024
        assert gpus[0].arch is None
        assert gpus[0].pci_bus_id is None

    def test_placeholder_extras_become_none(self) -> None:
        gpu = parse_query_csv(NOT_SUPPORTED_EXTRAS, with_extras=True)[0]
        assert gpu.arch is None
        assert gpu.uuid is None
        # A real value alongside placeholders still comes through.
        assert gpu.pci_bus_id == "00000000:03:00.0"

    def test_blank_and_malformed_lines_skipped(self) -> None:
        text = DUAL_4090 + "\n\ngarbage\n"
        assert len(parse_query_csv(text, with_extras=True)) == 2

    def test_empty_input(self) -> None:
        assert parse_query_csv("", with_extras=True) == []

    def test_tolerates_fewer_columns_than_requested(self) -> None:
        # Asked for extras, driver returned only the core fields.
        gpus = parse_query_csv(OLD_DRIVER_CORE_ONLY, with_extras=True)
        assert len(gpus) == 1
        assert gpus[0].name == "Tesla T4"
        assert gpus[0].arch is None


class TestParseCudaVersion:
    def test_structured_version_block(self) -> None:
        assert parse_cuda_version(VERSION_BLOCK) == "12.8"

    def test_banner(self) -> None:
        assert parse_cuda_version(BANNER) == "12.2"

    def test_absent(self) -> None:
        assert parse_cuda_version("no version here") is None
