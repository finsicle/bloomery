# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for backend recommendation."""

from __future__ import annotations

import pytest

from bloomery.probe.backend import (
    arch_issues,
    nvidia_backend,
    parse_version,
    resolve,
)
from bloomery.probe.types import Backend, GpuInfo, Vendor


def gpu(vendor: Vendor, *, arch: str | None = None, name: str = "GPU") -> GpuInfo:
    return GpuInfo(index=0, vendor=vendor, name=name, arch=arch)


class TestParseVersion:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("12.8", (12, 8)),
            ("570.86.15", (570, 86, 15)),
            ("13.0", (13, 0)),
            ("7.2.1", (7, 2, 1)),
            ("12", (12,)),
            ("", None),
            (None, None),
            ("not-a-version", None),
        ],
    )
    def test_parses(self, text: str | None, expected: tuple[int, ...] | None) -> None:
        assert parse_version(text) == expected

    def test_comparable(self) -> None:
        assert parse_version("12.8") > parse_version("12.6")
        assert parse_version("13.0") > parse_version("12.8")


class TestNvidiaBackend:
    @pytest.mark.parametrize(
        ("cuda", "expected"),
        [
            ("13.0", Backend.CU130),
            ("13.2", Backend.CU130),
            ("12.8", Backend.CU128),
            ("12.9", Backend.CU128),
            ("12.6", Backend.CU126),
            ("12.7", Backend.CU126),
            ("12.2", Backend.CU118),
            ("11.8", Backend.CU118),
        ],
    )
    def test_from_cuda_version(self, cuda: str, expected: Backend) -> None:
        chosen, reason = nvidia_backend(cuda, "570.86.15")
        assert chosen is expected
        assert cuda in reason

    def test_single_component_cuda_version(self) -> None:
        chosen, _ = nvidia_backend("13", None)
        assert chosen is Backend.CU130

    @pytest.mark.parametrize(
        ("driver", "expected"),
        [
            ("580.10", Backend.CU130),
            ("570.86.15", Backend.CU128),
            ("525.60.13", Backend.CU128),
            ("470.199.02", Backend.CU118),
        ],
    )
    def test_falls_back_to_driver_version(self, driver: str, expected: Backend) -> None:
        chosen, reason = nvidia_backend(None, driver)
        assert chosen is expected
        assert "CUDA version unreported" in reason

    def test_nothing_readable_picks_safe_default(self) -> None:
        chosen, reason = nvidia_backend(None, None)
        assert chosen is Backend.CU128
        assert "unreadable" in reason

    def test_cuda_below_all_thresholds_uses_driver(self) -> None:
        chosen, _ = nvidia_backend("10.2", "470.199.02")
        assert chosen is Backend.CU118


class TestResolve:
    def test_no_gpus_is_cpu(self) -> None:
        chosen, reason = resolve([])
        assert chosen is Backend.CPU
        assert "no supported GPU" in reason

    def test_amd(self) -> None:
        chosen, _ = resolve([gpu(Vendor.AMD, arch="gfx1100")])
        assert chosen is Backend.ROCM
        assert chosen.uv_torch_backend == "rocm7.2"

    def test_apple_maps_to_mps_but_installs_cpu_wheel(self) -> None:
        chosen, _ = resolve([gpu(Vendor.APPLE)])
        assert chosen is Backend.MPS
        # Apple wheels come from plain PyPI; there is no mps index.
        assert chosen.uv_torch_backend == "cpu"
        assert chosen.accelerated is True

    def test_intel(self) -> None:
        assert resolve([gpu(Vendor.INTEL)])[0] is Backend.XPU

    def test_nvidia_wins_over_amd_on_mixed_host(self) -> None:
        chosen, _ = resolve(
            [gpu(Vendor.AMD, arch="gfx1100"), gpu(Vendor.NVIDIA)],
            cuda_version="12.8",
        )
        assert chosen is Backend.CU128

    def test_cpu_backend_not_accelerated(self) -> None:
        assert Backend.CPU.accelerated is False


class TestArchIssues:
    def test_supported_arch_is_silent(self) -> None:
        assert arch_issues([gpu(Vendor.AMD, arch="gfx1100")]) == []
        assert arch_issues([gpu(Vendor.AMD, arch="gfx942")]) == []

    def test_case_insensitive(self) -> None:
        assert arch_issues([gpu(Vendor.AMD, arch="GFX1100")]) == []

    def test_rdna2_without_the_override_is_an_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Measured, not assumed: this is a run that cannot start, not a caution.

        On an RX 6700 XT with ROCm 7.1 and no override set, torch reports the
        device as available and bf16-capable, and the first matmul of any dtype
        segfaults. Calling that a warning understates it by the whole distance
        between "rough edges" and "core dumped".
        """
        monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
        issues = arch_issues([gpu(Vendor.AMD, arch="gfx1030", name="RX 6900 XT")])
        assert len(issues) == 1
        assert issues[0].level == "error"
        assert "HSA_OVERRIDE_GFX_VERSION=10.3.0" in (issues[0].hint or "")

    def test_rdna2_with_the_matching_override_is_only_a_note(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Otherwise doctor exits 1 on a machine that trains perfectly well."""
        monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
        issues = arch_issues([gpu(Vendor.AMD, arch="gfx1030", name="RX 6900 XT")])
        assert len(issues) == 1
        assert issues[0].level == "info"

    def test_an_override_for_a_different_card_is_still_an_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Set is not the same as correct.

        A value copied from another machine, left over from a different card, or
        simply mistyped leaves the GPU exactly as unusable as no override at all
        — and reporting it as fine would wave through the failure this check
        exists to catch.
        """
        monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
        issues = arch_issues([gpu(Vendor.AMD, arch="gfx1030", name="RX 6900 XT")])
        assert len(issues) == 1
        assert issues[0].level == "error"
        # Both numbers, or the message does not say what to change it from.
        assert "11.0.0" in issues[0].message
        assert "10.3.0" in issues[0].message

    def test_a_supported_card_is_silent_either_way(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
        assert arch_issues([gpu(Vendor.AMD, arch="gfx1100")]) == []

    def test_the_gfx_name_may_carry_feature_suffixes(self) -> None:
        """torch reports gfx1031:sramecc-:xnack-, not the bare name."""
        from bloomery.probe.backend import override_hint

        assert override_hint("gfx1031:sramecc-:xnack-") == "10.3.0"
        assert override_hint("gfx1100") is None
        assert override_hint(None) is None

    def test_unknown_arch_gets_generic_warning(self) -> None:
        issues = arch_issues([gpu(Vendor.AMD, arch="gfx1250")])
        assert len(issues) == 1
        assert "supported target list" in issues[0].message

    def test_ignores_non_amd(self) -> None:
        assert arch_issues([gpu(Vendor.NVIDIA, arch="8.9")]) == []

    def test_ignores_amd_without_arch(self) -> None:
        assert arch_issues([gpu(Vendor.AMD)]) == []
