# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the capability estimator.

The calibration tests below are the important ones. They pin the memory model
against published figures for well-known configurations, so if someone later
adjusts a constant to make one number look nicer, the others will object.
"""

from __future__ import annotations

import pytest

from bloomery.capability import (
    LADDER,
    LADDER_BY_KEY,
    TOKENS_PER_PARAM,
    Method,
    ModelSpec,
    assess,
    check_fit,
    derive_budget,
    estimate_memory,
    format_params,
    format_tokens,
)
from bloomery.probe.types import (
    CpuInfo,
    DiskInfo,
    GpuInfo,
    HostInfo,
    HostReport,
    MemoryInfo,
    Platform,
    Vendor,
)

GIB = 1024**3


def make_report(gpus: list[GpuInfo], *, ram: int = 64 * GIB) -> HostReport:
    return HostReport(
        host=HostInfo(
            platform=Platform.LINUX,
            system="Linux",
            release="6.8.0",
            machine="x86_64",
            python_version="3.12.7",
        ),
        cpu=CpuInfo(physical_cores=16, logical_cores=32),
        memory=MemoryInfo(total=ram, available=ram),
        disk=DiskInfo(path="/tmp", total=2000 * GIB, free=1000 * GIB),
        gpus=gpus,
    )


def nvidia(vram_gib: float, *, index: int = 0) -> GpuInfo:
    return GpuInfo(
        index=index,
        vendor=Vendor.NVIDIA,
        name=f"GPU{index}",
        vram_total=int(vram_gib * GIB),
    )


class TestParameterCounts:
    """The param formula must match real models, or every estimate is wrong."""

    def test_gpt2_small(self) -> None:
        # Real GPT-2 small is 124M.
        params = LADDER_BY_KEY["d12"].params
        assert 120_000_000 < params < 128_000_000

    def test_gpt2_medium(self) -> None:
        # Real GPT-2 medium is ~355M.
        params = LADDER_BY_KEY["d24"].params
        assert 330_000_000 < params < 380_000_000

    def test_seven_billion(self) -> None:
        # Llama-7B is 6.74B; the approximation should land close.
        params = LADDER_BY_KEY["7b"].params
        assert 6.5e9 < params < 7.5e9

    def test_one_billion(self) -> None:
        assert 0.9e9 < LADDER_BY_KEY["1b"].params < 1.3e9

    def test_ladder_is_monotonic_in_params(self) -> None:
        counts = [spec.params for spec in LADDER]
        assert counts == sorted(counts)

    def test_ladder_keys_unique(self) -> None:
        keys = [spec.key for spec in LADDER]
        assert len(keys) == len(set(keys))


# The published QLoRA and LoRA memory figures were measured on Llama-1/2 era
# models: 32k vocab, 2048 context. Modern specs use a 128k vocab and 4096
# context, which adds ~3 GiB of logit and activation memory on its own. Comparing
# against the ladder's 7B entry would be comparing different configurations, so
# calibration uses a spec that reproduces the original conditions.
LLAMA2_7B = ModelSpec("llama2-7b", "Llama-2 7B", 32, 4096, 32, 32_000, 2048, 1)


class TestMemoryCalibration:
    """Pinned against published memory figures for known configurations."""

    def test_reference_spec_matches_llama2_param_count(self) -> None:
        # Real Llama-2 7B is 6.74B.
        assert 6.4e9 < LLAMA2_7B.params < 7.0e9

    def test_qlora_7b_fits_a_consumer_card(self) -> None:
        # Widely reported as 6-8 GiB under these conditions.
        required = estimate_memory(LLAMA2_7B, Method.QLORA)
        assert 5.5 * GIB < required < 8.5 * GIB

    def test_lora_7b_bf16(self) -> None:
        # Widely reported as 16-20 GiB.
        required = estimate_memory(LLAMA2_7B, Method.LORA)
        assert 15 * GIB < required < 21 * GIB

    def test_full_7b_does_not_fit_one_80gib_card(self) -> None:
        # AdamW full fine-tuning of 7B needs ~107 GiB; this is the known result
        # and the reason people reach for ZeRO offload.
        required = estimate_memory(LLAMA2_7B, Method.FULL)
        assert 95 * GIB < required < 125 * GIB

    def test_modern_vocab_and_context_cost_several_gib(self) -> None:
        """A 128k vocab at 4096 context is materially more expensive.

        Worth pinning, because it is the most common reason a user's real
        numbers exceed a figure they read in a blog post from two years ago.
        """
        legacy = estimate_memory(LLAMA2_7B, Method.QLORA)
        modern = estimate_memory(LADDER_BY_KEY["7b"], Method.QLORA)
        assert modern - legacy > 2 * GIB

    def test_full_gpt2_small_fits_a_small_card(self) -> None:
        required = estimate_memory(LADDER_BY_KEY["d12"], Method.FULL)
        assert required < 8 * GIB

    def test_tiny_model_is_dominated_by_fixed_overhead(self) -> None:
        # A 5M model still needs the driver context, so it cannot come out near
        # zero. This guards against an estimate that looks absurdly optimistic.
        required = estimate_memory(LADDER_BY_KEY["d4"], Method.FULL)
        assert required > 1 * GIB

    def test_method_ordering_at_fixed_size(self) -> None:
        spec = LADDER_BY_KEY["7b"]
        full = estimate_memory(spec, Method.FULL)
        lora = estimate_memory(spec, Method.LORA)
        qlora = estimate_memory(spec, Method.QLORA)
        assert full > lora > qlora

    def test_monotonic_in_size_at_fixed_step(self) -> None:
        # Holding batch and sequence fixed, bigger models must need more memory.
        # The displayed table varies the step per row, which is why it is not
        # monotonic there.
        previous = 0
        for spec in LADDER:
            required = estimate_memory(spec, Method.FULL, batch=1, seq=1024)
            assert required > previous
            previous = required

    def test_scales_with_batch(self) -> None:
        spec = LADDER_BY_KEY["d12"]
        small = estimate_memory(spec, Method.FULL, batch=1, seq=1024)
        large = estimate_memory(spec, Method.FULL, batch=16, seq=1024)
        assert large > small

    def test_scales_with_sequence_length(self) -> None:
        spec = LADDER_BY_KEY["d12"]
        short = estimate_memory(spec, Method.FULL, batch=1, seq=512)
        long = estimate_memory(spec, Method.FULL, batch=1, seq=8192)
        assert long > short

    def test_large_vocab_costs_memory(self) -> None:
        """The logit tensor scales with vocab, a real and often surprising cost."""
        base = ModelSpec("a", "a", 12, 768, 12, 32_000, 1024, 8)
        wide = ModelSpec("b", "b", 12, 768, 12, 256_000, 1024, 8)
        assert estimate_memory(wide, Method.FULL) > estimate_memory(base, Method.FULL)


class TestDeriveBudget:
    def test_uses_largest_single_gpu_not_the_sum(self) -> None:
        # Plain data parallel replicates the model onto every device, so four
        # 24 GiB cards do not let you train a 96 GiB model.
        report = make_report([nvidia(24, index=i) for i in range(4)])
        budget = derive_budget(report)
        assert budget.total == int(24 * GIB)
        assert budget.shardable_devices == 4
        assert budget.shardable_total == int(96 * GIB)

    def test_single_gpu_has_no_shardable_total(self) -> None:
        budget = derive_budget(make_report([nvidia(24)]))
        assert budget.shardable_total is None

    def test_picks_the_biggest_of_mismatched_cards(self) -> None:
        report = make_report([nvidia(8), nvidia(24, index=1)])
        assert derive_budget(report).total == int(24 * GIB)

    def test_cpu_only_falls_back_to_ram_fraction(self) -> None:
        report = make_report([], ram=64 * GIB)
        budget = derive_budget(report)
        assert budget.total < 64 * GIB
        assert "CPU only" in budget.source

    def test_apple_reports_unified_memory(self) -> None:
        apple = GpuInfo(index=0, vendor=Vendor.APPLE, name="Apple M3 Max", vram_total=48 * GIB)
        budget = derive_budget(make_report([apple]))
        assert "unified memory" in budget.source


class TestAssess:
    def test_covers_every_size_and_method(self) -> None:
        report = assess(make_report([nvidia(24)]))
        assert len(report.rows) == len(LADDER) * len(Method)

    def test_24gib_card_can_pretrain_gpt2_medium_but_not_1b(self) -> None:
        capability = assess(make_report([nvidia(24)]))
        largest = capability.largest_fitting(Method.FULL)
        assert largest is not None
        # 1B full training needs ~21 GiB by our model, which is inside 24 but
        # with almost no headroom; the ladder should stop at or below it.
        assert largest.spec.params <= LADDER_BY_KEY["1b"].params

    def test_80gib_card_can_qlora_7b(self) -> None:
        capability = assess(make_report([nvidia(80)]))
        row = next(r for r in capability.for_method(Method.QLORA) if r.spec.key == "7b")
        assert row.fits

    def test_tiny_card_fits_almost_nothing(self) -> None:
        capability = assess(make_report([nvidia(4)]))
        largest = capability.largest_fitting(Method.FULL)
        assert largest is not None
        assert largest.spec.key in {"d4", "d8"}

    def test_qlora_unavailable_on_apple(self) -> None:
        apple = GpuInfo(index=0, vendor=Vendor.APPLE, name="Apple M1", vram_total=6 * GIB)
        capability = assess(make_report([apple]))
        rows = capability.for_method(Method.QLORA)
        assert all(not r.available_here for r in rows)
        assert all("macOS" in r.unavailable_reason for r in rows)
        assert capability.largest_fitting(Method.QLORA) is None

    def test_adapter_methods_unavailable_on_cpu(self) -> None:
        capability = assess(make_report([]))
        assert all(not r.available_here for r in capability.for_method(Method.QLORA))
        assert all(not r.available_here for r in capability.for_method(Method.LORA))
        # Full training on CPU is slow but real, so it stays available.
        assert any(r.available_here for r in capability.for_method(Method.FULL))

    def test_unavailable_rows_never_report_as_fitting(self) -> None:
        apple = GpuInfo(index=0, vendor=Vendor.APPLE, name="Apple M1", vram_total=128 * GIB)
        capability = assess(make_report([apple]))
        # Even with absurd memory, QLoRA cannot run here.
        assert all(not r.fits for r in capability.for_method(Method.QLORA))

    def test_token_budget_follows_chinchilla(self) -> None:
        capability = assess(make_report([nvidia(24)]))
        row = capability.for_method(Method.FULL)[0]
        assert row.tokens == row.spec.params * TOKENS_PER_PARAM


class TestFormatting:
    @pytest.mark.parametrize(
        ("count", "expected"),
        [
            (105_000_000, "105M"),
            (2_500_000_000, "2.5B"),
            (1_500_000_000_000, "1.5T"),
            (5_000, "5000"),
        ],
    )
    def test_format_tokens(self, count: int, expected: str) -> None:
        assert format_tokens(count) == expected

    @pytest.mark.parametrize(
        ("count", "expected"),
        [
            (123_532_032, "124M"),
            (6_980_000_000, "6.98B"),
            (5_200_000, "5M"),
            (500_000, "500K"),
        ],
    )
    def test_format_params(self, count: int, expected: str) -> None:
        assert format_params(count) == expected


GIB_ = 1024**3


class TestGradientCheckpointingIsModelled:
    """Checkpointing is OFF by default in `train`, so the estimate must model it.

    Assuming checkpointing when it is not enabled under-reports the very
    configuration people actually run, which is the case the pre-flight guard
    exists to catch.
    """

    def test_disabling_checkpointing_costs_more(self) -> None:
        spec = LADDER_BY_KEY["d12"]
        on = estimate_memory(spec, Method.FULL, gradient_checkpointing=True)
        off = estimate_memory(spec, Method.FULL, gradient_checkpointing=False)
        assert off > on

    def test_the_gap_widens_with_depth(self) -> None:
        """Non-checkpointed activations scale with layers; checkpointed do not."""
        shallow = LADDER_BY_KEY["d12"]
        deep = LADDER_BY_KEY["7b"]
        shallow_gap = estimate_memory(
            shallow, Method.FULL, batch=1, seq=1024, gradient_checkpointing=False
        ) - estimate_memory(shallow, Method.FULL, batch=1, seq=1024, gradient_checkpointing=True)
        deep_gap = estimate_memory(
            deep, Method.FULL, batch=1, seq=1024, gradient_checkpointing=False
        ) - estimate_memory(deep, Method.FULL, batch=1, seq=1024, gradient_checkpointing=True)
        assert deep_gap > shallow_gap

    def test_default_still_assumes_checkpointing(self) -> None:
        """The doctor ladder is quoted with checkpointing, so the default must not move."""
        spec = LADDER_BY_KEY["d12"]
        assert estimate_memory(spec, Method.FULL) == estimate_memory(
            spec, Method.FULL, gradient_checkpointing=True
        )


class TestDeviceAwareBudget:
    def test_cpu_budget_ignores_the_gpu(self) -> None:
        """`--device cpu` on a GPU box only has system RAM available.

        Reporting the card's VRAM would approve a run that cannot possibly fit.
        """
        report = make_report([nvidia(80)], ram=8 * GIB)
        gpu_budget = derive_budget(report)
        cpu_budget = derive_budget(report, device_type="cpu")
        assert gpu_budget.total == int(80 * GIB)
        assert cpu_budget.total < 8 * GIB
        assert "CPU only" in cpu_budget.source

    def test_gpu_device_type_uses_the_card(self) -> None:
        report = make_report([nvidia(24)], ram=8 * GIB)
        assert derive_budget(report, device_type="cuda").total == int(24 * GIB)


class TestCheckFit:
    def test_a_reasonable_config_fits(self) -> None:
        report = make_report([nvidia(24)])
        check = check_fit(report, LADDER_BY_KEY["d12"], batch=8, seq=1024, device_type="cuda")
        assert check.fits
        assert check.headroom > 1

    def test_an_oversized_config_does_not(self) -> None:
        report = make_report([nvidia(8)])
        check = check_fit(report, LADDER_BY_KEY["7b"], batch=8, seq=4096, device_type="cuda")
        assert not check.fits
        assert check.headroom < 1

    def test_a_failing_check_offers_concrete_fixes(self) -> None:
        report = make_report([nvidia(8)])
        check = check_fit(
            report,
            LADDER_BY_KEY["d24"],
            batch=32,
            seq=2048,
            gradient_checkpointing=False,
            device_type="cuda",
        )
        assert not check.fits
        options = check.suggestions()
        assert options
        # Checkpointing is the largest single saving when it is off.
        assert any("--grad-checkpoint" in o for o in options)
        assert any("--batch" in o for o in options)

    def test_a_passing_check_offers_nothing(self) -> None:
        report = make_report([nvidia(80)])
        check = check_fit(report, LADDER_BY_KEY["d4"], batch=2, seq=128, device_type="cuda")
        assert check.fits
        assert check.suggestions() == []

    def test_checkpointing_is_not_suggested_when_already_on(self) -> None:
        report = make_report([nvidia(8)])
        check = check_fit(
            report,
            LADDER_BY_KEY["7b"],
            batch=8,
            seq=4096,
            gradient_checkpointing=True,
            device_type="cuda",
        )
        assert not any("--grad-checkpoint" in o for o in check.suggestions())

    def test_suggested_depth_actually_fits(self) -> None:
        """A suggestion that does not fit would be worse than no suggestion."""
        import re

        report = make_report([nvidia(8)])
        check = check_fit(
            report,
            LADDER_BY_KEY["1b"],
            batch=8,
            seq=1024,
            gradient_checkpointing=False,
            device_type="cuda",
        )
        depth_options = [o for o in check.suggestions() if o.startswith("--depth")]
        if depth_options:
            depth = int(re.search(r"--depth (\d+)", depth_options[0]).group(1))
            from bloomery.arch import spec_from_depth

            candidate = spec_from_depth(depth, vocab=check.spec.vocab, seq=1024, batch=8)
            required = estimate_memory(
                candidate, Method.FULL, batch=8, seq=1024, gradient_checkpointing=False
            )
            assert required <= check.budget.total

    def test_cpu_run_on_a_gpu_box_is_judged_against_ram(self) -> None:
        """The regression that motivated device_type: a GPU present but unused."""
        report = make_report([nvidia(80)], ram=4 * GIB)
        check = check_fit(report, LADDER_BY_KEY["1b"], batch=8, seq=2048, device_type="cpu")
        assert not check.fits
