# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for weighted sampling, forgetting detection and precision selection."""

from __future__ import annotations

import contextlib
import math
from pathlib import Path
from typing import Any

import pytest

from bloomery.data import build_dataset, eot_id, synthetic_documents
from bloomery.mixture import Component, MixtureError, create, single
from bloomery.train.device import DeviceChoice, clear_precision_cache
from bloomery.train.mixing import (
    ForgettingTracker,
    MixtureSampler,
    weighted_mean,
)


@pytest.fixture
def cpu() -> DeviceChoice:
    import torch

    return DeviceChoice(
        device=torch.device("cpu"), dtype=torch.float32, autocast=False, reason="test"
    )


@pytest.fixture(scope="module")
def two_datasets(tokenizer: Any, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Two packed corpora sharing one tokenizer, so they are blendable."""
    root = tmp_path_factory.mktemp("blend")
    out = {}
    for name, seed in (("alpha", 1), ("beta", 2)):
        documents = synthetic_documents(400, seed=seed)
        out[name] = build_dataset(
            documents,
            tokenizer,
            out_dir=root / name,
            eot=eot_id(tokenizer),
            val_fraction=0.1,
        )
    return out


class TestMixtureSampler:
    def test_batch_shape_matches_request(self, two_datasets: Any, cpu: DeviceChoice) -> None:
        blend = create("b", [Component("alpha", 0.7), Component("beta", 0.3)])
        sampler = MixtureSampler(blend, two_datasets, "train", seq=32, seed=0)
        batch = sampler.batch(16, cpu.device)
        assert tuple(batch.shape) == (16, 32)

    def test_single_component_passes_straight_through(
        self, two_datasets: Any, cpu: DeviceChoice
    ) -> None:
        sampler = MixtureSampler(single("alpha"), two_datasets, "train", seq=32, seed=0)
        assert sampler.names == ("alpha",)
        assert tuple(sampler.batch(8, cpu.device).shape) == (8, 32)

    def test_long_run_proportions_track_the_weights(self, two_datasets: Any) -> None:
        """The multinomial draw must converge on the requested split.

        A batch of 8 cannot be divided 70/30 exactly; over many batches it has to
        average out, or replay data silently gets a different share than was
        asked for — which would defeat the entire point of the blend.
        """
        blend = create("b", [Component("alpha", 0.7), Component("beta", 0.3)])
        sampler = MixtureSampler(blend, two_datasets, "train", seq=16, seed=7)

        totals = {"alpha": 0, "beta": 0}
        for _ in range(400):
            for name, count in sampler.draw_counts(8).items():
                totals[name] += count

        drawn = sum(totals.values())
        assert drawn == 400 * 8
        assert totals["alpha"] / drawn == pytest.approx(0.7, abs=0.03)
        assert totals["beta"] / drawn == pytest.approx(0.3, abs=0.03)

    def test_draw_counts_always_sums_to_the_batch_size(self, two_datasets: Any) -> None:
        blend = create("b", [Component("alpha", 0.9), Component("beta", 0.1)])
        sampler = MixtureSampler(blend, two_datasets, "train", seq=16, seed=3)
        for size in (1, 2, 7, 16, 64):
            assert sum(sampler.draw_counts(size).values()) == size

    def test_single_component_draw_counts(self, two_datasets: Any) -> None:
        sampler = MixtureSampler(single("alpha"), two_datasets, "train", seq=16, seed=0)
        assert sampler.draw_counts(12) == {"alpha": 12}

    def test_effective_weights_are_normalised(self, two_datasets: Any) -> None:
        blend = create("b", [Component("alpha", 60), Component("beta", 40)])
        sampler = MixtureSampler(blend, two_datasets, "train", seq=16, seed=0)
        assert sum(sampler.effective_weights.values()) == pytest.approx(1.0)
        assert sampler.effective_weights["alpha"] == pytest.approx(0.6)

    def test_tokens_by_component(self, two_datasets: Any) -> None:
        blend = create("b", [Component("alpha", 1), Component("beta", 1)])
        sampler = MixtureSampler(blend, two_datasets, "train", seq=16, seed=0)
        by_component = sampler.tokens_by_component()
        assert set(by_component) == {"alpha", "beta"}
        assert sampler.tokens == sum(by_component.values())

    def test_unknown_dataset_is_named(self, two_datasets: Any) -> None:
        blend = create("b", [Component("alpha", 1), Component("ghost", 1)])
        with pytest.raises(KeyError, match="ghost"):
            MixtureSampler(blend, two_datasets, "train", seq=16, seed=0)

    def test_unusable_component_is_dropped_and_reweighted(
        self, tokenizer: Any, two_datasets: Any, tmp_path: Path, cpu: DeviceChoice
    ) -> None:
        """A component too small for the sequence length is dropped, not fatal.

        The remaining weights must renormalise to 1, or every batch would come
        back short and the effective batch size would silently change.
        """
        tiny = build_dataset(
            synthetic_documents(3, seed=5),
            tokenizer,
            out_dir=tmp_path / "tiny",
            eot=eot_id(tokenizer),
            val_fraction=0.0,
        )
        datasets = {**two_datasets, "tiny": tiny}
        blend = create("b", [Component("alpha", 0.5), Component("tiny", 0.5)])

        # 512 tokens is far more than three short documents hold.
        sampler = MixtureSampler(blend, datasets, "train", seq=512, seed=0)

        assert "tiny" in sampler.skipped
        assert sampler.names == ("alpha",)
        assert sampler.effective_weights == pytest.approx({"alpha": 1.0})
        # And the batch is still the size that was requested.
        assert tuple(sampler.batch(4, cpu.device).shape) == (4, 512)

    def test_all_components_unusable_is_an_error(self, two_datasets: Any) -> None:
        blend = create("b", [Component("alpha", 1)])
        with pytest.raises(ValueError, match="no usable"):
            MixtureSampler(blend, two_datasets, "train", seq=10_000_000, seed=0)

    def test_components_draw_different_offsets(self, two_datasets: Any, cpu: DeviceChoice) -> None:
        """Per-component seed offsets stop two components sampling in lockstep."""
        blend = create("b", [Component("alpha", 1), Component("beta", 1)])
        sampler = MixtureSampler(blend, two_datasets, "train", seq=16, seed=0)
        samplers = sampler.component_samplers()
        first = samplers["alpha"].batch(4, cpu.device)
        second = samplers["beta"].batch(4, cpu.device)
        assert not first.equal(second)


class TestForgettingTracker:
    def test_first_measurement_is_never_a_regression(self) -> None:
        tracker = ForgettingTracker()
        assert tracker.update({"a": 2.0}) == {}
        assert tracker.best["a"] == 2.0

    def test_improvement_updates_the_best(self) -> None:
        tracker = ForgettingTracker()
        tracker.update({"a": 2.0})
        assert tracker.update({"a": 1.5}) == {}
        assert tracker.best["a"] == 1.5

    def test_regression_is_reported_with_its_delta(self) -> None:
        tracker = ForgettingTracker(tolerance=0.0)
        tracker.update({"a": 1.0})
        regressed = tracker.update({"a": 1.4})
        assert regressed == pytest.approx({"a": 0.4})

    def test_small_wobble_is_within_tolerance(self) -> None:
        """Validation loss on a small split is noisy; flagging every tick is useless."""
        tracker = ForgettingTracker(tolerance=0.05)
        tracker.update({"a": 1.0})
        assert tracker.update({"a": 1.02}) == {}

    def test_regression_does_not_move_the_best(self) -> None:
        tracker = ForgettingTracker(tolerance=0.0)
        tracker.update({"a": 1.0})
        tracker.update({"a": 2.0})
        assert tracker.best["a"] == 1.0

    def test_catches_one_component_regressing_while_another_improves(self) -> None:
        """The whole point: an aggregate can fall while an old corpus degrades."""
        tracker = ForgettingTracker(tolerance=0.0)
        tracker.update({"new": 2.0, "old": 1.0})
        regressed = tracker.update({"new": 1.0, "old": 1.5})
        assert set(regressed) == {"old"}

    def test_ignores_non_finite_losses(self) -> None:
        tracker = ForgettingTracker()
        assert tracker.update({"a": float("nan")}) == {}
        assert "a" not in tracker.best

    def test_improvement_from_first_to_best(self) -> None:
        tracker = ForgettingTracker(tolerance=0.0)
        tracker.update({"a": 3.0})
        tracker.update({"a": 1.0})
        assert tracker.improvement()["a"] == pytest.approx(2.0)

    def test_improvement_is_zero_when_never_beaten(self) -> None:
        """Zero, not negative: `best` only ever moves downward."""
        tracker = ForgettingTracker(tolerance=0.0)
        tracker.update({"a": 1.0})
        tracker.update({"a": 5.0})
        assert tracker.improvement()["a"] == pytest.approx(0.0)


class TestWeightedMean:
    def test_weights_the_components(self) -> None:
        assert weighted_mean({"a": 1.0, "b": 2.0}, {"a": 0.5, "b": 0.5}) == pytest.approx(1.5)
        assert weighted_mean({"a": 1.0, "b": 2.0}, {"a": 0.9, "b": 0.1}) == pytest.approx(1.1)

    def test_skips_non_finite(self) -> None:
        result = weighted_mean({"a": 1.0, "b": float("nan")}, {"a": 0.5, "b": 0.5})
        assert result == pytest.approx(1.0)

    def test_no_usable_components(self) -> None:
        assert math.isnan(weighted_mean({}, {}))

    def test_ignores_zero_weight_components(self) -> None:
        result = weighted_mean({"a": 1.0, "b": 99.0}, {"a": 1.0, "b": 0.0})
        assert result == pytest.approx(1.0)


class TestResolve:
    def test_refuses_mismatched_tokenizers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tokenizer: Any
    ) -> None:
        """Blending different token spaces would train on nonsense without erroring."""
        from bloomery import paths
        from bloomery.data import train_tokenizer
        from bloomery.train.mixing import resolve

        monkeypatch.setenv("BLOOMERY_HOME", str(tmp_path / "home"))
        documents = synthetic_documents(300, seed=3)

        # Same corpus, deliberately two different tokenizers.
        for name, vocab in (("one", 300), ("two", 500)):
            local = train_tokenizer(documents, vocab_size=vocab, out_dir=paths.tokenizer_dir(name))
            build_dataset(
                documents,
                local,
                out_dir=paths.tokens_dir(name),
                eot=eot_id(local),
                val_fraction=0.1,
            )

        blend = create("mismatch", [Component("one", 1), Component("two", 1)])
        with pytest.raises(MixtureError, match="different tokenizers") as caught:
            resolve(blend)
        # The message must name the actual figures, since the fix is not guessable.
        assert "vocab" in str(caught.value)
        assert "Re-prepare" in str(caught.value)

    def test_unprepared_dataset_says_how_to_prepare_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from bloomery.train.mixing import resolve

        monkeypatch.setenv("BLOOMERY_HOME", str(tmp_path / "home2"))
        with pytest.raises(MixtureError, match="bloomery prepare"):
            resolve(single("never-made"))


class TestPrecisionSelection:
    """Regression cover for a 46x slowdown that shipped unnoticed.

    bf16 was selected whenever it produced arithmetically correct results. On
    hardware without native bf16 it is emulated: correct and far slower. A short
    test run cannot see the difference, so these tests pin the decision rule
    rather than the timing.

    These cover the decision made *after* the hardware gate passes, so they stub
    it open. The gate itself — which stops a bf16 matmul ever running on a CPU
    that would take an illegal-instruction trap on it — is covered by
    TestCpuBf16Gate in test_train.py.
    """

    def test_requires_speed_not_just_correctness(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import torch

        from bloomery.train import device as dev

        clear_precision_cache()
        monkeypatch.delenv(dev.ENV_PRECISION, raising=False)
        monkeypatch.setattr(dev, "_cpu_bf16_is_native", lambda: True)
        monkeypatch.setattr(dev, "_bf16_probe_says_faster", lambda _d, **_k: False)

        dtype, autocast, reason = dev.select_precision(torch.device("cpu"))
        assert dtype is torch.float32
        assert autocast is False
        assert "not usably faster" in reason

    def test_uses_bf16_when_it_is_genuinely_faster(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import torch

        from bloomery.train import device as dev

        clear_precision_cache()
        monkeypatch.delenv(dev.ENV_PRECISION, raising=False)
        monkeypatch.setattr(dev, "_cpu_bf16_is_native", lambda: True)
        monkeypatch.setattr(dev, "_bf16_probe_says_faster", lambda _d, **_k: True)

        dtype, autocast, _ = dev.select_precision(torch.device("cpu"))
        assert dtype is torch.bfloat16
        assert autocast is True

    def test_env_override_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import torch

        from bloomery.train import device as dev

        clear_precision_cache()
        monkeypatch.setenv(dev.ENV_PRECISION, "bf16")
        dtype, autocast, reason = dev.select_precision(torch.device("cpu"))
        assert dtype is torch.bfloat16
        assert autocast is True
        assert "forced by" in reason

    def test_unknown_override_is_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import torch

        from bloomery.train import device as dev

        clear_precision_cache()
        monkeypatch.setenv(dev.ENV_PRECISION, "float8-ish")
        dtype, _, reason = dev.select_precision(torch.device("cpu"))
        assert dtype in (torch.float32, torch.bfloat16)
        assert "forced by" not in reason

    def test_result_is_memoised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import torch

        from bloomery.train import device as dev

        clear_precision_cache()
        monkeypatch.delenv(dev.ENV_PRECISION, raising=False)
        calls = {"n": 0}

        def counted(_device: Any, **_kwargs: Any) -> bool:
            calls["n"] += 1
            return False

        monkeypatch.setattr(dev, "_cpu_bf16_is_native", lambda: True)
        monkeypatch.setattr(dev, "_bf16_probe_says_faster", counted)

        dev.select_precision(torch.device("cpu"))
        dev.select_precision(torch.device("cpu"))
        assert calls["n"] == 1

    def test_cuda_rejects_emulated_bf16(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`is_bf16_supported()` defaults to including_emulation=True.

        Accepting that default would reintroduce, on CUDA, exactly the
        correct-but-slow trap this module exists to avoid on CPU and Metal.
        """
        import torch

        from bloomery.train import device as dev

        seen: dict[str, object] = {}

        def fake_is_bf16_supported(including_emulation: bool = True) -> bool:
            seen["including_emulation"] = including_emulation
            # Native support absent; only emulation available.
            return bool(including_emulation)

        clear_precision_cache()
        monkeypatch.delenv(dev.ENV_PRECISION, raising=False)
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", fake_is_bf16_supported)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        monkeypatch.setattr(torch.cuda, "device", lambda _d: contextlib.nullcontext())

        dtype, autocast, reason = dev._probe_precision(torch.device("cuda"))
        assert seen["including_emulation"] is False
        assert dtype is torch.float16
        assert "without native bf16" in reason
        assert autocast is True
        clear_precision_cache()

    def test_cuda_cache_key_is_per_device(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two cards of different generations must not share one answer."""
        import torch

        from bloomery.train.device import _cache_key

        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        assert _cache_key(torch.device("cuda:0")) != _cache_key(torch.device("cuda:1"))
        assert _cache_key(torch.device("cpu")) == "cpu"
        assert _cache_key(torch.device("mps")) == "mps"

    def test_real_probe_returns_a_usable_answer(self) -> None:
        """Whatever this machine is, the probe must produce a workable dtype."""
        import torch

        clear_precision_cache()
        from bloomery.train.device import select_precision

        dtype, autocast, reason = select_precision(torch.device("cpu"))
        assert dtype in (torch.float32, torch.bfloat16)
        assert isinstance(autocast, bool)
        assert reason
        clear_precision_cache()
