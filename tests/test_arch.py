# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the depth dial and config derivation."""

from __future__ import annotations

import pytest

from bloomery.arch import (
    actual_param_count,
    intermediate_size,
    resolve_spec,
    spec_from_depth,
    suggested_lr,
)
from bloomery.capability import LADDER, LADDER_BY_KEY


class TestSpecFromDepth:
    @pytest.mark.parametrize(
        ("depth", "hidden", "heads"),
        [(4, 256, 4), (8, 512, 8), (12, 768, 12), (26, 1664, 26)],
    )
    def test_derives_width_and_heads(self, depth: int, hidden: int, heads: int) -> None:
        spec = spec_from_depth(depth, vocab=8192)
        assert spec.layers == depth
        assert spec.hidden == hidden
        assert spec.heads == heads

    def test_depth_12_matches_gpt2_small_shape(self) -> None:
        """The dial should land on a known-good shape at depth 12."""
        spec = spec_from_depth(12, vocab=50257)
        preset = LADDER_BY_KEY["d12"]
        assert (spec.layers, spec.hidden, spec.heads) == (
            preset.layers,
            preset.hidden,
            preset.heads,
        )

    def test_head_dim_is_always_64(self) -> None:
        for depth in range(1, 40):
            spec = spec_from_depth(depth, vocab=1024)
            assert spec.hidden % spec.heads == 0
            assert spec.hidden // spec.heads == 64

    def test_rejects_nonsense_depth(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            spec_from_depth(0, vocab=1024)
        with pytest.raises(ValueError, match="at least 1"):
            spec_from_depth(-3, vocab=1024)


class TestResolveSpec:
    def test_named_preset(self) -> None:
        spec = resolve_spec(size="d12")
        assert spec.key == "d12"
        assert spec.layers == 12

    def test_depth(self) -> None:
        assert resolve_spec(depth=6, vocab=2048).layers == 6

    def test_vocab_overrides_preset(self) -> None:
        """The tokenizer that was actually trained wins over the preset."""
        spec = resolve_spec(size="d12", vocab=777)
        assert spec.vocab == 777

    def test_seq_and_batch_override(self) -> None:
        spec = resolve_spec(size="d4", seq=64, batch=3)
        assert (spec.seq, spec.batch) == (64, 3)

    def test_size_and_depth_together_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            resolve_spec(size="d12", depth=8)

    def test_unknown_size_lists_the_options(self) -> None:
        with pytest.raises(ValueError, match="unknown size") as caught:
            resolve_spec(size="enormous")
        assert "d12" in str(caught.value)

    def test_defaults_to_a_tiny_model(self) -> None:
        spec = resolve_spec()
        assert spec.layers == 4


class TestIntermediateSize:
    def test_approximates_eight_thirds(self) -> None:
        for hidden in (256, 512, 768, 2048, 4096):
            inner = intermediate_size(hidden)
            ratio = inner / hidden
            assert 2.5 < ratio < 2.85

    def test_is_hardware_friendly_multiple(self) -> None:
        for hidden in (256, 512, 768, 1664):
            assert intermediate_size(hidden) % 64 == 0


class TestParamAgreement:
    """The estimator and the real model must not drift apart.

    ``doctor`` prints memory numbers derived from ``ModelSpec.params``. If the
    config this module builds has a materially different parameter count, those
    numbers stop describing what actually gets trained.
    """

    @pytest.mark.parametrize("spec", LADDER, ids=lambda s: s.key)
    def test_estimate_within_five_percent_of_actual(self, spec) -> None:  # noqa: ANN001
        estimate = spec.params
        actual = actual_param_count(spec)
        assert abs(actual - estimate) / actual < 0.05, (
            f"{spec.key}: estimate {estimate:,} vs actual {actual:,}"
        )

    @pytest.mark.parametrize("depth", [4, 8, 12, 16, 24])
    def test_holds_for_derived_specs_too(self, depth: int) -> None:
        spec = spec_from_depth(depth, vocab=8192)
        assert abs(actual_param_count(spec) - spec.params) / actual_param_count(spec) < 0.05

    def test_actual_count_matches_a_real_model(self) -> None:
        """The arithmetic must match what transformers actually builds."""
        from bloomery.train.loop import build_model

        spec = spec_from_depth(2, vocab=512, seq=64)
        model = build_model(spec, eos_token_id=0, seed=0)
        # Tied embeddings mean the lm_head shares storage, so named_parameters
        # reports it once.
        built = sum(p.numel() for p in model.parameters())
        assert actual_param_count(spec) == built


class TestSuggestedLr:
    def test_anchor(self) -> None:
        spec = spec_from_depth(12, vocab=8192)  # 768 wide
        assert suggested_lr(spec) == pytest.approx(6e-4, rel=0.01)

    def test_falls_as_width_grows(self) -> None:
        narrow = suggested_lr(spec_from_depth(4, vocab=8192))
        wide = suggested_lr(spec_from_depth(32, vocab=8192))
        assert narrow > wide

    def test_stays_in_a_sane_band(self) -> None:
        for depth in (1, 4, 12, 32, 64):
            lr = suggested_lr(spec_from_depth(depth, vocab=8192))
            assert 1e-5 < lr < 5e-3
