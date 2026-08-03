# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for what a batch is scored against.

Every one of these pins a specific way the preference loss can be wrong while
still producing a plausible number and a run that appears to train.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from bloomery.arch import spec_from_depth  # noqa: E402
from bloomery.train.loop import LoraSettings, attach_adapter, build_model  # noqa: E402
from bloomery.train.objective import (  # noqa: E402
    IGNORE_INDEX,
    causal_loss,
    preference_loss,
    sequence_logprobs,
)

VOCAB = 64
SEQ = 12


@pytest.fixture
def plain() -> Any:
    """A dropout-free model, so a reference forward is deterministic.

    Rebuilt per test, not shared: ``get_peft_model`` replaces layers on the model
    it is given rather than wrapping a copy, so a fixture handed to one test
    comes back adapted for the next.
    """
    return build_model(spec_from_depth(1, vocab=VOCAB, seq=SEQ), eos_token_id=0, seed=0)


def preference_batch(pairs: int = 3, prompt: int = 4) -> dict[str, Any]:
    """Two stacked halves sharing a prompt, in the shape PreferenceSampler emits."""
    generator = torch.Generator().manual_seed(0)
    ids = torch.randint(1, VOCAB, (2 * pairs, SEQ), generator=generator)
    # The halves must agree on the prompt, which is the whole premise.
    ids[pairs:, :prompt] = ids[:pairs, :prompt]

    labels = ids.clone()
    labels[:, :prompt] = IGNORE_INDEX
    # One padded tail, so the padding path is exercised rather than assumed.
    labels[:, -2:] = IGNORE_INDEX
    attention = torch.ones_like(ids)
    attention[:, -2:] = 0
    return {"input_ids": ids, "labels": labels, "attention_mask": attention}


class TestSequenceLogprobs:
    def test_it_agrees_with_the_model_s_own_loss(self, plain: Any) -> None:
        """Pins the shift, which runs opposite to every other batch here.

        The samplers deliberately leave labels unshifted because the model shifts
        internally. This function shifts by hand, so an off-by-one is invisible
        except against the model's own arithmetic: cross-entropy is the mean
        negative log-probability over the scored positions, so multiplying it
        back out must reproduce the sum this returns.
        """
        drawn = preference_batch()
        scored = int((drawn["labels"][:, 1:] != IGNORE_INDEX).sum())

        with torch.no_grad():
            by_hand = float(sequence_logprobs(plain, drawn).sum())
            reported = float(plain(**drawn).loss)

        assert by_hand == pytest.approx(-reported * scored, rel=1e-3)

    def test_padding_contributes_nothing(self, plain: Any) -> None:
        """Or a longer row is scored on positions that are not there."""
        drawn = preference_batch()
        with torch.no_grad():
            before = sequence_logprobs(plain, drawn)
            # Change what sits under the padding. Nothing may move.
            drawn["input_ids"] = drawn["input_ids"].clone()
            drawn["input_ids"][:, -1] = (drawn["input_ids"][:, -1] + 7) % VOCAB
            after = sequence_logprobs(plain, drawn)
        assert torch.allclose(before, after)


class TestPreferenceLoss:
    def adapted(self, plain: Any) -> Any:
        """A fresh LoRA policy over the same weights, with dropout off.

        Dropout would make the policy stochastic and the ln(2) identity below
        approximate, which would turn an exact check into a loose one.
        """
        return attach_adapter(plain, LoraSettings(r=4, alpha=8, dropout=0.0))

    def test_an_untouched_policy_scores_exactly_ln_two(self, plain: Any) -> None:
        """The identity that says the reference is wired to the right weights.

        LoRA initialises so that it changes nothing, so policy and reference are
        the same distribution and every reward is zero. -logsigmoid(0) is ln 2.
        A reference pointing anywhere else — the wrong model, a stale copy, the
        policy itself with adapters left on — breaks this immediately.
        """
        model = self.adapted(plain)
        try:
            parts = preference_loss(model, preference_batch(), beta=0.1)
        finally:
            model.unload()

        assert float(parts.loss.detach()) == pytest.approx(math.log(2), abs=1e-5)
        assert parts.extras["reward_margin"] == pytest.approx(0.0, abs=1e-6)
        assert parts.extras["reward_chosen"] == pytest.approx(0.0, abs=1e-6)
        assert parts.extras["reward_rejected"] == pytest.approx(0.0, abs=1e-6)
        # Undecided, not wrong: a strict comparison would call this 0%.
        assert parts.extras["reward_accuracy"] == pytest.approx(0.5)

    def test_the_reference_is_the_base_and_does_not_move(self, plain: Any) -> None:
        """Training must not drag the thing it is measured against with it."""
        drawn = preference_batch()
        # Measured before attaching anything: get_peft_model edits the model it
        # is given, so afterwards there is no untouched handle left to compare
        # against.
        with torch.no_grad():
            base = sequence_logprobs(plain, drawn)

        model = self.adapted(plain)
        try:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if "lora_B" in name:
                        parameter.add_(torch.randn_like(parameter) * 0.05)

                moved = sequence_logprobs(model, drawn)
                model.eval()
                with model.disable_adapter():
                    reference = sequence_logprobs(model, drawn)
        finally:
            model.unload()

        assert not torch.allclose(moved, reference), "the adapter changed nothing"
        assert torch.allclose(reference, base, atol=1e-5)

    def test_everything_stays_in_float32_under_autocast(self, plain: Any) -> None:
        """bf16 log-probabilities carry an error the size of the signal.

        Guards the cross_entropy formulation: swapping it for log_softmax puts
        these back in bf16 and nothing else in the suite would notice.

        Runs only where bf16 is genuinely usable on this CPU. A bf16 matmul on a
        CPU whose kernels use instructions it does not implement does not raise —
        it takes an illegal-instruction trap and kills the process where it
        stands. This test did exactly that to a Windows CI runner: `0xc000001d`,
        exit 132, pytest dead at 72% through the suite.

        The gate is ``_probe_precision`` and not ``select_precision``, which are
        different questions. ``select_precision`` answers "what will bloomery
        use here", and honours ``BLOOMERY_PRECISION`` before probing anything —
        so a forced ``bf16`` on a CPU that cannot do it would walk straight past
        this guard into the trap it exists to prevent. ``_probe_precision``
        answers "can this CPU actually do it", which is the only question that
        keeps the process alive. An environment variable is a user asserting
        something about their hardware, and a test must not take anyone's word
        for a claim that can kill the process rather than fail it.

        Both halves of the probe matter: a runner that *claimed* bf16 survived an
        8x8 matmul and died on a 384x384 one, which is why the capability check
        alone is not the gate.

        A consequence worth naming: on hardware without usable bf16 this body
        does not run at all, and at the time of writing that included every CI
        platform here. So it cannot be the only thing guarding the formulation —
        see :meth:`test_the_autocast_asymmetry_this_relies_on`, which pins the
        same premise with no matmul and therefore runs everywhere.
        """
        from bloomery.train.device import _probe_precision

        dtype, _, reason = _probe_precision(torch.device("cpu"))
        if dtype is not torch.bfloat16:
            pytest.skip(f"bf16 is not usable on this CPU: {reason}")

        model = self.adapted(plain)
        try:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                drawn = preference_batch()
                assert sequence_logprobs(model, drawn).dtype is torch.float32
                assert preference_loss(model, drawn, beta=0.1).loss.dtype is torch.float32
        finally:
            model.unload()

    def test_the_autocast_asymmetry_this_relies_on(self) -> None:
        """The premise behind choosing cross_entropy, checked without a matmul.

        `sequence_logprobs` returns float32 under autocast for exactly one
        reason: torch promotes `cross_entropy` to fp32 and leaves `log_softmax`
        in bf16. That is a property of torch, not of this code — but this code
        depends on it, so if it ever changed the precision would regress
        silently and every other assertion here would still pass.

        No model and no matmul, so unlike the test above this runs everywhere,
        including on the CPUs where a bf16 matmul is fatal rather than slow.
        """
        import torch.nn.functional as functional

        logits = torch.randn(4, 16, dtype=torch.bfloat16)
        targets = torch.randint(0, 16, (4,))

        with torch.autocast("cpu", dtype=torch.bfloat16):
            promoted = functional.cross_entropy(logits, targets, reduction="none")
            left_alone = functional.log_softmax(logits, dim=-1)

        assert promoted.dtype is torch.float32, "cross_entropy is no longer promoted"
        assert left_alone.dtype is torch.bfloat16, "log_softmax is no longer the risky one"

    def test_training_it_raises_the_margin_rather_than_lowering_it(self, plain: Any) -> None:
        """Pins the sign, which the ln(2) identity cannot: at zero it is symmetric.

        With the two rewards swapped the loss still falls — it is being minimised
        either way — while the model learns to prefer the rejected answer. Only
        the direction the margin travels says which happened.
        """
        model = self.adapted(plain)
        try:
            drawn = preference_batch()
            optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=2.0)
            first = preference_loss(model, drawn, beta=0.1)
            for _ in range(25):
                optimizer.zero_grad()
                parts = preference_loss(model, drawn, beta=0.1)
                parts.loss.backward()
                optimizer.step()
            last = preference_loss(model, drawn, beta=0.1)
        finally:
            model.unload()

        assert float(last.loss.detach()) < float(first.loss.detach()), "nothing was learned at all"
        assert last.extras["reward_margin"] > first.extras["reward_margin"]
        assert last.extras["reward_chosen"] > last.extras["reward_rejected"]

    def test_it_counts_the_tokens_it_scored(self, plain: Any) -> None:
        model = self.adapted(plain)
        try:
            drawn = preference_batch()
            parts = preference_loss(model, drawn, beta=0.1)
        finally:
            model.unload()
        assert parts.supervised_tokens == int((drawn["labels"] != IGNORE_INDEX).sum())


class TestPreferenceScoring:
    """The eval metric has to move when the model's preference moves.

    Every other test of it checks shape and range, which a function returning a
    constant 0.5 would also satisfy.
    """

    def test_it_tracks_which_answer_the_model_actually_prefers(self, plain: Any) -> None:
        from bloomery.evaluate import score_preferences
        from bloomery.train.device import DeviceChoice, select_precision
        from bloomery.train.loop import LoraSettings, attach_adapter

        drawn = preference_batch(pairs=4)

        class OneBatch:
            """A sampler that keeps handing back the same pairs, so the only
            thing that changes between measurements is the model."""

            def batch(self, size: int, device: Any) -> dict[str, Any]:
                return drawn

        dtype, autocast, reason = select_precision(torch.device("cpu"))
        choice = DeviceChoice(
            device=torch.device("cpu"), dtype=dtype, autocast=autocast, reason=reason
        )

        model = attach_adapter(plain, LoraSettings(r=4, alpha=8, dropout=0.0))
        try:
            before, _, pairs = score_preferences(model, OneBatch(), choice, batch=4, batches=1)
            assert pairs == 4

            # Train it to prefer the chosen half of this very batch. Whatever it
            # scored before, it must score higher after.
            optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=2.0)
            for _ in range(30):
                optimizer.zero_grad()
                preference_loss(model, drawn, beta=0.1).loss.backward()
                optimizer.step()

            after, _, _ = score_preferences(model, OneBatch(), choice, batch=4, batches=1)
        finally:
            model.unload()

        assert after > before, f"the score did not follow the model: {before} -> {after}"
        assert after == pytest.approx(1.0), "it was trained on exactly these pairs"

    def test_the_two_measures_can_disagree(self, plain: Any) -> None:
        """Which is the whole reason both are reported.

        Summed log-probability falls as an answer lengthens, so a model can win
        on the sum by preferring brevity. Here the rejected answers are made
        much shorter than the chosen ones, so the sum favours them while the
        per-token mean is unaffected by the length difference.
        """
        from bloomery.evaluate import score_preferences
        from bloomery.train.device import DeviceChoice, select_precision
        from bloomery.train.objective import IGNORE_INDEX

        drawn = preference_batch(pairs=4)
        # Shorten the rejected half: mask most of its scored positions away.
        labels = drawn["labels"].clone()
        labels[4:, 6:] = IGNORE_INDEX
        drawn = {**drawn, "labels": labels}

        class OneBatch:
            def batch(self, size: int, device: Any) -> dict[str, Any]:
                return drawn

        dtype, autocast, reason = select_precision(torch.device("cpu"))
        choice = DeviceChoice(
            device=torch.device("cpu"), dtype=dtype, autocast=autocast, reason=reason
        )
        by_sum, by_token, _ = score_preferences(plain, OneBatch(), choice, batch=4, batches=1)

        # The short answers accumulate fewer negative terms, so they win on the
        # sum. Per token, length buys them nothing.
        assert by_sum < by_token, f"length bias did not show: {by_sum} vs {by_token}"


class TestCausalLoss:
    def test_it_is_the_model_s_own_loss(self, plain: Any) -> None:
        drawn = preference_batch()
        with torch.no_grad():
            assert float(causal_loss(plain, drawn).loss) == pytest.approx(
                float(plain(**drawn).loss)
            )

    def test_it_counts_only_unmasked_positions(self, plain: Any) -> None:
        drawn = preference_batch()
        with torch.no_grad():
            parts = causal_loss(plain, drawn)
        assert parts.supervised_tokens == int((drawn["labels"] != IGNORE_INDEX).sum())
        assert parts.extras == {}
