# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What a batch is scored against.

The training loop does not know what it is optimising. It draws a batch, gets a
number to call backward on, and gets told how many tokens that number covered —
everything else it does, from gradient accumulation to the checkpoint dance, is
the same whatever the objective.

Keeping that here rather than branching inside the loop is what stops a second
objective becoming a second loop. A parallel loop would have to reimplement the
non-finite guard, the unscale-then-clip ordering, resume, the run.jsonl contract
and the forgetting tracker, and would drift from this one the first time either
was fixed.

These are plain functions over ``(model, batch)``. No run directory, no
optimizer, no metrics writer — so they can be checked against arithmetic rather
than against a training run.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

# Positions cross-entropy skips. The value is torch's own convention, not ours.
IGNORE_INDEX = -100

# A batch is a plain dict so it can be splatted straight into a model that takes
# input_ids/labels/attention_mask. What is in it depends on the sampler.
Batch = dict[str, "torch.Tensor"]


@dataclass(frozen=True, slots=True)
class LossParts:
    """One microbatch's contribution.

    ``supervised_tokens`` is what the loss was actually computed over, which is
    not the batch size once anything is masked. It lives here rather than in the
    loop because only the objective knows which positions counted.

    ``extras`` are objective-specific numbers for the metrics stream — empty for
    plain language modelling, and the reward statistics for preference training.
    They are floats rather than tensors so nothing keeps a graph alive.
    """

    loss: Any
    supervised_tokens: int
    extras: dict[str, float] = field(default_factory=dict)


# What the loop is handed. Everything objective-specific — which sampler drew the
# batch, which hyperparameters apply — is closed over before the loop starts, so
# the step body never asks what it is training.
Objective = Callable[[Any, Batch], LossParts]


def causal_loss(model: Any, drawn: Batch) -> LossParts:
    """Next-token prediction, scored by the model itself.

    The labels are handed over unshifted because the model shifts internally —
    see :meth:`BatchSampler.batch`, which builds them that way on purpose.
    """
    labels = drawn["labels"]
    return LossParts(
        loss=model(**drawn).loss,
        supervised_tokens=int((labels != IGNORE_INDEX).sum().item()),
    )


def sequence_logprobs(model: Any, drawn: Batch) -> Any:
    """Total log-probability the model assigns to each row's scored tokens.

    Shifted by hand here, which is the opposite of every other batch in this
    project: the samplers deliberately leave labels unshifted because the model
    does it internally when computing its own loss (see
    :meth:`BatchSampler.batch`). Nothing computes the loss for us here, so the
    alignment is ours to get right — position ``t`` of the logits predicts token
    ``t + 1``.

    ``F.cross_entropy`` rather than ``log_softmax(...).gather(...)``, and that is
    a measurement rather than a preference. Logits arrive from a bf16 matmul
    under autocast; cross-entropy is on torch's fp32 autocast list and
    log_softmax is not, so the obvious formulation silently stays in bf16.
    Summed over 512 positions at a 128k vocabulary, against an fp64 reference:

        F.cross_entropy (fp32)            +0.114 nats
        log_softmax, fp32 accumulation    -0.853 nats
        log_softmax, bf16 accumulation    -4.353 nats

    A DPO logit is O(1) and these enter it scaled by beta, so the last is error
    of the same size as the signal. It also avoids materialising a separate fp32
    log-softmax over the whole vocabulary, which at a realistic batch is a
    gigabyte that cross-entropy never allocates.
    """
    import torch.nn.functional as functional

    logits = model(
        input_ids=drawn["input_ids"],
        attention_mask=drawn.get("attention_mask"),
        # Nothing generates from these rows, so the cache is allocated for every
        # forward and read by none of them. Four sequence-forwards per
        # microbatch makes that worth turning off rather than tolerating.
        use_cache=False,
    ).logits

    predicted = logits[:, :-1]
    targets = drawn["labels"][:, 1:]
    token_logprobs = -functional.cross_entropy(
        predicted.reshape(-1, predicted.shape[-1]),
        targets.reshape(-1),
        reduction="none",
        ignore_index=IGNORE_INDEX,
    ).view(targets.shape)
    # Deliberately not cast to float32 here. It already is one — autocast
    # promotes cross-entropy — and casting at the end would hide it if this were
    # ever rewritten in a way that did not, which is the whole failure above.
    # The dtype is asserted in the tests for that reason.
    # cross_entropy returns 0 at ignored positions, so summing needs no mask.
    return token_logprobs.sum(dim=-1)


def preference_loss(model: Any, drawn: Batch, *, beta: float) -> LossParts:
    """Direct preference optimization against the model's own untuned self.

    The batch is two stacked halves — chosen answers, then rejected ones — and
    row ``i`` pairs with row ``i + half``. The objective raises the model's
    log-probability of the preferred answer relative to the other, measured
    against a reference that does not move, so that it cannot win by simply
    becoming more confident about everything.

    The reference is this same model with its adapters switched off, which costs
    no second copy of the weights. It is computed under ``eval()`` as well as
    ``no_grad()``: a base model with dropout of its own — GPT-2 has three kinds —
    would otherwise give a reference that is different every time it is asked,
    and that noise lands directly in the only training signal there is.
    """
    import torch
    import torch.nn.functional as functional

    half = drawn["input_ids"].shape[0] // 2
    policy = sequence_logprobs(model, drawn)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), model.disable_adapter():
            reference = sequence_logprobs(model, drawn)
    finally:
        if was_training:
            model.train()

    # The reward is how far the model has moved from its reference on this
    # answer. Reported per side, not only as the margin: the best-known way for
    # a DPO run to go wrong is for both to fall while the margin still grows,
    # which a margin-only view reports as a healthy run.
    chosen_reward = beta * (policy[:half] - reference[:half])
    rejected_reward = beta * (policy[half:] - reference[half:])
    loss = -functional.logsigmoid(chosen_reward - rejected_reward).mean()

    with torch.no_grad():
        return LossParts(
            loss=loss,
            supervised_tokens=int((drawn["labels"] != IGNORE_INDEX).sum().item()),
            extras={
                "reward_margin": float((chosen_reward - rejected_reward).mean()),
                # Ties count as half. At step zero a fresh adapter makes both
                # rewards exactly zero, and a strict comparison would report 0%
                # accuracy for a model that is simply undecided.
                "reward_accuracy": float(
                    (
                        (chosen_reward > rejected_reward).float()
                        + 0.5 * (chosen_reward == rejected_reward).float()
                    ).mean()
                ),
                "reward_chosen": float(chosen_reward.mean()),
                "reward_rejected": float(rejected_reward.mean()),
            },
        )
