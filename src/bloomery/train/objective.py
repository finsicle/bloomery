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
