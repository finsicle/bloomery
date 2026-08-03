# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Scoring a checkpoint that already exists, against a dataset that already exists.

Training reports a validation loss as it goes, which answers "is this run still
improving". It does not answer "is this checkpoint better than that one", because
two runs report losses over their own splits and those numbers are not
comparable. This scores any checkpoint against any dataset, so they are.

What gets reported depends on the dataset's format, the same way the training
objective does — a corpus of text can only be scored by how well the model
predicts it, and a corpus of preference pairs can be scored by whether the model
prefers the better answer, which is a question about behaviour rather than about
fit.

Deliberately no standard benchmark suites. HellaSwag, ARC and MMLU are scored at
chance by models of the size this trains, so importing them would produce numbers
that look like rigour and carry no signal — the same trade `bench` refuses when
it runs real training steps instead of reading a peak-FLOPS table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from bloomery.train.device import DeviceChoice


class EvalError(RuntimeError):
    """A checkpoint could not be scored against this dataset."""


@dataclass(frozen=True, slots=True)
class EvalReport:
    checkpoint: str
    dataset: str
    format: str
    split: str
    examples: int
    device: str

    # Text corpora. Mean loss over the sampled batches, and its exponential —
    # which is a perplexity only because the loss is a log-likelihood.
    loss: float | None = None
    perplexity: float | None = None

    # Preference corpora. Both are the share of pairs where the model scores the
    # chosen answer above the rejected one; they differ in how the two are
    # compared. See :func:`score_preferences` for why one number is not enough.
    accuracy: float | None = None
    accuracy_per_token: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "checkpoint": self.checkpoint,
            "dataset": self.dataset,
            "format": self.format,
            "split": self.split,
            "examples": self.examples,
            "device": self.device,
        }
        for name in ("loss", "perplexity", "accuracy", "accuracy_per_token"):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        return payload


def run(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    *,
    name: str,
    checkpoint: str,
    choice: DeviceChoice,
    split: str = "val",
    batch: int = 8,
    seq: int = 512,
    batches: int = 20,
    seed: int = 1337,
) -> EvalReport:
    """Score a loaded model against a prepared dataset.

    Takes the model and dataset already loaded rather than paths, so the command
    layer keeps the reporting and this keeps the arithmetic — the same split
    ``bench`` uses.
    """
    from bloomery.data.shards import FORMAT_DPO
    from bloomery.train.loop import evaluate, sampler_for

    del tokenizer  # loaded by the caller for its own checks; not needed here
    model.to(choice.device)

    try:
        sampler = sampler_for(dataset, split, seq=seq, seed=seed)
    except (ValueError, FileNotFoundError) as exc:
        raise EvalError(f"cannot read the {split!r} split of {name!r}: {exc}") from exc

    common = {
        "checkpoint": checkpoint,
        "dataset": name,
        "format": dataset.format,
        "split": split,
        "device": choice.label(),
    }

    if dataset.format == FORMAT_DPO:
        by_sum, by_token, scored = score_preferences(
            model, sampler, choice, batch=batch, batches=batches
        )
        return EvalReport(examples=scored, accuracy=by_sum, accuracy_per_token=by_token, **common)

    result = evaluate(model, sampler, choice, batch=batch, batches=batches)
    return EvalReport(
        examples=batch * batches,
        loss=result.loss,
        # exp() of a loss is a perplexity only because this loss is a
        # log-likelihood — see _perplexity in the training loop, which refuses
        # to publish one for an objective where it would not be.
        perplexity=math.exp(min(result.loss, 20)) if math.isfinite(result.loss) else None,
        **common,
    )


def score_preferences(
    model: Any,
    sampler: Any,
    choice: DeviceChoice,
    *,
    batch: int,
    batches: int,
) -> tuple[float, float, int]:
    """How often this model prefers the answer a person preferred.

    Needs no reference model, unlike the reward accuracy a DPO run reports. That
    one measures movement away from where training started; this one measures the
    model as it stands, which is what a comparison between two checkpoints needs.

    Two numbers, because summed log-probability carries a length bias: a longer
    answer accumulates more negative terms and scores lower for being longer.
    DPO optimises the sum, so the sum is what training moved — but a model can
    raise it by learning that shorter is better, which is not a preference for
    anything. The per-token mean removes the bias. When the two disagree, the
    gap is the finding.

    Returns ``(by_sum, by_token, pairs_scored)``.
    """
    import torch

    from bloomery.train.loop import autocast_for
    from bloomery.train.objective import IGNORE_INDEX, sequence_logprobs

    model.eval()
    wins_by_sum = 0.0
    wins_by_token = 0.0
    scored = 0

    with torch.no_grad():
        for _ in range(batches):
            drawn = sampler.batch(batch, choice.device)
            half = drawn["input_ids"].shape[0] // 2
            with autocast_for(choice):
                totals = sequence_logprobs(model, drawn)

            lengths = (drawn["labels"][:, 1:] != IGNORE_INDEX).sum(dim=-1).clamp(min=1)
            per_token = totals / lengths

            for values, bucket in ((totals, "sum"), (per_token, "token")):
                chosen, rejected = values[:half], values[half:]
                # Ties count as half, for the reason they do during training: two
                # answers the model cannot separate are undecided, not wrong.
                won = float(
                    ((chosen > rejected).float() + 0.5 * (chosen == rejected).float()).sum()
                )
                if bucket == "sum":
                    wins_by_sum += won
                else:
                    wins_by_token += won
            scored += half

    model.train()
    if not scored:
        raise EvalError("no preference pairs were scored; the split may be empty")
    return wins_by_sum / scored, wins_by_token / scored, scored
