# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sampling across a weighted blend, and watching for forgetting.

Two jobs here.

**Sampling.** Draw each training batch across several datasets in proportion to
their weights, so replay data is interleaved with new data rather than appended
after it. Interleaving is the whole point: training on A then B forgets A,
training on a blend of A and B does not.

**Detection.** Evaluate every component's held-out split *separately*. A single
aggregate validation loss can fall while the model quietly gets worse at the
oldest corpus in the blend — the aggregate is dominated by whichever component
has the largest weight. Per-component loss makes forgetting visible while there
is still time to raise the replay share, which is the actionable part and the
reason this module exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from bloomery.data.shards import DatasetInfo
from bloomery.mixture import Mixture
from bloomery.train.device import DeviceChoice

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

    from bloomery.train.loop import BatchSampler, PreferenceSampler

# A component has to get worse by more than this before it is called a
# regression. Validation loss on a small held-out split is noisy; flagging every
# upward tick would make the warning meaningless.
REGRESSION_TOLERANCE = 0.02


@dataclass(frozen=True, slots=True)
class ResolvedMixture:
    """A mixture with every component loaded and verified compatible."""

    mixture: Mixture
    datasets: dict[str, DatasetInfo]
    tokenizer: Any
    eos_token_id: int
    # Which component's tokenizer is in use. All components share it by
    # construction; recorded so the run is reproducible.
    tokenizer_from: str

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)


def resolve(mixture: Mixture) -> ResolvedMixture:
    """Load every component of a mixture, refusing incompatible token spaces.

    This is the guard that matters most in this module. Each dataset is packed
    with its own tokenizer, so token id 4,211 in one corpus and in another are
    unrelated symbols unless the tokenizers are identical. Blending them would
    not error — it would train, converge to nothing useful, and give no clue why.

    So components must agree on both vocabulary size and the tokenizer file
    itself, compared by hash. Mismatches are refused with the actual numbers,
    because the fix (re-run ``prepare`` for one corpus with the other's
    tokenizer) is not guessable from a generic error.
    """
    import hashlib

    from bloomery import paths
    from bloomery.data import eot_id, load_dataset, load_tokenizer
    from bloomery.mixture import MixtureError

    datasets: dict[str, DatasetInfo] = {}
    fingerprints: dict[str, tuple[int, str]] = {}

    for name in mixture.datasets:
        tokens_path = paths.tokens_dir(name)
        tokenizer_path = paths.tokenizer_dir(name)
        if not tokens_path.is_dir() or not tokenizer_path.is_dir():
            raise MixtureError(
                f"mixture {mixture.name!r} references dataset {name!r}, which has not "
                f"been prepared. Run: bloomery prepare --name {name} --source ..."
            )
        info = load_dataset(tokens_path)
        datasets[name] = info

        tokenizer_file = tokenizer_path / "tokenizer.json"
        digest = (
            hashlib.sha256(tokenizer_file.read_bytes()).hexdigest()[:16]
            if tokenizer_file.is_file()
            else "missing"
        )
        fingerprints[name] = (info.vocab_size, digest)

    distinct = set(fingerprints.values())
    if len(distinct) > 1:
        detail = "\n".join(
            f"  {name}: vocab {vocab}, tokenizer {digest}"
            for name, (vocab, digest) in fingerprints.items()
        )
        raise MixtureError(
            f"mixture {mixture.name!r} blends datasets with different tokenizers, so "
            f"their token ids mean different things:\n{detail}\n"
            "Re-prepare the components against one tokenizer before blending them."
        )

    # Safe to take the tokenizer from any component now that they are identical.
    source = mixture.datasets[0]
    tokenizer = load_tokenizer(paths.tokenizer_dir(source))
    return ResolvedMixture(
        mixture=mixture,
        datasets=datasets,
        tokenizer=tokenizer,
        eos_token_id=eot_id(tokenizer),
        tokenizer_from=source,
    )


def check_tokenizer_matches(resolved: ResolvedMixture, model_tokenizer: Any, source: str) -> None:
    """Refuse a corpus that the model cannot read.

    The same failure the mixture guard above exists to prevent, one level out. A
    model's embedding table is indexed by *its* tokenizer's ids; hand it a corpus
    packed with another and token 4,211 addresses an unrelated symbol. Nothing
    raises. The run trains, the loss falls to somewhere unremarkable, and the
    model comes out worse than it started with no indication why.

    Compared on what the tokenizers do rather than on the files they came from,
    because one side was written by this project and the other was not — see
    :func:`bloomery.data.fingerprint`.
    """
    from bloomery.data import fingerprint, id_space
    from bloomery.mixture import MixtureError

    if fingerprint(resolved.tokenizer) == fingerprint(model_tokenizer):
        return

    corpus = ", ".join(resolved.mixture.datasets)
    raise MixtureError(
        f"the corpus was packed with a different tokenizer than {source} uses, so its "
        f"token ids mean different things to that model:\n"
        f"  corpus  {corpus}: {id_space(resolved.tokenizer):,} ids, "
        f"tokenizer {fingerprint(resolved.tokenizer)}\n"
        f"  model   {source}: {id_space(model_tokenizer):,} ids, "
        f"tokenizer {fingerprint(model_tokenizer)}\n"
        # Every component, not just the first. A replay blend is the arrangement
        # this is most often used with, and naming one of its datasets sends the
        # user round the same refusal again for the next.
        "Re-pack every component against the model's own tokenizer:\n"
        + "\n".join(
            f"  bloomery prepare --name {name} --source ... --tokenizer {source}"
            for name in resolved.mixture.datasets
        )
    )


class MixtureSampler:
    """Draws batches across a blend in proportion to component weights.

    Component counts per batch come from a multinomial draw rather than a fixed
    split, so the long-run proportions match the weights exactly even when the
    batch is too small to divide cleanly. A batch of 8 across a 60/15/25 blend
    cannot be split exactly; over a thousand batches the multinomial gets there.
    """

    def __init__(
        self,
        mixture: Mixture,
        datasets: dict[str, DatasetInfo],
        split: str,
        *,
        seq: int,
        seed: int,
    ) -> None:
        import numpy as np

        from bloomery.train.loop import sampler_for

        self.mixture = mixture
        self.split = split
        self._rng = np.random.default_rng(seed)
        # Uniform across the blend, because resolve() refuses a mixed one. The
        # two batch shapes have different keys and different meanings for a row,
        # so there is no sensible way to concatenate one with the other.
        self.preference = any(
            info.preference for name, info in datasets.items() if name in mixture.datasets
        )

        weights = mixture.weights()
        self._names: list[str] = []
        self._samplers: list[BatchSampler | PreferenceSampler] = []
        self._probs: list[float] = []
        self.skipped: dict[str, str] = {}

        for index, name in enumerate(mixture.datasets):
            info = datasets.get(name)
            if info is None:
                raise KeyError(f"mixture {mixture.name!r} references unknown dataset {name!r}")
            try:
                # Offset the seed per component so two components do not draw
                # the same offsets from differently sized arrays.
                sampler = sampler_for(info, split, seq=seq, seed=seed + index * 7919)
            except (ValueError, FileNotFoundError) as exc:
                self.skipped[name] = str(exc)
                continue
            self._names.append(name)
            self._samplers.append(sampler)
            self._probs.append(weights[name])

        if not self._samplers:
            detail = "; ".join(f"{k}: {v}" for k, v in self.skipped.items())
            raise ValueError(f"no usable {split!r} data in mixture {mixture.name!r}. {detail}")

        # Renormalise over the components that survived, so dropping an unusable
        # component reweights the rest instead of silently shrinking the batch.
        total = sum(self._probs)
        self._probs = [p / total for p in self._probs]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._names)

    @property
    def effective_weights(self) -> dict[str, float]:
        """Weights actually in use, after any unusable component was dropped."""
        return dict(zip(self._names, self._probs, strict=True))

    @property
    def tokens(self) -> int:
        return sum(s.tokens for s in self._samplers)

    def tokens_by_component(self) -> dict[str, int]:
        return {name: s.tokens for name, s in zip(self._names, self._samplers, strict=True)}

    def draw_counts(self, size: int) -> dict[str, int]:
        """How many sequences of the next batch come from each component.

        Public because it is the only way to confirm the blend is being honoured:
        a batch tensor is a concatenation with no record of where each row came
        from, so without this the sampling proportions are unobservable.
        """
        if len(self._samplers) == 1:
            return {self._names[0]: size}
        counts = self._rng.multinomial(size, self._probs)
        return dict(zip(self._names, (int(c) for c in counts), strict=True))

    def batch(self, size: int, device: torch.device) -> dict[str, torch.Tensor]:
        """One training batch, drawn across components by weight.

        Concatenated per field, so a blend can mix a conversation corpus with a
        plain one: the conversation component's labels carry the completion mask
        and the plain component's are simply its inputs. Every component was
        built with the same sequence length, so the rows line up.
        """
        import torch

        if len(self._samplers) == 1:
            return self._samplers[0].batch(size, device)

        counts = self.draw_counts(size)
        by_name = dict(zip(self._names, self._samplers, strict=True))
        chunks = [by_name[name].batch(count, device) for name, count in counts.items() if count > 0]

        if self.preference:
            # A preference batch is two stacked halves, and row i pairs with row
            # i + half. Concatenating whole chunks would put one component's
            # chosen rows opposite another's rejected ones — every comparison
            # between two unrelated answers, and nothing that looks wrong.
            return {
                key: torch.cat(
                    [chunk[key][: len(chunk[key]) // 2] for chunk in chunks]
                    + [chunk[key][len(chunk[key]) // 2 :] for chunk in chunks],
                    dim=0,
                )
                for key in chunks[0]
            }
        return {key: torch.cat([chunk[key] for chunk in chunks], dim=0) for key in chunks[0]}

    def component_samplers(self) -> dict[str, BatchSampler | PreferenceSampler]:
        """Per-component samplers, for evaluating each split on its own."""
        return dict(zip(self._names, self._samplers, strict=True))


@dataclass
class ForgettingTracker:
    """Watches each component's validation loss for regression.

    Tracks the best loss seen per component and reports anything that has drifted
    meaningfully above its own best. That framing matters: a component can be
    regressing while the aggregate improves, and the aggregate is what a naive
    training loop reports.
    """

    tolerance: float = REGRESSION_TOLERANCE
    best: dict[str, float] = field(default_factory=dict)
    first: dict[str, float] = field(default_factory=dict)

    def update(self, losses: dict[str, float]) -> dict[str, float]:
        """Record an evaluation. Returns the regression delta per affected component.

        A component appears in the result only when it is worse than its own best
        by more than the tolerance. The value is the absolute increase.
        """
        import math

        regressed: dict[str, float] = {}
        for name, loss in losses.items():
            if not math.isfinite(loss):
                continue
            self.first.setdefault(name, loss)
            previous_best = self.best.get(name)
            if previous_best is None:
                self.best[name] = loss
                continue
            if loss > previous_best * (1 + self.tolerance):
                regressed[name] = loss - previous_best
            else:
                self.best[name] = min(previous_best, loss)
        return regressed

    def improvement(self) -> dict[str, float]:
        """How much each component improved from its first measurement to its best.

        Never negative: `best` only ever moves downward, so a component that
        never beat its opening measurement reports exactly zero. Zero is
        therefore the signal that a component is being crowded out of the blend.
        """
        return {
            name: self.first[name] - self.best[name] for name in self.best if name in self.first
        }


def evaluate_components(
    model: Any,
    samplers: dict[str, BatchSampler],
    choice: DeviceChoice,
    *,
    batch: int,
    batches: int,
) -> dict[str, float]:
    """Mean validation loss for each component, measured independently."""
    from bloomery.train.loop import evaluate

    return {
        name: evaluate(model, sampler, choice, batch=batch, batches=batches)
        for name, sampler in samplers.items()
    }


def weighted_mean(losses: dict[str, float], weights: dict[str, float]) -> float:
    """Aggregate per-component losses back into one number, by sampling weight.

    Reported alongside the per-component figures, never instead of them.
    """
    import math

    total = 0.0
    used = 0.0
    for name, loss in losses.items():
        weight = weights.get(name, 0.0)
        if weight > 0 and math.isfinite(loss):
            total += loss * weight
            used += weight
    return total / used if used else float("nan")
