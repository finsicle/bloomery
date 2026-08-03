# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The pretraining loop.

Written out explicitly rather than delegating to ``Trainer``. The loop is the
part a person learning to train a model most needs to be able to read, and it is
the part bloomery most needs to control — the metrics protocol, the checkpoint
layout and the resume semantics are all decisions the UI depends on.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bloomery.arch import suggested_lr, to_llama_config
from bloomery.capability import ModelSpec
from bloomery.data.shards import DatasetInfo, open_mask, open_preference_split, open_split
from bloomery.mixture import Mixture
from bloomery.train import checkpoint
from bloomery.train import objective as objective_mod
from bloomery.train.device import DeviceChoice
from bloomery.train.metrics import MetricsWriter, Throughput

# mixing imports BatchSampler and evaluate from this module, but does so inside
# its functions, so importing it here at module level does not cycle.
from bloomery.train.mixing import (  # noqa: E402 - ordering explained above
    ForgettingTracker,
    MixtureSampler,
    evaluate_components,
    weighted_mean,
)
from bloomery.train.objective import IGNORE_INDEX, Batch, LossParts, Objective

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

# Re-exported from bloomery.train.objective, which owns them now that what a
# batch contains depends on what is being optimised. Kept importable from here
# because the samplers below build them and every caller already looks here.
__all__ = ["IGNORE_INDEX", "Batch", "LossParts", "Objective"]


@dataclass(slots=True)
class TrainConfig:
    steps: int = 500
    batch: int = 8
    seq: int = 256
    grad_accum: int = 1
    lr: float | None = None  # None means derive from model width
    min_lr_fraction: float = 0.1
    warmup_fraction: float = 0.05
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_every: int = 100
    eval_batches: int = 20
    save_every: int = 0  # 0 means only at the end
    log_every: int = 10
    seed: int = 1337
    gradient_checkpointing: bool = False
    # How hard preference training is held to the reference model. Named
    # dpo_beta and not beta because beta1 and beta2 sit right above it and mean
    # something else entirely. Ignored unless the dataset is preference pairs.
    dpo_beta: float = 0.1

    @property
    def tokens_per_step(self) -> int:
        return self.batch * self.seq * self.grad_accum

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TrainResult:
    run_dir: Path
    checkpoint: Path
    steps: int
    final_loss: float
    best_val_loss: float | None
    tokens_seen: int
    tokens_per_second: float
    # Tokens the loss was actually computed over. Equal to tokens_seen for a
    # plain corpus; a fraction of it once a completion mask is in play.
    trained_tokens: int = 0
    # Microbatches dropped for a non-finite loss, whether an unplaceable
    # window or a diverging run.
    skipped_microbatches: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    # Last measured validation loss per blend component, and any component that
    # ended worse than its own best. Empty for a single-dataset run with no
    # validation split.
    per_component_loss: dict[str, float] = field(default_factory=dict)
    regressed: dict[str, float] = field(default_factory=dict)


class ModelLoadError(RuntimeError):
    """A model could not be opened, or adapters could not be attached."""


def build_model(spec: ModelSpec, *, eos_token_id: int, seed: int) -> Any:
    """Create a randomly initialised Llama model.

    This is the "from nothing" step: weights come from the initialiser, not from
    anybody else's checkpoint.
    """
    import torch
    from transformers import LlamaForCausalLM

    torch.manual_seed(seed)
    config = to_llama_config(spec, eos_token_id=eos_token_id)
    return LlamaForCausalLM(config)


@dataclass(frozen=True, slots=True)
class LoraSettings:
    """How to size the adapters trained against a frozen base."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05

    def to_dict(self) -> dict[str, Any]:
        return {"r": self.r, "alpha": self.alpha, "dropout": self.dropout}


ADAPTER_CONFIG = "adapter_config.json"


def is_adapter_dir(path: Path) -> bool:
    """Whether this checkpoint holds adapters rather than a whole model."""
    return (path / ADAPTER_CONFIG).is_file()


def load_model(source: str | Path) -> Any:
    """Load a causal language model from a checkpoint or a repository id.

    ``AutoModelForCausalLM`` rather than a named class: this is asked to open
    models this project did not build, and Qwen, Mistral and Phi are not Llama
    classes. Naming one would work on our own checkpoints and fail on everything
    the feature exists for.

    An adapter checkpoint is not a model — it is a small diff against one — so it
    is loaded by first loading the base it names and then applying the adapters
    on top. Doing that here rather than at each call site means inference,
    resuming, and adapting an already-adapted model all handle it the same way.
    """
    from transformers import AutoModelForCausalLM

    path = Path(source)
    if path.is_dir() and is_adapter_dir(path):
        return _load_adapted(path)

    try:
        return AutoModelForCausalLM.from_pretrained(str(source))
    except Exception as exc:  # noqa: BLE001 - transformers raises many types here
        raise ModelLoadError(
            f"could not load a model from {source!r}: {exc}\n"
            "Give a local checkpoint directory or a Hugging Face repository id."
        ) from exc


def _load_adapted(path: Path) -> Any:
    """Load the base a set of adapters was trained against, then apply them."""
    import json

    from transformers import AutoModelForCausalLM

    try:
        from peft import PeftModel
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise ModelLoadError(
            f"{path} holds LoRA adapters, which need the adapt extra: "
            'install it with  uv pip install -e ".[adapt]"'
        ) from exc

    config = json.loads((path / ADAPTER_CONFIG).read_text(encoding="utf-8"))
    base = config.get("base_model_name_or_path")
    if not base:
        raise ModelLoadError(
            f"{path} does not record which model its adapters were trained against, "
            "so there is nothing to apply them to."
        )

    try:
        model = AutoModelForCausalLM.from_pretrained(str(base))
    except Exception as exc:  # noqa: BLE001 - transformers raises many types here
        raise ModelLoadError(
            f"{path} holds adapters for {base!r}, which could not be loaded: {exc}\n"
            "Adapters are a diff against a base model; the base has to be available too."
        ) from exc

    try:
        return PeftModel.from_pretrained(model, str(path), is_trainable=True)
    except Exception as exc:  # noqa: BLE001 - peft raises many types here
        raise ModelLoadError(
            f"the adapters in {path} could not be applied to {base!r}: {exc}\n"
            "They may have been trained against a different model, or the adapter "
            "file may be incomplete."
        ) from exc


def merge_adapters(model: Any) -> Any:
    """Fold any LoRA adapters into the weights they modify.

    Export needs one set of weights, not a base plus a diff: GGUF has no notion
    of an adapter, and a runtime reading the file would get the untouched base.

    A model with no adapters is returned unchanged, so a caller does not have to
    know which shape it was handed.
    """
    merge = getattr(model, "merge_and_unload", None)
    if merge is None:
        return model
    # eval() first: dropout is active on a freshly loaded adapter, and merging
    # while it is would fold a randomly masked version of the update.
    model.eval()
    return merge()


def attach_adapter(model: Any, settings: LoraSettings) -> Any:
    """Freeze the model and train low-rank adapters against it instead.

    The optimizer built later skips parameters that do not require a gradient, so
    freezing here is all that is needed for the rest of the loop to follow.
    """
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise ModelLoadError(
            'training adapters needs the adapt extra: install it with  uv pip install -e ".[adapt]"'
        ) from exc

    config = LoraConfig(
        r=settings.r,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def trainable_fraction(model: Any) -> tuple[int, int]:
    """(trainable, total) parameters, for reporting what is actually being trained."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


class BatchSampler:
    """Samples random fixed-length windows from a packed token array.

    The memmap is recreated on every draw. That looks wasteful and is not: numpy
    memmaps accumulate memory when sampled repeatedly from a long-lived handle,
    and reopening costs far less than the leak does over a long run.
    """

    def __init__(self, info: DatasetInfo, split: str, *, seq: int, seed: int) -> None:
        import numpy as np

        self._info = info
        self._split = split
        self._seq = seq
        self._rng = np.random.default_rng(seed)

        length = len(open_split(info, split))
        if length <= seq + 1:
            raise ValueError(
                f"split {split!r} holds {length} tokens, which is not enough for a "
                f"sequence length of {seq}. Use a shorter --seq or a larger corpus."
            )
        self._length = length
        self._supervised_blocks = self._index_supervision(info, split, seq)

    @staticmethod
    def _index_supervision(info: DatasetInfo, split: str, seq: int) -> Any | None:
        """Where in a masked corpus a window is guaranteed to find something.

        Which windows carry a loss cannot be answered by redrawing and hoping:
        with prompts several times the sequence length, most draws land in one,
        and a bounded number of retries leaves a real chance of returning a
        window with nothing supervised in it at all.

        So the offsets that do work are recorded once. One entry per window-sized
        block rather than per token — for a corpus of a billion tokens at a
        sequence length of 512, two million entries instead of a billion — and
        the mask is walked in chunks so indexing it never holds the whole thing.
        """
        import numpy as np

        mask = open_mask(info, split)
        if mask is None:
            return None

        blocks = len(mask) // seq
        if blocks == 0:
            return None

        # Chunked so a large corpus is not materialised to build its own index.
        per_chunk = max(1, (1 << 22) // seq)
        flags: list[Any] = []
        for start in range(0, blocks, per_chunk):
            stop = min(start + per_chunk, blocks)
            window = np.asarray(mask[start * seq : stop * seq]).reshape(stop - start, seq)
            flags.append(window.any(axis=1))
        usable = np.flatnonzero(np.concatenate(flags)) * seq

        if usable.size == 0:
            raise ValueError(
                f"split {split!r} has no window of {seq} tokens containing anything to "
                "learn from — every response is shorter than the gap between them. "
                "Use a shorter --seq, or check that the corpus was packed against the "
                "right chat template."
            )
        return usable

    @property
    def tokens(self) -> int:
        return self._length

    def batch(self, size: int, device: torch.device) -> Batch:
        """One batch of windows, with the labels to score them against.

        For a plain corpus the labels *are* the inputs, which is what next-token
        prediction means and what this returned before there was anything else.
        For a conversation corpus the prompt positions carry ``IGNORE_INDEX``, so
        the loss counts only what the model was meant to produce.

        The labels are deliberately not shifted. The model does that internally,
        so a shifted label tensor here would move the mask a position away from
        the token it belongs to.

        A window holding no supervised token at all is redrawn. One prompt longer
        than the sequence length is enough to produce one — and cross-entropy
        over nothing but ignored positions is a mean of an empty set, which comes
        back NaN. That NaN is not caught by the gradient (it arrives as zeros) but
        it does reach the reported loss, run.jsonl and the final result, where it
        looks like a diverged model rather than an unlucky draw.
        """
        import numpy as np
        import torch

        data = open_split(self._info, self._split)
        mask = open_mask(self._info, self._split)

        starts = self._rng.integers(0, self._length - self._seq - 1, size=size)
        if mask is not None:
            starts = self._with_supervision(mask, starts)

        window = np.stack([data[i : i + self._seq] for i in starts]).astype(np.int64)
        inputs = torch.from_numpy(window).to(device, non_blocking=True)

        if mask is None:
            return {"input_ids": inputs, "labels": inputs}

        keep = np.stack([mask[i : i + self._seq] for i in starts]).astype(bool)
        labels = np.where(keep, window, IGNORE_INDEX)
        return {
            "input_ids": inputs,
            "labels": torch.from_numpy(labels).to(device, non_blocking=True),
        }

    def _with_supervision(self, mask: Any, starts: Any) -> Any:
        """Replace any window that would carry no loss at all.

        The uniform draw stays the normal path, so ordinary sampling is unchanged
        and windows are not biased toward responses. Only a barren row is moved,
        and it is moved onto a recorded offset rather than redrawn — a redraw
        might land in another prompt, and with prompts several times the sequence
        length it usually would.
        """
        import numpy as np

        if self._supervised_blocks is None:  # pragma: no cover - masked implies an index
            return starts

        barren = np.array([not mask[i : i + self._seq].any() for i in starts], dtype=bool)
        if not barren.any():
            return starts

        replacements = self._rng.choice(self._supervised_blocks, size=len(starts))
        return np.where(barren, replacements, starts)


class PreferenceSampler:
    """Draws whole preference pairs and lays them out as one batch.

    A batch of ``size`` pairs comes back as ``2 * size`` rows: the chosen answers
    first, then the rejected ones in the same order. Row ``i`` and row
    ``i + size`` are the two halves of one comparison, and the loss relies on
    that pairing, so it is stated here rather than left to be inferred.

    This is the first batch in the project carrying an ``attention_mask``. The
    packed and SFT paths never need one because a window is dense by
    construction; a preference pair has whatever length it has, so short ones are
    padded and the padding has to be excluded from attention as well as from the
    loss.
    """

    def __init__(self, info: DatasetInfo, split: str, *, seq: int, seed: int) -> None:
        import numpy as np

        self._info = info
        self._split = split
        self._seq = seq
        self._rng = np.random.default_rng(seed)

        _, index = open_preference_split(info, split)
        prompt, chosen, rejected = (index[:, column, 1] for column in range(3))
        # An answer alone longer than the window leaves no room for the prompt it
        # answers, and truncating into the answer would change what is being
        # compared. Such a pair is dropped rather than mangled.
        self._usable = np.flatnonzero(np.maximum(chosen, rejected) < seq)
        self._dropped = int(len(index) - len(self._usable))
        if self._usable.size == 0:
            raise ValueError(
                f"split {split!r} has no preference pair whose answers fit in {seq} "
                "tokens. Use a longer --seq, or check that the corpus was packed "
                "against the right chat template."
            )
        self._tokens = int(prompt.sum() + chosen.sum() + rejected.sum())
        # How much prompt has to go to make room, per usable pair. Driven by the
        # longer of the two answers and applied to both, or the two rows would
        # condition on different prompts and compare nothing.
        room = seq - np.maximum(chosen, rejected)[self._usable]
        self._keep = np.minimum(prompt[self._usable], room)
        self._truncated = int((self._keep < prompt[self._usable]).sum())

    @property
    def tokens(self) -> int:
        return self._tokens

    @property
    def dropped(self) -> int:
        """Pairs whose answers do not fit the window at all."""
        return self._dropped

    @property
    def truncated(self) -> int:
        """Pairs whose prompt is cut to make room, identically on both sides."""
        return self._truncated

    def batch(self, size: int, device: torch.device) -> Batch:
        """``2 * size`` rows: the chosen answers, then the rejected ones.

        Right-padded, which is the opposite of what generation wants. Left
        padding would need the position ids corrected to match, and that is a
        second thing to get wrong for no gain here — nothing generates from these
        rows, they only ever produce log-probabilities.

        The prompt is truncated from the left when it does not fit, keeping the
        end that the answer actually follows from. Both rows lose the same
        number of tokens, so what they are conditioned on stays identical.
        """
        import numpy as np
        import torch

        streams, index = open_preference_split(self._info, self._split)
        picks = self._rng.choice(len(self._usable), size=size)

        ids = np.zeros((2 * size, self._seq), dtype=np.int64)
        labels = np.full((2 * size, self._seq), IGNORE_INDEX, dtype=np.int64)
        attention = np.zeros((2 * size, self._seq), dtype=np.int64)

        for row, pick in enumerate(picks):
            example = self._usable[pick]
            keep = int(self._keep[pick])
            start, length = index[example, 0]
            prompt = streams["prompt"][start + length - keep : start + length]

            for offset, part in ((0, "chosen"), (size, "rejected")):
                start, length = index[example, 1 if part == "chosen" else 2]
                answer = streams[part][start : start + length]
                span = keep + len(answer)
                ids[row + offset, :keep] = prompt
                ids[row + offset, keep:span] = answer
                # Only the answer is scored. The prompt is what both rows share,
                # so a loss over it would be identical on each side and cancel —
                # it would add noise to the comparison and nothing else.
                labels[row + offset, keep:span] = answer
                attention[row + offset, :span] = 1

        return {
            "input_ids": torch.from_numpy(ids).to(device, non_blocking=True),
            "labels": torch.from_numpy(labels).to(device, non_blocking=True),
            "attention_mask": torch.from_numpy(attention).to(device, non_blocking=True),
        }


def sampler_for(
    info: DatasetInfo, split: str, *, seq: int, seed: int
) -> BatchSampler | PreferenceSampler:
    """The sampler this dataset's format calls for.

    One factory rather than a branch at each construction site, so a caller that
    does not care about the format — which is all of them — does not have to
    learn about it.
    """
    if info.preference:
        return PreferenceSampler(info, split, seq=seq, seed=seed)
    return BatchSampler(info, split, seq=seq, seed=seed)


def lr_at(step: int, config: TrainConfig, peak: float) -> float:
    """Linear warmup into a cosine decay, floored at a fraction of peak.

    Decaying to zero tends to waste the tail of a run; a floor keeps the last
    steps doing useful work.
    """
    warmup = max(1, int(config.steps * config.warmup_fraction))
    floor = peak * config.min_lr_fraction

    if step < warmup:
        return peak * (step + 1) / warmup
    if step >= config.steps:
        return floor

    progress = (step - warmup) / max(1, config.steps - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return floor + (peak - floor) * cosine


def autocast_for(choice: DeviceChoice) -> Any:
    import torch

    if not choice.autocast:
        return nullcontext()
    return torch.autocast(device_type=choice.type, dtype=choice.dtype)


@dataclass(frozen=True, slots=True)
class EvalResult:
    """A held-out score, plus whatever else the objective measured.

    A bare float would be enough for language modelling. Preference training has
    a second number worth seeing — how often the preferred answer actually scores
    higher — and it comes from the same forward passes, so computing it in a
    second evaluation pass would double the most expensive non-training work in
    a run.
    """

    loss: float
    extras: dict[str, float] = field(default_factory=dict)
    # Batches that actually contributed. Not the number asked for: a batch whose
    # loss came back non-finite is dropped, so a caller reporting how much was
    # measured has to be told rather than assume.
    counted: int = 0


def evaluate(
    model: Any,
    sampler: BatchSampler | PreferenceSampler,
    choice: DeviceChoice,
    *,
    batch: int,
    batches: int,
    objective: Objective = objective_mod.causal_loss,
) -> EvalResult:
    """Mean loss over a fixed number of held-out batches."""
    import torch

    was_training = model.training
    model.eval()
    total = 0.0
    counted = 0
    extras: dict[str, float] = {}
    try:
        with torch.no_grad():
            for _ in range(batches):
                drawn = sampler.batch(batch, choice.device)
                with autocast_for(choice):
                    parts = objective(model, drawn)
                if torch.isfinite(parts.loss):
                    total += parts.loss.item()
                    counted += 1
                    for name, value in parts.extras.items():
                        extras[name] = extras.get(name, 0.0) + value
    finally:
        # Restored, not assumed. This used to end with a bare model.train(),
        # which put a model that arrived in eval mode into training mode — and
        # left one that arrived in training mode stuck in eval if anything here
        # raised. Called mid-run the first is invisible because the loop keeps
        # the model training anyway; called from `eval` on a freshly loaded
        # checkpoint it silently changes what the caller handed over.
        if was_training:
            model.train()
    if not counted:
        return EvalResult(loss=float("nan"))
    return EvalResult(
        loss=total / counted,
        extras={name: value / counted for name, value in extras.items()},
        counted=counted,
    )


def _perplexity(val_loss: float, objective: Objective) -> float | None:
    """``exp(loss)``, but only where the loss is a log-likelihood.

    A preference loss is not one. Publishing ``exp`` of it gives a plausible
    number — a fresh DPO run would report a perplexity of 2.0 — into a file this
    project calls a contract, and a plausible wrong number is worse than an
    absent one.
    """
    if objective is not objective_mod.causal_loss:
        return None
    return round(math.exp(min(val_loss, 20)), 2) if math.isfinite(val_loss) else None


def train(
    *,
    spec: ModelSpec,
    datasets: dict[str, DatasetInfo],
    mixture: Mixture,
    tokenizer: Any,
    run_dir: Path,
    config: TrainConfig,
    choice: DeviceChoice,
    eos_token_id: int,
    resume_from: Path | None = None,
    base_model: str | Path | None = None,
    adapter: LoraSettings | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> TrainResult:
    """Train a model from random initialisation, resume one, or continue someone else's.

    Always trains against a :class:`~bloomery.mixture.Mixture`. A plain single
    dataset is a one-component blend — see :func:`bloomery.mixture.single` — so
    there is one sampling path, one evaluation path and one metrics shape rather
    than two that drift apart.

    ``base_model`` continues training a model that already exists, from a local
    directory or a Hugging Face repository id. ``adapter`` trains LoRA adapters
    against it with the base frozen, instead of every weight.

    The three ways of getting a model differ only in how it is constructed.
    Sampling, the schedule, evaluation, forgetting detection, metrics and
    checkpointing are the same afterwards, deliberately: a second training path
    would be a second set of behaviours to keep in step.

    Emits every event to ``run.jsonl`` and, if given, to ``on_event`` for live
    display.
    """
    import time

    import torch

    run_dir.mkdir(parents=True, exist_ok=True)
    peak_lr = config.lr if config.lr is not None else suggested_lr(spec)

    if resume_from is not None:
        model = load_model(resume_from)
    elif base_model is not None:
        model = load_model(base_model)
    else:
        model = build_model(spec, eos_token_id=eos_token_id, seed=config.seed)

    if adapter is not None:
        model = attach_adapter(model, adapter)

    model.to(choice.device)
    model.train()

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # Checkpointing and the KV cache are mutually exclusive during training.
        model.config.use_cache = False

    # No weight decay on biases or normalisation weights: decaying a scale
    # parameter toward zero fights what it is there to do.
    decay: list[Any] = []
    no_decay: list[Any] = []
    for _name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if param.ndim < 2 else decay).append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=peak_lr,
        betas=(config.beta1, config.beta2),
        fused=choice.type == "cuda",
    )

    start_step = 0
    tokens_seen = 0
    # Distinct from tokens_seen, which counts the window. Under a completion
    # mask most of the window is prompt, and reporting that as tokens learned
    # from would overstate the run several-fold. run.jsonl is a published
    # contract, so this is a new number rather than a quiet redefinition.
    trained_tokens = 0
    # Microbatches dropped for a non-finite loss. Published because the two
    # causes look identical from inside the loop and very different from
    # outside: an occasional drop is an unlucky window, a persistent one is a
    # diverging run that would otherwise be invisible.
    skipped_microbatches = 0
    best_val: float | None = None
    resume_state: checkpoint.ResumeState | None = None
    if resume_from is not None:
        resume_state = checkpoint.load_resume_state(resume_from, optimizer)
        start_step = resume_state.step
        tokens_seen = resume_state.tokens_seen
        trained_tokens = resume_state.trained_tokens
        best_val = resume_state.best_val_loss

    # fp16 needs loss scaling to keep small gradients from flushing to zero.
    # bf16 has the exponent range to avoid this, so the scaler stays disabled.
    scaler = torch.amp.GradScaler(choice.type, enabled=choice.dtype == torch.float16)

    train_sampler = MixtureSampler(mixture, datasets, "train", seq=config.seq, seed=config.seed)

    # A missing or too-small validation split is not fatal: training still works,
    # you just lose the forgetting signal. Say so rather than aborting.
    val_sampler: MixtureSampler | None
    try:
        val_sampler = MixtureSampler(mixture, datasets, "val", seq=config.seq, seed=config.seed + 1)
    except (ValueError, FileNotFoundError, KeyError):
        val_sampler = None

    # Seeded from the checkpoint when resuming, so a component that was already
    # regressing before the interruption keeps its history instead of having its
    # first post-resume reading mistaken for a baseline.
    tracker = ForgettingTracker()
    if resume_state is not None:
        tracker.best.update(resume_state.component_best)
        tracker.first.update(resume_state.component_first)
    last_per_component: dict[str, float] = {}
    last_regressed: dict[str, float] = {}
    # Chosen from the data, not from a flag. A field on TrainConfig would be a
    # second source of truth for one fact — and it is written into every
    # checkpoint's run.json, so a run could record an objective its own dataset
    # contradicts. A mixed blend is refused when the mixture is resolved, so
    # any() and all() agree here.
    objective: Objective = objective_mod.causal_loss
    if any(info.preference for info in datasets.values()):
        # The reference the objective scores against is this model with its
        # adapters switched off, so there have to be adapters. The CLI refuses
        # --method full on preference data, but train() is reachable without it
        # — from a library caller, or from a test — and without this the run
        # builds the model, the optimizer, the samplers and the metrics writer
        # before dying on the first microbatch with an AttributeError naming a
        # peft method, which says nothing about what was actually wrong.
        if not hasattr(model, "disable_adapter"):
            raise ValueError(
                "preference training scores a model against its own untuned self, "
                "which here means the same weights with the LoRA adapters switched "
                "off — so it needs adapters, and this model has none.\n"
                "Pass a LoraSettings adapter, or use `bloomery adapt --method lora`."
            )
        objective = partial(objective_mod.preference_loss, beta=config.dpo_beta)
    history: list[dict[str, Any]] = []
    throughput = Throughput()
    final_loss = float("nan")
    params = sum(p.numel() for p in model.parameters())

    def record(event: dict[str, Any]) -> None:
        history.append(event)
        if on_event is not None:
            on_event(event)

    with MetricsWriter(run_dir / "run.jsonl") as metrics:
        record(
            metrics.emit(
                "start",
                model=spec.key,
                params=params,
                device=str(choice.device),
                dtype=str(choice.dtype).removeprefix("torch."),
                precision_reason=choice.reason,
                vocab=spec.vocab,
                train_tokens=train_sampler.tokens,
                val_tokens=val_sampler.tokens if val_sampler else 0,
                peak_lr=peak_lr,
                resumed_from=str(resume_from) if resume_from else None,
                mixture=mixture.name,
                mixture_version=mixture.version,
                mixture_summary=mixture.describe(),
                replay_share=round(mixture.replay_share(), 4),
                components=train_sampler.effective_weights,
                component_tokens=train_sampler.tokens_by_component(),
                # Named so a component silently dropped from the blend cannot be
                # mistaken for one that was never in it.
                components_skipped=train_sampler.skipped or None,
                config=config.to_dict(),
            )
        )

        for step in range(start_step, config.steps):
            lr = lr_at(step, config, peak_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr

            began = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            contributed = 0
            step_extras: dict[str, float] = {}

            for _ in range(config.grad_accum):
                drawn = train_sampler.batch(config.batch, choice.device)
                with autocast_for(choice):
                    parts = objective(model, drawn)
                    # Average over accumulation so the gradient matches what a
                    # single large batch would have produced.
                    scaled = parts.loss / config.grad_accum
                if not torch.isfinite(scaled):
                    # Usually a window the sampler could not place on anything
                    # supervised, where the score is a mean over an empty set.
                    # But a diverging run reaches here too — fp16 overflow, a
                    # learning rate far too high — and that used to be visible as
                    # a NaN in the reported loss. Skipping without counting would
                    # make a run that diverges every step look like a healthy one
                    # with a slightly smaller batch, so the count is published.
                    skipped_microbatches += 1
                    continue
                scaler.scale(scaled).backward()
                step_loss += scaled.item()
                contributed += 1
                # What the loss was actually computed over. With a conversation
                # corpus most of the window is prompt, so the window size stops
                # describing what the model learned from. Only the objective can
                # answer it, since only it knows which positions counted.
                trained_tokens += parts.supervised_tokens
                for name, value in parts.extras.items():
                    step_extras[name] = step_extras.get(name, 0.0) + value

            if contributed:
                if config.grad_clip > 0:
                    # Unscale first, or the clip threshold is applied to inflated
                    # fp16 gradients and does nothing.
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

                scaler.step(optimizer)
                scaler.update()
                final_loss = step_loss
            else:
                # Nothing was scored, so there is nothing to step on: the
                # gradients are the zeros from zero_grad, and stepping them
                # applies weight decay for no reason. Reporting a loss of 0.0
                # here — which is what an untouched accumulator holds — would be
                # worse than the NaN this guard replaced, because it reads as a
                # perfect model rather than a broken step.
                step_loss = float("nan")
                final_loss = step_loss

            tokens_seen += config.tokens_per_step
            rate = throughput.update(config.tokens_per_step, time.perf_counter() - began)

            if config.log_every and (step + 1) % config.log_every == 0:
                record(
                    metrics.emit(
                        "step",
                        step=step + 1,
                        loss=round(step_loss, 5),
                        lr=round(lr, 8),
                        tokens=tokens_seen,
                        trained_tokens=trained_tokens,
                        skipped_microbatches=skipped_microbatches,
                        tokens_per_second=round(rate, 1),
                        # Averaged over the microbatches that contributed, not
                        # over grad_accum: the non-finite guard above can drop
                        # some, and dividing by the wrong denominator would make
                        # these shrink on exactly the steps already going wrong.
                        **{
                            name: round(total / contributed, 5)
                            for name, total in step_extras.items()
                            if contributed
                        },
                    )
                )

            should_eval = (
                val_sampler is not None
                and config.eval_every
                and (step + 1) % config.eval_every == 0
            )
            if should_eval and val_sampler is not None:
                # Each component is measured on its own held-out split. The
                # aggregate is reported too, but it is dominated by whichever
                # component carries the most weight, so on its own it can fall
                # while an older corpus is being forgotten.
                scored = evaluate_components(
                    model,
                    val_sampler.component_samplers(),
                    choice,
                    batch=config.batch,
                    batches=config.eval_batches,
                    objective=objective,
                )
                per_component = {name: result.loss for name, result in scored.items()}
                regressed = tracker.update(per_component)
                last_per_component = per_component
                last_regressed = regressed
                val_loss = weighted_mean(per_component, val_sampler.effective_weights)
                val_extras: dict[str, float] = {}
                for result in scored.values():
                    for name, value in result.extras.items():
                        val_extras[name] = val_extras.get(name, 0.0) + value / len(scored)

                if math.isfinite(val_loss) and (best_val is None or val_loss < best_val):
                    best_val = val_loss

                record(
                    metrics.emit(
                        "eval",
                        step=step + 1,
                        val_loss=round(val_loss, 5),
                        best_val_loss=round(best_val, 5) if best_val is not None else None,
                        perplexity=_perplexity(val_loss, objective),
                        per_component={k: round(v, 5) for k, v in per_component.items()},
                        regressed={k: round(v, 5) for k, v in regressed.items()} or None,
                        **{f"val_{name}": round(v, 5) for name, v in val_extras.items()},
                    )
                )

            if config.save_every and (step + 1) % config.save_every == 0:
                path = checkpoint.save(
                    checkpoint.checkpoint_dir(run_dir, step + 1),
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    step=step + 1,
                    tokens_seen=tokens_seen,
                    trained_tokens=trained_tokens,
                    best_val_loss=best_val,
                    component_best=dict(tracker.best),
                    component_first=dict(tracker.first),
                    extra={
                        "spec": asdict(spec),
                        "config": config.to_dict(),
                        "mixture": mixture.to_dict(),
                    },
                )
                record(metrics.emit("checkpoint", step=step + 1, path=str(path)))

        final_path = checkpoint.save(
            checkpoint.checkpoint_dir(run_dir),
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            step=config.steps,
            tokens_seen=tokens_seen,
            trained_tokens=trained_tokens,
            best_val_loss=best_val,
            component_best=dict(tracker.best),
            component_first=dict(tracker.first),
            extra={
                "spec": asdict(spec),
                "config": config.to_dict(),
                "mixture": mixture.to_dict(),
            },
        )

        record(
            metrics.emit(
                "done",
                step=config.steps,
                final_loss=round(final_loss, 5),
                best_val_loss=round(best_val, 5) if best_val is not None else None,
                tokens=tokens_seen,
                # Distinct from `tokens`, which counts the window. Under a
                # completion mask most of that window is prompt, so publishing
                # only the window overstates the run several-fold — which is the
                # whole reason this counter exists.
                trained_tokens=trained_tokens,
                skipped_microbatches=skipped_microbatches,
                tokens_per_second=round(throughput.value or 0.0, 1),
                checkpoint=str(final_path),
                per_component={k: round(v, 5) for k, v in tracker.best.items()} or None,
                # Same shape as the `eval` event's field. run.jsonl is a
                # published contract, so one key must not carry a dict in one
                # event and a list in another.
                regressed={k: round(last_regressed[k], 5) for k in sorted(last_regressed)} or None,
                improvement={k: round(v, 5) for k, v in tracker.improvement().items()} or None,
            )
        )

    return TrainResult(
        run_dir=run_dir,
        checkpoint=final_path,
        steps=config.steps,
        final_loss=final_loss,
        best_val_loss=best_val,
        tokens_seen=tokens_seen,
        trained_tokens=trained_tokens,
        skipped_microbatches=skipped_microbatches,
        tokens_per_second=throughput.value or 0.0,
        history=history,
        per_component_loss=last_per_component,
        regressed=last_regressed,
    )
