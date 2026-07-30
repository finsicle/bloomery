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
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bloomery.arch import suggested_lr, to_llama_config
from bloomery.capability import ModelSpec
from bloomery.data.shards import DatasetInfo, open_split
from bloomery.train import checkpoint
from bloomery.train.device import DeviceChoice
from bloomery.train.metrics import MetricsWriter, Throughput

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch


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
    history: list[dict[str, Any]] = field(default_factory=list)


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

    @property
    def tokens(self) -> int:
        return self._length

    def batch(self, size: int, device: torch.device) -> torch.Tensor:
        import numpy as np
        import torch

        data = open_split(self._info, self._split)
        starts = self._rng.integers(0, self._length - self._seq - 1, size=size)
        window = np.stack([data[i : i + self._seq] for i in starts]).astype(np.int64)
        tensor = torch.from_numpy(window)
        return tensor.to(device, non_blocking=True)


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


def evaluate(
    model: Any,
    sampler: BatchSampler,
    choice: DeviceChoice,
    *,
    batch: int,
    batches: int,
) -> float:
    """Mean loss over a fixed number of held-out batches."""
    import torch

    model.eval()
    total = 0.0
    counted = 0
    with torch.no_grad():
        for _ in range(batches):
            inputs = sampler.batch(batch, choice.device)
            with autocast_for(choice):
                loss = model(input_ids=inputs, labels=inputs).loss
            if torch.isfinite(loss):
                total += loss.item()
                counted += 1
    model.train()
    return total / counted if counted else float("nan")


def train(
    *,
    spec: ModelSpec,
    dataset: DatasetInfo,
    tokenizer: Any,
    run_dir: Path,
    config: TrainConfig,
    choice: DeviceChoice,
    eos_token_id: int,
    resume_from: Path | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> TrainResult:
    """Train a model from random initialisation, or resume one.

    Emits every event to ``run.jsonl`` and, if given, to ``on_event`` for live
    display.
    """
    import time

    import torch
    from transformers import LlamaForCausalLM

    run_dir.mkdir(parents=True, exist_ok=True)
    peak_lr = config.lr if config.lr is not None else suggested_lr(spec)

    if resume_from is not None:
        model = LlamaForCausalLM.from_pretrained(str(resume_from))
    else:
        model = build_model(spec, eos_token_id=eos_token_id, seed=config.seed)
    model.to(choice.device)  # type: ignore[arg-type]
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
    best_val: float | None = None
    if resume_from is not None:
        state = checkpoint.load_resume_state(resume_from, optimizer)
        start_step, tokens_seen, best_val = state.step, state.tokens_seen, state.best_val_loss

    # fp16 needs loss scaling to keep small gradients from flushing to zero.
    # bf16 has the exponent range to avoid this, so the scaler stays disabled.
    scaler = torch.amp.GradScaler(choice.type, enabled=choice.dtype == torch.float16)

    train_sampler = BatchSampler(dataset, "train", seq=config.seq, seed=config.seed)
    val_sampler: BatchSampler | None
    try:
        val_sampler = BatchSampler(dataset, "val", seq=config.seq, seed=config.seed + 1)
    except (ValueError, FileNotFoundError):
        val_sampler = None

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
                config=config.to_dict(),
            )
        )

        for step in range(start_step, config.steps):
            lr = lr_at(step, config, peak_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr

            began = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0

            for _ in range(config.grad_accum):
                inputs = train_sampler.batch(config.batch, choice.device)
                with autocast_for(choice):
                    loss = model(input_ids=inputs, labels=inputs).loss
                    # Average over accumulation so the gradient matches what a
                    # single large batch would have produced.
                    scaled = loss / config.grad_accum
                scaler.scale(scaled).backward()
                step_loss += scaled.item()

            if config.grad_clip > 0:
                # Unscale first, or the clip threshold is applied to inflated
                # fp16 gradients and does nothing.
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            tokens_seen += config.tokens_per_step
            final_loss = step_loss
            rate = throughput.update(config.tokens_per_step, time.monotonic() - began)

            if config.log_every and (step + 1) % config.log_every == 0:
                record(
                    metrics.emit(
                        "step",
                        step=step + 1,
                        loss=round(step_loss, 5),
                        lr=round(lr, 8),
                        tokens=tokens_seen,
                        tokens_per_second=round(rate, 1),
                    )
                )

            should_eval = (
                val_sampler is not None
                and config.eval_every
                and (step + 1) % config.eval_every == 0
            )
            if should_eval and val_sampler is not None:
                val_loss = evaluate(
                    model,
                    val_sampler,
                    choice,
                    batch=config.batch,
                    batches=config.eval_batches,
                )
                if math.isfinite(val_loss) and (best_val is None or val_loss < best_val):
                    best_val = val_loss
                record(
                    metrics.emit(
                        "eval",
                        step=step + 1,
                        val_loss=round(val_loss, 5),
                        best_val_loss=round(best_val, 5) if best_val is not None else None,
                        perplexity=round(math.exp(min(val_loss, 20)), 2)
                        if math.isfinite(val_loss)
                        else None,
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
                    best_val_loss=best_val,
                    extra={"spec": asdict(spec), "config": config.to_dict()},
                )
                record(metrics.emit("checkpoint", step=step + 1, path=str(path)))

        final_path = checkpoint.save(
            checkpoint.checkpoint_dir(run_dir),
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            step=config.steps,
            tokens_seen=tokens_seen,
            best_val_loss=best_val,
            extra={"spec": asdict(spec), "config": config.to_dict()},
        )

        record(
            metrics.emit(
                "done",
                step=config.steps,
                final_loss=round(final_loss, 5),
                best_val_loss=round(best_val, 5) if best_val is not None else None,
                tokens=tokens_seen,
                tokens_per_second=round(throughput.value or 0.0, 1),
                checkpoint=str(final_path),
            )
        )

    return TrainResult(
        run_dir=run_dir,
        checkpoint=final_path,
        steps=config.steps,
        final_loss=final_loss,
        best_val_loss=best_val,
        tokens_seen=tokens_seen,
        tokens_per_second=throughput.value or 0.0,
        history=history,
    )
