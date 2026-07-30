# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Measured training throughput.

``bloomery doctor`` deliberately does not estimate wall-clock time. Doing that
from a table of peak-FLOPS figures per GPU produces a number that is wrong on any
machine with a thermal limit, a shared GPU, a different driver, or an
architecture the table predates — and a confidently wrong time estimate is worse
than none.

So instead we run the real thing briefly and measure. A few seconds of actual
forward and backward passes gives a tokens-per-second figure that already
accounts for this machine's memory bandwidth, its cooling, and whatever else is
running on it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from bloomery.arch import actual_param_count
from bloomery.capability import TOKENS_PER_PARAM, ModelSpec
from bloomery.train.device import DeviceChoice
from bloomery.train.loop import autocast_for, build_model

# Discarded before timing starts: the first steps pay for kernel autotuning,
# allocator warmup and, on Metal, shader compilation.
DEFAULT_WARMUP = 3
DEFAULT_STEPS = 10


@dataclass(frozen=True, slots=True)
class BenchResult:
    spec: ModelSpec
    device: str
    dtype: str
    params: int
    batch: int
    seq: int
    steps: int
    seconds: float
    tokens_per_second: float
    peak_memory: int | None

    @property
    def seconds_per_step(self) -> float:
        return self.seconds / self.steps if self.steps else float("nan")

    def hours_for_tokens(self, tokens: int) -> float:
        if self.tokens_per_second <= 0:
            return float("nan")
        return tokens / self.tokens_per_second / 3600

    def hours_for_chinchilla(self) -> float:
        """Wall clock to train this size on a compute-optimal token budget."""
        return self.hours_for_tokens(self.params * TOKENS_PER_PARAM)


def _peak_memory(device_type: str) -> int | None:
    import torch

    try:
        if device_type == "cuda":
            return int(torch.cuda.max_memory_allocated())
        if device_type == "mps":
            return int(torch.mps.current_allocated_memory())
    except Exception:  # noqa: BLE001 - availability varies by build
        return None
    return None


def _synchronize(device_type: str) -> None:
    """Wait for queued work before reading the clock.

    Both CUDA and Metal dispatch asynchronously, so timing without this measures
    how fast Python can submit work, not how fast the device completes it.
    """
    import torch

    try:
        if device_type == "cuda":
            torch.cuda.synchronize()
        elif device_type == "mps":
            torch.mps.synchronize()
    except Exception:  # noqa: BLE001
        pass


def run(
    spec: ModelSpec,
    choice: DeviceChoice,
    *,
    batch: int | None = None,
    seq: int | None = None,
    steps: int = DEFAULT_STEPS,
    warmup: int = DEFAULT_WARMUP,
    gradient_checkpointing: bool = False,
    seed: int = 1337,
) -> BenchResult:
    """Time real training steps on random tokens.

    Random data is fine here: throughput depends on tensor shapes and the device,
    not on what the tokens mean.
    """
    import torch

    batch = batch or spec.batch
    seq = seq or spec.seq

    if choice.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model = build_model(spec, eos_token_id=0, seed=seed)
    model.to(choice.device)
    model.train()
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=choice.type == "cuda")
    generator = torch.Generator().manual_seed(seed)

    def one_step() -> None:
        inputs = torch.randint(
            0, spec.vocab, (batch, seq), generator=generator, dtype=torch.long
        ).to(choice.device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_for(choice):
            loss = model(input_ids=inputs, labels=inputs).loss
        loss.backward()
        optimizer.step()

    for _ in range(max(0, warmup)):
        one_step()
    _synchronize(choice.type)

    began = time.perf_counter()
    for _ in range(steps):
        one_step()
    _synchronize(choice.type)
    elapsed = time.perf_counter() - began

    tokens = batch * seq * steps
    if elapsed <= 0:
        # Should be unreachable with perf_counter, but a zero here would be
        # reported as "0 tok/s", which reads as a failure rather than as an
        # unmeasurably fast run.
        raise RuntimeError("benchmark completed in unmeasurable time; increase --steps or --batch")
    return BenchResult(
        spec=spec,
        device=str(choice.device),
        dtype=str(choice.dtype).removeprefix("torch."),
        params=actual_param_count(spec),
        batch=batch,
        seq=seq,
        steps=steps,
        seconds=elapsed,
        tokens_per_second=tokens / elapsed,
        peak_memory=_peak_memory(choice.type),
    )


def to_dict(result: BenchResult) -> dict[str, Any]:
    return {
        "model": result.spec.key,
        "device": result.device,
        "dtype": result.dtype,
        "params": result.params,
        "batch": result.batch,
        "seq": result.seq,
        "steps": result.steps,
        "seconds": round(result.seconds, 3),
        "tokens_per_second": round(result.tokens_per_second, 1),
        "seconds_per_step": round(result.seconds_per_step, 4),
        "peak_memory_bytes": result.peak_memory,
        "chinchilla_hours": round(result.hours_for_chinchilla(), 2),
    }
