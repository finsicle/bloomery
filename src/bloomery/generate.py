# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sampling from a trained checkpoint.

Being able to talk to the thing you just trained is not a nicety. A loss curve
tells you the optimiser is working; only generated text tells you whether the
model learned anything you wanted. For a first from-scratch run, that gap is the
whole lesson.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    max_new_tokens: int = 80
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 40
    repetition_penalty: float = 1.1
    seed: int | None = None

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0


@dataclass(slots=True)
class LoadedModel:
    model: Any
    tokenizer: Any
    device: Any

    @property
    def params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())


def load(checkpoint: Path, *, device: str | None = None) -> LoadedModel:
    """Load a checkpoint for inference.

    Plain ``from_pretrained``, because bloomery checkpoints are ordinary Hugging
    Face directories. Anything else in the ecosystem can load them the same way.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"{checkpoint} is not a model directory (no config.json)")

    from bloomery.train.device import select_device

    resolved = torch.device(device) if device else select_device()

    model = AutoModelForCausalLM.from_pretrained(str(checkpoint))
    model.to(resolved)  # type: ignore[arg-type]
    model.eval()
    # Re-enabled explicitly: training turns the cache off when gradient
    # checkpointing is on, and that setting is persisted in config.json.
    model.config.use_cache = True

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    return LoadedModel(model=model, tokenizer=tokenizer, device=resolved)


def complete(loaded: LoadedModel, prompt: str, config: SamplingConfig | None = None) -> str:
    """Continue a prompt, returning only the newly generated text."""
    import torch

    config = config or SamplingConfig()
    if config.seed is not None:
        torch.manual_seed(config.seed)

    tokenizer = loaded.tokenizer
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {k: v.to(loaded.device) for k, v in encoded.items()}
    prompt_length = encoded["input_ids"].shape[1]

    kwargs: dict[str, Any] = {
        "max_new_tokens": config.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if config.greedy:
        kwargs["do_sample"] = False
    else:
        kwargs.update(
            do_sample=True,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=config.repetition_penalty,
        )

    with torch.no_grad():
        output = loaded.model.generate(**encoded, **kwargs)

    # Slice off the prompt so callers get the continuation, not an echo.
    new_tokens = output[0][prompt_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
