# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Model architecture, derived from a single complexity dial.

Bloomery trains Llama-architecture models. That is a deliberate constraint, not
a limitation we inherited: emitting a standard architecture from the first byte
means the checkpoint is loadable by ``transformers`` immediately, convertible to
GGUF by ``llama.cpp`` without a bespoke converter, and servable by vLLM or
Ollama. Inventing an architecture would mean owning a converter forever.

The *depth* dial is borrowed in spirit from nanochat: pick how many layers you
want and everything else — width, heads, MLP size, learning rate — follows. One
number a beginner can reason about beats twelve they will get wrong.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from bloomery.capability import LADDER_BY_KEY, ModelSpec

if TYPE_CHECKING:  # pragma: no cover - typing only
    from transformers import LlamaConfig

# Width per layer of depth. Gives 256 wide at depth 4 and 768 at depth 12, which
# lands the depth-12 model on GPT-2 small's shape.
_WIDTH_PER_DEPTH = 64

# Attention head width. 64 keeps every size compatible with SDPA and flash
# kernels, which are picky about head dimensions.
_HEAD_DIM = 64

# SwiGLU uses three matrices rather than two, so the conventional 4x expansion
# becomes 8/3 to keep the parameter count the same.
_MLP_EXPANSION = 8 / 3
_MLP_MULTIPLE = 64

# Learning rate anchor: a 768-wide model trains well around 6e-4. Scaled by
# 1/sqrt(width) from there, the standard width-scaling rule.
_LR_ANCHOR_WIDTH = 768
_LR_AT_ANCHOR = 6e-4


def round_to(value: float, multiple: int) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def spec_from_depth(
    depth: int,
    *,
    vocab: int,
    seq: int = 1024,
    batch: int = 8,
    key: str | None = None,
    label: str | None = None,
) -> ModelSpec:
    """Derive a full model shape from a layer count.

    Raises on a depth that cannot produce a valid attention configuration,
    rather than silently rounding to something the user did not ask for.
    """
    if depth < 1:
        raise ValueError(f"depth must be at least 1, got {depth}")

    hidden = round_to(depth * _WIDTH_PER_DEPTH, _HEAD_DIM)
    heads = hidden // _HEAD_DIM
    if heads < 1:
        raise ValueError(f"depth {depth} is too small to form an attention head")

    return ModelSpec(
        key=key or f"d{depth}",
        label=label or f"depth {depth}",
        layers=depth,
        hidden=hidden,
        heads=heads,
        vocab=vocab,
        seq=seq,
        batch=batch,
    )


def spec_from_model_config(
    config: Any,
    *,
    seq: int,
    batch: int,
    params: int | None = None,
    label: str | None = None,
) -> ModelSpec:
    """Describe a model that already exists, rather than one to be built.

    Deliberately not routed through :func:`resolve_spec`, which exists to turn
    ``--size``/``--depth`` into a shape and falls back to depth 4 when given
    neither. Here the shape is not a choice — it is a fact about a checkpoint,
    and inventing one would silently misreport what is about to be trained.

    ``params`` should be the real parameter count from the loaded model. Without
    it :attr:`ModelSpec.params` falls back to a closed form that assumes this
    project's own conventions, which a foreign checkpoint need not follow.

    ``seq`` is what the run will actually use, not the model's maximum: memory
    scales with the sequence length trained on, and the two are rarely equal.
    """
    missing = [
        name
        for name in ("num_hidden_layers", "hidden_size", "num_attention_heads", "vocab_size")
        if getattr(config, name, None) is None
    ]
    if missing:
        raise ValueError(
            f"this model's config does not describe its shape: missing {', '.join(missing)}. "
            "Only decoder-style causal language models are supported."
        )

    model_type = getattr(config, "model_type", None) or "model"
    return ModelSpec(
        key=model_type,
        label=label or model_type,
        layers=config.num_hidden_layers,
        hidden=config.hidden_size,
        heads=config.num_attention_heads,
        vocab=config.vocab_size,
        seq=seq,
        batch=batch,
        params_override=params,
    )


def resolve_spec(
    *,
    size: str | None = None,
    depth: int | None = None,
    vocab: int | None = None,
    seq: int | None = None,
    batch: int | None = None,
) -> ModelSpec:
    """Turn CLI-shaped input into a concrete model shape.

    Accepts either a named ladder preset or a raw depth. Vocab normally comes
    from the tokenizer that was actually trained, so it overrides the preset.
    """
    if size and depth:
        raise ValueError("give either --size or --depth, not both")

    if size:
        preset = LADDER_BY_KEY.get(size)
        if preset is None:
            known = ", ".join(LADDER_BY_KEY)
            raise ValueError(f"unknown size {size!r}; choose one of: {known}")
        base = preset
    else:
        base = spec_from_depth(depth or 4, vocab=vocab or 8192, seq=seq or 512)

    return ModelSpec(
        key=base.key,
        label=base.label,
        layers=base.layers,
        hidden=base.hidden,
        heads=base.heads,
        vocab=vocab if vocab is not None else base.vocab,
        seq=seq if seq is not None else base.seq,
        batch=batch if batch is not None else base.batch,
        note=base.note,
    )


def intermediate_size(hidden: int) -> int:
    """SwiGLU inner width, rounded for hardware-friendly shapes."""
    return round_to(hidden * _MLP_EXPANSION, _MLP_MULTIPLE)


def suggested_lr(spec: ModelSpec) -> float:
    """A peak learning rate that trains stably at this width.

    Width scaling only. It is a sane starting point, not a tuned value.
    """
    return _LR_AT_ANCHOR * math.sqrt(_LR_ANCHOR_WIDTH / spec.hidden)


def to_llama_config(spec: ModelSpec, *, eos_token_id: int = 0) -> LlamaConfig:
    """Build a ``LlamaConfig`` for a randomly initialised model.

    Embeddings are tied, matching the parameter-count model in
    :mod:`bloomery.capability`, and saving a full vocab x hidden matrix — which
    at small scale is a large fraction of the whole model.
    """
    from transformers import LlamaConfig  # noqa: PLC0415 - optional extra

    return LlamaConfig(
        vocab_size=spec.vocab,
        hidden_size=spec.hidden,
        intermediate_size=intermediate_size(spec.hidden),
        num_hidden_layers=spec.layers,
        num_attention_heads=spec.heads,
        # Multi-head rather than grouped-query. GQA saves KV-cache memory at
        # inference but complicates nothing else at these sizes, and MHA keeps
        # the parameter count matching the estimator.
        num_key_value_heads=spec.heads,
        max_position_embeddings=spec.seq,
        rms_norm_eps=1e-5,
        # RoPE base is left at the default, which is 10000.0 — the value we want.
        # Passing it explicitly is a trap: transformers 5 renamed `rope_theta` to
        # `rope_parameters`, so an explicit kwarg only survives via a deprecation
        # shim that will eventually go away.
        tie_word_embeddings=True,
        attention_bias=False,
        mlp_bias=False,
        bos_token_id=eos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=eos_token_id,
    )


def actual_param_count(spec: ModelSpec) -> int:
    """Exact parameter count for the config this module would build.

    :attr:`ModelSpec.params` is a closed-form approximation used for memory
    estimates before anything is built. This is the real number, and the two are
    held within a few percent of each other by a test — if they drift, the
    numbers ``bloomery doctor`` prints stop describing what actually gets
    trained.
    """
    hidden = spec.hidden
    inner = intermediate_size(hidden)
    head_dim = hidden // spec.heads

    embedding = spec.vocab * hidden  # tied, so counted once
    attention = 2 * hidden * hidden + 2 * hidden * (spec.heads * head_dim)
    mlp = 3 * hidden * inner
    norms = 2 * hidden  # input and post-attention RMSNorm, one weight each

    return embedding + spec.layers * (attention + mlp + norms) + hidden
