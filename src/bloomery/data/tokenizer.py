# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Training and loading a byte-level BPE tokenizer.

Training your own tokenizer is part of what "from scratch" means. A model built
on someone else's vocabulary inherits their assumptions about what a word is,
which matters if your corpus is code, a non-English language, or a narrow
domain.

Byte-level BPE so there is no unknown token: any input encodes, including
emoji and broken UTF-8. That property is worth more than a slightly better
compression ratio, because an out-of-vocabulary crash mid-training is a much
worse failure than a few extra tokens.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

# Tokenizers are held as `Any` rather than pinned to a concrete class.
# transformers restructured its tokenizer hierarchy in 5.x — AutoTokenizer now
# returns one of several backend classes — so naming a single type here would
# break across the version range this project supports. We only ever use the
# small, stable surface: calling it, `decode`, `save_pretrained`, `vocab_size`
# and `len()`.
Tokenizer = Any

# Marks a document boundary. Also serves as BOS, EOS and PAD: at pretraining
# scale there is nothing to gain from distinguishing them, and a single special
# token keeps the vocabulary honest.
END_OF_TEXT = "<|endoftext|>"

# Batch size for feeding the trainer. Large enough to amortise the call
# overhead, small enough not to hold the corpus in memory twice.
_TRAIN_BATCH = 1000


def train_tokenizer(
    documents: Iterable[str],
    *,
    vocab_size: int,
    out_dir: Path,
    min_frequency: int = 2,
) -> Tokenizer:
    """Train a byte-level BPE tokenizer and save it in Hugging Face format.

    Saved so that ``AutoTokenizer.from_pretrained(out_dir)`` works, which is
    what makes the resulting checkpoint usable by anything else in the
    ecosystem.
    """
    from tokenizers import Tokenizer, decoders, pre_tokenizers, processors, trainers
    from tokenizers.models import BPE
    from transformers import PreTrainedTokenizerFast

    if vocab_size < 256 + 1:
        # Byte-level BPE needs 256 base symbols before it can merge anything.
        raise ValueError(f"vocab_size must be at least 257, got {vocab_size}")

    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[END_OF_TEXT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(_batched(documents), trainer=trainer)

    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=END_OF_TEXT,
        eos_token=END_OF_TEXT,
        unk_token=None,
        pad_token=END_OF_TEXT,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    wrapped.save_pretrained(str(out_dir))
    return wrapped


def _batched(documents: Iterable[str]) -> Iterator[list[str]]:
    batch: list[str] = []
    for document in documents:
        batch.append(document)
        if len(batch) >= _TRAIN_BATCH:
            yield batch
            batch = []
    if batch:
        yield batch


def load_tokenizer(path: Path) -> Tokenizer:
    """Load a tokenizer previously saved by :func:`train_tokenizer`."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(path))


def eot_id(tokenizer: Tokenizer) -> int:
    """The end-of-text id, which is also the document separator.

    Falls back to 0 only if the tokenizer somehow has no special token
    registered; that is the id the trainer assigns it, since it is added first.
    """
    for candidate in (tokenizer.eos_token_id, tokenizer.bos_token_id):
        if candidate is not None:
            return int(candidate)
    resolved = tokenizer.convert_tokens_to_ids(END_OF_TEXT)
    if isinstance(resolved, list):
        resolved = resolved[0] if resolved else None
    return int(resolved) if resolved is not None else 0
