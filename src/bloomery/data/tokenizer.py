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


def load_tokenizer(path: str | Path) -> Tokenizer:
    """Load a tokenizer from a directory or a Hugging Face repository id.

    Takes ``str`` as well as ``Path`` because a repository id is not a path.
    Routing ``Qwen/Qwen2-0.5B`` through ``Path`` rewrites the separator on
    Windows and the lookup then fails, in a way that would never show up on a
    machine where the separator happens to be a slash.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(path))


def adopt_tokenizer(source: str | Path, out_dir: Path) -> Tokenizer:
    """Take an existing model's tokenizer instead of training a new one.

    ``source`` is a local directory or a Hugging Face repository id.

    Continuing to train a model means writing into an embedding table indexed by
    *its* tokenizer's ids, so a corpus for that model has to be packed with that
    tokenizer. This is what makes such a corpus possible; without it every
    dataset carries a tokenizer bloomery invented, and no existing model can
    read it.

    The tokenizer is saved into the dataset alongside the tokens so the dataset
    stays self-describing — the same shape ``prepare`` already produces, and the
    same shape the mixture loader expects.
    """
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(source))
    except Exception as exc:  # noqa: BLE001 - transformers raises many types here
        raise ValueError(
            f"could not load a tokenizer from {source!r}: {exc}\n"
            "Give a local model directory or a Hugging Face repository id."
        ) from exc

    if tokenizer.eos_token_id is None and tokenizer.bos_token_id is None:
        raise ValueError(
            f"the tokenizer at {source!r} declares neither an end-of-text nor a "
            "beginning-of-text token, so documents cannot be separated in the "
            "packed stream."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(out_dir))
    return tokenizer


def id_space(tokenizer: Tokenizer) -> int:
    """The smallest array width that can hold every id this tokenizer emits.

    Counting entries is not enough. ``vocab_size`` excludes added tokens and
    ``len()`` includes them, but both are counts, and an id is not bounded by
    how many of them there are: a tokenizer can hold 32,001 entries where one
    sits at id 128,255, which several published models do because they reserve a
    block of special tokens at the top.

    Sizing the packed array from a count is then how that token silently wraps in
    ``uint16``, coming back as whatever it collides with, with nothing reporting
    it. So take the highest id actually in use.
    """
    counts = [int(tokenizer.vocab_size), len(tokenizer)]
    # Guarded rather than assumed: every tokenizer `prepare` writes has this,
    # but widening an array is a decision worth making from what is actually
    # there, not from a call that might not exist on some future backend.
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        counts.append(max(get_vocab().values(), default=-1) + 1)
    return max(counts)


# Text the fallback fingerprint runs through a tokenizer to see what it does with
# it. Chosen to move under the transformations that separate two tokenizers
# sharing a vocabulary: letter case, accents and other combining marks, runs of
# whitespace, punctuation attachment, digit grouping, and non-Latin scripts.
_PROBE = (
    "The Quick brown FOX\n"
    "  spaced\tout  \n"
    "café CAFÉ café\n"
    "hello, world! (don't) [x]=1;\n"
    "1234567890 3.14 -7\n"
    "日本語 Ελληνικά Привет 🙂\n"
)


def fingerprint(tokenizer: Tokenizer) -> str:
    """Identify a tokenizer by what it does, not by the file it came from.

    The mixture loader compares datasets by hashing ``tokenizer.json`` directly,
    which is right there: both sides were written by this project, so identical
    tokenizers give identical bytes.

    That does not hold when one side is someone else's model. Saving a tokenizer
    re-serialises it, and a re-serialised file need not be byte-identical to the
    original even though the tokenizer is the same. Hashing the backend's own
    canonical form compares the vocabulary and merges instead, which is what
    actually has to match.

    A tokenizer with no fast backend has no such canonical form, and that case is
    reachable: ``AutoTokenizer`` returns a slow tokenizer for any model that
    ships no fast implementation, and ``adopt_tokenizer`` will happily take one.
    Hashing its vocabulary alone would be unsafe in the worst direction — two
    tokenizers can share a vocabulary and still assign different ids to the same
    text, because normalisation happens before the lookup. One that lowercases
    and one that does not would fingerprint identically, and this comparison is
    what stands between a user and training on ids that mean nothing to their
    model.

    So the fallback also asks the tokenizer what it actually produces for a fixed
    probe. That captures normalisation, case folding and pre-tokenisation by
    observing them rather than by trying to enumerate them. It is not a proof of
    equality — two tokenizers agreeing on the probe could still differ on some
    other input — but it fails in the safe direction far more often, and a
    difference it does see is a real one.
    """
    import hashlib

    # Included explicitly, because it is invisible to everything else here. The
    # template lives on the wrapper rather than the backend, so it is absent from
    # the canonical form below and from the tokenizer.json the mixture loader
    # hashes. Two tokenizers differing only in how they lay out a conversation
    # would otherwise compare equal — and a corpus formatted one way, trained
    # against a model expecting another, is a model taught a shape it will never
    # be prompted in. Valid ids, wrong format, no error.
    template = getattr(tokenizer, "chat_template", None) or ""

    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        canonical = backend.to_str() + template
    else:
        canonical = repr(
            (
                sorted(tokenizer.get_vocab().items()),
                _probe_ids(tokenizer),
                template,
            )
        )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _probe_ids(tokenizer: Tokenizer) -> list[int] | str:
    """What this tokenizer makes of :data:`_PROBE`, for the fallback fingerprint."""
    try:
        encoded = tokenizer(_PROBE, add_special_tokens=False)["input_ids"]
    except Exception as exc:  # noqa: BLE001 - a tokenizer that cannot encode is a difference too
        return f"unencodable: {type(exc).__name__}"
    return [int(i) for i in encoded]


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
