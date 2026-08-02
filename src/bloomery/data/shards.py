# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Packing tokenized text into flat memory-mapped arrays.

One contiguous array of token ids per split, separated by the end-of-text token,
plus a small JSON sidecar. Training samples random windows out of it.

The format is deliberately dull. A flat array on disk memory-maps in constant
time regardless of size, needs no index, survives being copied between machines,
and can be inspected with numpy in one line. Anything cleverer would buy
throughput we do not need and cost debuggability we do.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sized
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bloomery.data.tokenizer import id_space

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

# See bloomery.data.tokenizer for why this is not a concrete class.
Tokenizer = Any

META_NAME = "meta.json"
SPLITS = ("train", "val")

# How a dataset is laid out. "packed" is a plain token stream, every token
# trained on. "sft" carries a parallel byte per token saying whether it does.
# "dpo" is not a stream at all: preference pairs have to stay whole, so they are
# stored as three streams plus an index of where each example sits in them.
FORMAT_PACKED = "packed"
FORMAT_SFT = "sft"
FORMAT_DPO = "dpo"

# The parts of a preference example, in the order the index records their
# lengths. The prompt is stored once rather than inside each answer: DPO only
# means anything if both answers condition on a token-identical prompt, and one
# copy makes that structural instead of a rule someone has to keep.
PREFERENCE_PARTS = ("prompt", "chosen", "rejected")

# The mask is one byte per token. uint8 rather than a bitfield because it is
# read as a window alongside the tokens and the arithmetic stays obvious; at one
# eighth of the token array even for uint16 ids, packing it would save little.
MASK_DTYPE = "uint8"

# uint16 halves the on-disk size but only addresses 65536 ids.
_UINT16_LIMIT = 1 << 16

# Documents per encode call. Batching matters: the fast tokenizer releases the
# GIL and parallelises internally, so one-at-a-time encoding is several times
# slower.
_ENCODE_BATCH = 512


def dtype_for_vocab(vocab_size: int) -> str:
    return "uint16" if vocab_size < _UINT16_LIMIT else "uint32"


@dataclass(frozen=True, slots=True)
class SplitInfo:
    name: str
    tokens: int
    documents: int


@dataclass(frozen=True, slots=True)
class DatasetInfo:
    """What ended up on disk, as recorded in ``meta.json``."""

    root: Path
    dtype: str
    vocab_size: int
    splits: tuple[SplitInfo, ...]
    # "packed" for a plain corpus, "sft" when a completion mask sits beside the
    # tokens. Defaulted so a dataset written before masking existed still loads.
    format: str = FORMAT_PACKED
    # Why records did not make it in, by reason. Provenance rather than layout,
    # and it belongs here because it is the question asked months later when a
    # model underperforms and the corpus looks smaller than the file it came
    # from. Empty for a format that drops nothing.
    dropped: Mapping[str, int] = field(default_factory=dict)

    def split(self, name: str) -> SplitInfo:
        for info in self.splits:
            if info.name == name:
                return info
        raise KeyError(f"no split named {name!r}")

    @property
    def total_tokens(self) -> int:
        return sum(info.tokens for info in self.splits)

    @property
    def masked(self) -> bool:
        return self.format == FORMAT_SFT

    @property
    def preference(self) -> bool:
        return self.format == FORMAT_DPO

    def bin_path(self, name: str) -> Path:
        return self.root / f"{name}.bin"

    def mask_path(self, name: str) -> Path:
        return self.root / f"{name}.mask.bin"

    def part_path(self, name: str, part: str) -> Path:
        """Where one side of a preference split lives."""
        return self.root / f"{name}.{part}.bin"

    def index_path(self, name: str) -> Path:
        return self.root / f"{name}.index.bin"

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "dtype": self.dtype,
            "vocab_size": self.vocab_size,
            "format": self.format,
            "splits": [
                {"name": s.name, "tokens": s.tokens, "documents": s.documents} for s in self.splits
            ],
        }
        if self.dropped:
            payload["dropped"] = dict(self.dropped)
        return payload


def _val_stride(
    documents: Iterable[str],
    *,
    val_fraction: float,
    min_val_documents: int,
) -> int:
    """How often to divert a document into the validation split.

    Derived from the real corpus size whenever the corpus is a sized collection.
    Deriving it from the fraction alone — ``1 / 0.01`` — means a 60-document
    corpus never reaches a multiple of 100 and ends up with no validation set at
    all, which turned a perfectly reasonable ``prepare`` into a hard error.

    Returns 0 for "no validation split".
    """
    if val_fraction <= 0:
        return 0

    total = len(documents) if isinstance(documents, Sized) else None
    if total is None:
        # Streaming corpus: the fraction is all we have to go on.
        return max(2, int(round(1 / val_fraction)))

    if total < 2:
        raise ValueError(
            "corpus too small to hold out a validation split; "
            "add more documents or pass val_fraction=0"
        )

    # At least one document, and never so many that training is left with none.
    wanted = min(max(min_val_documents, round(total * val_fraction)), total - 1)
    return max(2, total // wanted)


def encode_documents(
    documents: Iterable[str],
    tokenizer: Tokenizer,
    *,
    eot: int,
) -> Iterator[list[int]]:
    """Encode documents, appending the separator to each."""
    batch: list[str] = []

    def flush() -> Iterator[list[int]]:
        if not batch:
            return
        encoded = tokenizer(batch, add_special_tokens=False)["input_ids"]
        for ids in encoded:
            yield [*ids, eot]

    for document in documents:
        batch.append(document)
        if len(batch) >= _ENCODE_BATCH:
            yield from flush()
            batch = []
    yield from flush()


def build_dataset(
    documents: Iterable[str],
    tokenizer: Tokenizer,
    *,
    out_dir: Path,
    eot: int,
    val_fraction: float = 0.01,
    min_val_documents: int = 1,
) -> DatasetInfo:
    """Tokenize documents and write ``train.bin`` and ``val.bin``.

    The validation split is taken by striding through the corpus rather than by
    slicing off the tail, so a corpus whose files are ordered — by date, by
    source, by topic — does not end up with a validation set that looks nothing
    like training. A held-out set that is not representative reports a loss you
    cannot act on.
    """
    import numpy as np

    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")

    out_dir.mkdir(parents=True, exist_ok=True)
    # The full id space, not the base vocabulary. A tokenizer that came with
    # someone else's model can have added tokens whose ids sit above
    # ``vocab_size``, and sizing the array from the smaller number wraps them.
    vocab = id_space(tokenizer)
    dtype = dtype_for_vocab(vocab)
    np_dtype = np.dtype(dtype)

    stride = _val_stride(
        documents,
        val_fraction=val_fraction,
        min_val_documents=min_val_documents,
    )

    handles = {name: (out_dir / f"{name}.bin").open("wb") for name in SPLITS}
    counts = dict.fromkeys(SPLITS, 0)
    documents_seen = dict.fromkeys(SPLITS, 0)

    try:
        for position, ids in enumerate(encode_documents(documents, tokenizer, eot=eot)):
            target = "val" if stride and (position + 1) % stride == 0 else "train"
            array = np.asarray(ids, dtype=np_dtype)
            array.tofile(handles[target])
            counts[target] += array.size
            documents_seen[target] += 1
    finally:
        for handle in handles.values():
            handle.close()

    if counts["train"] == 0:
        raise ValueError("no training tokens produced; is the corpus empty?")
    if val_fraction > 0 and documents_seen["val"] < min_val_documents:
        raise ValueError(
            "corpus too small to hold out a validation split; "
            "add more documents or pass val_fraction=0"
        )

    info = DatasetInfo(
        root=out_dir,
        dtype=dtype,
        # Recorded as the id space too, so the number that describes the packed
        # array is the same one that sized it.
        vocab_size=vocab,
        splits=tuple(
            SplitInfo(name=name, tokens=counts[name], documents=documents_seen[name])
            for name in SPLITS
        ),
    )
    (out_dir / META_NAME).write_text(json.dumps(info.to_dict(), indent=2) + "\n")
    return info


class SftError(ValueError):
    """A conversation corpus could not be packed."""


def encode_conversations(
    examples: Iterable[Any],
    tokenizer: Tokenizer,
    *,
    eot: int,
) -> Iterator[tuple[list[int], list[int]]]:
    """Encode conversations, yielding ids and a per-token mask.

    The mask marks assistant turns and nothing else, so the model learns to
    produce a response rather than to reproduce the question it was asked.

    Boundaries come from character offsets rather than from tokenizing the
    prompt and the response separately. BPE merges across a join, so the length
    of a separately tokenized prefix is not reliably where that prefix ends
    inside the whole — the mask would land a token or two off, silently, and
    the model would be trained partly on the wrong side of its own turn. The
    conversation is rendered once, tokenized once, and the spans are matched in
    character space where no merging can happen.
    """
    if getattr(tokenizer, "chat_template", None) is None:
        raise SftError(
            "this tokenizer has no chat template, so there is no way to know how "
            "conversations should be formatted for the model.\n"
            "Take the tokenizer from an instruct model:\n"
            "  bloomery prepare --name ... --chat --tokenizer <an instruct model>"
        )
    if not getattr(tokenizer, "is_fast", False):
        raise SftError(
            "packing conversations needs a fast tokenizer, because the mask is "
            "computed from character offsets that only a fast tokenizer reports.\n"
            "The model this corpus is for does not ship one."
        )

    for example in examples:
        messages = [dict(turn) for turn in example.messages]
        # Rendered with tokenize=False deliberately: on transformers 5.x
        # apply_chat_template(tokenize=True) returns Encoding objects rather
        # than a flat list of ids.
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        spans = _response_spans(tokenizer, messages, text)
        if not spans:
            continue

        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = list(encoded["input_ids"])
        mask = [
            1 if any(start <= a and b <= end for start, end in spans) else 0
            for a, b in encoded["offset_mapping"]
        ]
        if not any(mask):
            continue
        # The separator is masked: it belongs to the packing, not to the reply,
        # and the template already taught the model how a turn ends.
        yield [*ids, eot], [*mask, 0]


def _response_spans(
    tokenizer: Tokenizer,
    messages: list[dict[str, str]],
    text: str,
) -> list[tuple[int, int]]:
    """Character spans of the assistant turns within the rendered conversation."""
    spans: list[tuple[int, int]] = []
    for index, turn in enumerate(messages):
        if turn["role"] != "assistant":
            continue
        before = tokenizer.apply_chat_template(
            messages[:index], tokenize=False, add_generation_prompt=True
        )
        through = tokenizer.apply_chat_template(messages[: index + 1], tokenize=False)
        # A template that does not render prefixes as prefixes cannot be masked
        # by offset. Better to refuse than to mask an arbitrary region.
        if not text.startswith(before) or not text.startswith(through):
            raise SftError(
                "this chat template does not render a conversation prefix as a prefix "
                "of the whole, so the assistant turns cannot be located reliably. "
                "Masking them by guesswork would train the model on the wrong half."
            )
        if len(through) > len(before):
            spans.append((len(before), len(through)))
    return spans


def build_sft_dataset(
    examples: Iterable[Any],
    tokenizer: Tokenizer,
    *,
    out_dir: Path,
    eot: int,
    val_fraction: float = 0.01,
    min_val_documents: int = 1,
) -> DatasetInfo:
    """Pack conversations, writing tokens and the completion mask beside them.

    Same layout as :func:`build_dataset` plus ``{split}.mask.bin``, so every
    reader of a packed corpus reads this one too — a blend of a conversation
    corpus and a plain one works, with the plain component trained on in full.
    """
    import numpy as np

    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")

    out_dir.mkdir(parents=True, exist_ok=True)
    vocab = id_space(tokenizer)
    dtype = dtype_for_vocab(vocab)
    np_dtype = np.dtype(dtype)
    mask_dtype = np.dtype(MASK_DTYPE)

    stride = _val_stride(
        examples,
        val_fraction=val_fraction,
        min_val_documents=min_val_documents,
    )

    handles = {name: (out_dir / f"{name}.bin").open("wb") for name in SPLITS}
    masks = {name: (out_dir / f"{name}.mask.bin").open("wb") for name in SPLITS}
    counts = dict.fromkeys(SPLITS, 0)
    trained = dict.fromkeys(SPLITS, 0)
    seen = dict.fromkeys(SPLITS, 0)

    try:
        encoded = encode_conversations(examples, tokenizer, eot=eot)
        for position, (ids, mask) in enumerate(encoded):
            target = "val" if stride and (position + 1) % stride == 0 else "train"
            np.asarray(ids, dtype=np_dtype).tofile(handles[target])
            np.asarray(mask, dtype=mask_dtype).tofile(masks[target])
            counts[target] += len(ids)
            trained[target] += sum(mask)
            seen[target] += 1
    finally:
        for handle in (*handles.values(), *masks.values()):
            handle.close()

    if counts["train"] == 0:
        raise SftError(
            "no conversations could be packed. Records need an assistant turn to "
            "learn from: either a `messages` list containing one, or a `prompt` "
            "and `completion` pair."
        )
    if val_fraction > 0 and seen["val"] < min_val_documents:
        raise ValueError(
            "corpus too small to hold out a validation split; "
            "add more conversations or pass val_fraction=0"
        )

    info = DatasetInfo(
        root=out_dir,
        dtype=dtype,
        vocab_size=vocab,
        format=FORMAT_SFT,
        splits=tuple(
            SplitInfo(name=name, tokens=counts[name], documents=seen[name]) for name in SPLITS
        ),
    )
    (out_dir / META_NAME).write_text(json.dumps(info.to_dict(), indent=2) + "\n")
    return info


# Why a preference pair did not make it into the dataset. Recorded per reason
# and not as one total, so that a corpus which came out smaller than its file
# says which problem to go and look at.
#
# In practice this is the one that fires. A template that cannot be split at all
# is caught by _response_spans, which raises for the whole corpus rather than
# dropping pairs one at a time — that is a property of the tokenizer, not of any
# single record.
DROP_PROMPT_DIVERGED = "the prompt differed between the two answers"
DROP_NO_ANSWER_TOKENS = "no answer tokens after the prompt"


def encode_preference(
    example: Any,
    tokenizer: Tokenizer,
    *,
    eot: int,
) -> tuple[tuple[list[int], list[int], list[int]] | None, str]:
    """Encode one preference pair into prompt, chosen and rejected ids.

    Returns the triple and an empty reason, or ``None`` and why not.

    Each answer is rendered as a whole conversation and tokenized once, and the
    prompt/answer boundary is then located in **character** space through
    :func:`_response_spans` — the same reasoning as :func:`encode_conversations`,
    and for the same reason: BPE merges across a join, so tokenizing the prompt
    on its own gives a length that is not where that prompt ends inside the
    whole. Splitting by that length would put the boundary a token or two off.

    Then the invariant DPO actually rests on is checked rather than assumed: the
    two renderings must produce identical prompt tokens. If they do not, the two
    answers are being compared under different conditions and the pair is
    dropped. The one stored prompt makes the invariant structural afterwards;
    this is what establishes it in the first place.
    """
    chosen = _encode_side(tokenizer, example, example.chosen)
    rejected = _encode_side(tokenizer, example, example.rejected)
    if chosen is None or rejected is None:
        # Defensive rather than expected: a template whose assistant turn cannot
        # be located makes _response_spans raise for the corpus before it gets
        # here. Without this, an empty span list would be an IndexError.
        return None, DROP_NO_ANSWER_TOKENS

    chosen_ids, chosen_prompt = chosen
    rejected_ids, rejected_prompt = rejected
    if chosen_ids[:chosen_prompt] != rejected_ids[:rejected_prompt]:
        return None, DROP_PROMPT_DIVERGED

    # The separator is part of the packing, not of the answer — but unlike the
    # SFT path it is trained on here, because stopping where a good answer stops
    # is part of what is being preferred.
    chosen_answer = [*chosen_ids[chosen_prompt:], eot]
    rejected_answer = [*rejected_ids[rejected_prompt:], eot]
    return (chosen_ids[:chosen_prompt], chosen_answer, rejected_answer), ""


def _encode_side(
    tokenizer: Tokenizer,
    example: Any,
    content: str,
) -> tuple[list[int], int] | None:
    """One answer rendered in place, as ids plus how many of them are prompt."""
    messages = [dict(turn) for turn in example.with_response(content)]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    spans = _response_spans(tokenizer, messages, text)
    if not spans:
        return None

    # The last span is the answer just appended; earlier ones are assistant turns
    # that were already part of the prompt.
    start = spans[-1][0]
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(encoded["input_ids"])
    # A token straddling the boundary belongs to neither side cleanly. Counting
    # it as prompt is what makes the divergence check above catch it, since the
    # two answers would then differ inside that shared token.
    prompt_length = sum(1 for _, b in encoded["offset_mapping"] if b <= start)
    if prompt_length >= len(ids):
        return None
    return ids, prompt_length


def build_preference_dataset(
    examples: Iterable[Any],
    tokenizer: Tokenizer,
    *,
    out_dir: Path,
    eot: int,
    val_fraction: float = 0.01,
    min_val_documents: int = 1,
) -> DatasetInfo:
    """Pack preference pairs as three token streams and an index.

    Not a single stream like the other two formats. A window sampled at random
    out of a concatenation would cut across examples, and half of one pair
    against half of another compares nothing. Each example has to stay whole, so
    the index records the three lengths and the streams hold the tokens
    back to back.
    """
    import numpy as np

    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")

    out_dir.mkdir(parents=True, exist_ok=True)
    vocab = id_space(tokenizer)
    dtype = dtype_for_vocab(vocab)
    np_dtype = np.dtype(dtype)

    stride = _val_stride(examples, val_fraction=val_fraction, min_val_documents=min_val_documents)

    handles = {
        (name, part): (out_dir / f"{name}.{part}.bin").open("wb")
        for name in SPLITS
        for part in PREFERENCE_PARTS
    }
    indexes = {name: (out_dir / f"{name}.index.bin").open("wb") for name in SPLITS}
    counts = dict.fromkeys(SPLITS, 0)
    seen = dict.fromkeys(SPLITS, 0)
    dropped: dict[str, int] = {}

    try:
        position = 0
        for example in examples:
            encoded, reason = encode_preference(example, tokenizer, eot=eot)
            if encoded is None:
                dropped[reason] = dropped.get(reason, 0) + 1
                continue
            target = "val" if stride and (position + 1) % stride == 0 else "train"
            position += 1
            for part, ids in zip(PREFERENCE_PARTS, encoded, strict=True):
                np.asarray(ids, dtype=np_dtype).tofile(handles[(target, part)])
                counts[target] += len(ids)
            np.asarray([len(ids) for ids in encoded], dtype=np.int64).tofile(indexes[target])
            seen[target] += 1
    finally:
        for handle in (*handles.values(), *indexes.values()):
            handle.close()

    if seen["train"] == 0:
        raise SftError(
            "no preference pairs could be packed. Records need a prompt and two "
            "different answers to it:\n"
            '  {"prompt": ..., "chosen": ..., "rejected": ...}\n'
            + (f"dropped: {dropped}" if dropped else "")
        )
    if val_fraction > 0 and seen["val"] < min_val_documents:
        raise ValueError(
            "corpus too small to hold out a validation split; add more pairs or pass val_fraction=0"
        )

    info = DatasetInfo(
        root=out_dir,
        dtype=dtype,
        vocab_size=vocab,
        format=FORMAT_DPO,
        dropped=dropped,
        splits=tuple(
            SplitInfo(name=name, tokens=counts[name], documents=seen[name]) for name in SPLITS
        ),
    )
    (out_dir / META_NAME).write_text(json.dumps(info.to_dict(), indent=2) + "\n")
    return info


def open_preference_split(info: DatasetInfo, name: str) -> tuple[dict[str, np.memmap], np.ndarray]:
    """The three token streams of a preference split, and where each example sits.

    The index is read whole rather than memory-mapped: it is three int64 per
    example, so a corpus of a million pairs is 24 MB, and every draw needs
    arbitrary rows of it. Returned as start offsets rather than lengths, since
    that is what every caller wants and the cumulative sum should happen once.
    """
    import numpy as np

    if not info.preference:
        raise ValueError(f"{info.root} is a {info.format!r} dataset, not preference pairs")

    index_path = info.index_path(name)
    if not index_path.is_file():
        raise FileNotFoundError(f"missing {index_path}")
    lengths = np.fromfile(index_path, dtype=np.int64).reshape(-1, len(PREFERENCE_PARTS))

    streams = {}
    for part in PREFERENCE_PARTS:
        path = info.part_path(name, part)
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}")
        streams[part] = np.memmap(path, dtype=np.dtype(info.dtype), mode="r")

    starts = np.zeros_like(lengths)
    starts[1:] = np.cumsum(lengths, axis=0)[:-1]
    return streams, np.stack([starts, lengths], axis=-1)


def load_dataset(root: Path) -> DatasetInfo:
    """Read the sidecar written by :func:`build_dataset`."""
    meta_path = root / META_NAME
    if not meta_path.is_file():
        raise FileNotFoundError(f"no {META_NAME} in {root}; run `bloomery prepare` first")
    payload = json.loads(meta_path.read_text())
    return DatasetInfo(
        root=root,
        dtype=payload["dtype"],
        vocab_size=int(payload["vocab_size"]),
        # Defaulted, not indexed: every dataset prepared before masking existed
        # has no such key, and demanding one would break all of them at once.
        format=str(payload.get("format", FORMAT_PACKED)),
        dropped={str(k): int(v) for k, v in (payload.get("dropped") or {}).items()},
        splits=tuple(
            SplitInfo(name=s["name"], tokens=int(s["tokens"]), documents=int(s["documents"]))
            for s in payload["splits"]
        ),
    )


def open_split(info: DatasetInfo, name: str) -> np.memmap:
    """Memory-map a split for reading.

    Re-opened per access in the training loop rather than held open, which is
    the documented way to avoid a memory leak when sampling repeatedly from a
    memmap.
    """
    import numpy as np

    path = info.bin_path(name)
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    return np.memmap(path, dtype=np.dtype(info.dtype), mode="r")


def open_mask(info: DatasetInfo, name: str) -> np.memmap | None:
    """Memory-map a split's completion mask, or None for a plain corpus.

    Separate from :func:`open_split` rather than a flag on it, because the mask
    has its own dtype: one byte per token regardless of how wide the ids are.

    None means every token is trained on. That is what a corpus prepared without
    ``--chat`` means, and it is what makes a blend of a conversation corpus and a
    plain one work — the plain component simply has no mask.
    """
    import numpy as np

    if not info.masked:
        return None
    path = info.mask_path(name)
    if not path.is_file():
        raise FileNotFoundError(
            f"{info.root} says it holds conversations but {path.name} is missing, "
            "so there is no way to tell which tokens to learn from. Re-run "
            "`bloomery prepare --chat` for it."
        )
    return np.memmap(path, dtype=np.dtype(MASK_DTYPE), mode="r")
