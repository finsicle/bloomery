# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Corpus ingestion, tokenizer training and token packing."""

from __future__ import annotations

from bloomery.data.corpus import (
    Example,
    PreferenceExample,
    count_bytes,
    iter_documents,
    iter_examples,
    iter_preferences,
    looks_like_conversations,
    looks_like_preferences,
    synthetic_documents,
)
from bloomery.data.shards import (
    FORMAT_DPO,
    FORMAT_PACKED,
    FORMAT_SFT,
    DatasetInfo,
    SftError,
    SplitInfo,
    build_dataset,
    build_preference_dataset,
    build_sft_dataset,
    dtype_for_vocab,
    load_dataset,
    open_mask,
    open_preference_split,
    open_split,
)
from bloomery.data.tokenizer import (
    END_OF_TEXT,
    adopt_tokenizer,
    eot_id,
    fingerprint,
    id_space,
    load_tokenizer,
    train_tokenizer,
)

__all__ = [
    "END_OF_TEXT",
    "FORMAT_DPO",
    "FORMAT_PACKED",
    "FORMAT_SFT",
    "DatasetInfo",
    "Example",
    "PreferenceExample",
    "SftError",
    "SplitInfo",
    "adopt_tokenizer",
    "build_dataset",
    "build_preference_dataset",
    "build_sft_dataset",
    "count_bytes",
    "dtype_for_vocab",
    "eot_id",
    "fingerprint",
    "id_space",
    "iter_documents",
    "iter_examples",
    "iter_preferences",
    "looks_like_conversations",
    "looks_like_preferences",
    "load_dataset",
    "load_tokenizer",
    "open_mask",
    "open_preference_split",
    "open_split",
    "synthetic_documents",
    "train_tokenizer",
]
