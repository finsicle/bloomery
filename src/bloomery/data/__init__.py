# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Corpus ingestion, tokenizer training and token packing."""

from __future__ import annotations

from bloomery.data.corpus import (
    count_bytes,
    iter_documents,
    synthetic_documents,
)
from bloomery.data.shards import (
    DatasetInfo,
    SplitInfo,
    build_dataset,
    dtype_for_vocab,
    load_dataset,
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
    "DatasetInfo",
    "SplitInfo",
    "adopt_tokenizer",
    "build_dataset",
    "count_bytes",
    "dtype_for_vocab",
    "eot_id",
    "fingerprint",
    "id_space",
    "iter_documents",
    "load_dataset",
    "load_tokenizer",
    "open_split",
    "synthetic_documents",
    "train_tokenizer",
]
