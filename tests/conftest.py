# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared fixtures.

Tokenizer training and model construction are the slow parts, so anything that
can be shared across a module is built once. The suite has to stay fast enough
that people actually run it before pushing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch", reason="training tests need the [train] extra")

from bloomery.data import (  # noqa: E402
    build_dataset,
    eot_id,
    synthetic_documents,
    train_tokenizer,
)

# Enough documents that a small vocabulary has something to merge, few enough
# that training the tokenizer stays under a couple of seconds.
_DOCUMENTS = 800
_VOCAB = 400


@pytest.fixture(scope="session")
def documents() -> list[str]:
    return synthetic_documents(_DOCUMENTS, seed=0)


@pytest.fixture(scope="session")
def tokenizer(documents: list[str], tmp_path_factory: pytest.TempPathFactory) -> Any:
    out = tmp_path_factory.mktemp("tokenizer")
    return train_tokenizer(documents, vocab_size=_VOCAB, out_dir=out)


@pytest.fixture(scope="session")
def dataset(documents: list[str], tokenizer: Any, tmp_path_factory: pytest.TempPathFactory) -> Any:
    out = tmp_path_factory.mktemp("tokens")
    return build_dataset(
        documents,
        tokenizer,
        out_dir=out,
        eot=eot_id(tokenizer),
        val_fraction=0.1,
    )


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    path = tmp_path / "run"
    path.mkdir()
    return path
