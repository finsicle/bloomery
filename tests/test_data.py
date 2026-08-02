# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for corpus ingestion, tokenizer training and token packing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from bloomery.data import (
    END_OF_TEXT,
    adopt_tokenizer,
    build_dataset,
    count_bytes,
    dtype_for_vocab,
    eot_id,
    fingerprint,
    id_space,
    iter_documents,
    load_dataset,
    open_split,
    synthetic_documents,
)


class TestIterDocuments:
    def test_reads_a_text_file(self, tmp_path: Path) -> None:
        path = tmp_path / "a.txt"
        path.write_text("hello world")
        assert list(iter_documents(path)) == ["hello world"]

    def test_walks_a_directory_deterministically(self, tmp_path: Path) -> None:
        (tmp_path / "b.txt").write_text("second")
        (tmp_path / "a.txt").write_text("first")
        (tmp_path / "nested").mkdir()
        (tmp_path / "nested" / "c.md").write_text("third")
        # Sorted, so a re-run produces the same train/val split.
        assert list(iter_documents(tmp_path)) == ["first", "second", "third"]

    def test_jsonl_one_document_per_line(self, tmp_path: Path) -> None:
        path = tmp_path / "corpus.jsonl"
        path.write_text(json.dumps({"text": "one"}) + "\n" + json.dumps({"text": "two"}) + "\n")
        assert list(iter_documents(path)) == ["one", "two"]

    def test_jsonl_custom_field(self, tmp_path: Path) -> None:
        path = tmp_path / "c.jsonl"
        path.write_text(json.dumps({"body": "content"}) + "\n")
        assert list(iter_documents(path, jsonl_field="body")) == ["content"]

    def test_jsonl_skips_malformed_lines(self, tmp_path: Path) -> None:
        """One bad line must not cost the whole ingest."""
        path = tmp_path / "c.jsonl"
        path.write_text('{"text": "good"}\nnot json at all\n{"text": "also good"}\n')
        assert list(iter_documents(path)) == ["good", "also good"]

    def test_jsonl_bare_strings(self, tmp_path: Path) -> None:
        path = tmp_path / "c.jsonl"
        path.write_text('"just a string"\n')
        assert list(iter_documents(path)) == ["just a string"]

    def test_ignores_unknown_extensions(self, tmp_path: Path) -> None:
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        (tmp_path / "notes.txt").write_text("kept")
        assert list(iter_documents(tmp_path)) == ["kept"]

    def test_skips_whitespace_only_files(self, tmp_path: Path) -> None:
        (tmp_path / "blank.txt").write_text("   \n\n  ")
        assert list(iter_documents(tmp_path)) == []

    def test_survives_invalid_utf8(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.txt"
        path.write_bytes(b"before \xff\xfe after")
        documents = list(iter_documents(path))
        assert len(documents) == 1
        assert "before" in documents[0]


class TestCountBytes:
    def test_counts_only_ingestible_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("12345")
        (tmp_path / "b.bin").write_bytes(b"0" * 1000)
        assert count_bytes(tmp_path) == 5

    def test_single_file(self, tmp_path: Path) -> None:
        path = tmp_path / "a.txt"
        path.write_text("abc")
        assert count_bytes(path) == 3

    def test_non_ingestible_single_file_is_zero(self, tmp_path: Path) -> None:
        path = tmp_path / "a.bin"
        path.write_bytes(b"abc")
        assert count_bytes(path) == 0


class TestSyntheticDocuments:
    def test_deterministic_for_a_seed(self) -> None:
        assert synthetic_documents(20, seed=7) == synthetic_documents(20, seed=7)

    def test_seed_changes_output(self) -> None:
        assert synthetic_documents(20, seed=1) != synthetic_documents(20, seed=2)

    def test_count(self) -> None:
        assert len(synthetic_documents(53)) == 53

    def test_documents_look_like_prose(self) -> None:
        for document in synthetic_documents(10):
            assert document.endswith(".")
            assert len(document.split()) > 8


class TestDtypeForVocab:
    @pytest.mark.parametrize(
        ("vocab", "expected"),
        [
            (256, "uint16"),
            (8192, "uint16"),
            (65535, "uint16"),
            (65536, "uint32"),
            (131072, "uint32"),
        ],
    )
    def test_picks_narrowest_safe_dtype(self, vocab: int, expected: str) -> None:
        assert dtype_for_vocab(vocab) == expected


class TestTokenizer:
    def test_round_trips_text(self, tokenizer: Any) -> None:
        text = "Ana found a happy cat in the garden."
        decoded = tokenizer.decode(
            tokenizer(text, add_special_tokens=False)["input_ids"],
            skip_special_tokens=True,
        )
        assert decoded == text

    def test_byte_level_has_no_unknown_token(self, tokenizer: Any) -> None:
        """Anything must encode, including text the corpus never contained."""
        for text in ("日本語", "🔥 emoji", "\x00\x01 control"):
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            assert len(ids) > 0

    def test_eot_is_registered(self, tokenizer: Any) -> None:
        assert eot_id(tokenizer) == tokenizer.convert_tokens_to_ids(END_OF_TEXT)

    def test_reloads_from_disk(self, tokenizer: Any, tmp_path: Path) -> None:
        from bloomery.data import load_tokenizer

        out = tmp_path / "tok"
        tokenizer.save_pretrained(str(out))
        reloaded = load_tokenizer(out)
        assert len(reloaded) == len(tokenizer)

    def test_rejects_impossibly_small_vocab(self, tmp_path: Path) -> None:
        from bloomery.data import train_tokenizer

        with pytest.raises(ValueError, match="at least 257"):
            train_tokenizer(["text"], vocab_size=100, out_dir=tmp_path)


class TestBuildDataset:
    def test_writes_both_splits(self, dataset: Any) -> None:
        assert {s.name for s in dataset.splits} == {"train", "val"}
        assert dataset.split("train").tokens > 0
        assert dataset.split("val").tokens > 0

    def test_meta_round_trips(self, dataset: Any) -> None:
        reloaded = load_dataset(dataset.root)
        assert reloaded.dtype == dataset.dtype
        assert reloaded.vocab_size == dataset.vocab_size
        assert reloaded.total_tokens == dataset.total_tokens

    def test_token_ids_are_in_range(self, dataset: Any) -> None:
        data = open_split(dataset, "train")
        assert int(data.max()) < dataset.vocab_size

    def test_separator_appears_between_documents(self, dataset: Any, tokenizer: Any) -> None:
        data = open_split(dataset, "train")
        separators = int((data[:] == eot_id(tokenizer)).sum())
        # One per document in this split.
        assert separators == dataset.split("train").documents

    def test_val_split_is_strided_not_a_tail_slice(
        self, documents: list[str], tokenizer: Any, tmp_path: Path
    ) -> None:
        """A corpus ordered by source must not get an unrepresentative val set."""
        info = build_dataset(
            documents,
            tokenizer,
            out_dir=tmp_path / "strided",
            eot=eot_id(tokenizer),
            val_fraction=0.1,
        )
        expected = round(len(documents) * 0.1)
        assert abs(info.split("val").documents - expected) <= 2

    def test_val_fraction_zero_puts_everything_in_train(
        self, documents: list[str], tokenizer: Any, tmp_path: Path
    ) -> None:
        info = build_dataset(
            documents,
            tokenizer,
            out_dir=tmp_path / "novalue",
            eot=eot_id(tokenizer),
            val_fraction=0.0,
        )
        assert info.split("val").tokens == 0
        assert info.split("train").documents == len(documents)

    def test_empty_corpus_is_an_error(self, tokenizer: Any, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="no training tokens"):
            build_dataset([], tokenizer, out_dir=tmp_path / "e", eot=0, val_fraction=0.0)

    def test_corpus_too_small_for_a_val_split(self, tokenizer: Any, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="too small"):
            build_dataset(
                ["only one document"],
                tokenizer,
                out_dir=tmp_path / "small",
                eot=0,
                val_fraction=0.001,
            )

    def test_rejects_out_of_range_fraction(self, tokenizer: Any, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="val_fraction"):
            build_dataset(["a"], tokenizer, out_dir=tmp_path / "x", eot=0, val_fraction=1.0)

    def test_load_dataset_without_meta(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="prepare"):
            load_dataset(tmp_path)

    def test_unknown_split(self, dataset: Any) -> None:
        with pytest.raises(KeyError):
            dataset.split("test")


class TestIdSpace:
    """Sizing the packed array from the wrong number wraps ids in silence."""

    def test_counts_added_tokens(self, documents: list[str], tmp_path: Path) -> None:
        """Trains its own tokenizer: add_tokens mutates in place, and the shared
        fixture is session-scoped, so doing this to it would leak the extra
        tokens into every later test that touches it."""
        from bloomery.data import train_tokenizer

        local = train_tokenizer(documents[:200], vocab_size=300, out_dir=tmp_path / "local")
        base = local.vocab_size
        local.add_tokens(["<|extra_one|>", "<|extra_two|>"])

        assert len(local) > base
        assert id_space(local) == len(local)

    def test_an_id_above_the_entry_count_still_widens_the_array(self) -> None:
        """Counting entries is not the same as bounding ids.

        A tokenizer can hold a few thousand entries with one of them at id
        128,255 — several published models reserve a block of special tokens at
        the top. Both counts then say uint16, and that token wraps.
        """

        class Sparse:
            vocab_size = 32_000

            def __len__(self) -> int:
                return 32_001

            def get_vocab(self) -> dict[str, int]:
                return {"a": 0, "<|reserved|>": 128_255}

        assert id_space(Sparse()) == 128_256
        assert dtype_for_vocab(id_space(Sparse())) == "uint32"

    def test_a_vocabulary_just_under_the_limit_is_not_packed_as_uint16(self) -> None:
        """The concrete failure: base vocab below 65,536, added tokens above it.

        Sizing from ``vocab_size`` alone gives uint16, and every added token's id
        comes back as whatever it collides with, with nothing reporting it.
        """

        class Tokenizer:
            vocab_size = 65_530

            def __len__(self) -> int:
                return 65_540

        assert dtype_for_vocab(Tokenizer.vocab_size) == "uint16"
        assert dtype_for_vocab(id_space(Tokenizer())) == "uint32"


class TestAdoptTokenizer:
    """Packing a corpus for a model that already exists."""

    def test_adopting_reproduces_the_source_tokenizer(self, tmp_path: Path, tokenizer: Any) -> None:
        """What makes a corpus for an existing model possible at all."""
        source = tmp_path / "source"
        tokenizer.save_pretrained(str(source))
        original = tokenizer

        adopted = adopt_tokenizer(source, tmp_path / "adopted")
        adopted_dir = tmp_path / "adopted"

        assert (adopted_dir / "tokenizer.json").is_file()
        assert fingerprint(adopted) == fingerprint(original)
        assert adopted("hello world")["input_ids"] == original("hello world")["input_ids"]

    def test_a_source_that_is_not_a_tokenizer_is_refused(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="could not load a tokenizer"):
            adopt_tokenizer(empty, tmp_path / "out")


class TestFingerprint:
    """Identity by behaviour, not by the bytes a file happens to have.

    The mixture guard hashes ``tokenizer.json`` directly, which is right when
    both sides were written by this project. It is not right when one side is
    someone else's model: saving re-serialises, and a re-serialised file need
    not be byte-identical to the original.
    """

    def test_survives_a_save_and_reload(self, tmp_path: Path, tokenizer: Any) -> None:
        from bloomery.data import load_tokenizer

        before = fingerprint(tokenizer)
        tokenizer.save_pretrained(str(tmp_path / "saved"))
        assert fingerprint(load_tokenizer(tmp_path / "saved")) == before

    def test_differs_for_a_different_vocabulary(self, tmp_path: Path, documents: Any) -> None:
        from bloomery.data import train_tokenizer

        small = train_tokenizer(documents, vocab_size=300, out_dir=tmp_path / "a")
        large = train_tokenizer(documents, vocab_size=500, out_dir=tmp_path / "b")
        assert fingerprint(small) != fingerprint(large)
