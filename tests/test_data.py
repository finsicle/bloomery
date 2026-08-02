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
    FORMAT_PACKED,
    FORMAT_SFT,
    adopt_tokenizer,
    build_dataset,
    count_bytes,
    dtype_for_vocab,
    eot_id,
    fingerprint,
    id_space,
    iter_documents,
    iter_examples,
    iter_preferences,
    load_dataset,
    looks_like_conversations,
    looks_like_preferences,
    open_mask,
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


class TestFingerprintWithoutAFastBackend:
    """A tokenizer with no fast backend has no canonical serialisation.

    Reachable: AutoTokenizer returns a slow tokenizer for any model shipping no
    fast implementation, and ``adopt_tokenizer`` takes whatever it is given.
    Hashing the vocabulary alone would be unsafe in the worst direction, because
    normalisation happens before the lookup — so two tokenizers can share a
    vocabulary and still assign different ids to the same text.
    """

    class Slow:
        """Only the surface ``fingerprint`` uses, and deliberately no backend."""

        def __init__(self, vocab: dict[str, int], *, lowercase: bool) -> None:
            self._vocab = vocab
            self._lowercase = lowercase

        def get_vocab(self) -> dict[str, int]:
            return dict(self._vocab)

        def __call__(self, text: str, **_: Any) -> dict[str, list[int]]:
            prepared = text.lower() if self._lowercase else text
            return {"input_ids": [self._vocab.get(ch, 0) for ch in prepared]}

    VOCAB = {"A": 1, "a": 2, "B": 3, "b": 4}

    def test_the_same_vocabulary_with_different_normalisation_differs(self) -> None:
        """The case that made the vocabulary-only fingerprint unsafe.

        A false match here does not raise. It lets the run start on ids that
        address the wrong symbols, which is what this comparison exists to stop.
        """
        folding = self.Slow(self.VOCAB, lowercase=True)
        faithful = self.Slow(self.VOCAB, lowercase=False)

        assert folding.get_vocab() == faithful.get_vocab()
        assert fingerprint(folding) != fingerprint(faithful)

    def test_two_identical_slow_tokenizers_still_match(self) -> None:
        """Failing safe must not mean failing always."""
        assert fingerprint(self.Slow(self.VOCAB, lowercase=True)) == fingerprint(
            self.Slow(self.VOCAB, lowercase=True)
        )

    def test_a_different_vocabulary_still_differs(self) -> None:
        other = dict(self.VOCAB) | {"C": 5}
        assert fingerprint(self.Slow(self.VOCAB, lowercase=False)) != fingerprint(
            self.Slow(other, lowercase=False)
        )

    def test_a_tokenizer_that_cannot_encode_does_not_crash_the_comparison(self) -> None:
        """A refusal to encode is itself a difference worth recording."""

        class Broken(TestFingerprintWithoutAFastBackend.Slow):
            def __call__(self, text: str, **_: Any) -> dict[str, list[int]]:
                raise RuntimeError("no")

        assert fingerprint(Broken(self.VOCAB, lowercase=False))
        assert fingerprint(Broken(self.VOCAB, lowercase=False)) != fingerprint(
            self.Slow(self.VOCAB, lowercase=False)
        )


# A minimal template with the shape every real one has: role markers, a turn
# terminator, and a cue that the assistant speaks next.
CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}<|end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


@pytest.fixture
def chat_tokenizer(documents: list[str], tmp_path: Path) -> Any:
    """Its own tokenizer, so setting a template cannot leak into other tests."""
    from bloomery.data import train_tokenizer

    tokenizer = train_tokenizer(documents[:400], vocab_size=400, out_dir=tmp_path / "chat-tok")
    tokenizer.chat_template = CHAT_TEMPLATE
    return tokenizer


def write_conversations(path: Path, count: int = 40) -> Path:
    records = [
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": f"question number {i}"},
                    {"role": "assistant", "content": f"answer number {i}"},
                    {"role": "user", "content": "say more"},
                    {"role": "assistant", "content": f"more about {i}"},
                ]
            }
        )
        for i in range(count)
    ]
    path.write_text("\n".join(records), encoding="utf-8")
    return path


class TestIterExamples:
    def test_reads_a_messages_list(self, tmp_path: Path) -> None:
        path = write_conversations(tmp_path / "c.jsonl", count=3)
        examples = list(iter_examples(path))
        assert len(examples) == 3
        assert examples[0].messages[0]["role"] == "user"
        assert examples[0].has_response

    def test_reads_a_prompt_and_completion_pair(self, tmp_path: Path) -> None:
        """The same thing with the roles implied, which is how many corpora ship."""
        path = tmp_path / "p.jsonl"
        path.write_text(json.dumps({"prompt": "why", "completion": "because"}), encoding="utf-8")
        (example,) = list(iter_examples(path))
        assert [turn["role"] for turn in example.messages] == ["user", "assistant"]
        assert example.messages[1]["content"] == "because"

    def test_a_record_with_no_reply_is_skipped(self, tmp_path: Path) -> None:
        """With completion-only masking there is nothing in it to learn from.

        Packing it would contribute tokens that are entirely masked, quietly
        diluting the corpus with examples that teach nothing.
        """
        path = tmp_path / "c.jsonl"
        path.write_text(
            json.dumps({"messages": [{"role": "user", "content": "hello?"}]}), encoding="utf-8"
        )
        assert list(iter_examples(path)) == []

    def test_an_unknown_role_is_refused_rather_than_guessed(self, tmp_path: Path) -> None:
        """Masking is decided by role, so guessing means training on the wrong half."""
        path = tmp_path / "c.jsonl"
        path.write_text(
            json.dumps({"messages": [{"role": "wizard", "content": "hi"}]}), encoding="utf-8"
        )
        assert list(iter_examples(path)) == []

    def test_conversations_are_recognised_as_such(self, tmp_path: Path) -> None:
        """So the plain path can say what the file is instead of finding nothing."""
        assert looks_like_conversations(write_conversations(tmp_path / "c.jsonl", count=2))
        plain = tmp_path / "plain.jsonl"
        plain.write_text(json.dumps({"text": "just prose"}), encoding="utf-8")
        assert not looks_like_conversations(plain)


def write_preferences(path: Path, count: int = 40) -> Path:
    records = [
        json.dumps(
            {
                "prompt": f"question number {i}",
                "chosen": f"a good answer to {i}",
                "rejected": f"a poor answer to {i}",
            }
        )
        for i in range(count)
    ]
    path.write_text("\n".join(records), encoding="utf-8")
    return path


class TestIterPreferences:
    def test_reads_a_string_prompt(self, tmp_path: Path) -> None:
        path = write_preferences(tmp_path / "p.jsonl", count=3)
        examples = list(iter_preferences(path))
        assert len(examples) == 3
        assert [turn["role"] for turn in examples[0].messages] == ["user"]
        assert examples[0].chosen.startswith("a good")

    def test_reads_a_messages_prompt(self, tmp_path: Path) -> None:
        """A preference pair can follow several turns, not only a single question."""
        path = tmp_path / "p.jsonl"
        path.write_text(
            json.dumps(
                {
                    "prompt": [
                        {"role": "user", "content": "hello"},
                        {"role": "assistant", "content": "hi"},
                        {"role": "user", "content": "and now?"},
                    ],
                    "chosen": "this",
                    "rejected": "that",
                }
            ),
            encoding="utf-8",
        )
        (example,) = list(iter_preferences(path))
        assert [turn["role"] for turn in example.messages] == ["user", "assistant", "user"]

    def test_a_pair_with_identical_answers_is_skipped(self, tmp_path: Path) -> None:
        """It carries no preference: the loss is constant and its gradient zero."""
        path = tmp_path / "p.jsonl"
        path.write_text(
            json.dumps({"prompt": "q", "chosen": "same", "rejected": "same"}), encoding="utf-8"
        )
        assert list(iter_preferences(path)) == []

    def test_a_missing_side_is_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "p.jsonl"
        path.write_text(json.dumps({"prompt": "q", "chosen": "only one"}), encoding="utf-8")
        assert list(iter_preferences(path)) == []

    def test_a_prompt_already_ending_in_a_reply_is_refused(self, tmp_path: Path) -> None:
        """Both answers are the assistant's turn, so the prompt must stop before one.

        Otherwise the record renders two assistant turns in a row, which no chat
        template describes and which puts the mask where neither answer is.
        """
        path = tmp_path / "p.jsonl"
        path.write_text(
            json.dumps(
                {
                    "prompt": [
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": "already answered"},
                    ],
                    "chosen": "this",
                    "rejected": "that",
                }
            ),
            encoding="utf-8",
        )
        assert list(iter_preferences(path)) == []

    def test_an_unknown_role_is_refused_rather_than_guessed(self, tmp_path: Path) -> None:
        path = tmp_path / "p.jsonl"
        path.write_text(
            json.dumps(
                {
                    "prompt": [{"role": "wizard", "content": "hi"}],
                    "chosen": "this",
                    "rejected": "that",
                }
            ),
            encoding="utf-8",
        )
        assert list(iter_preferences(path)) == []

    def test_preferences_are_recognised_as_such(self, tmp_path: Path) -> None:
        """So a path that cannot read them names the flag that can."""
        assert looks_like_preferences(write_preferences(tmp_path / "p.jsonl", count=2))
        assert not looks_like_preferences(write_conversations(tmp_path / "c.jsonl", count=2))
        assert not looks_like_conversations(write_preferences(tmp_path / "q.jsonl", count=2))


class TestPreferenceEncoding:
    """The prompt has to come out token-identical for both answers.

    DPO compares two answers under one condition. If the prompt tokenizes
    differently depending on what follows it, the comparison is between two
    different questions and the training signal is noise wearing a loss value.
    """

    def example(self, chosen: str = "the better answer", rejected: str = "worse") -> Any:
        from bloomery.data import PreferenceExample

        return PreferenceExample(
            ({"role": "user", "content": "a question"},), chosen=chosen, rejected=rejected
        )

    def test_the_prompt_is_a_real_prefix_of_both_renderings(self, chat_tokenizer: Any) -> None:
        """The property shards.py says may not be assumed, so it is checked.

        BPE merges across a join: tokenizing a prefix on its own does not tell
        you where that prefix ends inside the whole. This asserts the boundary
        found in character space lands where the tokens actually split.
        """
        from bloomery.data import eot_id
        from bloomery.data.shards import encode_preference

        eot = eot_id(chat_tokenizer)
        encoded, reason = encode_preference(self.example(), chat_tokenizer, eot=eot)
        assert encoded is not None, reason
        prompt, chosen, rejected = encoded

        for content, answer in (("the better answer", chosen), ("worse", rejected)):
            messages = [
                {"role": "user", "content": "a question"},
                {"role": "assistant", "content": content},
            ]
            text = chat_tokenizer.apply_chat_template(messages, tokenize=False)
            whole = list(chat_tokenizer(text, add_special_tokens=False)["input_ids"])
            assert whole == [*prompt, *answer[:-1]], "the boundary is not where the tokens split"
        assert chosen[-1] == eot and rejected[-1] == eot

    class Stub:
        """One token per character, and ">" marks where the assistant speaks.

        A stub rather than a trained tokenizer because the behaviour under test
        is a property of BPE in general, not of whichever merges a fixture
        happened to learn — and a fixture that does not happen to merge at the
        join would make the test pass by doing nothing.

        With ``contagious``, the two characters before the marker become one
        token whenever the answer begins with a vowel: the prompt tokenizes
        differently depending on what follows it, which is exactly what
        `shards.py:253` says may not be assumed away.
        """

        is_fast = True
        chat_template = "x"

        def __init__(self, contagious: bool = False) -> None:
            self.contagious = contagious

        def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
            text = "".join(
                (">" + m["content"] if m["role"] == "assistant" else m["content"]) for m in messages
            )
            return text + ">" if kwargs.get("add_generation_prompt") else text

        def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
            offsets = [(i, i + 1) for i in range(len(text))]
            marker = text.find(">")
            if self.contagious and marker > 1 and text[marker + 1 : marker + 2] in set("aeiou"):
                offsets = [*offsets[: marker - 2], (marker - 2, marker), *offsets[marker:]]
            return {
                "input_ids": [sum(ord(c) for c in text[a:b]) for a, b in offsets],
                "offset_mapping": offsets,
            }

    def test_a_prompt_that_retokenizes_differently_is_dropped(self) -> None:
        """The failure this whole path exists to catch.

        When the prompt's own tokens depend on which answer follows, the two
        answers are no longer being compared under one condition. Nothing about
        the resulting file looks wrong — the loss is a number, training runs, and
        what it converges to is noise.
        """
        from bloomery.data.shards import DROP_PROMPT_DIVERGED, encode_preference

        encoded, reason = encode_preference(
            self.example(chosen="alpha", rejected="beta"), self.Stub(contagious=True), eot=0
        )
        assert encoded is None
        assert reason == DROP_PROMPT_DIVERGED

    def test_a_stable_boundary_still_packs(self) -> None:
        """The guard must not reject every pair, only the ones that diverge."""
        from bloomery.data.shards import encode_preference

        encoded, reason = encode_preference(self.example("alpha", "beta"), self.Stub(), eot=7)
        assert encoded is not None, reason
        prompt, chosen, rejected = encoded
        assert prompt == [ord(c) for c in "a question>"]
        assert chosen == [*(ord(c) for c in "alpha"), 7]
        assert rejected == [*(ord(c) for c in "beta"), 7]


class TestPreferencePacking:
    def build(self, tokenizer: Any, tmp_path: Path, count: int = 40) -> Any:
        from bloomery.data import eot_id, iter_preferences
        from bloomery.data.shards import build_preference_dataset

        path = write_preferences(tmp_path / "p.jsonl", count=count)
        return build_preference_dataset(
            iter_preferences(path),
            tokenizer,
            out_dir=tmp_path / "packed",
            eot=eot_id(tokenizer),
            val_fraction=0.1,
        )

    def test_every_pair_round_trips(self, chat_tokenizer: Any, tmp_path: Path) -> None:
        """Read every example back and check it is the one that went in."""
        from bloomery.data import eot_id, iter_preferences, load_dataset
        from bloomery.data.shards import encode_preference, open_preference_split

        info = self.build(chat_tokenizer, tmp_path)
        assert load_dataset(tmp_path / "packed").format == "dpo"

        eot = eot_id(chat_tokenizer)
        wanted = [
            encode_preference(e, chat_tokenizer, eot=eot)[0]
            for e in iter_preferences(tmp_path / "p.jsonl")
        ]
        wanted = [w for w in wanted if w is not None]

        seen = []
        for split in ("train", "val"):
            streams, index = open_preference_split(info, split)
            assert len(index) == info.split(split).documents
            for row in index:
                seen.append(
                    tuple(
                        streams[part][start : start + length].tolist()
                        for part, (start, length) in zip(
                            ("prompt", "chosen", "rejected"), row, strict=True
                        )
                    )
                )

        assert len(seen) == len(wanted)
        # The two splits partition the corpus: every packed example is one that
        # went in, exactly once, with its three parts still together.
        assert sorted(seen) == sorted(tuple(map(list, w)) for w in wanted)

    def test_the_index_lands_on_the_right_boundaries(
        self, chat_tokenizer: Any, tmp_path: Path
    ) -> None:
        """An off-by-one here reads one example's answer against another's prompt."""
        from bloomery.data.shards import open_preference_split

        info = self.build(chat_tokenizer, tmp_path)
        streams, index = open_preference_split(info, "train")
        for column, part in enumerate(("prompt", "chosen", "rejected")):
            starts = index[:, column, 0]
            lengths = index[:, column, 1]
            assert starts[0] == 0
            # Back to back with no gaps and no overlap.
            assert list(starts[1:]) == list((starts + lengths)[:-1])
            assert starts[-1] + lengths[-1] == len(streams[part])

    def test_the_split_token_count_covers_all_three_streams(
        self, chat_tokenizer: Any, tmp_path: Path
    ) -> None:
        from bloomery.data.shards import open_preference_split

        info = self.build(chat_tokenizer, tmp_path)
        for split in ("train", "val"):
            streams, _ = open_preference_split(info, split)
            assert info.split(split).tokens == sum(len(s) for s in streams.values())

    def merging(self, chat_tokenizer: Any) -> Any:
        """The real tokenizer, except answers starting with "!" merge at the join.

        Contrived, and it has to be: a well-behaved template packs every pair.
        The accounting only matters for the pairs that cannot be split, so
        exercising it needs a tokenizer that fails to split some.
        """

        class Merging:
            is_fast = True
            chat_template = CHAT_TEMPLATE

            def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
                return str(chat_tokenizer.apply_chat_template(messages, **kwargs))

            def __call__(self, text: str, **kwargs: Any) -> Any:
                encoded = chat_tokenizer(text, **kwargs)
                if "!" not in text or "offset_mapping" not in encoded:
                    return encoded
                # Glue the token before the "!" to the one after it, so the
                # boundary token now spans both sides of the join.
                ids = list(encoded["input_ids"])
                offsets = list(encoded["offset_mapping"])
                cut = next(i for i, (a, b) in enumerate(offsets) if text[a:b].startswith("!"))
                merged = (offsets[cut - 1][0], offsets[cut][1])
                return {
                    "input_ids": [*ids[: cut - 1], ids[cut], *ids[cut + 1 :]],
                    "offset_mapping": [*offsets[: cut - 1], merged, *offsets[cut + 1 :]],
                }

            def __getattr__(self, name: str) -> Any:
                return getattr(chat_tokenizer, name)

            def __len__(self) -> int:
                return len(chat_tokenizer)

        return Merging()

    def test_dropped_pairs_are_recorded_rather_than_vanishing(
        self, chat_tokenizer: Any, tmp_path: Path
    ) -> None:
        """A corpus quietly smaller than its file is what you cannot debug later."""
        from bloomery.data import PreferenceExample, eot_id, load_dataset
        from bloomery.data.shards import DROP_PROMPT_DIVERGED, build_preference_dataset

        good = [
            PreferenceExample(
                ({"role": "user", "content": f"q{i}"},), chosen=f"good {i}", rejected=f"bad {i}"
            )
            for i in range(20)
        ]
        bad = [
            PreferenceExample(({"role": "user", "content": "q"},), chosen="!merged", rejected="ok")
            for _ in range(3)
        ]
        info = build_preference_dataset(
            [*good, *bad],
            self.merging(chat_tokenizer),
            out_dir=tmp_path / "packed",
            eot=eot_id(chat_tokenizer),
            val_fraction=0.1,
        )
        assert info.dropped == {DROP_PROMPT_DIVERGED: 3}
        assert sum(s.documents for s in info.splits) == 20
        # And it survives the sidecar, which is where it is read months later.
        assert load_dataset(tmp_path / "packed").dropped == {DROP_PROMPT_DIVERGED: 3}

    def test_a_corpus_of_nothing_usable_is_an_error(
        self, chat_tokenizer: Any, tmp_path: Path
    ) -> None:
        """Rather than an empty dataset that fails later, further from the cause."""
        from bloomery.data import PreferenceExample, SftError, eot_id
        from bloomery.data.shards import build_preference_dataset

        with pytest.raises(SftError, match="no preference pairs"):
            build_preference_dataset(
                [
                    PreferenceExample(
                        ({"role": "user", "content": "q"},), chosen="!merged", rejected="ok"
                    )
                ],
                self.merging(chat_tokenizer),
                out_dir=tmp_path / "packed",
                eot=eot_id(chat_tokenizer),
                val_fraction=0.0,
            )

    def test_a_packed_dataset_is_not_readable_as_preferences(
        self, dataset: Any, tmp_path: Path
    ) -> None:
        from bloomery.data.shards import open_preference_split

        with pytest.raises(ValueError, match="not preference pairs"):
            open_preference_split(dataset, "train")


class TestSftPacking:
    """The mask is the feature. If it is wrong the run still trains, and the
    model learns to repeat questions instead of answering them."""

    def _build(self, tmp_path: Path, tokenizer: Any, **kwargs: Any) -> Any:
        from bloomery.data import build_sft_dataset, eot_id

        path = write_conversations(tmp_path / "c.jsonl", count=kwargs.pop("count", 40))
        return build_sft_dataset(
            iter_examples(path),
            tokenizer,
            out_dir=tmp_path / "tokens",
            eot=eot_id(tokenizer),
            **kwargs,
        )

    def test_only_the_assistant_turns_are_trained_on(
        self, tmp_path: Path, chat_tokenizer: Any
    ) -> None:
        """Multi-turn on purpose: a single exchange hides boundary errors."""
        import numpy as np

        info = self._build(tmp_path, chat_tokenizer, val_fraction=0.1)
        ids = np.asarray(open_split(info, "train"))
        mask = np.asarray(open_mask(info, "train"))

        trained = chat_tokenizer.decode(ids[mask == 1])
        masked = chat_tokenizer.decode(ids[mask == 0])

        assert "answer number" in trained
        assert "more about" in trained, "the second reply must be trained on too"
        assert "question number" not in trained, "the prompt was not masked"
        assert "say more" not in trained
        assert "<|user|>" not in trained, "role markers belong to the prompt"
        assert "question number" in masked

    def test_the_reply_terminator_is_trained_on(self, tmp_path: Path, chat_tokenizer: Any) -> None:
        """Otherwise the model never learns to stop, and runs on past its answer."""
        import numpy as np

        info = self._build(tmp_path, chat_tokenizer, val_fraction=0.1)
        ids = np.asarray(open_split(info, "train"))
        mask = np.asarray(open_mask(info, "train"))
        assert "<|end|>" in chat_tokenizer.decode(ids[mask == 1])

    def test_the_mask_lines_up_with_the_tokens(self, tmp_path: Path, chat_tokenizer: Any) -> None:
        info = self._build(tmp_path, chat_tokenizer, val_fraction=0.1)
        for split in ("train", "val"):
            assert len(open_split(info, split)) == len(open_mask(info, split))

    def test_some_but_not_all_of_the_corpus_is_trained_on(
        self, tmp_path: Path, chat_tokenizer: Any
    ) -> None:
        """A share at either extreme means the boundaries were not found."""
        import numpy as np

        info = self._build(tmp_path, chat_tokenizer, val_fraction=0.1)
        share = float(np.asarray(open_mask(info, "train")).mean())
        assert 0.1 < share < 0.9, share

    def test_the_dataset_says_it_is_masked(self, tmp_path: Path, chat_tokenizer: Any) -> None:
        info = self._build(tmp_path, chat_tokenizer, val_fraction=0.1)
        assert info.format == FORMAT_SFT
        assert info.masked
        assert load_dataset(info.root).masked

    def test_a_tokenizer_with_no_chat_template_is_refused(
        self, tmp_path: Path, tokenizer: Any
    ) -> None:
        """Without one there is no telling how the model expects to be addressed."""
        from bloomery.data import SftError

        with pytest.raises(SftError, match="no chat template"):
            self._build(tmp_path, tokenizer, val_fraction=0.1)


class TestPackedDatasetsStillWork:
    """The pretraining path must be untouched by any of this."""

    def test_a_plain_dataset_has_no_mask(self, dataset: Any) -> None:
        assert not dataset.masked
        assert dataset.format == FORMAT_PACKED
        assert open_mask(dataset, "train") is None

    def test_meta_written_before_masking_existed_still_loads(self, tmp_path: Path) -> None:
        """Every dataset already on disk lacks the key. Demanding one breaks them all."""
        root = tmp_path / "old"
        root.mkdir()
        (root / "meta.json").write_text(
            json.dumps(
                {
                    "dtype": "uint16",
                    "vocab_size": 400,
                    "splits": [{"name": "train", "tokens": 10, "documents": 1}],
                }
            ),
            encoding="utf-8",
        )
        info = load_dataset(root)
        assert info.format == FORMAT_PACKED
        assert not info.masked
