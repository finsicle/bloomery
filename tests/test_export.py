# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for GGUF export.

The round trip is what matters. A GGUF that writes without error and holds the
wrong numbers is the failure this has to exclude: it loads, it generates, and
what comes out is noise with no indication that the file is at fault.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("gguf", reason="export tests need the [export] extra")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from gguf import GGMLQuantizationType as QT  # noqa: E402
from gguf import GGUFReader, quants  # noqa: E402

from bloomery.arch import spec_from_depth  # noqa: E402
from bloomery.export import (  # noqa: E402
    QUANTIZATIONS,
    ExportError,
    architecture_of,
    to_gguf,
    write_modelfile,
)
from bloomery.train.loop import build_model  # noqa: E402


@pytest.fixture(scope="module")
def model(tokenizer: Any) -> Any:
    """A real model, small enough to export in a moment."""
    spec = spec_from_depth(2, vocab=len(tokenizer), seq=64)
    return build_model(spec, eos_token_id=0, seed=0)


def read(path: Path) -> dict[str, Any]:
    reader = GGUFReader(str(path))
    return {tensor.name: tensor for tensor in reader.tensors}


def as_float(tensor: Any) -> np.ndarray:
    data = np.asarray(tensor.data)
    if tensor.tensor_type not in (QT.F32, QT.F16):
        data = quants.dequantize(data, tensor.tensor_type)
    return data.astype(np.float32)


class TestRoundTrip:
    """What was written must be what the model held."""

    @pytest.mark.parametrize(
        ("quantization", "share"),
        # As a fraction of each tensor's own spread. An absolute bound is
        # meaningless here: a freshly built model initialises to a standard
        # deviation around 0.02, so a tolerance of 1.0 would admit all zeros,
        # or the wrong tensor entirely.
        [("f16", 0.01), ("q8_0", 0.05), ("q4_0", 0.6)],
    )
    def test_weights_survive(
        self, model: Any, tokenizer: Any, tmp_path: Path, quantization: str, share: float
    ) -> None:
        out = tmp_path / f"{quantization}.gguf"
        to_gguf(model, tokenizer, out, quantization=quantization)
        written = read(out)
        state = model.state_dict()

        for source, name in (
            ("model.embed_tokens.weight", "token_embd.weight"),
            ("model.layers.0.self_attn.q_proj.weight", "blk.0.attn_q.weight"),
            ("model.layers.0.mlp.down_proj.weight", "blk.0.ffn_down.weight"),
            ("model.layers.1.mlp.gate_proj.weight", "blk.1.ffn_gate.weight"),
            ("model.norm.weight", "output_norm.weight"),
        ):
            expected = state[source].to(torch.float32).numpy()
            got = as_float(written[name]).reshape(expected.shape)

            spread = float(np.abs(expected).max())
            error = float(np.abs(got - expected).max())
            assert error <= share * spread, (
                f"{name} drifted by {error} at {quantization}, "
                f"more than {share:.0%} of its {spread:.4f} range"
            )
            # And it must actually be this tensor, not merely something small.
            # Correlation is undefined for a constant tensor — the norms
            # initialise to all ones — so those are checked for equality
            # instead, which is the stronger claim anyway.
            if float(expected.std()) > 0:
                assert np.corrcoef(got.ravel(), expected.ravel())[0, 1] > 0.9, (
                    f"{name} does not track the weights it came from at {quantization}"
                )
            else:
                assert np.allclose(got, expected), f"{name} changed at {quantization}"

    def test_the_output_head_is_written_though_it_is_tied(
        self, model: Any, tokenizer: Any, tmp_path: Path
    ) -> None:
        """The safetensors on disk hold no lm_head, and GGUF needs one anyway.

        Tying means the file has a single embedding matrix serving both ends.
        A GGUF without an output projection loads and produces nothing useful.
        """
        assert model.config.tie_word_embeddings
        out = tmp_path / "tied.gguf"
        to_gguf(model, tokenizer, out)
        written = read(out)

        assert "output.weight" in written
        embedding = as_float(written["token_embd.weight"])
        assert np.array_equal(as_float(written["output.weight"]), embedding)

    def test_normalisation_weights_are_not_quantized(
        self, model: Any, tokenizer: Any, tmp_path: Path
    ) -> None:
        """A small share of the bytes and a large share of the behaviour."""
        out = tmp_path / "q4.gguf"
        to_gguf(model, tokenizer, out, quantization="q4_0")
        written = read(out)
        assert written["output_norm.weight"].tensor_type == QT.F32
        assert written["blk.0.attn_norm.weight"].tensor_type == QT.F32
        # And the thing that should be quantized, was.
        assert written["blk.0.ffn_down.weight"].tensor_type == QT.Q4_0

    def test_quantizing_actually_shrinks_the_file(
        self, model: Any, tokenizer: Any, tmp_path: Path
    ) -> None:
        sizes = {}
        for quantization in QUANTIZATIONS:
            out = tmp_path / f"size-{quantization}.gguf"
            sizes[quantization] = to_gguf(
                model, tokenizer, out, quantization=quantization
            ).bytes_written
        assert sizes["f16"] > sizes["q8_0"] > sizes["q4_0"], sizes


class TestTokenizerTravels:
    """A GGUF carries its own vocabulary; a wrong one is silent nonsense."""

    def test_every_token_is_written_in_order(
        self, model: Any, tokenizer: Any, tmp_path: Path
    ) -> None:
        out = tmp_path / "vocab.gguf"
        to_gguf(model, tokenizer, out)
        field = GGUFReader(str(out)).fields["tokenizer.ggml.tokens"]

        expected = [
            token for token, _ in sorted(tokenizer.get_vocab().items(), key=lambda kv: kv[1])
        ]
        assert len(field.data) == len(expected)
        first = bytes(field.parts[field.data[0]]).decode("utf-8", "replace")
        assert first == expected[0]

    def test_the_merge_rules_are_written(self, model: Any, tokenizer: Any, tmp_path: Path) -> None:
        """BPE is the merges, not just the vocabulary.

        Found by loading an export into Ollama, not by any test here: llama.cpp
        refuses a file without them outright — "cannot find tokenizer merges in
        model file" — and every assertion about tensors, vocabulary and metadata
        passed while the artefact would not load at all.
        """
        out = tmp_path / "merges.gguf"
        to_gguf(model, tokenizer, out)
        fields = GGUFReader(str(out)).fields

        assert "tokenizer.ggml.merges" in fields
        assert len(fields["tokenizer.ggml.merges"].data) > 0

    def test_a_merge_is_a_space_joined_pair(
        self, model: Any, tokenizer: Any, tmp_path: Path
    ) -> None:
        """The shape llama.cpp expects, and the one the library has changed.

        tokenizers has emitted merges as both "a b" strings and ["a", "b"] pairs
        across versions; either is read, and only the joined form is written.
        """
        out = tmp_path / "shape.gguf"
        to_gguf(model, tokenizer, out)
        field = GGUFReader(str(out)).fields["tokenizer.ggml.merges"]
        first = bytes(field.parts[field.data[0]]).decode("utf-8", "replace")
        assert " " in first, first
        assert len(first.split(" ")) == 2, first

    def test_the_pre_tokenizer_is_declared(
        self, model: Any, tokenizer: Any, tmp_path: Path
    ) -> None:
        """The field llama.cpp's converter refuses to guess for our tokenizers.

        It picks this by hashing a tokenizer's output against a table of known
        models; ours is a new hash by construction. We know the answer, so we
        state it — and a GGUF without it cannot be tokenized correctly at all.
        """
        out = tmp_path / "pre.gguf"
        to_gguf(model, tokenizer, out)
        fields = GGUFReader(str(out)).fields
        assert "tokenizer.ggml.pre" in fields
        assert "tokenizer.ggml.model" in fields

    def test_a_chat_template_is_carried_across(
        self, model: Any, tokenizer: Any, tmp_path: Path
    ) -> None:
        """Otherwise a fine-tuned model is prompted in a shape it never saw."""
        tokenizer.chat_template = (
            "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}{% endfor %}"
        )
        try:
            out = tmp_path / "chat.gguf"
            to_gguf(model, tokenizer, out)
            assert "tokenizer.chat_template" in GGUFReader(str(out)).fields
        finally:
            tokenizer.chat_template = None


class TestMetadataLlamaCppWillAccept:
    """Fields whose wrong value produces a file that writes and will not load.

    The same class of failure as the missing merges: nothing here errors at
    write time, and llama.cpp refuses the result.
    """

    def test_the_architecture_is_the_one_llama_cpp_registers(
        self, model: Any, tokenizer: Any, tmp_path: Path
    ) -> None:
        """A mistral checkpoint must still say llama.

        Its tensor layout is Llama's, and llama.cpp has one registry entry for
        the family. Writing `general.architecture = "mistral"` produces
        "unknown model architecture" on load — there is no such entry.
        """
        out = tmp_path / "arch.gguf"
        model.config.model_type = "mistral"
        try:
            result = to_gguf(model, tokenizer, out)
        finally:
            model.config.model_type = "llama"

        field = GGUFReader(str(out)).fields["general.architecture"]
        written = bytes(field.parts[field.data[0]]).decode()
        assert written == "llama", written
        # Reported as what it actually is, which is a different question.
        assert result.architecture == "mistral"

    def test_the_token_list_matches_the_embedding_rows(
        self, model: Any, tokenizer: Any, tmp_path: Path
    ) -> None:
        """Fewer entries than rows and llama.cpp reads the two as disagreeing."""
        out = tmp_path / "rows.gguf"
        to_gguf(model, tokenizer, out)
        reader = GGUFReader(str(out))
        tokens = reader.fields["tokenizer.ggml.tokens"]
        embedding = next(t for t in reader.tensors if t.name == "token_embd.weight")

        assert len(tokens.data) == model.config.vocab_size
        assert len(tokens.data) == int(embedding.shape[1])

    def test_a_sparse_vocabulary_keeps_every_token_at_its_own_id(
        self, model: Any, tokenizer: Any, tmp_path: Path
    ) -> None:
        """Sorting only reproduces ids when they happen to be contiguous.

        Published checkpoints reserve blocks of ids. Sorting a vocabulary with a
        gap shifts every token after it by one, so the model tokenizes to ids
        that mean something else — with nothing reporting it.
        """

        class Sparse:
            """Ids 0, 1 and 9, as a reserved block would leave them."""

            chat_template = None
            all_special_tokens: list[str] = []
            is_fast = True
            backend_tokenizer = tokenizer.backend_tokenizer

            def get_vocab(self) -> dict[str, int]:
                return {"a": 0, "b": 1, "z": 9}

        model.config.vocab_size = 10
        original = model.get_input_embeddings().weight.shape[0]
        try:
            out = tmp_path / "sparse.gguf"
            to_gguf(model, tokenizer, out)  # a sanity export, real tokenizer
            from bloomery.export import _vocabulary

            tokens, _ = _vocabulary(Sparse(), Sparse)
        finally:
            model.config.vocab_size = original

        assert len(tokens) == 10
        assert tokens[0] == "a"
        assert tokens[1] == "b"
        assert tokens[9] == "z", "the gap shifted a token onto the wrong id"
        assert tokens[5].startswith("["), "an unnamed row still needs an entry"


class TestRefusals:
    def test_an_architecture_we_cannot_map_is_refused_by_name(self) -> None:
        """`adapt --method full` can produce one, and a wrong mapping is silent."""

        class NotLlama:
            model_type = "mamba"

        with pytest.raises(ExportError, match="mamba"):
            architecture_of(NotLlama())

    def test_a_model_with_no_output_layer_is_refused(self, tokenizer: Any, tmp_path: Path) -> None:
        """Refused, rather than reaching the user as a KeyError traceback."""

        class Hollow:
            config = type("C", (), {"model_type": "llama", "num_hidden_layers": 1})()

            def state_dict(self) -> dict[str, Any]:
                return {}

        with pytest.raises(ExportError, match="nothing to write"):
            to_gguf(Hollow(), tokenizer, tmp_path / "hollow.gguf")

    def test_an_unknown_quantization_is_refused(
        self, model: Any, tokenizer: Any, tmp_path: Path
    ) -> None:
        with pytest.raises(ExportError, match="unknown quantization"):
            to_gguf(model, tokenizer, tmp_path / "x.gguf", quantization="q3_k_xxl")


class TestModelfile:
    def test_it_points_at_the_gguf_beside_it(self, tokenizer: Any, tmp_path: Path) -> None:
        path = write_modelfile(tmp_path, tokenizer)
        text = path.read_text(encoding="utf-8")
        assert text.startswith("FROM ./model.gguf")

    def test_it_declares_a_stop_token(self, tokenizer: Any, tmp_path: Path) -> None:
        """Without one a runtime generates past the end of the reply."""
        text = write_modelfile(tmp_path, tokenizer).read_text(encoding="utf-8")
        assert "PARAMETER stop" in text
