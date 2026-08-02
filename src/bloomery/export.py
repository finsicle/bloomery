# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Writing a checkpoint out as GGUF, for llama.cpp and Ollama.

Written here rather than handed to llama.cpp's ``convert_hf_to_gguf.py``, which
cannot read the models this project exists to produce. That converter identifies
a tokenizer by hashing its output on a probe string and matching a table of known
models; a tokenizer bloomery trained is a new hash by construction, and the
converter refuses it rather than guessing.

Refusing is the right call for a general tool — the hash decides which
pre-tokenizer regex to apply, and applying the wrong one produces a model that
loads and generates rubbish. But bloomery already knows the answer the hash is
trying to recover: its tokenizers are byte-level BPE with the standard regex. So
the field is set directly, and the file is written here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

# Architectures whose tensor layout this writer knows. `adapt --method full` can
# produce a checkpoint of whatever base it was given, and those are not all
# Llama — mapping one wrongly yields a file that loads and generates nonsense,
# which is far worse than refusing to write it.
#
# Mistral is included because its tensor layout is Llama's; the name differs and
# nothing else does.
SUPPORTED_ARCHITECTURES = frozenset({"llama", "mistral"})

# What goes in `general.architecture`, which is not the same question as which
# checkpoints can be read. llama.cpp has one registry entry for this family and
# it is called llama — there is no `mistral` in it, so writing the model_type
# through produces a file that fails to load with "unknown model architecture".
GGUF_ARCHITECTURE = "llama"

# What `--quantize` accepts, mapped to the ggml type. Everything here is
# implemented in the `gguf` package's own quantizer, which is the reference one.
# The K-quants (q4_K, q5_K, q6_K) are deliberately absent: that package can read
# them but raises NotImplementedError when asked to write one, so they need
# llama.cpp's `llama-quantize` and the command says so.
QUANTIZATIONS: tuple[str, ...] = ("f16", "q8_0", "q4_0")

# Quantized formats work in blocks along the last dimension. A tensor whose last
# dimension is not a whole number of blocks cannot be quantized and is written at
# f16 instead — correct, slightly larger, and reported rather than silent.
_BLOCK = 32


class ExportError(RuntimeError):
    """A checkpoint could not be written as GGUF."""


@dataclass(frozen=True, slots=True)
class ExportResult:
    path: Path
    architecture: str
    quantization: str
    tensors: int
    parameters: int
    bytes_written: int
    context_length: int
    vocab_size: int
    # Tensors that could not take the requested quantization and fell back to
    # f16, by name. Empty for every shape this project produces.
    unquantized: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "architecture": self.architecture,
            "quantization": self.quantization,
            "tensors": self.tensors,
            "parameters": self.parameters,
            "bytes": self.bytes_written,
            "context_length": self.context_length,
            "vocab_size": self.vocab_size,
            "unquantized": list(self.unquantized),
        }


def architecture_of(config: Any) -> str:
    """The model family this config describes, refusing what cannot be mapped."""
    model_type = getattr(config, "model_type", None)
    if model_type not in SUPPORTED_ARCHITECTURES:
        known = ", ".join(sorted(SUPPORTED_ARCHITECTURES))
        raise ExportError(
            f"this checkpoint is a {model_type!r} model, and GGUF export understands "
            f"{known}.\n"
            "Writing it as one of those would produce a file that loads and generates "
            "nonsense, so it is refused instead. A checkpoint from `train` is always "
            "supported; one from `adapt` is whatever model it continued."
        )
    return str(model_type)


def _tensor_type(name: str, array: np.ndarray, quantization: str) -> Any:
    """The ggml type to store one tensor as.

    Normalisation weights stay at f32 and the token embedding at f16, which is
    what llama.cpp's own quantizer does: they are a small share of a model's
    bytes and a large share of its behaviour, so quantizing them costs quality
    for almost no size.
    """
    from gguf import GGMLQuantizationType as QT

    if array.ndim == 1:
        return QT.F32
    if quantization == "f16" or name == "token_embd.weight":
        return QT.F16
    if array.shape[-1] % _BLOCK:
        return QT.F16
    return {"q8_0": QT.Q8_0, "q4_0": QT.Q4_0}[quantization]


def to_gguf(
    model: Any,
    tokenizer: Any,
    out_path: Path,
    *,
    quantization: str = "f16",
) -> ExportResult:
    """Write a loaded model and its tokenizer as a single GGUF file."""
    import numpy as np
    import torch
    from gguf import GGMLQuantizationType as QT
    from gguf import GGUFWriter, get_tensor_name_map, quants
    from gguf.constants import MODEL_ARCH

    if quantization not in QUANTIZATIONS:
        raise ExportError(
            f"unknown quantization {quantization!r}; choose one of: {', '.join(QUANTIZATIONS)}"
        )

    config = model.config
    architecture = architecture_of(config)

    state = model.state_dict()
    # Tying means the file on disk holds no lm_head, but loading materialises it,
    # so it is usually here. Deriving it from the embedding when it is not keeps
    # both paths working: GGUF always wants an output projection.
    if "lm_head.weight" not in state:
        embedding = state.get("model.embed_tokens.weight")
        if embedding is None:
            # Refused rather than left to raise KeyError, which would reach the
            # user as a traceback past every handler this module's callers have.
            raise ExportError(
                "this checkpoint has neither an output projection nor a token "
                "embedding, so there is nothing to write as GGUF's output layer. "
                "It does not look like a causal language model."
            )
        state = dict(state)
        state["lm_head.weight"] = embedding

    # The official mapping rather than a hand-written table, so a rename upstream
    # is not something this has to notice.
    names = get_tensor_name_map(MODEL_ARCH.LLAMA, config.num_hidden_layers)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # GGUF_ARCHITECTURE, not the model_type: llama.cpp registers this family
    # under one name, and `architecture` is kept for what we report back.
    writer = GGUFWriter(str(out_path), GGUF_ARCHITECTURE)
    try:
        writer.add_block_count(config.num_hidden_layers)
        # The sequence length the model was trained at, not a capability of the
        # architecture. For a bloomery run this is often 512 or 1024, and a
        # runtime will hold it as a hard limit.
        writer.add_context_length(config.max_position_embeddings)
        writer.add_embedding_length(config.hidden_size)
        writer.add_feed_forward_length(config.intermediate_size)
        writer.add_head_count(config.num_attention_heads)
        writer.add_head_count_kv(getattr(config, "num_key_value_heads", config.num_attention_heads))
        writer.add_layer_norm_rms_eps(config.rms_norm_eps)
        writer.add_file_type(_file_type(quantization))
        _add_rope(writer, config)
        _add_tokenizer(writer, tokenizer, config)

        parameters = 0
        written = 0
        unquantized: list[str] = []
        for source, tensor in state.items():
            mapped = names.get_name(source.removesuffix(".weight"))
            if mapped is None:
                # Buffers such as rotary inverse frequencies are recomputed by
                # the runtime and have no GGUF name; skipping them is correct.
                continue
            name = f"{mapped}.weight"
            array = tensor.to(torch.float32).numpy()
            parameters += int(array.size)

            wanted = _tensor_type(name, array, quantization)
            # Only a shape that could not be blocked counts as a fallback. The
            # embedding is held at f16 by policy, not by failure.
            if (
                quantization != "f16"
                and array.ndim > 1
                and wanted is QT.F16
                and name != "token_embd.weight"
            ):
                unquantized.append(name)
            data = array.astype(np.float32) if wanted is QT.F32 else quants.quantize(array, wanted)
            writer.add_tensor(name, data, raw_dtype=wanted)
            written += 1

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()

    return ExportResult(
        path=out_path,
        architecture=architecture,
        quantization=quantization,
        tensors=written,
        parameters=parameters,
        bytes_written=out_path.stat().st_size,
        context_length=config.max_position_embeddings,
        vocab_size=config.vocab_size,
        unquantized=tuple(unquantized),
    )


MODELFILE = "Modelfile"
GGUF_NAME = "model.gguf"


def write_modelfile(directory: Path, tokenizer: Any, *, gguf_name: str = GGUF_NAME) -> Path:
    """Describe the exported model to Ollama.

    Written rather than run: `ollama create` needs Ollama installed, and an
    export that fails because a tool the user has not installed is missing would
    be a poor trade for a file that is six lines of text.

    The template is carried across when the checkpoint has one. Without it a
    fine-tuned model gets prompted as raw text, which is the same mismatch
    `bloomery chat` avoids by applying the template itself.
    """
    lines = [
        f"FROM ./{gguf_name}",
        "",
    ]
    template = getattr(tokenizer, "chat_template", None)
    if template:
        # Ollama's own template syntax is Go's, not Jinja, so the tokenizer's
        # template cannot be handed over directly. The GGUF carries it in its
        # metadata, where Ollama reads it; this note says where it went so the
        # absence of a TEMPLATE line does not read as an omission.
        lines += [
            "# This model's chat template travels inside the GGUF metadata,",
            "# which Ollama reads directly. Nothing to declare here.",
            "",
        ]

    eos = getattr(tokenizer, "eos_token", None)
    if eos:
        lines.append(f'PARAMETER stop "{eos}"')
    lines.append("")

    path = directory / MODELFILE
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _file_type(quantization: str) -> Any:
    from gguf import LlamaFileType

    return {
        "f16": LlamaFileType.MOSTLY_F16,
        "q8_0": LlamaFileType.MOSTLY_Q8_0,
        "q4_0": LlamaFileType.MOSTLY_Q4_0,
    }[quantization]


def _add_rope(writer: Any, config: Any) -> None:
    """Record the RoPE base, reading it rather than assuming the default.

    `to_llama_config` deliberately does not pass this, so it takes whatever the
    transformers default is — and transformers 5 renamed ``rope_theta`` to
    ``rope_parameters``. Hardcoding 10000.0 here would be right until it was not.
    """
    theta = getattr(config, "rope_theta", None)
    if theta is None:
        parameters = getattr(config, "rope_parameters", None)
        if isinstance(parameters, dict):
            theta = parameters.get("rope_theta")
    if theta is not None:
        writer.add_rope_freq_base(float(theta))


def _vocabulary(tokenizer: Any, config: Any) -> tuple[list[str], set[str]]:
    """The token list, indexed by id, sized to the embedding table.

    Built by position rather than by sorting the vocabulary, because sorting only
    reproduces the ids when they happen to be contiguous. Tokenizers this project
    trains are; ones that arrive with a published checkpoint need not be. They
    reserve blocks of ids, and they are routinely shorter than ``vocab_size`` —
    the embedding has rows nothing is named for.

    Sorting a sparse vocabulary silently shifts every token after the first gap
    by one, so the model tokenizes to ids that mean something else. Nothing
    errors; the output is simply wrong.
    """
    vocab = tokenizer.get_vocab()
    size = int(getattr(config, "vocab_size", 0)) or (max(vocab.values(), default=-1) + 1)

    by_id: dict[int, str] = {}
    for token, index in vocab.items():
        position = int(index)
        if 0 <= position < size:
            by_id[position] = token

    # An unnamed row still needs an entry, or the token list is shorter than the
    # embedding and llama.cpp reads the two as disagreeing about the vocabulary.
    tokens = [by_id.get(index, f"[UNUSED_{index}]") for index in range(size)]
    added = {token for token in getattr(tokenizer, "all_special_tokens", []) or [] if token}
    return tokens, added


def _merges(tokenizer: Any) -> list[str]:
    """The BPE merge rules, as GGUF wants them: one space-joined pair per entry.

    Read from the backend's own serialisation rather than from a file beside the
    checkpoint, because a tokenizer saved by ``save_pretrained`` may write
    ``tokenizer.json`` and nothing else — there is no ``merges.txt`` to find.

    The tokenizers library has emitted these as both ``"a b"`` strings and
    ``["a", "b"]`` pairs across versions, so both are accepted.
    """
    import json

    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        raise ExportError(
            "this tokenizer has no fast backend, so its merge rules cannot be read. "
            "GGUF needs them: a BPE vocabulary without merges cannot tokenize."
        )

    model = json.loads(backend.to_str()).get("model", {})
    raw = model.get("merges")
    if not raw:
        raise ExportError(
            "this tokenizer reports no BPE merges. A byte-level BPE tokenizer always "
            "has them, so the checkpoint's tokenizer.json is probably incomplete."
        )

    merges: list[str] = []
    for entry in raw:
        merges.append(" ".join(entry) if isinstance(entry, list) else str(entry))
    return merges


def _add_tokenizer(writer: Any, tokenizer: Any, config: Any) -> None:
    """Write the vocabulary, and say which pre-tokenizer produced it.

    ``gpt2`` and ``default`` are the byte-level BPE settings, which is what every
    tokenizer this project trains is — see the module docstring for why naming it
    outright is safe here and is not for a general converter.
    """
    tokens, added = _vocabulary(tokenizer, config)

    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("default")
    writer.add_token_list(tokens)
    # BPE is the merge rules, not just the vocabulary. Without them llama.cpp
    # refuses the file outright — "cannot find tokenizer merges in model file" —
    # and a writer that emits only the token list produces something every test
    # about tensors and vocabulary passes and no runtime will load.
    writer.add_token_merges(_merges(tokenizer))
    # 1 is a normal token, 3 is a control token. Marking the specials keeps a
    # runtime from printing them back to the user as text.
    writer.add_token_types([3 if token in added else 1 for token in tokens])

    for setter, value in (
        (writer.add_bos_token_id, getattr(config, "bos_token_id", None)),
        (writer.add_eos_token_id, getattr(config, "eos_token_id", None)),
        (writer.add_pad_token_id, getattr(config, "pad_token_id", None)),
    ):
        if value is not None:
            setter(int(value))

    template = getattr(tokenizer, "chat_template", None)
    if template:
        writer.add_chat_template(template)
