# bloomery

**Train a language model from nothing, on your own machine.**

> ⚠️ **Status: pre-alpha.** `bloomery doctor` works today. Nothing trains yet.
> See [Roadmap](#roadmap) for what's being built and in what order.

---

## What this is

A *bloomery* is the earliest kind of iron furnace — the small, buildable one
that turned raw ore into usable metal long before industrial blast furnaces
existed. Its output is called a **bloom**: a rough mass of iron, ready to be
refined.

That's the idea here. Point it at a folder of text and get back a language
model that didn't exist before. Then keep going — continued pretraining,
instruction tuning, export — all from the same place, all on hardware you own.

## Why another training tool

There are good local fine-tuning GUIs. There are good from-scratch pretraining
libraries. There is nothing that is both.

|                        | From scratch | Continued pretraining | SFT / RL | Local GUI |
| ---------------------- | :----------: | :-------------------: | :------: | :-------: |
| Transformer Lab        |      ❌       |           ~           |    ✅     |     ✅     |
| Unsloth Studio         |      ❌       |           ❌           |    ✅     |     ✅     |
| LLaMA-Factory          |      ❌       |           ✅           |    ✅     |     ✅     |
| Oumi                   |      ✅       |           ✅           |    ✅     |     ❌     |
| nanochat               |      ✅       |           ✅           |    ✅     |     ❌     |
| **bloomery**           |    **✅**     |         **✅**         |  **✅**   |   **✅**   |

The tools with the capability are CLI and YAML. The tools with the interface
start from someone else's weights. Bloomery is from-scratch first, with
fine-tuning as the natural next step rather than the headline.

## Planned capabilities

- **Pretrain from random init** — train your own tokenizer, shard a corpus,
  pick a size, watch loss come down. A ~10M-parameter model on a laptop in
  minutes; GPT-2 class on a single consumer GPU overnight.
- **Continued pretraining** on an existing base model, with **replay mixtures**
  as a first-class versioned object so adding a new corpus doesn't silently
  degrade everything the model already knew.
- **Instruction tuning and preference optimization** — SFT, DPO, ORPO, KTO,
  full or LoRA/QLoRA.
- **Honest pre-flight estimates** — VRAM, wall clock, and token budget computed
  from *your* detected hardware, shown before you start, not discovered at hour
  six.
- **Explicit resource allocation** — choose which GPUs, how many CPU cores, how
  much RAM a job may use.
- **Export anywhere** — safetensors, GGUF with quantization, Ollama, MLX.

## Hardware

| Platform                 | Status  |
| ------------------------ | ------- |
| Linux + NVIDIA           | primary |
| Linux + AMD (ROCm)       | primary |
| Windows + NVIDIA (WSL2)  | planned |
| Windows + AMD (WSL2)     | planned |
| macOS (Apple Silicon)    | planned, small models only |
| CPU only                 | toy models only |

## Roadmap

| Milestone | Scope                                                          |
| --------- | -------------------------------------------------------------- |
| **M0**    | `bloomery doctor` — hardware probe and install script           |
| **M1**    | Vertical slice: train a tiny model from scratch, then chat with it |
| **M2**    | Job queue, cancel, resume, resource limits, crash recovery      |
| **M3**    | Data layer — ingest, tokenize, shard, mixture builder           |
| **M4**    | Continued pretraining and SFT engine                            |
| **M5**    | Export — GGUF, quantization, Ollama, Hugging Face               |
| **M6**    | AMD hardening, Windows, macOS MPS                               |
| **M7**    | Evaluation and inference playground                             |

## Install

Needs [uv](https://docs.astral.sh/uv/). The script installs it if it's missing.

```bash
git clone https://github.com/aswinsam/bloomery.git
cd bloomery
./scripts/install.sh
```

On Windows, `.\scripts\install.ps1` — though WSL2 is the supported path.

This installs the core package only, which is a few megabytes, then immediately
reports what your machine can do. Adding PyTorch is a separate, much larger
step, and there is no point spending that download before you know which backend
you need:

```bash
uv pip install --torch-backend=auto -e ".[train,serve]"
```

`--torch-backend=auto` inspects your CUDA driver, AMD GPU version or Intel GPU
and resolves the matching wheel index by itself. That one flag is most of what
made dropping Docker viable.

## `bloomery doctor`

The first thing that works, and the thing to paste into a bug report.

```
╭─ gpus ───────────────────────────────────────────────────────────────────────╮
│ #   vendor  device              memory   arch      source                    │
│ 0   amd     Radeon RX 7900 XTX  24 GiB   gfx1100   amd-smi                   │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ what this machine can train ────────────────────────────────────────────────╮
│ budget  Radeon RX 7900 XTX — 24 GiB VRAM                                     │
│                                                                              │
│ pretrain from scratch  up to 1B (1.07B)  ·  needs ~21.5B tokens              │
│ fine-tune with QLoRA   up to 7B (6.98B)                                      │
│                                                                              │
│ model               params  tokens     step     Full     LoRA  QLoRA         │
│ tiny                    5M    105M   32×512  1.6 GiB  1.5 GiB  1.5 GiB       │
│ GPT-2 small class     124M    2.5B   8×1024  4.7 GiB  3.0 GiB  2.8 GiB       │
│ nanochat d26          595M   11.9B   4×2048   13 GiB  5.1 GiB  4.2 GiB       │
│ 1B                   1.07B   21.5B   2×2048   21 GiB  5.8 GiB  4.2 GiB       │
│ 7B                   6.98B  139.6B   1×4096  117 GiB   20 GiB  9.2 GiB       │
╰──────────────────────────────────────────────────────────────────────────────╯
```

It detects OS (including WSL2), CPU, RAM, free disk, and GPUs across NVIDIA,
AMD, Apple and Intel — then estimates, from *your* measured VRAM, which model
sizes are actually trainable. Memory estimates are conservative and assume
AdamW, bf16 and gradient checkpointing.

Two things it does that are easy to skip and expensive to omit:

- **Reads the PCI bus directly.** So "you have no GPU" and "you have a GPU whose
  driver isn't loaded" produce different messages. The second is the state most
  new AMD users are actually in.
- **Checks the installed PyTorch against the detected hardware.** A CPU-only
  wheel on a machine with two H100s is a silent, expensive mistake.

`--json` gives machine-readable output. Exit status is non-zero when something
would stop a training run, so it works as a check in a script.

## Development

```bash
uv pip install -e ".[dev]"
pytest && ruff check . && ruff format --check . && mypy src
```

The GPU vendor parsers are fixture-driven, against captured `nvidia-smi`,
`amd-smi` and `rocm-smi` output plus synthetic sysfs trees. That is deliberate:
the AMD path has to be testable and CI-covered on machines with no AMD hardware,
which is most of them.

## License

**GNU Affero General Public License v3.0 or later** — see [LICENSE](LICENSE).

In plain terms: use it, modify it, run it, sell services around it. If you
distribute a modified version — or offer one to users over a network — you have
to make your source available under the same terms. You cannot build a
proprietary product on top of bloomery and keep it closed.

Contributions are accepted under a [Contributor License Agreement](CLA.md),
which keeps open the possibility of offering commercial licenses to
organisations that cannot use AGPL software. You keep the copyright in your
work.

### A note on model licenses

Whatever license bloomery ends up under governs *the tool only*. It does not
govern models you produce with it. If you continue-pretrain or fine-tune an
existing base model, that base model's terms (Llama Community License, Gemma
Terms of Use, and so on) flow through to your result independently. Check them.

## Credits

Bloomery stands on:

- [nanochat](https://github.com/karpathy/nanochat) (MIT) — the from-scratch
  reference this project's pretraining path is modelled on
- [PyTorch](https://pytorch.org) (BSD-3)
- [transformers](https://github.com/huggingface/transformers),
  [peft](https://github.com/huggingface/peft),
  [trl](https://github.com/huggingface/trl),
  [accelerate](https://github.com/huggingface/accelerate) (Apache-2.0)
- [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT) — GGUF export
