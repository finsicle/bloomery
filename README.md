# bloomery

**Train a language model from nothing, on your own machine.**

> ⚠️ **Status: pre-alpha.** This repository is a skeleton. There is no working
> code yet. Watch or star if you want to know when M1 lands — see
> [Roadmap](#roadmap) for what's being built and in what order.

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

Not yet. When it exists it will be one command, and it will resolve the right
PyTorch build for your GPU automatically.

## License

**Not yet decided.** No license file is present, which means default copyright
applies and no usage rights are granted for now. This will be resolved before
the first release.

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
