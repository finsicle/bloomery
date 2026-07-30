# bloomery

**Train a language model from nothing, on your own machine.**

> ⚠️ **Status: pre-alpha.** Pretraining from scratch works today, end to end:
> tokenizer, packing, training, checkpoint, generation. No web UI yet, and no
> support for adapting existing models. See [Roadmap](#roadmap).

```bash
bloomery demo
```

Trains a language model from random weights on your machine in about a minute,
then talks to it. No download, no GPU required.

```
1/4 generated 6,000 synthetic documents
2/4 trained a 486-token vocabulary
3/4 packed 125,482 training tokens
4/4 training 3M params on mps (bfloat16)

  step  300  loss 0.680  15,828 tok/s

trained in 614,400 tokens · final loss 0.680 · val 0.668

samples (the model has never seen these prompts)

  Ana found a shy ball in the river. The bear wanted it too. They shared it and felt calm.
  One morning Ben walked to the forest. A clever bear was waiting there. Hugo felt proud.
  The sleepy cat lost its key. Ben looked in the forest and found it. The dog was excited.
```

That is a 3-million-parameter model that did not exist a minute earlier. It is a
toy trained on a synthetic grammar — note that it loses track of *who* the story
is about, which is exactly what 3M parameters buys you — but every step of the
path is the real one.

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
| **bloomery**           |    **✅**     |          M4           |    M4    |    M3     |

The tools with the capability are CLI and YAML. The tools with the interface
start from someone else's weights. Bloomery is from-scratch first, with
fine-tuning as the natural next step rather than the headline. Cells marked with
a milestone are not built yet — see [Roadmap](#roadmap).

### Working today

- **Pretrain from random init** — train your own byte-level BPE tokenizer, pack a
  corpus, pick a size with one dial, watch loss come down, then generate from the
  result. Checkpoints are standard Hugging Face directories.
- **Honest pre-flight estimates** — VRAM and token budget computed from *your*
  measured hardware, before you start rather than at hour six. `train` refuses a
  configuration that will not fit and names what to change.
- **Measured throughput** — `bench` runs real training steps instead of guessing
  from a peak-FLOPS table.
- **CPU thread caps** — `--cores` works identically on every platform.
- **Replay mixtures** — weighted, versioned dataset blends with per-component
  forgetting detection.

### Not built yet

- **Web UI and job queue** (M3), including per-job memory limits and GPU choice.
- **Continued pretraining, SFT and preference optimization** (M4).
- **Export to GGUF, Ollama, MLX** (M5).

## Hardware

| Platform                | Status                                                    |
| ----------------------- | --------------------------------------------------------- |
| macOS (Apple Silicon)   | training verified on an M1 via Metal, small models only    |
| CPU only                | training verified; toy models only                         |
| Linux + NVIDIA          | detection covered by tests; training not yet run on real hardware |
| Linux + AMD (ROCm)      | detection covered by tests; training not yet run on real hardware |
| Windows (WSL2)          | detection covered by tests; supported path on Windows      |
| Windows (native)        | best-effort                                                |

Being precise rather than optimistic: the GPU probe is exercised against captured
vendor output on every CI run across three operating systems, but only Metal and
CPU have had a real model trained on them so far. If you have NVIDIA or AMD
hardware, `bloomery doctor --json` in an issue is genuinely useful.

## Roadmap

| Milestone | Scope                                                             | Status |
| --------- | ----------------------------------------------------------------- | ------ |
| **M0**    | `bloomery doctor` — hardware probe, capability estimator, installers | done |
| **M1**    | Pretrain from scratch end to end, then generate from it            | done |
| **M2**    | Replay mixtures — weighted, versioned dataset blends               | done |
| **M3**    | Web UI, job queue, cancel/resume, resource limits                  | next |
| **M4**    | Continued pretraining and SFT on existing models                   |        |
| **M5**    | Export — GGUF, quantization, Ollama                                |        |
| **M6**    | AMD hardening on real hardware, Windows, evaluation                |        |

M2 came before the web UI because weighted replay is what makes "keep adding
datasets" work instead of quietly degrading the model — and it is the part no
comparable tool has.

## Install

Needs [uv](https://docs.astral.sh/uv/). The script installs it if it's missing.

```bash
git clone https://github.com/finsicle/bloomery.git
cd bloomery
./scripts/install.sh
```

On Windows, `.\scripts\install.ps1` — though WSL2 is the supported path.

This installs the core package only, which is a few megabytes, then immediately
reports what your machine can do. Adding PyTorch is a separate, much larger
step, and there is no point spending that download before you know which backend
you need:

```bash
uv pip install --torch-backend=auto -e ".[train]"
```

`--torch-backend=auto` inspects your CUDA driver, AMD GPU version or Intel GPU
and resolves the matching wheel index by itself. That one flag is most of what
made dropping Docker viable.

## Training something real

```bash
bloomery prepare --name mine --source ./my-text --vocab 8192
bloomery train   --data mine --depth 8 --steps 5000
bloomery chat    --run run1
```

`prepare` trains a byte-level BPE tokenizer on your corpus and packs it into
memory-mapped token shards. Do it once per corpus; every model you train on it
reuses the result.

`train` starts from random weights. **`--depth` is the only shape knob you need**
— width, head count, MLP size and learning rate are all derived from it, so
there are not twelve numbers to get wrong. `--size d12` picks a named preset
instead.

Checkpoints are ordinary Hugging Face directories. `AutoModelForCausalLM.from_pretrained`
loads them, which is the whole reason bloomery emits a Llama-architecture model
rather than inventing one — GGUF conversion, vLLM and Ollama all work without a
bespoke converter.

`train` estimates memory before it starts and refuses a configuration that
cannot fit, because an out-of-memory error arrives whenever the allocator
happens to hit the ceiling — which can be well into a run, after the tokenizer,
the packing and the model build have all succeeded:

```text
memory     ~14.4 GiB needed of 8.0 GiB  NVIDIA RTX 3070 — 8 GiB VRAM
error this configuration needs about 14.4 GiB but only 8.0 GiB is available.

try one of:
  --grad-checkpoint  (saves ~4.2 GiB, costs ~30% speed)
  --batch 8  (~7.9 GiB, fits)
  --depth 10  (the largest depth that fits as configured)
```

The estimate is deliberately conservative; `--force` starts anyway. Note that
gradient checkpointing is off by default, and the estimate accounts for that —
turning it on is usually the largest single saving available.

```bash
bloomery bench --size d12
```

Measures real training throughput on your machine and tells you what a
compute-optimal run would actually cost in hours.

## Keep adding datasets, without forgetting

Training on corpus A, then B, then C makes the model *worse* at A. That is
catastrophic forgetting, and it is why "just keep adding data" does not work
literally. The defence is replay: mix a share of the earlier data back into every
later run.

Replay only helps if the blend is a real object you can name, version and reuse.
So it is one:

```bash
bloomery mix create --name blend --add new:0.8 --replay old:0.2
bloomery train --mix blend --name run1 --depth 8 --steps 5000
```

Weights are raw numbers, normalised on use — `80/20` and `0.8/0.2` are the same
blend. Components you mark `--replay` are the ones that exist to stop forgetting,
and that share gets reported.

**Every component is evaluated on its own held-out split.** This is the part that
matters. A single aggregate validation loss is dominated by whichever component
carries the most weight, so it can fall while the oldest corpus in the blend
quietly degrades:

```text
  step 150  val loss 0.7004  ppl 2.01  new 0.696 old 0.742
  step 200  val loss 0.6913  ppl 2.00  new 0.687 old 0.731
  forgetting  old is 0.0180 above its best
```

That warning names the corpus that is getting worse while there is still time to
raise its weight. Doing so creates the next version rather than editing the
current one:

```bash
bloomery mix add blend --replay old:0.3 --note "raise replay after old regressed"
bloomery mix show blend          # weights, replay share, and full lineage
```

Versions are immutable on disk. `mix show` prints the whole chain, so "what was
run 7 actually trained on" has an answer.

Two guards worth knowing about:

- **Mismatched tokenizers are refused.** Each corpus is packed with its own
  tokenizer, so blending two of them means token id 4,211 refers to different
  symbols in each. That would not error — it would train, converge to nothing,
  and give no clue why. Components must agree on vocabulary size and tokenizer
  hash.
- **A blend with no replay component gets a warning**, because that is the
  configuration that forgets.

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
