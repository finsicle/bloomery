# Contributing to bloomery

Thanks for looking. The project is pre-alpha — there is no working code yet, so
the most useful contributions right now are design feedback and hardware
reports.

## Before you write code

Open an issue first. The architecture is still settling and a short discussion
will save you from building something that gets rewritten next week.

## Clean-room rule

Some of the closest projects to this one — Transformer Lab, Unsloth Studio —
are AGPL-3.0 licensed. Reading them to understand a problem is fine and
encouraged. **Copying code from them is not.** Even a modest amount of pasted
AGPL code would force a license change on this entire project.

If you contribute code, it must be your own work or come from a permissively
licensed source (MIT, BSD, Apache-2.0) that you credit in the pull request.

## Hardware reports are contributions

The hardest part of this project is not the training code. It is making
`import torch` work correctly across NVIDIA, AMD, Apple and CPU-only machines
on three operating systems.

If you have hardware — especially an AMD card, an unusual driver version, or a
Windows setup — opening an issue with the output of `bloomery doctor` (once it
exists) is genuinely valuable.

## Development setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:aswinsam/bloomery.git
cd bloomery
uv venv
uv pip install -e ".[dev]"
```

To run the training tests you also need PyTorch, resolved for your GPU:

```bash
uv pip install --torch-backend=auto -e ".[dev,train]"
```

## Checks

```bash
ruff check .
ruff format --check .
mypy src
pytest
```

## Commit messages

Plain imperative subject lines, no trailers.

```
add rocm device enumeration to doctor
```

## Licensing of contributions

Bloomery is **AGPL-3.0-or-later**. Contributions are accepted under a
[Contributor License Agreement](CLA.md) — you keep your copyright, and the
project gains the right to also offer commercial licenses to organisations that
cannot use AGPL software.

Signing happens on your first pull request: a bot comments with a link and you
confirm there. Nothing to print or email.

If you'd rather not sign, issues, reviews, hardware reports and design
discussion are all still very welcome and require no agreement.

New source files should carry the two-line header:

```python
# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
```
