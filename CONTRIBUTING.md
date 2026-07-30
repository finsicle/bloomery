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

Once the training path exists, add the framework layer with the backend
resolved for your GPU:

```bash
uv pip install --torch-backend=auto -e ".[dev,train,serve]"
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

The project license is not yet decided. Until it is, please do not submit
substantial code contributions — the terms you would be contributing under
aren't defined yet, which isn't fair to you. Issues and discussion are very
welcome in the meantime.
