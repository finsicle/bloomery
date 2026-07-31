# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Command line entry point."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, NoReturn

import typer
from rich.console import Console

from bloomery import __version__, paths
from bloomery.capability import LADDER_BY_KEY, Method, assess, format_params
from bloomery.probe import probe_host_report
from bloomery.probe.types import GIB
from bloomery.render import render_report

app = typer.Typer(
    name="bloomery",
    help="Train a language model from nothing, on your own machine.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

SIZES = ", ".join(LADDER_BY_KEY)


def _quiet_transformers() -> None:
    """Silence progress bars and advisory warnings.

    Called before any transformers import work in a CLI path. Its default
    chattiness buries our own output, and a shard-writing progress bar for a
    3 MB model is noise.
    """
    import os

    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:  # noqa: BLE001 - never fail a command over logging setup
        pass


def _die(message: str) -> NoReturn:
    """Report a user-facing error and stop.

    Annotated NoReturn so that everything after a call is provably unreachable.
    Several handlers bind a name inside a `try` and read it afterwards; their
    correctness depends on this raising, and `-> None` gave neither mypy nor a
    reader that guarantee.
    """
    console.print(f"[bold red]error[/bold red] {message}")
    raise typer.Exit(code=1)


# --------------------------------------------------------------------------- #
# version / doctor
# --------------------------------------------------------------------------- #


@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(f"bloomery {__version__}")


@app.command()
def doctor(
    as_json: bool = typer.Option(
        False, "--json", help="Emit machine-readable output instead of a report."
    ),
    skip_torch: bool = typer.Option(
        False, "--no-torch", help="Skip probing the installed PyTorch."
    ),
) -> None:
    """Report what this machine is, and what it can train.

    Exits non-zero if anything would stop a training run, so it can be used as a
    check in a script. Paste the output of `bloomery doctor --json` into a bug
    report.
    """
    report = probe_host_report(include_torch=not skip_torch)
    capability = assess(report)

    if as_json:
        payload = {
            "bloomery_version": __version__,
            "home": str(paths.home()),
            "probe": report.to_dict(),
            "capability": {
                "budget": {
                    "bytes": capability.budget.total,
                    "source": capability.budget.source,
                    "shardable_devices": capability.budget.shardable_devices,
                    "shardable_bytes": capability.budget.shardable_total,
                },
                "models": [
                    {
                        "key": row.spec.key,
                        "label": row.spec.label,
                        "method": row.method.value,
                        "params": row.spec.params,
                        "tokens_chinchilla": row.tokens,
                        "required_bytes": row.required,
                        "fits": row.fits,
                        "available": row.available_here,
                        "unavailable_reason": row.unavailable_reason or None,
                    }
                    for row in capability.rows
                ],
                "largest_from_scratch": _spec_summary(capability, Method.FULL),
                "largest_qlora": _spec_summary(capability, Method.QLORA),
            },
        }
        console.print_json(json.dumps(payload))
    else:
        render_report(console, report, capability)

    if any(issue.level == "error" for issue in report.issues):
        raise typer.Exit(code=1)


def _spec_summary(capability: Any, method: Method) -> dict[str, Any] | None:
    row = capability.largest_fitting(method)
    if row is None:
        return None
    return {
        "key": row.spec.key,
        "label": row.spec.label,
        "params": row.spec.params,
        "required_bytes": row.required,
    }


# --------------------------------------------------------------------------- #
# prepare
# --------------------------------------------------------------------------- #


@app.command()
def prepare(
    name: str = typer.Option(..., "--name", "-n", help="Name for this dataset."),
    source: Path | None = typer.Option(
        None, "--source", "-s", help="File or directory of .txt/.md/.jsonl documents."
    ),
    synthetic: int = typer.Option(
        0, "--synthetic", help="Instead of a source, generate N synthetic documents."
    ),
    vocab: int = typer.Option(8192, "--vocab", help="Tokenizer vocabulary size."),
    val_fraction: float = typer.Option(
        0.01, "--val-fraction", help="Fraction of documents held out for validation."
    ),
) -> None:
    """Train a tokenizer and pack a corpus into token shards.

    Do this once per corpus. Training runs then read the packed tokens directly,
    so the cost of tokenizing is paid a single time however many models you
    train on it.
    """
    _quiet_transformers()
    from bloomery.data import (
        build_dataset,
        count_bytes,
        eot_id,
        iter_documents,
        synthetic_documents,
        train_tokenizer,
    )

    if bool(source) == bool(synthetic):
        _die("give exactly one of --source or --synthetic")

    if source is not None:
        if not source.exists():
            _die(f"{source} does not exist")
        size = count_bytes(source)
        if size == 0:
            _die(f"no .txt, .md or .jsonl files found under {source}")
        console.print(f"reading [bold]{size / 1e6:.1f} MB[/bold] from {source}")
        documents = list(iter_documents(source))
    else:
        console.print(f"generating [bold]{synthetic}[/bold] synthetic documents")
        documents = synthetic_documents(synthetic)

    if not documents:
        _die("no documents found")

    tok_path = paths.tokenizer_dir(name)
    tokens_path = paths.tokens_dir(name)

    with console.status(f"training a {vocab}-token vocabulary over {len(documents)} documents"):
        tokenizer = train_tokenizer(documents, vocab_size=vocab, out_dir=tok_path)
    actual_vocab = len(tokenizer)
    console.print(f"tokenizer  [bold]{actual_vocab}[/bold] tokens → {tok_path}")
    if actual_vocab < vocab:
        console.print(
            f"[dim]corpus supported only {actual_vocab} merges, fewer than the {vocab} "
            "requested; that is normal for a small corpus[/dim]"
        )

    with console.status("tokenizing and packing"):
        try:
            info = build_dataset(
                documents,
                tokenizer,
                out_dir=tokens_path,
                eot=eot_id(tokenizer),
                val_fraction=val_fraction,
            )
        except ValueError as exc:
            _die(str(exc))

    train_split = info.split("train")
    val_split = info.split("val")
    console.print(
        f"tokens     [bold]{train_split.tokens:,}[/bold] train / "
        f"{val_split.tokens:,} val  ({info.dtype}) → {tokens_path}"
    )
    console.print(f"\nnext:  [bold]bloomery train --data {name} --name run1[/bold]")


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #


@app.command()
def train(
    data: str | None = typer.Option(
        None, "--data", "-d", help="Single dataset name from `prepare`."
    ),
    mix: str | None = typer.Option(
        None, "--mix", "-m", help="Mixture name from `mix create`. Use instead of --data."
    ),
    mix_version: int | None = typer.Option(
        None, "--mix-version", help="Pin a mixture version. Defaults to the newest."
    ),
    name: str = typer.Option("run1", "--name", "-n", help="Name for this run."),
    depth: int | None = typer.Option(
        None, "--depth", help="Layer count. Everything else is derived from it."
    ),
    size: str | None = typer.Option(None, "--size", help=f"Named preset: {SIZES}."),
    steps: int = typer.Option(500, "--steps", help="Optimizer steps."),
    batch: int = typer.Option(16, "--batch", help="Sequences per step."),
    seq: int = typer.Option(256, "--seq", help="Tokens per sequence."),
    grad_accum: int = typer.Option(1, "--grad-accum", help="Micro-batches per step."),
    lr: float | None = typer.Option(None, "--lr", help="Peak LR. Default derives from width."),
    eval_every: int = typer.Option(100, "--eval-every", help="Steps between evaluations."),
    save_every: int = typer.Option(
        0, "--save-every", help="Steps between snapshots (0 = end only)."
    ),
    cores: int | None = typer.Option(None, "--cores", help="Cap CPU threads."),
    device: str | None = typer.Option(None, "--device", help="Force cuda / mps / cpu."),
    grad_checkpoint: bool = typer.Option(
        False, "--grad-checkpoint", help="Trade compute for memory."
    ),
    resume: bool = typer.Option(False, "--resume", help="Continue from the latest checkpoint."),
    force: bool = typer.Option(
        False, "--force", help="Start even if the memory estimate says it will not fit."
    ),
    seed: int = typer.Option(1337, "--seed"),
) -> None:
    """Train a model from random initialisation.

    No pretrained weights are involved: the tokenizer is yours, the data is
    yours, and the parameters start from the initialiser.
    """
    _quiet_transformers()
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    from bloomery import mixture as mix_mod
    from bloomery.arch import actual_param_count, resolve_spec
    from bloomery.capability import check_fit
    from bloomery.probe import probe_host_report
    from bloomery.train import checkpoint as ckpt
    from bloomery.train.device import choose, thread_limit
    from bloomery.train.loop import TrainConfig
    from bloomery.train.loop import train as run_training
    from bloomery.train.mixing import resolve as resolve_mixture

    if bool(data) == bool(mix):
        _die("give exactly one of --data or --mix")

    # A single dataset is a one-component blend, so the rest of the flow has a
    # single path rather than a branch that can drift.
    try:
        blend = mix_mod.load(mix, mix_version) if mix else mix_mod.single(data or "")
        resolved = resolve_mixture(blend)
    except mix_mod.MixtureError as exc:
        _die(str(exc))
    except FileNotFoundError as exc:
        _die(str(exc))

    tokenizer = resolved.tokenizer

    try:
        spec = resolve_spec(size=size, depth=depth, vocab=len(tokenizer), seq=seq, batch=batch)
    except ValueError as exc:
        _die(str(exc))

    thread_limit(cores)
    choice = choose(device)

    run_path = paths.run_dir(name)
    resume_from = None
    if resume:
        candidate = ckpt.checkpoint_dir(run_path)
        if not ckpt.is_resumable(candidate):
            _die(f"nothing resumable at {candidate}")
        resume_from = candidate

    config = TrainConfig(
        steps=steps,
        batch=batch,
        seq=seq,
        grad_accum=grad_accum,
        lr=lr,
        eval_every=eval_every,
        save_every=save_every,
        seed=seed,
        gradient_checkpointing=grad_checkpoint,
    )

    train_tokens = sum(i.split("train").tokens for i in resolved.datasets.values())
    console.print(
        f"model      [bold]{spec.label}[/bold]  "
        f"{spec.layers}L × {spec.hidden}d × {spec.heads}h  "
        f"{format_params(actual_param_count(spec))} params"
    )
    console.print(f"device     {choice.label()}  [dim]{choice.reason}[/dim]")
    console.print(
        f"data       {train_tokens:,} train tokens  "
        f"[dim]{config.tokens_per_step:,} tokens/step[/dim]"
    )
    if len(blend.components) > 1:
        console.print(f"mixture    [bold]{blend.describe()}[/bold]")
        replay = blend.replay_share()
        if replay == 0:
            console.print(
                "[yellow]           no component is marked as replay — "
                "an older corpus can degrade unnoticed[/yellow]"
            )
        else:
            console.print(f"[dim]           replay share {replay * 100:.0f}%[/dim]")
    # Pre-flight memory check. An out-of-memory error arrives whenever the
    # allocator happens to hit the ceiling, which can be well into a run after
    # the tokenizer, the packing and the model build have all succeeded. The
    # estimate is cheap and deliberately conservative.
    fit = check_fit(
        probe_host_report(include_torch=False),
        spec,
        batch=batch,
        seq=seq,
        gradient_checkpointing=grad_checkpoint,
        device_type=choice.type,
    )
    console.print(
        f"memory     ~{fit.required / GIB:.1f} GiB needed of "
        f"{fit.budget.total / GIB:.1f} GiB  [dim]{fit.budget.source}[/dim]"
    )
    if not fit.fits:
        lines = [
            f"this configuration needs about {fit.required / GIB:.1f} GiB but only "
            f"{fit.budget.total / GIB:.1f} GiB is available ({fit.budget.source}).",
        ]
        if fit.suggestions():
            lines.append("\ntry one of:")
            lines.extend(f"  {option}" for option in fit.suggestions())
        lines.append(
            "\nThe estimate is conservative and assumes AdamW in bf16. "
            "Pass --force to start anyway."
        )
        if not force:
            _die("\n".join(lines))
        console.print("[yellow]--force given; starting despite the estimate[/yellow]")
    elif fit.spare < 0.25:
        console.print(
            f"[yellow]           only {fit.spare:.0%} spare — "
            "an out-of-memory error is plausible[/yellow]"
        )
    console.print()

    columns = (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    )
    with Progress(*columns, console=console) as progress:
        task = progress.add_task("training", total=steps, completed=0)

        def on_event(event: dict[str, Any]) -> None:
            if event["event"] == "step":
                progress.update(
                    task,
                    completed=event["step"],
                    description=(
                        f"loss {event['loss']:.3f}  {event['tokens_per_second']:,.0f} tok/s"
                    ),
                )
            elif event["event"] == "eval":
                line = (
                    f"  [dim]step {event['step']}[/dim]  "
                    f"val loss [bold]{event['val_loss']:.4f}[/bold]  "
                    f"ppl {event['perplexity']}"
                )
                per = event.get("per_component") or {}
                if len(per) > 1:
                    line += (
                        "  [dim]"
                        + " ".join(f"{k} {v:.3f}" for k, v in sorted(per.items()))
                        + "[/dim]"
                    )
                progress.console.print(line)
                # The point of per-component evaluation: name the corpus that is
                # getting worse, while there is still time to raise its weight.
                for component, delta in (event.get("regressed") or {}).items():
                    progress.console.print(
                        f"  [yellow]forgetting[/yellow]  {component} is {delta:.4f} above its best"
                    )

        try:
            result = run_training(
                spec=spec,
                datasets=resolved.datasets,
                mixture=blend,
                tokenizer=tokenizer,
                run_dir=run_path,
                config=config,
                choice=choice,
                eos_token_id=resolved.eos_token_id,
                resume_from=resume_from,
                on_event=on_event,
            )
        except ValueError as exc:
            progress.stop()
            _die(str(exc))
        except KeyboardInterrupt:
            progress.stop()
            console.print("\n[yellow]interrupted[/yellow] — no checkpoint written this step")
            raise typer.Exit(code=130) from None

    console.print()
    console.print(f"final loss [bold]{result.final_loss:.4f}[/bold]")
    if result.best_val_loss is not None:
        console.print(f"best val   [bold]{result.best_val_loss:.4f}[/bold]")
    console.print(f"throughput {result.tokens_per_second:,.0f} tok/s")
    if len(result.per_component_loss) > 1:
        console.print()
        console.print("per component")
        for component, loss in sorted(result.per_component_loss.items()):
            flag = (
                f"  [yellow]+{result.regressed[component]:.4f} over best[/yellow]"
                if component in result.regressed
                else ""
            )
            console.print(f"  {component:<24} {loss:.4f}{flag}")
        if result.regressed:
            console.print(
                "\n[yellow]Some components ended worse than their best.[/yellow] "
                "Raise their weight, or mark them as replay:\n"
                f"  [bold]bloomery mix add {blend.name} "
                f"--replay {next(iter(result.regressed))}:0.2[/bold]"
            )
    console.print(f"checkpoint {result.checkpoint}")
    console.print(f"\nnext:  [bold]bloomery chat --run {name}[/bold]")


# --------------------------------------------------------------------------- #
# mix
# --------------------------------------------------------------------------- #

mix_app = typer.Typer(
    name="mix",
    help="Build and version weighted dataset blends.",
    no_args_is_help=True,
)
app.add_typer(mix_app)


def _parse_components(domain: list[str], replay: list[str]) -> list[Any]:
    """Turn ``NAME:WEIGHT`` strings into components, tagged by role."""
    from bloomery.mixture import ROLE_DOMAIN, ROLE_REPLAY, Component, parse_spec

    components = []
    for specs, role in ((domain, ROLE_DOMAIN), (replay, ROLE_REPLAY)):
        for raw in specs:
            dataset, weight = parse_spec(raw)
            components.append(Component(dataset=dataset, weight=weight, role=role))
    return components


@mix_app.command("create")
def mix_create(
    name: str = typer.Option(..., "--name", "-n", help="Name for this mixture."),
    add: list[str] = typer.Option(
        [], "--add", help="Component as NAME:WEIGHT, e.g. domain:0.6. Repeatable."
    ),
    replay: list[str] = typer.Option(
        [], "--replay", help="Same, but marked as replay data. Repeatable."
    ),
    note: str = typer.Option("", "--note", help="Why this blend exists."),
) -> None:
    """Create version 1 of a blend.

    Weights are raw numbers, normalised on use — 60/15/25 and 0.6/0.15/0.25 are
    the same blend. Mark the components that exist to prevent forgetting with
    --replay; that share is what gets reported and warned about.
    """
    from bloomery import mixture as mix_mod

    if not add and not replay:
        _die("give at least one --add or --replay component")

    try:
        components = _parse_components(add, replay)
        blend = mix_mod.create(name, components, note=note)
        path = mix_mod.save(blend)
    except mix_mod.MixtureError as exc:
        _die(str(exc))

    console.print(f"created    [bold]{blend.describe()}[/bold]")
    console.print(f"replay     {blend.replay_share() * 100:.0f}%")
    console.print(f"saved      {path}")
    console.print(f"\nnext:  [bold]bloomery train --mix {name} --name run1[/bold]")


@mix_app.command("list")
def mix_list() -> None:
    """List every blend, newest version of each."""
    from bloomery import mixture as mix_mod

    blends = mix_mod.list_all()
    if not blends:
        console.print("No mixtures yet. Create one with [bold]bloomery mix create[/bold].")
        return
    for blend in blends:
        versions = mix_mod.versions(blend.name)
        console.print(
            f"[bold]{blend.name}[/bold]  v{blend.version}  "
            f"[dim]({len(versions)} version(s), replay "
            f"{blend.replay_share() * 100:.0f}%)[/dim]"
        )
        console.print(f"  {blend.describe()}")


@mix_app.command("show")
def mix_show(
    name: str = typer.Argument(..., help="Mixture name."),
    version: int | None = typer.Option(None, "--version", help="Defaults to newest."),
) -> None:
    """Show a blend, its weights and its full version history."""
    from bloomery import mixture as mix_mod

    try:
        blend = mix_mod.load(name, version)
    except mix_mod.MixtureError as exc:
        _die(str(exc))

    weights = blend.weights()
    console.print(f"[bold]{blend.name}[/bold] v{blend.version}")
    if blend.note:
        console.print(f"[dim]{blend.note}[/dim]")
    console.print()
    for component in sorted(blend.components, key=lambda c: -c.weight):
        role = "[cyan]replay[/cyan]" if component.role == "replay" else "domain"
        console.print(
            f"  {component.dataset:<24} {weights[component.dataset] * 100:>5.1f}%  "
            f"[dim](raw {component.weight:g})[/dim]  {role}"
        )
    console.print(f"\n  replay share  {blend.replay_share() * 100:.0f}%")

    chain = mix_mod.lineage(blend)
    if len(chain) > 1:
        console.print("\nlineage")
        for entry in chain:
            marker = "→" if entry.version == blend.version else " "
            suffix = f"  [dim]{entry.note}[/dim]" if entry.note else ""
            console.print(f"  {marker} v{entry.version}  {entry.describe()}{suffix}")


@mix_app.command("add")
def mix_add(
    name: str = typer.Argument(..., help="Mixture name."),
    add: list[str] = typer.Option([], "--add", help="NAME:WEIGHT to add or reweight."),
    replay: list[str] = typer.Option([], "--replay", help="Same, marked as replay."),
    remove: list[str] = typer.Option([], "--remove", help="Dataset name to drop."),
    note: str = typer.Option("", "--note", help="Why this version exists."),
) -> None:
    """Create the next version of a blend.

    Versions are immutable, so this never edits the current one — it writes a new
    version that records the current one as its parent. That chain is what lets
    you say exactly what a past run was trained on.
    """
    from bloomery import mixture as mix_mod

    if not add and not replay and not remove:
        _die("give at least one --add, --replay or --remove")

    try:
        blend = mix_mod.load(name)
        for dataset in remove:
            blend = blend.without_component(dataset, note=note)
        for component in _parse_components(add, replay):
            blend = blend.with_component(
                component.dataset, component.weight, role=component.role, note=note
            )
        # Collapse the intermediate steps into one new version, so adding three
        # components produces v2 rather than v2, v3 and v4.
        current = mix_mod.load(name)
        final = mix_mod.Mixture(
            name=blend.name,
            version=current.version + 1,
            components=blend.components,
            parent_version=current.version,
            note=note,
            created_at=blend.created_at,
        )
        path = mix_mod.save(final)
    except mix_mod.MixtureError as exc:
        _die(str(exc))

    console.print(f"created    [bold]{final.describe()}[/bold]")
    console.print(f"parent     v{final.parent_version}")
    console.print(f"replay     {final.replay_share() * 100:.0f}%")
    console.print(f"saved      {path}")


# --------------------------------------------------------------------------- #
# chat
# --------------------------------------------------------------------------- #


@app.command()
def chat(
    run: str | None = typer.Option(None, "--run", "-r", help="Run name to load."),
    checkpoint: Path | None = typer.Option(
        None, "--checkpoint", "-c", help="Path to a checkpoint directory."
    ),
    prompt: str | None = typer.Option(
        None, "--prompt", "-p", help="One-shot prompt. Omit for an interactive session."
    ),
    max_new_tokens: int = typer.Option(80, "--max-new-tokens"),
    temperature: float = typer.Option(0.8, "--temperature", help="0 for greedy decoding."),
    top_p: float = typer.Option(0.95, "--top-p"),
    top_k: int = typer.Option(40, "--top-k"),
    seed: int | None = typer.Option(None, "--seed"),
    device: str | None = typer.Option(None, "--device"),
) -> None:
    """Generate text from a checkpoint.

    A loss curve tells you the optimiser worked. This tells you whether the model
    learned anything you wanted.
    """
    _quiet_transformers()
    from bloomery.generate import SamplingConfig, complete, load
    from bloomery.train import checkpoint as ckpt

    if bool(run) == bool(checkpoint):
        _die("give exactly one of --run or --checkpoint")

    target = checkpoint if checkpoint else ckpt.checkpoint_dir(paths.run_dir(run or ""))
    try:
        with console.status(f"loading {target}"):
            loaded = load(target, device=device)
    except (FileNotFoundError, OSError) as exc:
        _die(str(exc))

    config = SamplingConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
    )
    console.print(
        f"[dim]{format_params(loaded.params)} params on {loaded.device}"
        f"{' · greedy' if config.greedy else f' · T={temperature}'}[/dim]\n"
    )

    if prompt is not None:
        console.print(f"[bold]{prompt}[/bold]", end="")
        console.print(complete(loaded, prompt, config))
        return

    console.print("[dim]Type a prompt. Ctrl-C or an empty line to quit.[/dim]\n")
    while True:
        try:
            line = console.input("[bold cyan]> [/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print()
            return
        if not line:
            return
        console.print(complete(loaded, line, config).strip() or "[dim](nothing)[/dim]")
        console.print()


# --------------------------------------------------------------------------- #
# bench
# --------------------------------------------------------------------------- #


@app.command()
def bench(
    size: str | None = typer.Option(None, "--size", help=f"Named preset: {SIZES}."),
    depth: int | None = typer.Option(None, "--depth", help="Layer count."),
    vocab: int = typer.Option(8192, "--vocab"),
    batch: int | None = typer.Option(None, "--batch"),
    seq: int | None = typer.Option(None, "--seq"),
    steps: int = typer.Option(10, "--steps", help="Timed steps, after warmup."),
    cores: int | None = typer.Option(None, "--cores"),
    device: str | None = typer.Option(None, "--device"),
    grad_checkpoint: bool = typer.Option(False, "--grad-checkpoint"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Measure real training throughput on this machine.

    `doctor` estimates memory but refuses to estimate time, because a
    peak-FLOPS table cannot know about your cooling, your driver or what else is
    running. This runs actual training steps and measures.
    """
    _quiet_transformers()
    from bloomery import bench as bench_mod
    from bloomery.arch import resolve_spec
    from bloomery.train.device import choose, thread_limit

    if not size and not depth:
        size = "d12"

    try:
        spec = resolve_spec(size=size, depth=depth, vocab=vocab, seq=seq, batch=batch)
    except ValueError as exc:
        _die(str(exc))

    thread_limit(cores)
    choice = choose(device)

    with console.status(f"benchmarking {spec.label} on {choice.label()}"):
        try:
            result = bench_mod.run(
                spec,
                choice,
                batch=batch,
                seq=seq,
                steps=steps,
                gradient_checkpointing=grad_checkpoint,
            )
        except RuntimeError as exc:
            _die(f"{exc}\n\nTry a smaller --batch or --seq, or --grad-checkpoint.")

    if as_json:
        console.print_json(json.dumps(bench_mod.to_dict(result)))
        return

    console.print(f"model        [bold]{spec.label}[/bold]  {format_params(result.params)} params")
    console.print(f"device       {result.device} ({result.dtype})")
    console.print(
        f"step         {result.batch}×{result.seq} = {result.batch * result.seq:,} tokens"
    )
    console.print(f"throughput   [bold]{result.tokens_per_second:,.0f} tok/s[/bold]")
    console.print(f"per step     {result.seconds_per_step * 1000:,.0f} ms")
    if result.peak_memory:
        console.print(f"peak memory  {result.peak_memory / 1024**3:.2f} GiB")
    console.print()
    hours = result.hours_for_chinchilla()
    console.print(
        f"[dim]At this rate a compute-optimal run for this size "
        f"({result.params * 20 / 1e9:.1f}B tokens) would take "
        f"[/dim][bold]{hours:,.1f} hours[/bold][dim] on this machine.[/dim]"
    )


# --------------------------------------------------------------------------- #
# demo
# --------------------------------------------------------------------------- #


@app.command()
def demo(
    steps: int = typer.Option(300, "--steps"),
    depth: int = typer.Option(4, "--depth"),
    documents: int = typer.Option(6000, "--documents"),
    device: str | None = typer.Option(None, "--device"),
    keep: bool = typer.Option(False, "--keep", help="Keep the dataset and run afterwards."),
) -> None:
    """Train a tiny model from scratch, end to end, in about a minute.

    Uses a small synthetic corpus so it needs no download and no GPU. Its point
    is to prove the whole path works on your machine — tokenizer, packing,
    training, checkpoint, generation — before you spend hours on real data.
    """
    _quiet_transformers()
    import shutil

    from bloomery import mixture as mix_mod
    from bloomery.arch import actual_param_count, resolve_spec
    from bloomery.data import (
        build_dataset,
        eot_id,
        synthetic_documents,
        train_tokenizer,
    )
    from bloomery.generate import SamplingConfig, complete, load
    from bloomery.train.device import choose
    from bloomery.train.loop import TrainConfig
    from bloomery.train.loop import train as run_training

    name = "demo"
    docs = synthetic_documents(documents)
    console.print(f"[bold]1/4[/bold] generated {len(docs):,} synthetic documents")

    with console.status("training tokenizer"):
        tokenizer = train_tokenizer(docs, vocab_size=1024, out_dir=paths.tokenizer_dir(name))
    console.print(f"[bold]2/4[/bold] trained a {len(tokenizer)}-token vocabulary")

    with console.status("packing tokens"):
        info = build_dataset(
            docs,
            tokenizer,
            out_dir=paths.tokens_dir(name),
            eot=eot_id(tokenizer),
            val_fraction=0.05,
        )
    console.print(f"[bold]3/4[/bold] packed {info.split('train').tokens:,} training tokens")

    spec = resolve_spec(depth=depth, vocab=len(tokenizer), seq=128, batch=16)
    choice = choose(device)
    console.print(
        f"[bold]4/4[/bold] training {format_params(actual_param_count(spec))} params "
        f"on {choice.label()}"
    )
    console.print()

    result = run_training(
        spec=spec,
        datasets={name: info},
        mixture=mix_mod.single(name),
        tokenizer=tokenizer,
        run_dir=paths.run_dir(name),
        config=TrainConfig(
            steps=steps,
            batch=16,
            seq=128,
            eval_every=max(1, steps // 3),
            eval_batches=10,
            # Drive the cadence here rather than filtering events downstream: a
            # modulo on the step number only lines up when it happens to be a
            # multiple of log_every, which silently swallows most of the output.
            log_every=max(1, steps // 6),
        ),
        choice=choice,
        eos_token_id=eot_id(tokenizer),
        on_event=lambda e: (
            console.print(
                f"  step {e['step']:>4}  loss [bold]{e['loss']:.3f}[/bold]  "
                f"{e['tokens_per_second']:,.0f} tok/s"
            )
            if e["event"] == "step"
            else None
        ),
    )

    console.print()
    console.print(
        f"trained in [bold]{result.tokens_seen:,}[/bold] tokens · "
        f"final loss [bold]{result.final_loss:.3f}[/bold] · "
        f"val [bold]{result.best_val_loss:.3f}[/bold]"
    )
    console.print()

    loaded = load(result.checkpoint, device=device)
    console.print("[bold]samples[/bold] [dim](the model has never seen these prompts)[/dim]\n")
    for seed_text in ("Ana found", "One morning", "The sleepy"):
        continuation = complete(
            loaded, seed_text, SamplingConfig(max_new_tokens=30, temperature=0.7, seed=0)
        )
        console.print(f"  [dim]{seed_text}[/dim][bold]{continuation.rstrip()}[/bold]")

    console.print()
    if keep:
        console.print(f"[dim]kept: {paths.dataset_dir(name)} and {paths.run_dir(name)}[/dim]")
    else:
        shutil.rmtree(paths.dataset_dir(name), ignore_errors=True)
        shutil.rmtree(paths.run_dir(name), ignore_errors=True)
        console.print("[dim]cleaned up. pass --keep to keep the dataset and checkpoint.[/dim]")

    console.print(
        "\nThat was a toy. For something real:\n"
        "  [bold]bloomery prepare --name mine --source ./my-text --vocab 8192[/bold]\n"
        "  [bold]bloomery train --data mine --depth 8 --steps 5000[/bold]"
    )


def _force_utf8_output() -> None:
    """Make stdout and stderr UTF-8 regardless of the platform's locale.

    Windows defaults to the locale codec — cp1252 for most installs — whenever
    output is not a console, which is exactly what happens when a command is
    redirected to a file or captured by the job runner. The report uses "→" and
    "×", none of which cp1252 can encode, so the command died with
    UnicodeEncodeError partway through printing rather than doing its work.

    errors="replace" as well as the encoding: losing a character from a progress
    line is a far better outcome than killing a twelve-hour training run over it.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # A stream that cannot be reconfigured, such as one already replaced
            # by a test harness. Nothing to do, and not worth failing over.
            continue


def main() -> None:
    _force_utf8_output()
    app()


if __name__ == "__main__":
    main()
