# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Command line entry point."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from bloomery import __version__, paths
from bloomery.capability import LADDER_BY_KEY, Method, assess, format_params
from bloomery.probe import probe_host_report
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


def _die(message: str) -> None:
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
    data: str = typer.Option(..., "--data", "-d", help="Dataset name from `prepare`."),
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

    from bloomery.arch import actual_param_count, resolve_spec
    from bloomery.data import load_dataset, load_tokenizer
    from bloomery.train import checkpoint as ckpt
    from bloomery.train.device import choose, thread_limit
    from bloomery.train.loop import TrainConfig
    from bloomery.train.loop import train as run_training

    tokens_path = paths.tokens_dir(data)
    tok_path = paths.tokenizer_dir(data)
    if not tokens_path.is_dir() or not tok_path.is_dir():
        _die(f"no dataset named {data!r}. Run `bloomery prepare --name {data} ...` first.")

    try:
        info = load_dataset(tokens_path)
    except FileNotFoundError as exc:
        _die(str(exc))

    tokenizer = load_tokenizer(tok_path)
    from bloomery.data import eot_id

    eos = eot_id(tokenizer)

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

    console.print(
        f"model      [bold]{spec.label}[/bold]  "
        f"{spec.layers}L × {spec.hidden}d × {spec.heads}h  "
        f"{format_params(actual_param_count(spec))} params"
    )
    console.print(f"device     {choice.label()}  [dim]{choice.reason}[/dim]")
    console.print(
        f"data       {info.split('train').tokens:,} train tokens  "
        f"[dim]{config.tokens_per_step:,} tokens/step[/dim]"
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
                progress.console.print(
                    f"  [dim]step {event['step']}[/dim]  "
                    f"val loss [bold]{event['val_loss']:.4f}[/bold]  "
                    f"ppl {event['perplexity']}"
                )

        try:
            result = run_training(
                spec=spec,
                dataset=info,
                tokenizer=tokenizer,
                run_dir=run_path,
                config=config,
                choice=choice,
                eos_token_id=eos,
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
    console.print(f"checkpoint {result.checkpoint}")
    console.print(f"\nnext:  [bold]bloomery chat --run {name}[/bold]")


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
        dataset=info,
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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
