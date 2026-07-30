# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Command line entry point."""

from __future__ import annotations

import json

import typer
from rich.console import Console

from bloomery import __version__, paths
from bloomery.capability import Method, assess
from bloomery.probe import probe_host_report
from bloomery.render import render_report

app = typer.Typer(
    name="bloomery",
    help="Train a language model from nothing, on your own machine.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


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


def _spec_summary(capability, method: Method) -> dict | None:  # noqa: ANN001
    row = capability.largest_fitting(method)
    if row is None:
        return None
    return {
        "key": row.spec.key,
        "label": row.spec.label,
        "params": row.spec.params,
        "required_bytes": row.required,
    }


def main() -> None:
    app()


if __name__ == "__main__":
    main()
