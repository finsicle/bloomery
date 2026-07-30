# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Command line entry point.

Currently a placeholder. The first real command will be `bloomery doctor`,
which probes the host for GPUs, drivers, memory and cores and reports which
PyTorch backend it would resolve to.
"""

from __future__ import annotations

import typer
from rich.console import Console

from bloomery import __version__

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
def doctor() -> None:
    """Report host hardware and the PyTorch backend it maps to."""
    console.print("[yellow]Not implemented yet.[/yellow] This is milestone M0.")
    console.print("Follow along at https://github.com/aswinsam/bloomery")
    raise typer.Exit(code=1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
