# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Terminal rendering for the doctor report."""

from __future__ import annotations

from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from bloomery.capability import (
    CapabilityReport,
    Method,
    format_params,
    format_tokens,
)
from bloomery.probe.types import GIB, Backend, HostReport, Issue, Platform, Vendor

_LEVEL_STYLE = {"error": "bold red", "warn": "yellow", "info": "cyan"}
_LEVEL_MARK = {"error": "✗", "warn": "!", "info": "i"}


def _kv_table() -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", justify="right")
    table.add_column()
    return table


def _or_unknown(value: object, suffix: str = "") -> str:
    if value is None:
        return "[dim]unknown[/dim]"
    return f"{value}{suffix}"


def render_host(report: HostReport) -> Panel:
    table = _kv_table()

    platform_label = {
        Platform.LINUX: "Linux",
        Platform.WSL2: "Linux under WSL2",
        Platform.WINDOWS: "Windows (native)",
        Platform.MACOS: "macOS",
        Platform.UNKNOWN: "unknown",
    }[report.host.platform]

    table.add_row("platform", platform_label)
    if report.host.distro:
        table.add_row("os", report.host.distro)
    if report.host.wsl_kernel:
        table.add_row("wsl kernel", report.host.wsl_kernel)
    table.add_row("arch", report.host.machine)
    table.add_row("python", report.host.python_version)

    cores = report.cpu
    if cores.physical_cores or cores.logical_cores:
        detail = f"{_or_unknown(cores.physical_cores)} physical"
        if cores.logical_cores:
            detail += f" / {cores.logical_cores} logical"
        table.add_row("cpu", detail)
    if cores.model:
        table.add_row("", f"[dim]{cores.model}[/dim]")

    memory = report.memory
    if memory.total_gib is not None:
        detail = f"{memory.total_gib:.0f} GiB total"
        if memory.available_gib is not None:
            detail += f" / {memory.available_gib:.0f} GiB available"
        table.add_row("ram", detail)

    if report.disk.free_gib is not None:
        table.add_row("disk", f"{report.disk.free_gib:.0f} GiB free at {report.disk.path}")

    return Panel(table, title="host", title_align="left", border_style="dim")


def render_gpus(report: HostReport) -> Panel:
    if not report.gpus:
        body: Group | Text = Text("No GPU detected.", style="yellow")
        return Panel(body, title="gpus", title_align="left", border_style="dim")

    table = Table(box=None, pad_edge=False, expand=False)
    table.add_column("#", style="dim", width=2)
    table.add_column("vendor")
    table.add_column("device")
    table.add_column("memory", justify="right")
    table.add_column("arch", style="dim")
    table.add_column("source", style="dim")

    for gpu in report.gpus:
        vendor_style = {
            Vendor.NVIDIA: "green",
            Vendor.AMD: "red",
            Vendor.APPLE: "white",
            Vendor.INTEL: "blue",
        }.get(gpu.vendor, "white")

        if gpu.vram_gib is None:
            memory = "[dim]unknown[/dim]"
        elif gpu.vendor is Vendor.APPLE:
            memory = f"{gpu.vram_gib:.0f} GiB*"
        else:
            memory = f"{gpu.vram_gib:.0f} GiB"

        table.add_row(
            str(gpu.index),
            Text(gpu.vendor.value, style=vendor_style),
            gpu.name,
            memory,
            gpu.arch or "",
            gpu.source,
        )

    extras = _kv_table()
    if report.nvidia_driver:
        extras.add_row("nvidia driver", report.nvidia_driver)
    if report.nvidia_cuda_version:
        extras.add_row("cuda (driver max)", report.nvidia_cuda_version)
    if report.rocm_version:
        extras.add_row("rocm", report.rocm_version)

    parts: list[RenderableType] = [table]
    if extras.row_count:
        parts.extend([Text(""), extras])
    if any(g.vendor is Vendor.APPLE for g in report.gpus):
        parts.append(
            Text(
                "\n* unified memory available to the GPU, not a separate pool",
                style="dim italic",
            )
        )

    return Panel(Group(*parts), title="gpus", title_align="left", border_style="dim")


def render_backend(report: HostReport) -> Panel:
    table = _kv_table()
    style = "bold green" if report.backend.accelerated else "bold yellow"
    table.add_row("recommended", Text(report.backend.value, style=style))
    table.add_row("because", report.backend_reason)

    if report.torch:
        torch_info = report.torch
        table.add_row("installed torch", torch_info.version)
        reachable = []
        if torch_info.cuda_available:
            label = "hip" if torch_info.hip_version else "cuda"
            reachable.append(f"{label} ({torch_info.device_count} device(s))")
        if torch_info.mps_available:
            reachable.append("mps")
        if torch_info.xpu_available:
            reachable.append("xpu")
        table.add_row(
            "torch can reach",
            ", ".join(reachable) if reachable else Text("cpu only", style="yellow"),
        )
    else:
        table.add_row("installed torch", "[dim]not installed[/dim]")

    install = Text()
    install.append("\ninstall with\n  ", style="dim")
    install.append(
        f'uv pip install --torch-backend={report.backend.uv_torch_backend} -e ".[train,serve]"',
        style="bold",
    )
    if report.backend is Backend.MPS:
        # Worth spelling out: the recommendation says mps but the flag says cpu,
        # which looks like a contradiction until you know Apple builds come
        # from plain PyPI.
        install.append(
            "\n\ncpu is correct here — PyTorch's PyPI wheel carries Metal support,\n"
            "so there is no separate mps index to point at.",
            style="dim italic",
        )
    install.append(
        "\n\n--torch-backend=auto does its own detection and is authoritative;\n"
        "the value above is what we expect it to choose.",
        style="dim italic",
    )

    return Panel(
        Group(table, install), title="pytorch backend", title_align="left", border_style="dim"
    )


def _cell(required: int, budget: int, available: bool, reason: str) -> Text:
    if not available:
        return Text("n/a", style="dim")
    gib = required / GIB
    if required <= budget:
        headroom = budget / required if required else 0
        style = "green" if headroom >= 1.5 else "yellow"
        return Text(f"{gib:.1f} GiB", style=style)
    return Text(f"{gib:.0f} GiB", style="dim red")


def render_capability(capability: CapabilityReport) -> Panel:
    table = Table(box=None, pad_edge=False, expand=False)
    table.add_column("model", no_wrap=True)
    table.add_column("params", justify="right", style="dim")
    table.add_column("tokens", justify="right", style="dim")
    # Batch and sequence length are load-bearing: the logit tensor scales with
    # both, so without this column the memory numbers look non-monotonic down
    # the table and the reader reasonably concludes the tool is broken.
    table.add_column("step", justify="right", style="dim")
    for method in Method:
        table.add_column(method.label, justify="right")

    by_method = {m: {r.spec.key: r for r in capability.for_method(m)} for m in Method}

    for spec in [r.spec for r in capability.for_method(Method.FULL)]:
        cells = []
        for method in Method:
            row = by_method[method][spec.key]
            cells.append(
                _cell(row.required, row.budget, row.available_here, row.unavailable_reason)
            )
        table.add_row(
            spec.label,
            format_params(spec.params),
            format_tokens(spec.params * 20),
            f"{spec.batch}×{spec.seq}",
            *cells,
        )

    scratch = capability.largest_fitting(Method.FULL)
    qlora = capability.largest_fitting(Method.QLORA)

    headline = Text()
    headline.append("budget  ", style="dim")
    headline.append(capability.budget.source, style="bold")
    if capability.budget.shardable_total:
        headline.append(
            f"\n        {capability.budget.shardable_devices} devices, "
            f"{capability.budget.shardable_total / GIB:.0f} GiB combined — "
            "reachable with FSDP or ZeRO-3, not with plain data parallel",
            style="dim italic",
        )
    headline.append("\n\n")

    if scratch:
        headline.append("pretrain from scratch  ", style="dim")
        headline.append(
            f"up to {scratch.spec.label} ({format_params(scratch.spec.params)})",
            style="bold green",
        )
        headline.append(f"  ·  needs ~{format_tokens(scratch.tokens)} tokens\n", style="dim")
    else:
        headline.append("pretrain from scratch  nothing on the ladder fits\n", style="bold red")

    if qlora:
        headline.append("fine-tune with QLoRA   ", style="dim")
        headline.append(
            f"up to {qlora.spec.label} ({format_params(qlora.spec.params)})",
            style="bold green",
        )
        headline.append("\n")

    footer = Text(
        "\nstep is batch×sequence, chosen per size — memory scales with it, so the\n"
        "columns are not monotonic down the table.\n"
        "Full covers pretraining from scratch and full fine-tuning. LoRA and QLoRA\n"
        "adapt an existing model; they cannot train one from random init.\n"
        "Assumes AdamW, bf16, gradient checkpointing and a flash-attention kernel.\n"
        "Conservative by design. Throughput is not estimated — measure it with "
        "`bloomery bench`.",
        style="dim italic",
    )

    return Panel(
        Group(headline, table, footer),
        title="what this machine can train",
        title_align="left",
        border_style="dim",
    )


def render_issues(issues: list[Issue]) -> Panel | None:
    if not issues:
        return None

    order = {"error": 0, "warn": 1, "info": 2}
    body = Table.grid(padding=(0, 1))
    body.add_column(width=1)
    body.add_column()

    for issue in sorted(issues, key=lambda i: order.get(i.level, 3)):
        style = _LEVEL_STYLE.get(issue.level, "white")
        text = Text(issue.message, style=style)
        if issue.hint:
            text.append(f"\n{issue.hint}", style="dim")
        body.add_row(Text(_LEVEL_MARK.get(issue.level, "-"), style=style), text)

    worst = min((order.get(i.level, 3) for i in issues), default=3)
    border = ["red", "yellow", "cyan", "dim"][worst]
    return Panel(body, title="notes", title_align="left", border_style=border)


def render_report(console: Console, report: HostReport, capability: CapabilityReport) -> None:
    console.print(render_host(report))
    console.print(render_gpus(report))
    console.print(render_backend(report))
    console.print(render_capability(capability))
    issues = render_issues(report.issues)
    if issues is not None:
        console.print(issues)
