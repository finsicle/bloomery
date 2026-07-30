# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared helpers for the probe.

Every external call goes through :func:`run`. Vendor tools hang more often than
you would like — a wedged ``nvidia-smi`` on a machine with a half-loaded driver
is a classic — so nothing here is allowed to block indefinitely or raise.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 8.0


@dataclass(slots=True)
class Result:
    ok: bool
    stdout: str = ""
    stderr: str = ""
    returncode: int | None = None

    def lines(self) -> list[str]:
        return [stripped for line in self.stdout.splitlines() if (stripped := line.strip())]


def which(name: str) -> str | None:
    """Locate an executable, or None."""
    return shutil.which(name)


def run(
    cmd: list[str],
    *,
    timeout: float = DEFAULT_TIMEOUT,
) -> Result:
    """Run a command and capture its output.

    Never raises. A missing binary, a non-zero exit, a timeout and a permission
    error all come back as ``ok=False``, because from the probe's point of view
    they mean the same thing: this source has nothing to tell us.
    """
    if not cmd:
        return Result(ok=False, stderr="empty command")
    if which(cmd[0]) is None:
        return Result(ok=False, stderr=f"{cmd[0]}: not found")

    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        log.debug("timeout running %s", " ".join(cmd))
        return Result(ok=False, stderr=f"{cmd[0]}: timed out after {timeout}s")
    except (OSError, ValueError) as exc:  # pragma: no cover - platform dependent
        log.debug("error running %s: %s", " ".join(cmd), exc)
        return Result(ok=False, stderr=f"{cmd[0]}: {exc}")

    return Result(
        ok=proc.returncode == 0,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
        returncode=proc.returncode,
    )


def read_text(path: str | Path) -> str | None:
    """Read a small file, or None if unreadable.

    Used for sysfs, where files can vanish or reject reads between the glob and
    the open.
    """
    try:
        return Path(path).read_text(errors="replace").strip()
    except (OSError, UnicodeError):
        return None


def read_int(path: str | Path) -> int | None:
    raw = read_text(path)
    if raw is None:
        return None
    try:
        return int(raw.strip(), 0)
    except ValueError:
        return None


def parse_int(value: Any) -> int | None:
    """Best-effort int from vendor-tool output.

    Handles ``"16384"``, ``"16384 MB"``, ``"[N/A]"``, ``None`` and floats.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value).strip()
    if not text or text.startswith("["):
        return None
    digits = ""
    for ch in text:
        if ch.isdigit():
            digits += ch
        elif digits:
            break
    return int(digits) if digits else None


def find_key(obj: Any, candidates: tuple[str, ...]) -> Any:
    """Depth-first search a nested dict/list for the first matching key.

    Vendor JSON schemas move fields between releases and rename them without
    much ceremony. Rather than pin to one layout, we look for any of a set of
    plausible key names anywhere in the tree. Matching is case-insensitive and
    ignores separators, so ``marketName``, ``market_name`` and
    ``"Market Name"`` all hit the same candidate.

    Breadth-first, so a shallow match wins over a deeper one.
    """
    wanted = {_norm_key(c) for c in candidates}
    stack: list[Any] = [obj]
    while stack:
        node = stack.pop(0)
        if isinstance(node, dict):
            for key, value in node.items():
                if _norm_key(str(key)) in wanted and value not in (
                    None,
                    "",
                    "N/A",
                    "[N/A]",
                    "NA",
                ):
                    return value
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return None


def _norm_key(key: str) -> str:
    """Reduce a key to comparable form.

    Strips every non-alphanumeric character rather than a fixed list, because
    vendor keys carry units in parentheses — rocm-smi's
    ``"VRAM Total Memory (B)"`` has to match the candidate
    ``vram_total_memory_b``.
    """
    return "".join(ch for ch in key if ch.isalnum()).lower()


def gfx_name(target_version: int) -> str | None:
    """Turn a KFD ``gfx_target_version`` integer into a gfx name.

    The encoding is ``major * 10000 + minor * 100 + step``, with minor and step
    rendered as single hex digits. So 110000 is gfx1100, 90402 is gfx942, and
    90010 is gfx90a.
    """
    if target_version <= 0:
        return None
    major, rest = divmod(target_version, 10000)
    minor, step = divmod(rest, 100)
    if minor > 15 or step > 15:
        return None
    return f"gfx{major}{minor:x}{step:x}"
