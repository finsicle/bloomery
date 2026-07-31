# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The bf16 probe, run as a throwaway process.

Deciding whether bf16 is worth using means running a bf16 matmul and timing it.
On some CPUs that matmul is fatal rather than slow: the kernel reaches for an
instruction the chip does not implement and the process takes a hardware trap —
SIGILL on Unix, ``0xc000001d`` on Windows. It cannot be caught, because there is
nothing left to catch it.

Asking the CPU first is not sufficient either. A Windows runner that reported
AVX512-BF16 support, and survived an 8x8 bf16 matmul, still died on a 384x384
one — the larger shape dispatches to a different kernel, and support for one
instruction set says nothing about the other.

So the probe runs here, in a process whose death costs nothing. The parent reads
the verdict from the exit status: anything other than a clean report means "do
not use bf16", which is the answer that was wanted from a crash anyway.

Not part of the public interface. Invoked as ``python -m
bloomery.train._bf16_probe <device>``.
"""

from __future__ import annotations

import sys

VERDICT_FAST = "bf16"
VERDICT_SLOW = "fp32"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("usage: python -m bloomery.train._bf16_probe <device>", file=sys.stderr)
        return 2

    import torch

    from bloomery.train.device import _bf16_is_faster, _bf16_works

    device = torch.device(argv[0])

    # Either call may take the process down. That is the point of running here.
    if not _bf16_works(device):
        print(VERDICT_SLOW)
        return 0
    print(VERDICT_FAST if _bf16_is_faster(device) else VERDICT_SLOW)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main())
