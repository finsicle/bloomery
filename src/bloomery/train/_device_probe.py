# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The accelerator usability probe, run as a throwaway process.

The sibling of :mod:`bloomery.train._bf16_probe`, asking a blunter question: can
this device execute anything at all?

That is not the same as whether torch can see it. Measured on an RX 6700 XT
(gfx1031) with ROCm 7.1 and torch 2.13+rocm7.2, with no ``HSA_OVERRIDE_GFX_VERSION``
set::

    torch.cuda.is_available()                    True
    torch.cuda.get_device_name(0)                AMD Radeon RX 6700 XT
    torch.cuda.is_bf16_supported(emulation=False) True
    a = torch.randn(512, 512, device="cuda"); a @ a
    Segmentation fault (core dumped)

Every question torch was asked came back positive, and the first operation took
the process down — in fp32, fp16 and bf16 alike, so it is not a precision
question but a "this device does not work" one.

The bf16 probe's own docstring already says why asking cannot settle this: a
trap is not an exception, and there is nothing left in the process to catch it.
That reasoning was applied to the CPU path and not to this one, which is the gap
this module closes. The verdict is read from the exit status, and death is a
legitimate answer.

Not part of the public interface. Invoked as ``python -m
bloomery.train._device_probe <device>``.
"""

from __future__ import annotations

import sys

VERDICT_OK = "ok"

# Large enough to reach a real kernel. The bf16 probe learned this the hard way:
# an 8x8 matmul survived on a CPU that died on a 384x384 one, because the larger
# shape dispatches somewhere else entirely.
_SIZE = 512


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("usage: python -m bloomery.train._device_probe <device>", file=sys.stderr)
        return 2

    import torch

    device = torch.device(argv[0])

    # Any line below may take the process down. That is the point of running here.
    tensor = torch.randn(_SIZE, _SIZE, device=device)
    # float() forces a synchronise. Without it the kernel is still queued and a
    # fault would surface later, in the parent, where it cannot be survived.
    total = float((tensor @ tensor).sum())
    if total != total:  # NaN: it ran, but produced nothing usable
        return 1

    print(VERDICT_OK)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main())
