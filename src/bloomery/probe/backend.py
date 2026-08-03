# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Recommend a PyTorch build for the detected hardware.

This is advisory, not authoritative. The installer calls
``uv pip install --torch-backend=auto``, and uv performs its own detection
against the CUDA driver, AMD GPU version and Intel GPU presence. Duplicating
that as the source of truth would guarantee the two drift apart.

What this does is *explain* the outcome: if a user ends up on a CPU build with an
expensive GPU installed, they need to know which check failed and what to do,
and a silent ``auto`` cannot tell them that.
"""

from __future__ import annotations

import os

from bloomery.probe.types import Backend, GpuInfo, Issue, Vendor

# The variable that makes an unsupported AMD card present itself as a supported
# one. Read here rather than only suggested, because whether it is already set
# is the difference between a card that trains and a card that dies on its first
# operation.
HSA_OVERRIDE = "HSA_OVERRIDE_GFX_VERSION"

# nvidia-smi reports the highest CUDA version the installed driver supports.
# Highest match wins. CUDA 12 minor-version compatibility means any 12.x driver
# can run any cu12x build, so these thresholds are conservative, not exact.
_CUDA_TO_BACKEND: tuple[tuple[tuple[int, int], Backend], ...] = (
    ((13, 0), Backend.CU130),
    ((12, 8), Backend.CU128),
    ((12, 6), Backend.CU126),
    ((11, 8), Backend.CU118),
)

# Fallback when the CUDA version could not be read but the driver version could.
_DRIVER_TO_BACKEND: tuple[tuple[int, Backend], ...] = (
    (580, Backend.CU130),
    (525, Backend.CU128),
    (450, Backend.CU118),
)

# ROCm's officially supported targets, as of ROCm 7.2. Cards outside this set
# frequently work anyway via HSA_OVERRIDE_GFX_VERSION, so absence here is a
# warning rather than a refusal. This list moves every release — treat a
# mismatch as "check the docs", not "impossible".
ROCM_SUPPORTED_ARCHS = frozenset(
    {
        "gfx90a",  # MI200
        "gfx942",  # MI300
        "gfx950",  # MI350
        "gfx1100",  # RX 7900 XTX / XT
        "gfx1101",  # RX 7800 XT
        "gfx1102",  # RX 7700 / 7600
        "gfx1200",  # RX 9060
        "gfx1201",  # RX 9070
    }
)

# RDNA1/RDNA2 consumer cards. These generally run once told to impersonate a
# supported target.
_OVERRIDE_HINTS: dict[str, str] = {
    "gfx1010": "10.3.0",
    "gfx1011": "10.3.0",
    "gfx1012": "10.3.0",
    "gfx1030": "10.3.0",
    "gfx1031": "10.3.0",
    "gfx1032": "10.3.0",
    "gfx1034": "10.3.0",
    "gfx1035": "10.3.0",
    "gfx1036": "10.3.0",
}


def parse_version(text: str | None) -> tuple[int, ...] | None:
    """Parse a dotted version into a comparable tuple. Trailing junk is dropped."""
    if not text:
        return None
    parts: list[int] = []
    for chunk in str(text).strip().split("."):
        digits = ""
        for ch in chunk:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) if parts else None


def nvidia_backend(cuda_version: str | None, driver_version: str | None) -> tuple[Backend, str]:
    """Pick a CUDA wheel tag from the driver's reported capability."""
    cuda = parse_version(cuda_version)
    if cuda:
        # "13" has to compare as (13, 0), not as the one-element tuple (13,),
        # which would sort below every two-element threshold.
        padded = (cuda + (0,) * (2 - len(cuda))) if len(cuda) < 2 else cuda
        for cuda_threshold, backend in _CUDA_TO_BACKEND:
            if padded[:2] >= cuda_threshold:
                return backend, f"driver supports CUDA {cuda_version}"

    driver = parse_version(driver_version)
    if driver:
        for driver_threshold, backend in _DRIVER_TO_BACKEND:
            if driver[0] >= driver_threshold:
                return backend, f"driver {driver_version}, CUDA version unreported"

    return (
        Backend.CU128,
        "NVIDIA GPU found but driver version unreadable; cu128 is the safe default",
    )


def resolve(
    gpus: list[GpuInfo],
    *,
    cuda_version: str | None = None,
    driver_version: str | None = None,
) -> tuple[Backend, str]:
    """Choose a backend for the detected GPUs.

    Order is NVIDIA, AMD, Intel, Apple. On a machine with cards from two
    vendors, one process can only sensibly use one backend, and NVIDIA is the
    best supported, so it wins.
    """
    vendors = {gpu.vendor for gpu in gpus}

    if Vendor.NVIDIA in vendors:
        return nvidia_backend(cuda_version, driver_version)
    if Vendor.AMD in vendors:
        return Backend.ROCM, "AMD GPU detected"
    if Vendor.INTEL in vendors:
        return Backend.XPU, "Intel GPU detected"
    if Vendor.APPLE in vendors:
        return Backend.MPS, "Apple Silicon detected; Metal support ships in the PyPI build"
    return Backend.CPU, "no supported GPU detected"


def override_hint(arch: str | None) -> str | None:
    """The ``HSA_OVERRIDE_GFX_VERSION`` value that usually makes this card work.

    ``None`` for a card that is either already supported or not one we have a
    known answer for. The gfx name may carry feature suffixes — torch reports
    ``gfx1031:sramecc-:xnack-`` — so only the part before the first colon is
    matched.
    """
    if not arch:
        return None
    return _OVERRIDE_HINTS.get(arch.split(":")[0].strip().lower())


def arch_issues(gpus: list[GpuInfo]) -> list[Issue]:
    """Report AMD cards outside ROCm's supported set.

    The level depends on whether the override is already in effect, because the
    two states could hardly be further apart. Measured on an RX 6700 XT
    (gfx1031) running ROCm 7.1 with torch 2.13+rocm7.2: with the variable set,
    training runs at 80,000 tokens/second. Without it, ``torch.cuda.is_available()``
    still returns True, ``is_bf16_supported()`` still returns True, and the first
    matmul of any dtype segfaults the process.

    So an unsupported card with no override is an error rather than a caution —
    it is not "rough edges", it is a run that cannot start.
    """
    issues: list[Issue] = []
    # The value, not merely whether one is set. An override naming a different
    # architecture than this card needs is no better than none — the card still
    # takes the process down on its first operation — so treating "set" as "safe"
    # would wave through the exact failure this check exists to catch.
    setting = os.environ.get(HSA_OVERRIDE, "").strip()
    for gpu in gpus:
        if gpu.vendor is not Vendor.AMD or not gpu.arch:
            continue
        arch = gpu.arch.lower()
        if arch in ROCM_SUPPORTED_ARCHS:
            continue
        override = override_hint(arch)
        if override and setting == override:
            issues.append(
                Issue(
                    level="info",
                    message=f"{gpu.name} ({arch}) is running through {HSA_OVERRIDE}.",
                    hint=(
                        "Not an officially supported ROCm target, so expect rough edges — "
                        "but the override is set and the card should work."
                    ),
                )
            )
        elif override and setting:
            issues.append(
                Issue(
                    level="error",
                    message=(
                        f"{HSA_OVERRIDE} is set to {setting}, but {gpu.name} ({arch}) "
                        f"needs {override}. The wrong architecture is no safer than none: "
                        "operations on this GPU will still crash the process."
                    ),
                    hint=(
                        f"export {HSA_OVERRIDE}={override}\n"
                        "  One variable covers the whole process, so a machine with two "
                        "cards wanting different values can only run one of them at a time."
                    ),
                )
            )
        elif override:
            issues.append(
                Issue(
                    level="error",
                    message=(
                        f"{gpu.name} ({arch}) is not a supported ROCm target, and "
                        f"{HSA_OVERRIDE} is not set. Any operation on this GPU will "
                        "crash the process."
                    ),
                    hint=(
                        f"export {HSA_OVERRIDE}={override}\n"
                        "  Then re-run. This makes the card present itself as a "
                        "supported one, which is what ROCm's kernels are built for."
                    ),
                )
            )
        else:
            issues.append(
                Issue(
                    level="warn",
                    message=f"{gpu.name} ({arch}) is not in ROCm's supported target list.",
                    hint=(
                        "Check the ROCm compatibility matrix for your card. "
                        f"{HSA_OVERRIDE} may help."
                    ),
                )
            )
    return issues
