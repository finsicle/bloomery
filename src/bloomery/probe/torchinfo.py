# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What an already-installed PyTorch reports about itself.

Optional. On a fresh install torch is absent and that is not a problem — the
rest of the probe works without it. When torch *is* present this is the section
that catches the install actually being wrong: a CPU-only wheel on a machine
with two H100s is a silent, expensive mistake, and comparing what we detected
against what torch believes is the only way to spot it.
"""

from __future__ import annotations

from contextlib import suppress

from bloomery.probe.types import Backend, GpuInfo, Issue, TorchInfo, Vendor


def probe() -> TorchInfo | None:
    """Interrogate the installed torch, or None if it is not importable.

    Careful to avoid initialising a CUDA or HIP context. ``torch.cuda.is_available``
    is safe; anything that allocates is not, because doing so would claim VRAM
    from a training run that might be in progress.
    """
    try:
        import torch  # noqa: PLC0415 - optional dependency, imported on demand
    except Exception:  # noqa: BLE001 - a broken install raises all sorts
        return None

    version = getattr(torch, "__version__", "unknown")
    build = None
    if "+" in str(version):
        build = str(version).partition("+")[2]

    info = TorchInfo(version=str(version), build=build)

    # Each probe is suppressed independently: a broken CUDA install must not
    # stop us reporting that Metal or XPU works, and a half-installed torch can
    # raise from any of these.
    with suppress(Exception):
        info.cuda_version = getattr(torch.version, "cuda", None)
        info.hip_version = getattr(torch.version, "hip", None)

    with suppress(Exception):
        info.cuda_available = bool(torch.cuda.is_available())
        if info.cuda_available:
            info.device_count = int(torch.cuda.device_count())

    with suppress(Exception):
        info.mps_available = bool(torch.backends.mps.is_available())

    with suppress(Exception):
        info.xpu_available = bool(torch.xpu.is_available())
        if info.xpu_available and not info.device_count:
            info.device_count = int(torch.xpu.device_count())

    return info


def consistency_issues(
    torch_info: TorchInfo | None,
    gpus: list[GpuInfo],
    recommended: Backend,
) -> list[Issue]:
    """Compare what we detected against what torch can actually reach."""
    if torch_info is None:
        return []

    issues: list[Issue] = []
    vendors = {gpu.vendor for gpu in gpus}
    accelerated_gpus = vendors & {Vendor.NVIDIA, Vendor.AMD, Vendor.INTEL, Vendor.APPLE}

    reachable = torch_info.cuda_available or torch_info.mps_available or torch_info.xpu_available

    if accelerated_gpus and not reachable:
        install = (
            f"uv pip install --torch-backend={recommended.uv_torch_backend} --force-reinstall torch"
        )
        issues.append(
            Issue(
                level="error",
                message=(
                    f"PyTorch {torch_info.version} cannot see any of the "
                    f"{len(gpus)} detected GPU(s). Training would fall back to CPU."
                ),
                hint=f"Reinstall against the right backend:\n  {install}",
            )
        )
    elif (
        torch_info.cuda_available
        and Vendor.NVIDIA in vendors
        and torch_info.device_count < len([g for g in gpus if g.vendor is Vendor.NVIDIA])
    ):
        issues.append(
            Issue(
                level="warn",
                message=(
                    f"torch sees {torch_info.device_count} CUDA device(s) but "
                    f"{len([g for g in gpus if g.vendor is Vendor.NVIDIA])} are installed."
                ),
                hint="CUDA_VISIBLE_DEVICES may be restricting them.",
            )
        )

    # A ROCm torch reports through the CUDA API, so hip_version is how you tell
    # the two builds apart.
    if Vendor.AMD in vendors and torch_info.cuda_available and not torch_info.hip_version:
        issues.append(
            Issue(
                level="warn",
                message="An AMD GPU is present but torch is a CUDA build, not ROCm.",
                hint="uv pip install --torch-backend=rocm7.2 --force-reinstall torch",
            )
        )

    return issues
