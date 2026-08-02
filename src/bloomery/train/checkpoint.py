# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Saving and resuming runs.

Checkpoints are written in Hugging Face layout — ``save_pretrained`` for the
model and tokenizer — plus one extra file holding optimizer state and step
count. That split is deliberate: the model directory is independently loadable
by anything in the ecosystem, while the training state stays a bloomery detail.

Writes go to a temporary path and are then renamed. A checkpoint half-written
when a process is killed is worse than no checkpoint, because it looks resumable
and is not.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

TRAINER_STATE = "trainer_state.pt"
RUN_META = "run.json"


@dataclass(frozen=True, slots=True)
class ResumeState:
    step: int
    best_val_loss: float | None
    tokens_seen: int
    # Per-component forgetting history. Without it a resumed run treats its first
    # evaluation as a baseline, so a component that was already degrading before
    # the checkpoint is reported as healthy.
    component_best: dict[str, float] = field(default_factory=dict)
    component_first: dict[str, float] = field(default_factory=dict)


def checkpoint_dir(run_dir: Path, step: int | None = None) -> Path:
    """``latest`` for the rolling checkpoint, ``step-000123`` for a snapshot."""
    return run_dir / ("latest" if step is None else f"step-{step:06d}")


def save(
    directory: Path,
    *,
    model: Any,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    step: int,
    tokens_seen: int,
    best_val_loss: float | None,
    component_best: dict[str, float] | None = None,
    component_first: dict[str, float] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a complete, resumable checkpoint atomically."""
    import torch

    staging = directory.with_name(directory.name + ".tmp")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    model.save_pretrained(str(staging), safe_serialization=True)
    tokenizer.save_pretrained(str(staging))

    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "step": step,
            "tokens_seen": tokens_seen,
            "best_val_loss": best_val_loss,
            "component_best": component_best or {},
            "component_first": component_first or {},
        },
        staging / TRAINER_STATE,
    )

    (staging / RUN_META).write_text(
        json.dumps(
            {
                "step": step,
                "tokens_seen": tokens_seen,
                "best_val_loss": best_val_loss,
                **(extra or {}),
            },
            indent=2,
            default=str,
        )
        + "\n"
    )

    # Replace only once everything is on disk.
    if directory.exists():
        shutil.rmtree(directory)
    staging.rename(directory)
    return directory


def load_resume_state(
    directory: Path, optimizer: torch.optim.Optimizer | None = None
) -> ResumeState:
    """Restore step counters, and optimizer state if an optimizer is given."""
    import torch

    state_path = directory / TRAINER_STATE
    if not state_path.is_file():
        raise FileNotFoundError(f"no {TRAINER_STATE} in {directory}")

    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])

    return ResumeState(
        step=int(payload.get("step", 0)),
        best_val_loss=payload.get("best_val_loss"),
        tokens_seen=int(payload.get("tokens_seen", 0)),
        # Absent in checkpoints written before forgetting was tracked; an empty
        # history simply means the next evaluation establishes the baseline.
        component_best=dict(payload.get("component_best") or {}),
        component_first=dict(payload.get("component_first") or {}),
    )


def is_resumable(directory: Path) -> bool:
    """Whether a run can be picked up from this directory.

    Needs the optimizer state, plus weights in one of the two shapes a run
    writes: a whole model, or the adapters a LoRA run produced. Checking only for
    ``config.json`` would report every adapter checkpoint as unresumable, which
    is the shape a long adaptation run leaves behind.
    """
    if not (directory / TRAINER_STATE).is_file():
        return False
    return (directory / "config.json").is_file() or (directory / "adapter_config.json").is_file()
