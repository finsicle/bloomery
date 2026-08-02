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
    # Tokens the loss was actually computed over. Restored alongside tokens_seen
    # or a resumed run reports only the portion after the resume, while its
    # sibling counter stays cumulative.
    trained_tokens: int = 0
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
    trained_tokens: int = 0,
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
            "trained_tokens": trained_tokens,
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
                "trained_tokens": trained_tokens,
                "best_val_loss": best_val_loss,
                **(extra or {}),
            },
            indent=2,
            default=str,
        )
        + "\n"
    )

    # Replace only once everything is on disk — and never leave a moment with no
    # checkpoint at all. Deleting the old one first opens a window where a kill,
    # or a rename that fails, loses the good checkpoint and puts nothing in its
    # place. The old one is moved aside instead, and only discarded once the new
    # one is where it belongs.
    #
    # Not academic: `export` reads runs/<name>/latest while a run may be saving,
    # and that window is exactly when it finds nothing there.
    previous = directory.with_name(directory.name + ".previous")

    # Recover before clearing. A save killed between the two renames below
    # leaves the checkpoint under `previous` and nothing at `directory`; going
    # straight to rmtree here would then delete the only copy that exists, which
    # is a worse outcome than the window this whole dance is closing.
    restore_interrupted(directory)
    if previous.exists():
        shutil.rmtree(previous)

    if directory.exists():
        directory.rename(previous)
    try:
        staging.rename(directory)
    except OSError:
        # Put the old one back rather than leaving the caller with neither.
        if previous.exists() and not directory.exists():
            previous.rename(directory)
        raise
    shutil.rmtree(previous, ignore_errors=True)
    return directory


def restore_interrupted(directory: Path) -> bool:
    """Put back a checkpoint left aside by a save that did not finish.

    The promotion above is two renames, and a process killed between them leaves
    the good checkpoint at ``<name>.previous`` with nothing at ``<name>``. An
    atomic directory exchange would remove even that gap, but the syscall for it
    is Linux-only, and this project runs on three platforms.

    So the gap is made recoverable instead of impossible: the state it leaves is
    unambiguous — a complete checkpoint under a known name — and this puts it
    back. Called before any save clears the way, and available to a reader that
    finds nothing where it expected a checkpoint.

    Returns whether anything was restored.
    """
    previous = directory.with_name(directory.name + ".previous")
    if directory.exists() or not previous.is_dir():
        return False
    previous.rename(directory)
    return True


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
        # Absent from a checkpoint written before masking existed, where every
        # token was trained on; falling back to the window count keeps such a
        # run's total honest rather than restarting it at zero.
        trained_tokens=int(payload.get("trained_tokens", payload.get("tokens_seen", 0))),
        # Absent in checkpoints written before forgetting was tracked; an empty
        # history simply means the next evaluation establishes the baseline.
        component_best=dict(payload.get("component_best") or {}),
        component_first=dict(payload.get("component_first") or {}),
    )


def is_resumable(directory: Path) -> bool:
    """Whether a run can be picked up from this directory.

    Recovers first: a save killed mid-promotion leaves the checkpoint beside
    this path rather than at it, and reporting "nothing to resume" then would
    discard a complete checkpoint that is sitting right there.

    Needs the optimizer state, plus weights in one of the two shapes a run
    writes: a whole model, or the adapters a LoRA run produced. Checking only for
    ``config.json`` would report every adapter checkpoint as unresumable, which
    is the shape a long adaptation run leaves behind.
    """
    restore_interrupted(directory)
    if not (directory / TRAINER_STATE).is_file():
        return False
    return (directory / "config.json").is_file() or (directory / "adapter_config.json").is_file()
