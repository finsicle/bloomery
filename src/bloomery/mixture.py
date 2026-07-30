# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Weighted, versioned dataset blends.

The problem this solves
-----------------------
"Keep adding datasets and keep training" does not work if taken literally.
Training on corpus A, then B, then C makes the model worse at A while it learns
C — catastrophic forgetting. The standard defence is *replay*: mix a share of the
earlier data back into every subsequent run.

Replay only works if the blend is a real object you can name, version and reuse.
If it lives in a shell history as ``--data a --data b`` then nobody can say what
run 7 was actually trained on, and nobody can reproduce it.

So a mixture here is a first-class, immutable, versioned record:

    blend v3 = domain_v3 @ 60% + domain_v1 @ 15% + general @ 25%

Adding a corpus produces v4 with v3 recorded as its parent, rather than editing
v3 in place. The lineage is the audit trail — it is the difference between
"we trained on some data" and "we trained on exactly this".
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bloomery import paths

# A component either teaches something new or exists to stop the model
# forgetting something old. Recording which is cheap and makes the replay share
# reportable.
ROLE_DOMAIN = "domain"
ROLE_REPLAY = "replay"
ROLES = (ROLE_DOMAIN, ROLE_REPLAY)

_SPEC_RE = re.compile(r"^\s*([A-Za-z0-9._-]+)\s*[:=]\s*([0-9]*\.?[0-9]+)\s*%?\s*$")


class MixtureError(ValueError):
    """Raised for a mixture that cannot be used as specified."""


@dataclass(frozen=True, slots=True)
class Component:
    """One dataset's share of a blend.

    ``weight`` is a raw number, not a fraction. Weights are normalised at use
    time so that adding a component does not silently rescale the others in the
    stored file — the numbers you typed are the numbers that get saved.
    """

    dataset: str
    weight: float
    role: str = ROLE_DOMAIN

    def __post_init__(self) -> None:
        if not self.dataset:
            raise MixtureError("component dataset name cannot be empty")
        if self.weight <= 0:
            raise MixtureError(
                f"component {self.dataset!r} has weight {self.weight}; "
                "weights must be positive (drop the component instead of zeroing it)"
            )
        if self.role not in ROLES:
            raise MixtureError(f"unknown role {self.role!r}; expected one of {ROLES}")

    def to_dict(self) -> dict[str, Any]:
        return {"dataset": self.dataset, "weight": self.weight, "role": self.role}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Component:
        return cls(
            dataset=str(payload["dataset"]),
            weight=float(payload["weight"]),
            role=str(payload.get("role", ROLE_DOMAIN)),
        )


@dataclass(frozen=True, slots=True)
class Mixture:
    """An immutable, versioned blend of datasets."""

    name: str
    version: int
    components: tuple[Component, ...]
    parent_version: int | None = None
    note: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.components:
            raise MixtureError("a mixture needs at least one component")
        seen: set[str] = set()
        for component in self.components:
            if component.dataset in seen:
                raise MixtureError(
                    f"dataset {component.dataset!r} appears twice; "
                    "combine the weights into one component"
                )
            seen.add(component.dataset)

    # ----------------------------------------------------------------- weights

    @property
    def total_weight(self) -> float:
        return sum(c.weight for c in self.components)

    def weights(self) -> dict[str, float]:
        """Normalised sampling probability per dataset. Sums to 1."""
        total = self.total_weight
        return {c.dataset: c.weight / total for c in self.components}

    def replay_share(self) -> float:
        """Fraction of sampling devoted to components marked as replay.

        The number to watch. Published guidance lands somewhere around
        5-30% depending on how far the new domain is from the old; a blend with
        0% replay is the one that forgets.
        """
        total = self.total_weight
        replay = sum(c.weight for c in self.components if c.role == ROLE_REPLAY)
        return replay / total

    @property
    def datasets(self) -> tuple[str, ...]:
        return tuple(c.dataset for c in self.components)

    # ---------------------------------------------------------------- evolution

    def with_component(
        self,
        dataset: str,
        weight: float,
        *,
        role: str = ROLE_DOMAIN,
        note: str = "",
    ) -> Mixture:
        """Return the next version with a component added or its weight replaced.

        Never mutates. The returned mixture records this one as its parent, so a
        run trained on v4 can be traced back through v3 to the original blend.
        """
        replaced = [c for c in self.components if c.dataset != dataset]
        replaced.append(Component(dataset=dataset, weight=weight, role=role))
        return Mixture(
            name=self.name,
            version=self.version + 1,
            components=tuple(replaced),
            parent_version=self.version,
            note=note,
            created_at=_now(),
        )

    def without_component(self, dataset: str, *, note: str = "") -> Mixture:
        """Return the next version with a component removed."""
        remaining = tuple(c for c in self.components if c.dataset != dataset)
        if len(remaining) == len(self.components):
            raise MixtureError(f"mixture {self.name!r} has no component {dataset!r}")
        if not remaining:
            raise MixtureError("cannot remove the last component of a mixture")
        return Mixture(
            name=self.name,
            version=self.version + 1,
            components=remaining,
            parent_version=self.version,
            note=note,
            created_at=_now(),
        )

    # -------------------------------------------------------------- persistence

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "parent_version": self.parent_version,
            "note": self.note,
            "created_at": self.created_at,
            "components": [c.to_dict() for c in self.components],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Mixture:
        return cls(
            name=str(payload["name"]),
            version=int(payload["version"]),
            components=tuple(Component.from_dict(c) for c in payload["components"]),
            parent_version=(
                None if payload.get("parent_version") is None else int(payload["parent_version"])
            ),
            note=str(payload.get("note", "")),
            created_at=str(payload.get("created_at", "")),
        )

    def describe(self) -> str:
        """One-line human summary, e.g. ``blend v3: domain 60% + general 25% (replay)``."""
        weights = self.weights()
        parts = []
        for component in sorted(self.components, key=lambda c: -c.weight):
            share = f"{weights[component.dataset] * 100:.0f}%"
            suffix = " (replay)" if component.role == ROLE_REPLAY else ""
            parts.append(f"{component.dataset} {share}{suffix}")
        return f"{self.name} v{self.version}: " + " + ".join(parts)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #

_VERSION_RE = re.compile(r"^v(\d+)\.json$")


def mixtures_root() -> Path:
    return paths.mixtures_dir()


def mixture_dir(name: str) -> Path:
    return mixtures_root() / paths.slug(name)


def create(
    name: str,
    components: list[Component],
    *,
    note: str = "",
) -> Mixture:
    """Build version 1 of a new mixture. Does not write it to disk."""
    return Mixture(
        name=name,
        version=1,
        components=tuple(components),
        parent_version=None,
        note=note,
        created_at=_now(),
    )


def single(dataset: str) -> Mixture:
    """A one-component mixture wrapping a plain dataset.

    Training always runs against a mixture. Treating a single dataset as a
    degenerate blend means there is one sampling path, one eval path and one set
    of metrics rather than two that drift apart.
    """
    return Mixture(
        name=dataset,
        version=1,
        components=(Component(dataset=dataset, weight=1.0),),
        created_at=_now(),
    )


def save(mixture: Mixture) -> Path:
    """Write a version. Refuses to overwrite an existing one.

    Versions are immutable by construction: reproducing run 7 requires that the
    blend it names still means what it meant.
    """
    directory = mixture_dir(mixture.name)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"v{mixture.version}.json"
    if path.exists():
        raise MixtureError(
            f"{mixture.name} v{mixture.version} already exists at {path}; "
            "versions are immutable, so create a new one instead"
        )
    path.write_text(json.dumps(mixture.to_dict(), indent=2) + "\n")
    return path


def versions(name: str) -> list[int]:
    """Every stored version number for a mixture, ascending."""
    directory = mixture_dir(name)
    if not directory.is_dir():
        return []
    found = []
    for path in directory.iterdir():
        match = _VERSION_RE.match(path.name)
        if match:
            found.append(int(match.group(1)))
    return sorted(found)


def load(name: str, version: int | None = None) -> Mixture:
    """Load a mixture. Defaults to the newest version."""
    available = versions(name)
    if not available:
        raise MixtureError(f"no mixture named {name!r}. Create one with `bloomery mix create`.")
    wanted = available[-1] if version is None else version
    if wanted not in available:
        listed = ", ".join(f"v{v}" for v in available)
        raise MixtureError(f"{name!r} has no v{wanted}; available: {listed}")
    path = mixture_dir(name) / f"v{wanted}.json"
    return Mixture.from_dict(json.loads(path.read_text()))


def list_all() -> list[Mixture]:
    """The newest version of every stored mixture, by name."""
    root = mixtures_root()
    if not root.is_dir():
        return []
    out = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        if versions(directory.name):
            out.append(load(directory.name))
    return out


def lineage(mixture: Mixture) -> list[Mixture]:
    """Walk back through parent versions, oldest first.

    Stops on a missing parent rather than raising: a mixture whose history was
    partly deleted is still usable, and a broken chain should not block a run.
    """
    chain = [mixture]
    current = mixture
    # Version files are plain JSON on disk and can be hand-edited, so a parent
    # pointer that loops back on itself is reachable. Without this guard that
    # would hang the CLI rather than fail.
    seen = {mixture.version}
    while current.parent_version is not None:
        if current.parent_version in seen:
            break
        try:
            current = load(current.name, current.parent_version)
        except (MixtureError, OSError, json.JSONDecodeError):
            break
        seen.add(current.version)
        chain.append(current)
    return list(reversed(chain))


def parse_spec(spec: str) -> tuple[str, float]:
    """Parse a ``name:weight`` CLI argument.

    Accepts ``domain:0.6``, ``domain=60``, ``domain:60%``. Weights are raw
    numbers, normalised later, so ``60/15/25`` and ``0.6/0.15/0.25`` are the
    same blend.
    """
    match = _SPEC_RE.match(spec)
    if not match:
        raise MixtureError(f"cannot parse {spec!r}; expected NAME:WEIGHT, for example domain:0.6")
    dataset, raw = match.group(1), match.group(2)
    weight = float(raw)
    if weight <= 0:
        raise MixtureError(f"weight for {dataset!r} must be positive, got {weight}")
    return dataset, weight
