# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""HTTP API and live event stream over the job engine."""

from __future__ import annotations

from typing import Any

__all__ = ["EventHub", "create_app"]


def __getattr__(name: str) -> Any:
    """Import lazily so the core CLI does not pay for FastAPI.

    ``bloomery serve`` is an optional extra. Importing this package eagerly would
    make every ``bloomery doctor`` run load a web framework it will never use,
    and would turn a missing optional dependency into an import error at startup.
    """
    if name == "create_app":
        from bloomery.server.app import create_app

        return create_app
    if name == "EventHub":
        from bloomery.server.events import EventHub

        return EventHub
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
