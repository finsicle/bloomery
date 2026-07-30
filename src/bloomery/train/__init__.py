# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Training: device selection, the loop, checkpoints and the metrics protocol."""

from __future__ import annotations

from bloomery.train.device import DeviceChoice, choose, thread_limit
from bloomery.train.metrics import MetricsWriter, Throughput, read_events

__all__ = [
    "DeviceChoice",
    "MetricsWriter",
    "Throughput",
    "choose",
    "read_events",
    "thread_limit",
]
