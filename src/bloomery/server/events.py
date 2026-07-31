# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fan-out of job events to connected clients.

The supervisor already reports every state change through its ``on_change`` hook,
so this subscribes to that rather than polling the database. Polling would be
both slower to react and heavier at rest, and the interesting states — a job
starting, a job dying — are exactly the ones a poll interval smears over.

The hook fires on the supervisor's thread, which is not the event loop's thread.
Everything here is therefore written to be safe to call from either, and the
handoff happens through the loop's own scheduling.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from typing import Any

log = logging.getLogger(__name__)

# How many events a slow client may fall behind before it starts losing them.
# A browser that stops reading must not be able to grow the server's memory
# without limit, and for a live view the newest events are the ones that matter.
QUEUE_LIMIT = 256


class EventHub:
    """Broadcasts events to every current subscriber.

    Each subscriber gets its own bounded queue. One client on a slow connection
    therefore cannot hold up the others, and cannot make the server buffer without
    limit — it drops its own oldest events instead.
    """

    def __init__(self, *, queue_limit: int = QUEUE_LIMIT) -> None:
        self.queue_limit = queue_limit
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = asyncio.Lock()
        # Set on close. Readers wait on this *alongside* their queue, so a
        # handler parked on an empty queue still learns that shutdown started.
        self._closing = asyncio.Event()

    def bind(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Record the loop that publishes must be marshalled onto.

        Called during startup, from the loop's own thread.
        """
        self._loop = loop or asyncio.get_running_loop()

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    # ------------------------------------------------------------------ publish

    def publish_threadsafe(self, event: dict[str, Any]) -> None:
        """Publish from any thread, including the supervisor's.

        This is what the supervisor's ``on_change`` hook calls. It must never
        raise into the caller: a broken listener that killed the scheduler would
        turn a display problem into a training problem.
        """
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(self._publish, event)
        except RuntimeError:  # pragma: no cover - loop shut down mid-call
            log.debug("event dropped: loop is gone")

    def _publish(self, event: dict[str, Any]) -> None:
        """Deliver to every subscriber. Runs on the event loop's thread."""
        for queue in list(self._subscribers):
            if queue.full():
                # Drop this subscriber's oldest rather than blocking everyone
                # else. A live view wants the newest events, not a complete one.
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(event)

    # ---------------------------------------------------------------- subscribe

    @contextlib.asynccontextmanager
    async def subscribe(self) -> AsyncIterator[asyncio.Queue[dict[str, Any]]]:
        """Register a queue for the caller's lifetime."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self.queue_limit)
        async with self._lock:
            self._subscribers.add(queue)
        try:
            yield queue
        finally:
            async with self._lock:
                self._subscribers.discard(queue)

    async def next_event(self, queue: asyncio.Queue[dict[str, Any]]) -> dict[str, Any] | None:
        """The next event for this subscriber, or None once the hub is closing.

        A reader parked on ``queue.get()`` has no way to notice shutdown on its
        own. Uvicorn waits for active tasks *before* running lifespan teardown,
        so a websocket handler blocked here would sit through the whole graceful
        shutdown period and then be force-cancelled — every restart paying that
        timeout, and a client dropped rather than closed.

        The wake-up is a separate flag rather than a sentinel pushed onto the
        queue, because a full queue rejects ``put_nowait``: a sentinel would go
        missing exactly when the server is busiest, which is the worst time for
        shutdown to hang.
        """
        if self._closing.is_set():
            # Anything still buffered is dropped. These are status updates for a
            # live view and the connection is ending either way; draining them
            # into a socket that may not be reading is the stall this method
            # exists to prevent.
            return None

        getter: asyncio.Task[dict[str, Any]] = asyncio.ensure_future(queue.get())
        closing: asyncio.Task[bool] = asyncio.ensure_future(self._closing.wait())
        try:
            await asyncio.wait({getter, closing}, return_when=asyncio.FIRST_COMPLETED)
            # Shutdown wins a tie. Both can finish together when close() lands
            # while an event is queued, and handing that event back would put a
            # send on a socket we already know is going away — a smaller version
            # of the stall this method exists to prevent. It would also
            # contradict the drop-what-is-buffered rule above, for no reason
            # beyond which task the loop happened to schedule first.
            if self._closing.is_set():
                return None
            if getter.done() and not getter.cancelled():
                return getter.result()
            return None
        finally:
            for task in (getter, closing):
                if not task.done():
                    task.cancel()
            # Awaiting the cancellations keeps "Task was destroyed but it is
            # pending" out of the log on shutdown.
            await asyncio.gather(getter, closing, return_exceptions=True)

    async def close(self) -> None:
        """Wake every reader and drop them.

        Setting the flag before clearing the subscribers, so a reader that wakes
        immediately finds the hub already closing rather than racing the set.
        """
        self._closing.set()
        async with self._lock:
            self._subscribers.clear()
