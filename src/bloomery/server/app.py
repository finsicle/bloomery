# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The HTTP API.

A thin layer over :mod:`bloomery.jobs`. Scheduling, cancellation and process
supervision all live there and are not reimplemented here — this module turns
requests into supervisor calls and results into JSON, and nothing else. Anything
that looks like policy in here is a bug.

Section 13 of the AGPL applies to this file more than any other in the project:
running a modified version and letting people interact with it over a network
obliges you to offer them the source. That is why ``/api/source`` exists and why
every response carries a link to it — the obligation attaches to the served
surface, not to the repository the operator happens to have cloned.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from bloomery import paths
from bloomery.jobs import JobKind, JobStatus, JobStore, Supervisor
from bloomery.server.events import EventHub

log = logging.getLogger(__name__)

# Where the corresponding source can be obtained. Overridable because an operator
# running a modified build must be able to point at *their* source, which is
# precisely what section 13 requires of them.
ENV_SOURCE_URL = "BLOOMERY_SOURCE_URL"
DEFAULT_SOURCE_URL = "https://github.com/finsicle/bloomery"

API_PREFIX = "/api"


# --------------------------------------------------------------------------- #
# request models
# --------------------------------------------------------------------------- #


class ResourceSpec(BaseModel):
    cores: int | None = Field(default=None, ge=1)
    memory_bytes: int | None = Field(default=None, ge=1)
    gpus: list[int] = Field(default_factory=list)

    @field_validator("gpus")
    @classmethod
    def _non_negative(cls, value: list[int]) -> list[int]:
        if any(index < 0 for index in value):
            raise ValueError("gpu indices must be non-negative")
        return value


class JobRequest(BaseModel):
    """A job to queue.

    ``params`` is deliberately free-form: the runner already filters it against a
    fixed table of known flags, so an unknown key is dropped rather than reaching
    a command line. Duplicating that whitelist here would give two places to keep
    in step and no extra safety.
    """

    kind: JobKind
    params: dict[str, Any] = Field(default_factory=dict)
    name: str = Field(default="", max_length=200)
    resources: ResourceSpec | None = None


# --------------------------------------------------------------------------- #
# application state
# --------------------------------------------------------------------------- #


class ServerState:
    """Everything a request handler may need, created once at startup."""

    def __init__(
        self,
        *,
        store: JobStore | None = None,
        home: Path | None = None,
        supervisor: Supervisor | None = None,
    ) -> None:
        self.home = home or paths.home()
        self.store = store or JobStore()
        self.hub = EventHub()
        self.supervisor = supervisor or Supervisor(
            store=self.store,
            home=self.home,
            on_change=self._on_job_change,
        )
        # A supervisor supplied by a caller (tests, mostly) still needs its
        # events routed here, or the socket would stay silent.
        if supervisor is not None and supervisor.on_change is None:
            supervisor.on_change = self._on_job_change

    def _on_job_change(self, job: Any) -> None:
        """Called on the supervisor's thread, never the event loop's."""
        self.hub.publish_threadsafe({"type": "job", "job": job.to_dict()})

    def source_url(self) -> str:
        return os.environ.get(ENV_SOURCE_URL, "").strip() or DEFAULT_SOURCE_URL


def get_state(app: FastAPI) -> ServerState:
    state: ServerState = app.state.bloomery
    return state


# --------------------------------------------------------------------------- #
# the app
# --------------------------------------------------------------------------- #


def create_app(
    *,
    store: JobStore | None = None,
    home: Path | None = None,
    supervisor: Supervisor | None = None,
    start_supervisor: bool = True,
) -> FastAPI:
    """Build the application.

    Constructed by a factory rather than at import time so tests can supply their
    own store and home, and so importing this module does not touch the user's
    filesystem.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        state = get_state(app)
        state.hub.bind(asyncio.get_running_loop())
        if start_supervisor:
            state.supervisor.start()
        try:
            yield
        finally:
            # Running jobs are deliberately left alone: a training run that
            # outlives the server is a feature, and reconcile() accounts for it
            # on the next start.
            state.supervisor.stop()
            await state.hub.close()

    app = FastAPI(
        title="bloomery",
        summary="Train a small language model on the hardware you already own.",
        license_info={
            "name": "AGPL-3.0-or-later",
            "url": "https://www.gnu.org/licenses/agpl-3.0.html",
        },
        lifespan=lifespan,
    )
    app.state.bloomery = ServerState(store=store, home=home, supervisor=supervisor)

    def state() -> ServerState:
        return get_state(app)

    _register_routes(app, state)
    return app


def _register_routes(app: FastAPI, state: Any) -> None:
    Dep = Depends(state)

    # ----------------------------------------------------------------- licence

    @app.get(f"{API_PREFIX}/source")
    def source(st: ServerState = Dep) -> dict[str, str]:
        """Where to get the source for the version running here.

        Required by AGPL section 13, which is not satisfied by the licence text
        alone: anyone interacting with this server over a network must be offered
        the corresponding source of *this* build.
        """
        return {
            "license": "AGPL-3.0-or-later",
            "license_url": "https://www.gnu.org/licenses/agpl-3.0.html",
            "source_url": st.source_url(),
            "notice": (
                "This is free software. You may use, study, modify and share it. "
                "If you run a modified version and let others use it over a "
                "network, you must offer them its source."
            ),
        }

    @app.get("/health")
    def health(st: ServerState = Dep) -> dict[str, Any]:
        return {"status": "ok", "jobs": st.store.counts()}

    # -------------------------------------------------------------------- jobs

    @app.get(f"{API_PREFIX}/jobs")
    def list_jobs(
        st: ServerState = Dep,
        status: JobStatus | None = None,
        kind: JobKind | None = None,
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> dict[str, Any]:
        jobs = st.store.find(status=status, kind=kind, limit=limit)
        return {"jobs": [job.to_dict() for job in jobs], "counts": st.store.counts()}

    @app.post(f"{API_PREFIX}/jobs", status_code=201)
    def create_job(request: JobRequest, st: ServerState = Dep) -> dict[str, Any]:
        params = dict(request.params)
        if request.resources is not None:
            params["resources"] = request.resources.model_dump()
        job = st.supervisor.submit(request.kind, params, name=request.name)
        return job.to_dict()

    @app.get(f"{API_PREFIX}/jobs/{{job_id}}")
    def get_job(job_id: str, st: ServerState = Dep) -> dict[str, Any]:
        job = st.store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no such job: {job_id}")
        return job.to_dict()

    @app.delete(f"{API_PREFIX}/jobs/{{job_id}}")
    def cancel_job(job_id: str, st: ServerState = Dep) -> dict[str, Any]:
        job = st.store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no such job: {job_id}")
        changed = st.supervisor.cancel(job_id)
        if not changed:
            raise HTTPException(
                status_code=409,
                detail=f"job {job_id} already finished with status {job.status.value}",
            )
        current = st.store.get(job_id)
        return current.to_dict() if current else {"id": job_id, "status": "cancelled"}

    @app.get(f"{API_PREFIX}/jobs/{{job_id}}/log")
    def job_log(
        job_id: str,
        st: ServerState = Dep,
        tail: int = Query(default=200, ge=1, le=10_000),
    ) -> dict[str, Any]:
        job = st.store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no such job: {job_id}")
        if not job.log_path:
            return {"id": job_id, "lines": []}
        path = Path(job.log_path)
        if not path.is_file():
            return {"id": job_id, "lines": []}
        # errors="replace" because a log is diagnostic output, and refusing to
        # show it because one byte is malformed helps nobody.
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        return {"id": job_id, "lines": lines[-tail:], "truncated": len(lines) > tail}

    # -------------------------------------------------------------------- host

    @app.get(f"{API_PREFIX}/host")
    def host() -> dict[str, Any]:
        """What this machine is, as the doctor command reports it."""
        from bloomery.probe import probe_host_report

        return probe_host_report().to_dict()

    # ------------------------------------------------------------------ stream

    @app.websocket(f"{API_PREFIX}/stream")
    async def stream(websocket: WebSocket) -> None:
        """Live job transitions.

        The first message is a snapshot, so a client that connects mid-run does
        not have to wait for the next transition to know what is happening.
        """
        st = get_state(app)
        await websocket.accept()
        try:
            async with st.hub.subscribe() as queue:
                await websocket.send_json(
                    {
                        "type": "snapshot",
                        "jobs": [job.to_dict() for job in st.store.find(limit=100)],
                        "source_url": st.source_url(),
                    }
                )
                while True:
                    event = await queue.get()
                    await websocket.send_json(event)
        except WebSocketDisconnect:
            return
        except Exception:  # noqa: BLE001 - a dead socket must not log a stack
            log.debug("stream closed", exc_info=True)
            return

    # ----------------------------------------------------------------- licence

    @app.middleware("http")
    async def add_source_header(request: Any, call_next: Any) -> Any:
        """Advertise the source on every response.

        A link the user has to go looking for is a link most will never see, and
        section 13 is about the offer being made rather than being available on
        request.
        """
        response = await call_next(request)
        response.headers["X-Bloomery-Source"] = get_state(app).source_url()
        response.headers["X-Bloomery-License"] = "AGPL-3.0-or-later"
        return response

    @app.exception_handler(ValueError)
    async def value_error(request: Any, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})
