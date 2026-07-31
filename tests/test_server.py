# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the HTTP API and the event stream.

The supervisor is left stopped for most of these: what is under test here is the
translation between requests and supervisor calls, and a scheduler running in the
background would make that non-deterministic. The cases that genuinely need a
process to run say so.

No pytest-asyncio: the few coroutines here are driven with ``asyncio.run``, which
is one less dependency for a handful of tests.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi", reason="the serve extra is not installed")

from fastapi.testclient import TestClient  # noqa: E402

from bloomery.jobs import JobKind, JobStatus, JobStore, Supervisor  # noqa: E402
from bloomery.server.app import (  # noqa: E402
    DEFAULT_SOURCE_URL,
    ENV_SOURCE_URL,
    create_app,
)
from bloomery.server.events import EventHub  # noqa: E402


@pytest.fixture
def store(tmp_path: Path) -> JobStore:
    return JobStore(tmp_path / "jobs.db")


@pytest.fixture
def client(store: JobStore, tmp_path: Path) -> Any:
    """A client whose supervisor never runs, so nothing spawns."""
    app = create_app(store=store, home=tmp_path, start_supervisor=False)
    with TestClient(app) as test_client:
        yield test_client


# --------------------------------------------------------------------------- #
# AGPL section 13
# --------------------------------------------------------------------------- #


class TestSourceOffer:
    """Section 13 obliges an operator to offer source to network users.

    The licence file alone does not discharge it: someone using this over a
    network never sees the repository. So the offer has to be part of what is
    served.
    """

    def test_source_endpoint(self, client: Any) -> None:
        payload = client.get("/api/source").json()
        assert payload["license"] == "AGPL-3.0-or-later"
        assert payload["source_url"] == DEFAULT_SOURCE_URL
        assert "network" in payload["notice"]

    def test_every_response_advertises_the_source(self, client: Any) -> None:
        for path in ("/health", "/api/jobs", "/api/source"):
            response = client.get(path)
            assert response.headers["X-Bloomery-Source"]
            assert response.headers["X-Bloomery-License"] == "AGPL-3.0-or-later"

    def test_a_modified_build_can_point_at_its_own_source(
        self, client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The obligation falls on whoever runs the modified version."""
        monkeypatch.setenv(ENV_SOURCE_URL, "https://example.invalid/my-fork")
        response = client.get("/api/source")
        assert response.json()["source_url"] == "https://example.invalid/my-fork"
        assert response.headers["X-Bloomery-Source"] == "https://example.invalid/my-fork"

    def test_a_blank_override_falls_back(
        self, client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ENV_SOURCE_URL, "   ")
        assert client.get("/api/source").json()["source_url"] == DEFAULT_SOURCE_URL

    def test_the_stream_offers_it_too(self, client: Any) -> None:
        """A client that only ever opens the socket still gets the offer."""
        with client.websocket_connect("/api/stream") as socket:
            assert socket.receive_json()["source_url"] == DEFAULT_SOURCE_URL


# --------------------------------------------------------------------------- #
# jobs
# --------------------------------------------------------------------------- #


class TestJobsApi:
    def test_health(self, client: Any) -> None:
        payload = client.get("/health").json()
        assert payload["status"] == "ok"

    def test_create_returns_201_and_the_queued_job(self, client: Any) -> None:
        response = client.post(
            "/api/jobs",
            json={"kind": "prepare", "name": "first", "params": {"synthetic": 10}},
        )
        assert response.status_code == 201
        job = response.json()
        assert job["status"] == "queued"
        assert job["kind"] == "prepare"
        assert job["name"] == "first"

    def test_created_jobs_are_listed(self, client: Any) -> None:
        client.post("/api/jobs", json={"kind": "prepare", "params": {}})
        client.post("/api/jobs", json={"kind": "train", "params": {}})
        payload = client.get("/api/jobs").json()
        assert len(payload["jobs"]) == 2
        assert payload["counts"]["queued"] == 2

    def test_list_filters_by_kind_and_status(self, client: Any) -> None:
        client.post("/api/jobs", json={"kind": "prepare", "params": {}})
        client.post("/api/jobs", json={"kind": "train", "params": {}})
        assert len(client.get("/api/jobs", params={"kind": "train"}).json()["jobs"]) == 1
        assert len(client.get("/api/jobs", params={"status": "queued"}).json()["jobs"]) == 2
        assert len(client.get("/api/jobs", params={"status": "succeeded"}).json()["jobs"]) == 0

    def test_get_one(self, client: Any) -> None:
        created = client.post("/api/jobs", json={"kind": "prepare", "params": {}}).json()
        assert client.get(f"/api/jobs/{created['id']}").json()["id"] == created["id"]

    def test_unknown_job_is_404(self, client: Any) -> None:
        response = client.get("/api/jobs/nope")
        assert response.status_code == 404
        assert "no such job" in response.json()["detail"]

    def test_cancel_a_queued_job(self, client: Any) -> None:
        created = client.post("/api/jobs", json={"kind": "train", "params": {}}).json()
        response = client.delete(f"/api/jobs/{created['id']}")
        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"

    def test_cancelling_a_finished_job_is_409(self, client: Any, store: JobStore) -> None:
        """Not 404 — the job exists; the request is what does not apply."""
        created = client.post("/api/jobs", json={"kind": "train", "params": {}}).json()
        store.mark_finished(created["id"], JobStatus.SUCCEEDED, exit_code=0)
        response = client.delete(f"/api/jobs/{created['id']}")
        assert response.status_code == 409
        assert "already finished" in response.json()["detail"]

    def test_cancelling_an_unknown_job_is_404(self, client: Any) -> None:
        assert client.delete("/api/jobs/nope").status_code == 404

    def test_resources_reach_the_job(self, client: Any, store: JobStore) -> None:
        created = client.post(
            "/api/jobs",
            json={
                "kind": "train",
                "params": {"data": "corpus"},
                "resources": {"cores": 4, "gpus": [0, 1]},
            },
        ).json()
        stored = store.get(created["id"])
        assert stored.params["resources"]["cores"] == 4
        assert stored.params["resources"]["gpus"] == [0, 1]

    def test_an_unknown_kind_is_rejected(self, client: Any) -> None:
        assert client.post("/api/jobs", json={"kind": "rm -rf", "params": {}}).status_code == 422

    def test_negative_gpu_indices_are_rejected(self, client: Any) -> None:
        response = client.post(
            "/api/jobs",
            json={"kind": "train", "params": {}, "resources": {"gpus": [-1]}},
        )
        assert response.status_code == 422

    def test_unknown_params_are_accepted_and_filtered_later(
        self, client: Any, store: JobStore
    ) -> None:
        """The runner's flag table is the single gate; duplicating it here would
        give two things to keep in step and no extra safety."""
        created = client.post(
            "/api/jobs",
            json={"kind": "train", "params": {"data": "c", "--rm": "-rf /"}},
        ).json()
        assert store.get(created["id"]).params["--rm"] == "-rf /"

        from bloomery.jobs import runner

        assert "--rm" not in runner.build_command(store.get(created["id"]))

    def test_limit_is_bounded(self, client: Any) -> None:
        assert client.get("/api/jobs", params={"limit": 0}).status_code == 422
        assert client.get("/api/jobs", params={"limit": 100_000}).status_code == 422


class TestJobLog:
    def test_missing_log_is_an_empty_list_not_an_error(self, client: Any) -> None:
        created = client.post("/api/jobs", json={"kind": "prepare", "params": {}}).json()
        assert client.get(f"/api/jobs/{created['id']}/log").json()["lines"] == []

    def test_unknown_job_is_404(self, client: Any) -> None:
        assert client.get("/api/jobs/nope/log").status_code == 404

    def test_tail_returns_the_end_and_says_it_truncated(
        self, client: Any, store: JobStore, tmp_path: Path
    ) -> None:
        created = client.post("/api/jobs", json={"kind": "prepare", "params": {}}).json()
        log = tmp_path / "job.log"
        log.write_text("\n".join(f"line {i}" for i in range(50)))
        job = store.get(created["id"])
        job.log_path = str(log)
        store.update(job)

        payload = client.get(f"/api/jobs/{created['id']}/log", params={"tail": 5}).json()
        assert payload["lines"] == [f"line {i}" for i in range(45, 50)]
        assert payload["truncated"] is True

    def test_undecodable_bytes_do_not_hide_the_log(
        self, client: Any, store: JobStore, tmp_path: Path
    ) -> None:
        """A log is diagnostic output; refusing to show it over one bad byte helps nobody."""
        created = client.post("/api/jobs", json={"kind": "prepare", "params": {}}).json()
        log = tmp_path / "job.log"
        log.write_bytes(b"before\n\xff\xfe bad bytes\nafter\n")
        job = store.get(created["id"])
        job.log_path = str(log)
        store.update(job)

        payload = client.get(f"/api/jobs/{created['id']}/log").json()
        assert "before" in payload["lines"]
        assert "after" in payload["lines"]


class TestHost:
    def test_reports_this_machine(self, client: Any) -> None:
        payload = client.get("/api/host").json()
        assert payload["host"]["system"]
        assert "gpus" in payload
        assert "memory" in payload


# --------------------------------------------------------------------------- #
# the stream
# --------------------------------------------------------------------------- #


class TestStream:
    def test_snapshot_arrives_first(self, client: Any) -> None:
        """A client connecting mid-run should not have to wait for a transition."""
        client.post("/api/jobs", json={"kind": "prepare", "params": {}})
        with client.websocket_connect("/api/stream") as socket:
            first = socket.receive_json()
        assert first["type"] == "snapshot"
        assert len(first["jobs"]) == 1

    def test_transitions_are_pushed(self, client: Any, store: JobStore) -> None:
        with client.websocket_connect("/api/stream") as socket:
            socket.receive_json()  # snapshot
            created = client.post("/api/jobs", json={"kind": "prepare", "params": {}}).json()
            event = socket.receive_json()
        assert event["type"] == "job"
        assert event["job"]["id"] == created["id"]
        assert event["job"]["status"] == "queued"


class TestEventHub:
    def test_publishing_before_bind_is_harmless(self) -> None:
        """The supervisor may report a change before the loop exists."""
        hub = EventHub()
        hub.publish_threadsafe({"type": "job"})  # must not raise

    def test_each_subscriber_gets_every_event(self) -> None:
        async def scenario() -> tuple[int, int]:
            hub = EventHub()
            hub.bind()
            async with hub.subscribe() as first, hub.subscribe() as second:
                hub.publish_threadsafe({"n": 1})
                await asyncio.sleep(0.05)
                return first.qsize(), second.qsize()

        assert asyncio.run(scenario()) == (1, 1)

    def test_a_slow_client_drops_its_own_oldest(self) -> None:
        """One stalled browser must not grow the server without limit."""

        async def scenario() -> tuple[int, dict[str, Any]]:
            hub = EventHub(queue_limit=4)
            hub.bind()
            async with hub.subscribe() as queue:
                for n in range(10):
                    hub.publish_threadsafe({"n": n})
                await asyncio.sleep(0.05)
                size = queue.qsize()
                newest = None
                while not queue.empty():
                    newest = queue.get_nowait()
                return size, newest

        size, newest = asyncio.run(scenario())
        assert size == 4, "queue grew past its limit"
        assert newest["n"] == 9, "kept the stale events instead of the fresh ones"

    def test_unsubscribing_stops_delivery(self) -> None:
        async def scenario() -> int:
            hub = EventHub()
            hub.bind()
            async with hub.subscribe():
                pass
            hub.publish_threadsafe({"n": 1})
            await asyncio.sleep(0.05)
            return hub.subscriber_count

        assert asyncio.run(scenario()) == 0


# --------------------------------------------------------------------------- #
# wiring
# --------------------------------------------------------------------------- #


class TestWiring:
    def test_the_package_does_not_import_fastapi_eagerly(self) -> None:
        """`bloomery doctor` must not pay for a web framework it never uses."""
        import subprocess
        import sys

        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, bloomery.server; print('fastapi' in sys.modules)",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
        assert completed.stdout.strip() == "False"

    def test_a_supplied_supervisor_still_feeds_the_stream(
        self, store: JobStore, tmp_path: Path
    ) -> None:
        supervisor = Supervisor(store=store, home=tmp_path)
        app = create_app(store=store, home=tmp_path, supervisor=supervisor, start_supervisor=False)
        with TestClient(app) as client, client.websocket_connect("/api/stream") as socket:
            socket.receive_json()  # snapshot
            supervisor.submit(JobKind.PREPARE, {})
            assert socket.receive_json()["job"]["kind"] == "prepare"

    def test_the_openapi_schema_builds(self, client: Any) -> None:
        schema = client.get("/openapi.json").json()
        assert schema["info"]["license"]["name"] == "AGPL-3.0-or-later"
        assert "/api/jobs" in schema["paths"]


class TestRealJobOverHttp:
    """One genuine run through the whole stack, because stubs prove nothing here."""

    def test_a_job_submitted_over_http_actually_runs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import time

        home = tmp_path / "home"
        monkeypatch.setenv("BLOOMERY_HOME", str(home))
        store = JobStore(home / "jobs.db")
        supervisor = Supervisor(store=store, home=home, poll_seconds=0.1)
        app = create_app(store=store, home=home, supervisor=supervisor, start_supervisor=True)

        with TestClient(app) as client:
            created = client.post(
                "/api/jobs",
                json={
                    "kind": "prepare",
                    "params": {"name": "over-http", "synthetic": 300, "vocab": 300},
                },
            ).json()

            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                current = client.get(f"/api/jobs/{created['id']}").json()
                if current["status"] in ("succeeded", "failed", "cancelled", "interrupted"):
                    break
                time.sleep(0.3)

            assert current["status"] == "succeeded", (
                current.get("error"),
                client.get(f"/api/jobs/{created['id']}/log").json()["lines"][-10:],
            )
            assert (home / "datasets" / "over-http" / "tokens" / "meta.json").is_file()
