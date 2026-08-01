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
import time
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

    def test_a_windows_log_does_not_come_back_with_stray_carriage_returns(
        self, client: Any, store: JobStore, tmp_path: Path
    ) -> None:
        """Runners write logs on every OS in the matrix, including CRLF ones."""
        created = client.post("/api/jobs", json={"kind": "prepare", "params": {}}).json()
        log = tmp_path / "job.log"
        log.write_bytes(b"first\r\nsecond\r\n")
        job = store.get(created["id"])
        job.log_path = str(log)
        store.update(job)

        assert client.get(f"/api/jobs/{created['id']}/log").json()["lines"] == [
            "first",
            "second",
        ]

    def test_a_progress_bar_redraw_is_not_a_new_log_line(
        self, client: Any, store: JobStore, tmp_path: Path
    ) -> None:
        """The deliberate difference from the old splitlines() reader.

        A bare CR is how every progress bar redraws its line. splitlines() broke
        on it, so one tqdm bar arrived as thousands of log entries that were
        never written. Only a newline ends a line now.
        """
        created = client.post("/api/jobs", json={"kind": "prepare", "params": {}}).json()
        log = tmp_path / "job.log"
        log.write_bytes(b"epoch 1: 10%\repoch 1: 90%\repoch 1: done\nsaved\n")
        job = store.get(created["id"])
        job.log_path = str(log)
        store.update(job)

        assert client.get(f"/api/jobs/{created['id']}/log").json()["lines"] == [
            "epoch 1: 10%\repoch 1: 90%\repoch 1: done",
            "saved",
        ]

    def test_form_feeds_and_unicode_separators_do_not_invent_lines(
        self, client: Any, store: JobStore, tmp_path: Path
    ) -> None:
        """splitlines() also broke on these; the byte-level split does not."""
        created = client.post("/api/jobs", json={"kind": "prepare", "params": {}}).json()
        log = tmp_path / "job.log"
        log.write_text("step 1\x0c50%\nstep 2\u2028done\n", encoding="utf-8")
        job = store.get(created["id"])
        job.log_path = str(log)
        store.update(job)

        assert client.get(f"/api/jobs/{created['id']}/log").json()["lines"] == [
            "step 1\x0c50%",
            "step 2\u2028done",
        ]

    def test_a_tail_of_a_large_log_reads_only_the_end(
        self, client: Any, store: JobStore, tmp_path: Path
    ) -> None:
        """Job logs are never rotated, so one only grows. Rereading it per poll is the bug."""
        created = client.post("/api/jobs", json={"kind": "prepare", "params": {}}).json()
        log = tmp_path / "job.log"
        log.write_text("".join(f"line {i}\n" for i in range(200_000)), encoding="utf-8")
        job = store.get(created["id"])
        job.log_path = str(log)
        store.update(job)

        payload = client.get(f"/api/jobs/{created['id']}/log", params={"tail": 3}).json()
        assert payload["lines"] == ["line 199997", "line 199998", "line 199999"]
        assert payload["truncated"] is True

    def test_a_line_longer_than_the_first_read_still_comes_back_whole(
        self, client: Any, store: JobStore, tmp_path: Path
    ) -> None:
        """The read budget grows until it has the lines; it must not return a fragment."""
        created = client.post("/api/jobs", json={"kind": "prepare", "params": {}}).json()
        log = tmp_path / "job.log"
        log.write_text("head\n" + ("x" * 300_000) + "\ntail line\n", encoding="utf-8")
        job = store.get(created["id"])
        job.log_path = str(log)
        store.update(job)

        payload = client.get(f"/api/jobs/{created['id']}/log", params={"tail": 2}).json()
        assert payload["lines"] == ["x" * 300_000, "tail line"]
        assert payload["truncated"] is True

    def test_a_whole_short_log_is_not_reported_as_truncated(
        self, client: Any, store: JobStore, tmp_path: Path
    ) -> None:
        created = client.post("/api/jobs", json={"kind": "prepare", "params": {}}).json()
        log = tmp_path / "job.log"
        log.write_text("only line\n", encoding="utf-8")
        job = store.get(created["id"])
        job.log_path = str(log)
        store.update(job)

        payload = client.get(f"/api/jobs/{created['id']}/log", params={"tail": 10}).json()
        assert payload["lines"] == ["only line"]
        assert payload["truncated"] is False

    def test_an_empty_log_is_not_truncated(
        self, client: Any, store: JobStore, tmp_path: Path
    ) -> None:
        created = client.post("/api/jobs", json={"kind": "prepare", "params": {}}).json()
        log = tmp_path / "job.log"
        log.write_bytes(b"")
        job = store.get(created["id"])
        job.log_path = str(log)
        store.update(job)

        payload = client.get(f"/api/jobs/{created['id']}/log").json()
        assert payload["lines"] == []
        assert payload["truncated"] is False

    def test_a_log_without_a_final_newline_keeps_its_last_line(
        self, client: Any, store: JobStore, tmp_path: Path
    ) -> None:
        """A log being appended to right now has no trailing newline yet."""
        created = client.post("/api/jobs", json={"kind": "prepare", "params": {}}).json()
        log = tmp_path / "job.log"
        log.write_text("first\nstill writing", encoding="utf-8")
        job = store.get(created["id"])
        job.log_path = str(log)
        store.update(job)

        assert client.get(f"/api/jobs/{created['id']}/log").json()["lines"] == [
            "first",
            "still writing",
        ]

    def test_multibyte_characters_survive_a_seek_into_the_middle(
        self, client: Any, store: JobStore, tmp_path: Path
    ) -> None:
        """The backward read seeks to an arbitrary byte; it must not cut a character in half."""
        created = client.post("/api/jobs", json={"kind": "prepare", "params": {}}).json()
        log = tmp_path / "job.log"
        log.write_text("".join(f"日本語のログ行 {i}\n" for i in range(5_000)), encoding="utf-8")
        job = store.get(created["id"])
        job.log_path = str(log)
        store.update(job)

        payload = client.get(f"/api/jobs/{created['id']}/log", params={"tail": 4}).json()
        assert payload["lines"] == [f"日本語のログ行 {i}" for i in range(4996, 5000)]
        assert "�" not in "".join(payload["lines"])


class TestHost:
    def test_reports_this_machine(self, client: Any) -> None:
        payload = client.get("/api/host").json()
        assert payload["host"]["system"]
        assert "gpus" in payload
        assert "memory" in payload


# --------------------------------------------------------------------------- #
# the web ui
# --------------------------------------------------------------------------- #


class TestWebUi:
    """The UI is static files mounted at the root.

    A mount at "/" matches every path, so the thing most worth testing is not
    that the page loads but that mounting it did not swallow the API.
    """

    def test_the_root_serves_the_page(self, client: Any) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert "bloomery" in response.text

    def test_the_api_still_answers_past_the_catch_all_mount(self, client: Any) -> None:
        """The regression guard. Mount the UI too early and every one of these 404s."""
        assert client.get("/health").json()["status"] == "ok"
        assert "jobs" in client.get("/api/jobs").json()
        assert client.get("/api/source").json()["license"] == "AGPL-3.0-or-later"
        assert client.get("/api/host").status_code == 200

    def test_an_unknown_path_is_still_a_404(self, client: Any) -> None:
        """html=True must not turn every typo into the index page."""
        assert client.get("/no-such-page").status_code == 404

    def test_the_assets_are_served(self, client: Any) -> None:
        for path, kind in (
            ("/app.js", "javascript"),
            ("/app.css", "css"),
            # app.js imports this one as a module, so a 404 here is a blank page.
            ("/format.js", "javascript"),
        ):
            response = client.get(path)
            assert response.status_code == 200, path
            assert kind in response.headers["content-type"], response.headers["content-type"]

    def test_the_page_carries_the_source_offer_too(self, client: Any) -> None:
        """Section 13 attaches to the served surface, and this is the surface."""
        response = client.get("/")
        assert response.headers["X-Bloomery-Source"] == DEFAULT_SOURCE_URL
        assert response.headers["X-Bloomery-License"] == "AGPL-3.0-or-later"

    def test_every_asset_the_page_needs_is_present(self) -> None:
        """The set of files STATIC_DIR must contain for the page to work at all.

        This reads the source tree, not a wheel: STATIC_DIR resolves inside
        src/ under an editable install, which is how the whole matrix runs. So
        it pins the file list, not the packaging. The `verify the web ui ships`
        step in .github/workflows/ci.yml is what inspects a built wheel, and it
        is the only thing that can — nothing here would notice a wheel that
        shipped without these.
        """
        from bloomery.server.app import STATIC_DIR

        assert (STATIC_DIR / "index.html").is_file()
        assert (STATIC_DIR / "app.js").is_file()
        assert (STATIC_DIR / "app.css").is_file()
        assert (STATIC_DIR / "format.js").is_file()

    def test_the_page_knows_how_much_history_the_snapshot_carries(self) -> None:
        """The page tells the user when it is showing a truncated list.

        It can only say that if its number matches the server's. If the server
        raised SNAPSHOT_LIMIT and the page did not, the notice would appear
        while jobs were still arriving; lower it and the notice never appears
        at all, so the missing jobs go unmentioned.
        """
        import re

        from bloomery.server.app import SNAPSHOT_LIMIT, STATIC_DIR

        source = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
        declared = re.search(r"const HISTORY_LIMIT = (\d+);", source)
        assert declared, "app.js does not declare HISTORY_LIMIT"
        assert int(declared.group(1)) == SNAPSHOT_LIMIT

    def test_the_form_and_the_runner_offer_the_same_parameters(self) -> None:
        """Neither table may hold a key the other does not.

        A key in the runner but not the form is a parameter nobody can reach. A
        key in the form but not the runner is worse: the runner drops what it
        does not recognise, so the user types a value, the job runs without it,
        and nothing reports the difference — not the UI, not the log, not the
        stored params.

        This is the one place the UI duplicates a table that lives in Python.
        """
        import re

        from bloomery.jobs.runner import _FLAGS, _SWITCHES
        from bloomery.server.app import STATIC_DIR

        source = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

        def arm_of(table: str, kind_name: str) -> str:
            """The array literal for one kind, sliced by bracket depth.

            Depth-counting rather than a regex: the arms sit one after another
            in the same object, and a non-greedy pattern happily runs from an
            empty `prepare: []` to the closing bracket of `train`.
            """
            start = source.index(f"const {table} = {{")
            opening = source.index(f"\n  {kind_name}: [", start) + len(f"\n  {kind_name}: ")
            depth = 0
            for offset, char in enumerate(source[opening:]):
                depth += (char == "[") - (char == "]")
                if depth == 0:
                    return source[opening : opening + offset + 1]
            raise AssertionError(f"unterminated {kind_name} arm in {table}")

        for kind in _FLAGS:
            # Sliced per kind so a key defined for `train` cannot satisfy an
            # assertion about `bench`.
            expected = set(_FLAGS[kind]) | set(_SWITCHES[kind])
            offered: set[str] = set()
            for table in ("FIELDS", "SWITCHES"):
                offered |= set(re.findall(r'key:\s*"([^"]+)"', arm_of(table, kind.value)))

            assert offered == expected, (
                f"{kind.value}: form and runner disagree — "
                f"only in form: {sorted(offered - expected)}, "
                f"only in runner: {sorted(expected - offered)}"
            )


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

    def test_a_parked_reader_wakes_on_close(self) -> None:
        """Otherwise every restart waits out uvicorn's graceful-shutdown timeout.

        Uvicorn waits for active tasks before running lifespan teardown, so a
        websocket handler blocked on an empty queue would sit through the whole
        grace period and then be force-cancelled — the client dropped rather
        than closed.
        """

        async def scenario() -> Any:
            hub = EventHub()
            hub.bind()
            async with hub.subscribe() as queue:
                reader = asyncio.create_task(hub.next_event(queue))
                await asyncio.sleep(0.05)
                assert not reader.done(), "reader should be parked"
                await hub.close()
                return await asyncio.wait_for(reader, timeout=5)

        assert asyncio.run(scenario()) is None

    def test_a_full_queue_still_learns_about_shutdown(self) -> None:
        """The reason the wake-up is a flag and not a sentinel event.

        A sentinel would be pushed with put_nowait, which a full queue rejects —
        so it would go missing exactly when the server is busiest, which is the
        worst moment for shutdown to hang.
        """

        async def scenario() -> Any:
            hub = EventHub(queue_limit=2)
            hub.bind()
            async with hub.subscribe() as queue:
                for n in range(6):
                    hub.publish_threadsafe({"n": n})
                await asyncio.sleep(0.05)
                assert queue.full()
                await hub.close()

                last: Any = object()
                while last is not None:
                    last = await asyncio.wait_for(hub.next_event(queue), timeout=5)
                return last

        assert asyncio.run(scenario()) is None

    def test_subscribing_after_close_does_not_park_forever(self) -> None:
        async def scenario() -> Any:
            hub = EventHub()
            hub.bind()
            await hub.close()
            async with hub.subscribe() as queue:
                return await asyncio.wait_for(hub.next_event(queue), timeout=5)

        assert asyncio.run(scenario()) is None

    def test_close_drops_whatever_was_still_buffered(self) -> None:
        """Shutdown wins over delivery, deliberately.

        These are status updates for a live view and the connection is ending
        anyway. Draining them into a socket that may not be reading is the exact
        stall the wake-up exists to prevent.
        """

        async def scenario() -> Any:
            hub = EventHub()
            hub.bind()
            async with hub.subscribe() as queue:
                hub.publish_threadsafe({"n": 1})
                await asyncio.sleep(0.05)
                assert queue.qsize() == 1
                await hub.close()
                return await asyncio.wait_for(hub.next_event(queue), timeout=5)

        assert asyncio.run(scenario()) is None

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

    def test_shutdown_is_not_delayed_by_an_open_socket(
        self, store: JobStore, tmp_path: Path
    ) -> None:
        """The behaviour the wake-up exists for, at the level a user would feel."""
        app = create_app(store=store, home=tmp_path, start_supervisor=False)
        started = time.monotonic()
        with TestClient(app) as test_client, test_client.websocket_connect("/api/stream") as socket:
            socket.receive_json()  # snapshot; the handler is now parked
        elapsed = time.monotonic() - started
        assert elapsed < 5, f"shutdown took {elapsed:.1f}s with a socket open"

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
