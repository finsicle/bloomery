// SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Run with:  node --test tests/
//
// node:test and node:assert are built into node, so this needs no package.json,
// no npm install and no dependency. The project still has no build step: node
// is a test tool here, never something a user needs in order to run bloomery.

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  describe as describeError,
  elapsed,
  formatBytes,
  formatDuration,
  terminalise,
} from "../src/bloomery/server/static/format.js";

const ESC = String.fromCharCode(0x1b);

test("terminalise strips the colour codes rich writes", () => {
  const raw = `tokens ${ESC}[1m861${ESC}[0m train`;
  assert.equal(terminalise(raw), "tokens 861 train");
});

test("terminalise strips cursor and erase sequences", () => {
  const raw = `${ESC}[?25l${ESC}[2Ktokenizing and packing`;
  assert.equal(terminalise(raw), "tokenizing and packing");
});

test("a progress bar shows its last redraw, as a terminal would", () => {
  // The server keeps every redraw because a bare CR is content, not a line
  // break. What a person should see is where the bar ended up.
  assert.equal(terminalise("epoch: 10%\repoch: 90%\repoch: done"), "epoch: done");
});

test("text that merely looks like an escape is left alone", () => {
  // The regex must anchor on ESC. Bracketed text is ordinary log output.
  assert.equal(terminalise("path/to/file [not-ansi] ok"), "path/to/file [not-ansi] ok");
  assert.equal(terminalise("plain line"), "plain line");
});

test("terminalise leaves an empty line empty", () => {
  assert.equal(terminalise(""), "");
});

test("formatDuration reads as a person would say it", () => {
  assert.equal(formatDuration(0), "0s");
  assert.equal(formatDuration(45), "45s");
  assert.equal(formatDuration(90), "1m 30s");
  assert.equal(formatDuration(3661), "1h 01m");
  assert.equal(formatDuration(null), "—");
  assert.equal(formatDuration(undefined), "—");
});

test("formatDuration does not render a negative clock", () => {
  // Clock skew between the server's timestamps and the browser is possible.
  assert.equal(formatDuration(-5), "0s");
});

test("elapsed ticks from started_at while a job is running", () => {
  // The server only recomputes duration_seconds when it sends a frame, so a
  // running job's timer would otherwise sit frozen between transitions.
  const job = {
    status: "running",
    started_at: "2026-08-01T00:00:00+00:00",
    duration_seconds: 1, // stale on purpose
  };
  const now = Date.parse("2026-08-01T00:00:30+00:00");
  assert.equal(elapsed(job, now), 30);
});

test("elapsed trusts the server once a job has finished", () => {
  const job = {
    status: "succeeded",
    started_at: "2026-08-01T00:00:00+00:00",
    duration_seconds: 12.5,
  };
  const now = Date.parse("2026-08-01T09:00:00+00:00");
  assert.equal(elapsed(job, now), 12.5);
});

test("elapsed is unknown for a job that has not started", () => {
  assert.equal(elapsed({ status: "queued", started_at: null }), null);
});

test("elapsed falls back rather than returning NaN on an unparseable time", () => {
  const job = { status: "running", started_at: "not a date", duration_seconds: 7 };
  assert.equal(elapsed(job, Date.now()), 7);
});

test("formatBytes picks a readable unit", () => {
  assert.equal(formatBytes(0), "0 B");
  assert.equal(formatBytes(512), "512 B");
  assert.equal(formatBytes(1024), "1.0 KiB");
  assert.equal(formatBytes(6 * 1024 ** 3), "6.0 GiB");
  assert.equal(formatBytes(228 * 1024 ** 3), "228 GiB");
});

test("formatBytes shows nothing rather than zero when a value is missing", () => {
  // A GPU that does not report VRAM must not read as a GPU with none.
  assert.equal(formatBytes(null), "—");
  assert.equal(formatBytes(undefined), "—");
});

test("describe flattens FastAPI's validation errors into a sentence", () => {
  const detail = [
    { loc: ["body", "resources", "cores"], msg: "Input should be greater than 0" },
  ];
  assert.equal(describeError(detail), "resources.cores: Input should be greater than 0");
});

test("describe passes a plain string detail through", () => {
  assert.equal(describeError("no such job: abc"), "no such job: abc");
  assert.equal(describeError(null), "");
});
