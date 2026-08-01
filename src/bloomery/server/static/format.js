// SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The parts of the client that are pure functions of their input, split out so
// they can be tested without a browser. Everything here must stay free of
// document/window references: tests/format.test.js imports this module into
// node, and app.js imports it in the browser. Native ES modules on both sides,
// so there is still nothing to build.

// A job is finished when it reaches one of these. `interrupted` is not `failed`:
// it means the server died mid-run and never learned how the job ended.
export const TERMINAL = new Set(["succeeded", "failed", "cancelled", "interrupted"]);

// Escape sequences a terminal would act on rather than print: CSI (colour,
// cursor moves, erase) and OSC (window titles and the like). Written with
// \u escapes rather than the literal bytes, which would be invisible here.
export const ANSI = /\u001b\[[0-9;?]*[ -/]*[@-~]|\u001b\][^\u0007\u001b]*(?:\u0007|\u001b\\)/g;

// Jobs write console output meant for a terminal, so a log line arrives with
// colour codes in it and a progress bar arrives as a series of redraws
// separated by bare carriage returns. Render what a terminal would be showing
// once the line is done: the last redraw, with the control codes removed.
//
// The server deliberately keeps those bytes exactly as written — a bare CR is
// content, not a line break. Deciding what they should look like is this
// layer's job, and stripping rather than colourising keeps rendering to
// textContent, which is what makes log output safe to display at all.
export function terminalise(line) {
  return line.split("\r").pop().replace(ANSI, "");
}

export function formatDuration(seconds) {
  if (seconds === null || seconds === undefined) return "—";
  const total = Math.max(0, Math.floor(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h) return `${h}h ${String(m).padStart(2, "0")}m`;
  if (m) return `${m}m ${String(s).padStart(2, "0")}s`;
  return `${s}s`;
}

// The server computes duration_seconds only when it sends a frame, so a running
// job's elapsed time would sit frozen between transitions. Derive it locally
// instead, and fall back to the server's value once the job has finished.
//
// `now` is a parameter so this stays a pure function and a test does not have
// to move the clock.
export function elapsed(job, now = Date.now()) {
  if (!job.started_at) return null;
  if (TERMINAL.has(job.status)) return job.duration_seconds;
  const started = Date.parse(job.started_at);
  if (Number.isNaN(started)) return job.duration_seconds;
  return (now - started) / 1000;
}

export function formatBytes(n) {
  if (!n && n !== 0) return "—";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let value = n;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value < 10 && unit > 0 ? value.toFixed(1) : Math.round(value)} ${units[unit]}`;
}

// FastAPI validation errors arrive as a list of objects rather than a string.
export function describe(detail) {
  if (!detail) return "";
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((item) => `${(item.loc || []).slice(1).join(".")}: ${item.msg}`).join("; ");
  }
  return JSON.stringify(detail);
}
