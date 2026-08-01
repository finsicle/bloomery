// SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The whole client. No build step, no dependencies — what you are reading is
// what runs, which is the honest answer to the AGPL's offer of source.

import {
  TERMINAL,
  describe,
  elapsed,
  formatBytes,
  formatDuration,
  terminalise,
} from "./format.js";

const SIZES = ["d4", "d8", "d12", "d24", "d26", "1b", "3b", "7b"];

// Exactly the parameters the runner accepts, per kind. It filters anything else
// out silently, so offering an extra field would produce a job that ignores what
// was typed with no error anywhere. Keep in step with _FLAGS in jobs/runner.py.
const FIELDS = {
  prepare: [
    { key: "name", label: "Dataset name" },
    { key: "source", label: "Source", hint: "file or directory" },
    { key: "synthetic", label: "Synthetic docs", type: "number", min: 1, hint: "instead of a source" },
    { key: "vocab", label: "Vocab size", type: "number", min: 1 },
    { key: "val_fraction", label: "Val fraction", type: "number", min: 0, step: "0.01" },
  ],
  train: [
    { key: "data", label: "Data" },
    { key: "mix", label: "Mixture", hint: "instead of data" },
    { key: "mix_version", label: "Mixture version" },
    { key: "name", label: "Run name" },
    { key: "size", label: "Size", choices: SIZES, hint: "named preset" },
    { key: "depth", label: "Depth", type: "number", min: 1 },
    { key: "steps", label: "Steps", type: "number", min: 1 },
    { key: "batch", label: "Batch", type: "number", min: 1, hint: "sequences per step" },
    { key: "seq", label: "Sequence length", type: "number", min: 1 },
    { key: "grad_accum", label: "Grad accum", type: "number", min: 1 },
    { key: "lr", label: "Learning rate", hint: "default derives from width" },
    { key: "eval_every", label: "Eval every", type: "number", min: 1 },
    { key: "save_every", label: "Save every", type: "number", min: 1 },
    { key: "cores", label: "Cores", type: "number", min: 1, hint: "--cores" },
    { key: "device", label: "Device", choices: ["cuda", "mps", "cpu"] },
    { key: "seed", label: "Seed", type: "number" },
  ],
  bench: [
    { key: "size", label: "Size", choices: SIZES, hint: "named preset" },
    { key: "depth", label: "Depth", type: "number", min: 1 },
    { key: "vocab", label: "Vocab size", type: "number", min: 1 },
    { key: "batch", label: "Batch", type: "number", min: 1 },
    { key: "seq", label: "Sequence length", type: "number", min: 1 },
    { key: "steps", label: "Steps", type: "number", min: 1 },
    { key: "cores", label: "Cores", type: "number", min: 1 },
    { key: "device", label: "Device", choices: ["cuda", "mps", "cpu"] },
  ],
};

// Boolean flags, emitted as bare switches. Mirrors _SWITCHES in jobs/runner.py.
const SWITCHES = {
  prepare: [],
  train: [
    { key: "grad_checkpoint", label: "Gradient checkpointing" },
    { key: "resume", label: "Resume from latest checkpoint" },
    { key: "force", label: "Force" },
  ],
  bench: [{ key: "grad_checkpoint", label: "Gradient checkpointing" }],
};

const state = {
  jobs: new Map(), // id -> job, authoritative
  open: null, // id of the expanded job, if any
  logTimer: null,
  gpus: [],
};

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------- formatting

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  // textContent throughout, never innerHTML: job names, params and log lines
  // are all attacker-influenced if this is ever exposed beyond localhost.
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

// ---------------------------------------------------------------- the form

function buildFields() {
  const kind = $("kind").value;
  const host = $("fields");
  host.replaceChildren();

  for (const field of FIELDS[kind]) {
    const label = el("label");
    const caption = el("span", "caption");
    caption.append(el("span", null, field.label));
    if (field.hint) caption.append(el("span", "hint", field.hint));
    label.append(caption);

    let input;
    if (field.choices) {
      input = el("select");
      input.append(el("option", null, ""));
      for (const choice of field.choices) {
        const option = el("option", null, choice);
        option.value = choice;
        input.append(option);
      }
    } else {
      input = el("input");
      input.type = field.type || "text";
      if (field.min !== undefined) input.min = field.min;
      if (field.step) input.step = field.step;
    }
    input.name = `param_${field.key}`;
    label.append(input);
    host.append(label);
  }

  if (SWITCHES[kind].length) {
    const box = el("div", "switches");
    for (const sw of SWITCHES[kind]) {
      const label = el("label");
      const input = el("input");
      input.type = "checkbox";
      input.name = `switch_${sw.key}`;
      label.append(input, sw.label);
      box.append(label);
    }
    host.append(box);
  }
}

function collect(form) {
  const params = {};
  for (const [name, raw] of new FormData(form).entries()) {
    if (!name.startsWith("param_")) continue;
    const value = String(raw).trim();
    // The runner skips empty values anyway; not sending them keeps the stored
    // params honest about what was actually asked for.
    if (value !== "") params[name.slice(6)] = value;
  }
  for (const sw of SWITCHES[$("kind").value]) {
    if (form.elements[`switch_${sw.key}`]?.checked) params[sw.key] = true;
  }

  const body = { kind: $("kind").value, params };
  const label = form.elements.label.value.trim();
  if (label) body.name = label;

  const cores = form.elements.res_cores.value.trim();
  const gib = form.elements.res_memory_gib.value.trim();
  const gpus = state.gpus.filter((g) => form.elements[`gpu_${g.index}`]?.checked).map((g) => g.index);
  if (cores || gib || gpus.length) {
    body.resources = { gpus };
    if (cores) body.resources.cores = Number(cores);
    if (gib) body.resources.memory_bytes = Number(gib) * 1024 ** 3;
  }
  return body;
}

async function submit(event) {
  event.preventDefault();
  const form = event.target;
  const button = $("submit-button");
  const error = $("submit-error");
  error.textContent = "";
  button.disabled = true;
  try {
    const response = await fetch("/api/jobs", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(collect(form)),
    });
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(describe(detail.detail) || `submit failed (${response.status})`);
    }
    // No need to insert it: the supervisor publishes the new job and it arrives
    // over the socket like any other transition.
    form.reset();
    buildFields();
  } catch (problem) {
    error.textContent = problem.message;
  } finally {
    button.disabled = false;
  }
}

// ---------------------------------------------------------------- job list

function sorted() {
  return [...state.jobs.values()].sort((a, b) =>
    String(b.created_at).localeCompare(String(a.created_at)),
  );
}

function renderCounts() {
  const counts = {};
  for (const job of state.jobs.values()) {
    counts[job.status] = (counts[job.status] || 0) + 1;
  }
  const host = $("counts");
  host.replaceChildren();
  // Only show statuses that exist, but always show running and queued so the
  // header does not jump around while a queue drains.
  const order = ["running", "queued", "succeeded", "failed", "cancelled", "interrupted"];
  for (const status of order) {
    const n = counts[status] || 0;
    if (!n && status !== "running" && status !== "queued") continue;
    const span = el("span");
    span.append(el("b", null, n), ` ${status}`);
    host.append(span);
  }
}

function renderJobs() {
  const host = $("jobs");
  const jobs = sorted();
  $("jobs-empty").hidden = jobs.length > 0;
  host.replaceChildren();
  for (const job of jobs) host.append(renderJob(job));
  renderCounts();
}

function renderJob(job) {
  const box = el("div", "job");
  box.dataset.id = job.id;

  const head = el("button", "job-head");
  head.type = "button";
  head.setAttribute("aria-expanded", String(state.open === job.id));
  head.append(
    el("span", `pill ${job.status}`, job.status),
    el("span", "job-kind", job.kind),
    el("span", "job-name", job.name || job.id.slice(0, 8)),
    el("span", "job-when", formatDuration(elapsed(job))),
  );
  head.addEventListener("click", () => toggle(job.id));
  box.append(head);

  if (state.open === job.id) box.append(renderDetail(job));
  return box;
}

function renderDetail(job) {
  const body = el("div", "job-body");

  const facts = el("dl", "facts");
  const add = (term, value) => {
    const cell = el("div");
    cell.append(el("dt", null, term), el("dd", null, value ?? "—"));
    facts.append(cell);
  };
  add("id", job.id);
  add("created", job.created_at);
  add("started", job.started_at);
  add("finished", job.finished_at);
  add("exit code", job.exit_code);
  add("run dir", job.run_dir);
  body.append(facts);

  const params = Object.entries(job.params || {}).filter(([key]) => key !== "resources");
  if (params.length) {
    const list = el("dl", "facts");
    for (const [key, value] of params) {
      const cell = el("div");
      cell.append(el("dt", null, key), el("dd", null, formatValue(value)));
      list.append(cell);
    }
    body.append(list);
  }

  if (job.error) body.append(el("div", "job-error", job.error));
  // Set when a memory cap was asked for but could not be applied — worth
  // surfacing, because the job is running with weaker limits than requested.
  if (job.limits_note) body.append(el("div", "note", job.limits_note));

  const logHead = el("div", "log-head");
  logHead.append(el("span", null, "log"));
  const truncated = el("span", "hint", "");
  truncated.id = "log-truncated";
  logHead.append(truncated);
  if (!TERMINAL.has(job.status)) {
    const cancel = el("button", "quiet", "Cancel");
    cancel.type = "button";
    cancel.style.marginLeft = "auto";
    cancel.addEventListener("click", () => cancelJob(job.id, cancel));
    logHead.append(cancel);
  }
  body.append(logHead);

  const log = el("pre", "log", "");
  log.id = "log";
  body.append(log);
  return body;
}

function formatValue(value) {
  if (value === true) return "yes";
  if (value === false) return "no";
  if (value && typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function toggle(id) {
  state.open = state.open === id ? null : id;
  stopLogPolling();
  renderJobs();
  if (state.open) pollLog(state.open);
}

async function cancelJob(id, button) {
  button.disabled = true;
  try {
    const response = await fetch(`/api/jobs/${encodeURIComponent(id)}`, { method: "DELETE" });
    // 409 means it finished between the render and the click. That is a race,
    // not an error worth shouting about — the socket will deliver the real
    // state along in a moment.
    if (response.ok) {
      const job = await response.json();
      // A 200 can carry just {id, status} if the row vanished mid-cancel, so
      // merge rather than replace.
      upsert({ ...state.jobs.get(id), ...job });
    }
  } catch {
    button.disabled = false;
  }
}

// ---------------------------------------------------------------- the log

function stopLogPolling() {
  if (state.logTimer) clearTimeout(state.logTimer);
  state.logTimer = null;
}

async function pollLog(id) {
  if (state.open !== id) return;
  let payload;
  try {
    const response = await fetch(`/api/jobs/${encodeURIComponent(id)}/log?tail=200`);
    if (!response.ok) return;
    payload = await response.json();
  } catch {
    return; // the socket handler will resync; no need to shout here
  }
  if (state.open !== id) return;

  const log = $("log");
  if (log) {
    // Only follow the tail if the reader was already at the bottom; otherwise
    // scrolling back through a running job's log would yank them forward every
    // poll.
    const pinned = log.scrollTop + log.clientHeight >= log.scrollHeight - 20;
    log.textContent = payload.lines.length
      ? payload.lines.map(terminalise).join("\n")
      : "(nothing yet)";
    if (pinned) log.scrollTop = log.scrollHeight;
  }
  const note = $("log-truncated");
  // `truncated` is absent entirely when the job has no log file, so a plain
  // truthiness check is right here — undefined must not read as "complete".
  if (note) note.textContent = payload.truncated ? "earlier lines not shown" : "";

  // Stop once the job is finished: the file will not change again, and this is
  // reading from disk on a machine that is probably busy training.
  const job = state.jobs.get(id);
  if (job && TERMINAL.has(job.status)) return;
  state.logTimer = setTimeout(() => pollLog(id), 1500);
}

// ---------------------------------------------------------------- the socket

function upsert(job) {
  const previous = state.jobs.get(job.id);
  state.jobs.set(job.id, job);

  if (!previous || previous.status !== job.status || state.open === job.id) {
    renderJobs();
  } else {
    // Cheap path: only the elapsed time moved, so do not rebuild the row and
    // lose focus or scroll position.
    tick();
    renderCounts();
  }

  // One last read on the transition into a terminal state, to catch whatever
  // the process wrote just before exiting.
  if (state.open === job.id && previous && !TERMINAL.has(previous.status) && TERMINAL.has(job.status)) {
    pollLog(job.id);
  }
}

function setLink(status, detail) {
  const lamp = $("link");
  lamp.className = `link ${status}`;
  lamp.title = detail;
}

function connect(attempt = 0) {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${scheme}://${location.host}/api/stream`);

  socket.addEventListener("open", () => setLink("live", "live"));

  socket.addEventListener("message", (event) => {
    const frame = JSON.parse(event.data);
    if (frame.type === "snapshot") {
      // Replace rather than merge. The hub drops the oldest events for a slow
      // subscriber without marking the gap, so after a reconnect the snapshot
      // is the only trustworthy account of what exists.
      state.jobs = new Map(frame.jobs.map((job) => [job.id, job]));
      if (frame.source_url) showSource(frame.source_url);
      renderJobs();
      if (state.open) pollLog(state.open);
    } else if (frame.type === "job") {
      upsert(frame.job);
    }
  });

  socket.addEventListener("close", () => {
    setLink("lost", "reconnecting");
    // There is no heartbeat in the protocol, so a dropped connection is only
    // discovered here. Back off, but stay responsive for the common case of a
    // server restart during development.
    const wait = Math.min(1000 * 2 ** attempt, 15000);
    setTimeout(() => connect(attempt + 1), wait);
  });

  socket.addEventListener("error", () => socket.close());
}

function showSource(url) {
  const note = $("source-note");
  note.replaceChildren();
  const link = el("a", null, "Source");
  link.href = url;
  link.rel = "noreferrer";
  note.append(link, " · ");
}

// ---------------------------------------------------------------- the host

async function loadHost() {
  const box = $("host");
  let report;
  try {
    // Fetched once. This endpoint re-probes the machine on every call, so
    // polling it would mean shelling out to vendor tools on a loop.
    report = await (await fetch("/api/host")).json();
  } catch {
    box.replaceChildren(el("p", "empty", "Could not read the hardware."));
    return;
  }

  box.replaceChildren();
  const facts = el("dl");
  const add = (term, value) => {
    const cell = el("div");
    cell.append(el("dt", null, term), el("dd", null, value ?? "—"));
    facts.append(cell);
  };
  add("cpu", report.cpu?.model);
  add("cores", `${report.cpu?.physical_cores ?? "?"} physical / ${report.cpu?.logical_cores ?? "?"} logical`);
  add("memory", `${formatBytes(report.memory?.available)} free of ${formatBytes(report.memory?.total)}`);
  add("disk", `${formatBytes(report.disk?.free)} free of ${formatBytes(report.disk?.total)}`);
  add("backend", report.backend_reason ? `${report.backend} — ${report.backend_reason}` : report.backend);
  box.append(facts);

  state.gpus = report.gpus || [];
  if (state.gpus.length) {
    const list = el("dl");
    for (const gpu of state.gpus) {
      const cell = el("div");
      cell.append(
        el("dt", null, `gpu ${gpu.index}`),
        el("dd", null, `${gpu.name} · ${formatBytes(gpu.vram_total)}`),
      );
      list.append(cell);
    }
    box.append(list);
  }

  for (const issue of report.issues || []) {
    const note = el("div", `issue ${issue.level}`, issue.message);
    if (issue.hint) note.append(el("span", "hint", issue.hint));
    box.append(note);
  }

  renderGpuPicker();
}

function renderGpuPicker() {
  const box = $("gpu-picker");
  box.replaceChildren();
  if (!state.gpus.length) {
    box.append(el("span", "hint", "No GPUs detected."));
    return;
  }
  box.append(el("span", "hint", "Visible GPUs:"));
  for (const gpu of state.gpus) {
    const label = el("label");
    const input = el("input");
    input.type = "checkbox";
    input.name = `gpu_${gpu.index}`;
    label.append(input, `${gpu.index}: ${gpu.name}`);
    box.append(label);
  }
}

// ---------------------------------------------------------------- lifecycle

// Running jobs need their elapsed time redrawn even when nothing is happening.
function tick() {
  for (const job of state.jobs.values()) {
    if (TERMINAL.has(job.status)) continue;
    const row = document.querySelector(`.job[data-id="${CSS.escape(job.id)}"] .job-when`);
    if (row) row.textContent = formatDuration(elapsed(job));
  }
}

$("kind").addEventListener("change", buildFields);
$("submit").addEventListener("submit", submit);

buildFields();
renderJobs();
setLink("", "connecting");
connect();
loadHost();
setInterval(tick, 1000);
