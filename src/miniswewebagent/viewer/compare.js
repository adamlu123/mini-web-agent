// Compare tab: diffs two run folders (via GET /api/compare) and lets the
// user drill into any task's steps side by side by reusing the existing
// /api/task and /artifact endpoints for each run. Kept independent of
// app.js/state so the Trace tab is untouched by this feature.

const FILTER_LABELS = {
  all: "All",
  regressed: "Regressed",
  improved: "Improved",
  same_success: "Same pass",
  same_fail: "Same fail",
  unknown: "Unknown",
};

const STATUS_LABELS = {
  success: "Pass",
  failure: "Fail",
  unknown: "Unknown",
  missing: "Missing",
};

const compareState = {
  runs: [],
  runIdA: null,
  runIdB: null,
  data: null,
  filter: "all",
  selectedTaskId: null,
  detail: { a: null, b: null },
  folderNames: { a: null, b: null },
  stepIndex: { a: 0, b: 0 },
};

let compareRunsLoaded = false;

const traceTabBtn = document.getElementById("traceTabBtn");
const compareTabBtn = document.getElementById("compareTabBtn");
const traceView = document.getElementById("traceView");
const compareView = document.getElementById("compareView");

const compareRunA = document.getElementById("compareRunA");
const compareRunB = document.getElementById("compareRunB");
const compareLoadBtn = document.getElementById("compareLoadBtn");
const compareStatusText = document.getElementById("compareStatusText");
const leaderboardBody = document.getElementById("leaderboardBody");
const compareFilterChips = document.getElementById("compareFilterChips");
const compareTaskList = document.getElementById("compareTaskList");
const compareEmptyState = document.getElementById("compareEmptyState");
const compareDetail = document.getElementById("compareDetail");
const compareTaskTitle = document.getElementById("compareTaskTitle");
const compareTaskMeta = document.getElementById("compareTaskMeta");
const compareColumns = document.getElementById("compareColumns");

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    let message = response.statusText;
    try {
      const payload = await response.json();
      message = payload.error || message;
    } catch {
      message = await response.text();
    }
    throw new Error(message);
  }
  return response.json();
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function setCompareStatus(message) {
  compareStatusText.textContent = message;
}

function switchTab(target) {
  const showCompare = target === "compare";
  traceView.hidden = showCompare;
  compareView.hidden = !showCompare;
  traceTabBtn.classList.toggle("active", !showCompare);
  traceTabBtn.setAttribute("aria-selected", String(!showCompare));
  compareTabBtn.classList.toggle("active", showCompare);
  compareTabBtn.setAttribute("aria-selected", String(showCompare));

  if (showCompare && !compareRunsLoaded) {
    loadCompareRuns().catch((error) => setCompareStatus(`Failed to load runs: ${error.message}`));
  }
}

traceTabBtn.addEventListener("click", () => switchTab("trace"));
compareTabBtn.addEventListener("click", () => switchTab("compare"));

function populateRunSelect(select) {
  const previous = select.value;
  select.innerHTML = "";
  for (const run of compareState.runs) {
    const option = document.createElement("option");
    option.value = run.id;
    option.textContent = `${run.name} (${run.taskCount})`;
    select.appendChild(option);
  }
  if (compareState.runs.some((run) => run.id === previous)) {
    select.value = previous;
  }
}

async function loadCompareRuns() {
  setCompareStatus("Loading runs...");
  const payload = await fetchJson("/api/runs");
  compareState.runs = payload.runs || [];
  compareRunsLoaded = true;

  populateRunSelect(compareRunA);
  populateRunSelect(compareRunB);

  if (compareState.runs.length >= 2) {
    // Runs are newest-first (see /api/runs); default to comparing the two
    // most recent runs, with the older one as the baseline.
    compareRunA.value = compareState.runs[1].id;
    compareRunB.value = compareState.runs[0].id;
    setCompareStatus("Pick two runs, then press Compare runs.");
  } else {
    setCompareStatus("Need at least two runs under this root to compare.");
  }
}

compareLoadBtn.addEventListener("click", () => {
  loadComparison().catch((error) => setCompareStatus(`Compare failed: ${error.message}`));
});

async function loadComparison() {
  const runIdA = compareRunA.value;
  const runIdB = compareRunB.value;
  if (!runIdA || !runIdB) {
    setCompareStatus("Select both runs first.");
    return;
  }
  if (runIdA === runIdB) {
    setCompareStatus("Pick two different runs to compare.");
    return;
  }

  setCompareStatus(`Comparing ${runIdA} (baseline) vs ${runIdB}...`);
  const params = new URLSearchParams({ runs: `${runIdA},${runIdB}`, baseline: runIdA });
  compareState.data = await fetchJson(`/api/compare?${params.toString()}`);
  compareState.runIdA = runIdA;
  compareState.runIdB = runIdB;
  compareState.selectedTaskId = null;
  compareState.detail = { a: null, b: null };
  compareState.filter = "all";
  updateFilterChipUi();

  renderLeaderboard();
  renderCompareTaskList();
  renderCompareDetail();
  setCompareStatus(summaryLine());
}

function summaryLine() {
  const diff = compareState.data?.diffSummary?.[compareState.runIdB];
  const total = compareState.data?.tasks?.length ?? 0;
  if (!diff) return `Loaded ${total} task(s).`;
  return (
    `${total} task(s) · ${diff.regressed} regressed · ${diff.improved} improved · ` +
    `${diff.sameSuccess} same-pass · ${diff.sameFail} same-fail · ${diff.unknown} unknown`
  );
}

function formatRate(bucket) {
  if (!bucket || !bucket.total) return "–";
  return `${Math.round(bucket.successRate * 100)}% (${bucket.success}/${bucket.total})`;
}

function renderLeaderboard() {
  const data = compareState.data;
  leaderboardBody.innerHTML = "";
  if (!data || !data.leaderboard.length) {
    leaderboardBody.innerHTML = '<tr><td colspan="5" class="empty-hint">No comparison loaded yet.</td></tr>';
    return;
  }

  for (const row of data.leaderboard) {
    const tr = document.createElement("tr");
    const isBaseline = row.runId === data.baselineId;
    tr.innerHTML = `
      <td>${escapeHtml(row.runId)}${isBaseline ? ' <span class="baseline-chip">baseline</span>' : ""}</td>
      <td>${formatRate(row.overall)}</td>
      <td>${formatRate(row.byLevel.easy)}</td>
      <td>${formatRate(row.byLevel.medium)}</td>
      <td>${formatRate(row.byLevel.hard)}</td>
    `;
    leaderboardBody.appendChild(tr);
  }
}

function classificationForTask(task) {
  if (!compareState.data) return "unknown";
  return task.flipsVsBaseline[compareState.runIdB] || "unknown";
}

function applyCompareFilter(tasks) {
  if (compareState.filter === "all") return tasks;
  return tasks.filter((task) => classificationForTask(task) === compareState.filter);
}

function updateFilterChipCounts() {
  const tasks = compareState.data?.tasks || [];
  const counts = { all: tasks.length, regressed: 0, improved: 0, same_success: 0, same_fail: 0, unknown: 0 };
  for (const task of tasks) {
    const key = classificationForTask(task);
    counts[key] = (counts[key] || 0) + 1;
  }
  for (const chip of compareFilterChips.querySelectorAll("button[data-filter]")) {
    const filter = chip.dataset.filter;
    chip.textContent = `${chip.dataset.label} (${counts[filter] || 0})`;
  }
}

function updateFilterChipUi() {
  updateFilterChipCounts();
  for (const chip of compareFilterChips.querySelectorAll("button[data-filter]")) {
    chip.classList.toggle("active", chip.dataset.filter === compareState.filter);
  }
}

compareFilterChips.addEventListener("click", (event) => {
  const chip = event.target.closest("button[data-filter]");
  if (!chip) return;
  compareState.filter = chip.dataset.filter;
  updateFilterChipUi();
  renderCompareTaskList();
});

function renderCompareTaskList() {
  const data = compareState.data;
  compareTaskList.innerHTML = "";
  if (!data) return;

  const tasks = applyCompareFilter(data.tasks);
  if (!tasks.length) {
    compareTaskList.innerHTML = '<li class="task-empty">No tasks match this filter.</li>';
    return;
  }

  for (const task of tasks) {
    const classification = classificationForTask(task);
    const statusA = STATUS_LABELS[task.statuses[compareState.runIdA]] || "–";
    const statusB = STATUS_LABELS[task.statuses[compareState.runIdB]] || "–";

    const li = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "task-item";
    if (task.taskId === compareState.selectedTaskId) {
      button.classList.add("selected");
    }
    button.dataset.taskId = task.taskId;
    button.innerHTML = `
      <span class="task-header">
        <span class="task-tag classification-${classification}">${FILTER_LABELS[classification] || classification}</span>
        <strong>${escapeHtml(task.taskId)}</strong>
      </span>
      <span class="task-title">${escapeHtml(task.title || "(no task text)")}</span>
      <span class="task-meta">level: ${escapeHtml(task.level)} · A: ${statusA} · B: ${statusB}</span>
    `;
    li.appendChild(button);
    compareTaskList.appendChild(li);
  }
}

compareTaskList.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-task-id]");
  if (!button) return;
  compareState.selectedTaskId = button.dataset.taskId;
  renderCompareTaskList();
  loadCompareDetail().catch((error) => setCompareStatus(`Failed to load task detail: ${error.message}`));
});

function findSelectedTask() {
  if (!compareState.data || !compareState.selectedTaskId) return null;
  return compareState.data.tasks.find((task) => task.taskId === compareState.selectedTaskId) || null;
}

async function loadCompareDetail() {
  const task = findSelectedTask();
  if (!task) return;

  // runDirNames maps runId -> the task's folder name on disk, which is what
  // /api/task and /artifact expect; it can differ from taskId in general
  // (see run_compare.py), so always go through it, falling back to taskId
  // only if a run has no entry (task absent from that run).
  const folderA = task.runDirNames[compareState.runIdA] || task.taskId;
  const folderB = task.runDirNames[compareState.runIdB] || task.taskId;
  compareState.folderNames = { a: folderA, b: folderB };
  compareState.stepIndex = { a: 0, b: 0 };

  setCompareStatus(`Loading ${task.taskId}...`);
  const [detailA, detailB] = await Promise.all([
    fetchJson(`/api/task?run=${encodeURIComponent(compareState.runIdA)}&task=${encodeURIComponent(folderA)}`),
    fetchJson(`/api/task?run=${encodeURIComponent(compareState.runIdB)}&task=${encodeURIComponent(folderB)}`),
  ]);
  compareState.detail = { a: detailA, b: detailB };
  renderCompareDetail();
  setCompareStatus(`Showing ${task.taskId}`);
}

function columnTemplate(side, label) {
  return `
    <div class="compare-column panel" data-side="${side}">
      <div class="compare-column-header">
        <h3>${escapeHtml(label)}</h3>
        <span class="task-tag" id="compareStatus-${side}">-</span>
      </div>
      <div class="step-nav compare-step-nav">
        <button type="button" class="compare-prev" data-side="${side}">Prev</button>
        <span id="compareStepLabel-${side}">Step - / -</span>
        <button type="button" class="compare-next" data-side="${side}">Next</button>
      </div>
      <div class="image-frame compare-image-frame">
        <img id="compareStepImage-${side}" alt="Step screenshot" hidden />
        <p id="compareStepImageEmpty-${side}">No screenshot for this step.</p>
      </div>
      <div class="info-block">
        <h4>Thought</h4>
        <pre id="compareThought-${side}">-</pre>
      </div>
      <div class="info-block">
        <h4>Action</h4>
        <pre id="compareAction-${side}">-</pre>
      </div>
      <div class="info-block two-col">
        <div>
          <h4>Page URL</h4>
          <pre id="compareUrl-${side}">-</pre>
        </div>
        <div>
          <h4>Page Title</h4>
          <pre id="compareTitle-${side}">-</pre>
        </div>
      </div>
      <div class="info-block">
        <h4>Final response</h4>
        <pre id="compareFinal-${side}">-</pre>
      </div>
    </div>
  `;
}

function renderCompareDetail() {
  const task = findSelectedTask();
  if (!task || !compareState.detail.a || !compareState.detail.b) {
    compareEmptyState.hidden = false;
    compareDetail.hidden = true;
    return;
  }

  compareEmptyState.hidden = true;
  compareDetail.hidden = false;
  compareTaskTitle.textContent = task.taskId;
  compareTaskMeta.textContent = `${task.title || "(no task text)"} · level: ${task.level}`;

  compareColumns.innerHTML =
    columnTemplate("a", `Run A (baseline) — ${compareState.runIdA}`) +
    columnTemplate("b", `Run B — ${compareState.runIdB}`);

  renderColumnStep("a");
  renderColumnStep("b");
}

function buildCompareArtifactUrl(side, step) {
  const runId = side === "a" ? compareState.runIdA : compareState.runIdB;
  const folder = compareState.folderNames[side];
  if (!runId || !folder || !step?.screenshotRelPath) return "";
  const params = new URLSearchParams({ run: runId, task: folder, file: step.screenshotRelPath });
  return `/artifact?${params.toString()}`;
}

function renderColumnStep(side) {
  const detail = compareState.detail[side];
  const statusEl = document.getElementById(`compareStatus-${side}`);
  const finalEl = document.getElementById(`compareFinal-${side}`);
  const labelEl = document.getElementById(`compareStepLabel-${side}`);
  const thoughtEl = document.getElementById(`compareThought-${side}`);
  const actionEl = document.getElementById(`compareAction-${side}`);
  const urlEl = document.getElementById(`compareUrl-${side}`);
  const titleEl = document.getElementById(`compareTitle-${side}`);
  const imgEl = document.getElementById(`compareStepImage-${side}`);
  const imgEmptyEl = document.getElementById(`compareStepImageEmpty-${side}`);
  const prevBtn = compareColumns.querySelector(`button.compare-prev[data-side="${side}"]`);
  const nextBtn = compareColumns.querySelector(`button.compare-next[data-side="${side}"]`);

  if (!detail) return;

  statusEl.textContent = detail.status || "-";
  statusEl.className = `task-tag ${detail.status || ""}`;
  finalEl.textContent = (detail.finalResult || "").trim() || "(no final response captured)";

  const steps = detail.steps || [];
  if (!steps.length) {
    labelEl.textContent = "Step - / -";
    thoughtEl.textContent = "-";
    actionEl.textContent = "-";
    urlEl.textContent = "-";
    titleEl.textContent = "-";
    imgEl.hidden = true;
    imgEl.removeAttribute("src");
    imgEmptyEl.hidden = false;
    prevBtn.disabled = true;
    nextBtn.disabled = true;
    return;
  }

  const index = Math.max(0, Math.min(compareState.stepIndex[side], steps.length - 1));
  compareState.stepIndex[side] = index;
  const step = steps[index];

  labelEl.textContent = `Step ${index + 1} / ${steps.length}`;
  thoughtEl.textContent = (step.thought || "").trim() || "(no thought recorded)";
  actionEl.textContent = (step.action || "").trim() || "(no action recorded)";
  urlEl.textContent = step.url || "-";
  titleEl.textContent = step.title || "-";

  const screenshotUrl = buildCompareArtifactUrl(side, step);
  if (screenshotUrl) {
    imgEl.src = screenshotUrl;
    imgEl.hidden = false;
    imgEmptyEl.hidden = true;
  } else {
    imgEl.hidden = true;
    imgEl.removeAttribute("src");
    imgEmptyEl.hidden = false;
  }

  prevBtn.disabled = index <= 0;
  nextBtn.disabled = index >= steps.length - 1;
}

compareColumns.addEventListener("click", (event) => {
  const button = event.target.closest("button.compare-prev, button.compare-next");
  if (!button) return;
  const side = button.dataset.side;
  const delta = button.classList.contains("compare-prev") ? -1 : 1;
  compareState.stepIndex[side] = (compareState.stepIndex[side] || 0) + delta;
  renderColumnStep(side);
});
