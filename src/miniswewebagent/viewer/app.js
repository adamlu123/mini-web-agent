const state = {
  runs: [],
  tasks: [],
  filteredTasks: [],
  selectedRun: null,
  selectedTask: null,
  taskDetail: null,
  stepIndex: 0,
  judgeIndex: 0,
};

const runSelect = document.getElementById("runSelect");
const taskSearch = document.getElementById("taskSearch");
const refreshBtn = document.getElementById("refreshBtn");
const statusText = document.getElementById("statusText");
const rootDir = document.getElementById("rootDir");
const runCount = document.getElementById("runCount");
const taskCount = document.getElementById("taskCount");
const taskList = document.getElementById("taskList");

const taskId = document.getElementById("taskId");
const taskStatus = document.getElementById("taskStatus");
const taskSteps = document.getElementById("taskSteps");
const taskExit = document.getElementById("taskExit");
const taskText = document.getElementById("taskText");
const taskStartUrl = document.getElementById("taskStartUrl");
const taskWarnings = document.getElementById("taskWarnings");
const taskWarningsText = document.getElementById("taskWarningsText");
const taskFinal = document.getElementById("taskFinal");
const judgeSelect = document.getElementById("judgeSelect");
const judgeModel = document.getElementById("judgeModel");
const judgeStatus = document.getElementById("judgeStatus");
const judgeFile = document.getElementById("judgeFile");
const judgeUpdatedAt = document.getElementById("judgeUpdatedAt");
const judgeResponse = document.getElementById("judgeResponse");

const prevStepBtn = document.getElementById("prevStepBtn");
const nextStepBtn = document.getElementById("nextStepBtn");
const stepLabel = document.getElementById("stepLabel");
const stepImage = document.getElementById("stepImage");
const stepImageEmpty = document.getElementById("stepImageEmpty");
const stepThought = document.getElementById("stepThought");
const stepAction = document.getElementById("stepAction");
const stepUrl = document.getElementById("stepUrl");
const stepTitle = document.getElementById("stepTitle");
const stepSuccess = document.getElementById("stepSuccess");
const stepException = document.getElementById("stepException");
const stepConsole = document.getElementById("stepConsole");
const stepAria = document.getElementById("stepAria");
const rawResult = document.getElementById("rawResult");

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

function setStatus(message) {
  statusText.textContent = message;
}

function formatMultiline(value, fallback = "-") {
  const text = String(value || "").trim();
  return text || fallback;
}

function buildArtifactUrl(step) {
  if (!state.selectedRun || !state.selectedTask || !step?.screenshotRelPath) {
    return "";
  }
  const params = new URLSearchParams({
    run: state.selectedRun,
    task: state.selectedTask,
    file: step.screenshotRelPath,
  });
  return `/artifact?${params.toString()}`;
}

function renderTaskList() {
  taskList.innerHTML = "";
  taskCount.textContent = String(state.filteredTasks.length);

  if (!state.filteredTasks.length) {
    taskList.innerHTML = '<li class="task-empty">No tasks match this run/search.</li>';
    return;
  }

  for (const task of state.filteredTasks) {
    const li = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "task-item";
    if (task.id === state.selectedTask) {
      button.classList.add("selected");
    }
    button.dataset.taskId = task.id;
    button.innerHTML = `
      <span class="task-header">
        <span class="task-tag ${task.status}">${task.status}</span>
        <strong>${escapeHtml(task.taskId)}</strong>
      </span>
      <span class="task-title">${escapeHtml(task.title || "(no task text)")}</span>
      <span class="task-meta">${task.stepCount} step(s)</span>
      <span class="task-preview">${escapeHtml(task.finalPreview || "No final response captured.")}</span>
    `;
    li.appendChild(button);
    taskList.appendChild(li);
  }
}

function renderTaskDetail() {
  const detail = state.taskDetail;
  if (!detail) {
    taskId.textContent = "-";
    taskStatus.textContent = "-";
    taskSteps.textContent = "-";
    taskExit.textContent = "-";
    taskText.textContent = "-";
    taskStartUrl.textContent = "-";
    taskWarnings.hidden = true;
    taskFinal.textContent = "-";
    renderJudgeDetail();
    rawResult.textContent = "-";
    renderStep();
    return;
  }

  taskId.textContent = detail.taskId;
  taskStatus.textContent = detail.status;
  taskSteps.textContent = String(detail.stepCount);
  taskExit.textContent = detail.exitStatus || "-";
  taskText.textContent = detail.task || "-";
  taskStartUrl.textContent = detail.startUrl || "-";
  taskFinal.textContent = formatMultiline(detail.finalResult);
  rawResult.textContent = JSON.stringify(detail.result, null, 2);

  const warnings = [detail.runException, detail.closeException].filter(Boolean).join("\n\n");
  taskWarnings.hidden = !warnings;
  taskWarningsText.textContent = warnings || "-";

  renderJudgeDetail();
  renderStep();
}

function renderJudgeDetail() {
  const judges = state.taskDetail?.judges || [];
  judgeSelect.innerHTML = "";

  if (!judges.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No judge result for this task";
    judgeSelect.appendChild(option);
    judgeSelect.disabled = true;
    judgeModel.textContent = "-";
    judgeStatus.textContent = "-";
    judgeFile.textContent = "-";
    judgeUpdatedAt.textContent = "-";
    judgeResponse.textContent = "-";
    return;
  }

  state.judgeIndex = Math.max(0, Math.min(state.judgeIndex, judges.length - 1));
  judges.forEach((judge, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `${judge.model} · ${judge.status}`;
    judgeSelect.appendChild(option);
  });
  judgeSelect.value = String(state.judgeIndex);
  judgeSelect.disabled = false;

  const judge = judges[state.judgeIndex];
  judgeModel.textContent = judge.model || "-";
  judgeStatus.textContent = judge.status || "-";
  judgeFile.textContent = judge.fileName || "-";
  judgeUpdatedAt.textContent = judge.updatedAt || "-";
  judgeResponse.textContent = formatMultiline(judge.response, "(no judge response recorded)");
}

function renderStep() {
  const steps = state.taskDetail?.steps || [];
  if (!steps.length) {
    stepLabel.textContent = "Step - / -";
    stepThought.textContent = "-";
    stepAction.textContent = "-";
    stepUrl.textContent = "-";
    stepTitle.textContent = "-";
    stepSuccess.textContent = "-";
    stepException.textContent = "-";
    stepConsole.textContent = "-";
    stepAria.textContent = "-";
    stepImage.removeAttribute("src");
    stepImage.hidden = true;
    stepImageEmpty.hidden = false;
    prevStepBtn.disabled = true;
    nextStepBtn.disabled = true;
    return;
  }

  state.stepIndex = Math.max(0, Math.min(state.stepIndex, steps.length - 1));
  const step = steps[state.stepIndex];
  stepLabel.textContent = `Step ${state.stepIndex + 1} / ${steps.length}`;
  stepThought.textContent = formatMultiline(step.thought, "(no thought recorded)");
  stepAction.textContent = formatMultiline(step.action, "(no action recorded)");
  stepUrl.textContent = formatMultiline(step.url);
  stepTitle.textContent = formatMultiline(step.title);
  stepSuccess.textContent = step.success == null ? "-" : String(step.success);
  stepException.textContent = formatMultiline(step.exception);
  stepConsole.textContent = formatMultiline(step.consoleOutput || step.recentConsole);
  stepAria.textContent = formatMultiline(step.ariaSnapshot);

  const screenshotUrl = buildArtifactUrl(step);
  if (screenshotUrl) {
    stepImage.src = screenshotUrl;
    stepImage.hidden = false;
    stepImageEmpty.hidden = true;
  } else {
    stepImage.removeAttribute("src");
    stepImage.hidden = true;
    stepImageEmpty.hidden = false;
  }

  prevStepBtn.disabled = state.stepIndex <= 0;
  nextStepBtn.disabled = state.stepIndex >= steps.length - 1;
}

function applyTaskFilter() {
  const query = taskSearch.value.trim().toLowerCase();
  state.filteredTasks = state.tasks.filter((task) => {
    if (!query) return true;
    return [task.taskId, task.title, task.finalPreview].some((value) =>
      String(value || "").toLowerCase().includes(query)
    );
  });

  if (!state.filteredTasks.some((task) => task.id === state.selectedTask)) {
    state.selectedTask = state.filteredTasks[0]?.id || null;
    state.taskDetail = null;
  }

  renderTaskList();
}

function populateRunSelect() {
  runSelect.innerHTML = "";
  for (const run of state.runs) {
    const option = document.createElement("option");
    option.value = run.id;
    option.textContent = `${run.name} (${run.taskCount})`;
    runSelect.appendChild(option);
  }
  runSelect.value = state.selectedRun || "";
}

async function loadRuns() {
  setStatus("Loading runs...");
  const payload = await fetchJson("/api/runs");
  state.runs = payload.runs || [];
  rootDir.textContent = payload.rootDir || "-";
  runCount.textContent = String(state.runs.length);

  if (!state.runs.length) {
    state.selectedRun = null;
    state.tasks = [];
    state.filteredTasks = [];
    state.selectedTask = null;
    state.taskDetail = null;
    populateRunSelect();
    renderTaskList();
    renderTaskDetail();
    setStatus("No trace runs found under the configured root.");
    return;
  }

  if (!state.selectedRun || !state.runs.some((run) => run.id === state.selectedRun)) {
    state.selectedRun = state.runs[0].id;
  }

  populateRunSelect();
  await loadTasks();
}

async function loadTasks() {
  if (!state.selectedRun) return;
  setStatus(`Loading tasks for ${state.selectedRun}...`);
  const payload = await fetchJson(`/api/tasks?run=${encodeURIComponent(state.selectedRun)}`);
  state.tasks = payload.tasks || [];
  if (!state.tasks.some((task) => task.id === state.selectedTask)) {
    state.selectedTask = state.tasks[0]?.id || null;
    state.taskDetail = null;
  }
  applyTaskFilter();
  if (state.selectedTask) {
    await loadTaskDetail();
  } else {
    renderTaskDetail();
    setStatus("This run has no task artifacts.");
  }
}

async function loadTaskDetail() {
  if (!state.selectedRun || !state.selectedTask) return;
  setStatus(`Loading task ${state.selectedTask}...`);
  state.taskDetail = await fetchJson(
    `/api/task?run=${encodeURIComponent(state.selectedRun)}&task=${encodeURIComponent(state.selectedTask)}`
  );
  state.stepIndex = 0;
  state.judgeIndex = 0;
  renderTaskList();
  renderTaskDetail();
  setStatus(`Showing ${state.taskDetail.taskId}`);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

runSelect.addEventListener("change", async () => {
  state.selectedRun = runSelect.value || null;
  state.selectedTask = null;
  state.taskDetail = null;
  await loadTasks();
});

taskSearch.addEventListener("input", () => {
  applyTaskFilter();
  renderTaskDetail();
});

refreshBtn.addEventListener("click", async () => {
  await loadRuns();
});

taskList.addEventListener("click", async (event) => {
  const button = event.target.closest("button[data-task-id]");
  if (!button) return;
  state.selectedTask = button.dataset.taskId || null;
  await loadTaskDetail();
});

prevStepBtn.addEventListener("click", () => {
  state.stepIndex -= 1;
  renderStep();
});

nextStepBtn.addEventListener("click", () => {
  state.stepIndex += 1;
  renderStep();
});

judgeSelect.addEventListener("change", () => {
  state.judgeIndex = Number.parseInt(judgeSelect.value || "0", 10) || 0;
  renderJudgeDetail();
});

loadRuns().catch((error) => {
  setStatus(`Failed to load trace viewer data: ${error.message}`);
});
