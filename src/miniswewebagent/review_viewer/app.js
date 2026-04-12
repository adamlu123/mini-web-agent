const state = {
  runs: [],
  judgeFiles: [],
  tasks: [],
  filteredTasks: [],
  selectedRun: null,
  selectedJudge: null,
  selectedTask: null,
  taskDetail: null,
  selectedImageIndex: 0,
  pendingImageView: null,
  imageView: {
    baseScale: 1,
    zoom: 1,
    minZoom: 1,
    maxZoom: 8,
    panX: 0,
    panY: 0,
    pointers: new Map(),
    dragging: false,
    dragPointerId: null,
    dragStartX: 0,
    dragStartY: 0,
    startPanX: 0,
    startPanY: 0,
    pinchStartDistance: 0,
    pinchStartZoom: 1,
    pinchImagePoint: null,
  },
};

const runSelect = document.getElementById("runSelect");
const judgeSelect = document.getElementById("judgeSelect");
const taskSearch = document.getElementById("taskSearch");
const refreshBtn = document.getElementById("refreshBtn");
const statusText = document.getElementById("statusText");
const runsRoot = document.getElementById("runsRoot");
const judgeRoot = document.getElementById("judgeRoot");
const taskCount = document.getElementById("taskCount");
const taskList = document.getElementById("taskList");

const taskId = document.getElementById("taskId");
const taskJudgeStatus = document.getElementById("taskJudgeStatus");
const taskStartUrl = document.getElementById("taskStartUrl");
const taskText = document.getElementById("taskText");

const imageCountLabel = document.getElementById("imageCountLabel");
const imageList = document.getElementById("imageList");
const imageViewport = document.getElementById("imageViewport");
const selectedImage = document.getElementById("selectedImage");
const selectedImageEmpty = document.getElementById("selectedImageEmpty");
const prevImageBtn = document.getElementById("prevImageBtn");
const nextImageBtn = document.getElementById("nextImageBtn");
const imagePositionLabel = document.getElementById("imagePositionLabel");
const zoomOutBtn = document.getElementById("zoomOutBtn");
const zoomInBtn = document.getElementById("zoomInBtn");
const zoomResetBtn = document.getElementById("zoomResetBtn");
const zoomLabel = document.getElementById("zoomLabel");

const judgeMeta = document.getElementById("judgeMeta");
const judgeStatus = document.getElementById("judgeStatus");
const judgeModel = document.getElementById("judgeModel");
const judgeReason = document.getElementById("judgeReason");
const judgeResponse = document.getElementById("judgeResponse");
const finalResponse = document.getElementById("finalResponse");
const lastAction = document.getElementById("lastAction");
const exitStatus = document.getElementById("exitStatus");
const lastThought = document.getElementById("lastThought");

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    let message = response.statusText;
    const bodyText = await response.text();
    try {
      const payload = JSON.parse(bodyText);
      message = payload.error || message;
    } catch {
      message = bodyText || message;
    }
    throw new Error(message);
  }
  return response.json();
}

function setStatus(message) {
  statusText.textContent = message;
}

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatText(value, fallback = "-") {
  const text = String(value || "").trim();
  return text || fallback;
}

function buildArtifactUrl(image) {
  if (!state.selectedRun || !state.selectedTask || !image?.relPath) {
    return "";
  }
  const params = new URLSearchParams({
    run: state.selectedRun,
    task: state.selectedTask,
    file: image.relPath,
  });
  return `/artifact?${params.toString()}`;
}

function hasLoadedImage() {
  return Boolean(selectedImage.src) && !selectedImage.hidden && selectedImage.naturalWidth > 0;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function resetPinchState() {
  state.imageView.pinchStartDistance = 0;
  state.imageView.pinchStartZoom = state.imageView.zoom;
  state.imageView.pinchImagePoint = null;
}

function clampImagePan() {
  if (!hasLoadedImage()) {
    return;
  }
  const view = state.imageView;
  const viewportWidth = imageViewport.clientWidth;
  const viewportHeight = imageViewport.clientHeight;
  const totalScale = view.baseScale * view.zoom;
  const scaledWidth = selectedImage.naturalWidth * totalScale;
  const scaledHeight = selectedImage.naturalHeight * totalScale;

  if (scaledWidth <= viewportWidth) {
    view.panX = (viewportWidth - scaledWidth) / 2;
  } else {
    view.panX = clamp(view.panX, viewportWidth - scaledWidth, 0);
  }

  if (scaledHeight <= viewportHeight) {
    view.panY = (viewportHeight - scaledHeight) / 2;
  } else {
    view.panY = clamp(view.panY, viewportHeight - scaledHeight, 0);
  }
}

function applyImageTransform() {
  if (!hasLoadedImage()) {
    zoomLabel.textContent = "100%";
    return;
  }
  clampImagePan();
  const totalScale = state.imageView.baseScale * state.imageView.zoom;
  selectedImage.style.transform = `translate(${state.imageView.panX}px, ${state.imageView.panY}px) scale(${totalScale})`;
  zoomLabel.textContent = `${Math.round(state.imageView.zoom * 100)}%`;
}

function updateZoomButtons() {
  const enabled = hasLoadedImage();
  zoomOutBtn.disabled = !enabled;
  zoomInBtn.disabled = !enabled;
  zoomResetBtn.disabled = !enabled;
}

function captureCurrentImageView() {
  if (!hasLoadedImage()) {
    return null;
  }
  const rect = imageViewport.getBoundingClientRect();
  const view = state.imageView;
  const totalScale = view.baseScale * view.zoom;
  if (!rect.width || !rect.height || !totalScale) {
    return null;
  }

  const centerX = rect.width / 2;
  const centerY = rect.height / 2;
  return {
    zoom: view.zoom,
    focusX: (centerX - view.panX) / totalScale,
    focusY: (centerY - view.panY) / totalScale,
  };
}

function updateImageNav() {
  const images = state.taskDetail?.images || [];
  const hasImages = images.length > 0;
  prevImageBtn.disabled = !hasImages || state.selectedImageIndex <= 0;
  nextImageBtn.disabled = !hasImages || state.selectedImageIndex >= images.length - 1;
  imagePositionLabel.textContent = hasImages
    ? `${state.selectedImageIndex + 1} / ${images.length}`
    : "- / -";
}

function resetImageView() {
  if (!hasLoadedImage()) {
    zoomLabel.textContent = "100%";
    updateZoomButtons();
    return;
  }
  const view = state.imageView;
  const viewportWidth = imageViewport.clientWidth;
  const viewportHeight = imageViewport.clientHeight;
  const naturalWidth = selectedImage.naturalWidth;
  const naturalHeight = selectedImage.naturalHeight;
  if (!viewportWidth || !viewportHeight || !naturalWidth || !naturalHeight) {
    return;
  }

  view.baseScale = Math.min(viewportWidth / naturalWidth, viewportHeight / naturalHeight);
  view.zoom = 1;
  view.minZoom = 1;
  view.panX = (viewportWidth - naturalWidth * view.baseScale) / 2;
  view.panY = (viewportHeight - naturalHeight * view.baseScale) / 2;
  view.pointers.clear();
  view.dragging = false;
  view.dragPointerId = null;
  resetPinchState();
  imageViewport.classList.remove("dragging");
  applyImageTransform();
  updateZoomButtons();
}

function applyCapturedImageView(capturedView) {
  if (!hasLoadedImage() || !capturedView) {
    resetImageView();
    return;
  }
  const view = state.imageView;
  const viewportWidth = imageViewport.clientWidth;
  const viewportHeight = imageViewport.clientHeight;
  const naturalWidth = selectedImage.naturalWidth;
  const naturalHeight = selectedImage.naturalHeight;
  if (!viewportWidth || !viewportHeight || !naturalWidth || !naturalHeight) {
    resetImageView();
    return;
  }

  view.baseScale = Math.min(viewportWidth / naturalWidth, viewportHeight / naturalHeight);
  view.zoom = clamp(capturedView.zoom || 1, view.minZoom, view.maxZoom);
  const totalScale = view.baseScale * view.zoom;
  view.panX = viewportWidth / 2 - (capturedView.focusX || 0) * totalScale;
  view.panY = viewportHeight / 2 - (capturedView.focusY || 0) * totalScale;
  view.pointers.clear();
  view.dragging = false;
  view.dragPointerId = null;
  resetPinchState();
  imageViewport.classList.remove("dragging");
  applyImageTransform();
  updateZoomButtons();
}

function zoomAt(nextZoom, clientX, clientY) {
  if (!hasLoadedImage()) {
    return;
  }
  const view = state.imageView;
  const clampedZoom = clamp(nextZoom, view.minZoom, view.maxZoom);
  if (Math.abs(clampedZoom - view.zoom) < 0.001) {
    return;
  }

  const rect = imageViewport.getBoundingClientRect();
  const targetX = clientX ?? rect.left + rect.width / 2;
  const targetY = clientY ?? rect.top + rect.height / 2;
  const frameX = targetX - rect.left;
  const frameY = targetY - rect.top;

  const currentScale = view.baseScale * view.zoom;
  const imageX = (frameX - view.panX) / currentScale;
  const imageY = (frameY - view.panY) / currentScale;

  view.zoom = clampedZoom;
  const nextScale = view.baseScale * view.zoom;
  view.panX = frameX - imageX * nextScale;
  view.panY = frameY - imageY * nextScale;
  applyImageTransform();
}

function panBy(deltaX, deltaY) {
  if (!hasLoadedImage()) {
    return;
  }
  state.imageView.panX += deltaX;
  state.imageView.panY += deltaY;
  applyImageTransform();
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

function populateJudgeSelect() {
  judgeSelect.innerHTML = "";
  if (!state.judgeFiles.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No judge files found";
    judgeSelect.appendChild(option);
    judgeSelect.disabled = true;
    return;
  }

  for (const file of state.judgeFiles) {
    const option = document.createElement("option");
    option.value = file.id;
    option.textContent = `${file.kind} · ${file.name}`;
    judgeSelect.appendChild(option);
  }
  judgeSelect.disabled = false;
  judgeSelect.value = state.selectedJudge || "";
}

function renderTaskList() {
  taskList.innerHTML = "";
  taskCount.textContent = String(state.filteredTasks.length);

  if (!state.filteredTasks.length) {
    taskList.innerHTML = '<li class="empty-state">No tasks match the current run and search.</li>';
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
    button.classList.add(task.judgeStatus || "unknown");
    button.textContent = task.taskId;
    li.appendChild(button);
    taskList.appendChild(li);
  }
}

function renderImageList() {
  const images = state.taskDetail?.images || [];
  imageList.innerHTML = "";
  imageCountLabel.textContent = `${images.length} image(s)`;

  if (!images.length) {
    state.pendingImageView = null;
    imageViewport.hidden = true;
    selectedImage.hidden = true;
    selectedImage.removeAttribute("src");
    selectedImage.style.transform = "";
    selectedImageEmpty.hidden = false;
    imageList.innerHTML = '<p class="empty-state">No screenshots under screenshots/.</p>';
    updateZoomButtons();
    updateImageNav();
    zoomLabel.textContent = "100%";
    return;
  }

  state.selectedImageIndex = Math.max(0, Math.min(state.selectedImageIndex, images.length - 1));

  for (const [index, image] of images.entries()) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "image-link";
    if (index === state.selectedImageIndex) {
      button.classList.add("selected");
    }
    button.dataset.imageIndex = String(index);
    button.textContent = image.name;
    imageList.appendChild(button);
  }

  const selected = images[state.selectedImageIndex];
  const imageUrl = buildArtifactUrl(selected);
  if (imageUrl) {
    imageViewport.hidden = false;
    selectedImage.src = imageUrl;
    selectedImage.hidden = false;
    selectedImageEmpty.hidden = true;
    updateZoomButtons();
  } else {
    imageViewport.hidden = true;
    selectedImage.hidden = true;
    selectedImage.removeAttribute("src");
    selectedImage.style.transform = "";
    selectedImageEmpty.hidden = false;
    updateZoomButtons();
  }
  updateImageNav();
}

function renderTaskDetail() {
  const detail = state.taskDetail;
  if (!detail) {
    taskId.textContent = "-";
    taskJudgeStatus.textContent = "-";
    taskStartUrl.textContent = "-";
    taskText.textContent = "-";
    judgeMeta.textContent = "-";
    judgeStatus.textContent = "-";
    judgeModel.textContent = "-";
    judgeReason.textContent = "-";
    judgeResponse.textContent = "-";
    finalResponse.textContent = "-";
    lastAction.textContent = "-";
    exitStatus.textContent = "-";
    lastThought.textContent = "-";
    renderImageList();
    return;
  }

  taskId.textContent = detail.taskId || "-";
  taskJudgeStatus.textContent = detail.judge?.status || "-";
  taskStartUrl.textContent = detail.startUrl || "-";
  taskText.textContent = detail.task || "-";

  const judge = detail.judge || {};
  judgeMeta.textContent = judge.fileId || "No judge file selected.";
  judgeStatus.textContent = judge.status || "unknown";
  judgeModel.textContent = judge.model || "-";
  judgeReason.textContent = formatText(judge.reason, "No judge reason found for this task in the selected judge file.");
  judgeResponse.textContent = formatText(judge.response, "No judge response found.");
  finalResponse.textContent = formatText(detail.finalResponse, "No final response found.");
  lastAction.textContent = formatText(detail.lastAction, "No action history found.");
  exitStatus.textContent = formatText(detail.exitStatus, "No exit status found.");
  lastThought.textContent = formatText(detail.lastThought, "No thought history found.");

  renderImageList();
}

function applyTaskFilter() {
  const query = taskSearch.value.trim().toLowerCase();
  state.filteredTasks = state.tasks.filter((task) => {
    if (!query) {
      return true;
    }
    return [task.taskId, task.title, task.judgeReasonPreview]
      .some((value) => String(value || "").toLowerCase().includes(query));
  });

  if (!state.filteredTasks.some((task) => task.id === state.selectedTask)) {
    state.selectedTask = state.filteredTasks[0]?.id || null;
    state.taskDetail = null;
  }

  renderTaskList();
}

async function loadRuns() {
  setStatus("Loading run folders...");
  const payload = await fetchJson("/api/runs");
  state.runs = payload.runs || [];
  runsRoot.textContent = payload.runsRoot || "-";
  judgeRoot.textContent = payload.judgeRoot || "-";

  if (!state.runs.length) {
    state.selectedRun = null;
    state.tasks = [];
    state.filteredTasks = [];
    state.selectedTask = null;
    state.taskDetail = null;
    populateRunSelect();
    renderTaskList();
    renderTaskDetail();
    setStatus("No run folders found.");
    return;
  }

  if (!state.selectedRun || !state.runs.some((run) => run.id === state.selectedRun)) {
    state.selectedRun = state.runs[0].id;
  }
  populateRunSelect();
}

async function loadJudgeFiles() {
  setStatus("Loading judge files...");
  const payload = await fetchJson("/api/judges");
  state.judgeFiles = payload.judgeFiles || [];

  if (!state.judgeFiles.length) {
    state.selectedJudge = null;
    populateJudgeSelect();
    return;
  }

  if (!state.selectedJudge || !state.judgeFiles.some((file) => file.id === state.selectedJudge)) {
    state.selectedJudge = state.judgeFiles[0].id;
  }
  populateJudgeSelect();
}

async function loadTasks() {
  if (!state.selectedRun) {
    return;
  }
  setStatus(`Loading tasks for ${state.selectedRun}...`);
  const params = new URLSearchParams({ run: state.selectedRun });
  if (state.selectedJudge) {
    params.set("judge", state.selectedJudge);
  }
  const payload = await fetchJson(`/api/tasks?${params.toString()}`);
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
  if (!state.selectedRun || !state.selectedTask) {
    return;
  }
  setStatus(`Loading task ${state.selectedTask}...`);
  const params = new URLSearchParams({
    run: state.selectedRun,
    task: state.selectedTask,
  });
  if (state.selectedJudge) {
    params.set("judge", state.selectedJudge);
  }
  state.taskDetail = await fetchJson(`/api/task?${params.toString()}`);
  state.selectedImageIndex = 0;
  state.pendingImageView = null;
  renderTaskList();
  renderTaskDetail();
  setStatus(`Showing ${state.taskDetail.taskId}`);
}

runSelect.addEventListener("change", async () => {
  state.selectedRun = runSelect.value || null;
  state.selectedTask = null;
  state.taskDetail = null;
  await loadTasks();
});

judgeSelect.addEventListener("change", async () => {
  state.selectedJudge = judgeSelect.value || null;
  state.selectedTask = null;
  state.taskDetail = null;
  await loadTasks();
});

taskSearch.addEventListener("input", () => {
  applyTaskFilter();
  renderTaskDetail();
});

refreshBtn.addEventListener("click", async () => {
  await initialize();
});

taskList.addEventListener("click", async (event) => {
  const button = event.target.closest("button[data-task-id]");
  if (!button) {
    return;
  }
  state.selectedTask = button.dataset.taskId || null;
  await loadTaskDetail();
});

imageList.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-image-index]");
  if (!button) {
    return;
  }
  state.pendingImageView = captureCurrentImageView();
  state.selectedImageIndex = Number.parseInt(button.dataset.imageIndex || "0", 10) || 0;
  renderImageList();
});

prevImageBtn.addEventListener("click", () => {
  if (state.selectedImageIndex <= 0) {
    return;
  }
  state.pendingImageView = captureCurrentImageView();
  state.selectedImageIndex -= 1;
  renderImageList();
});

nextImageBtn.addEventListener("click", () => {
  const images = state.taskDetail?.images || [];
  if (state.selectedImageIndex >= images.length - 1) {
    return;
  }
  state.pendingImageView = captureCurrentImageView();
  state.selectedImageIndex += 1;
  renderImageList();
});

zoomOutBtn.addEventListener("click", () => {
  const rect = imageViewport.getBoundingClientRect();
  zoomAt(state.imageView.zoom / 1.2, rect.left + rect.width / 2, rect.top + rect.height / 2);
});

zoomInBtn.addEventListener("click", () => {
  const rect = imageViewport.getBoundingClientRect();
  zoomAt(state.imageView.zoom * 1.2, rect.left + rect.width / 2, rect.top + rect.height / 2);
});

zoomResetBtn.addEventListener("click", () => {
  resetImageView();
});

selectedImage.addEventListener("load", () => {
  const pendingView = state.pendingImageView;
  state.pendingImageView = null;
  if (pendingView) {
    applyCapturedImageView(pendingView);
    return;
  }
  resetImageView();
});

imageViewport.addEventListener(
  "wheel",
  (event) => {
    if (!hasLoadedImage()) {
      return;
    }
    event.preventDefault();
    const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
    zoomAt(state.imageView.zoom * factor, event.clientX, event.clientY);
  },
  { passive: false }
);

imageViewport.addEventListener("pointerdown", (event) => {
  if (!hasLoadedImage()) {
    return;
  }
  imageViewport.focus();
  imageViewport.setPointerCapture(event.pointerId);
  state.imageView.pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });

  if (state.imageView.pointers.size === 1) {
    state.imageView.dragging = true;
    state.imageView.dragPointerId = event.pointerId;
    state.imageView.dragStartX = event.clientX;
    state.imageView.dragStartY = event.clientY;
    state.imageView.startPanX = state.imageView.panX;
    state.imageView.startPanY = state.imageView.panY;
    imageViewport.classList.add("dragging");
  } else if (state.imageView.pointers.size === 2) {
    state.imageView.dragging = false;
    state.imageView.dragPointerId = null;
    imageViewport.classList.remove("dragging");
    const points = Array.from(state.imageView.pointers.values());
    const centerX = (points[0].x + points[1].x) / 2;
    const centerY = (points[0].y + points[1].y) / 2;
    const rect = imageViewport.getBoundingClientRect();
    const currentScale = state.imageView.baseScale * state.imageView.zoom;
    state.imageView.pinchStartDistance = Math.hypot(points[0].x - points[1].x, points[0].y - points[1].y);
    state.imageView.pinchStartZoom = state.imageView.zoom;
    state.imageView.pinchImagePoint = {
      x: (centerX - rect.left - state.imageView.panX) / currentScale,
      y: (centerY - rect.top - state.imageView.panY) / currentScale,
    };
  }
});

imageViewport.addEventListener("pointermove", (event) => {
  if (!hasLoadedImage() || !state.imageView.pointers.has(event.pointerId)) {
    return;
  }
  state.imageView.pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });

  if (state.imageView.pointers.size === 2 && state.imageView.pinchImagePoint) {
    const points = Array.from(state.imageView.pointers.values());
    const centerX = (points[0].x + points[1].x) / 2;
    const centerY = (points[0].y + points[1].y) / 2;
    const distance = Math.hypot(points[0].x - points[1].x, points[0].y - points[1].y);
    if (!state.imageView.pinchStartDistance) {
      return;
    }
    const nextZoom = state.imageView.pinchStartZoom * (distance / state.imageView.pinchStartDistance);
    const clampedZoom = clamp(nextZoom, state.imageView.minZoom, state.imageView.maxZoom);
    const rect = imageViewport.getBoundingClientRect();
    const nextScale = state.imageView.baseScale * clampedZoom;
    state.imageView.zoom = clampedZoom;
    state.imageView.panX = centerX - rect.left - state.imageView.pinchImagePoint.x * nextScale;
    state.imageView.panY = centerY - rect.top - state.imageView.pinchImagePoint.y * nextScale;
    applyImageTransform();
    return;
  }

  if (state.imageView.dragging && event.pointerId === state.imageView.dragPointerId) {
    const deltaX = event.clientX - state.imageView.dragStartX;
    const deltaY = event.clientY - state.imageView.dragStartY;
    state.imageView.panX = state.imageView.startPanX + deltaX;
    state.imageView.panY = state.imageView.startPanY + deltaY;
    applyImageTransform();
  }
});

function releasePointer(event) {
  state.imageView.pointers.delete(event.pointerId);
  if (state.imageView.pointers.size < 2) {
    resetPinchState();
  }
  if (event.pointerId === state.imageView.dragPointerId || state.imageView.pointers.size === 0) {
    state.imageView.dragging = false;
    state.imageView.dragPointerId = null;
    imageViewport.classList.remove("dragging");
  }
}

imageViewport.addEventListener("pointerup", releasePointer);
imageViewport.addEventListener("pointercancel", releasePointer);

imageViewport.addEventListener("keydown", (event) => {
  if (!hasLoadedImage()) {
    return;
  }
  const rect = imageViewport.getBoundingClientRect();
  const centerX = rect.left + rect.width / 2;
  const centerY = rect.top + rect.height / 2;
  switch (event.key) {
    case "+":
    case "=":
      event.preventDefault();
      zoomAt(state.imageView.zoom * 1.12, centerX, centerY);
      break;
    case "-":
    case "_":
      event.preventDefault();
      zoomAt(state.imageView.zoom / 1.12, centerX, centerY);
      break;
    case "0":
      event.preventDefault();
      resetImageView();
      break;
    case "ArrowLeft":
      event.preventDefault();
      panBy(40, 0);
      break;
    case "ArrowRight":
      event.preventDefault();
      panBy(-40, 0);
      break;
    case "ArrowUp":
      event.preventDefault();
      panBy(0, 40);
      break;
    case "ArrowDown":
      event.preventDefault();
      panBy(0, -40);
      break;
    default:
      break;
  }
});

window.addEventListener("resize", () => {
  if (hasLoadedImage()) {
    resetImageView();
  }
});

async function initialize() {
  imageViewport.hidden = true;
  updateZoomButtons();
  updateImageNav();
  state.pendingImageView = null;
  await loadRuns();
  await loadJudgeFiles();
  await loadTasks();
}

initialize().catch((error) => {
  setStatus(`Failed to load review viewer data: ${error.message}`);
});
