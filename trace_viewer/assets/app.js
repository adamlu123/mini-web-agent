(() => {
  "use strict";

  const catalog = window.TRACE_CATALOG;
  const state = {
    task: null,
    events: [],
    blocks: [],
    visibleBlocks: [],
    query: "",
    role: "all",
    errorsOnly: false,
    expandAll: false,
    showPrompts: false,
    outlineIds: [],
    observer: null,
    scrollHandler: null,
    activeBlockId: "",
    imageTrigger: null,
    images: [],
    imageIndex: -1,
  };

  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => [...document.querySelectorAll(selector)];

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function viewerTitle() {
    return catalog.title || "CUAWright traces";
  }

  function collectionMeta() {
    if (catalog.collection) return catalog.collection;
    const title = String(catalog.title || "");
    const lowered = title.toLowerCase();
    let benchmark = "generic";
    let benchmarkLabel = "Evaluation";
    let metricLabel = "Score";
    let metricKind = "score";
    if (lowered.includes("cadgenbench")) {
      benchmark = "cadgenbench";
      benchmarkLabel = "CADGenBench";
      metricLabel = "Official score";
      metricKind = "aggregate";
    } else if (lowered.includes("benchcad")) {
      benchmark = "benchcad";
      benchmarkLabel = "BenchCAD";
      metricLabel = "IoU";
      metricKind = "overlap";
    } else if (lowered.includes("weavebench")) {
      benchmark = "weavebench";
      benchmarkLabel = "WeaveBench";
      metricLabel = "Task score";
      metricKind = "binary";
    } else if (lowered.includes("osworld")) {
      benchmark = "osworld-v2";
      benchmarkLabel = "OSWorld V2";
      metricLabel = "Partial reward";
      metricKind = "reward";
    }
    const titleParts = title.split("·").map((part) => part.trim()).filter(Boolean);
    return {
      benchmark,
      benchmarkLabel,
      metricLabel,
      metricKind,
      modelLabel: titleParts.length > 1 ? titleParts.slice(1).join(" · ") : "",
    };
  }

  function metricLabel(task) {
    return task.metricLabel || collectionMeta().metricLabel || "Score";
  }

  function benchmarkLabel(task) {
    return (
      task.benchmarkLabel ||
      collectionMeta().benchmarkLabel ||
      task.benchmark ||
      "Evaluation"
    );
  }

  function evaluationState(task) {
    return String(task.evaluationStatus || "").trim().toLowerCase();
  }

  function formatScore(value) {
    if (!Number.isFinite(value)) return "—";
    return value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
  }

  function formatDuration(seconds) {
    if (!Number.isFinite(seconds)) return "—";
    const total = Math.max(0, Math.round(seconds));
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    const remainder = total % 60;
    if (hours) return `${hours}h ${minutes}m`;
    if (minutes) return `${minutes}m ${remainder}s`;
    return `${remainder}s`;
  }

  function fragmentId(value) {
    return String(value || "")
      .replace(/[^A-Za-z0-9_-]+/g, "-")
      .replace(/^-+|-+$/g, "");
  }

  function icon(name, extra = "") {
    return `<span class="material-symbols-rounded ${extra}" aria-hidden="true">${name}</span>`;
  }

  function statusClass(task) {
    const evaluation = evaluationState(task);
    const metricKind = task.metricKind || collectionMeta().metricKind;
    if (task.traceStatus === "missing" || task.status !== "evaluated") return "failed";
    if (
      evaluation.includes("pending") ||
      evaluation.includes("trace available") ||
      evaluation.includes("unavailable")
    ) {
      return "pending";
    }
    if (metricKind === "binary") return task.score >= 1 ? "pass" : "zero";
    if (task.strictPass) return "pass";
    if (task.score === 0) return "zero";
    return "partial";
  }

  function statusText(task) {
    const evaluation = evaluationState(task);
    const metricKind = task.metricKind || collectionMeta().metricKind;
    if (task.traceStatus === "missing" || task.status !== "evaluated") return "Trace unavailable";
    if (evaluation.includes("pending")) return "Evaluation pending";
    if (evaluation.includes("trace available")) return "Trace available";
    if (evaluation.includes("unavailable")) return "Evaluation unavailable";
    if (metricKind === "binary") return task.score >= 1 ? "Pass" : "Not passed";
    if (metricKind === "overlap") return "IoU recorded";
    if (task.strictPass) return "Full reward";
    if (task.score === 0) return "Zero reward";
    if (task.score == null) return "Trace available";
    return "Partial reward";
  }

  function collectionDescription() {
    const benchmark = collectionMeta().benchmark;
    const descriptions = {
      "osworld-v2":
        "Desktop computer-use trajectories with task instructions, model actions, visual evidence, specialist reviews, and partial reward.",
      weavebench:
        "End-to-end software workflow trajectories with commands, artifacts, verification records, and task-level outcomes.",
      benchcad:
        "Vision-to-CAD reconstruction trajectories with reference evidence, generated geometry, verification, and IoU results.",
      cadgenbench:
        "CAD generation trajectories with modeling steps, rendered evidence, public validation, and available benchmark results.",
    };
    return descriptions[benchmark] || "Inspectable model trajectories, evidence, and evaluation outcomes.";
  }

  function renderCollectionHero() {
    const meta = collectionMeta();
    const summary = catalog.summary || {};
    const mean = Number.isFinite(summary.mean) ? formatScore(summary.mean) : "—";
    $("#collection-hero").innerHTML = `
      <span class="collection-eyebrow">${escapeHtml(
        meta.benchmarkLabel || "Trajectory collection",
      )}</span>
      <h1>${escapeHtml(viewerTitle())}</h1>
      <p>${escapeHtml(collectionDescription())}</p>
      <div class="collection-stats">
        <span><b>${escapeHtml(summary.taskCount ?? catalog.tasks.length)}</b> tasks</span>
        <span><b>${escapeHtml(summary.evaluated ?? "—")}</b> evaluated</span>
        <span><b>${escapeHtml(mean)}</b> mean ${escapeHtml(
          meta.metricLabel || "score",
        ).toLowerCase()}</span>
      </div>`;
  }

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = src;
      script.onload = resolve;
      script.onerror = () => reject(new Error(`Could not load ${src}`));
      document.head.append(script);
    });
  }

  function populateTaskSelect(selected = "") {
    $("#task-select").innerHTML = [
      '<option value="">Choose a task…</option>',
      ...catalog.tasks.map(
        (task) =>
          `<option value="${escapeHtml(task.id)}" ${
            task.id === selected ? "selected" : ""
          }>${escapeHtml(task.id)} · ${escapeHtml(
            task.title || task.id,
          )}</option>`,
      ),
    ].join("");
  }

  function openTask(taskId) {
    location.href = `index.html?task=${encodeURIComponent(taskId)}`;
  }

  function softwareIcon(label) {
    const value = String(label || "").toLowerCase();
    if (value.includes("mail")) return "mail";
    if (value.includes("calendar")) return "calendar_month";
    if (value.includes("calc") || value.includes("spreadsheet")) return "table_chart";
    if (value.includes("writer") || value.includes("document") || value.includes("pdf")) {
      return "description";
    }
    if (value.includes("terminal") || value.includes("conda")) return "terminal";
    if (value.includes("game")) return "sports_esports";
    if (value.includes("travel") || value.includes("hotel")) return "hotel";
    if (value.includes("chrome") || value.includes("browser") || value.includes("web")) {
      return "language";
    }
    return "visibility";
  }

  function taskRow(task) {
    const software = task.relatedSoftware || [];
    const thumbnail = task.thumbnail;
    const fallbackLabel = software[0] || benchmarkLabel(task);
    return `
      <button class="task-row" type="button" data-task="${escapeHtml(task.id)}">
        <span class="task-row-thumbnail ${thumbnail?.path ? "" : "fallback"}">
          ${
            thumbnail?.path
              ? `<img src="${escapeHtml(thumbnail.path)}" alt="" width="${
                  thumbnail.width || 160
                }" height="${thumbnail.height || 100}" loading="lazy">`
              : `${icon(softwareIcon(fallbackLabel))}
                 <small>${escapeHtml(fallbackLabel)}</small>`
          }
        </span>
        <span class="task-row-main">
          <span class="task-row-id">${escapeHtml(task.id)}</span>
          <span class="task-row-instruction">${escapeHtml(task.instruction)}</span>
          ${
            software.length
              ? `<span class="task-row-software">${software
                  .slice(0, 4)
                  .map((item) => `<span>${escapeHtml(item)}</span>`)
                  .join("")}</span>`
              : ""
          }
        </span>
        <span class="task-row-score">${formatScore(task.score)}</span>
        <span class="task-row-calls">${
          task.callsUsed == null ? "—" : `${task.callsUsed} calls`
        }</span>
        <span class="task-row-status"><span class="status-chip ${statusClass(
          task,
        )}">${escapeHtml(statusText(task))}</span></span>
        <span class="task-row-arrow">${icon("chevron_right")}</span>
      </button>`;
  }

  function renderTaskList() {
    state.task = null;
    state.images = [];
    populateTaskSelect();
    renderCollectionHero();
    $("#workspace").classList.add("hidden");
    $("#task-list-view").classList.remove("hidden");
    $("#step-nav").classList.add("hidden");
    $("#task-score").textContent =
      `${catalog.summary.evaluated}/${catalog.summary.taskCount}`;
    $("#score-label").textContent = "Traces";
    $("#task-status").className = "status-chip partial";
    $("#task-status").textContent = "Browse";
    $("#mobile-outline-toggle").classList.add("hidden");
    $("#loading").classList.add("hidden");
    document.title = viewerTitle();

    const update = () => {
      const query = $("#task-list-search").value.trim().toLowerCase();
      const filter = $("#task-status-filter").value;
      const tasks = catalog.tasks.filter((task) => {
        if (
          query &&
          !`${task.id} ${task.instruction}`.toLowerCase().includes(query)
        ) {
          return false;
        }
        if (filter === "strict" && !task.strictPass) return false;
        if (
          filter === "partial" &&
          !(task.status === "evaluated" && task.score > 0 && !task.strictPass)
        ) {
          return false;
        }
        if (
          filter === "zero" &&
          !(task.status === "evaluated" && task.score === 0)
        ) {
          return false;
        }
        if (filter === "unevaluated" && task.status === "evaluated") return false;
        return true;
      });
      $("#task-list-count").textContent =
        `${tasks.length} task${tasks.length === 1 ? "" : "s"}`;
      $("#task-list").innerHTML = tasks.map(taskRow).join("");
      $$(".task-row").forEach((row) => {
        row.onclick = () => openTask(row.dataset.task);
      });
    };
    $("#task-list-search").oninput = update;
    $("#task-status-filter").oninput = update;
    update();
  }

  function parseReport(text) {
    if (!text) return null;
    try {
      const value = JSON.parse(text);
      if (value && typeof value === "object" && value.report) return value;
    } catch {
      return null;
    }
    return null;
  }

  function buildBlocks(events, subagents) {
    const blocks = [];
    const specialistCounts = {};
    const usedArtifacts = new Set();
    const hardBoundaries = new Set([
      "specialist_context_prepared",
      "submission_accepted",
      "terminal_orchestration_completed",
      "evaluation",
      "infrastructure_failure",
    ]);
    for (let index = 0; index < events.length; index += 1) {
      const event = events[index];
      if (event.kind !== "specialist_context_prepared") {
        blocks.push(event);
        continue;
      }
      const invocation = event.raw?.invocation_id || `${event.role}-${index}`;
      const children = [];
      let completed = null;
      let cursor = index + 1;
      const limit = Math.min(events.length, index + 500);
      for (; cursor < limit; cursor += 1) {
        const candidate = events[cursor];
        if (hardBoundaries.has(candidate.kind)) break;
        children.push(candidate);
        if (
          candidate.kind === "specialist_completed" &&
          candidate.raw?.invocation_id === invocation
        ) {
          completed = candidate;
          break;
        }
      }
      const artifact = (subagents || []).find(
        (candidate) =>
          candidate.invocationId === invocation &&
          candidate.role === event.role &&
          !usedArtifacts.has(
            candidate.artifactKey || `${candidate.role}-${candidate.invocationId}`,
          ),
      );
      if (artifact) {
        usedArtifacts.add(
          artifact.artifactKey || `${artifact.role}-${artifact.invocationId}`,
        );
      }
      const finalModel = [...children]
        .reverse()
        .find((candidate) => candidate.kind === "model" && candidate.role === event.role);
      specialistCounts[event.role] = (specialistCounts[event.role] || 0) + 1;
      const identity =
        artifact?.artifactKey ||
        `${event.raw?.phase_id || "phase"}-${invocation}-${specialistCounts[event.role]}`;
      blocks.push({
        id: `specialist-${fragmentId(identity)}`,
        kind: "specialist",
        role: event.role,
        title: `${event.role} consultation #${specialistCounts[event.role]}`,
        invocation: artifact?.artifactKey || `${event.raw?.phase_id || "phase"}/${invocation}`,
        elapsedLabel: event.elapsedLabel,
        timestamp: event.timestamp,
        request: artifact?.request || "",
        report: artifact?.report || parseReport(finalModel?.text),
        rawReport: finalModel?.text || "",
        completed,
        children,
        error: !completed || Boolean(completed.error),
      });
      index = completed ? cursor : Math.max(index, cursor - 1);
    }
    return blocks;
  }

  async function loadTask(taskId) {
    const entry = catalog.tasks.find((task) => task.id === taskId);
    if (!entry) {
      renderTaskList();
      return;
    }
    $("#loading").classList.remove("hidden");
    window.TRACE_TASK_META = null;
    window.TRACE_TASK_CHUNKS = [];
    await loadScript(entry.dataPath);
    for (const chunk of window.TRACE_TASK_META.chunks || []) {
      await loadScript(chunk);
    }
    state.task = window.TRACE_TASK_META;
    state.events = window.TRACE_TASK_CHUNKS.flat();
    state.blocks = buildBlocks(state.events, state.task.subagents || []);
    state.query = "";
    state.role = "all";
    state.errorsOnly = false;
    state.expandAll = false;
    state.showPrompts = false;
    state.activeBlockId = "";
    state.images = collectEvidence(state.events);
    state.imageIndex = -1;
    renderReview();
  }

  function blockText(block) {
    if (block.kind === "specialist") {
      return [
        block.title,
        block.report?.report,
        block.report?.status,
        ...block.children.map((child) => child.searchText || ""),
      ]
        .join(" ")
        .toLowerCase();
    }
    return block.searchText || JSON.stringify(block).toLowerCase();
  }

  function blockHasError(block) {
    if (block.error || block.kind === "response_retry") return true;
    if (block.kind === "specialist") {
      return (block.children || []).some(
        (child) => child.error || child.kind === "response_retry",
      );
    }
    return false;
  }

  function matchesFilters(block) {
    if (state.errorsOnly && !blockHasError(block)) return false;
    if (state.role !== "all") {
      const role = block.role || "System";
      if (role !== state.role) return false;
    }
    return !state.query || blockText(block).includes(state.query);
  }

  function renderReview() {
    const task = state.task;
    populateTaskSelect(task.id);
    $("#task-list-view").classList.add("hidden");
    $("#workspace").classList.remove("hidden");
    $("#step-nav").classList.remove("hidden");
    $("#workspace").classList.remove("outline-open");
    $("#mobile-outline-toggle").classList.remove("hidden");
    $("#mobile-outline-toggle").setAttribute("aria-expanded", "false");
    $("#task-score").textContent = formatScore(task.score);
    $("#score-label").textContent = metricLabel(task);
    $("#task-status").className = `status-chip ${statusClass(task)}`;
    $("#task-status").textContent = statusText(task);
    $("#loading").classList.add("hidden");
    document.title = `Task ${task.id} · ${viewerTitle()}`;

    state.visibleBlocks = state.blocks.filter(matchesFilters);
    $("#transcript").innerHTML = [
      renderOverview(task),
      state.visibleBlocks.length
        ? state.visibleBlocks.map(renderBlock).join("")
        : '<div class="empty-filter">No transcript sections match these filters.</div>',
    ].join("");
    $("#outline-summary").textContent =
      `${state.visibleBlocks.filter((block) => block.kind === "model").length} model calls · ` +
      `${state.visibleBlocks.length} timeline sections` +
      (state.visibleBlocks.length === state.blocks.length
        ? ""
        : ` · filtered from ${state.blocks.length}`);
    renderOutline();
    state.activeBlockId = "";
    bindTranscriptActions();
    syncControlState();
    observeSections();
  }

  function renderOverview(task) {
    const title =
      task.title &&
      !task.instruction.startsWith(task.title) &&
      task.title !== task.id
        ? task.title
        : `Task ${task.id}`;
    const metrics = [
      [metricLabel(task), formatScore(task.score)],
      ["Model steps", task.callsUsed ?? "—"],
      ["Duration", formatDuration(task.durationSeconds)],
      ["Visual evidence", state.images.length],
    ];
    const identity = [
      benchmarkLabel(task),
      task.modelLabel || collectionMeta().modelLabel,
      task.reasoningEffort ? `${task.reasoningEffort} reasoning` : "",
      task.category,
      task.taskType,
      ...(task.relatedSoftware || []).map((item) => `software: ${item}`),
    ].filter(Boolean);
    const instructionOpen = task.instruction.length <= 520 ? "open" : "";
    return `
      <section id="overview" class="review-section overview" data-outline>
        <div class="identity-row">
          ${identity.map((item) => `<span>${escapeHtml(item)}</span>`).join("")}
        </div>
        <h1>${escapeHtml(title)}</h1>
        <p class="task-id-line">Task ${escapeHtml(task.id)} · ${escapeHtml(
          statusText(task),
        )}</p>
        <div class="metric-row">
          ${metrics
            .map(
              ([label, value]) =>
                `<div class="metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(
                  value,
                )}</strong></div>`,
            )
            .join("")}
        </div>
        <details class="fold task-instruction-panel" ${instructionOpen}>
          <summary>
            <span>Task instruction</span>
            <span class="instruction-preview">${escapeHtml(
              task.instruction,
            )}</span>
          </summary>
          <div class="prose">${formatReport(task.instruction)}</div>
        </details>
        ${renderPromptContext(task)}
        ${renderResultSummary(task)}
        ${renderRoleUsage(task.roleUsage || {})}
        ${renderEvidenceStrip()}
      </section>`;
  }

  function renderPromptContext(task) {
    if (!state.showPrompts) return "";
    const prompts = task.prompts || {};
    if (!prompts.available) {
      return `
        <section class="prompt-context unavailable">
          <div class="prompt-context-heading">
            <span class="eyebrow">Model-visible launch context</span>
            <h2>System and user prompts</h2>
          </div>
          <p>${escapeHtml(prompts.note || "Prompt data was not retained for this trajectory.")}</p>
        </section>`;
    }
    const system = prompts.system
      ? `<article class="prompt-card">
           <h3>System prompt</h3>
           <div class="prompt-text">${escapeHtml(prompts.system)}</div>
         </article>`
      : "";
    const user = (prompts.user || [])
      .map(
        (text, index) => `
          <article class="prompt-card">
            <h3>User prompt ${index + 1}</h3>
            <div class="prompt-text">${escapeHtml(text)}</div>
          </article>`,
      )
      .join("");
    return `
      <section class="prompt-context">
        <div class="prompt-context-heading">
          <div>
            <span class="eyebrow">Model-visible launch context</span>
            <h2>System and user prompts</h2>
          </div>
          <span class="prompt-source">${escapeHtml(prompts.source || "")}</span>
        </div>
        <p class="prompt-note">${escapeHtml(prompts.note || "")}</p>
        <div class="prompt-grid">${system}${user}</div>
      </section>`;
  }

  function renderResultSummary(task) {
    const result = task.resultSummary || {};
    const specialists = Object.entries(result.specialists || {});
    const evaluation = task.evaluation;
    const scoreProvenance =
      task.scoreProvenance || (evaluation ? "Recorded evaluator output" : "Selection manifest");
    return `
      <section class="outcome-summary ${statusClass(task)}">
        <div class="outcome-heading">
          <div>
            <span class="eyebrow">Result</span>
            <h2>${escapeHtml(statusText(task))}</h2>
          </div>
          <div class="outcome-score">
            <span>${escapeHtml(metricLabel(task))}</span>
            <strong>${formatScore(task.score)}</strong>
          </div>
        </div>
        <div class="outcome-grid">
          <div><span>Evaluation</span><strong>${escapeHtml(
            result.evaluatorStatus || task.evaluationStatus || "not recorded",
          )}</strong></div>
          <div><span>Submission</span><strong>${escapeHtml(
            result.submissionStatus || "not recorded",
          )}</strong></div>
          <div><span>Verification</span><strong>${escapeHtml(
            result.verificationStatus || "not recorded",
          )}</strong></div>
          <div><span>Score source</span><strong>${escapeHtml(scoreProvenance)}</strong></div>
        </div>
        ${
          result.stopReason
            ? `<p class="outcome-note"><strong>Stop reason:</strong> ${escapeHtml(
                result.stopReason,
              )}</p>`
            : ""
        }
        ${
          specialists.length
            ? `<div class="specialist-statuses">${specialists
                .map(
                  ([role, value]) =>
                    `<span class="specialist-status ${roleClass(role)}"><b>${escapeHtml(
                      role,
                    )}</b> ${escapeHtml(value)}</span>`,
                )
                .join("")}</div>`
            : ""
        }
        ${renderEvaluatorRubric(evaluation, task)}
      </section>`;
  }

  function rubricStatusText(status) {
    const labels = {
      satisfied: "Satisfied",
      partial: "Partially satisfied",
      not_satisfied: "Not satisfied",
      credited: "Credit recorded",
      recorded: "Recorded aggregate",
      unknown: "Unknown",
    };
    return labels[status] || String(status || "Unknown");
  }

  function rubricStatusIcon(status) {
    const icons = {
      satisfied: "check_circle",
      partial: "adjust",
      not_satisfied: "cancel",
      credited: "flag",
      recorded: "fact_check",
      unknown: "help",
    };
    return icons[status] || "help";
  }

  function rubricMetric(value) {
    return value !== null && value !== undefined && value !== "" &&
      Number.isFinite(Number(value))
      ? formatScore(Number(value))
      : "—";
  }

  function renderEvaluatorRubric(evaluation, task = {}) {
    const structuredEvaluation =
      evaluation && typeof evaluation === "object" ? evaluation : null;
    const rubric = structuredEvaluation?.rubric;
    if (!rubric?.rows?.length) {
      const hasOverallScore =
        task.score !== null &&
        task.score !== undefined &&
        Number.isFinite(Number(task.score));
      return `
        <section class="rubric-section empty-rubric">
          <div>
            <span class="eyebrow">Evaluator rubric</span>
            <h3>No per-criterion breakdown recorded</h3>
          </div>
          <p>${
            structuredEvaluation || hasOverallScore
              ? "The saved evaluator result contains the overall score but no public rubric rows."
              : "No public per-criterion rubric was retained for this trajectory."
          }</p>
        </section>`;
    }
    const reportedCount =
      (rubric.satisfied || 0) +
      (rubric.partial || 0) +
      (rubric.notSatisfied || 0) +
      (rubric.credited || 0);
    return `
      <section class="rubric-section">
        <header class="rubric-header">
          <div>
            <span class="eyebrow">Evaluator rubric</span>
            <h3>${
              reportedCount
                ? "What was satisfied and what was not"
                : "Criteria known; outcomes not individually recorded"
            }</h3>
          </div>
          <div class="rubric-counts">
            ${
              rubric.satisfied
                ? `<span class="satisfied">${rubric.satisfied} satisfied</span>`
                : ""
            }
            ${
              rubric.partial
                ? `<span class="partial">${rubric.partial} partial</span>`
                : ""
            }
            ${
              rubric.notSatisfied
                ? `<span class="not-satisfied">${rubric.notSatisfied} not satisfied</span>`
                : ""
            }
            ${
              rubric.unknown
                ? `<span class="unknown">${rubric.unknown} not individually reported</span>`
                : ""
            }
          </div>
        </header>
        ${
          rubric.sourceSummary || rubric.aggregation
            ? `<div class="rubric-method">
                 ${
                   rubric.sourceSummary
                     ? `<p><strong>Evaluator:</strong> ${escapeHtml(
                         rubric.sourceSummary,
                       )}</p>`
                     : ""
                 }
                 ${
                   rubric.aggregation
                     ? `<p><strong>Scoring:</strong> ${escapeHtml(
                         rubric.aggregation,
                       )}</p>`
                     : ""
                 }
               </div>`
            : ""
        }
        ${
          rubric.outcomeNote
            ? `<p class="rubric-outcome-note">${escapeHtml(rubric.outcomeNote)}</p>`
            : ""
        }
        <div class="rubric-table">
          ${(rubric.rows || [])
            .map(
              (row) => `
                <article class="rubric-row ${escapeHtml(row.status)}">
                  <div class="rubric-status">
                    ${icon(rubricStatusIcon(row.status))}
                    <span>${escapeHtml(rubricStatusText(row.status))}</span>
                  </div>
                  <div class="rubric-main">
                    <h4>${escapeHtml(row.label || row.id)}</h4>
                    ${
                      row.description
                        ? `<p>${escapeHtml(row.description)}</p>`
                        : ""
                    }
                    ${
                      row.evidence || row.details
                        ? `<details>
                             <summary>Recorded evidence</summary>
                             ${row.evidence ? `<pre>${escapeHtml(row.evidence)}</pre>` : ""}
                             ${row.details ? `<pre>${escapeHtml(row.details)}</pre>` : ""}
                           </details>`
                        : ""
                    }
                  </div>
                  <div class="rubric-metric">
                    <span>${
                      row.kind === "phase"
                        ? "Contribution"
                        : row.kind === "aggregate"
                          ? "Overall"
                          : "Score"
                    }</span>
                    <strong>${rubricMetric(
                      row.kind === "phase" ? row.contribution : row.score,
                    )}</strong>
                  </div>
                  <div class="rubric-metric">
                    <span>Weight</span>
                    <strong>${
                      row.weight !== null &&
                      row.weight !== undefined &&
                      row.weight !== "" &&
                      Number.isFinite(Number(row.weight))
                        ? `${(Number(row.weight) * 100).toFixed(0)}%`
                        : "—"
                    }</strong>
                  </div>
                  <div class="rubric-metric">
                    <span>Weighted</span>
                    <strong>${rubricMetric(row.contribution)}</strong>
                  </div>
                </article>`,
            )
            .join("")}
        </div>
      </section>`;
  }

  function renderRoleUsage(roleUsage) {
    const entries = Object.entries(roleUsage).filter(([, count]) =>
      Number.isFinite(Number(count)),
    );
    if (!entries.length) return "";
    return `
      <section class="role-usage" aria-label="Model calls by role">
        <span class="role-usage-label">Calls by role</span>
        ${entries
          .map(
            ([role, count]) =>
              `<span class="role-usage-item ${roleClass(role)}">
                 ${escapeHtml(role_name(role))} <b>${escapeHtml(count)}</b>
               </span>`,
          )
          .join("")}
      </section>`;
  }

  function role_name(role) {
    const value = String(role || "");
    return value ? value[0].toUpperCase() + value.slice(1) : "System";
  }

  function collectEvidence(events) {
    const byImage = new Map();
    for (const event of events) {
      for (const image of event.images || []) {
        const key = image.sha256 || image.path;
        if (!key) continue;
        const existing = byImage.get(key);
        if (existing) {
          existing.occurrences += 1;
          continue;
        }
        byImage.set(key, {
          ...image,
          eventId: event.id,
          role: event.role,
          step: event.step,
          elapsedLabel: event.elapsedLabel,
          occurrences: 1,
        });
      }
    }
    return [...byImage.values()].map((image, index) => ({
      ...image,
      galleryIndex: index,
    }));
  }

  function renderEvidenceStrip() {
    if (!state.images.length) return "";
    return `
      <section class="evidence-strip">
        <div class="evidence-strip-heading">
          <div>
            <span class="eyebrow">Visual evidence</span>
            <h2>${state.images.length} distinct image${state.images.length === 1 ? "" : "s"}</h2>
          </div>
          <span>Open any image, then use previous/next.</span>
        </div>
        <div class="evidence-filmstrip">
          ${state.images
            .map(
              (image) => `
                <button class="evidence-thumb" type="button"
                        data-image-index="${image.galleryIndex}">
                  <img src="${escapeHtml(image.path)}" alt="${escapeHtml(
                    image.caption || "Trace image",
                  )}" width="${image.width}" height="${image.height}" loading="lazy">
                  <span>${escapeHtml(
                    image.evidenceKind || "observation",
                  )} · ${escapeHtml(image.role || "System")} ${escapeHtml(
                    image.step ?? "",
                  )}</span>
                </button>`,
            )
            .join("")}
        </div>
      </section>`;
  }

  function renderBlock(block) {
    if (block.kind === "model") return renderTurn(block);
    if (block.kind === "specialist") return renderSpecialist(block);
    return renderLandmark(block);
  }

  function roleClass(role) {
    return String(role || "System").replace(/[^A-Za-z0-9_-]/g, "");
  }

  function renderTurn(turn, nested = false) {
    const body = [
      ...(turn.reasoning || []).map(
        (summary) => `
          <section class="reasoning-summary">
            <h3>${icon("psychology")} Reasoning summary</h3>
            <div class="prose">${formatReport(summary.text ?? summary)}</div>
          </section>`,
      ),
      turn.text
        ? `<section class="response-text"><h3>${icon("chat")} Model response</h3>
             <div class="prose">${formatReport(turn.text)}</div></section>`
        : "",
      ...(turn.toolCalls || []).map(renderToolCall),
      ...(turn.observations || []).map(renderObservation),
      ...(turn.images || []).map((image) => renderImage(image, turn)),
      ...(turn.attachmentWarnings || []).map(
        (warning) => `
          <div class="attachment-warning">${icon("broken_image")}
            <div><strong>Image unavailable</strong>
              <span>${escapeHtml(warning.sourcePath || "unknown source")}</span>
              <span>${escapeHtml(warning.error || "decode failed")}</span>
            </div>
          </div>`,
      ),
    ].filter(Boolean);
    return `
      <section id="${escapeHtml(turn.id)}"
               class="review-section turn ${nested ? "nested-turn" : ""} ${
                 turn.error ? "has-error" : ""
               }"
               data-role="${escapeHtml(turn.role)}" data-outline>
        <header class="turn-header">
          <div class="turn-identity">
            <span class="role-mark ${roleClass(turn.role)}">${escapeHtml(turn.role)}</span>
            <span class="turn-kind">Model turn</span>
            <strong>Call ${escapeHtml(turn.step ?? "—")}</strong>
            ${turn.stage ? `<span>${escapeHtml(turn.stage)}</span>` : ""}
          </div>
          ${
            nested
              ? ""
              : `<a class="turn-link" href="?task=${encodeURIComponent(
                  state.task.id,
                )}#${escapeHtml(turn.id)}" title="Link to this turn">#</a>`
          }
          <time>${escapeHtml(turn.elapsedLabel || "")}</time>
        </header>
        <div class="turn-body">
          ${body.join("") || '<p class="muted">No visible content recorded for this turn.</p>'}
        </div>
      </section>`;
  }

  function controlCommand(command) {
    const match = String(command || "")
      .trim()
      .match(
        /^(?:python3?|\/usr\/bin\/python3?)\s+\/opt\/cuawright-tools\/(binder|critic|verifier|image_show|ask_user|submit)\.py(?:\s|$)/,
      );
    if (!match) return null;
    const labels = {
      binder: "Binder consultation",
      critic: "Critic consultation",
      verifier: "Verifier consultation",
      image_show: "Visual inspection",
      ask_user: "Ask user",
      submit: "Submit task",
    };
    return { operation: match[1], label: labels[match[1]] };
  }

  function renderToolCall(call) {
    const command = call.arguments?.command;
    const control = controlCommand(command);
    const label = control?.label || (call.name === "run_command" ? "Shell command" : call.name);
    const content = command || JSON.stringify(call.arguments || {}, null, 2);
    if (control && ["binder", "critic", "verifier"].includes(control.operation)) {
      return `
        <div class="specialist-invocation ${roleClass(control.operation)}">${icon("account_tree")}
          <strong>Invoked ${escapeHtml(control.operation)}</strong>
          <span>Request and report follow in the next consultation section.</span>
        </div>`;
    }
    return `
      <section class="action-block ${control ? "control-action" : ""}">
        <h3>
          <span class="content-kind">Action</span>
          ${icon(control ? "account_tree" : "terminal")} ${escapeHtml(label)}
          ${
            control
              ? `<span class="control-chip">${escapeHtml(control.operation)}</span>`
              : ""
          }
        </h3>
        <div class="code-panel">
          <div class="code-toolbar">
            <span>${escapeHtml(
              call.arguments?.timeout != null
                ? `timeout ${call.arguments.timeout}s`
                : "action",
            )}</span>
            <button class="copy-button" type="button"
                    data-copy="${escapeHtml(content)}">${icon(
                      "content_copy",
                    )}<span>Copy</span></button>
          </div>
          <pre><code class="language-shell">${highlightShell(content)}</code></pre>
        </div>
      </section>`;
  }

  function highlightShell(text) {
    const source = String(text ?? "");
    const pattern =
      /(?:'(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*"|#[^\n]*|\$\{[^}]+\}|\$[A-Za-z_][A-Za-z0-9_]*|--?[A-Za-z0-9][A-Za-z0-9_-]*|\b(?:sudo|python3?|bash|sh|git|curl|jq|find|grep|sed|awk|tar|cp|mv|rm|mkdir|cd|export)\b)/g;
    let output = "";
    let last = 0;
    for (const match of source.matchAll(pattern)) {
      output += escapeHtml(source.slice(last, match.index));
      const token = match[0];
      let kind = "shell-option";
      if (token.startsWith("#")) kind = "shell-comment";
      else if (token.startsWith("'") || token.startsWith('"')) kind = "shell-string";
      else if (token.startsWith("$")) kind = "shell-variable";
      else if (!token.startsWith("-")) kind = "shell-command";
      output += `<span class="${kind}">${escapeHtml(token)}</span>`;
      last = match.index + token.length;
    }
    return output + escapeHtml(source.slice(last));
  }

  function renderObservation(observation) {
    const isError = observation.status === "error";
    const output = observation.output || "";
    const errorOutput = observation.errorOutput || "";
    const open = isError || output.length + errorOutput.length < 800;
    return `
      <details class="fold observation ${isError ? "error" : ""}" ${open ? "open" : ""}>
        <summary>
          ${icon(isError ? "error" : "subdirectory_arrow_right")}
          <span class="content-kind">Observation</span>
          <span>${escapeHtml(
            observation.tool ? `${observation.tool} result` : "Recorded tool result",
          )}</span>
          <span class="visibility-chip">${escapeHtml(
            observation.visibility === "host_recorded"
              ? "host recorded"
              : "visibility unknown",
          )}</span>
          <span class="summary-meta">${Number(
            observation.outputChars || 0,
          ).toLocaleString()} stdout chars · exit ${escapeHtml(
            observation.exitCode ?? "—",
          )}${
            observation.truncated ? " · abbreviated in viewer" : ""
          }</span>
        </summary>
        ${
          output
            ? `<div class="output-label">stdout</div><pre>${escapeHtml(output)}</pre>`
            : ""
        }
        ${
          errorOutput
            ? `<div class="output-label">stderr</div><pre class="stderr">${escapeHtml(
              errorOutput,
            )}</pre>`
            : ""
        }
        ${
          !output && !errorOutput
            ? '<div class="empty-output">Command returned no text.</div>'
            : ""
        }
      </details>`;
  }

  function renderImage(image, turn) {
    const gallery = state.images.find(
      (candidate) => (candidate.sha256 || candidate.path) === (image.sha256 || image.path),
    );
    return `
      <figure class="image-panel">
        <button class="image-open" type="button"
                data-image-index="${gallery?.galleryIndex ?? 0}">
          <img src="${escapeHtml(image.path)}" alt="${escapeHtml(
            image.caption || "Image supplied to the model",
          )}" width="${image.width}" height="${image.height}" loading="lazy">
        </button>
        <figcaption>
          <span>${escapeHtml(image.caption || "Image supplied to the model")}</span>
          <span>${escapeHtml(image.evidenceKind || "observation")} ·
            ${escapeHtml(turn.role || "System")} call ${escapeHtml(turn.step ?? "—")} ·
            ${image.width}×${image.height}</span>
        </figcaption>
      </figure>`;
  }

  function renderSpecialist(block) {
    const internal = block.children.filter((child) => child.kind === "model");
    const completed = block.completed?.raw || {};
    const report = block.report;
    return `
      <section id="${escapeHtml(block.id)}"
               class="review-section specialist ${roleClass(block.role)} ${
                 block.error ? "has-error" : ""
               }" data-role="${escapeHtml(block.role)}" data-outline>
        <header class="specialist-header">
          <div>
            <div class="eyebrow">Specialist consultation</div>
            <h2>${escapeHtml(block.title)}</h2>
            <p>${escapeHtml(block.invocation || "")}</p>
          </div>
          <div class="specialist-stats">
            ${
              completed.status
                ? `<span class="status-chip ${
                    completed.status === "complete" || completed.status === "ready"
                      ? "pass"
                      : completed.status === "concerns"
                        ? "partial"
                        : "pending"
                  }">${escapeHtml(completed.status)}</span>`
                : '<span class="status-chip failed">Incomplete trace group</span>'
            }
            ${
              completed.steps_used != null
                ? `<span>${completed.steps_used} calls</span>`
                : ""
            }
          </div>
        </header>
        ${
          block.request
            ? `<details class="fold specialist-request">
                 <summary>${icon("outgoing_mail")} Request and purpose</summary>
                 <div class="prose">${formatReport(block.request)}</div>
               </details>`
            : ""
        }
        ${
          report
            ? `<section class="specialist-report">
                 <h3>${icon("assignment")} Returned report</h3>
                 <div class="prose">${formatReport(report.report)}</div>
                 ${
                   (report.unresolved || []).length
                     ? `<h4>Unresolved</h4><ul>${report.unresolved
                         .map((item) => `<li>${escapeHtml(item)}</li>`)
                         .join("")}</ul>`
                     : ""
                 }
               </section>`
            : block.rawReport
              ? `<section class="specialist-report">
                   <h3>${icon("assignment")} Returned report (raw)</h3>
                   <div class="prose">${formatReport(block.rawReport)}</div>
                 </section>`
              : '<p class="muted">No specialist report was recovered.</p>'
        }
        <details class="fold specialist-internals" open>
          <summary>${icon("account_tree")} Inspect ${internal.length} internal model turns</summary>
          <div class="nested-turns">${internal.map((turn) => renderTurn(turn, true)).join("")}</div>
        </details>
      </section>`;
  }

  function landmarkTitle(event) {
    const names = {
      compaction: "Context compacted",
      phase_started: "Phase started",
      phase_evaluated: "Phase evaluated",
      submission_accepted: "Submission accepted",
      task_submitted: "Task submitted",
      terminal_orchestration_completed: "Orchestration completed",
      evaluation: "Evaluation",
      infrastructure_failure: "Infrastructure failure",
      response_retry: "Provider response retried",
      agent_loop_stopped: "Agent loop stopped",
    };
    return names[event.kind] || event.title || event.kind.replaceAll("_", " ");
  }

  function landmarkIcon(event) {
    if (event.kind === "evaluation") return "fact_check";
    if (event.kind.includes("submission")) return "send";
    if (event.kind === "compaction") return "compress";
    if (event.error) return "error";
    return "flag";
  }

  function renderLandmark(event) {
    const secondary = [
      "agent_loop_stopped",
      "text_completion_result",
      "specialist_report_accepted",
      "specialist_completed",
    ].includes(event.kind);
    return `
      <section id="${escapeHtml(event.id)}"
               class="review-section landmark ${event.error ? "has-error" : ""} ${
                 secondary ? "secondary-landmark" : ""
               }"
               data-role="${escapeHtml(event.role || "System")}" ${
                 secondary ? "" : "data-outline"
               }>
        <header>
          ${icon(landmarkIcon(event))}
          <div><div class="eyebrow">${escapeHtml(event.role || "System")} · ${escapeHtml(
            event.elapsedLabel || "",
          )}</div>
          <h2>${escapeHtml(landmarkTitle(event))}</h2></div>
          <span class="landmark-kind">${escapeHtml(event.kind.replaceAll("_", " "))}</span>
        </header>
        ${event.text ? `<div class="prose">${formatReport(event.text)}</div>` : ""}
        ${event.report?.report ? `<div class="prose">${formatReport(event.report.report)}</div>` : ""}
        ${event.raw ? `<div class="structured">${formatStructured(event.raw)}</div>` : ""}
      </section>`;
  }

  function formatStructured(value) {
    if (typeof value === "string") return escapeHtml(value);
    return `<pre>${escapeHtml(JSON.stringify(value, null, 2))}</pre>`;
  }

  function formatReport(text) {
    const lines = String(text ?? "").split(/\r?\n/);
    let output = "";
    let listOpen = false;
    let codeOpen = false;
    let code = [];
    const closeList = () => {
      if (listOpen) {
        output += "</ul>";
        listOpen = false;
      }
    };
    for (const rawLine of lines) {
      const line = rawLine.trimEnd();
      if (line.trim().startsWith("```")) {
        if (codeOpen) {
          output += `<pre>${escapeHtml(code.join("\n"))}</pre>`;
          code = [];
          codeOpen = false;
        } else {
          closeList();
          codeOpen = true;
        }
        continue;
      }
      if (codeOpen) {
        code.push(rawLine);
        continue;
      }
      const bullet = line.match(/^\s*[-*]\s+(.*)$/);
      if (bullet) {
        if (!listOpen) {
          output += "<ul>";
          listOpen = true;
        }
        output += `<li>${formatInline(bullet[1])}</li>`;
        continue;
      }
      closeList();
      if (!line.trim()) continue;
      const heading = line.match(/^\s*#{1,4}\s+(.*)$/);
      output += heading
        ? `<h4>${formatInline(heading[1])}</h4>`
        : `<p>${formatInline(line)}</p>`;
    }
    closeList();
    if (codeOpen) output += `<pre>${escapeHtml(code.join("\n"))}</pre>`;
    return output;
  }

  function formatInline(text) {
    return escapeHtml(text)
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');
  }

  function outlineLabel(block) {
    if (block.kind === "specialist") return block.title;
    if (block.kind === "model") {
      const call = `${block.role || "System"} ${block.step ?? "—"}`;
      const command = block.toolCalls?.[0]?.arguments?.command;
      const control = controlCommand(command);
      if (control) return `${call} · ${control.label}`;
      if (command) return `${call} · ${command}`;
      const summary = block.reasoning?.[0]?.text || block.text;
      return `${call} · ${summary || "Model response"}`;
    }
    return landmarkTitle(block);
  }

  function renderOutline() {
    const blocks = state.visibleBlocks;
    const item = (block, label = outlineLabel(block)) =>
      `<a href="#${escapeHtml(block.id)}" data-target="${escapeHtml(block.id)}"
          class="${block.error ? "error" : ""}">
         <span class="outline-label">${escapeHtml(label)}</span>
         <small>${escapeHtml(block.elapsedLabel || "")}</small>
       </a>`;
    const chronology = [
      '<a href="#overview" data-target="overview"><span>Task and outcome</span></a>',
    ];
    let actorRun = [];
    const flushActorRun = () => {
      if (!actorRun.length) return;
      const first = actorRun[0].step ?? "—";
      const last = actorRun.at(-1).step ?? first;
      const label = first === last ? `Actor call ${first}` : `Actor calls ${first}–${last}`;
      chronology.push(
        `<details class="outline-group" ${actorRun.length <= 8 ? "open" : ""}>
           <summary><span>${escapeHtml(label)}</span><small>${actorRun.length}</small></summary>
           <div>${actorRun.map((turn) => item(turn)).join("")}</div>
         </details>`,
      );
      actorRun = [];
    };
    const hiddenLandmarks = new Set([
      "agent_loop_stopped",
      "text_completion_result",
      "specialist_report_accepted",
      "specialist_completed",
    ]);
    for (const block of blocks) {
      if (block.kind === "model" && block.role === "Actor") {
        actorRun.push(block);
        continue;
      }
      flushActorRun();
      if (block.kind === "specialist") {
        const internal = block.children.filter((child) => child.kind === "model");
        chronology.push(
          `<details class="outline-group outline-specialist">
             <summary><span>${escapeHtml(block.title)}</span>
               <small>${internal.length} calls</small></summary>
             <div>
               ${item(block, "Report and status")}
               ${internal.map((turn) => item(turn)).join("")}
             </div>
           </details>`,
        );
      } else if (!hiddenLandmarks.has(block.kind)) {
        chronology.push(item(block));
      }
    }
    flushActorRun();
    $("#outline").innerHTML = chronology.join("");
    state.outlineIds = $$("#outline a").map((link) => link.dataset.target);
    $$("#outline a").forEach((link) => {
      link.onclick = (event) => {
        event.preventDefault();
        scrollToSection(link.dataset.target, true);
        history.replaceState(null, "", `?task=${encodeURIComponent(state.task.id)}#${link.dataset.target}`);
        if (window.matchMedia("(max-width: 680px)").matches) {
          $("#workspace").classList.remove("outline-open");
          $("#mobile-outline-toggle").setAttribute("aria-expanded", "false");
        }
      };
    });
  }

  function bindTranscriptActions() {
    $$(".copy-button").forEach((button) => {
      button.onclick = () => copyText(button.dataset.copy || "", button);
    });
    $$(".image-open, .evidence-thumb").forEach((button) => {
      button.onclick = () => {
        state.imageTrigger = button;
        openImage(Number(button.dataset.imageIndex || 0));
      };
    });
    if (state.expandAll) {
      $$("#transcript details.fold").forEach((details) => {
        details.open = true;
      });
    }
  }

  function openImage(index) {
    if (!state.images.length) return;
    state.imageIndex = Math.max(0, Math.min(state.images.length - 1, index));
    const image = state.images[state.imageIndex];
    $("#dialog-image").src = image.path;
    $("#dialog-image").alt = image.caption || "Trace image";
    $("#dialog-caption").innerHTML = `
      <strong>${escapeHtml(image.caption || "Trace image")}</strong>
      <span>${state.imageIndex + 1} / ${state.images.length} · ${escapeHtml(
        image.evidenceKind || "observation",
      )} · ${escapeHtml(image.role || "System")} call ${escapeHtml(
        image.step ?? "—",
      )} · ${escapeHtml(image.elapsedLabel || "")}</span>`;
    $("#previous-image").disabled = state.imageIndex === 0;
    $("#next-image").disabled = state.imageIndex === state.images.length - 1;
    if (!$("#image-dialog").open) $("#image-dialog").showModal();
  }

  function copyText(text, button) {
    const done = () => {
      const label = button.querySelector("span:last-child");
      if (!label) return;
      const previous = label.textContent;
      label.textContent = "Copied";
      setTimeout(() => {
        label.textContent = previous;
      }, 900);
    };
    if (navigator.clipboard?.writeText) {
      navigator.clipboard.writeText(text).then(done).catch(() => fallbackCopy(text, done));
    } else {
      fallbackCopy(text, done);
    }
  }

  function fallbackCopy(text, done) {
    const area = document.createElement("textarea");
    area.value = text;
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.append(area);
    area.select();
    document.execCommand("copy");
    area.remove();
    done();
  }

  function contextForId(id) {
    if (id === "overview") {
      return {
        id,
        role: benchmarkLabel(state.task),
        title: state.task.title || `Task ${state.task.id}`,
        elapsedLabel: statusText(state.task),
      };
    }
    for (const block of state.blocks) {
      if (block.id === id) return block;
      if (block.kind === "specialist") {
        const child = (block.children || []).find((item) => item.id === id);
        if (child) return child;
      }
    }
    return null;
  }

  function contextMarkup(block) {
    if (!block) return "";
    const role = block.role || "System";
    const title =
      block.id === "overview"
        ? block.title
        : block.kind === "model"
        ? `Call ${block.step ?? "—"} · ${outlineLabel(block).split(" · ").slice(1).join(" · ")}`
        : outlineLabel(block);
    return `
      <span class="role-mark ${roleClass(role)}">${escapeHtml(role)}</span>
      <strong>${escapeHtml(title)}</strong>
      <span>${escapeHtml(block.elapsedLabel || "")}</span>`;
  }

  function keepOutlineLinkVisible(link) {
    const scroller = $(".outline-scroll");
    if (!link || !scroller) return;
    const linkRect = link.getBoundingClientRect();
    const scrollRect = scroller.getBoundingClientRect();
    if (linkRect.top < scrollRect.top) {
      scroller.scrollTop -= scrollRect.top - linkRect.top + 8;
    } else if (linkRect.bottom > scrollRect.bottom) {
      scroller.scrollTop += linkRect.bottom - scrollRect.bottom + 8;
    }
  }

  function updateStepNavigation() {
    const index = state.outlineIds.indexOf(state.activeBlockId);
    const total = state.outlineIds.length;
    $("#step-position").textContent =
      index >= 0 && total ? `${index + 1} / ${total}` : `— / ${total}`;
    $("#previous-step").disabled = index <= 0;
    $("#next-step").disabled = index < 0 || index >= total - 1;
  }

  function setActiveContext(id) {
    if (!id || id === state.activeBlockId) return;
    state.activeBlockId = id;
    const block = contextForId(id);
    const markup = contextMarkup(block);
    $("#current-context").innerHTML = markup;
    $("#current-context").classList.toggle("hidden", !markup);
    $("#outline-current").innerHTML = markup;
    $$("#outline a").forEach((link) => {
      const active = link.dataset.target === id;
      link.classList.toggle("active", active);
      const group = link.closest("details.outline-group");
      if (active) {
        link.setAttribute("aria-current", "location");
        if (group) group.open = true;
        keepOutlineLinkVisible(link);
      } else {
        link.removeAttribute("aria-current");
      }
      group?.classList.toggle("contains-active", Boolean(active));
    });
    updateStepNavigation();
  }

  function scrollToSection(id, smooth = false) {
    const target = document.getElementById(id);
    if (!target) return;
    let parent = target.closest("details");
    while (parent) {
      parent.open = true;
      parent = parent.parentElement?.closest("details");
    }
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        const root = $("#transcript-scroll");
        const delta =
          target.getBoundingClientRect().top -
          root.getBoundingClientRect().top -
          12;
        root.scrollBy({ top: delta, behavior: smooth ? "smooth" : "auto" });
        setActiveContext(id);
      });
    });
  }

  function syncCurrentSection() {
    const root = $("#transcript-scroll");
    const sections = $$("[data-outline]");
    if (!sections.length) return;
    const threshold = root.getBoundingClientRect().top + 18;
    let current = sections[0];
    for (const section of sections) {
      if (section.getBoundingClientRect().top <= threshold) current = section;
      else break;
    }
    setActiveContext(current.id);
  }

  function observeSections() {
    const root = $("#transcript-scroll");
    state.observer?.disconnect();
    if (state.scrollHandler) {
      root.removeEventListener("scroll", state.scrollHandler);
    }
    let ticking = false;
    state.scrollHandler = () => {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(() => {
        syncCurrentSection();
        ticking = false;
      });
    };
    root.addEventListener("scroll", state.scrollHandler, { passive: true });
    state.observer = new IntersectionObserver(
      () => syncCurrentSection(),
      {
        root,
        rootMargin: "-8px 0px -72% 0px",
        threshold: [0, 0.01, 0.5, 1],
      },
    );
    $$("[data-outline]").forEach((section) => state.observer.observe(section));
    const requested = location.hash.slice(1);
    if (requested) {
      setTimeout(() => scrollToSection(requested, false), 80);
    } else {
      setTimeout(syncCurrentSection, 0);
    }
  }

  function moveOutline(direction) {
    const links = $$("#outline a");
    if (!links.length) return;
    const current = links.findIndex((link) => link.classList.contains("active"));
    const start = current < 0 ? (direction > 0 ? -1 : links.length) : current;
    const next = links[Math.max(0, Math.min(links.length - 1, start + direction))];
    next?.click();
  }

  function syncControlState() {
    $("#error-filter").setAttribute("aria-pressed", String(state.errorsOnly));
    $("#error-filter").setAttribute(
      "aria-label",
      state.errorsOnly
        ? "Show all trajectory events"
        : "Show only failed actions and infrastructure issues",
    );
    $("#error-filter").title = state.errorsOnly
      ? "Show all events"
      : "Show failed actions only";
    $("#failure-filter-icon").textContent = state.errorsOnly
      ? "filter_alt_off"
      : "warning";
    $("#expand-all").setAttribute("aria-pressed", String(state.expandAll));
    $("#expand-all").setAttribute(
      "aria-label",
      state.expandAll ? "Collapse all details" : "Expand all details",
    );
    $("#expand-all").title = state.expandAll
      ? "Collapse all details"
      : "Expand all details";
    $("#expand-icon").textContent = state.expandAll
      ? "unfold_less"
      : "unfold_more";
    $("#role-filter").value = state.role;
    $("#prompt-toggle").checked = state.showPrompts;
  }

  function initControls() {
    $("#collection-name").textContent = viewerTitle();
    $("#task-select").onchange = () => {
      if ($("#task-select").value) openTask($("#task-select").value);
    };
    $("#all-tasks-button").onclick = () => {
      location.href = "index.html";
    };
    $("#mobile-outline-toggle").onclick = () => {
      const open = !$("#workspace").classList.contains("outline-open");
      $("#workspace").classList.toggle("outline-open", open);
      $("#mobile-outline-toggle").setAttribute("aria-expanded", String(open));
      if (open) {
        requestAnimationFrame(() => {
          const active = $("#outline a.active");
          keepOutlineLinkVisible(active);
        });
      }
    };
    $("#trace-search").oninput = () => {
      state.query = $("#trace-search").value.trim().toLowerCase();
      renderReview();
    };
    $("#role-filter").oninput = () => {
      state.role = $("#role-filter").value;
      renderReview();
    };
    $("#prompt-toggle").onchange = () => {
      state.showPrompts = $("#prompt-toggle").checked;
      renderReview();
    };
    $("#error-filter").onclick = () => {
      state.errorsOnly = !state.errorsOnly;
      renderReview();
    };
    $("#expand-all").onclick = () => {
      state.expandAll = !state.expandAll;
      $$("#transcript details.fold").forEach((details) => {
        details.open = state.expandAll;
      });
      syncControlState();
    };
    $("#previous-step").onclick = () => moveOutline(-1);
    $("#next-step").onclick = () => moveOutline(1);
    $("#close-image").onclick = () => {
      $("#image-dialog").close();
    };
    $("#previous-image").onclick = () => openImage(state.imageIndex - 1);
    $("#next-image").onclick = () => openImage(state.imageIndex + 1);
    $("#image-dialog").addEventListener("close", () => state.imageTrigger?.focus());
    $("#image-dialog").onclick = (event) => {
      if (event.target === $("#image-dialog")) $("#image-dialog").close();
    };
    document.addEventListener("keydown", (event) => {
      if (event.target.matches("input, select, textarea")) return;
      if ($("#image-dialog").open && event.key === "ArrowLeft") {
        event.preventDefault();
        openImage(state.imageIndex - 1);
      } else if ($("#image-dialog").open && event.key === "ArrowRight") {
        event.preventDefault();
        openImage(state.imageIndex + 1);
      } else if (event.key === "j") {
        event.preventDefault();
        moveOutline(1);
      } else if (event.key === "k") {
        event.preventDefault();
        moveOutline(-1);
      }
    });
  }

  async function main() {
    initControls();
    const taskId = new URLSearchParams(location.search).get("task");
    if (!taskId) {
      renderTaskList();
      return;
    }
    try {
      await loadTask(taskId);
    } catch (error) {
      $("#loading").classList.add("hidden");
      $("#task-score").textContent = "—";
      $("#score-label").textContent = "Load error";
      $("#task-status").className = "status-chip zero";
      $("#task-status").textContent = error.message;
    }
  }

  main();
})();
