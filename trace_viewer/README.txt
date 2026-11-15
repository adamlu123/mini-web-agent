CUAWright CLI Trace Viewer
============================

1. Extract the entire ZIP.
2. Open index.html in a modern browser.
3. No local server or network connection is required.

The viewer includes provider-returned reasoning summaries, normalized tool
calls, host-recorded command results, explicit images supplied to the model,
specialist reports, verification records, and evaluator outcomes. Reasoning
summaries are not full hidden chain-of-thought.

Each trajectory starts with benchmark/model identity, the benchmark-specific
metric, submission and verification state, role usage, and a deduplicated visual
evidence strip. The left outline remains the primary way to jump through model
turns, specialist consultations, submissions, errors, and evaluation landmarks.

It intentionally omits raw API requests/responses, encrypted reasoning,
credentials, cookies, host paths, environment configuration, provider retries,
internal request identifiers, evaluator caches, and command-output data that
was not shown to the model.

Keyboard shortcuts in a task timeline:
  j  next visible event
  k  previous visible event
  left/right arrow  previous/next image while the image viewer is open

The package manifest records source roots, hashes, file limits, and the
omitted-data policy.
