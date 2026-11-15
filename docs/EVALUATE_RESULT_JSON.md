# Evaluation result artifacts

CUAWright writes the complete benchmark result to:

```text
evaluation/result.json
```

When the benchmark returns a numeric score, it also writes:

```text
evaluation/score.txt
```

`evaluation/result.json` records the benchmark, task ID, submission and
verification provenance, score, complete evaluator payload, run directory,
model-cost snapshot, and prompt hashes. A benchmark evaluator may return either
a numeric score or a mapping containing `score` plus benchmark-specific
components. Mapping values must remain JSON-serializable.

Manual examination utilities are diagnostic and do not replace the maintained
CUAWright evaluation path.
