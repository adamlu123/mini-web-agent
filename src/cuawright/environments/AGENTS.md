# CUAWright environment adapters

Keep benchmark-specific setup, validation, prompts, lifecycle, and packaging
inside the corresponding environment package. The shared Agent, model adapters,
tool schema, control broker, and budget semantics remain benchmark-neutral.

WeaveBench currently lives in `weavebench/`. Run it through the pinned external
integration checkout:

```bash
cuawright setup weavebench
cuawright doctor weavebench
cuawright run weavebench --task TASK_ID
```

The public-comparable agent profile is fixed to GPT-5.5 with `high` reasoning.
Do not copy WeaveBench tasks, VM images, judge containers, or upstream runtime
code into this repository. Preserve its official staging and agent-as-judge
flow.
