# BenchCAD adapter

BenchCAD source and dataset artifacts are external and commit-pinned.
CUAWright owns the isolated environment, prompts, rendering helpers, direct
condition, full Actor-orchestrator condition, scoring bridge, and packaging.

```bash
cuawright setup benchcad
cuawright doctor benchcad
cuawright run benchcad full --record RECORD_ID
cuawright run benchcad direct --record RECORD_ID
```

Retain both full and direct conditions. Do not copy benchmark datasets or
ground-truth STEP files into the package.
