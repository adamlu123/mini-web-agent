# CADGenBench adapter

CADGenBench source, fixtures, and official scoring remain external and
commit-pinned. CUAWright owns the isolated MCP bridge, protected checkpoints,
public validation, rendering, prompts, and submission packaging.

```bash
cuawright setup cadgenbench
cuawright doctor cadgenbench
cuawright run cadgenbench --fixture FIXTURE_ID
cuawright package cadgenbench --manifest MANIFEST --run-dir RUN_DIR
```

Do not duplicate Build123d-MCP resources into another KB or expose alternate
orchestration modes.
