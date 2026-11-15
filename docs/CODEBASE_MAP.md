# CUAWright codebase map

| Path | Purpose |
|---|---|
| `src/cuawright/runtime/` | Benchmark-neutral turns, command results, attachments and narrow runtime protocols |
| `src/cuawright/agents/` | Benchmark-neutral agent loop plus terminal orchestration and compaction composition |
| `src/cuawright/providers/` | Closed provider specification, construction and adapter selection façade |
| `src/cuawright/models/` | Provider request/response implementations and cost accounting |
| `src/cuawright/config/` | Persistent benchmark path store plus complete prompt sources, immutable prompt assets, named bundle registry and hashes |
| `src/cuawright/benchmarks.py` | Closed descriptor registry for the five supported benchmark families and their stable CLI/health handlers |
| `src/cuawright/operations/` | Public launch, scheduling, process, manifest and result helpers shared by CLI, batch and panel entrypoints |
| `src/cuawright/health/` | Versioned factual retained-manifest and trace-event health contract |
| `src/cuawright/environments/desktop/` | Benchmark-neutral desktop guest workspace, secure file bridge, command execution and control-client provisioning |
| `src/cuawright/environments/osworld/` | OSWorld task, evaluator, lifecycle, release compatibility and guest knowledge adapter |
| `src/cuawright/environments/weavebench/` | WeaveBench external DesktopEnv adapter and byte-stable translated prompt bundle |
| `src/cuawright/environments/benchcad/` | Isolated BenchCAD full/direct adapter |
| `src/cuawright/environments/cadgenbench/` | Isolated CADGenBench adapter |
| `src/cuawright/environments/odysseys/` | Clean external-source identity, task, prompt, complete action trajectory and immutable external-judge adapter |
| `src/cuawright/environments/terminal.py` | Neutral compatibility import surface for terminal benchmark composition |
| `src/cuawright/run/` | Task composition plus thin benchmark-specific entrypoints |
| `src/cuawright/releases/` | Pinned benchmark and integration identities |
| `src/cuawright/cli.py` | Fixed `setup`, `doctor` and `run` parsing plus descriptor dispatch |
| `tests/` | Maintained offline unit, integration and golden-request coverage |
| `scripts/` | Diagnostics, packaging and evidence tooling |
| `trace_viewer/` | Static trace-viewer application |

External benchmark sources and runtime artifacts are stored outside the
repository. CUAWright adapters validate their identity before importing or
executing them.
