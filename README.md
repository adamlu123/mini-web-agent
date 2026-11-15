# CUAWright

CUAWright is a terminal-first agent harness for persistent computer-use
benchmarks. The maintained implementation lives under `src/cuawright/` and
exposes one model-visible tool, `run_command`.

Supported benchmarks:

- OSWorld V2
- WeaveBench
- BenchCAD, including full and direct conditions
- CADGenBench
- Odysseys

Benchmark source repositories, datasets, evaluators, VM images, websites, and
container build inputs remain external and are pinned through CUAWright release
metadata.

```bash
# Minimal package, core agent loop/contracts, CLI help/config, and prompt registry.
uv sync --dev

# Full OSWorld/Odysseys/BenchCAD/CADGenBench, inspection, and legacy runtime.
uv sync --extra full --dev

cuawright setup osworld ...
cuawright setup weavebench ...
cuawright setup benchcad ...
cuawright setup cadgenbench ...
cuawright setup odysseys \
  --source /path/to/Odysseys --source-commit FULL_SHA \
  --dataset /path/to/Odysseys/data/odysseys.json \
  --examples /path/to/Odysseys/data/odysseys_cua_final \
  --judge /path/to/Odysseys/scripts/python/run_full_trajectory_per_rubric.py \
  --runtime-source /path/to/pinned/desktop-runtime \
  --vm /path/to/ubuntu.qcow2 --api-key-file /path/to/key

cuawright doctor

cuawright run osworld --task 001 --result-root results/osworld-001
cuawright run weavebench --domain DOC --task TASK_ID --result-root results/weave
cuawright run benchcad full --record RECORD_ID --result-root results/benchcad
cuawright run benchcad direct --record RECORD_ID --result-root results/benchcad-direct
cuawright run cadgenbench --fixture 101 --result-root results/cadgenbench
cuawright run odysseys --task TASK_SHA1 --result-root results/odysseys
```

For a benchmark-focused environment, replace `full` with `osworld`,
`odysseys`, `benchcad`, or `cadgenbench`. The `providers`, `inspection`, and
`cloud-legacy` extras are also available for provider-only integrations,
document/media tooling, and retained legacy infrastructure respectively:

```bash
uv sync --extra osworld --dev
python -m pip install "cuawright[benchcad]"
```

WeaveBench must use its own environment because its pinned host runner requires
`gymnasium>=0.29` and `tqdm>=4.66`, while pinned OSWorld requires
`gymnasium~=0.28.1` and `tqdm~=4.65.0`:

```bash
uv sync --extra weavebench --dev
```

Do not combine `weavebench` with `osworld` or `full`; the lock metadata declares
those combinations incompatible. Benchmark extras include the currently
supported model provider clients they need. `full` is the explicit compatible
union used by shared OSWorld/Odysseys/BenchCAD/CADGenBench deployments and
preserves the previous broad installation. See
[`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md) for the import-based boundary
map.

`setup` records and verifies external benchmark paths. Odysseys requires a
clean, explicitly pinned source commit and snapshots its selected judge/task
inputs into each run. `doctor` is read-only.
`run` never downloads or patches dependencies. BenchCAD preserves its explicit
`full` and `direct` conditions. The default topology uses the
persistent Actor with a fresh evaluator-aligned Binder and an independent
conservative Verifier. Critic is inactive. OSWorld exposes the topology
ablation `--no-subagents`, which runs one linear Actor. WeaveBench, BenchCAD,
and CADGenBench reject that flag:

```bash
cuawright run osworld --task 001 --no-subagents --result-root results/osworld-linear
```

The reproducible OSWorld baseline is explicitly linear, not the global default.
Use the [pinned 105-task, width-48 recipe](docs/OSWORLD_LINEAR_BASELINE.md).
See the [retained GPT-5.5, GPT-5.6 Sol, and GPT-6 Astra run comparison](docs/OSWORLD_MODEL_RUN_COMPARISON.md)
for exact scores, prompts, token accounting, pricing, and harness differences.
Both OSWorld entrypoints honor the configured VM path (or an explicit
`--path-to-vm`); they no longer guess a legacy image from the checkout location.
Batch and regression-panel launchers share public operations helpers, while the
standalone health gate consumes a versioned retained-artifact contract. The
WeaveBench health adapter reports an explicit incomplete reason until its
external result layout retains that canonical terminal manifest.

See [`AGENTS.md`](AGENTS.md), [`docs/CODEBASE_MAP.md`](docs/CODEBASE_MAP.md),
[`docs/ODYSSEYS.md`](docs/ODYSSEYS.md),
and the benchmark-local `AGENTS.md` files under
`src/cuawright/environments/`.
