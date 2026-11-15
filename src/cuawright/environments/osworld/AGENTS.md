# OSWorld adapter

OSWorld source, gated tasks, assets, websites, and VM images are external,
release-pinned runtime dependencies. CUAWright owns only the adapter,
compatibility boundary, guest tools, prompt overlay, setup, and doctor logic.

```bash
cuawright setup osworld
cuawright doctor osworld
cuawright run osworld --task 001
```

Activate the verified checkout before importing `desktop_env` or `task_loader`.
Do not add root-level vendored OSWorld code or silently patch/download during a
run. Preserve the sole `run_command` tool and exact benchmark evaluator.

## Clean one-task bootstrap

`cuawright setup osworld` installs or verifies the pinned source checkout and
records paths. It does **not** download the VM image or task-asset cache and
does not repair a QCOW2. Acquire those before setup; never repair a shared source
image in place.

Pin the harness commit that is actually being tested and record it in the run
audit. `docs/OSWORLD_LINEAR_BASELINE.md` describes reproduction of the historical
September 5 baseline and therefore intentionally requires the older
`osworld-linear-baseline-20260905` tag. Do not apply that historical tag check
to a current-master smoke test. Conversely, do not call a current-master run a
reproduction of that historical baseline.

Install the locked runtime:

```bash
uv sync --frozen --extra osworld --dev
```

Authenticate to Hugging Face without printing the token. The gated task dataset
requires that the account has accepted its access conditions:

```bash
uvx --from huggingface_hub hf auth login
```

Prepare the required repaired June-release image. The script downloads and
verifies the public archive, extracts the source QCOW2, clones it, repairs only
the clone's ext4 partition, and checks it read-only:

```bash
scripts/tools/prepare_osworld_v2_qcow.sh /absolute/output/directory
```

For one task, download only its immutable setup assets:

```bash
TASK_ID=099
ASSETS=/absolute/cache/osworld_v2_assets
uvx --from huggingface_hub hf download \
  xlangai/osworld_v2_assets_gated --repo-type dataset \
  --include "task_${TASK_ID}/**" --local-dir "$ASSETS"
```

Official June task classes can reference setup and evaluator assets that are
absent from the public `xlangai/osworld_v2_assets` repository. Use the gated
`xlangai/osworld_v2_assets_gated` repository with an account authorized for the
benchmark. Downloading the selected task directory is required for official
setup and evaluation; evaluator-only files remain host-side inputs to the
benchmark and must never be opened, summarized, or exposed to the model.

Install the pinned source and gated task classes:

```bash
SOURCE=/absolute/data/osworld-v2-2026.06.24/source
uv run scripts/tools/download_osworld_v2_tasks.py \
  --benchmark-release osworld-v2-2026.06.24 \
  --target-dir /absolute/task-class-staging

uv run cuawright setup osworld \
  --benchmark-release v2026.06.24 \
  --source "$SOURCE" \
  --seed-repository "$PWD" \
  --tasks-from /absolute/task-class-staging \
  --vm /absolute/output/directory/osworld-v2-ubuntu-x86-repaired.qcow2 \
  --assets "$ASSETS" \
  --api-key-file /absolute/model-key-file

uv run cuawright doctor osworld
```

The June release source commit is part of this repository's Git history, so a
clean checkout should use itself as `--seed-repository`. This avoids depending
on authentication to the historical source URL while still checking out and
verifying the exact required commit. Do not substitute a public repository that
does not contain that commit.

Run one linear GPT-5.5 xhigh/AIServices task:

```bash
uv run cuawright run osworld \
  --task "$TASK_ID" --no-subagents \
  --result-root "/absolute/results/task-$TASK_ID"
```

This generic command uses the default canonical Direct prompt. It does **not**
reproduce the GPT-5.5 E-L1 condition below. E-L1 requires the explicit
`--prompt-profile e-l1-direct` flag.

## September 2026 GPT-5.5 E-L1 run provenance

The full E-L1 comparison was launched from the `master` worktree while
`master` pointed at commit
`03128983ef9dcdd57f8f5df2768b323e17042f55`. The worktree later advanced
through `8bc821a07e9d208f5e944c4e516869fcadf63fc3` and subsequent commits;
that did not change the code already imported by the main processes. Treat
the commit in `batch-manifest.json`, not the worktree's current `HEAD`, as
the authoritative launch pin.

Main-run condition:

- Result root:
  `/data/iso-osworld-v2/OSWorld-V2/results-gpt55-e-l1-direct-full105-w48-03128983-20260915T2305Z-r2`
- Standard 105-task selection, excluding GitLab tasks `026` and `041` and
  nested-Docker task `082`
- Width `48`; OpenAI provider through AIServices; model `gpt-5.5`; reasoning
  effort `xhigh`
- Linear terminal-only Actor, `--no-subagents`, prompt profile
  `e-l1-direct`
- Actor/composed-system prompt SHA-256
  `ce78a5f41660ad00c58dff44aa993a36e2faf247d3e1ebc746131acc3eee9e85`
  and sole tool-schema SHA-256
  `975a500121a85eb5cb201b3483d1b3c3ad85b2bd4f04b102db95483bbf060c0a`
- June benchmark release `v2026.06.24`, repaired comparison QCOW2, gated
  official assets, website suffix `10.217.29.172.nip.io`
- Task-requested proxy behavior enabled; external proxy-config SHA-256
  `46093a4b743da7b4ab4c3bb977feefba4ce8887be7d7d80efebcf5800ff893b6`
- One runner-level retry for objective `InfrastructureError` only; no quality
  retries and no OpenAI policy-feedback continuation

Equivalent launch shape, with credentials kept external:

```bash
uv run python -m cuawright.run.batch \
  --result-root /data/iso-osworld-v2/OSWorld-V2/results-gpt55-e-l1-direct-full105-w48-03128983-20260915T2305Z-r2 \
  --width 48 \
  --path-to-vm /data/iso-osworld-v2/osworld-v2-ubuntu-x86-hf-6e16459a-20260825-repaired.qcow2 \
  --assets-dir /data/iso-osworld-v2/OSWorld-V2/cache/osworld_v2_assets \
  --api-key-file /absolute/model-key-file \
  --proxy-config /absolute/temporary-proxy-config.json \
  --base-url https://maif-gateway.azure-api.net/api \
  --model gpt-5.5 --reasoning-effort xhigh \
  --model-infrastructure aiservices \
  --website-host-suffix 10.217.29.172.nip.io \
  --no-subagents --prompt-profile e-l1-direct
```

Policy-failure recovery is a separate, mixed condition and must not be called
the clean main run. It was launched from `master` commit
`8bc821a07e9d208f5e944c4e516869fcadf63fc3` for tasks `038`, `043`, and
`046`, at width `3`, with the same model, reasoning, prompt, tools, VM, assets,
website, and proxy condition plus `--openai-policy-feedback-retries 10`.
Its result root is
`/data/iso-osworld-v2/OSWorld-V2/results-gpt55-e-l1-policy-feedback3-w3-8bc821a0-20260916T0110Z`.
Recovery scores may fill previously unevaluated tasks only; they must never
replace an existing main-run score.

Final observed outcomes:

- Clean main: 101/105 evaluated, score sum `64.47285312461935`,
  fixed-105 `61.402717261542236%`, evaluated-only `63.834508044177575%`.
  Native evaluator misses were `038`, `043`, `046`, and `075`.
- Mixed recovery: 103/105 evaluated, score sum `65.96285312461936`,
  fixed-105 `62.82176488058987%`, evaluated-only `64.0416049753586%`.
  This adds task `043 = 0.5` from a fresh recovery run and task `075 = 0.99`
  from an explicitly labelled evaluator-transport recovery. Task `043`
  encountered no policy block during its successful attempt, so its score is
  not evidence that policy feedback worked. Tasks `038` and `046` remained
  unevaluated after ten bounded policy-feedback continuations each.

Do not report `62.8218%` as the clean E-L1 result. The clean reproducible
headline is `61.4027%` fixed-105; `62.8218%` is the mixed-recovery composite.

Before paying for the model call, verify the repaired image SHA-256 is
recorded, `qemu-img check` succeeds, a read-only `e2fsck -fn` is clean, the
downloaded source image still has SHA-256
`28b617987f3edf14edd835069cdde6a6e4708be8e4ae4eb957200d5da335186a`,
the source is commit `43997ed9a88b81151bbb496f1fd1f3b9febf312c`, 108 task
classes are present, the selected task assets exist, Docker/KVM is available,
and the result path is new. After launch, verify the VM, first request/response,
trace growth, model pin, and final evaluator result. Do not retry for quality.

An independently repaired ext4 filesystem does not reproduce a stable
whole-QCOW byte hash because repair metadata can vary. Such a run is a
`fresh-repair` smoke condition. To reproduce the historical September
comparison image exactly, obtain the already repaired immutable QCOW2 and
require SHA-256
`b4674bda74e70d485578bd794f816a8d1f846be44d755fdb9c4aa04acb4814de`;
do not generate a new repair and call it that historical image condition.

For the Docker provider, `/dev/kvm` must exist on the host and the Docker daemon
must be usable. The invoking user does not need direct read/write permission on
`/dev/kvm` when the daemon launches the container with that device. Verify with
`docker info` and the documented no-model VM preflight rather than rejecting a
host solely because `test -r /dev/kvm` or `test -w /dev/kvm` fails.
