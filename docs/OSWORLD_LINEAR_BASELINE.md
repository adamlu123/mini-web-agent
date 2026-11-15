# Reproducible OSWorld linear baseline

This is the same harness lineage as the August 29–30 native-inline run
(`2ef21b1a`, 60.36%) and September 3 explicit linear control (`79e0d507`,
task 089 replacement `d2be7e73`, 59.04%). These are historical observations,
not guaranteed reproduction scores.

## Fixed contract

- One persistent Actor; explicitly pass `--no-subagents`. No Binder, Critic,
  Verifier, experimental helpers, or task-specific prompt heuristics.
- Sole provider-visible tool: `run_command`; GPT-5.5, `xhigh`, OpenAI Responses
  through AIServices. Shared 300-model-call budget per attempt, compaction 40.
- All 105 tasks, excluding 026, 041, 082; width **48**.
- Existing single fresh-VM retry on `InfrastructureError`; no quality retries
  or best-score selection. Preserve every attempt and cost. Report missing
  outcomes separately from official evaluated scores.
- Linear composed system SHA256:
  `0ff8702a60527ace118bb8cab8a6ef02852763843b51de3fb30ab104000f08fe`.
  Actor loop, model adapters, guest tools, phase topology propagation, and
  submission semantics are unchanged from the September 3 control.

## Reliability recovery

The legacy `osworld-v2-ubuntu-x86.qcow2` was inadvertently retained in persistent
configuration, including the September 3 control. Use the repaired image below,
never repair or mutate a shared source QCOW in place. Both direct OSWorld and
batch entrypoints now use the configured VM path or require `--path-to-vm`;
neither guesses the legacy basename from the checkout location.

Keep the `89ff6659` / `2e3953bb` correction: exact UTF-8 output-decode and
subprocess-timeout HTTP 500 signatures remain command errors with an
uncertain-side-effect warning, not infrastructure aborts. They never cause
automatic command replay. Unknown server/transport failures and confirmed
global-home read-only failures still fail closed. Shared PNG/EXIF normalization
is retained; there is no demonstrated regression from it.

The global product default remains orchestrated for compatibility. This
baseline's explicit linear invocation avoids incidental topology drift without
changing other benchmarks.

## Pin, preflight, and run

Use a clean, detached worktree at the immutable commit resolved by tag
`osworld-linear-baseline-20260905`. Never launch queued work from mutable master.
Install the existing lockfile dependencies with
`uv sync --frozen --extra full --dev` when provisioning a new checkout.
External gated tasks/assets are not in this repo.

Required release: source commit
`43997ed9a88b81151bbb496f1fd1f3b9febf312c`, 108 task classes, supported subset 105.
Required repaired QCOW SHA256:
`b4674bda74e70d485578bd794f816a8d1f846be44d755fdb9c4aa04acb4814de`.

Before paying for the batch, boot one fresh repaired-image VM using the normal
Docker provider and real task setup, without calling a model or evaluator.
Verify home/Desktop/control-store writability, free disk space, kernel error
evidence, ordinary commands/errors/timeouts, both known command-local HTTP 500
cases followed by successful commands, Python control client dispatch and typed
image delivery, required websites and assets. Close only that probe VM. A
passing no-model probe does not guarantee every later task's infrastructure.

Example deployment paths (substitute only location, not content identity):

```bash
export CUAWRIGHT_OSWORLD_ROOT=/home/v-theolee/.local/share/cuawright/osworld/osworld-v2-2026.06.24/source
VM=/data/iso-osworld-v2/osworld-v2-ubuntu-x86-hf-6e16459a-20260825-repaired.qcow2
ASSETS=/data/iso-osworld-v2/OSWorld-V2/cache/osworld_v2_assets
KEY=/data/iso-osworld-v2/.a-creds
PROXY=/data/iso-osworld-v2/OSWorld-V2/evaluation_examples/settings/proxy/dataimpulse.json
PIN=$(git rev-parse HEAD)
test "$PIN" = "$(git rev-parse 'osworld-linear-baseline-20260905^{commit}')"
test -z "$(git status --porcelain --untracked-files=no)"
test "$(git -C "$CUAWRIGHT_OSWORLD_ROOT" rev-parse HEAD)" = 43997ed9a88b81151bbb496f1fd1f3b9febf312c
printf '%s  %s\n' b4674bda74e70d485578bd794f816a8d1f846be44d755fdb9c4aa04acb4814de "$VM" | sha256sum --check
test -d "$ASSETS" && test -s "$KEY" && test -r "$PROXY"
RESULT_ROOT="/data/iso-osworld-v2/OSWorld-V2/results-linear-baseline-105-w48-${PIN:0:8}-$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "$RESULT_ROOT"
python -m cuawright.run.batch \
  --result-root "$RESULT_ROOT" --width 48 --no-subagents \
  --path-to-vm "$VM" --assets-dir "$ASSETS" \
  --api-key-file "$KEY" --proxy-config "$PROXY" \
  --model gpt-5.5 --reasoning-effort xhigh \
  --model-infrastructure aiservices \
  --website-host-suffix 10.217.29.172.nip.io
```

Before launch, keep a read-only run manifest alongside the results freezing full
commit/tag, source/task and asset content inventories, image path/hash, lockfile
and control-client hashes, prompt hash, task list, provider/model, width, budgets,
selection/retry rules, and command. Keep credentials out of the manifest.
Do not overwrite historical manifests or substitute these outcomes into frozen
research controls. Check actual first requests/responses, guest commands, VM
mounts, worker count, trace growth, and the existing five-minute health reports;
a PID alone is not a healthy launch.

Known official task 024/086 grading/runtime incompatibilities and instruction
violations must be documented with results, not fixed by changing evaluators,
injecting hidden/reference answers, silently rescoring, or selecting retries.
