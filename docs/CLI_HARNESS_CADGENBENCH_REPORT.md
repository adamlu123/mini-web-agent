# CadGenBench Integration and Evaluation Report

**Report date:** 2026-08-22
**Status:** Full 81-task generation and official hosted scoring are complete.
The public methodology status remains unvalidated.

## Executive summary

CUAWright's persistent Actor-orchestrator was adapted from OSWorld V2 to
CadGenBench without changing the core role graph or the model-facing harness
tool names:

```text
run_command
read_image
call_binder
call_critic
call_verifier
submit_task
```

CAD operations are invoked as ordinary shell commands through a persistent,
fixture-local Build123d-MCP service:

```text
Responses API
  -> run_command
  -> cad-mcp CLI
  -> in-container HTTP MCP service
  -> persistent build123d/OCCT worker
```

The final run produced:

| Result | Value |
|---|---:|
| Public tasks | 81 |
| Generation tasks | 49 |
| Editing tasks | 32 |
| Locally valid and watertight candidates | **81/81** |
| Protected valid checkpoints | 208 |
| MCP restarts | 0 |
| Container recreations | 0 |
| Container failures | 0 |
| Gateway response retries | 14 |
| Model responses | 8,343 |
| Mean responses per task | 103.0 |
| Estimated final-run model cost | **$768.14** |

The first official 32-task and 48-task partial submissions were statistically
consistent with the public GPT-5.5 Build123d-MCP comparator. The 48-task result
was modestly higher on the same fixtures, but its confidence interval included
zero:

```text
CUAWright command-MCP, same 48: 0.3851
Public GPT-5.5 MCP, same 48:    0.3691
Paired delta:                   +0.0160
Bootstrap 95% CI:               [-0.0213, +0.0524]
```

The full-board system result was:

```text
CUAWright command MCP: 0.4979
Validated GPT-5.5 MCP:  0.4573
Paired delta:           +0.0406
Wins/losses:            48 / 33
Bootstrap 95% CI:       [+0.0072, +0.0764]
```

The overall paired interval excludes zero. The gain came primarily from editing:

```text
Generation delta: +0.0156, CI [-0.0206, +0.0512]
Editing delta:    +0.0789, CI [+0.0184, +0.1503]
```

The full 81-task archive is:

```text
/home/v-theolee/experiment81.zip
SHA-256: fe1ea371e617c6ec4ac7ce10da70916b8589646d07ec4aa9bbcae4cbd1ed4746
```

All 81 scrubbed STEP files passed the official public validity gate.

> **Official hosted result**
>
> <https://huggingai4engineering-cadgenbench.hf.space/reports/experiment_experiment_20260822-234139.html>
>
> The hosted row reports `0.4979` overall, `0.3874` generation, `0.6671`
> editing, and 100% validity. Its methodology status is `unvalidated`.

## What this result demonstrates

The result demonstrates **operational portability** of the Actor-orchestrator:

- the same persistent Actor controls execution;
- Binder, Critic, and Verifier remain advisory;
- every role uses the same generic shell and image harness interface;
- the architecture operated a substantially different CAD environment;
- it remained stable at width 48;
- it produced one valid candidate for every public fixture.

It does **not** yet isolate a score gain caused by the multi-role architecture.
The public comparator differs in client, input policy, checkpoint handling,
compute, and orchestration. A matched Actor-only ablation is still required for
a causal architecture claim.

<details>
<summary><strong>Benchmark, dataset, and public comparator pins</strong></summary>

## Public pins

| Item | Pin |
|---|---|
| CadGenBench source | `33304cf771fc5639144b1df9611e347251052cf8` |
| Public data revision | `f76f965585817c621d6ea0d150d745adf670e66e` |
| Task count | 81: 49 generation, 32 editing |
| Comparator harness | `pzfreo/cadgenbench-build123d` |
| Comparator commit | `0a8a78da20c726181a6210924096a8c208eafd4a` |
| Build123d-MCP | `0.3.59` |
| build123d | `0.11.0` |
| build123d-drafting-helpers | `0.12.0` |
| bd_warehouse | `0.2.0` |
| augura | `0.1.5` |

Public references:

- [CadGenBench](https://github.com/huggingface/cadgenbench)
- [CadGenBench data](https://huggingface.co/datasets/HuggingAI4Engineering/cadgenbench-data)
- [CadGenBench submissions](https://huggingface.co/datasets/HuggingAI4Engineering/cadgenbench-submissions)
- [Public Build123d-MCP comparator harness](https://github.com/pzfreo/cadgenbench-build123d/tree/0a8a78da20c726181a6210924096a8c208eafd4a)

The validated online GPT-5.5 comparator reports:

```text
Overall:    0.4573
Generation: 0.3718
Editing:    0.5881
Validity:   100%
```

CadGenBench does not standardize model calls, tokens, wall time, parallelism, or
cost. Leaderboard rows therefore compare complete systems rather than a
compute-normalized model capability.

</details>

<details>
<summary><strong>Architecture and model-visible environment</strong></summary>

## Role topology

```text
Persistent main Actor
  -> fresh Binder
  -> fresh Critic
  -> Actor builds/edits CAD
  -> fresh Verifier
  -> Actor repairs direct findings
  -> optional re-verification
  -> submit
```

The core implementation in `src/cuawright/config/prompts.py` is unchanged from the
original branch base. The CAD profile composes the existing main-Actor prompt
with domain instructions and environment routing.

## Command-only MCP transport

The final design deliberately avoids registering every MCP operation as a
Responses API tool. The model calls only `run_command`; the in-container
`cad-mcp` CLI handles MCP protocol details:

```bash
cad-mcp tools
cad-mcp schema measure
cad-mcp call measure --json '{}'
cad-mcp execute /workspace/build.py
cad-mcp call render_view --json '{"direction":"iso"}'
```

Build123d-MCP runs as the task container's service:

```text
transport: streamable HTTP
bind:      127.0.0.1:8765
published Docker port: none
network mode: none
```

MCP image results are written under `/workspace/.cad-mcp/images` and attached
to the model through the ordinary `read_image` harness tool.

## Specialist permissions

Binder, Critic, and Verifier receive `run_command` and `read_image`, matching the
master Actor-orchestrator capability model. Prompt instructions, not technical
tool removal, prohibit specialist mutation of judge-relevant state.

## Model visibility contract

Every run records:

```json
{
  "public_fixture_inputs_mounted": true,
  "public_submission_outputs_mounted": false,
  "public_scores_mounted": false,
  "public_per_fixture_breakdowns_mounted": false,
  "public_trajectory_logs_mounted": false,
  "trusted_public_evaluation_host_side_only": true
}
```

Private ground truth, official scores, prior submissions, public competitor
trajectories, credentials, and upload mechanisms were not visible to the model.

</details>

<details>
<summary><strong>Input and prompt policy</strong></summary>

## Generation

All fixture-declared public drawings are staged and attached. This matters for:

```text
127: input.png + input2.png
134: input.png + input2.png
145: input.png + input2.png
```

The public comparator attached only `input.png`. Using all declared images is
permitted by the public dataset contract but prevents a strict input-identical
comparison on those three fixtures.

## Editing

Editing fixtures receive:

```text
/input/input.step
/input/edit_description.txt
/input/renders/front.png
/input/renders/top.png
/input/renders/right.png
/input/renders/iso.png
```

## Prompt provenance

The detailed generation and editing procedures in
`cuawright.environments.cadgenbench/public_prompts.py` were pinned from the public comparator
harness. They include:

- exact dimension extraction;
- section/profile-first construction;
- feature-count verification;
- validity after geometry changes;
- render/measure/cross-section loops;
- snapshot and rollback discipline;
- editing preservation and `shape_compare`.

CUAWright adds role-specific instructions and requires one initial batched read
of:

```text
workflow_hints
build123d://quickref
build123d://skill/modeling
```

The prompts are generic across tasks; there are no fixture-specific prompt
branches.

</details>

<details>
<summary><strong>Isolation, validity, checkpointing, and recovery</strong></summary>

## Container isolation

Each fixture receives one container and one persistent CAD session:

```text
network_mode=none
read-only root
read-only /input
non-root host-mapped UID/GID
all Linux capabilities dropped
no-new-privileges
12 GiB memory limit
4 CPU cgroup allocation
1024 PID/thread limit
private 2 GiB tmpfs workspace
```

The image used for the final run:

```text
Image ID: sha256:adc36f75b54e757e04ca95cc6cb8833af5633e7d763f011ef9a9e89fa3fd7b54
Size:     3,240,990,346 bytes
```

## Host-protected checkpoints

After every `run_command`, the host checks for a changed canonical
`output.step`/`output.stp`. A changed candidate is independently checked with
the public validity gate and archived only if valid.

The final evaluator restores the latest host-protected valid STEP before
scoring. Later invalid experiments, failed exports, timeouts, or exhausted
budgets cannot overwrite the scorer-facing candidate.

## Recovery

Recovery handles:

```text
MCP service failure
container exit
command transport failure
command timeout requiring container replacement
```

On recovery:

```text
recreate fixture container around persistent workspace
restore latest host checkpoint
restart local HTTP MCP service
tell Actor to re-import output.step
continue under the same model budget
```

The final 81-task waves required no MCP restart or container recreation.

</details>

## Iterative experiment history

| Stage | Coverage | Result | Interpretation |
|---|---:|---:|---|
| Initial staged adapter | 8 tasks | 5/8 candidates under early limits | Rejected: wrong staged architecture |
| Native MCP panel | 8 tasks | All 8 eventually valid; estimated mean about `0.421` | Non-official estimate only |
| Direct-MCP wave | 32 generation tasks | Candidate mean `0.4049` | First official private-GT evidence |
| Command-MCP single smoke | Fixture 116 | Valid, 2 checkpoints | Proved command transport end-to-end |
| Command-MCP width-48 wave | 48 generation tasks | Candidate mean `0.3851` | 48/48 valid; official partial report |
| Remaining wave | 1 generation + 32 editing | 33/33 valid | Completed full 81 coverage |
| Full 81 upload | 81 tasks | **0.4979** | 100% valid; `+0.0406` over validated GPT-5.5 |

<details>
<summary><strong>Commit and design chronology</strong></summary>

## Branch commits

| Commit | Purpose |
|---|---|
| `3e92a77e` | Initial CadGenBench adapter |
| `367c602e` | Native MCP Actor-orchestrator profile |
| `91e939e0` | MCP transport recovery |
| `4ab6655b` | Whole-container recovery |
| `0dee3406` | Container crash/resource observability |
| `ace7e57d` | 12 GiB task memory bound |

## Important uncommitted final-tree changes

The final command-MCP architecture and full-81 results were generated from a
dirty worktree after `ace7e57d`. Relevant changes include:

- replacement of direct Responses MCP tools with `cad-mcp`;
- in-container HTTP MCP service;
- all declared generation images;
- full command capability for specialists;
- command-triggered host checkpointing;
- workflow-resource guidance;
- updated tests and documentation.

At report time:

```text
HEAD:   ace7e57dd90cfac594e6780b5c22dcbb0a2c8ed3
Base:   8b1709736cc63488fdd651272e2625f51ea24960
Master: a7d449e4c2f90b4d9ec41fba27cc89053372012f
```

Current master is one documentation/reproduction-pin commit ahead of the branch;
its harness code is unchanged from the branch base. The final source must be
committed and tagged before claiming exact reproducibility.

</details>

<details>
<summary><strong>Eight-task pilot cost and early reliability history</strong></summary>

The selected eight valid outputs used:

```text
Calls:               670
Input tokens:        27,221,896
Cached input:        24,471,424
Output tokens:        1,234,846
Estimated cost:          $63.03
Mean cost/task:           $7.88
Mean task runtime:       64.0 minutes
```

Including discarded infrastructure-affected attempts:

```text
Calls:             1,201
Estimated cost:  $116.00
Attempts:             16
```

The original 81-task cost projection from that panel was approximately
`$638.21`. The clean final 81-task waves ultimately consumed more output and
reasoning tokens, with an estimated cost of `$768.14`.

</details>

## Official score progression

### First official 32-task partial submission

The hosted row included 49 missing zeros, so its leaderboard aggregate was not
the meaningful comparison. Recomputed on the 32 submitted fixtures:

```text
CUAWright direct MCP: 0.404923
Public GPT-5.5 MCP:    0.394341
Paired delta:          +0.010582
Wins/losses:           16 / 16
Bootstrap 95% CI:      [-0.027827, +0.048207]
```

Component means:

| Component | CUAWright | Public MCP | Delta |
|---|---:|---:|---:|
| Shape | 0.2869 | 0.2828 | +0.0041 |
| Surface F1 | 0.2465 | 0.2428 | +0.0037 |
| Volume IoU | 0.3273 | 0.3228 | +0.0046 |
| Interface | 0.3957 | 0.4022 | -0.0066 |
| Topology | 0.6595 | 0.6017 | +0.0578 |

The confidence interval included zero. The gain was not statistically
established.

<details>
<summary><strong>Largest 32-task paired movements</strong></summary>

Largest wins over public GPT-5.5 included:

```text
121  +0.2325
113  +0.1782
125  +0.1553
124  +0.1552
111  +0.1520
114  +0.1455
```

Largest losses included:

```text
118  -0.2621
112  -0.1454
105  -0.1447
117  -0.1222
116  -0.1132
```

The broad win/loss spread suggested architecture-dependent task variance rather
than a uniform model ceiling.

</details>

### Official 48-task command-MCP partial submission

Hosted report:

<https://huggingai4engineering-cadgenbench.hf.space/reports/experiment_experiment_20260822-194005.html>

Official row:

```text
81-task aggregate:                0.2282
Generation score incl. missing:  0.3772
Validity over 81:                59.26% (48 present)
Missing:                         33
Methodology validation:          unvalidated
```

The correct mean over the 48 submitted candidates was:

```text
CUAWright command MCP: 0.385104
Public GPT-5.5 same 48: 0.369111
Paired delta:            +0.015993
Wins/losses:             26 / 22
Bootstrap 95% CI:        [-0.021308, +0.052369]
```

Component means versus public GPT-5.5:

| Component | CUAWright | Public MCP | Delta |
|---|---:|---:|---:|
| Shape | 0.2795 | 0.2746 | +0.0049 |
| Surface F1 | 0.2372 | 0.2396 | -0.0024 |
| Volume IoU | 0.3217 | 0.3096 | +0.0121 |
| Interface | 0.3909 | 0.3605 | +0.0305 |
| Topology | 0.5848 | 0.5755 | +0.0093 |

Again, the confidence interval included zero.

### Command transport versus prior direct MCP

On the original 32 fixtures:

```text
Command MCP: 0.406050
Direct MCP:  0.404923
Delta:       +0.001127
Wins/losses: 17 / 15
95% CI:      [-0.036017, +0.039563]
```

This supports transport parity, not an uplift claim. The component mix changed:

```text
Shape:     +0.0013
Interface: +0.0468
Topology:  -0.0905
```

<details>
<summary><strong>Multi-image fixture results</strong></summary>

The three multi-image fixtures all exceeded the public comparator:

| Fixture | Command MCP | Public GPT-5.5 | Delta |
|---|---:|---:|---:|
| 127 | 0.4585 | 0.3378 | +0.1207 |
| 134 | 0.3922 | 0.1882 | +0.2040 |
| 145 | 0.3490 | 0.2193 | +0.1297 |

This is not a causal image-policy A/B. Fixture 127 was effectively unchanged
from the earlier CUAWright run, while 134 and 145 had no prior CUAWright
official score.

</details>

### Full 81-task official result

```text
Overall:                0.4979
Generation:             0.3874
Editing:                0.6671
Validity:               100%
Methodology validation: unvalidated
Archive SHA-256:
fe1ea371e617c6ec4ac7ce10da70916b8589646d07ec4aa9bbcae4cbd1ed4746
```

Hosted report:

<https://huggingai4engineering-cadgenbench.hf.space/reports/experiment_experiment_20260822-234139.html>

Paired against the validated online GPT-5.5 Build123d-MCP submission:

| Scope | CUAWright | GPT-5.5 baseline | Delta | Wins-losses | Bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| All 81 | 0.4979 | 0.4573 | **+0.0406** | 48-33 | **[+0.0072, +0.0764]** |
| Generation 49 | 0.3874 | 0.3718 | +0.0156 | 26-23 | [-0.0206, +0.0512] |
| Editing 32 | 0.6671 | 0.5881 | **+0.0789** | 22-10 | **[+0.0184, +0.1503]** |

The full-board and editing intervals exclude zero; the generation interval does
not. This establishes a system-level paired improvement over that public
baseline, but does not isolate which architectural or adapter difference caused
it.

Effective scored components:

| Scope/component | CUAWright | GPT-5.5 baseline | Delta |
|---|---:|---:|---:|
| Overall effective shape component | 0.3615 | 0.3122 | +0.0493 |
| Overall interface | 0.6111 | 0.5858 | +0.0253 |
| Overall topology | 0.7377 | 0.7223 | +0.0154 |
| Generation shape | 0.2867 | 0.2827 | +0.0041 |
| Generation interface | 0.3955 | 0.3650 | +0.0305 |
| Generation topology | 0.5729 | 0.5638 | +0.0091 |
| Editing normalized shape | 0.4761 | 0.3574 | +0.1186 |
| Editing interface | 0.9414 | 0.9240 | +0.0174 |
| Editing topology | 0.9901 | 0.9650 | +0.0251 |

Generation uses raw shape similarity with `0.4 / 0.4 / 0.2`
shape/interface/topology weights. Editing uses baseline-relative normalized
shape with `0.5 / 0.3 / 0.2` weights. Raw editing preservation metrics also
improved:

```text
Surface F1: +0.0197
Volume IoU: +0.0212
```

<details>
<summary><strong>Largest full-board paired movements</strong></summary>

Largest wins:

| Fixture | Type | CUAWright | Baseline | Delta |
|---|---|---:|---:|---:|
| 203 | Editing | 0.9702 | 0.1592 | +0.8110 |
| 247 | Editing | 0.7716 | 0.3929 | +0.3787 |
| 238 | Editing | 0.7772 | 0.4000 | +0.3772 |
| 249 | Editing | 0.8853 | 0.5510 | +0.3343 |
| 101 | Generation | 0.7729 | 0.4444 | +0.3285 |
| 214 | Editing | 0.8620 | 0.5669 | +0.2951 |
| 245 | Editing | 0.5331 | 0.2429 | +0.2902 |
| 121 | Generation | 0.6406 | 0.3761 | +0.2645 |

Largest losses:

| Fixture | Type | CUAWright | Baseline | Delta |
|---|---|---:|---:|---:|
| 118 | Generation | 0.1844 | 0.5483 | -0.3639 |
| 230 | Editing | 0.5090 | 0.8315 | -0.3225 |
| 137 | Generation | 0.3232 | 0.5838 | -0.2606 |
| 116 | Generation | 0.1650 | 0.3627 | -0.1977 |
| 128 | Generation | 0.4686 | 0.6326 | -0.1640 |
| 119 | Generation | 0.1276 | 0.2826 | -0.1550 |
| 106 | Generation | 0.3471 | 0.4987 | -0.1516 |
| 109 | Generation | 0.7662 | 0.9118 | -0.1456 |

The median paired delta was only about `+0.0044`; the positive mean is partly
driven by several large editing recoveries, especially fixture 203. This
heterogeneity is important when interpreting the aggregate.

</details>

<details>
<summary><strong>Full 81 runtime and token accounting</strong></summary>

## Generation wave: 48 tasks at width 48

```text
Completed/valid:      48/48
Wave wall time:       127.2 minutes
Mean task runtime:     70.0 minutes
Median task runtime:   67.9 minutes
Model responses:      5,456
Mean responses/task:  113.7
Protected checkpoints: 155
Response retries:         7
MCP/container failures:   0
```

Usage:

```text
Input tokens:       203,543,985
Cached input:       168,530,304
Output tokens:        8,827,678
Reasoning tokens:     7,630,818
```

## Remaining wave: fixture 150 plus 32 editing tasks

```text
Completed/valid:      33/33
Wave wall time:       115.0 minutes
Mean task runtime:     56.9 minutes
Median task runtime:   53.3 minutes
Model responses:      2,887
Mean responses/task:   87.5
Protected checkpoints:  53
Response retries:         7
MCP/container failures:   0
```

Usage:

```text
Input tokens:       115,512,017
Cached input:       100,685,952
Output tokens:        3,983,357
Reasoning tokens:     3,215,124
```

## Combined clean full-81 run

```text
Model responses:      8,343
Mean responses/task:  103.0
Input tokens:       319,056,002
Cached input:       269,216,256
Uncached input:      49,839,746
Output tokens:       12,811,035
Reasoning tokens:    10,845,942
Protected checkpoints: 208
Response retries:        14
```

Using the same accounting rates as the pilot:

```text
Uncached input: $5.00 / million
Cached input:   $0.50 / million
Output:        $30.00 / million

Estimated clean full-run cost: $768.14
Estimated mean cost/task:        $9.48
```

The first remaining-33 launcher was externally interrupted before producing
candidates. Its early model usage is not included in the clean final-run cost.
The retry was run from scratch.

</details>

<details>
<summary><strong>Submission packaging and anonymization</strong></summary>

## 48-task partial archive

```text
/home/v-theolee/experiment48.zip
SHA-256: 5db233f1e8081d35cfe4d4b1ade9666c239ae7d50b84d317d27193f81cb75b86
```

Contents:

```text
81 fixture directories
48 output.step candidates
33 intentional missing candidates
```

The official report row uses the same SHA-256.

## Full 81-task archive

```text
/home/v-theolee/experiment81.zip
SHA-256: fe1ea371e617c6ec4ac7ce10da70916b8589646d07ec4aa9bbcae4cbd1ed4746
```

Contents:

```text
81 fixture directories
81 output.step candidates
81/81 official public validity passes
```

## Scrubbing policy

- metadata values use only `Experiment`;
- `agent_url` and `notes` are null;
- STEP headers use fixed blank metadata;
- model-authored object labels are normalized to `part`;
- ZIP timestamps are fixed to `2000-01-01`;
- ZIP origin is standardized to MS-DOS;
- archive comments and extra fields are absent;
- no Unix owner/permission metadata;
- payload scan rejects user, CUAWright, model, role, MCP, path, URL, commit,
  prompt, and architecture identifiers.

Artifact anonymization cannot make an upload untrackable. The hosting service
can retain account, IP, request, and upload logs, and the authenticated username
may be public.

</details>

<details>
<summary><strong>Codebase inventory and divergence from master</strong></summary>

## Source shape

Relative to original base `8b170973`, including the final untracked command-MCP
files:

```text
Approximately 28 files
Approximately 7,152 additions
743 deletions
```

Production-only split:

```text
New CAD-specific production code: about 3,517 lines
Shared runner/budget production changes: +850 / -171
```

The shared runner grew from 3,352 to 4,009 lines. Core modules remained
unchanged:

```text
src/cuawright/config/prompts.py
src/cuawright/config/prompts.py
src/cuawright/environments/control.py
src/cuawright/environments/osworld/guest_tools.py
```

## Major files

| File | Responsibility |
|---|---|
| `cuawright.environments.cadgenbench/task.py` | Dataset discovery, manifests, asset provenance, image policy |
| `cuawright.environments.cadgenbench/environment.py` | Container lifecycle, command execution, checkpoints, recovery, evaluation |
| `cuawright.environments.cadgenbench/cad_mcp_cli.py` | Deterministic command-line MCP client |
| `cuawright.environments.cadgenbench/public_prompts.py` | Pinned public generation/editing prompts |
| `cuawright.environments.cadgenbench/prompts.py` | CUAWright CAD role prompts |
| `cuawright.environments.cadgenbench/packaging.py` | Provenance-checked submission packaging |
| `cuawright.environments.cadgenbench/container/Dockerfile` | Pinned CAD runtime |
| `src/cuawright/run/task.py` | Benchmark profile dispatch and shared orchestration integration |
| `tests/test_cadgenbench_harness.py` | Task, transport, checkpoint, recovery, packaging, and runtime tests |

## Infrastructure versus benchmark shaping

Approximate production-code classification:

| Category | Approximate share |
|---|---:|
| Benchmark contract/runtime infrastructure | 62% |
| Reliability, isolation, and observability | 25% |
| CAD-specific prompts and policy | 11% |
| Miscellaneous integration glue | 2% |

By code volume, roughly 85-90% is adapter/infrastructure. The smaller
benchmark-shaped portion has disproportionate behavioral leverage because it
contains prompts, image policy, budgets, and checkpoint semantics.

</details>

<details>
<summary><strong>Adversarial review and claim boundaries</strong></summary>

## Strongest defensible claim

> CUAWright's persistent Actor-orchestrator control plane and command-only
> model interface transferred successfully from OSWorld to a specialized CAD
> environment, producing valid outputs across all 81 public tasks.

## Claims not established

- that Binder/Critic/Verifier caused a score improvement;
- that CUAWright outperforms an Actor-only control;
- that the small paired uplift over public GPT-5.5 is statistically reliable;
- that the result isolates architecture from CAD tooling, prompts, input policy,
  and checkpointing.

## Likely reviewer criticisms

1. **No matched ablation.** The public comparator differs in more than roles.
2. **Substantial domain adaptation.** The CAD backend, prompts, checkpointing,
   and runtime are specialized.
3. **Benchmark-aware prompt.** It explicitly discusses public scoring
   components and CAD failure modes.
4. **Host checkpoint advantage.** Automatic public validity checks protect the
   final artifact.
5. **Iterative benchmark development.** Early official task scores informed
   later changes; the final benchmark is not a clean untouched test set.
6. **Input mismatch.** Three fixtures receive additional declared images not
   used by the comparator.
7. **Compute mismatch.** The leaderboard has no standardized compute budget.
8. **Dirty final source tree.** The final transport and image-policy changes
   must be committed and tagged for reproducibility.

## Countervailing evidence

- no private GT, official scores, or competitor trajectories were model-visible;
- no fixture-specific prompts;
- one attempt per final fixture;
- no best-of-N or score-guided selection;
- the detailed CAD prompt came from the public comparator;
- the final result covers the full public benchmark rather than a selected
  favorable subset;
- the model-facing tool surface and core role topology are unchanged.

## Recommended publication framing

Describe the result as **system-level architectural portability**, not an
architecture-only uplift. For a causal claim, run:

```text
same model
same prompt
same images
same cad-mcp transport
same checkpoint layer
same 150-response budget
Actor-only versus Actor-orchestrator
```

</details>

<details>
<summary><strong>Validation and test evidence</strong></summary>

The final command-MCP implementation passed:

```text
89 targeted unit/reliability tests
Python compile checks
git diff --check
48/48 HTTP MCP service preflight
single-task persistent-state/render/checkpoint/recovery smoke
48/48 generation validity
33/33 remaining-task validity
81/81 scrubbed-archive official validity
```

Width-48 preflight confirmed every isolated container reached the local HTTP MCP
service with all 36 tools and cleaned up successfully.

No final-run task required:

```text
MCP restart
container recreation
crash recovery
behavioral rerun
```

</details>

## Result and artifact index

| Artifact | Location |
|---|---|
| Width-48 result root | `results-cadgenbench-command-mcp-wave48-20260822T122837Z` |
| Remaining-33 result root | `results-cadgenbench-command-mcp-remaining33-retry-20260822T205834Z` |
| Width-48 summary | `results-cadgenbench-command-mcp-wave48-20260822T122837Z/wave-summary.json` |
| Remaining-33 summary | `results-cadgenbench-command-mcp-remaining33-retry-20260822T205834Z/wave-summary.json` |
| 48-task hosted report | `https://huggingai4engineering-cadgenbench.hf.space/reports/experiment_experiment_20260822-194005.html` |
| 48-task paired analysis | Session artifact `cadgenbench-experiment48-analysis.json` |
| 32-task paired analysis | Session artifact `cadgenbench-wave32-submission/official-analysis.json` |
| Full 81 hosted report | `https://huggingai4engineering-cadgenbench.hf.space/reports/experiment_experiment_20260822-234139.html` |
| Full 81 paired analysis | Session artifact `cadgenbench-experiment81-analysis.json` |
| Full 81 ZIP | `/home/v-theolee/experiment81.zip` |

## Remaining follow-up

1. Commit and tag the final command-MCP source tree.
2. Rebase or merge the documentation-only current master promotion commit.
3. Run a matched Actor-only ablation if making a causal architecture claim.
4. Submit methodology for public validation if the row should be treated as a
   validated leaderboard result.
