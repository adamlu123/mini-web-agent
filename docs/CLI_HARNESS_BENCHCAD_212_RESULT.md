# GPT-5.5 xhigh CUAWright on BenchCAD Vision2Code: 212-Record Result

## Executive result

CUAWright with GPT-5.5 xhigh achieved **0.738998 mean voxel IoU** on a
deterministically selected 212-record BenchCAD Vision2Code panel that was frozen
before model execution and contains exactly two records from each of the 106
public CAD families.

The same GPT-5.5 xhigh model under the canonical direct, single-response
condition achieved **0.437804** on the identical records. The paired uplift was
**+0.301194 IoU**, with a 10,000-resample family-cluster bootstrap 95% interval
of **[+0.256152, +0.346501]**. CUAWright strictly beat Direct on 189 records
and lost on 23.

| Condition | Mean IoU | Median | Executable | Statuses |
| --- | ---: | ---: | ---: | --- |
| Canonical Direct, GPT-5.5 xhigh | 0.437804 | 0.443535 | 206/212 (97.17%) | 206 ok, 5 exec-fail, 1 no-code |
| **Full CUAWright, GPT-5.5 xhigh** | **0.738998** | **0.862728** | 206/212 (97.17%) | 206 ok, 6 policy-invalid |
| Public issue #52 submission, same records | 0.727166 | 0.832095 | 195/212 (91.98%) | 195 ok, 17 exec-fail |

The public issue #52 submission is unnamed, open, and unaccepted. Its author
claims 0.7552 on the full 17,900-record release. Its 212-record value above was
independently reproduced with the same pinned scorer used for Direct and
CUAWright.

## Scope of the claim

This is a **system-level, high-test-time-compute result**, not a same-compute
model comparison and not an official full-split leaderboard score.

The defensible claim is:

> On a deterministically selected, pre-execution-frozen, family-balanced
> 212-record BenchCAD Vision2Code panel covering all 106 public families and
> matching the benchmark's difficulty distribution, GPT-5.5 xhigh with
> CUAWright achieved 0.7390 voxel IoU versus 0.4378 under the canonical direct
> condition. The paired family-cluster bootstrap uplift was +0.3012, 95% CI
> [+0.2562, +0.3465].

Do not claim:

- a formal BenchCAD leaderboard rank;
- a compute-matched win over Anthropic or OpenAI tool runs;
- a full 17,900-record score;
- a same-action-space model uplift;
- that the 212-record point estimate has the precision of a five-run
  1,000-record provider evaluation.

## Frozen panel

Panel ID: `benchcad-panel-212-v1`.

Selection was fixed before model execution:

1. sort the 106 family names by
   `SHA256("cuawright-benchcad-panel-212-v1" + "|" + family)`;
2. assign ranks 0-37 to easy+medium, 38-71 to easy+hard, and 72-105 to
   medium+hard;
3. select the minimum
   `SHA256(seed + "|" + stem)` record in each required family/difficulty
   bucket.

This yielded:

| Difficulty | Records | Panel share | Population share |
| --- | ---: | ---: | ---: |
| Easy | 72 | 33.96% | 34.00% |
| Medium | 72 | 33.96% | 33.96% |
| Hard | 68 | 32.08% | 32.04% |

Other population checks:

| Property | Panel | Population |
| --- | ---: | ---: |
| Standard variant | 90.57% | 91.07% |
| No engineering-standard label | 50.94% | 49.62% |
| XY base plane | 55.19% | 53.83% |
| YZ base plane | 27.83% | 24.32% |
| XZ base plane | 16.98% | 21.84% |

The sample is intentionally family-balanced rather than corpus-random. Family
sizes in the full release range from 86 to 251. Reweighting the panel by those
population family sizes changed the estimates only slightly:

| Condition | Family-balanced | Population-family-weighted |
| --- | ---: | ---: |
| Direct | 0.437804 | 0.438078 |
| CUAWright | 0.738998 | 0.738676 |
| Issue #52 | 0.727166 | 0.725542 |

The panel is therefore a strong broad-family preliminary result, but it is only
1.18% of the 17,900-record release and cannot replace a random 1,000-record or
full-release evaluation.

## Difficulty results

| Difficulty | Direct | CUAWright | Issue #52 |
| --- | ---: | ---: | ---: |
| Easy | 0.433772 | **0.782836** | 0.720631 |
| Medium | 0.446235 | 0.748252 | **0.751855** |
| Hard | 0.433148 | 0.682784 | **0.707945** |

CUAWright's gain over Direct persisted at every difficulty level. Performance
fell on hard records, as expected.

## Benchmark and runtime pins

- BenchCAD repository:
  `52087ef4b08811c21c09b8b16d0ae36c70844865`
- BenchCAD HF dataset:
  `5919f578ab09ec283603a082fab07c7639ab56eb`
- Dataset config: `code_gen`
- Dataset rows: 17,900 across 18 verified parquet shards
- Panel manifest SHA-256:
  `7dc67cee15b73318c66384b193aab91712aebcf2825711717ec91ed55358fd9b`
- Ordered panel-ID SHA-256:
  `e8a95f18780ac266cbdfa7dd5015414ca9aa24d8960523df6e30c962e454c403`
- Runtime image:
  `sha256:cb311aa0259c74ede9d41729442faeae4aea96abfd90a3a9ce013040ddc84a5a`
- Base CUAWright commit:
  `b665b8097fe51d744cc7a29e6478414eb4317d0f`
- Model: `gpt-5.5`
- Reasoning effort: `xhigh`
- CUAWright maximum: 300 model/tool steps
- Actor compaction interval: 40
- Generation width: 54
- Offline scoring width: 48

The run used the frozen uncommitted source snapshot archived in each result
root. The Git commit containing this report banks that exact runtime source hash
ledger plus the compact evidence package. It also includes one post-run
correctness fix for provenance-bound handling of a missing `prediction.py`;
that branch was not exercised by any of the 206 scored CUAWright predictions
or 212 scored Direct predictions and does not change the reported arithmetic.

## Environment and scoring

The model-facing surface remained CLI-based:

- `run_command`
- `read_image`
- existing Binder, Critic, Actor, and verifier submission tools

Candidate execution and rendering were normal commands:

```bash
benchcad-run /workspace/prediction.py --step /workspace/candidate.step
benchcad-render /workspace/candidate.step /workspace/candidate.png
```

The agent sandbox had no network, credentials, GT, scorer, dataset cache, prior
predictions, or Docker socket. `/workspace` used a private size-limited tmpfs
volume.

Final scoring used two isolated containers:

1. submitted Python executed without GT or full scorer source;
2. the generated STEP was compared with an immutable GT snapshot without
   submitted code.

The exact prediction, input, GT, manifest, dataset revision, benchmark commit,
and image ID were hash-bound. Scoring used BenchCAD's normalized 64-cubed voxel
IoU. Every parse, execution, policy, or missing-submission failure contributed
zero to the all-record mean.

These isolation properties were validated locally against the executed image.
The exact OCI image is banked as a content-addressed local archive with inspect
and history metadata, but it has not been published. The original Dockerfile
used version tags rather than registry digests for its base images, so an
independent rebuild is not guaranteed to reproduce the executed image byte for
byte; use the banked archive for exact local reproduction.

## Failure accounting

Direct:

- 206 executable results;
- 5 generated programs failed execution;
- 1 response contained no parseable code;
- one separate hinge attempt exhausted ten upstream HTTP 504 responses and
  produced no response. A clean infrastructure retry succeeded and was selected.

CUAWright:

- 206 executable results;
- 6 terminal model-endpoint `invalid_prompt` policy rejections, all scored zero;
- no other terminal failures;
- one `double_simplex_sprocket` attempt hit guest disk exhaustion, retried from
  clean state, and recovered. Usage includes both attempts.

Issue #52:

- 195 executable results;
- 17 generated programs failed execution.

## Inference usage and estimated cost

The estimate uses:

- $5 per million uncached input tokens;
- $0.50 per million cached input tokens;
- $30 per million output tokens;
- reasoning tokens are already part of output and are not double-counted.

These are list-price assumptions, not a gateway billing statement.

The committed evidence bundle includes one telemetry row per Direct and Full
attempt plus per-record HTTP ledgers and a deterministic extraction-method
description. Those ledgers exactly reconcile the aggregate call, token, role,
retry, timing, and estimated-cost values below.

| Condition | Calls | Input | Cached input | Output | Reasoning | Estimated cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct selected runs | 212 | 123,808 | 0 | 3,176,586 | 3,108,358 | $95.92 |
| CUAWright, every attempt | 33,176 | 608,412,660 | 382,417,536 | 29,975,600 | 22,169,913 | $2,220.45 |
| Combined | 33,388 | 608,536,468 | 382,417,536 | 33,152,186 | 25,278,271 | $2,316.37 |

CUAWright averaged 156.5 successful model calls and an estimated $10.47 per
attempted record. Its overall input cache rate was 62.85%.

CUAWright role costs:

| Role | Calls | Input cache rate | Estimated cost |
| --- | ---: | ---: | ---: |
| Actor | 25,164 | 64.7% | $1,777.33 |
| Binder Critic | 2,830 | 46.4% | $200.65 |
| Binder | 2,891 | 46.8% | $155.39 |
| Audit | 2,048 | 46.7% | $75.78 |
| Recheck | 243 | 32.5% | $11.30 |

The cost is driven by test-time compute rather than broken caching. Without
cache discounts, the estimated CUAWright cost would have been approximately
$3,941.

## Timing and reliability

| Condition | Width | Wall time | Task p50 | Task p95 | Task max |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct | 54 | 1:00:35 | 209s | 449s | 2,400s |
| CUAWright | 54 | 9:27:26 | 5,395s | 9,771s | 17,179s |

HTTP status accounting:

Direct selected runs:

- 212 successful HTTP 200 responses;
- 2 HTTP 502 and 28 HTTP 504 retries;
- the unselected hinge attempt had ten additional HTTP 504s and no successful
  response.

CUAWright:

- 33,176 HTTP 200 responses;
- 13 HTTP 429, 107 HTTP 502, 7 HTTP 503, and 2 HTTP 504 retries;
- 6 terminal HTTP 400 model-policy rejections.

The width-54 no-model load test completed 54/54 CAD execution, render, and score
cycles in 79.2 seconds. Peak sampled container memory was 23.08 GiB and minimum
host available RAM was 431.45 GiB.

## Public with-tools comparison

These are contextual point estimates, not matched evaluations.

Anthropic primary system cards:

| Model | No tools | With tools | Reported protocol |
| --- | ---: | ---: | --- |
| Sonnet 5 | 0.266 | 0.373 | random 1,000, five-run mean, max effort |
| Opus 4.8 | 0.277 | 0.521 | random 1,000, five-run mean, max effort |
| Mythos Preview | 0.356 | 0.610 | random 1,000, five-run mean, max effort |
| Mythos 5 | 0.378 | 0.678 | random 1,000, five-run mean, max effort |
| Opus 5 | 0.366 | 0.821 | random 1,000, five-run mean, max effort |

Anthropic discloses a container, image files, standard Python libraries, an
image-cropping tool, iterative rendering/visual verification, adaptive thinking
at max effort, and five-run averages with graphical confidence intervals.
Anthropic does not disclose subset IDs, token/tool/candidate/time limits,
stopping rules, exact tool packages, raw predictions, or traces.

OpenAI first-party GPT-5.6 launch table:

| Model | No tools | Python tool |
| --- | ---: | ---: |
| GPT-5.5 | 0.444 | 0.558 |
| GPT-5.6 Luna | 0.631 | 0.739 |
| GPT-5.6 Terra | 0.623 | 0.782 |
| GPT-5.6 Sol | 0.706 | 0.834 |

OpenAI does not disclose BenchCAD sample size/IDs, selection, repetitions,
effort, prompts/images, tool environment, calls/tokens/candidates/time,
stopping rules, scorer commit, raw predictions, or traces.

CUAWright's 0.7390 is numerically:

- above Mythos 5 with tools at 0.678;
- below Opus 5 with tools at 0.821;
- nearly identical to GPT-5.6 Luna with Python tool at 0.739;
- above the self-reported GPT-5.5 Python-tool value of 0.558.

These comparisons support broad capability-band placement only. BenchCAD itself
states that a standardized public with-tools division remains future work.

## Artifacts

Committed compact result:

- `benchmark_releases/benchcad-vision2code-212-gpt55-xhigh-v1.json`
- `benchmark_releases/benchcad-vision2code-212-gpt55-xhigh-v1/`
- this report

Complete local result package:

```text
/data/iso-osworld-v2/worktrees/benchcad/results-benchcad-panel-212-v1/
```

Authoritative aggregate:

```text
final-report/
├── README.md
├── summary.json
├── records.tsv
├── records.jsonl
├── families.tsv
├── direct-submission.jsonl
├── full-submission.jsonl
├── provenance.json
└── SHA256SUMS
```

Frozen panel:

```text
/data/iso-osworld-v2/worktrees/benchcad/cache/benchcad-panel-212-v1/
```

- manifest:
  `7dc67cee15b73318c66384b193aab91712aebcf2825711717ec91ed55358fd9b`
- ordered IDs:
  `e8a95f18780ac266cbdfa7dd5015414ca9aa24d8960523df6e30c962e454c403`
- complete provenance:
  `panel-provenance.json`

All local final-report checksums pass. Result generation independently verified
score arithmetic, prediction hashes, submissions, bootstrap intervals, and
artifact checksums in two passes.

The exact executed OCI archive is locally banked at:

```text
/data/iso-osworld-v2/worktrees/benchcad/banked-artifacts/
  benchcad-vision2code-212-gpt55-xhigh-v1/
```

Archive SHA-256:
`a8b1d3faad9aaeb58910637320c31b9a5663d6305f62ebe2277b5ae65b4a791f`.

## Primary references

- BenchCAD source at the pinned revision:
  <https://github.com/BenchCAD/BenchCAD-main/tree/52087ef4b08811c21c09b8b16d0ae36c70844865>
- Official BenchCAD dataset:
  <https://huggingface.co/datasets/BenchCAD/BenchCAD>
- Live BenchCAD leaderboard:
  <https://benchcad.com/leaderboard>
- Public issue #52 submission:
  <https://github.com/BenchCAD/BenchCAD-main/issues/52>
- Anthropic Claude Fable 5 / Mythos 5 system card:
  <https://www.anthropic.com/claude-fable-5-mythos-5-system-card>
- Anthropic Claude Sonnet 5 system card:
  <https://www.anthropic.com/claude-sonnet-5-system-card>
- Anthropic Claude Opus 5 system card:
  <https://www.anthropic.com/claude-opus-5-system-card>
- OpenAI GPT-5.6 launch and BenchCAD table:
  <https://openai.com/index/gpt-5-6/>
- OpenAI Code Interpreter documentation:
  <https://developers.openai.com/api/docs/guides/tools-code-interpreter>
