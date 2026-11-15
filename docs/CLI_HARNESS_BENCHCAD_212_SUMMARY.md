# BenchCAD Vision2Code: GPT-5.5 xhigh + CUAWright

## Result

CUAWright was evaluated on a deterministic 212-record Vision2Code panel frozen
before execution. The panel contains exactly two records from each of
BenchCAD's 106 public CAD families and matches the full release's
easy/medium/hard distribution.

| Condition | Mean IoU | Median | Executable |
| --- | ---: | ---: | ---: |
| Canonical Direct, GPT-5.5 xhigh | 0.4378 | 0.4435 | 206/212 |
| **Full CUAWright, GPT-5.5 xhigh** | **0.7390** | **0.8627** | 206/212 |
| Public issue #52 submission, same records | 0.7272 | 0.8321 | 195/212 |

CUAWright improved over Direct by **+0.3012 IoU**, with a paired
family-cluster bootstrap 95% interval of **[+0.2562, +0.3465]**. It won on
189 records and lost on 23.

Difficulty means:

| Difficulty | Direct | CUAWright | Issue #52 |
| --- | ---: | ---: | ---: |
| Easy | 0.4338 | **0.7828** | 0.7206 |
| Medium | 0.4462 | 0.7483 | **0.7519** |
| Hard | 0.4331 | 0.6828 | **0.7079** |

## Cost and runtime

- CUAWright: 33,176 successful model calls, approximately **$2,220** estimated
  list-price cost, 9h 27m wall time at width 54.
- Direct: 212 selected calls, approximately **$96**, 1h 1m wall time.
- CUAWright input cache rate: 62.85%.
- Six CUAWright tasks were rejected by model policy and scored zero. No other
  terminal CUAWright failures occurred.

This is a high-test-time-compute system result, not a compute-matched model
comparison.

## Public comparison

The `0.7390` result is directionally comparable to public Python-tool
Vision2Code evaluations, but it is not a formal leaderboard rank:

- OpenAI GPT-5.5 no tools: 0.444.
- OpenAI GPT-5.5 with Python tool: 0.558.
- Anthropic Mythos 5 with tools: 0.678.
- Anthropic Opus 5 with tools: 0.821.
- OpenAI GPT-5.6 Luna with Python tool: 0.739.
- OpenAI GPT-5.6 Terra with Python tool: 0.782.
- OpenAI GPT-5.6 Sol with Python tool: 0.834.

Provider subsets, record IDs, raw predictions, tool budgets, token budgets,
candidate limits, and stopping rules are not public. Anthropic reports random
1,000-record, five-run averages; this result is one run on a family-balanced
212-record panel. OpenAI's primary GPT-5.6 comparison table does not disclose
the GPT-5.5 effort setting, sample size, or run count; therefore the public
0.444/0.558 values should not be described as specifically `xhigh`.

Our same-model canonical Direct result, 0.4378, is only 0.0062 below the public
GPT-5.5 no-tools point estimate of 0.444, which is a useful sample/scorer sanity
check despite the undisclosed public protocol.

Population-family reweighting changes CUAWright only from 0.7390 to 0.7387,
so equal family weighting is not materially inflating the headline.

## Saved evidence

- Full report:
  [`CLI_HARNESS_BENCHCAD_212_RESULT.md`](CLI_HARNESS_BENCHCAD_212_RESULT.md)
- Compact release:
  `benchmark_releases/benchcad-vision2code-212-gpt55-xhigh-v1.json`
- Record-level evidence:
  `benchmark_releases/benchcad-vision2code-212-gpt55-xhigh-v1/`
- Git tag: `benchcad-vision2code-212-gpt55-xhigh-v1`
- Banking commit: `fa1a6a097346439caca70e93f26ef3dc4a77406d`

The small evidence ZIP contains reports, panel IDs/hashes, all 212 record-level
scores, exact Direct/Full submissions, family results, telemetry, provenance,
and checksums. The exact 628 MiB OCI image is stored separately under `/data`
and is referenced by SHA-256 from the evidence bundle.

Shareable evidence ZIP:

```text
/data/iso-osworld-v2/worktrees/benchcad/banked-artifacts/
  benchcad-vision2code-212-gpt55-xhigh-v1/
  benchcad-vision2code-212-gpt55-xhigh-v1-evidence.zip
```

ZIP SHA-256:
`113e422d649d1640a59e95dcc89bcb89117d34bee8bda328c1078d0c37e87cb5`
(227,753 bytes).

## Would another 100 records help?

Yes, but modestly:

- Under an IID same-design approximation, increasing from 212 to 312 records
  would reduce sampling standard error by roughly
  `sqrt(212/312) = 0.824`, about 18%. That is only a heuristic: the existing
  panel is family-clustered while the proposed validation panel is
  corpus-proportional, so the actual combined uncertainty must be estimated
  with a design-aware stratified/two-sample analysis.
- It would not make the result directly comparable to Anthropic's random
  1,000-record, five-run protocol or to a full 17,900-record leaderboard run.
- It would not resolve unknown provider tool/token budgets.

The highest-value use of another 100 records is an **independent,
corpus-proportional random validation panel** drawn from the remaining 17,688
records, rather than another family-balanced extension. Report it separately
and also provide a post-stratified combined estimate. Run Direct and CUAWright
on the same 100 records; issue #52 can be rescored on them without model cost.

At the observed average, another 100 records would cost approximately:

- CUAWright: about $1,047.
- Direct: about $45.
- Combined: about $1,092.

It would strengthen the representativeness check, but reaching a random 1,000
records remains the meaningful step for provider-scale sample comparability.
