# GPT-5.5 xhigh CLI Harness: Pinned 106-Task Result

## Result

- Task release: `v2026.06.24`
- Attempted tasks: 106; excluded GitLab tasks: `026,041`
- Evaluated: 103; failed-as-zero: `060,082,086`
- Partial score sum: `58.33904089974321`
- Effective partial score: `0.5503683103749359`
- Evaluated-task mean: `0.5663984553373127`
- Strict full completion: `19/106` (`0.1792452830188679`)
- Official matched GPT-5.5 xhigh batch-tool baseline: `0.4933412387948141`
- Relative partial-score uplift over the matched official baseline: `11.56%`

## Provenance

The observed tally is reproducible from these immutable result roots:

- `results-actor-repair48-20260811-134300`
- `results-actor-repair-rest58-20260811-155139`
- `results-actor-repair-gateway12-retry-20260812-005203`

Initial panels ran at `ff500d9`; the 12 gateway-exhausted retries ran at
`b091c78`. The default code is pinned by tag `cli-harness-v2-55pct-106` at the
current documentation commit, whose harness behavior is `b091c78`. A complete
single-commit 106-task rerun has not been performed.

## Usage and estimated cost

- Successful model calls retained in response logs: `18126`
- Input tokens: `532096464`, of which cached: `350042240`
- Output tokens: `12592839`
- GPT-5.5 list-price estimate: `$1463.08`, or `$13.80` per attempted task

This includes initial attempts, clean-VM retries, and the 12-task gateway rerun;
it excludes evaluator-model and host infrastructure costs.

## Failed tasks

- `060`: model endpoint rejected a request under cyber policy; no identical retry.
- `082`: nested Docker startup repeatedly severed the guest controller channel.
- `086`: guest ext4 root remounted read-only; evaluator JAR upload then failed.

All three are counted as zero in the effective score.

## Research interpretation

No evaluator code/cache, hidden answer, prior score, or official trajectory was
observed in model prompts or complete command traces. The harness intentionally
uses a broader CLI/native-tool action space than the official screenshot +
pyautogui baseline; describe the result as a system-level CLI-harness uplift,
not a same-action-space model uplift.
