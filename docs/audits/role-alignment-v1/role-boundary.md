# Structural role boundary

## Role knowledge

- **Binder:** task instruction, initialized VM, visible sources/state, exact
  task-visible facts, canonical paths/surfaces; cannot know hidden evaluator
  predicates or future Actor completion.
- **Critic:** Binder draft/evidence plus same VM; can challenge exact facts,
  sets, paths, serialization, and native structure; cannot know evaluator or
  Actor trajectory.
- **Adjudicator:** resolves Critic disputes but still cannot resolve
  evaluator-only conflicts; unresolved probes can disappear in the baseline
  protocol.
- **Actor:** instruction, final bindings, live VM, compaction handoff; does not
  receive Critic provenance or hidden evaluator predicates.
- **Verifier:** instruction, final bindings, unresolved probes, live VM;
  validates Binder’s contract, not the complete evaluator.

## Proof semantics

Natural-language `prove` clauses are guidance, not executable proof
obligations. A Binder/Verifier pass can therefore differ from evaluator pass.

Examples:

- task 102 passed broad binding checks but only one strict document passed;
- task 002 failed on folded serialized `DTSTART`;
- task 046 score changed through an XCF layer-name predicate.

## Actor guidance

Sufficient for deterministic tasks where the exact surface and execution route
are known. Insufficient for:

- domain/route failure;
- unresolved canonical-surface identity;
- hidden evaluator predicates.

Actor receives execution pressure, not an action-level execution plan.

## Principles

- Binder/Critic improve contract formation, not evaluator access.
- unresolved facts remain uncertainty;
- proof must reread the exact scored surface;
- native/API/hash/render evidence outranks screenshots and success messages;
- Actor needs canonical surface identity and a feasible route;
- Fresh Verifier can consistently pass a wrong Binder contract.

## Evidence

- `cli_harness/prompts.py`
- `cli_harness/binding_critic.py`
- `cli_harness/submission_verifier.py`
- `docs/CLI_HARNESS_ARCHITECTURE.md`
- `docs/CLI_HARNESS_BASELINE_106_COMPACT_TRACE_AUDIT.md`
- `docs/CLI_HARNESS_HISTORICAL_GAIN_ANALYSIS.md`

