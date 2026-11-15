# 28-Way Offline Audit of the 0.55037 Profile

Twenty-eight GPT-5.6 Luna agents audited 40 representative low, regressed,
failed, and repaired tasks using final bindings, Critic output, compact traces,
Verifier artifacts, narrow command outputs, and evaluator code.

## Primary attribution

| Class | Tasks | IDs |
|---|---:|---|
| Evaluator contract/configuration/proxy dominates | 14 | 009,010,021,023,024,033,035,036,037,044,061,080,089,092 |
| Actor execution dominates | 13 | 018,047,048,049,050,051,053,054,055,056,058,059,090 |
| Mixed Binder and Actor | 5 | 020,042,046,062,102 |
| Domain/reference-limited semantics | 3 | 067,099,106 |
| Infrastructure or model refusal | 5 | 060,071,081,082,086 |

Primary attribution is not exclusivity: many tasks contain secondary defects.
The machine-readable task matrix is `docs/cli-harness-28way-audit.json`.

## Binder/Critic alignment

Recurring reducible failures:

- unresolved exact sets became executable contracts (`046,099`);
- representative rather than exhaustive formatting checks (`102`);
- wrong canonical surface/profile (`033,036,089`);
- missing exact native serialization requirements (`042`);
- incomplete source inventories or contributor sets (`009,023,035,062`).

Recurring non-reducible or evaluator-specific conditions:

- exact punctuation, keys, strings, hashes, seats, coordinates, and hidden files
  (`009,010,021,023,024,037,099`);
- evaluator conditions directly contradicting visible sources or task wording
  (`024,035,080`);
- narrow parser/proxy acceptance of one visually equivalent representation
  (`044,061,092`);
- hidden expert image/voxel references (`061,106`).

No generic prompt can reliably infer those arbitrary constants. They require
task/evaluator correction or explicit task-visible specification.

## Actor execution

Common failures remained:

- no canonical artifact or incomplete application state (`047,048,059`);
- wrong native geometry/serialization despite plausible appearance
  (`042,049,051,058,090`);
- visual/semantic underperformance (`053,054,067,099,106`);
- reference copying or broken project-to-export provenance (`055,056`);
- proceeding without confirmed evidence or final persistence (`018`);
- authentic source acquisition and exact native audio structure (`050`).

Minimum-deliverable and finalization reminders help, but cannot replace
domain-specific capability or exact native readback.

## Verifier and original-Actor repair

- The Verifier reliably found obvious missing files and directly visible defects.
- It frequently passed evaluator-zero state when Binder omitted or contradicted
  evaluator conditions (`010,021,023,024,035,037,044,058,061,080,081,099,102`).
- Sixteen audited tasks entered Actor repair. Most remained failed or unknown;
  useful repairs were local/partial (`050,054,062,106`), while complete missing
  artifacts, games, or heavy semantic tasks were not recovered.
- Repair cannot resolve wrong Binder facts, hidden evaluator constants,
  read-only filesystems, unavailable services, or missing domain capability.

## Infrastructure findings

- `060,071,086`: ext4 root remounted read-only; `060` then attempted unsafe
  privilege escalation and hit model policy refusal.
- `081`: task service failed to start, so evaluator HTTP retrieval returned no
  project despite direct database edits.
- `082`: starting nested Docker repeatedly severed the guest controller channel.
- The setup compatibility shim matches shell `&&` as background `&`, making
  blocking setup commands asynchronous; this amplified task 086 I/O/races.

## Highest-value generic changes

1. Preserve unresolved exact facts as blocking probes rather than guesses.
2. Bind canonical evaluator-facing app/profile/path and verify the same surface.
3. Require exhaustive source/format manifests when the task exposes them.
4. Add native parser/readback checks for geometry, formulas, tracks, and groups.
5. Treat root `ro`, `EROFS`, missing services, and evaluator postconfig transport
   failures as infrastructure and retry before scoring.
6. Fix setup execution: reject nonzero return codes and distinguish `&&` from
   actual background commands.
7. Require evidence provenance for Verifier failures and continuity of coverage
   gaps through recheck.

Do not add task-specific constants, evaluator-derived values, format-specific
harness readers, or broad generic checklists that claim evaluator emulation.
