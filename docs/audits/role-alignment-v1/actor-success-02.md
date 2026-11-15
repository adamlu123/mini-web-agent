# Actor browser/state controls

## Productive behavior

- **013:** inspected the initialized form, cleared state, wrote exact answers,
  and submitted; proof was weaker than the actual evaluator-visible state.
- **017:** constructed a direct route URL with all stops and walking mode, but
  verified URL substrings rather than the evaluator’s active DOM.
- **022:** handled an API 404 by using the application’s exposed state,
  uploaded through the real portal, submitted, reloaded, and reread state.
- **052:** success depended on preserving the initialized TravelHub session; a
  later new-session reproduction scored zero.
- **065:** exact contact identity was score-critical.
- **073:** cross-app identity and final ReviewSphere readback were critical.
- **100:** Actor solved the puzzle, but later same-session CDP verification
  mutated the anti-DevTools state.

## General principles

1. Treat initialized browser/session/profile as task state.
2. Mutate the evaluator-facing persistent surface, not a proxy.
3. Fallback within the initialized application.
4. Read back exactly what the evaluator-facing surface stores.
5. Do not attach CDP to anti-CDP judged sessions.
6. Preserve exact identity fields.
7. Use minimum-deliverable pressure without arbitrary timing gates.

## Evidence

- `reliability64-alignment-audit/task-013.md`
- `reliability64-alignment-audit/task-017.md`
- `reliability64-alignment-audit/task-022.md`
- `/home/v-theolee/osworld-harness-handoff.md`
- `generalized60-causal-audit.md`

