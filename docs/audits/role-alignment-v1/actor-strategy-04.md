# Actor canonical-surface failures: tasks 020, 033, 036, 062, 081, 098

## Outcomes

- **020:** visible checkout differed from serialized cart.
- **033:** semantic trajectory existed, but provenance/canonical log visibility
  did not.
- **036:** review replaced/reconstructed the initialized Zotero library.
- **062:** Actor and evaluator used different Zotero databases.
- **081:** SQLite was edited without refreshing the live service/API.
- **098:** confirmation page looked submitted while canonical API state retained
  a blank SEVIS field.

## Successful controls

- 013: same cookie and same `/api/state`.
- generalized 081: service API write followed by service API readback.
- older 062: evaluator-facing snap DB.
- 066: same editable PPTX saved, parsed, and rendered.

## Surface-continuity rule

1. Resolve identity first: cookie, profile, DB path, project ID, service URL.
2. Treat alternate stores/hooks/caches/screenshots as discovery only.
3. Mutate through the canonical service/app/file.
4. Save/commit, then reload/restart the same surface.
5. Read back using the same authenticated/native endpoint.
6. Submit only when identity and persisted readback agree.

The canonical surface is the evaluator-facing read path, not whichever surface
looks correct to Actor.

## Evidence

- selected task traces/verification for 020, 033, 036, 062, 081, 098
- successful controls 013, 066 and historical 062/081 comparisons

