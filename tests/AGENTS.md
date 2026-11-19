# NISPO Test Policy

Tests must protect behavior, not wording.

Allowed: public API contracts, determinism, batch safety, unsupported-input
handling, OPSIN/RDKit roundtrip equivalence, benchmark telemetry contracts.

Forbidden: exact generated-name strings, synonym/retained-name choice, prefix
order, punctuation style, parent-choice internals, selector/ranking internals,
compaction style, or private helper behavior unless it is a stable public or
benchmark contract.

## Provenance

Preserve recorded origins and historical evidence. Do not add panel molecules,
reference names, individual failures or derived results without further user
authorization. Complete-panel evaluation is limited to frozen aggregate runs.
Use independently motivated graph controls or explicitly authorised sources.
Keep nomenclature-selection checks outside pytest.
