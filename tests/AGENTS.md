# NISPO Test Policy

Tests must protect behavior, not wording.

Allowed: public API contracts, determinism, batch safety, unsupported-input
handling, OPSIN/RDKit roundtrip equivalence, benchmark telemetry contracts.

Forbidden: exact generated-name strings, synonym/retained-name choice, prefix
order, punctuation style, parent-choice internals, selector/ranking internals,
compaction style, or private helper behavior unless it is a stable public or
benchmark contract.
