# NISPO Active Benchmark Goal

Improve NISPO on the active benchmark recorded in
`/Users/nicholas/projects/iupac_naming/benchmark/curriculum_state.json`.

Use the `nispo-engine-strategy` skill and follow
`/Users/nicholas/projects/iupac_naming/AGENTS.md`.

## Objective

Maximize OPSIN roundtrip success:

`SMILES -> NISPO name -> OPSIN -> SMILES`

The active set is currently a failure-focused benchmark, so report recovered
success rate on that set. Aim first for `>=10%`, then `>=25%`, `>=50%`,
`>=60%`, `>=75%`, and `>=90%` success while preserving general behavior.

Accuracy comes first. Speed is secondary during discovery; up to about
`25 ms/molecule/worker` is acceptable for substantial gains.

## Useful Paths

- Benchmark root: `/Users/nicholas/projects/iupac_naming`
- NISPO repo/package: `/Users/nicholas/projects/iupac_naming/nispo`
- Active state: `/Users/nicholas/projects/iupac_naming/benchmark/curriculum_state.json`
- Short brief: `/Users/nicholas/projects/iupac_naming/nispo/010_OPTIMIZATION_BRIEF.md`
- Full audit: `/Users/nicholas/projects/iupac_naming/nispo/BLUE_BOOK_REFACTOR_AUDIT.md`
- OPSIN: `/Users/nicholas/projects/iupac_naming/opsin`
- Blue Book refs: `/Users/nicholas/projects/iupac_naming/iupac_blue_book`
- SureChEMBL data: `/Users/nicholas/projects/iupac_naming/benchmark/data/surechembl`

## Operating Rules

- Run benchmark commands from `/Users/nicholas/projects/iupac_naming`.
- Use the exact benchmark command recorded in `curriculum_state.json`.
- Preserve `from nispo import smiles_to_iupac, smiles_to_iupac_batch`.
- `smiles_to_iupac_batch` must return one output per input; return `""` for
  unsupported molecules instead of aborting.
- Do not mutate benchmark tasks or scoring.
- Do not use answer lookups, exact-SMILES special cases, row-specific fixes,
  generated-name lookup tables, external answer files, memorized rows, or
  creation-time failure metadata.
- SureChEMBL names, including `!` variants, are only for broad pattern analysis.
- NISPO may adapt public/reference rules into deterministic resources, but must
  not depend on OPSIN internals at runtime.

## Search Strategy

Keep discovery broad. Use current benchmark evidence, selected regenerated
diagnostics, RDKit graph analysis, local batched OPSIN diagnostics, NISPO
resources, OPSIN behavior, the Blue Book, and broad SureChEMBL patterns to find
the highest-upside mechanism.

Prioritize general mechanisms that can move many rows: parent/core selection,
ring-system classification, fused/spiro/bridged/ring-assembly naming,
characteristic-group seniority, retained parent resources, heteroatom
replacement coverage, locants, unsaturation, substituent naming, stereo mapping,
candidate ranking, and OPSIN-parseable rendering.

## Validation And Pause

For meaningful changes, run focused tests or probes plus the exact active
benchmark. Judge transition counts, success rate, regressions, parse-failure
churn, and speed.

At natural checkpoints report score, speed, run path, tests, commits, dominant
remaining failures, and the next broad mechanism. If progress stalls, stop small
patching and reset strategy against the brief and audit.
