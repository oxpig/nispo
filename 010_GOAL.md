# NISPO Active Benchmark Goal

Improve NISPO on the active benchmark recorded in
`/Users/nicholas/projects/iupac_naming/benchmark/curriculum_state.json`.

Use the `nispo-engine-strategy` skill and follow
`/Users/nicholas/projects/iupac_naming/AGENTS.md`.

## Objective

Maximize OPSIN roundtrip success:

`SMILES -> NISPO name -> OPSIN -> SMILES`

The active set is failure-focused, so report recovered success rate on that set.
Accuracy is primary. Track speed and avoid unnecessary slowdowns.

## Useful Paths

- Benchmark root: `/Users/nicholas/projects/iupac_naming`
- NISPO repo/package: `/Users/nicholas/projects/iupac_naming/nispo`
- Active state: `/Users/nicholas/projects/iupac_naming/benchmark/curriculum_state.json`
- Short brief: `/Users/nicholas/projects/iupac_naming/nispo/010_OPTIMIZATION_BRIEF.md`
- Principles: `/Users/nicholas/projects/iupac_naming/nispo/NISPO_IMPLEMENTATION_PRINCIPLES.md`
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

## Search Policy

Prioritize broad, reusable, deterministic naming-engine improvements. Use the
Blue Book directory and `BLUE_BOOK_REFACTOR_AUDIT.md` for general rule ideas and
implementation fixes. Prefer changes that can be explained as nomenclature
rules: parent selection, ring-system classification, characteristic-group
seniority, locants, unsaturation, substituent naming, retained/systematic
resource use, rendering, or stereo.

## Validation

For meaningful changes, run focused tests or probes plus the exact active
benchmark. Judge transition counts, success rate, regressions, parse-failure
churn, and speed.

At checkpoints or when blocked, report files changed, validation run, benchmark
run path, score, speed, dominant remaining failures, and next intended
direction.
