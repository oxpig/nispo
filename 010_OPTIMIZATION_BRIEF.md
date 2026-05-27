# NISPO Active Failure Optimization Brief

Date: 2026-05-27

This is the compact operating brief for the current benchmark optimization
loop. The filename is kept for existing agent instructions, but the contents
track the active benchmark in `benchmark/curriculum_state.json`.

Use `BLUE_BOOK_REFACTOR_AUDIT.md` as the detailed roadmap and
`NISPO_IMPLEMENTATION_PRINCIPLES.md` as the compact architecture guide.

## Active Benchmark

Read the authoritative active set and command from:

`/Users/nicholas/projects/iupac_naming/benchmark/curriculum_state.json`

Current active set:

`/Users/nicholas/projects/iupac_naming/benchmark/sets/011_latest_failures_from_010`

Current command:

```bash
conda run -n rdkit python benchmark/run_benchmark.py benchmark/sets/011_latest_failures_from_010 --workers 8
```

Source run:

`benchmark/sets/010_failed_largest_fused_1_2/runs/20260527T173003Z`

Source run summary:

- source molecules: `65355`
- source matches: `35112`
- source success rate: `53.725%`
- source failures: `30243`
- source speed: `9.914` ms/molecule/worker

Active set composition:

- molecules: `30243`
- `empty_name`: `6191`
- `mismatch`: `19543`
- `opsin_parse_failed`: `4509`

This is a failure-only set made from the latest current failures. Treat gains
as recovered successes on this set, not as overall NISPO population accuracy.

## Objective

Improve OPSIN roundtrip success:

`SMILES -> NISPO name -> OPSIN -> SMILES`

Implement general deterministic naming-engine improvements. The benchmark is a
search guide, not a source of row-specific answers. Prefer changes that explain
families of failures through reusable IUPAC/Blue Book rules.

## Broad Rule Sources

Use the audit and Blue Book for general implementation ideas:

- `BLUE_BOOK_REFACTOR_AUDIT.md`: implementation fixes and phase roadmap,
- `NISPO_IMPLEMENTATION_PRINCIPLES.md`: descriptor-driven architecture,
- `iupac_blue_book/`: parent selection, ring systems, characteristic groups,
  locants, unsaturation, substituent naming, and stereo rules,
- `opsin/`: parser behavior, resource vocabulary, and batched diagnostics.

OPSIN is a benchmark oracle and diagnostic reference only. Do not add OPSIN as
a runtime dependency.

## Useful Mechanism Families

Keep discovery broad. High-value mechanisms may include:

- typed candidates and ranking traces,
- descriptor-driven parent/core selection,
- ring-system classification,
- fused, spiro, bridged, and ring-assembly naming,
- retained and systematic resource ranking,
- characteristic-group seniority,
- locant maps and unsaturation,
- cyclic and fused substituent naming,
- stereo mapping through selected parent locants,
- OPSIN-parseable rendering and local validity checks.

Prefer mechanisms that can move many rows. Avoid optimizing isolated rows unless
the fix is a clear instance of a general rule.

## Validation

For meaningful changes:

1. Run focused tests or probes for the touched rule family.
2. Run the exact active benchmark command from `curriculum_state.json`.
3. Report transition counts:
   - `empty_name -> match`
   - `mismatch -> match`
   - `opsin_parse_failed -> match`
   - `match -> mismatch`
   - `match -> empty_name`
   - `match -> opsin_parse_failed`
4. Report success rate, recovered matches, regressions, parse-failure churn,
   speed, and run path.

## Rules

- Preserve `from nispo import smiles_to_iupac, smiles_to_iupac_batch`.
- `smiles_to_iupac_batch` must return exactly one output per input.
- Return `""` for unsupported molecules instead of aborting.
- Do not use exact-SMILES special cases.
- Do not use benchmark row labels as runtime data.
- Do not use generated names or OPSIN outputs as lookup tables.
- Do not depend on OPSIN internals at runtime.
- Do not change benchmark scoring to improve score.
- Do not run OPSIN once per row or candidate in runtime code.

## Key Principle

Every score-improving change should be explainable as a reusable nomenclature
rule: parent selection, ring-system classification, characteristic-group
seniority, locants, unsaturation, substituent naming, rendering, resources, or
stereo.
