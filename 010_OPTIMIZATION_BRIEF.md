# NISPO 010 Optimization Brief

Date: 2026-05-27

This is the compact operating brief for the next benchmark optimization loop.
Use `BLUE_BOOK_REFACTOR_AUDIT.md` only when deeper design context is needed.

## Active Benchmark

Read the active set and command from:

`/Users/nicholas/projects/iupac_naming/benchmark/curriculum_state.json`

Current active set:

`/Users/nicholas/projects/iupac_naming/benchmark/sets/010_failed_largest_fused_1_2`

Current command:

```bash
conda run -n rdkit python benchmark/run_benchmark.py benchmark/sets/010_failed_largest_fused_1_2 --workers 8
```

Baseline:

- molecules: `65355`
- largest fused system = 1 ring: `24464`
- largest fused system = 2 fused rings: `40891`
- success rate: `0.0%`
- failures: `17651 empty_name`, `40845 mismatch`, `6859 opsin_parse_failed`
- speed: `7.84` ms/molecule/worker

This is a failure-only set. Treat gains as recovered successes on this set,
not as overall NISPO population accuracy.

## Objective

Improve OPSIN roundtrip success:

`SMILES -> NISPO name -> OPSIN -> SMILES`

Implement general deterministic naming-engine improvements. The benchmark is a
search guide, not a source of row-specific answers. Prefer changes that explain
families of failures through reusable IUPAC/Blue Book rules.

## Search Strategy

Keep the search broad. Let current evidence decide the next mechanism.

Useful mechanisms include:

- parent/core selection,
- typed name candidates and ranking traces,
- ring-system descriptors,
- fused, spiro, bridged, and ring-assembly classification,
- retained fused-parent resources,
- replacement heterocycle coverage,
- characteristic-group seniority,
- locants and unsaturation,
- cyclic and fused substituent naming,
- stereo locant mapping,
- OPSIN-parseable rendering and local validity checks.

Prefer mechanisms that can move many rows. Avoid optimizing isolated rows unless
the fix is a clear instance of a general rule.

## Current High-Value Areas

The active set is dominated by:

1. two-fused-ring mismatches, where NISPO emits a name but parent choice,
   locants, heteroatom handling, suffix handling, or rendering is wrong;
2. one-ring or unfused multi-ring empty names, which likely need better parent
   selection and ring-as-substituent/ring-assembly handling;
3. characteristic-group conflicts, where suffix seniority must guide parent
   selection instead of being patched late;
4. OPSIN parse failures, where malformed names should be prevented by renderer
   validity checks;
5. stereo-bearing mismatches, where descriptors must be mapped through the
   selected parent locants.

## Suggested First Moves

Before large behavior changes, improve observability if it is missing:

- candidate descriptors,
- ring-system descriptors,
- seniority vectors,
- shared locant maps,
- ranking traces,
- selected-row diagnostics.

Then change one broad rule family at a time and measure transitions.

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
- Do not use exact-SMILES special cases.
- Do not use benchmark row labels as runtime data.
- Do not use generated names or OPSIN outputs as lookup tables.
- Do not depend on OPSIN internals at runtime.
- Do not change benchmark scoring to improve score.
- Do not run OPSIN once per row or candidate in runtime code.

## Key Principle

Every score-improving change should be explainable as a reusable nomenclature
rule: parent selection, ring-system classification, characteristic-group
seniority, locants, unsaturation, substituent naming, rendering, or stereo.
