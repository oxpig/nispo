# NISPO 010 Optimization Brief

Date: 2026-05-27

Use this as the short operating brief for the next benchmark optimization loop.
The full roadmap remains `BLUE_BOOK_REFACTOR_AUDIT.md`; this file is the
compact version for day-to-day work.

## Active Benchmark

Current active set:

`/Users/nicholas/projects/iupac_naming/benchmark/sets/010_failed_largest_fused_1_2`

Run exactly:

```bash
conda run -n rdkit python benchmark/run_benchmark.py benchmark/sets/010_failed_largest_fused_1_2 --workers 8
```

Set contents:

- total molecules: `65355`
- largest fused system = 1 ring: `24464`
- largest fused system = 2 fused rings: `40891`
- baseline score: `0.0`
- baseline failures:
  - `empty_name`: `17651`
  - `mismatch`: `40845`
  - `opsin_parse_failed`: `6859`
- baseline speed: `7.84` ms/molecule/worker

This is a failure-only set. Score improvements should be read as recovered
matches, not as overall population accuracy.

## Objective

Optimize NISPO on the 0-3 largest-fused-ring-system problem by focusing first
on the 1-ring and 2-fused-ring failures collected in set 010.

The strategic goal is not to memorize this set. The goal is to implement
general IUPAC/Blue Book naming rules that should also help broad benchmarks.

## Core Judgment

Do not pure hill climb on benchmark score.

Use the benchmark to choose high-value work, but implement general rules:

1. typed parent candidates,
2. ring-system descriptors,
3. characteristic-group descriptors,
4. seniority vectors,
5. shared locant maps,
6. renderer validity checks.

The most likely path to 98% success on the 0-3 fused-ring population is a
narrow, benchmark-guided Blue Book refactor, not row-specific fixes.

## Highest-Value Error Modes

Prioritize these in order.

1. Two-fused-ring mismatches
   - largest pool in set 010,
   - often NISPO emits a name, but parent choice or rendering is wrong,
   - likely includes benzofused heterocycles, indoles, quinolines, lactams,
     benzodiazines, benzoxazoles, benzothiazoles, and related fused parents.

2. One-ring and unfused multi-ring empty names
   - largest fused component is 1, but molecule may contain several independent
     rings,
   - needs ring-as-substituent and component parent selection,
   - should be handled with a `RingAssemblyDescriptor`, not fused logic.

3. Characteristic-group parent conflicts
   - amide, ester, carboxyl, carbamate, sulfonyl, nitrile, hydroxy, amine,
   - ring parent selection must not override suffix seniority incorrectly,
   - implement through `CharacteristicGroupCandidate` and `SeniorityVector`.

4. OPSIN parse failures
   - these are often malformed generated names,
   - add local renderer validity checks before returning a non-empty name,
   - avoid increasing parse failures while converting `empty_name` rows.

5. Stereo-bearing mismatches
   - do not patch stereo strings late,
   - map stereo descriptors through the selected parent locants.

## Recommended First Work Slice

Start with infrastructure that does not change output:

1. Create `nispo/nomenclature/candidates.py`.
2. Create `nispo/nomenclature/trace.py`.
3. Add optional descriptor/trace fields to `NameCandidate`.
4. Create `nispo/nomenclature/ring_systems.py` as an adapter over current
   topology utilities.
5. Create `nispo/nomenclature/locants.py`.
6. Add enum values for:
   - monocycle,
   - unfused ring assembly,
   - fused,
   - bridged,
   - spiro-only,
   - component-spiro,
   - fused+spiro.
7. Cache ring descriptors in `NamingContext`.
8. Populate descriptors for current candidates without changing rendered names.
9. Add a diagnostic command or helper to print descriptors for selected rows.

Only after this should selection or rendering behavior change.

## Recommended Second Work Slice

Add seniority diagnostics without changing output:

1. Create `nispo/nomenclature/seniority.py`.
2. Add `SeniorityVector`.
3. Populate vectors for existing candidates.
4. Trace cases where current numeric priority disagrees with seniority.
5. Inspect set 010 rows plus broad guard rows before changing selection.

Only then start moving one low-risk rule family to seniority-based selection.

## Behavior-Changing Priorities

After descriptors and traces exist, work in this order:

1. Characteristic-group registry
   - centralize suffix/prefix decisions,
   - handle oxo/one, hydroxy/ol, amino/amine, cyano/nitrile,
     carboxy/carboxylic acid, carbamoyl/amide, sulfonyl/sulfone.

2. Two-fused-ring retained/generated parent cleanup
   - prefer resources for retained fused parents where available,
   - use general descriptor predicates, not exact SMILES,
   - preserve suffix-bearing parent decisions.

3. Ring assembly and ring substituents
   - handle independent ring components,
   - choose senior parent component,
   - render other rings as substituents.

4. Renderer validity
   - reject malformed generated names locally,
   - reduce `opsin_parse_failed` without increasing `empty_name` unnecessarily.

5. Stereo locant mapping
   - map stereochemistry through selected parent locants,
   - keep current working stereo cases as regression tests.

## Validation

For every meaningful behavior change:

1. Run focused tests for the touched rule family.
2. Run the active benchmark command from `benchmark/curriculum_state.json`.
3. Compare transition matrix:
   - `empty_name -> match`
   - `mismatch -> match`
   - `opsin_parse_failed -> match`
   - `match -> mismatch`
   - `match -> empty_name`
   - `match -> opsin_parse_failed`
4. Track speed in ms/molecule/worker.

Because set 010 is failure-only, also guard broad behavior with targeted checks
or full runs from:

- `001`: small high-scoring broad-druglike molecules,
- `004`: large functionalized heteroaromatic molecules,
- `007`: broad cyclic population,
- `009`: bare ring-system population.

## Success Criteria For This Loop

Good progress:

- set 010 recovered matches increase,
- speed remains below `20` ms/molecule/worker,
- no broad guard set has a meaningful match-to-failure regression,
- recovered rows come from general structural families, not isolated rows.

Strong progress:

- large `mismatch -> match` movement in two-fused-ring systems,
- reduced `empty_name` in one-ring/unfused ring assemblies,
- no growth in `opsin_parse_failed`,
- no regression in pure spiro-only or functionalized broad-druglike probes.

## Do Not Do

- Do not use exact-SMILES special cases.
- Do not use benchmark row labels as runtime data.
- Do not use generated names or OPSIN outputs as lookup tables.
- Do not depend on OPSIN internals at runtime.
- Do not change benchmark scoring to improve score.
- Do not run OPSIN once per row or candidate in runtime code.
- Do not optimize set 010 at the expense of 001/004/007/009.

## Key Principle

Every score-improving change should be explainable as a reusable nomenclature
rule. If the change cannot be described in terms of parent selection,
ring-system classification, characteristic-group seniority, locants,
unsaturation, substituent naming, or stereochemistry, it is probably too
benchmark-specific.
