# NISPO Implementation Principles

Date: 2026-05-27

Use this as the compact architecture brief for NISPO improvement work. It is
shorter than `BLUE_BOOK_REFACTOR_AUDIT.md` and is intended to fit into context
during benchmark optimization. If this file conflicts with `AGENTS.md`,
`AGENTS.md` wins.

## Purpose

NISPO converts SMILES to deterministic IUPAC-style names that roundtrip through
OPSIN:

```text
SMILES -> NISPO name -> OPSIN -> SMILES
```

The goal is not strict preferred-IUPAC-name perfection. The goal is broad,
general, deterministic naming behavior that is chemically coherent and OPSIN
parseable. Preferred Blue Book behavior is valuable when it also supports
roundtrip stability, but OPSIN-compatible general nomenclature is acceptable
when it gives reliable structure recovery.

## Reference Hierarchy

Use references in this order:

1. `AGENTS.md`: project contract, benchmark rules, integrity rules.
2. This file: compact implementation principles.
3. `010_OPTIMIZATION_BRIEF.md`: active benchmark context.
4. `BLUE_BOOK_REFACTOR_AUDIT.md`: detailed roadmap and phase plan.
5. `iupac_blue_book/`: rule source for nomenclature concepts.
6. `opsin/`: parse behavior, resource vocabulary, and validation oracle.

OPSIN may guide tests and deterministic resource creation. NISPO must not call
OPSIN or depend on OPSIN internals at runtime.

## Target Pipeline

The best-practice NISPO engine should be a descriptor-driven pipeline:

```text
RDKit molecule
  -> MoleculeContext
  -> graph, feature, ring, locant, group, stereo descriptors
  -> typed parent and substituent candidates
  -> shared seniority and name-choice comparison
  -> renderer for the selected candidate
  -> local syntax/shape checks
  -> external OPSIN roundtrip validation in tests/benchmarks
```

The central design rule is: structure first, string last. Naming decisions
should be made from graph facts and typed descriptors, not from partially
rendered strings.

## Core Design Principles

1. Every naming path proposes candidates.

   Acyclic, monocyclic, fused, bridged, spiro, ring-assembly, fragment, branch,
   and retained-name logic should all emit comparable candidates. Individual
   rule modules should not make final cross-family parent decisions.

2. Parent selection is centralized.

   The Blue Book repeatedly treats naming as hierarchical selection: choose the
   senior parent, then number it, then attach suffixes/prefixes/stereo. NISPO
   should use a shared `SeniorityVector` or equivalent comparator rather than
   hidden priority ladders scattered across modules.

3. Descriptors are the internal language.

   A mature candidate should carry:

   - parent scope,
   - topology class,
   - component/ring-system descriptor,
   - characteristic-group descriptor,
   - substituent attachment map,
   - locant map,
   - unsaturation/hydrogenation state,
   - stereo sites,
   - renderer family,
   - decision trace.

4. Locants are data, not formatting.

   Locant assignment must be represented once and consumed by parent selection,
   suffixes, prefixes, unsaturation, substituents, and stereo. Avoid private
   atom-number maps inside rule families.

5. Characteristic groups need a registry.

   P-33/P-35 suffix and prefix behavior should be represented by one registry
   that knows seniority, suffix form, prefix form, multiplicity, valence
   expectations, and parent compatibility. Ring-vs-chain and ring-vs-ring
   decisions must respect principal characteristic groups.

6. Ring systems must be classified before rendering.

   The engine should distinguish monocycles, unfused ring assemblies, fused
   systems, bridged fused systems, non-fused bridged/von Baeyer systems, free
   spiro systems, non-free spiro systems, and mixed fused/spiro/bridged systems.
   A rule that helps one topology class should not silently rerank another.

7. Retained names and systematic names are separate candidate types.

   Retained resources are legitimate nomenclature data. They should not become
   exact answer lookups. Resource records may carry structural metadata such as
   ring count, heteroatom pattern, component class, and locant coverage.

8. Stereo is parent-aware.

   Stereochemical descriptors are added after the parent and numbering are
   chosen. Stereo code should map RDKit stereo sites through the selected
   locant map rather than attaching fallback strings late.

9. Unsupported chemistry returns empty.

   A wrong but parseable name is worse than no name for engine quality. Return
   `""` for unsupported molecules rather than guessing. Batch APIs must still
   return exactly one output per input.

10. Runtime stays local and deterministic.

    Use RDKit and local NISPO resources at runtime. Use OPSIN only in tests,
    probes, benchmarks, and offline deterministic resource generation.

## Blue Book Rule Map For NISPO

These Blue Book areas are the highest-value implementation vocabulary:

- P-14: locants, enclosing marks, indicated hydrogen, formatting order.
- P-21: acyclic parent hydrides and skeletal replacement.
- P-22: monocycles, retained ring names, Hantzsch-Widman heterocycles.
- P-23: bridged/von Baeyer systems.
- P-24: spiro, polyspiro, component spiro, primed locants.
- P-25: fused systems, fusion components, fusion locants, bridged fused names.
- P-28: ring assemblies and linked cyclic systems.
- P-29: substituent prefixes, `yl`, `ylidene`, `ylidyne`, free valences.
- P-31: unsaturation, hydro/dehydro, indicated hydrogen interactions.
- P-33: functional suffixes and principal characteristic groups.
- P-35: prefix forms corresponding to characteristic groups.
- P-44/P-45: parent seniority and final name choice.
- P-91/P-93: stereochemical descriptors tied to the chosen numbering.

Do not try to implement these as isolated string recipes. Each should add or
consume descriptors in the shared pipeline.

## OPSIN Lessons

OPSIN's public resources and parser structure show useful boundaries:

- token/resource data is separated from parsing code,
- group stems, cyclic groups, fusion components, hetero atoms, Hantzsch-Widman
  atoms/suffixes, suffix rules, unsaturators, locants, spiro, fusion,
  von Baeyer, indicated hydrogen, and stereo are distinct vocabularies,
- successful parsing depends on renderer grammar as much as chemical intent,
- diagnostics over many names must be batched because process startup dominates.

NISPO should learn from this separation without becoming an OPSIN clone. It is
a generator, not a parser. It only needs to generate one deterministic,
OPSIN-compatible name per molecule.

## Current NISPO Shape

Keep these strengths:

- stable public API in `nispo/api.py` and `nispo/__init__.py`,
- `NamingEngine` and `RuleSpec` dispatcher,
- `NameCandidate` concept,
- `MoleculeContext` cache,
- RDKit graph/topology utilities,
- retained/generated resource support,
- benchmark and OPSIN roundtrip workflow,
- focused tests for API, formatting, templates, fused ranking, and smoke cases.

Refine these weaknesses:

- large policy surfaces in `parent_rules.py`, `parent/rule_helpers.py`,
  `polycyclic_rules.py`, `spiro_component_rules.py`, and branch modules,
- numeric priority standing in for parent seniority,
- rule families constructing private locants and final strings too early,
- local special gates that should be descriptor predicates,
- functional-group, ring-substituent, and parent-selection logic disagreeing,
- repeated graph/substructure scans that should be cached in `MoleculeContext`.

## Suggested Internal Ownership

New work should converge toward a small `nispo/nomenclature/` layer:

```text
nomenclature/
  candidates.py       typed candidates and descriptor attachment
  trace.py            candidate decision traces
  ring_systems.py     ring-system classification descriptors
  locants.py          selected and alternative locant maps
  seniority.py        parent and name-choice comparison vectors
  functional_groups.py characteristic-group registry
  monocycles.py       P-22/Hantzsch-Widman descriptors/renderers
  von_baeyer.py       P-23 bridge descriptors/renderers
  spiro.py            P-24 spiro/component-spiro descriptors/renderers
  fused.py            P-25 fused descriptors/renderers
  unsaturation.py     P-31 hydro/dehydro/ene/yne/indicated-H state
  substituents.py     P-29 substituent descriptors
  stereo.py           parent-aware stereo descriptors/renderers
```

Existing modules can remain as adapters while this layer grows.

## Implementation Standard

For each meaningful change:

1. State the general rule being implemented.
2. Identify the descriptor or comparator field it affects.
3. Prefer adding observability before changing behavior.
4. Change one broad rule family at a time.
5. Keep retained data, generated templates, and systematic renderers separate.
6. Add focused tests or probes for the rule family.
7. Run the exact active benchmark command for behavior changes.
8. Report transition counts, success rate, regressions, parse-failure churn,
   speed, and run path.

Good changes usually reduce one of these:

- duplicate topology checks,
- duplicate locant assignment,
- hidden parent-selection policy,
- string-first rendering,
- unregistered name-producing surfaces,
- repeated expensive graph/substructure work,
- malformed names that OPSIN cannot parse.

## Integrity Rules

Never improve score by using:

- exact-SMILES special cases,
- benchmark row labels,
- answer lookups,
- generated-name lookup tables,
- OPSIN outputs as runtime data,
- external answer files,
- creation-time failure metadata,
- benchmark/scorer changes.

Benchmark failures are examples of missing rule coverage, not training labels.

## Performance Principles

Accuracy comes first during discovery, but the architecture should also get
faster over time:

- cache descriptor views in `MoleculeContext`,
- compute expensive graph decompositions lazily,
- prune candidate families by descriptor applicability,
- use seniority to avoid enumerating clearly lower-value candidates,
- measure candidate counts when adding bridge/spiro/fused enumeration,
- batch OPSIN diagnostics outside runtime.

## Final Heuristic

If a patch cannot be explained in terms of parent selection, ring-system
classification, characteristic-group seniority, locants, unsaturation,
substituent naming, rendering, resources, or stereo, it is probably too narrow
for NISPO.
