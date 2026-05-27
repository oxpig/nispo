# NISPO Blue Book Refactor Audit

Date: 2026-05-27

Scope: this is an architecture and nomenclature-rule audit of the NISPO package in
`/Users/nicholas/projects/iupac_naming/nispo`. It is written for future
implementation work against the active ring-system benchmark, but it is not a
benchmark result and does not change runtime code.

## How Future Agents Should Use This Document

This document is intended to be the ground-truth roadmap for future NISPO
refactor goal runs. It does not replace `AGENTS.md`; it sits below it. If this
document and `AGENTS.md` disagree, `AGENTS.md` wins.

Future agents should use it in this order:

1. Read `/Users/nicholas/projects/iupac_naming/AGENTS.md`.
2. Read this document's executive conclusion, target architecture, current
   phase, and validation protocol.
3. Identify the smallest phase or subphase that can make meaningful progress.
4. Implement only general deterministic nomenclature behavior.
5. Validate with focused tests/probes and, for meaningful behavior changes, the
   exact active benchmark command from `benchmark/curriculum_state.json`.
6. Judge the result by transition matrix and speed, not score alone.
7. Record what phase was advanced, what changed, what improved, what regressed,
   and which phase should be next.

This document should be treated as a living plan. Agents may update it when a
phase is completed, when evidence contradicts a proposed approach, or when a
better general Blue Book abstraction is found. Updates should preserve the same
integrity rules: no answer lookup logic, no benchmark-specific name tables, no
row-specific fixes, and no changes to benchmark scoring.

### Ground-Truth Constraints

The following constraints are non-negotiable for future goal runs:

- Preserve:
  - `from nispo import smiles_to_iupac`
  - `from nispo import smiles_to_iupac_batch`
- `smiles_to_iupac_batch` returns exactly one output per input.
- Unsupported molecules return `""` rather than aborting a batch.
- OPSIN roundtrip is the primary objective.
- Preferred IUPAC names are secondary to deterministic names that OPSIN parses
  and roundtrips.
- Runtime NISPO must not depend on OPSIN internals.
- OPSIN diagnostics must be batched when used over many candidates.
- SureChEMBL names may be used only for broad pattern analysis, not runtime
  answers or benchmark labels.
- Do not use exact-SMILES special cases, generated answer lookup tables, row
  labels, scorer changes, or creation-time failure metadata to improve score.
- Keep generated artifacts and benchmark runs out of commits unless explicitly
  requested.

### Goal-Run Operating Pattern

Each future goal run should have a short written starting state:

```text
Phase:
Hypothesis:
Target modules:
Expected benchmark transitions:
Focused tests/probes:
Benchmark command:
Speed risk:
Rollback plan:
```

Each run should finish with a similarly short result:

```text
Implemented:
General rule advanced:
Tests/probes:
Benchmark summary:
Transition matrix highlights:
Speed:
Regressions:
Next phase:
```

If a change cannot be justified as a reusable nomenclature rule, it should not
be merged into runtime behavior.

## Executive Conclusion

Yes, a substantial part of NISPO can be refactored into more general rules from
the IUPAC Blue Book, and doing so should be beneficial if it is staged behind
the existing benchmark discipline.

The current code already has the right high-level shape for a rule engine:

- a public API layer,
- a candidate-oriented naming engine,
- a molecule context/cache,
- explicit rule registration,
- RDKit graph/topology utilities,
- OPSIN roundtrip benchmarking,
- retained/generated resource support.

The main weakness is that many naming decisions are still encoded as local
heuristics inside large modules rather than as reusable Blue Book concepts.
This makes each successful benchmark improvement tend to land as a narrow patch
for one ring family, one topology, or one observed failure mode. The recent
ring-system work is a good example: the exocyclic parent-selection improvement
and the relative `cis` fallback were chemically sensible, but their current
shape is closer to targeted rescue logic than a general nomenclature layer.

My estimate is that roughly 6,500 to 8,500 lines of the current name-decision
surface could be progressively reorganized around general Blue Book rule
objects. That does not mean those lines should be rewritten at once. It means
the conceptual ownership of that logic should move from ad hoc producer
functions toward typed parent candidates, ring-system descriptors, locant maps,
seniority vectors, characteristic-group records, and stereochemical descriptors.

The public API, benchmark tools, OPSIN oracle workflow, template store, and a
large fraction of the caching/resource infrastructure should stay mostly
stable. The refactor should be an adapter-style migration, not a reset.

The likely benefits are:

- higher OPSIN roundtrip accuracy on ring systems without exact-SMILES special
  cases,
- fewer local fixes that only work for one scaffold,
- clearer parent-selection behavior,
- more predictable handling of fused, bridged, spiro, and heterocyclic systems,
- better speed from shared topology/candidate caches and earlier candidate
  pruning,
- easier debugging because every chosen name can carry a decision trace.

The main risk is that a broad Blue Book implementation can become slower and
more complex than the current rule stack. The safe path is to introduce typed
intermediate representations first, keep existing renderers alive, then migrate
one rule family at a time with transition-matrix checks.

## Current Benchmark Context

The active ring-system benchmark set is:

`/Users/nicholas/projects/iupac_naming/benchmark/sets/009_chembl_ring_systems`

The latest summary inspected during this audit was:

- rows: `49670`
- matches: `43356`
- score: `0.8728810146970002`
- `empty_name`: `3563`
- `mismatch`: `2217`
- `opsin_parse_failed`: `534`
- speed: `13.529` ms/molecule/worker

This is already in a range where brittle special cases are less attractive.
Moving from about 87.3% toward 95% likely requires better general parent
selection and ring-system naming rather than more exact scaffold patches.

## Cross-Benchmark Structural Findings

A full 001-009 sweep on 2026-05-27 gives a clearer picture than the active
ring-system set alone. It shows that the recent ring work is valuable, but it
also shows why the next development phase should be a general Blue Book
refactor rather than another local fused/spiro patch.

Latest observed set-level movement versus each set's immediately previous run:

| Set | Current score | Score movement | Match movement | Structural signal |
| --- | ---: | ---: | ---: | --- |
| `001_moljson_v0` | `0.9823` | `-0.81 pp` | `-74` | small broad-druglike regression |
| `002_surechembl_ha40_random_50k` | `0.9398` | `+32.40 pp` | `+16198` | broad historic improvement |
| `003_surechembl_ha40_failures_100k` | `0.8877` | `+5.97 pp` | `+5971` | failure-set improvement |
| `004_surechembl_ha50_random_500k` | `0.9190` | `-0.06 pp` | `-322` | broad high-HA regression |
| `005_murcko_ha25_usual_single_fused_spiro` | `0.9688` | `+5.28 pp` | `+6245` | fused/spiro improvement |
| `006_hard_ha25_usual_single_fused_spiro_100k` | `0.9376` | `+14.17 pp` | `+14165` | hard fused/spiro improvement |
| `007_all_ha25_usual_single_cyclic` | `0.9689` | `+1.12 pp` | `+11981` | broad cyclic improvement |
| `008_failures_ha25_usual_single_cyclic` | `0.6517` | `0.00 pp` | `0` | stable since last 008 run |
| `009_chembl_ring_systems` | `0.8729` | `0.00 pp` | `0` | stable since last 009 run |

The structural aggregation from that sweep has six important lessons.

1. Fused-and-spiro systems improved strongly.

   The largest gains are in molecules where fused and spiro topology interact:

   | Set | `fused_and_spiro` movement |
   | --- | ---: |
   | `003` | `27.4% -> 83.2%` |
   | `004` | `61.7% -> 83.2%` |
   | `005` | `56.8% -> 94.0%` |
   | `006` | `59.0% -> 94.6%` |
   | `007` | `59.0% -> 94.6%` |

   This validates the direction of the recent work. It also warns against
   overfitting: the improvement came from recognizing a topology family, not
   from memorizing exact rows. Future work should generalize that pattern into
   typed topology descriptors and a shared parent-selection comparator.

2. Pure spiro-only systems are now a regression risk.

   `spiro_only` movement was negative in several sets:

   | Set | `spiro_only` movement |
   | --- | ---: |
   | `001` | `-2.82 pp` |
   | `004` | `-1.11 pp` |
   | `006` | `-1.25 pp` |
   | `007` | `-2.42 pp` |

   The architecture should separate simple spiro, component spiro, polyspiro,
   and fused+spiro descriptors before ranking. A rule that helps
   fused+spiro systems should not silently change pure spiro parent choice.

3. Unfused multi-ring molecules are mostly unsolved by fused-ring logic.

   Unfused multi-ring systems regressed or barely moved in the broad sets:

   | Set | `multi_ring_unfused` movement |
   | --- | ---: |
   | `001` | `-1.38 pp` |
   | `004` | `-0.09 pp` |
   | `007` | `-0.29 pp` |
   | `008` current score | `54.1%` |

   These structures need parent-component selection, ring-as-substituent
   naming, and seniority-based component comparison. They should not be routed
   through fused/spiro-specific heuristics.

4. General ring complexity remains the largest active-set gap.

   On `009_chembl_ring_systems`, current score falls sharply as ring count
   increases:

   | Ring count | Current `009` score |
   | ---: | ---: |
   | 1 | `90.6%` |
   | 2 | `95.4%` |
   | 3 | `92.7%` |
   | 4 | `89.4%` |
   | 5 | `83.0%` |
   | 6 | `66.6%` |
   | 7+ | `32.9%` |

   The high-ring-count problem should be addressed with a general
   ring-system classifier and descriptor renderer, not more retained scaffold
   additions alone. Retained names help common cases, but generated descriptors
   need to cover the long tail.

5. The broad `004` regression is mostly parent-selection interference.

   Regression samples in `004` are usually large fused heteroaromatic
   druglike molecules with amide, ester, carbamate, sulfonamide, nitrile, or
   other characteristic groups. In aggregate:

   | Feature in `004` | Movement |
   | --- | ---: |
   | `fused_only` | `-1.98 pp` |
   | `amide` | `-1.96 pp` |
   | `carboxyl_or_ester` | `-1.29 pp` |
   | `ha_31_plus` | `-1.55 pp` |
   | `rings_3` | `-1.95 pp` |

   This argues for a unified parent-candidate seniority engine. Ring rules
   should propose parents; characteristic-group rules should propose suffix
   and substituent decisions; a single comparator should decide seniority.
   Local ring renderers should not independently override carbonyl-bearing
   parent decisions.

6. NISPO is producing more names, but some become parse failures.

   Several sets show fewer `empty_name` rows but more `opsin_parse_failed`
   rows. This is progress only if the new names can be regularized. The
   refactor should therefore include a renderer-validity layer: generated
   descriptors must pass local syntax invariants before being emitted, and
   renderer changes should be validated by parse-failure transition counts.

Design implication:

The next general solution is not "add more fused rules." It is to make all
parent candidates comparable across acyclic, monocyclic, fused, bridged,
spiro, unfused multi-ring, characteristic-group, and stereochemical contexts.
The Blue Book refactor should therefore prioritize shared descriptors,
seniority vectors, locant maps, and characteristic-group handling before
adding more string-level naming branches.

## Repo Shape

The package has about 18.5k lines across Python source, tests, and tools. The
largest rule-bearing files are:

| Area | File | Approx. lines | Role |
| --- | --- | ---: | --- |
| Parent orchestration | `nispo/parent_rules.py` | 2102 | Main rule classes and rule registration |
| Parent helpers | `nispo/parent/rule_helpers.py` | 1569 | Shared helper logic, heuristics, candidate filters |
| Polycyclic naming | `nispo/polycyclic_rules.py` | 1216 | Bicyclic, tricyclic, secondary-bridge, replacement logic |
| Spiro components | `nispo/spiro_component_rules.py` | 910 | Component-spiro generation and ranking |
| Ring parents | `nispo/ring_rules.py` | 775 | Monocycles, retained names, Hantzsch-Widman-like cases |
| Acyclic parents | `nispo/acyclic/rules.py` | 782 | Chain parent generation and functional-chain ranking |
| Functional groups | `nispo/branch/functional_groups.py` | 841 | Prefix/suffix feature detection and rendering |
| Ring substituents | `nispo/branch/ring_substituents.py` | 741 | Ring substituent naming |
| Fragment naming | `nispo/fragment_rules.py` | 843 | Disconnected/fragment-like naming paths |

The repository also includes useful central infrastructure:

- `nispo/api.py`
- `nispo/rule_engine.py`
- `nispo/rule_dispatch.py`
- `nispo/rule_priority.py`
- `nispo/naming_context.py`
- `nispo/structural_profile.py`
- `nispo/ring_topology.py`
- `nispo/graph_model.py`
- `nispo/resource_templates.py`
- `nispo/template_store.py`
- `nispo/template_index.py`

An existing architecture-debt tool reports 31 issues, mostly around
rule-producing surfaces that are not represented as first-class registry/rule
objects. That aligns with the manual audit: NISPO has a good engine shell, but
too many rule decisions are still hidden in producer functions and helper
branches.

The repo's own broader audit tool reports a high infrastructure score:

- `audit_closure_score`: `97 / 100`
- template store/resource layer: `15 / 15`
- generated fused resource consolidation: `10 / 10`
- molecule context/cache ownership: `10 / 10`
- explicit routing/module decomposition: `17 / 20`
- Blue Book-style priority metadata: `10 / 10`
- structured rendering layer: `10 / 10`
- inverse suffix resource layer: `10 / 10`
- fragment assembly: `5 / 5`
- selected-row diagnostics: `5 / 5`
- focused tests: `5 / 5`

This score is important because it changes the interpretation of the task. The
repo is not primarily missing generic software-engineering cleanup. It has
already solved much of the infrastructure problem. The remaining gap is a
nomenclature-model gap: the current data structures do not yet represent enough
of the Blue Book concepts needed to make rule behavior broadly general.

## Deeper Module Audit

### Public API And Engine Boundary

Files:

- `nispo/api.py`
- `nispo/rule_engine.py`
- `nispo/rule_dispatch.py`
- `nispo/rule_priority.py`

Current state:

- The public API is small and should be preserved.
- `NameCandidate` currently carries `name`, `family`, `priority`, `sort_key`,
  and `stereo_prefix`.
- `NamingEngine` sorts rules by numeric priority and returns the best candidate
  from the first priority bucket that yields a name.
- `RuleSpec` gives the engine a clean family/priority/gates/producer boundary.
- `RulePriority` already contains metadata fields that look like an early
  seniority model.

Refactor assessment:

- Keep this layer.
- Expand `NameCandidate` so it can optionally point at a typed descriptor.
- Keep `RuleSpec` as the compatibility layer while new rule families migrate.
- Do not replace priority ordering in one step. First make priority decisions
  explainable, then migrate final selection to seniority vectors.

Blue Book gap:

- P-44 seniority is only approximated by priority buckets and local sort keys.
- Candidate comparison should be based on structured parent-selection criteria,
  not only rule order.

### Molecule Context And Cache Layer

Files:

- `nispo/naming_context.py`
- `nispo/molecule_cache.py`
- `nispo/structural_profile.py`
- `nispo/ring_topology.py`
- `nispo/graph_model.py`

Current state:

- `MoleculeContext` is the strongest place to host shared views.
- It already caches templates, acyclic names, polycyclic records, substructure
  matches, ring-system sets, ring topologies, and structural profiles.
- `StructuralProfile` is deliberately cheap and useful for gating expensive
  producers.
- `RingSystemTopology` contains atoms, graph, edge count, cycle rank, and branch
  atoms.

Refactor assessment:

- This should become the source of all reusable Blue Book views.
- Add descriptor-level caches rather than recomputing topology and feature
  decisions inside parent producers.
- Keep the cheap `StructuralProfile`, but separate dispatch signals from naming
  decisions.

Blue Book gap:

- The context can say "generic bicyclic topology" or "spiro topology", but it
  cannot yet say "free spiro union", "non-free spiro union", "fused system",
  "bridged fused system", "von Baeyer system", "ring assembly", "main bridge",
  "secondary bridge", or "candidate locant maps".

### Parent Rule Orchestration

File:

- `nispo/parent_rules.py`

Current state:

- This is the main orchestration file and is over 2,000 lines.
- It registers more than 25 parent rule surfaces.
- It contains specific functional parent exceptions, retained parents,
  generated fused parents, generated hydro parents, component spiro parents,
  generic bicyclic/spiro/polycyclic parents, exact parents, and acyclic fallback
  rules.
- It also contains several gating heuristics:
  - `_should_defer_acyclic_skeleton_to_exocyclic_ring_parent`
  - `_should_defer_group14_hydride_to_polycycle`
  - `_should_defer_natural_product_to_generic_polycycle`
  - `_too_large_for_exhaustive_ring_parent_search`
  - `_high_rank_hetero_retained_fused_search_unpromising`
  - `_has_generic_polycyclic_stereo_fallback`

Refactor assessment:

- This file should shrink over time.
- It should become mostly registration and adapters from old producers to new
  descriptor producers.
- Most of the chemical logic should move into parent selection, ring-system,
  characteristic-group, locant, and stereo modules.

Blue Book gap:

- P-44 parent seniority and P-33/P-35 characteristic-group treatment are
  scattered across separate branches. The code can make correct local decisions,
  but it has no single object that says why a ring parent should beat a chain
  parent, or why a chain parent should beat a ring parent.

### Parent Helper Layer

File:

- `nispo/parent/rule_helpers.py`

Current state:

- This file holds many naming helpers, ranking helpers, and candidate filters.
- It acts as a shared utility module, but several helpers are really naming
  policy.
- It includes logic for fused-parent ranking, skeleton/substituent tradeoffs,
  functional prefixes, generated template handling, and resource fallback.

Refactor assessment:

- Split by Blue Book responsibility:
  - locant helpers,
  - characteristic-group helpers,
  - ring-system ranking helpers,
  - rendering helpers,
  - resource/template helpers.
- Keep truly generic utility functions.
- Move policy helpers into typed modules with tests.

Blue Book gap:

- Helper names often encode the immediate benchmark symptom rather than the
  underlying rule. The desired form is "choose parent by P-44 criteria" or
  "rank hetero locants by replacement seniority", not "defer this particular
  family".

### Monocyclic Ring Rules

File:

- `nispo/ring_rules.py`

Current state:

- Handles monocyclic carbocycles and heterocycles.
- Contains retained/Hantzsch-Widman-like tables and procedural cases for
  saturated, aromatic, diazine, azole, pyran, replacement, charge adjustment,
  and substituent suffixes.
- Exposes `ring_parent_candidates` and `ring_substituent_suffix`.

Refactor assessment:

- Keep the current functions as behavior-preserving renderers initially.
- Extract P-22 tables into data structures.
- Build a general monocycle descriptor and Hantzsch-Widman renderer.
- Make retained names a preference layer rather than the only route.

Blue Book gap:

- P-22 provides a general procedure for Hantzsch-Widman names, heteroatom
  prefix citation order, heteroatom locant selection, indicated hydrogen, and
  skeletal replacement for larger rings. The current code covers many common
  cases, but not as a single reusable grammar.

### Polycyclic Rules

File:

- `nispo/polycyclic_rules.py`

Current state:

- Generates bicyclic, spiro, tricyclic, and higher polycyclic candidate records.
- Contains graph path enumeration, bridgehead selection, bridge descriptor
  construction, secondary bridge preference keys, unsaturation decoration,
  replacement prefixes, and substituent names.
- Comments explicitly note OPSIN-compatible generic fallbacks.

Refactor assessment:

- This is a high-value refactor target.
- The graph algorithms should become a typed P-23/von Baeyer descriptor engine.
- Existing sequence generation can be preserved, but output should be a
  `BridgeSystemDescriptor` before rendering to text.

Blue Book gap:

- P-23 defines bridgehead, bridge, main ring, main bridge, secondary bridge,
  independent/dependent secondary bridges, main bridge selection, secondary
  bridge ordering, numbering, and hetero replacement. The current file contains
  many of those concepts implicitly. They should be explicit.

### Spiro Component Rules

File:

- `nispo/spiro_component_rules.py`

Current state:

- Builds component-spiro and component-dispiro names.
- Splits graph components at spiro atoms.
- Combines simple, polycyclic, and fused components.
- Handles primed locants and a component-spiro stereo prefix.
- Uses candidate caps to control combinatorial blowup.

Refactor assessment:

- This should become a renderer over a shared `RingSystemDescriptor`.
- Component splitting, component ordering, priming, and spiro locants should be
  general P-24 objects.
- The current code is valuable but too separate from the generic spiro and
  fused logic.

Blue Book gap:

- P-24 distinguishes free and non-free spiro unions, monospiro, polyspiro,
  component spiro systems, component order, priming, hetero replacement, and
  indicated hydrogen. Current support is useful but partly family-specific.

### Fused Template And Resource Layer

Files:

- `nispo/resource_templates.py`
- `nispo/template_store.py`
- `nispo/template_index.py`
- `nispo/fused_template_families.py`
- `nispo/resources/*`

Current state:

- The resource layer is strong.
- `TemplateStore` owns template compilation and selection.
- Generated fused resources are compactly indexed.
- Resource sizes are non-trivial, especially the generated fused index record
  table at about 22 MB.
- Tests confirm expected resource counts and candidate materialization.

Refactor assessment:

- Keep this layer.
- Treat resources as retained-name and scaffold-candidate evidence.
- Do not let resources become exact answer lookup behavior.
- Add descriptor metadata to templates where useful: system class, ring count,
  fusion graph class, component names, locant coverage, hetero locant quality.

Blue Book gap:

- P-25 fused-system construction is not yet represented as a general component
  decomposition and fusion-descriptor engine. Retained/generated templates fill
  much of the gap, but systematic behavior should exist behind them.

### Acyclic And Branch Rules

Files:

- `nispo/acyclic/rules.py`
- `nispo/branch_rules.py`
- `nispo/branch/functional_groups.py`
- `nispo/branch/functional_suffixes.py`
- `nispo/branch/ring_substituents.py`
- `nispo/branch/carbon.py`
- `nispo/branch/hetero_atoms.py`

Current state:

- Acyclic path search and branch rendering are pragmatic and reasonably
  modular.
- `branch_rules.py` has a central branch dispatch facade.
- `NameParts`, `Prefix`, and `Suffix` exist and are useful.
- Functional suffix resources are already used for some inverse suffix/prefix
  decisions.
- Functional group handling is still distributed across many functions.

Refactor assessment:

- Preserve branch dispatch and `NameParts`.
- Introduce a central characteristic-group registry.
- Use it to decide suffix, prefix, principal group, parent-chain requirement,
  and ring-parent compatibility.
- Make acyclic chain selection consume the same group records as ring parent
  selection.

Blue Book gap:

- P-33/P-35/P-44 require a shared view of suffixes, prefixes, functional parent
  compounds, principal characteristic groups, and parent seniority. Current code
  has many correct local fragments but not one registry with those semantics.

### Stereochemistry

Files:

- `nispo/stereochemistry.py`
- stereo-related helpers in `rule_engine.py`, `acyclic/rules.py`,
  `branch/ring_substituents.py`, and `spiro_component_rules.py`

Current state:

- Stereo is mostly applied late.
- Some rules can pass a `stereo_prefix` manually.
- `rule_engine.py` chooses fallback atom order by family.
- Ring substituent and component-spiro code contain local relative stereo logic.

Refactor assessment:

- Add parent-aware stereo descriptors.
- Compute stereo sites once, then map them through the selected locant map.
- Keep current late finalization as a fallback while migrating.

Blue Book gap:

- P-91/P-93 stereo descriptors depend on locants and parent structure. Ring
  junctions, bridgeheads, spiro atoms, relative cis/trans, pseudoasymmetric
  `r/s`, and fused-system stereochemistry cannot be generalized cleanly while
  stereo is only a final string patch.

### Tests And Diagnostics

Files:

- `tests/test_*.py`
- `tools/explain_rows.py`
- `tools/probe_rule_candidates.py`
- `tools/profile_scaffold_gaps.py`
- `tools/architecture_debt_count.py`
- `tools/audit_refactor_score.py`

Current state:

- There are focused tests for API behavior, formatting, branch smoke, fragment
  assembly, fused ranking, generated fused resource compaction, OPSIN smoke, and
  template store behavior.
- There are diagnostic tools for selected rows, candidate probing, scaffold
  gap profiling, and architecture debt.

Refactor assessment:

- Keep all existing tests.
- Add descriptor tests before behavior changes.
- Add Blue Book example tests only where OPSIN accepts the names and roundtrips.
- Add transition-matrix checks to every phase.

Blue Book gap:

- Tests currently assert important local behavior, but they do not yet encode a
  systematic rule ladder: P-22 monocycles, P-23 von Baeyer, P-24 spiro, P-25
  fused systems, P-31 hydrogenation, P-44 parent selection, and P-93 stereo.

## Current Architectural Strengths

### Candidate Engine

`rule_engine.py` has a clean core abstraction:

- `NameCandidate`
- `NamingEngine`
- priority buckets
- deterministic candidate selection
- final stereochemical prefix application

This should not be thrown away. It should be expanded so a candidate is not just
a rendered string plus priority, but a typed nomenclature decision.

### Dispatch And Gating

`rule_dispatch.py` already has a `RuleSpec` structure with family, priority,
gates, and producer. The gates keep expensive paths from running on every
molecule. This is important for the speed target.

The issue is not the presence of gates. The issue is that some gates encode
chemical decisions that should live in reusable descriptors. For example,
topology gates, charged-ring gates, and exocyclic-feature gates should ask a
central feature profile rather than duplicate molecule-specific reasoning.

### Shared Context

`naming_context.py` caches repeated molecule views:

- templates,
- acyclic names,
- polycyclic records,
- substructure matches,
- topology flags,
- structural profiles.

This is the natural place to add richer Blue Book views:

- ring-system descriptors,
- parent-candidate pools,
- locant maps,
- characteristic-group candidates,
- stereochemical site descriptors.

### Ring Topology Layer

`ring_topology.py` already computes ring-system topology objects with atoms,
graph, edge count, cycle rank, and branch atoms. This is a strong base for a
general P-22/P-23/P-24/P-25 classifier.

The missing piece is a richer classification:

- monocycle,
- fused system,
- bridged fused system,
- von Baeyer non-fused bridged system,
- free spiro system,
- non-free spiro/fused-spiro system,
- ring assembly,
- mixed spiro/fused/bridged system.

### Resource Layer

The template/resource system is useful and should remain. NISPO should be
allowed to adapt OPSIN-compatible retained names and tables into deterministic
local resources. The important line is that resources must not become benchmark
answer lookups or exact-SMILES special cases.

## Current Architectural Weaknesses

### Parent Selection Is Mixed With Rendering

The biggest issue is that functions often do all of these at once:

1. identify a possible parent,
2. decide whether it outranks another parent,
3. assign locants,
4. choose suffix/prefix treatment,
5. render the final name,
6. apply fallback heuristics.

This makes it difficult to generalize. Blue Book parent selection, especially
P-44, is inherently a ranking problem over candidate parent structures. NISPO
needs a first-class parent descriptor before rendering.

### Numeric Rule Priority Is Too Coarse

`rule_priority.py` is useful, but much of the actual seniority logic is spread
across local sort keys and producer ordering. P-44 parent selection is not just
"this family has priority 40". It uses ordered criteria such as:

- principal characteristic groups,
- ring versus chain seniority,
- heterocyclic versus carbocyclic parents,
- nitrogen and heteroatom seniority,
- greater number of rings,
- greater number of skeletal atoms,
- type of ring system, where spiro, fused, bridged fused, non-fused bridged,
  and ring assemblies are ordered by rule.

NISPO should use a structured seniority vector. Numeric rule priority can remain
as a coarse family gate, but final selection should be explainable in Blue Book
terms.

### Ring-System Families Are Stringly Typed

Several APIs use string family labels such as `"bicyclic"`, `"spiro"`, and
`"kekulized"` when fetching polycyclic records. This works, but it makes it
hard to express mixed systems and shared ranking criteria.

A `RingSystemDescriptor` should carry explicit fields:

- atoms,
- bonds,
- ring count,
- cycle rank,
- ring-system class,
- bridgeheads,
- bridges,
- spiro atoms,
- fused bonds,
- fusion graph,
- hetero atoms,
- unsaturation state,
- possible locant maps.

### Monocycle Logic Is Part Table, Part Grammar

`ring_rules.py` contains retained names and Hantzsch-Widman-like logic in
procedural maps. This is workable for common systems, but it does not scale
cleanly to uncommon heterocycles, replacement variants, or systematic fallback
names.

P-22 is a good candidate for a grammar/data split:

- table-driven heteroatom prefix seniority,
- ring-size stems,
- saturation/mancude choices,
- retained-name overrides,
- replacement-nomenclature fallback for larger rings.

### Polycyclic Bridge Logic Is Improving But Still Local

`polycyclic_rules.py` now contains useful ordering logic for secondary bridges,
including P-23.2.6-style concepts. This should become a typed von Baeyer engine
rather than remaining as local ranking functions over generated records.

The engine should explicitly represent:

- main ring,
- main bridge,
- independent secondary bridges,
- dependent secondary bridges,
- bridge lengths,
- bridge attachment locants,
- numbering path,
- hetero replacement sites,
- unsaturation/hydro sites.

### Spiro Logic Is Separate From General Ring Seniority

`spiro_component_rules.py` does useful component-spiro work, but spiro systems
should participate in the same ring-system classification and parent-selection
machinery as fused and bridged systems. P-24 defines free spiro systems,
polyspiro systems, component spiro names, priming, and numbering. These are
general rules, not a separate collection of emergency producers.

### Functional Groups Are Detected In Several Places

Characteristic groups and exocyclic features affect:

- whether a ring or chain is the parent,
- whether a group is named as suffix or prefix,
- which locants are needed,
- whether a ring parent should be deferred to a functional acyclic parent,
- whether a chain is merely an exocyclic group attached to a senior ring parent.

At present, this logic is split across:

- `structural_profile.py`,
- `branch/functional_groups.py`,
- `acyclic/rules.py`,
- `parent_rules.py`,
- `parent/rule_helpers.py`.

P-33, P-35, and P-44 should be represented by one characteristic-group registry
that can answer suffix/prefix/seniority questions consistently.

### Stereochemistry Is A Final String Patch

The current engine applies stereochemical prefixes late. That is useful, but
ring-system stereochemistry depends on the chosen parent and its locants.
Recent work added a relative `cis` fallback for a narrow bicyclic case. P-91 and
P-93 suggest a broader layer:

- tetrahedral `R/S`,
- pseudoasymmetric `r/s`,
- relative `cis/trans`,
- bridgehead and ring-junction descriptors,
- fused-ring junction descriptors,
- spiro-center descriptors,
- endo/exo/syn/anti where applicable.

This should be locant-aware and parent-aware rather than a fallback attached to
one generated parent family.

## Blue Book Rule Areas To Encode As General Modules

### 1. Parent Selection And Seniority

Primary sources: P-44, with support from P-33, P-35, P-41, and P-43.

Implementation target:

- create a `SeniorityVector`,
- make every parent candidate carry the vector,
- compare candidates through a central comparator,
- keep current priority values as coarse rule-family dispatch only.

Example vector fields:

```python
SeniorityVector(
    principal_group_rank=...,
    parent_kind_rank=...,
    ring_system_type_rank=...,
    heterocycle_rank=...,
    nitrogen_count=...,
    heteroatom_seniority_tuple=...,
    ring_count=...,
    skeletal_atom_count=...,
    locant_quality=...,
    unsaturation_quality=...,
    retained_name_preference=...,
)
```

The comparator should be deterministic and should expose a trace explaining why
one candidate beat another.

### 2. Ring-System Classification

Primary sources: P-22, P-23, P-24, P-25, P-28, P-44.

Implementation target:

- build one cached `RingSystemDescriptor` per ring system,
- classify topology before any naming rule renders a string,
- expose the same descriptor to monocycle, spiro, fused, and bridged producers.

This should replace several family-specific topology tests with one source of
truth.

### 3. Locant And Numbering Engine

Primary sources: P-14 generally, plus the family-specific numbering rules in
P-22/P-23/P-24/P-25/P-31/P-93.

Implementation target:

- create a `LocantMap` object,
- separate atom-numbering generation from name rendering,
- let candidate producers emit multiple possible locant maps,
- rank locant maps by Blue Book criteria,
- make stereo and functional-group layers consume the selected locants.

This is essential for generalizing stereochemistry and heteroatom placement.

### 4. Monocyclic Parent Grammar

Primary source: P-22.

Implementation target:

- convert retained monocycle data and Hantzsch-Widman tables into a structured
  grammar,
- render 3-10 member heterocycles by heteroatom prefixes, seniority, ring size,
  and saturation state,
- support larger heteromonocycles through skeletal replacement fallback,
- keep retained names as preferred overrides where OPSIN roundtrip benefits.

This would reduce the hard-coded branching in `ring_rules.py`.

### 5. Von Baeyer Bridged Ring Engine

Primary source: P-23.

Implementation target:

- build a real bridge decomposition from the ring-system descriptor,
- select main ring, main bridge, and secondary bridges by P-23.2.6 criteria,
- assign numbering once,
- render bicyclo/tricyclo/tetracyclo descriptors from bridge records,
- apply skeletal replacement for hetero atoms,
- apply hydro/dehydro/indicated-H later through the unsaturation layer.

This should absorb much of the current `polycyclic_rules.py` complexity.

### 6. Spiro And Polyspiro Engine

Primary source: P-24.

Implementation target:

- classify free spiro systems separately from non-free spiro/fused systems,
- support monospiro and polyspiro descriptor generation,
- implement component-spiro naming as a renderer over the same descriptor,
- handle component ordering and priming centrally,
- expose spiro locants to stereochemistry.

This should consolidate `spiro_component_rules.py` with other ring-system
logic.

### 7. Fused And Bridged-Fused Engine

Primary source: P-25.

Implementation target:

- represent fused components, fusion bonds, fusion descriptors, and parent
  components explicitly,
- keep retained fused resources as data,
- use systematic construction only when the component decomposition is clear,
- distinguish fused, bridged fused, and non-fused bridged systems before
  choosing P-23 or P-25 behavior,
- centralize fused-system numbering and heteroatom locant ranking.

This is likely one of the largest accuracy opportunities, but also one of the
highest-risk areas. It should come after typed ring descriptors and locants.

### 8. Unsaturation, Hydro, Dehydro, And Indicated Hydrogen

Primary source: P-31, with P-14 and P-25 support.

Implementation target:

- create an unsaturation descriptor tied to the selected parent skeleton,
- decide whether a parent is saturated, mancude, partially hydrogenated, or
  needs hydro/dehydro prefixes,
- rank low locants for unsaturation after parent numbering is known,
- avoid duplicating partial-hydro logic in fused and bridged rule modules.

### 9. Characteristic-Group Registry

Primary sources: P-33, P-35, P-44.

Implementation target:

- create a registry of characteristic groups with suffix/prefix behavior,
  seniority, valence expectations, and multiplicity handling,
- represent exocyclic groups as records, not just boolean profile flags,
- let parent selection decide whether a functional feature is part of a parent
  chain, a suffix on a ring parent, or a substituent prefix.

This is the general version of the recent exocyclic ring-parent precedence
work.

### 10. Stereochemical Descriptor Layer

Primary sources: P-91 and P-93.

Implementation target:

- compute stereochemical sites independently from rendering,
- attach them to the selected parent locant map,
- render absolute, relative, ring-junction, and spiro descriptors through one
  layer,
- support `cis/trans` beyond one bicyclic fallback,
- preserve OPSIN roundtrip as the deciding validation criterion.

## Blue Book Requirements Matrix

This matrix turns the local Blue Book audit into concrete implementation
requirements. It is intentionally pragmatic: NISPO does not need to become a
complete preferred-IUPAC generator before it improves. It needs to encode the
general rules that repeatedly explain active benchmark failures.

| Blue Book area | Rule meaning for NISPO | Current implementation | Required refactor |
| --- | --- | --- | --- |
| P-14 general locants and formatting | Low locants, enclosing marks, multiplicative prefixes, indicated hydrogen order | Formatting helpers exist; locant ranking is spread across modules | Central `LocantMap`, locant ranking keys, and rendering helpers consumed by all rule families |
| P-22 monocycles | Hantzsch-Widman names, skeletal replacement, heteroatom order, indicated H | `ring_rules.py` has useful retained and procedural cases | `MonocycleDescriptor` plus table-driven Hantzsch-Widman/replacement renderer |
| P-23 von Baeyer systems | Bridgeheads, main ring, main bridge, secondary bridges, hetero replacement | `polycyclic_rules.py` has graph algorithms and local keys | `VonBaeyerDescriptor` with explicit bridge model and numbering |
| P-24 spiro systems | Free spiro unions, mono/polyspiro, component spiro, primed locants | `polycyclic_rules.py` and `spiro_component_rules.py` both handle parts | Unified `SpiroDescriptor` and component-spiro renderer |
| P-25 fused systems | Fused components, fusion descriptors, fused numbering, bridged fused systems | Retained/generated template resources are strong; systematic construction is limited | `FusedSystemDescriptor` behind resource-first candidate ranking |
| P-28 ring assemblies | Directly joined cyclic systems, bi/ter/quater assemblies, primed/composite locants | Mostly branch/substituent behavior; not a clear parent family | Ring assembly classification in `RingSystemDescriptor`, later renderer if benchmark warrants |
| P-29 substituent derivation | `yl`, `ylidene`, `ylidyne`, free-valence locants | Branch and ring substituent functions cover many cases | Shared substituent descriptor using selected parent locants |
| P-31 hydrogenation/unsaturation | `ene`, `yne`, hydro/dehydro, indicated H priority | Handled in ring, fused, polycyclic, and formatting helpers | Parent-aware `UnsaturationDescriptor` and renderer |
| P-33 suffixes | Principal characteristic groups and suffixes | Some suffix resources and functional rules exist | Central characteristic-group registry with suffix seniority |
| P-35 prefixes | Characteristic-group prefixes and prefix forms | `branch/functional_groups.py` has many local prefix functions | Prefixes generated from the same characteristic-group registry |
| P-44 parent seniority | Parent choice across rings, chains, characteristic groups, ring types | Named priorities and local sort keys approximate this | `SeniorityVector` and parent-selection comparator |
| P-45 name choice after parent selection | Prefix locants, repeated parent structures, final preferred form | Some local sort keys and render helpers | Later-stage name-choice scoring after parent seniority is stable |
| P-91 stereodescriptor principles | Which stereodescriptors exist and when they are omitted | `stereochemistry.py` and local stereo fallbacks | `StereoSiteDescriptor` layer tied to selected parent locants |
| P-93 configuration specification | Ring, von Baeyer, spiro, fused, and ring assembly stereo | Local simple-ring/component-spiro handling | General parent-aware stereo renderer |

The highest-impact rows for the active ring-system benchmark are P-22, P-23,
P-24, P-25, P-31, P-33/P-35, P-44, and P-93.

## Refactorable Surface Estimate

The following estimate is based on file roles, line counts, function surfaces,
and the architecture debt report. It is not a rewrite recommendation; it is a
map of where Blue Book concepts should eventually own behavior.

| Area | Current line surface | Refactorability | Recommended treatment |
| --- | ---: | --- | --- |
| `parent_rules.py` | about 2100 | Very high | Shrink into registration/adapters; move policy to descriptors |
| `parent/rule_helpers.py` | about 1570 | Very high | Split into locants, parent selection, features, rendering |
| `polycyclic_rules.py` | about 1210 | Very high | Convert to P-23 descriptor engine plus renderer |
| `spiro_component_rules.py` | about 910 | High | Convert to P-24 descriptor/render layer |
| `ring_rules.py` | about 775 | High | Convert to P-22 grammar plus retained overrides |
| `acyclic/rules.py` | about 780 | Medium-high | Integrate with characteristic groups and seniority |
| `branch/functional_groups.py` | about 840 | High | Convert much of it to characteristic-group registry |
| `branch/ring_substituents.py` | about 740 | Medium-high | Consume ring descriptors and locant maps |
| `rule_engine.py` | about 140 | Medium | Keep, extend candidate data |
| `rule_dispatch.py` | about 120 | Medium | Keep, migrate gates to descriptors |
| `naming_context.py` | about 180 | High leverage | Expand as descriptor cache |
| `ring_topology.py` | about 60 | High leverage | Keep as graph kernel feeding richer descriptors |
| `template_store.py` and resources | large data surface | Low-medium | Keep; add metadata, do not replace with code |

Approximate conclusion:

- 35-45% of the repository is stable infrastructure and should mostly remain.
- 35-45% is naming policy that should be reorganized into Blue Book concepts.
- 10-20% is fallback or compatibility logic that should be retired only after
  descriptor-based rules prove stable.

The greatest accuracy opportunity is not in changing the engine loop. It is in
changing what candidates know about themselves before the engine compares them.

## Target End-State Architecture

The desired end state is a descriptor-driven naming engine. The engine should
not be a collection of unrelated string producers. It should be a pipeline that
turns a molecule into reusable nomenclature facts, proposes typed candidates,
ranks those candidates by Blue Book-inspired criteria, renders the selected
candidate, and validates behavior externally through OPSIN benchmarks.

### End-State Pipeline

The target pipeline is:

```text
RDKit molecule
  -> MoleculeContext
  -> graph, feature, ring-system, locant, characteristic-group, stereo views
  -> typed parent and substituent candidates
  -> seniority and name-choice comparison
  -> rendered IUPAC-style name
  -> external OPSIN roundtrip validation in tests/benchmarks
```

Runtime naming should stay deterministic and local. OPSIN is not in the runtime
pipeline. OPSIN remains the benchmark oracle.

### General Solution: Shared Candidate Semantics

The refactor should make every naming path speak the same internal language.
At present, acyclic chains, monocyclic parents, fused systems, bridged systems,
spiro systems, fragments, and branch substituents can all make local string
decisions. That makes local improvements easy but makes global regressions
likely. The target architecture should instead use shared candidate semantics:

1. Every rule family proposes one or more typed candidates.
2. Every candidate carries the same core fields:
   - parent scope,
   - topology descriptor,
   - characteristic-group descriptor,
   - substituent attachment map,
   - locant map,
   - unsaturation/hydrogenation state,
   - stereochemical sites,
   - seniority vector,
   - renderer family,
   - decision trace.
3. A shared seniority comparator chooses between candidates.
4. The selected candidate is rendered by the renderer that owns its descriptor
   family.
5. The final string is checked by local syntax invariants before it is
   returned.

This solves the main problem shown by the 001-009 sweep: a fused/spiro-specific
improvement should not be able to accidentally outrank a better
characteristic-group parent or a simpler pure-spiro parent. The right place for
that decision is the shared comparator, not the individual rule file.

The most important shared fields are:

| Field | Purpose | Regression prevented |
| --- | --- | --- |
| `parent_scope` | Distinguish acyclic, ring, fused, spiro, fragment, substituent parent | fused rules stealing unfused or acyclic cases |
| `topology_class` | Distinguish monocycle, unfused multi-ring, fused, bridged, spiro, fused+spiro | pure spiro regressions from fused+spiro logic |
| `component_count` | Represent independent parent components before rendering | unfused multi-ring misrouting |
| `characteristic_group_rank` | Encode suffix/prefix seniority | carbonyl-containing broad-set regressions |
| `preferred_suffix_site` | Keep suffix-bearing group and parent choice coupled | amide/ester parent interference |
| `locant_quality` | Compare numbering choices uniformly | local locant maps disagreeing |
| `renderer_validity` | Reject internally malformed generated names | new `opsin_parse_failed` churn |

The comparator should initially mirror existing priorities so behavior stays
stable. Once descriptors are populated, phases can replace one heuristic at a
time with an explicit Blue Book rule. This is the safest path to broad
improvement because it converts rule interactions from hidden control flow into
data that can be inspected and tested.

### Layer Responsibilities

| Layer | Responsibility | Should not do |
| --- | --- | --- |
| Public API | Accept SMILES and return names or empty strings | expose internal descriptors as public API |
| Engine | Route rule families, compare candidates, finalize name | encode chemical special cases directly |
| Context/cache | Own molecule-wide reusable views | make final naming policy decisions |
| Graph/topology | Provide graph facts and ring-system topology | render names |
| Descriptors | Represent parent, ring, locant, functional, stereo facts | depend on benchmark row identities |
| Seniority | Compare parent candidates by rule criteria | perform molecule graph searches |
| Renderers | Convert selected descriptors into strings | choose parent family by hidden heuristics |
| Resources | Provide retained/template names and static nomenclature data | become exact answer lookup tables |
| Diagnostics | Explain candidate choices and benchmark transitions | change runtime naming behavior |

### Desired Package Shape

The repo should converge toward this package shape:

```text
nispo/
  api.py
  rule_engine.py
  rule_dispatch.py
  rule_priority.py
  naming_context.py
  nomenclature/
    candidates.py
    trace.py
    features.py
    functional_groups.py
    locants.py
    monocycles.py
    parent_selection.py
    ring_systems.py
    seniority.py
    stereo.py
    substituents.py
    unsaturation.py
    von_baeyer.py
    spiro.py
    fused.py
  resources/
  branch/
  acyclic/
  parent/
  tests/
  tools/
```

Existing modules do not need to be moved immediately. The goal is to make each
new descriptor module own one concept, then migrate legacy functions toward
those modules as tests and benchmarks permit.

### End-State Rule Invariants

A mature NISPO rule should satisfy these properties:

- It is general across a structural family, not tied to one benchmark molecule.
- It can explain which Blue Book rule family it implements or approximates.
- It emits typed descriptors before rendering strings.
- It only proposes candidates; cross-family parent choice is made by the shared
  comparator.
- It declares its structural applicability as predicates over descriptors, not
  as scattered string checks.
- It uses shared locant maps instead of constructing private atom-number maps.
- It uses the shared characteristic-group registry for suffix/prefix decisions.
- It uses the shared seniority comparator for parent selection.
- It has focused tests that assert structure-level behavior.
- It does not require OPSIN at runtime.
- It can be disabled or bypassed without corrupting other rule families.

### End-State Data Invariants

Resource and generated data should satisfy these properties:

- Static tables represent nomenclature data, not benchmark answers.
- Retained names are stored as retained names with source family metadata.
- Generated templates are compact and indexed by structural features.
- Resource records may carry descriptor metadata, such as ring count or
  component class, but must not carry active benchmark row answers.
- If data is derived from OPSIN-compatible resources, that derivation should be
  documented and deterministic.

### End-State Testing Shape

The test suite should eventually have these groups:

- API contract tests.
- Formatting and `NameParts` tests.
- Descriptor construction tests.
- P-22 monocycle tests.
- P-23 von Baeyer tests.
- P-24 spiro tests.
- P-25 fused tests.
- P-31 unsaturation/hydrogenation tests.
- P-33/P-35 characteristic-group tests.
- P-44 parent-selection tests.
- P-91/P-93 stereo tests.
- Resource/template tests.
- Branch/substituent tests.
- OPSIN roundtrip smoke tests.
- Focused regression tests for benchmark-derived general rules.

The test suite should not become a list of benchmark row answers. A benchmark
failure can motivate a test only when the test asserts a general rule or a
representative family behavior.

## Proposed New Internal Data Model

The refactor should not start by changing final names. It should start by
making current decisions inspectable.

### ParentCandidateDescriptor

```python
@dataclass(frozen=True)
class ParentCandidateDescriptor:
    family: str
    parent_scope: ParentScope
    parent_kind: str
    atoms: frozenset[int]
    component_atoms: tuple[frozenset[int], ...]
    base_name: str
    rendered_name: str | None
    ring_system: RingSystemDescriptor | None
    locants: LocantMap | None
    characteristic_groups: tuple[CharacteristicGroupCandidate, ...]
    preferred_suffix_group: CharacteristicGroupCandidate | None
    unsaturation: UnsaturationDescriptor | None
    stereo_sites: tuple[StereoSiteDescriptor, ...]
    seniority: SeniorityVector
    renderer_family: str
    renderer_validity: RendererValidity
    render_ops: tuple[RenderOp, ...]
    diagnostics: tuple[str, ...]
```

Existing producers can initially fill only the fields they know. Over time, the
engine should rank descriptors first and render strings second.

Important design rule:

`family` describes the producer. `parent_scope`, `ring_system`,
`characteristic_groups`, `seniority`, and `renderer_validity` describe the
chemical candidate. Ranking must depend on the chemical candidate fields, not
on which legacy file produced the string. This distinction is what prevents a
new fused-rule producer from outranking a better acyclic or characteristic
group parent merely because it has a higher local priority.

### RingSystemDescriptor

```python
@dataclass(frozen=True)
class RingSystemDescriptor:
    atoms: frozenset[int]
    bonds: frozenset[tuple[int, int]]
    cycle_rank: int
    ring_count: int
    system_class: RingSystemClass
    component_count: int
    component_descriptors: tuple[RingComponentDescriptor, ...]
    rings: tuple[frozenset[int], ...]
    fused_bonds: tuple[tuple[int, int], ...]
    spiro_atoms: tuple[int, ...]
    bridgeheads: tuple[int, ...]
    bridge_paths: tuple[BridgePath, ...]
    hetero_atoms: tuple[int, ...]
    candidate_locants: tuple[LocantMap, ...]
```

This should be computed once per molecule in `NamingContext`.

`system_class` should be a true enum rather than a string convention. It should
cover at least:

- `acyclic`,
- `monocycle`,
- `unfused_ring_assembly`,
- `fused`,
- `bridged_fused`,
- `von_baeyer`,
- `spiro`,
- `component_spiro`,
- `fused_and_spiro`,
- `multi_fragment`.

The 001-009 sweep shows why this matters. Pure spiro, fused+spiro, and
unfused multi-ring molecules behave differently in benchmarks and should not
share hidden fallback branches.

### CharacteristicGroupCandidate

```python
@dataclass(frozen=True)
class CharacteristicGroupCandidate:
    group_key: str
    atoms: frozenset[int]
    attachment_atoms: tuple[int, ...]
    suffix_term: str | None
    prefix_term: str | None
    seniority_rank: int
    can_be_principal_suffix: bool
    requires_parent_chain: bool
    can_attach_to_ring_parent: bool
```

This lets the parent selector compare a ring parent bearing a suffix against an
acyclic functional parent without re-deriving chemistry in separate files.

### Decision Trace

Every final candidate should optionally carry a trace:

```python
NomenclatureDecisionTrace(
    candidate_family="von_baeyer",
    accepted_by=["ring_system_class=non_fused_bridged", "p23_bridge_descriptor"],
    rejected_competitors=[
        ("acyclic", "lower parent_kind_rank"),
        ("generic_fused", "not fused system"),
    ],
    seniority_vector=(...),
)
```

This is for diagnostics and tests, not for runtime output.

## Multi-Phase Development Plan

The plan below is designed to refactor toward general Blue Book rules while
preserving OPSIN roundtrip performance and avoiding benchmark-specific
shortcuts. Each phase has a narrow architectural goal, concrete module targets,
expected benchmark effect, and validation gate.

### Phase 0: Baseline, Instrument, And Classify

Goal:

- make later refactors measurable and reversible.

Primary code targets:

- `tools/explain_rows.py`
- `tools/probe_rule_candidates.py`
- `tools/profile_scaffold_gaps.py`
- `tests/`

Actions:

- Record the exact active benchmark command from
  `benchmark/curriculum_state.json`.
- Preserve the latest active-set summary and transition matrix as the starting
  baseline.
- Preserve a lightweight 001-009 score table when doing general refactor work,
  because the active set alone does not catch broad druglike regressions.
- Add a small descriptor/probe fixture set with representatives for:
  - simple monocycle,
  - heteromonocycle,
  - bicyclo/von Baeyer,
  - tricyclo with secondary bridge,
  - simple spiro,
  - component spiro,
  - fused+spiro,
  - pure spiro-only,
  - unfused multi-ring assembly,
  - retained fused,
  - generated fused,
  - high ring-count fused system,
  - large fused heteroaromatic amide,
  - large fused heteroaromatic ester or carbamate,
  - exocyclic carbonyl/thione/imino ring parent,
  - stereospecified ring junction.
- Add a diagnostic mode that can print internal descriptors for selected SMILES
  without persisting generated benchmark answers.
- Add Blue Book example tests only when local OPSIN parses the example and
  roundtrips to the intended graph.

Deliverables:

- a stable baseline note in the audit or a small diagnostics README,
- focused probe rows,
- descriptor dump command,
- no behavior change.

Validation gate:

- focused tests pass,
- active benchmark score and speed are unchanged within normal run noise.
- for cross-family refactors, run or sample 001/004/007/009 before accepting:
  - `001` protects small broad-druglike behavior,
  - `004` protects high-heavy-atom functionalized druglike behavior,
  - `007` protects broad cyclic behavior,
  - `009` protects bare ring-system behavior.

Expected benchmark effect:

- none. This phase exists to prevent uncontrolled regression.

### Phase 1: Descriptor-Aware Candidates With No Selection Change

Goal:

- introduce a typed internal representation while preserving current names.

Primary code targets:

- `nispo/rule_engine.py`
- `nispo/name_parts.py`
- new `nispo/nomenclature/candidates.py`
- new `nispo/nomenclature/trace.py`

Actions:

- Add `ParentCandidateDescriptor`.
- Add `NomenclatureDecisionTrace`.
- Add an optional `descriptor` field to `NameCandidate`.
- Keep `NameCandidate.name` as the source of runtime output.
- Add adapters so existing producers can attach partial descriptors without
  changing their output.
- Add tests that descriptor attachment does not change rendered names.

Deliverables:

- descriptor dataclasses,
- compatibility adapter from old `NameCandidate`,
- debug-only decision trace object,
- no ranking change.

Validation gate:

- `tests/test_api_contract.py`,
- `tests/test_roundtrip_smoke.py`,
- one focused name-equivalence test over representative probe rows,
- active benchmark has no meaningful transition changes.

Expected benchmark effect:

- no accuracy change,
- no meaningful speed change if descriptor generation is lazy.

### Phase 2: RingSystemDescriptor And LocantMap Cache

Goal:

- make P-22/P-23/P-24/P-25 topology concepts first-class and cached.

Primary code targets:

- `nispo/ring_topology.py`
- `nispo/naming_context.py`
- new `nispo/nomenclature/ring_systems.py`
- new `nispo/nomenclature/locants.py`

Actions:

- Create `RingSystemDescriptor`.
- Create `LocantMap`.
- Populate descriptors from current `RingSystemTopology`.
- Classify ring systems as:
  - acyclic,
  - monocycle,
  - fused,
  - bridged fused,
  - non-fused bridged/von Baeyer,
  - free spiro,
  - non-free spiro,
  - ring assembly,
  - mixed or unknown.
- Add fields for fused bonds, spiro atoms, bridgehead candidates, ring count,
  cycle rank, hetero atoms, unsaturation summary, and external attachments.
- Cache descriptors in `MoleculeContext`.
- Change only gates at first: `generic_bicyclic_topology`,
  `generic_polycyclic_topology`, and `spiro_topology` should read descriptors
  instead of duplicating graph checks.

Deliverables:

- cached descriptor layer,
- descriptor tests for known topologies,
- gates migrated to descriptor predicates.

Validation gate:

- descriptor tests,
- branch smoke tests,
- fused ranking tests,
- active benchmark transition matrix has no large `match -> mismatch` churn.

Expected benchmark effect:

- small speed improvement is possible from less repeated topology work,
- no intended accuracy change.

### Phase 3: Blue Book Seniority Vector

Goal:

- replace hidden priority policy with explainable P-44-style parent selection.

Primary code targets:

- `nispo/rule_priority.py`
- new `nispo/nomenclature/seniority.py`
- `nispo/rule_engine.py`
- selected candidate producers in `parent_rules.py`

Actions:

- Implement `SeniorityVector`.
- Encode P-44 parent criteria as ordered vector fields:
  - principal characteristic group count and seniority,
  - senior atom/class,
  - ring over chain when same class applies,
  - parent scope,
  - component count and component seniority,
  - ring-system type,
  - heterocycle presence,
  - nitrogen presence,
  - heteroatom seniority order,
  - ring count,
  - skeletal atom count,
  - heteroatom count,
  - unsaturation and indicated-H quality,
  - characteristic-group suffix locants,
  - stereochemical completeness where relevant.
- Populate vectors for old candidates without changing selection.
- Add trace output showing when numeric priority overrides seniority.
- Add a comparison report for broad regressions:
  - candidate that old logic chose,
  - candidate that new seniority would choose,
  - characteristic group that changed priority,
  - topology class that changed priority,
  - rendered-name validity before OPSIN.
- Move one low-risk family to seniority comparison only after diagnostics are
  stable.

Deliverables:

- central seniority comparator,
- traces for parent-selection decisions,
- tests for P-44 ordering examples.

Validation gate:

- existing focused tests pass,
- P-44 example tests pass,
- active benchmark transition matrix shows no broad regressions.

Expected benchmark effect:

- initially none,
- later small positive movement from better ring-vs-chain decisions.
- fewer 001/004 regressions once carbonyl-bearing ring parents and acyclic
  functional parents are compared by the same vector.

### Phase 4: Characteristic Group Registry And Exocyclic Parent Generalization

Goal:

- generalize the recent exocyclic ring-parent rescue into P-33/P-35/P-44
  characteristic-group logic.

Primary code targets:

- new `nispo/nomenclature/functional_groups.py`
- `nispo/structural_profile.py`
- `nispo/branch/functional_groups.py`
- `nispo/branch/functional_suffixes.py`
- `nispo/acyclic/rules.py`
- `nispo/parent_rules.py`

Actions:

- Add `CharacteristicGroupCandidate`.
- Build a registry for suffix/prefix pairs:
  - oxo/one,
  - thioxo/thione,
  - imino/imine where OPSIN-compatible,
  - hydroxy/ol,
  - amino/amine,
  - cyano/nitrile,
  - carboxy/carboxylic acid,
  - carbamoyl/amide,
  - sulfonyl/sulfone,
  - sulfinyl/sulfoxide,
  - phosphoryl/phosphine oxide style groups where supported.
- Represent attachment atoms, principal group eligibility, suffix term, prefix
  term, seniority rank, parent-chain requirement, and ring-parent compatibility.
- Replace `_should_defer_acyclic_skeleton_to_exocyclic_ring_parent` with a
  general comparison:
  - ring parent plus characteristic group,
  - acyclic functional parent,
  - retained functional parent,
  - substituent prefix fallback.
- Add explicit rules for when carbonyl-bearing functional groups should remain
  suffix-bearing parents instead of being reduced to prefixes on a newly
  selected ring parent.
- Add broad-set regression probes for:
  - fused heteroaromatic amides,
  - carbamates and esters attached to fused parents,
  - sulfonamides and sulfonyl substituents,
  - nitriles on fused heteroaromatics,
  - charged zwitterionic functional groups.
- Keep the current successful exocyclic benchmark cases as tests.

Deliverables:

- central characteristic-group records,
- shared suffix/prefix decisions,
- parent-selection integration for ring-vs-chain cases.

Validation gate:

- focused tests for exocyclic carbonyl/thione/imino/nitrile cases,
- active benchmark transition matrix with attention to:
  - `empty_name -> match`,
  - `mismatch -> match`,
  - `match -> mismatch`.

Expected benchmark effect:

- likely positive on heterocycles and fused systems with exocyclic functions,
- should reduce narrow special-case pressure in `parent_rules.py`,
- potential speed improvement from one shared feature pass,
- lower risk of 004-style regressions where fused ring selection disrupts
  amide/carboxyl/ester parent decisions.

### Phase 5: P-22 Monocycle Grammar

Goal:

- turn common monocycle logic into a reusable Hantzsch-Widman and replacement
  grammar.

Primary code targets:

- `nispo/ring_rules.py`
- new `nispo/nomenclature/monocycles.py`
- `nispo/constants.py`
- resource files if table extraction is useful.

Actions:

- Extract P-22 heteroatom prefix order.
- Extract ring-size stems and saturation/mancude rules.
- Represent retained monocycle names as overrides.
- Generate Hantzsch-Widman candidates for 3-10 member heterocycles.
- Generate skeletal replacement fallback for larger heteromonocycles.
- Assign heteroatom locants by:
  - lowest set,
  - heteroatom seniority order,
  - then unsaturation and indicated-H locants.
- Preserve existing retained names when OPSIN roundtrip is better.

Deliverables:

- `MonocycleDescriptor`,
- Hantzsch-Widman renderer,
- replacement renderer for larger rings,
- retained-name preference adapter.

Validation gate:

- P-22 example tests,
- existing ring and branch tests,
- active benchmark comparison focused on single-ring and ring-substituent rows.

Expected benchmark effect:

- better uncommon heteromonocycle coverage,
- fewer `empty_name` single-ring cases,
- better components for later fused/spiro work.

### Phase 6: P-23 Von Baeyer Descriptor Engine

Goal:

- make bicyclic/tricyclic/polycyclic bridge naming explicit and general.

Primary code targets:

- `nispo/polycyclic_rules.py`
- new `nispo/nomenclature/von_baeyer.py`
- `nispo/nomenclature/ring_systems.py`
- `nispo/nomenclature/locants.py`

Actions:

- Add `BridgePath`, `BridgeheadPair`, and `VonBaeyerDescriptor`.
- Use ring-system graph descriptors to enumerate bridgehead pairs and paths.
- Select:
  - main ring with maximum skeletal atoms,
  - main bridge with maximum non-main-ring atoms,
  - independent secondary bridges before dependent bridges,
  - longest independent secondary bridge first,
  - minimum dependent bridge count,
  - lowest secondary bridge locant set,
  - lowest locants in citation order.
- Assign numbering once and expose it as `LocantMap`.
- Render bicyclo/tricyclo/tetracyclo descriptors from the typed bridge model.
- Add hetero replacement through the shared locant map.
- Move unsaturation rendering to the unsaturation layer when available.

Deliverables:

- typed bridge descriptor engine,
- compatibility renderer matching current names where possible,
- tests for P-23 examples and current generic bicyclic/tricyclic successes.

Validation gate:

- P-23 tests,
- focused probes for 2-5 ring systems,
- active benchmark transition matrix.

Expected benchmark effect:

- likely positive for bridged and high-cycle-rank ring systems,
- lower risk of row-specific bridge fixes,
- possible speed improvement once bridge enumeration is pruned by descriptors.

### Phase 7: P-24 Spiro And Component-Spiro Unification

Goal:

- unify generic spiro, polyspiro, component-spiro, and component-dispiro logic.

Primary code targets:

- `nispo/spiro_component_rules.py`
- `nispo/polycyclic_rules.py`
- new `nispo/nomenclature/spiro.py`
- `nispo/nomenclature/locants.py`

Actions:

- Add `SpiroDescriptor`.
- Distinguish free spiro unions from non-free spiro unions.
- Distinguish pure spiro-only systems from fused+spiro systems before
  candidate ranking.
- Implement monospiro descriptor generation for simple rings.
- Implement linear and branched polyspiro descriptors.
- Represent component spiro names as renderers over component descriptors.
- Centralize primed locants and component order.
- Share spiro locants with stereo and characteristic-group layers.
- Retain candidate caps, but base them on descriptor rank rather than nested
  producer loops.
- Add negative-control probes where pure spiro-only systems must keep their
  old successful parent choice while fused+spiro systems improve.

Deliverables:

- one spiro descriptor model,
- generic spiro renderer,
- component-spiro renderer,
- spiro stereo locant handoff.

Validation gate:

- P-24 tests,
- current component-spiro benchmark wins remain wins,
- active benchmark transition matrix with spiro family diagnostics,
- no net `spiro_only` regression on 001/004/006/007 sample guards.

Expected benchmark effect:

- positive on spiro systems with fused/bridged components,
- neutral or positive on pure spiro-only systems,
- fewer parse failures from malformed primed locants,
- clearer route to stereo improvements.

### Phase 8: P-31 Unsaturation, Hydro, Dehydro, And Indicated Hydrogen

Goal:

- make hydrogenation and unsaturation a parent-aware layer rather than a set of
  fused/polycyclic special renderers.

Primary code targets:

- new `nispo/nomenclature/unsaturation.py`
- `nispo/formatting.py`
- `nispo/polycyclic_rules.py`
- `nispo/ring_rules.py`
- `nispo/branch/ring_substituents.py`
- generated/hydro fused parent rules in `parent_rules.py`

Actions:

- Add `UnsaturationDescriptor`.
- Represent double/triple bond locants against the selected `LocantMap`.
- Prefer low locants for indicated hydrogen, suffixes, then unsaturation per
  P-14/P-31 ordering.
- Support:
  - saturated parents with ene/yne endings,
  - mancude parents with hydro/dehydro prefixes,
  - partially hydrogenated retained parents,
  - von Baeyer unsaturation with compound locants,
  - spiro unsaturation after the last bracket,
  - ring assembly unsaturation where needed.
- Keep current formatting helpers as renderers.

Deliverables:

- central unsaturation descriptor,
- hydro/dehydro renderer,
- tests for P-31 examples that OPSIN accepts.

Validation gate:

- formatting tests,
- ring/fused/spiro probes,
- active benchmark transition matrix with parse-failure churn monitored.

Expected benchmark effect:

- positive for partially saturated fused and spiro systems,
- reduced mismatch from wrong hydro/dehydro placement,
- possible parse-failure reduction.

### Phase 9: P-25 Fused And Bridged-Fused Construction

Goal:

- add systematic fused-system construction behind retained/generated resources.

Primary code targets:

- new `nispo/nomenclature/fused.py`
- `nispo/resource_templates.py`
- `nispo/template_store.py`
- `nispo/parent_rules.py`
- `nispo/branch/ring_substituents.py`

Actions:

- Add `FusedSystemDescriptor`.
- Decompose fused systems into component rings/ring systems.
- Represent fusion bonds, fusion atoms, peripheral atoms, interior atoms, and
  component orientation.
- Rank components using retained names, Hantzsch-Widman names, ring seniority,
  heteroatom priority, and P-25 fusion rules.
- Generate fusion descriptors where the decomposition is unambiguous.
- Keep retained and generated templates as preferred candidates.
- Use systematic fused construction only when resources do not provide a stable
  OPSIN-roundtripping name or when a systematic candidate has a better validated
  roundtrip.

Deliverables:

- fused descriptor model,
- component decomposition tests,
- conservative systematic fused renderer,
- resource-plus-systematic parent candidate ranking.

Validation gate:

- P-25 example tests,
- generated fused compaction tests,
- template store tests,
- active benchmark transition matrix with careful `match -> mismatch` review.

Expected benchmark effect:

- potentially large positive effect on the ring-system benchmark,
- highest risk phase because fused naming has many edge cases,
- should not begin until Phases 1-8 are stable.

### Phase 10: P-91/P-93 Parent-Aware Stereochemistry

Goal:

- generalize relative and absolute stereo rendering across rings and ring
  systems.

Primary code targets:

- `nispo/stereochemistry.py`
- new `nispo/nomenclature/stereo.py`
- `nispo/rule_engine.py`
- `nispo/spiro_component_rules.py`
- `nispo/branch/ring_substituents.py`

Actions:

- Add `StereoSiteDescriptor`.
- Compute stereogenic atoms, bonds, ring junctions, bridgeheads, and spiro
  centers independently from rendering.
- Map stereo sites to the selected parent locants.
- Support:
  - absolute `R/S`,
  - pseudoasymmetric `r/s`,
  - alkene `E/Z`,
  - relative ring `cis/trans`,
  - bridgehead `endo/exo/syn/anti` only where OPSIN-compatible and justified,
  - fused-ring junction stereo.
- Keep current late `apply_stereochemical_prefix` path as fallback.
- Prefer OPSIN roundtrip over strict PIN preference where the two diverge.

Deliverables:

- parent-aware stereo descriptor layer,
- migrated component-spiro stereo,
- migrated simple ring-substituent relative stereo,
- tests for P-93 examples that OPSIN accepts.

Validation gate:

- stereo-focused tests,
- current successful stereo benchmark cases remain matches,
- active benchmark transition matrix with stereo mismatches inspected.

Expected benchmark effect:

- positive on mismatch rows where graph connectivity is right but stereo is
  missing or mislocated,
- lower risk of narrow stereo rescue code.

### Phase 11: Branch, Prefix, And Ring-Substituent Consolidation

Goal:

- make substituent naming consume the same parent, locant, and characteristic
  group abstractions as main parent naming.

Primary code targets:

- `nispo/branch_rules.py`
- `nispo/branch/ring_substituents.py`
- `nispo/branch/functional_groups.py`
- `nispo/branch/carbon.py`
- `nispo/branch/hetero_atoms.py`
- `nispo/name_parts.py`

Actions:

- Make branch rules emit `SubstituentCandidateDescriptor` where useful.
- Use the characteristic-group registry for functional prefixes.
- Use ring-system descriptors for ring substituent names.
- Add a `RingAssemblyDescriptor` for independent rings connected by single
  bonds or linker chains.
- Treat unfused multi-ring molecules as component-selection problems:
  - choose the senior parent component,
  - render other ring components as substituent descriptors,
  - compare substituent multiplicative prefixes through `NameParts`,
  - preserve suffix-bearing characteristic groups on the correct component.
- Share locant maps for derived `yl`, `ylidene`, and `ylidyne` forms.
- Normalize use of `di` versus `bis` through `NameParts`/rendering helpers.
- Keep old branch functions as fallbacks during migration.

Deliverables:

- descriptor-aware branch candidates,
- shared functional prefix registry,
- fewer direct calls into ring and polycyclic producers from branch code.

Validation gate:

- branch smoke tests,
- fragment assembly tests,
- representative ring-substituent probes,
- active benchmark transition matrix,
- no net `multi_ring_unfused` regression on 001/004/007 sample guards.

Expected benchmark effect:

- positive on ring systems with substituent-rich side chains,
- positive on unfused multi-ring rows that are not addressed by fused/spiro
  engines,
- maintainability improvement more certain than immediate score gain.

### Phase 12: Performance And Candidate-Pruning Pass

Goal:

- convert generalization into speed rather than extra combinatorial cost.

Primary code targets:

- `nispo/naming_context.py`
- `nispo/template_index.py`
- `nispo/polycyclic_rules.py`
- `nispo/spiro_component_rules.py`
- new descriptor modules.

Actions:

- Profile candidate counts by family.
- Cache all descriptor construction in `MoleculeContext`.
- Avoid repeated RDKit substructure scans for the same query.
- Prune candidate generation by seniority:
  - do not enumerate lower-seniority parent families if a higher-seniority
    family has a validated candidate and no later suffix/stereo layer can
    change the decision,
  - cap bridge/spiro/fused decompositions by descriptor score,
  - prefer topology signatures before template matching.
- Keep OPSIN out of runtime naming.
- Batch OPSIN only in diagnostics/benchmark tooling.

Deliverables:

- profiling report,
- candidate count metrics,
- pruning rules documented with seniority rationale,
- speed-focused tests/probes where possible.

Validation gate:

- active benchmark remains at or above the accuracy target reached by earlier
  phases,
- speed below 20 ms/molecule/worker during accuracy work,
- after reaching 95% accuracy, optimize toward below 12 ms/molecule/worker.

Expected benchmark effect:

- accuracy should not drop,
- speed should improve if descriptor caches and pruning replace repeated local
  searches.

### Phase 13: Retire Legacy Policy And Reduce File Size

Goal:

- remove obsolete helper policy after descriptor-based rules prove stable.

Primary code targets:

- `nispo/parent_rules.py`
- `nispo/parent/rule_helpers.py`
- `nispo/polycyclic_rules.py`
- `nispo/spiro_component_rules.py`
- `nispo/ring_rules.py`

Actions:

- Delete or demote special gates that have descriptor equivalents.
- Remove duplicated family-specific locant logic.
- Move policy helpers into the relevant nomenclature modules.
- Keep only compatibility adapters needed for stable public behavior.
- Re-run `tools/architecture_debt_count.py` and expect reductions in:
  - `unregistered_name_producing_rule_surfaces`,
  - `rule_or_naming_paths_bypassing_central_registries`.

Deliverables:

- smaller `parent_rules.py`,
- smaller `parent/rule_helpers.py`,
- documented descriptor ownership,
- lower architecture debt count.

Validation gate:

- full focused test suite,
- active benchmark,
- no unexpected broad regression,
- architecture debt report improves.

Expected benchmark effect:

- mostly maintainability,
- possible speed improvement from less duplicated work.

## Phase Dependency Summary

The phases should not all run in strict linear isolation, but several
dependencies are important:

| Dependency | Reason |
| --- | --- |
| Phase 1 before all behavior changes | candidates need descriptors and traces before selection changes are safe |
| Phase 2 before P-23/P-24/P-25 | ring-system classes, locants, bridgeheads, and spiro atoms must be shared |
| Phase 3 before broad parent-priority changes | P-44 selection should be implemented once, not by another layer of priorities |
| Phase 4 before large ring/chain fixes | characteristic groups drive many ring-vs-chain decisions |
| Phase 5 before P-25 component construction | fused systems need reliable monocyclic component names |
| Phase 6 before P-7/P-9 spiro stereo work | von Baeyer descriptors supply bridgehead and locant maps |
| Phase 7 before component-spiro stereo work | spiro locants and priming must be stable |
| Phase 8 before aggressive fused/template pruning | hydro/dehydro and indicated-H can change which candidate roundtrips |
| Phase 10 after locants are stable | stereo without selected parent locants is brittle |
| Phase 12 after accuracy phases | speed optimization should not prune candidates before rule behavior is correct |

The first implementation milestone should be Phases 1-3. That milestone changes
almost no naming behavior but creates the machinery needed to make every later
rule change inspectable.

## Milestone Targets

### Milestone A: Descriptor Foundation

Includes:

- Phase 0,
- Phase 1,
- Phase 2,
- Phase 3 in diagnostic mode.

Target outcome:

- no meaningful benchmark score movement,
- no broad speed regression,
- parent candidates can be explained by descriptor and seniority trace,
- ring systems can be classified consistently.

Exit criteria:

- active benchmark has no unexpected broad `match -> mismatch` transition,
- descriptor tests cover all core topology classes,
- architecture debt count should not increase.

### Milestone B: Parent Selection And Functional Generalization

Includes:

- Phase 3 selection enablement,
- Phase 4.

Target outcome:

- exocyclic ring-parent logic becomes general,
- acyclic-vs-ring parent choice is traceable,
- several `empty_name` and `mismatch` rows improve without exact-SMILES logic.

Exit criteria:

- current successful exocyclic cases remain successful,
- active benchmark improves or stays neutral with no material match regression,
- speed remains below 20 ms/molecule/worker.

### Milestone C: Core Ring-System Rule Engines

Includes:

- Phase 5,
- Phase 6,
- Phase 7,
- Phase 8.

Target outcome:

- monocycles, von Baeyer systems, spiro systems, and unsaturation/hydrogenation
  all share locant and descriptor layers,
- fewer local branch-specific bridge/spiro/hydro fixes are needed,
- candidate generation is pruned by descriptor quality.

Exit criteria:

- active benchmark shows material `empty_name -> match` and `mismatch -> match`
  movement,
- no unacceptable `match -> mismatch` churn,
- speed remains below 20 ms/molecule/worker.

### Milestone D: Fused Systems And Stereo

Includes:

- Phase 9,
- Phase 10.

Target outcome:

- fused systems get a systematic fallback behind retained/generated templates,
- stereo is rendered against selected parent locants,
- mismatch failures from stereo and fused numbering decline.

Exit criteria:

- P-25 and P-93 focused tests pass,
- active benchmark reaches or approaches the 95% target,
- regressions are explainable by trace and are either fixed or accepted only
  when they are small and offset by broad gains.

### Milestone E: Consolidation And Speed

Includes:

- Phase 11,
- Phase 12,
- Phase 13.

Target outcome:

- branch/substituent rules consume the same descriptors as parent naming,
- candidate generation is faster,
- old policy helpers are retired,
- benchmark speed moves toward 12 ms/molecule/worker after accuracy target is
  reached.

Exit criteria:

- score remains at or above the achieved accuracy target,
- speed improves materially,
- architecture debt count drops below the current 31.

## Phase Definitions Of Done

This section is written for future goal runs. A phase is not done just because
code was added. It is done when the implementation, tests, benchmark evidence,
and documentation all support the same conclusion.

| Phase | Definition of done |
| --- | --- |
| Phase 0 | Baseline command recorded, 001/004/007/009 guard expectations known, representative probes exist, descriptor dump plan exists, no runtime behavior change |
| Phase 1 | `NameCandidate` can carry descriptors, old candidates still render identically, traces are available in diagnostics |
| Phase 2 | Ring systems are classified from cached descriptors, pure spiro/fused+spiro/unfused assemblies are distinct, old topology gates use descriptor predicates, descriptor tests cover core topologies |
| Phase 3 | Parent candidates have seniority vectors, traces explain legacy priority overrides, at least one low-risk family can be compared by vector, broad carbonyl-bearing parent choices are explainable |
| Phase 4 | Exocyclic functional parent decisions use `CharacteristicGroupCandidate`, old deferral helper is gone or reduced to compatibility, current exocyclic wins remain, amide/ester/carboxyl probes do not regress |
| Phase 5 | Monocycle naming has table-driven P-22 logic for common heterocycles, retained overrides are separate from grammar, Hantzsch-Widman tests pass |
| Phase 6 | Von Baeyer descriptors explicitly represent bridgeheads, main bridge, secondary bridges, locants, hetero replacement, and compatible rendering |
| Phase 7 | Generic spiro and component-spiro share a descriptor model, pure spiro and fused+spiro remain separate in selection, free/non-free spiro classification is explicit, primed locants are centralized |
| Phase 8 | Unsaturation/hydro/dehydro/indicated-H decisions are parent-aware and consumed by monocycle, von Baeyer, spiro, and fused candidates |
| Phase 9 | Fused descriptors exist, resources remain preferred, systematic fused fallback is conservative and covered by P-25 tests |
| Phase 10 | Stereo descriptors map through selected parent locants, at least simple ring, spiro, and fused junction cases use the shared layer |
| Phase 11 | Branch and ring-substituent naming can consume parent descriptors and characteristic-group records where applicable, unfused multi-ring assemblies have descriptor-aware parent/substituent selection |
| Phase 12 | Candidate counts and runtime hotspots are measured, pruning is justified by seniority or topology, speed improves without accuracy loss |
| Phase 13 | Legacy policy helpers are retired, architecture debt count falls, source files have clearer ownership boundaries |

For every behavior-changing phase, the minimum evidence is:

- focused tests for the changed rule family,
- selected molecule probes showing expected rendered names,
- benchmark transition matrix,
- speed comparison,
- written note of accepted regressions, if any.

## Full Repo Refactor Quality Model

The desired refactor is not just "more Blue Book rules". It should make the repo
easier to understand, safer to modify, and faster to run.

### Code Quality Standards

Every new module should:

- have one clear responsibility,
- avoid hidden global mutable state,
- prefer immutable dataclasses or plain tuples for descriptors,
- expose small public functions and keep helper functions private,
- use existing `MoleculeContext` caches instead of ad hoc molecule attributes,
- keep RDKit operations batched and reused where possible,
- keep string rendering separate from graph search,
- avoid broad exception swallowing in candidate producers,
- return no candidate rather than raising for unsupported chemistry,
- include tests or probes proportional to risk.

Every refactor should reduce at least one of these problems:

- duplicate topology checks,
- duplicate locant assignment,
- hidden parent-selection policy,
- string-first rendering before structural ranking,
- special-case gates in `parent_rules.py`,
- branch/ring/acyclic functional-group disagreement,
- repeated substructure scans,
- unregistered name-producing surfaces.

### Chemistry Quality Standards

Every new nomenclature rule should:

- name the Blue Book rule family it implements,
- state whether it targets preferred IUPAC behavior or OPSIN-compatible general
  nomenclature,
- prefer deterministic general behavior over local benchmark rescue,
- use RDKit graph facts for structural decisions,
- use local resources only as nomenclature resources,
- keep retained names and systematic names as separate candidate types,
- avoid silently changing stereochemistry,
- leave unsupported molecules unnamed rather than guessing.

### Performance Standards

During accuracy work:

- keep the active benchmark below 20 ms/molecule/worker unless the user accepts
  a temporary slowdown for a major accuracy step,
- measure candidate counts when adding graph enumeration,
- cache descriptor views in `MoleculeContext`,
- do not run OPSIN in runtime naming,
- batch OPSIN diagnostics,
- prefer topology signatures before expensive template or path enumeration.

After the 95% accuracy target is reached:

- focus on speed,
- target below 12 ms/molecule/worker,
- remove redundant fallback paths,
- prune lower-seniority candidate families earlier,
- profile before optimizing.

### Documentation Standards

When a future agent changes architecture, it should update one of:

- this audit document,
- a module docstring,
- a focused test name,
- a small diagnostics note.

Documentation should explain:

- what general rule was implemented,
- which Blue Book rule family motivated it,
- where behavior is intentionally OPSIN-compatible rather than strict PIN,
- how it was validated,
- what remains incomplete.

## Future Agent Handoff Protocol

A future agent should leave the repo in a state where the next agent can resume
without rediscovering context. At the end of each meaningful goal run, record:

- phase advanced,
- files changed,
- tests run,
- benchmark command run,
- benchmark summary,
- transition highlights,
- speed,
- open regressions,
- structural-family movements for touched families,
- next recommended phase,
- whether the audit should be updated.

If the run changes behavior but cannot run the active benchmark, the final note
must say why and must list the strongest focused validation that was run.

If the run only changes documentation, tests or benchmarks are not required, but
the final note should say that no runtime code changed.

## Recommended Work Slices For Goal Mode

Future goal-mode runs should avoid trying to complete an entire large phase in
one pass. These are good slices:

1. Add descriptor dataclasses and no behavior change.
2. Cache ring-system descriptors and migrate one gate.
3. Add seniority vectors in diagnostic-only mode.
4. Add the broad-regression guard probes for 001/004/007/009.
5. Convert one exocyclic group family to `CharacteristicGroupCandidate`.
6. Convert one simple monocycle family to the P-22 grammar.
7. Convert bicyclo descriptor generation to typed bridge records.
8. Convert simple monospiro generation to `SpiroDescriptor`.
9. Add `RingAssemblyDescriptor` for unfused multi-ring cases.
10. Move one hydro/dehydro decision into `UnsaturationDescriptor`.
11. Add descriptor-aware stereo for one parent family.
12. Retire one legacy helper after descriptor code covers it.

Each slice should be independently testable and should leave the repo in a
coherent state.

## Suggested Package Layout

A conservative layout would add a small internal namespace without moving
existing modules immediately:

```text
nispo/
  nomenclature/
    __init__.py
    candidates.py
    features.py
    fused.py
    functional_groups.py
    locants.py
    monocycles.py
    parent_selection.py
    ring_systems.py
    seniority.py
    spiro.py
    stereo.py
    substituents.py
    trace.py
    unsaturation.py
    von_baeyer.py
```

Existing modules would migrate gradually:

| Existing file | Future role |
| --- | --- |
| `rule_engine.py` | Keep as central engine; make candidates descriptor-aware |
| `rule_dispatch.py` | Keep rule registration/gating; gates consume descriptors |
| `naming_context.py` | Expand as cache for descriptors and feature registries |
| `structural_profile.py` | Fold into or feed `nomenclature/features.py` |
| `ring_topology.py` | Feed `nomenclature/ring_systems.py` |
| `ring_rules.py` | Become P-22 monocycle renderers/resources |
| `polycyclic_rules.py` | Become P-23 von Baeyer renderer over ring descriptors |
| `spiro_component_rules.py` | Become P-24 spiro/component-spiro renderer over ring descriptors |
| `parent_rules.py` | Shrink to rule registration and family adapters |
| `parent/rule_helpers.py` | Split into typed feature, locant, and selection helpers |
| `branch/functional_groups.py` | Become data-driven characteristic-group renderer |
| `acyclic/rules.py` | Consume characteristic-group registry and parent selector |
| `branch/ring_substituents.py` | Consume `substituents.py`, ring descriptors, and locant maps |

## Direct Answers To The User's Design Question

### Could The Recent Specific Feature Be Implemented Generally?

Yes. The successful substructure-specific feature should be recast as:

> When a ring system is the senior parent and a non-ring feature is a recognized
> exocyclic characteristic group or substituent, attach that feature to the ring
> parent through the characteristic-group registry unless P-44 seniority and the
> suffix hierarchy require an acyclic parent.

That rule can cover many scaffold families:

- monocyclic heterocycles with exocyclic carbonyl/thione/imine groups,
- fused heterocycles with exocyclic oxo/thioxo/imino groups,
- bridged systems with terminal functional groups,
- spiro systems with exocyclic functional groups,
- partially saturated retained parents with exocyclic suffix/prefix treatment.

The current targeted implementation should be preserved as proof that the rule
helps, but the long-term code should not ask "is this exact substructure one of
the benchmark rescues?" It should ask:

- what is the senior parent candidate?
- is the non-ring feature a characteristic group?
- is it principal, suffix, prefix, or replacement-compatible?
- what locant does it receive on the chosen parent?
- does OPSIN roundtrip the rendered result?

### Would This Be Beneficial?

Yes, with staged validation.

Accuracy benefit:

- current failures are dominated by no-name and mismatch cases, which often
  indicate missing parent families, wrong parent priority, or incomplete
  locant/stereo handling,
- general P-22/P-23/P-24/P-25/P-44 rules target exactly those failure modes,
- resources can still provide retained names, but general rules will catch
  scaffolds outside the retained-resource surface.

Speed benefit:

- shared descriptors reduce repeated topology and substructure checks,
- seniority vectors can prune low-value candidates earlier,
- a single feature registry avoids repeated functional-group scans,
- better gates reduce work in expensive polycyclic/spiro paths.

Maintainability benefit:

- benchmark improvements become reusable nomenclature rules,
- failures can be grouped by missing rule area,
- future agents can inspect decision traces instead of reverse-engineering
  nested helper logic.

Risk:

- a naive general-rule implementation could be slower,
- strict preferred-IUPAC behavior may not always be the best OPSIN roundtrip
  behavior,
- stereochemistry can silently become worse if locants are wrong,
- broad changes can create match-to-mismatch regressions.

The correct compromise is to implement Blue Book rules as general deterministic
NISPO rules, while continuing to use OPSIN roundtrip as the oracle.

## Validation Protocol

Every meaningful phase should be validated by:

1. Focused unit tests for the changed family.
2. Small probes for representative benchmark failures.
3. The exact active benchmark command from
   `benchmark/curriculum_state.json`.
4. Transition-matrix review, not just score review.

For general refactors, the active benchmark is necessary but not sufficient.
The 001-009 sweep showed that a change can help 009 or fused/spiro sets while
regressing broad druglike molecules. Use these structural guardrails when the
change touches candidate ranking, parent selection, characteristic groups,
spiro logic, fused logic, or branch/ring substituents:

| Guard | What it protects |
| --- | --- |
| `001` sample or full run | small broad-druglike molecules that were already high-scoring |
| `004` sample or full run | large functionalized heteroaromatic molecules |
| `007` sample or full run | broad cyclic population |
| `009` active/full run | bare ring-system coverage and high ring-count gaps |

Class-level movements to inspect:

- `fused_and_spiro`: should retain the large gains.
- `spiro_only`: should not regress when fused+spiro improves.
- `multi_ring_unfused`: should not be routed through fused/spiro logic.
- `amide`, `carboxyl_or_ester`, and `sulfonyl`: should not lose correct parent
  selection when a ring candidate is introduced.
- `rings_5`, `rings_6`, and `rings_7_plus`: should move toward generated
  high-ring-count coverage without depending only on retained names.
- `opsin_parse_failed`: should not increase simply because more malformed names
  are emitted.

Watch these transitions separately:

- `empty_name -> match`
- `mismatch -> match`
- `opsin_parse_failed -> match`
- `match -> mismatch`
- `match -> opsin_parse_failed`
- `match -> empty_name`

Speed should be tracked as:

- milliseconds per molecule per worker,
- candidate counts per family,
- time spent in topology/feature/profile construction,
- time spent in ring-system producers.

## What Not To Do

Do not:

- add exact-SMILES lookup tables,
- use benchmark row labels or generated answers as runtime data,
- depend on OPSIN internals at runtime,
- replace resources with benchmark-specific memorization,
- rewrite the whole engine in one branch,
- optimize score by changing benchmark/scorer behavior,
- run OPSIN once per candidate during normal naming.

Do:

- adapt stable nomenclature tables into deterministic NISPO resources,
- use OPSIN only as parse/roundtrip oracle in diagnostics and benchmark runs,
- keep public API compatibility,
- keep `smiles_to_iupac_batch` returning exactly one output per input,
- return `""` for unsupported molecules rather than aborting.

## Immediate High-Value Work Items

1. Add `ParentCandidateDescriptor` and `SeniorityVector` while preserving current
   output.
2. Add a cached `RingSystemDescriptor` in `NamingContext`.
3. Add explicit structural guard probes for the observed regression families:
   - small broad-druglike `001` ring regressions,
   - `004` fused heteroaromatic amides and esters,
   - pure spiro-only systems,
   - unfused multi-ring assemblies,
   - `009` high ring-count ring systems.
4. Convert the current exocyclic deferral rule into a
   `CharacteristicGroupCandidate` decision.
5. Move the relative bicyclic `cis` fallback toward a parent-aware
   `StereoSiteDescriptor` layer.
6. Type the current polycyclic bridge records and expose bridgehead/main-bridge
   choices explicitly.
7. Replace string family selectors for polycyclic records with enums or
   descriptor predicates.
8. Add debug traces that explain why a parent candidate was chosen or rejected.
9. Add `RingAssemblyDescriptor` for independent multi-ring systems before
   routing them through branch/substituent rendering.
10. Add Blue Book example tests for P-22, P-23, P-24, P-25, P-44, and P-93 cases
   that OPSIN can parse.

## Recommended First Implementation Slice

The best first slice is Phase 1 plus the smallest useful part of Phase 2 and a
small Phase 0 guardrail update. It should be small and low-risk:

1. Create `nispo/nomenclature/candidates.py` with descriptor dataclasses.
2. Create `nispo/nomenclature/trace.py` with diagnostic trace dataclasses.
3. Add optional descriptor and trace fields to `NameCandidate`.
4. Create `nispo/nomenclature/ring_systems.py` as an adapter around current
   `ring_topology.py`.
5. Create `nispo/nomenclature/locants.py` with a minimal `LocantMap`.
6. Add enum values for monocycle, unfused ring assembly, fused, bridged,
   spiro-only, component-spiro, and fused+spiro.
7. Cache ring descriptors in `NamingContext`.
8. Populate descriptors for existing ring-parent candidates without changing
   rendered names.
9. Add a diagnostic tool that prints descriptors for selected SMILES.
10. Add a tiny structural guard probe list covering the 001/004/007/009
    families above.
11. Run focused tests and the active benchmark to verify no behavior change.

Only after this should ranking or rendering behavior change.

This slice is intentionally not about improving score. Its purpose is to make
future score-improving work safer. A small temporary speed cost is acceptable if
descriptor creation is lazy and the active benchmark remains below the project
speed guardrail.

The second implementation slice should be Phase 3 in diagnostic-only mode. It
should compute seniority vectors and traces for all candidates but keep legacy
selection unchanged. Only after comparing traces for 001/004 regressions and
005/006/007 fused+spiro gains should the engine start selecting by vector.

## Bottom Line

The repo is ready for a Blue Book driven refactor. The current engine has enough
structure to support it, and the benchmark suite is now broad enough to show
which generalizations are real. The active 009 set exposes the hard ring-system
gap, while 001 and 004 expose broad-druglike regressions that a ring-only plan
would miss.

The highest-value change is not a new single naming rule. It is to separate
"what candidate parent and characteristic groups exist?" from "which candidate
is senior?" and from "how do I render the final string?" Blue Book seniority,
topology, locants, characteristic groups, unsaturation, and stereochemistry
should act as shared layers consumed by every family.

The general solution is:

1. make all producers emit typed descriptors,
2. compare descriptors with one seniority vector,
3. render only the selected descriptor,
4. reject locally malformed names before output,
5. validate against both active ring systems and broad regression guards.

Done carefully, this should improve both accuracy and speed. Done as a broad
rewrite, it will likely regress. The correct path is an incremental descriptor
and seniority-vector migration, with OPSIN roundtrip and transition matrices as
the guardrails.
