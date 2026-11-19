"""Bounded parent and branch constructions used by the independent fast API."""

import re
from functools import lru_cache


def principal_acid_name(mol, Chem, query):
    """Name a bounded neutral acid using the complete principal-site comparison."""
    matches = mol.GetSubstructMatches(query, maxMatches=3)
    if not 1 <= len(matches) <= 2:
        return ''
    if len(Chem.GetMolFrags(mol)) != 1 or any(a.GetFormalCharge() for a in mol.GetAtoms()):
        return ''
    owners = {match[0] for match in matches}
    centers = {match[1] for match in matches}
    if any(not mol.GetAtomWithIdx(owner).IsInRing() for owner in owners):
        from .mapped_monoacid_chain import mapped_monoacid_chain_decision

        decision = mapped_monoacid_chain_decision(mol, Chem)
        if decision is None:
            from .mapped_principal_general import mapped_principal_decision

            decision = mapped_principal_decision(mol, Chem)
        if (decision is not None and decision.principal_centers and decision.principal_centers <= centers
                and decision.name.endswith(' acid')):
            return decision.name
        if len(matches) == 1 and mol.HasSubstructMatch(_annular_thioamide_query(Chem)):
            return _annular_thioamide_acid(mol, Chem, centers)
        return ''
    from .extended_inventory import selection_inventory
    from .mapped_ring_parents import ring_systems

    inventory = selection_inventory(mol)
    groups = inventory.principal
    if (not inventory.supported or inventory.source_frontier.status not in {'selected', 'tie'}
            or any(group.kind != 'acid' for group in groups) or {g.center for g in groups} != centers):
        return ''
    core = next((core for core in ring_systems(mol) if owners <= core), None)
    if core is None or len(core) > (14 if len(centers) == 2 else 10):
        return ''
    return _mapped_ring_name(mol, Chem, groups, core)


@lru_cache(maxsize=1)
def _annular_thioamide_query(Chem):
    return Chem.MolFromSmarts('[N;R]@[C;R](=[S;D1;+0])')


def _annular_thioamide_acid(mol, Chem, centers):
    from .extended_inventory import selection_inventory
    from .principal_carboxyl import acyclic_acid_record, principal_carboxyl_group

    if (selection_inventory(mol).supported or mol.GetStereoGroups()
            or any(a.GetChiralTag() or a.HasQuery() for a in mol.GetAtoms())
            or any(b.GetStereo() or b.HasQuery() for b in mol.GetBonds())):
        return ''
    group = principal_carboxyl_group(mol)
    if group and {group[0]} == centers:
        record = acyclic_acid_record(mol, Chem, group)
        if not record.preference_issues and 'thioxo' in record.name:
            return record.name
    return ''


def _mapped_ring_name(mol, Chem, groups, core=None):
    from .mapped_ring_parents import records

    scope = frozenset(range(mol.GetNumAtoms()))
    rows = records(mol, Chem, scope=scope, core_filter=core, principal=groups,
                   general_branches=True)
    centers = {g.center for g in groups}
    names = {row.name for row in rows if row.principal_centers == centers
             and row.hydrogen_budget_proved and row.annotation_coverage
             and row.construction is not None
             and row.construction.atoms | row.construction.embedded_atoms == scope}
    return names.pop() if len(names) == 1 else ''


# These vocabulary signals restrict the extra work to names emitted by the
# fast ring paths in question. The replacement is constructed from atoms,
# including its suffix and annotations, independently of those signals.
_RING_SIGNALS = re.compile(
    r'tetralon|naphthoquinon|epoxybenzen|indolizidin|'
    r'imidazo\[4,5-d\]pyridin|pyrido\[2,3-d\]pyrimidin|'
    r'oxo-\d+,\d+-dihydropyrrolo\[3,2-c\]pyridin|oxo-1H-1,2,4-triazol|'
    r'7-aza-8-oxabicyclo\[4\.3\.0\]nonan'
)
_BENZOXAZIN_YL = re.compile(r'3,4-dihydro-2H-1,4-benzoxazin-\d+-yl')
_CYCLOPENTAPYRIDINE = '6,7-dihydro-cyclopenta[b]pyridine'
_FUSED_CARBONYL_ATTACHMENT = re.compile(r'-(?:one|thione)-\d+[a-z]?\'*-yl\b')


@lru_cache(maxsize=1)
def _triketone_query(Chem):
    ketone = '[C;X3;H0;!R](=[O;D1;+0])'
    return Chem.MolFromSmarts(f'{ketone}-{ketone}-{ketone}')


@lru_cache(maxsize=1)
def _ester_query(Chem):
    return Chem.MolFromSmarts('[#6,#7]-[CX3;!R](=[OX1])-[O;X2;H0;+0]-[#6]')


def bounded_ester_name(mol, Chem, *, allow_fused_aryl_carbamate=False):
    """Use the mapped writer for small closed ester and carbamate scopes."""
    matches = mol.GetSubstructMatches(_ester_query(Chem), maxMatches=3)
    if not 1 <= len(matches) <= 2:
        return ''
    if len(matches) == 1 and not allow_fused_aryl_carbamate:
        host = mol.GetAtomWithIdx(matches[0][0])
        if host.GetAtomicNum() == 7 and not host.IsInRing():
            aromatic_hosts = {a.GetIdx() for a in host.GetNeighbors() if a.GetIsAromatic()}
            if aromatic_hosts:
                rings = tuple(map(set, mol.GetRingInfo().AtomRings()))
                # A fused aryl N-ligand would load the full template bank.
                # Its explicit repair path can opt in after seeing the name.
                if any(aryl & aromatic_hosts and any(aryl & other for other in rings if aryl != other)
                       for aryl in rings):
                    return ''
    mixed_classes = False
    if len(matches) == 2:
        owners = {match[0] for match in matches}
        owner_elements = {mol.GetAtomWithIdx(i).GetAtomicNum() for i in owners}
        mixed_classes = owner_elements == {6, 7}
        if not mixed_classes and (
                owner_elements != {6}
                or any(mol.GetAtomWithIdx(arm).GetTotalNumHs() != 3 for _, _, _, _, arm in matches)
                or not any(owners <= set(ring) for ring in mol.GetRingInfo().AtomRings())):
            return ''
    from .mapped_ester_general import mapped_ester_decision

    decision = mapped_ester_decision(mol, Chem)
    centers = {match[1] for match in matches}
    if mixed_classes and decision is not None:
        carbon_centers = {center for host, center, *_ in matches
                          if mol.GetAtomWithIdx(host).GetAtomicNum() == 6}
        if decision.principal_centers == carbon_centers:
            return decision.name
    return decision.name if decision is not None and decision.principal_centers == centers else ''


def _unproved_compound_sulfanyl_branch(mol):
    # P-16.5.1.1 requires enclosures for a compound sulfanyl prefix. The
    # extra direct consumers below do not yet carry that prefix-nature
    # evidence through every nested branch. Preserve full delivery for a
    # neutral acyclic divalent S rather than infer presentation from text.
    return any(atom.GetAtomicNum() == 16 and not atom.IsInRing()
               and not atom.GetFormalCharge() and atom.GetDegree() == 2
               and atom.GetTotalValence() == 2
               and all(bond.GetBondTypeAsDouble() == 1 for bond in atom.GetBonds())
               for atom in mol.GetAtoms())


def bounded_fused_carbonyl_name(mol, Chem):
    """Build a complete numbered parent before declining the independent route."""
    from .extended_inventory import selection_inventory
    from .ring_state_parent import _SUPPORTED_GROUPS

    inventory = selection_inventory(mol)
    groups = inventory.principal
    if (inventory.supported and groups and not _unproved_compound_sulfanyl_branch(mol)
            and all(group.kind in {'ester', 'carbamate'} for group in groups)):
        from .mapped_ester_general import mapped_ester_decision

        decision = mapped_ester_decision(mol, Chem, require_parent_proof=True)
        if (decision is not None and decision.principal_centers == {group.center for group in groups}
                and decision.atoms == frozenset(range(mol.GetNumAtoms()))):
            return decision.name
    # Explicit principal groups must have a suffix writer. Atom coverage alone
    # cannot establish that an unsupported acid derivative was rendered.
    if not groups or any(group.kind not in _SUPPORTED_GROUPS for group in groups):
        return ''
    complete = _mapped_ring_name(mol, Chem, groups)
    if complete:
        return complete
    return _proved_carbonyl_subset_name(mol, Chem, inventory)


def _proved_carbonyl_subset_name(mol, Chem, inventory):
    """Discharge every competing source parent with optimistic graph bounds.

    A connected ring parent cannot span separate ring systems. An acyclic
    parent cannot cross a ring; its connected nonring component bounds the
    number of citable sites. Heteroatomic chains stay barriers at a tied site
    count. Strict P-44 dominance is required before a missing renderer can be
    ignored. No name text, corpus outcome or size threshold ranks parents.
    """
    from .mapped_ring_parents import records, parent_facts, ring_systems
    from .parent_selection import parent_frontier
    from .proposal_plans import graph_proposal_bound, strict_graph_dominance

    groups = inventory.principal
    if (not inventory.supported or len(groups) < 2
            or _unproved_compound_sulfanyl_branch(mol)
            or inventory.source_frontier.status == 'barrier'
            or any(group.kind not in {'ketone', 'pseudoketone', 'lactam', 'lactone'} for group in groups)
            or len({group.seniority for group in groups}) != 1):
        return ''
    scope = frozenset(range(mol.GetNumAtoms()))
    centers = {group.center for group in groups}
    systems = ring_systems(mol)
    if sum(bool(centers.intersection(core)) for core in systems) != 1:
        # Multiple principal-bearing rings can admit a ring assembly or a
        # multiplied functional parent. Their P-51 proof is not supplied by
        # a comparison of separate single-framework ring bounds.
        return ''
    rows = []
    for row in records(mol, Chem, scope=scope, principal=groups, general_branches=True):
        core = frozenset(row.parent_atoms)
        if (core not in systems or row.principal_centers != centers.intersection(core)
                or not row.principal_centers or not row.hydrogen_budget_proved
                or not row.annotation_coverage or row.construction is None
                or row.construction.atoms | row.construction.embedded_atoms != scope):
            continue
        facts = parent_facts(mol, row, groups[0].seniority)
        if facts is not None:
            rows.append((row, facts))
    if not rows:
        return ''
    rows = parent_frontier(rows, lambda option: option[1])
    if len({row.name for row, _facts in rows}) != 1:
        return ''
    selected, facts = rows[0]
    selected_core = frozenset(selected.parent_atoms)
    for core in systems:
        local = tuple(group for group in groups if group.center in core)
        if not local or core == selected_core:
            continue
        if strict_graph_dominance(facts, graph_proposal_bound(mol, core, local, cyclic=True)) is None:
            return ''
    # Keep every potentially skeletal nonring heteroatom in the upper bound.
    # Only terminal carbonyl O and halogens/H cannot extend this parent chain.
    remaining = {atom.GetIdx() for atom in mol.GetAtoms()
                 if not atom.IsInRing() and atom.GetAtomicNum() not in {1, 9, 17, 35, 53}
                 and not (atom.GetAtomicNum() == 8 and atom.GetDegree() == 1
                          and atom.GetBonds()[0].GetBondTypeAsDouble() == 2
                          and atom.GetNeighbors()[0].GetAtomicNum() == 6)}
    while remaining:
        component, stack = set(), [next(iter(remaining))]
        while stack:
            idx = stack.pop()
            if idx not in remaining:
                continue
            remaining.remove(idx)
            component.add(idx)
            stack.extend(n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                         if n.GetIdx() in remaining)
        local = tuple(group for group in groups if group.center in component)
        if not local:
            continue
        bound = graph_proposal_bound(mol, component, local)
        if len(local) == len(selected.principal_centers) and any(
                mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in component):
            return ''
        if strict_graph_dominance(facts, bound) is None:
            return ''
    return selected.name


def repaired_parent_name(mol, Chem, name):
    if 'aziridin-1-yl' in name and mol.GetNumHeavyAtoms() <= 52 and any(
            a.GetAtomicNum() == 7 and a.GetChiralTag() for a in mol.GetAtoms()):
        from .mapped_monoacid_chain import mapped_monoacid_chain_decision

        decision = mapped_monoacid_chain_decision(mol, Chem)
        if decision is not None:
            return decision.name
    if 'azaniumyl' in name and mol.GetNumHeavyAtoms() == 25 and mol.GetRingInfo().NumRings() == 4:
        from .bluebook_lambda_amine_methods import aryl_aminium_decision

        decision = aryl_aminium_decision(mol, Chem)
        if decision is not None:
            return decision.name
    if 'dione' in name and mol.GetNumHeavyAtoms() <= 40:
        match = mol.GetSubstructMatch(_triketone_query(Chem))
        if match:
            from .mapped_principal_general import mapped_principal_decision

            decision = mapped_principal_decision(mol, Chem)
            if decision is not None and decision.principal_centers == set(match[::2]):
                return decision.name
    if 'perhydropyrazino[' in name and '(formyl)' in name and mol.GetNumHeavyAtoms() <= 48:
        from .mapped_principal_general import mapped_principal_decision

        decision = mapped_principal_decision(mol, Chem)
        if decision is not None and decision.name.endswith('aldehyde'):
            return decision.name
    if ('pyrimido[1,2-b]pyrazol' in name and 'carbonyl' in name
            and mol.GetNumHeavyAtoms() <= 40):
        from .hidden_amide_pseudoketones import hidden_amide_pseudoketone_decision

        decision = hidden_amide_pseudoketone_decision(mol, Chem)
        if decision is not None:
            return decision.name
    if ('pyrrolo[3,2-c]pyridin' in name and 'hydropyrrolo' not in name
            and mol.GetNumHeavyAtoms() <= 22 and mol.GetRingInfo().NumRings() == 2
            and mol.HasSubstructMatch(_saturated_ring_nitrogen_query(Chem))):
        from .mapped_ring_parents import heteropolycyclic_hydride

        record = heteropolycyclic_hydride(mol, Chem)
        if record is not None and record.hydrogen_budget_proved and record.annotation_coverage:
            return record.name
    if 'hydrazinyl' in name and 'carboxy' in name and not name.endswith(' acid'):
        from .hydrazone_methods import hydrazono_acid_decision

        decision = hydrazono_acid_decision(mol, Chem)
        if decision is not None:
            return decision.name
    if 'spirocyclo' in name and 'thia' in name:
        from .spiro_component_rules import spiro_sulfamate_carbamate

        decision = spiro_sulfamate_carbamate(mol, Chem)
        if decision is not None:
            return decision.name
    if 'dioxo-' in name and 'bicyclo[' in name:
        from .relative_cycle_evidence import fused_ring_junction_evidence

        rings = mol.GetRingInfo().AtomRings()
        if len(rings) == 2 and fused_ring_junction_evidence(mol, Chem, set(rings[0]) | set(rings[1])):
            from .mapped_principal_general import mapped_principal_decision

            decision = mapped_principal_decision(mol, Chem)
            if decision is not None and decision.name.startswith(('(cis)-', '(trans)-')):
                return decision.name
    if 'benzoxazin' in name:
        match = _BENZOXAZIN_YL.search(name)
        if match is not None:
            replacement = _benzoxazin_yl(mol, Chem, match)
            if replacement:
                return replacement
    if _CYCLOPENTAPYRIDINE in name:
        replacement = _cyclopentapyridine_hydride(mol, Chem, name)
        if replacement:
            return replacement
    carbonyl_attachment = bool('one-' in name and _FUSED_CARBONYL_ATTACHMENT.search(name))
    if ('penam-' in name or 'naphthoquinon' in name or 'benzazepan' in name and 'dione' in name
            or 'carbonyloxy)propane' in name and 'tetrahydropyrido[' in name
            or carbonyl_attachment and ' ' in name and name.endswith('ate')):
        ester = bounded_ester_name(mol, Chem, allow_fused_aryl_carbamate=True)
        if ester:
            return ester
    if ('cephem' in name or 'norpinene' in name or 'pyrimido[5,4-d]pyrazol' in name and 'amino' in name
            or 'oxadiazol' in name and '5-oxo-1,2,4-oxadiazol' in name
            or carbonyl_attachment):
        from .mapped_principal_general import mapped_principal_decision

        if mol.GetNumHeavyAtoms() <= 40:
            decision = mapped_principal_decision(mol, Chem)
            if decision is not None:
                return decision.name
    if 'spiro[' in name and 'pyrrolo' in name and '-one]' in name:
        from .naming_context import MoleculeContext
        from .spiro_component_rules import component_spiro_parent_candidates

        candidates = component_spiro_parent_candidates(MoleculeContext(mol, Chem), 0)
        if len(candidates) == 1 and candidates[0].render_rank == 0:
            return candidates[0].name
    if '-imidazolin]' in name:
        repaired = _imidazoline_spiro_name(mol, Chem, name)
        if repaired:
            return repaired
    if 'phospha' in name and 'thiaethen' in name:
        from .phosphorus_thioester_parent import anionic_dithiophosphate_name

        return anionic_dithiophosphate_name(mol, Chem)
    stacked_amide = ('amino)carbonyl' in name or name.startswith('1-oxo-1-')
                     and name.endswith(('amino)methane', 'amino)-methane')))
    if stacked_amide and mol.GetNumHeavyAtoms() <= 30:
        repaired = _principal_amide_or_carbamate(mol, Chem)
        if repaired:
            return repaired
    if 'carboxy-' in name and '1-oxaethyl)imino' in name and mol.GetNumHeavyAtoms() <= 32:
        from .functional_inventory import _neutral_carboxyl_query

        repaired = principal_acid_name(mol, Chem, _neutral_carboxyl_query(Chem))
        if repaired:
            return repaired
    aldehyde = '(formyl)' in name
    if len(name) > 300:
        return ''
    ring_repair = _RING_SIGNALS.search(name) or 'anilin' in name and _has_ring_yne(mol)
    if not ring_repair and (not aldehyde or mol.GetRingInfo().NumRings() > (3 if 'spiro[' in name else 2)):
        return ''
    if mol.GetNumHeavyAtoms() > 48:
        return ''
    from .extended_inventory import selection_inventory

    inventory = selection_inventory(mol)
    if not inventory.supported or not inventory.principal:
        return ''
    if not ring_repair and any(g.kind != 'aldehyde' for g in inventory.principal):
        return ''
    return _mapped_ring_name(mol, Chem, inventory.principal)


def _principal_amide_or_carbamate(mol, Chem):
    from .extended_inventory import selection_inventory

    inventory = selection_inventory(mol)
    groups = inventory.principal
    if not inventory.supported or len(groups) != 1 or groups[0].kind not in {'amide', 'carbamate'}:
        return ''
    if groups[0].kind == 'carbamate':
        from .mapped_ester_general import mapped_ester_decision

        # A direct ring N-carbamate can reuse its numbered ring. Acyclic
        # carbamates may require a separate search over complete parent chains.
        if not any(mol.GetAtomWithIdx(i).GetAtomicNum() == 7 and mol.GetAtomWithIdx(i).IsInRing()
                   for i in groups[0].atoms):
            return ''
        decision = mapped_ester_decision(mol, Chem)
        return decision.name if decision is not None and decision.principal_centers == {groups[0].center} else ''
    if (inventory.source_frontier.status != 'selected' or groups[0].owner < 0
            or not mol.GetAtomWithIdx(groups[0].owner).IsInRing()):
        return ''
    from .mapped_ring_parents import ring_systems

    core = next((ring for ring in ring_systems(mol) if groups[0].owner in ring), None)
    if core is None or len(core) > 10:
        return ''
    return _mapped_ring_name(mol, Chem, groups, core)


def _has_ring_yne(mol):
    return any(b.GetBondTypeAsDouble() == 3 and b.IsInRing() for b in mol.GetBonds())


@lru_cache(maxsize=1)
def _saturated_ring_nitrogen_query(Chem):
    return Chem.MolFromSmarts('[N;R;+0;X3]')


def _cyclopentapyridine_hydride(mol, Chem, name):
    from .mapped_ring_parents import _core_graph, ring_systems

    if name.count(_CYCLOPENTAPYRIDINE) != 1:
        return ''
    cores = [ring for ring in ring_systems(mol) if len(ring) == 9]
    if len(cores) != 1:
        return ''
    prepared = _core_graph(mol, Chem, cores[0])
    if prepared is None:
        return ''
    graph, _ = prepared
    # Require the same fused topology and the specific C5 methylene state.
    if Chem.MolToSmiles(graph) != _cyclopentapyridine_graph(Chem):
        return ''
    return name.replace(_CYCLOPENTAPYRIDINE, '5H-cyclopenta[b]pyridine')


@lru_cache(maxsize=1)
def _cyclopentapyridine_graph(Chem):
    return Chem.MolToSmiles(Chem.MolFromSmiles('c1cc2c(nc1)C=CC2'))


def _benzoxazin_yl(mol, Chem, match):
    from .branch_rules import collect_subtree
    from .component_mapping import _complete_rooted_ring
    from .mapped_ring_parents import ring_systems

    cores = [core for core in ring_systems(mol) if len(core) == 10]
    if len(cores) != 1 or len(_BENZOXAZIN_YL.findall(match.string)) != 1:
        return ''
    core = cores[0]
    text = match.string
    branches = set()
    for root in core:
        for atom in mol.GetAtomWithIdx(root).GetNeighbors():
            host = atom.GetIdx()
            if host in core or atom.GetAtomicNum() != 6:
                continue
            scope = frozenset(collect_subtree(mol, root, host))
            if scope != core:
                continue
            branch = _complete_rooted_ring(mol, Chem, root, scope)
            if branch.startswith('3,4-dihydro-2H-1,4-benzoxazin-'):
                branches.add(branch)
    return text[:match.start()] + branches.pop() + text[match.end():] if len(branches) == 1 else ''


def _imidazoline_spiro_name(mol, Chem, name):
    from .mapped_ring_parents import _core_graph, ring_systems
    from .spiro_component_rules import (
        _proved_hydride_component_names, _submol_for_atoms, _two_component_splits_by_articulation,
        ring_system_graph,
    )

    # The contracted fast-engine word leaves the position of the double bond
    # implicit. Re-render that component from its capped graph and map.
    cores = [core for core in ring_systems(mol) if len(core) >= 7]
    if len(cores) != 1 or name.count('-imidazolin]') != 1:
        return ''
    core = cores[0]
    prepared = _core_graph(mol, Chem, core)
    if prepared is None:
        return ''
    graph, _ = prepared
    atoms = frozenset(range(graph.GetNumAtoms()))
    adjacency = ring_system_graph(graph, Chem, atoms)
    splits = _two_component_splits_by_articulation(adjacency, set(atoms))
    if len(splits) != 1:
        return ''
    junction, sides = next(iter(splits.items()))
    options = set()
    for side in sides:
        if len(side) != 4 or sorted(graph.GetAtomWithIdx(i).GetAtomicNum() for i in side) != [6, 6, 7, 7]:
            continue
        sub = _submol_for_atoms(graph, Chem, side | {junction})
        if sub is None:
            continue
        fragment, _, mapping = sub
        for row in _proved_hydride_component_names(fragment, Chem, mapping[junction]):
            if (row.junction_proved and row.parent_name == '4,5-dihydro-1H-imidazole'
                    and f",{row.spiro_locant}'-imidazolin]" in name):
                options.add(row.parent_name)
    if len(options) != 1:
        return ''
    return name.replace('-imidazolin]', '-' + options.pop() + ']')


