"""Reuse numbered polycyclic hydrides in complete original-graph constructions."""

from .component_mapping import (
    MappedComponent, _general_branch_stereo_is_cited, decorate_component,
    mapped_general_branch,
)
from .formatting import attach_suffix
from .molecule_cache import mol_cache_dict, mol_cache_set
from .naming_methods import MethodDecision
from .naming_result import SelectionLimitation


_RULE = 'P-23.2/P-25/P-29/P-41'


def _hydride_numberings(mol, Chem, core):
    """Lift an already numbered, hydrogen-capped core back to source atoms."""
    cache = mol_cache_dict(mol, 'numbered_polycycle_hydrides')
    if core not in cache:
        cache[core] = _build_hydride_numberings(mol, Chem, core)
    return cache[core]


def _build_hydride_numberings(mol, Chem, core):
    from .mapped_ring_parents import _core_graph
    from .polycyclic_rules import (
        _kekulized_polycyclic_view, _secondary_bridge_polycyclic_candidate_records,
    )
    from .primitive_three_face_fusion import general_component_fusion

    prepared = _core_graph(mol, Chem, core)
    if prepared is None:
        return ()
    capped, source = prepared
    local_core = frozenset(range(len(source)))
    fused = general_component_fusion(capped, Chem, include_locants=True)
    if fused is not None and {idx for idx, _ in fused.atom_locants} == local_core:
        return ((fused.name, tuple((source[i], loc) for i, loc in fused.atom_locants)),)
    view = _kekulized_polycyclic_view(capped, Chem)
    rows = _secondary_bridge_polycyclic_candidate_records(
        view, Chem, local_core, decorate_parent=False)
    return tuple(
        (name, tuple((source[i], str(loc)) for loc, i in enumerate(sequence, 1)))
        for name, sequence, unsaturation, topology in rows
        if not unsaturation and not topology
    )


def numbered_polycycle_component(mol, Chem, core, scope, *, attachment=None,
                                 carbonyl=None, sulfur_oxo=()):
    """Build a parent or yl prefix from the core and disjoint original branches."""
    if not core <= scope:
        return None
    cut = None
    if attachment is not None:
        outside = [b for b in mol.GetAtomWithIdx(attachment).GetBonds()
                   if b.GetOtherAtomIdx(attachment) not in scope]
        if (len(outside) != 1 or outside[0].GetBondTypeAsDouble() != 1
                or outside[0].GetStereo()):
            return None
        cut = (attachment, outside[0].GetOtherAtomIdx(attachment))

    occupied, branches = set(core), []
    for host in sorted(core):
        for bond in mol.GetAtomWithIdx(host).GetBonds():
            root = bond.GetOtherAtomIdx(host)
            edge = (host, root)
            if root in core or edge == cut:
                continue
            if root not in scope or bond.GetStereo():
                return None
            if edge == carbonyl:
                occupied.add(root)
                continue
            if edge in sulfur_oxo:
                word, owned = 'oxo', frozenset({root})
            elif bond.GetBondTypeAsDouble() == 1:
                word, owned = mapped_general_branch(mol, Chem, root, host)
            else:
                return None
            if (not word or root not in owned or owned & occupied or not owned <= scope
                    or any(c.isspace() for c in word)):
                return None
            branches.append((host, word))
            occupied.update(owned)
    if occupied != scope:
        return None

    for parent, mapping in _hydride_numberings(mol, Chem, core):
        locants = dict(mapping)
        if attachment is not None:
            parent = attach_suffix(parent, 'yl', (locants[attachment],))
        elif carbonyl is not None:
            parent = attach_suffix(parent, 'one', (locants[carbonyl[0]],))
        component = MappedComponent(
            parent, mapping, core, tuple((locants[i], word) for i, word in branches),
            scope - core, systematic_parent=True)
        name = decorate_component(mol, Chem, component)
        if name and _general_branch_stereo_is_cited(mol, Chem, scope, name):
            return name, component
    return None


def symmetric_bridged_amide_arm(mol, Chem, root, nitrogen, scope):
    """Name a saturated equal-bridge amide arm with one pendant ring."""
    from .complete_functional_parent import _ordinary_external_amide_core
    from .component_mapping import _complete_rooted_ring
    from .extended_inventory import selection_inventory
    from .mapped_ring_parents import ring_systems
    from .polycyclic_rules import _bicyclic_parent_sequence_candidates, _ring_system_graph

    n = mol.GetAtomWithIdx(nitrogen)
    if n.IsInRing() or n.GetDegree() != 2 or n.GetTotalNumHs() != 1:
        return ''
    inventory = selection_inventory(mol)
    if not inventory.supported or not any(
            group.kind == 'amide' and group.ligand == nitrogen and _ordinary_external_amide_core(mol, group)
            for group in inventory.principal):
        return ''
    core = next((atoms for atoms in ring_systems(mol) if root in atoms), None)
    if core is None or not core <= scope:
        return ''
    graph = _ring_system_graph(mol, Chem, core)
    if not graph or any(len(neighbors) not in (2, 3) for neighbors in graph.values()):
        return ''
    heads = {i for i, neighbors in graph.items() if len(neighbors) == 3}
    if len(heads) != 2 or root not in heads:
        return ''
    other, = heads - {root}
    exits = set()
    for i in core:
        atom = mol.GetAtomWithIdx(i)
        if (atom.GetAtomicNum() != 6 or atom.GetFormalCharge() or atom.GetIsotope()
                or atom.GetChiralTag() or atom.GetNumRadicalElectrons() or atom.HasQuery()
                or any(b.GetBondTypeAsDouble() != 1 or b.GetStereo() or b.HasQuery() for b in atom.GetBonds())):
            return ''
        exits.update((i, a.GetIdx()) for a in atom.GetNeighbors() if a.GetIdx() not in core)
    pendant = exits - {(root, nitrogen)}
    if len(pendant) != 1 or len(exits) != 2:
        return ''
    host, child = next(iter(pendant))
    owned = scope - core
    if host != other or child not in owned or not mol.GetAtomWithIdx(child).IsInRing():
        return ''
    numberings = [(parent, sequence) for parent, sequence, lengths in
                  _bicyclic_parent_sequence_candidates(mol, core, graph)
                  if sequence[0] == root and len(set(lengths)) == 1]
    if not numberings or len({(name, order.index(other)) for name, order in numberings}) != 1:
        return ''
    # Only the bridgeheads carry substituents. The surviving numberings
    # agree at both positions and all other core atoms are unannotated.
    name, order = numberings[0]
    branch = _complete_rooted_ring(mol, Chem, child, owned)
    if not branch or any(c.isspace() for c in branch):
        return ''
    locants = {i: str(j) for j, i in enumerate(order, 1)}
    component = MappedComponent(
        attach_suffix(name, 'yl', (locants[root],)), tuple(locants.items()), core,
        ((locants[other], branch),), owned, systematic_parent=True,
        prefix_branch_ownership=((child, other, owned, branch),))
    result = decorate_component(mol, Chem, component)
    return result if result and _general_branch_stereo_is_cited(mol, Chem, scope, result) else ''


def _annular_acylsulfonamide(mol, core):
    """Prove one annular C(=O)–NH–S(=O)2 unit and its external oxygens."""
    def terminal_oxo(atom):
        return tuple((atom.GetIdx(), b.GetOtherAtomIdx(atom.GetIdx())) for b in atom.GetBonds()
                     if b.GetBondTypeAsDouble() == 2 and b.GetOtherAtom(atom).GetAtomicNum() == 8
                     and b.GetOtherAtom(atom).GetDegree() == 1
                     and b.GetOtherAtomIdx(atom.GetIdx()) not in core)

    sites = []
    for idx in core:
        sulfur = mol.GetAtomWithIdx(idx)
        if sulfur.GetAtomicNum() != 16 or sulfur.GetDegree() != 4 or sulfur.GetTotalValence() != 6:
            continue
        oxo = terminal_oxo(sulfur)
        if len(oxo) != 2:
            continue
        annular = [a for a in sulfur.GetNeighbors() if a.GetIdx() in core
                   and mol.GetBondBetweenAtoms(idx, a.GetIdx()).GetBondTypeAsDouble() == 1]
        if sorted(a.GetAtomicNum() for a in annular) != [6, 7]:
            continue
        nitrogen = next(a for a in annular if a.GetAtomicNum() == 7)
        if nitrogen.GetDegree() != 2 or nitrogen.GetTotalNumHs() != 1:
            continue
        carbon = next(a for a in nitrogen.GetNeighbors() if a.GetIdx() != idx)
        if (carbon.GetAtomicNum() != 6 or carbon.GetIdx() not in core
                or mol.GetBondBetweenAtoms(nitrogen.GetIdx(), carbon.GetIdx()).GetBondTypeAsDouble() != 1
                or carbon.GetDegree() != 3
                or any(a.GetIdx() not in core and a.GetAtomicNum() != 8 for a in carbon.GetNeighbors())):
            continue
        carbonyl = terminal_oxo(carbon)
        if len(carbonyl) == 1:
            sites.append((carbonyl[0], oxo))
    return sites[0] if len(sites) == 1 else None


def _decision(name, component, principal_centers=()):
    return MethodDecision(
        name, 'substitutive', frozenset(principal_centers),
        component.atoms | component.embedded_atoms, _RULE, component.atom_locants,
        component_roles=(('numbered-parent', component.atoms, component.atom_locants),
                         ('external-components', component.embedded_atoms, ())),
        preference_issues=(SelectionLimitation(
            'numbered-polycyclic-parent', _RULE,
            'The original ring and branches are mapped. Preferred parent and method comparison remains open.'),))


def numbered_polycycle_decision(mol, Chem):
    """Recover a linked polycyclic graph when the ordinary writers return no name."""
    if mol.GetRingInfo().NumRings() < 5 or mol.GetStereoGroups():
        return None
    from .mapped_ring_parents import ring_systems
    from .polycyclic_rules import _graph_cycle_rank, _ring_system_graph

    systems = ring_systems(mol)
    if len(systems) < 2:
        return None
    cores = frozenset(core for core in systems
                      if 4 <= _graph_cycle_rank(_ring_system_graph(mol, Chem, core)) <= 9)
    if (not cores or len(Chem.GetMolFrags(mol)) != 1
            or any(a.GetAtomicNum() == 1 or a.GetIsotope() or a.GetFormalCharge()
                   or a.GetNumRadicalElectrons() or a.HasQuery() for a in mol.GetAtoms())
            or any(b.HasQuery() for b in mol.GetBonds())):
        return None
    from .extended_inventory import selection_inventory
    from .mapped_principal_general import mapped_principal_decision
    from .mapped_ring_construction import mapped_substituted_ring

    inventory = selection_inventory(mol)
    annular = {} if inventory.supported else {
        core: unit for core in cores if (unit := _annular_acylsulfonamide(mol, core)) is not None}
    if not inventory.supported and not annular:
        return None
    # The earlier writers may have cached misses with their ordinary resolver.
    graph = Chem.Mol(mol)
    mol_cache_set(graph, 'numbered_polycycle_recovery', cores)
    scope = frozenset(range(graph.GetNumAtoms()))
    if inventory.supported and inventory.principal:
        joined = mapped_principal_decision(graph, Chem, carbon_branch_resolver=mapped_general_branch)
        if (joined is not None and joined.atoms == scope
                and _general_branch_stereo_is_cited(graph, Chem, scope, joined.name)):
            return joined
        return None
    if inventory.supported:
        row = mapped_substituted_ring(graph, Chem)
        if row is not None and _general_branch_stereo_is_cited(graph, Chem, scope, row.name):
            return _decision(row.name, row.construction)
    for core in sorted(cores if inventory.supported else annular, key=lambda c: (-len(c), tuple(sorted(c)))):
        carbonyl, sulfur_oxo = annular.get(core, (None, ()))
        result = numbered_polycycle_component(
            graph, Chem, core, scope, carbonyl=carbonyl, sulfur_oxo=sulfur_oxo)
        if result is not None:
            name, component = result
            return _decision(name, component, (carbonyl[0],) if carbonyl is not None else ())
    return None
