"""Source-owned acyclic acid parent proposals for the shared selector.

This producer is independent of specific substituent or ring families. It
renders only fully mapped original-graph branches and does not assert that all
possible nomenclature methods or whole-parent classes have been compared.
"""

from .naming_methods import MethodDecision
from .naming_result import SelectionLimitation


def has_acyclic_carboxy_motif(mol):
    """Cheap necessary condition; detailed acid roles are checked later."""
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6 or atom.IsInRing():
            continue
        neighbors = tuple(atom.GetNeighbors())
        if (any(n.GetAtomicNum() == 8 and mol.GetBondBetweenAtoms(
                    atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 2 for n in neighbors)
                and any(n.GetAtomicNum() == 8 and n.GetDegree() == 1
                        and n.GetTotalNumHs(includeNeighbors=True) == 1
                        and mol.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 1
                        for n in neighbors)
                and any(n.GetAtomicNum() == 6 and not n.IsInRing() for n in neighbors)):
            return True
    return False


def _acid_branch(mol, Chem, root, host):
    imidate = _imidate_alkoxy_branch(mol, Chem, root, host)
    if imidate is not None:
        return imidate
    linked = _spiro_alkoxy_branch(mol, Chem, root, host)
    if linked is not None:
        return linked
    atom = mol.GetAtomWithIdx(root)
    if (atom.GetAtomicNum() == 15 and atom.GetDegree() == 1 and not atom.GetIsotope()
            and not atom.GetFormalCharge() and atom.GetTotalNumHs(includeNeighbors=True) == 2
            and mol.GetBondBetweenAtoms(root, host).GetBondTypeAsDouble() == 1):
        return 'phosphanyl', frozenset({root})
    from .component_mapping import mapped_branch, mapped_branch_ring_atoms, _record_branch_ring_atoms
    text, owned = mapped_branch(mol, Chem, root, host)
    if not mol.GetAtomWithIdx(root).IsInRing() or text and mapped_branch_ring_atoms(
            mol, Chem, root, host, text, owned):
        return text, owned
    # An explicitly numbered replacement ring can supply an exact general
    # prefix under a mapped acid without asserting preferred fusion.
    from .branch_rules import collect_subtree
    from .mapped_ring_parents import records
    scope = frozenset(collect_subtree(mol, root, host))
    rows = records(mol, Chem, scope=scope, attachment=root)
    if len(rows) == 1:
        row = rows[0]
        component = row.construction
        if (component is not None and row.hydrogen_budget_proved and row.annotation_coverage
                and component.atoms | component.embedded_atoms == scope):
            _record_branch_ring_atoms(mol, Chem, root, host, row.name, scope,
                                      (i for i in scope if mol.GetAtomWithIdx(i).IsInRing()))
            return row.name, scope
    from .spiro_component_rules import mapped_component_spiro_yl
    spiro, spiro_atoms = mapped_component_spiro_yl(mol, Chem, root, host, scope, allow_replacement=True)
    if spiro:
        return spiro, spiro_atoms
    return text, owned


def _imidate_alkoxy_branch(mol, Chem, root, host):
    """Render a terminal NH imidate as a singly attached alkoxy branch."""
    oxygen = mol.GetAtomWithIdx(root)
    if oxygen.GetAtomicNum() != 8 or oxygen.IsInRing() or oxygen.GetDegree() != 2:
        return None
    center = next(a for a in oxygen.GetNeighbors() if a.GetIdx() != host)
    if (center.GetAtomicNum() != 6 or center.IsInRing() or center.GetDegree() != 3
            or any(b.GetBondTypeAsDouble() != 1 for b in oxygen.GetBonds())):
        return None
    ports = {b.GetOtherAtom(center).GetAtomicNum(): b for b in center.GetBonds()
             if b.GetOtherAtomIdx(center.GetIdx()) != root}
    if ports.keys() != {6, 7} or ports[6].GetBondTypeAsDouble() != 1:
        return None
    nitrogen = ports[7].GetOtherAtom(center)
    if (ports[7].GetBondTypeAsDouble() != 2 or ports[7].GetStereo()
            or nitrogen.GetDegree() != 1 or nitrogen.GetTotalNumHs() != 1):
        return None
    attachment = ports[6].GetOtherAtomIdx(center.GetIdx())
    if not mol.GetAtomWithIdx(attachment).IsInRing():
        return None
    from .branch_rules import collect_subtree
    from .component_mapping import _record_branch_ring_atoms
    from .formatting import enclose_name_unit
    from .mapped_ring_parents import records

    ring_scope = frozenset(collect_subtree(mol, attachment, center.GetIdx()))
    owned = ring_scope | {root, center.GetIdx(), nitrogen.GetIdx()}
    if (ring_scope & {root, center.GetIdx(), nitrogen.GetIdx(), host}
            or owned != frozenset(collect_subtree(mol, root, host))):
        return None
    rows = records(mol, Chem, scope=ring_scope, attachment=attachment, general_branches=True)
    names = {r.name for r in rows if r.hydrogen_budget_proved and r.annotation_coverage
             and r.construction is not None
             and r.construction.atoms | r.construction.embedded_atoms == ring_scope}
    if len(names) != 1:
        return None
    word = 'imino' + enclose_name_unit(names.pop()) + 'methoxy'
    _record_branch_ring_atoms(mol, Chem, root, host, word, owned,
                              (i for i in ring_scope if mol.GetAtomWithIdx(i).IsInRing()))
    return word, owned


def _spiro_alkoxy_branch(mol, Chem, root, host):
    """Name a singly attached O–CH2–spiro branch with original stereo."""
    oxygen = mol.GetAtomWithIdx(root)
    if (oxygen.GetAtomicNum() != 8 or oxygen.GetDegree() != 2 or oxygen.IsInRing()
            or mol.GetRingInfo().NumRings() < 3):
        return None
    methylene = next(a for a in oxygen.GetNeighbors() if a.GetIdx() != host)
    if (methylene.GetAtomicNum() != 6 or methylene.IsInRing()
            or methylene.GetDegree() != 2 or methylene.GetTotalNumHs() != 2):
        return None
    attachment = next(a for a in methylene.GetNeighbors() if a.GetIdx() != root)
    if not attachment.IsInRing() or any(
            b.GetBondTypeAsDouble() != 1 for b in (*oxygen.GetBonds(), *methylene.GetBonds())):
        return None
    from .branch_rules import collect_subtree
    from .component_mapping import _record_branch_ring_atoms, mapped_branch_ring_atoms
    from .formatting import enclose_name_unit
    from .molecule_cache import mol_cache_dict, naming_cache_state
    from .spiro_component_rules import mapped_component_spiro_yl

    cache = mol_cache_dict(mol, 'spiro_alkoxy_branch')
    key = (id(Chem), root, host, naming_cache_state(mol))
    if key in cache:
        return cache[key]
    scope = frozenset(collect_subtree(mol, attachment.GetIdx(), methylene.GetIdx()))
    child, linker = attachment.GetIdx(), methylene.GetIdx()
    text, atoms = mapped_component_spiro_yl(mol, Chem, child, linker,
                                            scope, allow_replacement=True, general_branches=True)
    result = None
    if text and atoms == scope:
        ring_atoms = mapped_branch_ring_atoms(mol, Chem, child, linker, text, atoms)
        if ring_atoms:
            result = (enclose_name_unit(text) + 'methoxy', atoms | {root, linker})
            _record_branch_ring_atoms(mol, Chem, root, host, *result, ring_atoms)
    cache[key] = result
    return result


def _acid_branch_annotations_proved(mol, Chem, root, host, atoms, text, bond_sites):
    if text.startswith('imino') and _imidate_alkoxy_branch(mol, Chem, root, host) == (text, atoms):
        return True
    if not bond_sites and _spiro_alkoxy_branch(mol, Chem, root, host) == (text, atoms):
        return True
    if mol.GetAtomWithIdx(root).GetAtomicNum() == 8 and bond_sites:
        from .oxime_components import oxime_ether_oxy_prefix
        from .stereochemistry import _rendered_mapped_branch_bonds

        if (oxime_ether_oxy_prefix(mol, Chem, root, host, atoms) == (text, atoms)
                and not any(int(mol.GetAtomWithIdx(i).GetChiralTag()) for i in atoms)):
            return bond_sites <= _rendered_mapped_branch_bonds(mol, text).keys()
    if mol.GetAtomWithIdx(root).IsInRing():
        from .retained_acid_rules import mapped_ring_stereo_prefix_proved
        from .spiro_component_rules import mapped_component_spiro_yl
        return (
            not bond_sites and mapped_ring_stereo_prefix_proved(mol, Chem, root, host, atoms, text)
            or mapped_component_spiro_yl(
                mol, Chem, root, host, atoms, allow_replacement=True) == (text, atoms)
        )
    from .branch.amido_components import mapped_amido_prefix
    return mapped_amido_prefix(mol, Chem, root, host, atoms, chain_only=True) == (text, atoms)


def _configured_aziridine_inventory(mol, Chem):
    """Classify acid sites while preserving configured annular N on the source."""
    from .extended_inventory import selection_inventory
    from .stereochemistry import assigned_atom_stereochemistry

    sites = [a.GetIdx() for a in mol.GetAtoms() if a.GetChiralTag() and a.GetAtomicNum() != 6]
    if len(sites) != 1:
        return None
    idx = sites[0]
    atom = mol.GetAtomWithIdx(idx)
    if (atom.GetAtomicNum() != 7 or atom.GetDegree() != 3 or atom.GetTotalNumHs() != 0
            or dict(assigned_atom_stereochemistry(mol, Chem)).get(idx) not in {'R', 'S'}):
        return None
    rings = [ring for ring in mol.GetRingInfo().AtomRings() if idx in ring]
    if (len(rings) != 1 or len(rings[0]) != 3
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in rings[0] if i != idx)
            or any(b.GetBondTypeAsDouble() != 1 for b in atom.GetBonds())):
        return None
    # Only class recognition uses the achiral copy. Branch and parent
    # stereodescriptors are rendered below from the unchanged original graph.
    classification_graph = Chem.Mol(mol)
    classification_graph.GetAtomWithIdx(idx).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    return selection_inventory(classification_graph)


def mapped_monoacid_chain_decision(mol, Chem):
    """Propose a source-owned chain monoacid or terminal diacid."""
    if (mol is None or not has_acyclic_carboxy_motif(mol)
            or mol.GetNumHeavyAtoms() > 80 or mol.GetStereoGroups()
            or len(Chem.GetMolFrags(mol)) != 1
            or any(a.GetAtomicNum() not in {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}
                   or a.GetFormalCharge() or a.GetIsotope() or a.GetNumRadicalElectrons()
                   or a.HasQuery() for a in mol.GetAtoms())
            or any(b.HasQuery() for b in mol.GetBonds())):
        return None
    if any(atom.GetAtomMapNum() for atom in mol.GetAtoms()):
        # Caller maps are graph-representation labels, not name locants. Some
        # branch and parent witnesses canonicalize their input, so clear maps
        # only on a private copy before evaluating those witnesses. RDKit's
        # copy preserves atom indices, bonds, isotope/stereo state and native
        # properties: every returned center, ownership set and atom locant
        # still refers to the corresponding original atom. The original graph
        # (including its caller-assigned maps) is never mutated.
        mol = Chem.Mol(mol)
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum():
                atom.SetAtomMapNum(0)
    from .extended_inventory import selection_inventory

    inventory = selection_inventory(mol)
    branch_resolver = _acid_branch
    general_branches = False
    if not inventory.supported:
        alternative = _configured_aziridine_inventory(mol, Chem)
        if alternative is not None:
            from .component_mapping import mapped_general_branch

            inventory = alternative
            branch_resolver = mapped_general_branch
            general_branches = True
    frontier = inventory.source_frontier
    if (not inventory.supported or frontier.status not in {'selected', 'tie'}
            or frontier.unknown_pairs or frontier.cycle or not 1 <= len(frontier.groups) <= 2
            or any(group.kind != 'acid' for group in frontier.groups)
            or tuple(frontier.groups) != tuple(inventory.principal)):
        return None
    acids = frontier.groups
    if any(acid.owner < 0 or mol.GetAtomWithIdx(acid.owner).GetAtomicNum() != 6
           or mol.GetAtomWithIdx(acid.owner).IsInRing() for acid in acids):
        return None
    from .parent_rules import _ester_acyl_parent_paths

    centers = frozenset(acid.center for acid in acids)
    acid_atoms = {i for acid in acids for i in acid.atoms}
    if len(acids) == 1:
        acid = acids[0]
        paths = [(acid.center, *path)
                 for path in _ester_acyl_parent_paths(mol, Chem, acid.center, acid.owner)]
    else:
        path = tuple(Chem.GetShortestPath(mol, acids[0].center, acids[1].center))
        if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6
               or mol.GetAtomWithIdx(i).IsInRing() for i in path):
            return None
        paths = [path, path[::-1]] if path else []
    if not paths or len(paths) > 64:
        return None
    # All candidate chains already have the greatest length. Compare the
    # graph-only multiple-bond criteria before asking a writer to render any
    # of the tied maxima, so a failed rendering cannot promote an inferior path.
    def unsaturation(path):
        orders = tuple(mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble()
                       for a, b in zip(path, path[1:]))
        return (sum(order in {2, 3} for order in orders), orders.count(2))

    maximum = max(map(unsaturation, paths))
    paths = [path for path in paths if unsaturation(path) == maximum]
    tagged = {a.GetIdx() for a in mol.GetAtoms() if int(a.GetChiralTag())}
    from .stereochemistry import assigned_atom_stereochemistry, assigned_bond_stereochemistry
    assigned = dict(assigned_atom_stereochemistry(mol, Chem, atom_order='idx', include_pseudo=True))
    if set(assigned) != tagged or any(value not in {'R', 'S'} for value in assigned.values()):
        return None
    geometry = {frozenset((a, b)) for a, b, _ in assigned_bond_stereochemistry(mol, Chem)}
    if any(int(b.GetStereo()) and frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) not in geometry
           for b in mol.GetBonds()):
        return None
    from .complete_functional_parent import _render_functional_candidate
    from .branch_rules import collect_subtree

    proposals = []
    scope = frozenset(a.GetIdx() for a in mol.GetAtoms())
    for path in paths:
        proposal = _render_functional_candidate(
            mol, Chem, acids, 'acid', path, False, False,
            acid_atoms - centers, [], mapped_ring_prefixes=True,
            charge_role_proofs=(), locanted_acid_fallback=True, branch_resolver=branch_resolver,
            general_branches=general_branches,
        )
        if proposal is None or not proposal.name or proposal.principal_centers != centers:
            return None
        component = proposal.construction
        if (component is None or component.atoms & component.embedded_atoms
                or component.atoms | component.embedded_atoms != scope
                or not acid_atoms <= component.atoms):
            return None
        claimed = set()
        embedded_stereo = set()
        embedded_geometry = set()
        for root, host, owned, text in component.prefix_branch_ownership:
            owned = frozenset(owned)
            if (host not in path or root not in owned or claimed & owned
                    or frozenset(collect_subtree(mol, root, host)) != owned
                    or branch_resolver(mol, Chem, root, host) != (text, owned)):
                return None
            cuts = {(idx, neighbor.GetIdx()) for idx in owned
                    for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors()
                    if neighbor.GetAtomicNum() > 1 and neighbor.GetIdx() not in owned}
            if cuts != {(root, host)}:
                return None
            bond_sites = {ends for ends in geometry if ends & owned}
            if tagged & owned or bond_sites:
                if general_branches:
                    from .component_mapping import _general_branch_stereo_is_cited

                    proved = _general_branch_stereo_is_cited(mol, Chem, owned, text)
                else:
                    proved = _acid_branch_annotations_proved(mol, Chem, root, host, owned, text,
                                                            bond_sites)
                if (any(not ends <= owned for ends in bond_sites)
                        or not proved):
                    return None
                embedded_stereo.update(tagged & owned)
                embedded_geometry.update(bond_sites)
            claimed.update(owned)
        if (claimed != set(component.embedded_atoms) or tagged - set(path) != embedded_stereo
                or any(not ends <= set(path) and ends not in embedded_geometry for ends in geometry)):
            return None
        proposals.append(proposal)
    if len(proposals) > 1:
        if any(row.parent_selection is None for row in proposals):
            return None
        from .parent_selection import parent_frontier
        proposals = parent_frontier(proposals, lambda row: row.parent_selection)
    if len({row.name for row in proposals}) != 1:
        return None
    proposal = proposals[0]
    issues = [SelectionLimitation(
        'monoacid-parent', 'P-41/P-44/P-45/P-51/P-65',
        'The acid-bearing chain and every emitted prefix are source-mapped; '
        'complete competing parent and naming-method selection remains unresolved.')]
    try:
        potential = tuple(Chem.FindPotentialStereo(mol))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    if any(item.specified != Chem.StereoSpecified.Specified for item in potential):
        issues.append(SelectionLimitation(
            'stereochemistry', 'P-91', 'Potential stereochemistry is unassigned.'))
    return MethodDecision(
        proposal.name, 'mapped-monoacid-chain' if len(acids) == 1 else 'mapped-diacid-chain',
        centers, scope,
        'P-41/P-44/P-65', proposal.atom_locants,
        preference_issues=tuple(issues),
    )


def requires_full(mol, Chem):
    """Hand off only when the complete writer selected this exact proposal."""
    if not has_acyclic_carboxy_motif(mol):
        return False
    proof = mapped_monoacid_chain_decision(mol, Chem)
    if proof is None:
        return False
    from .bounded_mixed_acid_ester_fast_guard import _actual_principal_decision
    actual = _actual_principal_decision(mol, Chem)
    return bool(actual is not None and actual.output_name == proof.name
                and actual.comparison == proof and actual.prepared is None
                and set(proof.preference_issues) <= set(actual.preference_issues))
