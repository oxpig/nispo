"""Mapped names for bounded urea and chalcogenurea domains (P-66.1.6.1)."""

from __future__ import annotations

from .molecule_iteration import iter_atoms, iter_bonds
from .component_mapping import mapped_branch
from .formatting import format_prefixes, multiply_component, prefix_alphabetization_key
from .naming_methods import MethodDecision


_PARENT_BY_CORE_CHALCOGEN = {
    8: ("urea", "P-66.1.6.1.1.1/P-66.1.6.1.1.2"),
    16: ("thiourea", "P-66.1.6.1.3.1/P-43.1"),
    34: ("selenourea", "P-66.1.6.1.3.1/P-43.1"),
    52: ("tellurourea", "P-66.1.6.1.3.1/P-43.1"),
}


def urea_decision(mol, Chem):
    decision = _bounded_urea_decision(mol, Chem)
    if decision is not None:
        return decision
    from .extended_inventory import selection_inventory

    inventory = selection_inventory(mol)
    if (not inventory.supported or inventory.source_frontier.status != 'selected'
            or len(inventory.principal) != 1 or inventory.principal[0].kind != 'urea'):
        return None
    group = inventory.principal[0]
    nitrogens = tuple(i for i in group.atoms if mol.GetAtomWithIdx(i).GetAtomicNum() == 7)
    if len(nitrogens) != 2 or any(mol.GetAtomWithIdx(i).IsInRing() for i in nitrogens):
        return None  # A ring N can require the heterocyclic carboxamide construction.
    return _mapped_urea_substituents(mol, Chem, group, nitrogens)


def _mapped_urea_substituents(mol, Chem, group, nitrogens):
    from .component_mapping import mapped_general_branch
    from .naming_result import SelectionLimitation

    core = frozenset(group.atoms)
    if (mol.GetStereoGroups() or any(a.GetIsotope() or a.GetFormalCharge()
                                    or a.GetNumRadicalElectrons() for a in iter_atoms(mol))
            or any(int(mol.GetAtomWithIdx(i).GetChiralTag()) for i in core)):
        return None
    owned = set(core)
    by_nitrogen = {}
    branch_roles = []
    for i in nitrogens:
        prefixes = []
        for bond in mol.GetAtomWithIdx(i).GetBonds():
            root = bond.GetOtherAtomIdx(i)
            if root == group.center or mol.GetAtomWithIdx(root).GetAtomicNum() == 1:
                continue
            if bond.GetBondType() != Chem.BondType.SINGLE:
                return None
            text, atoms = mapped_general_branch(mol, Chem, root, i)
            if not text or atoms & owned:
                return None
            prefixes.append(text)
            owned.update(atoms)
            branch_roles.append(('N-substituent', atoms, ()))
        by_nitrogen[i] = tuple(prefixes)
    heavy = frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1)
    if owned != heavy:
        return None
    # The two equivalent N sites can be numbered either way. The first
    # criterion is the lowest set of locants, then the citation-order set.
    options = []
    for first, second in (nitrogens, nitrogens[::-1]):
        locants = ((first, 'N'), (second, "N'"))
        prefixes = [(locant, text) for idx, locant in locants for text in by_nitrogen[idx]]
        ordered = sorted(prefixes, key=lambda p: (prefix_alphabetization_key(p[1]), p[0]))
        key = (tuple(sorted(loc for loc, _ in prefixes)), tuple(loc for loc, _ in ordered))
        options.append((key, format_prefixes(prefixes, hierarchical_enclosures=True), locants))
    _, prefix, locants = min(options)
    return MethodDecision(prefix + 'urea', 'substitutive', frozenset({group.center}), heavy,
                          'P-41/P-66.1.6.1', locants,
                          component_roles=(('urea-core', core, locants), *branch_roles),
                          preference_issues=(SelectionLimitation('urea-substitution', 'P-66.1.6.1',
                              'The principal urea and all N substituents are mapped.'),))


def _bounded_urea_decision(mol, Chem):
    """Return a mapped urea/chalcogenurea construction when proved.

    Scope: one neutral acyclic X=C(N)(N) unit for X = O, S, Se, or Te; the N
    substituents are ordinary saturated carbon-only branches, terminal
    halogens, an exact terminal nitroso/nitro group, or one completely mapped
    saturated carbon/oxygen ring branch at the urea nitrogen.
    All heavy atoms must be owned by the core or one of those branches.
    Unsupported core labels, explicit isotope-H, and unproved component types
    decline.
    """
    if mol.GetNumHeavyAtoms() > 128 or mol.GetStereoGroups():
        return None
    # Explicit hydrogens would need their own component-level isotope proof.
    if any(a.GetAtomicNum() == 1 for a in iter_atoms(mol)):
        return None
    motifs = []
    for carbon in iter_atoms(mol):
        if carbon.GetAtomicNum() != 6 or carbon.GetFormalCharge() != 0:
            continue
        cidx = carbon.GetIdx()
        chalcogens = []
        nitrogens = []
        valid = True
        for neighbor in carbon.GetNeighbors():
            idx = neighbor.GetIdx()
            bond = mol.GetBondBetweenAtoms(cidx, idx)
            if (neighbor.GetAtomicNum() in _PARENT_BY_CORE_CHALCOGEN
                    and bond.GetBondType() == Chem.BondType.DOUBLE):
                chalcogens.append(idx)
            elif neighbor.GetAtomicNum() == 7 and bond.GetBondType() == Chem.BondType.SINGLE:
                nitrogens.append(idx)
            else:
                valid = False
        if valid and len(chalcogens) == 1 and len(nitrogens) == 2:
            motifs.append((cidx, chalcogens[0], tuple(nitrogens)))
    if len(motifs) != 1:
        return None
    core_carbon, chalcogen, nitrogens = motifs[0]
    core = {core_carbon, chalcogen, *nitrogens}
    for idx in core:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetFormalCharge() or atom.GetNumRadicalElectrons():
            return None
        # The central carbon and chalcogen have no mapped isotope descriptor;
        # nitrogen stereochemistry/isotopes also lack a proved grammar here.
        if atom.GetIsotope() or int(atom.GetChiralTag()):
            return None
    if any(int(b.GetStereo()) for b in iter_bonds(mol)):
        return None
    if any(a.GetAtomicNum() in {6, 7, 8, 16, 34, 52} and a.GetIsotope()
           and a.GetAtomicNum() != 6 for a in iter_atoms(mol)):
        return None

    n_substitution_counts = sorted(
        sum(a.GetAtomicNum() > 1 and a.GetIdx() != core_carbon
            for a in mol.GetAtomWithIdx(nitrogen).GetNeighbors())
        for nitrogen in nitrogens)
    if n_substitution_counts == [1, 2]:
        from .trisubstituted_urea import trisubstituted_urea_decision
        mapped_trisubstituted = trisubstituted_urea_decision(
            mol, Chem, core_carbon, chalcogen, nitrogens)
        if mapped_trisubstituted is not None:
            return mapped_trisubstituted

    mapped_disubstituted = _mapped_ring_disubstituted_urea(
        mol, Chem, core_carbon, chalcogen, nitrogens)
    if mapped_disubstituted is not None:
        return mapped_disubstituted

    side_names = []
    side_atoms = []
    from .locant_presentation import SubstitutionSite, substitution_locant_presentation
    substitutions = []
    owned = set(core)
    cyclic_branch_used = False
    for nitrogen in nitrogens:
        n_atom = mol.GetAtomWithIdx(nitrogen)
        if (n_atom.GetFormalCharge() or n_atom.GetNumRadicalElectrons()
                or n_atom.GetIsotope() or int(n_atom.GetChiralTag())):
            return None
        branches = [a.GetIdx() for a in n_atom.GetNeighbors()
                    if a.GetIdx() != core_carbon and a.GetAtomicNum() > 1]
        if len(branches) > 2:
            return None
        names = []
        branch_owned = set()
        for root in branches:
            root_atom = mol.GetAtomWithIdx(root)
            halogen = {9:'fluoro',17:'chloro',35:'bromo',53:'iodo'}.get(root_atom.GetAtomicNum())
            if root_atom.GetAtomicNum() == 6:
                atoms = _ordinary_carbon_branch(mol, root, nitrogen, Chem)
                if not atoms:
                    atoms = _bounded_oxygenated_ring_branch(mol, root, nitrogen, Chem)
                    if not atoms:
                        return None
                    cyclic_branch_used = True
            elif root_atom.GetAtomicNum() == 7:
                from .branch_rules import collect_subtree
                from .nitrogen_oxo_amide_prefixes import direct_nitrogen_oxo_substituent

                subtree = frozenset(collect_subtree(mol, root, nitrogen))
                typed = direct_nitrogen_oxo_substituent(
                    mol, root, nitrogen, subtree,
                )
                if typed is None:
                    return None
                atoms = typed.atoms
            elif (halogen and root_atom.GetDegree() == 1 and not root_atom.GetIsotope()
                  and not root_atom.GetFormalCharge() and not root_atom.GetNumRadicalElectrons()):
                atoms = frozenset({root})
            else:
                return None
            if branch_owned.intersection(atoms):
                return None
            text, mapped_atoms = ((halogen, atoms) if halogen else
                                  mapped_branch(mol, Chem, root, nitrogen))
            if not text or set(mapped_atoms) != atoms or branch_owned.intersection(mapped_atoms):
                return None
            names.append(text)
            substitutions.append(SubstitutionSite(nitrogen,root,frozenset(atoms),text))
            branch_owned.update(atoms)
        side_names.append(tuple(names))
        side_atoms.append(frozenset(branch_owned))
        owned.update(branch_owned)
    heavy = {a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1}
    if owned != heavy:
        return None
    # The N-site selection proof for this extension covers one cyclic
    # substituent on an otherwise unsubstituted urea parent. Other combinations
    # retain their established writers until their complete comparison is proved.
    if cyclic_branch_used and sum(map(len, side_names)) != 1:
        return None

    n_order = _named_nitrogen_order(nitrogens, side_names)
    n_map = {n_order[0]: "N", n_order[1]: "N′"}
    bindings = ((core_carbon,"1"),(chalcogen,"X"),*n_map.items())
    omission = substitution_locant_presentation(mol,Chem,bindings,substitutions)
    prefix = (_unlocanted_substitution_prefix(mol, substitutions)
              if omission is not None else _substitution_prefix(side_names))
    if prefix is None:
        return None
    parent_name, rule = _PARENT_BY_CORE_CHALCOGEN[
        mol.GetAtomWithIdx(chalcogen).GetAtomicNum()]
    name = prefix + parent_name
    roles = (("urea-core", frozenset(core),
              ((n_order[0], "N"), (n_order[1], "N′"))),) + tuple(
        (f"substituents-at-{n_map[nitrogen]}", side_atoms[nitrogens.index(nitrogen)], ())
        for nitrogen in n_order
    )
    from .naming_result import SelectionLimitation
    issues = ((SelectionLimitation(
        'urea-ring-substitution', 'P-66.1.6.1/P-44/P-45',
        'The complete mapped cyclic substituent is available; preferred parent and locant selection remain unresolved.'),)
        if cyclic_branch_used else ())
    if any(item.specified != Chem.StereoSpecified.Specified
           for item in Chem.FindPotentialStereo(mol)):
        issues += (SelectionLimitation(
            'stereochemistry', 'P-91.2.1.2.1/P-91.2.2',
            'One or more potential stereogenic units have no source configuration.'),)
    return MethodDecision(name, "substitutive", frozenset({core_carbon}),
                          frozenset(heavy), rule,
                          ((n_order[0], "N"), (n_order[1], "N′")),
                          component_roles=roles, preference_issues=issues)


def _mapped_ring_disubstituted_urea(mol, Chem, carbonyl, oxygen, nitrogens):
    """One senior acyclic urea with one separately mapped C arm on each N.

    New ring-containing arms must have an original-atom ring record, complete
    annotation/valence coverage, a single cut, and no local preference issue.
    A non-ring root is limited to the unique 1-(ring-yl)ethyl construction.
    The overall PIN status remains unresolved: these component proofs do not
    establish every comparison made by the enclosing parent selection.
    """
    from .branch_rules import collect_subtree
    from .component_mapping import mapped_branch_ring_atoms
    from .extended_inventory import selection_inventory
    from .locant_presentation import SubstitutionSite, substitution_locant_presentation
    from .mapped_ring_parents import records
    from .naming_result import SelectionLimitation

    if (mol.GetAtomWithIdx(oxygen).GetAtomicNum() != 8
            or any(mol.GetAtomWithIdx(i).IsInRing() for i in (carbonyl, *nitrogens))
            or any(a.GetIsotope() or a.GetFormalCharge() or a.GetNumRadicalElectrons()
                   for a in iter_atoms(mol))):
        return None
    core = frozenset({carbonyl, oxygen, *nitrogens})
    roots = []
    for nitrogen in nitrogens:
        atom = mol.GetAtomWithIdx(nitrogen)
        children = [n.GetIdx() for n in atom.GetNeighbors()
                    if n.GetIdx() != carbonyl and n.GetAtomicNum() > 1]
        if (atom.GetTotalValence() != 3 or atom.GetTotalNumHs(includeNeighbors=True) != 1
                or len(children) != 1 or mol.GetAtomWithIdx(children[0]).GetAtomicNum() != 6
                or mol.GetBondBetweenAtoms(nitrogen, children[0]).GetBondTypeAsDouble() != 1):
            return None
        roots.append(children[0])
    if not any(mol.GetAtomWithIdx(i).IsInRing()
               for root, nitrogen in zip(roots, nitrogens)
               for i in collect_subtree(mol, root, nitrogen)):
        return None
    inventory = selection_inventory(mol)
    frontier = inventory.source_frontier
    if (not inventory.supported or frontier.status != "selected"
            or len(frontier.groups) != 1 or frontier.groups[0].kind != "urea"
            or frontier.groups[0].center != carbonyl
            or frozenset(frontier.groups[0].atoms) != core):
        return None

    def one_cut(atoms, root, blocked):
        return {(i, n.GetIdx()) for i in atoms for n in mol.GetAtomWithIdx(i).GetNeighbors()
                if n.GetAtomicNum() > 1 and n.GetIdx() not in atoms} == {(root, blocked)}

    def mapped_ring(root, blocked, atoms):
        if not mol.GetAtomWithIdx(root).IsInRing() or not one_cut(atoms, root, blocked):
            return None
        options = records(mol, Chem, scope=atoms, attachment=root)
        if len(options) != 1:
            return None
        record = options[0]
        construction = record.construction
        if (not record.hydrogen_budget_proved or not record.annotation_coverage
                or record.preference_issues or construction is None
                or construction.atoms | construction.embedded_atoms != atoms):
            return None
        text, owned = mapped_branch(mol, Chem, root, blocked)
        ring_atoms = frozenset(i for i in atoms if mol.GetAtomWithIdx(i).IsInRing())
        if (text != record.name or frozenset(owned) != atoms or not ring_atoms
                or mapped_branch_ring_atoms(mol, Chem, root, blocked, text, owned) != ring_atoms):
            return None
        return text

    sides, side_atoms = [], []
    owned = set(core)
    substitutions = []
    for root, nitrogen in zip(roots, nitrogens):
        atoms = frozenset(collect_subtree(mol, root, nitrogen))
        if not atoms or atoms & owned or not one_cut(atoms, root, nitrogen):
            return None
        atom = mol.GetAtomWithIdx(root)
        if atom.IsInRing():
            text = mapped_ring(root, nitrogen, atoms)
        else:
            # The root-containing acyclic chain has exactly two carbons;
            # the other attached component is one independently mapped ring.
            children = [n.GetIdx() for n in atom.GetNeighbors()
                        if n.GetIdx() != nitrogen and n.GetAtomicNum() > 1]
            methyl = [i for i in children if not mol.GetAtomWithIdx(i).IsInRing()]
            rings = [i for i in children if mol.GetAtomWithIdx(i).IsInRing()]
            if (atom.GetAtomicNum() != 6 or atom.GetTotalValence() != 4
                    or atom.GetTotalNumHs(includeNeighbors=True) != 1 or int(atom.GetChiralTag())
                    or len(children) != 2 or len(methyl) != 1 or len(rings) != 1):
                return None
            methyl_atom = mol.GetAtomWithIdx(methyl[0])
            if (methyl_atom.GetAtomicNum() != 6 or methyl_atom.GetTotalValence() != 4
                    or methyl_atom.GetTotalNumHs(includeNeighbors=True) != 3
                    or methyl_atom.GetDegree() != 1 or int(methyl_atom.GetChiralTag())
                    or any(mol.GetBondBetweenAtoms(root, i).GetBondTypeAsDouble() != 1
                           for i in children)):
                return None
            ring_atoms = frozenset(collect_subtree(mol, rings[0], root))
            if ring_atoms | {root, methyl[0]} != atoms or ring_atoms & {root, methyl[0]}:
                return None
            ring_text = mapped_ring(rings[0], root, ring_atoms)
            text = (format_prefixes([("1", ring_text)], hierarchical_enclosures=True) + "ethyl"
                    if ring_text else None)
        if not text:
            return None
        sides.append((text,))
        side_atoms.append(atoms)
        substitutions.append(SubstitutionSite(nitrogen, root, atoms, text))
        owned.update(atoms)
    heavy = frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1)
    if frozenset(owned) != heavy:
        return None
    n_order = _named_nitrogen_order(nitrogens, sides)
    n_map = {n_order[0]: "N", n_order[1]: "N′"}
    bindings = ((carbonyl, "1"), (oxygen, "X"), *n_map.items())
    omission = substitution_locant_presentation(mol, Chem, bindings, substitutions)
    prefix = (_unlocanted_substitution_prefix(mol, substitutions)
              if omission is not None else _substitution_prefix(sides))
    if prefix is None:
        return None
    roles = (("urea-core", core, ((n_order[0], "N"), (n_order[1], "N′"))),) + tuple(
        (f"substituents-at-{n_map[nitrogen]}", side_atoms[nitrogens.index(nitrogen)], ())
        for nitrogen in n_order)
    issue = SelectionLimitation(
        "urea-ring-substitution", "P-66.1.6.1/P-44/P-45",
        "Both ring-containing N substituents are mapped; the complete preferred name remains unresolved.")
    stereo_issues = ((SelectionLimitation(
        'stereochemistry', 'P-91.2.1.2.1/P-91.2.2',
        'One or more potential stereogenic units have no source configuration.'),)
        if any(item.specified != Chem.StereoSpecified.Specified
               for item in Chem.FindPotentialStereo(mol)) else ())
    return MethodDecision(prefix + "urea", "substitutive", frozenset({carbonyl}), heavy,
                          "P-66.1.6.1.1.1/P-66.1.6.1.1.2/P-46.1.2", 
                          ((n_order[0], "N"), (n_order[1], "N′")),
                          component_roles=roles, preference_issues=(issue, *stereo_issues))


def _bounded_oxygenated_ring_branch(mol, root, blocked, Chem):
    """Prove one complete neutral saturated C/O ring branch on the input graph."""
    from .branch_rules import collect_subtree
    from .mapped_ring_parents import records

    if (not mol.GetAtomWithIdx(root).IsInRing()
            or mol.GetAtomWithIdx(blocked).IsInRing()):
        return None
    atoms = frozenset(collect_subtree(mol, root, blocked))
    if not atoms or blocked in atoms:
        return None
    for idx in atoms:
        atom = mol.GetAtomWithIdx(idx)
        if (atom.GetAtomicNum() not in {6, 8} or atom.GetFormalCharge()
                or atom.GetNumRadicalElectrons() or atom.GetIsotope()
                or atom.GetIsAromatic()):
            return None
        if atom.GetAtomicNum() == 6 and atom.GetTotalValence() != 4:
            return None
        if atom.GetAtomicNum() == 8 and (
                atom.GetTotalValence() != 2
                or any(n.GetAtomicNum() != 6 for n in atom.GetNeighbors())):
            return None
    cuts = {(idx, neighbor.GetIdx()) for idx in atoms
            for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors()
            if neighbor.GetIdx() not in atoms}
    if cuts != {(root, blocked)}:
        return None
    if any(bond.GetBondType() != Chem.BondType.SINGLE or int(bond.GetStereo())
           for bond in iter_bonds(mol)
           if bond.GetBeginAtomIdx() in atoms and bond.GetEndAtomIdx() in atoms):
        return None
    candidates = records(mol, Chem, scope=atoms, attachment=root)
    if len(candidates) != 1:
        return None
    record = candidates[0]
    construction = record.construction
    if (not record.annotation_coverage or not record.hydrogen_budget_proved
            or record.preference_issues or construction is None
            or construction.atoms | construction.embedded_atoms != atoms):
        return None
    return atoms


def _ordinary_carbon_branch(mol, root, blocked, Chem):
    """Acyclic saturated C/H fragment with one cut at the urea nitrogen."""
    atoms = set()
    stack = [root]
    while stack:
        idx = stack.pop()
        if idx in atoms:
            continue
        atom = mol.GetAtomWithIdx(idx)
        if (atom.GetAtomicNum() != 6 or atom.GetFormalCharge()
                or atom.GetNumRadicalElectrons() or atom.IsInRing()):
            return None
        atoms.add(idx)
        for neighbor in atom.GetNeighbors():
            other = neighbor.GetIdx()
            if other == blocked:
                continue
            if neighbor.GetAtomicNum() == 1:
                if neighbor.GetIsotope():
                    return None
                continue
            bond = mol.GetBondBetweenAtoms(idx, other)
            if neighbor.GetAtomicNum() != 6 or bond.GetBondType() != Chem.BondType.SINGLE:
                return None
            stack.append(other)
    return frozenset(atoms)


def _unlocanted_substitution_prefix(mol, substitutions):
    """Multiply whole N ligands after the source-proved locant omission.

    Four decyl ligands remain four components; the numeral prefix must not
    merge with decyl into the different fourteen-carbon organyl word.
    """
    from collections import Counter
    from .organyl_prefix_components import organyl_prefix_nature
    counts = Counter(sub.text for sub in substitutions)
    parts = []
    for text in sorted(counts, key=prefix_alphabetization_key):
        natures = {organyl_prefix_nature(mol, sub.root, sub.atoms, text)
                   for sub in substitutions if sub.text == text}
        nature = next(iter(natures)) if len(natures) == 1 else None
        piece = multiply_component(text, counts[text], nature=nature)
        if not piece and nature is not None:
            # Existing isotope and characteristic vocabularies retain their
            # established grammar outside the typed organyl writer's scope.
            piece = multiply_component(text, counts[text])
        if not piece:
            return None
        # Different unlocanted ligands remain separately owned components.
        if parts and counts[text] == 1:
            from .formatting import enclose_name_unit
            piece = enclose_name_unit(piece)
        parts.append(piece)
    return ''.join(parts)


def _name_order(left, right):
    """P-14.5 ordering, with the explicit P-82.2.2.1 isotope tie-break."""
    a = prefix_alphabetization_key(left)
    b = prefix_alphabetization_key(right)
    if a[:3] < b[:3]:
        return -1
    if a[:3] > b[:3]:
        return 1
    if left == right:
        return 0
    left_iso = _has_isotope_descriptor(left)
    right_iso = _has_isotope_descriptor(right)
    if left_iso != right_iso:
        modified = left if left_iso else right
        unmodified = right if left_iso else left
        if _strip_isotope_descriptors(modified) != unmodified:
            return None
        return -1 if left_iso else 1
    # P-14.5 excludes stereodescriptors from alphabetization. If two distinct
    # stereochemical/annotated forms remain tied, do not invent an order.
    return None


def _has_isotope_descriptor(name):
    import re
    return bool(re.search(r"\(\d+(?:,\d+)*-?\d*[A-Z][a-z]?", name))


def _strip_isotope_descriptors(name):
    import re
    return re.sub(r"\(\d+(?:,\d+)*-?\d*[A-Z][a-z]?\d*\)", "", name)


def _substitution_prefix(sides):
    left, right = sides
    if len(left) > 2 or len(right) > 2:
        return None
    if not left and not right:
        return ""
    # A single substituted nitrogen is locanted N; two equivalent substituents
    # on it use N,N. The equivalent nitrogen sides are otherwise unordered.
    if left and not right or right and not left:
        side = left or right
        if len(side) == 1:
            return _n_piece("N", side[0])
        if side[0] == side[1]:
            return _repeat_piece("N,N", side[0])
        relation = _name_order(side[0], side[1])
        if relation is None:
            return None
        first, second = (side[0], side[1]) if relation < 0 else (side[1], side[0])
        return _n_piece("N", first) + "-" + _n_piece("N", second)
    if len(left) == 1 and len(right) == 1:
        order = _name_order(left[0], right[0])
        if order is None:
            return None
        if order == 0:
            return _repeat_piece("N,N′", left[0])
        first, second = (left[0], right[0]) if order < 0 else (right[0], left[0])
        return _n_piece("N", first) + "-" + _n_piece("N′", second)
    # Unlike groups on one N and mixed occupancy across N/N′ are declined:
    # their source-supported N/N′ citation choice is not established here.
    return None


def _named_nitrogen_order(nitrogens, sides):
    """Map printed N/N′ roles onto the original N vertices deterministically."""
    if bool(sides[0]) != bool(sides[1]):
        occupied = 0 if sides[0] else 1
        return nitrogens[occupied], nitrogens[1 - occupied]
    if len(sides[0]) == len(sides[1]) == 1:
        relation = _name_order(sides[0][0], sides[1][0])
        if relation is None:
            return (nitrogens[0], nitrogens[1])
        if relation > 0:
            return nitrogens[1], nitrogens[0]
        return nitrogens[0], nitrogens[1]
    # Same-site substitutions use that occupied nitrogen as N; equivalent
    # unoccupied sites are resolved by original index for traceability.
    order = sorted(range(2), key=lambda i: (not bool(sides[i]), i))
    return nitrogens[order[0]], nitrogens[order[1]]


def _n_piece(locant, name):
    return format_prefixes([(locant, name)], hierarchical_enclosures=True)


def _repeat_piece(locants, name):
    return format_prefixes([(locant, name) for locant in locants.split(",")],
                           hierarchical_enclosures=True)
