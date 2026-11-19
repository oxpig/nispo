"""Mapped single-carbon diaryl ketone general names (P-64.2.2.1).

Both aromatic organyl arms are original-graph one-cut components. The
one-carbon parent and its oxo atom have no competing skeletal locant, so the
otherwise redundant parent and suffix locants can be omitted (P-14.3.4.2).
This proves a general construction, not the entire preferred method or ring
parent comparison.
"""


def diaryl_ketone_site(mol):
    """Cheap necessary graph condition for the narrower proof below."""
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6 or atom.IsInRing() or atom.GetDegree() != 3:
            continue
        neighbors = list(atom.GetNeighbors())
        oxy = [n for n in neighbors if n.GetAtomicNum() == 8 and n.GetDegree() == 1
                and mol.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 2]
        aryl = [n for n in neighbors if n.GetAtomicNum() == 6 and n.GetIsAromatic()
                and n.IsInRing() and mol.GetBondBetweenAtoms(
                    atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 1]
        if len(oxy) == 1 and len(aryl) == 2:
            return atom.GetIdx(), oxy[0].GetIdx(), tuple(n.GetIdx() for n in aryl)
    return None


def diaryl_methanone_scope(mol, Chem):
    """Admit only source-closed class, bonding and annotation patterns."""
    from .extended_inventory import selection_inventory

    site = diaryl_ketone_site(mol)
    if (site is None or mol.GetNumHeavyAtoms() > 100 or len(Chem.GetMolFrags(mol)) != 1
            or mol.GetStereoGroups() or any(a.GetAtomicNum() not in {6, 7, 8, 9, 17, 35, 53}
            or a.GetIsotope() or a.GetFormalCharge() or a.GetNumRadicalElectrons()
            or a.GetAtomMapNum() or int(a.GetChiralTag()) for a in mol.GetAtoms())):
        return None
    carbon, oxygen, roots = site
    inventory = selection_inventory(mol)
    # A singleton supported inventory avoids assuming an unimplemented
    # within-class comparison or projecting priority from source name text.
    if (not inventory.supported or len(inventory.groups) != 1
            or inventory.groups[0].kind != 'ketone' or inventory.groups[0].center != carbon
            or tuple(inventory.principal) != tuple(inventory.groups)
            or len(inventory.source_group_facts) != 1
            or inventory.source_group_facts[0] is None
            or inventory.source_group_facts[0].class_rank != 16):
        return None
    azo = []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        order = bond.GetBondTypeAsDouble()
        if bond.GetIsAromatic() or order == 1:
            continue
        ids = {a.GetIdx(), b.GetIdx()}
        if order == 2 and ids == {carbon, oxygen}:
            continue
        if (order == 2 and a.GetAtomicNum() == b.GetAtomicNum() == 7
                and not a.IsInRing() and not b.IsInRing()):
            azo.append(bond)
            continue
        return None
    if len(azo) > 1:
        return None
    azo_atoms = ({azo[0].GetBeginAtomIdx(), azo[0].GetEndAtomIdx()} if azo else set())
    # The selected source scope does not infer the class of other nitrogen,
    # oxygen or halogen bonding environments from a coincidental match.
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        if z == 7 and not (atom.GetIsAromatic() and atom.IsInRing()) and atom.GetIdx() not in azo_atoms:
            return None
        if z == 8 and atom.GetIdx() != oxygen and not (atom.GetDegree() == 2
                and all(b.GetBondTypeAsDouble() == 1 or b.GetIsAromatic() for b in atom.GetBonds())):
            return None
        if z in {9, 17, 35, 53} and (atom.GetDegree() != 1
                or atom.GetBonds()[0].GetBondTypeAsDouble() != 1):
            return None
    potential = Chem.FindPotentialStereo(mol)
    if azo:
        if (len(potential) != 1 or str(potential[0].type) != 'Bond_Double'
                or potential[0].centeredOn != azo[0].GetIdx()
                or str(potential[0].specified) != 'Specified'):
            return None
    elif potential:
        return None
    if any(int(b.GetStereo()) and (not azo or b.GetIdx() != azo[0].GetIdx())
           or int(b.GetBondDir()) and not ({b.GetBeginAtomIdx(), b.GetEndAtomIdx()} & azo_atoms)
           for b in mol.GetBonds()):
        return None
    return carbon, oxygen, roots, tuple(azo)


def _proved_ring_arm(mol, Chem, root, host, text, atoms):
    """Join the ring writer's hydrogen, annotations and original branches."""
    from collections import Counter
    from .bounded_mixed_acid_ester_fast_guard import _one_cut_branch
    from .component_mapping import mapped_branch_ring_atoms
    from .mapped_ring_parents import records

    candidates = records(mol, Chem, scope=atoms, attachment=root)
    if len(candidates) != 1:
        return False
    record = candidates[0]
    component = record.construction
    if (record.name != text or component is None or not record.hydrogen_budget_proved
            or not record.annotation_coverage or record.preference_issues
            or component.atoms & component.embedded_atoms
            or component.atoms | component.embedded_atoms != atoms):
        return False
    locants = dict(component.atom_locants)
    owned, prefixes = set(), Counter()
    for branch_root, branch_host, branch_atoms, branch_text in component.prefix_branch_ownership:
        branch_atoms = frozenset(branch_atoms)
        if (branch_host not in component.atoms or branch_host not in locants
                or owned & branch_atoms or not _one_cut_branch(
                    mol, Chem, (branch_root, branch_host, branch_atoms, branch_text))):
            return False
        rings = {i for i in branch_atoms if mol.GetAtomWithIdx(i).IsInRing()}
        if not rings <= mapped_branch_ring_atoms(
                mol, Chem, branch_root, branch_host, branch_text, branch_atoms):
            return False
        prefixes[(locants[branch_host], branch_text)] += 1
        owned.update(branch_atoms)
    return owned == set(component.embedded_atoms) and prefixes == Counter(component.prefixes)


def diaryl_methanone_decision(mol, Chem):
    from .bounded_mixed_acid_ester_fast_guard import _one_cut_branch
    from .component_mapping import mapped_branch, mapped_branch_ring_atoms
    from .formatting import format_prefixes
    from .naming_methods import MethodDecision
    from .naming_result import SelectionLimitation
    from .stereochemistry import assigned_bond_stereochemistry, _rendered_mapped_branch_bonds

    scope = diaryl_methanone_scope(mol, Chem)
    if scope is None:
        return None
    carbon, oxygen, roots, azo = scope
    arms = []
    for root in roots:
        text, atoms = mapped_branch(mol, Chem, root, carbon)
        row = root, carbon, atoms, text
        rings = {i for i in atoms if mol.GetAtomWithIdx(i).IsInRing()}
        if (not _one_cut_branch(mol, Chem, row)
                or not rings <= mapped_branch_ring_atoms(mol, Chem, root, carbon, text, atoms)
                or not _proved_ring_arm(mol, Chem, root, carbon, text, frozenset(atoms))):
            return None
        arms.append((text, frozenset(atoms)))
    (text_a, atoms_a), (text_b, atoms_b) = arms
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    if (text_a == text_b or atoms_a & atoms_b or {carbon, oxygen} & (atoms_a | atoms_b)
            or atoms_a | atoms_b | {carbon, oxygen} != heavy):
        return None
    name = format_prefixes([('', text_a), ('', text_b)], omit_locants=True,
                           separate_unlocanted=True, hierarchical_enclosures=True) + 'methanone'
    if azo:
        assigned = {(frozenset((a, b)), label)
                    for a, b, label in assigned_bond_stereochemistry(mol, Chem)}
        owned = set(_rendered_mapped_branch_bonds(mol, name).items())
        if len(assigned) != 1 or assigned != owned:
            return None
    elif assigned_bond_stereochemistry(mol, Chem):
        return None
    roles = (('methanone', frozenset({carbon, oxygen}), ((carbon, '1'),)),
             ('organyl-a', atoms_a, ()), ('organyl-b', atoms_b, ()))
    issues = (SelectionLimitation('diaryl-ketone-parent', 'P-44/P-45/P-64.2.2.1',
        'The complete ketone and both substituent arms are mapped; whole-parent and method preference remain unresolved.'),)
    return MethodDecision(name, 'substitutive', frozenset({carbon}), heavy,
        'P-41/P-14.3.4.2/P-64.2.2.1/P-68.3.1.3', ((carbon, '1'),),
        component_roles=roles, preference_issues=issues)
