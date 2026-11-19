"""Current cyclic hydride numberings joined to one original characteristic unit."""

from .component_mapping import MappedComponent, decorate_component
from .formatting import attach_suffix
from .naming_methods import MethodDecision
from .naming_result import SelectionLimitation


def _unit(mol):
    units = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if atom.GetAtomicNum() == 7 and atom.GetDegree() == 1 and atom.GetTotalNumHs() == 2:
            host = atom.GetNeighbors()[0]
            if host.GetAtomicNum() == 6 and atom.GetBonds()[0].GetBondTypeAsDouble() == 1:
                units.append((host.GetIdx(), frozenset({idx}), 'amine', 'P-62.2.1.2'))
        if atom.GetAtomicNum() != 6 or atom.IsInRing():
            continue
        neighbors = tuple(atom.GetNeighbors())
        carbon = [a for a in neighbors if a.GetAtomicNum() == 6
                  and mol.GetBondBetweenAtoms(idx, a.GetIdx()).GetBondTypeAsDouble() == 1]
        triple_n = [a for a in neighbors if a.GetAtomicNum() == 7
                    and a.GetDegree() == 1
                    and mol.GetBondBetweenAtoms(idx, a.GetIdx()).GetBondTypeAsDouble() == 3]
        if len(neighbors) == 2 and len(carbon) == len(triple_n) == 1:
            units.append((carbon[0].GetIdx(), frozenset({idx, triple_n[0].GetIdx()}),
                          'carbonitrile', 'P-66.5.1.1.3'))
        oxo = [a for a in neighbors if a.GetAtomicNum() == 8 and a.GetDegree() == 1
               and mol.GetBondBetweenAtoms(idx, a.GetIdx()).GetBondTypeAsDouble() == 2]
        hydroxy = [a for a in neighbors if a.GetAtomicNum() == 8 and a.GetDegree() == 1
                   and a.GetTotalNumHs() == 1
                   and mol.GetBondBetweenAtoms(idx, a.GetIdx()).GetBondTypeAsDouble() == 1]
        if len(neighbors) == 3 and len(carbon) == len(oxo) == len(hydroxy) == 1:
            units.append((carbon[0].GetIdx(), frozenset({idx, oxo[0].GetIdx(), hydroxy[0].GetIdx()}),
                          'carboxylic acid', 'P-65.1.2'))
    return units[0] if len(units) == 1 else None


def cyclic_suffix_construction(mol, Chem, native_numberings):
    """Construct a one-group saturated polycyclic derivative with mapped stereo.

    Native complete hydride writers supply words and numberings, without an
    earlier preferred functional-parent barrier. An exhaustive graph
    isomorphism transfers that numbering to the original source; characteristic
    atoms remain separate from the parent. No nuclide or enhanced stereo is
    stripped from a deliverable name.
    """
    if (mol.GetStereoGroups() or len(Chem.GetMolFrags(mol)) != 1
            or mol.GetRingInfo().NumRings() < 2
            or any(a.GetIsotope() or a.GetFormalCharge() or a.GetNumRadicalElectrons()
                   for a in mol.GetAtoms())):
        return None
    unit = _unit(mol)
    if unit is None:
        return None
    host, group, suffix, rule = unit
    core = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1) - group
    if (host not in core or not core
            or any(not mol.GetAtomWithIdx(i).IsInRing()
                   or mol.GetAtomWithIdx(i).GetAtomicNum() not in {6, 7, 8} for i in core)
            or any(b.GetBondTypeAsDouble() != 1 for b in mol.GetBonds()
                   if b.GetBeginAtomIdx() in core and b.GetEndAtomIdx() in core)):
        return None
    order = tuple(sorted(core))
    old_to_new = {old: new for new, old in enumerate(order)}
    graph = Chem.RWMol()
    for old in order:
        graph.AddAtom(Chem.Atom(mol.GetAtomWithIdx(old).GetAtomicNum()))
    for bond in mol.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if left in core and right in core:
            graph.AddBond(old_to_new[left], old_to_new[right], Chem.BondType.SINGLE)
    parent = graph.GetMol()
    try:
        Chem.SanitizeMol(parent)
    except (ValueError, RuntimeError):
        return None
    if (len(Chem.GetMolFrags(parent)) != 1
            or any(parent.GetAtomWithIdx(new).GetTotalNumHs(includeNeighbors=True) !=
                   mol.GetAtomWithIdx(old).GetTotalNumHs(includeNeighbors=True) + (old == host)
                   for new, old in enumerate(order))):
        return None
    smiles = Chem.MolToSmiles(parent, canonical=True, isomericSmiles=False)
    canonical_parent = Chem.MolFromSmiles(smiles)
    if canonical_parent is None:
        return None
    matches = canonical_parent.GetSubstructMatches(parent, uniquify=False, useChirality=False, maxMatches=4097)
    if not matches or len(matches) >= 4097:
        return None
    candidates = []
    for parent_name, locants in native_numberings(smiles):
        numbered = dict(locants)
        if set(numbered) != set(range(len(order))):
            continue
        for match in matches:
            bindings = {old: numbered[match[new]] for new, old in enumerate(order)}
            locant = bindings[host]
            name = attach_suffix(parent_name, suffix, (locant,))
            complete_map = dict(bindings)
            complete_map.update((i, '') for i in group)
            name = decorate_component(mol, Chem, MappedComponent(
                name, tuple(complete_map.items()), core | group, systematic_parent=True))
            if name:
                candidates.append((int(locant), name, bindings))
    if not candidates:
        return None
    _, name, bindings = min(candidates, key=lambda row: (row[0], row[1]))
    return MethodDecision(name=name, method='substitutive', principal_centers=group,
        atoms=core | group, rule=rule + '/P-23/P-24/P-29/P-9', atom_locants=tuple(bindings.items()),
        component_roles=(('cyclic-parent', core, tuple(bindings.items())),
                         ('characteristic-suffix', group, ())),
        preference_issues=(SelectionLimitation('cyclic-parent-construction', 'P-44/P-45/P-51',
            'A complete current hydride numbering and suffix join are constructed; the global preferred construction/parent comparison remains unresolved.'),))
