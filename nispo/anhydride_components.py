"""Mapped P-65.7 single-linkage chalcogen anhydride construction.

The source-defined scope is intentionally limited to neutral acyclic linkages
between two ordinary carboxylic-acid residues. Each residue is selected and
rendered by the shared mapped acid-component writer; bridge atoms remain an
independent role on the original graph.
"""

from .molecule_iteration import iter_atoms
from .acid_derivative_methods import _acid_component
from .formatting import prefix_alphabetization_key
from .naming_methods import MethodDecision


_BRIDGE_NUMBERS = frozenset({8, 16, 34, 52})
_SINGLE_BRIDGE_CLASS = {16: "thioanhydride", 34: "selenoanhydride",
                        52: "telluroanhydride"}


def _single_bond(mol, left, right):
    bond = mol.GetBondBetweenAtoms(left, right)
    return bond is not None and bond.GetBondTypeAsDouble() == 1


def _acid_name(name):
    if name.endswith(" acid"):
        return name[:-5]
    return ""


def _classify_bridge(numbers):
    if len(numbers) == 1:
        return _SINGLE_BRIDGE_CLASS.get(numbers[0])
    pair = tuple(numbers)
    if pair == (8, 8):
        return "peroxyanhydride"
    if pair == (16, 16):
        return "dithioperoxyanhydride"
    if set(pair) == {8, 16}:
        return "thioperoxyanhydride"
    return None


def _find_endpoints(mol):
    """Find exactly two C(=O) sites singly attached to a bridge chalcogen."""
    endpoints = []
    for atom in iter_atoms(mol):
        if atom.GetAtomicNum() != 6:
            continue
        center = atom.GetIdx()
        carbonyl_o = [n.GetIdx() for n in atom.GetNeighbors()
                      if n.GetAtomicNum() == 8
                      and mol.GetBondBetweenAtoms(center, n.GetIdx()).GetBondTypeAsDouble() == 2]
        bridge_neighbors = [n.GetIdx() for n in atom.GetNeighbors()
                            if n.GetAtomicNum() in _BRIDGE_NUMBERS
                            and _single_bond(mol, center, n.GetIdx())]
        if len(carbonyl_o) == 1 and len(bridge_neighbors) == 1:
            endpoints.append((center, bridge_neighbors[0]))
    return endpoints


def _bridge_path(mol, left, right):
    """Return a linear one/two-atom chalcogen path between acyl centers."""
    path = [left]
    previous = None
    current = left
    while current != right and len(path) <= 2:
        atom = mol.GetAtomWithIdx(current)
        nexts = [n.GetIdx() for n in atom.GetNeighbors()
                 if n.GetIdx() != previous and n.GetAtomicNum() in _BRIDGE_NUMBERS
                 and _single_bond(mol, current, n.GetIdx())]
        if len(nexts) != 1:
            return None
        previous, current = current, nexts[0]
        if current in path:
            return None
        path.append(current)
    if current != right or not 1 <= len(path) <= 2:
        return None
    # Each bridge atom must have exactly its two bonds in the linkage: an
    # endpoint contributes one carbonyl carbon, while a central atom in a
    # future longer chain would be checked by the path-length guard above.
    for pos, idx in enumerate(path):
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetDegree() != 2:
            return None
    return tuple(path)


def _role_names(parts, bridge_path, mol, class_term):
    names = [_acid_name(part.name) for part in parts]
    if any(not name for name in names):
        return None
    if len(bridge_path) == 1:
        if parts[0].identity == parts[1].identity:
            return f"{names[0]} {class_term}"
        return " ".join(sorted(names, key=prefix_alphabetization_key) + [class_term])

    numbers = tuple(mol.GetAtomWithIdx(i).GetAtomicNum() for i in bridge_path)
    if parts[0].identity == parts[1].identity:
        return f"{names[0]} {class_term}"
    # For an unlike two-atom bridge, the source requires element symbols on
    # acid-group names when the attachment needs to be distinguished. Put the
    # S-linked residue first, as in the explicit P-65.7.4 S–O example; this is
    # graph-role based and independent of atom numbering.
    if set(numbers) == {8, 16}:
        labels = ["S" if number == 16 else "O" for number in numbers]
        # P-65.7.2's alphabetical acid-name order remains the ordering rule;
        # P-65.7.4's element symbols identify attachment without replacing it.
        order = sorted(range(2), key=lambda i: prefix_alphabetization_key(names[i]))
        return " ".join(f"{labels[i]}-{names[i]}" for i in order) + f" {class_term}"
    return " ".join(sorted(names, key=prefix_alphabetization_key) + [class_term])


def anhydride_decision(mol, Chem):
    """Return a mapped single-linkage chalcogen anhydride decision or None."""
    if (mol.GetStereoGroups()
            or any(a.GetFormalCharge() or a.GetNumRadicalElectrons()
                   for a in iter_atoms(mol))):
        return None
    # P-82.6.4's typed oxygen-isotope roles are specific to ester formation;
    # they do not establish isotope grammar for anhydride acid endings or
    # bridges. Carbon isotopes remain owned by their mapped acid components.
    if any(a.GetIsotope() and a.GetAtomicNum() != 6 for a in iter_atoms(mol)):
        return None
    endpoints = _find_endpoints(mol)
    if len(endpoints) != 2:
        return None
    (c1, b1), (c2, b2) = endpoints
    if c1 == c2:
        return None
    bridge_path = _bridge_path(mol, b1, b2)
    if bridge_path is None:
        return None
    if any(mol.GetAtomWithIdx(i).GetIsotope()
           or int(mol.GetAtomWithIdx(i).GetChiralTag())
           or mol.GetAtomWithIdx(i).IsInRing() for i in bridge_path):
        return None
    class_term = _classify_bridge(tuple(mol.GetAtomWithIdx(i).GetAtomicNum()
                                        for i in bridge_path))
    if class_term is None:
        return None
    # The selected bridge must be the complete connection between the two
    # mapped acid residues; _acid_component retains original atom ownership.
    parts = [_acid_component(mol, Chem, c1, b1),
             _acid_component(mol, Chem, c2, b2)]
    if any(part is None for part in parts):
        return None
    first, second = parts
    bridge_atoms = frozenset(bridge_path)
    atoms = frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1)
    if (first.atoms & second.atoms or first.atoms & bridge_atoms
            or second.atoms & bridge_atoms
            or first.atoms | second.atoms | bridge_atoms != atoms):
        return None
    name = _role_names(parts, bridge_path, mol, class_term)
    if not name:
        return None
    roles = (("acid:1", first.atoms, first.atom_locants),
             ("acid:2", second.atoms, second.atom_locants),
             ("bridge", bridge_atoms, ()))
    return MethodDecision(name, "functional-class", frozenset({c1, c2}), atoms,
                          "P-65.7.3" if len(bridge_path) == 1 else "P-65.7.4",
                          first.atom_locants + second.atom_locants,
                          component_roles=roles)
