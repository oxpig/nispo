"""Bounded mapped P-31.2 dehydro nomenclature for the benzene parent."""
from __future__ import annotations

from collections import defaultdict

from .molecule_iteration import iter_atoms, iter_bonds
from .component_mapping import mapped_branch
from .formatting import format_prefixes, prefix_alphabetization_key
from .naming_methods import MethodDecision


RULE = "P-14.4(e)-(g)/P-14.5/P-22.1.2/P-22.1.3/P-31.2.1/P-31.2.2/P-31.2.4.1"
_HALOGENS = {9, 17, 35, 53}


def dehydro_parent_decision(mol, Chem):
    """Name a bounded benzene didehydro parent with simple ring substituents.

    The parent proof is one six-carbon cycle with one triple, two double, and
    three single bonds. At each of the four remaining ring atoms, admit no
    substituent, one halogen, or one short unbranched saturated alkyl group.
    Both triple-end orientations are compared under P-14.4 before rendering.
    This opt-in producer does not displace ordinary en/yne routes.
    """
    if mol.GetNumHeavyAtoms() > 22 or mol.GetNumAtoms() != mol.GetNumHeavyAtoms():
        return None
    if mol.GetStereoGroups():
        return None
    for atom in iter_atoms(mol):
        if (atom.GetAtomicNum() not in {6} | _HALOGENS or atom.GetFormalCharge()
                or atom.GetNumRadicalElectrons() or atom.GetIsotope()
                or int(atom.GetChiralTag())):
            return None
    if any(int(bond.GetStereo()) or int(bond.GetBondDir()) for bond in iter_bonds(mol)):
        return None

    # Aromatic input bonds have order 1.5; Kekulize a private copy before
    # proving the explicit valence operation. The original molecule is read-only.
    probe = Chem.Mol(mol)
    try:
        Chem.Kekulize(probe, clearAromaticFlags=True)
    except Exception:
        return None
    rings = [tuple(ring) for ring in probe.GetRingInfo().AtomRings()]
    if (len(rings) != 1 or len(rings[0]) != 6
            or any(probe.GetAtomWithIdx(idx).GetAtomicNum() != 6 for idx in rings[0])):
        return None
    ring = frozenset(rings[0])
    ring_bonds = [bond for bond in probe.GetBonds()
                  if bond.GetBeginAtomIdx() in ring and bond.GetEndAtomIdx() in ring]
    if len(ring_bonds) != 6 or sorted(b.GetBondTypeAsDouble() for b in ring_bonds) != [
            1.0, 1.0, 1.0, 2.0, 2.0, 3.0]:
        return None
    triple = [bond for bond in ring_bonds if bond.GetBondTypeAsDouble() == 3.0]
    if len(triple) != 1:
        return None
    triple_endpoints = {triple[0].GetBeginAtomIdx(), triple[0].GetEndAtomIdx()}

    # Validate all substituent components before applying parent numbering.
    branches = []
    used = set(ring)
    ring_adj = {idx: set() for idx in ring}
    for bond in ring_bonds:
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ring_adj[a].add(b)
        ring_adj[b].add(a)
    if any(len(neighbors) != 2 for neighbors in ring_adj.values()):
        return None
    for idx in ring:
        atom = probe.GetAtomWithIdx(idx)
        external = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() not in ring]
        if len(external) > 1:
            return None
        if external:
            if idx in triple_endpoints:
                return None
            root = external[0]
            if root in used:
                return None
            branch, text = _substituent_branch(probe, Chem, root, idx)
            if branch is None:
                return None
            used.update(branch)
            branches.append((idx, text, branch))
        expected_h = 0 if idx in triple_endpoints or external else 1
        if atom.GetTotalNumHs(includeNeighbors=True) != expected_h:
            return None
        if external:
            bond = probe.GetBondBetweenAtoms(idx, external[0])
            if bond is None or bond.GetBondTypeAsDouble() != 1.0:
                return None
    if used != {atom.GetIdx() for atom in probe.GetAtoms()}:
        return None

    a, b = triple[0].GetBeginAtomIdx(), triple[0].GetEndAtomIdx()
    candidates = []
    for one, two in ((a, b), (b, a)):
        locant_to_atom = {1: one, 2: two}
        previous, current = one, two
        valid = True
        for locant in range(3, 7):
            choices = sorted(neighbor for neighbor in ring_adj[current] if neighbor != previous)
            if len(choices) != 1:
                valid = False
                break
            previous, current = current, choices[0]
            if current in locant_to_atom.values():
                valid = False
                break
            locant_to_atom[locant] = current
        if not valid or ring_adj[current] - {previous} != {one}:
            continue
        if set(locant_to_atom.values()) != set(ring):
            continue
        prefix_items = [(str(next(k for k, v in locant_to_atom.items() if v == owner)), text)
                        for owner, text, _branch in branches]
        all_locants = tuple(sorted(int(loc) for loc, _text in prefix_items))
        by_name = defaultdict(list)
        for loc, text in prefix_items:
            by_name[text].append(int(loc))
        ordered_names = sorted(by_name, key=prefix_alphabetization_key)
        per_prefix = tuple(tuple(sorted(by_name[name])) for name in ordered_names)
        # P-14.4(e): dehydro locants; (f): all prefix locants together; (g):
        # if needed, prefixes in citation order receive low locants first.
        comparison_key = ((1, 2), all_locants, per_prefix)
        prefix_text = format_prefixes(prefix_items)
        name = f"{prefix_text}-1,2-didehydrobenzene" if prefix_text else "1,2-didehydrobenzene"
        atom_locants = tuple((idx, str(locant))
                             for locant, idx in sorted(locant_to_atom.items()))
        candidates.append((comparison_key, name, atom_locants, locant_to_atom))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    best_key = candidates[0][0]
    best_names = {item[1] for item in candidates if item[0] == best_key}
    if len(best_names) != 1:
        return None
    _, name, atom_locants, locant_to_atom = candidates[0]
    parent_atoms = frozenset(ring)
    role_entries = [("dehydro-parent", parent_atoms, atom_locants)]
    for owner, text, branch in branches:
        role_entries.append((f"substituent:{text}", frozenset(branch), ()))
    atoms = frozenset(used)
    return MethodDecision(
        name=name,
        method="substitutive",
        principal_centers=frozenset(),
        atoms=atoms,
        rule=RULE,
        atom_locants=atom_locants,
        component_roles=tuple(role_entries),
    )


def _substituent_branch(mol, Chem, root, host):
    """Recognize one halogen or a small unbranched saturated C1-C4 alkyl."""
    atom = mol.GetAtomWithIdx(root)
    if atom.GetAtomicNum() in _HALOGENS:
        if (atom.GetDegree() != 1 or atom.GetIsotope() or atom.GetFormalCharge()
                or mol.GetBondBetweenAtoms(root, host).GetBondTypeAsDouble() != 1.0):
            return None, ""
        name, owned = mapped_branch(mol, Chem, root, host)
        return (frozenset(owned), name) if name and owned == {root} else (None, "")
    if atom.GetAtomicNum() != 6:
        return None, ""
    atoms, todo = {root}, [root]
    while todo:
        idx = todo.pop()
        current = mol.GetAtomWithIdx(idx)
        if current.GetAtomicNum() != 6 or current.IsInRing():
            return None, ""
        for neighbor in current.GetNeighbors():
            other = neighbor.GetIdx()
            if other == host or neighbor.GetAtomicNum() == 1:
                continue
            if neighbor.GetAtomicNum() != 6:
                return None, ""
            if other not in atoms:
                atoms.add(other)
                todo.append(other)
    if len(atoms) > 4:
        return None, ""
    internal_bonds = [bond for bond in iter_bonds(mol)
                      if bond.GetBeginAtomIdx() in atoms and bond.GetEndAtomIdx() in atoms]
    if len(internal_bonds) != len(atoms) - 1 or any(
            bond.GetBondTypeAsDouble() != 1.0 for bond in internal_bonds):
        return None, ""
    external = [(idx, neighbor.GetIdx()) for idx in atoms
                for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors()
                if neighbor.GetIdx() not in atoms]
    if external != [(root, host)] and set(external) != {(root, host)}:
        return None, ""
    if any(sum(n.GetIdx() in atoms for n in mol.GetAtomWithIdx(idx).GetNeighbors()) > 2
           for idx in atoms):
        return None, ""
    if mol.GetBondBetweenAtoms(root, host).GetBondTypeAsDouble() != 1.0:
        return None, ""
    name, owned = mapped_branch(mol, Chem, root, host)
    if not name or frozenset(owned) != frozenset(atoms):
        return None, ""
    return frozenset(atoms), name
