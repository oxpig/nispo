"""Mapped P-32 preferred prefixes for bounded unsaturated carbon trees.

This module names one complete carbon branch attached through a single bond.
It delegates the candidate path comparison to the P-46 frontier and keeps all
surviving atom-map alternatives; it does not select an arbitrary graph path.
"""

from dataclasses import dataclass

from .constants import ROOTS
from .formatting import attach_suffix, format_prefixes, loc_unsat, unsaturated_parent
from .unsaturated_substituent_selection import unsaturated_substituent_frontier


RULE = "P-32.1.1/P-57.1.1.1/P-57.1.6.1/P-46.1"
_ALKYL = {
    1: "methyl", 2: "ethyl", 3: "propyl", 4: "butyl", 5: "pentyl",
    6: "hexyl", 7: "heptyl", 8: "octyl", 9: "nonyl", 10: "decyl",
}


@dataclass(frozen=True, slots=True)
class UnsaturatedSubstituent:
    """Preferred prefix text plus complete original-graph ownership facts."""

    text: str
    atoms: frozenset[int]
    root_atom_id: int
    attachment_atom_id: int
    atom_locants: tuple[tuple[int, str], ...]
    component_roles: tuple
    prefix_roles: tuple
    nature: str
    candidate_mappings: tuple
    rule: str = RULE
    proof: tuple = ()


def _scope(mol, Chem, root_id, attachment_id):
    if (isinstance(root_id, bool) or not isinstance(root_id, int)
            or isinstance(attachment_id, bool) or not isinstance(attachment_id, int)
            or not (0 <= root_id < mol.GetNumAtoms())
            or not (0 <= attachment_id < mol.GetNumAtoms())
            or root_id == attachment_id):
        return None
    bond = mol.GetBondBetweenAtoms(root_id, attachment_id)
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return None
    seen, stack = set(), [root_id]
    while stack:
        idx = stack.pop()
        if idx in seen:
            continue
        seen.add(idx)
        for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
            if neighbor.GetAtomicNum() <= 1:
                continue
            other = neighbor.GetIdx()
            if other == attachment_id:
                if idx != root_id:
                    return None
                continue
            if neighbor.GetAtomicNum() > 1 and other not in seen:
                stack.append(other)
    if len(seen) > 32:
        return None
    for idx in seen:
        atom = mol.GetAtomWithIdx(idx)
        if (atom.GetAtomicNum() != 6 or atom.GetFormalCharge() != 0
                or atom.GetNumRadicalElectrons() or atom.GetIsotope()
                or int(atom.GetChiralTag()) or atom.IsInRing()
                or atom.GetTotalValence() != 4):
            return None
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() == 1 and neighbor.GetIsotope():
                return None
            if neighbor.GetAtomicNum() > 1 and neighbor.GetIdx() not in seen | {attachment_id}:
                return None
            if neighbor.GetIdx() == attachment_id and idx != root_id:
                return None
    return frozenset(seen)


def _component_name(mol, Chem, atom_ids, host_id):
    """Prove a linear, saturated terminal alkyl component of 1–10 carbons."""
    atoms = frozenset(atom_ids)
    if not 1 <= len(atoms) <= 10:
        return None
    degree = {}
    host_bonds = 0
    for idx in atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() != 6 or atom.GetFormalCharge() or atom.GetIsotope():
            return None
        if int(atom.GetChiralTag()) or atom.IsInRing() or atom.GetNumRadicalElectrons():
            return None
        internal = 0
        for neighbor in atom.GetNeighbors():
            other = neighbor.GetIdx()
            bond = mol.GetBondBetweenAtoms(idx, other)
            if other in atoms:
                if bond.GetBondType() != Chem.BondType.SINGLE:
                    return None
                internal += 1
            elif other == host_id:
                if bond.GetBondType() != Chem.BondType.SINGLE:
                    return None
                host_bonds += 1
            elif neighbor.GetAtomicNum() > 1:
                return None
        degree[idx] = internal
    if host_bonds != 1 or any(value > 2 for value in degree.values()):
        return None
    attach_atoms = [idx for idx in atoms if mol.GetBondBetweenAtoms(idx, host_id)]
    if len(attach_atoms) != 1 or degree[attach_atoms[0]] > 1:
        return None
    seen, stack = set(), [next(iter(atoms))]
    while stack:
        idx = stack.pop()
        if idx in seen:
            continue
        seen.add(idx)
        stack.extend(n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                     if n.GetIdx() in atoms)
    if seen != set(atoms):
        return None
    return _ALKYL.get(len(atoms))


def _prefix_names(mol, Chem, candidates):
    names = {}
    for candidate in candidates:
        for prefix in candidate.prefixes:
            key = tuple(sorted(prefix.atom_ids))
            text = _component_name(mol, Chem, key, prefix.host_atom_id)
            if text is None or (key in names and names[key] != text):
                return None
            names[key] = text
    return names


def unsaturated_carbon_stem(length, free_locant, ene, yne):
    """P-29/P-32 suffix construction shared by complete carbon components."""
    if length not in ROOTS:
        return None
    if (length == 2 and ene == (1,) and not yne):
        # P-32.3 specifically gives ethenyl as the preferred systematic term.
        if free_locant == 1:
            return "ethenyl"
    root = ROOTS[length]
    if ene and yne:
        parent_stem = unsaturated_parent(root,list(ene),list(yne))
    elif ene:
        parent_stem = loc_unsat(root, list(ene), "ene", terminal_e=False)
    elif yne:
        parent_stem = loc_unsat(root, list(yne), "yne", terminal_e=False)
    else:
        return None
    return attach_suffix(parent_stem, "yl", (free_locant,))


def _render(mol, candidate):
    ene = candidate.double_bond_locants
    yne = tuple(loc for loc in candidate.multiple_bond_locants if loc not in ene)
    stem = unsaturated_carbon_stem(len(candidate.path_atom_ids),
                                   candidate.free_valence_locants[0], ene, yne)
    if not stem:
        return None
    prefixes = [(str(prefix.locant), prefix.proved_name or "")
                for prefix in candidate.prefixes]
    if any(not text for _, text in prefixes):
        return None
    return format_prefixes(prefixes) + stem


def unsaturated_carbon_substituent(mol, Chem, root_id, attachment_id):
    """Return a proved mapped P-32 prefix for one unsaturated carbon branch.

    The branch must be a complete neutral, unannotated carbon tree with one
    attachment bond, at least one isolated C=C/C#C, and no cycles, cumulenes,
    isotopic atoms, or stereochemical assignments. A name is returned only
    when the complete Job 146 method-2 P-46 frontier is resolved and every
    surviving candidate has identical rendered text.
    """
    atoms = _scope(mol, Chem, root_id, attachment_id)
    if atoms is None:
        return None
    has_unsaturation = False
    for bond in mol.GetBonds():
        if bond.GetBeginAtomIdx() in atoms and bond.GetEndAtomIdx() in atoms:
            if int(bond.GetStereo()):
                return None
            if bond.GetBondTypeAsDouble() in {2.0, 3.0}:
                has_unsaturation = True
    if not has_unsaturation:
        return None
    initial = unsaturated_substituent_frontier(
        mol, root_id, attachment_id, atoms, method=2,
    )
    if initial is None or initial.status == "incomplete" or not initial.frontier:
        return None
    names = _prefix_names(mol, Chem, initial.frontier)
    if names is None:
        return None
    selected = unsaturated_substituent_frontier(
        mol, root_id, attachment_id, atoms, method=2, prefix_names=names,
    )
    if (selected is None or selected.status in {"unresolved", "incomplete"}
            or not selected.frontier):
        return None
    rendered = tuple(_render(mol, candidate) for candidate in selected.frontier)
    if any(not text for text in rendered) or len(set(rendered)) != 1:
        return None
    nature = "compound" if any(candidate.prefixes for candidate in selected.frontier) else "simple"
    if len({candidate.path_atom_ids for candidate in selected.frontier}) > 1:
        # Preserve every tied original-graph construction; only locants shared
        # by all mappings appear in the compact common map.
        all_maps = [dict(candidate.atom_locants) for candidate in selected.frontier]
        common = set(all_maps[0].items())
        for mapping in all_maps[1:]:
            common &= set(mapping.items())
        atom_locants = tuple(sorted(((idx, str(loc)) for idx, loc in common),
                                    key=lambda row: (int(row[1]), row[0])))
    else:
        candidate = selected.frontier[0]
        atom_locants = tuple((idx, str(loc)) for idx, loc in candidate.atom_locants)
    prefix_roles = tuple(
        ("saturated-alkyl-prefix", frozenset(prefix.atom_ids), ())
        for prefix in selected.frontier[0].prefixes
    )
    whole_role = ("unsaturated-carbon-substituent", atoms, atom_locants)
    mappings = tuple(
        (candidate.path_atom_ids, candidate.atom_locants,
         candidate.free_valence_locants, candidate.multiple_bond_locants,
         candidate.double_bond_locants,
         tuple((p.atom_ids, p.host_atom_id, p.locant, p.proved_name)
               for p in candidate.prefixes))
        for candidate in selected.frontier
    )
    proof = (
        ("rules", RULE), ("selector", "Job 146 unsaturated method-2 frontier"),
        ("frontier_status", selected.status), ("candidate_mappings", mappings),
        ("branch_scope", tuple(sorted(atoms))),
    )
    return UnsaturatedSubstituent(
        rendered[0], atoms, root_id, attachment_id, atom_locants,
        (whole_role,), prefix_roles, nature, mappings, RULE, proof,
    )
