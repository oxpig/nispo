"""Numbered beta-carboline graph to the preferred fused hydride H state.

The old beta-carboline word is a general-nomenclature alias, not a preferred
fusion component (P-25.2.1 and the 1998 fused-ring recommendations FR-2.2.5).
For an otherwise aromatic indole with a fully hydrogenated pyridine ring,
P-58.2.1.2 assigns the lowest available indicated-H locant before the hydro
locants. Other hydrogenation, charge, isotope and direct functional-group
states are intentionally left to their existing producers.
"""

_ALIASES = frozenset({
    "b-carbolin", "b-carboline", "beta-carbolin", "beta-carboline",
    "betacarbolin", "betacarboline",
})
_LABELS = ("1", "2", "3", "4", "4a", "4b", "5", "6", "7", "8", "8a", "9", "9a")
_RING_EDGES = frozenset(
    frozenset((a, b)) for a, b in zip(_LABELS, _LABELS[1:] + _LABELS[:1])
) | frozenset({frozenset(("4a", "9a")), frozenset(("4b", "8a"))})
_HYDRO = frozenset({"1", "2", "3", "4"})
_REFERENCE_H = {label: int(label in {"1", "3", "4", "5", "6", "7", "8", "9"})
                for label in _LABELS}


def preferred_saturated_pyridoindole_parent(mol, template, match):
    """Prove one neutral hydrogenated frame and return its fused hydride name.

    The capped H population counts each external single bond as one H on the
    corresponding parent, so this applies to substituted and chiral examples
    without using a particular molecule or changing its substituents. A direct
    functional suffix may require joint indicated-H accommodation and is left
    outside the scope of this producer.
    """
    name, query, locants = template.get("name"), template.get("mol"), template.get("locants", {})
    if (name not in _ALIASES or query is None or query.GetNumAtoms() != len(_LABELS)
            or len(match) != len(_LABELS) or len(set(match)) != len(_LABELS)
            or set(locants) != set(range(len(_LABELS))) or set(locants.values()) != set(_LABELS)):
        return ""
    at = {locants[i]: i for i in range(len(_LABELS))}
    if any(query.GetAtomWithIdx(at[loc]).GetAtomicNum() != (7 if loc in {"2", "9"} else 6)
           or not query.GetAtomWithIdx(at[loc]).GetIsAromatic()
           or query.GetAtomWithIdx(at[loc]).GetFormalCharge()
           or query.GetAtomWithIdx(at[loc]).GetTotalNumHs() != _REFERENCE_H[loc]
           for loc in _LABELS):
        return ""
    edges = frozenset(frozenset((locants[b.GetBeginAtomIdx()], locants[b.GetEndAtomIdx()]))
                      for b in query.GetBonds())
    if edges != _RING_EDGES or any(not bond.GetIsAromatic() for bond in query.GetBonds()):
        return ""
    core = set(match)
    for loc, j in at.items():
        atom, reference = mol.GetAtomWithIdx(match[j]), query.GetAtomWithIdx(j)
        if (atom.GetAtomicNum() != reference.GetAtomicNum() or atom.GetFormalCharge()
                or atom.GetIsotope() or atom.GetNumRadicalElectrons()
                or atom.GetIsAromatic() != (loc not in _HYDRO)
                or any(n.GetAtomicNum() == 1 and n.GetIsotope() for n in atom.GetNeighbors())):
            return ""
        external = [bond for bond in atom.GetBonds() if bond.GetOtherAtomIdx(atom.GetIdx()) not in core]
        if any(bond.GetBondTypeAsDouble() != 1 for bond in external):
            return ""
        # Direct heteroatom and carbonyl substituents may be suffixes; their
        # joint indicated-H choice is not resolved by this hydride-only proof.
        for bond in external:
            neighbor = bond.GetOtherAtom(atom)
            if neighbor.GetAtomicNum() != 6 or any(
                not b.GetIsAromatic() and b.GetBondTypeAsDouble() > 1
                and b.GetOtherAtom(neighbor).GetAtomicNum() != 6
                for b in neighbor.GetBonds()
            ):
                return ""
        capped_h = atom.GetTotalNumHs() + len(external)
        if capped_h != reference.GetTotalNumHs() + (loc in _HYDRO):
            return ""
    for edge in _RING_EDGES:
        left, right = tuple(edge)
        bond = mol.GetBondBetweenAtoms(match[at[left]], match[at[right]])
        if bond is None or (not bond.GetIsAromatic() if not (edge & _HYDRO)
                            else bond.GetBondTypeAsDouble() != 1):
            return ""
    return "2,3,4,9-tetrahydro-1H-pyrido[3,4-b]indole"
