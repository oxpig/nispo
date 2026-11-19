"""Mapped P-63.3.1 substituent units for neutral peroxides/disulfides."""


def _single(mol, a, b):
    bond = mol.GetBondBetweenAtoms(a, b)
    return bond is not None and bond.GetBondTypeAsDouble() == 1


def _linker_atom(atom, element):
    return (atom.GetAtomicNum() == element and not atom.GetFormalCharge()
            and not atom.GetNumRadicalElectrons() and not atom.GetIsotope()
            and not int(atom.GetChiralTag()) and not atom.IsInRing()
            and atom.GetDegree() == 2 and atom.GetTotalValence() == 2)


def peroxide_bridge(mol, root, blocked):
    """Describe a directly attached C–O–O–C or C–S–S–C bridge.

    Return `(bridge_element, distal_carbon)` or `None`. Terminal O–H groups,
    mixed O/S linkers, charges, linker annotations, and non-single bonds are
    rejected at recognition time.
    """
    if root < 0 or blocked < 0 or not _single(mol, root, blocked):
        return None
    atom = mol.GetAtomWithIdx(root)
    host = mol.GetAtomWithIdx(blocked)
    element = atom.GetAtomicNum()
    if element not in {8, 16} or host.GetAtomicNum() != 6 or not _linker_atom(atom, element):
        return None
    if host.GetFormalCharge() or host.GetNumRadicalElectrons():
        return None
    heavy = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
    if len(heavy) != 2 or host.GetIdx() not in {n.GetIdx() for n in heavy}:
        return None
    bridge = next((n for n in heavy if n.GetIdx() != blocked), None)
    if bridge is None or not _linker_atom(bridge, element) or not _single(mol, root, bridge.GetIdx()):
        return None
    bridge_heavy = [n for n in bridge.GetNeighbors() if n.GetAtomicNum() > 1]
    distal = [n for n in bridge_heavy if n.GetIdx() != root]
    if (len(bridge_heavy) != 2 or len(distal) != 1
            or distal[0].GetAtomicNum() != 6
            or not _single(mol, bridge.GetIdx(), distal[0].GetIdx())):
        return None
    return element, distal[0].GetIdx()


def mapped_peroxide_prefix(mol, Chem, root, blocked, atoms):
    """Render a P-63.3.1 peroxy/disulfanyl prefix with mapped carbon ownership.

    `root` is the bridge atom attached to the parent-side carbon; `blocked` is
    that carbon. The returned prefix unit is undecorated at its outer boundary
    (for example `methylperoxy` or `methyldisulfanyl`); the caller encloses it
    as a complete prefix unit. Distal carbon isotope/stereo annotations are
    produced by `mapped_branch` against the original molecule.
    """
    bridge = peroxide_bridge(mol, root, blocked)
    if bridge is None:
        return "", frozenset()
    element, distal = bridge
    from .component_mapping import mapped_branch

    bridge_atom = next(n.GetIdx() for n in mol.GetAtomWithIdx(root).GetNeighbors()
                       if n.GetIdx() != blocked and n.GetAtomicNum() == element)
    text, owned = mapped_branch(mol, Chem, distal, bridge_atom)
    if not text:
        return "", frozenset()
    expected = set(owned) | {root, bridge_atom}
    if expected != set(atoms):
        return "", frozenset()
    term = "peroxy" if element == 8 else "disulfanyl"
    # P-63.3.1 compounds the distal substituent name additively with the
    # bridge term. A complex substituent remains an inner enclosed unit.
    compound = f"({text}){term}" if any(mark in text for mark in "-()[]{} ") else f"{text}{term}"
    return compound, frozenset(expected)
