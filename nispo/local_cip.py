"""Local CIP labelling for opposite sites of an isolated symmetric ring."""

from .molecule_iteration import iter_atoms, iter_bonds


def assigned_chiral_centers(mol, Chem):
    """Use the normal native stereo census with a smaller equivalent CIP problem."""
    try:
        labels = _opposite_ring_labels(mol, Chem)
    except (ValueError, RuntimeError):
        labels = None
    if labels is not None:
        # Retain RDKit's original-graph stereo perception and use its own
        # computed descriptors for both original atom IDs.
        observed = {feature.centeredOn for feature in Chem.FindPotentialStereo(mol)
                    if feature.type == Chem.StereoType.Atom_Tetrahedral
                    and feature.specified == Chem.StereoSpecified.Specified}
        if observed == {i for i, _ in labels}:
            for i, descriptor in labels:
                mol.GetAtomWithIdx(i).SetProp('_CIPCode', descriptor)
            return labels
    return Chem.FindMolChiralCenters(mol, includeUnassigned=False, useLegacyImplementation=False)


def _opposite_ring_labels(mol, Chem):
    if type(mol) is not Chem.Mol or mol.GetNumAtoms() <= 8 or mol.GetStereoGroups():
        return None
    rings = mol.GetRingInfo().AtomRings()
    symmetric_rings = tuple(ring for ring in rings if len(ring) in (4, 6))
    if not symmetric_rings:
        return None
    atoms = tuple(iter_atoms(mol))
    tetrahedral = {Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW}
    tagged = [i for i, atom in enumerate(atoms) if atom.GetChiralTag()]
    if len(tagged) != 2 or any(atoms[i].GetChiralTag() not in tetrahedral for i in tagged):
        return None
    ring = next((row for row in symmetric_rings if all(i in row for i in tagged)), ())
    if not ring or abs(ring.index(tagged[0]) - ring.index(tagged[1])) != len(ring) // 2:
        return None
    core = set(ring)
    if any(row != ring and not core.isdisjoint(row) for row in rings):
        return None
    if any(atom.GetAtomicNum() <= 1 or atom.GetIsotope() or atom.GetFormalCharge()
           or atom.GetNumRadicalElectrons() or atom.GetAtomMapNum()
           or atom.HasQuery() or atom.HasProp('_UnknownStereo') for atom in atoms):
        return None
    bonds = tuple(iter_bonds(mol))
    covalent = {Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC}
    if any(bond.GetStereo() or bond.GetBondDir() or bond.HasQuery()
           or bond.GetBondType() not in covalent for bond in bonds):
        return None

    ligands = {}
    for pos, i in enumerate(ring):
        atom = atoms[i]
        if (atom.GetAtomicNum() != 6 or atom.GetIsAromatic()
                or {n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() in core}
                != {ring[pos - 1], ring[(pos + 1) % len(ring)]}
                or any(b.GetBondType() != Chem.BondType.SINGLE for b in atom.GetBonds())):
            return None
        if i not in tagged:
            if atom.GetDegree() != 2 or atom.GetTotalNumHs() != 2:
                return None
            continue
        if atom.GetDegree() != 3 or atom.GetTotalNumHs() != 1:
            return None
        root, = (n for n in atom.GetNeighbors() if n.GetIdx() not in core)
        # A cut bond keeps the remote ligand on its own branch of the CIP
        # digraph. The two ring paths are equivalent up to the other tagged
        # site, where they differ only in the auxiliary configuration.
        if mol.GetBondBetweenAtoms(i, root.GetIdx()).IsInRing():
            return None
        if not _outranks_ring_methylene(root, i):
            return None
        ligands[root.GetIdx()] = 9 if not ligands else 17
    if len(ligands) != 2:
        return None

    # The two markers preserve the strict Sequence Rule 1a orders of the
    # external ligands against the ring paths and H at both tagged sites.
    # Keep the cycle closure and native ordering of every incident bond.
    probe = Chem.RWMol()
    source = sorted(core | ligands.keys())
    index = {i: probe.AddAtom(Chem.Atom(ligands[i]) if i in ligands else Chem.Atom(atoms[i]))
             for i in source}
    for bond in bonds:
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in index and b in index:
            probe.AddBond(index[a], index[b], Chem.BondType.SINGLE)
    probe = probe.GetMol()
    for i in tagged:
        target = probe.GetAtomWithIdx(index[i])
        if tuple(index[n.GetIdx()] for n in atoms[i].GetNeighbors()) != tuple(
                n.GetIdx() for n in target.GetNeighbors()):
            return None
        if target.HasProp('_CIPCode'):
            target.ClearProp('_CIPCode')
    Chem.SanitizeMol(probe)
    labels = [(source[i], descriptor) for i, descriptor in Chem.FindMolChiralCenters(
        probe, includeUnassigned=False, useLegacyImplementation=False)]
    if {i for i, descriptor in labels if descriptor in {'r', 's'}} != set(tagged):
        return None
    return labels


def _outranks_ring_methylene(root, host):
    number = root.GetAtomicNum()
    if number != 6:
        return number > 6
    # The opposing ring ligand has attached atoms C,H,H. Actual heavy
    # neighbors are a lower bound for the carbon ligand's CIP sphere.
    numbers = [n.GetAtomicNum() for n in root.GetNeighbors() if n.GetIdx() != host]
    return any(n > 6 for n in numbers) or numbers.count(6) >= 2
