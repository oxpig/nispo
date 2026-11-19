"""Mapped P-65.7.2 construction for ordinary borinic/carboxylic anhydrides.

The source example is ``(CH3)2B-O-CO-CH3`` (acetic dimethylborinic
anhydride).  This producer admits one neutral trigonal borinic centre, one
ordinary carboxylic acid residue, and two simple hydrocarbon B-organyl arms.
It keeps the original graph as the ownership authority; the temporary capped
borinic acid is used only to obtain the retained component name.
"""

from .molecule_iteration import iter_atoms, iter_bonds
from .naming_methods import MethodDecision


RULE = "P-41/P-65.7.0/P-65.7.2/P-68.1.4.1/P-68.1.5"


def _single(mol, left, right):
    bond = mol.GetBondBetweenAtoms(left, right)
    return bond is not None and bond.GetBondTypeAsDouble() == 1.0


def _simple_acid_carbon_residue(mol, root, carbonyl):
    """Accept a connected saturated carbon tree or plain phenyl residue."""
    if root is None:
        return frozenset()
    seen, pending = {root}, [root]
    while pending:
        idx = pending.pop()
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() != 6 or atom.GetFormalCharge() or atom.GetIsotope():
            return None
        for neighbor in atom.GetNeighbors():
            other = neighbor.GetIdx()
            if other == carbonyl or neighbor.GetAtomicNum() == 1:
                continue
            if neighbor.GetAtomicNum() != 6:
                return None
            if other not in seen:
                seen.add(other)
                pending.append(other)
    atoms = frozenset(seen)
    rings = [tuple(ring) for ring in mol.GetRingInfo().AtomRings()
             if set(ring) & atoms]
    ring_atoms = {idx for idx in atoms if mol.GetAtomWithIdx(idx).IsInRing()}
    if rings:
        if (len(atoms) != 6 or len(rings) != 1 or len(rings[0]) != 6
                or set(rings[0]) != set(atoms) or ring_atoms != set(atoms)
                or any(not mol.GetAtomWithIdx(i).GetIsAromatic() for i in atoms)
                or any(not mol.GetBondBetweenAtoms(a, b).GetIsAromatic()
                       for a, b in zip(rings[0], rings[0][1:] + rings[0][:1]))):
            return None
    else:
        if any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in atoms):
            return None
        if any(not _single(mol, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
               for bond in iter_bonds(mol)
               if bond.GetBeginAtomIdx() in atoms and bond.GetEndAtomIdx() in atoms):
            return None
        if sum(1 for bond in iter_bonds(mol)
               if bond.GetBeginAtomIdx() in atoms and bond.GetEndAtomIdx() in atoms) != len(atoms)-1:
            return None
    return atoms


def _recognize(mol):
    if (mol.GetNumHeavyAtoms() > 64 or mol.GetStereoGroups()
            or any(atom.GetFormalCharge() or atom.GetNumRadicalElectrons()
                   for atom in iter_atoms(mol))
            or any(atom.GetIsotope() or int(atom.GetChiralTag())
                   for atom in iter_atoms(mol))
            or any(int(bond.GetStereo()) or int(bond.GetBondDir())
                   for bond in iter_bonds(mol))):
        return None
    if mol.GetNumAtoms():
        connected, pending = {0}, [0]
        while pending:
            idx = pending.pop()
            for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
                nidx = neighbor.GetIdx()
                if nidx not in connected:
                    connected.add(nidx)
                    pending.append(nidx)
        if connected != set(range(mol.GetNumAtoms())):
            return None
    heavy = {atom.GetIdx() for atom in iter_atoms(mol) if atom.GetAtomicNum() > 1}
    borons = [atom for atom in iter_atoms(mol) if atom.GetAtomicNum() == 5]
    if len(borons) != 1:
        return None
    boron = borons[0]
    bidx = boron.GetIdx()
    if boron.IsInRing() or boron.GetDegree() != 3 or boron.GetTotalValence() != 3:
        return None
    oxygens = [atom for atom in iter_atoms(mol) if atom.GetAtomicNum() == 8]
    if len(oxygens) != 2:  # bridge O and carboxyl carbonyl O
        return None
    bridges = [atom for atom in boron.GetNeighbors()
               if atom.GetAtomicNum() == 8 and _single(mol, bidx, atom.GetIdx())]
    if len(bridges) != 1:
        return None
    bridge = bridges[0]
    oidx = bridge.GetIdx()
    if bridge.GetDegree() != 2 or bridge.IsInRing():
        return None
    other = [atom for atom in bridge.GetNeighbors() if atom.GetIdx() != bidx]
    if len(other) != 1 or other[0].GetAtomicNum() != 6:
        return None
    carbonyl = other[0]
    cidx = carbonyl.GetIdx()
    if not _single(mol, oidx, cidx):
        return None
    carbonyl_oxygens = [atom for atom in carbonyl.GetNeighbors()
                        if atom.GetAtomicNum() == 8
                        and mol.GetBondBetweenAtoms(cidx, atom.GetIdx()).GetBondTypeAsDouble() == 2]
    if len(carbonyl_oxygens) != 1:
        return None
    carbonyl_o = carbonyl_oxygens[0]
    if carbonyl_o.GetDegree() != 1:
        return None
    remaining = [atom for atom in carbonyl.GetNeighbors()
                 if atom.GetIdx() not in {oidx, carbonyl_o.GetIdx()}]
    if len(remaining) > 1 or any(atom.GetAtomicNum() != 6 for atom in remaining):
        return None
    if remaining and not _single(mol, cidx, remaining[0].GetIdx()):
        return None
    acid_atoms = _simple_acid_carbon_residue(
        mol, remaining[0].GetIdx() if remaining else None, cidx)
    if acid_atoms is None:
        return None
    arms = [atom for atom in boron.GetNeighbors() if atom.GetIdx() != oidx]
    if len(arms) != 2 or any(atom.GetAtomicNum() != 6 for atom in arms):
        return None
    # The shared retained borinic acid renderer proves simple connected
    # alkyl/phenyl branches; its cap is applied only on a temporary graph.
    covered = {bidx, oidx, cidx, carbonyl_o.GetIdx(), *acid_atoms}
    if len(covered) != 4 + len(acid_atoms):
        return None
    for arm in arms:
        branch = {arm.GetIdx()}
        todo = [arm.GetIdx()]
        while todo:
            idx = todo.pop()
            for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
                nidx = neighbor.GetIdx()
                if nidx in {bidx, oidx} or neighbor.GetAtomicNum() == 1:
                    continue
                if neighbor.GetAtomicNum() != 6:
                    return None
                if nidx not in branch:
                    branch.add(nidx)
                    todo.append(nidx)
        if covered & branch:
            return None
        covered.update(branch)
    if covered != heavy:
        return None
    # Require a single connected molecular graph and no explicit isotopic H.
    if any(atom.GetAtomicNum() == 1 and atom.GetIsotope() for atom in iter_atoms(mol)):
        return None
    return bidx, oidx, cidx, carbonyl_o.GetIdx(), frozenset(acid_atoms), frozenset(covered)


def _capped_borinic_name(mol, Chem, boron, bridge, carbonyl, carbonyl_o, acid_atoms):
    """Render borinic component after removing the carboxyl residue at O."""
    remove = set(acid_atoms) | {carbonyl, carbonyl_o}
    rw = Chem.RWMol(mol)
    for idx in sorted(remove, reverse=True):
        rw.RemoveAtom(idx)
    capped = rw.GetMol()
    # The source bridge O is divalent in the original anhydride. Once its
    # carbonyl side is removed it is the OH oxygen of the retained borinic
    # acid; restore implicit-H valence on this copy alone.
    bridge_after = bridge - sum(idx < bridge for idx in remove)
    capped_bridge = capped.GetAtomWithIdx(bridge_after)
    capped_bridge.SetNoImplicit(False)
    capped_bridge.SetNumExplicitHs(0)
    try:
        Chem.SanitizeMol(capped)
    except (ValueError, RuntimeError):
        return None
    from .boron_acid_methods import boron_acid_decision
    decision = boron_acid_decision(capped, Chem)
    if decision is None or not decision.name.endswith("borinic acid"):
        return None
    return decision.name


def borinic_anhydride_decision(mol, Chem=None):
    """Return an original-graph P-65.7.2 decision for the bounded motif."""
    found = _recognize(mol)
    if found is None:
        return None
    if Chem is None:
        from rdkit import Chem
    boron, bridge, carbonyl, carbonyl_o, acid_carbons, heavy = found
    from .acid_derivative_methods import _acid_component
    acid = _acid_component(mol, Chem, carbonyl, bridge)
    expected_acid = frozenset({carbonyl, carbonyl_o}) | acid_carbons
    if acid is None or acid.atoms != expected_acid or not acid.name.endswith(" acid"):
        return None
    borinic = _capped_borinic_name(mol, Chem, boron, bridge, carbonyl,
                                   carbonyl_o, acid_carbons)
    if not borinic:
        return None
    from .formatting import prefix_alphabetization_key
    acid_name = acid.name[:-5]
    borinic_name = borinic[:-5]
    first, second = sorted((acid_name, borinic_name), key=prefix_alphabetization_key)
    name = f"{first} {second} anhydride"
    bridge_atoms = frozenset({bridge})
    borinic_atoms = frozenset({boron}) | frozenset(
        idx for idx in heavy if idx not in acid.atoms and idx != bridge and idx != boron
    )
    roles = (
        ("carboxylic-acid-residue", acid.atoms, acid.atom_locants),
        ("anhydride-oxygen", bridge_atoms, ()),
        ("borinic-acid-residue", borinic_atoms, ()),
    )
    if (acid.atoms & bridge_atoms or acid.atoms & borinic_atoms
            or bridge_atoms & borinic_atoms
            or acid.atoms | bridge_atoms | borinic_atoms != heavy):
        return None
    return MethodDecision(
        name=name, method="functional-class",
        principal_centers=frozenset({carbonyl, boron}), atoms=heavy,
        rule=RULE, atom_locants=acid.atom_locants, component_roles=roles,
    )

