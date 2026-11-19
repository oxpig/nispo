"""A bounded P-51.3 multiplicative route for symmetric sulfone ethers.

This producer handles an otherwise unsupported, but general, connector: two
unsubstituted benzene parents linked through equivalent, saturated carbon
paths, two sulfonyl groups and an ether oxygen.  Each carbon path is numbered
from its sulfonyl end.  The path may have any length in ``ROOTS`` and the
ether may join any atom on it.  Branched paths and an internal sulfonyl
attachment are outside this writer's scope.

P-44.1.2.2 selects the rings rather than hydrocarbon chains, P-51.3.1 and
P-63.6 admit multiplication for the identical parent structures, and
P-15.3.1.2.2.1/.4 specify the order and numbering of the concatenated linker.
The central oxygen and two sulfonyl groups are three heterounits, below the
four required by P-51.4.1.1.  P-68.4.3.2 concerns directly concatenated
sulfonyl units, not the carbon-separated units recognized here.

The producer proves ownership of every atom and checks each bond before it
renders.  It declines isotopes, charge, radicals, explicit atom stereotags,
enhanced stereo and all other unsupported chemistry.  Unknown stereogenic
units are not assigned configurations: their omitted P-91 descriptor is
reported as a selection limitation to the caller.
"""

from dataclasses import dataclass

from .constants import ROOTS
from .naming_result import SelectionLimitation


@dataclass(frozen=True, slots=True)
class SulfoneEtherArm:
    """Original-atom ownership in one ring-to-central-oxygen branch."""

    ring: frozenset[int]
    sulfur: int
    oxo_oxygens: frozenset[int]
    # Locants 1..n, numbered from the sulfur end under P-15.3.1.2.2.4.
    carbon_path: tuple[int, ...]
    ether_locant: int

    @property
    def atoms(self):
        return self.ring | {self.sulfur} | self.oxo_oxygens | set(self.carbon_path)


@dataclass(frozen=True, slots=True)
class SymmetricSulfoneEtherProof:
    """Evidence for two disjoint identical branches and their central atom."""

    central_oxygen: int
    arms: tuple[SulfoneEtherArm, SulfoneEtherArm]
    unknown_stereocenters: frozenset[int]
    atoms: frozenset[int]


@dataclass(frozen=True, slots=True)
class SymmetricSulfoneEtherDecision:
    name: str
    proof: SymmetricSulfoneEtherProof
    preference_issues: tuple[SelectionLimitation, ...] = ()
    rule: str = "P-44.1.2.2/P-51.3.1/P-15.3.1.2.2"


def _other_index(bond, index):
    return bond.GetOtherAtomIdx(index)


def _is_single(bond, Chem):
    return bond.GetBondType() == Chem.BondType.SINGLE and not bond.GetIsAromatic()


def _recognized_benzenes(mol, Chem):
    """Return two disjoint, monosubstituted benzene rings or decline."""
    rings = tuple(frozenset(ring) for ring in mol.GetRingInfo().AtomRings())
    if len(rings) != 2 or any(len(ring) != 6 for ring in rings) or rings[0] & rings[1]:
        return None
    for ring in rings:
        for idx in ring:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetAtomicNum() != 6 or not atom.GetIsAromatic():
                return None
            in_ring = [bond for bond in atom.GetBonds() if _other_index(bond, idx) in ring]
            if (len(in_ring) != 2 or any(not bond.GetIsAromatic() for bond in in_ring)):
                return None
        outside = [(idx, _other_index(bond, idx)) for idx in ring
                   for bond in mol.GetAtomWithIdx(idx).GetBonds()
                   if _other_index(bond, idx) not in ring]
        if len(outside) != 1:
            return None
        ring_idx, external_idx = outside[0]
        bond = mol.GetBondBetweenAtoms(ring_idx, external_idx)
        if (mol.GetAtomWithIdx(external_idx).GetAtomicNum() != 16
                or not _is_single(bond, Chem)):
            return None
        # These checks also rule out valence/explicit-hydrogen variants whose
        # graph has the right skeleton but not the represented benzene hydrogens.
        if any(mol.GetAtomWithIdx(idx).GetTotalNumHs(includeNeighbors=True)
               != (0 if idx == ring_idx else 1) for idx in ring):
            return None
    return rings


def _sulfone_parts(mol, sulfur_idx, rings, Chem):
    """Prove one C-S(=O)2-C unit and return its owned atoms."""
    sulfur = mol.GetAtomWithIdx(sulfur_idx)
    if sulfur.GetDegree() != 4 or sulfur.GetTotalNumHs(includeNeighbors=True):
        return None
    oxos, single_carbons = [], []
    for bond in sulfur.GetBonds():
        neighbor = bond.GetOtherAtom(sulfur)
        if (neighbor.GetAtomicNum() == 8 and bond.GetBondType() == Chem.BondType.DOUBLE
                and neighbor.GetDegree() == 1 and not neighbor.GetTotalNumHs(includeNeighbors=True)):
            oxos.append(neighbor.GetIdx())
        elif neighbor.GetAtomicNum() == 6 and _is_single(bond, Chem):
            single_carbons.append(neighbor.GetIdx())
        else:
            return None
    if len(oxos) != 2 or len(single_carbons) != 2:
        return None
    ring_hits = [(ring, idx) for ring in rings for idx in single_carbons if idx in ring]
    if len(ring_hits) != 1:
        return None
    ring, ring_atom = ring_hits[0]
    carbon_atom = next(idx for idx in single_carbons if idx != ring_atom)
    if mol.GetAtomWithIdx(carbon_atom).GetIsAromatic() or mol.GetAtomWithIdx(carbon_atom).IsInRing():
        return None
    return ring, carbon_atom, frozenset(oxos)


def _carbon_component(mol, start, eligible, Chem):
    seen, todo = set(), [start]
    while todo:
        idx = todo.pop()
        if idx in seen:
            continue
        seen.add(idx)
        for bond in mol.GetAtomWithIdx(idx).GetBonds():
            neighbor = _other_index(bond, idx)
            if neighbor in eligible:
                if not _is_single(bond, Chem):
                    return None
                if neighbor not in seen:
                    todo.append(neighbor)
    return frozenset(seen)


def _carbon_path(mol, component, start, sulfur_idx, ether_idx, Chem):
    """Prove an unbranched saturated carbon path and all its external bonds."""
    adjacency = {}
    ether_sites = []
    for idx in component:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() != 6 or atom.GetIsAromatic() or atom.IsInRing():
            return None
        carbon_neighbors = []
        external = []
        for bond in atom.GetBonds():
            other = _other_index(bond, idx)
            if not _is_single(bond, Chem):
                return None
            if other in component:
                carbon_neighbors.append(other)
            else:
                external.append(other)
        if any(other not in (sulfur_idx, ether_idx) for other in external):
            return None
        if sulfur_idx in external and idx != start:
            return None
        if ether_idx in external:
            ether_sites.append(idx)
        # Hydrogen count is a separate safeguard against nonstandard valence.
        if atom.GetTotalNumHs(includeNeighbors=True) != 4 - atom.GetDegree():
            return None
        adjacency[idx] = carbon_neighbors
    if len(ether_sites) != 1 or any(len(neighbors) > 2 for neighbors in adjacency.values()):
        return None
    if len(component) == 1:
        if adjacency[start]:
            return None
    elif len(adjacency[start]) != 1 or sum(len(v) == 1 for v in adjacency.values()) != 2:
        return None
    path, previous, current = [], None, start
    while current is not None:
        path.append(current)
        following = [idx for idx in adjacency[current] if idx != previous]
        if len(following) > 1 or (following and following[0] in path):
            return None
        previous, current = current, following[0] if following else None
    if len(path) != len(component) or len(path) not in ROOTS:
        return None
    return tuple(path), path.index(ether_sites[0]) + 1


def _unknown_stereocenters(mol, Chem, carbon_atoms):
    """Reject expressed/unsupported stereo and retain unresolved centres."""
    if mol.GetStereoGroups():
        return None
    if any(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for atom in mol.GetAtoms()):
        return None
    if any(bond.GetStereo() != Chem.BondStereo.STEREONONE
           or bond.GetBondDir() != Chem.BondDir.NONE for bond in mol.GetBonds()):
        return None
    unknown = set()
    try:
        for info in Chem.FindPotentialStereo(mol):
            if (str(info.type) != 'Atom_Tetrahedral' or str(info.specified) != 'Unspecified'
                    or info.centeredOn not in carbon_atoms):
                return None
            unknown.add(info.centeredOn)
    except (AttributeError, RuntimeError, ValueError):
        return None
    return frozenset(unknown)


def symmetric_sulfone_ether_decision(mol, Chem):
    """Construct a proven multiplicative name, or return None outside scope.

    ``mol`` must be a sanitized RDKit molecule.  The result includes atom
    ownership and any stereochemical limitation so a principal-selection
    caller can preserve the latter in its public naming result.
    """
    if not mol or len(Chem.GetMolFrags(mol)) != 1:
        return None
    atoms = tuple(mol.GetAtoms())
    if any(atom.GetAtomicNum() not in (6, 8, 16) or atom.GetFormalCharge()
           or atom.GetIsotope() or atom.GetNumRadicalElectrons() for atom in atoms):
        return None
    sulfurs = [atom.GetIdx() for atom in atoms if atom.GetAtomicNum() == 16]
    oxygens = [atom.GetIdx() for atom in atoms if atom.GetAtomicNum() == 8]
    if len(sulfurs) != 2 or len(oxygens) != 5:
        return None
    rings = _recognized_benzenes(mol, Chem)
    if rings is None:
        return None
    parts = [_sulfone_parts(mol, idx, rings, Chem) for idx in sulfurs]
    if any(part is None for part in parts) or parts[0][0] == parts[1][0]:
        return None
    oxos = parts[0][2] | parts[1][2]
    if len(oxos) != 4:
        return None
    central = set(oxygens) - oxos
    if len(central) != 1:
        return None
    ether_idx = next(iter(central))
    ether = mol.GetAtomWithIdx(ether_idx)
    if (ether.GetDegree() != 2 or ether.GetTotalNumHs(includeNeighbors=True)
            or ether.IsInRing() or any(not _is_single(bond, Chem) for bond in ether.GetBonds())
            or any(neighbor.GetAtomicNum() != 6 for neighbor in ether.GetNeighbors())):
        return None
    ring_atoms = rings[0] | rings[1]
    carbon_atoms = frozenset(atom.GetIdx() for atom in atoms
                             if atom.GetAtomicNum() == 6 and atom.GetIdx() not in ring_atoms)
    components = [_carbon_component(mol, part[1], carbon_atoms, Chem) for part in parts]
    if (any(component is None for component in components) or components[0] & components[1]
            or components[0] | components[1] != carbon_atoms):
        return None
    arms = []
    for sulfur, part, component in zip(sulfurs, parts, components):
        path_result = _carbon_path(mol, component, part[1], sulfur, ether_idx, Chem)
        if path_result is None:
            return None
        path, locant = path_result
        arms.append(SulfoneEtherArm(part[0], sulfur, part[2], path, locant))
    if (len(arms[0].carbon_path), arms[0].ether_locant) != (
            len(arms[1].carbon_path), arms[1].ether_locant):
        return None
    # Verify the ownership partition is complete, pairwise disjoint, and is
    # tied to the actual ether neighbours (not a coincidental local match).
    owned = (set(arms[0].atoms), set(arms[1].atoms), {ether_idx})
    if (owned[0] & owned[1] or owned[0] & owned[2] or owned[1] & owned[2]
            or set.union(*owned) != set(range(mol.GetNumAtoms()))
            or {atom.GetIdx() for atom in ether.GetNeighbors()} != {
                arm.carbon_path[arm.ether_locant - 1] for arm in arms}):
        return None
    unknown = _unknown_stereocenters(mol, Chem, carbon_atoms)
    if unknown is None:
        return None
    n, locant = len(arms[0].carbon_path), arms[0].ether_locant
    carbon_prefix = 'methylene' if n == 1 else f'{ROOTS[n]}ane-{locant},1-diyl'
    name = f"1,1'-[oxybis({carbon_prefix}sulfonyl)]dibenzene"
    issues = ()
    if unknown:
        issues = (SelectionLimitation(
            'stereochemistry', 'P-91.2.1.2.1',
            'One or more stereogenic centres have unspecified configuration; '
            'the constitutional name omits the required unknown-configuration descriptors.'),)
    proof = SymmetricSulfoneEtherProof(ether_idx, tuple(arms), unknown,
                                      frozenset(range(mol.GetNumAtoms())))
    return SymmetricSulfoneEtherDecision(name, proof, issues)
