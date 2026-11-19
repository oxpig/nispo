"""Native whole-graph proof for relative configurations on saturated rings.

The proof does not infer relative configuration from CIP label case.  It
retains all outside annotations and checks that reversing both native ring
tetrahedral tags denotes exactly the same original isomeric graph, while
reversing either individual tag denotes a distinct graph.
"""

from dataclasses import dataclass
import re

from .molecule_cache import mol_cache_dict


@dataclass(frozen=True, slots=True)
class NativeRelativeCycleEvidence:
    cycle: tuple[int, ...]
    tagged_atoms: frozenset[int]
    descriptor: str
    canonical_graph: str
    simultaneous_inverse_graph: str
    rule: str = 'P-93.5.1.1/P-93.5.1.2'


def native_relative_cycle_evidence(mol, Chem, sequence):
    """Prove an isolated saturated cycle with two singly substituted sites.

    Canonical comparisons use private whole-graph copies. Atom-map labels
    are identifiers, so they do not distinguish otherwise equivalent ring
    paths; every chemical isotope, charge and outside stereo tag is retained.
    """
    sequence = tuple(sequence)
    core = frozenset(sequence)
    if (len(core) != len(sequence) or not 3 <= len(core) <= 12
            or mol.GetStereoGroups()):
        return None
    rings = tuple(frozenset(ring) for ring in mol.GetRingInfo().AtomRings())
    if core not in rings or any(core & ring for ring in rings if ring != core):
        return None
    tetra = {Chem.ChiralType.CHI_TETRAHEDRAL_CW,
             Chem.ChiralType.CHI_TETRAHEDRAL_CCW}
    tagged = set()
    for position, idx in enumerate(sequence):
        atom = mol.GetAtomWithIdx(idx)
        if (atom.GetAtomicNum() not in {6, 8} or atom.GetIsAromatic()
                or atom.GetFormalCharge() or atom.GetNumRadicalElectrons()
                or atom.GetTotalValence() != (4 if atom.GetAtomicNum() == 6 else 2)):
            return None
        inside = {n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() in core}
        if inside != {sequence[position - 1], sequence[(position + 1) % len(sequence)]}:
            return None
        if any(mol.GetBondBetweenAtoms(idx, other).GetBondType() != Chem.BondType.SINGLE
               for other in inside):
            return None
        pendant = [n for n in atom.GetNeighbors()
                   if n.GetIdx() not in core and n.GetAtomicNum() > 1]
        if atom.GetChiralTag() in tetra:
            if len(pendant) != 1 or atom.GetTotalNumHs(includeNeighbors=True) != 1:
                return None
            if mol.GetBondBetweenAtoms(idx, pendant[0].GetIdx()).GetBondType() != Chem.BondType.SINGLE:
                return None
            tagged.add(idx)
        elif atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED or pendant:
            return None
    if len(tagged) != 2:
        return None

    from .stereochemistry import relative_monocycle_configuration
    descriptor = relative_monocycle_configuration(mol, Chem, sequence)
    return _relative_evidence(mol, Chem, sequence, tagged, descriptor)


def _relative_evidence(mol, Chem, sequence, tagged, descriptor):
    if not descriptor:
        return None

    def canonical(inverted):
        copy = Chem.Mol(mol)
        for atom in copy.GetAtoms():
            atom.SetAtomMapNum(0)
            if atom.GetIdx() in inverted:
                atom.InvertChirality()
        expected_tags = tuple(atom.GetChiralTag() for atom in copy.GetAtoms())
        # RDKit may normalize its stored cis/trans enums to Z/E while
        # refreshing the same original double-bond geometry.
        bond_equivalent = {Chem.BondStereo.STEREOCIS: Chem.BondStereo.STEREOZ,
                           Chem.BondStereo.STEREOTRANS: Chem.BondStereo.STEREOE}
        expected_bonds = tuple(bond_equivalent.get(bond.GetStereo(), bond.GetStereo())
                               for bond in copy.GetBonds())
        # Native tag inversion does not invalidate every RDKit stereo cache.
        # Recompute before serializing so a single-site inversion cannot
        # accidentally reuse the original canonical isomeric representation.
        copy.ClearComputedProps()
        Chem.AssignStereochemistry(copy, cleanIt=True, force=True)
        if (tuple(atom.GetChiralTag() for atom in copy.GetAtoms()) != expected_tags
                or tuple(bond_equivalent.get(bond.GetStereo(), bond.GetStereo())
                         for bond in copy.GetBonds()) != expected_bonds):
            return None
        return Chem.MolToSmiles(copy, canonical=True, isomericSmiles=True)

    original = canonical(())
    inverse = canonical(tagged)
    singles = tuple(canonical({idx}) for idx in tagged)
    if (original is None or original != inverse or any(value is None for value in singles)
            or original in singles):
        return None
    return NativeRelativeCycleEvidence(sequence, frozenset(tagged), descriptor,
                                       original, inverse)


def mapped_native_relative_cycle(mol, Chem, locant_by_atom, name):
    """Return proved relative descriptor/ownership for a complete parent map.

    ``None`` leaves existing CIP and other relative grammars to their current
    producers. An existing parent-level cis/trans word still owns the same
    original atoms, so their individual CIP labels must not be added again.
    """
    from .stereochemistry import _parent_stereo_texts_in_name

    existing = _parent_stereo_texts_in_name(name, 'RSrs')
    for sequence in mol.GetRingInfo().AtomRings():
        # This unlocanted face word belongs to one complete cyclic parent,
        # not one of several cycles sharing an assembly's numbering map.
        if frozenset(sequence) != frozenset(locant_by_atom):
            continue
        proof = _cached_cycle_evidence(mol, Chem, sequence)
        if proof is None:
            continue
        if re.match(r'^\((?:cis|trans)\)-', name):
            return '', proof.tagged_atoms
        if any(str(locant_by_atom[idx]) + descriptor in existing
               for idx in proof.tagged_atoms for descriptor in 'RSrs'):
            continue
        return proof.descriptor, proof.tagged_atoms
    return None


def mapped_component_relative_cycle(mol, Chem, locant_by_atom):
    """Prove the one numbered skeletal cycle, allowing mapped suffix atoms."""
    skeleton = {idx for idx, locant in locant_by_atom.items()
                if locant.isdecimal() and mol.GetAtomWithIdx(idx).GetAtomicNum() > 1}
    for sequence in mol.GetRingInfo().AtomRings():
        if skeleton == set(sequence):
            return _cached_cycle_evidence(mol, Chem, sequence)
    return fused_ring_junction_evidence(mol, Chem, frozenset(locant_by_atom))


def fused_ring_junction_evidence(mol, Chem, core):
    """Prove the face relationship of the two junction ligands of a fused pair."""
    key = frozenset(core)
    cache = mol_cache_dict(mol, 'relative_fused_ring_junction')
    if key not in cache:
        cache[key] = _fused_ring_junction_evidence(mol, Chem, key)
    return cache[key]


def _fused_ring_junction_evidence(mol, Chem, core):
    if mol.GetStereoGroups():
        return None
    rings = tuple(tuple(r) for r in mol.GetRingInfo().AtomRings() if set(r) & core)
    if len(rings) != 2 or frozenset(rings[0]) | frozenset(rings[1]) != core:
        return None
    left, right = map(frozenset, rings)
    junctions = left & right
    if len(junctions) != 2 or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6
                                 or mol.GetAtomWithIdx(i).GetFormalCharge()
                                 or mol.GetAtomWithIdx(i).GetIsAromatic() for i in core):
        return None
    if any(b.GetBondType() != Chem.BondType.SINGLE for i in core
           for b in mol.GetAtomWithIdx(i).GetBonds() if b.GetOtherAtomIdx(i) in core):
        return None
    if {i for i in core if mol.GetAtomWithIdx(i).GetChiralTag()} != junctions:
        return None
    # Orient both junction tetrahedra against one ring. At each junction the
    # third internal neighbor lies in the other ring. Comparing those faces
    # also compares the faces of the two remaining junction ligands.
    sequence = rings[0]
    faces = []
    for i in sorted(junctions):
        atom = mol.GetAtomWithIdx(i)
        pos = sequence.index(i)
        preceding, following = sequence[pos - 1], sequence[(pos + 1) % len(sequence)]
        if not {preceding, following} & (junctions - {i}):
            return None
        stored = [a.GetIdx() for a in atom.GetNeighbors()]
        other = [j for j in stored if j in right - left]
        external = [j for j in stored if j not in core]
        if not external and len(stored) == 3 and atom.GetTotalNumHs() == 1:
            external = [-1]
            stored.append(-1)
        if len(other) != 1 or len(external) != 1 or len(stored) != 4:
            return None
        if atom.GetChiralTag() not in (Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                                       Chem.ChiralType.CHI_TETRAHEDRAL_CCW):
            return None
        permutation = [stored.index(j) for j in (preceding, following, other[0], external[0])]
        inversions = sum(a > b for k, a in enumerate(permutation) for b in permutation[k + 1:])
        faces.append((int(atom.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CW) + inversions) % 2)
    return _relative_evidence(mol, Chem, sequence, junctions,
                              'cis' if faces[0] == faces[1] else 'trans')


def _cached_cycle_evidence(mol, Chem, sequence):
    cache = mol_cache_dict(mol, 'native_relative_cycle_evidence')
    if sequence not in cache:
        cache[sequence] = native_relative_cycle_evidence(mol, Chem, sequence)
    return cache[sequence]
