"""P-93 original-graph evidence for directly joined cyclic assemblies.

The chemical assembly proof owns the parent. This module supplies assigned
R/S/r/s and double-junction E/Z or seqCis/seqTrans after whole-graph CIP.
It never labels a capped component or treats an E/Z enum as a CIP result.
"""
from __future__ import annotations

from dataclasses import dataclass

from .locant_order import writing_locant_key
from .molecule_cache import exact_graph_computation
from .molecule_iteration import iter_atoms, iter_bonds


# RDKit Descriptor.h serializes seqTrans/seqCis as e/z. Those are backend
# encodings, not the preferred Blue Book P-91.2.1.1/P-93.5.1.4.2.2 spelling.
_BOND_DESCRIPTORS = {'E':'E', 'Z':'Z', 'e':'seqTrans', 'z':'seqCis'}


@dataclass(frozen=True, slots=True)
class AssemblyStereoCensus:
    assigned_atoms: tuple = ()
    assigned_bonds: tuple = ()
    potential_features: tuple = ()
    issue: str = ''


@dataclass(frozen=True, slots=True)
class AssemblyStereoEvidence:
    components: tuple[frozenset[int], ...]
    parent_atoms: frozenset[int]
    parent_locants: tuple
    component_atoms: frozenset[int]
    component_locants: tuple
    junctions: tuple
    assigned_atoms: tuple
    assigned_bonds: tuple
    potential_features: tuple
    isotopes: bool = False


def has_assembly_annotations(mol):
    return (bool(mol.GetStereoGroups()) or any(a.GetIsotope() or int(a.GetChiralTag())
            for a in iter_atoms(mol)) or any(int(b.GetStereo()) for b in iter_bonds(mol)))


def assembly_stereo_census(mol, Chem, *, allow_isotopes=False):
    """Keep native flags, potential units and actual CIP in one full scope.

    Native graph changes invalidate the cache, including maps and raw stereo.
    Caller input is untouched. The recursion bound is an explicit resource
    barrier, never an invitation to fall back to incomplete legacy labels.
    """
    def compute():
        if not allow_isotopes and any(a.GetIsotope() for a in iter_atoms(mol)):
            return AssemblyStereoCensus(issue='Assembly isotope/stereochemical composition remains unproved.')
        if mol.GetStereoGroups():
            return AssemblyStereoCensus(issue='Assembly enhanced relative/ensemble stereochemistry remains unproved.')
        tagged_atoms, tagged_bonds = set(), set()
        for atom in iter_atoms(mol):
            if atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED:
                continue
            if atom.GetChiralTag() not in {Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                                           Chem.ChiralType.CHI_TETRAHEDRAL_CCW}:
                return AssemblyStereoCensus(issue='Assembly non-tetrahedral stereochemistry remains unproved.')
            tagged_atoms.add(atom.GetIdx())
        for bond in iter_bonds(mol):
            if not int(bond.GetStereo()):
                continue
            if (bond.GetBondType() != Chem.BondType.DOUBLE or bond.GetStereo() not in
                    {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ,
                     Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS}):
                return AssemblyStereoCensus(issue='Assembly unknown geometry or stereogenic-axis construction remains unproved.')
            tagged_bonds.add(bond.GetIdx())
        from rdkit.Chem import rdCIPLabeler
        graph = Chem.Mol(mol)
        for atom in graph.GetAtoms():
            atom.SetAtomMapNum(0)
        for obj in (*graph.GetAtoms(), *graph.GetBonds()):
            if obj.HasProp('_CIPCode'):
                obj.ClearProp('_CIPCode')
        try:
            rdCIPLabeler.AssignCIPLabels(graph, maxRecursiveIterations=1_000_000)
            atoms = tuple((a.GetIdx(), a.GetProp('_CIPCode')) for a in graph.GetAtoms()
                          if a.HasProp('_CIPCode') and a.GetProp('_CIPCode') in {'R', 'S', 'r', 's'})
            bonds = tuple((b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetProp('_CIPCode'))
                          for b in graph.GetBonds() if b.HasProp('_CIPCode')
                          and b.GetProp('_CIPCode') in _BOND_DESCRIPTORS)
            potential = tuple((str(f.type), f.centeredOn, str(f.specified))
                              for f in Chem.FindPotentialStereo(graph))
        except (ValueError, RuntimeError):
            return AssemblyStereoCensus(issue='Whole-assembly CIP assignment did not complete within its resource bound.')
        if (tagged_atoms != {i for i, _ in atoms} or tagged_bonds !=
                {graph.GetBondBetweenAtoms(a, b).GetIdx() for a, b, _ in bonds}):
            return AssemblyStereoCensus(atoms, bonds, potential,
                'A flagged assembly unit lacks complete whole-graph CIP assignment, including coupled cyclic configurations.')
        if any(kind not in {'Atom_Tetrahedral', 'Bond_Double'} for kind, _i, _state in potential):
            return AssemblyStereoCensus(atoms, bonds, potential,
                                       'Another potential assembly stereogenic-unit method remains unproved.')
        return AssemblyStereoCensus(atoms, bonds, potential)
    return exact_graph_computation(mol, Chem, 'assembly_stereo_census', (allow_isotopes,), compute)


def build_assembly_stereo_evidence(mol, Chem, components, parent_atoms, mapping,
                                   component_atoms, connectors, *, allow_isotopes=False):
    """Bind selected parent, all mapped sites and original connecting bonds."""
    from .mapped_ring_parents import ring_systems
    try:
        if (not isinstance(components, tuple) or not 2 <= len(components) <= 6
                or not isinstance(parent_atoms, frozenset) or not isinstance(component_atoms, frozenset)
                or any(not isinstance(c, frozenset) or not c for c in components)
                or len(set(components)) != len(components)
                or any(a & b for i, a in enumerate(components) for b in components[i+1:])
                or frozenset().union(*components) != parent_atoms
                or not set(components) <= set(ring_systems(mol))
                or not parent_atoms <= component_atoms
                or set(mapping) != set(component_atoms)
                or any(not isinstance(i, int) or not 0 <= i < mol.GetNumAtoms()
                       or not isinstance(loc, str) or not loc for i, loc in mapping.items())
                or any(writing_locant_key(mapping[i]) is None for i in parent_atoms)
                or len({mapping[i] for i in parent_atoms}) != len(parent_atoms)):
            return None
        owner = {a:i for i, core in enumerate(components) for a in core}
        actual = {frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
                  for b in iter_bonds(mol) if b.GetBeginAtomIdx() in parent_atoms
                  and b.GetEndAtomIdx() in parent_atoms
                  and owner[b.GetBeginAtomIdx()] != owner[b.GetEndAtomIdx()]}
        if (len(connectors) != len(components)-1
                or len({frozenset(pair) for pair in connectors}) != len(connectors)
                or actual != {frozenset(pair) for pair in connectors}):
            return None
        junctions = []
        for a, b in connectors:
            bond = mol.GetBondBetweenAtoms(a, b)
            if (bond is None or a not in owner or b not in owner or owner[a] == owner[b]
                    or bond.IsInRing() or bond.GetBondTypeAsDouble() not in {1, 2}):
                return None
            junctions.append((a, b, int(bond.GetBondTypeAsDouble())))
        if any(order == 2 for _a, _b, order in junctions) and len(components) != 2:
            return None
        census = assembly_stereo_census(mol, Chem, allow_isotopes=allow_isotopes)
        if census.issue:
            return None
        return AssemblyStereoEvidence(components, parent_atoms,
            tuple(sorted((i, mapping[i]) for i in parent_atoms)), component_atoms,
            tuple(sorted(mapping.items())), tuple(junctions), census.assigned_atoms,
            census.assigned_bonds, census.potential_features, allow_isotopes)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return None


def assembly_stereo_plan(mol, Chem, component, locants):
    """Writer plan with revalidated actual compound junction locants."""
    parameters = (component, tuple(locants.items()))
    return exact_graph_computation(mol, Chem, 'assembly_stereo_plan', parameters,
        lambda: _assembly_stereo_plan(mol, Chem, component, locants))


def _assembly_stereo_plan(mol, Chem, component, locants):
    evidence = component.assembly_stereo_evidence
    if (not isinstance(evidence, AssemblyStereoEvidence)
            or len(locants) != len(component.atom_locants)
            or not isinstance(evidence.junctions, tuple)
            or any(not isinstance(row, tuple) or len(row) != 3 for row in evidence.junctions)):
        return None
    if evidence.isotopes:
        from .assembly_isotopes import validate_assembly_isotope_evidence
        if not validate_assembly_isotope_evidence(mol, Chem, component):
            return None
    expected = build_assembly_stereo_evidence(mol, Chem, evidence.components,
        evidence.parent_atoms, locants, component.atoms,
        tuple((a, b) for a, b, _order in evidence.junctions), allow_isotopes=evidence.isotopes)
    if expected != evidence:
        return None
    text, atoms, bonds = [], [], []
    for idx, descriptor in evidence.assigned_atoms:
        if idx in component.atoms:
            locant = locants[idx]
            if writing_locant_key(locant) is None:
                return None
            text.append(((writing_locant_key(locant), ()), locant+descriptor))
            atoms.append((idx, locant, descriptor))
    junctions = {frozenset((a, b)) for a, b, order in evidence.junctions if order == 2}
    for a, b, descriptor in evidence.assigned_bonds:
        if not {a, b} & component.atoms:
            continue
        # Other double-bond methods (internal unsaturation, a cross-boundary
        # bond, suffix geometry) require their own source construction.
        if not {a, b} <= component.atoms or frozenset((a, b)) not in junctions:
            return None
        first, second = sorted((locants[a], locants[b]), key=writing_locant_key)
        locant = first+'('+second+')'
        source_descriptor = _BOND_DESCRIPTORS[descriptor]
        text.append(((writing_locant_key(first), (writing_locant_key(second),)), locant+source_descriptor))
        bonds.append((a, b, locant, source_descriptor))
    return tuple(text), tuple(atoms), tuple(bonds)


def assembly_stereo_numbering(mol, Chem, component):
    """P-14.4(j), after junction, suffix and prefix chemical criteria.

    Compare homologous assigned units only. Different descriptor-family
    layouts retain a visible comparison obligation at the caller, rather than
    gaining an arbitrary cross-family or ASCII preference.
    """
    plan = assembly_stereo_plan(mol, Chem, component, dict(component.atom_locants))
    if plan is None:
        return None
    rows = [((writing_locant_key(locant), ()), 'ordinary' if descriptor in {'R','S'}
             else 'pseudo', descriptor) for _idx,locant,descriptor in plan[1]]
    locants = dict(component.atom_locants)
    for a,b,_compound,descriptor in plan[2]:
        first,second = sorted((locants[a],locants[b]),key=writing_locant_key)
        rows.append(((writing_locant_key(first),(writing_locant_key(second),)),
                     'bond' if descriptor in {'E','Z'} else 'enantiomorphic-bond',descriptor))
    rows.sort()
    return (tuple(key for key,_family,_descriptor in rows),
            tuple(family for _key,family,_descriptor in rows),
            tuple(0 if descriptor in {'Z','R','r','seqCis'} else 1
                  for _key,_family,descriptor in rows))
