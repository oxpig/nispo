"""Graph-mapped fused cyclic sultams named as sulfur heterones.

The admitted hydride is the reduced, benzene-fused 1,2-benzothiazole
family. Its two annular sulfur oxo groups are cited by the heterone suffix,
not as detachable oxo prefixes (P-64.4, P-66.1.5.2.1).
"""

from .component_mapping import mapped_branch
from .formatting import format_prefixes, join_prefix_parent
from .functional_inventory import functional_inventory
from .mapped_ring_parents import ring_systems, skeletons
from .molecule_iteration import iter_atoms, iter_bonds
from .naming_methods import MethodDecision
from .naming_result import SelectionLimitation


RULE = "P-41/P-64.4/P-66.1.5.2.1"


def _junior_or_no_other_groups(mol, Chem, sulfur, oxygens):
    """Prove the remaining original groups are junior to a heterone.

    The ordinary inventory does not recognize an annular SO2 group. On a
    recognition copy only, replacing that unit with a saturated carbon
    allows the shared inventory to inspect the other characteristic groups.
    No proxy atom or proxy text participates in the actual naming decision.
    """
    graph = Chem.RWMol(mol)
    substitute = graph.GetAtomWithIdx(sulfur)
    substitute.SetAtomicNum(6)
    substitute.SetNumExplicitHs(0)
    substitute.SetNoImplicit(False)
    for index in sorted(oxygens, reverse=True):
        graph.RemoveAtom(index)
    proxy = graph.GetMol()
    try:
        Chem.SanitizeMol(proxy)
    except (ValueError, RuntimeError):
        return False
    inventory = functional_inventory(proxy)
    return (inventory.supported
            and not any(group.seniority[0] <= 16 for group in inventory.groups))


def _benzothiazole_frame(mol, Chem, core, sulfur, nitrogen):
    """Prove the exact saturated 2,3 bond and the fused benzene state."""
    rings = [frozenset(ring) for ring in mol.GetRingInfo().AtomRings()
             if set(ring) <= core]
    if (len(core) != 9 or len(rings) != 2
            or sorted(map(len, rings)) != [5, 6]
            or len(rings[0] & rings[1]) != 2):
        return None
    benzene = next(ring for ring in rings if len(ring) == 6)
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6
           or not mol.GetAtomWithIdx(i).GetIsAromatic() for i in benzene):
        return None
    matches = []
    for frame in skeletons(mol, Chem, core):
        if frame[0] != '1,2-benzothiazole':
            continue
        mapping = dict(zip(frame[1], frame[2]))
        if mapping.get(sulfur) != '1' or mapping.get(nitrogen) != '2':
            continue
        inverse = {locant: idx for idx, locant in mapping.items()}
        carbon3 = inverse.get('3')
        if carbon3 is None:
            continue
        if (mol.GetAtomWithIdx(nitrogen).GetIsAromatic()
                or mol.GetAtomWithIdx(carbon3).GetAtomicNum() != 6
                or mol.GetAtomWithIdx(carbon3).GetIsAromatic()):
            continue
        bond = mol.GetBondBetweenAtoms(nitrogen, carbon3)
        if bond is None or bond.GetBondTypeAsDouble() != 1:
            continue
        if any(mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble() != 1
               for a, b in ((sulfur, nitrogen), (carbon3, inverse['3a']),
                            (sulfur, inverse['7a']))):
            continue
        matches.append(mapping)
    return matches[0] if len(matches) == 1 else None


def cyclic_sultam_heterone_decision(mol, Chem):
    """Name the proved 2,3-dihydro-1,2-benzothiazole sultam family."""
    if mol.GetNumHeavyAtoms() > 128 or mol.GetStereoGroups():
        return None
    sulfur_atoms = [a for a in iter_atoms(mol) if a.GetAtomicNum() == 16]
    if len(sulfur_atoms) != 1 or not sulfur_atoms[0].IsInRing():
        return None
    if any(a.GetFormalCharge() or a.GetNumRadicalElectrons() or a.GetIsotope()
           or int(a.GetChiralTag()) or a.GetAtomicNum() not in {1, 6, 7, 8, 9, 16, 17, 35, 53}
           for a in iter_atoms(mol)):
        return None
    if any(int(b.GetStereo()) for b in iter_bonds(mol)):
        return None
    sulfur = sulfur_atoms[0]
    sidx = sulfur.GetIdx()
    neighbors = [n for n in sulfur.GetNeighbors() if n.GetAtomicNum() > 1]
    oxygens = [n.GetIdx() for n in neighbors if n.GetAtomicNum() == 8
               and n.GetDegree() == 1
               and mol.GetBondBetweenAtoms(sidx, n.GetIdx()).GetBondTypeAsDouble() == 2]
    ring_neighbors = [n for n in neighbors if n.IsInRing()
                      and mol.GetBondBetweenAtoms(sidx, n.GetIdx()).GetBondTypeAsDouble() == 1]
    nitrogens = [n for n in ring_neighbors if n.GetAtomicNum() == 7]
    if len(neighbors) != 4 or len(oxygens) != 2 or len(ring_neighbors) != 2 or len(nitrogens) != 1:
        return None
    cores = [core for core in ring_systems(mol) if sidx in core]
    if len(cores) != 1:
        return None
    core = cores[0]
    mapping = _benzothiazole_frame(mol, Chem, core, sidx, nitrogens[0].GetIdx())
    if mapping is None or not _junior_or_no_other_groups(mol, Chem, sidx, oxygens):
        return None
    owned = set(core) | set(oxygens)
    prefixes, roles = [], [
        ('heterocyclic_parent', frozenset(core), tuple(sorted(mapping.items()))),
        ('sulfur_oxo_suffix', frozenset(oxygens), ()),
    ]
    for idx in sorted(core):
        for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
            other = neighbor.GetIdx()
            if other in core or other in oxygens or neighbor.GetAtomicNum() == 1:
                continue
            if mol.GetBondBetweenAtoms(idx, other).GetBondTypeAsDouble() != 1:
                return None
            prefix, atoms = mapped_branch(mol, Chem, other, idx)
            if not prefix or not atoms or owned & atoms:
                return None
            prefixes.append((mapping[idx], prefix))
            owned.update(atoms)
            roles.append(('substituent:' + mapping[idx], atoms, ()))
    heavy = frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1)
    if owned != heavy:
        return None
    body = '2,3-dihydro-1lambda6,2-benzothiazole-1,1-dione'
    name = join_prefix_parent(format_prefixes(prefixes, hierarchical_enclosures=True), body)
    if not name:
        return None
    unspecified = tuple(site for site in Chem.FindPotentialStereo(mol)
                        if str(site.specified) != 'Specified')
    issues = ((SelectionLimitation(
        'stereochemistry', 'P-91.1/P-93.1.1',
        f'The input leaves {len(unspecified)} potentially stereogenic unit(s) unspecified; '
        'the name does not assign their configurations.'),) if unspecified else ())
    return MethodDecision(name, 'substitutive', frozenset({sidx}), heavy, RULE,
                          ((sidx, '1'),), component_roles=tuple(roles),
                          preference_issues=issues)
