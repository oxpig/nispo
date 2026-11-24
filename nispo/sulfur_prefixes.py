"""P-63.1.5/P-65.3/P-66 subordinate sulfur groups on original atoms.

Characteristic groups are typed before generic heteroatom-prefix rendering.
Each N-substituent retains its own mapped scope; no final-name substitution
or capped-fragment stereochemistry is used.
"""
from types import SimpleNamespace


def characteristic_sulfur_prefix(mol, Chem, root, blocked, atoms):
    """Return a mapped prefix, ``None`` for other sulfur chemistry.

    An empty prefix declines a recognized but unproved local construction.
    The bounded functions attach to carbon and have unmodified S/O/N suffix
    atoms; carbon-side isotope/configuration descriptors remain supported.
    """
    sulfur = mol.GetAtomWithIdx(root)
    if sulfur.GetAtomicNum() != 16 or sulfur.IsInRing():
        return None
    host_element = mol.GetAtomWithIdx(blocked).GetAtomicNum()
    if host_element == 7:
        from .component_mapping import has_annotations

        # P-65.3.2.3: an unsubstituted sulfamoyl unit can attach to N.
        # Its single cut must own S, two terminal oxo O and terminal NH2.
        nitrogen = [a for a in sulfur.GetNeighbors() if a.GetIdx() != blocked
                    and a.GetAtomicNum() == 7
                    and mol.GetBondBetweenAtoms(root, a.GetIdx()).GetBondTypeAsDouble() == 1]
        oxygen = [a for a in sulfur.GetNeighbors() if a.GetAtomicNum() == 8
                  and mol.GetBondBetweenAtoms(root, a.GetIdx()).GetBondTypeAsDouble() == 2]
        if (len(atoms) == 4 and sulfur.GetDegree() == 4
                and mol.GetBondBetweenAtoms(root, blocked).GetBondTypeAsDouble() == 1
                and len(nitrogen) == 1 and nitrogen[0].GetDegree() == 1 and nitrogen[0].GetTotalNumHs() == 2
                and len(oxygen) == 2 and all(a.GetDegree() == 1 for a in oxygen)
                and atoms == {root, nitrogen[0].GetIdx(), *(a.GetIdx() for a in oxygen)}
                and not has_annotations(mol, atoms)
                and all(not mol.GetAtomWithIdx(i).GetFormalCharge()
                        and not mol.GetAtomWithIdx(i).GetNumRadicalElectrons() for i in atoms)):
            return 'sulfamoyl', atoms
        return None
    if host_element != 6:
        return None
    if mol.GetBondBetweenAtoms(root, blocked).GetBondTypeAsDouble() != 1:
        return None
    from .hydrazide_components import unsubstituted_hydrazide_prefix
    hydrazide = unsubstituted_hydrazide_prefix(mol, root, blocked, atoms)
    if hydrazide is not None:
        return hydrazide
    from .sulfinamide_prefix_components import sulfinamide_subordinate_prefix
    sulfinamide = sulfinamide_subordinate_prefix(mol, Chem, root, blocked, atoms)
    if sulfinamide is not None:
        return sulfinamide
    neighbors = [a for a in sulfur.GetNeighbors() if a.GetAtomicNum() > 1]
    oxo = [a for a in neighbors if a.GetAtomicNum() == 8
           and mol.GetBondBetweenAtoms(root, a.GetIdx()).GetBondTypeAsDouble() == 2
           and len([n for n in a.GetNeighbors() if n.GetAtomicNum() > 1]) == 1]
    hydroxy = [a for a in neighbors if a.GetAtomicNum() == 8
               and mol.GetBondBetweenAtoms(root, a.GetIdx()).GetBondTypeAsDouble() == 1
               and len([n for n in a.GetNeighbors() if n.GetAtomicNum() > 1]) == 1
               and a.GetTotalNumHs(includeNeighbors=True) == 1]
    nitrogen = [a for a in neighbors if a.GetAtomicNum() == 7
                and mol.GetBondBetweenAtoms(root, a.GetIdx()).GetBondTypeAsDouble() == 1]
    terminal = len(neighbors) == 1 and sulfur.GetTotalNumHs(includeNeighbors=True) == 1
    acid = (len(oxo) in {1, 2} and len(hydroxy) == 1
            and len(neighbors) == 2 + len(oxo))
    amide = len(oxo) == 2 and len(nitrogen) == 1 and len(neighbors) == 4
    if not (terminal or acid or amide):
        return None
    characteristic = {root, *(a.GetIdx() for a in oxo + hydroxy + nitrogen)}
    if any(mol.GetAtomWithIdx(i).GetFormalCharge() or mol.GetAtomWithIdx(i).GetIsotope()
           or mol.GetAtomWithIdx(i).GetNumRadicalElectrons() or int(mol.GetAtomWithIdx(i).GetChiralTag())
           or any(n.GetAtomicNum() == 1 and n.GetIsotope()
                  for n in mol.GetAtomWithIdx(i).GetNeighbors())
           for i in characteristic):
        return '', frozenset()
    if terminal or acid:
        if characteristic != set(atoms):
            return '', frozenset()
        return ('sulfanyl' if terminal else 'sulfo' if len(oxo) == 2 else 'sulfino'), atoms

    from .sulfur_selection import sulfonamide_n_sites
    from .component_mapping import MappedComponent, decorate_component, mapped_branch
    nidx = nitrogen[0].GetIdx()
    group = SimpleNamespace(kind='sulfonamide', center=root, owner=blocked, ligand=nidx)
    roots = sulfonamide_n_sites(mol, group)
    if roots is None:
        return '', frozenset()
    if (sulfur.GetTotalValence() == 6 and not sulfur.GetTotalNumHs(includeNeighbors=True)
            and all(a.GetDegree() == 1 for a in oxo)
            and not nitrogen[0].IsInRing()
            and nitrogen[0].GetTotalValence() == 3
            and nitrogen[0].GetTotalNumHs(includeNeighbors=True) == 2 - len(roots)
            and all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in roots)):
        from .carboxamide_prefixes import acyl_amide_ligands
        from .component_mapping import _record_branch_ring_atoms

        built = acyl_amide_ligands(mol, Chem, nidx, roots, atoms - characteristic, general=True)
        if built is not None:
            text, rings = built
            name = text + 'sulfamoyl'
            _record_branch_ring_atoms(mol, Chem, root, blocked, name, atoms, rings)
            return name, atoms
    prefixes, embedded = [], set()
    for child in roots:
        text, owned = mapped_branch(mol, Chem, child, nidx)
        if not text or owned & (characteristic | embedded):
            return '', frozenset()
        prefixes.append(('N', text))
        embedded.update(owned)
    if characteristic | embedded != set(atoms):
        return '', frozenset()
    mapping = {i: ('N' if i == nidx else '') for i in characteristic}
    name = decorate_component(mol, Chem, MappedComponent(
        'sulfamoyl', tuple(mapping.items()), frozenset(mapping), tuple(prefixes),
        frozenset(embedded), systematic_parent=True, nitrogen_site_locants=True))
    return name, atoms if name else frozenset()
