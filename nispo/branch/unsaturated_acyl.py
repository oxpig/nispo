"""Original-graph acid-derived acyl prefixes for bounded carbon trees."""
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class UnsaturatedAcylProof:
    text: str
    root: int
    host: int
    oxygen: int
    atoms: frozenset[int]
    carbon_ligand: frozenset[int]
    candidate_mappings: tuple
    rule: str = 'P-44.3.2/P-44.4.1/P-45.2/P-65.1.7.3.1/P-65.1.7.4.1'


def unsaturated_acyl_prefix(mol, Chem, root, host):
    """Prove a whole ordinary O-attached acyl before rendering its suffix.

    The acid-derived carbonyl is carbon 1. Method-1 ligand paths all start at
    the alpha carbon; adding carbonyl carbon 1 to each path preserves their
    length and chemical ranking. The bounded P-46 frontier has the same
    applicable P-44/P-45 comparisons for these unannotated carbon-only trees.
    Potential stereogenic units are excluded because P-44.4.1.12 would need
    a separate comparison before the prefix tie-breaks.
    """
    from ..functionalized_carbon_prefixes import _scope
    from ..unsaturated_substituent_selection import unsaturated_substituent_frontier
    from ..unsaturated_substituent_methods import _prefix_names
    from ..constants import ROOTS
    from ..formatting import format_prefixes, loc_unsat

    if (mol is None or mol.GetStereoGroups() or not isinstance(root, int)
            or not isinstance(host, int) or root == host
            or not 0 <= root < mol.GetNumAtoms() or not 0 <= host < mol.GetNumAtoms()):
        return None
    acyl, linking = mol.GetAtomWithIdx(root), mol.GetAtomWithIdx(host)
    bond = mol.GetBondBetweenAtoms(root, host)
    if (acyl.GetAtomicNum() != 6 or acyl.IsInRing() or acyl.GetFormalCharge()
            or acyl.GetIsotope() or acyl.GetNumRadicalElectrons() or int(acyl.GetChiralTag())
            or linking.GetAtomicNum() != 8 or linking.IsInRing() or linking.GetFormalCharge()
            or linking.GetIsotope() or linking.GetNumRadicalElectrons()
            or linking.GetTotalNumHs(includeNeighbors=True) or bond is None
            or bond.GetBondTypeAsDouble() != 1):
        return None
    neighbors = [n for n in acyl.GetNeighbors() if n.GetIdx() != host and n.GetAtomicNum() > 1]
    oxygens = [n for n in neighbors if n.GetAtomicNum() == 8
               and mol.GetBondBetweenAtoms(root, n.GetIdx()).GetBondTypeAsDouble() == 2]
    carbons = [n for n in neighbors if n.GetAtomicNum() == 6
               and mol.GetBondBetweenAtoms(root, n.GetIdx()).GetBondTypeAsDouble() == 1]
    if len(neighbors) != 2 or len(oxygens) != 1 or len(carbons) != 1:
        return None
    oxygen, alpha = oxygens[0].GetIdx(), carbons[0].GetIdx()
    oxo = mol.GetAtomWithIdx(oxygen)
    if (oxo.GetDegree() != 1 or oxo.GetFormalCharge() or oxo.GetIsotope()
            or oxo.GetNumRadicalElectrons() or oxo.GetTotalNumHs(includeNeighbors=True)):
        return None
    whole, ligand = _scope(mol, root, host), _scope(mol, alpha, root)
    if (whole is None or ligand is None or whole != ligand | {root, oxygen}
            or not 2 <= len(ligand) <= 32):
        return None
    if (linking.HasQuery() or linking.GetAtomMapNum()
            or any(mol.GetAtomWithIdx(i).HasQuery()
                   or mol.GetAtomWithIdx(i).GetAtomMapNum()
                   or mol.GetAtomWithIdx(i).HasProp('_UnknownStereo') for i in whole)
            or any(b.HasQuery() for b in mol.GetBonds()
                   if b.GetBeginAtomIdx() in whole or b.GetEndAtomIdx() in whole)
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in ligand)
            or any(n.GetAtomicNum() == 1 and n.GetIsotope()
                   for i in whole for n in mol.GetAtomWithIdx(i).GetNeighbors())):
        return None
    bonds = [b for b in mol.GetBonds()
             if b.GetBeginAtomIdx() in ligand and b.GetEndAtomIdx() in ligand]
    if not any(b.GetBondTypeAsDouble() in {2, 3} for b in bonds):
        return None
    try:
        potential = Chem.FindPotentialStereo(mol)
        for site in potential:
            if site.type == Chem.StereoType.Atom_Tetrahedral and site.centeredOn in whole:
                return None
            if site.type == Chem.StereoType.Bond_Double:
                stereo_bond = mol.GetBondWithIdx(site.centeredOn)
                if {stereo_bond.GetBeginAtomIdx(), stereo_bond.GetEndAtomIdx()} & whole:
                    return None
            elif site.type != Chem.StereoType.Atom_Tetrahedral:
                # Other stereo types need a separate source-mapped scope.
                return None
    except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
        return None
    initial = unsaturated_substituent_frontier(mol, alpha, root, ligand, method=1)
    if initial is None or initial.status == 'incomplete' or not initial.frontier:
        return None
    names = _prefix_names(mol, Chem, initial.frontier)
    if names is None:
        return None
    selected = unsaturated_substituent_frontier(mol, alpha, root, ligand, method=1,
                                               prefix_names=names)
    if selected is None or selected.status in {'unresolved', 'incomplete'} or not selected.frontier:
        return None
    rendered, mappings = [], []
    for candidate in selected.frontier:
        length = len(candidate.path_atom_ids) + 1
        root_name = ROOTS.get(length)
        if not root_name or candidate.path_atom_ids[0] != alpha:
            return None
        double = tuple(i + 1 for i in candidate.double_bond_locants)
        triple = tuple(i + 1 for i in candidate.multiple_bond_locants
                       if i not in candidate.double_bond_locants)
        if not double and not triple:
            return None
        if double and triple:
            stem = loc_unsat(root_name, list(double), 'ene') + '-' + loc_unsat('', list(triple), 'yne')
        elif double:
            stem = loc_unsat(root_name, list(double), 'ene')
        else:
            stem = loc_unsat(root_name, list(triple), 'yne')
        prefixes = [(str(prefix.locant + 1), prefix.proved_name or '')
                    for prefix in candidate.prefixes]
        if any(not text for _, text in prefixes):
            return None
        rendered.append(format_prefixes(prefixes) + stem + 'oyl')
        mappings.append((root, candidate.path_atom_ids, candidate.atom_locants,
                         candidate.multiple_bond_locants, candidate.double_bond_locants))
    if len(set(rendered)) != 1:
        return None
    return UnsaturatedAcylProof(rendered[0], root, host, oxygen, whole, ligand, tuple(mappings))
