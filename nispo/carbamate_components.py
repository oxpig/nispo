"""Graph proofs for bounded subordinate alkoxycarbonylamino components."""
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AlkoxycarbonylaminoProof:
    root: int
    host: int
    carbonyl: int
    oxo: int
    ester_oxygen: int
    organyl_root: int
    organyl_atoms: frozenset[int]
    atoms: frozenset[int]
    text: str
    rule: str = 'P-41/P-65.2.1.1/P-65.6.3/P-66.1.1.4.3'


def mapped_alkoxycarbonylamino_proof(mol, Chem, root, host):
    """Prove a simple NH-C(=O)-O-alkyl branch on the unmodified graph.

    Only fully unannotated, neutral methyl, ethyl, propyl, butyl and
    tert-butyl O-arms are admitted. The actual mapped writer must reproduce
    the independently assembled component text and exactly own its scope.
    """
    from .branch_rules import collect_subtree
    from .carbamate_methods import _carbon_organyl
    from .component_mapping import mapped_branch
    from .extended_inventory import selection_inventory
    from .formatting import alkyl_to_alkoxy

    nitrogen = mol.GetAtomWithIdx(root)
    bond = mol.GetBondBetweenAtoms(root, host)
    if (nitrogen.GetAtomicNum() != 7 or nitrogen.IsInRing()
            or nitrogen.GetFormalCharge() or nitrogen.GetIsotope()
            or nitrogen.GetNumRadicalElectrons() or int(nitrogen.GetChiralTag())
            or nitrogen.GetTotalNumHs(includeNeighbors=True) != 1
            or mol.GetAtomWithIdx(host).GetAtomicNum() != 6
            or bond is None or bond.GetBondTypeAsDouble() != 1):
        return None
    scope = frozenset(collect_subtree(mol, root, host))
    if (not scope or host in scope or mol.GetStereoGroups()
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() == 1
                   or mol.GetAtomWithIdx(i).GetIsotope()
                   or mol.GetAtomWithIdx(i).GetFormalCharge()
                   or mol.GetAtomWithIdx(i).GetNumRadicalElectrons()
                   or int(mol.GetAtomWithIdx(i).GetChiralTag()) for i in scope)
            or any(int(b.GetStereo()) or int(b.GetBondDir()) for b in mol.GetBonds()
                   if b.GetBeginAtomIdx() in scope and b.GetEndAtomIdx() in scope)):
        return None
    nitrogens = [a for a in nitrogen.GetNeighbors()
                 if a.GetIdx() != host and a.GetAtomicNum() > 1]
    if len(nitrogens) != 1 or nitrogens[0].GetAtomicNum() != 6:
        return None
    carbon = nitrogens[0]
    carbonyl = carbon.GetIdx()
    others = [a for a in carbon.GetNeighbors() if a.GetIdx() != root]
    oxo = [a for a in others if a.GetAtomicNum() == 8
           and mol.GetBondBetweenAtoms(carbonyl, a.GetIdx()).GetBondTypeAsDouble() == 2]
    oxygen = [a for a in others if a.GetAtomicNum() == 8
              and mol.GetBondBetweenAtoms(carbonyl, a.GetIdx()).GetBondTypeAsDouble() == 1]
    if (carbon.IsInRing() or carbon.GetDegree() != 3
            or mol.GetBondBetweenAtoms(root, carbonyl).GetBondTypeAsDouble() != 1
            or len(others) != 2 or len(oxo) != 1 or len(oxygen) != 1
            or oxo[0].GetDegree() != 1 or oxygen[0].GetDegree() != 2
            or oxygen[0].GetTotalNumHs(includeNeighbors=True)):
        return None
    bridge = oxygen[0].GetIdx()
    alkyl = [a for a in oxygen[0].GetNeighbors() if a.GetIdx() != carbonyl]
    if (len(alkyl) != 1 or alkyl[0].GetAtomicNum() != 6
            or mol.GetBondBetweenAtoms(bridge, alkyl[0].GetIdx()).GetBondTypeAsDouble() != 1):
        return None
    organyl_root = alkyl[0].GetIdx()
    organyl, organyl_atoms = _carbon_organyl(mol, Chem, organyl_root, bridge)
    if (organyl not in {'methyl', 'ethyl', 'propyl', 'butyl', 'tert-butyl'}
            or any(mol.GetAtomWithIdx(i).IsInRing() for i in organyl_atoms)
            or set(scope) != {root, carbonyl, oxo[0].GetIdx(), bridge, *organyl_atoms}):
        return None
    for item in Chem.FindPotentialStereo(mol):
        if item.type == Chem.StereoType.Atom_Tetrahedral and item.centeredOn in scope:
            return None
        if item.type == Chem.StereoType.Bond_Double:
            stereo_bond = mol.GetBondWithIdx(item.centeredOn)
            if {stereo_bond.GetBeginAtomIdx(), stereo_bond.GetEndAtomIdx()} & scope:
                return None
    inventory = selection_inventory(mol)
    target = [g for g in inventory.groups if g.kind == 'carbamate' and g.center == carbonyl]
    frontier = getattr(inventory, 'source_frontier', None)
    if (not inventory.supported or len(target) != 1 or len(inventory.principal) != 1
            or frontier is None or frontier.status not in {'selected', 'tie'}
            or frontier.unknown_pairs or frontier.cycle
            or tuple(frontier.groups) != tuple(inventory.principal)
            or inventory.principal[0].seniority >= target[0].seniority
            or set(inventory.principal[0].atoms) & scope):
        return None
    alkoxy = alkyl_to_alkoxy(organyl)
    expected = (f'({alkoxy}carbonyl)amino' if '-' in alkoxy
                else f'{alkoxy}carbonylamino')
    text, owned = mapped_branch(mol, Chem, root, host)
    if text != expected or owned != scope:
        return None
    return AlkoxycarbonylaminoProof(root, host, carbonyl, oxo[0].GetIdx(), bridge,
        organyl_root, frozenset(organyl_atoms), scope, text)
