"""Conservative name-only reconstruction of small mixed alkane-diyl esters.

IUPAC Blue Book P-65.6.3.3.3 and .3.3.3.2 describe a multivalent alcoholic
component followed by acid-anion words, with locants when necessary:
https://iupac.qmul.ac.uk/BlueBook/PDF/P6.pdf

An independent parser supplies the named diol and each carboxylate. Both
possible placements are constructed; only one resulting graph identity is
accepted. No original molecule, expected graph, NISPO producer or ranking is
available to this helper. This bounded supplement does not verify PIN choice.
"""
from __future__ import annotations

from collections.abc import Callable
from itertools import permutations
import re

from rdkit import Chem


_NAME = re.compile(
    r"((ethane|propane|butane|pentane|hexane)-([1-6]),([1-6])-diyl) "
    r"([a-z]+ate) ([a-z]+ate)"
)
_CARBONS = {'ethane': 2, 'propane': 3, 'butane': 4, 'pentane': 5, 'hexane': 6}
_MULTIPLIER = re.compile(r"(?:di|tri|tetra|penta|hexa|bis|tris)")


def _ordinary(mol):
    return (mol is not None and 0 < mol.GetNumAtoms() <= 12
            and len(Chem.GetMolFrags(mol)) == 1 and not mol.GetRingInfo().NumRings()
            and not mol.GetStereoGroups()
            and all(a.GetAtomicNum() in {6, 8} and not a.GetIsotope()
                    and not a.GetNumRadicalElectrons() and not int(a.GetChiralTag())
                    and not a.HasQuery() and not a.GetAtomMapNum() for a in mol.GetAtoms())
            and all(not int(b.GetStereo()) and not int(b.GetBondDir()) and not b.HasQuery()
                    for b in mol.GetBonds()))


def _carbon_chain(mol, count):
    carbons = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6}
    if len(carbons) != count:
        return False
    edges = [b for b in mol.GetBonds()
             if b.GetBeginAtomIdx() in carbons and b.GetEndAtomIdx() in carbons]
    return (len(edges) == count-1 and all(b.GetBondType() == Chem.BondType.SINGLE for b in edges)
            and all(sum(n.GetIdx() in carbons for n in mol.GetAtomWithIdx(i).GetNeighbors()) <= 2
                    for i in carbons))


def _diol_sites(mol, count):
    if (not _ordinary(mol) or any(a.GetFormalCharge() for a in mol.GetAtoms())
            or not _carbon_chain(mol, count)
            or any(b.GetBondType() != Chem.BondType.SINGLE for b in mol.GetBonds())):
        return None
    oxygens = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 8]
    if len(oxygens) != 2 or any(a.GetDegree() != 1 or a.GetTotalNumHs() != 1
                              or a.GetNeighbors()[0].GetAtomicNum() != 6 for a in oxygens):
        return None
    if len({a.GetNeighbors()[0].GetIdx() for a in oxygens}) != 2:
        return None
    return tuple(a.GetIdx() for a in oxygens)


def _carboxylate(mol):
    if not _ordinary(mol):
        return None
    oxygens = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 8]
    charged = [a for a in mol.GetAtoms() if a.GetFormalCharge()]
    count = sum(a.GetAtomicNum() == 6 for a in mol.GetAtoms())
    if (len(oxygens) != 2 or len(charged) != 1 or not 1 <= count <= 8
            or not _carbon_chain(mol, count)):
        return None
    leaving = charged[0]
    if (leaving.GetAtomicNum() != 8 or leaving.GetFormalCharge() != -1
            or leaving.GetDegree() != 1 or leaving.GetTotalNumHs()):
        return None
    carbonyl = leaving.GetNeighbors()[0]
    other = next(a for a in oxygens if a.GetIdx() != leaving.GetIdx())
    if (carbonyl.GetAtomicNum() != 6 or other.GetDegree() != 1
            or other.GetNeighbors()[0].GetIdx() != carbonyl.GetIdx()
            or mol.GetBondBetweenAtoms(carbonyl.GetIdx(), other.GetIdx()).GetBondType() != Chem.BondType.DOUBLE
            or mol.GetBondBetweenAtoms(carbonyl.GetIdx(), leaving.GetIdx()).GetBondType() != Chem.BondType.SINGLE
            or sum(b.GetBondType() == Chem.BondType.DOUBLE for b in mol.GetBonds()) != 1):
        return None
    return leaving.GetIdx(), carbonyl.GetIdx()


def recover_mixed_diyl_ester(name: str, parse_component: Callable) -> str | None:
    """Return a unique uncharged ester graph or decline the bounded grammar.

    ``parse_component`` receives only a component name and returns an
    independently parsed SMILES, molecule or ``None``. Inputs are copied.
    Multiplicity, explicit site assignments, isotopes, stereo, rings and
    non-carboxylate acid components are outside this supplement's scope.
    """
    if not isinstance(name, str):
        return None
    match = _NAME.fullmatch(name)
    if match is None:
        return None
    root, first, second = match[2], int(match[3]), int(match[4])
    if first == second or max(first, second) > _CARBONS[root]:
        return None
    acid_words = match[5], match[6]
    if acid_words[0] == acid_words[1] or any(_MULTIPLIER.match(word) for word in acid_words):
        return None
    try:
        def parsed(component):
            value = parse_component(component)
            if value is None:
                return None
            return Chem.MolFromSmiles(value) if isinstance(value, str) else Chem.Mol(value)

        diol = parsed(match[1][:-4]+'diol')
        sites = _diol_sites(diol, _CARBONS[root])
        if sites is None:
            return None
        acids = [parsed(word) for word in acid_words]
        roles = [_carboxylate(acid) for acid in acids]
        if any(role is None for role in roles):
            return None
        identities = set()
        for assignment in permutations(range(2)):
            graph = Chem.Mol(diol)
            for site, acid_index in zip(sites, assignment, strict=True):
                leaving, carbonyl = roles[acid_index]
                fragment = Chem.RWMol(acids[acid_index])
                fragment.RemoveAtom(leaving)
                carbonyl -= leaving < carbonyl
                offset = graph.GetNumAtoms()
                assembled = Chem.RWMol(Chem.CombineMols(graph, fragment.GetMol()))
                oxygen = assembled.GetAtomWithIdx(site)
                oxygen.SetNumExplicitHs(0)
                oxygen.SetNoImplicit(True)
                assembled.AddBond(site, offset+carbonyl, Chem.BondType.SINGLE)
                graph = assembled.GetMol()
            Chem.SanitizeMol(graph)
            if (any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in graph.GetAtoms())
                    or len(Chem.GetMolFrags(graph)) != 1
                    or Chem.FindMolChiralCenters(graph, includeUnassigned=True)):
                return None
            identities.add(Chem.MolToSmiles(graph, isomericSmiles=True))
        return identities.pop() if len(identities) == 1 else None
    except (TypeError, ValueError, RuntimeError, AttributeError):
        return None
