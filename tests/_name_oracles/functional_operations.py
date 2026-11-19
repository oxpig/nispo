"""Independent name-only proofs for two bounded OPSIN grammar gaps.

P-72.2.2.2.3 removes two hydrons from a primary amine (aminediide).
P-65.7.6.1 joins two identical monobasic acids and one dibasic acid by
two anhydride bridges. Sources: https://iupac.qmul.ac.uk/BlueBook/P7.html
and https://iupac.qmul.ac.uk/BlueBook/P6.html .

These operations use independently parsed name components, never an input
molecule or production naming code. Annotations and ambiguous sites decline.
"""
from collections import Counter
import re

from rdkit import Chem


def _plain_base(value, elements):
    if value is None:
        return None
    graph = Chem.MolFromSmiles(value) if isinstance(value, str) else Chem.Mol(value)
    if (graph is None or not graph.GetNumAtoms() or len(Chem.GetMolFrags(graph)) != 1
            or graph.GetStereoGroups() or Chem.GetMolSubstanceGroups(graph)
            or any(a.GetAtomicNum() not in elements or a.GetFormalCharge()
                   or a.GetIsotope() or a.GetNumRadicalElectrons() or a.GetChiralTag()
                   or a.GetAtomMapNum() or a.HasQuery() for a in graph.GetAtoms())
            or any(b.GetStereo() or b.GetBondDir() or b.HasQuery() for b in graph.GetBonds())):
        return None
    return Chem.AddHs(graph)


def _census(graph):
    return Counter(a.GetAtomicNum() for a in Chem.AddHs(graph).GetAtoms())


def recover_amine_dianion(name, parse_raw):
    """Decode a sole unannotated primary hydrocarbylamine dianion."""
    if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9,\-]*aminediide", name):
        return None
    try:
        base = _plain_base(parse_raw(name[:-5]), {1, 6, 7})
        if base is None:
            return None
        nitrogens = [a for a in base.GetAtoms() if a.GetAtomicNum() == 7]
        if len(nitrogens) != 1:
            return None
        nitrogen = nitrogens[0]
        hydrogens = [a.GetIdx() for a in nitrogen.GetNeighbors() if a.GetAtomicNum() == 1]
        if (nitrogen.GetDegree() != 3 or len(hydrogens) != 2
                or any(b.GetBondType() != Chem.BondType.SINGLE for b in nitrogen.GetBonds())):
            return None
        expected = _census(base)
        expected[1] -= 2
        working = Chem.RWMol(base)
        atom = working.GetAtomWithIdx(nitrogen.GetIdx())
        atom.SetFormalCharge(-2)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)
        for index in sorted(hydrogens, reverse=True):
            working.RemoveAtom(index)
        graph = working.GetMol()
        Chem.SanitizeMol(graph)
        if _census(graph) != +expected or Chem.GetFormalCharge(graph) != -2:
            return None
        return Chem.MolToSmiles(Chem.RemoveHs(graph), isomericSmiles=True)
    except (TypeError, ValueError, RuntimeError):
        return None


def _acid_sites(graph, count):
    """Require only saturated acyclic carbon and the stated carboxyl groups."""
    if (graph is None or graph.GetRingInfo().NumRings()
            or sum(a.GetAtomicNum() == 8 for a in graph.GetAtoms()) != 2 * count):
        return ()
    sites = []
    for carbon in graph.GetAtoms():
        if carbon.GetAtomicNum() != 6:
            continue
        oxo = [b.GetOtherAtom(carbon) for b in carbon.GetBonds()
               if b.GetBondType() == Chem.BondType.DOUBLE
               and b.GetOtherAtom(carbon).GetAtomicNum() == 8]
        hydroxyl = [a for a in carbon.GetNeighbors() if a.GetAtomicNum() == 8
                    and graph.GetBondBetweenAtoms(carbon.GetIdx(), a.GetIdx()).GetBondType()
                    == Chem.BondType.SINGLE
                    and a.GetDegree() == 2
                    and sum(n.GetAtomicNum() == 1 for n in a.GetNeighbors()) == 1]
        if len(oxo) == len(hydroxyl) == 1 and oxo[0].GetDegree() == 1:
            oxygen = hydroxyl[0]
            hydrogen = next(a for a in oxygen.GetNeighbors() if a.GetAtomicNum() == 1)
            sites.append((carbon.GetIdx(), oxygen.GetIdx(), hydrogen.GetIdx()))
    if len(sites) != count or any(
            b.GetBondType() != Chem.BondType.SINGLE
            and not (b.GetBondType() == Chem.BondType.DOUBLE
                     and {b.GetBeginAtom().GetAtomicNum(), b.GetEndAtom().GetAtomicNum()} == {6, 8})
            for b in graph.GetBonds()):
        return ()
    return tuple(sites)


def recover_symmetric_dianhydride(name, parse_raw):
    """Decode two identical monobasic end groups around one dibasic acid."""
    if not isinstance(name, str):
        return None
    match = re.fullmatch(r"di([a-z][a-z0-9,\-]*) ([a-z][a-z0-9,\-]*) dianhydride", name)
    if match is None:
        return None
    try:
        terminal = _plain_base(parse_raw(match[1] + " acid"), {1, 6, 8})
        central = _plain_base(parse_raw(match[2] + " acid"), {1, 6, 8})
        ends, middle = _acid_sites(terminal, 1), _acid_sites(central, 2)
        if not ends or not middle:
            return None
        # Both ends are identical, so the two possible assignments are the
        # same graph. Each linkage eliminates exactly one water molecule.
        joined = Chem.CombineMols(Chem.CombineMols(central, terminal), terminal)
        expected = _census(joined)
        expected[1] -= 4
        expected[8] -= 2
        working = Chem.RWMol(joined)
        removed = []
        for index, (_, oxygen, hydrogen) in enumerate(middle):
            offset = central.GetNumAtoms() + index * terminal.GetNumAtoms()
            acyl, end_oxygen, end_hydrogen = (offset + atom for atom in ends[0])
            working.AddBond(oxygen, acyl, Chem.BondType.SINGLE)
            removed.extend((hydrogen, end_oxygen, end_hydrogen))
        for index in sorted(removed, reverse=True):
            working.RemoveAtom(index)
        graph = working.GetMol()
        Chem.SanitizeMol(graph)
        if (_census(graph) != +expected or Chem.GetFormalCharge(graph)
                or len(Chem.GetMolFrags(graph)) != 1):
            return None
        return Chem.MolToSmiles(Chem.RemoveHs(graph), isomericSmiles=True)
    except (TypeError, ValueError, RuntimeError):
        return None
