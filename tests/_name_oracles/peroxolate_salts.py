"""Name-only proof for one plain light-alkali alkylperoxolate salt.

IUPAC P-72.2.2.2.2 defines peroxolate by removing the hydroxy hydron from
peroxol; P-77.2.1 gives binary cation/anion salt names. General source:
https://iupac.qmul.ac.uk/BlueBook/P7.html#P-72.2.2.2.2 .

Only an independently parsed neutral peroxol and explicit metal(1+) are
used. This intentionally excludes other heteroatoms, rings, annotations,
additional components and ambiguous peroxide sites. It never sees an
original test graph or calls NISPO. Raw-parser disagreement remains visible.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable
import re

from rdkit import Chem


_SALT = re.compile(r"(lithium|sodium|potassium) ([a-z0-9][a-z0-9,\-]*)peroxolate")
_METALS = {"lithium": 3, "sodium": 11, "potassium": 19}


def _copy_parsed(value):
    if value is None:
        return None
    return Chem.MolFromSmiles(value) if isinstance(value, str) else Chem.Mol(value)


def _has_annotations(graph):
    return (bool(graph.GetStereoGroups())
            or any(a.GetIsotope() or a.GetNumRadicalElectrons() or int(a.GetChiralTag())
                   or a.GetAtomMapNum() or a.HasQuery() for a in graph.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) or b.HasQuery()
                   for b in graph.GetBonds()))


def recover_alkali_peroxolate_salt(name: str, parse_raw: Callable) -> str | None:
    """Recover one proved cation/peroxolate graph from independent name parts.

    ``parse_raw`` must independently parse each supplied name without using
    this supplement. Its returned molecules are copied before editing.
    """
    if not isinstance(name, str):
        return None
    match = _SALT.fullmatch(name)
    if match is None or "perox" in match[2]:
        return None
    metal_name, parent = match.groups()
    try:
        base = _copy_parsed(parse_raw(parent + "peroxol"))
        if (base is None or not base.GetNumAtoms() or len(Chem.GetMolFrags(base)) != 1
                or base.GetRingInfo().NumRings() or _has_annotations(base)
                or any(a.GetFormalCharge() or a.GetAtomicNum() not in {1, 6, 8} for a in base.GetAtoms())
                or any(b.GetBondType() not in {Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
                                              Chem.BondType.TRIPLE} for b in base.GetBonds())
                or not any(a.GetAtomicNum() == 6 for a in base.GetAtoms())
                or sum(a.GetAtomicNum() == 8 for a in base.GetAtoms()) != 2):
            return None
        expanded = Chem.AddHs(base)
        if any(a.GetTotalValence() != {1: 1, 6: 4, 8: 2}[a.GetAtomicNum()]
               for a in expanded.GetAtoms()):
            return None
        candidates = []
        for terminal in expanded.GetAtoms():
            if terminal.GetAtomicNum() != 8 or terminal.GetDegree() != 2:
                continue
            hs = [a for a in terminal.GetNeighbors() if a.GetAtomicNum() == 1]
            os = [a for a in terminal.GetNeighbors() if a.GetAtomicNum() == 8]
            if len(hs) != 1 or len(os) != 1 or os[0].GetDegree() != 2:
                continue
            carbon = [a for a in os[0].GetNeighbors() if a.GetAtomicNum() == 6]
            if len(carbon) != 1:
                continue
            pairs = ((terminal, hs[0]), (terminal, os[0]), (os[0], carbon[0]))
            if all(expanded.GetBondBetweenAtoms(a.GetIdx(), b.GetIdx()).GetBondType()
                   == Chem.BondType.SINGLE for a, b in pairs):
                candidates.append((terminal.GetIdx(), hs[0].GetIdx()))
        if len(candidates) != 1:
            return None
        cation = _copy_parsed(parse_raw(metal_name + "(1+)"))
        if cation is None or cation.GetNumAtoms() != 1 or cation.GetNumBonds() or _has_annotations(cation):
            return None
        metal = cation.GetAtomWithIdx(0)
        if (metal.GetAtomicNum() != _METALS[metal_name] or metal.GetFormalCharge() != 1
                or metal.GetTotalNumHs(includeNeighbors=True)):
            return None
        terminal_id, hydrogen_id = candidates[0]
        working = Chem.RWMol(expanded)
        oxygen = working.GetAtomWithIdx(terminal_id)
        oxygen.SetFormalCharge(-1)
        oxygen.SetNumExplicitHs(0)
        oxygen.SetNoImplicit(True)
        working.RemoveAtom(hydrogen_id)
        anion = working.GetMol()
        Chem.SanitizeMol(anion)
        combined = Chem.CombineMols(Chem.RemoveHs(anion), cation)
        Chem.SanitizeMol(combined)
        expected = Counter(a.GetAtomicNum() for a in expanded.GetAtoms())
        expected[1] -= 1
        expected[_METALS[metal_name]] += 1
        observed = Counter(a.GetAtomicNum() for a in Chem.AddHs(combined).GetAtoms())
        if (observed != +expected or len(Chem.GetMolFrags(combined)) != 2
                or sum(a.GetFormalCharge() for a in combined.GetAtoms()) != 0
                or sum(a.GetFormalCharge() == -1 for a in combined.GetAtoms()) != 1
                or _has_annotations(combined)):
            return None
        return Chem.MolToSmiles(combined, isomericSmiles=True)
    except (TypeError, ValueError, RuntimeError):
        return None
