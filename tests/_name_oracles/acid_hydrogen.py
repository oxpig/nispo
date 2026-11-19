"""Name-only recovery of one uniquely placed carboxylic-acid H isotope.

Independent source: IUPAC Blue Book P-82.2.1 and P-82.2.4,
https://iupac.qmul.ac.uk/BlueBook/P8.html#P-82.2.4 . These rules place a
nuclide descriptor before the name part containing its atom. A single 2H/3H
descriptor immediately before ``carboxylic acid`` therefore identifies its
OH hydrogen when the independently parsed base has exactly one such group.

The decoder is intentionally narrower than the source rule: no other isotope,
charge, radical, stereo, disconnected component or ambiguous acid site is
admitted. It receives a name and an independent base-name parser only; it
never receives an original molecule or calls a production naming function.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable
import re

from rdkit import Chem


# Parentheses/brackets and prime locants admit compound ring parent names.
# Uppercase letters are deliberately excluded from the parent: other nuclide
# blocks, indicated-H and absolute stereodescriptors need a separate decoder.
_ACID_HYDROGEN = re.compile(r"([a-z0-9,'\[\]()\-]+)\(([23])H(?:1)?\)(carboxylic acid|oic acid)")


def recover_acid_hydrogen_isotope(name: str, parse_base: Callable) -> str | None:
    """Recover a single suffix H isotope iff its acid OH site is unique.

    The callback receives only the unlabelled base name and may return SMILES,
    an RDKit molecule or None. A returned molecule is copied before editing.
    """
    if not isinstance(name, str):
        return None
    match = _ACID_HYDROGEN.fullmatch(name)
    if match is None:
        return None
    try:
        parsed = parse_base(match[1] + match[3])
        if parsed is None:
            return None
        base = Chem.MolFromSmiles(parsed) if isinstance(parsed, str) else Chem.Mol(parsed)
        if base is None or not base.GetNumAtoms() or len(Chem.GetMolFrags(base)) != 1:
            return None
        if (base.GetStereoGroups()
                or any(a.GetIsotope() or a.GetFormalCharge() or a.GetNumRadicalElectrons()
                       or int(a.GetChiralTag()) or a.HasQuery() for a in base.GetAtoms())
                or any(int(b.GetStereo()) or int(b.GetBondDir()) or b.HasQuery()
                       for b in base.GetBonds())):
            return None
        graph = Chem.AddHs(base)
        candidates = []
        for carbon in graph.GetAtoms():
            if carbon.GetAtomicNum() != 6 or carbon.GetDegree() != 3 or carbon.GetTotalValence() != 4:
                continue
            carbonyl, hydroxyl, parent = [], [], []
            for atom in carbon.GetNeighbors():
                bond = graph.GetBondBetweenAtoms(carbon.GetIdx(), atom.GetIdx())
                if atom.GetAtomicNum() == 8 and bond.GetBondType() == Chem.BondType.DOUBLE:
                    if atom.GetDegree() == 1:
                        carbonyl.append(atom)
                elif atom.GetAtomicNum() == 8 and bond.GetBondType() == Chem.BondType.SINGLE:
                    hydrogens = [a for a in atom.GetNeighbors() if a.GetAtomicNum() == 1]
                    if atom.GetDegree() == 2 and len(hydrogens) == 1:
                        hydrogen = hydrogens[0]
                        if graph.GetBondBetweenAtoms(atom.GetIdx(), hydrogen.GetIdx()).GetBondType() == Chem.BondType.SINGLE:
                            hydroxyl.append(hydrogen.GetIdx())
                elif atom.GetAtomicNum() == 6 and bond.GetBondType() == Chem.BondType.SINGLE:
                    parent.append(atom)
            if len(carbonyl) == len(hydroxyl) == len(parent) == 1:
                candidates.append(hydroxyl[0])
        if len(candidates) != 1:
            return None
        mass = int(match[2])
        graph.GetAtomWithIdx(candidates[0]).SetIsotope(mass)
        result = Chem.RemoveHs(graph)
        census = Counter((a.GetAtomicNum(), a.GetIsotope()) for a in result.GetAtoms() if a.GetIsotope())
        if census != {(1, mass): 1}:
            return None
        return Chem.MolToSmiles(result, isomericSmiles=True)
    except (TypeError, ValueError, RuntimeError):
        return None
