"""Name-only reconstruction for one unambiguous alcohol-suffix H isotope.

P-82.2.1 permits inserting a nuclide descriptor before the part modified.
P-82.3.1 illustrates the descriptor immediately before ``ol``;
P-82.6.1.1 illustrates the corresponding hydrogen-labelled alcohol spelling.
The rule sources and the independent origin are recorded in provenance.json.

This is not a general IUPAC parser.  It delegates the unlabelled constitution
to an independent parser, supports only one terminal 2H/3H ol insertion, and
declines whenever the hydrogen's placement is not unique.  No original input
molecule, expected output structure or production naming function is used.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from rdkit import Chem


# A deliberately small grammar excludes nested blocks, other isotope labels,
# additional characteristic-group words and stereochemical descriptors.  In
# particular, parsing the unlabelled base must not assign CIP descriptors
# whose meaning could change after this isotope is installed.
_OL_HYDROGEN = re.compile(r"([a-z][a-z0-9,\-]*)\(([23])H(?:1)?\)ol")
_HYDROXY_HYDROGEN = re.compile(r"(\d+)-\(([23])H(?:1)?\)hydroxy([a-z][a-z0-9,\-]*)")


def recover_ol_hydrogen_isotope(name: str, parse_base: Callable) -> str | None:
    """Return the uniquely reconstructed isomeric SMILES, else ``None``.

    ``parse_base`` receives only the derived unlabelled name and returns an
    independently parsed RDKit molecule, a SMILES string, or ``None``.  The
    callback's molecule is copied before a hydrogen is labelled.
    """
    if not isinstance(name, str):
        return None
    match = _OL_HYDROGEN.fullmatch(name)
    prefix = _HYDROXY_HYDROGEN.fullmatch(name)
    base_name = match[1]+"ol" if match else None
    if prefix is not None:
        match = prefix
        base_name = prefix[1]+'-hydroxy'+prefix[3]
    if match is None:
        return None
    parsed = parse_base(base_name)
    if parsed is None:
        return None
    try:
        base = Chem.MolFromSmiles(parsed) if isinstance(parsed, str) else Chem.Mol(parsed)
        if base is None or not base.GetNumAtoms() or len(Chem.GetMolFrags(base)) != 1:
            return None
        # The admitted spelling contains exactly this single nuclide token.
        # Other labels, charges, radicals and stereo groups need a different
        # independently proved parser scope.
        if (base.GetStereoGroups()
                or any(atom.GetIsotope() or atom.GetFormalCharge()
                       or atom.GetNumRadicalElectrons() or int(atom.GetChiralTag())
                       or atom.HasQuery() for atom in base.GetAtoms())
                or any(int(bond.GetStereo()) or int(bond.GetBondDir()) or bond.HasQuery()
                       for bond in base.GetBonds())):
            return None
        graph = Chem.AddHs(base)
        candidates = []
        for oxygen in graph.GetAtoms():
            if oxygen.GetAtomicNum() != 8 or oxygen.GetDegree() != 2:
                continue
            hydrogens = [atom for atom in oxygen.GetNeighbors() if atom.GetAtomicNum() == 1]
            carbons = [atom for atom in oxygen.GetNeighbors() if atom.GetAtomicNum() == 6]
            if len(hydrogens) != 1 or len(carbons) != 1:
                continue
            if any(graph.GetBondBetweenAtoms(oxygen.GetIdx(), atom.GetIdx()).GetBondType()
                   != Chem.BondType.SINGLE for atom in (*hydrogens, *carbons)):
                continue
            candidates.append(hydrogens[0].GetIdx())
        if len(candidates) != 1:
            return None
        graph.GetAtomWithIdx(candidates[0]).SetIsotope(int(match[2]))
        return Chem.MolToSmiles(Chem.RemoveHs(graph), isomericSmiles=True)
    except (TypeError, ValueError, RuntimeError):
        return None
