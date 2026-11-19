"""P-63.1.5 principal thiol, selenol and tellurol suffixes."""

from .naming_methods import MethodDecision


def _branched_carbon_selenol_tellurol(mol, Chem, carbon, groups, branches):
    """Use the corresponding alcohol's chain choice for a saturated tree."""
    if (branches or len(groups) != 1 or groups[0].GetAtomicNum() not in {34, 52}
            or any(b.GetBondTypeAsDouble() != 1 for b in mol.GetBonds())):
        return None
    marker = Chem.RWMol(mol)
    atom = marker.GetAtomWithIdx(groups[0].GetIdx())
    atom.SetAtomicNum(8)
    marker = marker.GetMol()
    try:
        Chem.SanitizeMol(marker)
    except (RuntimeError, ValueError):
        return None
    import re
    from .api import mol_to_iupac

    alcohol = mol_to_iupac(marker)
    match = re.fullmatch(r'([a-z0-9,()\[\]-]+an)-(\d+)-ol', alcohol)
    if match is None:
        return None
    suffix = 'selenol' if groups[0].GetAtomicNum() == 34 else 'tellurol'
    name = f'{match[1]}e-{match[2]}-{suffix}'
    group = frozenset((groups[0].GetIdx(),))
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms())
    return MethodDecision(name, 'substitutive', group, heavy, 'P-63.1.5/P-41/P-44/P-45', (),
                          component_roles=(('carbon-parent', frozenset(carbon), ()),
                                           ('principal-chalcogen', group, ())))


def chalcogenol_decision(mol, Chem):
    """Name neutral C-bound chalcogenols on one carbon chain or carbon ring.

    All non-parent atoms are identical terminal X-H groups or halogens. This
    proves the characteristic-group class and leaves more senior groups,
    isotopes, stereochemistry and competing complex parents to other routes.
    """
    if (mol.GetNumHeavyAtoms() > 64 or mol.GetStereoGroups()
            or len(Chem.GetMolFrags(mol)) != 1
            or any(a.GetAtomicNum() not in {6, 9, 16, 17, 34, 35, 52, 53}
                   or a.GetFormalCharge() or a.GetIsotope() or a.GetNumRadicalElectrons()
                   or int(a.GetChiralTag()) or a.HasQuery() for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) or b.HasQuery() for b in mol.GetBonds())):
        return None
    carbon = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6}
    groups = [a for a in mol.GetAtoms() if a.GetAtomicNum() in {16, 34, 52}]
    if (not carbon or not groups or len({a.GetAtomicNum() for a in groups}) != 1
            or any(a.IsInRing() or a.GetDegree() != 1 or a.GetTotalValence() != 2
                   or a.GetTotalNumHs(includeNeighbors=True) != 1
                   or a.GetNeighbors()[0].GetIdx() not in carbon
                   or a.GetBonds()[0].GetBondTypeAsDouble() != 1 for a in groups)):
        return None
    halogens = {9: 'fluoro', 17: 'chloro', 35: 'bromo', 53: 'iodo'}
    branches = [a for a in mol.GetAtoms() if a.GetAtomicNum() in halogens]
    if any(a.GetDegree() != 1 or a.GetNeighbors()[0].GetIdx() not in carbon
           or a.GetBonds()[0].GetBondTypeAsDouble() != 1 for a in branches):
        return None
    neighbors = {i: [n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in carbon]
                 for i in carbon}
    rings = mol.GetRingInfo().AtomRings()
    cyclic = bool(rings)
    if cyclic:
        if len(rings) != 1 or frozenset(rings[0]) != carbon or not 3 <= len(carbon) <= 10:
            return None
        if any(len(neighbors[i]) != 2 for i in carbon):
            return None
        aromatic = all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in carbon)
        if aromatic and len(carbon) != 6:
            return None
        if not aromatic and any(mol.GetBondBetweenAtoms(i, j).GetBondTypeAsDouble() != 1
                                for i in carbon for j in neighbors[i]):
            return None
        starts = carbon
    else:
        if any(len(row) > 2 for row in neighbors.values()):
            return _branched_carbon_selenol_tellurol(mol, Chem, carbon, groups, branches)
        starts = {i for i in carbon if len(neighbors[i]) <= 1}
        if len(carbon) > 1 and len(starts) != 2:
            return None
        aromatic = False
    from .branch_rules import parent_chain_name
    from .formatting import (attach_suffix, complete_parent_name, format_prefixes,
                             join_prefix_parent, multiplied_suffix, prefix_alphabetization_key)
    from .constants import ROOTS

    suffix = {16: 'thiol', 34: 'selenol', 52: 'tellurol'}[groups[0].GetAtomicNum()]
    suffix = multiplied_suffix(len(groups), suffix)
    if not suffix:
        return None
    rows = []
    for start in starts:
        following = neighbors[start] if cyclic else [neighbors[start][0] if neighbors[start] else None]
        for first in following:
            seq, previous, current = [start], start, first
            while current is not None and current != start and len(seq) <= len(carbon):
                seq.append(current)
                nexts = [i for i in neighbors[current] if i != previous]
                if len(nexts) > 1:
                    break
                previous, current = current, nexts[0] if nexts else None
            if len(seq) != len(carbon) or frozenset(seq) != carbon:
                continue
            if cyclic and current != start or not cyclic and current is not None:
                continue
            mapping = {idx: str(i) for i, idx in enumerate(seq, 1)}
            suffix_locs = tuple(sorted(int(mapping[a.GetNeighbors()[0].GetIdx()]) for a in groups))
            prefixes = [(mapping[a.GetNeighbors()[0].GetIdx()], halogens[a.GetAtomicNum()]) for a in branches]
            if cyclic:
                root = ROOTS.get(len(seq))
                if not root:
                    continue
                parent = 'benzene' if aromatic else 'cyclo' + root + 'ane'
            else:
                parent = complete_parent_name(parent_chain_name(mol, seq))
            if not parent:
                continue
            # P-14.3.4.2/.5/.6: a mononuclear parent needs no locant;
            # complete uniform halogen substitution also omits locants when
            # the single suffix site is fixed by a two-atom chain or ring.
            uniform = (bool(branches) and len(groups) == 1 and (cyclic or len(seq) <= 2)
                       and len({a.GetAtomicNum() for a in branches}) == 1
                       and all(mol.GetAtomWithIdx(i).GetTotalNumHs(includeNeighbors=True) == 0 for i in carbon))
            omit = len(seq) == 1 or (len(groups) == 1 and (not prefixes or uniform)
                                     and (cyclic or len(seq) <= 2))
            ending = attach_suffix(parent, suffix, () if omit else suffix_locs)
            name = (join_prefix_parent(format_prefixes(prefixes, omit_locants=len(seq) == 1 or uniform), ending)
                    if prefixes else ending)
            unsaturation = tuple(i for i, (a, b) in enumerate(zip(seq, seq[1:]), 1)
                                 if mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble() > 1)
            order = sorted({text for _loc, text in prefixes}, key=prefix_alphabetization_key)
            score = (suffix_locs, unsaturation, tuple(sorted(int(loc) for loc, _text in prefixes)),
                     tuple(tuple(sorted(int(loc) for loc, value in prefixes if value == text)) for text in order))
            rows.append((score, name, mapping))
    if not rows:
        return None
    best = min(row[0] for row in rows)
    winners = [row for row in rows if row[0] == best]
    if len({row[1] for row in winners}) != 1:
        return None
    _score, name, mapping = winners[0]
    suffix_atoms = frozenset(a.GetIdx() for a in groups)
    roles = [('carbon-parent', frozenset(carbon), tuple(mapping.items())),
             ('principal-chalcogens', suffix_atoms, ())]
    roles.extend(('halogen', frozenset((a.GetIdx(),)), ()) for a in branches)
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms())
    return MethodDecision(name, 'substitutive', frozenset(suffix_atoms), heavy,
                          'P-63.1.5/P-41/P-44/P-45', tuple(mapping.items()),
                          component_roles=tuple(roles))
