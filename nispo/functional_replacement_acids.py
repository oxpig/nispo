"""Source-bounded carbonic, carbamic and carboxylic replacement acids."""

from collections import Counter

from .molecule_cache import exact_graph_computation
from .molecule_iteration import iter_atoms, iter_bonds
from .naming_methods import MethodDecision


_ELEMENTS = {8: 'O', 16: 'S', 34: 'Se', 52: 'Te'}
_INFICES = {16: 'thio', 34: 'seleno', 52: 'telluro'}
_MULT = {1: '', 2: 'di', 3: 'tri', 4: 'tetra'}


def _plain(mol, Chem):
    # Several independent functional-class writers require this same native
    # eligibility check. Annotation or stereo edits invalidate its result.
    return exact_graph_computation(mol, Chem, 'functional_plain_graph', (id(Chem),),
                                   lambda: _plain_uncached(mol, Chem))


def _plain_uncached(mol, Chem):
    return (mol.GetNumHeavyAtoms() <= 64 and len(Chem.GetMolFrags(mol)) == 1
            and not mol.GetStereoGroups()
            and all(not (a.GetFormalCharge() or a.GetIsotope() or a.GetNumRadicalElectrons()
                         or int(a.GetChiralTag()) or a.HasQuery()) for a in iter_atoms(mol))
            and all(not (int(b.GetStereo()) or int(b.GetBondDir()) or b.HasQuery()) for b in iter_bonds(mol)))


def _out(mol, name, center, rule, method='substitutive'):
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    return MethodDecision(name, method, frozenset((center,)), heavy, rule, (),
                          component_roles=(('source-acid', heavy, ()),))


def _organyl_units(mol, Chem, host, roots, seen):
    """Own disjoint carbon branches and bounded amino/hydroxy variants."""
    from .phosphorus_ester_methods import _carbon_component
    from .component_mapping import mapped_branch
    from .prefix_sequences import PrefixUnit, mononuclear_prefix_sequence

    units = []
    for root in roots:
        if mol.GetBondBetweenAtoms(host, root).GetBondTypeAsDouble() != 1:
            return None
        group = _carbon_component(mol, root, host)
        if group is None:
            group = _hydroxy_amino_organyl(mol, host, root, seen)
        if group is None or group & seen:
            return None
        text, owned = mapped_branch(mol, Chem, root, host)
        if not text or frozenset(owned) != group:
            return None
        path = all(sum(n.GetIdx() in group for n in mol.GetAtomWithIdx(i).GetNeighbors()) <= 2 for i in group)
        carbon_only = all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in group)
        units.append(PrefixUnit(text, 'simple' if carbon_only and
                                (text in {'phenyl', 'tert-butyl'} or path) else 'compound'))
        seen.update(group)
    seq = mononuclear_prefix_sequence(units) if units else None
    if units and seq is None:
        return None
    return seq.text if seq else ''


def _hydroxy_amino_organyl(mol, host, root, seen):
    """Prove branches containing only subordinate amino/hydroxy decoration."""
    group, pending = set(), [root]
    while pending:
        idx = pending.pop()
        if idx in group:
            continue
        if idx in seen:
            return None
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() not in {6, 7, 8, 9, 17, 35, 53}:
            return None
        if atom.GetAtomicNum() == 8 and (atom.GetDegree() != 1 or atom.GetTotalNumHs() != 1):
            return None
        if atom.GetAtomicNum() == 7 and (atom.GetDegree() != 1 or atom.GetTotalNumHs() != 2):
            return None
        if atom.GetAtomicNum() in {9, 17, 35, 53} and atom.GetDegree() != 1:
            return None
        group.add(idx)
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            if other.GetIdx() == host:
                continue
            if other.GetIdx() in seen:
                return None
            if (atom.GetAtomicNum() != 6 or other.GetAtomicNum() != 6) and bond.GetBondTypeAsDouble() != 1:
                return None
            pending.append(other.GetIdx())
    return frozenset(group)


def _center(mol):
    """Yield possible neutral C(=X)(single)(single) centers."""
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6 or atom.IsInRing() or atom.GetDegree() != 3:
            continue
        double = [n for n in atom.GetNeighbors() if n.GetAtomicNum() in _ELEMENTS and n.GetDegree() == 1
                  and mol.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 2]
        singles = [n for n in atom.GetNeighbors()
                   if mol.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 1]
        if len(double) == 1 and len(singles) == 2:
            yield atom, double[0], singles


def _acid_ligand(mol, atom):
    """Return acid H ligand kind, terminal element, and its owned atoms."""
    if atom.GetAtomicNum() not in _ELEMENTS or atom.IsInRing():
        return None
    if atom.GetDegree() == 1 and atom.GetTotalNumHs(includeNeighbors=True) == 1:
        return ('ordinary', atom.GetAtomicNum(), frozenset((atom.GetIdx(),)))
    if atom.GetDegree() != 2 or atom.GetTotalNumHs(includeNeighbors=True):
        return None
    tails = [n for n in atom.GetNeighbors() if n.GetAtomicNum() in _ELEMENTS and n.GetDegree() == 1
             and n.GetTotalNumHs(includeNeighbors=True) == 1
             and mol.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 1]
    if len(tails) != 1:
        return None
    if atom.GetAtomicNum() == tails[0].GetAtomicNum() == 8:
        kind = 'peroxo'
    else:
        pair = {atom.GetAtomicNum(), tails[0].GetAtomicNum()}
        kind = {frozenset((8, 16)): 'thioperoxo',
                frozenset((8, 34)): 'selenoperoxo',
                frozenset((8, 52)): 'telluroperoxo'}.get(frozenset(pair))
        if kind is None:
            return None
    return (kind, (atom.GetAtomicNum(), tails[0].GetAtomicNum()),
            frozenset((atom.GetIdx(), tails[0].GetIdx())))


def carbonic_replacement_acid_decision(mol, Chem):
    """P-65.2.1.2 carbonic/carbamic acids with wholly mapped ligands."""
    if not _plain(mol, Chem):
        return None
    choices = []
    for carbon, double, singles in _center(mol):
        n = [a for a in singles if a.GetAtomicNum() == 7]
        if len(n) > 1:
            continue
        carbamic = bool(n)
        acid_atoms = [a for a in singles if a not in n]
        if len(acid_atoms) != (1 if carbamic else 2):
            continue
        units = [_acid_ligand(mol, a) for a in acid_atoms]
        if any(unit is None for unit in units):
            continue
        seen = {carbon.GetIdx(), double.GetIdx()}
        for _kind, _element, owned in units:
            seen.update(owned)
        prefix = ''
        if carbamic:
            nitrogen = n[0]
            if nitrogen.IsInRing() or nitrogen.GetTotalValence() != 3:
                continue
            seen.add(nitrogen.GetIdx())
            roots = [a.GetIdx() for a in nitrogen.GetNeighbors() if a.GetIdx() != carbon.GetIdx()]
            if len(roots) > 2 or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in roots):
                continue
            prefix = _organyl_units(mol, Chem, nitrogen.GetIdx(), roots, seen)
            if prefix is None:
                continue
        if seen != {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}:
            continue
        special = [unit for unit in units if unit[0] != 'ordinary']
        if special:
            # Direct P-65.2.1.2 source patterns: peroxide and its mono-thio
            # analogues, with no additional chalcogen replacement at =O.
            if double.GetAtomicNum() != 8 or any(unit[0] not in {special[0][0], 'ordinary'} for unit in units):
                continue
            if any(unit[0] == 'ordinary' and unit[1] != 8 for unit in units):
                continue
            kind = special[0][0]
            count = len(special)
            if kind == 'peroxo':
                stem = _MULT.get(count, '') + 'peroxoic'
                locator = ''
            else:
                stem = '(' + kind + 'ic)' if count == 1 else 'bis(' + kind + 'ic)'
                locator = ' ' + ','.join(sorted(_ELEMENTS[a] + _ELEMENTS[b] for _, (a, b), _ in special)) + '-'
            name = prefix + ('carbamo' if carbamic else 'carbono') + stem + (locator or ' ') + 'acid'
        else:
            numbers = [double.GetAtomicNum(), *(unit[1] for unit in units)]
            counts = Counter(number for number in numbers if number != 8)
            if not counts:
                if carbamic:
                    name = prefix + 'carbamic acid'
                else:
                    name = 'carbonic acid'
            else:
                parts = []
                for element in sorted(counts, key=lambda number: _INFICES[number]):
                    multiplier = _MULT.get(counts[element])
                    if multiplier is None:
                        break
                    parts.append(multiplier + _INFICES[element])
                else:
                    stem = ''.join(parts) + 'ic'
                    acid_elements = [unit[1] for unit in units]
                    if len(set(numbers)) == 1:
                        locator = ''
                    elif double.GetAtomicNum() == 8 and 8 in acid_elements:
                        locator = ','.join(_ELEMENTS[x] for x in acid_elements if x != 8)
                    else:
                        locator = ','.join(sorted((_ELEMENTS[x] for x in acid_elements),
                                                  key=lambda x: ('O', 'S', 'Se', 'Te').index(x)))
                    name = prefix + ('carbamo' if carbamic else 'carbono') + stem
                    name += ' ' + (locator + '-' if locator else '') + 'acid'
                if not parts or len(parts) != len(counts):
                    continue
        rule = 'P-65.2.1.1/P-41' if carbamic and name.endswith('carbamic acid') else 'P-65.2.1.2/P-41'
        method = 'retained-parent' if rule.startswith('P-65.2.1.1') else 'functional-replacement'
        choices.append(_out(mol, name, carbon.GetIdx(), rule, method))
    return choices[0] if len(choices) == 1 else None


def carboxylic_chalcogen_acid_decision(mol, Chem):
    """P-65.1.5.2 on carbon-only chains and simple carbon rings."""
    if not _plain(mol, Chem) or any(a.GetAtomicNum() not in {6, 8, 16, 34, 52} for a in mol.GetAtoms()):
        return None
    # One C(=X)-YH center; carbonic/carbamic acids are handled separately.
    centers = []
    for carbon in mol.GetAtoms():
        if carbon.GetAtomicNum() != 6 or carbon.IsInRing():
            continue
        double = [a for a in carbon.GetNeighbors() if a.GetAtomicNum() in _ELEMENTS and a.GetDegree() == 1
                  and mol.GetBondBetweenAtoms(carbon.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() == 2]
        single = [a for a in carbon.GetNeighbors() if a.GetAtomicNum() in _ELEMENTS and a.GetDegree() == 1
                  and a.GetTotalNumHs(includeNeighbors=True) == 1
                  and mol.GetBondBetweenAtoms(carbon.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() == 1]
        if len(double) == len(single) == 1 and (double[0].GetAtomicNum() != 8 or single[0].GetAtomicNum() != 8):
            centers.append((carbon, double[0], single[0]))
    if len(centers) != 1:
        return None
    carbon, double, single = centers[0]
    carbon_ids = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6}
    if carbon_ids | {double.GetIdx(), single.GetIdx()} != {a.GetIdx() for a in mol.GetAtoms()}:
        return None
    adjacent = [a for a in carbon.GetNeighbors() if a.GetAtomicNum() == 6]
    if len(adjacent) > 1 or carbon.GetDegree() != 2 + len(adjacent):
        return None
    rings = mol.GetRingInfo().AtomRings()
    if rings:
        ring = frozenset(rings[0]) if len(rings) == 1 else frozenset()
        if (not 3 <= len(ring) <= 10 or carbon_ids != ring | {carbon.GetIdx()}
                or len(adjacent) != 1 or adjacent[0].GetIdx() not in ring):
            return None
        aromatic = all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
        if aromatic and len(ring) == 6:
            parent = 'benzenecarbo'
        elif (not aromatic and all(mol.GetBondBetweenAtoms(i, j).GetBondTypeAsDouble() == 1
                                   for i in ring for j in ring if mol.GetBondBetweenAtoms(i, j))):
            from .constants import ROOTS
            parent = 'cyclo' + ROOTS[len(ring)] + 'anecarbo'
        else:
            return None
    else:
        from .constants import ROOTS
        if len(carbon_ids) not in ROOTS:
            return None
        simple_path = (all(sum(n.GetIdx() in carbon_ids for n in mol.GetAtomWithIdx(i).GetNeighbors()) <= 2
                           for i in carbon_ids)
                       and all(b.GetBondTypeAsDouble() == 1 for b in mol.GetBonds()
                               if b.GetBeginAtomIdx() in carbon_ids and b.GetEndAtomIdx() in carbon_ids))
        if simple_path:
            parent = ROOTS[len(carbon_ids)] + 'ane'
        else:
            # Both chalcogens are terminal and the rest is a carbon-only
            # parent, so the corresponding ordinary acid has the same
            # parent selection and numbering. Request its systematic stem,
            # never retaining acetic/formic acid for the replacement name.
            marker = Chem.RWMol(mol)
            for atom, hydrogens in ((double, 0), (single, 1)):
                replacement = marker.GetAtomWithIdx(atom.GetIdx())
                replacement.SetAtomicNum(8)
                replacement.SetNumExplicitHs(hydrogens)
                replacement.SetNoImplicit(True)
            marker = marker.GetMol()
            try:
                Chem.SanitizeMol(marker)
            except (RuntimeError, ValueError):
                return None
            from .api import mol_to_iupac
            acid_name = mol_to_iupac(marker)
            if not acid_name.endswith('oic acid'):
                return None
            parent = acid_name[:-len('oic acid')] + 'e'
    counts = Counter(x for x in (double.GetAtomicNum(), single.GetAtomicNum()) if x != 8)
    infix = ''.join(_MULT[counts[x]] + _INFICES[x] for x in sorted(counts, key=lambda value: _INFICES[value]))
    locator = '' if double.GetAtomicNum() == single.GetAtomicNum() else _ELEMENTS[single.GetAtomicNum()] + '-'
    # In an acyclic parent, enclosing the doubly replaced *single* suffix
    # distinguishes it from a multiplicative acid suffix (P-13.2.2.2 and
    # P-16.3.5). The ring-attached carbodithioic suffix has no ambiguity.
    if not rings and len(counts) == 1 and next(iter(counts.values())) == 2:
        name = parent + '(' + infix + 'ic) acid'
    else:
        name = parent + infix + 'ic ' + locator + 'acid'
    return _out(mol, name, carbon.GetIdx(), 'P-65.1.5.2/P-41')
