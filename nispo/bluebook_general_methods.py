"""Graph-scoped constructions for retained parents and functional classes.

Each writer proves complete atom ownership and declines unhandled competing
functions or annotations. Published-name expectations live in development.
"""
from collections import Counter

from .bluebook_azoxy_methods import _decision, _plain
from .component_mapping import mapped_branch
from .constants import MULT
from .formatting import format_prefixes


def _bond_order(mol, first, second):
    bond = mol.GetBondBetweenAtoms(first, second)
    return bond.GetBondTypeAsDouble() if bond else 0


def _sanitize(editable, Chem):
    result = editable.GetMol()
    try:
        Chem.SanitizeMol(result)
    except (ValueError, RuntimeError):
        return None
    return result


def retained_cyanic_or_formonitrile(mol, Chem):
    """The two unsubstituted HCN and HOCN parents, with nuclides."""
    if (len(Chem.GetMolFrags(mol)) != 1 or mol.GetStereoGroups()
            or any(a.GetFormalCharge() or a.GetNumRadicalElectrons() or a.HasQuery()
                   or int(a.GetChiralTag()) for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) or b.HasQuery() for b in mol.GetBonds())):
        return None
    atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    hydrogens = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 1]
    if len(hydrogens) > 1 or any(a.GetDegree() != 1 or a.GetIsotope() not in {2, 3} for a in hydrogens):
        return None
    composition = Counter(a.GetAtomicNum() for a in atoms)
    if composition not in (Counter({6: 1, 7: 1}), Counter({6: 1, 7: 1, 8: 1})):
        return None
    carbon = next(a for a in atoms if a.GetAtomicNum() == 6)
    nitrogen = next(a for a in atoms if a.GetAtomicNum() == 7)
    if nitrogen.GetDegree() != 1 or _bond_order(mol, carbon.GetIdx(), nitrogen.GetIdx()) != 3:
        return None
    if len(atoms) == 2:
        if carbon.GetTotalNumHs(includeNeighbors=True) != 1:
            return None
        h_host = carbon.GetIdx()
        base, rule = 'formonitrile', 'P-66.5.1.2.1/P-65.1.8'
    else:
        oxygen = next(a for a in atoms if a.GetAtomicNum() == 8)
        if (carbon.GetTotalNumHs(includeNeighbors=True) or oxygen.GetDegree() != 1 + len(hydrogens)
                or oxygen.GetTotalNumHs(includeNeighbors=True) != 1
                or _bond_order(mol, carbon.GetIdx(), oxygen.GetIdx()) != 1):
            return None
        h_host = oxygen.GetIdx()
        base, rule = 'cyanic acid', 'P-65.2.2'
    if any(a.GetNeighbors()[0].GetIdx() != h_host for a in hydrogens):
        return None
    # These parents contain one atom of each element, so element locants and
    # multiplicities are unambiguous (P-82.2.1).
    isotopes = sorted((a.GetSymbol(), a.GetIsotope()) for a in atoms + hydrogens if a.GetIsotope())
    if isotopes:
        base = '(' + ','.join(str(mass) + symbol for symbol, mass in isotopes) + ')' + base
        rule += '/P-82.2.1'
    return _decision(mol, base, (carbon.GetIdx(),), rule, method='retained-parent')


def nitrile_chalcogenide_decision(mol, Chem):
    """One terminal R-C#N(+)-E(-), deriving the named neutral nitrile.

    The unmodified nitrile must itself be selected as the principal class.
    Its ordinary, subordinate substituents need not be hydrocarbons.
    """
    if (not _plain(mol, Chem, charged=True) or mol.GetNumAtoms() != mol.GetNumHeavyAtoms()
            or any(a.GetAtomicNum() not in {6, 7, 8, 9, 16, 17, 34, 35, 52, 53} for a in mol.GetAtoms())):
        return None
    sites = []
    for n in mol.GetAtoms():
        if n.GetAtomicNum() != 7 or n.GetFormalCharge() != 1 or n.GetDegree() != 2:
            continue
        carbon = [a for a in n.GetNeighbors() if a.GetAtomicNum() == 6
                  and _bond_order(mol, n.GetIdx(), a.GetIdx()) == 3]
        ends = [a for a in n.GetNeighbors() if a.GetAtomicNum() in {8, 16, 34, 52}
                and a.GetFormalCharge() == -1 and a.GetDegree() == 1
                and _bond_order(mol, n.GetIdx(), a.GetIdx()) == 1]
        if len(carbon) == len(ends) == 1:
            sites.append((carbon[0], n, ends[0]))
    if len(sites) != 1:
        return None
    carbon, nitrogen, end = sites[0]
    if any(a.GetFormalCharge() and a.GetIdx() not in {nitrogen.GetIdx(), end.GetIdx()} for a in mol.GetAtoms()):
        return None
    copy = Chem.RWMol(mol)
    copy.GetAtomWithIdx(nitrogen.GetIdx()).SetFormalCharge(0)
    copy.RemoveAtom(end.GetIdx())
    parent = _sanitize(copy, Chem)
    if parent is None:
        return None
    # A dinitrile name alone cannot identify which of two inequivalent
    # nitrile nitrogens bears the oxide. Admit it only if the parent graph
    # proves the sites equivalent; otherwise leave the construction open.
    nitrile_carbons = [a.GetNeighbors()[0].GetIdx() for a in parent.GetAtoms()
                       if a.GetAtomicNum() == 7 and a.GetDegree() == 1
                       and a.GetNeighbors()[0].GetAtomicNum() == 6
                       and _bond_order(parent, a.GetIdx(), a.GetNeighbors()[0].GetIdx()) == 3]
    selected_carbon = carbon.GetIdx() - int(end.GetIdx() < carbon.GetIdx())
    if not nitrile_carbons or selected_carbon not in nitrile_carbons:
        return None
    if len(nitrile_carbons) > 1:
        ranks = Chem.CanonicalRankAtoms(parent, breakTies=False)
        if any(ranks[i] != ranks[selected_carbon] for i in nitrile_carbons):
            return None
    from .api import mol_to_iupac
    name = mol_to_iupac(parent)
    if not name.endswith('nitrile'):
        return None
    term = {8: 'oxide', 16: 'sulfide', 34: 'selenide', 52: 'telluride'}[end.GetAtomicNum()]
    return _decision(mol, name + ' ' + term, (carbon.GetIdx(),), 'P-66.5.4.1/P-74.2.2.2.1.2',
                     method='functional-class')


def _typed_mononuclear_units(mol, Chem, center, *, allow_pseudohalide=False):
    from .prefix_sequences import PrefixUnit, mononuclear_prefix_sequence
    halogens = {9: 'fluoro', 17: 'chloro', 35: 'bromo', 53: 'iodo'}
    seen = {center.GetIdx()}
    units = []
    for atom in center.GetNeighbors():
        if _bond_order(mol, center.GetIdx(), atom.GetIdx()) != 1:
            return None
        if atom.GetAtomicNum() in halogens and atom.GetDegree() == 1:
            text, owned, nature = halogens[atom.GetAtomicNum()], frozenset({atom.GetIdx()}), 'simple'
        elif allow_pseudohalide and atom.GetAtomicNum() in {7, 8, 16}:
            other = [a for a in atom.GetNeighbors() if a.GetIdx() != center.GetIdx()]
            if len(other) != 1 or other[0].GetAtomicNum() != 6 or other[0].GetDegree() != 2:
                return None
            carbon = other[0]
            if atom.GetAtomicNum() == 7 and _bond_order(mol, atom.GetIdx(), carbon.GetIdx()) == 2:
                ends = [a for a in carbon.GetNeighbors() if a.GetIdx() != atom.GetIdx()
                        and a.GetAtomicNum() in {8, 16, 34, 52} and a.GetDegree() == 1
                        and _bond_order(mol, carbon.GetIdx(), a.GetIdx()) == 2]
                if len(ends) != 1:
                    return None
                text = {8: 'isocyanato', 16: 'isothiocyanato',
                        34: 'isoselenocyanato', 52: 'isotellurocyanato'}[ends[0].GetAtomicNum()]
            elif atom.GetAtomicNum() in {8, 16} and _bond_order(mol, atom.GetIdx(), carbon.GetIdx()) == 1:
                ends = [a for a in carbon.GetNeighbors() if a.GetAtomicNum() == 7 and a.GetDegree() == 1
                        and _bond_order(mol, carbon.GetIdx(), a.GetIdx()) == 3]
                if len(ends) != 1:
                    return None
                text = {8: 'cyanato', 16: 'thiocyanato'}[atom.GetAtomicNum()]
            else:
                return None
            owned, nature = frozenset({atom.GetIdx(), carbon.GetIdx(), ends[0].GetIdx()}), 'simple'
        elif atom.GetAtomicNum() == 6:
            nitrogens = [a for a in atom.GetNeighbors() if a.GetAtomicNum() == 7 and a.GetDegree() == 1
                         and _bond_order(mol, atom.GetIdx(), a.GetIdx()) == 3]
            if allow_pseudohalide and atom.GetDegree() == 2 and len(nitrogens) == 1:
                text, owned, nature = 'cyano', frozenset({atom.GetIdx(), nitrogens[0].GetIdx()}), 'simple'
                if seen & owned:
                    return None
                seen.update(owned)
                units.append(PrefixUnit(text, nature))
                continue
            text, owned = mapped_branch(mol, Chem, atom.GetIdx(), center.GetIdx())
            if not text or not owned or any(mol.GetAtomWithIdx(i).GetAtomicNum() not in {6, 9, 17, 35, 53}
                                            for i in owned):
                return None
            all_carbon = all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in owned)
            nature = 'simple' if all_carbon and (text in {'phenyl', 'benzyl'} or text.isalpha()) else 'compound'
        else:
            return None
        if seen & owned:
            return None
        seen.update(owned)
        units.append(PrefixUnit(text, nature))
    if seen != set(range(mol.GetNumAtoms())):
        return None
    return mononuclear_prefix_sequence(units) if units else None


def boron_silicon_halide_decision(mol, Chem):
    """Halo and hydrocarbon substituents on one B/Si hydride parent."""
    if (not _plain(mol, Chem) or mol.GetNumAtoms() != mol.GetNumHeavyAtoms()
            or any(a.GetAtomicNum() not in {5, 6, 7, 8, 9, 14, 16, 17, 34, 35, 52, 53}
                   for a in mol.GetAtoms())):
        return None
    centers = [a for a in mol.GetAtoms() if a.GetAtomicNum() in {5, 14}]
    if len(centers) != 1 or not any(a.GetAtomicNum() in {7, 9, 17, 35, 53} for a in mol.GetAtoms()):
        return None
    center = centers[0]
    if center.IsInRing() or center.GetTotalValence() != {5: 3, 14: 4}[center.GetAtomicNum()]:
        return None
    sequence = _typed_mononuclear_units(mol, Chem, center, allow_pseudohalide=True)
    if sequence is None:
        return None
    name = sequence.text + {5: 'borane', 14: 'silane'}[center.GetAtomicNum()]
    return _decision(mol, name, (center.GetIdx(),), 'P-67.1.2.5.1/P-16.5.1.3.1')


def hydride_addition_anion_decision(mol, Chem):
    """Mononuclear -uide anions with hydrocarbon/halogen substituents."""
    if (not _plain(mol, Chem, charged=True) or mol.GetNumAtoms() != mol.GetNumHeavyAtoms()
            or sum(a.GetFormalCharge() for a in mol.GetAtoms()) != -1):
        return None
    charged = [a for a in mol.GetAtoms() if a.GetFormalCharge()]
    if len(charged) != 1 or charged[0].GetFormalCharge() != -1:
        return None
    center = charged[0]
    hydrides = {5: ('boran', 4), 14: ('silan', 5), 15: ('phosphan', 4),
                16: ('sulfan', 3), 33: ('arsan', 4), 34: ('selan', 3),
                51: ('stiban', 4), 52: ('tellan', 3), 53: ('iodan', 2)}
    info = hydrides.get(center.GetAtomicNum())
    if info is None or center.IsInRing() or center.GetDegree() + center.GetTotalNumHs() != info[1]:
        return None
    sequence = _typed_mononuclear_units(mol, Chem, center)
    if sequence is None:
        return None
    return _decision(mol, sequence.text + info[0] + 'uide', (center.GetIdx(),), 'P-72.3',
                     method='substitutive')


def n_chalcogen_amine_decision(mol, Chem):
    """An N-chalcogen substituent on a carbon-derived amine."""
    if (not _plain(mol, Chem) or mol.GetNumAtoms() != mol.GetNumHeavyAtoms()
            or any(a.GetAtomicNum() not in {6, 7, 16, 34, 52} for a in mol.GetAtoms())):
        return None
    nitrogens = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    chalcogens = [a for a in mol.GetAtoms() if a.GetAtomicNum() in {16, 34, 52}]
    if len(nitrogens) != 1 or len(chalcogens) != 1:
        return None
    n, x = nitrogens[0], chalcogens[0]
    if (n.IsInRing() or n.GetDegree() not in {2, 3}
            or n.GetDegree() + n.GetTotalNumHs() != 3
            or x.GetDegree() != 1 or x.GetTotalNumHs() != 1
            or _bond_order(mol, n.GetIdx(), x.GetIdx()) != 1):
        return None
    roots = [a for a in n.GetNeighbors() if a.GetIdx() != x.GetIdx()]
    if len(roots) not in {1, 2} or any(root.GetAtomicNum() != 6 for root in roots):
        return None
    copy = Chem.RWMol(mol)
    copy.RemoveAtom(x.GetIdx())
    parent = _sanitize(copy, Chem)
    if parent is None:
        return None
    from .api import mol_to_iupac
    name = mol_to_iupac(parent)
    if not name.endswith(('amine', 'aniline')):
        return None
    prefix = {16: 'sulfanyl', 34: 'selanyl', 52: 'tellanyl'}[x.GetAtomicNum()]
    if len(roots) == 1:
        if name.startswith('N-'):
            return None
        result = 'N-' + prefix + name
    else:
        choices = []
        for root in roots:
            text, atoms = mapped_branch(mol, Chem, root.GetIdx(), n.GetIdx())
            if not text or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in atoms):
                continue
            leading = format_prefixes([('N', text)], hierarchical_enclosures=True)
            if name.startswith(leading):
                choices.append(format_prefixes([('N', text), ('N', prefix)],
                                                hierarchical_enclosures=True) + name[len(leading):])
        if len(set(choices)) != 1:
            return None
        result = choices[0]
    return _decision(mol, result, (n.GetIdx(),), 'P-68.3.1.1.1.6/P-62.2')


def phosphane_carboxylic_acid_decision(mol, Chem):
    """One carboxylic-acid group directly bonded to P, As or Sb."""
    if (not _plain(mol, Chem) or mol.GetNumAtoms() != mol.GetNumHeavyAtoms()
            or any(a.GetAtomicNum() not in {6, 8, 15, 33, 51} for a in mol.GetAtoms())):
        return None
    elements = [a for a in mol.GetAtoms() if a.GetAtomicNum() in {15, 33, 51}]
    if len(elements) != 1:
        return None
    atom = elements[0]
    if atom.IsInRing() or atom.GetTotalValence() != 3:
        return None
    sites = []
    for c in atom.GetNeighbors():
        if c.GetAtomicNum() != 6 or c.GetDegree() != 3 or _bond_order(mol, atom.GetIdx(), c.GetIdx()) != 1:
            continue
        oxo = [a for a in c.GetNeighbors() if a.GetAtomicNum() == 8 and a.GetDegree() == 1
               and _bond_order(mol, c.GetIdx(), a.GetIdx()) == 2]
        oh = [a for a in c.GetNeighbors() if a.GetAtomicNum() == 8 and a.GetDegree() == 1
              and a.GetTotalNumHs() == 1 and _bond_order(mol, c.GetIdx(), a.GetIdx()) == 1]
        if len(oxo) == len(oh) == 1:
            sites.append((c, oxo[0], oh[0]))
    if len(sites) != 1 or sum(a.GetAtomicNum() == 8 for a in mol.GetAtoms()) != 2:
        return None
    core = sites[0]
    from .prefix_sequences import PrefixUnit, mononuclear_prefix_sequence
    seen = {atom.GetIdx(), *(a.GetIdx() for a in core)}
    units = []
    for neighbor in atom.GetNeighbors():
        if neighbor.GetIdx() == core[0].GetIdx():
            continue
        if neighbor.GetAtomicNum() != 6 or _bond_order(mol, atom.GetIdx(), neighbor.GetIdx()) != 1:
            return None
        text, owned = mapped_branch(mol, Chem, neighbor.GetIdx(), atom.GetIdx())
        if not text or owned & seen or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in owned):
            return None
        units.append(PrefixUnit(text, 'simple' if text.isalpha() else 'compound'))
        seen.update(owned)
    if seen != set(range(mol.GetNumAtoms())):
        return None
    sequence = mononuclear_prefix_sequence(units) if units else None
    if units and sequence is None:
        return None
    name = (sequence.text if sequence else '') + {15: 'phosphane', 33: 'arsane', 51: 'stibane'}[atom.GetAtomicNum()]
    return _decision(mol, name + 'carboxylic acid', (core[0].GetIdx(),), 'P-68.3.2.3.1/P-41')


def polyalene_parent_decision(mol, Chem):
    """Mancude unsubstituted identical ortho-fused hydrocarbon rings."""
    if (not _plain(mol, Chem) or mol.GetNumAtoms() != mol.GetNumHeavyAtoms()
            or any(a.GetAtomicNum() != 6 for a in mol.GetAtoms())):
        return None
    rings = tuple(frozenset(r) for r in mol.GetRingInfo().AtomRings())
    if len(rings) != 2 or len(rings[0]) != len(rings[1]) or not 5 <= len(rings[0]) <= 10:
        return None
    if rings[0] | rings[1] != frozenset(range(mol.GetNumAtoms())):
        return None
    shared = rings[0] & rings[1]
    if len(shared) != 2 or mol.GetBondBetweenAtoms(*shared) is None:
        return None
    from .fusion_components import _mancude
    if not _mancude(mol):
        return None
    size = len(rings[0])
    name = 'naphthalene' if size == 6 else MULT[size][:-1] + 'alene'
    return _decision(mol, name, (), 'P-25.1.2.3/P-25.1.1', method='ring-fusion')


def carbonic_imido_hydrazono_decision(mol, Chem):
    """Functional replacement of C=O in carbonic/carbamic acid parents."""
    if not _plain(mol, Chem) or mol.GetNumAtoms() != mol.GetNumHeavyAtoms():
        return None
    from .functional_replacement_acids import _acid_ligand
    choices = []
    for carbon in mol.GetAtoms():
        if carbon.GetAtomicNum() != 6 or carbon.GetDegree() != 3 or carbon.IsInRing():
            continue
        imines = [a for a in carbon.GetNeighbors() if a.GetAtomicNum() == 7
                  and _bond_order(mol, carbon.GetIdx(), a.GetIdx()) == 2]
        singles = [a for a in carbon.GetNeighbors() if _bond_order(mol, carbon.GetIdx(), a.GetIdx()) == 1]
        if len(imines) != 1 or len(singles) != 2:
            continue
        imine = imines[0]
        seen = {carbon.GetIdx(), imine.GetIdx()}
        other = [a for a in imine.GetNeighbors() if a.GetIdx() != carbon.GetIdx()]
        if imine.GetDegree() == 1 and imine.GetTotalNumHs() == 1:
            imido = 'imido'
        elif (len(other) == 1 and other[0].GetAtomicNum() == 7 and other[0].GetDegree() == 1
              and other[0].GetTotalNumHs() == 2 and imine.GetTotalNumHs() == 0
              and _bond_order(mol, imine.GetIdx(), other[0].GetIdx()) == 1):
            imido = 'hydrazono'
            seen.add(other[0].GetIdx())
        else:
            continue
        amines = [a for a in singles if a.GetAtomicNum() == 7]
        if len(amines) > 1:
            continue
        carbamic = bool(amines)
        groups = [a for a in singles if a not in amines]
        units = [_acid_ligand(mol, atom) for atom in groups]
        if len(groups) != (1 if carbamic else 2) or any(unit is None for unit in units):
            continue
        for _kind, _element, owned in units:
            seen.update(owned)
        prefixes = []
        if carbamic:
            amino = amines[0]
            if amino.IsInRing() or amino.GetTotalValence() != 3:
                continue
            seen.add(amino.GetIdx())
            for root in amino.GetNeighbors():
                if root.GetIdx() == carbon.GetIdx():
                    continue
                if root.GetAtomicNum() != 6:
                    break
                text, owned = mapped_branch(mol, Chem, root.GetIdx(), amino.GetIdx())
                if not text or owned & seen or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in owned):
                    break
                prefixes.append(('N', text))
                seen.update(owned)
            else:
                pass
            if len(prefixes) != amino.GetDegree() - 1:
                continue
        if seen != set(range(mol.GetNumAtoms())):
            continue
        special = [unit for unit in units if unit[0] != 'ordinary']
        if special:
            # A single mixed O/S peroxide arm has an explicit position locant.
            if len(special) != 1 or len(units) != 1 or special[0][0] != 'thioperoxo':
                continue
            pair = special[0][1]
            locator = ''.join({8: 'O', 16: 'S'}[x] for x in pair)
            stem = ('carbam' if carbamic else 'carbon') + imido + '(thioperoxoic) ' + locator + '-acid'
        else:
            counts = Counter(unit[1] for unit in units if unit[1] != 8)
            infices = {16: 'thio', 34: 'seleno', 52: 'telluro'}
            replacement = ''.join((MULT[count] if count > 1 else '') + infices[number]
                                  for number, count in sorted(counts.items(), key=lambda item: infices[item[0]]))
            if imido == 'imido':
                stem = ('carbamimid' if carbamic else 'carbonimid')
            else:
                stem = ('carbamohydrazon' if carbamic else 'carbonohydrazon')
            stem += ('o' + replacement if replacement else '') + 'ic acid'
        name = format_prefixes(prefixes, hierarchical_enclosures=True) + stem
        choices.append(_decision(mol, name, (carbon.GetIdx(),), 'P-65.2.1.3/P-41',
                                 method='functional-replacement'))
    return choices[0] if len(choices) == 1 else None


def amide_competing_acid_halide_decision(mol, Chem):
    """One acyl halide and a subordinate primary carboxamide."""
    if (not _plain(mol, Chem) or mol.GetNumAtoms() != mol.GetNumHeavyAtoms()
            or any(a.GetAtomicNum() not in {6, 7, 8, 9, 16, 17, 35, 53} for a in mol.GetAtoms())):
        return None
    nitrogens = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    halogens = [a for a in mol.GetAtoms() if a.GetAtomicNum() in {9, 17, 35, 53}]
    if len(nitrogens) != 1 or len(halogens) != 1:
        return None
    n, leaving = nitrogens[0], halogens[0]
    if n.GetDegree() != 1 or n.GetTotalNumHs() != 2 or leaving.GetDegree() != 1:
        return None
    amide = n.GetNeighbors()[0]
    if amide.GetAtomicNum() != 6 or not any(a.GetAtomicNum() == 8 and a.GetDegree() == 1
            and _bond_order(mol, amide.GetIdx(), a.GetIdx()) == 2 for a in amide.GetNeighbors()):
        return None
    center = leaving.GetNeighbors()[0]
    if center.GetIdx() == amide.GetIdx() or center.GetAtomicNum() not in {6, 16} or center.IsInRing():
        return None
    oxos = [a for a in center.GetNeighbors() if a.GetAtomicNum() == 8 and a.GetDegree() == 1
            and _bond_order(mol, center.GetIdx(), a.GetIdx()) == 2]
    if len(oxos) != (1 if center.GetAtomicNum() == 6 else 2):
        return None
    if len([a for a in center.GetNeighbors() if a.GetAtomicNum() == 6]) != 1:
        return None
    # No extra oxygen or sulfur characteristic group can compete for class.
    expected_o = len(oxos) + 1
    if (sum(a.GetAtomicNum() == 8 for a in mol.GetAtoms()) != expected_o
            or sum(a.GetAtomicNum() == 16 for a in mol.GetAtoms()) != (center.GetAtomicNum() == 16)):
        return None
    copy = Chem.RWMol(mol)
    replacement = copy.GetAtomWithIdx(leaving.GetIdx())
    replacement.SetAtomicNum(8)
    replacement.SetNoImplicit(False)
    replacement.SetNumExplicitHs(0)
    acid = _sanitize(copy, Chem)
    if acid is None:
        return None
    from .api import mol_to_iupac
    from .bluebook_lambda_amine_methods import _acid_to_acyl
    name = mol_to_iupac(acid)
    if center.GetAtomicNum() == 6:
        acyl = _acid_to_acyl(name)
    elif name.endswith('sulfonic acid'):
        acyl = name[:-len('ic acid')] + 'yl'
    else:
        acyl = ''
    if not acyl:
        return None
    halide = {9: 'fluoride', 17: 'chloride', 35: 'bromide', 53: 'iodide'}[leaving.GetAtomicNum()]
    return _decision(mol, acyl + ' ' + halide, (center.GetIdx(),), 'P-41/P-65.5.1', method='functional-class')


def _extended_semicarbazone_ylidene(mol, Chem, carbon, imine_n):
    """Map a saturated cycloalkylidene or halo-substituted carbon tree.

    The carbon parent is a longest path through the ylidene atom; path
    numberings are compared before choosing the spelling. More complicated
    ring systems and competing functions do not enter this construction.
    """
    from .constants import HALO, ROOTS
    from .formatting import prefix_alphabetization_key

    host = mol.GetAtomWithIdx(carbon)
    if host.IsInRing():
        rings = [frozenset(ring) for ring in mol.GetRingInfo().AtomRings() if carbon in ring]
        if len(rings) != 1 or len(rings[0]) not in ROOTS or len(rings[0]) < 3:
            return None
        ring = rings[0]
        for idx in ring:
            atom = mol.GetAtomWithIdx(idx)
            neighbors = {neighbor.GetIdx() for neighbor in atom.GetNeighbors()}
            if (atom.GetAtomicNum() != 6 or len(neighbors & ring) != 2
                    or neighbors - ring != ({imine_n} if idx == carbon else set())
                    or any(_bond_order(mol, idx, j) != 1 for j in neighbors & ring)):
                return None
        return 'cyclo' + ROOTS[len(ring)] + 'ylidene', ring

    allowed = {6, 9, 17, 35, 53}
    owned, stack = {carbon}, [carbon]
    while stack:
        idx = stack.pop()
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() not in allowed or atom.IsInRing():
            return None
        for neighbor in atom.GetNeighbors():
            j = neighbor.GetIdx()
            if j == imine_n and idx == carbon:
                continue
            if neighbor.GetAtomicNum() not in allowed or _bond_order(mol, idx, j) != 1:
                return None
            if j not in owned:
                owned.add(j)
                stack.append(j)
    carbons = {idx for idx in owned if mol.GetAtomWithIdx(idx).GetAtomicNum() == 6}
    if len(carbons) not in ROOTS:
        return None
    graph = {idx: {n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors() if n.GetIdx() in carbons}
             for idx in carbons}
    if sum(map(len, graph.values())) != 2 * (len(carbons) - 1):
        return None
    if any(mol.GetAtomWithIdx(idx).GetDegree() != 1 for idx in owned - carbons):
        return None
    ends = [idx for idx in carbons if len(graph[idx]) <= 1]
    if len(carbons) == 1:
        paths = [(carbon,)]
    else:
        paths = []
        for start in ends:
            for finish in ends:
                if start == finish:
                    continue
                queue = [(start, (start,))]
                while queue:
                    idx, path = queue.pop()
                    if idx == finish:
                        if carbon in path:
                            paths.append(path)
                        break
                    queue.extend((j, path + (j,)) for j in graph[idx] if j not in path)
    candidates = []
    for path in paths:
        path_set = set(path)
        seen = set(path)
        prefixes = []
        for locant, idx in enumerate(path, 1):
            for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
                j = neighbor.GetIdx()
                if j not in owned or j in path_set:
                    continue
                if neighbor.GetAtomicNum() == 6:
                    text, atoms = mapped_branch(mol, Chem, j, idx)
                else:
                    text, atoms = HALO.get(neighbor.GetSymbol(), ''), frozenset({j})
                if not text or not atoms or not atoms <= owned or atoms & seen:
                    break
                seen.update(atoms)
                prefixes.append((str(locant), text))
            else:
                continue
            break
        else:
            if seen != owned:
                continue
            yloc = path.index(carbon) + 1
            parent = ROOTS[len(path)] + ('ylidene' if yloc == 1 else f'an-{yloc}-ylidene')
            name = format_prefixes(prefixes, hierarchical_enclosures=True) + parent
            grouped = {}
            for locant, text in prefixes:
                grouped.setdefault(text, []).append(int(locant))
            prefix_key = tuple((prefix_alphabetization_key(text), tuple(sorted(locs)))
                               for text, locs in sorted(grouped.items(),
                                                        key=lambda item: prefix_alphabetization_key(item[0])))
            key = (-len(path), yloc, -len(prefixes),
                   tuple(sorted(int(loc) for loc, _ in prefixes)), prefix_key)
            candidates.append((key, name))
    if not candidates:
        return None
    best = min(key for key, _ in candidates)
    names = {name for key, name in candidates if key == best}
    return (names.pop(), frozenset(owned)) if len(names) == 1 else None


def semicarbazone_decision(mol, Chem):
    """Ylidenehydrazinecarboxamides and their chalcogen analogues."""
    if (not _plain(mol, Chem) or mol.GetNumAtoms() != mol.GetNumHeavyAtoms()
            or any(a.GetAtomicNum() not in {6, 7, 8, 9, 16, 17, 34, 35, 52, 53} for a in mol.GetAtoms())):
        return None
    if (sum(a.GetAtomicNum() == 7 for a in mol.GetAtoms()) != 3
            or sum(a.GetAtomicNum() in {8, 16, 34, 52} for a in mol.GetAtoms()) != 1):
        return None
    from .hydrazone_methods import HydrazoneSite, _ylidene_component
    candidates = []
    for carbonyl in mol.GetAtoms():
        if carbonyl.GetAtomicNum() != 6 or carbonyl.IsInRing() or carbonyl.GetDegree() != 3:
            continue
        oxy = [a for a in carbonyl.GetNeighbors() if a.GetAtomicNum() in {8, 16, 34, 52} and a.GetDegree() == 1
               and _bond_order(mol, carbonyl.GetIdx(), a.GetIdx()) == 2]
        ns = [a for a in carbonyl.GetNeighbors() if a.GetAtomicNum() == 7
              and _bond_order(mol, carbonyl.GetIdx(), a.GetIdx()) == 1]
        if len(oxy) != 1 or len(ns) != 2:
            continue
        for n1, amide_n in ((ns[0], ns[1]), (ns[1], ns[0])):
            n2s = [a for a in n1.GetNeighbors() if a.GetAtomicNum() == 7 and a.GetIdx() != amide_n.GetIdx()
                   and _bond_order(mol, n1.GetIdx(), a.GetIdx()) == 1]
            if (len(n2s) != 1 or n1.GetDegree() not in {2, 3}
                    or n1.GetDegree() + n1.GetTotalNumHs() != 3):
                continue
            n2 = n2s[0]
            carbons = [a for a in n2.GetNeighbors() if a.GetAtomicNum() == 6
                       and _bond_order(mol, n2.GetIdx(), a.GetIdx()) == 2]
            if len(carbons) != 1 or n2.GetDegree() != 2 or n2.GetTotalNumHs():
                continue
            unit = _ylidene_component(mol, Chem, HydrazoneSite(carbons[0].GetIdx(), n2.GetIdx(), n1.GetIdx()))
            if unit is None:
                unit = _extended_semicarbazone_ylidene(mol, Chem, carbons[0].GetIdx(), n2.GetIdx())
            if unit is None:
                # A substituted benzylidene is the aryl counterpart of the
                # chain writer. Halogens cannot outrank the amide class.
                host = carbons[0]
                roots = [a for a in host.GetNeighbors() if a.GetIdx() != n2.GetIdx()]
                if (host.GetTotalNumHs() != 1 or len(roots) != 1 or not roots[0].GetIsAromatic()
                        or _bond_order(mol, host.GetIdx(), roots[0].GetIdx()) != 1):
                    continue
                aryl, aryl_atoms = mapped_branch(mol, Chem, roots[0].GetIdx(), host.GetIdx())
                if not aryl or not aryl.endswith('phenyl') or any(
                        mol.GetAtomWithIdx(i).GetAtomicNum() not in {6, 9, 17, 35, 53} for i in aryl_atoms):
                    continue
                yname = aryl[:-len('phenyl')] + 'benzylidene'
                yatoms = set(aryl_atoms) | {host.GetIdx()}
            elif len(unit) == 2:
                yname, yatoms = unit
            else:
                yname, yatoms, _mapping, _locant = unit
            seen = set(yatoms) | {carbonyl.GetIdx(), oxy[0].GetIdx(), n1.GetIdx(), n2.GetIdx(), amide_n.GetIdx()}
            prefixes = [('2', yname)]
            n1_roots = [a for a in n1.GetNeighbors() if a.GetIdx() not in {carbonyl.GetIdx(), n2.GetIdx()}]
            if len(n1_roots) > 1:
                continue
            for root in n1_roots:
                text, owned = mapped_branch(mol, Chem, root.GetIdx(), n1.GetIdx())
                if (root.GetAtomicNum() != 6 or not text or seen & owned
                        or any(mol.GetAtomWithIdx(i).GetAtomicNum() not in {6, 9, 17, 35, 53} for i in owned)):
                    break
                prefixes.append(('1', text))
                seen.update(owned)
            else:
                pass
            if len(prefixes) != 1 + len(n1_roots):
                continue
            roots = [a for a in amide_n.GetNeighbors() if a.GetIdx() != carbonyl.GetIdx()]
            if len(roots) > 2:
                continue
            for root in roots:
                if root.GetAtomicNum() != 6:
                    break
                text, owned = mapped_branch(mol, Chem, root.GetIdx(), amide_n.GetIdx())
                if not text or seen & owned or any(
                        mol.GetAtomWithIdx(i).GetAtomicNum() not in {6, 9, 17, 35, 53} for i in owned):
                    break
                prefixes.append(('N', text))
                seen.update(owned)
            else:
                if seen != set(range(mol.GetNumAtoms())):
                    continue
                suffix = {8: 'carboxamide', 16: 'carbothioamide',
                          34: 'carboselenoamide', 52: 'carbotelluroamide'}[oxy[0].GetAtomicNum()]
                name = format_prefixes(prefixes, hierarchical_enclosures=True) + 'hydrazine-1-' + suffix
                candidates.append(_decision(mol, name, (carbonyl.GetIdx(),), 'P-68.3.1.2.5/P-41'))
    return candidates[0] if len(candidates) == 1 else None


def modified_carbonic_diester_decision(mol, Chem):
    """A fully esterified carbonic-acid chalcogen analogue."""
    if (not _plain(mol, Chem) or mol.GetNumAtoms() != mol.GetNumHeavyAtoms()
            or any(a.GetAtomicNum() not in {6, 8, 9, 16, 17, 34, 35, 52, 53} for a in mol.GetAtoms())):
        return None
    from .formatting import COMPLEX_MULT, enclose_name_unit, prefix_alphabetization_key
    elements = {8: 'O', 16: 'S', 34: 'Se', 52: 'Te'}
    infices = {16: 'thio', 34: 'seleno', 52: 'telluro'}
    results = []
    for carbon in mol.GetAtoms():
        if carbon.GetAtomicNum() != 6 or carbon.GetDegree() != 3 or carbon.IsInRing():
            continue
        double = [a for a in carbon.GetNeighbors() if a.GetAtomicNum() in elements and a.GetDegree() == 1
                  and _bond_order(mol, carbon.GetIdx(), a.GetIdx()) == 2]
        arms = [a for a in carbon.GetNeighbors() if a.GetAtomicNum() in elements and a.GetDegree() == 2
                and _bond_order(mol, carbon.GetIdx(), a.GetIdx()) == 1]
        if len(double) != 1 or len(arms) != 2:
            continue
        numbers = [double[0].GetAtomicNum(), *(a.GetAtomicNum() for a in arms)]
        replacements = Counter(n for n in numbers if n != 8)
        if not replacements:
            continue
        seen = {carbon.GetIdx(), double[0].GetIdx(), *(a.GetIdx() for a in arms)}
        organyl = []
        for arm in arms:
            root = [a for a in arm.GetNeighbors() if a.GetIdx() != carbon.GetIdx()]
            if len(root) != 1 or root[0].GetAtomicNum() != 6 or _bond_order(mol, arm.GetIdx(), root[0].GetIdx()) != 1:
                break
            text, atoms = mapped_branch(mol, Chem, root[0].GetIdx(), arm.GetIdx())
            if not text or seen & atoms or any(
                    mol.GetAtomWithIdx(i).GetAtomicNum() not in {6, 8, 9, 17, 35, 53} for i in atoms):
                break
            # Halo and ether groups are subordinate to the ester. Decline
            # other oxygen functionality instead of guessing its seniority or
            # treating an acid/ester inside the branch as a simple ether.
            if any(atom.GetAtomicNum() == 8 and
                   (atom.GetDegree() != 2 or atom.GetTotalNumHs() or
                    any(bond.GetBondTypeAsDouble() != 1 for bond in atom.GetBonds()))
                   for atom in (mol.GetAtomWithIdx(i) for i in atoms)):
                break
            seen.update(atoms)
            organyl.append((text, elements[arm.GetAtomicNum()]))
        else:
            if len(organyl) != 2 or seen != set(range(mol.GetNumAtoms())):
                continue
            # Element locants distinguish otherwise possible chalcogen
            # positions; all-same-element structures have no such ambiguity.
            use_locants = len(set(numbers)) > 1
            grouped = {}
            for text, locant in organyl:
                grouped.setdefault(text, []).append(locant)
            words = []
            for text in sorted(grouped, key=prefix_alphabetization_key):
                locants = sorted(grouped[text], key=lambda s: ('O', 'S', 'Se', 'Te').index(s))
                loc = ','.join(locants) + '-' if use_locants else ''
                if len(locants) == 1:
                    owner = text if text.isalpha() else enclose_name_unit(text)
                elif text.isalpha():
                    owner = MULT[len(locants)] + text
                else:
                    owner = COMPLEX_MULT[len(locants)] + enclose_name_unit(text)
                words.append(loc + owner)
            replacement = ''.join((MULT[count] if count > 1 else '') + infices[number]
                                  for number, count in sorted(replacements.items(), key=lambda row: infices[row[0]]))
            name = ' '.join(words) + ' carbono' + replacement + 'ate'
            results.append(_decision(mol, name, (carbon.GetIdx(),), 'P-65.6.3.3.7.2.1', method='functional-class'))
    return results[0] if len(results) == 1 else None


def modified_carboxylate_decision(mol, Chem):
    """Monoanions of carbon-only thio acids and peroxy acid analogues."""
    if (not _plain(mol, Chem, charged=True) or mol.GetNumAtoms() != mol.GetNumHeavyAtoms()
            or any(a.GetAtomicNum() not in {6, 8, 16, 34, 52} for a in mol.GetAtoms())):
        return None
    charged = [a for a in mol.GetAtoms() if a.GetFormalCharge()]
    if len(charged) != 1 or charged[0].GetFormalCharge() != -1 or charged[0].GetDegree() != 1:
        return None
    end = charged[0]
    if end.GetAtomicNum() not in {8, 16, 34, 52}:
        return None
    inner = end.GetNeighbors()[0]
    peroxide = inner.GetAtomicNum() in {8, 16, 34, 52}
    if peroxide:
        if (inner.GetDegree() != 2 or _bond_order(mol, inner.GetIdx(), end.GetIdx()) != 1):
            return None
        centers = [a for a in inner.GetNeighbors() if a.GetIdx() != end.GetIdx()]
        if len(centers) != 1:
            return None
        center = centers[0]
    else:
        center = inner
    if center.GetAtomicNum() != 6 or center.GetDegree() not in {2, 3}:
        return None
    terminal = [a for a in center.GetNeighbors() if a.GetAtomicNum() in {8, 16, 34, 52}
                and a.GetDegree() == 1 and _bond_order(mol, center.GetIdx(), a.GetIdx()) == 2]
    attachment = inner if peroxide else end
    if len(terminal) != 1 or _bond_order(mol, center.GetIdx(), attachment.GetIdx()) != 1:
        return None
    owned = {center.GetIdx(), terminal[0].GetIdx(), end.GetIdx()}
    if peroxide:
        owned.add(inner.GetIdx())
    if any(a.GetAtomicNum() != 6 and a.GetIdx() not in owned for a in mol.GetAtoms()):
        return None
    copy = Chem.RWMol(mol)
    terminal_copy = copy.GetAtomWithIdx(end.GetIdx())
    terminal_copy.SetFormalCharge(0)
    terminal_copy.SetNumExplicitHs(1)
    terminal_copy.SetNoImplicit(True)
    double_number = terminal[0].GetAtomicNum()
    if peroxide:
        # Name an ordinary peroxy acid with the same carbon parent, then
        # reinsert explicitly proved chalcogen replacements below.
        copy.GetAtomWithIdx(terminal[0].GetIdx()).SetAtomicNum(8)
        copy.GetAtomWithIdx(inner.GetIdx()).SetAtomicNum(8)
        terminal_copy.SetAtomicNum(8)
    acid = _sanitize(copy, Chem)
    if acid is None:
        return None
    from .api import mol_to_iupac
    neutral = mol_to_iupac(acid)
    if peroxide:
        if not neutral.endswith('peroxoic acid'):
            return None
        parent = neutral[:-len('peroxoic acid')]
        pair = (inner.GetAtomicNum(), end.GetAtomicNum())
        if pair == (8, 8):
            infix = {8: '', 16: 'thio', 34: 'seleno', 52: 'telluro'}[double_number]
            name = parent + 'peroxo' + infix + 'ate'
        elif double_number == 8 and set(pair) == {8, 16}:
            locator = ''.join({8: 'O', 16: 'S'}[n] for n in pair)
            name = parent + '(' + locator + '-thioperoxoate)'
        else:
            return None
    else:
        # Monothio/selenium/tellurium carboxylates are resonance descriptions
        # and are cited without the acid's O/S/Se/Te tautomer locant.
        import re
        match = re.fullmatch(r'(.+(?:thio|seleno|telluro))ic (?:O|S|Se|Te)-acid', neutral)
        if match is None:
            return None
        name = match.group(1) + 'ate'
    return _decision(mol, name, (center.GetIdx(),), 'P-72.2.2.2.1.1', method='functional-replacement')


def noncarbon_acid_pseudohalide_decision(mol, Chem):
    """P/As/Sb and sulfur oxoacid pseudohalides with uniform pseudogroups."""
    if not _plain(mol, Chem) or mol.GetNumAtoms() != mol.GetNumHeavyAtoms():
        return None
    centers = [a for a in mol.GetAtoms() if a.GetAtomicNum() in {15, 33, 51, 16}]
    # An isothiocyanato terminus is also sulfur, so determine the core by
    # its direct oxo/organic/pseudohalide neighborhood, not element count.
    possibilities = []
    for center in centers:
        if center.IsInRing():
            continue
        oxos, halides, roots, pseudos = [], [], [], []
        ok = True
        for atom in center.GetNeighbors():
            order = _bond_order(mol, center.GetIdx(), atom.GetIdx())
            if atom.GetAtomicNum() == 8 and atom.GetDegree() == 1 and order == 2:
                oxos.append(atom.GetIdx())
            elif atom.GetAtomicNum() in {9, 17, 35, 53} and atom.GetDegree() == 1 and order == 1:
                halides.append(atom)
            elif atom.GetAtomicNum() == 7 and atom.GetDegree() == 2 and order == 1:
                mids = [a for a in atom.GetNeighbors() if a.GetIdx() != center.GetIdx() and a.GetAtomicNum() == 6
                        and _bond_order(mol, atom.GetIdx(), a.GetIdx()) == 2]
                ends = ([a for a in mids[0].GetNeighbors() if a.GetIdx() != atom.GetIdx()
                         and a.GetAtomicNum() in {8, 16, 34, 52} and a.GetDegree() == 1
                         and _bond_order(mol, mids[0].GetIdx(), a.GetIdx()) == 2] if len(mids) == 1 else [])
                if len(ends) != 1:
                    ok = False
                    break
                cls = {8: ('isocyanate', 'isocyanatido'), 16: ('isothiocyanate', 'isothiocyanatido'),
                       34: ('isoselenocyanate', 'isoselenocyanatido'),
                       52: ('isotellurocyanate', 'isotellurocyanatido')}[ends[0].GetAtomicNum()]
                pseudos.append((cls, {atom.GetIdx(), mids[0].GetIdx(), ends[0].GetIdx()}))
            elif atom.GetAtomicNum() == 6 and order == 1:
                nitrogens = [a for a in atom.GetNeighbors() if a.GetAtomicNum() == 7 and a.GetDegree() == 1
                             and _bond_order(mol, atom.GetIdx(), a.GetIdx()) == 3]
                if atom.GetDegree() == 2 and len(nitrogens) == 1:
                    pseudos.append((('cyanide', 'cyanido'), {atom.GetIdx(), nitrogens[0].GetIdx()}))
                else:
                    roots.append(atom.GetIdx())
            else:
                ok = False
                break
        if not ok or not pseudos or len({row[0] for row in pseudos}) != 1:
            continue
        count = len(pseudos)
        if center.GetAtomicNum() == 16:
            if len(oxos) != 2 or roots or center.GetTotalNumHs() or count + len(halides) != 2:
                continue
            stem = 'sulfuric'
            acyl = 'sulfuryl'
        else:
            if len(oxos) > 1 or len(roots) > 2:
                continue
            fixed = len(roots) + center.GetTotalNumHs()
            if fixed not in {0, 1, 2} or fixed + count + len(halides) != 3:
                continue
            names = {15: ('phosphor', 'phosphon', 'phosphin'),
                     33: ('arsor', 'arson', 'arsin'), 51: ('stibor', 'stibon', 'stibin')}
            stem = names[center.GetAtomicNum()][fixed] + ('ic' if oxos else 'ous')
            acyl = 'phosphoryl' if center.GetAtomicNum() == 15 and oxos and not fixed else ''
        seen = {center.GetIdx(), *oxos, *(a.GetIdx() for a in halides)}
        for _cls, atoms in pseudos:
            if atoms & seen:
                ok = False
                break
            seen.update(atoms)
        if not ok:
            continue
        from .functional_replacement_acids import _organyl_units
        prefix = _organyl_units(mol, Chem, center.GetIdx(), roots, seen)
        if prefix is None or seen != set(range(mol.GetNumAtoms())):
            continue
        cls, infix = pseudos[0][0]
        if halides:
            if len(halides) != 1 or not stem.endswith(('ic', 'ous')):
                continue
            ending = 'ic' if stem.endswith('ic') else 'ous'
            base = stem[:-len(ending)]
            joined = base + ('o' if count > 1 or infix[0] not in 'aeiou' else '')
            replacement = (MULT[count] if count > 1 else '') + infix[:-1]
            acid = joined + replacement + ending
            word = {9: 'fluoride', 17: 'chloride', 35: 'bromide', 53: 'iodide'}[halides[0].GetAtomicNum()]
            name = prefix + acid + ' ' + word
        else:
            base = acyl if acyl and count > 1 else stem
            name = prefix + base + ' ' + (MULT[count] if count > 1 else '') + cls
        possibilities.append(_decision(mol, name, (center.GetIdx(),), 'P-67.1.2.5.1/P-41',
                                       method='functional-class'))
    return possibilities[0] if len(possibilities) == 1 else None


def mancude_azocine_decision(mol, Chem):
    """Mancude azocines, including bounded alkyl and halo substitution."""
    if not _plain(mol, Chem) or mol.GetNumAtoms() != mol.GetNumHeavyAtoms():
        return None
    rings = mol.GetRingInfo().AtomRings()
    if len(rings) != 1 or len(rings[0]) != 8:
        return None
    ring_atoms = frozenset(rings[0])
    if (any(mol.GetAtomWithIdx(i).GetAtomicNum() not in {6, 7} for i in ring_atoms)
            or not any(mol.GetAtomWithIdx(i).GetAtomicNum() == 7 for i in ring_atoms)):
        return None
    copy = Chem.Mol(mol)
    try:
        Chem.Kekulize(copy, clearAromaticFlags=True)
    except (ValueError, RuntimeError):
        return None
    doubles = [b for b in copy.GetBonds() if b.GetBondTypeAsDouble() == 2
               and b.GetBeginAtomIdx() in ring_atoms and b.GetEndAtomIdx() in ring_atoms]
    endpoints = [i for b in doubles for i in (b.GetBeginAtomIdx(), b.GetEndAtomIdx())]
    if len(doubles) != 4 or len(set(endpoints)) != 8:
        return None
    from .cycle_utils import ordered_ring
    from .formatting import prefix_alphabetization_key
    from .ring_rules import _medium_aromatic_heterocycle_candidates
    ring = ordered_ring(mol, rings[0])
    counts = Counter(mol.GetAtomWithIdx(i).GetSymbol() for i in ring_atoms)
    candidates = _medium_aromatic_heterocycle_candidates(mol, ring, counts)
    if not candidates:
        return None
    halogens = {9: 'fluoro', 17: 'chloro', 35: 'bromo', 53: 'iodo'}
    rows = []
    for parent, seq in candidates:
        seen = set(ring_atoms)
        prefixes = []
        for locant, idx in enumerate(seq, 1):
            for atom in mol.GetAtomWithIdx(idx).GetNeighbors():
                root = atom.GetIdx()
                if root in ring_atoms:
                    continue
                if _bond_order(mol, idx, root) != 1:
                    break
                if atom.GetAtomicNum() in halogens and atom.GetDegree() == 1:
                    text, owned = halogens[atom.GetAtomicNum()], frozenset({root})
                elif atom.GetAtomicNum() == 6:
                    text, owned = mapped_branch(mol, Chem, root, idx)
                    if (text not in {'methyl', 'ethyl'} or
                            any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in owned)):
                        break
                else:
                    break
                if not owned or seen & owned:
                    break
                seen.update(owned)
                prefixes.append((str(locant), text))
            else:
                continue
            break
        else:
            if seen != set(range(mol.GetNumAtoms())):
                continue
            nitrogen_locs = tuple(i + 1 for i, idx in enumerate(seq)
                                  if mol.GetAtomWithIdx(idx).GetAtomicNum() == 7)
            grouped = {}
            for locant, text in prefixes:
                grouped.setdefault(text, []).append(int(locant))
            prefix_key = tuple((prefix_alphabetization_key(text), tuple(sorted(locs)))
                               for text, locs in sorted(grouped.items(),
                                                        key=lambda item: prefix_alphabetization_key(item[0])))
            key = (nitrogen_locs, tuple(sorted(int(loc) for loc, _ in prefixes)), prefix_key)
            lead = format_prefixes(prefixes, hierarchical_enclosures=True)
            name = lead + ('-' if lead and parent[0].isdigit() else '') + parent
            rows.append((key, name, seq))
    if not rows:
        return None
    best = min(row[0] for row in rows)
    winners = [row for row in rows if row[0] == best]
    if len({row[1] for row in winners}) != 1:
        return None
    # A mancude name alone does not identify every possible localized-bond
    # arrangement. Do not map a different input graph onto the default
    # 1=2, 3=4, 5=6, 7=8 construction merely because its prefix locants are
    # lower. The existing localized-bond writer can handle the other cases.
    aligned = [row for row in winners if all(
        _bond_order(copy, row[2][i], row[2][(i + 1) % 8]) == (2 if i % 2 == 0 else 1)
        for i in range(8))]
    if not aligned:
        return None
    winners = aligned
    _, name, seq = winners[0]
    locants = tuple((idx, str(i + 1)) for i, idx in enumerate(seq))
    return _decision(mol, name, (), 'P-22.2.2.1', method='ring-parent', locants=locants)


def boron_cyanatido_anion_decision(mol, Chem):
    """Partially neutralised boric acid bearing cyanatido replacements."""
    if (not _plain(mol, Chem, charged=True) or mol.GetNumAtoms() != mol.GetNumHeavyAtoms()
            or any(a.GetAtomicNum() not in {5, 6, 7, 8} for a in mol.GetAtoms())):
        return None
    borons = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 5]
    if len(borons) != 1 or borons[0].GetDegree() != 3 or borons[0].GetFormalCharge():
        return None
    boron = borons[0]
    anions, hydrogens, groups = [], [], []
    seen = {boron.GetIdx()}
    from .bluebook_hydride_assembly_methods import _cyano_replacement
    for atom in boron.GetNeighbors():
        if _bond_order(mol, boron.GetIdx(), atom.GetIdx()) != 1:
            return None
        if atom.GetAtomicNum() == 8 and atom.GetDegree() == 1 and atom.GetFormalCharge() == -1:
            anions.append(atom.GetIdx())
            seen.add(atom.GetIdx())
        elif (atom.GetAtomicNum() == 8 and atom.GetDegree() == 1 and not atom.GetFormalCharge()
              and atom.GetTotalNumHs() == 1):
            hydrogens.append(atom.GetIdx())
            seen.add(atom.GetIdx())
        elif replacement := _cyano_replacement(mol, boron.GetIdx(), atom.GetIdx()):
            if replacement[0] != 'cyanatido' or replacement[1] & seen:
                return None
            groups.append(replacement)
            seen.update(replacement[1])
        else:
            return None
    if (not anions or not groups or len(groups) > 2 or seen != set(range(mol.GetNumAtoms()))
            or any(a.GetFormalCharge() != (-1 if a.GetIdx() in anions else 0) for a in mol.GetAtoms())):
        return None
    htext = (MULT[len(hydrogens)] if len(hydrogens) > 1 else '') + 'hydrogen ' if hydrogens else ''
    infix = (MULT[len(groups)] if len(groups) > 1 else '') + 'cyanatido'
    return _decision(mol, htext + 'boro' + infix[:-1] + 'ate', (boron.GetIdx(),),
                     'P-67.1.3.1/P-67.1.2.4', method='functional-replacement')


def isotopic_benzene_suffix_decision(mol, Chem):
    """P-82.6.3.2: identify an unnumbered retained-parent suffix atom.

    The source examples require a systematic benzene suffix when the
    benzonitrile or benzoic-acid carbon is isotopically substituted. A ring
    with direct, unlabelled halogens uses the same systematic construction;
    other substitution combinations stay with the mapped general path.
    """
    if (len(Chem.GetMolFrags(mol)) != 1 or mol.GetStereoGroups()
            or mol.GetNumAtoms() != mol.GetNumHeavyAtoms()
            or any(a.GetFormalCharge() or a.GetNumRadicalElectrons() or a.HasQuery()
                   or int(a.GetChiralTag()) for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) or b.HasQuery() for b in mol.GetBonds())):
        return None
    rings = mol.GetRingInfo().AtomRings()
    if len(rings) != 1 or len(rings[0]) != 6:
        return None
    ring = frozenset(rings[0])
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 or mol.GetAtomWithIdx(i).GetIsotope()
           or not mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
        return None
    external = [n for i in ring for n in mol.GetAtomWithIdx(i).GetNeighbors()
                if n.GetIdx() not in ring]
    centers = [a for a in external if a.GetAtomicNum() == 6 and a.GetIsotope()]
    halogens = [a for a in external if a.GetAtomicNum() in {9, 17, 35, 53}]
    if (len(centers) != 1 or len(external) != 1 + len(halogens)
            or any(a.GetDegree() != 1 or a.GetIsotope() for a in halogens)):
        return None
    carbon = centers[0]
    if (carbon.GetAtomicNum() != 6 or not carbon.GetIsotope() or carbon.IsInRing()
            or sum(_bond_order(mol, carbon.GetIdx(), i) == 1 for i in ring) != 1):
        return None
    outside = [n for n in carbon.GetNeighbors() if n.GetIdx() not in ring]
    if (len(outside) == 1 and outside[0].GetAtomicNum() == 7 and outside[0].GetDegree() == 1
            and _bond_order(mol, carbon.GetIdx(), outside[0].GetIdx()) == 3):
        nitrogen = outside[0]
        if mol.GetNumAtoms() != 8 + len(halogens):
            return None
        descriptor = str(carbon.GetIsotope()) + 'C'
        if nitrogen.GetIsotope():
            descriptor += ',' + str(nitrogen.GetIsotope()) + 'N'
        suffix = '(' + descriptor + ')carbonitrile'
        retained = 'benzonitrile'
    elif (len(outside) == 2 and mol.GetNumAtoms() == 9 + len(halogens)
          and all(a.GetAtomicNum() == 8 and a.GetDegree() == 1 and not a.GetIsotope() for a in outside)
          and sorted(_bond_order(mol, carbon.GetIdx(), a.GetIdx()) for a in outside) == [1, 2]
          and any(a.GetTotalNumHs() == 1 for a in outside)):
        suffix = '(' + str(carbon.GetIsotope()) + 'C)carboxylic acid'
        retained = 'benzoic acid'
    else:
        return None
    if halogens:
        parent = Chem.Mol(mol)
        for atom in parent.GetAtoms():
            atom.SetIsotope(0)
        from .api import mol_to_iupac
        parent_name = mol_to_iupac(parent)
        if not parent_name.endswith(retained) or parent_name == retained:
            return None
        prefix = parent_name[:-len(retained)]
        name = prefix + 'benzene-1-' + suffix
    else:
        name = 'benzene' + suffix
    return _decision(mol, name, (carbon.GetIdx(),), 'P-82.6.3.2/P-82.2.1', method='substitutive')


def isotopic_halomethane_decision(mol, Chem):
    """Mononuclear halomethanes with a substituted carbon nuclide."""
    if (len(Chem.GetMolFrags(mol)) != 1 or mol.GetStereoGroups()
            or any(a.GetFormalCharge() or a.GetNumRadicalElectrons() or a.HasQuery()
                   or int(a.GetChiralTag()) or a.GetAtomicNum() == 1 for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) or b.HasQuery() for b in mol.GetBonds())):
        return None
    carbons = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 6]
    if len(carbons) != 1 or not carbons[0].GetIsotope():
        return None
    c = carbons[0]
    halogens = {9: 'fluoro', 17: 'chloro', 35: 'bromo', 53: 'iodo'}
    if (not 1 <= c.GetDegree() <= 4 or c.GetTotalValence() != 4
            or any(a.GetAtomicNum() not in halogens or a.GetDegree() != 1 or a.GetIsotope()
                   or _bond_order(mol, c.GetIdx(), a.GetIdx()) != 1 for a in c.GetNeighbors())
            or c.GetDegree() != mol.GetNumAtoms() - 1):
        return None
    from .prefix_sequences import PrefixUnit, mononuclear_prefix_sequence
    sequence = mononuclear_prefix_sequence([PrefixUnit(halogens[a.GetAtomicNum()], 'simple')
                                            for a in c.GetNeighbors()])
    if sequence is None:
        return None
    return _decision(mol, sequence.text + f'({c.GetIsotope()}C)methane', (c.GetIdx(),),
                     'P-14.3.4.2/P-82.2.1')


def isotopic_formate_decision(mol, Chem):
    """The monocarbon formate ion with an isotopically substituted carbon."""
    if (len(Chem.GetMolFrags(mol)) != 1 or mol.GetNumAtoms() != 3 or mol.GetStereoGroups()
            or any(a.GetNumRadicalElectrons() or a.HasQuery() or int(a.GetChiralTag()) for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) or b.HasQuery() for b in mol.GetBonds())):
        return None
    cs = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 6]
    oxygens = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 8]
    if len(cs) != 1 or len(oxygens) != 2 or not cs[0].GetIsotope():
        return None
    carbon = cs[0]
    if carbon.GetFormalCharge() or carbon.GetTotalNumHs() != 1 or any(a.GetIsotope() for a in oxygens):
        return None
    states = sorted((a.GetFormalCharge(), _bond_order(mol, carbon.GetIdx(), a.GetIdx())) for a in oxygens)
    if states != [(-1, 1), (0, 2)]:
        return None
    return _decision(mol, f'({carbon.GetIsotope()}C)formate', (carbon.GetIdx(),), 'P-82.2.1/P-77',
                     method='functional-class')


def isotopic_methoxybenzene_decision(mol, Chem):
    """Nuclides on the methyl group of a single methoxybenzene parent."""
    if (len(Chem.GetMolFrags(mol)) != 1 or mol.GetStereoGroups()
            or any(a.GetFormalCharge() or a.GetNumRadicalElectrons() or a.HasQuery()
                   or int(a.GetChiralTag()) for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) or b.HasQuery() for b in mol.GetBonds())):
        return None
    rings = mol.GetRingInfo().AtomRings()
    if len(rings) != 1 or len(rings[0]) != 6:
        return None
    ring = frozenset(rings[0])
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 or not mol.GetAtomWithIdx(i).GetIsAromatic()
           or mol.GetAtomWithIdx(i).GetIsotope() for i in ring):
        return None
    oxygens = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 8]
    methyls = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 6 and a.GetIdx() not in ring]
    if len(oxygens) != 1 or len(methyls) != 1:
        return None
    oxygen, methyl = oxygens[0], methyls[0]
    if (oxygen.GetDegree() != 2 or oxygen.GetIsotope() or methyl.IsInRing()
            or _bond_order(mol, oxygen.GetIdx(), methyl.GetIdx()) != 1
            or sum(a.GetIdx() in ring for a in oxygen.GetNeighbors()) != 1):
        return None
    h_atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 1]
    if ring | {oxygen.GetIdx(), methyl.GetIdx(), *(a.GetIdx() for a in h_atoms)} != frozenset(range(mol.GetNumAtoms())):
        return None
    if any(a.GetDegree() != 1 or a.GetNeighbors()[0].GetIdx() != methyl.GetIdx()
           or a.GetIsotope() not in {2, 3} for a in h_atoms):
        return None
    if methyl.GetDegree() != 1 + len(h_atoms) or methyl.GetTotalValence() != 4:
        return None
    isotopes = Counter((a.GetSymbol(), a.GetIsotope()) for a in [methyl, *h_atoms] if a.GetIsotope())
    if not isotopes:
        return None
    words = [f'{mass}{symbol}' + (str(count) if symbol == 'H' or count > 1 else '')
             for (symbol, mass), count in sorted(isotopes.items())]
    return _decision(mol, '(' + ','.join(words) + ')methoxybenzene', (oxygen.GetIdx(),), 'P-82.2.1')


def isotopic_phenylketone_decision(mol, Chem):
    """Mapped carbon nuclides on a simple phenylalkan-1-one chain."""
    if (len(Chem.GetMolFrags(mol)) != 1 or mol.GetStereoGroups()
            or any(a.GetFormalCharge() or a.GetNumRadicalElectrons() or a.HasQuery()
                   or int(a.GetChiralTag()) or a.GetAtomicNum() == 1 for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) or b.HasQuery() for b in mol.GetBonds())):
        return None
    if any(a.GetAtomicNum() not in {6, 8} for a in mol.GetAtoms()):
        return None
    rings = mol.GetRingInfo().AtomRings()
    if len(rings) != 1 or len(rings[0]) != 6:
        return None
    ring = frozenset(rings[0])
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 or not mol.GetAtomWithIdx(i).GetIsAromatic()
           or mol.GetAtomWithIdx(i).GetIsotope() for i in ring):
        return None
    oxygens = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 8]
    if len(oxygens) != 1 or oxygens[0].GetDegree() != 1 or oxygens[0].GetIsotope():
        return None
    oxygen = oxygens[0]
    carbonyl = oxygen.GetNeighbors()[0]
    if (carbonyl.GetAtomicNum() != 6 or carbonyl.GetIdx() in ring
            or _bond_order(mol, carbonyl.GetIdx(), oxygen.GetIdx()) != 2):
        return None
    adjacent_ring = [a for a in carbonyl.GetNeighbors() if a.GetIdx() in ring]
    chain = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6 and a.GetIdx() not in ring}
    if len(adjacent_ring) != 1 or len(chain) < 2 or any(
            n.GetIdx() not in ring and n.GetIdx() != carbonyl.GetIdx()
            for i in ring for n in mol.GetAtomWithIdx(i).GetNeighbors()):
        return None
    from .bluebook_fusion_functional_methods import _carbon_path
    path = _carbon_path(mol, chain, carbonyl.GetIdx())
    from .constants import ROOTS
    if not path or len(path) not in ROOTS or any(
            _bond_order(mol, a, b) != 1 for a, b in zip(path, path[1:])):
        return None
    labels = [(i + 1, mol.GetAtomWithIdx(idx).GetIsotope()) for i, idx in enumerate(path)
              if mol.GetAtomWithIdx(idx).GetIsotope()]
    if not labels or len({mass for _, mass in labels}) != 1:
        return None
    locants = ','.join(str(loc) for loc, _ in labels)
    descriptor = f"({locants}-{labels[0][1]}C" + (str(len(labels)) if len(labels) > 1 else '') + ')'
    name = '1-phenyl' + descriptor + ROOTS[len(path)] + 'an-1-one'
    return _decision(mol, name, (carbonyl.GetIdx(),), 'P-82.2.1/P-82.6.1')
