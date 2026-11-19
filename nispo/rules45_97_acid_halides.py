"""Mapped P-67 modified acids and their halide/pseudohalide classes.

Ligands are classified by their original bonds. A principal construction is
emitted only after disjoint ligand ownership closes the whole graph; higher
carbon acid/amide functions are excluded from the organyl grammar. Prefix
operations retain their own attachment direction and atom ownership.
"""
from collections import Counter

from .bluebook_lambda_amine_methods import _plain, _out, _sanitize
from .component_mapping import mapped_branch
from .constants import MULT
from .formatting import format_prefixes, enclose_name_unit
from .molecule_iteration import iter_atoms, iter_bonds

_HALOGEN = {9: ('fluoride', 'fluorido'), 17: ('chloride', 'chlorido'),
            35: ('bromide', 'bromido'), 53: ('iodide', 'iodido')}
_CHALCOGEN = {8: '', 16: 'thio', 34: 'seleno', 52: 'telluro'}
_PSEUDO = {8: ('isocyanate', 'isocyanatido'),
           16: ('isothiocyanate', 'isothiocyanatido'),
           34: ('isoselenocyanate', 'isoselenocyanatido'),
           52: ('isotellurocyanate', 'isotellurocyanatido')}
_CHALCOGEN_MODIFIED_INFIXES = frozenset(
    infix for element, (_word, infix) in _PSEUDO.items() if element != 8)
_PSEUDO_SENIORITY = {word: rank for rank, (word, _infix) in
                     enumerate(_PSEUDO.values())}


def _order(mol, a, b):
    bond = mol.GetBondBetweenAtoms(a, b)
    return bond.GetBondTypeAsDouble() if bond else 0


def _carbon_arm(mol, Chem, root, host):
    from .phosphorus_ester_methods import _carbon_component
    atoms = _carbon_component(mol, root, host)
    if atoms is None:
        # The P-acid branch grammar also maps ordinary carbon annotations and
        # an acyl carbon at the attachment. A free carboxylic acid is excluded.
        from .phosphorus_acid_methods import _hydrocarbon_branch
        atoms = _hydrocarbon_branch(mol, root, host)
        if atoms is None:
            from .functional_replacement_acids import _hydroxy_amino_organyl
            atoms = _hydroxy_amino_organyl(mol, host, root, {host})
            if atoms is None:
                atoms = _ether_heteroaryl_organyl(mol, root, host)
                if atoms is None:
                    return None
    name, owned = mapped_branch(mol, Chem, root, host)
    if not name or frozenset(owned) != atoms:
        return None
    return name, atoms


def _ether_heteroaryl_organyl(mol, root, host):
    """Lower-class ethers and neutral heteroaryl ligands, with one exit."""
    if mol.GetAtomWithIdx(root).GetAtomicNum() != 6 or _order(mol, root, host) != 1:
        return None
    from .branch_rules import collect_subtree
    atoms = frozenset(collect_subtree(mol, root, host))
    if host in atoms:
        return None
    for idx in atoms:
        atom = mol.GetAtomWithIdx(idx)
        element = atom.GetAtomicNum()
        if element == 6:
            continue
        if element in _HALOGEN:
            if atom.GetDegree() != 1 or any(_order(mol, idx, n.GetIdx()) != 1 for n in atom.GetNeighbors()):
                return None
        elif element == 7:
            if not atom.IsInRing() or not atom.GetIsAromatic():
                return None
        elif element in {8, 16, 34, 52}:
            if atom.IsInRing() and atom.GetIsAromatic() and atom.GetTotalValence() == 2:
                continue
            if element != 8 or atom.GetDegree() != 2 or any(n.GetAtomicNum() != 6
                    or _order(mol, idx, n.GetIdx()) != 1 for n in atom.GetNeighbors()):
                return None
        else:
            return None
    exits = {(i, n.GetIdx()) for i in atoms for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() not in atoms}
    return atoms if exits == {(root, host)} else None


def _infix_name(stem, terms, ending):
    """Order and elide infixes, retaining P-67.1.2.4.1 enclosures.

    A chalcogen-modified infix is one enclosed unit. When it ends the acid
    name, its suffix is inside that unit; a following infix remains outside.
    Numerical multipliers precede the enclosed unit (P-16.5.1.1).
    """
    counts = Counter(terms)
    pieces = [(term, MULT[count] if count > 1 else '')
              for term, count in sorted(counts.items())]
    if not pieces:
        return stem + ending
    first = pieces[0][1] + pieces[0][0]
    text = stem + ('o' if first[0] not in 'aeiou' else '')
    for index, (term, multiplier) in enumerate(pieces):
        terminal = index + 1 == len(pieces)
        following = (pieces[index + 1][1] + pieces[index + 1][0]) if not terminal else ending
        retain = terminal and ending == 'ic' and term in {
            'thio', 'seleno', 'telluro', 'peroxo'}
        piece = term
        if piece.endswith('o') and following[0] in 'aeiou' and not retain:
            piece = piece[:-1]
        if term in _CHALCOGEN_MODIFIED_INFIXES:
            text += multiplier + '(' + piece + (ending if terminal else '') + ')'
            if terminal:
                return text
        else:
            text += multiplier + piece
    return text + ending


def _organyl_prefix(mol, Chem, center, direct, imido):
    from .formatting import prefix_alphabetization_key
    nitrogen_sites = sorted(imido, key=lambda row: (not bool(row[1]), prefix_alphabetization_key(row[1]) if row[1] else ()))
    nitrogen_rows = [('N' + "'" * index, text)
                     for index, (_nitrogen, text, _atoms) in enumerate(nitrogen_sites) if text]
    if nitrogen_rows:
        prefixes = [('P', text) for _root, text, _atoms in direct]
        prefixes.extend((locant, text) for locant, text in nitrogen_rows)
        return format_prefixes(prefixes, hierarchical_enclosures=True)
    from .prefix_sequences import PrefixUnit, mononuclear_prefix_sequence
    units = []
    for _root, text, atoms in direct:
        simple = all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in atoms) and (text in {'phenyl', 'tert-butyl'} or all(
            sum(n.GetIdx() in atoms for n in mol.GetAtomWithIdx(i).GetNeighbors()) <= 2
            for i in atoms))
        units.append(PrefixUnit(text, 'simple' if simple else 'compound'))
    if not units:
        return ''
    sequence = mononuclear_prefix_sequence(units)
    return sequence.text if sequence is not None else None


def _oxoacid_site(mol, Chem, center):
    """Classify one central atom without inferring chemistry from name text."""
    idx, element = center.GetIdx(), center.GetAtomicNum()
    if center.IsInRing():
        return None
    used = {idx}
    direct, imido, oxo, hydroxyl, halides, pseudo, nitrido = [], [], [], [], [], [], []
    oxide_charge_pair = False
    for atom in center.GetNeighbors():
        j, number, bond = atom.GetIdx(), atom.GetAtomicNum(), _order(mol, idx, atom.GetIdx())
        if number == 6 and bond == 1:
            row = _carbon_arm(mol, Chem, j, idx)
            if row is None:
                return None
            text, atoms = row
            direct.append((j, text, atoms))
        elif number in _CHALCOGEN and atom.GetDegree() == 1 and bond == 2 and not atom.GetTotalNumHs():
            if atom.GetFormalCharge():
                return None
            oxo.append(number); atoms = frozenset({j})
        elif (element == 7 and number == 8 and atom.GetDegree() == 1 and bond == 1
              and center.GetFormalCharge() == 1 and atom.GetFormalCharge() == -1
              and not atom.GetTotalNumHs()):
            oxide_charge_pair = True; oxo.append(8); atoms = frozenset({j})
        elif (number in _CHALCOGEN and atom.GetDegree() == 1 and bond == 1
              and atom.GetTotalNumHs() == 1 and not atom.GetFormalCharge()):
            hydroxyl.append((j, number)); atoms = frozenset({j})
        elif number in _HALOGEN and atom.GetDegree() == 1 and bond == 1 and not atom.GetFormalCharge():
            halides.append(number); atoms = frozenset({j})
        elif number == 7 and bond == 3 and atom.GetDegree() == 1 and not atom.GetTotalNumHs() and not atom.GetFormalCharge():
            nitrido.append(j); atoms = frozenset({j})
        elif number == 7 and bond == 2 and not atom.IsInRing() and not atom.GetFormalCharge():
            others = [a for a in atom.GetNeighbors() if a.GetIdx() != idx]
            if not others and atom.GetTotalNumHs() == 1:
                imido.append((j, '', frozenset())); atoms = frozenset({j})
            elif len(others) == 1 and not atom.GetTotalNumHs() and _order(mol, j, others[0].GetIdx()) == 1:
                row = _carbon_arm(mol, Chem, others[0].GetIdx(), j)
                if row is None:
                    return None
                text, owned = row
                imido.append((j, text, owned)); atoms = owned | {j}
            else:
                return None
        elif number == 7 and bond == 1 and atom.GetDegree() == 2 and not atom.GetTotalNumHs() and not atom.GetFormalCharge():
            others = [a for a in atom.GetNeighbors() if a.GetIdx() != idx]
            carbon = others[0] if len(others) == 1 else None
            ends = [a for a in carbon.GetNeighbors() if a.GetIdx() != j] if carbon is not None else []
            if (carbon is None or carbon.GetAtomicNum() != 6 or carbon.GetDegree() != 2
                    or _order(mol, j, carbon.GetIdx()) != 2 or len(ends) != 1
                    or ends[0].GetAtomicNum() not in _PSEUDO or ends[0].GetDegree() != 1
                    or _order(mol, carbon.GetIdx(), ends[0].GetIdx()) != 2):
                return None
            pseudo.append(_PSEUDO[ends[0].GetAtomicNum()])
            atoms = frozenset({j, carbon.GetIdx(), ends[0].GetIdx()})
        else:
            return None
        if atoms & used:
            return None
        used.update(atoms)
    if used != set(range(mol.GetNumAtoms())):
        return None
    charged = {a.GetIdx() for a in iter_atoms(mol) if a.GetFormalCharge()}
    if oxide_charge_pair:
        if charged != {idx, *(a.GetIdx() for a in center.GetNeighbors() if a.GetFormalCharge() == -1)} or len(charged) != 2:
            return None
    elif charged:
        return None
    return dict(center=idx, element=element, direct=direct, imido=imido,
                oxo=oxo, hydroxyl=hydroxyl, halides=halides, pseudo=pseudo,
                nitrido=nitrido, oxide_charge_pair=oxide_charge_pair)


def modified_oxoacid_decision(mol, Chem):
    # Labels on carbon-only ligand components are emitted by mapped_branch
    # from the original graph; do not place them on the acid's central atom.
    ordinary = Chem.Mol(mol)
    for atom in iter_atoms(ordinary):
        if atom.GetAtomicNum() == 6:
            if atom.GetIsotope() and Chem.GetPeriodicTable().GetMassForIsotope(6, atom.GetIsotope()) <= 0:
                return None
            atom.SetIsotope(0)
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    if not _plain(ordinary, Chem, {6, 7, 8, 9, 15, 16, 17, 34, 35, 52, 53}, charges=(0, -1, 1)):
        return None
    candidates = [a for a in iter_atoms(mol) if a.GetAtomicNum() in {7, 15, 16}
                  and not a.IsInRing() and a.GetDegree() >= 2
                  and (a.GetAtomicNum() != 15 or any(n.GetAtomicNum() == 7
                       and _order(mol, a.GetIdx(), n.GetIdx()) in {2, 3} for n in a.GetNeighbors()))]
    answers = []
    for center in candidates:
        site = _oxoacid_site(mol, Chem, center)
        if site is None:
            continue
        element = site['element']; direct = site['direct']; imido = site['imido']
        oxo = site['oxo']; hydroxyl = site['hydroxyl']; halides = site['halides']; pseudo = site['pseudo']; nitrido = site['nitrido']
        fixed = len(direct) + center.GetTotalNumHs()
        if element == 15:
            if fixed not in {0, 1, 2} or len(nitrido) > 1 or len(oxo) + len(imido) + len(nitrido) != 1:
                continue
            if fixed + len(hydroxyl) + len(halides) + len(pseudo) != 3 - len(nitrido):
                continue
            stem = ('phosphor', 'phosphon', 'phosphin')[fixed]
            ending = 'ic'
        elif element == 16:
            if direct or nitrido or center.GetTotalNumHs() or len(oxo) + len(imido) not in {1, 2}:
                continue
            if len(hydroxyl) + len(halides) + len(pseudo) != 2 or not (imido or pseudo):
                continue
            stem, ending = 'sulfur', 'ic' if len(oxo) + len(imido) == 2 else 'ous'
        else:
            if fixed != 1 or imido or nitrido or len(oxo) > 1:
                continue
            if len(hydroxyl) + len(halides) + len(pseudo) != 2:
                continue
            if not oxo and not hydroxyl:
                continue
            stem, ending = 'azon', 'ic' if oxo else 'ous'
        prefix = _organyl_prefix(mol, Chem, center.GetIdx(), direct, imido)
        if prefix is None:
            continue
        infixes = ['imido'] * len(imido) + ['nitrido'] * len(nitrido)
        infixes += [_CHALCOGEN[e] for e in oxo if e != 8]
        infixes += [_CHALCOGEN[e] for _j, e in hydroxyl if e != 8]
        if hydroxyl:
            infixes += [_HALOGEN[e][1] for e in halides] + [infix for _word, infix in pseudo]
            name = prefix + _infix_name(stem, infixes, ending) + ' acid'
        elif halides:
            infixes += [infix for _word, infix in pseudo]
            acid = _infix_name(stem, infixes, ending)
            words = [(MULT[count] if count > 1 else '') + _HALOGEN[e][0]
                     for e, count in sorted(Counter(halides).items(), key=lambda item: _HALOGEN[item[0]][0])]
            # The retained phosphoryl/sulfuryl exception applies to the
            # ordinary oxo parent, not its imido replacement.
            if (not prefix and not infixes and len(words) == 1 and
                    (element == 15 and fixed == 0 or element == 16 and len(oxo) == 2)):
                acid = 'phosphoryl' if element == 15 else 'sulfuryl'
            name = prefix + acid + ' ' + ' '.join(words)
        elif pseudo:
            # P-67.1.2.5.1: the senior pseudohalide supplies the class term;
            # its lower chalcogen analogues remain replacement infixes.
            senior = min(_PSEUDO_SENIORITY[word] for word, _infix in pseudo)
            infixes += [infix for word, infix in pseudo
                        if _PSEUDO_SENIORITY[word] != senior]
            acid = _infix_name(stem, infixes, ending)
            words = [(MULT[count] if count > 1 else '') + word for word, count in
                     sorted(Counter(word for word, _infix in pseudo
                                    if _PSEUDO_SENIORITY[word] == senior).items())]
            if (not prefix and not infixes and len(words) == 1 and
                    (element == 15 and fixed == 0 or element == 16 and len(oxo) == 2)):
                acid = 'phosphoryl' if element == 15 else 'sulfuryl'
            name = prefix + acid + ' ' + ' '.join(words)
        else:
            continue
        answers.append(_out(mol, name, (center.GetIdx(),),
            'P-13.2.2.1/P-67.1.2.3/P-67.1.2.4/P-67.1.2.5.1',
            'substitutive' if hydroxyl else 'functional-class'))
    return answers[0] if len(answers) == 1 else None


def cyanatosilicic_acid_decision(mol, Chem):
    if not _plain(mol, Chem, {6, 7, 8, 14, 16}):
        return None
    centers = [a for a in iter_atoms(mol) if a.GetAtomicNum() == 14]
    if len(centers) != 1:
        return None
    center = centers[0]
    if center.IsInRing() or center.GetDegree() != 4 or center.GetTotalValence() != 4:
        return None
    used, cyanato, hydroxyl, sulfur = {center.GetIdx()}, 0, 0, 0
    for atom in center.GetNeighbors():
        idx = atom.GetIdx()
        if _order(mol, center.GetIdx(), idx) != 1:
            return None
        if atom.GetAtomicNum() in {8, 16} and atom.GetDegree() == 1 and atom.GetTotalNumHs() == 1:
            hydroxyl += 1
            sulfur += atom.GetAtomicNum() == 16
            used.add(idx)
            continue
        if atom.GetAtomicNum() != 8 or atom.GetDegree() != 2 or atom.GetTotalNumHs():
            return None
        others = [a for a in atom.GetNeighbors() if a.GetIdx() != center.GetIdx()]
        carbon = others[0] if len(others) == 1 else None
        nitrogens = [a for a in carbon.GetNeighbors() if a.GetIdx() != idx] if carbon is not None else []
        if (carbon is None or carbon.GetAtomicNum() != 6 or carbon.GetDegree() != 2
                or _order(mol, idx, carbon.GetIdx()) != 1 or len(nitrogens) != 1
                or nitrogens[0].GetAtomicNum() != 7 or nitrogens[0].GetDegree() != 1
                or _order(mol, carbon.GetIdx(), nitrogens[0].GetIdx()) != 3):
            return None
        atoms = {idx, carbon.GetIdx(), nitrogens[0].GetIdx()}
        if atoms & used:
            return None
        used.update(atoms); cyanato += 1
    if not hydroxyl or not cyanato or used != set(range(mol.GetNumAtoms())):
        return None
    text = (MULT[cyanato] if cyanato > 1 else '') + 'cyanato'
    text += ((MULT[sulfur] if sulfur > 1 else '') + 'thio') if sulfur else ''
    return _out(mol, text + 'silicic acid', (center.GetIdx(),), 'P-67.1.2.4.1.3')


def branch_prefix(mol, Chem, root, blocked):
    """Graph-owned modified phosphorus acyl prefix attached through P."""
    atom = mol.GetAtomWithIdx(root)
    if (atom.GetAtomicNum() != 15 or atom.IsInRing() or atom.GetFormalCharge()
            or atom.GetNumRadicalElectrons() or atom.GetIsotope() or int(atom.GetChiralTag())
            or atom.GetTotalValence() != 5
            or _order(mol, root, blocked) != 1):
        return None
    # Modified imido/nitrido acyl groups use the same acid infixes. Residual
    # OH groups are concatenated back onto that acyl core (P-67.1.4.1.1.5).
    if any(a.GetAtomicNum() == 7 and _order(mol, root, a.GetIdx()) in {2, 3}
           for a in atom.GetNeighbors()):
        from .branch_rules import collect_subtree
        atoms = frozenset(collect_subtree(mol, root, blocked))
        if blocked in atoms:
            return None
        rows = sorted(atoms)
        rw = Chem.RWMol(mol)
        for idx in sorted(set(range(mol.GetNumAtoms())) - atoms, reverse=True):
            rw.RemoveAtom(idx)
        pi = rows.index(root)
        cap_p = rw.GetAtomWithIdx(pi)
        cap_p.SetNumExplicitHs(atom.GetTotalNumHs()); cap_p.SetNoImplicit(True)
        oxygen = rw.AddAtom(Chem.Atom(8)); rw.AddBond(pi, oxygen, Chem.BondType.SINGLE)
        capped = _sanitize(rw, Chem)
        if capped is None:
            return None
        decision = modified_oxoacid_decision(capped, Chem)
        if decision is None or not decision.name.endswith('ic acid'):
            return None
        hydroxyls = [a for a in atom.GetNeighbors() if a.GetIdx() != blocked
                     and a.GetAtomicNum() == 8 and a.GetDegree() == 1
                     and _order(mol, root, a.GetIdx()) == 1 and a.GetTotalNumHs() == 1]
        word = decision.name[:-len('ic acid')] + 'oyl'
        if hydroxyls:
            core = word if word.isalpha() else enclose_name_unit(word)
            word = (MULT[len(hydroxyls)] if len(hydroxyls) > 1 else '') + 'hydroxy' + core
        return word, atoms
    oxo, halides, atoms = [], [], {root}
    for neighbor in atom.GetNeighbors():
        idx = neighbor.GetIdx()
        if idx == blocked:
            continue
        if (neighbor.GetIsotope() or neighbor.GetFormalCharge() or neighbor.GetNumRadicalElectrons()
                or neighbor.GetDegree() != 1):
            return None
        if neighbor.GetAtomicNum() == 8 and _order(mol, root, idx) == 2:
            oxo.append(idx)
        elif neighbor.GetAtomicNum() in _HALOGEN and _order(mol, root, idx) == 1:
            halides.append(neighbor.GetAtomicNum())
        else:
            return None
        atoms.add(idx)
    if len(oxo) != 1 or len(halides) != 2:
        return None
    return _infix_name('phosphor', [_HALOGEN[e][1] for e in halides], 'oyl'), frozenset(atoms)


def carbon_acid_with_phosphorohalide(mol, Chem):
    """Restore a mapped P acid-derived prefix on the senior carbon acid.

    A private terminal-F proposal supplies the ordinary acyl-halide or acid
    parent. This initial grammar is a carbon path bearing one acid function
    and one modified P unit, so neither other prefixes nor a ring numbering
    tie can change citation order when the prefix is restored.
    """
    if not _plain(mol, Chem, {6, 7, 8, 9, 15, 16, 17, 35, 53}):
        return None
    ps = [a for a in iter_atoms(mol) if a.GetAtomicNum() == 15]
    if len(ps) > 1 or any(a.IsInRing() for a in iter_atoms(mol)):
        return None
    proposals = []
    # An imido N bonded to a carbon framework may itself be the root of a
    # sulfur acid-halide prefix. The same ordinary-acid proposal keeps that
    # carbon acid senior without inventing a substituted methane parent.
    for p in iter_atoms(mol):
        if p.GetAtomicNum() not in {7, 15} or p.IsInRing():
            continue
        for host in p.GetNeighbors():
            if host.GetAtomicNum() != 6 or _order(mol, p.GetIdx(), host.GetIdx()) != 1:
                continue
            prefix = branch_prefix(mol, Chem, p.GetIdx(), host.GetIdx()) if p.GetAtomicNum() == 15 else None
            if prefix is None and p.GetAtomicNum() == 7 and any(
                    a.GetAtomicNum() == 16 and _order(mol, p.GetIdx(), a.GetIdx()) == 2 for a in p.GetNeighbors()):
                text, owned = mapped_branch(mol, Chem, p.GetIdx(), host.GetIdx())
                prefix = (text, owned) if text else None
            if prefix:
                proposals.append((p, host.GetIdx(), *prefix))
    for p, root, word, owned in proposals:
        decision = _carbon_acid_proposal(mol, Chem, p, root, word, owned)
        if decision is not None:
            return decision
    return None


def _carbon_acid_proposal(mol, Chem, p, root, word, owned):
    owned = frozenset(owned)
    carbon = {a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() == 6}
    carbon -= owned
    if root not in carbon:
        return None
    if (any(sum(n.GetIdx() in carbon for n in mol.GetAtomWithIdx(i).GetNeighbors()) > 2 for i in carbon)
            or sum(b.GetBeginAtomIdx() in carbon and b.GetEndAtomIdx() in carbon for b in iter_bonds(mol)) != len(carbon) - 1):
        return None
    # The P ligand is at an end of the chosen complete carbon path. A direct
    # P-C(=O) linkage belongs to a different carbonic/parent-hydride rule.
    if any(_order(mol, root, n.GetIdx()) != 1 for n in mol.GetAtomWithIdx(root).GetNeighbors()):
        return None
    acid_sites = []
    seen = set(owned) | carbon
    for idx in carbon:
        others = [a for a in mol.GetAtomWithIdx(idx).GetNeighbors()
                  if a.GetIdx() not in carbon and a.GetIdx() != p.GetIdx()]
        if not others:
            continue
        oxo = [a for a in others if a.GetAtomicNum() == 8 and a.GetDegree() == 1 and _order(mol, idx, a.GetIdx()) == 2]
        leaving = [a for a in others if a.GetDegree() == 1 and _order(mol, idx, a.GetIdx()) == 1
                   and (a.GetAtomicNum() in _HALOGEN or a.GetAtomicNum() == 8 and a.GetTotalNumHs() == 1)]
        if len(oxo) != 1 or len(leaving) != 1 or len(others) != 2:
            return None
        acid_sites.append(idx); seen.update(a.GetIdx() for a in others)
    if len(acid_sites) != 1 or seen != set(range(mol.GetNumAtoms())):
        return None
    rw = Chem.RWMol(mol)
    marker = rw.GetAtomWithIdx(p.GetIdx())
    marker.SetAtomicNum(9); marker.SetNumExplicitHs(0); marker.SetNoImplicit(True)
    for idx in sorted(owned - {p.GetIdx()}, reverse=True):
        rw.RemoveAtom(idx)
    proxy = _sanitize(rw, Chem)
    if proxy is None:
        return None
    from .api import mol_to_iupac
    name = mol_to_iupac(proxy)
    if name.count('fluoro') != 1 or not name.endswith((' acid', ' chloride', ' bromide', ' iodide', ' fluoride')):
        return None
    replacement = (word if word.isalpha() and not word.startswith(('hydroxy', 'dihydroxy'))
                   else enclose_name_unit(word))
    name = name.replace('fluoro', replacement, 1)
    if len(carbon) == 2 and root != acid_sites[0] and name.startswith('2-'):
        name = name[2:]
    return _out(mol, name, acid_sites,
                'P-41/P-67.1.4.1.1.4/P-68.3.2.3.2.2',
                'substitutive' if name.endswith(' acid') else 'functional-class')


def phosphanylidene_sulfonamide_decision(mol, Chem):
    admission = Chem.Mol(mol)
    for atom in iter_atoms(admission):
        if atom.GetAtomicNum() == 6:
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    if not _plain(admission, Chem, {6, 7, 8, 9, 15, 16, 17, 34, 35, 52, 53}):
        return None
    ps = [a for a in iter_atoms(mol) if a.GetAtomicNum() == 15]
    sulfurs = [a for a in iter_atoms(mol) if a.GetAtomicNum() == 16 and not a.IsInRing()
               and sum(n.GetAtomicNum() == 8 and _order(mol, a.GetIdx(), n.GetIdx()) == 2 for n in a.GetNeighbors()) == 2]
    if len(ps) != 1 or len(sulfurs) != 1:
        return None
    p, sulfur = ps[0], sulfurs[0]
    if p.IsInRing() or p.GetTotalValence() not in {3, 5}:
        return None
    imido = [a for a in p.GetNeighbors() if a.GetAtomicNum() == 7 and _order(mol, p.GetIdx(), a.GetIdx()) == 2]
    if len(imido) != 1:
        return None
    nitrogen = imido[0]
    if nitrogen.GetDegree() != 2 or _order(mol, nitrogen.GetIdx(), sulfur.GetIdx()) != 1:
        return None
    so = [a for a in sulfur.GetNeighbors() if a.GetAtomicNum() == 8 and a.GetDegree() == 1
          and _order(mol, sulfur.GetIdx(), a.GetIdx()) == 2]
    sr = [a for a in sulfur.GetNeighbors() if a.GetAtomicNum() == 6 and _order(mol, sulfur.GetIdx(), a.GetIdx()) == 1]
    if len(so) != 2 or len(sr) != 1 or sulfur.GetDegree() != 4:
        return None
    sarm = _carbon_arm(mol, Chem, sr[0].GetIdx(), sulfur.GetIdx())
    if sarm is None:
        return None
    base_atoms = {sulfur.GetIdx(), nitrogen.GetIdx(), *(a.GetIdx() for a in so)} | set(sarm[1])
    used, rows = set(base_atoms) | {p.GetIdx()}, []
    for atom in p.GetNeighbors():
        idx = atom.GetIdx()
        if idx == nitrogen.GetIdx():
            continue
        if _order(mol, p.GetIdx(), idx) != 1:
            return None
        if atom.GetAtomicNum() == 6:
            arm = _carbon_arm(mol, Chem, idx, p.GetIdx())
            if arm is None:
                return None
            text, owned = arm
        elif atom.GetAtomicNum() == 7 and atom.GetTotalValence() == 3 and not atom.IsInRing():
            carbon = [a for a in atom.GetNeighbors() if a.GetIdx() != p.GetIdx()]
            if len(carbon) != 1 or atom.GetTotalNumHs() != 1:
                return None
            arm = _carbon_arm(mol, Chem, carbon[0].GetIdx(), idx)
            if arm is None:
                return None
            if any(mol.GetAtomWithIdx(i).GetAtomicNum() == 8 and any(
                    b.GetBondTypeAsDouble() == 2 for b in mol.GetAtomWithIdx(i).GetBonds()) for i in arm[1]):
                return None  # A carboxamide ligand needs its own seniority comparison.
            text, owned = mapped_branch(mol, Chem, idx, p.GetIdx())
            if not text or owned != arm[1] | {idx}:
                return None
            rings = [r for r in mol.GetRingInfo().AtomRings() if frozenset(r) == arm[1]]
            if len(arm[1]) == 6 and len(rings) == 1 and all(
                    mol.GetAtomWithIdx(i).GetIsAromatic() for i in arm[1]):
                text = 'anilino'
        else:
            return None
        if owned & used:
            return None
        rows.append(('', text)); used.update(owned)
    if used != set(range(mol.GetNumAtoms())) or len(rows) + p.GetTotalNumHs() != p.GetTotalValence() - 2:
        return None
    prefix = format_prefixes(rows, omit_locants=True, hierarchical_enclosures=True)
    descriptor = ('-λ' + str(p.GetTotalValence())) if p.GetTotalValence() != 3 else ''
    prefix += descriptor + ('-' if prefix or descriptor else '') + 'phosphanylidene'
    # Cap the original amide N after deleting its P ligand. Naming the capped
    # graph determines the sulfonamide parent without moving any output atoms.
    rw = Chem.RWMol(mol)
    cap = rw.GetAtomWithIdx(nitrogen.GetIdx()); cap.SetNumExplicitHs(2); cap.SetNoImplicit(True)
    for idx in sorted(set(range(mol.GetNumAtoms())) - base_atoms, reverse=True):
        rw.RemoveAtom(idx)
    base = _sanitize(rw, Chem)
    if base is None:
        return None
    from .api import mol_to_iupac
    name = mol_to_iupac(base)
    if not name.endswith('sulfonamide'):
        return None
    # Reuse the selected capped parent's maps, then decorate the original
    # molecule with the extra N prefix. This preserves citation order and
    # recalculates carbon stereodescriptors in their original graph context.
    from .complete_functional_parent import functional_parent_candidates
    from .component_mapping import MappedComponent, decorate_component
    back = sorted(base_atoms)
    group = frozenset(range(mol.GetNumAtoms())) - base_atoms
    answers = set()
    for record in functional_parent_candidates(base, Chem, include_single=True):
        if record.name != name or record.construction is None:
            continue
        component = record.construction
        core = frozenset(back[i] for i in component.atoms)
        ownership = tuple((back[root], back[host], frozenset(back[i] for i in atoms), text)
                          for root, host, atoms, text in component.prefix_branch_ownership)
        ownership += ((p.GetIdx(), nitrogen.GetIdx(), group, prefix),)
        mapped = MappedComponent(component.parent,
            tuple((back[i], loc) for i, loc in component.atom_locants), core,
            prefixes=component.prefixes + (('N', prefix),),
            embedded_atoms=frozenset(back[i] for i in component.embedded_atoms) | group,
            systematic_parent=component.systematic_parent, nitrogen_site_locants=True,
            prefix_branch_ownership=ownership)
        rendered = decorate_component(mol, Chem, mapped)
        if rendered:
            answers.add(rendered)
    return (_out(mol, answers.pop(), (sulfur.GetIdx(),), 'P-67.1.5.1')
            if len(answers) == 1 else None)


def select(mol, Chem, numbers):
    if numbers[8] >= 2 and (numbers[15] or numbers[7] and numbers[16]):
        found = carbon_acid_with_phosphorohalide(mol, Chem)
        if found is not None:
            return found
    if numbers[14] and numbers[7] and numbers[8] >= 2:
        found = cyanatosilicic_acid_decision(mol, Chem)
        if found is not None:
            return found
    if numbers[15] and numbers[16] and numbers[7] and numbers[8] >= 2:
        found = phosphanylidene_sulfonamide_decision(mol, Chem)
        if found is not None:
            return found
    if (numbers[15] or numbers[16] and numbers[7] or numbers[7] and numbers[8]
            and any(numbers[x] for x in _HALOGEN) or numbers[7] and numbers[8] and numbers[16]):
        return modified_oxoacid_decision(mol, Chem)
    return None
