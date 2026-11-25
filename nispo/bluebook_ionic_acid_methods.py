"""Bounded P-65--P-73 functional-class and ionic constructions.

All recognizers prove ownership of the original graph and decline when a
competing characteristic group, annotation, or unsupported linkage is present.
Private neutral/ordinary analogues are used only to select an existing parent.
"""
from dataclasses import dataclass

from .naming_methods import MethodDecision


def _plain(mol, Chem, allowed, charges=(0,)):
    return (mol.GetNumHeavyAtoms() <= 64 and len(Chem.GetMolFrags(mol)) == 1
            and not mol.GetStereoGroups()
            and all(a.GetAtomicNum() in allowed and a.GetFormalCharge() in charges
                    and not (a.GetIsotope() or a.GetNumRadicalElectrons()
                             or int(a.GetChiralTag()) or a.HasQuery()) for a in mol.GetAtoms())
            and all(not (int(b.GetStereo()) or int(b.GetBondDir()) or b.HasQuery())
                    for b in mol.GetBonds()))


def _out(mol, name, centers, rule, method='substitutive'):
    atoms = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    return MethodDecision(name, method, frozenset(centers), atoms, rule, (),
                          component_roles=(('source-functional-graph', atoms, ()),))


def _sanitize(copy, Chem):
    result = copy.GetMol()
    try:
        Chem.SanitizeMol(result)
    except (ValueError, RuntimeError):
        return None
    return result


def phosphono_carboxylic_acid_decision(mol, Chem):
    """P-42: an unsubstituted phosphono group is subordinate to COOH."""
    if not _plain(mol, Chem, {6, 7, 8, 15}):
        return None
    # Amino is subordinate to the carboxylic acid and alphabetizes before
    # either the private fluoro marker or the final phosphono prefix.
    nitrogens = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    if len(nitrogens) > 1 or any(a.GetDegree() != 1 or a.GetTotalNumHs() != 2
                                or a.GetNeighbors()[0].GetAtomicNum() != 6
                                or mol.GetBondBetweenAtoms(a.GetIdx(), a.GetNeighbors()[0].GetIdx()).GetBondTypeAsDouble() != 1
                                for a in nitrogens):
        return None
    phosphorus = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 15]
    if len(phosphorus) != 1:
        return None
    p = phosphorus[0]
    oxygen = [a for a in p.GetNeighbors() if a.GetAtomicNum() == 8]
    carbon = [a for a in p.GetNeighbors() if a.GetAtomicNum() == 6]
    if len(oxygen) != 3 or len(carbon) != 1 or p.GetDegree() != 4:
        return None
    double = [a for a in oxygen if mol.GetBondBetweenAtoms(p.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() == 2 and a.GetDegree() == 1]
    hydroxy = [a for a in oxygen if mol.GetBondBetweenAtoms(p.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() == 1 and a.GetDegree() == 1 and a.GetTotalNumHs() == 1]
    if len(double) != 1 or len(hydroxy) != 2:
        return None
    # Exactly one ordinary acid and no other oxygen-bearing characteristic
    # group: the private F marks the phosphono attachment without changing
    # the carbon parent, its numbering, or its only prefix class.
    oxygen_ids = {a.GetIdx() for a in oxygen}
    others = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 8 and a.GetIdx() not in oxygen_ids]
    if len(others) != 2:
        return None
    acid_centers = []
    for c in mol.GetAtoms():
        if c.GetAtomicNum() != 6:
            continue
        attached = [o for o in others if mol.GetBondBetweenAtoms(c.GetIdx(), o.GetIdx())]
        if (len(attached) == 2 and sorted(mol.GetBondBetweenAtoms(c.GetIdx(), o.GetIdx()).GetBondTypeAsDouble() for o in attached) == [1, 2]
                and any(o.GetDegree() == 1 and o.GetTotalNumHs() == 1 for o in attached)):
            acid_centers.append(c.GetIdx())
    if len(acid_centers) != 1 or carbon[0].GetIdx() == acid_centers[0]:
        return None
    copy = Chem.RWMol(mol)
    copy.GetAtomWithIdx(p.GetIdx()).SetAtomicNum(9)
    for idx in sorted((o.GetIdx() for o in oxygen), reverse=True):
        copy.RemoveAtom(idx)
    marker = _sanitize(copy, Chem)
    if marker is None:
        return None
    from .api import mol_to_iupac
    name = mol_to_iupac(marker)
    if not name.endswith(' acid') or name.count('fluoro') != 1:
        return None
    return _out(mol, name.replace('fluoro', 'phosphono'), acid_centers, 'P-42/P-67.1.5.1')


def thioamide_decision(mol, Chem):
    """Primary mono/dithioamides; retain heterocyclic amide parent selection."""
    if not _plain(mol, Chem, {6, 7, 8, 9, 16, 17, 35, 53}):
        return None
    sulfurs = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 16]
    if len(sulfurs) not in {1, 2}:
        return None
    sites = []
    for sulfur in sulfurs:
        if sulfur.GetDegree() != 1:
            return None
        c = sulfur.GetNeighbors()[0]
        if c.GetAtomicNum() != 6 or mol.GetBondBetweenAtoms(c.GetIdx(), sulfur.GetIdx()).GetBondTypeAsDouble() != 2:
            return None
        nitrogens = [n for n in c.GetNeighbors() if n.GetAtomicNum() == 7 and n.GetDegree() == 1
                     and n.GetTotalNumHs() == 2 and mol.GetBondBetweenAtoms(c.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 1]
        if len(nitrogens) != 1 or c.IsInRing():
            return None
        sites.append((c.GetIdx(), sulfur.GetIdx(), nitrogens[0].GetIdx()))
    if any(a.GetAtomicNum() == 8 and not a.IsInRing() for a in mol.GetAtoms()):
        return None
    if any(a.GetAtomicNum() == 7 and not a.IsInRing() and a.GetIdx() not in {s[2] for s in sites} for a in mol.GetAtoms()):
        return None
    copy = Chem.RWMol(mol)
    for _, sulfur, _ in sites:
        copy.GetAtomWithIdx(sulfur).SetAtomicNum(8)
    analogue = _sanitize(copy, Chem)
    if analogue is None:
        return None
    from .api import mol_to_iupac
    ordinary = mol_to_iupac(analogue)
    if len(sites) == 1:
        if ordinary == 'formamide':
            name = 'methanethioamide'
        elif ordinary == 'acetamide':
            name = 'ethanethioamide'
        elif ordinary.endswith('benzamide'):
            name = ordinary[:-len('benzamide')] + 'benzenecarbothioamide'
        elif ordinary.endswith('carboxamide'):
            name = ordinary[:-len('carboxamide')] + 'carbothioamide'
        elif ordinary.endswith('amide') and not ordinary.endswith('diamide'):
            # Restoring the elided e is a parent-hydride operation. A
            # functional-parent phrase such as a carbonohalidic amide has
            # another suffix grammar and must not become "ic ethioamide".
            if not ordinary[:-len('amide')].endswith(('an', 'en', 'yn')):
                return None
            name = ordinary[:-len('amide')] + 'ethioamide'
        else:
            return None
    else:
        if ordinary.endswith('dicarboxamide'):
            name = ordinary[:-len('dicarboxamide')] + 'dicarbothioamide'
        elif ordinary.endswith('diamide'):
            name = ordinary[:-len('diamide')] + 'dithioamide'
        else:
            return None
    return _out(mol, name, (s[0] for s in sites), 'P-66.1.4.1.1/P-41', 'functional-replacement')


def imidothioic_acid_decision(mol, Chem):
    """Replace =NH and -SH on a private acid for parent selection."""
    if not _plain(mol, Chem, {6, 7, 16, 9, 17, 35, 53}):
        return None
    sulfurs = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 16]
    nitrogens = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    if len(sulfurs) != 1 or len(nitrogens) != 1:
        return None
    s, n = sulfurs[0], nitrogens[0]
    if (s.GetDegree() != 1 or s.GetTotalNumHs() != 1 or n.GetDegree() != 1
            or n.GetTotalNumHs() != 1 or s.GetNeighbors()[0].GetIdx() != n.GetNeighbors()[0].GetIdx()):
        return None
    c = s.GetNeighbors()[0]
    if (c.GetAtomicNum() != 6 or c.IsInRing()
            or mol.GetBondBetweenAtoms(c.GetIdx(), s.GetIdx()).GetBondTypeAsDouble() != 1
            or mol.GetBondBetweenAtoms(c.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() != 2):
        return None
    if any(a.GetAtomicNum() not in {6, 9, 17, 35, 53} for a in c.GetNeighbors() if a.GetIdx() not in {s.GetIdx(), n.GetIdx()}):
        return None
    copy = Chem.RWMol(mol)
    for atom, h in ((n, 0), (s, 1)):
        replacement = copy.GetAtomWithIdx(atom.GetIdx())
        replacement.SetAtomicNum(8)
        replacement.SetNumExplicitHs(h)
        replacement.SetNoImplicit(True)
    acid = _sanitize(copy, Chem)
    if acid is None:
        return None
    from .api import mol_to_iupac
    from .imidic_acid_methods import _imidic_ending
    imidic = _imidic_ending(mol_to_iupac(acid), 1)
    if not imidic.endswith('imidic acid'):
        return None
    name = imidic[:-len('imidic acid')] + 'imidothioic acid'
    return _out(mol, name, (c.GetIdx(),), 'P-65.1.5.1/P-41', 'functional-replacement')


def symmetric_modified_anhydride_decision(mol, Chem):
    """Symmetrical C(=chalcogen)-O-C(=chalcogen) anhydrides."""
    if not _plain(mol, Chem, {6, 8, 9, 16, 17, 34, 35, 52, 53}):
        return None
    choices = []
    for bridge in mol.GetAtoms():
        if bridge.GetAtomicNum() != 8 or bridge.IsInRing() or bridge.GetDegree() != 2:
            continue
        left, right = bridge.GetNeighbors()
        if left.GetAtomicNum() != 6 or right.GetAtomicNum() != 6:
            continue
        if any(mol.GetBondBetweenAtoms(bridge.GetIdx(), c.GetIdx()).GetBondTypeAsDouble() != 1 for c in (left, right)):
            continue
        doubles = []
        for c in (left, right):
            terminal = [n for n in c.GetNeighbors() if n.GetAtomicNum() in {8, 16, 34, 52}
                        and n.GetDegree() == 1 and mol.GetBondBetweenAtoms(c.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 2]
            if len(terminal) != 1:
                break
            doubles.append(terminal[0])
        if len(doubles) != 2 or all(a.GetAtomicNum() == 8 for a in doubles):
            continue  # ordinary anhydrides already have a shared writer
        if any(a.GetAtomicNum() in {8, 16, 34, 52} and a.GetIdx() not in
               {bridge.GetIdx(), *(x.GetIdx() for x in doubles)} for a in mol.GetAtoms()):
            continue
        copy = Chem.RWMol(mol)
        copy.RemoveBond(left.GetIdx(), bridge.GetIdx())
        new_o = copy.AddAtom(Chem.Atom(8))
        copy.AddBond(left.GetIdx(), new_o, Chem.BondType.SINGLE)
        acid_pair = _sanitize(copy, Chem)
        if acid_pair is None:
            continue
        frags = Chem.GetMolFrags(acid_pair, asMols=True, sanitizeFrags=True)
        if len(frags) != 2 or Chem.MolToSmiles(frags[0]) != Chem.MolToSmiles(frags[1]):
            continue
        from .api import mol_to_iupac
        acid = mol_to_iupac(frags[0])
        # The acid H's element locant is not needed for the O-bridged product.
        import re
        stem = re.sub(r' (?:O|S|Se|Te)-acid$', '', acid)
        if stem == acid and acid.endswith(' acid'):
            stem = acid[:-5]
        if stem == acid:
            continue
        choices.append(_out(mol, stem + ' anhydride', (left.GetIdx(), right.GetIdx()), 'P-65.7.1', 'functional-class'))
    return choices[0] if len(choices) == 1 else None


def sulfur_cyanide_decision(mol, Chem):
    """P-66.5.1.3.2 sulfonyl/sulfinyl (and Se/Te analogue) cyanides."""
    if not _plain(mol, Chem, {6, 7, 8, 16, 34, 52}):
        return None
    hetero = [a for a in mol.GetAtoms() if a.GetAtomicNum() in {16, 34, 52}]
    nitrogens = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    if len(hetero) != 1 or len(nitrogens) != 1:
        return None
    center, n = hetero[0], nitrogens[0]
    if n.GetDegree() != 1:
        return None
    cyano = n.GetNeighbors()[0]
    if (cyano.GetAtomicNum() != 6 or cyano.GetDegree() != 2
            or mol.GetBondBetweenAtoms(n.GetIdx(), cyano.GetIdx()).GetBondTypeAsDouble() != 3
            or mol.GetBondBetweenAtoms(center.GetIdx(), cyano.GetIdx()) is None
            or mol.GetBondBetweenAtoms(center.GetIdx(), cyano.GetIdx()).GetBondTypeAsDouble() != 1):
        return None
    oxos = [a for a in center.GetNeighbors() if a.GetAtomicNum() == 8 and a.GetDegree() == 1
            and mol.GetBondBetweenAtoms(center.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() == 2]
    roots = [a for a in center.GetNeighbors() if a.GetAtomicNum() == 6 and a.GetIdx() != cyano.GetIdx()]
    if len(oxos) not in {1, 2} or len(roots) != 1 or center.GetDegree() != len(oxos) + 2:
        return None
    if sum(a.GetAtomicNum() == 8 for a in mol.GetAtoms()) != len(oxos):
        return None
    copy = Chem.RWMol(mol)
    copy.GetAtomWithIdx(center.GetIdx()).SetAtomicNum(16)
    copy.RemoveBond(center.GetIdx(), cyano.GetIdx())
    new_o = copy.AddAtom(Chem.Atom(8))
    copy.AddBond(center.GetIdx(), new_o, Chem.BondType.SINGLE)
    for idx in sorted((n.GetIdx(), cyano.GetIdx()), reverse=True):
        copy.RemoveAtom(idx)
    acid = _sanitize(copy, Chem)
    if acid is None:
        return None
    from .api import mol_to_iupac
    name = mol_to_iupac(acid)
    ending = 'sulfonic acid' if len(oxos) == 2 else 'sulfinic acid'
    if not name.endswith(ending):
        return None
    stem = name[:-len('ic acid')] + 'yl'
    if center.GetAtomicNum() == 34:
        stem = stem.replace('sulfonyl', 'selenonyl').replace('sulfinyl', 'seleninyl')
    elif center.GetAtomicNum() == 52:
        stem = stem.replace('sulfonyl', 'telluronyl').replace('sulfinyl', 'tellurinyl')
    return _out(mol, stem + ' cyanide', (center.GetIdx(),), 'P-66.5.1.3.2', 'functional-class')


def nitrogen_oxoester_decision(mol, Chem):
    """Carbon-only esters of nitric and nitrous acid."""
    if not _plain(mol, Chem, {6, 7, 8, 16, 34, 52}, charges=(-1, 0, 1)):
        return None
    ns = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    if len(ns) != 1 or sum(a.GetFormalCharge() for a in mol.GetAtoms()) != 0:
        return None
    n = ns[0]
    ligands = [a for a in n.GetNeighbors() if a.GetAtomicNum() in {8, 16, 34, 52}]
    if len(ligands) not in {2, 3} or n.GetDegree() != len(ligands):
        return None
    bridges = [a for a in ligands if a.GetDegree() == 2 and a.GetFormalCharge() == 0
               and mol.GetBondBetweenAtoms(n.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() == 1]
    terminal = [a for a in ligands if a.GetDegree() == 1 and a.GetAtomicNum() == 8]
    if len(bridges) != 1 or len(terminal) != len(ligands)-1:
        return None
    doubles = [a for a in terminal if mol.GetBondBetweenAtoms(n.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() == 2 and a.GetFormalCharge() == 0]
    negatives = [a for a in terminal if mol.GetBondBetweenAtoms(n.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() == 1 and a.GetFormalCharge() == -1]
    if len(doubles) != 1 or len(negatives) != len(ligands)-2 or n.GetFormalCharge() != len(negatives):
        return None
    bridge = bridges[0]
    roots = [a for a in bridge.GetNeighbors() if a.GetIdx() != n.GetIdx() and a.GetAtomicNum() == 6]
    if len(roots) != 1:
        return None
    from .phosphorus_ester_methods import _carbon_component
    from .component_mapping import mapped_branch
    group = _carbon_component(mol, roots[0].GetIdx(), bridge.GetIdx())
    if group is None or set(group) | {n.GetIdx(), *(a.GetIdx() for a in ligands)} != set(range(mol.GetNumAtoms())):
        return None
    word, mapped = mapped_branch(mol, Chem, roots[0].GetIdx(), bridge.GetIdx())
    if not word or frozenset(mapped) != group:
        return None
    replacement = {8: '', 16: 'thio', 34: 'seleno', 52: 'telluro'}[bridge.GetAtomicNum()]
    if replacement:
        word = bridge.GetSymbol() + '-' + word
    return _out(mol, word + ' ' + replacement + ('nitrate' if len(ligands) == 3 else 'nitrite'),
                (n.GetIdx(),), 'P-67.1.3.2', 'functional-class')


def pnictogen_chalcogen_ester_decision(mol, Chem):
    """One chalcogen replacement in fully/partially esterified P/As acids.

    Reuse the proven ordinary phosphorus ester mapping on a private analogue;
    each printed element locant is bound to the original linking atom.
    """
    if not _plain(mol, Chem, {6, 8, 15, 16, 33, 34, 52}):
        return None
    centers = [a for a in mol.GetAtoms() if a.GetAtomicNum() in {15, 33}]
    replaced = [a for a in mol.GetAtoms() if a.GetAtomicNum() in {16, 34, 52}]
    if len(centers) != 1 or len(replaced) != 1:
        return None
    center, replacement = centers[0], replaced[0]
    if mol.GetBondBetweenAtoms(center.GetIdx(), replacement.GetIdx()) is None:
        return None
    copy = Chem.RWMol(mol)
    copy.GetAtomWithIdx(center.GetIdx()).SetAtomicNum(15)
    copy.GetAtomWithIdx(replacement.GetIdx()).SetAtomicNum(8)
    ordinary = _sanitize(copy, Chem)
    if ordinary is None:
        return None
    from .phosphorus_ester_methods import _recognize, _acid_core_parts, _organyl_words
    site = _recognize(ordinary, Chem)
    if site is None:
        return None
    # Leave multi-hydrogen mixed-chalcogen positions to a writer that can
    # prove which acid hydrogen occupies which chalcogen.
    if replacement.GetIdx() in site['hydroxyls']:
        return None
    core = _acid_core_parts(ordinary, Chem, site)
    if core is None:
        from .functional_replacement_acids import _organyl_units
        seen = {center.GetIdx()}
        prefix = _organyl_units(ordinary, Chem, center.GetIdx(), site['p_roots'], seen)
        if prefix is None:
            return None
    else:
        prefix = core[0]
    endings = {'phosphoric acid': 'phosphoro', 'phosphonic acid': 'phosphono',
               'phosphinic acid': 'phosphino'}
    stem = endings[site['acid_kind']]
    if center.GetAtomicNum() == 33:
        stem = stem.replace('phosphor', 'arsor').replace('phosphon', 'arson').replace('phosphin', 'arsin')
    infix = {16: 'thio', 34: 'seleno', 52: 'telluro'}[replacement.GetAtomicNum()]
    words = _organyl_words(ordinary, Chem, site)
    if not words:
        return None
    from .formatting import prefix_alphabetization_key, multiply_component
    from .substituent_multiplication import terminally_substituted_carbon_component
    components = []
    for _, _word, rows in words:
        # The ordinary analogue groups identical organyls without regard to
        # linking element. Split those groups again on the original graph.
        buckets = {}
        for row in rows:
            number = mol.GetAtomWithIdx(row['oxygen']).GetAtomicNum()
            buckets.setdefault(number, []).append(row)
        for group in buckets.values():
            link = mol.GetAtomWithIdx(group[0]['oxygen'])
            word = group[0]['ordinary_text']
            if len(group) > 1:
                nature = ('compound' if terminally_substituted_carbon_component(
                    ordinary, group[0]['atoms']) else None)
                word = multiply_component(word, len(group), nature=nature)
                if not word:
                    return None
            elif any(x in word for x in '-()'):
                word = '(' + word + ')'
            locants = ','.join([link.GetSymbol()] * len(group))
            components.append((prefix_alphabetization_key(group[0]['ordinary_text']), locants + '-' + word))
    hydrogen = {0: '', 1: 'hydrogen', 2: 'dihydrogen', 3: 'trihydrogen'}[len(site['hydroxyls'])]
    text = [word for _, word in sorted(components)]
    if hydrogen:
        text.append(hydrogen)
    text.append(prefix + stem + infix + 'ate')
    return _out(mol, ' '.join(text), (center.GetIdx(),), 'P-67.1.3.2', 'functional-class')


@dataclass(frozen=True, slots=True)
class SourceIon:
    name: str
    atoms: frozenset[int]
    charge: int
    rule: str
    substituted: bool = False


def _ion(mol, name, rule, substituted=False):
    return SourceIon(name, frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1),
                     sum(a.GetFormalCharge() for a in mol.GetAtoms()), rule, substituted)


def diazonium_ion(mol, Chem):
    """Unsubstituted simple carbon chains and monocyclic diazonium ions."""
    if not _plain(mol, Chem, {6, 7, 8}, charges=(0, 1)):
        return None
    nitrogens = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    if not nitrogens or len(nitrogens) % 2:
        return None
    hosts, owned = [], set()
    for n in nitrogens:
        if n.GetFormalCharge() != 1 or n.GetDegree() != 2:
            continue
        terminal = [a for a in n.GetNeighbors() if a.GetAtomicNum() == 7 and a.GetDegree() == 1
                    and a.GetFormalCharge() == 0 and mol.GetBondBetweenAtoms(n.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() == 3]
        carbon = [a for a in n.GetNeighbors() if a.GetAtomicNum() == 6
                  and mol.GetBondBetweenAtoms(n.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() == 1]
        if len(terminal) != 1 or len(carbon) != 1:
            return None
        hosts.append(carbon[0].GetIdx())
        owned.update((n.GetIdx(), terminal[0].GetIdx()))
    if len(owned) != len(nitrogens) or sum(a.GetFormalCharge() for a in mol.GetAtoms()) != len(hosts):
        return None
    carbons = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6}
    if not carbons or len(carbons) > 10:
        return None
    oxo_hosts = []
    for oxygen in (a for a in mol.GetAtoms() if a.GetAtomicNum() == 8):
        if oxygen.GetDegree() != 1:
            return None
        host = oxygen.GetNeighbors()[0]
        if (host.GetIdx() not in carbons or oxygen.GetFormalCharge()
                or mol.GetBondBetweenAtoms(host.GetIdx(), oxygen.GetIdx()).GetBondTypeAsDouble() != 2):
            return None
        oxo_hosts.append(host.GetIdx())
    if len(oxo_hosts) > 4:
        return None
    adj = {i: [n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in carbons] for i in carbons}
    edge_count = sum(map(len, adj.values())) // 2
    rings = mol.GetRingInfo().AtomRings()
    from .constants import ROOTS
    if rings:
        if len(rings) != 1 or frozenset(rings[0]) != carbons or any(len(adj[i]) != 2 for i in carbons):
            return None
        aromatic = all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in carbons)
        if aromatic and len(carbons) == 6:
            parent = 'benzene'
        elif (not aromatic and len(carbons) >= 3 and all(
                mol.GetBondBetweenAtoms(i, j).GetBondTypeAsDouble() == 1 for i in carbons for j in adj[i])):
            parent = 'cyclo' + ROOTS[len(carbons)] + 'ane'
        else:
            return None
        paths = []
        for start in carbons:
            for first in adj[start]:
                path, prev, cur = [start], start, first
                while cur != start:
                    if cur in path:
                        return None
                    path.append(cur)
                    nxt = [x for x in adj[cur] if x != prev]
                    if len(nxt) != 1:
                        return None
                    prev, cur = cur, nxt[0]
                paths.append(path)
    else:
        if edge_count != len(carbons)-1:
            return None
        if any(b.GetBondTypeAsDouble() != 1 for b in mol.GetBonds()
               if b.GetBeginAtomIdx() in carbons and b.GetEndAtomIdx() in carbons):
            return None
        if any(len(adj[i]) > 2 for i in carbons):
            if len(hosts) != 1 or oxo_hosts:
                return None
            return _branched_diazonium_ion(mol, Chem, adj, hosts[0], owned)
        ends = [i for i in carbons if len(adj[i]) <= 1]
        if len(carbons) > 1 and len(ends) != 2:
            return None
        paths = []
        for start in ends:
            path, prev, cur = [], None, start
            while cur is not None:
                path.append(cur)
                nxt = [x for x in adj[cur] if x != prev]
                prev, cur = cur, (nxt[0] if nxt else None)
            if len(path) != len(carbons):
                return None
            paths.append(path)
        parent = ROOTS[len(carbons)] + 'ane'
    if not paths or len(hosts) > 4:
        return None
    locants, oxo_locants = min((tuple(sorted(path.index(h)+1 for h in hosts)),
                               tuple(sorted(path.index(h)+1 for h in oxo_hosts))) for path in paths)
    if len(hosts) == 1:
        suffix = 'diazonium' if len(carbons) <= 2 or rings else '-' + str(locants[0]) + '-diazonium'
    else:
        multiplier = {2: 'bis', 3: 'tris', 4: 'tetrakis'}[len(hosts)]
        suffix = '-' + ','.join(map(str, locants)) + '-' + multiplier + '(diazonium)'
    if oxo_locants:
        multiplier = {1: '', 2: 'di', 3: 'tri', 4: 'tetra'}[len(oxo_locants)]
        parent = ','.join(map(str, oxo_locants)) + '-' + multiplier + 'oxo' + parent
    return _ion(mol, parent + suffix, 'P-73.2.2.3', len(hosts) > 1 or bool(oxo_locants))


def _branched_diazonium_ion(mol, Chem, adj, host, nitrogen_ids):
    """Reuse a mapped fluorohydrocarbon only when its parent numbering is proved."""
    import re

    # Enumerate tree-end paths. The marker's longest parent must contain the
    # charged suffix site; its locant must also be the minimum over all such
    # longest paths, so ordinary prefix numbering cannot override P-73.
    ends = [idx for idx, neighbors in adj.items() if len(neighbors) == 1]
    paths = []
    for start in ends:
        stack = [(start, None, (start,))]
        while stack:
            current, previous, path = stack.pop()
            if current != start and len(adj[current]) == 1:
                paths.append(path)
            for nxt in adj[current]:
                if nxt != previous:
                    stack.append((nxt, current, path + (nxt,)))
    if not paths:
        return None
    longest = max(map(len, paths))
    site_paths = [path for path in paths if len(path) == longest and host in path]
    if not site_paths:
        return None
    suffix_locant = min(path.index(host) + 1 for path in site_paths)
    central = next((i for i in nitrogen_ids if mol.GetAtomWithIdx(i).GetFormalCharge() == 1), None)
    terminal = next((i for i in nitrogen_ids if i != central), None)
    if central is None or terminal is None:
        return None
    copy = Chem.RWMol(mol)
    marker = copy.GetAtomWithIdx(central)
    marker.SetAtomicNum(9)
    marker.SetFormalCharge(0)
    marker.SetNoImplicit(True)
    copy.RemoveAtom(terminal)
    analogue = _sanitize(copy, Chem)
    if analogue is None:
        return None
    from .api import mol_to_iupac
    fluorinated = mol_to_iupac(analogue)
    match = re.search(r'(?<![0-9])([0-9]+)-fluoro(-?)', fluorinated)
    if match is None or int(match.group(1)) != suffix_locant:
        return None
    start, end = match.span()
    # A last prefix is joined directly to the parent, so remove the hyphen
    # formerly separating it from the preceding prefix.
    if not match.group(2) and start and fluorinated[start - 1] == '-':
        start -= 1
    parent = fluorinated[:start] + fluorinated[end:]
    if not parent or 'fluoro' in parent or not parent.endswith('ane'):
        return None
    return _ion(mol, parent + '-' + str(suffix_locant) + '-diazonium', 'P-73.2.2.3', True)


def acid_derived_cation_ion(mol, Chem):
    """Monovalent carbon acylium ions of simple carboxylic acids."""
    if not _plain(mol, Chem, {6, 8, 16, 34, 52}, charges=(0, 1)):
        return None
    if sum(a.GetFormalCharge() for a in mol.GetAtoms()) != 1:
        return None
    hetero = [a for a in mol.GetAtoms() if a.GetAtomicNum() != 6]
    if len(hetero) != 1:
        return None
    terminal = hetero[0]
    if terminal.GetDegree() != 1:
        return None
    c = terminal.GetNeighbors()[0]
    if c.GetAtomicNum() != 6 or c.IsInRing():
        return None
    bond = mol.GetBondBetweenAtoms(c.GetIdx(), terminal.GetIdx())
    if not ((c.GetFormalCharge() == 1 and terminal.GetFormalCharge() == 0 and bond.GetBondTypeAsDouble() == 2)
            or (c.GetFormalCharge() == 0 and terminal.GetFormalCharge() == 1 and bond.GetBondTypeAsDouble() == 3)):
        return None
    copy = Chem.RWMol(mol)
    copy.GetAtomWithIdx(c.GetIdx()).SetFormalCharge(0)
    copy.GetAtomWithIdx(terminal.GetIdx()).SetFormalCharge(0)
    copy.GetBondBetweenAtoms(c.GetIdx(), terminal.GetIdx()).SetBondType(Chem.BondType.DOUBLE)
    o = copy.AddAtom(Chem.Atom(8))
    copy.AddBond(c.GetIdx(), o, Chem.BondType.SINGLE)
    acid = _sanitize(copy, Chem)
    if acid is None:
        return None
    from .api import mol_to_iupac
    name = mol_to_iupac(acid)
    special = {'formic acid': 'formylium', 'acetic acid': 'acetylium', 'benzoic acid': 'benzoylium'}
    if name in special:
        name = special[name]
    elif name.endswith('carboxylic acid'):
        name = name[:-len('carboxylic acid')] + 'carbonylium'
    elif name.endswith('oic acid'):
        name = name[:-len('oic acid')] + 'oylium'
    else:
        import re
        match = re.fullmatch(r'(.+(?:thio|seleno|telluro))ic (?:O|S|Se|Te)-acid', name)
        if not match:
            return None
        name = match.group(1) + 'ylium'
    return _ion(mol, name, 'P-73.2.3.1')


def heteroacid_derived_cation_ion(mol, Chem):
    """Acylium ions from wholly dehydroxylated sulfur and P oxoacids."""
    if not _plain(mol, Chem, {6, 8, 15, 16}, charges=(0, 1, 2)):
        return None
    centers = [a for a in mol.GetAtoms() if a.GetAtomicNum() in {15, 16}]
    if len(centers) != 1:
        return None
    center = centers[0]
    charge = center.GetFormalCharge()
    if charge not in {1, 2} or sum(a.GetFormalCharge() for a in mol.GetAtoms()) != charge:
        return None
    roots = [a for a in center.GetNeighbors() if a.GetAtomicNum() == 6
             and mol.GetBondBetweenAtoms(center.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() == 1]
    oxygens = [a for a in center.GetNeighbors() if a.GetAtomicNum() == 8 and a.GetDegree() == 1
               and mol.GetBondBetweenAtoms(center.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() == 2]
    if (center.GetDegree() != len(roots) + len(oxygens)
            or sum(a.GetAtomicNum() == 8 for a in mol.GetAtoms()) != len(oxygens)):
        return None
    if center.GetAtomicNum() == 16:
        if charge != 1 or len(roots) != 1 or len(oxygens) not in {1, 2}:
            return None
    elif len(oxygens) != 1 or len(roots) + charge != 3:
        return None
    copy = Chem.RWMol(mol)
    copy.GetAtomWithIdx(center.GetIdx()).SetFormalCharge(0)
    for _ in range(charge):
        idx = copy.AddAtom(Chem.Atom(8))
        copy.AddBond(center.GetIdx(), idx, Chem.BondType.SINGLE)
    acid = _sanitize(copy, Chem)
    if acid is None:
        return None
    from .api import mol_to_iupac
    name = mol_to_iupac(acid)
    endings = {
        'sulfinic acid': 'sulfinylium', 'sulfonic acid': 'sulfonylium',
        'phosphinic acid': 'phosphinoylium', 'phosphonic acid': 'phosphonobis(ylium)',
    }
    for acid_ending, ion_ending in endings.items():
        if name.endswith(acid_ending):
            return _ion(mol, name[:-len(acid_ending)] + ion_ending, 'P-73.2.3.1', charge > 1)
    return None


def hydroxy_chalcogen_anion_ion(mol, Chem):
    """Uniform anions of carbon-only thiols, selenols, tellurols and peroxols."""
    if not _plain(mol, Chem, {6, 8, 16, 34, 52}, charges=(0, -1)):
        return None
    charged = [a for a in mol.GetAtoms() if a.GetFormalCharge() == -1]
    if not 1 <= len(charged) <= 4 or sum(a.GetFormalCharge() for a in mol.GetAtoms()) != -len(charged):
        return None
    if any(a.GetDegree() != 1 or a.GetAtomicNum() not in {8, 16, 34, 52} for a in charged):
        return None
    if len({a.GetAtomicNum() for a in charged}) != 1:
        return None
    site = charged[0]
    chalcogens = [a for a in mol.GetAtoms() if a.GetAtomicNum() in {8, 16, 34, 52}]
    if site.GetAtomicNum() == 8:
        bridges = [a.GetNeighbors()[0] for a in charged]
        if (len(chalcogens) != 2 * len(charged)
                or len({a.GetIdx() for a in bridges}) != len(charged)
                or any(a.GetAtomicNum() != 8 or a.GetFormalCharge() or a.GetDegree() != 2
                       or sum(n.GetAtomicNum() == 6 for n in a.GetNeighbors()) != 1 for a in bridges)):
            return None  # ordinary retained and polyol anions have existing writers
    elif (len(chalcogens) != len(charged)
          or any(a.GetNeighbors()[0].GetAtomicNum() != 6 for a in charged)):
        return None
    copy = Chem.RWMol(mol)
    for site in charged:
        atom = copy.GetAtomWithIdx(site.GetIdx())
        atom.SetFormalCharge(0)
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
    neutral = _sanitize(copy, Chem)
    if neutral is None:
        return None
    from .api import mol_to_iupac
    parent = mol_to_iupac(neutral)
    endings = {8: 'peroxol', 16: 'thiol', 34: 'selenol', 52: 'tellurol'}
    ending = endings[charged[0].GetAtomicNum()]
    if not parent.endswith(ending):
        return None
    if len(charged) == 1:
        name = parent + 'ate'
    else:
        neutral_multiplier, ion_multiplier = {2: ('di', 'bis'), 3: ('tri', 'tris'),
                                              4: ('tetra', 'tetrakis')}[len(charged)]
        suffix = neutral_multiplier + ending
        if not parent.endswith(suffix):
            return None
        name = parent[:-len(suffix)] + ion_multiplier + '(' + ending + 'ate)'
    return _ion(mol, name, 'P-72.2.2.2.2', len(charged) > 1)


def ionic_source_decision(mol, Chem):
    ion = (diazonium_ion(mol, Chem) or acid_derived_cation_ion(mol, Chem)
           or heteroacid_derived_cation_ion(mol, Chem) or hydroxy_chalcogen_anion_ion(mol, Chem))
    if ion is None:
        return None
    centers = [a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge()]
    return _out(mol, ion.name, centers, ion.rule)


def pnictogen_halogen_ester_decision(mol, Chem):
    """Mono-halogen functional replacement in P/As acid esters."""
    if not _plain(mol, Chem, {6, 8, 9, 15, 17, 33, 35, 53}):
        return None
    centers = [a for a in mol.GetAtoms() if a.GetAtomicNum() in {15, 33}]
    halogens = [a for a in mol.GetAtoms() if a.GetAtomicNum() in {9, 17, 35, 53}]
    if len(centers) != 1 or len(halogens) != 1:
        return None
    center, halogen = centers[0], halogens[0]
    bond = mol.GetBondBetweenAtoms(center.GetIdx(), halogen.GetIdx())
    if halogen.GetDegree() != 1 or bond is None or bond.GetBondTypeAsDouble() != 1:
        return None
    copy = Chem.RWMol(mol)
    copy.GetAtomWithIdx(center.GetIdx()).SetAtomicNum(15)
    cap = copy.GetAtomWithIdx(halogen.GetIdx())
    cap.SetAtomicNum(8)
    cap.SetNoImplicit(False)
    cap.SetNumExplicitHs(0)
    ordinary = _sanitize(copy, Chem)
    if ordinary is None:
        return None
    from .phosphorus_ester_methods import _recognize, _acid_core_parts, _organyl_words
    site = _recognize(ordinary, Chem)
    if site is None or halogen.GetIdx() not in site['hydroxyls']:
        return None
    core = _acid_core_parts(ordinary, Chem, site)
    if core is None:
        from .functional_replacement_acids import _organyl_units
        prefix = _organyl_units(ordinary, Chem, center.GetIdx(), site['p_roots'], {center.GetIdx()})
        if prefix is None:
            return None
    else:
        prefix = core[0]
    stem = {'phosphoric acid': 'phosphoro', 'phosphonic acid': 'phosphono',
            'phosphinic acid': 'phosphino'}[site['acid_kind']]
    if center.GetAtomicNum() == 33:
        stem = stem.replace('phosphor', 'arsor').replace('phosphon', 'arson').replace('phosphin', 'arsin')
    infix = {9: 'fluorid', 17: 'chlorid', 35: 'bromid', 53: 'iodid'}[halogen.GetAtomicNum()]
    words = _organyl_words(ordinary, Chem, site)
    if not words:
        return None
    hydrogen_count = len(site['hydroxyls']) - 1
    hydrogen = {0: '', 1: 'hydrogen', 2: 'dihydrogen'}[hydrogen_count]
    text = [word for _, word, _ in words]
    if hydrogen:
        text.append(hydrogen)
    text.append(prefix + stem + infix + 'ate')
    return _out(mol, ' '.join(text), (center.GetIdx(),), 'P-67.1.3.2', 'functional-class')


def hypohalite_ester_decision(mol, Chem, *, sites=None):
    """Cite the esterifying chalcogen and every terminal oxo of its halogen acid."""
    from .halogen_oxoacids import ADJECTIVES, carbon_branch, derivative_query, oxo_atoms, senior_organyl_group
    from .extended_inventory import selection_inventory
    if (mol.GetStereoGroups() or mol.GetNumAtoms() != mol.GetNumHeavyAtoms()
            or any(a.HasQuery() or a.GetNumRadicalElectrons() for a in mol.GetAtoms())):
        return None
    if sites is None:
        sites = mol.GetSubstructMatches(derivative_query(Chem))
    for x, y in sites:
        halogen, link = mol.GetAtomWithIdx(x), mol.GetAtomWithIdx(y)
        if (link.GetAtomicNum() not in (8, 16, 34, 52) or link.GetDegree() != 2
                or link.GetTotalValence() != 2 or link.GetFormalCharge() or link.GetIsotope()
                or link.IsInRing() or link.GetChiralTag() or halogen.GetIsotope()):
            continue
        oxos = oxo_atoms(halogen, y)
        if oxos is None or any(mol.GetAtomWithIdx(i).GetIsotope() for i in oxos):
            continue
        root = next(a.GetIdx() for a in link.GetNeighbors() if a.GetIdx() != x)
        if mol.GetBondBetweenAtoms(root, y).GetBondTypeAsDouble() != 1:
            continue
        carbon = mol.GetAtomWithIdx(root)
        if any(b.GetBondTypeAsDouble() == 2 and b.GetOtherAtom(carbon).GetAtomicNum()
               in (8, 16, 34, 52) for b in carbon.GetBonds()):
            if link.GetAtomicNum() == 8 and oxos:
                from .bluebook_hydride_assembly_methods import mixed_anhydride_decision
                return mixed_anhydride_decision(mol, Chem)
            continue
        word, owned = carbon_branch(mol, Chem, root, y)
        if not word or len(owned | oxos | {x, y}) != mol.GetNumAtoms():
            continue
        if (any(g.seniority[0] <= 9 for g in selection_inventory(mol).groups)
                or senior_organyl_group(mol, Chem, owned, 9)):
            continue
        adjective = ADJECTIVES[halogen.GetAtomicNum()][len(oxos)]
        anion = adjective[:-3] + 'ite' if adjective.endswith('ous') else adjective[:-2] + 'ate'
        chalcogen = {8: '', 16: 'thio', 34: 'seleno', 52: 'telluro'}[link.GetAtomicNum()]
        # In higher acids the unlocanted thio name can put sulfur on a terminal
        # double bond. S-/Se-/Te- binds the organyl to the esterifying atom.
        if oxos and chalcogen:
            from .formatting import enclose_name_unit
            if word[0].isdigit() or '(' in word:
                word = enclose_name_unit(word)
            word = link.GetSymbol() + '-' + word
        return _out(mol, word + ' ' + chalcogen + anion, (x,), 'P-61.3.2.2/P-67.1.3.2', 'functional-class')
    return None
