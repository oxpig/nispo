"""Graph-derived preferred functional classes from P-65, P-67, P-68, P-72.

Each producer proves whole-graph ownership and the absence of an unhandled
senior group. Annotated or competing chemistry outside that proof defers.
"""
from collections import Counter

from .molecule_iteration import iter_atoms
from .bluebook_lambda_amine_methods import _plain, _out, _sanitize
from .component_mapping import mapped_branch
from .constants import MULT, ROOTS
from .formatting import format_prefixes, enclose_name_unit

_HALIDES = {9: 'fluoride', 17: 'chloride', 35: 'bromide', 53: 'iodide'}
_SIMPLE_ACYLOXY = {'(' + acyl + ')oxy': acyl + 'oxy'
                   for acyl in {'formyl', 'acetyl', 'benzoyl'} |
                   {root + 'anoyl' for root in ROOTS.values()}}


def _order(mol, a, b):
    bond = mol.GetBondBetweenAtoms(a, b)
    return bond.GetBondTypeAsDouble() if bond else 0


def systematic_aceylene(mol, Chem):
    """Use a source-numbered mancude ace parent before a bridged fallback."""
    if (not _plain(mol, Chem, {6}) or not all(a.IsInRing() for a in mol.GetAtoms())
            or mol.GetRingInfo().NumRings() < 3):
        return None
    from .fusion_components import _systematic_fused_records
    for name, (smiles, _locants) in _systematic_fused_records().items():
        if not name.startswith('ace') or not name.endswith('ylene'):
            continue
        query = Chem.MolFromSmiles(smiles)
        if query is None or query.GetNumAtoms() != mol.GetNumAtoms():
            continue
        for match in mol.GetSubstructMatches(query, uniquify=False):
            if all(mol.GetAtomWithIdx(idx).GetTotalNumHs() == query.GetAtomWithIdx(i).GetTotalNumHs()
                   for i, idx in enumerate(match)):
                return _out(mol, name, (), 'P-25.1.2.7/P-52.2.4')
    return None


def ring_n_carboxylic_acid(mol, Chem):
    """Carry the ring's free-valence/added-H map into a carboxylic suffix."""
    if not _plain(mol, Chem, {6, 7, 8, 9, 17, 35, 53}):
        return None
    sites = []
    for n in mol.GetAtoms():
        if n.GetAtomicNum() != 7 or not n.IsInRing() or n.GetTotalValence() != 3:
            continue
        for c in n.GetNeighbors():
            if c.GetAtomicNum() != 6 or c.IsInRing() or c.GetDegree() != 3 or _order(mol, n.GetIdx(), c.GetIdx()) != 1:
                continue
            oxo = [a for a in c.GetNeighbors() if a.GetAtomicNum() == 8 and a.GetDegree() == 1
                   and _order(mol, c.GetIdx(), a.GetIdx()) == 2]
            oh = [a for a in c.GetNeighbors() if a.GetAtomicNum() == 8 and a.GetDegree() == 1
                  and a.GetTotalNumHs() == 1 and _order(mol, c.GetIdx(), a.GetIdx()) == 1]
            if len(oxo) == len(oh) == 1:
                sites.append((n.GetIdx(), c.GetIdx(), oxo[0].GetIdx(), oh[0].GetIdx()))
    if len(sites) != 1:
        return None
    n, carbon, oxo, oh = sites[0]
    scope = frozenset(range(mol.GetNumAtoms())) - {carbon, oxo, oh}
    from .mapped_ring_parents import rooted_prefix
    prefix, atoms = rooted_prefix(mol, Chem, n, carbon, scope)
    if not prefix or atoms != scope:
        return None
    import re
    match = re.fullmatch(r'(.+)-([0-9]+[a-z]?(?:\([0-9a-z,H]+\))?)-yl', prefix)
    if match is None:
        return None
    stem, location = match.groups()
    name = stem + 'e-' + location + '-carboxylic acid'
    return _out(mol, name, (carbon,), 'P-65.1.8.1/P-58.2.2/P-52.2.4')


def mixed_ring_carbonyl_chalcogen(mol, Chem):
    """Select the oxygen carbonyl and map subordinate chalcogen ylidenes."""
    if not _plain(mol, Chem, {6, 7, 8, 9, 16, 17, 34, 35, 52, 53}) or mol.GetRingInfo().NumRings() != 1:
        return None
    replacements, oxygen_carbonyls = [], []
    for atom in mol.GetAtoms():
        if atom.GetDegree() != 1 or atom.GetAtomicNum() not in {8, 16, 34, 52}:
            continue
        host = atom.GetNeighbors()[0]
        if host.GetAtomicNum() != 6 or not host.IsInRing() or _order(mol, host.GetIdx(), atom.GetIdx()) != 2:
            continue
        if atom.GetAtomicNum() == 8:
            oxygen_carbonyls.append(host.GetIdx())
        else:
            replacements.append((host.GetIdx(), atom.GetIdx(), atom.GetAtomicNum()))
    if not replacements or not oxygen_carbonyls:
        return None
    rw = Chem.RWMol(mol)
    for host, end, _element in replacements:
        marker = rw.GetAtomWithIdx(end)
        marker.SetAtomicNum(9)
        marker.SetNumExplicitHs(0)
        marker.SetNoImplicit(True)
        rw.GetBondBetweenAtoms(host, end).SetBondType(Chem.BondType.SINGLE)
        cap = rw.GetAtomWithIdx(host)
        if cap.GetNoImplicit():
            cap.SetNumExplicitHs(cap.GetNumExplicitHs() + 1)
    proxy = _sanitize(rw, Chem)
    if proxy is None:
        return None
    from .naming_context import MoleculeContext
    from .ring_state_parent import select_ring_state
    selection = select_ring_state(MoleculeContext(proxy, Chem))
    if selection is None or not selection.selected.name.endswith('one'):
        return None
    record = selection.selected
    if record.principal_centers != frozenset(oxygen_carbonyls) or record.construction is None:
        return None
    labels = dict(record.atom_locants)
    names = {16: 'sulfanylidene', 34: 'selanylidene', 52: 'tellanylidene'}
    substitutions = {(labels[host], 'fluoro'): names[element]
                     for host, _end, element in replacements}
    prefixes = tuple((loc, substitutions.get((loc, text), text))
                     for loc, text in record.construction.prefixes)
    if sum((loc, text) in substitutions for loc, text in record.construction.prefixes) != len(replacements):
        return None
    from dataclasses import replace
    from .component_mapping import decorate_component
    component = replace(record.construction, prefixes=prefixes,
        prefix_branch_ownership=tuple((root, host, atoms, substitutions.get((labels[host], text), text))
            for root, host, atoms, text in record.construction.prefix_branch_ownership))
    name = decorate_component(mol, Chem, component)
    if not name:
        return None
    return _out(mol, name, oxygen_carbonyls, 'P-64.6.1/P-64.6.2/P-22.2/P-58.2')


def carboxamide_above_urea(mol, Chem):
    """Keep a single ordinary carboxamide senior to a urea on its N arm."""
    if not _plain(mol, Chem, {6, 7, 8}):
        return None
    from .extended_inventory import selection_inventory
    inventory = selection_inventory(mol)
    if (not inventory.supported or len(inventory.principal) != 1
            or inventory.principal[0].kind != 'amide'
            or not any(group.kind == 'urea' for group in inventory.groups)):
        return None
    group = inventory.principal[0]
    center, nitrogen = group.center, group.ligand
    n = mol.GetAtomWithIdx(nitrogen)
    if n.IsInRing() or n.GetTotalValence() != 3:
        return None
    from .branch_rules import collect_subtree
    prefixes, removed = [], set()
    for root in n.GetNeighbors():
        if root.GetIdx() == center:
            continue
        scope = frozenset(collect_subtree(mol, root.GetIdx(), nitrogen))
        if center in scope or nitrogen in scope or scope & removed:
            return None
        text, atoms = mapped_branch(mol, Chem, root.GetIdx(), nitrogen)
        if not text or atoms != scope:
            return None
        prefixes.append(('N', text))
        removed.update(scope)
    if not removed:
        return None
    rw = Chem.RWMol(mol)
    cap = rw.GetAtomWithIdx(nitrogen)
    cap.SetNumExplicitHs(0)
    cap.SetNoImplicit(False)
    for index in sorted(removed, reverse=True):
        rw.RemoveAtom(index)
    probe = _sanitize(rw, Chem)
    if probe is None:
        return None
    from .api import mol_to_iupac
    import re
    ordinary = mol_to_iupac(probe)
    if ordinary not in {'acetamide', 'formamide', 'benzamide'} and not re.fullmatch(
            r'(?:eth|prop|but|pent|hex|hept|oct|non|dec)anamide', ordinary):
        return None
    name = format_prefixes(prefixes, hierarchical_enclosures=True) + ordinary
    return _out(mol, name, (center,), 'P-66.1.6.1.1.3/P-66.1.6.1.1.5/P-41')


def imidoyl_halide(mol, Chem):
    """R-C(=NR')-X, with hydrocarbon R/R' and no competing class."""
    if not _plain(mol, Chem, {6, 7, 9, 17, 35, 53}):
        return None
    ns = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    if len(ns) != 1:
        return None
    n = ns[0]
    if n.IsInRing() or n.GetTotalValence() != 3:
        return None
    doubles = [a for a in n.GetNeighbors() if a.GetAtomicNum() == 6
               and _order(mol, a.GetIdx(), n.GetIdx()) == 2]
    if len(doubles) != 1:
        return None
    carbon = doubles[0]
    halogens = [a for a in carbon.GetNeighbors() if a.GetAtomicNum() in _HALIDES
                and a.GetDegree() == 1 and _order(mol, carbon.GetIdx(), a.GetIdx()) == 1]
    if carbon.IsInRing() or len(halogens) != 1:
        return None
    x = halogens[0]
    if any(a.GetAtomicNum() != 6 for a in carbon.GetNeighbors()
           if a.GetIdx() not in {n.GetIdx(), x.GetIdx()}):
        return None
    prefixes, removed = [], set()
    for root in n.GetNeighbors():
        if root.GetIdx() == carbon.GetIdx():
            continue
        text, atoms = mapped_branch(mol, Chem, root.GetIdx(), n.GetIdx())
        if (root.GetAtomicNum() != 6 or _order(mol, n.GetIdx(), root.GetIdx()) != 1
                or not text or not atoms or removed & atoms
                or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in atoms)):
            return None
        prefixes.append(('N', text))
        removed.update(atoms)
    if len(prefixes) > 1 or n.GetTotalNumHs() != 1-len(prefixes):
        return None
    # The acid selects its complete carbon parent before functional
    # replacement. The source graph continues to own N and the halogen.
    rw = Chem.RWMol(mol)
    for idx in (n.GetIdx(), x.GetIdx()):
        a = rw.GetAtomWithIdx(idx)
        a.SetAtomicNum(8)
        a.SetNumExplicitHs(0)
        a.SetNoImplicit(False)
    for idx in sorted(removed, reverse=True):
        rw.RemoveAtom(idx)
    acid = _sanitize(rw, Chem)
    if acid is None:
        return None
    from .api import mol_to_iupac
    name = mol_to_iupac(acid)
    retained = {'formic acid': 'methanimidoyl', 'acetic acid': 'ethanimidoyl',
                'benzoic acid': 'benzenecarboximidoyl'}
    if name in retained:
        acyl = retained[name]
    elif name.endswith('carboxylic acid'):
        acyl = name[:-len('carboxylic acid')] + 'carboximidoyl'
    elif name.endswith('oic acid'):
        acyl = name[:-len('oic acid')] + 'imidoyl'
    else:
        return None
    name = format_prefixes(prefixes, hierarchical_enclosures=True) + acyl + ' ' + _HALIDES[x.GetAtomicNum()]
    return _out(mol, name, (carbon.GetIdx(),), 'P-65.5.1/P-65.1.7/P-41', 'functional-class')


def polyoxide_anion(mol, Chem):
    """A terminal hydron removed from R-O-(O)n-H, with n >= 2."""
    if not _plain(mol, Chem, {6, 8}, charges=(0, -1)):
        return None
    charged = [a for a in mol.GetAtoms() if a.GetFormalCharge()]
    if len(charged) != 1 or charged[0].GetAtomicNum() != 8 or charged[0].GetDegree() != 1:
        return None
    path, previous, current = [], None, charged[0].GetIdx()
    while mol.GetAtomWithIdx(current).GetAtomicNum() == 8:
        atom = mol.GetAtomWithIdx(current)
        if current in path or atom.IsInRing() or atom.GetTotalNumHs():
            return None
        path.append(current)
        neighbors = [a.GetIdx() for a in atom.GetNeighbors() if a.GetIdx() != previous]
        if len(neighbors) != 1 or _order(mol, current, neighbors[0]) != 1:
            return None
        previous, current = current, neighbors[0]
    if len(path) < 3 or len(path) not in MULT:
        return None
    text, atoms = mapped_branch(mol, Chem, current, previous)
    if (not text or set(path) | set(atoms) != set(range(mol.GetNumAtoms()))
            or set(path) & set(atoms)
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in atoms)):
        return None
    # P-29/P-72 retain this preselected tertiary group for the source class.
    root = mol.GetAtomWithIdx(current)
    branches = [a for a in root.GetNeighbors() if a.GetIdx() != previous]
    if len(atoms) == 4 and len(branches) == 3 and all(a.GetDegree() == 1 for a in branches):
        text = 'tert-butyl'
    if not text.isalpha() and not text.startswith(('tert-', 'sec-')):
        text = enclose_name_unit(text)
    return _out(mol, text + MULT[len(path)] + 'oxidanide', (charged[0].GetIdx(),),
                'P-72.2.2.2.4/P-29.6', 'substitutive')


def phosphorus_amide_halide(mol, Chem):
    """Amide replacement precedes the senior acid-halide class term."""
    if not _plain(mol, Chem, {6, 7, 8, 9, 15, 16, 17, 34, 35, 52, 53}):
        return None
    ps = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 15]
    if len(ps) != 1 or ps[0].IsInRing():
        return None
    p = ps[0]
    from .phosphorus_ester_methods import _carbon_component
    core, used, p_arms, nitrogens, oxos, halides, pseudos = {p.GetIdx()}, {p.GetIdx()}, [], [], [], [], []

    def carbon_arm(root, host):
        atoms = _carbon_component(mol, root, host)
        if atoms is None or atoms & used:
            return None
        used.update(atoms)
        return root, atoms

    for a in p.GetNeighbors():
        idx, order = a.GetIdx(), _order(mol, p.GetIdx(), a.GetIdx())
        if a.GetAtomicNum() in {8, 16, 34, 52} and a.GetDegree() == 1 and order == 2:
            oxos.append(a)
        elif a.GetAtomicNum() in _HALIDES and a.GetDegree() == 1 and order == 1:
            halides.append(a.GetAtomicNum())
        elif a.GetAtomicNum() == 6 and order == 1:
            arm = carbon_arm(idx, p.GetIdx())
            if arm is None:
                return None
            p_arms.append(arm)
            continue
        elif a.GetAtomicNum() == 7 and order == 1:
            others = [n for n in a.GetNeighbors() if n.GetIdx() != p.GetIdx()]
            pseudo = None
            if len(others) == 1 and others[0].GetAtomicNum() == 6 and _order(mol, idx, others[0].GetIdx()) == 2:
                carbon = others[0]
                ends = [n for n in carbon.GetNeighbors() if n.GetIdx() != idx
                        and n.GetAtomicNum() in {8, 16, 34, 52} and n.GetDegree() == 1
                        and _order(mol, carbon.GetIdx(), n.GetIdx()) == 2]
                if len(ends) == 1 and carbon.GetDegree() == 2:
                    pseudo = ends[0]
            if pseudo is not None:
                word = {8: 'isocyanatido', 16: 'isothiocyanatido',
                        34: 'isoselenocyanatido', 52: 'isotellurocyanatido'}[pseudo.GetAtomicNum()]
                pseudos.append(word)
                core.update({carbon.GetIdx(), pseudo.GetIdx()})
                used.update({carbon.GetIdx(), pseudo.GetIdx()})
            else:
                if a.IsInRing() or a.GetTotalValence() != 3 or len(others) > 2:
                    return None
                arms = []
                for neighbor in others:
                    if neighbor.GetAtomicNum() != 6 or _order(mol, idx, neighbor.GetIdx()) != 1:
                        return None
                    arm = carbon_arm(neighbor.GetIdx(), idx)
                    if arm is None:
                        return None
                    arms.append(arm)
                if a.GetTotalNumHs() + len(arms) != 2:
                    return None
                nitrogens.append((idx, tuple(arms)))
        else:
            return None
        if idx in used:
            return None
        core.add(idx)
        used.add(idx)
    fixed = len(p_arms) + p.GetTotalNumHs()
    if (not halides or not nitrogens or len(oxos) > 1 or fixed not in {0, 1, 2}
            or fixed + len(nitrogens) + len(halides) + len(pseudos) != 3
            or used != set(range(mol.GetNumAtoms()))):
        return None
    from .phosphorus_amide_methods import _substitution
    kind = ('phosphoric', 'phosphonic', 'phosphinic')[fixed]
    substitution = _substitution(mol, Chem, dict(p=p.GetIdx(), p_arms=tuple(p_arms),
        nitrogens=tuple(nitrogens), ester_oxygens=(), kind=kind))
    if substitution is None:
        return None
    prefix, _nmap = substitution
    stem = ('phosphor', 'phosphon', 'phosphin')[fixed]
    amid = 'amid' if len(nitrogens) == 1 else 'o' + MULT[len(nitrogens)] + 'amid'
    pseudo_text = ''.join(('' if count == 1 else 'o' + MULT[count]) + word[:-1]
                          for word, count in sorted(Counter(pseudos).items()))
    replacement = '' if not oxos or oxos[0].GetAtomicNum() == 8 else {16: 'othio', 34: 'oseleno', 52: 'otelluro'}[oxos[0].GetAtomicNum()]
    acid = stem + amid + pseudo_text + replacement + ('ic' if oxos else 'ous')
    classes = [('' if count == 1 else MULT[count]) + _HALIDES[number]
               for number, count in sorted(Counter(halides).items(), key=lambda row: _HALIDES[row[0]])]
    return _out(mol, prefix + acid + ' ' + ' '.join(classes), (p.GetIdx(),),
                'P-67.1.2.5.1/P-67.1.2.6.1/P-41', 'functional-class')


def acidic_boron_acyloxy(mol, Chem):
    """Residual B-OH is senior to the B-O-C(O) anhydride linkage."""
    if not _plain(mol, Chem, {5, 6, 8}):
        return None
    bs = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 5]
    if len(bs) != 1 or bs[0].IsInRing() or bs[0].GetTotalValence() != 3:
        return None
    b = bs[0]
    hydroxyls, rows, used, acyl_count = [], [], {b.GetIdx()}, 0
    for a in b.GetNeighbors():
        if _order(mol, b.GetIdx(), a.GetIdx()) != 1:
            return None
        if a.GetAtomicNum() == 8 and a.GetDegree() == 1 and a.GetTotalNumHs() == 1:
            hydroxyls.append(a.GetIdx())
            used.add(a.GetIdx())
            continue
        if a.GetAtomicNum() == 8:
            others = [n for n in a.GetNeighbors() if n.GetIdx() != b.GetIdx()]
            if (len(others) != 1 or others[0].GetAtomicNum() != 6
                    or not any(n.GetAtomicNum() == 8 and n.GetDegree() == 1
                               and _order(mol, others[0].GetIdx(), n.GetIdx()) == 2
                               for n in others[0].GetNeighbors())):
                return None
            acyl_count += 1
        elif a.GetAtomicNum() != 6:
            return None
        text, atoms = mapped_branch(mol, Chem, a.GetIdx(), b.GetIdx())
        if not text or not atoms or used & atoms:
            return None
        if a.GetAtomicNum() == 8:
            # P-65.7.6: an unmodified simple acyl joins directly to oxy.
            # Complex acyl units retain their own attachment boundary.
            text = _SIMPLE_ACYLOXY.get(text, text)
        # A different free acid inside an arm needs the full acid comparison.
        if any(mol.GetAtomWithIdx(i).GetAtomicNum() == 8
               and mol.GetAtomWithIdx(i).GetTotalNumHs() for i in atoms):
            return None
        rows.append(('', text))
        used.update(atoms)
    if len(hydroxyls) not in {1, 2} or not acyl_count or used != set(range(mol.GetNumAtoms())):
        return None
    parent = 'borinic acid' if len(hydroxyls) == 1 else 'boronic acid'
    text = format_prefixes(rows, omit_locants=True, hierarchical_enclosures=True)
    return _out(mol, text + parent, (b.GetIdx(),), 'P-65.7.6.1/P-68.1.4.1/P-41')


def nonstandard_chalcogen_hydride(mol, Chem):
    """Mononuclear lambda4/lambda6 hydrides without a senior function."""
    if not _plain(mol, Chem, {6, 8, 16, 34, 52}):
        return None
    xs = [a for a in mol.GetAtoms() if a.GetAtomicNum() in {16, 34, 52}]
    if len(xs) != 1 or xs[0].IsInRing() or xs[0].GetTotalValence() not in {4, 6}:
        return None
    center = xs[0]
    used, rows = {center.GetIdx()}, []
    from .phosphorus_ester_methods import _carbon_component
    from .branch.carbon import _alkylidene_prefix
    for a in center.GetNeighbors():
        order = _order(mol, center.GetIdx(), a.GetIdx())
        if a.GetAtomicNum() == 6:
            if order == 2:
                # The ylidene writer also sees the actual double bond.
                text = _alkylidene_prefix(mol, a.GetIdx(), center.GetIdx())
                pending, atoms = [a.GetIdx()], set()
                while pending:
                    i = pending.pop()
                    if i in atoms:
                        continue
                    if mol.GetAtomWithIdx(i).GetAtomicNum() != 6:
                        return None
                    atoms.add(i)
                    pending.extend(n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors()
                                   if n.GetIdx() != center.GetIdx())
                if not text:
                    return None
            elif order == 1:
                atoms = _carbon_component(mol, a.GetIdx(), center.GetIdx())
                text, owned = mapped_branch(mol, Chem, a.GetIdx(), center.GetIdx())
                if atoms is None or set(owned) != set(atoms):
                    return None
            else:
                return None
        elif a.GetAtomicNum() == 8 and order == 1:
            others = [n for n in a.GetNeighbors() if n.GetIdx() != center.GetIdx()]
            if (len(others) != 1 or others[0].GetAtomicNum() != 6
                    or _carbon_component(mol, others[0].GetIdx(), a.GetIdx()) is None):
                return None
            text, atoms = mapped_branch(mol, Chem, a.GetIdx(), center.GetIdx())
        else:
            return None  # In particular, an oxo group has its own senior class.
        if not text or not atoms or used & set(atoms):
            return None
        rows.append(('', text))
        used.update(atoms)
    if used != set(range(mol.GetNumAtoms())):
        return None
    parent = {16: 'sulfane', 34: 'selane', 52: 'tellane'}[center.GetAtomicNum()]
    prefixes = format_prefixes(rows, omit_locants=True, hierarchical_enclosures=True,
                              separate_unlocanted=True)
    name = (prefixes + '-' if prefixes else '') + 'λ' + str(center.GetTotalValence()) + '-' + parent
    return _out(mol, name, (center.GetIdx(),), 'P-68.4.3.1/P-14.1/P-41')


def select(mol, Chem):
    numbers = Counter(a.GetAtomicNum() for a in iter_atoms(mol))
    from .rules45_97_methods import select as select_new_rules
    decision = select_new_rules(mol,Chem,numbers)
    if decision is not None:
        return decision
    from .functional_parent_extensions import select as select_extensions
    result = select_extensions(mol, Chem, numbers)
    if result:
        return result
    if len(numbers) == 1 and numbers[6] and mol.GetRingInfo().NumRings() >= 3:
        result = systematic_aceylene(mol, Chem)
        if result:
            return result
    if numbers[7] == 2:
        from .hydrazine_methods import hydrocarbyl_hydrazine_decision
        result = hydrocarbyl_hydrazine_decision(mol, Chem)
        if result:
            return result
    if mol.GetRingInfo().NumRings() and numbers[7] and numbers[8] >= 2:
        result = ring_n_carboxylic_acid(mol, Chem)
        if result:
            return result
    if mol.GetRingInfo().NumRings() and numbers[8] and any(numbers[x] for x in (16, 34, 52)):
        result = mixed_ring_carbonyl_chalcogen(mol, Chem)
        if result:
            return result
    if numbers[7] >= 3 and numbers[8] >= 2:
        result = carboxamide_above_urea(mol, Chem)
        if result:
            return result
    if numbers[7] and any(numbers[x] for x in _HALIDES):
        result = imidoyl_halide(mol, Chem)
        if result:
            return result
        if numbers[15]:
            result = phosphorus_amide_halide(mol, Chem)
            if result:
                return result
    if numbers[8] >= 3 and any(a.GetFormalCharge() == -1 for a in iter_atoms(mol)):
        result = polyoxide_anion(mol, Chem)
        if result:
            return result
    if numbers[5] and numbers[8] >= 3:
        result = acidic_boron_acyloxy(mol, Chem)
        if result:
            return result
    if numbers[5] and any(numbers[x] for x in (16, 34, 52)):
        from .boron_acid_methods import boron_acid_decision
        result = boron_acid_decision(mol, Chem)
        if result:
            return result
    if any(numbers[x] for x in (16, 34, 52)):
        return nonstandard_chalcogen_hydride(mol, Chem)
    return None
