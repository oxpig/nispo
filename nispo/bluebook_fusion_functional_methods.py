"""Whole-graph P-61/P-64/P-65/P-67 constructions with bounded evidence.

Each writer owns the complete original graph and declines annotations or
competing characteristic groups it cannot faithfully represent.
"""
from __future__ import annotations

from .bluebook_azoxy_methods import _decision, _hydrocarbon_parent, _plain
from .component_mapping import mapped_branch
from .constants import MULT, ROOTS
from .formatting import enclose_name_unit, format_prefixes


def _single(mol, left, right):
    b = mol.GetBondBetweenAtoms(left, right)
    return b is not None and b.GetBondTypeAsDouble() == 1


def _carbon_component(mol, root, blocked):
    seen, stack = set(), [root]
    while stack:
        i = stack.pop()
        if i in seen:
            continue
        atom = mol.GetAtomWithIdx(i)
        if atom.GetAtomicNum() != 6:
            return None
        seen.add(i)
        for n in atom.GetNeighbors():
            if n.GetIdx() != blocked and n.GetAtomicNum() == 6:
                if not _single(mol, i, n.GetIdx()):
                    return None
                stack.append(n.GetIdx())
    return frozenset(seen)


def _carbon_path(mol, atoms, start):
    neighbors = {i: [n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors()
                     if n.GetIdx() in atoms] for i in atoms}
    if start not in atoms or any(len(v) > 2 for v in neighbors.values()) or len(neighbors[start]) > 1:
        return None
    path, last, current = [], None, start
    while current is not None and current not in path:
        path.append(current)
        nxt = [n for n in neighbors[current] if n != last]
        last, current = current, nxt[0] if nxt else None
    return tuple(path) if current is None and set(path) == set(atoms) else None


def _isocyanato_groups(mol):
    groups = []
    for n in mol.GetAtoms():
        if n.GetAtomicNum() != 7 or n.GetDegree() != 2 or n.GetTotalNumHs():
            continue
        doubles = [a for a in n.GetNeighbors() if a.GetAtomicNum() == 6
                   and mol.GetBondBetweenAtoms(n.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() == 2]
        hosts = [a for a in n.GetNeighbors() if _single(mol, n.GetIdx(), a.GetIdx())]
        if len(doubles) != 1 or len(hosts) != 1:
            continue
        carbon = doubles[0]
        ends = [a for a in carbon.GetNeighbors() if a.GetIdx() != n.GetIdx()]
        if (carbon.GetDegree() != 2 or len(ends) != 1 or ends[0].GetAtomicNum() not in {8, 16, 34, 52}
                or ends[0].GetDegree() != 1
                or mol.GetBondBetweenAtoms(carbon.GetIdx(), ends[0].GetIdx()).GetBondTypeAsDouble() != 2):
            continue
        groups.append((n.GetIdx(), carbon.GetIdx(), ends[0].GetIdx(), hosts[0].GetIdx()))
    return tuple(groups)


def isocyanato_hydride_decision(mol, Chem):
    """One or more identical -N=C=E substituents on a simple hydride."""
    if not _plain(mol, Chem) or any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
        return None
    sites = _isocyanato_groups(mol)
    if not sites or len(sites) > 6:
        return None
    groups = frozenset(i for n, c, x, _host in sites for i in (n, c, x))
    if len(groups) != 3 * len(sites):
        return None
    if any(a.GetAtomicNum() == 7 and a.GetIdx() not in groups for a in mol.GetAtoms()):
        return None
    if len({mol.GetAtomWithIdx(x).GetAtomicNum() for _n, _c, x, _h in sites}) != 1:
        return None
    skeleton = frozenset(range(mol.GetNumAtoms())) - groups
    if not skeleton or any(mol.GetAtomWithIdx(h).GetIdx() not in skeleton for _n, _c, _x, h in sites):
        return None
    kinds = {mol.GetAtomWithIdx(i).GetAtomicNum() for i in skeleton}
    if kinds == {6}:
        # No independent organic branches can change alphabetical numbering.
        if any(sum(nb.GetIdx() in skeleton for nb in mol.GetAtomWithIdx(i).GetNeighbors()) > 2 for i in skeleton):
            return None
        rings = [i for i in skeleton if mol.GetAtomWithIdx(i).IsInRing()]
        if rings and set(rings) != set(skeleton):
            return None
        if any(b.GetBondTypeAsDouble() not in {1, 1.5} for b in mol.GetBonds()
               if b.GetBeginAtomIdx() in skeleton and b.GetEndAtomIdx() in skeleton):
            return None
    elif len(skeleton) == 1 and kinds <= {5, 14, 32, 50, 82}:
        pass
    else:
        return None
    replacement = {8: 'isocyanato', 16: 'isothiocyanato',
                   34: 'isoselenocyanato', 52: 'isotellurocyanato'}[mol.GetAtomWithIdx(sites[0][2]).GetAtomicNum()]
    if len(skeleton) == 1 and kinds != {6}:
        element = next(iter(kinds))
        hydride = {5: 'borane', 14: 'silane', 32: 'germane',
                   50: 'stannane', 82: 'plumbane'}[element]
        multiplier = MULT.get(len(sites), '') if len(sites) > 1 else ''
        name = multiplier + replacement + hydride
        return _decision(mol, name, (), 'P-61.8/P-14.3.4.2',
                         (("parent-hydride", skeleton, ()), ("isocyanato-groups", groups, ())))
    copy = Chem.RWMol(mol)
    for n, _c, _x, _host in sites:
        atom = copy.GetAtomWithIdx(n)
        atom.SetAtomicNum(9)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(True)
    for idx in sorted((i for _n, c, x, _host in sites for i in (c, x)), reverse=True):
        copy.RemoveAtom(idx)
    capped = copy.GetMol()
    try:
        Chem.SanitizeMol(capped)
    except (ValueError, RuntimeError):
        return None
    from .api import mol_to_iupac
    name = mol_to_iupac(capped)
    if name.count('fluoro') != 1:
        return None
    name = name.replace('fluoro', replacement)
    return _decision(mol, name, (), 'P-61.8',
                     (("parent-hydride", skeleton, ()), ("isocyanato-groups", groups, ())))


def isocyanato_carboxylic_acid_decision(mol, Chem):
    """Keep a carbon-bound isocyanato group subordinate to carboxylic acid."""
    if not _plain(mol, Chem) or any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
        return None
    sites = _isocyanato_groups(mol)
    if len(sites) != 1:
        return None
    n, c, x, host = sites[0]
    group = frozenset((n, c, x))
    parent = frozenset(range(mol.GetNumAtoms())) - group
    if (mol.GetAtomWithIdx(host).GetAtomicNum() != 6
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() not in {6, 8} for i in parent)):
        return None
    oxygens = [mol.GetAtomWithIdx(i) for i in parent if mol.GetAtomWithIdx(i).GetAtomicNum() == 8]
    if len(oxygens) != 2:
        return None
    acids = []
    for atom in mol.GetAtoms():
        if atom.GetIdx() in group or atom.GetAtomicNum() != 6 or atom.GetDegree() != 3:
            continue
        oxos = [a for a in atom.GetNeighbors() if a.GetAtomicNum() == 8 and a.GetDegree() == 1
                and mol.GetBondBetweenAtoms(atom.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() == 2]
        ohs = [a for a in atom.GetNeighbors() if a.GetAtomicNum() == 8 and a.GetDegree() == 1
               and a.GetTotalNumHs(includeNeighbors=True) == 1 and _single(mol, atom.GetIdx(), a.GetIdx())]
        if len(oxos) == len(ohs) == 1:
            acids.append(atom.GetIdx())
    if len(acids) != 1 or host == acids[0]:
        return None
    # The isocyanato group is the only prefix that can influence numbering;
    # a fluorine cap therefore preserves the selected carbon-acid locants.
    copy = Chem.RWMol(mol)
    cap = copy.GetAtomWithIdx(n)
    cap.SetAtomicNum(9)
    cap.SetNumExplicitHs(0)
    cap.SetNoImplicit(True)
    for idx in sorted((c, x), reverse=True):
        copy.RemoveAtom(idx)
    capped = copy.GetMol()
    try:
        Chem.SanitizeMol(capped)
    except (ValueError, RuntimeError):
        return None
    from .api import mol_to_iupac
    name = mol_to_iupac(capped)
    if name.count('fluoro') != 1 or not name.endswith(' acid'):
        return None
    prefix = {8: 'isocyanato', 16: 'isothiocyanato',
              34: 'isoselenocyanato', 52: 'isotellurocyanato'}[mol.GetAtomWithIdx(x).GetAtomicNum()]
    name = name.replace('fluoro', prefix)
    return _decision(mol, name, (acids[0],), 'P-61.8/P-41',
                     (("carboxylic-acid-parent", parent, ()), ("isocyanato-group", group, ())))


def isocyanato_noncarbon_acid_decision(mol, Chem):
    """Select a phosphoric or sulfuric acid/pseudohalide over isocyanate."""
    if not _plain(mol, Chem) or any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
        return None
    sites = _isocyanato_groups(mol)
    if not sites or len(sites) > 3 or len({host for _n, _c, _x, host in sites}) != 1:
        return None
    if len({mol.GetAtomWithIdx(x).GetAtomicNum() for _n, _c, x, _host in sites}) != 1:
        return None
    center = mol.GetAtomWithIdx(sites[0][3])
    if center.GetAtomicNum() not in {15, 16}:
        return None
    groups = frozenset(i for n, c, x, _host in sites for i in (n, c, x))
    if len(groups) != 3 * len(sites):
        return None
    oxos, ohs = [], []
    for atom in center.GetNeighbors():
        if atom.GetIdx() in groups:
            continue
        order = mol.GetBondBetweenAtoms(center.GetIdx(), atom.GetIdx()).GetBondTypeAsDouble()
        if atom.GetAtomicNum() != 8 or atom.GetDegree() != 1:
            return None
        if order == 2:
            oxos.append(atom.GetIdx())
        elif order == 1 and atom.GetTotalNumHs(includeNeighbors=True) == 1:
            ohs.append(atom.GetIdx())
        else:
            return None
    core = frozenset((center.GetIdx(), *oxos, *ohs))
    if core | groups != frozenset(range(mol.GetNumAtoms())):
        return None
    count = len(sites)
    h = center.GetTotalNumHs(includeNeighbors=True)
    element = mol.GetAtomWithIdx(sites[0][2]).GetAtomicNum()
    class_word = {8: 'isocyanate', 16: 'isothiocyanate',
                  34: 'isoselenocyanate', 52: 'isotellurocyanate'}[element]
    if center.GetAtomicNum() == 15 and len(oxos) == 1:
        if ohs and element == 8 and not h and count + len(ohs) == 3:
            infix = ('di' if count == 2 else '') + 'isocyanatidic acid'
            name = 'phosphoro' + infix if count == 2 else 'phosphor' + infix
        elif not ohs and h + count == 3:
            stem = {0: 'phosphoryl', 1: 'phosphonic', 2: 'phosphinic'}.get(h)
            if stem is None:
                return None
            name = stem + ' ' + (MULT.get(count, '') if count > 1 else '') + class_word
        else:
            return None
    elif center.GetAtomicNum() == 16 and len(oxos) == 2 and not h:
        if len(ohs) == 1 and count == 1 and element == 8:
            name = 'sulfurisocyanatidic acid'
        elif not ohs and count == 2:
            name = 'sulfuryl di' + class_word
        else:
            return None
    else:
        return None
    return _decision(mol, name, (center.GetIdx(),), 'P-61.8/P-67.1.2.4/P-67.1.2.5',
                     (("oxoacid-parent", core, ()), ("pseudohalide-groups", groups, ())),
                     method='functional-class' if not ohs else 'functional-replacement')


def aryl_isocyanato_sulfonyl_halide_decision(mol, Chem):
    """Benzene sulfonyl halide senior to an isocyanato substituent."""
    if not _plain(mol, Chem):
        return None
    sites = _isocyanato_groups(mol)
    rings = mol.GetRingInfo().AtomRings()
    if len(sites) != 1 or len(rings) != 1 or len(rings[0]) != 6:
        return None
    ring = frozenset(rings[0])
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 or not mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
        return None
    n, c, x, attached = sites[0]
    if attached not in ring:
        return None
    sulfur = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 16 and a.GetIdx() != x]
    if len(sulfur) != 1:
        return None
    s = sulfur[0]
    oxo = [a for a in s.GetNeighbors() if a.GetAtomicNum() == 8 and a.GetDegree() == 1
           and mol.GetBondBetweenAtoms(s.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() == 2]
    halides = [a for a in s.GetNeighbors() if a.GetAtomicNum() in {9, 17, 35, 53}
               and a.GetDegree() == 1 and _single(mol, s.GetIdx(), a.GetIdx())]
    hosts = [a.GetIdx() for a in s.GetNeighbors() if a.GetIdx() in ring and _single(mol, s.GetIdx(), a.GetIdx())]
    if len(oxo) != 2 or len(halides) != 1 or len(hosts) != 1 or s.GetDegree() != 4:
        return None
    group = frozenset({n, c, x, s.GetIdx(), halides[0].GetIdx(), *(a.GetIdx() for a in oxo)})
    if ring | group != frozenset(range(mol.GetNumAtoms())):
        return None
    from collections import deque
    queue = deque([(hosts[0], 0)])
    visited = set()
    distance = None
    while queue:
        i, d = queue.popleft()
        if i in visited:
            continue
        visited.add(i)
        if i == attached:
            distance = d
            break
        queue.extend((a.GetIdx(), d + 1) for a in mol.GetAtomWithIdx(i).GetNeighbors()
                     if a.GetIdx() in ring and a.GetIdx() not in visited)
    if distance not in {1, 2, 3}:
        return None
    term = {8: 'isocyanato', 16: 'isothiocyanato', 34: 'isoselenocyanato', 52: 'isotellurocyanato'}[mol.GetAtomWithIdx(x).GetAtomicNum()]
    halide = {9: 'fluoride', 17: 'chloride', 35: 'bromide', 53: 'iodide'}[halides[0].GetAtomicNum()]
    name = f'{distance + 1}-{term}benzene-1-sulfonyl {halide}'
    return _decision(mol, name, (s.GetIdx(),), 'P-61.8/P-65.5.1',
                     (("benzene", ring, ()), ("functional-groups", group, ())), method='functional-class')


def acyclic_pseudoketone_decision(mol, Chem):
    """Unbranched carbonyl parent linked to P, Si or a dialkylaminooxy arm."""
    if not _plain(mol, Chem) or any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
        return None
    choices = []
    for carbon in mol.GetAtoms():
        if carbon.GetAtomicNum() != 6 or carbon.IsInRing() or carbon.GetDegree() != 3:
            continue
        ci = carbon.GetIdx()
        oxos = [n for n in carbon.GetNeighbors() if n.GetAtomicNum() == 8 and n.GetDegree() == 1
                and mol.GetBondBetweenAtoms(ci, n.GetIdx()).GetBondTypeAsDouble() == 2]
        roots = [n for n in carbon.GetNeighbors() if n.GetAtomicNum() == 6 and _single(mol, ci, n.GetIdx())]
        ligands = [n for n in carbon.GetNeighbors() if n.GetAtomicNum() in {8, 14, 15} and _single(mol, ci, n.GetIdx())]
        if len(oxos) == len(roots) == len(ligands) == 1:
            choices.append((carbon, oxos[0], roots[0], ligands[0]))
    if len(choices) != 1:
        return None
    carbon, oxo, root, ligand = choices[0]
    ci, li = carbon.GetIdx(), ligand.GetIdx()
    chain = _carbon_component(mol, root.GetIdx(), ci)
    if chain is None:
        return None
    chain = chain | {ci}
    path = _carbon_path(mol, chain, ci)
    if not path or len(path) < 2 or len(path) not in ROOTS or any(mol.GetAtomWithIdx(i).IsInRing() for i in path):
        return None
    prefix, branch = mapped_branch(mol, Chem, li, ci)
    if not prefix or branch & chain or branch | chain | {oxo.GetIdx()} != frozenset(range(mol.GetNumAtoms())):
        return None
    if any(b.GetBondTypeAsDouble() != 1 for b in mol.GetBonds()
           if b.GetBeginAtomIdx() in branch and b.GetEndAtomIdx() in branch):
        return None
    if ligand.GetAtomicNum() == 15:
        # The standard P-H substituent is phosphanyl, never phosphino.
        if branch != {li} or ligand.GetTotalNumHs() != 2:
            return None
        prefix = 'phosphanyl'
    elif ligand.GetAtomicNum() == 14:
        if prefix == 'tms':
            prefix = 'trimethylsilyl'
        if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in branch if i != li):
            return None
    else:
        # R-C(=O)-O-NR2 is a pseudoester/pseudoketone, not an ordinary ester.
        if ligand.GetDegree() != 2:
            return None
        ns = [n for n in ligand.GetNeighbors() if n.GetIdx() != ci and n.GetAtomicNum() == 7]
        if len(ns) != 1 or any(mol.GetAtomWithIdx(i).GetAtomicNum() not in {6, 7, 8} for i in branch):
            return None
        if any(mol.GetAtomWithIdx(i).GetAtomicNum() in {7, 8} and i not in {li, ns[0].GetIdx()} for i in branch):
            return None
    rendered = ('1-' + enclose_name_unit(prefix) if ligand.GetAtomicNum() == 14
                else format_prefixes([('1', prefix)], hierarchical_enclosures=True))
    name = rendered + ROOTS[len(path)] + 'an-1-one'
    mapping = tuple((i, str(position)) for position, i in enumerate(path, 1))
    return _decision(mol, name, (ci,), 'P-64.3.2/P-64.5.2.2',
                     (("carbonyl-parent", chain | {oxo.GetIdx()}, mapping), ("heteroatom-arm", branch, ())),
                     locants=mapping)


def modified_chalcogen_acid_decision(mol, Chem):
    """One R-E(=X)[=Y]-ZH acid with exact terminal replacement positions."""
    if not _plain(mol, Chem) or any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
        return None
    central = [a for a in mol.GetAtoms() if a.GetAtomicNum() in {16, 34, 52}
               and any(n.GetAtomicNum() == 6 and _single(mol, a.GetIdx(), n.GetIdx()) for n in a.GetNeighbors())]
    if len(central) != 1:
        return None
    e = central[0]
    doubles, terminal, hosts = [], [], []
    for n in e.GetNeighbors():
        order = mol.GetBondBetweenAtoms(e.GetIdx(), n.GetIdx()).GetBondTypeAsDouble()
        if n.GetAtomicNum() == 6 and order == 1:
            hosts.append(n)
        elif n.GetAtomicNum() in {8, 16, 34, 52} and n.GetDegree() == 1 and order == 2:
            doubles.append(n)
        elif (n.GetAtomicNum() in {8, 16, 34, 52} and n.GetDegree() == 1 and order == 1
              and n.GetTotalNumHs(includeNeighbors=True) == 1):
            terminal.append(n)
        else:
            return None
    if len(hosts) != 1 or len(terminal) != 1 or len(doubles) not in {1, 2}:
        return None
    ligand_atoms = doubles + terminal
    replacements = [a.GetAtomicNum() for a in ligand_atoms if a.GetAtomicNum() != 8]
    if not replacements:
        return None
    text, arm = mapped_branch(mol, Chem, hosts[0].GetIdx(), e.GetIdx())
    if not text or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in arm):
        return None
    core = frozenset({e.GetIdx(), *(a.GetIdx() for a in ligand_atoms)})
    if arm | core != frozenset(range(mol.GetNumAtoms())):
        return None
    parent = _hydrocarbon_parent(mol, arm, hosts[0].GetIdx())
    if parent is None:
        return None
    base, loc, _ring = parent
    prefixes = {16: 'thio', 34: 'seleno', 52: 'telluro'}
    # Functional-replacement prefixes are cited alphabetically.
    infixes = []
    for number in sorted(set(replacements), key=lambda x: prefixes[x]):
        count = replacements.count(number)
        infixes.append((MULT.get(count, '') if count > 1 else '') + prefixes[number])
    stems = {(16, 1): 'sulfino', (16, 2): 'sulfono',
             (34, 1): 'selenino', (34, 2): 'selenono',
             (52, 1): 'tellurino', (52, 2): 'tellurono'}
    ending = stems[(e.GetAtomicNum(), len(doubles))] + ''.join(infixes) + 'ic'
    # Mixed terminal elements admit alternative attachment positions.
    locant = (' ' + terminal[0].GetSymbol() + '-acid' if len({a.GetAtomicNum() for a in ligand_atoms}) > 1
              else ' acid')
    name = base + ('-' + loc + '-' if loc else '') + ending + locant
    return _decision(mol, name, (e.GetIdx(),), 'P-65.3.1.3/P-41',
                     (("hydrocarbon-parent", arm, ()), ("modified-sulfur-acid", core, ())))


def hydrazine_carboxylic_acid_decision(mol, Chem):
    """One N-N parent bearing one carboxyl group and carbon-only N arms."""
    if not _plain(mol, Chem) or any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
        return None
    ns = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    if len(ns) != 2 or any(n.IsInRing() or n.GetTotalValence() != 3 for n in ns):
        return None
    first, second = ns
    if not _single(mol, first.GetIdx(), second.GetIdx()):
        return None
    acid_sites = []
    for n in ns:
        for c in n.GetNeighbors():
            if c.GetAtomicNum() != 6 or not _single(mol, n.GetIdx(), c.GetIdx()) or c.GetDegree() != 3:
                continue
            oxos = [a for a in c.GetNeighbors() if a.GetAtomicNum() == 8 and a.GetDegree() == 1
                    and mol.GetBondBetweenAtoms(c.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() == 2]
            ohs = [a for a in c.GetNeighbors() if a.GetAtomicNum() == 8 and a.GetDegree() == 1
                   and a.GetTotalNumHs(includeNeighbors=True) == 1 and _single(mol, c.GetIdx(), a.GetIdx())]
            if len(oxos) == len(ohs) == 1:
                acid_sites.append((n.GetIdx(), c.GetIdx(), oxos[0].GetIdx(), ohs[0].GetIdx()))
    if len(acid_sites) != 1:
        return None
    host, acid_c, oxo, oh = acid_sites[0]
    other = next(n.GetIdx() for n in ns if n.GetIdx() != host)
    core = frozenset({host, other, acid_c, oxo, oh})
    seen, prefixes = set(core), []
    for nidx, loc in ((host, '1'), (other, '2')):
        for root in mol.GetAtomWithIdx(nidx).GetNeighbors():
            if root.GetIdx() in core:
                continue
            if root.GetAtomicNum() != 6 or not _single(mol, nidx, root.GetIdx()):
                return None
            text, atoms = mapped_branch(mol, Chem, root.GetIdx(), nidx)
            if not text or atoms & seen or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in atoms):
                return None
            prefixes.append((loc, text))
            seen.update(atoms)
    if seen != set(range(mol.GetNumAtoms())):
        return None
    parent = 'hydrazine-1-carboxylic acid' if prefixes else 'hydrazinecarboxylic acid'
    name = format_prefixes(prefixes, hierarchical_enclosures=True) + parent
    return _decision(mol, name, (acid_c,), 'P-65.1.8.1/P-68.3.1.2.1',
                     (("hydrazine", frozenset({host, other}), ((host, '1'), (other, '2'))),
                      ("carboxylic-acid", frozenset({acid_c, oxo, oh}), ()),
                      ("substituents", frozenset(seen - set(core)), ())))


def amino_noncarbon_acid_decision(mol, Chem):
    """Amido derivatives of phosphoric/phosphonic acid and sulfuric acid."""
    if not _plain(mol, Chem) or any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
        return None
    centers = [a for a in mol.GetAtoms() if a.GetAtomicNum() in {15, 16}]
    if len(centers) != 1:
        return None
    e = centers[0]
    oxos, ohs, nitrogens, carbons = [], [], [], []
    for n in e.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(e.GetIdx(), n.GetIdx()).GetBondTypeAsDouble()
        if n.GetAtomicNum() == 8 and n.GetDegree() == 1 and bond == 2:
            oxos.append(n.GetIdx())
        elif n.GetAtomicNum() == 8 and n.GetDegree() == 1 and bond == 1 and n.GetTotalNumHs() == 1:
            ohs.append(n.GetIdx())
        elif n.GetAtomicNum() == 7 and bond == 1 and not n.IsInRing():
            nitrogens.append(n)
        elif n.GetAtomicNum() == 6 and bond == 1:
            carbons.append(n)
        else:
            return None
    if not ohs or not nitrogens or len(nitrogens) > 2:
        return None
    if e.GetAtomicNum() == 16:
        if len(oxos) != 2 or len(ohs) != 1 or len(nitrogens) != 1 or carbons:
            return None
        parent = 'sulfamic acid'
    else:
        if len(oxos) != 1 or len(carbons) > 1:
            return None
        substitutable = len(carbons) + e.GetTotalNumHs(includeNeighbors=True)
        if substitutable == 0 and len(ohs) == 2 and len(nitrogens) == 1:
            parent = 'phosphoramidic acid'
        elif substitutable == 0 and len(ohs) == 1 and len(nitrogens) == 2:
            parent = 'phosphorodiamidic acid'
        elif substitutable == 1 and len(ohs) == len(nitrogens) == 1:
            parent = 'phosphonamidic acid'
        else:
            return None
    core = {e.GetIdx(), *oxos, *ohs, *(n.GetIdx() for n in nitrogens)}
    seen = set(core)
    sides = []
    for n in nitrogens:
        if n.GetTotalValence() != 3:
            return None
        prefixes = []
        for root in n.GetNeighbors():
            if root.GetIdx() == e.GetIdx():
                continue
            if root.GetAtomicNum() != 6 or not _single(mol, n.GetIdx(), root.GetIdx()):
                return None
            text, arm = mapped_branch(mol, Chem, root.GetIdx(), n.GetIdx())
            if not text or arm & seen or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in arm):
                return None
            seen.update(arm)
            prefixes.append(text)
        if len(prefixes) > 2:
            return None
        sides.append(prefixes)
    from .formatting import prefix_alphabetization_key
    sides.sort(key=lambda row: (not bool(row), tuple(sorted(prefix_alphabetization_key(t) for t in row))))
    prefixes = [(loc, text) for loc, side in zip(('N', "N'"), sides) for text in side]
    for root in carbons:
        text, arm = mapped_branch(mol, Chem, root.GetIdx(), e.GetIdx())
        if not text or arm & seen or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in arm):
            return None
        prefixes.append(('P', text))
        seen.update(arm)
    if seen != set(range(mol.GetNumAtoms())):
        return None
    name = format_prefixes(prefixes, hierarchical_enclosures=True) + parent
    return _decision(mol, name, (e.GetIdx(),), 'P-67.1.2.4/P-41',
                     (("acid-core", frozenset(core), ()), ("organyl-substituents", frozenset(seen - core), ())))


def acidic_mixed_anhydride_decision(mol, Chem):
    """One carboxylic acyloxy arm on phosphonic or boronic acid."""
    if not _plain(mol, Chem) or any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
        return None
    centers = [a for a in mol.GetAtoms() if a.GetAtomicNum() in {5, 15}]
    if len(centers) != 1:
        return None
    e = centers[0]
    ohs, oxos, links = [], [], []
    for a in e.GetNeighbors():
        if a.GetAtomicNum() != 8:
            return None
        order = mol.GetBondBetweenAtoms(e.GetIdx(), a.GetIdx()).GetBondTypeAsDouble()
        if a.GetDegree() == 1 and a.GetTotalNumHs() == 1 and order == 1:
            ohs.append(a.GetIdx())
        elif a.GetDegree() == 1 and order == 2:
            oxos.append(a.GetIdx())
        elif a.GetDegree() == 2 and order == 1:
            links.append(a.GetIdx())
        else:
            return None
    if len(ohs) != 2 or len(links) != 1 or len(oxos) != (1 if e.GetAtomicNum() == 15 else 0):
        return None
    link = links[0]
    carbon = [a for a in mol.GetAtomWithIdx(link).GetNeighbors() if a.GetIdx() != e.GetIdx()]
    if len(carbon) != 1 or carbon[0].GetAtomicNum() != 6 or not _single(mol, link, carbon[0].GetIdx()):
        return None
    oxo = [a for a in carbon[0].GetNeighbors() if a.GetAtomicNum() == 8 and a.GetDegree() == 1
           and mol.GetBondBetweenAtoms(carbon[0].GetIdx(), a.GetIdx()).GetBondTypeAsDouble() == 2]
    if len(oxo) != 1:
        return None
    text, arm = mapped_branch(mol, Chem, link, e.GetIdx())
    core = frozenset({e.GetIdx(), *ohs, *oxos})
    if (not text or not arm or arm & core or arm | core != frozenset(range(mol.GetNumAtoms()))
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() not in {6, 8} for i in arm)
            or sum(mol.GetAtomWithIdx(i).GetAtomicNum() == 8 for i in arm) != 2):
        return None
    # Mapped branch spelling sometimes encloses an ordinary acyl stem even
    # though the combined acyloxy unit is itself enclosed at attachment.
    if text.startswith('(') and text.endswith(')oxy') and text.count('(') == text.count(')') == 1:
        text = text[1:-4] + 'oxy'
    parent = 'phosphonic acid' if e.GetAtomicNum() == 15 else 'boronic acid'
    return _decision(mol, enclose_name_unit(text) + parent, (e.GetIdx(),), 'P-67.3.1/P-41',
                     (("noncarbon-acid", core, ()), ("acyloxy", arm, ())))


def diazo_ester_decision(mol, Chem):
    """Acyclic carboxylates with one charged -C=N+=N- diazo group."""
    if not _plain(mol, Chem, charged=True) or any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
        return None
    plus = [a for a in mol.GetAtoms() if a.GetFormalCharge() == 1]
    minus = [a for a in mol.GetAtoms() if a.GetFormalCharge() == -1]
    if (len(plus) != 1 or len(minus) != 1 or plus[0].GetAtomicNum() != 7 or minus[0].GetAtomicNum() != 7
            or any(a.GetFormalCharge() not in {-1, 0, 1} for a in mol.GetAtoms())
            or minus[0].GetDegree() != 1 or plus[0].GetDegree() != 2):
        return None
    n, terminal = plus[0], minus[0]
    nn = mol.GetBondBetweenAtoms(n.GetIdx(), terminal.GetIdx())
    if nn is None or nn.GetBondTypeAsDouble() != 2:
        return None
    diazo_host = [a for a in n.GetNeighbors() if a.GetAtomicNum() == 6
                  and mol.GetBondBetweenAtoms(n.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() == 2]
    if len(diazo_host) != 1:
        return None
    sites = []
    for c in mol.GetAtoms():
        if c.GetAtomicNum() != 6 or c.GetDegree() != 3 or c.IsInRing():
            continue
        oxos = [a for a in c.GetNeighbors() if a.GetAtomicNum() == 8 and a.GetDegree() == 1
                and mol.GetBondBetweenAtoms(c.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() == 2]
        os = [a for a in c.GetNeighbors() if a.GetAtomicNum() == 8 and a.GetDegree() == 2
              and _single(mol, c.GetIdx(), a.GetIdx())]
        if len(oxos) == len(os) == 1:
            sites.append((c.GetIdx(), oxos[0].GetIdx(), os[0].GetIdx()))
    if len(sites) != 1:
        return None
    ci, oi, link = sites[0]
    roots = [a for a in mol.GetAtomWithIdx(link).GetNeighbors() if a.GetIdx() != ci]
    acyl_roots = [a for a in mol.GetAtomWithIdx(ci).GetNeighbors() if a.GetAtomicNum() == 6]
    if len(roots) != 1 or roots[0].GetAtomicNum() != 6 or len(acyl_roots) != 1:
        return None
    organyl, alcohol_arm = mapped_branch(mol, Chem, roots[0].GetIdx(), link)
    acid_arm = _carbon_component(mol, acyl_roots[0].GetIdx(), ci)
    if not organyl or acid_arm is None or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in alcohol_arm):
        return None
    chain = acid_arm | {ci}
    path = _carbon_path(mol, chain, ci)
    if (not path or len(path) < 2 or len(path) not in ROOTS or diazo_host[0].GetIdx() not in path
            or any(mol.GetAtomWithIdx(i).IsInRing() for i in path)
            or chain & alcohol_arm):
        return None
    group = frozenset({oi, link, n.GetIdx(), terminal.GetIdx()})
    if chain | alcohol_arm | group != frozenset(range(mol.GetNumAtoms())):
        return None
    position = path.index(diazo_host[0].GetIdx()) + 1
    if len(path) == 2 and position == 2:
        acid_name = 'diazoacetate'
    else:
        acid_name = str(position) + '-diazo' + ROOTS[len(path)] + 'anoate'
    name = organyl + ' ' + acid_name
    mapping = tuple((idx, str(pos)) for pos, idx in enumerate(path, 1))
    return _decision(mol, name, (ci,), 'P-61.4/P-65.6.3.2.1',
                     (("acid-chain", chain | {oi, n.GetIdx(), terminal.GetIdx()}, mapping),
                      ("ester-oxygen", frozenset({link}), ()), ("organyl", alcohol_arm, ())),
                     method='functional-class', locants=mapping)


def phosphole_hydrogenation_decision(mol, Chem):
    """Use a valence-equivalent numbered pyrrole H state for phospholes."""
    if (not _plain(mol, Chem) or any(a.GetAtomicNum() not in {6, 15, 9, 17, 35, 53} for a in mol.GetAtoms())
            or any(a.GetAtomicNum() == 1 for a in mol.GetAtoms())):
        return None
    rings = mol.GetRingInfo().AtomRings()
    phosphorus = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 15]
    if len(rings) != 1 or len(rings[0]) != 5 or len(phosphorus) != 1:
        return None
    ring = frozenset(rings[0])
    p = phosphorus[0]
    if (p.GetIdx() not in ring or p.GetDegree() != 2 or p.GetTotalNumHs(includeNeighbors=True) != 1
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() not in {6, 15} for i in ring)):
        return None
    ring_bonds = [b for b in mol.GetBonds() if b.GetBeginAtomIdx() in ring and b.GetEndAtomIdx() in ring]
    if (sum(b.GetBondTypeAsDouble() == 2 for b in ring_bonds) != 1
            or any(b.GetBondTypeAsDouble() not in {1, 2} for b in ring_bonds)
            or any(b.GetBondTypeAsDouble() != 1 for b in p.GetBonds())):
        return None
    # Any peripheral substituent must be a carbon or halogen prefix; no
    # characteristic group or second ring may acquire a different seniority.
    if any(a.GetAtomicNum() in {9, 17, 35, 53} and a.GetDegree() != 1 for a in mol.GetAtoms()):
        return None
    neutral = Chem.Mol(mol)
    analog = neutral.GetAtomWithIdx(p.GetIdx())
    analog.SetAtomicNum(7)
    analog.SetNumExplicitHs(1)
    analog.SetNoImplicit(True)
    try:
        Chem.SanitizeMol(neutral)
    except (ValueError, RuntimeError):
        return None
    from .api import mol_to_iupac
    name = mol_to_iupac(neutral)
    if name.count('dihydro-1H-pyrrole') != 1 or not name.endswith('pyrrole'):
        return None
    name = name.replace('1H-pyrrole', '1H-phosphole')
    heavy = frozenset(range(mol.GetNumAtoms()))
    return _decision(mol, name, (), 'P-31.2.3.1/P-22.2.2.1.4',
                     (("phosphole-ring", ring, ()), ("substituents", heavy - ring, ())))
