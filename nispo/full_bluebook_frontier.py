"""Full-mode, graph-closed constructions for selected Blue Book classes.

Every producer proves the whole graph within its documented scope and declines
annotations and competing functions it cannot compare.  No panel data is used.
"""
from collections import Counter

from .molecule_iteration import iter_atoms
from .naming_methods import MethodDecision
from .functional_replacement_acids import _plain


_HALIDES = {9: 'fluorid', 17: 'chlorid', 35: 'bromid', 53: 'iodid'}


def _decision(mol, name, centers, rule, method='substitutive'):
    atoms = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    return MethodDecision(name, method, frozenset(centers), atoms, rule, (),
                          component_roles=(('complete-source-graph', atoms, ()),))


def _bond(mol, a, b):
    bond = mol.GetBondBetweenAtoms(a, b)
    return bond.GetBondTypeAsDouble() if bond else 0


def _clean(mol, Chem, allowed):
    return (mol.GetNumAtoms() == mol.GetNumHeavyAtoms() and _plain(mol, Chem)
            and all(a.GetAtomicNum() in allowed for a in iter_atoms(mol)))


def _sanitize(rw, Chem):
    result = rw.GetMol()
    try:
        Chem.SanitizeMol(result)
    except (ValueError, RuntimeError):
        return None
    return result


def _n_prefixes(mol, Chem, nitrogen, blocked):
    """Render and account for N-bound carbon-only substituents."""
    from .component_mapping import mapped_branch
    from .formatting import format_prefixes
    from .phosphorus_ester_methods import _carbon_component
    used = set(blocked)
    rows = []
    for atom in nitrogen.GetNeighbors():
        idx = atom.GetIdx()
        if idx in blocked:
            continue
        if atom.GetAtomicNum() != 6 or _bond(mol, nitrogen.GetIdx(), idx) != 1:
            return None
        owned = _carbon_component(mol, idx, nitrogen.GetIdx())
        if not owned or owned & used:
            return None
        text, mapped = mapped_branch(mol, Chem, idx, nitrogen.GetIdx())
        if not text or frozenset(mapped) != owned:
            return None
        used.update(owned)
        rows.append(('N', text))
    return format_prefixes(rows, hierarchical_enclosures=True), used


def hetero_carbonitrile(mol, Chem):
    # Run the existing source-closed nitrile constructor before the generic
    # silicon/boron cyano-prefix route, while respecting its competing-group
    # and retained-parent exclusions.
    from .hetero_functional_methods import heterohydride_carbonitrile_decision
    return heterohydride_carbonitrile_decision(mol, Chem)


def cyanic_chalcogen_acid(mol, Chem):
    """Unsubstituted mono-chalcogen and homogeneous peroxycyanic acids."""
    if not _clean(mol, Chem, {6, 7, 8, 16, 34, 52}):
        return None
    ns = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    cs = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 6]
    if len(ns) != 1 or len(cs) != 1:
        return None
    c, n = cs[0], ns[0]
    if n.GetDegree() != 1 or _bond(mol, c.GetIdx(), n.GetIdx()) != 3 or c.GetDegree() != 2:
        return None
    chain = [a for a in c.GetNeighbors() if a.GetIdx() != n.GetIdx()]
    if len(chain) != 1 or _bond(mol, c.GetIdx(), chain[0].GetIdx()) != 1:
        return None
    first = chain[0]
    remaining = [a for a in first.GetNeighbors() if a.GetIdx() != c.GetIdx()]
    if not remaining and first.GetTotalNumHs(includeNeighbors=True) == 1:
        name = {16: 'thiocyanic acid', 34: 'selenocyanic acid',
                52: 'tellurocyanic acid'}.get(first.GetAtomicNum())
    elif (len(remaining) == 1 and first.GetDegree() == 2 and remaining[0].GetDegree() == 1
          and _bond(mol, first.GetIdx(), remaining[0].GetIdx()) == 1
          and first.GetAtomicNum() == remaining[0].GetAtomicNum()
          and remaining[0].GetTotalNumHs(includeNeighbors=True) == 1):
        name = {8: 'peroxycyanic acid', 16: 'dithioperoxycyanic acid',
                34: 'diselenoperoxycyanic acid', 52: 'ditelluroperoxycyanic acid'}.get(first.GetAtomicNum())
    else:
        return None
    if not name or mol.GetNumAtoms() != 3 + len(remaining):
        return None
    return _decision(mol, name, (c.GetIdx(),), 'P-65.2.2', 'functional-replacement')


def o_sulfinyl_hydroxylamine(mol, Chem):
    """O-sulfinyl hydroxylamine with an otherwise hydrocarbon acid residue."""
    if not _clean(mol, Chem, {6, 7, 8, 16}):
        return None
    ns = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    ss = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 16]
    if len(ns) != 1 or len(ss) != 1:
        return None
    n, s = ns[0], ss[0]
    if n.GetDegree() != 1 or n.GetTotalNumHs(includeNeighbors=True) != 2:
        return None
    o = n.GetNeighbors()[0]
    if o.GetAtomicNum() != 8 or o.GetDegree() != 2 or _bond(mol, o.GetIdx(), s.GetIdx()) != 1:
        return None
    oxo = [a for a in s.GetNeighbors() if a.GetAtomicNum() == 8 and a.GetDegree() == 1
           and _bond(mol, s.GetIdx(), a.GetIdx()) == 2]
    roots = [a for a in s.GetNeighbors() if a.GetAtomicNum() == 6 and _bond(mol, s.GetIdx(), a.GetIdx()) == 1]
    if len(oxo) != 1 or len(roots) != 1 or s.GetDegree() != 3:
        return None
    if sum(a.GetAtomicNum() == 8 for a in mol.GetAtoms()) != 2:
        return None
    copy = Chem.RWMol(mol)
    copy.RemoveAtom(n.GetIdx())
    acid = _sanitize(copy, Chem)
    if acid is None:
        return None
    from .api import mol_to_iupac
    name = mol_to_iupac(acid)
    if not name.endswith('sulfinic acid'):
        return None
    acyl = name[:-len('ic acid')] + 'yl'
    return _decision(mol, 'O-(' + acyl + ')hydroxylamine', (n.GetIdx(),),
                     'P-68.3.1.1.1.2/P-41')


def senior_function_with_o_sulfinyl_hydroxylamine(mol, Chem):
    """Keep a proved senior carbon function as parent and map the O-N unit."""
    if not _clean(mol, Chem, {6, 7, 8, 9, 16, 17, 53}):
        return None
    sulfurs = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 16]
    if len(sulfurs) != 1:
        return None
    sulfur = sulfurs[0]
    if sulfur.IsInRing() or sulfur.GetDegree() != 3:
        return None
    oxos = [a for a in sulfur.GetNeighbors() if a.GetAtomicNum() == 8 and a.GetDegree() == 1
            and _bond(mol, sulfur.GetIdx(), a.GetIdx()) == 2]
    bridges = [a for a in sulfur.GetNeighbors() if a.GetAtomicNum() == 8 and a.GetDegree() == 2
               and _bond(mol, sulfur.GetIdx(), a.GetIdx()) == 1]
    roots = [a for a in sulfur.GetNeighbors() if a.GetAtomicNum() == 6
             and _bond(mol, sulfur.GetIdx(), a.GetIdx()) == 1]
    if len(oxos) != 1 or len(bridges) != 1 or len(roots) != 1:
        return None
    bridge = bridges[0]
    nitrogens = [a for a in bridge.GetNeighbors() if a.GetIdx() != sulfur.GetIdx()
                 and a.GetAtomicNum() == 7 and a.GetDegree() == 1
                 and a.GetTotalNumHs(includeNeighbors=True) == 2
                 and _bond(mol, bridge.GetIdx(), a.GetIdx()) == 1]
    if len(nitrogens) != 1:
        return None
    removed = {oxos[0].GetIdx(), bridge.GetIdx(), nitrogens[0].GetIdx()}
    copy = Chem.RWMol(mol)
    marker = copy.GetAtomWithIdx(sulfur.GetIdx())
    marker.SetAtomicNum(35)
    marker.SetNumExplicitHs(0)
    marker.SetNoImplicit(True)
    for idx in sorted(removed, reverse=True):
        copy.RemoveAtom(idx)
    probe = _sanitize(copy, Chem)
    if probe is None:
        return None
    from .extended_inventory import selection_inventory
    inventory = selection_inventory(probe)
    endings = {'acid': ' acid', 'ester': 'ate', 'amide': 'amide',
               'nitrile': 'nitrile', 'aldehyde': 'al', 'ketone': 'one'}
    if (not inventory.supported or not inventory.principal
            or any(g.kind not in endings for g in inventory.principal)):
        return None
    from .api import mol_to_iupac
    name = mol_to_iupac(probe)
    if (name.count('bromo') != 1 or not any(name.endswith(endings[g.kind])
                                            for g in inventory.principal)):
        return None
    remaining = [i for i in range(mol.GetNumAtoms()) if i not in removed]
    centers = tuple(remaining[g.center] for g in inventory.principal)
    return _decision(mol, name.replace('bromo', '[(aminooxy)sulfinyl]', 1), centers,
                     'P-68.3.1.1.1.5/P-41')


def carbohydrazonic_acid(mol, Chem):
    """Hydrazonic acid groups on a carbon-only parent, using systematic acid names."""
    if not _clean(mol, Chem, {6, 7, 8, 9, 17, 35, 53}):
        return None
    sites = []
    for c in mol.GetAtoms():
        if c.GetAtomicNum() != 6 or c.IsInRing() or c.GetDegree() not in {2, 3}:
            continue
        oh = [a for a in c.GetNeighbors() if a.GetAtomicNum() == 8 and a.GetDegree() == 1
              and a.GetTotalNumHs(includeNeighbors=True) == 1 and _bond(mol, c.GetIdx(), a.GetIdx()) == 1]
        nn = [a for a in c.GetNeighbors() if a.GetAtomicNum() == 7 and a.GetDegree() == 2
              and _bond(mol, c.GetIdx(), a.GetIdx()) == 2]
        if len(oh) != 1 or len(nn) != 1:
            continue
        n = nn[0]
        ends = [a for a in n.GetNeighbors() if a.GetIdx() != c.GetIdx() and a.GetAtomicNum() == 7
                and a.GetDegree() == 1 and a.GetTotalNumHs(includeNeighbors=True) == 2
                and _bond(mol, n.GetIdx(), a.GetIdx()) == 1]
        roots = [a for a in c.GetNeighbors() if a.GetIdx() not in {oh[0].GetIdx(), n.GetIdx()}]
        if (len(ends) == 1 and ((len(roots) == 1 and roots[0].GetAtomicNum() == 6)
                                or (not roots and c.GetTotalNumHs(includeNeighbors=True) == 1))):
            sites.append((c.GetIdx(), n.GetIdx(), ends[0].GetIdx(), oh[0].GetIdx()))
    if not 1 <= len(sites) <= 6:
        return None
    active = {i for site in sites for i in site[1:]}
    if any(a.GetAtomicNum() in {7, 8} and a.GetIdx() not in active for a in mol.GetAtoms()):
        return None
    copy = Chem.RWMol(mol)
    for _c, n, _end, _oh in sites:
        atom = copy.GetAtomWithIdx(n)
        atom.SetAtomicNum(8)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(True)
    for idx in sorted((site[2] for site in sites), reverse=True):
        copy.RemoveAtom(idx)
    acid = _sanitize(copy, Chem)
    if acid is None:
        return None
    from .api import mol_to_iupac
    base = mol_to_iupac(acid)
    if base == 'oxalic acid':
        base = 'ethanedioic acid'
    if len(sites) == 1 and base == 'formic acid':
        name = 'methanehydrazonic acid'
    elif len(sites) == 1 and base.endswith('acetic acid'):
        name = base[:-len('acetic acid')] + 'ethanehydrazonic acid'
    elif len(sites) == 1 and base.endswith('benzoic acid'):
        name = base[:-len('benzoic acid')] + 'benzenecarbohydrazonic acid'
    elif base.endswith('carboxylic acid'):
        # This also preserves the di-/tri- multiplier and all original
        # attachment locants on ring or branched parent hydrides.
        name = base[:-len('carboxylic acid')] + 'carbohydrazonic acid'
    elif len(sites) == 2 and base.endswith('dioic acid'):
        name = base[:-len('dioic acid')] + 'dihydrazonic acid'
    elif len(sites) == 1 and base.endswith('oic acid'):
        name = base[:-len('oic acid')] + 'ehydrazonic acid'
    else:
        return None
    return _decision(mol, name, (site[0] for site in sites), 'P-65.1.3.2.1/P-41')


def halocarbonic_amide(mol, Chem):
    """Carbonic-acid halogen replacement with a primary or substituted amide."""
    if not _clean(mol, Chem, {6, 7, 8, *_HALIDES}):
        return None
    for carbon in mol.GetAtoms():
        c = carbon.GetIdx()
        if carbon.GetAtomicNum() != 6 or carbon.IsInRing() or carbon.GetDegree() != 3:
            continue
        ox = [a for a in carbon.GetNeighbors() if a.GetAtomicNum() == 8 and a.GetDegree() == 1 and _bond(mol, c, a.GetIdx()) == 2]
        halo = [a for a in carbon.GetNeighbors() if a.GetAtomicNum() in _HALIDES and a.GetDegree() == 1 and _bond(mol, c, a.GetIdx()) == 1]
        amine = [a for a in carbon.GetNeighbors() if a.GetAtomicNum() == 7 and not a.IsInRing() and _bond(mol, c, a.GetIdx()) == 1]
        if len(ox) != 1 or len(halo) != 1 or len(amine) != 1:
            continue
        n = amine[0]
        if n.GetTotalValence() != 3 or any(_bond(mol, n.GetIdx(), a.GetIdx()) != 1 for a in n.GetNeighbors()):
            continue
        result = _n_prefixes(mol, Chem, n, {c, n.GetIdx(), ox[0].GetIdx(), halo[0].GetIdx()})
        if result is None or result[1] != set(range(mol.GetNumAtoms())):
            continue
        return _decision(mol, result[0] + 'carbono' + _HALIDES[halo[0].GetAtomicNum()] + 'ic amide',
                         (c,), 'P-66.1.1.1.2.2/P-65.2.1.4', 'functional-replacement')
    return None


def sulfinimidic_acid(mol, Chem):
    """One R-S(=NH)-OH with a hydrocarbon residue and no competing group."""
    if not _clean(mol, Chem, {6, 7, 8, 16}):
        return None
    ss = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 16]
    ns = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    os = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 8]
    if len(ss) != 1 or len(ns) != 1 or len(os) != 1:
        return None
    s, n, o = ss[0], ns[0], os[0]
    if (s.IsInRing() or s.GetDegree() != 3 or n.GetDegree() != 1
            or n.GetTotalNumHs(includeNeighbors=True) != 1 or _bond(mol, s.GetIdx(), n.GetIdx()) != 2
            or o.GetDegree() != 1 or o.GetTotalNumHs(includeNeighbors=True) != 1
            or _bond(mol, s.GetIdx(), o.GetIdx()) != 1
            or len([a for a in s.GetNeighbors() if a.GetAtomicNum() == 6]) != 1):
        return None
    copy = Chem.RWMol(mol)
    atom = copy.GetAtomWithIdx(n.GetIdx())
    atom.SetAtomicNum(8)
    atom.SetNumExplicitHs(0)
    atom.SetNoImplicit(True)
    acid = _sanitize(copy, Chem)
    if acid is None:
        return None
    from .api import mol_to_iupac
    name = mol_to_iupac(acid)
    if not name.endswith('sulfinic acid'):
        return None
    return _decision(mol, name[:-len('sulfinic acid')] + 'sulfinimidic acid',
                     (s.GetIdx(),), 'P-65.3.1.4/P-41')


def sulfamoyl_halide(mol, Chem):
    """Sulfuric-acid amide/halide with completely mapped N-carbon arms."""
    if not _clean(mol, Chem, {6, 7, 8, 16, *_HALIDES}):
        return None
    centers = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 16]
    if len(centers) != 1:
        return None
    s = centers[0]
    ox = [a for a in s.GetNeighbors() if a.GetAtomicNum() == 8 and a.GetDegree() == 1 and _bond(mol, s.GetIdx(), a.GetIdx()) == 2]
    ns = [a for a in s.GetNeighbors() if a.GetAtomicNum() == 7 and _bond(mol, s.GetIdx(), a.GetIdx()) == 1]
    hs = [a for a in s.GetNeighbors() if a.GetAtomicNum() in _HALIDES and a.GetDegree() == 1 and _bond(mol, s.GetIdx(), a.GetIdx()) == 1]
    if s.IsInRing() or s.GetDegree() != 4 or len(ox) != 2 or len(ns) != 1 or len(hs) != 1:
        return None
    n = ns[0]
    if n.IsInRing() or n.GetTotalValence() != 3 or any(_bond(mol, n.GetIdx(), a.GetIdx()) != 1 for a in n.GetNeighbors()):
        return None
    core = {s.GetIdx(), n.GetIdx(), hs[0].GetIdx(), *(a.GetIdx() for a in ox)}
    result = _n_prefixes(mol, Chem, n, core)
    if result is None or result[1] != set(range(mol.GetNumAtoms())):
        return None
    return _decision(mol, result[0] + 'sulfamoyl ' + _HALIDES[hs[0].GetAtomicNum()] + 'e',
                     (s.GetIdx(),), 'P-67.1.2.5.1/P-41', 'functional-class')


def carbamoyl_pseudohalide(mol, Chem):
    """Carbonic-acid monoamide pseudohalides, including substituted amides."""
    if mol.GetNumAtoms() != mol.GetNumHeavyAtoms() or not _plain(mol, Chem):
        return None
    from .acyl_pseudohalide_methods import _pseudohalide_at
    for c in mol.GetAtoms():
        if c.GetAtomicNum() != 6 or c.IsInRing() or c.GetDegree() != 3:
            continue
        ox = [a for a in c.GetNeighbors() if a.GetAtomicNum() == 8 and a.GetDegree() == 1 and _bond(mol, c.GetIdx(), a.GetIdx()) == 2]
        ns = [a for a in c.GetNeighbors() if a.GetAtomicNum() == 7 and _bond(mol, c.GetIdx(), a.GetIdx()) == 1]
        if len(ox) != 1 or len(ns) != 2:
            continue
        for pseudo_root, amide in (ns, ns[::-1]):
            pseudo = _pseudohalide_at(mol, c.GetIdx(), pseudo_root.GetIdx())
            if pseudo is None or amide.IsInRing() or amide.GetTotalValence() != 3:
                continue
            core = {c.GetIdx(), ox[0].GetIdx(), amide.GetIdx()} | set(pseudo.atoms)
            result = _n_prefixes(mol, Chem, amide, core)
            if result is None or result[1] != set(range(mol.GetNumAtoms())):
                continue
            return _decision(mol, result[0] + 'carbamoyl ' + pseudo.kind,
                             (c.GetIdx(),), 'P-65.5.3.1/P-41', 'functional-class')
    return None


def polyazane_acid(mol, Chem):
    """A homogeneous nitrogen chain bearing carboxylic-acid suffixes."""
    if not _clean(mol, Chem, {6, 7, 8, 9, 17, 35, 53}):
        return None
    ns = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    if not 3 <= len(ns) <= 20 or any(n.IsInRing() or n.GetTotalValence() != 3 for n in ns):
        return None
    nitrogen_ids = {a.GetIdx() for a in ns}
    adj = {a.GetIdx(): [b.GetIdx() for b in a.GetNeighbors() if b.GetIdx() in nitrogen_ids]
           for a in ns}
    ends = [i for i, neighbors in adj.items() if len(neighbors) == 1]
    if len(ends) != 2 or any(len(v) not in {1, 2} for v in adj.values()):
        return None
    if any(_bond(mol, i, j) != 1 for i, neighbors in adj.items() for j in neighbors):
        return None
    path = [ends[0]]
    while len(path) < len(ns):
        nexts = [j for j in adj[path[-1]] if j not in path]
        if len(nexts) != 1:
            return None
        path.append(nexts[0])
    acids = []
    for n in ns:
        for c in n.GetNeighbors():
            if c.GetAtomicNum() != 6 or c.GetDegree() != 3 or _bond(mol, n.GetIdx(), c.GetIdx()) != 1:
                continue
            ox = [a for a in c.GetNeighbors() if a.GetAtomicNum() == 8 and a.GetDegree() == 1 and _bond(mol, c.GetIdx(), a.GetIdx()) == 2]
            oh = [a for a in c.GetNeighbors() if a.GetAtomicNum() == 8 and a.GetDegree() == 1 and a.GetTotalNumHs(includeNeighbors=True) == 1 and _bond(mol, c.GetIdx(), a.GetIdx()) == 1]
            if len(ox) == len(oh) == 1:
                acids.append((n.GetIdx(), c.GetIdx(), ox[0].GetIdx(), oh[0].GetIdx()))
    if not 1 <= len(acids) <= 6:
        return None
    core = nitrogen_ids | {idx for _host, center, ox, oh in acids for idx in (center, ox, oh)}
    from .component_mapping import mapped_branch
    from .phosphorus_ester_methods import _carbon_component
    from .formatting import format_prefixes, prefix_alphabetization_key
    used, branches = set(core), []
    for n in ns:
        for a in n.GetNeighbors():
            i = a.GetIdx()
            if i in core:
                continue
            if a.GetAtomicNum() != 6 or _bond(mol, n.GetIdx(), i) != 1:
                return None
            owned = _carbon_component(mol, i, n.GetIdx())
            if not owned or owned & used:
                return None
            text, mapped = mapped_branch(mol, Chem, i, n.GetIdx())
            if not text or frozenset(mapped) != owned:
                return None
            used.update(owned)
            branches.append((n.GetIdx(), text))
    if used != set(range(mol.GetNumAtoms())):
        return None
    from .constants import MULT
    parent = MULT[len(ns)] + 'azane'
    choices = []
    for order in (path, path[::-1]):
        mapping = {atom: i + 1 for i, atom in enumerate(order)}
        groups = [(str(mapping[i]), text) for i, text in branches]
        by_word = tuple(tuple(sorted(mapping[i] for i, text in branches if text == word))
                        for word in sorted({text for _, text in branches}, key=prefix_alphabetization_key))
        acid_locants = tuple(sorted(mapping[host] for host, _center, _ox, _oh in acids))
        key = (acid_locants, tuple(sorted(mapping[i] for i, _ in branches)), by_word)
        multiplier = MULT[len(acids)] if len(acids) > 1 else ''
        name = (format_prefixes(groups, hierarchical_enclosures=True) + parent + '-'
                + ','.join(map(str, acid_locants)) + '-' + multiplier + 'carboxylic acid')
        choices.append((key, name))
    return _decision(mol, min(choices)[1], (center for _host, center, _ox, _oh in acids),
                     'P-58.3/P-65.1.2.2/P-41')


def aromatic_benzenide(mol, Chem):
    """Unsubstituted benzene anion: all six neutral positions equivalent."""
    if (mol.GetNumAtoms() != 6 or len(Chem.GetMolFrags(mol)) != 1 or mol.GetStereoGroups()
            or any(a.GetAtomicNum() != 6 or a.GetIsotope() or a.GetNumRadicalElectrons()
                   or int(a.GetChiralTag()) or a.GetDegree() != 2 or not a.IsInRing()
                   for a in mol.GetAtoms())):
        return None
    charged = [a for a in mol.GetAtoms() if a.GetFormalCharge()]
    if len(charged) != 1 or charged[0].GetFormalCharge() != -1:
        return None
    neutral = Chem.Mol(mol)
    atom = neutral.GetAtomWithIdx(charged[0].GetIdx())
    atom.SetFormalCharge(0)
    atom.SetNumExplicitHs(1)
    atom.SetNoImplicit(True)
    try:
        Chem.SanitizeMol(neutral)
    except (ValueError, RuntimeError):
        return None
    if Chem.MolToSmiles(neutral) != 'c1ccccc1':
        return None
    return _decision(mol, 'benzenide', (charged[0].GetIdx(),), 'P-72.2.2.1/P-14.3.4.4')


def symmetric_cycloalkanide(mol, Chem):
    """Omit the charge locant when all parent cycloalkane positions agree."""
    if (mol.GetNumAtoms() != mol.GetNumHeavyAtoms() or not 3 <= mol.GetNumAtoms() <= 12
            or len(Chem.GetMolFrags(mol)) != 1 or mol.GetStereoGroups()
            or any(a.GetAtomicNum() != 6 or a.GetIsotope() or a.GetNumRadicalElectrons()
                   or int(a.GetChiralTag()) or a.GetDegree() != 2 or not a.IsInRing()
                   for a in mol.GetAtoms())
            or any(b.GetBondTypeAsDouble() != 1 or int(b.GetStereo()) for b in mol.GetBonds())
            or len(mol.GetRingInfo().AtomRings()) != 1):
        return None
    charged = [a for a in mol.GetAtoms() if a.GetFormalCharge()]
    if len(charged) != 1 or charged[0].GetFormalCharge() != -1:
        return None
    if any(a.GetTotalNumHs(includeNeighbors=True) != (1 if a.GetIdx() == charged[0].GetIdx() else 2)
           for a in mol.GetAtoms()):
        return None
    neutral = Chem.Mol(mol)
    atom = neutral.GetAtomWithIdx(charged[0].GetIdx())
    atom.SetFormalCharge(0)
    atom.SetNumExplicitHs(2)
    atom.SetNoImplicit(True)
    try:
        Chem.SanitizeMol(neutral)
    except (ValueError, RuntimeError):
        return None
    from .api import mol_to_iupac
    parent = mol_to_iupac(neutral)
    if not parent.startswith('cyclo') or not parent.endswith('ane') or not parent.isalpha():
        return None
    return _decision(mol, parent[:-1] + 'ide', (charged[0].GetIdx(),), 'P-72.2.2.1/P-14.3.4.4')


def hydroxy_halogen_parent(mol, Chem):
    """One λ3/λ5/λ7 halogen hydride with OH and carbon-only organyls."""
    if not _clean(mol, Chem, {6, 8, 17, 35, 53}):
        return None
    centers = [a for a in mol.GetAtoms() if a.GetAtomicNum() in {17, 35, 53}]
    if len(centers) != 1:
        return None
    center = centers[0]
    oh = [a for a in center.GetNeighbors() if a.GetAtomicNum() == 8 and a.GetDegree() == 1
          and a.GetTotalNumHs(includeNeighbors=True) == 1 and _bond(mol, center.GetIdx(), a.GetIdx()) == 1]
    roots = [a.GetIdx() for a in center.GetNeighbors() if a.GetAtomicNum() == 6]
    valence = center.GetDegree()
    if (center.IsInRing() or valence not in {3, 5, 7} or center.GetTotalValence() != valence
            or not oh or not roots or len(oh) + len(roots) != valence
            or len(oh) != sum(a.GetAtomicNum() == 8 for a in mol.GetAtoms())
            or any(_bond(mol, center.GetIdx(), i) != 1 for i in roots)):
        return None
    from .functional_replacement_acids import _organyl_units
    seen = {center.GetIdx(), *(a.GetIdx() for a in oh)}
    prefix = _organyl_units(mol, Chem, center.GetIdx(), roots, seen)
    if prefix is None or seen != set(range(mol.GetNumAtoms())):
        return None
    hydride = {17: 'chlorane', 35: 'bromane', 53: 'iodane'}[center.GetAtomicNum()]
    from .formatting import attach_suffix, multiplied_suffix
    suffix = multiplied_suffix(len(oh), 'ol')
    return _decision(mol, prefix + '-lambda' + str(valence) + '-' + attach_suffix(hydride, suffix),
                     (center.GetIdx(),), 'P-68.5.1/P-41')


def cyclic_sulfur_heterone(mol, Chem):
    """The five-membered maximally unsaturated sulfur oxide parent."""
    if not _clean(mol, Chem, {6, 8, 16}) or mol.GetNumAtoms() != 6:
        return None
    rings = mol.GetRingInfo().AtomRings()
    sulfur = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 16]
    oxygen = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 8]
    if len(rings) != 1 or len(rings[0]) != 5 or len(sulfur) != 1 or len(oxygen) != 1:
        return None
    s, o = sulfur[0], oxygen[0]
    ring = set(rings[0])
    if (s.GetIdx() not in ring or s.GetTotalNumHs(includeNeighbors=True) != 0
            or o.GetDegree() != 1 or _bond(mol, s.GetIdx(), o.GetIdx()) != 2
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in ring if i != s.GetIdx())):
        return None
    doubles = [b for b in mol.GetBonds() if b.GetBeginAtomIdx() in ring and b.GetEndAtomIdx() in ring
               and b.GetBondTypeAsDouble() == 2]
    if len(doubles) != 2 or len({i for b in doubles for i in (b.GetBeginAtomIdx(), b.GetEndAtomIdx())}) != 4:
        return None
    if any(mol.GetAtomWithIdx(i).GetTotalNumHs(includeNeighbors=True) != 1 for i in ring if i != s.GetIdx()):
        return None
    return _decision(mol, '1H-1lambda4-thiophen-1-one', (s.GetIdx(),), 'P-64.4.2/P-25.7.1.1')


def saturated_cyclic_sulfur_heterone(mol, Chem):
    """Unsubstituted saturated sulfur monocycle bearing one or two oxo groups."""
    if not _clean(mol, Chem, {6, 8, 16}):
        return None
    rings = mol.GetRingInfo().AtomRings()
    sulfurs = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 16]
    oxygens = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 8]
    if len(rings) != 1 or not 3 <= len(rings[0]) <= 8 or len(sulfurs) != 1 or len(oxygens) not in {1, 2}:
        return None
    s = sulfurs[0]
    ring = set(rings[0])
    if (s.GetIdx() not in ring or s.GetTotalNumHs(includeNeighbors=True) != 0
            or len(ring) + len(oxygens) != mol.GetNumAtoms()
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in ring if i != s.GetIdx())
            or any(o.GetDegree() != 1 or _bond(mol, s.GetIdx(), o.GetIdx()) != 2 for o in oxygens)
            or any(_bond(mol, a, b) != 1 for a in ring for b in ring if a < b and _bond(mol, a, b))):
        return None
    copy = Chem.RWMol(mol)
    for i in sorted((o.GetIdx() for o in oxygens), reverse=True):
        copy.RemoveAtom(i)
    hydride = _sanitize(copy, Chem)
    if hydride is None:
        return None
    from .api import mol_to_iupac
    parent = mol_to_iupac(hydride)
    if not parent.endswith('ane') or any(c in parent for c in ' -'):
        return None
    if len(oxygens) == 1:
        name = '1lambda4-' + parent[:-1] + '-1-one'
    else:
        name = '1lambda6-' + parent + '-1,1-dione'
    return _decision(mol, name, (s.GetIdx(),), 'P-64.4.2/P-41')


def substituted_cyclic_sulfur_heterone(mol, Chem):
    """A sulfur monocycle oxide with mapped carbon/halogen substituents."""
    if not _clean(mol, Chem, {6, 8, 9, 16, 17, 35, 53}):
        return None
    rings = mol.GetRingInfo().AtomRings()
    sulfurs = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 16]
    oxygens = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 8]
    if len(rings) != 1 or len(sulfurs) != 1 or len(oxygens) not in {1, 2}:
        return None
    sulfur = sulfurs[0]
    ring = set(rings[0])
    if (sulfur.GetIdx() not in ring or sulfur.GetTotalNumHs(includeNeighbors=True) != 0
            or not 3 <= len(ring) <= 8
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in ring if i != sulfur.GetIdx())
            or any(a.GetDegree() != 1 or _bond(mol, sulfur.GetIdx(), a.GetIdx()) != 2 for a in oxygens)):
        return None
    copy = Chem.RWMol(mol)
    for idx in sorted((a.GetIdx() for a in oxygens), reverse=True):
        copy.RemoveAtom(idx)
    hydride = _sanitize(copy, Chem)
    if hydride is None:
        return None
    ring_bonds = [b for b in hydride.GetBonds() if b.GetBeginAtom().IsInRing() and b.GetEndAtom().IsInRing()]
    if len(ring) == 5 and all(b.GetIsAromatic() for b in ring_bonds):
        parent, indicated = 'thiophene', '1H-'
    elif all(b.GetBondTypeAsDouble() == 1 for b in ring_bonds):
        parent = {3: 'thiirane', 4: 'thietane', 5: 'thiolane', 6: 'thiane',
                  7: 'thiepane', 8: 'thiocane'}[len(ring)]
        indicated = ''
    else:
        return None
    from .api import mol_to_iupac
    named = mol_to_iupac(hydride)
    if not named.endswith(parent):
        return None
    prefix = named[:-len(parent)]
    if prefix:
        prefix += '-'
    if len(oxygens) == 1:
        body = '1lambda4-' + parent[:-1] + '-1-one'
    else:
        body = '1lambda6-' + parent + '-1,1-dione'
    return _decision(mol, prefix + indicated + body, (sulfur.GetIdx(),), 'P-64.4.2/P-41')


def fused_sulfur_heterone(mol, Chem):
    """Mapped benzothiophene oxides and the retained thianthrene dioxide."""
    if not _clean(mol, Chem, {6, 8, 9, 16, 17, 35, 53}):
        return None
    oxygens = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 8]
    sulfurs = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 16]
    if len(oxygens) not in {1, 2} or len(sulfurs) not in {1, 2}:
        return None
    centers = [s for s in sulfurs if s.IsInRing() and all(
        o.GetDegree() == 1 and _bond(mol, s.GetIdx(), o.GetIdx()) == 2 for o in oxygens)]
    if len(centers) != 1:
        return None
    center = centers[0]
    if center.GetTotalNumHs(includeNeighbors=True) != 0 or sum(n.IsInRing() for n in center.GetNeighbors()) != 2:
        return None
    copy = Chem.RWMol(mol)
    removed = sorted((a.GetIdx() for a in oxygens), reverse=True)
    for idx in removed:
        copy.RemoveAtom(idx)
    hydride = _sanitize(copy, Chem)
    if hydride is None:
        return None
    rings = hydride.GetRingInfo().AtomRings()
    if len(sulfurs) == 1 and len(rings) == 2:
        from .api import mol_to_iupac
        parent = '1-benzothiophene'
        name = mol_to_iupac(hydride)
        if not name.endswith(parent):
            return None
        prefix = name[:-len(parent)]
        if prefix and not prefix.endswith('-'):
            prefix += '-'
        if len(oxygens) == 1:
            body = '1H-1lambda4-1-benzothiophen-1-one'
        else:
            body = '1H-1lambda6-1-benzothiophene-1,1-dione'
        return _decision(mol, prefix + body, (center.GetIdx(),), 'P-64.4.2/P-41')
    if (len(sulfurs) != 2 or len(oxygens) != 2 or len(rings) != 3
            or hydride.GetNumAtoms() != 14 or any(not a.IsInRing() for a in hydride.GetAtoms())):
        return None
    from .fusion_components import resolve_component
    from .fusion_networks import skeleton_key
    component = resolve_component('thianthrene')
    if component is None or component.atom_count != hydride.GetNumAtoms():
        return None
    labels = tuple(label for label, _ in component.label_elements)
    elements = tuple(element for _, element in component.label_elements)
    positions = {label: i for i, label in enumerate(labels)}
    expected_edges = tuple(tuple(positions[label] for label in edge) for edge in component.bonds)
    actual_edges = tuple((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in hydride.GetBonds())
    actual_numbers = tuple(a.GetAtomicNum() for a in hydride.GetAtoms())
    if skeleton_key(elements, expected_edges) != skeleton_key(actual_numbers, actual_edges):
        return None
    from .ring_state_parent import _reference_states
    actual_h = tuple(a.GetTotalNumHs(includeNeighbors=True) for a in hydride.GetAtoms())
    if not any(h == actual_h for h, _indicated, _bonds in _reference_states(actual_numbers, actual_edges)):
        return None
    return _decision(mol, '5H-lambda6-thianthrene-5,5-dione', (center.GetIdx(),), 'P-64.4.2/P-25.7')


def heteroimine_ester(mol, Chem):
    """A carbon ester with an N-organylimino Group 14/15 substituent.

    A bromine marker is used only on a private carbon/O graph to reuse the
    established ester parent and numbering. All original atom ownership and
    the heteroimine prefix are proved on the input graph.
    """
    hetero = {14: ('silyl', 1), 32: ('germyl', 1), 50: ('stannyl', 1), 82: ('plumbyl', 1),
              15: ('phosphanyl', 0), 33: ('arsanyl', 0), 51: ('stibanyl', 0), 83: ('bismuthanyl', 0)}
    if not _clean(mol, Chem, {6, 7, 8, *hetero}):
        return None
    sis = [a for a in mol.GetAtoms() if a.GetAtomicNum() in hetero]
    ns = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    if len(sis) != 1 or len(ns) != 1 or sum(a.GetAtomicNum() == 8 for a in mol.GetAtoms()) != 2:
        return None
    si, n = sis[0], ns[0]
    if (si.IsInRing() or si.GetDegree() != 2
            or si.GetTotalNumHs(includeNeighbors=True) != hetero[si.GetAtomicNum()][1]
            or n.GetDegree() != 2 or _bond(mol, si.GetIdx(), n.GetIdx()) != 2):
        return None
    nroots = [a for a in n.GetNeighbors() if a.GetIdx() != si.GetIdx()]
    attached = [a for a in si.GetNeighbors() if a.GetIdx() != n.GetIdx()]
    if (len(nroots) != 1 or len(attached) != 1 or nroots[0].GetAtomicNum() != 6
            or attached[0].GetAtomicNum() != 6 or _bond(mol, si.GetIdx(), attached[0].GetIdx()) != 1):
        return None
    from .component_mapping import mapped_branch
    from .phosphorus_ester_methods import _carbon_component
    n_group = _carbon_component(mol, nroots[0].GetIdx(), n.GetIdx())
    n_word, owned = mapped_branch(mol, Chem, nroots[0].GetIdx(), n.GetIdx())
    if not n_group or frozenset(owned) != n_group or not n_word or not n_word.isalpha():
        return None
    copy = Chem.RWMol(mol)
    marker = copy.GetAtomWithIdx(si.GetIdx())
    marker.SetAtomicNum(35)
    marker.SetNoImplicit(True)
    marker.SetNumExplicitHs(0)
    for idx in sorted(set(n_group) | {n.GetIdx()}, reverse=True):
        copy.RemoveAtom(idx)
    probe = _sanitize(copy, Chem)
    if probe is None:
        return None
    from .extended_inventory import selection_inventory
    inventory = selection_inventory(probe)
    if not inventory.groups or any(g.kind != 'ester' for g in inventory.groups) or len(inventory.groups) != 1:
        return None
    from .api import mol_to_iupac
    name = mol_to_iupac(probe)
    if name.count('bromo') != 1 or ' ' not in name or not name.endswith('ate'):
        return None
    prefix = '[(' + n_word + 'imino)' + hetero[si.GetAtomicNum()][0] + ']'
    deleted = set(n_group) | {n.GetIdx()}
    remaining = [i for i in range(mol.GetNumAtoms()) if i not in deleted]
    original_center = remaining[inventory.groups[0].center]
    return _decision(mol, name.replace('bromo', prefix, 1),
                     (original_center,), 'P-62.3.1.3/P-41', 'functional-class')


def modified_diacid_diester(mol, Chem):
    """Diesters with matching chalcogen replacements at both acid sites."""
    chalcogens = {8, 16, 34, 52}
    if not _clean(mol, Chem, {6} | chalcogens):
        return None
    from .phosphorus_ester_methods import _carbon_component
    from .component_mapping import mapped_branch
    from .formatting import format_prefixes, prefix_alphabetization_key
    sites = []
    for c in mol.GetAtoms():
        if c.GetAtomicNum() != 6 or c.GetDegree() != 3:
            continue
        double = [a for a in c.GetNeighbors() if a.GetAtomicNum() in chalcogens and a.GetDegree() == 1
                  and _bond(mol, c.GetIdx(), a.GetIdx()) == 2]
        bridges = [a for a in c.GetNeighbors() if a.GetAtomicNum() in chalcogens and a.GetDegree() == 2
                   and _bond(mol, c.GetIdx(), a.GetIdx()) == 1]
        if len(double) != 1 or len(bridges) != 1:
            continue
        bridge = bridges[0]
        roots = [a for a in bridge.GetNeighbors() if a.GetIdx() != c.GetIdx() and a.GetAtomicNum() == 6]
        others = [a for a in c.GetNeighbors() if a.GetIdx() not in {double[0].GetIdx(), bridge.GetIdx()}]
        if len(roots) != 1 or len(others) != 1 or others[0].GetAtomicNum() != 6:
            continue
        root = roots[0]
        group = _carbon_component(mol, root.GetIdx(), bridge.GetIdx())
        word, owned = mapped_branch(mol, Chem, root.GetIdx(), bridge.GetIdx())
        if not group or not word or frozenset(owned) != group:
            return None
        rooted = Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(group),
                                          rootedAtAtom=root.GetIdx(), canonical=True,
                                          isomericSmiles=True)
        sites.append((c.GetIdx(), double[0].GetIdx(), bridge.GetIdx(), word, group, rooted))
    if len(sites) != 2 or sites[0][4] & sites[1][4]:
        return None
    types = [(mol.GetAtomWithIdx(row[1]).GetAtomicNum(), mol.GetAtomWithIdx(row[2]).GetAtomicNum()) for row in sites]
    if types[0] != types[1] or types[0] == (8, 8):
        return None
    if len([a for a in mol.GetAtoms() if a.GetAtomicNum() in chalcogens]) != 4:
        return None
    same_organyl = sites[0][5] == sites[1][5]
    if same_organyl != (sites[0][3] == sites[1][3]):
        return None
    copy = Chem.RWMol(mol)
    for _c, double, bridge, _word, _group, _rooted in sites:
        for idx in (double, bridge):
            atom = copy.GetAtomWithIdx(idx)
            atom.SetAtomicNum(8)
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(idx == double)
    removed = set(sites[0][4]) | set(sites[1][4])
    for idx in sorted(removed, reverse=True):
        copy.RemoveAtom(idx)
    acid = _sanitize(copy, Chem)
    if acid is None:
        return None
    if not same_organyl:
        # Distinct ester groups need site locants unless the two acid ends
        # are related by an automorphism of the normalized parent acid.
        remaining = [i for i in range(mol.GetNumAtoms()) if i not in removed]
        ranks = Chem.CanonicalRankAtoms(acid, breakTies=False)
        centers = [remaining.index(row[0]) for row in sites]
        if ranks[centers[0]] != ranks[centers[1]]:
            return None
    from .api import mol_to_iupac
    base = mol_to_iupac(acid)
    if base == 'oxalic acid':
        base = 'ethanedioic acid'
    if base.endswith('dicarboxylic acid'):
        acid_stem = base[:-len('dicarboxylic acid')]
        ring_acid = True
    elif base.endswith('dioic acid'):
        acid_stem = base[:-len('dioic acid')]
        ring_acid = False
    else:
        return None
    element = {8: 'O', 16: 'S', 34: 'Se', 52: 'Te'}[types[0][1]]
    mixed_elements = types[0][0] != types[0][1]
    if same_organyl:
        loc = element if mixed_elements else ''
        if loc:
            words = format_prefixes([(loc, sites[0][3]), (loc, sites[1][3])], hierarchical_enclosures=True)
        else:
            word = sites[0][3]
            words = 'di' + word if word.isalpha() else 'bis(' + word + ')'
    else:
        words = []
        for word in sorted((sites[0][3], sites[1][3]), key=prefix_alphabetization_key):
            rendered = word if word.isalpha() else '(' + word + ')'
            words.append((element + '-' if mixed_elements else '') + rendered)
        words = ' '.join(words)
    prefixes = {16: 'thio', 34: 'seleno', 52: 'telluro'}
    replacements = sorted(prefixes[element] for element in types[0] if element != 8)
    if len(replacements) == 2 and replacements[0] == replacements[1]:
        replacement = 'di' + replacements[0]
    else:
        replacement = ''.join(replacements)
    if ring_acid:
        name = words + ' ' + acid_stem + 'dicarbo' + replacement + 'ate'
    else:
        name = words + ' ' + acid_stem + 'bis(' + replacement + 'ate)'
    return _decision(mol, name, (sites[0][0], sites[1][0]), 'P-65.6.3.3.7.1/P-41', 'functional-class')


def bicyclic_aza_cation(mol, Chem):
    """A substituted bridgehead N of a saturated aza-bicyclic hydride."""
    if (mol.GetNumAtoms() != mol.GetNumHeavyAtoms() or mol.GetNumHeavyAtoms() > 48
            or len(Chem.GetMolFrags(mol)) != 1 or mol.GetStereoGroups()
            or any(a.GetAtomicNum() not in {6, 7} or a.GetIsotope() or a.GetNumRadicalElectrons()
                   or int(a.GetChiralTag()) for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or b.GetBondTypeAsDouble() != 1 for b in mol.GetBonds())):
        return None
    ns = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    if len(ns) != 1 or ns[0].GetFormalCharge() != 1 or any(a.GetFormalCharge() for a in mol.GetAtoms() if a.GetIdx() != ns[0].GetIdx()):
        return None
    n = ns[0]
    roots = [a for a in n.GetNeighbors() if not a.IsInRing()]
    if not n.IsInRing() or n.GetDegree() != 4 or len(roots) != 1 or roots[0].GetAtomicNum() != 6:
        return None
    from .component_mapping import mapped_branch
    from .phosphorus_ester_methods import _carbon_component
    from .formatting import format_prefixes
    branch = _carbon_component(mol, roots[0].GetIdx(), n.GetIdx())
    text, mapped = mapped_branch(mol, Chem, roots[0].GetIdx(), n.GetIdx())
    if not branch or frozenset(mapped) != branch or not text or any(mol.GetAtomWithIdx(i).IsInRing() for i in branch):
        return None
    if any(not a.IsInRing() and a.GetIdx() not in branch for a in mol.GetAtoms()):
        return None
    copy = Chem.RWMol(mol)
    copy.GetAtomWithIdx(n.GetIdx()).SetFormalCharge(0)
    for idx in sorted(branch, reverse=True):
        copy.RemoveAtom(idx)
    neutral = _sanitize(copy, Chem)
    if neutral is None:
        return None
    from .api import mol_to_iupac
    name = mol_to_iupac(neutral)
    if not name.startswith('1-azabicyclo[') or not name.endswith('ane'):
        return None
    prefix = format_prefixes([('1', text)], hierarchical_enclosures=True)
    return _decision(mol, prefix + '-' + name[:-1] + '-1-ium', (n.GetIdx(),), 'P-73.4/P-41')


def group14_hydride_multiplication(mol, Chem):
    """Identical group-14/15 hydrides on distinct benzene positions."""
    parents = {14: ('silane', 4), 32: ('germane', 4), 50: ('stannane', 4), 82: ('plumbane', 4),
               15: ('phosphane', 3), 33: ('arsane', 3), 51: ('stibane', 3), 83: ('bismuthane', 3)}
    if not _clean(mol, Chem, {6, *parents}):
        return None
    rings = mol.GetRingInfo().AtomRings()
    hydrides = [a for a in mol.GetAtoms() if a.GetAtomicNum() in parents]
    if len(rings) != 1 or len(rings[0]) != 6 or not 2 <= len(hydrides) <= 6:
        return None
    ring = set(rings[0])
    if (len({a.GetAtomicNum() for a in hydrides}) != 1
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 or not mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)):
        return None
    attachments, used, methyl_counts = [], set(ring), []
    for atom in hydrides:
        hosts = [a for a in atom.GetNeighbors() if a.GetIdx() in ring]
        methyls = [a for a in atom.GetNeighbors() if a.GetIdx() not in ring]
        valence = parents[atom.GetAtomicNum()][1]
        if (len(hosts) != 1 or len(methyls) > valence - 1 or atom.GetTotalValence() != valence
                or atom.GetTotalNumHs(includeNeighbors=True) != valence - 1 - len(methyls)
                or _bond(mol, atom.GetIdx(), hosts[0].GetIdx()) != 1
                or any(a.GetAtomicNum() != 6 or a.GetDegree() != 1
                       or a.GetTotalNumHs(includeNeighbors=True) != 3
                       or _bond(mol, atom.GetIdx(), a.GetIdx()) != 1 for a in methyls)):
            return None
        attachments.append(hosts[0].GetIdx())
        methyl_counts.append(len(methyls))
        used.update({atom.GetIdx(), *(a.GetIdx() for a in methyls)})
    if (len(set(attachments)) != len(hydrides) or len(set(methyl_counts)) != 1
            or used != set(range(mol.GetNumAtoms()))):
        return None
    locants = []
    for start in ring:
        for neighbor in mol.GetAtomWithIdx(start).GetNeighbors():
            if neighbor.GetIdx() not in ring:
                continue
            path = [start, neighbor.GetIdx()]
            while len(path) < 6:
                nexts = [a.GetIdx() for a in mol.GetAtomWithIdx(path[-1]).GetNeighbors()
                         if a.GetIdx() in ring and a.GetIdx() != path[-2] and a.GetIdx() not in path]
                if len(nexts) != 1:
                    break
                path.append(nexts[0])
            if len(path) == 6:
                locants.append(tuple(sorted(path.index(i) + 1 for i in attachments)))
    if not locants:
        return None
    count = len(hydrides)
    positions = ','.join(map(str, min(locants)))
    # P-29 retains phenylene for a divalent benzene unit. Higher-valence
    # linkers use the systematic benzene-triyl/tetrayl names.
    central = (positions + '-phenylene' if count == 2 else 'benzenehexayl' if count == 6 else
               'benzene-' + positions + '-' + {3: 'triyl', 4: 'tetrayl', 5: 'pentayl'}[count])
    mult = {2: 'bis', 3: 'tris', 4: 'tetrakis', 5: 'pentakis', 6: 'hexakis'}[count]
    methyl = {0: '', 1: 'methyl', 2: 'dimethyl', 3: 'trimethyl'}[methyl_counts[0]]
    parent = methyl + parents[hydrides[0].GetAtomicNum()][0]
    return _decision(mol, '(' + central + ')' + mult + '(' + parent + ')', (),
                     'P-15.3.2.3/P-29/P-51.3', 'multiplicative')


def acyclic_hydride_multiplication(mol, Chem):
    """Identical Si/Ge/Sn/Pb/P/As/Sb/Bi hydrides joined by one carbon path."""
    parents = {14: ('silane', 4), 32: ('germane', 4), 50: ('stannane', 4), 82: ('plumbane', 4),
               15: ('phosphane', 3), 33: ('arsane', 3), 51: ('stibane', 3), 83: ('bismuthane', 3)}
    if (not _clean(mol, Chem, {6, *parents}) or any(a.IsInRing() for a in mol.GetAtoms())
            or any(b.GetBondTypeAsDouble() != 1 for b in mol.GetBonds())):
        return None
    hydrides = [a for a in mol.GetAtoms() if a.GetAtomicNum() in parents]
    if not 2 <= len(hydrides) <= 6 or len({a.GetAtomicNum() for a in hydrides}) != 1:
        return None
    if any(any(n.GetAtomicNum() != 6 for n in a.GetNeighbors()) for a in hydrides):
        return None
    carbon = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6}
    components = []
    remaining = set(carbon)
    while remaining:
        stack, group = [min(remaining)], set()
        while stack:
            idx = stack.pop()
            if idx in group:
                continue
            group.add(idx)
            stack.extend(n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                         if n.GetIdx() in carbon and n.GetIdx() not in group)
        remaining -= group
        components.append(group)
    candidates = [group for group in components if all(
        sum(n.GetIdx() in group for n in atom.GetNeighbors()) == 1 for atom in hydrides)]
    if len(candidates) != 1:
        return None
    linker = candidates[0]
    adjacency = {i: [n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in linker]
                 for i in linker}
    if any(len(neighbors) > 2 for neighbors in adjacency.values()):
        return None
    if len(linker) == 1:
        path = list(linker)
    else:
        ends = [i for i, neighbors in adjacency.items() if len(neighbors) == 1]
        if len(ends) != 2:
            return None
        path = [ends[0]]
        while len(path) < len(linker):
            choices = [i for i in adjacency[path[-1]] if i not in path]
            if len(choices) != 1:
                return None
            path.append(choices[0])
    used, attachments, methyl_counts = set(linker), [], []
    for atom in hydrides:
        hosts = [n for n in atom.GetNeighbors() if n.GetIdx() in linker]
        methyls = [n for n in atom.GetNeighbors() if n.GetIdx() not in linker]
        valence = parents[atom.GetAtomicNum()][1]
        if (len(hosts) != 1 or len(methyls) > valence - 1 or atom.GetTotalValence() != valence
                or atom.GetTotalNumHs(includeNeighbors=True) != valence - 1 - len(methyls)
                or any(a.GetAtomicNum() != 6 or a.GetDegree() != 1
                       or a.GetTotalNumHs(includeNeighbors=True) != 3 for a in methyls)):
            return None
        attachments.append(hosts[0].GetIdx())
        methyl_counts.append(len(methyls))
        used.update({atom.GetIdx(), *(a.GetIdx() for a in methyls)})
    if len(set(methyl_counts)) != 1 or used != set(range(mol.GetNumAtoms())):
        return None
    from .constants import ROOTS
    count = len(hydrides)
    if len(path) == 1:
        if count != 2:
            return None
        central = 'methylene'
    else:
        root = ROOTS.get(len(path))
        if root is None:
            return None
        locants = min(tuple(sorted(order.index(i) + 1 for i in attachments))
                      for order in (path, path[::-1]))
        ending = {2: 'diyl', 3: 'triyl', 4: 'tetrayl', 5: 'pentayl', 6: 'hexayl'}[count]
        central = '(' + root + 'ane-' + ','.join(map(str, locants)) + '-' + ending + ')'
    mult = {2: 'bis', 3: 'tris', 4: 'tetrakis', 5: 'pentakis', 6: 'hexakis'}[count]
    methyl = {0: '', 1: 'methyl', 2: 'dimethyl', 3: 'trimethyl'}[methyl_counts[0]]
    parent = methyl + parents[hydrides[0].GetAtomicNum()][0]
    return _decision(mol, central + mult + '(' + parent + ')', (),
                     'P-15.3.2.3/P-68.3.2.3.2/P-51.3', 'multiplicative')


def methylene_bis_disilane(mol, Chem):
    """The source-proved methylene linkage of two complete disilane parents."""
    if not _clean(mol, Chem, {6, 14}) or mol.GetNumAtoms() != 5:
        return None
    carbons = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 6]
    silicons = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 14]
    if len(carbons) != 1 or len(silicons) != 4:
        return None
    c = carbons[0]
    if c.GetDegree() != 2 or c.GetTotalNumHs(includeNeighbors=True) != 2:
        return None
    inner = list(c.GetNeighbors())
    if any(a.GetAtomicNum() != 14 or a.GetDegree() != 2
           or a.GetTotalNumHs(includeNeighbors=True) != 2 or _bond(mol, c.GetIdx(), a.GetIdx()) != 1
           for a in inner):
        return None
    ends = []
    for atom in inner:
        outer = [n for n in atom.GetNeighbors() if n.GetIdx() != c.GetIdx()]
        if (len(outer) != 1 or outer[0].GetAtomicNum() != 14 or outer[0].GetDegree() != 1
                or outer[0].GetTotalNumHs(includeNeighbors=True) != 3
                or _bond(mol, atom.GetIdx(), outer[0].GetIdx()) != 1):
            return None
        ends.append(outer[0].GetIdx())
    if len(set(ends)) != 2:
        return None
    return _decision(mol, "1,1'-methylenebis(disilane)", (),
                     'P-15.3.2.1/P-15.3.2.3', 'multiplicative')


def mixed_cation_acid_salt(fragments, components):
    """P-65.6.2.1 for a neutral salt of one fully ionized oxoacid species.

    Component names, charge balance and original atom ownership have already
    been established by the ionic assembly before this extension is invoked.
    """
    from .constants import MULT
    from .formatting import COMPLEX_MULT, enclose_name_unit, prefix_alphabetization_key
    pairs = tuple(zip(fragments, components, strict=True))
    cations = [component for _fragment, component in pairs if component.charge > 0]
    anions = [(fragment, component) for fragment, component in pairs if component.charge < 0]
    if (len({(c.charge, c.identity) for c in cations}) < 2 or not anions
            or len({(c.charge, c.identity) for _f, c in anions}) != 1):
        return ''
    for fragment, component in anions:
        charged = [a for a in fragment.GetAtoms() if a.GetFormalCharge()]
        if (not charged or component.charge != -len(charged)
                or any(a.GetAtomicNum() != 8 or a.GetFormalCharge() != -1 or a.GetDegree() != 1
                       for a in charged)):
            return ''
        centers = {a.GetNeighbors()[0].GetIdx() for a in charged}
        for index in centers:
            center = fragment.GetAtomWithIdx(index)
            if center.GetAtomicNum() not in {6, 15, 16, 34, 52}:
                return ''
            oxos = [a for a in center.GetNeighbors() if a.GetAtomicNum() == 8
                    and a.GetDegree() == 1 and _bond(fragment, index, a.GetIdx()) == 2]
            residual_acid = [a for a in center.GetNeighbors() if a.GetAtomicNum() == 8
                             and a.GetDegree() == 1 and a.GetTotalNumHs(includeNeighbors=True) == 1
                             and _bond(fragment, index, a.GetIdx()) == 1]
            if not oxos or residual_acid:
                return ''
    counts = Counter(c for _fragment, c in pairs)

    def term(component):
        count = counts[component]
        if count == 1:
            return component.name
        complex_unit = (component.substituted or component.charge > 0 and component.organic
                        or any(c.isdigit() or c in '-()[]{}' for c in component.name))
        multiplier = (COMPLEX_MULT if complex_unit else MULT).get(count)
        if not multiplier:
            return ''
        return multiplier + enclose_name_unit(component.name) if complex_unit else multiplier + component.name

    positive = sorted({c for c in cations}, key=lambda c: prefix_alphabetization_key(c.name))
    negative = anions[0][1]
    terms = [term(c) for c in positive] + [term(negative)]
    return ' '.join(terms) if all(terms) else ''


def chalcogen_terminal_suffix(mol, Chem):
    """Keep a two/three-atom homogeneous parent block before terminal -EH."""
    if not _clean(mol, Chem, {6, 8, 16, 34, 52}):
        return None
    from .heterochalcogen_chain_methods import _ordered_path, _organyl_component
    from .component_mapping import MappedComponent, decorate_component, mapped_branch
    from .constants import MULT
    from .formatting import attach_suffix
    hydrides = {8: 'oxidane', 16: 'sulfane', 34: 'selane', 52: 'tellane'}
    suffixes = {8: 'ol', 16: 'thiol', 34: 'selenol', 52: 'tellurol'}
    hetero = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in hydrides}
    carbons = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6}
    if not carbons or len(hetero) < 3:
        return None
    chain = _ordered_path(mol, hetero)
    if len(chain) != len(hetero):
        return None
    if any(mol.GetAtomWithIdx(i).IsInRing() or mol.GetAtomWithIdx(i).GetTotalValence() != 2 for i in chain):
        return None
    if any(_bond(mol, a, b) != 1 for a, b in zip(chain, chain[1:])):
        return None
    links = [(i, a.GetIdx()) for i in chain for a in mol.GetAtomWithIdx(i).GetNeighbors() if a.GetAtomicNum() == 6]
    if len(links) != 1 or links[0][0] not in {chain[0], chain[-1]}:
        return None
    if links[0][0] == chain[-1]:
        chain = tuple(reversed(chain))
    if (mol.GetAtomWithIdx(chain[-1]).GetTotalNumHs(includeNeighbors=True) != 1
            or any(mol.GetAtomWithIdx(i).GetTotalNumHs(includeNeighbors=True) for i in chain[:-1])):
        return None
    if _organyl_component(mol, links[0][1], chain[0], carbons) != frozenset(carbons):
        return None
    sequence = tuple(mol.GetAtomWithIdx(i).GetAtomicNum() for i in chain)
    if sequence == (8, 16, 8):
        # P-68.4.2.2's R-O-S-OH construction uses the specified SO-peroxol
        # suffix on oxidane, rather than splitting off the terminal oxygen.
        prefix, owned = mapped_branch(mol, Chem, links[0][1], chain[0])
        if not prefix or frozenset(owned) != frozenset(carbons):
            return None
        name = decorate_component(mol, Chem, MappedComponent(
            'oxidane-SO-thioperoxol', ((chain[0], '1'),), frozenset({chain[0]}),
            prefixes=(('', prefix),), embedded_atoms=frozenset(carbons | set(chain[1:])),
            systematic_parent=True, prefix_mode='unlocanted'))
        return _decision(mol, name, (chain[-1],), 'P-68.4.2.2/P-41') if name else None
    if sequence[-1] == sequence[-2]:
        return None  # Terminal pairs belong to the existing peroxol route.
    seniority = {8: 0, 16: 1, 34: 2, 52: 3}
    if (len(sequence) % 2 == 1 and len(set(sequence)) == 2
            and seniority[sequence[0]] > seniority[sequence[1]]
            and all(sequence[i] == sequence[i % 2] for i in range(len(sequence)))):
        return None  # Alternating preselected a[ba]n parent hydride.
    start = len(chain) - 2
    while start and sequence[start - 1] == sequence[start]:
        start -= 1
    base = chain[start:-1]
    if not 1 <= len(base) <= 3:
        return None
    # Do not break an earlier repeated chalcogen block through an unsupported
    # generic prefix writer. Single-atom upstream chains remain composable.
    if any(sequence[i] == sequence[i + 1] for i in range(max(0, start - 1))):
        return None
    prefix_root = chain[start - 1] if start else links[0][1]
    prefix_atoms = frozenset(carbons | set(chain[:start]))
    prefix, owned = mapped_branch(mol, Chem, prefix_root, base[0])
    if not prefix or frozenset(owned) != prefix_atoms:
        return None
    parent = (MULT[len(base)] if len(base) > 1 else '') + hydrides[sequence[start]]
    parent = attach_suffix(parent, suffixes[sequence[-1]])
    mapping = tuple((idx, str(i)) for i, idx in enumerate(reversed(base), 1))
    name = decorate_component(mol, Chem, MappedComponent(
        parent, mapping, frozenset(base), prefixes=(('', prefix),),
        embedded_atoms=frozenset(prefix_atoms | {chain[-1]}),
        systematic_parent=True, prefix_mode='unlocanted'))
    if not name:
        return None
    return _decision(mol, name, (chain[-1],), 'P-68.4.2.2/P-41')


def cyanohydrazide(mol, Chem):
    """Retained cyanic-acid hydrazide, senior to hydrazinecarbonitrile."""
    if not _clean(mol, Chem, {6, 7}) or mol.GetNumAtoms() != 4:
        return None
    carbons = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 6]
    if len(carbons) != 1:
        return None
    carbon = carbons[0]
    triple = [a for a in carbon.GetNeighbors() if a.GetAtomicNum() == 7 and a.GetDegree() == 1 and _bond(mol, carbon.GetIdx(), a.GetIdx()) == 3]
    single = [a for a in carbon.GetNeighbors() if a.GetAtomicNum() == 7 and a.GetDegree() == 2 and a.GetTotalNumHs(includeNeighbors=True) == 1 and _bond(mol, carbon.GetIdx(), a.GetIdx()) == 1]
    if len(triple) != 1 or len(single) != 1:
        return None
    ends = [a for a in single[0].GetNeighbors() if a.GetIdx() != carbon.GetIdx() and a.GetAtomicNum() == 7 and a.GetDegree() == 1 and a.GetTotalNumHs(includeNeighbors=True) == 2 and _bond(mol, single[0].GetIdx(), a.GetIdx()) == 1]
    if len(ends) != 1:
        return None
    return _decision(mol, 'cyanohydrazide', (carbon.GetIdx(),), 'P-66.5.1.1.3/P-66.3.1.2.1')


def cyclic_ketene(mol, Chem):
    """An exocyclic ketene whose ring is an all-carbon ylidene group."""
    if not _clean(mol, Chem, {6, 8}):
        return None
    oxos = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 8]
    if len(oxos) != 1 or oxos[0].GetDegree() != 1:
        return None
    oxygen = oxos[0]
    carbonyl = oxygen.GetNeighbors()[0]
    if (carbonyl.GetAtomicNum() != 6 or carbonyl.IsInRing() or carbonyl.GetDegree() != 2
            or _bond(mol, carbonyl.GetIdx(), oxygen.GetIdx()) != 2):
        return None
    roots = [a for a in carbonyl.GetNeighbors() if a.GetIdx() != oxygen.GetIdx()
             and a.GetAtomicNum() == 6 and a.IsInRing()
             and _bond(mol, carbonyl.GetIdx(), a.GetIdx()) == 2]
    if len(roots) != 1:
        return None
    from .component_mapping import mapped_branch
    root = roots[0].GetIdx()
    text, owned = mapped_branch(mol, Chem, root, carbonyl.GetIdx())
    if (not text or not text.endswith('ylidene') or not owned
            or set(owned) | {oxygen.GetIdx(), carbonyl.GetIdx()} != set(range(mol.GetNumAtoms()))
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in owned)):
        return None
    prefix = text if text.isalpha() else '(' + text + ')'
    return _decision(mol, prefix + 'methanone', (carbonyl.GetIdx(),), 'P-64.2.2.4/P-41')


def select(mol, Chem):
    """Return the first completely proved source-directed construction."""
    from .full_bluebook_followup import select as followup_select
    followup = followup_select(mol, Chem)
    if followup is not None:
        return followup
    numbers = Counter(a.GetAtomicNum() for a in iter_atoms(mol))
    has_charge = any(a.GetFormalCharge() for a in iter_atoms(mol))
    if has_charge:
        return aromatic_benzenide(mol, Chem) or symmetric_cycloalkanide(mol, Chem) or bicyclic_aza_cation(mol, Chem)
    if numbers[7] and numbers[6]:
        decision = cyanohydrazide(mol, Chem)
        if decision:
            return decision
        if any(numbers[z] for z in (5, 14, 15, 32, 33, 50, 51, 82, 83)):
            decision = hetero_carbonitrile(mol, Chem)
            if decision:
                return decision
        if numbers[8] or any(numbers[z] for z in (16, 34, 52)):
            decision = cyanic_chalcogen_acid(mol, Chem)
            if decision:
                return decision
        if numbers[8] and numbers[16]:
            for fn in (o_sulfinyl_hydroxylamine,
                       senior_function_with_o_sulfinyl_hydroxylamine, sulfinimidic_acid):
                decision = fn(mol, Chem)
                if decision:
                    return decision
        if numbers[8]:
            for fn in (carbohydrazonic_acid, halocarbonic_amide, carbamoyl_pseudohalide, polyazane_acid):
                decision = fn(mol, Chem)
                if decision:
                    return decision
    if sum(numbers[z] for z in (8, 16, 34, 52)) >= 3:
        decision = chalcogen_terminal_suffix(mol, Chem)
        if decision:
            return decision
    if numbers[7] and numbers[8] and numbers[16]:
        decision = sulfamoyl_halide(mol, Chem)
        if decision:
            return decision
    if numbers[8] and numbers[16]:
        decision = (cyclic_sulfur_heterone(mol, Chem)
                    or saturated_cyclic_sulfur_heterone(mol, Chem)
                    or substituted_cyclic_sulfur_heterone(mol, Chem)
                    or fused_sulfur_heterone(mol, Chem))
        if decision:
            return decision
    if any(numbers[z] >= 2 for z in (16, 34, 52)) and numbers[6]:
        decision = modified_diacid_diester(mol, Chem)
        if decision:
            return decision
    if numbers[8] == 1 and numbers[6] >= 3:
        decision = cyclic_ketene(mol, Chem)
        if decision:
            return decision
    if (numbers[7] and numbers[8] >= 2
            and any(numbers[z] for z in (14, 15, 32, 33, 50, 51, 82, 83))):
        decision = heteroimine_ester(mol, Chem)
        if decision:
            return decision
    if any(numbers[z] >= 2 for z in (14, 15, 32, 33, 50, 51, 82, 83)):
        decision = (group14_hydride_multiplication(mol, Chem)
                    or acyclic_hydride_multiplication(mol, Chem)
                    or methylene_bis_disilane(mol, Chem))
        if decision:
            return decision
    if numbers[8] and any(numbers[z] for z in (17, 35, 53)):
        return hydroxy_halogen_parent(mol, Chem)
    return None
