"""Mapped, source-bounded P-22, P-44, P-62, P-65, P-66 and P-82 methods."""
from .molecule_iteration import iter_atoms, iter_bonds
from .naming_methods import MethodDecision


def _plain(mol, Chem, allowed, charges=(0,), limit=80):
    return (mol.GetNumHeavyAtoms() <= limit and len(Chem.GetMolFrags(mol)) == 1
            and not mol.GetStereoGroups()
            and all(a.GetAtomicNum() in allowed and a.GetFormalCharge() in charges
                    and not (a.GetIsotope() or a.GetNumRadicalElectrons()
                             or int(a.GetChiralTag()) or a.HasQuery()) for a in iter_atoms(mol))
            and all(not (int(b.GetStereo()) or int(b.GetBondDir()) or b.HasQuery())
                    for b in iter_bonds(mol)))


def _sanitize(copy, Chem):
    result = copy.GetMol()
    try:
        Chem.SanitizeMol(result)
    except (ValueError, RuntimeError):
        return None
    return result


def _out(mol, name, centers, rule, method='substitutive'):
    atoms = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    return MethodDecision(name, method, frozenset(centers), atoms, rule, (),
                          component_roles=(('source-graph', atoms, ()),))


def aryl_aminium_decision(mol, Chem):
    """Unsubstituted phenyl parent with carbon-only N substituents."""
    if not _plain(mol, Chem, {6, 7}, charges=(0, 1)):
        return None
    ns = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    rings = mol.GetRingInfo().AtomRings()
    if len(ns) != 1:
        return None
    n = ns[0]
    if n.GetDegree() == 4 and len(rings) == 4:
        phenyls = [frozenset(ring) for ring in rings if len(ring) == 6
                   and all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6
                           and mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)]
        nidx = n.GetIdx()
        if (n.GetFormalCharge() == 1 and len(phenyls) == 4 and mol.GetNumAtoms() == 25
                and len(frozenset().union(*phenyls)) == 24
                and {a.GetIdx() for a in n.GetNeighbors()}
                == {i for ring in phenyls for i in ring
                    if mol.GetBondBetweenAtoms(i, nidx) is not None}
                and all([other.GetIdx() for i in ring for other in mol.GetAtomWithIdx(i).GetNeighbors()
                         if other.GetIdx() not in ring] == [nidx] for ring in phenyls)):
            return _out(mol, 'N,N,N-triphenylanilinium', (nidx,), 'P-73.1.2.1/P-62.6.1')
    if len(rings) != 1 or len(rings[0]) != 6:
        return None
    ring = frozenset(rings[0])
    if (n.GetFormalCharge() != 1 or n.IsInRing() or n.GetTotalValence() != 4
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 or not mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)):
        return None
    attached = [a for a in n.GetNeighbors() if a.GetIdx() in ring]
    if len(attached) != 1 or any(a.GetIdx() not in ring and a.GetIdx() != n.GetIdx()
                                  for i in ring for a in mol.GetAtomWithIdx(i).GetNeighbors()):
        return None
    roots = [a.GetIdx() for a in n.GetNeighbors() if a.GetIdx() not in ring]
    if len(roots) > 3 or len(roots) + n.GetTotalNumHs() != 3:
        return None
    from .phosphorus_ester_methods import _carbon_component
    from .component_mapping import mapped_branch
    from .formatting import format_prefixes, join_prefix_parent
    used = set(ring) | {n.GetIdx()}
    prefixes = []
    for root in roots:
        group = _carbon_component(mol, root, n.GetIdx())
        if group is None or group & used:
            return None
        text, owned = mapped_branch(mol, Chem, root, n.GetIdx())
        if not text or frozenset(owned) != group:
            return None
        prefixes.append(('N', text))
        used.update(group)
    if used != set(range(mol.GetNumAtoms())):
        return None
    name = join_prefix_parent(format_prefixes(prefixes), 'anilinium') if prefixes else 'anilinium'
    return _out(mol, name, (n.GetIdx(),), 'P-73.1.2.1/P-62.6.1')


def simple_amine_oxide_decision(mol, Chem):
    """Single tertiary amine oxide with a fully mapped carbon-only parent."""
    if not _plain(mol, Chem, {6, 7, 8}, charges=(-1, 0, 1)):
        return None
    ns = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    oxygens = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 8]
    if len(ns) != 1 or len(oxygens) != 1:
        return None
    n, o = ns[0], oxygens[0]
    bond = mol.GetBondBetweenAtoms(n.GetIdx(), o.GetIdx())
    if (n.GetFormalCharge() != 1 or o.GetFormalCharge() != -1 or n.GetDegree() != 4
            or o.GetDegree() != 1 or bond is None or bond.GetBondTypeAsDouble() != 1
            or n.GetIsAromatic() or n.GetTotalNumHs() != 0
            or any(a.GetAtomicNum() != 6 for a in n.GetNeighbors() if a.GetIdx() != o.GetIdx())):
        return None
    copy = Chem.RWMol(mol)
    copy.GetAtomWithIdx(n.GetIdx()).SetFormalCharge(0)
    copy.RemoveAtom(o.GetIdx())
    amine = _sanitize(copy, Chem)
    if amine is None:
        return None
    from .api import mol_to_iupac
    name = mol_to_iupac(amine)
    if n.IsInRing():
        # For these simple one-N saturated rings the ring nitrogen has locant 1.
        if not name.endswith(('piperidine', 'pyrrolidine', 'azepane', 'azocane', 'azetidine')):
            return None
        ending = ' 1-oxide'
    elif name.endswith(('amine', 'aniline')):
        ending = ' N-oxide'
    else:
        return None
    return _out(mol, name + ending, (n.GetIdx(),), 'P-62.5/P-41', 'functional-class')


def ring_amidine_decision(mol, Chem):
    """Cyclic amidines via the corresponding completely selected amide."""
    if not _plain(mol, Chem, {6, 7, 8, 9, 17, 35, 53}) or not mol.GetRingInfo().NumRings():
        return None
    sites = []
    for c in mol.GetAtoms():
        if c.GetAtomicNum() != 6 or c.IsInRing():
            continue
        double_n = [n for n in c.GetNeighbors() if n.GetAtomicNum() == 7
                    and n.GetDegree() in {1, 2} and n.GetTotalNumHs() in {0, 1}
                    and mol.GetBondBetweenAtoms(c.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 2]
        single_n = [n for n in c.GetNeighbors() if n.GetAtomicNum() == 7
                    and mol.GetBondBetweenAtoms(c.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 1]
        hosts = [n for n in c.GetNeighbors() if n.IsInRing() and n.GetAtomicNum() in {6, 7, 8}]
        if len(double_n) == len(single_n) == len(hosts) == 1 and c.GetDegree() == 3:
            sites.append((c.GetIdx(), double_n[0].GetIdx(), single_n[0].GetIdx()))
    if not 1 <= len(sites) <= 3:
        return None
    # Any non-ring O or other non-ring N is a competing characteristic group.
    ns = {idx for _, imino, amino in sites for idx in (imino, amino)}
    if any((a.GetAtomicNum() == 8 and not a.IsInRing())
           or (a.GetAtomicNum() == 7 and not a.IsInRing() and a.GetIdx() not in ns)
           for a in mol.GetAtoms()):
        return None
    imino_prefix, removed = '', set()
    for carbon, imino, amino in sites:
        n = mol.GetAtomWithIdx(imino)
        roots = [a.GetIdx() for a in n.GetNeighbors() if a.GetIdx() != carbon]
        if not roots:
            if n.GetTotalNumHs() != 1:
                return None
            continue
        if (len(sites) != 1 or len(roots) != 1 or n.GetTotalNumHs() != 0
                or mol.GetAtomWithIdx(amino).GetDegree() != 1):
            return None
        from .phosphorus_ester_methods import _carbon_component
        from .component_mapping import mapped_branch
        branch = _carbon_component(mol, roots[0], imino)
        if branch is None or any(mol.GetAtomWithIdx(i).IsInRing() for i in branch):
            return None
        imino_prefix, mapped = mapped_branch(mol, Chem, roots[0], imino)
        if not imino_prefix or frozenset(mapped) != branch:
            return None
        removed.update(branch)
    copy = Chem.RWMol(mol)
    for _, imino, _ in sites:
        atom = copy.GetAtomWithIdx(imino)
        atom.SetAtomicNum(8)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)
    for idx in sorted(removed, reverse=True):
        copy.RemoveAtom(idx)
    amide = _sanitize(copy, Chem)
    if amide is None:
        return None
    from .api import mol_to_iupac
    ordinary = mol_to_iupac(amide)
    # Functional replacement below changes only the principal amide suffix.
    # Every converted site must therefore belong to that suffix. Otherwise a
    # subordinate carbamoyl/benzoyl group would retain the scratch graph's O
    # instead of the original imino N. Leave that case to the unchanged-graph
    # construction; an explicit multiple suffix still owns all its sites.
    if len(sites) > 1:
        multiplicity = {2: 'di', 3: 'tri'}[len(sites)]
        if not ordinary.endswith(multiplicity + 'carboxamide'):
            return None
    if ordinary.endswith('benzamide'):
        name = ordinary[:-len('benzamide')] + 'benzenecarboximidamide'
    elif ordinary.endswith('carboxamide'):
        name = ordinary[:-len('carboxamide')] + 'carboximidamide'
    else:
        return None
    if imino_prefix:
        from .formatting import format_prefixes, join_prefix_parent
        name = join_prefix_parent(format_prefixes([("N'", imino_prefix)]), name)
    return _out(mol, name, (site[0] for site in sites), 'P-66.4.1.1/P-41', 'functional-replacement')


def _acid_to_acyl(acid):
    import re
    special = {'formic acid': 'formyl', 'acetic acid': 'acetyl', 'benzoic acid': 'benzoyl'}
    if acid in special:
        return special[acid]
    if acid.endswith('benzoic acid'):
        return acid[:-len('benzoic acid')] + 'benzoyl'
    if acid.endswith('carboxylic acid'):
        return acid[:-len('carboxylic acid')] + 'carbonyl'
    if acid.endswith('oic acid'):
        return acid[:-len('oic acid')] + 'oyl'
    match = re.fullmatch(r'(.+(?:thio|seleno|telluro))ic (?:O|S|Se|Te)-acid', acid)
    return match.group(1) + 'yl' if match else ''


def acid_halide_decision(mol, Chem):
    """Single acyl halide of an otherwise hydrocarbon acid or a simple ring acid."""
    if not _plain(mol, Chem, {6, 7, 8, 9, 16, 17, 34, 35, 52, 53}):
        return None
    halogens = [a for a in mol.GetAtoms() if a.GetAtomicNum() in {9, 17, 35, 53}]
    if any(a.GetDegree() != 1 for a in halogens):
        return None
    leaving_groups = []
    for halogen in halogens:
        neighbor = halogen.GetNeighbors()[0]
        if neighbor.GetAtomicNum() in {16, 34, 52} or (
                neighbor.GetAtomicNum() == 6 and any(
                    b.GetBondTypeAsDouble() == 2
                    and b.GetOtherAtom(neighbor).GetAtomicNum() in {8, 16, 34, 52}
                    for b in neighbor.GetBonds())):
            leaving_groups.append(halogen)
        elif neighbor.GetAtomicNum() != 6:
            return None
    if len(leaving_groups) != 1:
        return None
    leaving = leaving_groups[0]
    center = leaving.GetNeighbors()[0]
    if center.GetAtomicNum() not in {6, 16, 34, 52} or center.IsInRing():
        return None
    if mol.GetBondBetweenAtoms(center.GetIdx(), leaving.GetIdx()).GetBondTypeAsDouble() != 1:
        return None
    terminal = [a for a in center.GetNeighbors() if a.GetAtomicNum() in {8, 16, 34, 52}
                and a.GetDegree() == 1
                and mol.GetBondBetweenAtoms(center.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() == 2]
    roots = [a for a in center.GetNeighbors() if a.GetAtomicNum() == 6]
    if center.GetAtomicNum() == 6:
        if len(terminal) != 1 or len(roots) > 1 or center.GetDegree() != len(roots) + 2:
            return None
    elif len(terminal) not in {1, 2} or len(roots) != 1 or center.GetDegree() != len(terminal) + 2:
        return None
    permitted_hetero = {center.GetIdx(), *(a.GetIdx() for a in (*halogens, *terminal))}
    if any(a.GetAtomicNum() != 6 and a.GetIdx() not in permitted_hetero
           and not (a.GetAtomicNum() == 7 and a.IsInRing()) for a in mol.GetAtoms()):
        return None
    copy = Chem.RWMol(mol)
    copy.GetAtomWithIdx(leaving.GetIdx()).SetAtomicNum(8)
    copy.GetAtomWithIdx(leaving.GetIdx()).SetNoImplicit(False)
    copy.GetAtomWithIdx(leaving.GetIdx()).SetNumExplicitHs(0)
    center_element = center.GetAtomicNum()
    if center_element in {34, 52}:
        copy.GetAtomWithIdx(center.GetIdx()).SetAtomicNum(16)
    acid = _sanitize(copy, Chem)
    if acid is None:
        return None
    from .api import mol_to_iupac
    name = mol_to_iupac(acid)
    if center_element == 6:
        acyl = _acid_to_acyl(name)
    else:
        ending = 'sulfonic acid' if len(terminal) == 2 else 'sulfinic acid'
        if not name.endswith(ending):
            return None
        acyl = name[:-len('ic acid')] + 'yl'
        if center_element == 34:
            acyl = acyl.replace('sulfonyl', 'selenonyl').replace('sulfinyl', 'seleninyl')
        elif center_element == 52:
            acyl = acyl.replace('sulfonyl', 'telluronyl').replace('sulfinyl', 'tellurinyl')
    if not acyl:
        return None
    halide = {9: 'fluoride', 17: 'chloride', 35: 'bromide', 53: 'iodide'}[leaving.GetAtomicNum()]
    return _out(mol, acyl + ' ' + halide, (center.GetIdx(),), 'P-65.5.1/P-41', 'functional-class')


def mixed_thioamide_decision(mol, Chem):
    """One carboxamide and one thioacyl unit on the same amide nitrogen."""
    if not _plain(mol, Chem, {6, 7, 8, 16}):
        return None
    nitrogens = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    oxygens = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 8]
    sulfurs = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 16]
    if len(nitrogens) != len(oxygens) or len(oxygens) != len(sulfurs) or len(nitrogens) != 1:
        return None
    n, o, s = nitrogens[0], oxygens[0], sulfurs[0]
    if n.IsInRing() or n.GetDegree() != 2 or n.GetTotalNumHs() != 1:
        return None
    if o.GetDegree() != 1 or s.GetDegree() != 1:
        return None
    oc, sc = o.GetNeighbors()[0], s.GetNeighbors()[0]
    if (oc.GetAtomicNum() != 6 or sc.GetAtomicNum() != 6 or oc.GetIdx() == sc.GetIdx()
            or any(mol.GetBondBetweenAtoms(n.GetIdx(), c.GetIdx()) is None for c in (oc, sc))
            or any(mol.GetBondBetweenAtoms(n.GetIdx(), c.GetIdx()).GetBondTypeAsDouble() != 1 for c in (oc, sc))
            or mol.GetBondBetweenAtoms(oc.GetIdx(), o.GetIdx()).GetBondTypeAsDouble() != 2
            or mol.GetBondBetweenAtoms(sc.GetIdx(), s.GetIdx()).GetBondTypeAsDouble() != 2):
        return None
    copy = Chem.RWMol(mol)
    copy.RemoveBond(n.GetIdx(), sc.GetIdx())
    nitrogen = copy.GetAtomWithIdx(n.GetIdx())
    nitrogen.SetNoImplicit(False)
    nitrogen.SetNumExplicitHs(0)
    oh = copy.AddAtom(Chem.Atom(8))
    copy.AddBond(sc.GetIdx(), oh, Chem.BondType.SINGLE)
    split = _sanitize(copy, Chem)
    if split is None:
        return None
    fragments = Chem.GetMolFrags(split, asMols=True, sanitizeFrags=True)
    if len(fragments) != 2:
        return None
    acid = next((frag for frag in fragments if any(a.GetAtomicNum() == 16 for a in frag.GetAtoms())), None)
    amide = next((frag for frag in fragments if any(a.GetAtomicNum() == 7 for a in frag.GetAtoms())), None)
    if acid is None or amide is None:
        return None
    from .api import mol_to_iupac
    acyl, parent = _acid_to_acyl(mol_to_iupac(acid)), mol_to_iupac(amide)
    if not acyl or not parent.endswith(('amide', 'carboxamide')):
        return None
    return _out(mol, 'N-(' + acyl + ')' + parent, (oc.GetIdx(),), 'P-41/P-66.1.4.2')


def silyl_carboxylic_acid_decision(mol, Chem):
    """Select all carboxy suffixes before a terminal silyl substituent."""
    if not _plain(mol, Chem, {6, 8, 14}):
        return None
    silicons = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 14]
    if len(silicons) != 1:
        return None
    si = silicons[0]
    acids = []
    for c in mol.GetAtoms():
        if c.GetAtomicNum() != 6:
            continue
        oxo = [a for a in c.GetNeighbors() if a.GetAtomicNum() == 8 and a.GetDegree() == 1
               and mol.GetBondBetweenAtoms(c.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() == 2]
        oh = [a for a in c.GetNeighbors() if a.GetAtomicNum() == 8 and a.GetDegree() == 1 and a.GetTotalNumHs() == 1
              and mol.GetBondBetweenAtoms(c.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() == 1]
        if len(oxo) == len(oh) == 1:
            acids.append(c.GetIdx())
    if not acids or len(acids) > 2 or sum(a.GetAtomicNum() == 8 for a in mol.GetAtoms()) != 2*len(acids):
        return None
    if si.GetDegree() == 1 and si.GetTotalNumHs() == 3 and si.GetNeighbors()[0].GetIdx() not in acids:
        # With another carbon branch, changing fluoro to silyl could change
        # prefix alphabetization and a tied parent numbering. Leave that
        # comparison to a complete mapped substituent frontier.
        if any(sum(n.GetAtomicNum() == 6 for n in a.GetNeighbors()) > 2
               for a in mol.GetAtoms() if a.GetAtomicNum() == 6):
            return None
        copy = Chem.RWMol(mol)
        marker = copy.GetAtomWithIdx(si.GetIdx())
        marker.SetAtomicNum(9)
        marker.SetNumExplicitHs(0)
        marker.SetNoImplicit(True)
        acid = _sanitize(copy, Chem)
        if acid is None:
            return None
        from .api import mol_to_iupac
        name = mol_to_iupac(acid)
        if name.endswith(' acid') and name.count('fluoro') == 1:
            return _out(mol, name.replace('fluoro', 'silyl'), acids, 'P-44.1.1/P-44.1.2')
        return None
    # A carboxy group directly attached to silicon belongs to silane's suffix.
    if len(acids) != 1 or mol.GetBondBetweenAtoms(si.GetIdx(), acids[0]) is None:
        return None
    roots = [a.GetIdx() for a in si.GetNeighbors() if a.GetIdx() != acids[0]]
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in roots):
        return None
    from .functional_replacement_acids import _organyl_units
    seen = {si.GetIdx(), acids[0]} | {a.GetIdx() for a in mol.GetAtomWithIdx(acids[0]).GetNeighbors() if a.GetAtomicNum() == 8}
    prefix = _organyl_units(mol, Chem, si.GetIdx(), roots, seen)
    if prefix is None or seen != set(range(mol.GetNumAtoms())):
        return None
    return _out(mol, prefix + 'silanecarboxylic acid', acids, 'P-44.1.1/P-65.1.2.2')


def nonstandard_heteromonocycle_decision(mol, Chem):
    """Unsubstituted Hantzsch–Widman and large-ring lambda hydrides."""
    if not _plain(mol, Chem, {6, 7, 8, 15, 16, 34, 52, 53}, limit=32):
        return None
    rings = mol.GetRingInfo().AtomRings()
    if len(rings) != 1:
        return None
    ring = frozenset(rings[0])
    if ring != set(range(mol.GetNumAtoms())) or any(mol.GetAtomWithIdx(i).GetDegree() != 2 for i in ring):
        return None
    size = len(ring)
    if not 5 <= size <= 20 or any(a.GetIsAromatic() for a in mol.GetAtoms()):
        return None
    numbers = [a.GetAtomicNum() for a in mol.GetAtoms() if a.GetAtomicNum() != 6]
    if not numbers:
        return None
    element_order = {8: 0, 16: 1, 34: 2, 52: 3, 7: 4, 15: 5, 53: 6}
    if len(numbers) > 3 or any(x not in element_order for x in numbers):
        return None
    if len(numbers) == 2 and set(numbers) not in ({8, 15}, {16, 7}, {8, 16}, {16}):
        return None
    if len(numbers) == 3 and (numbers.count(8) != 1 or numbers.count(16) != 2):
        return None
    standard = {8: 2, 16: 2, 34: 2, 52: 2, 7: 3, 15: 3, 53: 1}
    nonstandard = {a.GetIdx(): int(a.GetTotalValence()) for a in mol.GetAtoms()
                   if a.GetAtomicNum() != 6 and int(a.GetTotalValence()) != standard[a.GetAtomicNum()]}
    if not 1 <= len(nonstandard) <= 2 or any(v not in {3, 4, 5, 6} for v in nonstandard.values()):
        return None
    bonds = list(mol.GetBonds())
    if any(b.GetBondTypeAsDouble() not in {1, 2} for b in bonds):
        return None
    double = [b for b in bonds if b.GetBondTypeAsDouble() == 2]
    if any(sum(b.GetBondTypeAsDouble() == 2 for b in a.GetBonds()) > 1 for a in mol.GetAtoms()):
        return None
    saturated = not double
    if not saturated and (size > 10 or len(double) != size // 2):
        return None
    if saturated and size <= 10 and not (len(numbers) == 1 and numbers[0] == 53 and size == 6):
        return None
    if size > 10 and not (saturated and numbers.count(8) == 1 and 1 <= numbers.count(16) <= 2):
        return None
    from .ring_state_parent import _hw_name
    from .constants import ROOTS
    import re
    adjacency = {i: [a.GetIdx() for a in mol.GetAtomWithIdx(i).GetNeighbors()] for i in ring}
    first_element = min(numbers, key=element_order.__getitem__)
    candidates = []
    for start in ring:
        if mol.GetAtomWithIdx(start).GetAtomicNum() != first_element:
            continue
        for first in adjacency[start]:
            path, previous, current = [start], start, first
            while current != start and len(path) <= size:
                path.append(current)
                following = [i for i in adjacency[current] if i != previous]
                if len(following) != 1:
                    break
                previous, current = current, following[0]
            if current != start or len(path) != size or frozenset(path) != ring:
                continue
            sequence = [mol.GetAtomWithIdx(i).GetAtomicNum() for i in path]
            hetero_locs = tuple(i+1 for i, z in enumerate(sequence) if z != 6)
            by_type = tuple(tuple(i+1 for i, z in enumerate(sequence) if z == element)
                            for element in sorted(set(numbers), key=element_order.__getitem__))
            lambda_locs = {path.index(idx) + 1: valence for idx, valence in nonstandard.items()}
            indicated = []
            if not saturated and size % 2:
                unmatched = [a for a in mol.GetAtoms()
                             if all(b.GetBondTypeAsDouble() != 2 for b in a.GetBonds())]
                if len(unmatched) != 1:
                    continue
                if unmatched[0].GetTotalNumHs() > 0:
                    indicated = [path.index(unmatched[0].GetIdx()) + 1]
            score = (hetero_locs, by_type, tuple(indicated), tuple(sorted(lambda_locs)))
            if size > 10:
                if size not in ROOTS:
                    continue
                oxy = sequence.index(8) + 1
                sulfur_locs = [i+1 for i, z in enumerate(sequence) if z == 16]
                sulfur_text = ','.join(str(loc) + ('lambda' + str(lambda_locs[loc])
                                                  if loc in lambda_locs else '') for loc in sulfur_locs)
                name = f'{oxy}-oxa-{sulfur_text}-' + ('di' if len(sulfur_locs) == 2 else '') + f'thiacyclo{ROOTS[size]}ane'
            else:
                if numbers == [53]:
                    # These two sourced words have fixed ring sizes. Other
                    # iodine cycles need their own size-dependent vocabulary.
                    base = ('iodinane' if saturated and size == 6 else
                            'iodole' if not saturated and size == 5 else '')
                elif set(numbers) == {8, 15}:
                    base = 'oxaphosphole' if size == 5 and not saturated else ''
                else:
                    base = _hw_name(sequence, saturated)
                if not base:
                    continue
                base = re.sub(r'^\d+(?:,\d+)*-', '', base)
                fields = [str(loc) + ('lambda' + str(lambda_locs[loc]) if loc in lambda_locs else '')
                          for loc in hetero_locs]
                name = (','.join(map(str, indicated)) + 'H-' if indicated else '') + ','.join(fields) + '-' + base
            candidates.append((score, name, path))
    if not candidates:
        return None
    best = min(score for score, _, _ in candidates)
    winners = [row for row in candidates if row[0] == best]
    if len({row[1] for row in winners}) != 1:
        return None
    _, name, _ = winners[0]
    return _out(mol, name, nonstandard, 'P-22.2.7.1/P-14.1', 'Hantzsch-Widman')


def isotopic_alkene_decision(mol, Chem):
    """P-82 isotope-created E/Z on a single linear alkene with one 2H."""
    if len(Chem.GetMolFrags(mol)) != 1 or mol.GetStereoGroups() or mol.GetRingInfo().NumRings():
        return None
    atoms = list(mol.GetAtoms())
    hydrogens = [a for a in atoms if a.GetAtomicNum() == 1]
    carbon = {a.GetIdx() for a in atoms if a.GetAtomicNum() == 6}
    if (len(hydrogens) != 1 or hydrogens[0].GetIsotope() != 2
            or not 2 <= len(carbon) <= 20
            or any(a.GetAtomicNum() not in {1, 6} or a.GetFormalCharge() or a.GetNumRadicalElectrons()
                   or (a.GetAtomicNum() == 6 and (a.GetIsotope() or int(a.GetChiralTag()))) for a in atoms)):
        return None
    d = hydrogens[0]
    if d.GetDegree() != 1 or d.GetNeighbors()[0].GetIdx() not in carbon:
        return None
    edges = [b for b in mol.GetBonds() if b.GetBeginAtomIdx() in carbon and b.GetEndAtomIdx() in carbon]
    double = [b for b in edges if b.GetBondTypeAsDouble() == 2]
    if (len(edges) != len(carbon) - 1 or len(double) != 1
            or any(b.GetBondTypeAsDouble() not in {1, 2} for b in edges)
            or d.GetNeighbors()[0].GetIdx() not in {double[0].GetBeginAtomIdx(), double[0].GetEndAtomIdx()}):
        return None
    adjacency = {i: [n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in carbon] for i in carbon}
    if any(len(row) > 2 for row in adjacency.values()):
        return None
    ends = [i for i in carbon if len(adjacency[i]) == 1]
    if len(ends) != 2:
        return None
    paths = []
    for end in ends:
        path, previous, current = [], -1, end
        while True:
            path.append(current)
            following = [i for i in adjacency[current] if i != previous]
            if not following:
                break
            previous, current = current, following[0]
        if len(path) != len(carbon):
            return None
        double_loc = min(path.index(double[0].GetBeginAtomIdx()) + 1,
                         path.index(double[0].GetEndAtomIdx()) + 1)
        paths.append((double_loc, path.index(d.GetNeighbors()[0].GetIdx()) + 1, path))
    locant, isotope_locant, _ = min(paths)
    stereo_copy = Chem.Mol(mol)
    Chem.AssignStereochemistry(stereo_copy, cleanIt=True, force=True)
    stereo = stereo_copy.GetBondWithIdx(double[0].GetIdx()).GetStereo()
    if stereo == Chem.BondStereo.STEREOE:
        descriptor = 'E'
    elif stereo == Chem.BondStereo.STEREOZ:
        descriptor = 'Z'
    else:
        return None
    from .constants import ROOTS
    if len(carbon) not in ROOTS:
        return None
    stem = ROOTS[len(carbon)]
    parent = stem + 'ene' if len(carbon) == 2 else stem + f'-{locant}-ene'
    name = f'({locant}{descriptor})-({isotope_locant}-2H1){parent}'
    atoms = frozenset(a.GetIdx() for a in mol.GetAtoms())
    return MethodDecision(name, 'substitutive', frozenset(), atoms, 'P-82.2.1/P-82.4/P-9', (),
                          component_roles=(('isotope-stereoisomer', atoms, ()),))


def sulfonyl_carboxamide_decision(mol, Chem):
    """Carboxamide senior to an N-linked sulfonyl or sulfinyl group."""
    if not _plain(mol, Chem, {6, 7, 8, 16, 34, 52}):
        return None
    ns = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    centers = [a for a in mol.GetAtoms() if a.GetAtomicNum() in {16, 34, 52}]
    if len(ns) != 1 or len(centers) != 1:
        return None
    n, center = ns[0], centers[0]
    ncenter = mol.GetBondBetweenAtoms(n.GetIdx(), center.GetIdx())
    if n.IsInRing() or n.GetDegree() != 2 or n.GetTotalNumHs() != 1 or ncenter is None or ncenter.GetBondTypeAsDouble() != 1:
        return None
    oxos = [a for a in center.GetNeighbors() if a.GetAtomicNum() == 8 and a.GetDegree() == 1
            and mol.GetBondBetweenAtoms(center.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() == 2]
    roots = [a for a in center.GetNeighbors() if a.GetAtomicNum() == 6]
    carbonyl = [a for a in n.GetNeighbors() if a.GetAtomicNum() == 6 and any(
        o.GetAtomicNum() == 8 and o.GetDegree() == 1
        and mol.GetBondBetweenAtoms(a.GetIdx(), o.GetIdx()).GetBondTypeAsDouble() == 2 for o in a.GetNeighbors())]
    if (len(oxos) not in {1, 2} or len(roots) != 1 or center.GetDegree() != len(oxos) + 2
            or len(carbonyl) != 1 or sum(a.GetAtomicNum() == 8 for a in mol.GetAtoms()) != len(oxos) + 1):
        return None
    copy = Chem.RWMol(mol)
    copy.RemoveBond(n.GetIdx(), center.GetIdx())
    copy.GetAtomWithIdx(center.GetIdx()).SetAtomicNum(16)
    nitrogen = copy.GetAtomWithIdx(n.GetIdx())
    nitrogen.SetNoImplicit(False)
    nitrogen.SetNumExplicitHs(0)
    o = copy.AddAtom(Chem.Atom(8))
    copy.AddBond(center.GetIdx(), o, Chem.BondType.SINGLE)
    split = _sanitize(copy, Chem)
    if split is None:
        return None
    fragments = Chem.GetMolFrags(split, asMols=True, sanitizeFrags=True)
    if len(fragments) != 2:
        return None
    acid = next((f for f in fragments if any(a.GetAtomicNum() == 16 for a in f.GetAtoms())), None)
    amide = next((f for f in fragments if any(a.GetAtomicNum() == 7 for a in f.GetAtoms())), None)
    if acid is None or amide is None:
        return None
    from .api import mol_to_iupac
    acid_name, parent = mol_to_iupac(acid), mol_to_iupac(amide)
    ending = 'sulfonic acid' if len(oxos) == 2 else 'sulfinic acid'
    if not acid_name.endswith(ending) or not parent.endswith('amide'):
        return None
    acyl = acid_name[:-len('ic acid')] + 'yl'
    if center.GetAtomicNum() == 34:
        acyl = acyl.replace('sulfonyl', 'selenonyl').replace('sulfinyl', 'seleninyl')
    elif center.GetAtomicNum() == 52:
        acyl = acyl.replace('sulfonyl', 'telluronyl').replace('sulfinyl', 'tellurinyl')
    return _out(mol, 'N-(' + acyl + ')' + parent, (carbonyl[0].GetIdx(),), 'P-41/P-66.1.1.3')


def silyl_diol_decision(mol, Chem):
    """Keep two alcohol suffixes on a carbon parent bearing terminal silyl."""
    if not _plain(mol, Chem, {6, 8, 14}) or mol.GetRingInfo().NumRings():
        return None
    silicons = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 14]
    oxygens = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 8]
    if len(silicons) != 1 or len(oxygens) != 2:
        return None
    si = silicons[0]
    if si.GetDegree() != 1 or si.GetTotalNumHs() != 3 or si.GetNeighbors()[0].GetAtomicNum() != 6:
        return None
    if any(o.GetDegree() != 1 or o.GetTotalNumHs() != 1 or o.GetNeighbors()[0].GetAtomicNum() != 6
           or mol.GetBondBetweenAtoms(o.GetIdx(), o.GetNeighbors()[0].GetIdx()).GetBondTypeAsDouble() != 1
           for o in oxygens):
        return None
    carbon = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 6]
    if (sum(sum(n.GetAtomicNum() == 6 for n in a.GetNeighbors()) == 3 for a in carbon) != 1
            or any(sum(n.GetAtomicNum() == 6 for n in a.GetNeighbors()) > 3 for a in carbon)
            or any(b.GetBondTypeAsDouble() != 1 for b in mol.GetBonds())):
        return None
    copy = Chem.RWMol(mol)
    marker = copy.GetAtomWithIdx(si.GetIdx())
    marker.SetAtomicNum(9)
    marker.SetNumExplicitHs(0)
    marker.SetNoImplicit(True)
    ordinary = _sanitize(copy, Chem)
    if ordinary is None:
        return None
    from .api import mol_to_iupac
    name = mol_to_iupac(ordinary)
    if not name.endswith('diol') or 'hydroxy' in name or name.count('fluoro') != 1:
        return None
    return _out(mol, name.replace('fluoro', 'silyl'), (o.GetNeighbors()[0].GetIdx() for o in oxygens),
                'P-44.1.1/P-44.1.2')


def symmetric_thioimide_decision(mol, Chem):
    """Identical thioacyl groups on one nitrogen under P-66.1.4.2."""
    if not _plain(mol, Chem, {6, 7, 16}):
        return None
    ns = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    sulfurs = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 16]
    if len(ns) != 1 or len(sulfurs) != 2:
        return None
    n = ns[0]
    if n.IsInRing() or n.GetDegree() != 2 or n.GetTotalNumHs() != 1:
        return None
    centers = []
    for s in sulfurs:
        if s.GetDegree() != 1:
            return None
        c = s.GetNeighbors()[0]
        if (c.GetAtomicNum() != 6 or mol.GetBondBetweenAtoms(c.GetIdx(), s.GetIdx()).GetBondTypeAsDouble() != 2
                or mol.GetBondBetweenAtoms(c.GetIdx(), n.GetIdx()) is None
                or mol.GetBondBetweenAtoms(c.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() != 1):
            return None
        centers.append(c.GetIdx())
    if len(set(centers)) != 2:
        return None
    # Prove the two rooted thioacyl components identical before choosing
    # either as the principal amide, avoiding an arbitrary parent tie break.
    from .branch_rules import collect_subtree
    groups = [frozenset(collect_subtree(mol, c, n.GetIdx())) for c in centers]
    if groups[0] & groups[1] or groups[0] | groups[1] | {n.GetIdx()} != set(range(mol.GetNumAtoms())):
        return None
    signatures = [Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(group), rootedAtAtom=center,
                                           canonical=True, isomericSmiles=True)
                  for group, center in zip(groups, centers, strict=True)]
    if signatures[0] != signatures[1]:
        return None
    copy = Chem.RWMol(mol)
    copy.RemoveBond(n.GetIdx(), centers[0])
    copy.GetAtomWithIdx(n.GetIdx()).SetNoImplicit(False)
    copy.GetAtomWithIdx(n.GetIdx()).SetNumExplicitHs(0)
    o = copy.AddAtom(Chem.Atom(8))
    copy.AddBond(centers[0], o, Chem.BondType.SINGLE)
    split = _sanitize(copy, Chem)
    if split is None:
        return None
    fragments = Chem.GetMolFrags(split, asMols=True, sanitizeFrags=True)
    if len(fragments) != 2:
        return None
    acid = next((f for f in fragments if any(a.GetAtomicNum() == 8 for a in f.GetAtoms())), None)
    amide = next((f for f in fragments if any(a.GetAtomicNum() == 7 for a in f.GetAtoms())), None)
    if acid is None or amide is None:
        return None
    from .api import mol_to_iupac
    acyl, parent = _acid_to_acyl(mol_to_iupac(acid)), mol_to_iupac(amide)
    if not acyl or not parent.endswith('thioamide'):
        return None
    return _out(mol, 'N-(' + acyl + ')' + parent, centers, 'P-66.1.4.2')
