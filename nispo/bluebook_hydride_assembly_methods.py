"""Source-scoped parent, acid-derivative, and assembly constructions.

Every producer accounts for the original atoms and declines competing classes
or annotations outside its proven scope.
"""
from collections import Counter
from itertools import product
import re

from .bluebook_azoxy_methods import _decision, _plain
from .component_mapping import mapped_branch
from .constants import MULT, ROOTS
from .formatting import enclose_name_unit, format_prefixes, prefix_alphabetization_key


def _single(mol, a, b):
    bond = mol.GetBondBetweenAtoms(a, b)
    return bond is not None and bond.GetBondTypeAsDouble() == 1


def _sanitize(copy, Chem):
    mol = copy.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except (ValueError, RuntimeError):
        return None
    return mol


def hydrazinyl_amine_decision(mol, Chem):
    """A hydrazinyl branch subordinate to one simple primary amine."""
    if not _plain(mol, Chem) or any(a.GetAtomicNum() not in {6, 7} for a in mol.GetAtoms()):
        return None
    nitrogens = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    if len(nitrogens) != 3:
        return None
    choices = []
    for outer in nitrogens:
        if outer.GetDegree() != 1 or outer.GetTotalNumHs(includeNeighbors=True) != 2:
            continue
        inner = outer.GetNeighbors()[0]
        if inner.GetAtomicNum() != 7 or inner.GetDegree() != 2 or inner.GetTotalNumHs(includeNeighbors=True) != 1:
            continue
        if not _single(mol, outer.GetIdx(), inner.GetIdx()):
            continue
        hosts = [a for a in inner.GetNeighbors() if a.GetIdx() != outer.GetIdx()]
        if len(hosts) != 1 or hosts[0].GetAtomicNum() != 6 or not _single(mol, inner.GetIdx(), hosts[0].GetIdx()):
            continue
        other = [a for a in nitrogens if a.GetIdx() not in {outer.GetIdx(), inner.GetIdx()}]
        if (len(other) != 1 or other[0].GetDegree() != 1
                or other[0].GetTotalNumHs(includeNeighbors=True) != 2
                or other[0].GetNeighbors()[0].GetAtomicNum() != 6):
            continue
        choices.append((outer.GetIdx(), inner.GetIdx(), other[0].GetIdx()))
    if len(choices) != 1:
        return None
    outer, inner, amine = choices[0]
    copy = Chem.RWMol(mol)
    marker = copy.GetAtomWithIdx(inner)
    marker.SetAtomicNum(9)
    marker.SetNumExplicitHs(0)
    marker.SetNoImplicit(True)
    copy.RemoveAtom(outer)
    neutral = _sanitize(copy, Chem)
    if neutral is None:
        return None
    from .api import mol_to_iupac
    name = mol_to_iupac(neutral)
    if name.count('fluoro') != 1 or not name.endswith('amine'):
        return None
    return _decision(mol, name.replace('fluoro', 'hydrazinyl'), (amine,),
                     'P-68.3.1.2.1/P-41', method='substitutive')


def hydroxylamine_suffix_decision(mol, Chem):
    """An unsubstituted hydroxylamine O-bound acid or primary amide."""
    if not _plain(mol, Chem) or any(a.GetAtomicNum() not in {6, 7, 8, 16} for a in mol.GetAtoms()):
        return None
    for n in mol.GetAtoms():
        if (n.GetAtomicNum() != 7 or n.GetDegree() != 1 or n.GetTotalNumHs(includeNeighbors=True) != 2
                or n.GetNeighbors()[0].GetAtomicNum() != 8):
            continue
        o = n.GetNeighbors()[0]
        if o.GetDegree() != 2 or not _single(mol, n.GetIdx(), o.GetIdx()):
            continue
        centers = [a for a in o.GetNeighbors() if a.GetIdx() != n.GetIdx()]
        if len(centers) != 1 or not _single(mol, o.GetIdx(), centers[0].GetIdx()):
            continue
        center = centers[0]
        others = [a for a in center.GetNeighbors() if a.GetIdx() != o.GetIdx()]
        oxos = [a for a in others if a.GetAtomicNum() == 8 and a.GetDegree() == 1
                and mol.GetBondBetweenAtoms(center.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() == 2]
        ohs = [a for a in others if a.GetAtomicNum() == 8 and a.GetDegree() == 1
               and a.GetTotalNumHs(includeNeighbors=True) == 1 and _single(mol, center.GetIdx(), a.GetIdx())]
        amides = [a for a in others if a.GetAtomicNum() == 7 and a.GetDegree() == 1
                  and a.GetTotalNumHs(includeNeighbors=True) == 2 and _single(mol, center.GetIdx(), a.GetIdx())]
        if center.GetAtomicNum() == 6 and len(oxos) == 1 and len(ohs) == 1 and len(others) == 2:
            suffix = 'carboxylic acid'
        elif center.GetAtomicNum() == 6 and len(oxos) == 1 and len(amides) == 1 and len(others) == 2:
            suffix = 'carboxamide'
        elif center.GetAtomicNum() == 16 and len(oxos) == 2 and len(ohs) == 1 and len(others) == 3:
            suffix = 'sulfonic acid'
        elif center.GetAtomicNum() == 16 and len(oxos) == 2 and len(amides) == 1 and len(others) == 3:
            suffix = 'sulfonamide'
        else:
            continue
        owned = {n.GetIdx(), o.GetIdx(), center.GetIdx(), *(a.GetIdx() for a in others)}
        if owned != set(range(mol.GetNumAtoms())):
            continue
        return _decision(mol, 'hydroxylamine-O-' + suffix, (center.GetIdx(),),
                         'P-68.3.1.1.1.4/P-41')
    return None


def numbered_group14_amine_decision(mol, Chem):
    """Number carbon substituents of a primary mononuclear Group 14 amine."""
    if not _plain(mol, Chem) or any(a.GetAtomicNum() not in {6, 7, 14, 32, 50, 82} for a in mol.GetAtoms()):
        return None
    centers = [a for a in mol.GetAtoms() if a.GetAtomicNum() in {14, 32, 50, 82}]
    nitrogens = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    if len(centers) != 1 or len(nitrogens) != 1:
        return None
    center, n = centers[0], nitrogens[0]
    if (n.GetDegree() != 1 or n.GetTotalNumHs(includeNeighbors=True) != 2
            or n.GetNeighbors()[0].GetIdx() != center.GetIdx() or center.IsInRing()
            or center.GetTotalValence() != 4 or not _single(mol, center.GetIdx(), n.GetIdx())):
        return None
    roots = [a for a in center.GetNeighbors() if a.GetIdx() != n.GetIdx()]
    if not roots or any(a.GetAtomicNum() != 6 or not _single(mol, center.GetIdx(), a.GetIdx()) for a in roots):
        return None
    prefixes, owned = [], {center.GetIdx(), n.GetIdx()}
    for root in roots:
        text, atoms = mapped_branch(mol, Chem, root.GetIdx(), center.GetIdx())
        if not text or atoms & owned or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in atoms):
            return None
        prefixes.append(('1', text))
        owned.update(atoms)
    if owned != set(range(mol.GetNumAtoms())):
        return None
    # Numbering of the parent-atom substituents is source-backed when at
    # least two of them occupy that atom; a lone substituent delegates.
    if len(prefixes) < 2:
        return None
    parent = {14: 'silanamine', 32: 'germanamine', 50: 'stannanamine', 82: 'plumbanamine'}[center.GetAtomicNum()]
    name = format_prefixes(prefixes, hierarchical_enclosures=True) + parent
    return _decision(mol, name, (n.GetIdx(),), 'P-62.2.1.3/P-68.2.5/P-14')


def methyl_diamine_numbering_decision(mol, Chem):
    """Resolve a carbon/N methyl locant tie in a saturated acyclic diamine."""
    if (not _plain(mol, Chem) or mol.GetRingInfo().NumRings()
            or mol.GetNumAtoms() != mol.GetNumHeavyAtoms()
            or any(a.GetAtomicNum() not in {6, 7} for a in mol.GetAtoms())
            or any(b.GetBondTypeAsDouble() != 1 for b in mol.GetBonds())):
        return None
    ns = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    primary = [a for a in ns if a.GetDegree() == 1 and a.GetTotalNumHs() == 2]
    secondary = [a for a in ns if a.GetDegree() == 2 and a.GetTotalNumHs() == 1]
    if len(ns) != 2 or len(primary) != 1 or len(secondary) != 1:
        return None
    n1, n2 = primary[0], secondary[0]
    if any(a.GetAtomicNum() != 6 for a in (*n1.GetNeighbors(), *n2.GetNeighbors())):
        return None
    terminal = [a for a in n2.GetNeighbors() if a.GetDegree() == 1 and a.GetTotalNumHs() == 3]
    if len(terminal) != 1:
        return None
    n_methyl = terminal[0].GetIdx()
    host1 = n1.GetNeighbors()[0].GetIdx()
    host2 = next(a.GetIdx() for a in n2.GetNeighbors() if a.GetIdx() != n_methyl)
    carbon = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6} - {n_methyl}
    from .multifunctional_parent import _paths
    paths = [p for p in _paths(mol, carbon) if host1 in p and host2 in p]
    if not paths:
        return None
    longest = max(map(len, paths))
    rows = []
    for path in paths:
        if len(path) != longest or len(path) not in ROOTS:
            continue
        remaining = carbon - set(path)
        if len(remaining) != 1:
            continue
        methyl = mol.GetAtomWithIdx(next(iter(remaining)))
        if methyl.GetDegree() != 1 or methyl.GetTotalNumHs() != 3:
            continue
        methyl_host = methyl.GetNeighbors()[0].GetIdx()
        if methyl_host not in path:
            continue
        for sequence in (path, tuple(reversed(path))):
            loc = {idx: i for i, idx in enumerate(sequence, 1)}
            amines = tuple(sorted((loc[host1], loc[host2])))
            skeletal = loc[methyl_host]
            attached_n = loc[host2]
            prefixes = format_prefixes([(str(skeletal), 'methyl'), (f'N^({attached_n})', 'methyl')],
                                       hierarchical_enclosures=True)
            name = f"{prefixes}{ROOTS[len(sequence)]}ane-{amines[0]},{amines[1]}-diamine"
            rows.append(((amines, skeletal, attached_n), name))
    if not rows:
        return None
    best = min(row[0] for row in rows)
    winners = {name for key, name in rows if key == best}
    if len(winners) != 1:
        return None
    return _decision(mol, winners.pop(), (n1.GetIdx(), n2.GetIdx()),
                     'P-62.2.4.1.2/P-14', method='substitutive')


def parent_ring_cation_decision(mol, Chem):
    """Protonated monocyclic hydrides, retaining parent indicated hydrogen."""
    if not _plain(mol, Chem, charged=True) or mol.GetNumAtoms() != mol.GetNumHeavyAtoms():
        return None
    rings = mol.GetRingInfo().AtomRings()
    if len(rings) != 1:
        return None
    ring = frozenset(rings[0])
    owned = set(ring)
    branches = []
    for idx in ring:
        for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
            root = neighbor.GetIdx()
            if root in ring:
                continue
            if (mol.GetAtomWithIdx(idx).GetAtomicNum() != 6 or neighbor.GetAtomicNum() != 6
                    or not _single(mol, idx, root)):
                return None
            text, atoms = mapped_branch(mol, Chem, root, idx)
            if (not text or atoms & owned or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6
                                                 or mol.GetAtomWithIdx(i).IsInRing() for i in atoms)):
                return None
            branches.append((idx, text))
            owned.update(atoms)
    if owned != set(range(mol.GetNumAtoms())):
        return None
    positive = [a for a in mol.GetAtoms() if a.GetFormalCharge() == 1]
    if (not 1 <= len(positive) <= 2 or any(a.GetFormalCharge() not in {0, 1} for a in mol.GetAtoms())
            or any(a.GetIdx() not in ring or a.GetAtomicNum() not in {7, 8, 16}
                   or a.GetTotalNumHs(includeNeighbors=True) != 1
                   for a in positive)):
        return None
    neutral = Chem.Mol(mol)
    for original in positive:
        atom = neutral.GetAtomWithIdx(original.GetIdx())
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(True)
    try:
        Chem.SanitizeMol(neutral)
    except (ValueError, RuntimeError):
        return None
    from .cycle_utils import ordered_ring
    from .ring_rules import ring_parent_candidates
    ordered = ordered_ring(neutral, neutral.GetRingInfo().AtomRings()[0])
    if not ordered:
        return None
    rows = []
    for parent, sequence in ring_parent_candidates(neutral, ordered):
        # Admit stable, source-numbered saturated O heterocycles and aromatic
        # azoles. Other parent classes retain their existing selectors.
        if not (re.fullmatch(r'(?:\d+H-)?(?:imidazole|pyrazole)', parent)
                or parent in {'1,4-dioxane', '1,3-dioxane', 'oxane'}):
            continue
        mapping = {idx: pos for pos, idx in enumerate(sequence, 1)}
        locs = tuple(sorted(mapping[a.GetIdx()] for a in positive))
        h = re.match(r'^(\d+)H-', parent)
        h_key = (int(h[1]),) if h else ()
        stem = parent[:-1] if len(locs) == 1 and parent.endswith('e') else parent
        suffix = 'ium' if len(locs) == 1 else MULT[len(locs)] + 'ium'
        prefixes = [(str(mapping[idx]), text) for idx, text in branches]
        branch_locs = tuple(sorted(mapping[idx] for idx, _text in branches))
        branch_by_name = tuple(tuple(sorted(mapping[idx] for idx, text in branches if text == word))
                               for word in sorted({text for _idx, text in branches},
                                                  key=prefix_alphabetization_key))
        prefix = format_prefixes(prefixes, hierarchical_enclosures=True)
        separator = '-' if prefix and stem[0].isdigit() else ''
        name = prefix + separator + f"{stem}-{','.join(map(str, locs))}-{suffix}"
        rows.append(((h_key, locs, branch_locs, branch_by_name), name, mapping))
    if not rows:
        return None
    best = min(row[0] for row in rows)
    winners = [row for row in rows if row[0] == best]
    if len({row[1] for row in winners}) != 1:
        return None
    _, name, mapping = winners[0]
    return _decision(mol, name, (a.GetIdx() for a in positive), 'P-73.1.1.2/P-22.2.2.1.4',
                     locants=tuple((i, str(v)) for i, v in mapping.items()))


def _cyano_replacement(mol, center, root):
    """Return a complete –CN or –O–CN replacement and its owned atoms."""
    atom = mol.GetAtomWithIdx(root)
    if not _single(mol, center, root):
        return None
    if atom.GetAtomicNum() == 8 and atom.GetDegree() == 2:
        others = [a for a in atom.GetNeighbors() if a.GetIdx() != center]
        if len(others) != 1 or not _single(mol, root, others[0].GetIdx()):
            return None
        carbon, word, atoms = others[0], 'cyanatido', {root}
    else:
        carbon, word, atoms = atom, 'cyanido', set()
    if carbon.GetAtomicNum() != 6 or carbon.GetDegree() != 2:
        return None
    nitrogens = [a for a in carbon.GetNeighbors() if a.GetAtomicNum() == 7 and a.GetDegree() == 1
                 and mol.GetBondBetweenAtoms(carbon.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() == 3]
    if len(nitrogens) != 1:
        return None
    return word, frozenset(atoms | {carbon.GetIdx(), nitrogens[0].GetIdx()})


def noncarbon_replacement_acid_decision(mol, Chem):
    """Cyanido/cyanatido and halo replacement while acid OH remains."""
    if not _plain(mol, Chem, charged=True) or any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
        return None
    composition = Counter(a.GetAtomicNum() for a in mol.GetAtoms())
    if composition == Counter({16: 1, 8: 3, 7: 1, 6: 1}):
        query = Chem.MolFromSmarts('[S;X4](=[O;X1])(=[O;X1])([O;H1;D1])[N+;X2]#[C-;X1]')
        matches = mol.GetSubstructMatches(query)
        if (len(matches) == 1 and len(matches[0]) == mol.GetNumAtoms()
                and sum(abs(a.GetFormalCharge()) for a in mol.GetAtoms()) == 2):
            return _decision(mol, 'sulfurisocyanidic acid', (matches[0][0],),
                             'P-67.1.2.3/P-67.1.2.4', method='functional-replacement')
    if any(a.GetFormalCharge() for a in mol.GetAtoms()):
        return None
    # The compound isothiocyanatido infix uses parentheses and elides the
    # final o of the sulfuric-acid stem (P-67.1.2.4.1.1). Keep this path to
    # the complete published sulfuric acid skeleton; other cases delegate.
    if composition == Counter({16: 2, 8: 3, 7: 1, 6: 1}):
        query = Chem.MolFromSmarts('[S;X4](=[O;X1])(=[O;X1])([O;H1;D1])[N;X2]=[C;X2]=[S;X1]')
        matches = mol.GetSubstructMatches(query)
        if len(matches) == 1 and len(matches[0]) == mol.GetNumAtoms():
            return _decision(mol, 'sulfur(isothiocyanatidic) acid', (matches[0][0],),
                             'P-67.1.2.3/P-67.1.2.4', method='functional-replacement')
    centers = [a for a in mol.GetAtoms() if a.GetAtomicNum() in {15, 16}]
    if len(centers) != 1:
        return None
    center = centers[0]
    oxos, ohs, groups, carbons = [], [], [], []
    for atom in center.GetNeighbors():
        idx = atom.GetIdx()
        order = mol.GetBondBetweenAtoms(center.GetIdx(), idx).GetBondTypeAsDouble()
        if atom.GetAtomicNum() == 8 and atom.GetDegree() == 1 and order == 2:
            oxos.append(idx)
        elif (atom.GetAtomicNum() == 8 and atom.GetDegree() == 1 and order == 1
              and atom.GetTotalNumHs(includeNeighbors=True) == 1):
            ohs.append(idx)
        elif atom.GetAtomicNum() in {9, 17, 35, 53} and atom.GetDegree() == 1 and order == 1:
            groups.append(({9: 'fluorido', 17: 'chlorido', 35: 'bromido', 53: 'iodido'}[atom.GetAtomicNum()],
                           frozenset({idx})))
        elif pseudo := _cyano_replacement(mol, center.GetIdx(), idx):
            groups.append(pseudo)
        elif atom.GetAtomicNum() == 6 and order == 1:
            carbons.append(atom)
        else:
            return None
    if not ohs or not groups or len(groups) > 2 or len({row[0] for row in groups}) != 1:
        return None
    if center.GetAtomicNum() == 15:
        if (len(oxos) != 1 or len(carbons) > 1
                or len(ohs) + len(groups) + len(carbons) + center.GetTotalNumHs(includeNeighbors=True) != 3):
            return None
        fixed = len(carbons) + center.GetTotalNumHs(includeNeighbors=True)
        stem = {0: 'phosphoro', 1: 'phosphono', 2: 'phosphino'}.get(fixed)
    else:
        if len(oxos) != 2 or len(ohs) != 1 or len(groups) != 1 or carbons:
            return None
        stem = 'sulfuro'
    if stem is None:
        return None
    used = {center.GetIdx(), *oxos, *ohs}
    for _word, atoms in groups:
        if atoms & used:
            return None
        used.update(atoms)
    prefixes = []
    for root in carbons:
        text, atoms = mapped_branch(mol, Chem, root.GetIdx(), center.GetIdx())
        if not text or atoms & used or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in atoms):
            return None
        prefixes.append(('', text))
        used.update(atoms)
    if used != set(range(mol.GetNumAtoms())):
        return None
    word = groups[0][0]
    infix = (MULT[len(groups)] if len(groups) > 1 else '') + word
    # P-67.1.2.3.5 elides the terminal o before the acid ending.
    name = format_prefixes(prefixes, hierarchical_enclosures=True, omit_locants=True,
                           separate_unlocanted=True) + stem + infix[:-1] + 'ic acid'
    return _decision(mol, name, (center.GetIdx(),), 'P-67.1.2.3/P-67.1.2.4',
                     method='functional-replacement')


def phosphorus_halide_decision(mol, Chem):
    """Functional-class P/As/Sb acid halides with carbon-only substituents."""
    if not _plain(mol, Chem) or any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
        return None
    centers = [a for a in mol.GetAtoms() if a.GetAtomicNum() in {15, 33, 51}]
    if len(centers) != 1:
        return None
    center = centers[0]
    oxos, halides, roots = [], [], []
    for atom in center.GetNeighbors():
        order = mol.GetBondBetweenAtoms(center.GetIdx(), atom.GetIdx()).GetBondTypeAsDouble()
        if atom.GetAtomicNum() == 8 and atom.GetDegree() == 1 and order == 2:
            oxos.append(atom.GetIdx())
        elif atom.GetAtomicNum() in {9, 17, 35, 53} and atom.GetDegree() == 1 and order == 1:
            halides.append(atom.GetAtomicNum())
        elif atom.GetAtomicNum() == 6 and order == 1:
            roots.append(atom)
        else:
            return None
    if (not halides or len(oxos) > 1 or len(roots) > 2
            or len(halides) + len(roots) + center.GetTotalNumHs(includeNeighbors=True) != 3):
        return None
    used = {center.GetIdx(), *oxos}
    used.update(a.GetIdx() for a in center.GetNeighbors() if a.GetAtomicNum() in {9, 17, 35, 53})
    prefixes = []
    for root in roots:
        text, atoms = mapped_branch(mol, Chem, root.GetIdx(), center.GetIdx())
        if not text or atoms & used or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in atoms):
            return None
        prefixes.append(('', text))
        used.update(atoms)
    if used != set(range(mol.GetNumAtoms())):
        return None
    fixed = len(roots) + center.GetTotalNumHs(includeNeighbors=True)
    acid_stems = {15: ('phosphor', 'phosphon', 'phosphin'),
                  33: ('arsor', 'arson', 'arsin'), 51: ('stibor', 'stibon', 'stibin')}
    if fixed not in {0, 1, 2}:
        return None
    acid = acid_stems[center.GetAtomicNum()][fixed] + ('ic' if oxos else 'ous')
    classes = {9: 'fluoride', 17: 'chloride', 35: 'bromide', 53: 'iodide'}
    counts = Counter(halides)
    words = [(MULT[count] if count > 1 else '') + classes[number]
             for number, count in sorted(counts.items(), key=lambda item: classes[item[0]])]
    if center.GetAtomicNum() == 15 and oxos and not fixed and len(counts) == 1:
        acid = 'phosphoryl'
    name = format_prefixes(prefixes, hierarchical_enclosures=True, omit_locants=True,
                           separate_unlocanted=True) + acid + ' ' + ' '.join(words)
    return _decision(mol, name, (center.GetIdx(),), 'P-61.3.2.2/P-67.1.2.5',
                     method='functional-class')


def _capped_acid_name(mol, Chem):
    if (mol.GetNumAtoms() == 3 and sorted(a.GetAtomicNum() for a in mol.GetAtoms()) == [6, 7, 8]
            and any(b.GetBondTypeAsDouble() == 3 for b in mol.GetBonds())):
        return 'cyanic acid'
    hydroxyl = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 8 and a.GetDegree() == 1
                and a.GetTotalNumHs(includeNeighbors=True) == 1]
    if (len(hydroxyl) != 1 or any(a.GetAtomicNum() not in {6, 8, 15, 16, 34, 52, 9, 17, 35, 53}
                                 for a in mol.GetAtoms())):
        return ''
    from .api import mol_to_iupac
    name = mol_to_iupac(mol)
    if re.search(r' (?:O|S|Se|Te)-acid$', name):
        return re.sub(r' (?:O|S|Se|Te)-acid$', ' acid', name)
    return name if name.endswith(' acid') else ''


def linear_dianhydride_decision(mol, Chem):
    """Two monobasic acyl residues joined to a dicarboxylic acid residue."""
    if (not _plain(mol, Chem) or any(a.GetAtomicNum() not in {6, 8, 16} for a in mol.GetAtoms())
            or any(a.GetAtomicNum() == 8 and a.GetDegree() == 1 and a.GetTotalNumHs() for a in mol.GetAtoms())):
        return None
    bridges = []
    for oxygen in mol.GetAtoms():
        if oxygen.GetAtomicNum() != 8 or oxygen.GetDegree() != 2 or oxygen.IsInRing():
            continue
        ends = tuple(oxygen.GetNeighbors())
        if any(a.GetAtomicNum() != 6 or not _single(mol, oxygen.GetIdx(), a.GetIdx()) for a in ends):
            continue
        if any(not any(n.GetAtomicNum() in {8, 16} and n.GetDegree() == 1
                           and mol.GetBondBetweenAtoms(a.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 2
                           for n in a.GetNeighbors()) for a in ends):
            continue
        bridges.append((oxygen.GetIdx(), tuple(a.GetIdx() for a in ends)))
    if len(bridges) != 2:
        return None
    results = []
    for chosen in product((0, 1), repeat=2):
        copy = Chem.RWMol(mol)
        central_ends = []
        for (bridge, ends), side in zip(bridges, chosen, strict=True):
            endpoint = ends[side]
            central_ends.append(endpoint)
            copy.RemoveBond(bridge, endpoint)
            cap = copy.AddAtom(Chem.Atom(8))
            copy.AddBond(endpoint, cap, Chem.BondType.SINGLE)
            copy.GetAtomWithIdx(bridge).SetNoImplicit(False)
        split = _sanitize(copy, Chem)
        if split is None:
            continue
        atom_sets = Chem.GetMolFrags(split, asMols=False, sanitizeFrags=True)
        if len(atom_sets) != 3:
            continue
        central = [j for j, atoms in enumerate(atom_sets) if set(central_ends) <= set(atoms)]
        if len(central) != 1:
            continue
        fragments = Chem.GetMolFrags(split, asMols=True, sanitizeFrags=True)
        middle = fragments[central[0]]
        # A simple dicarboxylic acid has exactly two carbonyl oxygens and
        # two acid hydroxy oxygens, with no other heteroatom to compete.
        if (sum(a.GetAtomicNum() == 8 for a in middle.GetAtoms()) != 4
                or any(a.GetAtomicNum() not in {6, 8} for a in middle.GetAtoms())
                or sum(a.GetAtomicNum() == 8 and a.GetDegree() == 1 and a.GetTotalNumHs() == 1
                       for a in middle.GetAtoms()) != 2):
            continue
        from .api import mol_to_iupac
        middle_name = mol_to_iupac(middle)
        if not middle_name.endswith(' acid'):
            continue
        ends = [_capped_acid_name(fragment, Chem) for j, fragment in enumerate(fragments) if j != central[0]]
        simple = {'acetic acid', 'propanoic acid', 'butanoic acid', 'ethanethioic acid'}
        if len(ends) != 2 or any(name not in simple for name in ends):
            continue
        words = sorted((name[:-5] for name in ends), key=prefix_alphabetization_key)
        middle_word = middle_name[:-5]
        if words[0] == words[1]:
            name = 'di' + words[0] + ' ' + middle_word + ' dianhydride'
        else:
            name = words[0] + ' ' + middle_word + ' ' + words[1] + ' dianhydride'
        results.append(name)
    if len(set(results)) != 1:
        return None
    return _decision(mol, results[0], (), 'P-65.7.2/P-65.7.6.1/P-41', method='functional-class')


def mixed_anhydride_decision(mol, Chem):
    """Two monobasic acid residues, including cyanic and chalcogen acids."""
    if not _plain(mol, Chem) or any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
        return None
    results = []
    for bridge in mol.GetAtoms():
        if bridge.GetAtomicNum() not in {8, 16, 34, 52} or bridge.GetDegree() != 2 or bridge.IsInRing():
            continue
        left, right = bridge.GetNeighbors()
        if any(not _single(mol, bridge.GetIdx(), atom.GetIdx()) for atom in (left, right)):
            continue
        # Ordinary C(=O)–O–C(=O) anhydrides already have a mapped writer.
        def ordinary_acyl(atom):
            return (atom.GetAtomicNum() == 6 and any(n.GetAtomicNum() == 8 and n.GetDegree() == 1
                    and mol.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 2
                    for n in atom.GetNeighbors()))
        if bridge.GetAtomicNum()==8 and ordinary_acyl(left) and ordinary_acyl(right):
            continue
        def acid_endpoint(atom):
            if atom.GetAtomicNum() == 6:
                return any(n.GetDegree() == 1 and (
                    n.GetAtomicNum() in {8, 16, 34, 52}
                    and mol.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 2
                    or n.GetAtomicNum() == 7
                    and mol.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 3)
                    for n in atom.GetNeighbors())
            if atom.GetAtomicNum() in {16, 34, 52}:
                return any(n.GetAtomicNum() == 8 and n.GetDegree() == 1
                           and mol.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 2
                           for n in atom.GetNeighbors())
            return (atom.GetAtomicNum() == 15 and atom.GetTotalValence() == 3
                    and not atom.IsInRing() and all(n.GetIdx() == bridge.GetIdx()
                    or n.GetAtomicNum() == 6 for n in atom.GetNeighbors()))
        if not acid_endpoint(left) or not acid_endpoint(right):
            continue
        copy = Chem.RWMol(mol)
        copy.RemoveBond(bridge.GetIdx(), right.GetIdx())
        copy.GetAtomWithIdx(bridge.GetIdx()).SetAtomicNum(8)
        cap = copy.AddAtom(Chem.Atom(8))
        copy.AddBond(right.GetIdx(), cap, Chem.BondType.SINGLE)
        bridge_copy = copy.GetAtomWithIdx(bridge.GetIdx())
        bridge_copy.SetNumExplicitHs(0)
        bridge_copy.SetNoImplicit(False)
        split = _sanitize(copy, Chem)
        if split is None:
            continue
        fragments = Chem.GetMolFrags(split, asMols=True, sanitizeFrags=True)
        if len(fragments) != 2:
            continue
        acid_names = [_capped_acid_name(fragment, Chem) for fragment in fragments]
        if any(not name for name in acid_names) or acid_names[0] == acid_names[1]:
            continue
        words = sorted((name[:-5] for name in acid_names), key=prefix_alphabetization_key)
        modifier = {8: '', 16: 'thio', 34: 'seleno', 52: 'telluro'}[bridge.GetAtomicNum()]
        results.append(_decision(mol, ' '.join(words) + ' ' + modifier + 'anhydride',
                                 (left.GetIdx(), right.GetIdx()), 'P-65.7.2/P-41',
                                 method='functional-class'))
    return results[0] if len(results) == 1 else None


def substituted_diaryloxy_multiplication(mol, Chem):
    """Identically substituted benzene/pyridine units linked through oxygen."""
    if not _plain(mol, Chem) or any(a.GetAtomicNum() not in {6, 7, 8, 9, 17, 35, 53} for a in mol.GetAtoms()):
        return None
    rings = tuple(frozenset(r) for r in mol.GetRingInfo().AtomRings())
    if (len(rings) != 2 or any(len(r) != 6 for r in rings) or rings[0] & rings[1]
            or any(not mol.GetAtomWithIdx(i).GetIsAromatic() for ring in rings for i in ring)):
        return None
    compositions = [Counter(mol.GetAtomWithIdx(i).GetAtomicNum() for i in ring) for ring in rings]
    if compositions[0] != compositions[1] or compositions[0] not in (Counter({6: 6}), Counter({6: 5, 7: 1})):
        return None
    parent = 'benzene' if compositions[0][6] == 6 else 'pyridine'
    oxygens = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 8]
    if len(oxygens) != 1 or oxygens[0].GetDegree() != 2:
        return None
    oxygen = oxygens[0]
    roots = [a.GetIdx() for a in oxygen.GetNeighbors()]
    if any(not _single(mol, oxygen.GetIdx(), i) for i in roots):
        return None
    if len([root for root in roots if root in rings[0]]) != 1 or len([root for root in roots if root in rings[1]]) != 1:
        return None
    halogens = {9: 'fluoro', 17: 'chloro', 35: 'bromo', 53: 'iodo'}
    owned = set(rings[0] | rings[1] | {oxygen.GetIdx()})
    unit_names, links = [], []
    for ring in rings:
        root = next(i for i in roots if i in ring)
        attached = []
        for idx in ring:
            for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
                ni = neighbor.GetIdx()
                if ni in ring or ni == oxygen.GetIdx():
                    continue
                if not _single(mol, idx, ni) or ni in owned:
                    return None
                if neighbor.GetAtomicNum() in halogens and neighbor.GetDegree() == 1:
                    text, atoms = halogens[neighbor.GetAtomicNum()], frozenset({ni})
                elif neighbor.GetAtomicNum() == 6:
                    text, atoms = mapped_branch(mol, Chem, ni, idx)
                    if (not text or any(mol.GetAtomWithIdx(a).GetAtomicNum() not in {6, 9, 17, 35, 53}
                                        or mol.GetAtomWithIdx(a).IsInRing() for a in atoms)):
                        return None
                else:
                    return None
                if atoms & owned:
                    return None
                attached.append((idx, text, ni))
                owned.update(atoms)
        if not attached:
            return None
        start = root if parent == 'benzene' else next(i for i in ring if mol.GetAtomWithIdx(i).GetAtomicNum() == 7)
        neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(start).GetNeighbors() if n.GetIdx() in ring]
        rows = []
        for second in neighbors:
            sequence = [start, second]
            while len(sequence) < len(ring):
                nexts = [n.GetIdx() for n in mol.GetAtomWithIdx(sequence[-1]).GetNeighbors()
                         if n.GetIdx() in ring and n.GetIdx() != sequence[-2]]
                if len(nexts) != 1 or nexts[0] in sequence:
                    break
                sequence.append(nexts[0])
            if len(sequence) != len(ring) or not mol.GetBondBetweenAtoms(sequence[-1], start):
                continue
            mapping = {idx: pos for pos, idx in enumerate(sequence, 1)}
            prefixes = [(str(mapping[host]), text) for host, text, _ in attached]
            locants = tuple(sorted(mapping[host] for host, _text, _idx in attached))
            by_name = tuple(tuple(sorted(mapping[host] for host, text, _idx in attached if text == word))
                            for word in sorted({row[1] for row in attached}, key=prefix_alphabetization_key))
            rows.append(((mapping[root], locants, by_name),
                         format_prefixes(prefixes, hierarchical_enclosures=True), mapping[root]))
        if not rows:
            return None
        best = min(row[0] for row in rows)
        winners = [row for row in rows if row[0] == best]
        if len({row[1] for row in winners}) != 1:
            return None
        unit_names.append(winners[0][1] + parent)
        links.append(winners[0][2])
    if (owned != set(range(mol.GetNumAtoms())) or unit_names[0] != unit_names[1]
            or links[0] != links[1]):
        return None
    name = f"{links[0]},{links[1]}'-oxybis" + enclose_name_unit(unit_names[0])
    return _decision(mol, name, (), 'P-15.3.1.3/P-15.3.2.4.1', method='multiplicative')


def benzene_silyl_multiplication(mol, Chem):
    """Symmetric trimethylsilane units joined by a benzene multiprefix."""
    if not _plain(mol, Chem) or any(a.GetAtomicNum() not in {6, 14} for a in mol.GetAtoms()):
        return None
    rings = mol.GetRingInfo().AtomRings()
    if (len(rings) != 1 or len(rings[0]) != 6
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 or not mol.GetAtomWithIdx(i).GetIsAromatic()
                   for i in rings[0])):
        return None
    ring = frozenset(rings[0])
    silicon = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 14]
    if not 2 <= len(silicon) <= 4:
        return None
    attachments, owned = [], set(ring)
    for si in silicon:
        hosts = [n for n in si.GetNeighbors() if n.GetIdx() in ring]
        methyls = [n for n in si.GetNeighbors() if n.GetIdx() not in ring]
        if (len(hosts) != 1 or len(methyls) != 3 or si.GetTotalValence() != 4
                or any(a.GetAtomicNum() != 6 or a.GetDegree() != 1 or a.GetTotalNumHs() != 3
                       or not _single(mol, si.GetIdx(), a.GetIdx()) for a in methyls)
                or not _single(mol, si.GetIdx(), hosts[0].GetIdx())):
            return None
        attachments.append(hosts[0].GetIdx())
        owned.update({si.GetIdx(), *(a.GetIdx() for a in methyls)})
    if len(set(attachments)) != len(attachments) or owned != set(range(mol.GetNumAtoms())):
        return None
    locant_sets = []
    for start in ring:
        for neighbor in mol.GetAtomWithIdx(start).GetNeighbors():
            second = neighbor.GetIdx()
            if second not in ring:
                continue
            seq = [start, second]
            while len(seq) < 6:
                nxt = [a.GetIdx() for a in mol.GetAtomWithIdx(seq[-1]).GetNeighbors()
                       if a.GetIdx() in ring and a.GetIdx() != seq[-2]]
                if len(nxt) != 1 or nxt[0] in seq:
                    break
                seq.append(nxt[0])
            if len(seq) == 6:
                locant_sets.append(tuple(sorted(seq.index(i) + 1 for i in attachments)))
    if not locant_sets:
        return None
    locants = ','.join(map(str, min(locant_sets)))
    count = len(silicon)
    central = 'benzene-' + locants + '-' + {2: 'diyl', 3: 'triyl', 4: 'tetrayl'}[count]
    name = enclose_name_unit(central) + {2: 'bis', 3: 'tris', 4: 'tetrakis'}[count] + '(trimethylsilane)'
    return _decision(mol, name, (), 'P-15.3.1.3/P-15.3.2.4.1/P-51.3', method='multiplicative')


def indicated_h_ring_assembly_decision(mol, Chem):
    """Two complete identical mancude ring systems, with whole-name H locants."""
    if (not _plain(mol, Chem) or mol.GetNumAtoms() != mol.GetNumHeavyAtoms()
            or mol.GetNumAtoms() > 40):
        return None
    from .fusion_components import _mancude
    from .fusion_networks import skeleton_key
    from .mapped_ring_parents import _core_graph, assembly_unit_skeletons, ring_systems
    from .parent_selection import locant_key
    from .tree_ring_assemblies import _component_locant

    def assembly_locant_key(value):
        # A component prime precedes a fusion letter in names such as 3'a.
        match = re.fullmatch(r"(\d+)('*)([a-z]?)", value)
        return locant_key(match[1] + match[3] + match[2]) if match else None

    systems = tuple(ring_systems(mol))
    if len(systems) != 2 or systems[0] & systems[1] or systems[0] | systems[1] != frozenset(range(mol.GetNumAtoms())):
        return None
    joins = []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in systems[0] and b in systems[1]:
            joins.append((a, b, bond))
        elif b in systems[0] and a in systems[1]:
            joins.append((b, a, bond))
    if len(joins) != 1 or joins[0][2].GetBondType() != Chem.BondType.SINGLE:
        return None
    parent_names, numberings, keys = [], [], []
    allowed = {'pyrrole', 'pyran', 'indene', 'indole', 'isoindole', 'azepine'}
    for core in systems:
        prepared = _core_graph(mol, Chem, core)
        if prepared is None or not _mancude(prepared[0]):
            return None
        atoms = sorted(core)
        positions = {idx: pos for pos, idx in enumerate(atoms)}
        edges = [(positions[b.GetBeginAtomIdx()], positions[b.GetEndAtomIdx()]) for b in mol.GetBonds()
                 if b.GetBeginAtomIdx() in core and b.GetEndAtomIdx() in core]
        keys.append(skeleton_key([mol.GetAtomWithIdx(i).GetAtomicNum() for i in atoms], edges))
        rows = tuple(assembly_unit_skeletons(mol, Chem, core))
        if not rows:
            return None
        fixed = min(row[6] for row in rows)
        rows = [row for row in rows if row[6] == fixed]
        vocabulary = {re.sub(r'^(?:\d+[a-z]?H,)*\d+[a-z]?H-', '', row[0]) for row in rows}
        if len(vocabulary) != 1 or next(iter(vocabulary)) not in allowed:
            return None
        parent_names.append(next(iter(vocabulary)))
        numberings.append(tuple({tuple(zip(row[1], row[2], strict=True)) for row in rows}))
    if keys[0] != keys[1] or parent_names[0] != parent_names[1]:
        return None
    candidates = []
    for order in ((0, 1), (1, 0)):
        for first in numberings[order[0]]:
            for second in numberings[order[1]]:
                maps = (dict(first), dict(second))
                junction = (joins[0][order[0]], joins[0][order[1]])
                local_join = tuple(maps[i].get(junction[i]) for i in range(2))
                if any(not loc for loc in local_join):
                    continue
                hlocs = []
                for i, system in enumerate(order):
                    core = systems[system]
                    for idx in core:
                        atom = mol.GetAtomWithIdx(idx)
                        if not atom.GetTotalNumHs(includeNeighbors=True):
                            continue
                        bonds = [b.GetBondTypeAsDouble() for b in atom.GetBonds()
                                 if b.GetOtherAtomIdx(idx) in core]
                        indicated = (atom.GetAtomicNum() == 7 and atom.GetIsAromatic()
                                     or atom.GetAtomicNum() in {6, 7} and not atom.GetIsAromatic()
                                     and bonds and all(order == 1 for order in bonds))
                        if indicated:
                            loc = _component_locant(i + 1, str(maps[i][idx]), 2)
                            if not loc or assembly_locant_key(loc) is None:
                                return None
                            hlocs.append(loc)
                joined = (_component_locant(1, str(local_join[0]), 2),
                          _component_locant(2, str(local_join[1]), 2))
                if any(not x or assembly_locant_key(x) is None for x in joined):
                    continue
                hlocs = tuple(sorted(hlocs, key=assembly_locant_key))
                htext = ','.join(loc + 'H' for loc in hlocs) + '-' if hlocs else ''
                name = htext + ','.join(joined) + '-bi' + parent_names[0]
                key = (tuple(sorted(map(assembly_locant_key, joined))), tuple(map(assembly_locant_key, joined)),
                       tuple(map(assembly_locant_key, hlocs)))
                candidates.append((key, name, maps, order))
    if not candidates:
        return None
    best = min(row[0] for row in candidates)
    winners = [row for row in candidates if row[0] == best]
    if len({row[1] for row in winners}) != 1:
        return None
    _, name, maps, order = winners[0]
    bindings = tuple((idx, _component_locant(i + 1, str(loc), 2))
                     for i, mapping in enumerate(maps) for idx, loc in mapping.items())
    roles = tuple((f'ring-{i + 1}', systems[system], ()) for i, system in enumerate(order))
    return _decision(mol, name, (), 'P-28.2.3/P-52.2.7.1', roles,
                     method='ring-assembly', locants=bindings)
