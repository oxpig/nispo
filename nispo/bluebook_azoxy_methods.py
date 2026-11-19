"""Source-directed whole-graph constructions for selected P-63/P-65/P-68 cases.

These producers admit only graph partitions whose atoms, charge and competing
classes they can account for. Unsupported decorations remain on existing
routes rather than being silently discarded.
"""
from __future__ import annotations

import re

from .component_mapping import mapped_branch
from .constants import MULT, ROOTS
from .formatting import enclose_name_unit, format_prefixes, prefix_alphabetization_key
from .molecule_iteration import iter_atoms, iter_bonds
from .naming_methods import MethodDecision


def _plain(mol, Chem, *, charged=False):
    return (mol.GetNumHeavyAtoms() <= 64 and len(Chem.GetMolFrags(mol)) == 1
            and not mol.GetStereoGroups()
            and all(not a.GetIsotope() and not a.GetNumRadicalElectrons()
                    and not int(a.GetChiralTag()) and not a.HasQuery()
                    and (charged or not a.GetFormalCharge()) for a in iter_atoms(mol))
            and all(not int(b.GetStereo()) and not int(b.GetBondDir()) and not b.HasQuery()
                    for b in iter_bonds(mol)))


def _decision(mol, name, centers, rule, roles=(), method='substitutive', locants=()):
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    return MethodDecision(name, method, frozenset(centers), heavy, rule,
                          tuple(locants), component_roles=tuple(roles))


def azoxy_decision(mol, Chem):
    """Neutral R-N=N(O)-R with fully mapped, nonionic organyl arms."""
    if not _plain(mol, Chem, charged=True):
        return None
    if any(a.GetAtomicNum() not in {6, 7, 8, 9, 16, 17, 34, 35, 52, 53} for a in mol.GetAtoms()):
        return None
    positive = [a for a in mol.GetAtoms() if a.GetFormalCharge() == 1]
    negative = [a for a in mol.GetAtoms() if a.GetFormalCharge() == -1]
    if (len(positive) != 1 or len(negative) != 1
            or any(a.GetFormalCharge() not in {-1, 0, 1} for a in mol.GetAtoms())):
        return None
    oxidized, o = positive[0], negative[0]
    if (oxidized.GetAtomicNum() != 7 or o.GetAtomicNum() != 8 or o.GetDegree() != 1
            or o.GetNeighbors()[0].GetIdx() != oxidized.GetIdx()):
        return None
    partners = [n for n in oxidized.GetNeighbors() if n.GetAtomicNum() == 7
                and mol.GetBondBetweenAtoms(oxidized.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 2]
    if len(partners) != 1:
        return None
    other = partners[0]
    bond = mol.GetBondBetweenAtoms(oxidized.GetIdx(), other.GetIdx())
    if (other.GetFormalCharge() or other.GetDegree() != 2 or oxidized.GetDegree() != 3
            or bond is None or bond.GetBondTypeAsDouble() != 2
            or oxidized.IsInRing() or other.IsInRing()
            or mol.GetBondBetweenAtoms(o.GetIdx(), oxidized.GetIdx()).GetBondTypeAsDouble() != 1
            or any(a.GetFormalCharge() for a in mol.GetAtoms() if a.GetIdx() not in {o.GetIdx(), oxidized.GetIdx()})):
        return None
    from .diazene_methods import _carbon_arm, _order_arms, _render_arms, _render_numbered_arms
    arms = []
    for n in (oxidized, other):
        roots = [a for a in n.GetNeighbors() if a.GetAtomicNum() == 6]
        if len(roots) != 1 or mol.GetBondBetweenAtoms(n.GetIdx(), roots[0].GetIdx()).GetBondTypeAsDouble() != 1:
            return None
        name, atoms = _carbon_arm(mol, Chem, roots[0].GetIdx(), n.GetIdx(), n.GetIdx())
        if not name:
            name, atoms = mapped_branch(mol, Chem, roots[0].GetIdx(), n.GetIdx())
        if not name:
            return None
        arms.append((name, atoms, n.GetIdx()))
    if arms[0][1] & arms[1][1]:
        return None
    covered = arms[0][1] | arms[1][1] | {o.GetIdx(), oxidized.GetIdx(), other.GetIdx()}
    if covered != frozenset(a.GetIdx() for a in mol.GetAtoms()):
        return None
    ordered = _order_arms(mol, arms)
    if ordered[0][0] == ordered[1][0]:
        base = _render_arms(ordered)
        suffix = ' oxide'
    else:
        mapping = {row[2]: str(i) for i, row in enumerate(ordered, 1)}
        base = _render_numbered_arms(mol, ordered, mapping)
        suffix = ' ' + mapping[oxidized.GetIdx()] + '-oxide'
    return _decision(mol, base + suffix, (oxidized.GetIdx(), other.GetIdx()),
                     'P-68.3.1.3.3',
                     (("diazene-oxide", frozenset({o.GetIdx(), oxidized.GetIdx(), other.GetIdx()}), ()),
                      ("organic-arms", arms[0][1] | arms[1][1], ())))


def cyanic_ester_decision(mol, Chem):
    """Organyl ester of a simple cyanic acid chalcogen analogue."""
    if not _plain(mol, Chem):
        return None
    atoms = tuple(mol.GetAtoms())
    nitrogens = [a for a in atoms if a.GetAtomicNum() == 7]
    links = [a for a in atoms if a.GetAtomicNum() in {8, 16, 34, 52}]
    if len(nitrogens) != 1 or len(links) != 1 or any(a.GetAtomicNum() not in {6, 7, 8, 16, 34, 52} for a in atoms):
        return None
    n, x = nitrogens[0], links[0]
    if n.GetDegree() != 1 or x.GetDegree() != 2:
        return None
    carbon = n.GetNeighbors()[0]
    if (carbon.GetAtomicNum() != 6 or carbon.GetDegree() != 2
            or mol.GetBondBetweenAtoms(n.GetIdx(), carbon.GetIdx()).GetBondTypeAsDouble() != 3
            or mol.GetBondBetweenAtoms(carbon.GetIdx(), x.GetIdx()) is None
            or mol.GetBondBetweenAtoms(carbon.GetIdx(), x.GetIdx()).GetBondTypeAsDouble() != 1):
        return None
    roots = [a for a in x.GetNeighbors() if a.GetIdx() != carbon.GetIdx()]
    if len(roots) != 1 or roots[0].GetAtomicNum() != 6 or mol.GetBondBetweenAtoms(x.GetIdx(), roots[0].GetIdx()).GetBondTypeAsDouble() != 1:
        return None
    name, arm = mapped_branch(mol, Chem, roots[0].GetIdx(), x.GetIdx())
    if not name or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in arm):
        return None
    if arm | {n.GetIdx(), x.GetIdx(), carbon.GetIdx()} != frozenset(a.GetIdx() for a in atoms):
        return None
    anion = {8: 'cyanate', 16: 'thiocyanate', 34: 'selenocyanate', 52: 'tellurocyanate'}[x.GetAtomicNum()]
    return _decision(mol, name + ' ' + anion, (carbon.GetIdx(),), 'P-65.6.3.3.7.2.1',
                     (("cyanic-acid-residue", frozenset({n.GetIdx(), carbon.GetIdx(), x.GetIdx()}), ()),
                      ("organyl", arm, ())), method='functional-class')


def thiohydroxylamine_decision(mol, Chem):
    """S-organyl thiohydroxylamines, when no competing class is present."""
    if not _plain(mol, Chem):
        return None
    ns = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    ss = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 16]
    if len(ns) != 1 or len(ss) != 1 or any(a.GetAtomicNum() not in {6, 7, 16} for a in mol.GetAtoms()):
        return None
    n, s = ns[0], ss[0]
    if (n.GetDegree() != 1 or n.GetTotalNumHs(includeNeighbors=True) != 2 or s.GetDegree() != 2
            or mol.GetBondBetweenAtoms(n.GetIdx(), s.GetIdx()) is None
            or mol.GetBondBetweenAtoms(n.GetIdx(), s.GetIdx()).GetBondTypeAsDouble() != 1):
        return None
    roots = [a for a in s.GetNeighbors() if a.GetIdx() != n.GetIdx()]
    if len(roots) != 1 or roots[0].GetAtomicNum() != 6 or mol.GetBondBetweenAtoms(s.GetIdx(), roots[0].GetIdx()).GetBondTypeAsDouble() != 1:
        return None
    name, arm = mapped_branch(mol, Chem, roots[0].GetIdx(), s.GetIdx())
    if not name or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in arm):
        return None
    if arm | {n.GetIdx(), s.GetIdx()} != frozenset(a.GetIdx() for a in mol.GetAtoms()):
        return None
    name = name if name.isalpha() else enclose_name_unit(name)
    return _decision(mol, 'S-' + name + '(thiohydroxylamine)', (n.GetIdx(), s.GetIdx()),
                     'P-68.3.1.1.1.6',
                     (("thiohydroxylamine", frozenset({n.GetIdx(), s.GetIdx()}), ()), ("organyl", arm, ())))


def _hydrocarbon_parent(mol, atoms, root):
    """A complete simple linear alkane or uniform carbocycle and root locant."""
    if not atoms or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in atoms):
        return None
    adj = {i: [n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in atoms] for i in atoms}
    bonds = [b for b in mol.GetBonds() if b.GetBeginAtomIdx() in atoms and b.GetEndAtomIdx() in atoms]
    if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in atoms):
        return ('benzene', '', True) if len(atoms) == 6 and all(len(v) == 2 for v in adj.values()) else None
    if any(b.GetBondTypeAsDouble() != 1 for b in bonds):
        return None
    if all(len(v) == 2 for v in adj.values()) and len(atoms) in ROOTS:
        return ('cyclo' + ROOTS[len(atoms)] + 'ane', '', True)
    if any(len(v) > 2 for v in adj.values()) or len(atoms) not in ROOTS:
        return None
    ends = [i for i, v in adj.items() if len(v) <= 1]
    if len(atoms) > 1 and len(ends) != 2:
        return None
    path, prev, cur = [], None, ends[0]
    while cur is not None:
        path.append(cur)
        follow = [i for i in adj[cur] if i != prev]
        prev, cur = cur, follow[0] if follow else None
    if set(path) != set(atoms):
        return None
    loc = min(path.index(root) + 1, len(path) - path.index(root))
    return (ROOTS[len(atoms)] + 'ane', str(loc) if len(atoms) >= 3 else '', False)


def mixed_peroxide_decision(mol, Chem):
    """Two different linked chalcogens and two complete hydrocarbon parents."""
    if not _plain(mol, Chem):
        return None
    hetero = [a for a in mol.GetAtoms() if a.GetAtomicNum() in {8, 16, 34, 52}]
    if len(hetero) != 2 or hetero[0].GetAtomicNum() == hetero[1].GetAtomicNum() or any(a.GetAtomicNum() not in {6, 8, 16, 34, 52} for a in mol.GetAtoms()):
        return None
    x, y = hetero
    bond = mol.GetBondBetweenAtoms(x.GetIdx(), y.GetIdx())
    if bond is None or bond.GetBondTypeAsDouble() != 1 or x.GetDegree() != 2 or y.GetDegree() != 2:
        return None
    arms = []
    for a, other in ((x, y), (y, x)):
        roots = [n for n in a.GetNeighbors() if n.GetIdx() != other.GetIdx()]
        if len(roots) != 1 or roots[0].GetAtomicNum() != 6 or mol.GetBondBetweenAtoms(a.GetIdx(), roots[0].GetIdx()).GetBondTypeAsDouble() != 1:
            return None
        text, atoms = mapped_branch(mol, Chem, roots[0].GetIdx(), a.GetIdx())
        if not text or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in atoms):
            return None
        parent = _hydrocarbon_parent(mol, atoms, roots[0].GetIdx())
        if parent is None:
            return None
        arms.append((text, atoms, parent, a))
    if arms[0][1] & arms[1][1] or arms[0][1] | arms[1][1] | {x.GetIdx(), y.GetIdx()} != frozenset(a.GetIdx() for a in mol.GetAtoms()):
        return None
    terms = {8: 'oxy', 16: 'sulfanyl', 34: 'selanyl', 52: 'tellanyl'}
    options = []
    for p, q in ((arms[0], arms[1]), (arms[1], arms[0])):
        org = q[0]
        first = (org[:-2] + 'oxy' if q[3].GetAtomicNum() == 8 and org.endswith('yl')
                 else (org if org.isalpha() else enclose_name_unit(org)) + terms[q[3].GetAtomicNum()])
        if q[3].GetAtomicNum() != 8:
            first = enclose_name_unit(first)
        prefix = first + terms[p[3].GetAtomicNum()]
        parent_name, loc, ring = p[2]
        result = (loc + '-' if loc else '') + enclose_name_unit(prefix) + parent_name
        score = (-int(ring), -len(p[1]), prefix_alphabetization_key(prefix), result)
        options.append((score, result))
    return _decision(mol, min(options)[1], (x.GetIdx(), y.GetIdx()), 'P-63.3.2/P-51.4.1.1',
                     (("chalcogen-linkage", frozenset({x.GetIdx(), y.GetIdx()}), ()),
                      ("hydrocarbon-arms", arms[0][1] | arms[1][1], ())))


def biphenyl_polyamine_decision(mol, Chem):
    """Two joined benzene units, primary amino groups, methyls or halogens."""
    if not _plain(mol, Chem) or any(a.GetAtomicNum() not in {6, 7, 9, 17, 35, 53} for a in mol.GetAtoms()):
        return None
    rings = [frozenset(r) for r in mol.GetRingInfo().AtomRings()]
    if (len(rings) != 2 or any(len(r) != 6 for r in rings) or rings[0] & rings[1]
            or not all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6 and mol.GetAtomWithIdx(i).GetIsAromatic()
                       for ring in rings for i in ring)):
        return None
    nitrogens = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    if not 2 <= len(nitrogens) <= 6:
        return None
    for n in nitrogens:
        if (n.GetDegree() != 1 or n.GetTotalNumHs(includeNeighbors=True) != 2
                or n.GetNeighbors()[0].GetIdx() not in rings[0] | rings[1]
                or mol.GetBondBetweenAtoms(n.GetIdx(), n.GetNeighbors()[0].GetIdx()).GetBondTypeAsDouble() != 1):
            return None
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms())
    core = rings[0] | rings[1]
    amino = {n.GetIdx() for n in nitrogens}
    extras = []
    for idx in heavy - core - amino:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetDegree() != 1 or atom.GetNeighbors()[0].GetIdx() not in core:
            return None
        host = atom.GetNeighbors()[0].GetIdx()
        if mol.GetBondBetweenAtoms(idx, host).GetBondTypeAsDouble() != 1:
            return None
        if atom.GetAtomicNum() == 6 and atom.GetTotalNumHs(includeNeighbors=True) == 3:
            text = 'methyl'
        else:
            text = {9: 'fluoro', 17: 'chloro', 35: 'bromo', 53: 'iodo'}.get(atom.GetAtomicNum())
        if not text:
            return None
        extras.append((host, idx, text))
    bridges = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()
               if ((b.GetBeginAtomIdx() in rings[0] and b.GetEndAtomIdx() in rings[1])
                   or (b.GetBeginAtomIdx() in rings[1] and b.GetEndAtomIdx() in rings[0]))
               and b.GetBondTypeAsDouble() == 1]
    if len(bridges) != 1:
        return None
    ends = bridges[0]
    anchors = [next(i for i in ends if i in ring) for ring in rings]
    choices = []
    for ring, anchor in zip(rings, anchors):
        adj = {i: [n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in ring] for i in ring}
        if any(len(v) != 2 for v in adj.values()):
            return None
        seqs = []
        for first in adj[anchor]:
            seq, previous, current = [anchor], anchor, first
            while current != anchor and len(seq) <= 6:
                seq.append(current)
                nxt = [i for i in adj[current] if i != previous]
                if len(nxt) != 1:
                    break
                previous, current = current, nxt[0]
            if current == anchor and len(seq) == 6:
                seqs.append(seq)
        if len(seqs) != 2:
            return None
        choices.append(seqs)
    candidates = []
    for swap in (False, True):
        a, b = (1, 0) if swap else (0, 1)
        for first in choices[a]:
            for second in choices[b]:
                mapping = {i: (loc, 0) for loc, i in enumerate(first, 1)}
                mapping.update({i: (loc, 1) for loc, i in enumerate(second, 1)})
                locs = sorted(mapping[n.GetNeighbors()[0].GetIdx()] for n in nitrogens)
                printed = ','.join(str(number) + "'" * prime for number, prime in locs)
                mapped = tuple((idx, str(number) + "'" * prime) for idx, (number, prime) in mapping.items())
                extra_locs = tuple(sorted(mapping[host] for host, _idx, _text in extras))
                extra_names = sorted({text for _host, _idx, text in extras}, key=prefix_alphabetization_key)
                per_name = tuple(tuple(sorted(mapping[host] for host, _idx, text in extras if text == wanted))
                                 for wanted in extra_names)
                prefixes = [(str(mapping[host][0]) + "'" * mapping[host][1], text)
                            for host, _idx, text in extras]
                prefix = format_prefixes(prefixes, hierarchical_enclosures=True)
                multiplier = MULT[len(nitrogens)].removesuffix('a')
                name = prefix + "[1,1'-biphenyl]-" + printed + '-' + multiplier + 'amine'
                candidates.append(((tuple(locs), extra_locs, per_name), name, mapped))
    score, name, mapping = min(candidates, key=lambda row: (row[0], row[1]))
    centers = frozenset(n.GetIdx() for n in nitrogens)
    return _decision(mol, name, centers, 'P-44.1.1/P-62.2.4.1/P-28.2.1',
                     (("biphenyl", core, mapping), ("amino-groups", centers, ()),
                      ("ring-substituents", frozenset(idx for _host, idx, _text in extras), ())),
                     locants=mapping)


def aromatic_ring_cation_decision(mol, Chem):
    """A monocyclic diazine/pyridine cation with all charges on its ring."""
    if not _plain(mol, Chem, charged=True) or any(a.GetAtomicNum() not in {6, 7} for a in mol.GetAtoms()):
        return None
    rings = mol.GetRingInfo().AtomRings()
    if len(rings) != 1 or len(rings[0]) != 6:
        return None
    ring = frozenset(rings[0])
    if any(not mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
        return None
    positive = [a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() == 1]
    if (not positive or len(positive) > 2 or any(i not in ring or mol.GetAtomWithIdx(i).GetAtomicNum() != 7 for i in positive)
            or any(a.GetFormalCharge() not in {0, 1} for a in mol.GetAtoms())):
        return None
    branches = []
    seen = set(ring)
    for host in ring:
        for atom in mol.GetAtomWithIdx(host).GetNeighbors():
            root = atom.GetIdx()
            if root in ring:
                continue
            if atom.GetAtomicNum() != 6 or mol.GetBondBetweenAtoms(host, root).GetBondTypeAsDouble() != 1:
                return None
            text, atoms = mapped_branch(mol, Chem, root, host)
            if not text or atoms & seen or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in atoms):
                return None
            branches.append((host, text, atoms))
            seen.update(atoms)
    if seen != set(range(mol.GetNumAtoms())):
        return None
    copy = Chem.RWMol(mol)
    for i in positive:
        n = copy.GetAtomWithIdx(i)
        n.SetFormalCharge(0)
        n.SetNumExplicitHs(0)
        n.SetNoImplicit(False)
    for i in reversed(range(mol.GetNumAtoms())):
        if i not in ring:
            copy.RemoveAtom(i)
    core = copy.GetMol()
    try:
        Chem.SanitizeMol(core)
    except (ValueError, RuntimeError):
        return None
    from .cycle_utils import ordered_ring
    from .ring_rules import ring_parent_candidates
    old = sorted(ring)
    ordered = ordered_ring(core, core.GetRingInfo().AtomRings()[0])
    if not ordered:
        return None
    candidates = []
    for parent, seq in ring_parent_candidates(core, ordered):
        if parent not in {'pyridine', 'pyridazine', 'pyrimidine', 'pyrazine'}:
            continue
        mapping = {old[idx]: number for number, idx in enumerate(seq, 1)}
        charge_locs = tuple(sorted(mapping[i] for i in positive))
        prefixes = [(str(mapping[host]), text) for host, text, _ in branches]
        branch_locs = tuple(sorted(mapping[host] for host, _text, _atoms in branches))
        prefix = format_prefixes(prefixes, hierarchical_enclosures=True)
        suffix = ('-'.join((parent[:-1], ','.join(map(str, charge_locs)),
                           ('ium' if len(positive) == 1 else MULT[len(positive)] + 'ium'))))
        candidates.append(((charge_locs, branch_locs, prefix), prefix + suffix, mapping))
    if not candidates:
        return None
    _key, name, mapping = min(candidates, key=lambda row: row[0])
    locants = tuple((i, str(loc)) for i, loc in mapping.items())
    return _decision(mol, name, positive, 'P-73.7/P-73.1.1.2',
                     (("heterocyclic-cation", ring, locants),
                      ("substituents", frozenset().union(*(row[2] for row in branches)), ())),
                     locants=locants)


def identical_polyspiro_decision(mol, Chem):
    """Identical rooted components with proved whole-system hydrogen state."""
    if (not _plain(mol, Chem) or mol.GetRingInfo().NumRings() < 4
            or any(a.GetAtomicNum() not in {6, 7, 8, 16} for a in mol.GetAtoms())):
        return None
    from .ring_topology import ring_system_graph
    from .spiro_component_rules import (_two_component_splits_by_articulation,
        _submol_for_atoms, _poly_component_names, _cycle_rank_for_atoms)
    atoms = set(range(mol.GetNumAtoms()))
    if set().union(*(set(r) for r in mol.GetRingInfo().AtomRings())) != atoms:
        return None
    graph = ring_system_graph(mol, Chem, atoms)
    splits = _two_component_splits_by_articulation(graph, atoms)
    if len(splits) != 1:
        return None
    spiro, parts = next(iter(splits.items()))
    if len(parts) != 2 or mol.GetAtomWithIdx(spiro).GetAtomicNum() != 6:
        return None
    saturated = all(b.GetBondTypeAsDouble() == 1 for b in mol.GetBonds())
    if saturated and any(a.GetAtomicNum() != 6 for a in mol.GetAtoms()):
        return None
    components = []
    for part in parts:
        subset = set(part) | {spiro}
        if _cycle_rank_for_atoms(graph, subset) < 2:
            return None
        row = _submol_for_atoms(mol, Chem, subset)
        if row is None:
            return None
        sub, back, forward = row
        marked = Chem.Mol(sub)
        marked.GetAtomWithIdx(forward[spiro]).SetIsotope(13)
        identity = Chem.MolToSmiles(marked, isomericSmiles=True, canonical=True)
        names = []
        if not saturated:
            from .mapped_ring_parents import skeletons
            bases = {sk[0] for sk in skeletons(sub, Chem, frozenset(range(sub.GetNumAtoms())))}
        for c in _poly_component_names(sub, Chem, forward[spiro]):
            if c.replacement_prefixes or not c.spiro_locant.isdigit():
                continue
            parent, indicated = c.parent_name, None
            if not saturated:
                match = re.fullmatch(r'(\d+),(\d+)-dihydro(.+)', parent)
                if match:
                    locs = (match.group(1), match.group(2))
                    if c.spiro_locant not in locs:
                        continue
                    indicated = locs[1] if locs[0] == c.spiro_locant else locs[0]
                    h_atoms = [back[i] for i, loc in c.locants.items() if loc == indicated]
                    if (len(h_atoms) != 1 or mol.GetAtomWithIdx(h_atoms[0]).GetAtomicNum() != 6
                            or mol.GetAtomWithIdx(h_atoms[0]).GetTotalNumHs(includeNeighbors=True) != 2):
                        continue
                    parent = match.group(3).removeprefix('-')
                normalized = re.sub(r'^\d+-', '', parent)
                if normalized not in bases:
                    continue
                # A retained parent may already contain an indicated-H site;
                # permit it only when its original H is replaced by the spiro
                # junction, or the dihydro proof accounts for the extra site.
                extra_ch2 = [back[i] for i in c.locants if back[i] != spiro
                             and mol.GetAtomWithIdx(back[i]).GetAtomicNum() == 6
                             and mol.GetAtomWithIdx(back[i]).GetTotalNumHs(includeNeighbors=True) == 2]
                if len(extra_ch2) != (1 if indicated else 0):
                    continue
                parent = re.sub(r'^(\d+)-(?=benzo)', r'[\1]', parent)
            names.append((c, parent, indicated))
        if not names:
            return None
        components.append((subset, back, identity, names))
    if components[0][2] != components[1][2]:
        return None
    choices = []
    for left, left_parent, left_h in components[0][3]:
        for right, right_parent, right_h in components[1][3]:
            if left_parent != right_parent or left_h != right_h:
                continue
            ll, rr = int(left.spiro_locant), int(right.spiro_locant)
            if ll != rr:
                continue
            key = (left.component_rank + right.component_rank,
                   left.numbering_key, right.numbering_key, ll,
                   left.component_order + right.component_order, left_parent)
            choices.append((key, left, right, left_parent, left_h))
    if not choices:
        return None
    _key, left, right, parent, indicated = min(choices, key=lambda row: row[0])
    locant = left.spiro_locant
    hydrogen = f"{indicated}H,{indicated}'H-" if indicated else ''
    name = f"{hydrogen}{locant},{locant}'-spirobi[{parent}]"
    mapping = [(components[0][1][i], loc) for i, loc in left.locants.items()]
    mapping += [(components[1][1][i], loc + "'") for i, loc in right.locants.items()
                if components[1][1][i] != spiro]
    return _decision(mol, name, (spiro,), 'P-24.3.1/P-24.3.2',
                     (("spiro-component-1", frozenset(components[0][0]), ()),
                      ("spiro-component-2", frozenset(components[1][0] - {spiro}), ())),
                     locants=mapping)


def chalcogen_oxoacid_diamide_decision(mol, Chem):
    """Complete sulfurous, selenic/selenous or telluric/tellurous diamides."""
    if not _plain(mol, Chem):
        return None
    atoms = list(mol.GetAtoms())
    centers = [a for a in atoms if a.GetAtomicNum() in {16, 34, 52}]
    if len(centers) != 1 or any(a.GetAtomicNum() not in {6, 7, 8, 16, 34, 52} for a in atoms):
        return None
    center = centers[0]
    oxygen, nitrogen = [], []
    for atom in center.GetNeighbors():
        order = mol.GetBondBetweenAtoms(center.GetIdx(), atom.GetIdx()).GetBondTypeAsDouble()
        if atom.GetAtomicNum() == 8 and atom.GetDegree() == 1 and order == 2:
            oxygen.append(atom.GetIdx())
        elif atom.GetAtomicNum() == 7 and not atom.IsInRing() and order == 1:
            nitrogen.append(atom)
        else:
            return None
    if len(nitrogen) != 2 or len(oxygen) not in {1, 2}:
        return None
    if center.GetAtomicNum() == 16 and len(oxygen) == 2:
        return None  # Existing sulfuric diamide writer has a broader branch scope.
    sides, owned = [], {center.GetIdx(), *oxygen}
    for n in nitrogen:
        if n.GetFormalCharge() or n.GetTotalValence() != 3:
            return None
        pieces = []
        owned.add(n.GetIdx())
        for root in n.GetNeighbors():
            if root.GetIdx() == center.GetIdx():
                continue
            if root.GetAtomicNum() != 6 or mol.GetBondBetweenAtoms(n.GetIdx(), root.GetIdx()).GetBondTypeAsDouble() != 1:
                return None
            text, branch = mapped_branch(mol, Chem, root.GetIdx(), n.GetIdx())
            if not text or branch & owned or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in branch):
                return None
            owned.update(branch)
            pieces.append(text)
        if len(pieces) > 2:
            return None
        sides.append(pieces)
    if owned != set(range(mol.GetNumAtoms())):
        return None
    sides.sort(key=lambda row: (not bool(row), tuple(sorted((prefix_alphabetization_key(x) for x in row)))))
    prefixes = [(loc, text) for loc, row in zip(('N', "N'"), sides) for text in row]
    parent = {(16, 1): 'sulfurous', (34, 1): 'selenous', (34, 2): 'selenic',
              (52, 1): 'tellurous', (52, 2): 'telluric'}.get((center.GetAtomicNum(), len(oxygen)))
    if parent is None:
        return None
    return _decision(mol, format_prefixes(prefixes, hierarchical_enclosures=True) + parent + ' diamide',
                     (center.GetIdx(),), 'P-67.1.2.6.1',
                     (("acid-center", frozenset({center.GetIdx(), *oxygen, *(n.GetIdx() for n in nitrogen)}), ()),
                      ("nitrogen-substituents", frozenset(owned - {center.GetIdx(), *oxygen, *(n.GetIdx() for n in nitrogen)}), ())),
                     method='functional-class')
