"""Mapped imine, repeated-ring and hydroxylamine parent assemblies."""
from itertools import product
from functools import lru_cache
from types import SimpleNamespace
from typing import NamedTuple

from .functional_parent_extensions import side, capped, order
from .bluebook_lambda_amine_methods import _plain, _out
from .constants import MULT, ROOTS
from .formatting import enclose_name_unit, format_prefixes


class _RingUnit(NamedTuple):
    identity: str
    parent: str
    locant: str
    has_principal_group: bool = False
    mononuclear: bool = False
    stereo: tuple = ()


def _phenol_unit(mol, atoms, root, centers):
    """A phenolic unit with every hydroxyl on one benzene parent."""
    ring = next((r for r in mol.GetRingInfo().AtomRings() if root in r), ())
    if (len(ring) != 6 or not set(ring) <= atoms or not centers <= set(ring)
            or not centers
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 or not mol.GetAtomWithIdx(i).GetIsAromatic()
                   for i in ring)):
        return None
    from .formatting import attach_suffix, multiplied_suffix

    parts = ()
    if len(atoms) != 6 + len(centers):
        from rdkit import Chem
        from .retained_hydrocarbon_derivatives import ring_pendants, ring_substitution

        parts = ring_pendants(mol, Chem, frozenset(ring), scope=atoms)
        if (parts is None or any(p.kind not in ('', 'alcohol') for p in parts)
                or {p.host for p in parts if p.kind == 'alcohol'} != centers):
            return None
    choices = []
    for direction in (ring, ring[::-1]):
        for start in range(6):
            seq = direction[start:] + direction[:start]
            locants = tuple(i for i, atom in enumerate(seq, 1) if atom in centers)
            choices.append((locants, seq.index(root) + 1, seq))
    locants, attachment = min((locs, loc) for locs, loc, _ in choices)
    if parts:
        # The linker gets its lowest available locant after the hydroxyls.
        # Compare ordinary substituents only between the remaining numberings.
        mono = len(locants) == 1
        pendants = [p for p in parts if not p.kind] if mono else parts
        rows = [ring_substitution('phenol' if mono else 'benzene',
                                  {idx: str(j) for j, idx in enumerate(seq, 1)}, pendants)
                for locs, loc, seq in choices if (locs, loc) == (locants, attachment)]
        if any(row is None for row in rows):
            return None
        best = min(key for key, _, _ in rows)
        names = {name for key, name, _ in rows if key == best}
        if len(names) != 1:
            return None
        parent = names.pop()
    else:
        parent = 'phenol' if len(locants) == 1 else attach_suffix(
            'benzene', multiplied_suffix(len(locants), 'ol'), locants)
    return _RingUnit('phenolic-unit', parent, str(attachment), True)


def unit(mol, Chem, atoms, root):
    """Choose a fully capped unit independently of its central linker."""
    fragment, original = capped(mol, Chem, atoms)
    if fragment is None:
        return None
    local = original.index(root)
    fragment.GetAtomWithIdx(local).SetAtomMapNum(1)
    identity = Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=True)
    fragment.GetAtomWithIdx(local).SetAtomMapNum(0)
    from .extended_inventory import selection_inventory
    inv = selection_inventory(fragment)
    if not inv.supported:
        return None
    from .naming_methods import _unit_parent_cache_entry
    if inv.principal:
        kinds = {g.kind for g in inv.principal}
        if len(kinds) != 1 or not kinds <= {'acid','alcohol'}:
            return None
        kind = next(iter(kinds))
        if kind == 'alcohol' and fragment.GetRingInfo().NumRings() == 1:
            phenol = _phenol_unit(fragment, frozenset(range(len(original))), local, {g.center for g in inv.principal})
            if phenol is not None:
                return phenol
        pool = _unit_parent_cache_entry(fragment, Chem, local, kind, compare=kind != 'alcohol')
        if kind != 'alcohol':
            if pool.comparison.reason or pool.comparison.selected is None:
                return None
            row = pool.comparison.selected
            return _RingUnit(identity, row[1], str(row[2]), True)
        # Alcohol units have one parent family here. Numbering is fixed by
        # the suffix before minimizing the attachment locant.
        rows = pool.rows
        if not rows or (fragment.GetRingInfo().NumRings() and len({r[1] for r in rows}) != 1):
            return None
        key = min((r[0], int(r[2])) for r in rows)
        winners = [r for r in rows if (r[0], int(r[2])) == key]
        if len({r[1] for r in winners}) != 1:
            return None
        row = winners[0]
        return _RingUnit(identity, row[1], str(row[2]), True, len(row[7]) == 1)
    rings = frozenset(i for ring in fragment.GetRingInfo().AtomRings() for i in ring)
    if rings != frozenset(range(fragment.GetNumAtoms())) or local not in rings:
        return None
    from .mapped_ring_parents import assembly_unit_skeletons
    rows = list(assembly_unit_skeletons(fragment, Chem, rings))
    if not rows or len({r[0] for r in rows}) != 1:
        return None
    locants = [dict(zip(r[1],r[2]))[local] for r in rows]
    if not all(s.isdigit() for s in locants):
        return None
    return _RingUnit(identity, rows[0][0], min(locants, key=int))


def _assembly_name(parent, locant, bridge, count=2, *, mononuclear=False, stereo=(), simple_bridge=False):
    # bis avoids implying a continuous hydride, and encloses parents with
    # internal punctuation (P-15.3.2.3/P-16.3).
    complex_parent = parent.startswith(('di','tri')) or parent[0].isdigit()
    if parent.endswith(('silane','phosphane','azane','sulfane','oxidane','hydroxylamine')):
        complex_parent = True
    enclosed = complex_parent or any(c.isdigit() or c in '()[]' for c in parent)
    multiplier = {2:'bis', 3:'tris', 4:'tetrakis'}[count] if complex_parent else MULT[count]
    suffix = multiplier + (enclose_name_unit(parent) if enclosed else parent)
    simple = {'methylene', 'oxy', 'peroxy', 'sulfanediyl', 'disulfanediyl', 'azanediyl',
              'diazenediyl', 'selanediyl', 'tellanediyl', 'sulfinyl', 'sulfonyl',
              'seleninyl', 'selenonyl', 'tellurinyl', 'telluronyl'}
    from .naming_methods import _LINK_HYDRIDES
    simple.update(hydride + ending for hydride in _LINK_HYDRIDES.values()
                  for ending in ('diyl', 'triyl', 'tetrayl'))
    connector = bridge if simple_bridge or bridge in simple else enclose_name_unit(bridge)
    locants = '' if mononuclear else ','.join(locant + "'" * i for i in range(count)) + '-'
    descriptors = [str(position) + "'" * i + descriptor for position, descriptor in stereo
                   for i in range(count)]
    leading = '(' + ','.join(descriptors) + ')-' if descriptors else ''
    return leading + locants + connector + suffix


def _branched_alcohol_paths(mol, carbon, hydroxyls, root, atom_stereo):
    """Retain every senior orientation before asking whether the linker is on it."""
    from collections import Counter
    from rdkit import Chem
    from .substituent_selection import mapped_saturated_carbon_prefix
    from .tree_path_frontier import maximum_weight_leaf_paths

    counts = Counter(hydroxyls)
    adjacency = {i: [] for i in carbon}
    orders = {}
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in carbon and b in carbon:
            order = bond.GetBondTypeAsDouble()
            if order not in (1, 2, 3):
                return ()
            orders[min(a, b), max(a, b)] = order
            adjacency[a].append(b)
            adjacency[b].append(a)
    weights = {edge: (0, 0, int(n > 1), int(n == 2)) for edge, n in orders.items()}
    best, paths = maximum_weight_leaf_paths(adjacency, {i: (counts[i], 1, 0, 0) for i in carbon}, weights)
    if (not best or best[0] != len(hydroxyls) or best[2] != sum(n > 1 for n in orders.values())):
        return ()  # Branches must be fully expressible as saturated alkyl groups.
    directions = []
    for path in paths:
        for seq in (path, path[::-1]):
            multiple = [(j, orders[min(a, b), max(a, b)]) for j, (a, b) in enumerate(zip(seq, seq[1:]), 1)]
            key = (tuple(sorted(seq.index(i) + 1 for i in hydroxyls)),
                   tuple(j for j, n in multiple if n > 1), tuple(j for j, n in multiple if n == 2))
            directions.append((key, seq))
    earliest = min(key for key, _ in directions)
    results = []
    for key, seq in directions:
        if key != earliest or root is not None and root not in seq:
            continue
        if any(i not in seq for i in carbon & atom_stereo.keys()):
            continue
        branches = []
        path_atoms = set(seq)
        for host in seq:
            for first in adjacency[host]:
                if first in path_atoms:
                    continue
                if len(adjacency[first]) == 1:
                    text = 'methyl'
                else:
                    scope = side(mol, first, host)
                    proof = mapped_saturated_carbon_prefix(mol, Chem, first, host, scope)
                    if not proof or not proof[0] or proof[1] != scope:
                        return ()
                    text = proof[0]
                branches.append((host, text))
        results.append((seq, branches))
    return results


def _chain_alcohol_unit(mol, atoms, root, atom_stereo, bond_stereo, *, allow_branches=False):
    """Number a complete alcohol unit on the original graph, including its configuration."""
    from .formatting import attach_suffix, multiplied_suffix, unsaturated_parent
    from .naming_methods import _linear_order

    carbon = {i for i in atoms if mol.GetAtomWithIdx(i).GetAtomicNum() == 6}
    if (root is not None and root not in carbon) or len(carbon) not in ROOTS or any(
            mol.GetAtomWithIdx(i).IsInRing() or mol.GetAtomWithIdx(i).GetTotalValence() != 4 for i in carbon):
        return None
    ends = [i for i in carbon if sum(n.GetIdx() in carbon for n in mol.GetAtomWithIdx(i).GetNeighbors()) == 1]
    path = _linear_order(mol, carbon, ends) if len(carbon) > 1 else tuple(carbon)
    hydroxyls = [next((n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in carbon), None)
                 for i in atoms - carbon]
    suffix = multiplied_suffix(len(hydroxyls), 'ol')
    if not suffix or not set(hydroxyls) <= carbon:
        return None
    if path:
        directions = ((path, ()), (path[::-1], ()))
    elif allow_branches and len(set(hydroxyls).intersection(ends)) <= 2:
        directions = _branched_alcohol_paths(mol, carbon, hydroxyls, root, atom_stereo)
    else:
        return None
    choices = []
    for seq, branches in directions:
        mapping = {i: j for j, i in enumerate(seq, 1)}
        locants = tuple(sorted(mapping[i] for i in hydroxyls))
        double, triple = [], []
        descriptors = [(mapping[i], atom_stereo[i]) for i in seq if i in atom_stereo]
        for j, (a, b) in enumerate(zip(seq, seq[1:]), 1):
            bond = mol.GetBondBetweenAtoms(a, b)
            degree = bond.GetBondTypeAsDouble()
            if degree == 2:
                double.append(j)
                if bond.GetIdx() in bond_stereo:
                    descriptors.append((j, bond_stereo[bond.GetIdx()]))
            elif degree == 3:
                triple.append(j)
            elif degree != 1:
                return None
        base = unsaturated_parent(ROOTS[len(seq)], double, triple)
        name = attach_suffix(base, suffix, locants if len(seq) > 1 else ())
        key = (locants, tuple(sorted(double + triple)), tuple(double), mapping.get(root, 0))
        if branches:
            from .formatting import prefix_alphabetization_key

            prefixes = [(str(mapping[i]), text) for i, text in branches]
            key += (-len(branches), tuple(sorted(mapping[i] for i, _ in branches)),
                    tuple(sorted((prefix_alphabetization_key(text), mapping[i]) for i, text in branches)))
            name = format_prefixes(prefixes, hierarchical_enclosures=True) + name
        row = _RingUnit('chain-alcohol', name, str(mapping.get(root, 0)), True, len(seq) == 1,
                        tuple(sorted(descriptors)))
        choices.append((key, row))
    if not choices:
        return None
    best = min(key for key, row in choices)
    winners = {row for key, row in choices if key == best}
    return next(iter(winners)) if len(winners) == 1 else None


def _alcohol_stereo(mol, Chem, tagged):
    atom_stereo, bond_stereo = {}, {}
    if tagged:
        from .stereochemistry import assigned_atom_stereochemistry

        atom_stereo = dict(assigned_atom_stereochemistry(mol, Chem))
        if atom_stereo.keys() != tagged or any(d not in ('R', 'S') for d in atom_stereo.values()):
            return None
    stereo_bonds = {b.GetIdx() for b in mol.GetBonds() if b.GetStereo()}
    if stereo_bonds:
        from .stereochemistry import assigned_bond_stereochemistry

        for a, b, descriptor in assigned_bond_stereochemistry(mol, Chem):
            if descriptor not in ('E', 'Z'):
                return None
            bond_stereo[mol.GetBondBetweenAtoms(a, b).GetIdx()] = descriptor
        if bond_stereo.keys() != stereo_bonds:
            return None
    return atom_stereo, bond_stereo


def straight_polyol(mol, Chem, *, allow_branches=False):
    """Select an acyclic parent containing all hydroxyls, then number unsaturation."""
    if mol.GetStereoGroups() or mol.GetRingInfo().NumRings() or len(Chem.GetMolFrags(mol)) != 1:
        return None
    centers, tagged = [], set()
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() or atom.GetIsotope() or atom.GetNumRadicalElectrons() or atom.HasQuery():
            return None
        if atom.GetAtomicNum() == 8:
            if atom.GetDegree() != 1 or atom.GetTotalNumHs() != 1 or atom.GetNeighbors()[0].GetAtomicNum() != 6:
                return None
            centers.append(atom.GetNeighbors()[0].GetIdx())
        elif atom.GetAtomicNum() != 6:
            return None
        if atom.GetChiralTag():
            tagged.add(atom.GetIdx())
    if len(centers) < 2 or any(b.HasQuery() for b in mol.GetBonds()):
        return None
    descriptors = _alcohol_stereo(mol, Chem, tagged)
    if descriptors is None:
        return None
    row = _chain_alcohol_unit(mol, frozenset(range(mol.GetNumAtoms())), None, *descriptors,
                              allow_branches=allow_branches)
    if row is None:
        return None
    stereo = '(' + ','.join(str(i) + d for i, d in row.stereo) + ')-' if row.stereo else ''
    return _out(mol, stereo + row.parent, centers, 'P-63.1.2/P-14.4')


@lru_cache(maxsize=2)
def _alcohol_linker_query(Chem):
    return Chem.MolFromSmarts('[$([CH2](-c)-c),$([OX2](-c)-c),'
                              '$([#14,#15,#16,#32,#33,#34,#50,#51,#52,#82,#83](-[#6])-[#6]),'
                              '$([S,Se,Te;D2;H0;+0](-[#6])-[S,Se,Te]);!R]')


def linked_alcohol_decision(mol, Chem):
    matches = mol.GetSubstructMatches(_alcohol_linker_query(Chem), maxMatches=3)
    if len(matches) == 1:
        return _linked_alcohol_units(mol, Chem, (matches[0][0],))
    if len(matches) == 2:
        # P-68.4.2. Only the two terminal atoms of a divalent homogeneous
        # chalcogen chain can attach to the repeated parents.
        first, last = (m[0] for m in matches)
        element = mol.GetAtomWithIdx(first).GetAtomicNum()
        stem = {16:'sulfanediyl', 34:'selanediyl', 52:'tellanediyl'}.get(element)
        if not stem or mol.GetAtomWithIdx(last).GetAtomicNum() != element:
            return None
        # Equivalent configured units on a homogeneous linker must also
        # make its ends equivalent. Stop before naming two unequal arms.
        ranks = Chem.CanonicalRankAtoms(mol, breakTies=False, includeChirality=True, includeAtomMaps=False)
        if ranks[first] != ranks[last]:
            return None
        chain = []
        previous = None
        current = first
        while True:
            atom = mol.GetAtomWithIdx(current)
            if (atom.GetDegree() != 2 or atom.GetTotalValence() != 2 or atom.IsInRing()
                    or atom.GetFormalCharge() or any(b.GetBondType() != Chem.BondType.SINGLE for b in atom.GetBonds())):
                return None
            chain.append(current)
            if current == last:
                break
            following = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() != previous and n.GetAtomicNum() == element]
            if len(following) != 1:
                return None
            previous, current = current, following[0]
        from .number_terms import basic_number_term

        multiplier = basic_number_term(len(chain))
        if multiplier:
            return _linked_alcohol_units(mol, Chem, tuple(chain), multiplier + stem)
    return None


def _linked_alcohol_units(mol, Chem, linker_atoms, bridge=''):
    """Number equivalent complete alcohol units on both sides of the proved linker."""
    from .naming_methods import _mononuclear_linker

    center = linker_atoms[0]
    linker_scope = frozenset(linker_atoms)
    element = mol.GetAtomWithIdx(center).GetAtomicNum()
    if (mol.GetStereoGroups() or mol.GetRingInfo().NumRings() > 2 or mol.GetAtomWithIdx(center).IsInRing()
            or len(Chem.GetMolFrags(mol)) != 1):
        return None
    centers = set()
    tagged = set()
    for atom in mol.GetAtoms():
        if (atom.GetIsotope() or atom.HasQuery() or atom.GetNumRadicalElectrons()
                or atom.GetFormalCharge() not in {-1, 0, 1, 2}):
            return None
        if atom.GetAtomicNum() in (9, 17, 35, 53):
            if (atom.GetFormalCharge() or atom.GetDegree() != 1 or atom.GetTotalValence() != 1
                    or not atom.GetNeighbors()[0].GetIsAromatic()):
                return None
            continue
        if atom.GetChiralTag():
            if atom.GetAtomicNum() != 6 or atom.GetChiralTag() not in (
                    Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW):
                return None
            tagged.add(atom.GetIdx())
        if atom.GetIdx() in linker_scope:
            continue
        if atom.GetAtomicNum() not in (6, 8):
            return None
        if atom.GetAtomicNum() == 8:
            if atom.GetDegree() != 1:
                return None
            host = atom.GetNeighbors()[0]
            if host.GetIdx() == center and element in {16, 34, 52}:
                continue  # The linker separately proves its oxo bonds and charges.
            if host.GetAtomicNum() != 6 or atom.GetTotalNumHs() != 1:
                return None
            centers.add(host.GetIdx())
        if atom.GetFormalCharge():
            return None
    if not centers or any(b.HasQuery() for b in mol.GetBonds()):
        return None
    descriptors = _alcohol_stereo(mol, Chem, tagged)
    if descriptors is None:
        return None
    atom_stereo, bond_stereo = descriptors
    stereo_bonds = bond_stereo.keys()
    arms = []
    for end in (center, linker_atoms[-1]) if bridge else (center,):
        for atom in mol.GetAtomWithIdx(end).GetNeighbors():
            if atom.GetAtomicNum() != 6:
                continue
            root = atom.GetIdx()
            atoms = side(mol, root, end)
            hydroxyls = centers.intersection(atoms)
            if hydroxyls:
                arms.append((atom, root, atoms, hydroxyls))
    if not 2 <= len(arms) <= 4 or len({(len(a), len(h)) for _, _, a, h in arms}) != 1:
        return None
    if atom_stereo or bond_stereo:
        # Different configurations cannot yield identical fully specified
        # repeated units. Compare the census before constructing their names.
        signatures = {(tuple(sorted(d for i, d in atom_stereo.items() if i in atoms)),
                       tuple(sorted(d for i, d in bond_stereo.items()
                                    if mol.GetBondWithIdx(i).GetBeginAtomIdx() in atoms)))
                      for _, _, atoms, _ in arms}
        if len(signatures) != 1:
            return None
    remaining = set(range(mol.GetNumAtoms()))
    units = []
    for atom, root, atoms, hydroxyls in arms:
        if atom.IsInRing() and not tagged and not stereo_bonds:
            row = _phenol_unit(mol, atoms, root, hydroxyls)
        else:
            row = _chain_alcohol_unit(mol, atoms, root, atom_stereo, bond_stereo, allow_branches=True)
        if row is None and not tagged and not stereo_bonds:
            row = unit(mol, Chem, atoms, root)
        if row is None or not row.has_principal_group or not atoms <= remaining or units and row != units[0]:
            return None
        remaining.difference_update(atoms)
        units.append(row)
    if centers.intersection(remaining):
        return None
    if tagged.intersection(remaining) or any(mol.GetBondWithIdx(i).GetBeginAtomIdx() in remaining or
            mol.GetBondWithIdx(i).GetEndAtomIdx() in remaining for i in stereo_bonds):
        return None
    if bridge:
        if remaining != linker_scope or len(units) != 2:
            return None
    elif element in (6, 8):
        linker = mol.GetAtomWithIdx(center)
        if (remaining != {center} or len(units) != 2 or linker.GetFormalCharge()
                or linker.GetTotalNumHs() != (2 if element == 6 else 0)
                or not all(a.GetIsAromatic() and a.GetAtomicNum() == 6 for a in linker.GetNeighbors())
                or any(b.GetBondTypeAsDouble() != 1 for b in linker.GetBonds())):
            return None
        bridge = 'methylene' if element == 6 else 'oxy'
    else:
        bridge = _mononuclear_linker(mol, remaining, center, len(units))
    if not bridge:
        return None
    parent = units[0]
    name = _assembly_name(parent.parent, parent.locant, bridge, len(units),
                          mononuclear=parent.mononuclear, stereo=parent.stereo, simple_bridge=len(linker_atoms) > 1)
    return _out(mol, name, centers, 'P-63.1.2/P-51.3.1/P-15.3.2.1', method='multiplicative')


def repeated_ring_parents(mol, Chem):
    if not _plain(mol, Chem, {6,7,8,9,14,15,16,17,32,33,34,35,50,51,52,53,82,83}) or mol.GetRingInfo().NumRings() < 2:
        return None
    from .mapped_ring_parents import ring_systems
    systems = ring_systems(mol)
    if len(systems) != 2:
        return None
    # Recover exactly one unit per terminal ring system. A branch crossing
    # back into the other ring is a connector, never a unit substituent.
    choices = []
    for core in systems:
        options = []
        for root in core:
            for a in mol.GetAtomWithIdx(root).GetNeighbors():
                if a.GetIdx() in core or order(mol,root,a.GetIdx()) != 1:
                    continue
                atoms = side(mol,root,a.GetIdx())
                if not core <= atoms or any(other <= atoms for other in systems if other != core):
                    continue
                row = unit(mol,Chem,atoms,root)
                if row:
                    options.append((atoms,root,a.GetIdx(),row))
        choices.append(options)
    names = set()
    for left,right in product(*choices):
        if left[0] & right[0] or left[3] != right[3]:
            continue
        connector = set(range(mol.GetNumAtoms())) - left[0] - right[0]
        if not connector or left[2] not in connector or right[2] not in connector:
            continue
        from .naming_methods import _connector_name, _linear_order
        units = [SimpleNamespace(root=r[1],connector=r[2]) for r in (left,right)]
        unit_record = left[3]
        bridge = _connector_name(mol, connector, units, functional_units=unit_record.has_principal_group,
                                 allow_urea=unit_record.parent.endswith('acid'))
        path = _linear_order(mol,connector,[left[2],right[2]])
        if path and all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in path):
            if len(path) not in ROOTS:
                continue
            doubles = [j for j,(a,b) in enumerate(zip(path,path[1:]),1) if order(mol,a,b) == 2]
            if doubles and all(order(mol,a,b) in {1,2} for a,b in zip(path,path[1:])):
                # A fully owned carbon path; no branches or lost H sites.
                from .formatting import unsaturated_parent, complete_parent_name
                base = complete_parent_name(unsaturated_parent(ROOTS[len(path)],doubles,[]))
                bridge = base+'-1,'+str(len(path))+'-diyl'
        if bridge:
            # O is named oxy in P-15.3.2.1 repeated-parent constructions.
            if len(connector) == 1 and mol.GetAtomWithIdx(next(iter(connector))).GetAtomicNum() == 8:
                bridge = 'oxy'
            names.add(_assembly_name(unit_record.parent, unit_record.locant, bridge))
    if len(names) != 1:
        return None
    return _out(mol,names.pop(),(),'P-15.3.2.1/P-15.3.2.3',method='multiplicative')


def imine_assembly(mol, Chem):
    if not _plain(mol,Chem,{6,7,8}) or sum(a.GetAtomicNum()==7 for a in mol.GetAtoms()) != 2:
        return None
    ns = [a for a in mol.GetAtoms() if a.GetAtomicNum()==7]
    ends = []
    for n in ns:
        imines = [a for a in n.GetNeighbors() if a.GetAtomicNum()==6 and order(mol,n.GetIdx(),a.GetIdx())==2]
        carbon = [a for a in n.GetNeighbors() if a.GetAtomicNum()==6 and order(mol,n.GetIdx(),a.GetIdx())==1]
        if len(imines)!=1 or len(carbon)!=1 or n.GetTotalNumHs():
            return None
        c=imines[0]; ring_roots=[a for a in c.GetNeighbors() if a.IsInRing() and order(mol,c.GetIdx(),a.GetIdx())==1]
        if len(ring_roots)==1 and c.GetTotalNumHs()==1:
            root=ring_roots[0].GetIdx(); blocked=c.GetIdx(); extra=True
        else:
            root=c.GetIdx(); blocked=n.GetIdx(); extra=False
        atoms=side(mol,root,blocked)
        row=unit(mol,Chem,atoms,root)
        if row is None or not row.parent.endswith(('acid','phenol','ol')):
            return None
        ends.append((atoms,root,blocked,row,carbon[0].GetIdx(),extra))
    left,right=ends
    if left[0]&right[0] or left[3]!=right[3] or left[5]!=right[5]:
        return None
    nids={n.GetIdx() for n in ns}
    outer={r[2] for r in ends} if left[5] else set()
    center=set(range(mol.GetNumAtoms()))-left[0]-right[0]-nids-outer
    from .naming_methods import _connector_name
    units=[SimpleNamespace(root=n.GetIdx(),connector=r[4]) for n,r in zip(ns,ends)]
    bridge=_connector_name(mol,center,units)
    if not bridge:
        return None
    group='azanylylidenemethanylylidene' if left[5] else 'azanylylidene'
    bridge=bridge+'bis('+group+')'
    return _out(mol,_assembly_name(left[3].parent,left[3].locant,bridge),(),
                'P-62.3.1.2/P-15.3',method='multiplicative')


def hydroxylamine_assembly(mol,Chem):
    if not _plain(mol,Chem,{6,7,8}):
        return None
    ns=[a for a in mol.GetAtoms() if a.GetAtomicNum()==7]
    if len(ns)!=2 or any(a.GetDegree()!=1 or a.GetTotalNumHs()!=2 for a in ns):
        return None  # N-organyl hydroxylamines use amine parents.
    os=[a.GetNeighbors()[0] for a in ns]
    if any(a.GetAtomicNum()!=8 or a.GetDegree()!=2 for a in os) or os[0].GetIdx()==os[1].GetIdx():
        return None
    core=set(range(mol.GetNumAtoms()))-{a.GetIdx() for a in ns+os}
    if not core or any(mol.GetAtomWithIdx(i).GetAtomicNum()!=6 for i in core):
        return None
    roots=[[n.GetIdx() for n in o.GetNeighbors() if n.GetAtomicNum()==6] for o in os]
    if any(len(r)!=1 for r in roots):
        return None
    from .naming_methods import _connector_name
    units=[SimpleNamespace(root=o.GetIdx(),connector=r[0]) for o,r in zip(os,roots)]
    bridge=_connector_name(mol,core,units)
    if not bridge:
        rings=mol.GetRingInfo().AtomRings()
        if len(rings)==1 and frozenset(rings[0])==core and len(core)==6 and all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in core):
            ring=rings[0]; distance=min((ring.index(roots[1][0])-ring.index(roots[0][0]))%6,
                                     (ring.index(roots[0][0])-ring.index(roots[1][0]))%6)
            bridge='benzene-1,'+str(distance+1)+'-diyl'
    if not bridge:
        return None
    return _out(mol,"O,O'-"+enclose_name_unit(bridge)+'bis(hydroxylamine)',
                (a.GetIdx() for a in ns),'P-68.3.1.1.1.2',method='multiplicative')


def replacement_spirobi(mol,Chem):
    """P-24.3.4 compares identical carbon skeletons before replacement."""
    if (not _plain(mol,Chem,{6,7,8,14,15,16,34,52}) or mol.GetRingInfo().NumRings()<4
            or any(b.GetBondTypeAsDouble()!=1 for b in mol.GetBonds())
            or all(a.GetAtomicNum()==6 for a in mol.GetAtoms())):
        return None
    atoms=set(range(mol.GetNumAtoms()))
    if set(i for r in mol.GetRingInfo().AtomRings() for i in r)!=atoms:
        return None
    from .ring_topology import ring_system_graph
    from .spiro_component_rules import _two_component_splits_by_articulation, _submol_for_atoms, _cycle_rank_for_atoms
    graph=ring_system_graph(mol,Chem,atoms)
    splits=_two_component_splits_by_articulation(graph,atoms)
    if len(splits)!=1:
        return None
    spiro,parts=next(iter(splits.items()))
    if len(parts)!=2 or mol.GetAtomWithIdx(spiro).GetAtomicNum()!=6:
        return None
    components=[]
    from .polycyclic_rules import bicyclic_parent_candidate_records
    for part in parts:
        subset=set(part)|{spiro}
        if _cycle_rank_for_atoms(graph,subset)!=2:
            return None
        row=_submol_for_atoms(mol,Chem,subset)
        if row is None:
            return None
        sub,back,forward=row
        abstract=Chem.RWMol(sub)
        for a in abstract.GetAtoms():
            a.SetAtomicNum(6);a.SetNumExplicitHs(0);a.SetNoImplicit(False)
        abstract=abstract.GetMol()
        try:Chem.SanitizeMol(abstract)
        except (ValueError,RuntimeError):return None
        mark=Chem.Mol(abstract);mark.GetAtomWithIdx(forward[spiro]).SetIsotope(13)
        identity=Chem.MolToSmiles(mark)
        candidates=[]
        for record in bicyclic_parent_candidate_records(abstract,Chem):
            name,seq=record[:2]
            if len(seq)!=len(subset) or not name.startswith('bicyclo'):
                continue
            mapping={back[i]:str(j) for j,i in enumerate(seq,1)}
            candidates.append((name,mapping))
        if not candidates:
            return None
        components.append((identity,candidates))
    if components[0][0]!=components[1][0]:
        return None
    replacement={8:'oxa',16:'thia',34:'selena',52:'tellura',7:'aza',15:'phospha',14:'sila'}
    hierarchy=(8,16,34,52,7,15,14)
    choices=[]
    for left,right in product(components[0][1],components[1][1]):
        if left[0]!=right[0] or left[1][spiro]!=right[1][spiro]:
            continue
        for a,b in ((left,right),(right,left)):
            mapping=dict(a[1]);mapping.update({i:loc+"'" for i,loc in b[1].items() if i!=spiro})
            prefixes=[(loc,replacement[mol.GetAtomWithIdx(i).GetAtomicNum()]) for i,loc in mapping.items()
                      if mol.GetAtomWithIdx(i).GetAtomicNum()!=6]
            loc_key=lambda loc:(int(loc.rstrip("'")),loc.endswith("'"))
            key=(int(a[1][spiro]),tuple(sorted(loc_key(loc) for loc,_text in prefixes)),
                 tuple(tuple(sorted(loc_key(loc) for i,loc in mapping.items()
                     if mol.GetAtomWithIdx(i).GetAtomicNum()==e)) for e in hierarchy))
            # Skeletal replacement prefixes follow element seniority, not
            # alphabetical order; one prefix owns both primed components.
            chunks=[]
            for e in hierarchy:
                locs=sorted((loc for i,loc in mapping.items() if mol.GetAtomWithIdx(i).GetAtomicNum()==e),key=loc_key)
                if locs:
                    from .constants import MULT
                    chunks.append(','.join(locs)+'-'+(MULT[len(locs)] if len(locs)>1 else '')+replacement[e])
            loc=a[1][spiro]
            text='-'.join(chunks)+'-'+loc+','+loc+"'-spirobi["+a[0]+']'
            choices.append((key,text))
    if not choices:
        return None
    best=min(c[0] for c in choices)
    names={text for key,text in choices if key==best}
    return _out(mol,names.pop(),(spiro,),'P-24.3.4') if len(names)==1 else None
