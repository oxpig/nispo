"""P-51.3.1 assemblies with separately numbered, identical functional parents."""
from dataclasses import dataclass, replace
from itertools import groupby

from .constants import ROOTS
from .formatting import COMPLEX_MULT, attach_suffix, enclose_name_unit, format_prefixes, join_prefix_parent
from .functional_parent_extensions import side
from .linked_imidic_parents import RingEnd
from .multiplicative_parent_extensions import _assembly_name, multiplicand_stereo
from .naming_methods import (MethodDecision, _block_name, _connector_name, _isolated_heterounits,
                            _mapped_connector_name, _mapped_ring_connector, _replacement_key, _replacement_text)
from .parent_selection import locant_key
from .rooted_carbon_parents import carbon_paths, hydride_numberings, skeleton_prefixes, substituent_numbering_key


@dataclass(frozen=True)
class FunctionalEnd(RingEnd):
    host: int  # Numbered parent atom bearing the characteristic group.


@dataclass(frozen=True)
class NumberedEnd:
    parent: str
    locant: str
    substituted: bool = False
    stereo: tuple = ()


def _site(mol, center):
    atom = mol.GetAtomWithIdx(center)
    if atom.GetAtomicNum() == 7:
        from .amine_components import ordinary_amine
        return ('amine', frozenset((center,)), center) if ordinary_amine(mol, center) else None
    singles, oxos, nitriles = [], [], []
    for b in atom.GetBonds():
        child = b.GetOtherAtom(atom)
        z, order = child.GetAtomicNum(), b.GetBondTypeAsDouble()
        if z in (7, 8) and order == 1:
            singles.append(child)
        elif z == 8 and order == 2 and child.GetDegree() == 1:
            oxos.append(child.GetIdx())
        elif z == 7 and order == 3 and child.GetDegree() == 1:
            nitriles.append(child.GetIdx())
    if nitriles:
        return 'nitrile', frozenset((center, *nitriles)), None
    if not singles and len(oxos) == 1 and atom.GetAtomicNum() == 6:
        carbons = sum(a.GetAtomicNum() == 6 for a in atom.GetNeighbors())
        if carbons == 2 or carbons == 1 and atom.GetTotalNumHs() == 1:
            kind = 'ketone' if carbons == 2 else 'aldehyde'
            return kind, frozenset((center, *oxos)), None
    if len(singles) != 1:
        return None
    ligand = singles[0]
    if ligand.GetAtomicNum() == 8 and ligand.GetDegree() == 1 and ligand.GetTotalNumHs() == 1:
        kind = 'acid' if oxos else 'alcohol'
    elif ligand.GetAtomicNum() == 7 and oxos and not ligand.IsInRing():
        kind = 'sulfonamide' if atom.GetAtomicNum() == 16 and len(oxos) == 2 else 'amide'
    else:
        return None
    return kind, frozenset((center, ligand.GetIdx(), *oxos)), ligand.GetIdx()


def _amine_host(mol, center):
    """Select a provably senior carbon parent of an N-substituted amine."""
    hosts = [a.GetIdx() for a in mol.GetAtomWithIdx(center).GetNeighbors() if a.GetAtomicNum() == 6]
    ring_hosts = [i for i in hosts if mol.GetAtomWithIdx(i).IsInRing()]
    if ring_hosts or len(hosts) == 1:
        candidates = ring_hosts or hosts
        return candidates[0] if len(candidates) == 1 else None
    # All competitors are carbon chains bearing the same characteristic N.
    # Leave equal-length competing ligands to the general parent comparison.
    sizes = [max((len(p) for p in carbon_paths(mol, {i}, range(mol.GetNumAtoms()))), default=0)
             for i in hosts]
    if all(sizes) and sizes.count(max(sizes)) == 1:
        return hosts[sizes.index(max(sizes))]
    return None


def _terminal_scope(mol, Chem, center, peer, group, systems):
    """Cut toward the other suffixes at the edge of this ring or carbon skeleton."""
    kind, core, _ = group
    path = Chem.GetShortestPath(mol, center, peer)
    host = center
    if kind == 'amine':
        host = _amine_host(mol, center)
        if host is None:
            return None
    elif kind not in ('alcohol', 'ketone'):
        # A carbaldehyde/carboxamide/etc. is external to its ring parent.
        ring_hosts = [a.GetIdx() for a in mol.GetAtomWithIdx(center).GetNeighbors()
                      if a.GetIdx() not in core and a.IsInRing()]
        if len(ring_hosts) > 1:
            return None
        if ring_hosts:
            host = ring_hosts[0]
    ring = next((r for r in systems if host in r), frozenset())
    if kind == 'amine':
        if host != path[1]:
            # The linker attaches to N; its preferred carbon parent lies on
            # another N ligand. The cut must retain that entire ligand.
            return FunctionalEnd(ring, side(mol, center, path[1]), center, path[1], host)
        path = path[1:]
    for i, j in zip(path, path[1:]):
        child = mol.GetAtomWithIdx(j)
        if ring:
            if i not in ring or j in ring:
                continue
        elif child.GetAtomicNum() == 6 and not child.IsInRing():
            continue
        bond = mol.GetBondBetweenAtoms(i, j)
        if j in core or bond.IsInRing() or bond.GetBondTypeAsDouble() != 1:
            return None
        atoms = side(mol, i, j)
        if core <= atoms:
            return FunctionalEnd(ring, atoms, i, j, host)
        break
    return None


def _same_rooted_units(mol, Chem, ends):
    """Keep one dummy for each cut so that the attachment and its stereo survive."""
    cuts = [mol.GetBondBetweenAtoms(e.root, e.connector).GetIdx() for e in ends]
    cut = Chem.FragmentOnBonds(mol, cuts, dummyLabels=[(0, 0)] * len(cuts))
    for atom in cut.GetAtoms():
        if atom.GetAtomMapNum():
            atom.SetAtomMapNum(0)
    roots = {e.root for e in ends}
    signatures = {Chem.MolFragmentToSmiles(cut, atomsToUse=indices)
                  for indices in Chem.GetMolFrags(cut) if roots.intersection(indices)}
    return len(signatures) == 1


def _ring_site_sets(mol, centers, kind, systems):
    """Group several characteristic functions that belong to the same ring parent."""
    if len(centers) < 4 or len(systems) < 2 or kind not in ('ketone', 'aldehyde', 'amine'):
        return ()
    by_atom = {i: ring for ring in systems for i in ring}
    members = {}
    for idx, center in enumerate(centers):
        if kind == 'ketone':
            host = center
        else:
            ring_carbons = [n.GetIdx() for n in mol.GetAtomWithIdx(center).GetNeighbors()
                            if n.GetAtomicNum() == 6 and n.IsInRing()]
            if len(ring_carbons) != 1:
                return ()
            host = ring_carbons[0]
        if host not in by_atom:
            return ()
        members.setdefault(by_atom[host], []).append(idx)
    counts = {len(indices) for indices in members.values()}
    if 2 <= len(members) <= 4 and len(counts) == 1 and min(counts) >= 2:
        return tuple(members.values())
    return ()


def _numbered_ring_set(mol, Chem, end, groups):
    """Number every suffix and the linker on the original ring parent."""
    from .mapped_ring_parents import records
    required = {g.center for g in groups}
    rows = records(mol, Chem, scope=end.atoms, core_filter=end.ring, principal=groups,
                   multiplicative_root=end.root, general_branches=True)
    if len(rows) != 1 or rows[0].principal_centers != required or not rows[0].annotation_coverage:
        return None
    row = rows[0]
    return NumberedEnd(row.name, dict(row.atom_locants)[end.root], bool(row.prefixes))


def _suffix_parent(hydride, kind, loc, cyclic, isotope):
    if hydride == 'benzene' and not isotope:
        return {'acid': 'benzoic acid', 'amide': 'benzamide', 'nitrile': 'benzonitrile',
                'alcohol': 'phenol', 'sulfonamide': 'benzenesulfonamide',
                'aldehyde': 'benzaldehyde', 'amine': 'aniline'}.get(kind, '')
    direct = {'alcohol': 'ol', 'amine': 'amine', 'ketone': 'one'}
    if kind in direct:
        suffix = direct[kind]
        name = attach_suffix(hydride, suffix, () if hydride == 'methane' else (loc,))
        return name[:-len(suffix)] + isotope + suffix if isotope else name
    if cyclic:
        suffix = {'acid': 'carboxylic acid', 'amide': 'carboxamide',
                  'nitrile': 'carbonitrile', 'sulfonamide': 'sulfonamide', 'aldehyde': 'carbaldehyde'}[kind]
        return attach_suffix(hydride, isotope + suffix, (loc,))
    if loc != '1' or hydride == 'methane':
        return ''
    if hydride == 'ethane' and not isotope:
        return {'acid': 'acetic acid', 'amide': 'acetamide', 'nitrile': 'acetonitrile',
                'aldehyde': 'acetaldehyde'}[kind]
    if kind == 'aldehyde':
        return attach_suffix(hydride, 'al')[:-2] + isotope + 'al'
    return attach_suffix(hydride, {'acid': 'oic acid', 'amide': 'amide', 'nitrile': 'nitrile'}[kind])


def _numbered_unit(mol, Chem, end, center, group, stereo, labels):
    from .halogen_oxoacids import carbon_branch

    kind, core, ligand = group
    host = end.host
    if not end.ring and mol.GetAtomWithIdx(host).GetAtomicNum() != 6:
        return None
    occupied = set(core)
    n_groups = []
    if kind in ('amide', 'sulfonamide', 'amine'):
        for atom in mol.GetAtomWithIdx(ligand).GetNeighbors():
            if atom.GetIdx() in (center, host) or atom.GetIdx() not in end.atoms:
                continue
            word, owned = carbon_branch(mol, Chem, atom.GetIdx(), ligand)
            if not word or not owned <= end.atoms or owned & occupied:
                return None
            n_groups.append(('N', word))
            occupied.update(owned)
    scope = end.atoms - occupied | {host}
    skeleton = end.ring or scope
    atom_stereo, bond_stereo = stereo
    # N ligands already carry their independently checked local descriptors.
    embedded = occupied - core
    local_atoms = {i: d for i, d in atom_stereo.items() if i in end.atoms and i not in embedded}
    local_labels = labels & (end.atoms - embedded)
    local_scope = end.atoms - embedded
    local_bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), d) for i, d in bond_stereo.items()
                   for b in (mol.GetBondWithIdx(i),) if b.GetBeginAtomIdx() in local_scope]
    if not local_atoms.keys() <= scope or any(a not in scope or b not in scope for a, b, _ in local_bonds):
        return None
    suffix_isotope = parent_isotope = ''
    suffix_atoms = core - {host}
    labelled_suffix = suffix_atoms & local_labels
    if labelled_suffix and (end.ring and host != center or kind in ('aldehyde', 'ketone', 'amine')):
        # Place labels on the external suffix atoms. A label on the numbered
        # carbon of a ketone or an acyclic aldehyde belongs to the parent.
        symbols = [mol.GetAtomWithIdx(i).GetSymbol() for i in suffix_atoms]
        atoms = sorted((mol.GetAtomWithIdx(i) for i in labelled_suffix), key=lambda a: a.GetAtomicNum())
        if any(symbols.count(a.GetSymbol()) != 1 for a in atoms):
            return None
        # One suffix descriptor can select an element repeated elsewhere in
        # the parent (e.g. the carbon of -13CHO). Place additional nuclides
        # on the whole parent only when their element has a unique owner.
        if len(atoms) > 1:
            elements = [mol.GetAtomWithIdx(i).GetAtomicNum() for i in end.atoms]
            atoms.sort(key=lambda a: elements.count(a.GetAtomicNum()) == 1)
            if any(elements.count(a.GetAtomicNum()) != 1 for a in atoms[1:]):
                return None
        suffix_isotope, *outer = [f'({a.GetIsotope()}{a.GetSymbol()})' for a in atoms]
        parent_isotope = ''.join(outer)
        local_labels -= labelled_suffix
    rows = []
    numberings = tuple(hydride_numberings(mol, Chem, host, skeleton, extended=True))
    senior_frame = min((row[2] for row in numberings), default=None)
    for hydride, mapping, frame, pi in numberings:
        if frame != senior_frame:
            continue
        loc = mapping[host]
        parent = _suffix_parent(hydride, kind, loc, bool(end.ring), suffix_isotope)
        if not parent:
            continue
        pendants = skeleton_prefixes(mol, Chem, mapping, scope,
                                    check_branch_stereo=bool(local_atoms or local_bonds))
        if pendants is None:
            continue
        subkey = substituent_numbering_key(mol, mapping, pendants)
        if subkey is None:
            continue
        prefix = format_prefixes(n_groups + pendants, hierarchical_enclosures=True)
        if local_labels:
            from .isotope_mapping import locanted_isotope_prefix
            if not all(i in mapping and mapping[i].isdigit() for i in local_labels):
                continue
            parent = locanted_isotope_prefix(mol, mapping.items()) + parent
        name = parent_isotope + join_prefix_parent(prefix, parent)
        descriptors = [(mapping[i], d) for i, d in local_atoms.items() if i in mapping]
        descriptors.extend((min((mapping[a], mapping[b]), key=locant_key), d)
                           for a, b, d in local_bonds if a in mapping and b in mapping)
        port = 'N' if kind == 'amine' and end.root == center else mapping.get(end.root, '')
        usable = bool(port) and all((a in mapping) == (b in mapping) for a, b, _ in local_bonds)
        data = (name, port, bool(n_groups or pendants),
                tuple(sorted(descriptors, key=lambda row: locant_key(row[0])))) if usable else None
        rows.append(((frame, locant_key(loc), pi, subkey), data))
    if not rows:
        return None
    best = min(key for key, *_ in rows)
    winners = {data for key, data in rows if key == best and data is not None}
    if winners:
        lowest_port = min(locant_key(row[1]) for row in winners)
        winners = {row for row in winners if locant_key(row[1]) == lowest_port}
    if len(winners) != 1:
        return None
    return NumberedEnd(*winners.pop())


def _linker_block(mol, Chem, block, spine, scope, owned, *, outward):
    """Number a substituted block from the end nearest the repeated parent."""
    from .halogen_oxoacids import carbon_branch
    from .constants import HALO

    element = mol.GetAtomWithIdx(block[0]).GetAtomicNum()
    term = _block_name(element, len(block), outward=outward)
    if not term:
        return ''
    prefixes = []
    for loc, idx in enumerate(reversed(block), 1):
        for bond in mol.GetAtomWithIdx(idx).GetBonds():
            atom = bond.GetOtherAtomIdx(idx)
            if atom in spine or atom not in scope:
                continue
            if not outward or element not in (6, 7) or bond.GetBondTypeAsDouble() != 1:
                return ''
            neighbor = mol.GetAtomWithIdx(atom)
            if neighbor.GetSymbol() in HALO and neighbor.GetDegree() == 1:
                name, atoms = HALO[neighbor.GetSymbol()], {atom}
            else:
                name, atoms = carbon_branch(mol, Chem, atom, idx)
            if not name or not atoms <= scope or atoms & owned:
                return ''
            owned.update(atoms)
            prefixes.append((str(loc), name))
    if not prefixes:
        return term
    prefix = format_prefixes(prefixes, omit_locants=len(block) == 1, hierarchical_enclosures=True)
    return enclose_name_unit(prefix + term)


def _blocks(mol, path):
    return tuple(tuple(atoms) for _, atoms in groupby(path, key=lambda i: mol.GetAtomWithIdx(i).GetAtomicNum()))


def _decorated_linker(mol, Chem, scope, ends):
    """Number corresponding outward carbon blocks from the multiplicand end."""
    if len(ends) != 2 or ends[0].connector == ends[1].connector:
        return ''
    path = Chem.GetShortestPath(mol, ends[0].connector, ends[1].connector)
    if not set(path) <= scope or any(mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble() != 1
                                     for a, b in zip(path, path[1:])):
        return ''
    blocks = _blocks(mol, path)
    mid = len(blocks) // 2
    if len(blocks) < 3 or len(blocks) % 2 != 1:
        return ''
    used = set(path)
    central = _linker_block(mol, Chem, blocks[mid], path, scope, used, outward=False)
    if not central:
        return ''
    arms = []
    for blocks_out in (tuple(b[::-1] for b in reversed(blocks[:mid])), blocks[mid + 1:]):
        terms = [_linker_block(mol, Chem, b, path, scope, used, outward=True) for b in blocks_out]
        if not all(terms):
            return ''
        arms.append(''.join(terms))
    if used != scope or arms[0] != arms[1]:
        return ''
    return central + 'bis' + enclose_name_unit(arms[0])


def _central_ring_linker(mol, Chem, scope, ends, ring):
    """Follow each equally substituted outward arm to one numbered central ring."""
    ring_ports = []
    arms = []
    spine = set(ring)
    target = next(iter(ring))
    for end in ends:
        path = ((end.connector,) if end.connector in ring else
                Chem.GetShortestPath(mol, end.connector, target))
        boundary = next(j for j, i in enumerate(path) if i in ring)
        arm, port = path[:boundary], path[boundary]
        if (spine.intersection(arm) or not scope.issuperset(arm)
                or any(mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble() != 1
                       for a, b in zip(arm, path[1:]))):
            return ''
        ring_ports.append(replace(end, root=arm[-1] if arm else end.root, connector=port))
        arms.append(arm[::-1])
        spine.update(arm)
    owned = set(spine)
    words = set()
    for arm in arms:
        terms = [_linker_block(mol, Chem, b, spine, scope, owned, outward=True) for b in _blocks(mol, arm)]
        if not all(terms):
            return ''
        words.add(''.join(terms))
    if len(words) != 1:
        return ''
    term = words.pop()
    central = _mapped_ring_connector(mol, Chem, scope - owned | ring, ring_ports)
    if not central:
        return ''
    return central + (COMPLEX_MULT[len(ends)] + enclose_name_unit(term) if term else '')


def repeated_functional_decision(mol, Chem, centers):
    from .extended_inventory import selection_inventory
    from .mapped_ring_parents import ring_systems

    if len(centers) < 2 or mol.GetStereoGroups():
        return None
    groups = [_site(mol, c) for c in centers]
    if not all(groups):
        return None
    inventory = None
    kinds = [g[0] for g in groups]
    if len(set(kinds)) != 1:
        if len(kinds) == len(set(kinds)):
            return None
        inventory = selection_inventory(mol)
        senior = {g.center for g in inventory.principal}
        kept = [(c, g) for c, g in zip(centers, groups) if c in senior]
        if (not inventory.supported or len(kept) < 2 or len(kept) != len(senior)
                or len({g[0] for _, g in kept}) != 1):
            return None
        centers, groups = zip(*kept)
    kind = groups[0][0]
    ring_count = mol.GetRingInfo().NumRings()
    systems = ring_systems(mol) if ring_count >= 2 and len(centers) >= 4 and kind in (
        'ketone', 'aldehyde', 'amine') else ()
    ring_sets = _ring_site_sets(mol, centers, kind, systems)
    if ring_sets:
        parts = ring_sets
    else:
        if len(centers) > 4:
            return None
        # These properties survive a single cut, even when the atoms on
        # the central side of the equivalent linking bonds are different.
        site_shapes = {(a.GetDegree(), a.GetTotalNumHs(), a.IsInRing(), a.GetIsAromatic(),
                        mol.GetAtomWithIdx(g[2]).GetDegree() if g[2] is not None else None)
                       for c, g in zip(centers, groups) for a in (mol.GetAtomWithIdx(c),)}
        if len(site_shapes) != 1:
            return None
        parts = tuple((i,) for i in range(len(centers)))
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    if ring_count and not systems:
        systems = ring_systems(mol)
    ends = []
    for n, indices in enumerate(parts):
        i = indices[0]
        peer = centers[parts[(n + 1) % len(parts)][0]]
        end = _terminal_scope(mol, Chem, centers[i], peer, groups[i], systems)
        if (end is None or len(end.atoms) * len(parts) >= mol.GetNumAtoms()
                or ends and len(end.atoms) != len(ends[0].atoms)
                or ring_sets and (end.root not in end.ring or any(centers[j] not in end.atoms for j in indices))):
            return None
        ends.append(end)
    scope = frozenset(range(mol.GetNumAtoms()))
    residues = frozenset.union(*(e.atoms for e in ends))
    connector = scope - residues
    if sum(len(e.atoms) for e in ends) != len(residues) or any(e.connector not in connector for e in ends):
        return None
    if (any(a.GetFormalCharge() or a.GetNumRadicalElectrons() or a.GetAtomicNum() == 1 or a.HasQuery()
            for a in mol.GetAtoms()) or any(b.HasQuery() for b in mol.GetBonds())
            or not _same_rooted_units(mol, Chem, ends)):
        return None
    if inventory is None:
        inventory = selection_inventory(mol)
    if inventory.supported and {g.center for g in inventory.principal} != set(centers):
        return None
    if kind in ('ketone', 'aldehyde', 'amine'):
        if not inventory.supported or any(g.kind != kind for g in inventory.principal):
            return None
        if kind == 'amine' and not ring_sets and any(e.root == c for e, c in zip(ends, centers)):
            from .amine_components import common_framework
            if common_framework(mol, inventory.principal):
                return None
    if not inventory.supported:
        from .functional_class_queries import only_lower_substituents
        # A heavier connector can prevent the general inventory from classifying
        # its organic ends. Prove every atom outside the mapped suffixes and
        # ordinary central hydride before admitting that independent construction.
        core = frozenset.union(*(g[1] for g in groups))
        hydrides = {i for i in connector if mol.GetAtomWithIdx(i).GetAtomicNum() in (14, 15, 32, 33, 50, 51, 82, 83)
                    and all(b.GetBondTypeAsDouble() == 1 and b.GetOtherAtomIdx(i) in residues
                            for b in mol.GetAtomWithIdx(i).GetBonds())}
        if not only_lower_substituents(mol, core | hydrides):
            return None
        if any(
                b.GetBondTypeAsDouble() > 1 and b.GetBeginAtomIdx() in connector and b.GetEndAtomIdx() in connector
                and any(a.GetAtomicNum() != 6 for a in (b.GetBeginAtom(), b.GetEndAtom()))
                and not any(a.GetAtomicNum() in (16, 34, 52) for a in (b.GetBeginAtom(), b.GetEndAtom()))
                for b in mol.GetBonds()):
            return None
    if len(systems) >= 4:
        from .linear_phane_methods import linear_phane_frontier
        if linear_phane_frontier(mol, Chem).mandatory_eligible:
            return None
    tagged = {a.GetIdx() for a in mol.GetAtoms() if a.GetChiralTag()}
    labels = {a.GetIdx() for a in mol.GetAtoms() if a.GetIsotope()}
    central_rings = [r for r in systems if r <= connector]
    ring_stereo = central_rings[0] if len(central_rings) == 1 else frozenset()
    stereo = multiplicand_stereo(mol, Chem, tagged, linker_atoms=ring_stereo)
    if (stereo is None or not (tagged & connector) <= ring_stereo or labels & connector or any(
            mol.GetBondWithIdx(i).GetBeginAtomIdx() in connector or
            mol.GetBondWithIdx(i).GetEndAtomIdx() in connector for i in stereo[1])):
        return None
    if not systems and len(ends) == 2 and ends[0].connector != ends[1].connector:
        path = Chem.GetShortestPath(mol, centers[0], centers[1])
        if kind == 'amine':
            path = path[1:-1]  # Characteristic amine nitrogens are not replacement heterounits.
        units = _isolated_heterounits(mol, path)
        if units is None or len(units) >= 4:
            if (units is None or groups[0][0] != 'acid' or len(path) not in ROOTS
                    or any(stereo) or labels or set(path) | groups[0][1] | groups[1][1] != scope
                    or any(mol.GetAtomWithIdx(i).GetAtomicNum() not in (6, 7, 8, 16) for i in path)
                    or any(mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble() != 1 for a, b in zip(path, path[1:]))):
                return None
            path = min((path, path[::-1]), key=lambda p: _replacement_key(mol, p))
            name = _replacement_text(mol, path) + f'-1,{len(path)}-dioic acid'
            return MethodDecision(name, 'skeletal-replacement', frozenset(centers), scope, 'P-51.3.1/P-51.4.1')
    if central_rings:
        if len(central_rings) != 1:
            return None
        ring = central_rings[0]
        if all(e.ring and e.connector in ring for e in ends):
            # A directly bonded assembly of like rings can cite the suffixes
            # on a single parent. Leave it to the P-28/P-44 selector.
            cores = {Chem.MolFragmentToSmiles(mol, atomsToUse=tuple(r)) for r in (ring, ends[0].ring)}
            if len(cores) == 1:
                return None
        bridge = _central_ring_linker(mol, Chem, connector, ends, ring)
    else:
        bridge = _connector_name(mol, connector, ends, functional_units=True)
        if not bridge:
            bridge = _decorated_linker(mol, Chem, connector, ends) or _mapped_connector_name(mol, Chem, connector, ends)
    if not bridge:
        return None
    # One rooted graph suffices for ordinary ends. Configured ends also have
    # to agree in the CIP assignments of the intact molecule.
    if ring_sets:
        # Stereo on a ring with several suffixes needs a descriptor for each
        # primed copy. The single-site writer handles that mapping separately.
        if any(stereo) or labels:
            return None
        local = tuple(g for g in inventory.principal if g.center in ends[0].atoms)
        parents = [_numbered_ring_set(mol, Chem, ends[0], local)]
    else:
        indexed = zip(ends, centers, groups)
        if not any(stereo):
            indexed = (next(indexed),)
        parents = [_numbered_unit(mol, Chem, e, c, g, stereo, labels) for e, c, g in indexed]
    if not all(parents) or len({(e.parent, e.locant, e.stereo) for e in parents}) != 1:
        return None
    end = parents[0]
    name = _assembly_name(end.parent, end.locant, bridge, len(ends), substituted=end.substituted, stereo=end.stereo)
    roles = tuple(('multiplicand', e.atoms, ((e.root, end.locant),)) for e in ends) + (('linker', connector, ()),)
    return MethodDecision(name, 'multiplicative', frozenset(centers), scope, 'P-51.3.1/P-15.3',
                          component_roles=roles)
