"""P-51.3.1 assemblies with separately numbered, identical functional parents."""
from dataclasses import dataclass
from itertools import groupby

from .constants import ROOTS
from .formatting import attach_suffix, enclose_name_unit, format_prefixes, join_prefix_parent
from .functional_parent_extensions import capped
from .linked_imidic_parents import _ring_end
from .multiplicative_parent_extensions import _assembly_name
from .naming_methods import (MethodDecision, _block_name, _connector_name, _isolated_heterounits,
                            _mapped_connector_name, _replacement_key, _replacement_text)
from .parent_selection import locant_key
from .rooted_carbon_parents import hydride_numberings, skeleton_prefixes, substituent_numbering_key


@dataclass(frozen=True)
class End:
    root: int
    connector: int
    atoms: frozenset
    parent: str
    locant: str
    substituted: bool = False


def _site(mol, center):
    atom = mol.GetAtomWithIdx(center)
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


def _ring_unit(mol, Chem, end, center, group):
    from .halogen_oxoacids import carbon_branch

    kind, core, ligand = group
    hosts = [a.GetIdx() for a in mol.GetAtomWithIdx(center).GetNeighbors() if a.GetIdx() in end.ring]
    if kind == 'alcohol':
        host = center
        if host not in end.ring:
            return None
    elif len(hosts) == 1:
        host = hosts[0]
    else:
        return None
    occupied = set(core)
    n_groups = []
    if kind in ('amide', 'sulfonamide'):
        for atom in mol.GetAtomWithIdx(ligand).GetNeighbors():
            if atom.GetIdx() == center:
                continue
            word, owned = carbon_branch(mol, Chem, atom.GetIdx(), ligand)
            if not word or not owned <= end.atoms or owned & occupied:
                return None
            n_groups.append(('N', word))
            occupied.update(owned)
    scope = end.atoms - occupied | end.ring
    rows = []
    for hydride, mapping, frame, pi in hydride_numberings(mol, Chem, end.root, end.ring):
        if kind == 'alcohol' and hydride != 'benzene':
            continue  # Other ring alcohols use the existing P-63 parent selector.
        loc = mapping[host]
        retained = {'acid': 'benzoic acid', 'amide': 'benzamide', 'nitrile': 'benzonitrile',
                    'alcohol': 'phenol', 'sulfonamide': 'benzenesulfonamide'}
        suffix = {'acid': 'carboxylic acid', 'amide': 'carboxamide',
                  'nitrile': 'carbonitrile', 'sulfonamide': 'sulfonamide'}
        parent = retained[kind] if hydride == 'benzene' else hydride + '-' + loc + '-' + suffix[kind]
        pendants = skeleton_prefixes(mol, Chem, mapping, scope)
        if pendants is None:
            continue
        subkey = substituent_numbering_key(mol, mapping, pendants)
        if subkey is None:
            continue
        prefix = format_prefixes(n_groups + pendants, hierarchical_enclosures=True)
        name = join_prefix_parent(prefix, parent)
        port = mapping[end.root]
        rows.append(((frame, locant_key(loc), pi, subkey, locant_key(port)),
                     name, port, bool(n_groups or pendants)))
    if not rows:
        return None
    best = min(key for key, *_ in rows)
    winners = {(name, port, sub) for key, name, port, sub in rows if key == best}
    if len(winners) != 1:
        return None
    return End(end.root, end.connector, end.atoms, *winners.pop())


def _chain_unit(mol, center, group):
    kind, core, ligand = group
    if kind == 'sulfonamide' or (kind == 'amide' and mol.GetAtomWithIdx(ligand).GetDegree() != 1):
        return None
    chain = [center]
    previous = set(core)
    while True:
        atom = mol.GetAtomWithIdx(chain[-1])
        next_bonds = [b for b in atom.GetBonds() if b.GetOtherAtomIdx(atom.GetIdx()) not in previous]
        if len(next_bonds) != 1 or next_bonds[0].GetBondTypeAsDouble() != 1:
            return None
        child = next_bonds[0].GetOtherAtom(atom)
        if child.GetAtomicNum() != 6:
            break
        previous.add(child.GetIdx())
        chain.append(child.GetIdx())
    n = len(chain)
    if n not in ROOTS or n < 2 and kind != 'alcohol':
        return None
    parent = ROOTS[n] + 'ane'
    if kind == 'alcohol':
        parent = attach_suffix(parent, 'ol', [] if n == 1 else [1])
    elif kind == 'nitrile':
        parent = 'acetonitrile' if n == 2 else parent + 'nitrile'
    elif kind == 'acid':
        parent = 'acetic acid' if n == 2 else parent[:-1] + 'oic acid'
    else:
        parent = 'acetamide' if n == 2 else parent[:-1] + 'amide'
    return End(chain[-1], child.GetIdx(), frozenset(chain) | core, parent, str(n))


def _decorated_linker(mol, Chem, scope, ends):
    """Number corresponding outward carbon blocks from the multiplicand end."""
    from .halogen_oxoacids import carbon_branch
    from .constants import HALO
    if len(ends) != 2 or ends[0].connector == ends[1].connector:
        return ''
    path = Chem.GetShortestPath(mol, ends[0].connector, ends[1].connector)
    if not set(path) <= scope or any(mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble() != 1
                                     for a, b in zip(path, path[1:])):
        return ''
    blocks = [tuple(values) for _, values in groupby(path, key=lambda i: mol.GetAtomWithIdx(i).GetAtomicNum())]
    mid = len(blocks) // 2
    if len(blocks) < 3 or len(blocks) % 2 != 1:
        return ''
    used = set(path)
    def block_text(block, outward):
        element = mol.GetAtomWithIdx(block[0]).GetAtomicNum()
        term = _block_name(element, len(block), outward=outward)
        if not term:
            return ''
        prefixes = []
        for loc, idx in enumerate(reversed(block), 1):
            for atom in mol.GetAtomWithIdx(idx).GetNeighbors():
                root = atom.GetIdx()
                if root in path or root not in scope:
                    continue
                if mol.GetBondBetweenAtoms(idx, root).GetBondTypeAsDouble() != 1 or element != 6:
                    return ''
                if atom.GetSymbol() in HALO and atom.GetDegree() == 1:
                    name, owned = HALO[atom.GetSymbol()], {root}
                else:
                    name, owned = carbon_branch(mol, Chem, root, idx)
                if not name or used & owned or not owned <= scope:
                    return ''
                used.update(owned)
                prefixes.append((str(loc), name))
        if prefixes:
            if not outward:
                return ''  # Central substitution is handled by the central-group writer.
            return enclose_name_unit(format_prefixes(prefixes, hierarchical_enclosures=True) + term)
        return term
    central = block_text(blocks[mid], False)
    if not central:
        return ''
    arms = []
    for blocks_out in (tuple(b[::-1] for b in reversed(blocks[:mid])), blocks[mid + 1:]):
        terms = [block_text(b, True) for b in blocks_out]
        if not all(terms):
            return ''
        arms.append(''.join(terms))
    if used != scope or arms[0] != arms[1]:
        return ''
    return central + 'bis' + enclose_name_unit(arms[0])


def repeated_functional_decision(mol, Chem, centers):
    rings = mol.GetRingInfo().AtomRings()
    if (not 2 <= len(centers) <= 4 or 0 < len(rings) < len(centers)
            or mol.GetStereoGroups() or len(Chem.GetMolFrags(mol)) != 1):
        return None
    groups = [_site(mol, c) for c in centers]
    if not all(groups) or len({g[0] for g in groups}) != 1:
        return None
    if rings and any(not mol.GetAtomWithIdx(c).IsInRing() and not any(
            a.IsInRing() for a in mol.GetAtomWithIdx(c).GetNeighbors()) for c in centers):
        return None
    if any(a.GetIsotope() or a.GetChiralTag() or a.GetFormalCharge() or a.GetNumRadicalElectrons()
           or a.GetAtomicNum() == 1 or a.HasQuery() for a in mol.GetAtoms()) or any(
           b.HasQuery() or b.GetStereo() for b in mol.GetBonds()):
        return None
    scope = frozenset(range(mol.GetNumAtoms()))
    if rings:
        from .mapped_ring_parents import ring_systems
        from .extended_inventory import selection_inventory
        systems = ring_systems(mol)
        if len(systems) != len(groups):
            return None
        inventory = selection_inventory(mol)
        if inventory.supported and {g.center for g in inventory.principal} != set(centers):
            return None
        all_rings = frozenset.union(*systems)
        ends = []
        for ring in systems:
            end = _ring_end(mol, ring, all_rings)
            candidates = [(c, g) for c, g in zip(centers, groups) if end and g[1] <= end.atoms]
            if len(candidates) != 1:
                return None
            ends.append(_ring_unit(mol, Chem, end, *candidates[0]))
    else:
        ends = [_chain_unit(mol, c, g) for c, g in zip(centers, groups)]
    if not all(ends) or len({(e.parent, e.locant) for e in ends}) != 1:
        return None
    residues = frozenset.union(*(e.atoms for e in ends))
    connector = scope - residues
    if sum(len(e.atoms) for e in ends) != len(residues) or any(e.connector not in connector for e in ends):
        return None
    if not rings:
        from .extended_inventory import selection_inventory
        inventory = selection_inventory(mol)
        if inventory.supported and {g.center for g in inventory.principal} != set(centers):
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
    if rings:
        # The central linker can be asymmetric. Require equality only of the
        # rooted multiplicands, including their complete pendant substitution.
        identities = set()
        for end in ends:
            part, ids = capped(mol, Chem, end.atoms)
            if part is None:
                return None
            part.GetAtomWithIdx(ids.index(end.root)).SetAtomMapNum(1)
            identities.add(Chem.MolToSmiles(part))
        if len(identities) != 1:
            return None
    if not rings and len(ends) == 2 and ends[0].connector != ends[1].connector:
        path = Chem.GetShortestPath(mol, centers[0], centers[1])
        units = _isolated_heterounits(mol, path)
        if units is None or len(units) >= 4:
            if (units is None or groups[0][0] != 'acid' or len(path) not in ROOTS
                    or set(path) | groups[0][1] | groups[1][1] != scope
                    or any(mol.GetAtomWithIdx(i).GetAtomicNum() not in (6, 7, 8, 16) for i in path)
                    or any(mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble() != 1 for a, b in zip(path, path[1:]))):
                return None
            path = min((path, path[::-1]), key=lambda p: _replacement_key(mol, p))
            name = _replacement_text(mol, path) + f'-1,{len(path)}-dioic acid'
            return MethodDecision(name, 'skeletal-replacement', frozenset(centers), scope, 'P-51.3.1/P-51.4.1')
    bridge = _connector_name(mol, connector, ends, functional_units=True)
    if not bridge:
        bridge = _decorated_linker(mol, Chem, connector, ends) or _mapped_connector_name(mol, Chem, connector, ends)
    if not bridge:
        return None
    end = ends[0]
    name = _assembly_name(end.parent, end.locant, bridge, len(ends), substituted=end.substituted)
    roles = tuple(('multiplicand', e.atoms, ()) for e in ends) + (('linker', connector, ()),)
    return MethodDecision(name, 'multiplicative', frozenset(centers), scope, 'P-51.3.1/P-15.3',
                          component_roles=roles)
