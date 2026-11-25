"""Source-graph suffixes for amidines and the two drawn forms of amidrazones."""

from dataclasses import dataclass
from itertools import product

from .component_mapping import MappedComponent, decorate_component, hydrogen_bindings, mapped_general_branch, nitrogen_locant
from .formatting import attach_suffix, multiplied_suffix
from .functional_class_queries import amidine_query, only_lower_substituents
from .naming_methods import MethodDecision
from .parent_selection import locant_key, locant_set
from .rooted_carbon_parents import grouped_site_orders, hydride_numberings, skeleton_prefixes, substituent_numbering_key


@dataclass(frozen=True)
class ImidicSite:
    center: int
    owner: int | None
    element: int
    oxos: tuple
    replacement: int
    kind: str
    # Primes identify original nitrogen atoms. The bridging =N– of a
    # hydrazonamide has no substituent locant in the P-66.4.2.1 drawing.
    nitrogens: tuple

    @property
    def atoms(self):
        return {self.center, *self.oxos, *(i for i, _ in self.nitrogens)}

    def suffix(self, external):
        if self.element == 6:
            if not external:
                return self.kind
            return {'imidamide': 'carboximidamide', 'hydrazonamide': 'carbohydrazonamide',
                    'imidohydrazide': 'carboximidohydrazide'}[self.kind]
        stem = {16: 'sulf', 34: 'selen', 52: 'tellur'}[self.element]
        if self.kind == 'diimidamide':
            return stem + 'onodiimidamide'
        if self.replacement in (16, 34, 52):
            return stem + 'onimido' + {16: 'thio', 34: 'seleno', 52: 'telluro'}[self.replacement] + 'amide'
        oxidation = 'on' if self.oxos else 'in'
        return stem + oxidation + ('o' if self.kind == 'hydrazonamide' else '') + self.kind


def _site(mol, center, imines, amino):
    atom = mol.GetAtomWithIdx(center)
    element = atom.GetAtomicNum()
    owners, oxos = [], []
    for bond in atom.GetBonds():
        other = bond.GetOtherAtom(atom)
        idx = other.GetIdx()
        if idx in imines or idx == amino or other.GetAtomicNum() == 1:
            continue
        if (other.GetAtomicNum() == 6 or element == 6 and other.GetAtomicNum() in {14, 32, 50, 82}) and bond.GetBondTypeAsDouble() == 1:
            owners.append(idx)
        elif (element != 6 and other.GetAtomicNum() in {8, 16, 34, 52} and other.GetDegree() == 1
              and bond.GetBondTypeAsDouble() == 2):
            oxos.append(idx)
        else:
            return None  # E.g. guanidines, carbamimidates and shared N units.
    if (len(owners) > 1 or len(oxos) > 1 or atom.GetChiralTag()
            or (element != 6 and (len(owners) != 1 or atom.GetTotalNumHs()))):
        return None
    replacement = mol.GetAtomWithIdx(oxos[0]).GetAtomicNum() if oxos else 0
    extensions = []
    for idx in (*imines, amino):
        n = mol.GetAtomWithIdx(idx)
        if n.GetTotalValence() != 3 or n.GetChiralTag():
            return None
        extensions.extend((idx, other.GetIdx()) for other in n.GetNeighbors()
                          if other.GetAtomicNum() == 7)
    imine = imines[0]
    if len(imines) == 2 and element != 6 and not oxos and not extensions:
        kind, ns = 'diimidamide', ((amino, 0), (imine, 1), (imines[1], 2))
    elif len(imines) != 1:
        return None
    elif not extensions:
        kind, ns = 'imidamide', ((amino, 0), (imine, 1))
    elif len(extensions) == 1:
        adjacent, terminal = extensions[0]
        n = mol.GetAtomWithIdx(terminal)
        if (n.IsInRing() or n.GetTotalValence() != 3 or n.GetChiralTag()
                or mol.GetBondBetweenAtoms(adjacent, terminal).GetBondTypeAsDouble() != 1):
            return None
        if adjacent == imine:
            if mol.GetAtomWithIdx(imine).GetIsotope():
                return None
            kind, ns = 'hydrazonamide', ((amino, 0), (terminal, 1), (imine, None))
        else:
            kind, ns = 'imidohydrazide', ((amino, 0), (terminal, 1), (imine, 2))
    else:
        return None
    if replacement not in (0, 8) and kind != 'imidamide':
        return None
    return ImidicSite(center, next(iter(owners), None), element, tuple(oxos), replacement, kind, ns)


def _nitrogen_branches(mol, Chem, sites, core):
    """One cut per N substituent. Retain its complete source atom set."""
    substituents, occupied = [], set()
    for site in sites:
        for nitrogen, _ in site.nitrogens:
            for bond in mol.GetAtomWithIdx(nitrogen).GetBonds():
                root = bond.GetOtherAtomIdx(nitrogen)
                atom = mol.GetAtomWithIdx(root)
                if root in site.atoms or atom.GetAtomicNum() == 1:
                    continue
                if (root in core or atom.GetAtomicNum() not in {6, 9, 17, 35, 53}
                        or bond.GetBondType() != Chem.BondType.SINGLE):
                    return None
                word, atoms = mapped_general_branch(mol, Chem, root, nitrogen)
                if not word or atoms & (core | occupied):
                    return None
                substituents.append((nitrogen, word, atoms))
                occupied.update(atoms)
    return substituents, occupied


def _principal(mol, sites, core):
    from .extended_inventory import selection_inventory

    inventory = selection_inventory(mol)
    # The attached =N–N and –N–N variants can register as ordinary amines
    # or imines in the generic inventory. Compare separate characteristic
    # groups and never promote these sites over an acid or ordinary amide.
    kinds = {s.kind for s in sites}
    cutoff = (12, 0) if 'imidohydrazide' in kinds else (11, 0, 1)
    if any(g.center not in core and (g.seniority <= cutoff or
           sites[0].element != 6 and g.seniority[0] == cutoff[0]) for g in inventory.groups):
        return False
    if inventory.supported:
        return True
    # When the inventory cannot classify the unit, independently bound the
    # rest of the graph to organyl, halo, hydroxy, ether and amine groups.
    return only_lower_substituents(mol, core)


def _site_numberings(sites, positions):
    """For geminal groups, compare both assignments of the N-prime series."""
    factor = 2**sum(s.kind == 'diimidamide' for s in sites)
    for orders in grouped_site_orders(sites, positions, factor=factor):
        locants, occurrences = {}, {}
        for position, order in orders:
            offset = 0
            for occurrence, site in enumerate(order):
                occurrences[site.center] = occurrence
                for idx, primes in site.nitrogens:
                    if primes is None:
                        locants[idx] = position
                    else:
                        locants[idx] = 'N' + "'" * (primes + offset) + (position if len(sites) > 1 else '')
                # The bridging imine N cannot take a substituent, but still
                # belongs to the preceding group in a geminal N-prime series.
                offset += len(site.nitrogens)
        interchangeable = [tuple(i for i, p in s.nitrogens if p) for s in sites if s.kind == 'diimidamide']
        for swaps in product((False, True), repeat=len(interchangeable)):
            result = locants.copy()
            for (a, b), swap in zip(interchangeable, swaps):
                if swap:
                    result[a], result[b] = result[b], result[a]
            yield result, occurrences


def imidic_parent_decision(mol, Chem):
    matches = mol.GetSubstructMatches(amidine_query(Chem))
    if not matches or mol.GetStereoGroups() or any(a.GetFormalCharge() or a.GetNumRadicalElectrons()
                                                   for a in mol.GetAtoms()):
        return None
    groups = {}
    for imine, center, amino in matches:
        groups.setdefault((center, amino), []).append(imine)
    sites = [_site(mol, c, ns, a) for (c, a), ns in groups.items()]
    if any(s is None for s in sites):
        return None
    core = set.union(*(s.atoms for s in sites))
    if len(core) != sum(len(s.atoms) for s in sites):
        return None
    secondary = []
    if len({(s.element, s.replacement, s.kind) for s in sites}) > 1:
        if any(s.element != 6 for s in sites):
            return None
        rank = ('imidamide', 'hydrazonamide', 'imidohydrazide')
        senior = min((s.kind for s in sites), key=rank.index)
        secondary = [s for s in sites if s.kind != senior]
        sites = [s for s in sites if s.kind == senior]
        for site in secondary:
            if any(mol.GetAtomWithIdx(i).GetIsotope() or
                   any(a.GetIdx() not in site.atoms for a in mol.GetAtomWithIdx(i).GetNeighbors())
                   for i, _ in site.nitrogens):
                return None
    if not _principal(mol, sites, core):
        return None
    branches = _nitrogen_branches(mol, Chem, sites, core)
    if branches is None:
        return None
    n_prefixes, arms = branches
    scope = frozenset(range(mol.GetNumAtoms()))
    if len(sites) > 1 and not secondary and all(s.owner is not None and
            mol.GetAtomWithIdx(s.owner).IsInRing() for s in sites):
        from .mapped_ring_parents import ring_systems
        hosts = [(ring, [s for s in sites if s.owner in ring]) for ring in ring_systems(mol)]
        hosts = [(ring, group) for ring, group in hosts if group]
        if len(hosts) > 1:
            from .linked_imidic_parents import linked_imidic_parent
            return linked_imidic_parent(mol, Chem, hosts, sites, n_prefixes, arms, scope)
    return _suffix_decision(mol, Chem, sites, secondary, core, n_prefixes, arms, scope)


def _suffix_decision(mol, Chem, sites, secondary, core, n_prefixes, arms, scope,
                     *, attachment=None, numberings=None, parent_facts=False):
    centers = {s.center for s in sites}
    owners = {s.owner for s in sites if s.owner is not None}
    external = (sites[0].element != 6 or len(sites) > 2
                or any(mol.GetAtomWithIdx(i).IsInRing() or mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in owners))
    local_masses = {i for site in sites for i in (site.center, *site.oxos)
                    if external and mol.GetAtomWithIdx(i).GetIsotope()}
    if external and (secondary or local_masses and len(sites) > 1):
        return None
    required = owners if external else centers
    if not required:
        return None
    framework = scope - core - arms
    if not external:
        framework |= centers | {s.center for s in secondary}
    # Explicit isotope hydrogen belongs to the atom it modifies, including
    # H on a formamidine carbon and H on any of the individual N sites.
    n_core = core - framework
    framework -= hydrogen_bindings(mol, dict.fromkeys(n_core, '')).keys()
    suffix = multiplied_suffix(len(sites), sites[0].suffix(external))
    if not suffix:
        return None
    if local_masses:
        masses = sorted((mol.GetAtomWithIdx(i).GetSymbol(), mol.GetAtomWithIdx(i).GetIsotope()) for i in local_masses)
        suffix = '(' + ','.join(f'{mass}{symbol}' for symbol, mass in masses) + ')' + suffix
    external_stereo = []
    if external:
        for site in sites:
            for i, prime in site.nitrogens:
                bond = mol.GetBondBetweenAtoms(site.center, i)
                if bond is not None and bond.GetStereo() and bond.GetBondTypeAsDouble() == 2:
                    external_stereo.append((site.center, i, prime))
    if external_stereo:
        from .stereochemistry import _potential_stereochemistry
        unique_bond = sum(info.type == Chem.StereoType.Bond_Double
                          for info in _potential_stereochemistry(mol, Chem)) == 1
    else:
        unique_bond = False
    if numberings is None:
        numberings = hydride_numberings(mol, Chem, next(iter(required)), framework, extended=True)
    frames = []
    for hydride, skeleton, frame_key, pi_key in numberings:
        if not required <= skeleton.keys() or any(s.center not in skeleton for s in secondary):
            continue
        positions = [skeleton[s.owner if external else s.center] for s in sites]
        ordered = tuple(sorted(positions, key=locant_key))
        if len(sites) > 1 and any(not loc.rstrip("'").isdecimal() for loc in positions):
            continue
        if max(positions.count(p) for p in positions) > 3:
            continue
        if not unique_bond and any(prime is None for _, _, prime in external_stereo) and (
                sites[0].element != 6 or len(set(positions)) < len(positions) and
                any("'" in pos for pos in positions)):
            continue
        port = locant_key(skeleton[attachment]) if attachment is not None else ()
        frames.append(((frame_key, locant_set(ordered), pi_key, port), hydride, skeleton, positions, ordered))
    options = []
    for rank, hydride, skeleton, positions, ordered in sorted(frames, key=lambda row: row[0]):
        # Compare every orientation tied on the earlier rules. Once a name is
        # proved, a worse suffix or attachment locant cannot win on prefixes.
        if options and rank != options[0][0][0]:
            break
        main = hydrogen_bindings(mol, skeleton)
        prefixes = skeleton_prefixes(mol, Chem, main, framework)
        if prefixes is None:
            continue
        for site in secondary:
            units = ('hydrazinyl', 'imino') if site.kind == 'imidohydrazide' else ('amino', 'hydrazinylidene')
            prefixes.extend((skeleton[site.center], word) for word in units)
        if external:
            omit = (attachment is None and len(sites) == 1 and (len(skeleton) == 1 or
                    ordered == ('1',) and not prefixes and (hydride == 'benzene' or
                    hydride.startswith('cyclo') and hydride.endswith('ane') or
                    sites[0].element != 6 and hydride == 'ethane')))
            parent = hydride + ('' if omit else '-' + ','.join(ordered) + '-') + suffix
        else:
            parent = attach_suffix(hydride, suffix)
        for nmap, occurrences in _site_numberings(sites, positions):
            mapping = dict(main)
            for site, loc in zip(sites, positions):
                mapping[site.center] = loc
                mapping.update(dict.fromkeys(site.oxos, loc))
            mapping.update(nmap)
            # Rank geminal stereobonds in their assigned occurrence order.
            # The writer attaches the occurrence prime to C in the descriptor.
            carbon_sites = {}
            for c, i, prime in external_stereo:
                if prime is None:
                    position = nmap[i]
                    carbon_sites[c] = 'C' + "'" * occurrences[c] + position
                    mapping[c] = mapping[i] = position + "'" * occurrences[c]
            mapping = hydrogen_bindings(mol, mapping)
            items = prefixes + [(nmap[i], word) for i, word, _ in n_prefixes]
            number_key = substituent_numbering_key(mol, mapping, items)
            if number_key is None:
                continue
            component = MappedComponent(parent, tuple(mapping.items()), frozenset(mapping),
                tuple((nitrogen_locant(loc), word) for loc, word in items),
                scope - mapping.keys(), systematic_parent=True,
                local_isotope_atoms=frozenset(local_masses),
                suffix_stereo_locants=tuple((c, i, '' if unique_bond else
                    carbon_sites[c] if prime is None else nmap[i]) for c, i, prime in external_stereo))
            name = decorate_component(mol, Chem, component)
            if name:
                options.append(((rank, number_key), name, mapping, items))
    if not options:
        return None
    best = min(k for k, *_ in options)
    winners = {name: (m, items) for k, name, m, items in options if k == best}
    if len(winners) != 1:
        return None
    name, (mapping, items) = winners.popitem()
    rule = 'P-66.4.1.1/P-66.4.1.4.1' if sites[0].kind in {'imidamide', 'diimidamide'} else 'P-66.4.2.1'
    decision = MethodDecision(name, 'substitutive', frozenset(centers), scope, rule, tuple(mapping.items()))
    if parent_facts:
        from .parent_selection import graph_parent_facts
        if external_stereo or any(mol.GetAtomWithIdx(i).GetIsotope() for i in mapping if i not in parent_facts):
            return None
        facts = graph_parent_facts(mol, parent_facts, {i: mapping[i] for i in parent_facts},
            principal_class=(11, 0, 1),
            suffix_centers=centers, suffix_locants=[mapping[s.owner] for s in sites],
            prefixes=items, kind='ring', name=name, allow_annotations=True)
        return (decision, facts) if facts is not None else None
    return decision
