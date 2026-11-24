"""Original-graph recognition and naming of oximes and their ethers (P-68.3.1.1.2)."""

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache


RULES = (
    "P-41",
    "P-43",
    "P-44.1.1/P-44.3/P-44.4",
    "P-62.3.1.1/P-62.4",
    "P-16.9.3",
    "P-68.3.1.1.2",
    "P-91.2.1.1/P-93.4.2.1.3",
)


class OximeType(str, Enum):
    OXIME = "oxime"          # C=N-OH; render as N-hydroxy imine
    O_ALKYL = "oxime-ether"  # C=N-O-C; render as N-alkoxy imine


@dataclass(frozen=True, slots=True)
class OximeSite:
    kind: str
    center: int
    atoms: tuple[int, ...]
    owner: int = -1
    nitrogen: int = -1
    oxygen: int = -1
    oxygen_substituent: int = -1


@dataclass(frozen=True, slots=True)
class MethodDecision:
    supported: bool
    operation: str
    reason: str
    site: OximeSite | None = None
    suffix: str = ""
    prefix: str = ""


def oxime_site(mol, carbon_idx):
    """Recognize C=N–OH or C=N–O–organyl without choosing the parent."""
    carbon = mol.GetAtomWithIdx(carbon_idx)
    if (carbon.GetAtomicNum() != 6 or carbon.GetFormalCharge()
            or carbon.GetNumRadicalElectrons()):
        return None
    doubles = [a for a in carbon.GetNeighbors()
               if a.GetAtomicNum() == 7 and _bond_order(mol, carbon_idx, a.GetIdx()) == 2]
    if len(doubles) != 1:
        return None
    nitrogen = doubles[0]
    if (nitrogen.GetFormalCharge() or nitrogen.GetNumRadicalElectrons()
            or nitrogen.IsInRing() or nitrogen.GetTotalValence() != 3):
        return None
    if len([a for a in nitrogen.GetNeighbors() if a.GetIdx() != carbon_idx]) != 1:
        return None
    oxygen = next((a for a in nitrogen.GetNeighbors()
                   if a.GetIdx() != carbon_idx and a.GetAtomicNum() == 8
                   and _bond_order(mol, nitrogen.GetIdx(), a.GetIdx()) == 1), None)
    if (oxygen is None or oxygen.GetFormalCharge() or oxygen.GetNumRadicalElectrons()
            or int(oxygen.GetChiralTag())):
        return None
    if oxygen.GetTotalValence() == 2 and oxygen.GetTotalNumHs(includeNeighbors=True) == 1:
        kind = OximeType.OXIME.value
        substituent = -1
        atoms = (carbon_idx, nitrogen.GetIdx(), oxygen.GetIdx())
    elif oxygen.GetTotalValence() == 2 and oxygen.GetTotalNumHs(includeNeighbors=True) == 0:
        heavy = [a for a in oxygen.GetNeighbors() if a.GetIdx() != nitrogen.GetIdx()]
        if len(heavy) != 1 or heavy[0].GetAtomicNum() != 6:
            return None
        ligand = heavy[0]
        # O-acyl, O-imidoyl and O-cyano derivatives are acid derivatives.
        # Aryl, alkenyl and alkynyl attachments are genuine ether arms.
        if (ligand.GetFormalCharge() or ligand.GetNumRadicalElectrons()
                or any(b.GetBondTypeAsDouble() in (2, 3)
                       and b.GetOtherAtom(ligand).GetAtomicNum() != 6 for b in ligand.GetBonds())):
            return None
        kind = OximeType.O_ALKYL.value
        substituent = heavy[0].GetIdx()
        atoms = (carbon_idx, nitrogen.GetIdx(), oxygen.GetIdx(), substituent)
    else:
        return None
    if carbon.IsInRing() and not all(
            n.GetAtomicNum() == 6 and n.IsInRing() and _bond_order(mol, carbon_idx, n.GetIdx()) == 1
            for n in carbon.GetNeighbors() if n.GetIdx() != nitrogen.GetIdx()):
        return None
    owner_c = [a.GetIdx() for a in carbon.GetNeighbors()
               if a.GetIdx() != nitrogen.GetIdx() and a.GetAtomicNum() == 6
               and _bond_order(mol, carbon_idx, a.GetIdx()) == 1]
    owner = owner_c[0] if len(owner_c) == 1 else -1
    return OximeSite(kind, carbon_idx, atoms, owner, nitrogen.GetIdx(), oxygen.GetIdx(), substituent)


@lru_cache(maxsize=2)
def oxime_query(Chem):
    return Chem.MolFromSmarts('[C;X3;+0]=[N;X2;!R;+0]-[O;X2;!R;+0]')


@lru_cache(maxsize=2)
def _hydroxy_hydrogen_query(Chem):
    return Chem.MolFromSmarts('[C;X3;+0]=[N;X2;!R;+0]-[O;X2;!R;+0]-[1#1,2#1,3#1]')


def isotopic_hydroxy_oxime_name(mol, Chem):
    if (mol.HasSubstructMatch(_hydroxy_hydrogen_query(Chem)) and not mol.GetStereoGroups()
            and not any(a.GetNumRadicalElectrons() or a.GetFormalCharge() for a in mol.GetAtoms())):
        return fast_oxime_name(mol, Chem)
    return ''


def fast_oxime_name(mol, Chem):
    """Compare fully mapped imine-chain candidates, or use the numbered ring writer."""
    from .extended_inventory import selection_inventory
    from .imine_family_components import chain_proposals
    from .method_parent_frontier import select_method_parent

    inventory = selection_inventory(mol)
    if not inventory.supported or not any(g.kind in {'oxime', 'oxime-ether'} for g in inventory.groups):
        return ''
    if len(inventory.principal) == 1 and inventory.principal[0].kind == 'ester':
        from .mapped_ester_general import mapped_ester_decision
        decision = mapped_ester_decision(mol, Chem)
        return decision.name if decision and decision.principal_centers == {inventory.principal[0].center} else ''
    if any(mol.GetAtomWithIdx(g.center).IsInRing()
           for g in inventory.groups if g.kind in {'oxime', 'oxime-ether'}):
        return cyclic_oxime_name(mol, Chem, inventory=inventory)
    if not all(g.kind in {'oxime', 'oxime-ether', 'imine'} for g in inventory.principal):
        from .mapped_principal_general import mapped_principal_decision
        decision = mapped_principal_decision(mol, Chem)
        if decision is not None and decision.principal_centers == {g.center for g in inventory.principal}:
            return decision.name
        return ''
    # All principal carbons are outside rings. A chain containing every
    # suffix site precedes candidates which would cite a site as a prefix.
    pool = chain_proposals(mol, Chem, inventory)
    best = select_method_parent(pool, lambda p: p.parent_selection,
                                proposal_set_complete=True, record_pairwise=False).selected
    return best.name if best else ''


def _ring_suffix_sites(mol, core, groups):
    """Attach a senior group only when its complete suffix unit belongs to this ring."""
    kinds = {g.kind for g in groups}
    suffix = {'amine': 'amine', 'alcohol': 'ol', 'ketone': 'one', 'acid': 'carboxylic acid',
              'amide': 'carboxamide', 'nitrile': 'carbonitrile', 'aldehyde': 'carbaldehyde'}
    if len(kinds) != 1 or groups[0].kind not in suffix:
        return None
    hosts, owned = [], set()
    for group in groups:
        atoms = set(group.atoms)
        outside = atoms - core
        if not outside or any(mol.GetAtomWithIdx(i).GetIsotope() for i in outside):
            return None
        ports = [(i, n.GetIdx()) for i in outside for n in mol.GetAtomWithIdx(i).GetNeighbors()
                 if n.GetIdx() not in outside]
        if len(ports) != 1 or ports[0][1] not in core or outside & owned:
            return None
        hosts.append(ports[0][1])
        owned.update(outside)
    return suffix[groups[0].kind], hosts, owned


def _oxime_ring_numberings(mol, Chem, core):
    """Yield complete ring numberings before the imine or senior suffix chooses its locant."""
    from .constants import ROOTS
    from .mapped_ring_parents import _core_graph, records, skeletons

    rings = [r for r in mol.GetRingInfo().AtomRings() if set(r) <= core]
    if len(rings) == 1 and all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in core):
        ring = rings[0]
        if (len(ring) not in ROOTS or any(mol.GetAtomWithIdx(i).GetFormalCharge()
                or mol.GetAtomWithIdx(i).GetNumRadicalElectrons() for i in core)):
            return
        if all(_bond_order(mol, ring[j-1], i) == 1 for j, i in enumerate(ring)):
            parent = 'cyclo' + ROOTS[len(ring)] + 'ane'
            for start in range(len(ring)):
                for step in (1, -1):
                    seq = (ring[(start + step*j) % len(ring)] for j in range(len(ring)))
                    yield parent, '', {i: str(j) for j, i in enumerate(seq, 1)}, ()
            return
        from .unsaturated_ring_parents import render_unsaturated_parent
        for row in skeletons(mol, Chem, core):
            if row[7] != 'unsaturated-monocycle':
                return
            state = row[8].unsaturation
            parent = render_unsaturated_parent(row[0], state)
            # Compare ordinary or compound bond locants after suffix locants.
            numbered = [((a,) if b == a + 1 else (a, b), order) for a, b, order in state.edge_locants]
            unsaturation = (tuple(sorted(pair for pair, _ in numbered)),
                            tuple(sorted(pair for pair, order in numbered if order == 2)))
            yield parent, '', dict(zip(row[1], row[2], strict=True)), unsaturation
        return
    capped = _core_graph(mol, Chem, core)
    if capped is None:
        return
    graph, original = capped
    local = frozenset(range(graph.GetNumAtoms()))
    parents = records(graph, Chem, core_filter=local, principal=())
    if len(parents) != 1:
        return
    parent = parents[0]
    construction = parent.construction
    if (construction is None or construction.atoms != local or construction.prefixes
            or construction.embedded_atoms or not parent.hydrogen_budget_proved):
        return
    numberings = graph.GetSubstructMatches(graph, uniquify=False, maxMatches=4097)
    if len(numberings) < 4097:
        for seq in numberings:
            mapping = {original[seq[i]]: loc for i, loc in parent.atom_locants}
            yield construction.parent, construction.hydro_prefix, mapping, ()


def cyclic_oxime_name(mol, Chem, *, inventory=None):
    """Name exocyclic oxime sites on one numbered ring system, including repeated sites."""
    matches = mol.GetSubstructMatches(oxime_query(Chem))
    if not matches or mol.GetStereoGroups() or len(Chem.GetMolFrags(mol)) != 1:
        return ''
    from .extended_inventory import selection_inventory
    from .component_mapping import MappedComponent, decorate_component, hydrogen_bindings, mapped_general_branch
    from .formatting import attach_suffix, enclose_name_unit, multiplied_suffix, prefix_alphabetization_key
    from .mapped_ring_parents import ring_systems
    from .nitrogen_acid_derivatives import _local_configuration
    from .parent_selection import locant_key, locant_set
    from .stereochemistry import _prepend_stereochemical_descriptors, assigned_atom_stereochemistry

    centers = {c for c, n, o in matches}
    core = next((ring for ring in ring_systems(mol) if centers <= ring), None)
    if core is None:
        return ''
    if inventory is None:
        inventory = selection_inventory(mol)
    if not inventory.supported or not inventory.principal:
        return ''
    sites = [g for g in inventory.groups if g.kind in {'oxime', 'oxime-ether'}]
    if {g.center for g in sites} != centers:
        return ''
    principal = all(g in sites for g in inventory.principal)
    if principal:
        suffix, suffix_hosts, embedded = 'imine', centers, set()
    else:
        unit = _ring_suffix_sites(mol, core, inventory.principal)
        if unit is None:
            return ''
        suffix, suffix_hosts, embedded = unit

    rendering = Chem.Mol(mol)
    oxygen_names, configurations = {}, {}
    for center, nitrogen, oxygen in matches:
        site = oxime_site(mol, center)
        if site is None or mol.GetAtomWithIdx(nitrogen).GetIsotope():
            return ''
        oxy, arm = _n_oxygen_substituent(mol, Chem, oxygen, nitrogen)
        configuration = _local_configuration(mol, Chem, center, nitrogen)
        if not oxy or arm & (core | embedded) or configuration is None:
            return ''
        embedded.update(arm | {nitrogen})
        oxygen_names[center] = oxy
        if configuration:
            configurations[center] = configuration
            rendering.GetBondBetweenAtoms(center, nitrogen).SetStereo(Chem.BondStereo.STEREONONE)
    if configurations and any(mol.GetAtomWithIdx(i).GetChiralTag() for i in core):
        before = dict(assigned_atom_stereochemistry(mol, Chem, include_pseudo=True))
        after = dict(assigned_atom_stereochemistry(rendering, Chem, include_pseudo=True))
        if any(before.get(i) != after.get(i) for i in core):
            return ''

    branches = []
    for host in core:
        for neighbor in mol.GetAtomWithIdx(host).GetNeighbors():
            root = neighbor.GetIdx()
            if root in core or root in embedded or neighbor.GetAtomicNum() == 1:
                continue
            text, arm = mapped_general_branch(mol, Chem, root, host)
            if not text or root not in arm or arm & (core | embedded):
                return ''
            branches.append((host, text))
            embedded.update(arm)
    needed = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1 or a.GetIsotope()}
    if set(hydrogen_bindings(mol, {i: '' for i in core})) | embedded != needed:
        return ''
    # A sole imine on a homogeneous saturated monocycle has no ring locant.
    omit = (principal and len(sites) == 1 and not branches and
            any(set(r) == core for r in mol.GetRingInfo().AtomRings())
            and all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6 and not mol.GetAtomWithIdx(i).GetIsotope()
                    and all(b.GetBondTypeAsDouble() == 1 for b in mol.GetAtomWithIdx(i).GetBonds()
                            if b.GetOtherAtomIdx(i) in core) for i in core))
    best, names = None, []
    for parent_word, hydro, mapping, unsaturation in _oxime_ring_numberings(mol, Chem, core):
        locants = tuple(sorted((mapping[i] for i in suffix_hosts), key=locant_key))
        prefixes = [(mapping[i], text) for i, text in branches]
        for center, oxy in oxygen_names.items():
            loc = mapping[center]
            if principal:
                prefixes.append(('N' + loc if len(sites) > 1 else 'N', oxy))
            else:
                prefixes.append((loc, (oxy if oxy.isalpha() else enclose_name_unit(oxy)) + 'imino'))
        key = (locant_set(locants), unsaturation, locant_set(loc for loc, _ in prefixes),
               tuple(locant_key(loc) for loc, _ in sorted(prefixes, key=lambda row: (
                   prefix_alphabetization_key(row[1]), locant_key(row[0])))))
        if best is not None and key > best:
            continue
        ending = attach_suffix(parent_word, multiplied_suffix(len(locants), suffix),
                               () if omit else locants)
        if not ending:
            continue
        bindings = hydrogen_bindings(mol, mapping)
        component = MappedComponent(ending, tuple(bindings.items()), frozenset(bindings),
                                    tuple(prefixes), frozenset(embedded), hydro_prefix=hydro,
                                    systematic_parent=True, nitrogen_site_locants=principal)
        name = decorate_component(rendering, Chem, component)
        if not name:
            continue
        if configurations:
            stereo = sorted(configurations, key=lambda i: locant_key(mapping[i]))
            name = _prepend_stereochemical_descriptors(name, [
                ('' if omit else mapping[i]) + configurations[i] for i in stereo])
        if best is None or key < best:
            best = key
            names.clear()
        names.append(name)
    return min(names) if names else ''


def _bond_order(mol, a, b):
    bond = mol.GetBondBetweenAtoms(a, b)
    return 0.0 if bond is None else bond.GetBondTypeAsDouble()


def oxime_ether_oxy_prefix(mol, Chem, root, host, atoms):
    """Name the C=N–O branch when the senior parent lies on the O-bound side."""
    oxygen = mol.GetAtomWithIdx(root)
    if (oxygen.GetDegree() != 2 or mol.GetAtomWithIdx(host).GetAtomicNum() != 6
            or oxygen.GetIsotope() or _bond_order(mol, root, host) != 1):
        return None
    nitrogen = next(a for a in oxygen.GetNeighbors() if a.GetIdx() != host)
    if nitrogen.GetAtomicNum() != 7:
        return None
    carbons = [a.GetIdx() for a in nitrogen.GetNeighbors() if a.GetAtomicNum() == 6
               and _bond_order(mol, nitrogen.GetIdx(), a.GetIdx()) == 2]
    if len(carbons) != 1:
        return None
    site = oxime_site(mol, carbons[0])
    if site is None or site.oxygen != root or site.oxygen_substituent != host:
        return None
    from .branch_rules import parent_chain_name
    from .component_mapping import MappedComponent, decorate_component, mapped_general_branch
    from .formatting import attach_suffix, prefix_alphabetization_key
    from .multifunctional_parent import _paths
    from .nitrogen_acid_derivatives import _local_configuration
    from .parent_selection import locant_set

    carbon = site.center
    if {(i, a.GetIdx()) for i in atoms for a in mol.GetAtomWithIdx(i).GetNeighbors()
        if a.GetAtomicNum() > 1 and a.GetIdx() not in atoms} != {(root, host)}:
        return None
    graph = Chem.Mol(mol)
    # The C=N descriptor belongs to the aminooxy unit, outside the
    # independently numbered ylidene fragment. Keep its original E/Z below.
    graph.GetBondBetweenAtoms(carbon, site.nitrogen).SetStereo(Chem.BondStereo.STEREONONE)
    geometry = _local_configuration(mol, Chem, carbon, site.nitrogen)
    if geometry is None:
        return None
    if mol.GetAtomWithIdx(carbon).IsInRing():
        from .constants import ROOTS
        from .mapped_ring_parents import records

        scope = atoms - {root, site.nitrogen}
        if (len(scope) in ROOTS and any(scope == frozenset(r) for r in mol.GetRingInfo().AtomRings())
                and all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6
                        and not mol.GetAtomWithIdx(i).GetIsotope()
                        and mol.GetAtomWithIdx(i).GetTotalNumHs() == 2 for i in scope - {carbon})
                and not mol.GetAtomWithIdx(carbon).GetIsotope()):
            return _aminooxy_prefix(mol, Chem, site, atoms,
                                    'cyclo' + ROOTS[len(scope)] + 'ylidene', scope, geometry)
        rows = records(graph, Chem, scope=scope, attachment=carbon, principal=(),
                       general_branches=True)
        if len(rows) != 1 or rows[0].construction is None:
            return None
        component = rows[0].construction
        if component.atoms | component.embedded_atoms != scope:
            return None
        if any(mol.GetAtomWithIdx(i).GetChiralTag() for i in scope):
            from .component_mapping import _general_branch_stereo_is_cited
            from .stereochemistry import assigned_atom_stereochemistry
            original = dict(assigned_atom_stereochemistry(mol, Chem, include_pseudo=True))
            rendered = dict(assigned_atom_stereochemistry(graph, Chem, include_pseudo=True))
            if (any(original.get(i) != rendered.get(i) for i in scope)
                    or not _general_branch_stereo_is_cited(mol, Chem, scope, rows[0].name)):
                return None
        return _aminooxy_prefix(mol, Chem, site, atoms, rows[0].name, scope, geometry)
    if any(mol.GetAtomWithIdx(i).GetChiralTag() for i in atoms):
        return None
    core = {carbon}
    stack = [carbon]
    while stack:
        idx = stack.pop()
        for a in mol.GetAtomWithIdx(idx).GetNeighbors():
            if (a.GetAtomicNum() == 6 and not a.IsInRing() and a.GetIdx() in atoms
                    and a.GetIdx() not in core):
                core.add(a.GetIdx())
                stack.append(a.GetIdx())
    if len(core) > 16:
        return None
    paths = [path for path in _paths(mol, core) if carbon in path]
    if not paths:
        return None
    longest = max(map(len, paths))
    best = []
    for path in paths:
        if len(path) != longest:
            continue
        for seq in dict.fromkeys((path, path[::-1])):
            mapping = {idx: str(i) for i, idx in enumerate(seq, 1)}
            framework = set(seq) | {root, site.nitrogen}
            occupied, prefixes = set(framework), []
            ports = ((idx, a.GetIdx()) for idx in seq for a in mol.GetAtomWithIdx(idx).GetNeighbors()
                     if a.GetIdx() not in framework and a.GetAtomicNum() > 1)
            complete = True
            for idx, other in ports:
                text, owned = mapped_general_branch(mol, Chem, other, idx)
                if not text or owned & occupied or not owned <= atoms:
                    complete = False
                    break
                occupied.update(owned)
                prefixes.append((mapping[idx], text))
            if not complete or occupied != atoms:
                continue
            parent = parent_chain_name(mol, seq)
            ylidene = attach_suffix(parent, 'ylidene', (mapping[carbon],))
            if not ylidene:
                continue
            body = decorate_component(graph, Chem, MappedComponent(
                ylidene, tuple(mapping.items()), frozenset(seq), tuple(prefixes),
                frozenset(atoms - set(seq) - {root, site.nitrogen}), systematic_parent=True))
            if not body:
                continue
            orders = [mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble() for a, b in zip(seq, seq[1:])]
            key = (int(mapping[carbon]), -sum(order > 1 for order in orders),
                   tuple(i for i, order in enumerate(orders, 1) if order > 1),
                   locant_set(loc for loc, _ in prefixes),
                   locant_set(loc for loc, _ in sorted(
                       prefixes, key=lambda item: prefix_alphabetization_key(item[1]))))
            best.append((key, body, frozenset(seq)))
    if not best:
        return None
    lowest = min(key for key, _, _ in best)
    winners = [(body, seq) for key, body, seq in best if key == lowest]
    if len({body for body, _ in winners}) != 1:
        return None
    body, seq = winners[0]
    return _aminooxy_prefix(mol, Chem, site, atoms, body, seq, geometry)


def _aminooxy_prefix(mol, Chem, site, atoms, body, seq, geometry):
    from .formatting import enclose_name_unit
    from .nitrogen_acid_derivatives import _record_configuration
    from .stereochemistry import assigned_bond_stereochemistry

    amino = enclose_name_unit(body) + 'amino'
    if geometry:
        amino = f'({geometry})-' + amino
    name = enclose_name_unit(amino) + 'oxy'
    _record_configuration(mol, site.center, site.nitrogen, name, geometry, atoms)
    for a, b, descriptor in assigned_bond_stereochemistry(mol, Chem):
        if {a, b} <= seq:
            _record_configuration(mol, a, b, name, descriptor, atoms)
    return name, atoms


def principal_scope(mol, groups):
    """Accept only a complete set of supported oxime sites in principal scope."""
    groups = tuple(groups)
    return bool(groups) and all(
        getattr(group, "kind", None) in {OximeType.OXIME.value, OximeType.O_ALKYL.value}
        and (site := oxime_site(mol, group.center)) is not None
        and site.kind == group.kind
        for group in groups
    )


def method_decision(mol, site, *, principal=True, senior_suffix=False):
    """Choose only the source-demonstrated operation for a validated site.

    The caller supplies the P-41/P-43 result. This function does not rank a
    candidate suffix against other characteristic groups.
    """
    found = oxime_site(mol, site.center if hasattr(site, "center") else int(site))
    if found is None:
        return MethodDecision(False, "unsupported", "not a neutral oxime or oxime ether")
    if senior_suffix or not principal:
        if found.kind == OximeType.OXIME.value:
            return MethodDecision(True, "prefix", "P-68.3.1.1.2 subordinate oxime", found, prefix="hydroxyimino")
        return MethodDecision(True, "prefix", "P-68.3.1.1.2 subordinate oxime ether", found, prefix="alkoxyimino")
    if found.kind == OximeType.OXIME.value:
        return MethodDecision(True, "imine-suffix", "N-hydroxy derivative of imine", found, suffix="imine", prefix="N-hydroxy")
    return MethodDecision(True, "imine-suffix", "alkoxy-substituted imine", found, suffix="imine", prefix="N-alkoxy")


def suffix_site_maps(mol, groups, parent, mapping):
    """Map each imine N and its N-substituent label without losing atom ids.

    The returned records are ordered by parent carbon locant. A rendering
    caller still has to name and claim the O-H/O-alkyl atom component.
    """
    result = []
    for group in groups:
        site = oxime_site(mol, group.center)
        if site is None or group.center not in parent or group.center not in mapping:
            return ()
        nloc = "N" + str(mapping[group.center]) if len(groups) > 1 else "N"
        result.append((mapping[group.center], site.nitrogen, site.oxygen, nloc, site.kind))
    return tuple(sorted(result, key=lambda item: _loc_key(item[0])))


def _loc_key(value):
    try:
        return (0, int(value), str(value))
    except (TypeError, ValueError):
        return (1, 0, str(value))


def _n_oxygen_substituent(mol, Chem, oxygen_idx, nitrogen_idx):
    """Name the complete N-bound hydroxy or organyloxy group on the input graph."""
    from .component_mapping import MappedComponent, decorate_component, mapped_general_branch, _general_branch_stereo_is_cited

    oxygen = mol.GetAtomWithIdx(oxygen_idx)
    if (oxygen.GetAtomicNum() != 8 or oxygen.GetFormalCharge()
            or oxygen.GetNumRadicalElectrons() or oxygen.GetChiralTag()
            or _bond_order(mol, oxygen_idx, nitrogen_idx) != 1):
        return "", frozenset()
    if oxygen.GetTotalNumHs(includeNeighbors=True):
        if oxygen.GetDegree() > 1:
            from .polyol_hydroxy_isotopes import hydroxy_isotope_prefix

            group = hydroxy_isotope_prefix(mol, Chem, host_atom_id=nitrogen_idx,
                                           oxygen_atom_id=oxygen_idx, host_locant='N',
                                           component_id='oxime-hydroxy')
            return (group.text, group.component_atom_ids) if group else ("", frozenset())
        if oxygen.GetTotalValence() != 2:
            return "", frozenset()
        name, atoms = 'hydroxy', frozenset({oxygen_idx})
    else:
        graph = mol
        if oxygen.GetIsotope():
            graph = Chem.Mol(mol)
            graph.GetAtomWithIdx(oxygen_idx).SetIsotope(0)
        name, atoms = mapped_general_branch(graph, Chem, oxygen_idx, nitrogen_idx)
        if not name or not _general_branch_stereo_is_cited(mol, Chem, atoms, name):
            return "", frozenset()
    if oxygen.GetIsotope():
        if any(mol.GetAtomWithIdx(i).GetAtomicNum() == 8 or mol.GetAtomWithIdx(i).GetIsotope()
               for i in atoms if i != oxygen_idx):
            return "", frozenset()
        name = decorate_component(mol, Chem, MappedComponent(
            name, ((oxygen_idx, ''),), frozenset({oxygen_idx}), embedded_atoms=atoms - {oxygen_idx}))
    return name, atoms


def oxime_imino_prefix(mol, Chem, root, blocked, atoms):
    """Bind the selected C=N branch to its N–OH or N–OR descriptor."""
    site = oxime_site(mol, blocked)
    if site is None or root != site.nitrogen or mol.GetAtomWithIdx(root).GetIsotope():
        return "", frozenset()
    text, owned = _n_oxygen_substituent(mol, Chem, site.oxygen, root)
    if not text or owned | {root} != atoms:
        return "", frozenset()
    from .formatting import enclose_name_unit
    from .molecule_cache import mol_cache_dict
    from .stereochemistry import assigned_bond_stereochemistry

    name = (text if text.isalpha() else enclose_name_unit(text)) + "imino"
    if mol.GetBondBetweenAtoms(root, blocked).GetStereo():
        label = next((d for a, b, d in assigned_bond_stereochemistry(mol, Chem)
                      if {a, b} == {root, blocked}), '')
        if not label:
            return "", frozenset()
        if mol.GetAtomWithIdx(blocked).IsInRing():
            # P-93.5.1.4.2.1(a) places the descriptor on the ring component.
            claims = mol_cache_dict(mol, 'oxime_parent_bond_stereo')
            claims[(root, blocked, name, atoms)] = label
        else:
            name = f"({label})-{name}"
            claims = mol_cache_dict(mol, 'rendered_mapped_branch_bond_stereo')
            claims[(root, blocked, name, atoms | {blocked})] = label
    return name, atoms


def original_cno_stereo(mol, Chem, site):
    """Return the recorded original C=N E/Z assignment; never infer on a cap."""
    carbon = site.center if hasattr(site, "center") else int(site)
    found = oxime_site(mol, carbon)
    if found is None:
        return ""
    from .stereochemistry import _indexed_bond_stereochemistry
    records = _indexed_bond_stereochemistry(mol, Chem)
    return next((descriptor for a, b, descriptor in records
                 if {a, b} == {found.center, found.nitrogen}), "")


def chain_proposals(mol, Chem, inventory, *, scope=None, principal=None):
    """Generate fully mapped acyclic chain parents for principal oximes.

    Oxygen branches are named from their original atoms (`hydroxy` or a
    supported alkoxy branch) at the N-site. P-41/P-43 ranking and non-chain
    parent competition remain with the caller. Ring parents, unsupported O
    substituents, and non-equivalent N-site grammar are declined.
    """
    from .multifunctional_parent import _paths
    from .component_mapping import MappedComponent, decorate_component, hydrogen_bindings, mapped_branch
    from .formatting import attach_suffix, complete_parent_name, multiplied_suffix
    from .branch_rules import parent_chain_name
    from .parent_selection import graph_parent_facts
    from .cross_route_selection import ParentProposal, _annotation_ownership

    groups = inventory.principal if principal is None else tuple(principal)
    if (not groups or not inventory.principal
            or inventory.principal[0].kind not in {OximeType.OXIME.value, OximeType.O_ALKYL.value}
            or not principal_scope(mol, groups)):
        return
    carbon = {
        atom.GetIdx() for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 6 and not atom.IsInRing()
        and (scope is None or atom.GetIdx() in scope)
    }
    for path in _paths(mol, carbon):
        for seq in (path, tuple(reversed(path))):
            covered = tuple(g for g in groups if g.center in seq)
            if not covered or len(covered) != len(groups):
                continue
            carbon_mapping = {idx: str(pos) for pos, idx in enumerate(seq, 1)}
            sites = suffix_site_maps(mol, covered, seq, carbon_mapping)
            if len(sites) != len(covered):
                continue
            mapping = dict(carbon_mapping)
            for _loc, nitrogen, _oxygen, nloc, _kind in sites:
                if nitrogen in mapping:
                    break
                mapping[nitrogen] = nloc
            else:
                prefixes, embedded, failed = [], set(), False
                for idx, locant in tuple(mapping.items()):
                    for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
                        root = neighbor.GetIdx()
                        if root in mapping or neighbor.GetAtomicNum() <= 1 or (scope is not None and root not in scope):
                            continue
                        if mol.GetBondBetweenAtoms(idx, root).GetBondTypeAsDouble() != 1:
                            failed = True
                            break
                        site = next((found for g in covered
                                     if (found := oxime_site(mol, g.center)) is not None
                                     and found.nitrogen == idx and found.oxygen == root), None)
                        if site is not None:
                            text, owned = _n_oxygen_substituent(mol, Chem, root, idx)
                        else:
                            text, owned = mapped_branch(mol, Chem, root, idx)
                        if not text or owned & (mapping.keys() | embedded) or any(c.isspace() for c in text):
                            failed = True
                            break
                        prefixes.append((locant, text))
                        embedded.update(owned)
                    if failed:
                        break
                mapping = hydrogen_bindings(mol, mapping)
                needed = {a.GetIdx() for a in mol.GetAtoms()
                          if a.GetAtomicNum() > 1 or a.GetIsotope() or int(a.GetChiralTag())}
                target = needed if scope is None else needed & set(scope)
                if failed or set(mapping) | embedded != target or not _annotation_ownership(mol, mapping):
                    continue
                parent = complete_parent_name(parent_chain_name(mol, seq))
                if not parent:
                    continue
                locants = tuple(sorted((carbon_mapping[g.center] for g in covered), key=int))
                suffix = multiplied_suffix(len(covered), "imine")
                if not suffix:
                    continue
                suffix_locants = () if len(covered) == 1 and len(seq) <= 2 else locants
                ending = attach_suffix(parent, suffix, suffix_locants)
                if not ending:
                    continue
                # Component decoration reads all atom and bond annotations.
                # Temporarily remove only C=N stereo so its generic alkene
                # formatter cannot invent a chain locant for hetero-linked E/Z.
                decoration_mol = Chem.Mol(mol)
                geometry = []
                # P-93.4.2.1.3 requires an E/Z locant when the name gives the
                # C=N site (for example, propan-1-imine); it is unlocanted
                # only when the name itself has no imine locant.
                stereo_locants = {} if not suffix_locants else {
                    g.center: carbon_mapping[g.center] for g in covered
                }
                for group in covered:
                    descriptor = original_cno_stereo(mol, Chem, group)
                    if descriptor:
                        locant = stereo_locants.get(group.center, "")
                        geometry.append((int(locant) if locant else 0, f"{locant}{descriptor}"))
                    site = oxime_site(mol, group.center)
                    bond = decoration_mol.GetBondBetweenAtoms(group.center, site.nitrogen)
                    if bond is not None:
                        bond.SetStereo(Chem.BondStereo.STEREONONE)
                # Removing C=N geometry for locant rendering can affect CIP
                # assignment at a coupled pseudoasymmetric atom. Keep the
                # candidate only when all mapped atom assignments are stable.
                from .stereochemistry import assigned_atom_stereochemistry
                original_atom_stereo = dict(assigned_atom_stereochemistry(
                    mol, Chem, atom_order="idx", include_pseudo=True,
                ))
                decoration_atom_stereo = dict(assigned_atom_stereochemistry(
                    decoration_mol, Chem, atom_order="idx", include_pseudo=True,
                ))
                if any(original_atom_stereo.get(idx, "") != decoration_atom_stereo.get(idx, "")
                       for idx in mapping):
                    continue
                component = MappedComponent(
                    ending, tuple(mapping.items()), frozenset(mapping), tuple(prefixes),
                    frozenset(embedded), systematic_parent=True,
                    nitrogen_site_locants=True,
                )
                name = decorate_component(decoration_mol, Chem, component)
                if not name or len(name) > 260:
                    continue
                if geometry:
                    geometry.sort()
                    prefix = "(" + ",".join(descriptor for _loc, descriptor in geometry) + ")-"
                    name = prefix + name
                facts = graph_parent_facts(
                    mol, seq, mapping, principal_class=groups[0].seniority,
                    suffix_centers=[g.center for g in covered], suffix_locants=locants,
                    prefixes=prefixes, kind="chain", name=name, allow_annotations=True,
                )
                if facts is not None:
                    yield ParentProposal(
                        name, "mapped-chain-oxime", groups[0].kind,
                        frozenset(g.center for g in covered), tuple(seq),
                        tuple(mapping.items()), facts, True, construction=component,
                    )
