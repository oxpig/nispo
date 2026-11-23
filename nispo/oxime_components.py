"""Bounded atom-mapped oxime recognition under Blue Book P-68.3.1.1.2.

The module recognizes neutral acyclic C=N-OH oximes and C=N-O-alkyl oxime
ethers. It exposes the source-selected naming operation but leaves shared
functional-group ranking and route integration to the caller.
"""

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


def oxime_site(mol, carbon_idx, *, allow_carboxymethyl=False, allow_cyclic_ether=False,
               allow_substituted_ether=False, subordinate=False):
    """Return a typed neutral C=N-OH/C=N-O-alkyl site, or None.

    C=N-N hydrazones, charged iminium species and C=N-O-Ar are outside this
    method. The cyclic writer can admit a substituted O-alkyl arm when it
    supplies a complete mapped branch.
    """
    carbon = mol.GetAtomWithIdx(carbon_idx)
    if (carbon.GetAtomicNum() != 6 or carbon.GetFormalCharge()
            or carbon.GetNumRadicalElectrons()
            or carbon.IsInRing() and not (allow_cyclic_ether or subordinate)):
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
            or oxygen.GetIsotope() or int(oxygen.GetChiralTag())):
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
        if (not (subordinate and all(b.GetBondTypeAsDouble() == 1 for b in ligand.GetBonds()))
                and not (allow_substituted_ether and carbon.IsInRing()
                 and all(b.GetBondTypeAsDouble() == 1 for b in ligand.GetBonds()))
                and not _simple_alkyl_substituent(mol, ligand.GetIdx(), oxygen.GetIdx())
                and not (allow_carboxymethyl and _carboxymethyl(mol, ligand, oxygen.GetIdx()))):
            return None
        kind = OximeType.O_ALKYL.value
        substituent = heavy[0].GetIdx()
        atoms = (carbon_idx, nitrogen.GetIdx(), oxygen.GetIdx(), substituent)
    else:
        return None
    if carbon.IsInRing() and not subordinate and (kind != OximeType.O_ALKYL.value or not all(
            n.GetAtomicNum() == 6 and n.IsInRing() and _bond_order(mol, carbon_idx, n.GetIdx()) == 1
            for n in carbon.GetNeighbors() if n.GetIdx() != nitrogen.GetIdx())):
        return None
    owner_c = [a.GetIdx() for a in carbon.GetNeighbors()
               if a.GetIdx() != nitrogen.GetIdx() and a.GetAtomicNum() == 6
               and _bond_order(mol, carbon_idx, a.GetIdx()) == 1]
    owner = owner_c[0] if len(owner_c) == 1 else -1
    return OximeSite(kind, carbon_idx, atoms, owner, nitrogen.GetIdx(), oxygen.GetIdx(), substituent)


@lru_cache(maxsize=1)
def _cyclic_oxime_ether_query(Chem):
    return Chem.MolFromSmarts('[C;R;+0]=[N;!R;+0]-[O;!R;+0]-[C;+0]')


def cyclic_oxime_ether_name(mol, Chem):
    """Join a mapped cyclic hydride and one exocyclic C=N–O–organyl unit."""
    matches = mol.GetSubstructMatches(_cyclic_oxime_ether_query(Chem), maxMatches=2)
    if (len(matches) != 1 or mol.GetStereoGroups() or len(Chem.GetMolFrags(mol)) != 1
            or any(a.GetFormalCharge() or a.GetNumRadicalElectrons() or a.HasQuery()
                   for a in mol.GetAtoms()) or any(b.HasQuery() for b in mol.GetBonds())):
        return ''
    center, nitrogen, oxygen, _ = matches[0]
    site = oxime_site(mol, center, allow_cyclic_ether=True, allow_substituted_ether=True)
    if (site is None or site.kind != OximeType.O_ALKYL.value
            or any(mol.GetAtomWithIdx(i).GetIsotope() or mol.GetAtomWithIdx(i).GetChiralTag()
                   for i in (nitrogen, oxygen))):
        return ''

    from .component_mapping import MappedComponent, decorate_component, hydrogen_bindings
    from .formatting import attach_suffix, enclose_name_unit, multiplied_suffix, prefix_alphabetization_key
    from .mapped_ring_parents import _core_graph, records, ring_systems
    from .nitrogen_acid_derivatives import _local_configuration
    from .parent_selection import locant_key, locant_set
    from .stereochemistry import _prepend_stereochemical_descriptors, assigned_atom_stereochemistry

    core = next((system for system in ring_systems(mol) if center in system), None)
    if core is None:
        return ''
    configuration = _local_configuration(mol, Chem, center, nitrogen)
    if configuration is None:
        return ''
    branches = _cyclic_oxime_ether_branches(mol, Chem, core, site)
    if branches is None:
        return ''
    oxy, other_prefixes, amino_sites, embedded = branches

    capped = _core_graph(mol, Chem, core)
    if capped is None:
        return ''
    graph, original = capped
    local = frozenset(range(graph.GetNumAtoms()))
    parents = records(graph, Chem, core_filter=local, principal=())
    if len(parents) != 1:
        return ''
    parent = parents[0]
    construction = parent.construction
    if (construction is None or construction.atoms != local or construction.embedded_atoms
            or construction.prefixes or not parent.hydrogen_budget_proved
            or {atom for atom, _ in parent.atom_locants} != local):
        return ''
    # Numberings equivalent on the capped parent still differ at original
    # substituent sites. Apply suffix and prefix locant comparisons after
    # mapping every symmetry-equivalent numbering back to the source atoms.
    numberings = graph.GetSubstructMatches(graph, uniquify=False, maxMatches=4097)
    if len(numberings) == 4097:
        return ''
    rendering = Chem.Mol(mol)
    if configuration:
        # Decorate the rest of the original graph before inserting this one
        # separately owned exocyclic stereo descriptor.
        rendering.GetBondBetweenAtoms(center, nitrogen).SetStereo(Chem.BondStereo.STEREONONE)
        if any(mol.GetAtomWithIdx(i).GetChiralTag() for i in core):
            before = dict(assigned_atom_stereochemistry(mol, Chem, include_pseudo=True))
            after = dict(assigned_atom_stereochemistry(rendering, Chem, include_pseudo=True))
            if any(before.get(i) != after.get(i) for i in core):
                return ''

    needed = {atom.GetIdx() for atom in mol.GetAtoms()
              if atom.GetAtomicNum() > 1 or atom.GetIsotope()}
    best = None
    for numbering in numberings:
        mapping = {original[numbering[i]]: locant for i, locant in parent.atom_locants}
        oxime_locant = mapping[center]
        prefixes = [(mapping[host], text) for host, text in other_prefixes]
        if amino_sites:
            suffix = multiplied_suffix(len(amino_sites), 'amine')
            suffix_locs = tuple(sorted((mapping[i] for i in amino_sites), key=locant_key))
            imino = (oxy if oxy.isalpha() else enclose_name_unit(oxy)) + 'imino'
            prefixes.append((oxime_locant, imino))
        else:
            suffix, suffix_locs = 'imine', (oxime_locant,)
            prefixes.append(('N', oxy))
        base = attach_suffix(construction.parent, suffix, suffix_locs) if suffix else ''
        if not base:
            continue
        bindings = hydrogen_bindings(mol, mapping)
        if bindings.keys() & embedded or bindings.keys() | embedded != needed:
            continue
        component = MappedComponent(
            base, tuple(bindings.items()), frozenset(bindings), tuple(prefixes), embedded,
            hydro_prefix=construction.hydro_prefix, systematic_parent=True,
            nitrogen_site_locants=not amino_sites,
        )
        name = decorate_component(rendering, Chem, component)
        if not name:
            continue
        if configuration:
            name = _prepend_stereochemical_descriptors(name, [oxime_locant + configuration])
        key = (locant_set(suffix_locs), locant_set(loc for loc, _ in prefixes),
               tuple(locant_key(loc) for loc, _ in sorted(prefixes, key=lambda row: (
                   prefix_alphabetization_key(row[1]), locant_key(row[0])))), name)
        if best is None or key < best:
            best = key
    return best[-1] if best is not None else ''


def _cyclic_oxime_ether_branches(mol, Chem, core, site):
    from .component_mapping import _general_branch_stereo_is_cited, mapped_general_branch

    def branch(root, host):
        text, owned = mapped_general_branch(mol, Chem, root, host)
        if (not text or any(c.isspace() for c in text) or root not in owned
                or {(i, neighbour.GetIdx()) for i in owned
                    for neighbour in mol.GetAtomWithIdx(i).GetNeighbors()
                    if neighbour.GetAtomicNum() > 1 and neighbour.GetIdx() not in owned} != {(root, host)}
                or not _general_branch_stereo_is_cited(mol, Chem, owned, text)):
            return '', frozenset()
        return text, owned

    oxy, arm = branch(site.oxygen, site.nitrogen)
    if not oxy or arm & (core | {site.nitrogen}):
        return None
    embedded = set(arm) | {site.nitrogen}
    prefixes, amino_sites = [], []
    for host in sorted(core):
        for neighbour in mol.GetAtomWithIdx(host).GetNeighbors():
            root = neighbour.GetIdx()
            if root in core or neighbour.GetAtomicNum() == 1 or (host, root) == (site.center, site.nitrogen):
                continue
            if mol.GetBondBetweenAtoms(host, root).GetBondTypeAsDouble() != 1:
                return None
            text, owned = branch(root, host)
            if not text or owned & (core | embedded):
                return None
            if text == 'amino' and owned == {root} and neighbour.GetAtomicNum() == 7:
                amino_sites.append(host)
            else:
                prefixes.append((host, text))
            embedded.update(owned)
    return oxy, tuple(prefixes), tuple(amino_sites), frozenset(embedded)


def _bond_order(mol, a, b):
    bond = mol.GetBondBetweenAtoms(a, b)
    return 0.0 if bond is None else bond.GetBondTypeAsDouble()


def _simple_alkyl_substituent(mol, root, blocked):
    """Require an ordinary acyclic, saturated C/H substituent on oxime O."""
    seen, stack = set(), [root]
    while stack:
        idx = stack.pop()
        if idx in seen:
            continue
        atom = mol.GetAtomWithIdx(idx)
        if (atom.GetAtomicNum() != 6 or atom.IsInRing() or atom.GetFormalCharge()
                or atom.GetNumRadicalElectrons() or atom.GetIsotope()
                or int(atom.GetChiralTag())):
            return False
        seen.add(idx)
        for neighbor in atom.GetNeighbors():
            other = neighbor.GetIdx()
            if other == blocked or other in seen or neighbor.GetAtomicNum() == 1:
                continue
            bond = mol.GetBondBetweenAtoms(idx, other)
            if bond.GetBondTypeAsDouble() != 1 or neighbor.GetAtomicNum() != 6:
                return False
            stack.append(other)
    return True


def _carboxymethyl(mol, methylene, oxygen):
    if (methylene.IsInRing() or methylene.GetFormalCharge() or methylene.GetIsotope()
            or methylene.GetNumRadicalElectrons() or methylene.GetTotalNumHs(includeNeighbors=True) != 2):
        return False
    children = [a for a in methylene.GetNeighbors()
                if a.GetIdx() != oxygen and a.GetAtomicNum() > 1]
    if len(children) != 1 or children[0].GetAtomicNum() != 6:
        return False
    center = children[0]
    return (not center.IsInRing() and not center.GetFormalCharge()
            and _bond_order(mol, methylene.GetIdx(), center.GetIdx()) == 1
            and center.GetDegree() == 3
            and sorted(_bond_order(mol, center.GetIdx(), a.GetIdx()) for a in center.GetNeighbors()
                       if a.GetAtomicNum() == 8 and a.GetDegree() == 1
                       and not a.GetFormalCharge() and not a.GetIsotope()
                       and a.GetTotalNumHs() == int(_bond_order(mol, center.GetIdx(), a.GetIdx()) == 1))
            == [1, 2])


def oxime_ether_oxy_prefix(mol, Chem, root, host, atoms):
    """Compose the original C=N–O arm under a carboxymethyl acid parent."""
    oxygen = mol.GetAtomWithIdx(root)
    if (oxygen.GetDegree() != 2 or _bond_order(mol, root, host) != 1
            or not _carboxymethyl(mol, mol.GetAtomWithIdx(host), root)):
        return None
    nitrogen = next(a for a in oxygen.GetNeighbors() if a.GetIdx() != host)
    if nitrogen.GetAtomicNum() != 7:
        return None
    carbons = [a.GetIdx() for a in nitrogen.GetNeighbors() if a.GetAtomicNum() == 6
               and _bond_order(mol, nitrogen.GetIdx(), a.GetIdx()) == 2]
    if len(carbons) != 1:
        return None
    site = oxime_site(mol, carbons[0], allow_carboxymethyl=True)
    if site is None or site.oxygen != root or site.oxygen_substituent != host:
        return None
    from .branch_rules import parent_chain_name
    from .component_mapping import MappedComponent, decorate_component, mapped_general_branch
    from .formatting import attach_suffix, enclose_name_unit, prefix_alphabetization_key
    from .multifunctional_parent import _paths
    from .nitrogen_acid_derivatives import _local_configuration, _record_configuration
    from .parent_selection import locant_set
    from .stereochemistry import assigned_bond_stereochemistry

    carbon = site.center
    if ({(i, a.GetIdx()) for i in atoms for a in mol.GetAtomWithIdx(i).GetNeighbors()
         if a.GetAtomicNum() > 1 and a.GetIdx() not in atoms} != {(root, host)}
            or any(mol.GetAtomWithIdx(i).GetChiralTag() for i in atoms)):
        return None
    graph = Chem.Mol(mol)
    # The C=N descriptor belongs to the aminooxy unit, outside the
    # independently numbered ylidene fragment. Keep its original E/Z below.
    graph.GetBondBetweenAtoms(carbon, site.nitrogen).SetStereo(Chem.BondStereo.STEREONONE)
    geometry = _local_configuration(mol, Chem, carbon, site.nitrogen)
    if geometry is None:
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
    amino = enclose_name_unit(body) + 'amino'
    if geometry:
        amino = f'({geometry})-' + amino
    name = enclose_name_unit(amino) + 'oxy'
    _record_configuration(mol, carbon, site.nitrogen, name, geometry, atoms)
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
    found = oxime_site(mol, site.center if hasattr(site, "center") else int(site),
                       subordinate=senior_suffix or not principal)
    if found is None:
        return MethodDecision(False, "unsupported", "not a neutral acyclic C=N-OH/C=N-O-alkyl site")
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
    """Render a directly mapped oxime N–O(H/alkyl) substituent."""
    from .component_mapping import mapped_branch
    from .formatting import alkyl_to_alkoxy

    oxygen = mol.GetAtomWithIdx(oxygen_idx)
    if (oxygen.GetAtomicNum() != 8 or oxygen.GetFormalCharge()
            or oxygen.GetNumRadicalElectrons() or oxygen.GetIsotope()
            or int(oxygen.GetChiralTag())
            or mol.GetBondBetweenAtoms(oxygen_idx, nitrogen_idx) is None
            or mol.GetBondBetweenAtoms(oxygen_idx, nitrogen_idx).GetBondTypeAsDouble() != 1):
        return "", frozenset()
    heavy = [a for a in oxygen.GetNeighbors() if a.GetIdx() != nitrogen_idx and a.GetAtomicNum() > 1]
    if not heavy:
        isotope_h = any(a.GetAtomicNum() == 1 and a.GetIsotope() for a in oxygen.GetNeighbors())
        if (not isotope_h and oxygen.GetTotalValence() == 2
                and oxygen.GetTotalNumHs(includeNeighbors=True) == 1):
            return "hydroxy", frozenset((oxygen_idx,))
        return "", frozenset()
    if len(heavy) != 1 or heavy[0].GetAtomicNum() != 6 or oxygen.GetTotalNumHs(includeNeighbors=True):
        return "", frozenset()
    child = heavy[0]
    if mol.GetBondBetweenAtoms(oxygen_idx, child.GetIdx()).GetBondTypeAsDouble() != 1:
        return "", frozenset()
    alkyl, owned = mapped_branch(mol, Chem, child.GetIdx(), oxygen_idx)
    if (not alkyl or any(mol.GetAtomWithIdx(i).GetIsotope()
                         or int(mol.GetAtomWithIdx(i).GetChiralTag()) for i in owned)):
        return "", frozenset()
    alkoxy = alkyl_to_alkoxy(alkyl)
    return (alkoxy, frozenset(set(owned) | {oxygen_idx})) if alkoxy else ("", frozenset())


def oxime_imino_prefix(mol, Chem, root, blocked, atoms):
    """Bind the selected C=N branch to its N–OH or N–OR descriptor."""
    site = oxime_site(mol, blocked, subordinate=True)
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
