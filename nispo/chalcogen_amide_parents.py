"""Substitutive oxyamide suffixes on carbon chains and numbered rings (P-66.1.1.2)."""
from dataclasses import dataclass, replace
from .functional_class_queries import amide_query

from .branch_rules import collect_subtree
from .component_mapping import decorate_component, mapped_general_branch
from .formatting import complete_parent_name
from .naming_methods import MethodDecision
from .naming_result import SelectionLimitation
from .sulfur_selection import sulfur_suffix_locants


_SUFFIX = {16: ('sulfinamide', 'sulfonamide'), 34: ('seleninamide', 'selenonamide'),
           52: ('tellurinamide', 'telluronamide')}
_ACYL = {16: ('sulfinyl', 'sulfonyl'), 34: ('seleninyl', 'selenonyl'),
         52: ('tellurinyl', 'telluronyl')}


@dataclass(frozen=True)
class _Oxyamide:
    center: int
    host: int
    nitrogen: int
    element: int
    oxos: int
    atoms: frozenset
    n_branches: tuple
    configuration: str

    @property
    def seniority(self):
        # P-41, class 7a, applied to the corresponding amides by P-66.1.1.2.
        return self.element, -self.oxos

    def prefix(self):
        from .formatting import enclose_name_unit, format_prefixes
        groups = format_prefixes([('', word) for word in self.n_branches],
                                 omit_locants=True, hierarchical_enclosures=True)
        if (self.element, self.oxos) == (16, 2):
            word = groups + 'sulfamoyl'
            if not groups:
                return word
        else:
            amino = enclose_name_unit(groups + 'amino') if groups else 'amino'
            word = amino + _ACYL[self.element][self.oxos - 1]
        if self.configuration:
            return '(' + self.configuration + ')-' + word
        # Keep the whole compound prefix together, including when multiplied.
        return enclose_name_unit(word) if word.isalpha() else word


def _oxyamide(mol, Chem, center, host, nitrogen, oxos):
    n_atom = mol.GetAtomWithIdx(nitrogen)
    core = {center, nitrogen, *oxos}
    owned, branches = set(core), []
    for bond in n_atom.GetBonds():
        root = bond.GetOtherAtomIdx(nitrogen)
        if root == center:
            continue
        if bond.GetBondType() != Chem.BondType.SINGLE or mol.GetAtomWithIdx(root).GetAtomicNum() != 6:
            return None
        word, atoms = mapped_general_branch(mol, Chem, root, nitrogen)
        if not word or atoms & owned or host in atoms:
            return None
        owned.update(atoms)
        branches.append(word)
    atom = mol.GetAtomWithIdx(center)
    configuration = ''
    if atom.GetChiralTag():
        from .stereochemistry import assigned_atom_stereochemistry
        configuration = dict(assigned_atom_stereochemistry(mol, Chem, atom_order='idx',
                                                            include_pseudo=True)).get(center)
        if len(oxos) != 1 or configuration not in {'R', 'S'}:
            return None
    return _Oxyamide(center, host, nitrogen, atom.GetAtomicNum(), len(oxos),
                     frozenset(owned), tuple(branches), configuration)


def _mixed_frames(mol, Chem, host, skeleton):
    """Parent orientations before suffix, substituent and N-site numbering."""
    from .branch_rules import parent_chain_name
    from .mapped_ring_parents import ring_systems, skeletons
    from .rooted_carbon_parents import carbon_paths

    if mol.GetAtomWithIdx(host).IsInRing():
        ring = next((r for r in ring_systems(mol) if host in r), None)
        if not ring or not ring <= skeleton:
            return
        aromatic = all(mol.GetAtomWithIdx(i).GetIsAromatic() and not (
            mol.GetAtomWithIdx(i).GetAtomicNum() != 6 and mol.GetAtomWithIdx(i).GetTotalNumHs())
            for i in ring)
        for hydride, seq, locants, _, _, saturated, key, *_ in skeletons(mol, Chem, ring):
            if saturated or aromatic:
                yield hydride, dict(zip(seq, locants)), (key,), ()
        return
    for path in carbon_paths(mol, {host}, skeleton):
        for seq in dict.fromkeys((path, path[::-1])):
            orders = [mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble() for a, b in zip(seq, seq[1:])]
            hydride = parent_chain_name(mol, seq)
            if hydride:
                yield (complete_parent_name(hydride), {idx: str(n) for n, idx in enumerate(seq, 1)},
                       (-len(seq), -sum(o > 1 for o in orders), -orders.count(2)),
                       (tuple(n for n, o in enumerate(orders, 1) if o > 1),
                        tuple(n for n, o in enumerate(orders, 1) if o == 2)))


def _frame_prefixes(mol, Chem, mapping, scope, units, suffix_centers):
    """Render every off-frame branch once, including nested junior oxyamides."""
    direct = [u for u in units if u.host in mapping]
    named = set(mapping).union(*(u.atoms for u in direct))
    prefixes = [(mapping[u.host], u.prefix()) for u in direct if u.center not in suffix_centers]
    for idx in mapping:
        for atom in mol.GetAtomWithIdx(idx).GetNeighbors():
            root = atom.GetIdx()
            if root in named or root not in scope:
                continue
            atoms = frozenset(collect_subtree(mol, root, idx))
            if not atoms <= scope or named & atoms:
                return None
            nested = [u for u in units if u.center in atoms]
            if nested:
                if atom.GetAtomicNum() != 6 or any(not u.atoms <= atoms for u in nested):
                    return None
                word = _mixed_branch(mol, Chem, root, atoms, nested)
            else:
                word, owned = mapped_general_branch(mol, Chem, root, idx)
                if owned != atoms:
                    return None
            if not word:
                return None
            prefixes.append((mapping[idx], word))
            named.update(atoms)
    return prefixes if named == scope else None


def _mixed_branch(mol, Chem, root, scope, units):
    from .component_mapping import MappedComponent
    from .formatting import attach_suffix, substituent_parent_suffix
    from .parent_selection import locant_key

    skeleton = scope.difference(*(u.atoms for u in units))
    options = []
    for hydride, mapping, frame_key, pi_key in _mixed_frames(mol, Chem, root, skeleton):
        prefixes = _frame_prefixes(mol, Chem, mapping, scope, units, ())
        if prefixes is None:
            continue
        loc = mapping[root]
        if hydride == 'benzene':
            if loc != '1':
                continue
            parent = 'phenyl'
        elif loc == '1' and hydride.endswith('ane') and (
                not mol.GetAtomWithIdx(root).IsInRing() or hydride.startswith('cyclo')):
            parent = substituent_parent_suffix(hydride)
        else:
            parent = attach_suffix(hydride, 'yl', (loc,))
        component = MappedComponent(parent, tuple(mapping.items()), frozenset(mapping), tuple(prefixes),
            scope - mapping.keys(), systematic_parent=True,
            prefix_mode='unlocanted' if len(mapping) == 1 else 'locanted')
        name = decorate_component(mol, Chem, component)
        if name:
            key = (frame_key, locant_key(loc), pi_key, _prefix_numbering(prefixes))
            options.append((key, name))
    if not options:
        return ''
    best = min(key for key, _ in options)
    winners = {name for key, name in options if key == best}
    return winners.pop() if len(winners) == 1 else ''


def _prefix_numbering(prefixes):
    from .formatting import prefix_alphabetization_key
    from .parent_selection import locant_key, locant_set
    citation = sorted(prefixes, key=lambda p: (prefix_alphabetization_key(p[1]), locant_key(p[0])))
    return locant_set(loc for loc, _ in prefixes), tuple(locant_key(loc) for loc, _ in citation)


def _mixed_amides(mol, Chem, units):
    from .component_mapping import MappedComponent
    from .constants import MULT
    from .parent_selection import locant_key, locant_set
    from .parent_annotations import stereo_numbering

    priority = min(u.seniority for u in units)
    senior = [u for u in units if u.seniority == priority]
    occupied = frozenset().union(*(u.atoms for u in units))
    if len(occupied) != sum(len(u.atoms) for u in units):
        return None
    scope = frozenset(range(mol.GetNumAtoms()))
    skeleton = scope - occupied
    multiplier = '' if len(senior) == 1 else MULT.get(len(senior))
    if multiplier is None:
        return None
    suffix = multiplier + _SUFFIX[senior[0].element][senior[0].oxos - 1]
    suffix_units = senior
    configuration = ''
    if any(u.configuration for u in senior):
        # A unique center needs no stereo locant. Otherwise keep the
        # chalcogen descriptor on its own acyl prefix to identify its atom.
        stereocenters = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
        if (len(senior) == 1 and len(stereocenters) == 1 and stereocenters[0][0] == senior[0].center
                and not any(b.GetStereo() for b in mol.GetBonds())):
            configuration = '(' + senior[0].configuration + ')-'
        else:
            suffix_units = []
    suffix_centers = frozenset(u.center for u in suffix_units)
    options = []
    for parent, mapping, frame_key, pi_key in _mixed_frames(mol, Chem, senior[0].host, skeleton):
        if any(u.host not in mapping for u in senior):
            continue
        sites = sorted((mapping[u.host] for u in suffix_units), key=locant_key)
        if len(sites) != len(set(sites)) and len(senior) > 1 and any(u.n_branches for u in senior):
            continue  # A geminal pair needs distinct N and N' site locants.
        prefixes = _frame_prefixes(mol, Chem, mapping, scope, units, suffix_centers)
        if prefixes is None:
            continue
        n_prefixes = []
        for u in suffix_units:
            loc = 'N' if len(senior) == 1 else 'N' + mapping[u.host]
            n_prefixes.extend((loc, text) for text in u.n_branches)
        # A junior unit is always present. Only methane can still omit its
        # suffix locant. Fused-ring locants may contain letters.
        locants = '' if len(mapping) == 1 and len(sites) == 1 else '-' + ','.join(sites) + '-'
        prefixes.extend(n_prefixes)
        hydride = parent + locants + suffix if suffix_units else parent
        component = MappedComponent(hydride, tuple(mapping.items()), frozenset(mapping),
            tuple(prefixes), scope - set(mapping), systematic_parent=True, nitrogen_site_locants=True)
        name = decorate_component(mol, Chem, component)
        if name:
            stereo = stereo_numbering(mol, mapping)
            if stereo is None:
                continue
            key = (frame_key, locant_set(sites), pi_key, _prefix_numbering(prefixes), stereo)
            options.append((key, configuration + name, tuple(mapping.items())))
    if not options:
        return None
    best = min(key for key, _, _ in options)
    winners = {name: locants for key, name, locants in options if key == best}
    if len(winners) != 1:
        return None
    name, locants = winners.popitem()
    issues = () if suffix_units else (SelectionLimitation('stereogenic-oxyamide-suffix', 'P-66.1.1.2/P-93',
        'The stereogenic chalcogen is cited in its own configured prefix.'),)
    return MethodDecision(name, 'substitutive' if suffix_units else 'general', suffix_centers, scope,
        'P-41/P-66.1.1.2/P-66.1.1.4.2', locants,
        component_roles=(('parent-and-substituents', skeleton, locants),
                         *(('oxyamide', u.atoms, ()) for u in units)), preference_issues=issues)


def _lower_substituents(mol, core, host):
    """Admit organyl, halo, hydroxy, ether, and amine substitution outside the unit."""
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() or atom.GetNumRadicalElectrons() or atom.GetIsotope() and atom.GetIdx() not in core:
            return False
        if atom.GetIdx() in core or atom.GetIdx() == host or atom.GetAtomicNum() == 6:
            continue
        if atom.GetAtomicNum() in {9, 17, 35, 53} and atom.GetDegree() == 1:
            continue
        if atom.GetIsAromatic() and atom.GetAtomicNum() in {7, 8, 16}:
            continue
        if (atom.GetAtomicNum() in {7, 8} and all(b.GetBondTypeAsDouble() == 1
                and b.GetOtherAtom(atom).GetAtomicNum() == 6 for b in atom.GetBonds())):
            continue
        return False
    return True


def _parent_components(mol, Chem, host, center, scope):
    if not mol.GetAtomWithIdx(host).IsInRing():
        from .rooted_carbon_parents import chain_orientations
        for key, component in chain_orientations(mol, Chem, host, center, scope):
            locant = int(dict(component.atom_locants)[host])
            positions = sulfur_suffix_locants(count=1, locants=(locant,), cyclic=False,
                path_length=len(component.atoms), parent_prefix_count=len(component.prefixes))
            yield key, complete_parent_name(component.parent) + positions, component
        return

    from .mapped_ring_parents import records
    for row in records(mol, Chem, scope=scope, attachment=host, principal=(),
                       general_branches=True, local_inventory=True):
        component = row.construction
        if (component is None or row.added_h_locants or not row.hydrogen_budget_proved
                or not row.annotation_coverage or component.atoms | component.embedded_atoms != scope):
            continue
        locant = dict(row.atom_locants)[host]
        ring = frozenset(row.parent_atoms)
        saturated_carbocycle = (any(ring == frozenset(r) for r in mol.GetRingInfo().AtomRings())
            and all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6 and all(
                bond.GetBondTypeAsDouble() == 1 for bond in mol.GetAtomWithIdx(i).GetBonds()
                if bond.GetOtherAtomIdx(i) in ring) for i in ring))
        omit = (locant == '1' and not component.prefixes
                and (row.parent == 'benzene' or saturated_carbocycle))
        yield (), (row.hydride or row.parent) + ('' if omit else '-' + locant + '-'), component


def _polyamide(mol, Chem, sites, matches):
    """Select mixed amides on the source graph or transfer an isovalent homogeneous numbering."""
    from .mapped_principal_general import mapped_principal_decision

    if any(a.GetIsotope() for a in mol.GetAtoms()):
        return None
    units, signatures = [], set()
    for center, host, nitrogen in sites:
        atom = mol.GetAtomWithIdx(center)
        oxo = {o for o, i, _, _ in matches if i == center}
        if (mol.GetAtomWithIdx(host).GetAtomicNum() != 6 or len(oxo) not in {1, 2}
                or atom.GetDegree() != len(oxo) + 2 or atom.GetTotalNumHs()
                or mol.GetAtomWithIdx(nitrogen).GetTotalValence() != 3):
            return None
        unit = frozenset({center, nitrogen, *oxo})
        if any(unit & other for _, other in units):
            return None
        units.append((center, unit))
        signatures.add((atom.GetAtomicNum(), len(oxo)))
    core = frozenset().union(*(atoms for _, atoms in units))
    if not _lower_substituents(mol, core, None):
        return None
    if len(signatures) > 1:
        if any(mol.GetAtomWithIdx(n).GetChiralTag() for _, _, n in sites):
            return None
        mixed = [_oxyamide(mol, Chem, center, host, nitrogen,
                           {o for o, i, _, _ in matches if i == center})
                 for center, host, nitrogen in sites]
        if any(u is None for u in mixed):
            return None
        decision = _mixed_amides(mol, Chem, mixed)
        if decision is not None and decision.method == 'general':
            from .molecule_cache import mol_cache_set
            mol_cache_set(mol, 'configured_mixed_oxyamide_alternative', decision)
        return decision
    if any(a.GetChiralTag() for a in mol.GetAtoms()) or any(b.GetStereo() for b in mol.GetBonds()):
        return None
    element, oxos = signatures.pop()
    graph = mol
    if element != 16 or oxos != 2:
        editable = Chem.RWMol(mol)
        for center, _ in units:
            editable.GetAtomWithIdx(center).SetAtomicNum(16)
            if oxos == 1:
                editable.AddBond(center, editable.AddAtom(Chem.Atom(8)), Chem.BondType.DOUBLE)
        graph = editable.GetMol()
        try:
            Chem.SanitizeMol(graph)
        except (RuntimeError, ValueError):
            return None
    decision = mapped_principal_decision(graph, Chem)
    centers = frozenset(i for i, _ in units)
    if (decision is None or decision.principal_centers != centers
            or decision.atoms != frozenset(range(graph.GetNumAtoms()))
            or not decision.name.endswith('sulfonamide')):
        return None
    if graph is mol:
        return decision
    scope = frozenset(range(mol.GetNumAtoms()))
    mapping = tuple((idx, loc) for idx, loc in decision.atom_locants if idx in scope)
    return MethodDecision(decision.name.removesuffix('sulfonamide') + _SUFFIX[element][oxos - 1],
        'substitutive', centers, scope, 'P-66.1.1.2/P-66.1.1.3.1.1', mapping,
        component_roles=(('hydride-and-substituents', scope - core, ()),
                         *(('oxyamide', atoms, ()) for _, atoms in units)),
        preference_issues=decision.preference_issues)


def chalcogen_amide_decision(mol, Chem):
    """Compose the chosen hydride, the element/oxidation-specific suffix and N branches."""
    matches = mol.GetSubstructMatches(amide_query(Chem))
    sites = {(center, host, nitrogen) for _, center, host, nitrogen in matches}
    if not sites or mol.GetStereoGroups():
        return None
    if len(sites) > 1:
        return _polyamide(mol, Chem, sites, matches)
    center, host, nitrogen = sites.pop()
    atom = mol.GetAtomWithIdx(center)
    oxo = {oxygen for oxygen, *_ in matches}
    if (atom.GetDegree() != len(oxo) + 2 or len(oxo) not in {1, 2}
            or atom.GetTotalNumHs() or any(mol.GetAtomWithIdx(i).GetFormalCharge() for i in oxo)):
        return None
    n_atom = mol.GetAtomWithIdx(nitrogen)
    if (n_atom.GetTotalValence() != 3 or any(b.GetBondTypeAsDouble() != 1 or
            b.GetOtherAtom(n_atom).GetAtomicNum() != 6 for b in n_atom.GetBonds()
            if b.GetOtherAtomIdx(nitrogen) != center)):
        return None
    core = frozenset({center, nitrogen, *oxo})
    if not _lower_substituents(mol, core, host):
        return None
    # Alkyl/aryl groups on the suffix N occupy disjoint, singly attached branches.
    scope = frozenset(collect_subtree(mol, host, center))
    if core & scope:
        return None
    embedded, prefixes, roles = set(core), [], [('oxyamide', core, ())]
    for neighbor in n_atom.GetNeighbors():
        root = neighbor.GetIdx()
        if root == center:
            continue
        name, owned = mapped_general_branch(mol, Chem, root, nitrogen)
        if not name or owned & (embedded | scope):
            return None
        embedded.update(owned)
        prefixes.append(('N', name))
        roles.append(('N-substituent', owned, ()))
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    if scope | embedded != heavy:
        return None
    configurations = ''
    if atom.GetChiralTag():
        assignments = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
        if (len(assignments) != 1 or assignments[0][0] != center
                or assignments[0][1] not in {'R', 'S'}):
            return None
        configurations = '(' + assignments[0][1] + ')-'
    ending = _SUFFIX[atom.GetAtomicNum()][len(oxo) - 1]
    if any(mol.GetAtomWithIdx(i).GetIsotope() for i in core):
        from .component_mapping import MappedComponent, _mapped_isotope_descriptor_plan
        mapping = dict.fromkeys(core, '')
        # The nuclides modify the suffix, whose two oxo positions are equivalent.
        unit = MappedComponent(ending, tuple(mapping.items()), core)
        plan = _mapped_isotope_descriptor_plan(mol, Chem, unit, mapping)
        if plan is None or not plan[0]:
            return None
        ending = '(' + ','.join(plan[0]) + ')' + ending
    options = []
    for key, parent, component in _parent_components(mol, Chem, host, center, scope):
        combined = replace(component, parent=parent + ending,
            prefixes=component.prefixes + tuple(prefixes), embedded_atoms=component.embedded_atoms | embedded,
            nitrogen_site_locants=True, substitution_locant_presentation=None)
        name = decorate_component(mol, Chem, combined)
        if name:
            options.append((key, configurations + name, component.atom_locants))
    if not options:
        return None
    lowest = min(key for key, _, _ in options)
    winners = [(name, mapping) for key, name, mapping in options if key == lowest]
    if len({name for name, _ in winners}) != 1:
        return None
    name, mapping = winners[0]
    rule = 'P-66.1.1.2/P-66.1.1.3.1.1'
    return MethodDecision(name, 'substitutive', frozenset({center}), heavy, rule, mapping,
        component_roles=(('parent-hydride', scope, mapping), *roles),
        preference_issues=(SelectionLimitation('oxyamide-parent', rule,
            'The suffix and N substituents are mapped on the selected carbon chain or ring.'),))
