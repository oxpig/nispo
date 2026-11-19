"""Original-graph P-82 roles, numbering and ownership for P-28 assemblies."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace

from .characteristic_isotope_selection import CharacteristicGroupMap, select_characteristic_isotope_frontier
from .molecule_cache import exact_graph_computation
from .molecule_iteration import iter_atoms


@dataclass(frozen=True, slots=True)
class AssemblyIsotopeSite:
    group: object
    role_map: CharacteristicGroupMap


@dataclass(frozen=True, slots=True)
class AssemblyIsotopeContext:
    components: tuple
    parent_atoms: frozenset[int]
    source_groups: tuple
    junctions: tuple = ()
    sites: tuple = ()
    selection: object | None = None
    selected_groups: tuple = ()
    ordinary_maps: tuple = ()
    chemical_key: tuple = ()
    isotope_census: tuple = ()
    issue: str = ''


@dataclass(frozen=True, slots=True)
class AssemblySuffixIsotopes:
    suffix: str
    local_atoms: frozenset[int] = frozenset()
    amide_evidence: object | None = None


@dataclass(frozen=True, slots=True)
class AssemblyIsotopeEvidence:
    context: AssemblyIsotopeContext
    parent_locants: tuple
    component_locants: tuple
    isotope_key: tuple
    suffix: AssemblySuffixIsotopes
    parent_claims: tuple
    # Actual root/host, whole owned subtree, actual text, unmodified source
    # word and original isotope atoms. No token search establishes ownership.
    branch_claims: tuple
    prefix_citation_order: tuple

    @property
    def covered_isotope_atoms(self):
        return (self.suffix.local_atoms | frozenset(row[0] for row in self.parent_claims)
                | frozenset(i for row in self.branch_claims for i in row[5]))


def _site(mol, group):
    if group.kind == 'amide':
        from .amide_isotopes import amide_isotope_site
        site = amide_isotope_site(mol, group)
        return None if site is None else AssemblyIsotopeSite(group, site.role_map)
    carbon = mol.GetAtomWithIdx(group.center)
    if (carbon.GetAtomicNum() != 6 or carbon.IsInRing() or carbon.GetFormalCharge()
            or carbon.GetNumRadicalElectrons() or group.owner < 0
            or mol.GetBondBetweenAtoms(group.center, group.owner) is None
            or mol.GetBondBetweenAtoms(group.center, group.owner).GetBondTypeAsDouble() != 1):
        return None
    if group.kind == 'acid':
        from .oxygen_characteristic_suffixes import oxygen_group
        oxygen = oxygen_group(mol, group)
        if oxygen is None:
            return None
        roles = (('carbon', (group.center,)),) + tuple((role, (idx,)) for role, idx in oxygen.roles)
        roles += (('hydroxyl_hydrogen', oxygen.attached_hydrogens),)
        atoms = frozenset(i for _role, ids in roles for i in ids)
        if atoms - set(oxygen.attached_hydrogens) != frozenset(group.atoms):
            return None
    elif group.kind == 'nitrile':
        nitrogen = tuple(a.GetIdx() for a in carbon.GetNeighbors() if a.GetAtomicNum() == 7
            and mol.GetBondBetweenAtoms(group.center, a.GetIdx()).GetBondTypeAsDouble() == 3)
        if (len(nitrogen) != 1 or carbon.GetTotalValence() != 4
                or mol.GetAtomWithIdx(nitrogen[0]).GetDegree() != 1
                or set(group.atoms) != {group.center, nitrogen[0]}):
            return None
        roles = (('carbon', (group.center,)), ('nitrogen', nitrogen))
        atoms = frozenset(group.atoms)
    else:
        return None
    return AssemblyIsotopeSite(group, CharacteristicGroupMap(group.kind+':'+str(group.center), atoms, roles))


def _unmodified_graph(mol, Chem):
    def compute():
        scratch = Chem.Mol(mol)
        for atom in scratch.GetAtoms():
            atom.SetIntProp('_assembly_original_id', atom.GetIdx())
            atom.SetIsotope(0)
        Chem.RemoveStereochemistry(scratch)
        scratch = Chem.RemoveHs(scratch)
        original = tuple(a.GetIntProp('_assembly_original_id') for a in scratch.GetAtoms())
        return scratch.ToBinary(Chem.PropertyPickleOptions.AllProps), original
    binary, original = exact_graph_computation(mol, Chem, 'assembly_unmodified_graph', (), compute)
    return Chem.Mol(binary), original


def prepare_assembly_isotopes(mol, Chem, inventory, groups, proof):
    """Recognize complete original sites and retain all ordinary chemical ties."""
    groups = tuple(groups)
    # Ring-info enumeration can reverse an otherwise identical assembly path
    # during mapped child construction. A partition is unordered evidence;
    # actual occurrence numbering remains with the complete P-28 frontier.
    components = tuple(sorted(proof.components, key=lambda core:tuple(sorted(core))))
    junctions = tuple(sorted((min(a,b),max(a,b),int(mol.GetBondBetweenAtoms(a,b).GetBondTypeAsDouble()))
                             for a,b in proof.connectors))
    def compute():
        context = AssemblyIsotopeContext(components, proof.parent_atoms, groups, junctions=junctions,
            isotope_census=tuple((a.GetIdx(), a.GetAtomicNum(), a.GetIsotope())
                                 for a in iter_atoms(mol) if a.GetIsotope()))
        sites = tuple(_site(mol, group) for group in groups)
        if any(site is None for site in sites):
            return replace(context, issue='An assembly characteristic group lacks complete original isotope roles.')
        selection = None
        selected = ()
        if sites:
            scope = frozenset().union(*(site.role_map.atoms for site in sites))
            selection = select_characteristic_isotope_frontier(mol,
                tuple(site.role_map for site in sites), scope)
            if selection is None or selection.barrier or len(selection.principal_frontier) != 1:
                return replace(context, sites=sites, selection=selection,
                    issue='The assembly characteristic-isotope profile comparison remains unresolved.')
            members = selection.multiplicity_groups[selection.principal_frontier[0]]
            selected = tuple(groups[i] for i in members)
        context = replace(context, sites=sites, selection=selection, selected_groups=selected)
        if any(site.group.kind == 'acid' and any(mol.GetAtomWithIdx(i).GetAtomicNum() == 8
                and mol.GetAtomWithIdx(i).GetIsotope() for i in site.role_map.atoms) for site in sites):
            return replace(context, issue='Systematic assembly acid oxygen-role descriptor grammar remains unproved.')
        scratch, original = _unmodified_graph(mol, Chem)
        from .functional_ring_assemblies import functional_assembly_outcome
        ordinary = functional_assembly_outcome(scratch, Chem)
        if ordinary.status != 'complete' or ordinary.proof is None:
            return replace(context, issue='The complete unmodified assembly chemical frontier is unavailable.')
        if {frozenset(original[i] for i in core) for core in ordinary.proof.components} != set(proof.components):
            return replace(context, issue='Removing isotope H did not preserve the original assembly partition.')
        key = min(row.chemical_key for row in ordinary.candidates)
        maps = tuple(sorted({tuple(sorted((original[i], loc) for i, loc in row.atom_locants
                                         if original[i] in proof.parent_atoms))
            for row in ordinary.candidates if row.chemical_key == key}))
        if not maps or any(frozenset(i for i, _loc in row) != proof.parent_atoms for row in maps):
            return replace(context, issue='An ordinary chemical tie lacks the complete original parent map.')
        return replace(context, ordinary_maps=maps, chemical_key=key)
    return exact_graph_computation(mol, Chem, 'assembly_isotope_context',
        (components, proof.parent_atoms, groups, junctions), compute)


def suffix_isotopes(mol, context):
    """P-82.2/.6: local unnumbered characteristic atoms, or mapped parent word."""
    groups = context.selected_groups
    if not groups:
        return AssemblySuffixIsotopes('')
    from .formatting import multiplied_suffix
    from .constants import MULT
    kind = groups[0].kind
    base = {'acid':'carboxylic acid', 'amide':'carboxamide', 'nitrile':'carbonitrile'}[kind]
    ordinary = multiplied_suffix(len(groups), base)
    if not ordinary:
        return None
    if kind == 'amide':
        from .amide_isotopes import amide_suffix_isotopes
        value = amide_suffix_isotopes(mol, groups, ordinary, external=True)
        return None if value is None else AssemblySuffixIsotopes(value[0], value[1], value[2])
    sites = tuple(site for site in context.sites if site.group in groups)
    modified = frozenset(i for site in sites for i in site.role_map.atoms if mol.GetAtomWithIdx(i).GetIsotope())
    local = bool(modified)
    if kind == 'nitrile' and not any(mol.GetAtomWithIdx(g.center).GetIsotope() for g in groups):
        nitrogen = tuple(a.GetIsotope() for a in iter_atoms(mol) if a.GetAtomicNum() == 7)
        # P-82.6.1.2/.3 allows whole-word omission only for one N or complete
        # uniform N modification. No invented external cyano-N locant emits.
        local = bool(modified) and not (len(nitrogen) == 1 or nitrogen and len(set(nitrogen)) == 1 and nitrogen[0])
    if not local:
        return AssemblySuffixIsotopes(ordinary)
    descriptor = _local_descriptor(mol, sites[0])
    if not descriptor or any(_local_descriptor(mol, site) != descriptor for site in sites):
        return None
    unit = descriptor+base
    if len(groups) > 1:
        multiplier = MULT.get(len(groups))
        if not multiplier:
            return None
        unit = multiplier+'['+unit+']'
    return AssemblySuffixIsotopes(unit, modified)


def _local_descriptor(mol, site):
    labels = Counter((mol.GetAtomWithIdx(i).GetSymbol(), mol.GetAtomWithIdx(i).GetIsotope())
                     for i in site.role_map.atoms if mol.GetAtomWithIdx(i).GetIsotope())
    return '('+','.join(str(mass)+symbol+(str(count) if count > 1 or
        symbol == 'H' and site.group.kind == 'amide' else '')
        for (symbol, mass), count in sorted(labels.items()))+')' if labels else ''


def assembly_isotope_preparation(mol, Chem, inventory, context):
    """Join local characteristic ownership to the shared physical branch plan."""
    from .ring_state_parent import _build_ring_preparation
    scratch = Chem.Mol(mol)
    for atom in scratch.GetAtoms():
        atom.SetIsotope(0)
    prepared = _build_ring_preparation(scratch, Chem, context.parent_atoms,
        inventory.groups, context.selected_groups, None, (), ())
    if prepared is None:
        return None
    aliases = list(prepared.aliases)
    mapped = {target for target, _host, _prefix in aliases}
    selected_atoms = set()
    for site in context.sites:
        group = site.group
        if group not in context.selected_groups:
            continue
        selected_atoms.update(site.role_map.atoms)
        for idx in group.atoms:
            if idx in mapped or group.kind == 'amide' and idx == group.ligand:
                continue
            number = mol.GetAtomWithIdx(idx).GetAtomicNum()
            # These characteristic aliases are internal bindings only. Local
            # descriptors own their modified atoms. The sole nonlocal cyano N
            # operation above proves complete uniformity before omitting it.
            prefix = 'C' if number == 6 else 'O' if number == 8 else 'N'
            aliases.append((idx, group.owner, prefix));mapped.add(idx)
    return replace(prepared, aliases=tuple(aliases),
                   suffix_atoms=prepared.suffix_atoms-selected_atoms)


def isotope_bindings(mol, Chem, context, parent_mapping):
    from .component_mapping import hydrogen_bindings
    from .extended_inventory import selection_inventory
    mapping = dict(parent_mapping)
    prepared = assembly_isotope_preparation(mol, Chem, selection_inventory(mol), context)
    if prepared is None:
        return None
    if context.selected_groups and context.selected_groups[0].kind == 'amide':
        from .amide_site_locants import external_amide_nitrogen_locants
        result = external_amide_nitrogen_locants(mol, Chem, context.selected_groups, mapping,
                                               locant_series=assembly_locant_series)
        if result is None:
            return None
        mapping.update(result.nitrogen_locants)
    for target, host, prefix in prepared.aliases:
        mapping[target] = prefix+mapping[host]
    return hydrogen_bindings(mol, mapping)


def _characteristic_prefix(mol, Chem, site, root, host, atoms):
    group = site.group
    if group.center != root or group.owner != host:
        return None
    if group.kind == 'amide':
        from .amide_isotopes import mapped_amide_prefix
        result = mapped_amide_prefix(mol, Chem, root, host, atoms)
        if result is not None:
            return result
        from .component_mapping import mapped_branch
        return mapped_branch(mol, Chem, root, host)
    if site.role_map.atoms != atoms:
        return '', frozenset()
    # One complete local characteristic unit, with graph-proved C/N/O/H roles.
    return _local_descriptor(mol, site)+{'acid':'carboxy', 'nitrile':'cyano'}[group.kind], atoms


def assembly_isotope_branch(mol, Chem, context, root, host):
    from .branch_rules import collect_subtree
    from .component_mapping import mapped_branch
    atoms = frozenset(collect_subtree(mol, root, host))
    for site in context.sites:
        if site.group not in context.selected_groups and site.group.center == root and site.group.owner == host:
            return _characteristic_prefix(mol, Chem, site, root, host, atoms)
    return mapped_branch(mol, Chem, root, host)


def assembly_isotope_branches(mol, Chem, context, seq, nmap, mapping, needed, prepared):
    from .component_mapping import hydrogen_bindings
    from .cross_route_selection import _annotation_ownership
    prefixes, ownership, embedded = [], [], set(prepared.suffix_atoms)
    for host in (*seq, *nmap):
        for root, allowed in prepared.branch_edges[host]:
            if root in embedded:
                continue
            if not allowed:
                return None
            value = assembly_isotope_branch(mol, Chem, context, root, host)
            if value is None:
                return None
            text, atoms = value
            if not text or any(c.isspace() for c in text) or atoms & (context.parent_atoms | embedded):
                return None
            prefixes.append((mapping[host], text));embedded.update(atoms)
            ownership.append((root, host, frozenset(atoms), text))
    bindings = hydrogen_bindings(mol, mapping)
    if set(bindings) | embedded != needed or not _annotation_ownership(mol, bindings):
        return None
    return tuple(prefixes), frozenset(embedded), tuple(ownership), bindings


def _ordinary_branch_word(mol, Chem, root, host):
    def compute():
        from .component_mapping import mapped_branch
        scratch, original = _unmodified_graph(mol, Chem)
        translated = {old:new for new,old in enumerate(original)}
        if root not in translated or host not in translated:
            return ''
        return mapped_branch(scratch, Chem, translated[root], translated[host])[0]
    return exact_graph_computation(mol, Chem, 'assembly_ordinary_branch_word', (root, host), compute)


def assembly_locant_series(values):
    from .locant_order import writing_locant_key
    keys = tuple(writing_locant_key(value) for value in values)
    return None if any(key is None for key in keys) else tuple(sorted(keys))


def build_assembly_isotope_evidence(mol, Chem, context, component, parent_mapping):
    from .component_mapping import _mapped_isotope_descriptor_plan
    from .formatting import prefix_alphabetization_key
    from .parent_annotations import isotope_numbering_key
    from .component_mapping import hydrogen_bindings
    parent = tuple(sorted(parent_mapping.items()))
    if context.issue or parent not in context.ordinary_maps:
        return None
    mapping = dict(component.atom_locants)
    expected = isotope_bindings(mol, Chem, context, parent_mapping)
    suffix = suffix_isotopes(mol, context)
    if (expected is None or mapping != expected or len(mapping) != len(component.atom_locants)
            or set(mapping) != set(component.atoms) or suffix is None
            or component.local_isotope_atoms != suffix.local_atoms
            or component.amide_isotope_evidence != suffix.amide_evidence
            or suffix.suffix and not component.parent.endswith(suffix.suffix)):
        return None
    plan = _mapped_isotope_descriptor_plan(mol, Chem, component, mapping,
                                           skeletal_atoms=context.parent_atoms)
    if plan is None:
        return None
    branches, owned, prefixes, words = [], set(), [], {}
    for root, host, atoms, text in component.prefix_branch_ownership:
        actual = assembly_isotope_branch(mol, Chem, context, root, host)
        if actual is None or actual != (text, atoms) or atoms & (component.atoms | owned):
            return None
        base = _ordinary_branch_word(mol, Chem, root, host)
        if not base:
            return None
        isotope_atoms = frozenset(i for i in atoms if mol.GetAtomWithIdx(i).GetIsotope())
        word = (base, bool(isotope_atoms))
        if text in words and words[text] != word:
            return None
        words[text] = word
        prefixes.append((mapping[host], text));owned.update(atoms)
        branches.append((root, host, atoms, text, base, isotope_atoms))
    if Counter(prefixes) != Counter(component.prefixes) or owned != set(component.embedded_atoms):
        return None
    # P-82.2.2.1 proves modified before natural copies of the same chemical
    # prefix. A mutual ordering of unlike modified profiles needs more proof.
    for base in {word[0] for word in words.values()}:
        if len({text for text, word in words.items() if word == (base, True)}) > 1:
            return None
    order = tuple(sorted(words, key=lambda text:(prefix_alphabetization_key(words[text][0])[:3],
                                               -int(words[text][1]))))
    if component.prefix_citation_order is not None and component.prefix_citation_order != order:
        return None
    key = isotope_numbering_key(mol, hydrogen_bindings(mol, dict(parent_mapping)),
                               include_substituent_groups=True, locant_series=assembly_locant_series)
    if key is None:
        return None
    result = AssemblyIsotopeEvidence(context, parent, tuple(sorted(mapping.items())), key,
                                    suffix, plan[1], tuple(branches), order)
    claims = [row[0] for row in result.parent_claims] + list(suffix.local_atoms)
    claims += [i for row in branches for i in row[5]]
    required = {i for i, _z, _mass in context.isotope_census}
    if len(claims) != len(set(claims)) or set(claims) != required:
        return None
    return result


def validate_assembly_isotope_evidence(mol, Chem, component):
    """Reconstruct once per complete component and exact native graph state.

    The full immutable component includes the actual writer fields and every
    supplied proof. Changed evidence cannot share a hit with the original.
    Unkeyable components retain the original full validation path.
    """
    return exact_graph_computation(mol, Chem, 'assembly_isotope_validation',
        (component,), lambda: _validate_assembly_isotope_evidence(mol, Chem, component))


def _validate_assembly_isotope_evidence(mol, Chem, component):
    evidence = component.assembly_isotope_evidence
    if not isinstance(evidence, AssemblyIsotopeEvidence):
        return False
    from .functional_ring_assemblies import _scope
    scope = _scope(mol, Chem)
    if scope is None:
        return False
    inventory, groups, proof, _phane = scope
    context = prepare_assembly_isotopes(mol, Chem, inventory, groups, proof)
    if context != evidence.context:
        return False
    expected = build_assembly_isotope_evidence(mol, Chem, context, component,
                                             dict(evidence.parent_locants))
    return expected is not None and expected == evidence and component.prefix_citation_order == evidence.prefix_citation_order
