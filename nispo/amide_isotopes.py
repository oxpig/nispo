"""Original ordinary-amide isotope roles and P-82.2.2.2 integration.

Class recognition, isotope-profile selection and component rendering remain
separate. No isotope-cleared molecule, generated-name rewrite or atom-index
tie-break selects the principal suffix.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .characteristic_isotope_selection import (
    CharacteristicGroupMap, select_characteristic_isotope_frontier,
)
from .molecule_cache import mol_cache_dict, naming_cache_state

RULE = 'P-82.2.1/P-82.2.2.2/P-82.2.5/P-82.6.3.2/P-66.1.1'


@dataclass(frozen=True, slots=True)
class AmideIsotopeSite:
    group: object
    oxygen: int
    isotope_hydrogens: tuple[int, ...]
    role_map: CharacteristicGroupMap


@dataclass(frozen=True, slots=True)
class AmideIsotopePartition:
    sites: tuple[AmideIsotopeSite, ...]
    selection: object
    selected_groups: tuple
    modified_atoms: frozenset[int]

    @property
    def active(self):
        return bool(self.modified_atoms)


@dataclass(frozen=True, slots=True)
class AmideIsotopeEvidence:
    sites: tuple[AmideIsotopeSite, ...]
    placement: str
    owned_isotope_atoms: frozenset[int]
    suffix_text: str = ''


@dataclass(frozen=True, slots=True)
class OrdinaryAmideNumberingProof:
    status: str
    parent_atoms: frozenset[int]
    equivalent_maps: tuple = ()
    reason: str = ''


def amide_isotope_site(mol, group):
    if group.kind != 'amide':
        return None
    carbon, nitrogen = mol.GetAtomWithIdx(group.center), mol.GetAtomWithIdx(group.ligand)
    if (carbon.GetAtomicNum() != 6 or carbon.IsInRing() or carbon.GetTotalValence() != 4
            or nitrogen.GetAtomicNum() != 7 or nitrogen.IsInRing() or nitrogen.GetTotalValence() != 3
            or any(atom.GetFormalCharge() or atom.GetNumRadicalElectrons() for atom in (carbon, nitrogen))):
        return None
    oxygen = [atom for atom in carbon.GetNeighbors() if atom.GetAtomicNum() == 8
              and mol.GetBondBetweenAtoms(group.center, atom.GetIdx()).GetBondTypeAsDouble() == 2]
    if (len(oxygen) != 1 or oxygen[0].GetDegree() != 1 or oxygen[0].GetFormalCharge()
            or oxygen[0].GetNumRadicalElectrons()
            or set(group.atoms) != {group.center, group.ligand, oxygen[0].GetIdx()}
            or mol.GetBondBetweenAtoms(group.center, group.ligand) is None
            or mol.GetBondBetweenAtoms(group.center, group.ligand).GetBondTypeAsDouble() != 1):
        return None
    heavy = {atom.GetIdx() for atom in carbon.GetNeighbors() if atom.GetAtomicNum() > 1}
    expected = {group.ligand, oxygen[0].GetIdx()}
    if group.owner >= 0:
        if (mol.GetAtomWithIdx(group.owner).GetAtomicNum() != 6
                or mol.GetBondBetweenAtoms(group.center, group.owner) is None
                or mol.GetBondBetweenAtoms(group.center, group.owner).GetBondTypeAsDouble() != 1):
            return None
        expected.add(group.owner)
    if heavy != expected or any(b.GetBondTypeAsDouble() != 1 for b in nitrogen.GetBonds()):
        return None
    hydrogens = tuple(atom.GetIdx() for atom in nitrogen.GetNeighbors()
                      if atom.GetAtomicNum() == 1 and atom.GetIsotope())
    for idx in hydrogens:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetDegree() != 1 or atom.GetFormalCharge() or atom.GetNumRadicalElectrons():
            return None
    roles = (('carbonyl_carbon', (group.center,)), ('carbonyl_oxygen', (oxygen[0].GetIdx(),)),
             ('amide_nitrogen', (group.ligand,)), ('nitrogen_hydrogen', hydrogens))
    atoms = frozenset(idx for _role, indices in roles for idx in indices)
    return AmideIsotopeSite(group, oxygen[0].GetIdx(), hydrogens,
        CharacteristicGroupMap('amide:' + str(group.center), atoms, roles))


def amide_isotope_partition(mol, groups=None):
    from .graph_model import graph_view
    if not any(graph_view(mol, None).isotopes):
        return None
    if groups is None:
        from .extended_inventory import selection_inventory
        inventory = selection_inventory(mol)
        if (not inventory.supported or not inventory.principal
                or any(group.kind != 'amide' for group in inventory.principal)):
            return None
        groups = inventory.principal
    groups = tuple(groups)
    key = (groups, naming_cache_state(mol))
    cache = mol_cache_dict(mol, 'amide_isotope_partitions')
    if key in cache:
        return cache[key]
    sites = tuple(amide_isotope_site(mol, group) for group in groups)
    result = None
    if sites and all(site is not None for site in sites):
        scope = frozenset().union(*(site.role_map.atoms for site in sites))
        selection = select_characteristic_isotope_frontier(mol,
            tuple(site.role_map for site in sites), scope)
        if selection is not None:
            selected = ()
            if not selection.barrier and len(selection.principal_frontier) == 1:
                cluster = selection.principal_frontier[0]
                selected = tuple(groups[i] for i in selection.multiplicity_groups[cluster])
            modified = frozenset(i for i in scope if mol.GetAtomWithIdx(i).GetIsotope())
            result = AmideIsotopePartition(sites, selection, selected, modified)
    cache[key] = result
    return result


def source_principal_amides(mol, inventory, groups):
    """Validate the original source class, optionally projected by P-82."""
    source = inventory.source_frontier
    if (source.status not in {'selected', 'tie'} or set(source.groups) != set(inventory.principal)):
        return False
    if set(groups) == set(source.groups):
        return True
    partition = amide_isotope_partition(mol, inventory.principal)
    return bool(partition is not None and partition.active
                and not partition.selection.barrier
                and set(groups) == set(partition.selected_groups))


def amide_parent_annotation_mapping(mol, atoms, mapping):
    """Map the physical parent; external labelled amides are whole groups.

    Their C/O/N role aliases serialize local annotations, not additional
    skeletal parent positions. P-82.5.2 counts a singly attached modified
    external unit at its parent attachment, as the shared branch key does.
    """
    partition = amide_isotope_partition(mol)
    if partition is None or not partition.active or partition.selection.barrier:
        return None
    from .component_mapping import hydrogen_bindings
    return hydrogen_bindings(mol, {idx: mapping[idx] for idx in atoms})


def local_amide_descriptor(mol, site):
    labels = Counter((mol.GetAtomWithIdx(i).GetSymbol(), mol.GetAtomWithIdx(i).GetIsotope())
                     for i in site.role_map.atoms if mol.GetAtomWithIdx(i).GetIsotope())
    return '(' + ','.join(str(mass) + symbol + (str(count) if count > 1 or symbol == 'H' else '')
        for (symbol, mass), count in sorted(labels.items())) + ')' if labels else ''


def amide_suffix_isotopes(mol, groups, ending, *, external, hydride='', location_text=''):
    """Return ending, local ownership and a typed parent/suffix boundary."""
    partition = amide_isotope_partition(mol, groups)
    if partition is None:
        if any(mol.GetAtomWithIdx(i).GetIsotope()
               or any(n.GetAtomicNum() == 1 and n.GetIsotope()
                      for n in mol.GetAtomWithIdx(i).GetNeighbors())
               for group in groups for i in group.atoms):
            return None
        return ending, frozenset(), None
    if not partition.active:
        return ending, frozenset(), None
    if partition.selection.barrier or len(partition.selection.multiplicity_groups) != 1:
        return None
    sites = partition.sites
    local = external and any(mol.GetAtomWithIdx(site.group.center).GetIsotope() for site in sites)
    oxygen_masses = {mol.GetAtomWithIdx(site.oxygen).GetIsotope() for site in sites} - {0}
    if external and oxygen_masses:
        all_oxygen = tuple(a.GetIsotope() for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
        # An unnumbered suffix oxygen cannot borrow an invented O^site
        # locant. Whole-name omission is established only by one oxygen or
        # complete uniform oxygen modification (P-82.6.1.2/.3).
        local = local or not (len(all_oxygen) == 1 or len(oxygen_masses) == 1
                              and all(mass == next(iter(oxygen_masses)) for mass in all_oxygen))
    if local:
        from .constants import MULT
        unit = local_amide_descriptor(mol, sites[0]) + 'carboxamide'
        if len(sites) > 1:
            if len(sites) not in MULT:
                return None
            unit = MULT[len(sites)] + '[' + unit + ']'
        ending = hydride + location_text + unit
        evidence = AmideIsotopeEvidence(sites, 'suffix-unit', partition.modified_atoms, unit)
        return ending, partition.modified_atoms, evidence
    evidence = AmideIsotopeEvidence(sites, 'parent-word', partition.modified_atoms)
    return ending, frozenset(), evidence


def validate_amide_isotope_evidence(mol, component):
    evidence = component.amide_isotope_evidence
    if not isinstance(evidence, AmideIsotopeEvidence) or not evidence.sites:
        return False
    sites = tuple(amide_isotope_site(mol, site.group) for site in evidence.sites)
    if sites != evidence.sites:
        return False
    partition = amide_isotope_partition(mol, tuple(site.group for site in sites))
    if (partition is None or not partition.active or partition.selection.barrier
            or len(partition.selection.multiplicity_groups) != 1
            or partition.modified_atoms != evidence.owned_isotope_atoms
            or not frozenset().union(*(site.role_map.atoms for site in sites)) <= component.atoms):
        return False
    if evidence.placement == 'suffix-unit':
        from .constants import MULT
        unit = local_amide_descriptor(mol, sites[0]) + 'carboxamide'
        if len(sites) > 1:
            unit = MULT.get(len(sites), '') + '[' + unit + ']'
        return (evidence.suffix_text == unit and component.parent.endswith(unit)
                and evidence.owned_isotope_atoms <= component.local_isotope_atoms)
    return (evidence.placement == 'parent-word'
            and not evidence.owned_isotope_atoms & component.local_isotope_atoms)


def mapped_amide_prefix(mol, Chem, root, host, atoms):
    """P-65/P-82 carbamoyl component on the actual directed branch."""
    from .extended_inventory import selection_inventory
    from .component_mapping import MappedComponent, hydrogen_bindings, mapped_branch, decorate_component
    inventory = selection_inventory(mol)
    group = next((g for g in inventory.groups if g.kind == 'amide'
                  and g.center == root and g.owner == host), None)
    if group is None:
        return None
    site = amide_isotope_site(mol, group)
    if site is None:
        return None
    partition = amide_isotope_partition(mol)
    if not any(mol.GetAtomWithIdx(i).GetIsotope() for i in site.role_map.atoms) and not (
            partition is not None and partition.active):
        return None
    mapping = {root: '', site.oxygen: 'O', group.ligand: 'N'}
    embedded, prefixes = set(), []
    for neighbor in mol.GetAtomWithIdx(group.ligand).GetNeighbors():
        if neighbor.GetAtomicNum() == 1 or neighbor.GetIdx() == root:
            continue
        text, owned = mapped_branch(mol, Chem, neighbor.GetIdx(), group.ligand)
        if not text or owned & (set(mapping) | embedded) or host in owned:
            return '', frozenset()
        embedded.update(owned)
        prefixes.append(('N', text))
    bindings = hydrogen_bindings(mol, mapping)
    if set(bindings) | embedded != set(atoms):
        return '', frozenset()
    component = MappedComponent('carbamoyl', tuple(bindings.items()), frozenset(bindings),
        tuple(prefixes), frozenset(embedded), nitrogen_site_locants=True,
        isotope_unlocanted_elements=(6, 7, 8), prefix_mode='amino', systematic_parent=True)
    text = decorate_component(mol, Chem, component)
    return (text, frozenset(atoms)) if text else ('', frozenset())


def mapped_amide_nitrogen_prefix(mol, Chem, root, carbonyl, atoms):
    """P-65/P-82 amino operation when the amide carbon is in the parent.

    An amide N is not reclassified as an ordinary amine. Its original C=O/N
    roles authorize this subordinate amino/oxo construction independently.
    """
    from .extended_inventory import selection_inventory
    from .component_mapping import MappedComponent, mapped_branch, hydrogen_bindings, decorate_component
    inventory = selection_inventory(mol)
    group = next((g for g in inventory.groups if g.kind == 'amide'
                  and g.center == carbonyl and g.ligand == root), None)
    if group is None or amide_isotope_site(mol, group) is None:
        return None
    partition = amide_isotope_partition(mol)
    if not any(mol.GetAtomWithIdx(i).GetIsotope() for i in atoms) and not (
            partition is not None and partition.active):
        return None
    prefixes, embedded = [], set()
    for neighbor in mol.GetAtomWithIdx(root).GetNeighbors():
        if neighbor.GetIdx() == carbonyl or neighbor.GetAtomicNum() == 1:
            continue
        text, owned = mapped_branch(mol, Chem, neighbor.GetIdx(), root)
        if not text or owned & (embedded | {root, carbonyl}):
            return '', frozenset()
        prefixes.append(('', text))
        embedded.update(owned)
    mapping = hydrogen_bindings(mol, {root: 'N'})
    if set(mapping) | embedded != set(atoms):
        return '', frozenset()
    component = MappedComponent('amino', tuple(mapping.items()), frozenset(mapping),
        tuple(prefixes), frozenset(embedded), isotope_unlocanted_elements=(7,),
        prefix_mode='amino', systematic_parent=True)
    text = decorate_component(mol, Chem, component)
    return (text, frozenset(atoms)) if text else ('', frozenset())


def ordinary_amide_numbering_proof(mol, Chem, partition, selected):
    """P-82.5.1 witness from the unmodified analogue's chemical frontier.

    The isotope selector and actual writer use the original graph only. A
    private unmodified copy supplies ordinary numbering witnesses; neither its
    generated spelling nor its atom-index tie order is an output candidate.
    Every chemical tie is retained before comparing the original mapping.
    """
    core = frozenset(selected.parent_atoms)
    all_groups = tuple(site.group for site in partition.sites)
    if set(partition.selected_groups) == set(all_groups):
        return OrdinaryAmideNumberingProof('proved', core,
            reason='The complete ordinary suffix set is unchanged; existing chemical numbering precedes isotope criteria.')
    owners = {group.owner for group in all_groups}
    if len(owners) == 1 and next(iter(owners)) in core:
        return OrdinaryAmideNumberingProof('proved', core,
            reason='Every ordinary suffix has the same original host; reducing its multiplicity cannot change the chemical attachment-locant order.')
    key = (all_groups, core, naming_cache_state(mol))
    cache = mol_cache_dict(mol, 'ordinary_amide_numbering_witnesses')
    if key not in cache:
        scratch = Chem.Mol(mol)
        for atom in scratch.GetAtoms():
            atom.SetIsotope(0)
        Chem.SanitizeMol(scratch)
        from .extended_inventory import selection_inventory
        inventory = selection_inventory(scratch)
        rows = []
        if any(mol.GetAtomWithIdx(i).IsInRing() for i in core):
            from .mapped_ring_parents import skeletons
            from .ring_state_parent import _render, _reference_states
            from .naming_context import MoleculeContext
            from types import SimpleNamespace
            groups = tuple(g for g in inventory.principal if g.owner in core)
            local = SimpleNamespace(groups=inventory.groups, principal=groups)
            ctx = MoleculeContext(scratch, Chem)
            for skeleton in skeletons(scratch, Chem, core):
                refs = (None,) if skeleton[5] or len(skeleton) > 8 else _reference_states(skeleton[3], skeleton[4])
                for reference in refs:
                    rows.extend(_render(ctx, skeleton, reference, local, principal=groups))
            if rows and len({row.parent for row in rows}) == 1:
                chemical = min(row.numbering_key[:-2] for row in rows)
                rows = [row for row in rows if row.numbering_key[:-2] == chemical]
            else:
                rows = []
        else:
            from .complete_functional_parent import functional_parent_frontier
            frontier = functional_parent_frontier(scratch, Chem)
            if frontier.status in {'complete', 'tie'}:
                rows = [row for row in frontier.candidates if frozenset(row.parent_atoms) == core]
        cache[key] = tuple(tuple((idx, str(dict(row.atom_locants)[idx])) for idx in sorted(core)) for row in rows)
    maps = cache[key]
    if maps:
        from .component_mapping import hydrogen_bindings
        from .parent_annotations import isotope_numbering_key
        ranked = [(isotope_numbering_key(mol, hydrogen_bindings(mol, dict(row)),
                    include_substituent_groups=True), row) for row in maps]
        if any(rank is None for rank, _row in ranked):
            return OrdinaryAmideNumberingProof('pending', core, maps,
                'A source-tied ordinary parent lacks a complete physical isotope-numbering map.')
        best = min(rank for rank, _row in ranked)
        maps = tuple(row for rank, row in ranked if rank == best)
    actual = tuple((idx, str(dict(selected.atom_locants)[idx])) for idx in sorted(core))
    if actual in maps:
        return OrdinaryAmideNumberingProof('proved', core, maps,
            'The original mapping is one of the complete unmodified chemical-numbering ties.')
    return OrdinaryAmideNumberingProof('pending', core, maps,
        'The changed suffix/parent construction has not retained a proved ordinary-numbering witness under P-82.5.1.')


def amide_isotope_outcome(mol, Chem):
    from .naming_methods import MethodDecision, NamingMethodOutcome, NamingMethodStatus
    partition = amide_isotope_partition(mol)
    if partition is None:
        from .extended_inventory import selection_inventory
        inventory = selection_inventory(mol)
        if (inventory.principal and all(g.kind == 'amide' for g in inventory.principal)
                and any(mol.GetAtomWithIdx(i).GetIsotope()
                    or any(n.GetAtomicNum() == 1 and n.GetIsotope()
                           for n in mol.GetAtomWithIdx(i).GetNeighbors())
                    for group in inventory.principal for i in group.atoms)):
            return NamingMethodOutcome(NamingMethodStatus.PENDING, rule=RULE,
                method='isotope-amide', reason='The recognized amide-core labels lack disjoint complete characteristic-group role maps.')
    if partition is None or not partition.active:
        return NamingMethodOutcome(NamingMethodStatus.OUTSIDE_SCOPE, reason='No mapped amide-core isotope operation.')
    def pending(reason, decision=None):
        return NamingMethodOutcome(NamingMethodStatus.PENDING, decision,
            rule=RULE, method='isotope-amide', reason=reason,
            evidence=(('amide_isotope_partition', partition),))
    if partition.selection.barrier or not partition.selected_groups:
        return pending('The characteristic-isotope profile comparison remains unresolved.')
    from .cross_route_selection import cross_route_selection_outcome
    outcome = cross_route_selection_outcome(mol, Chem)
    if outcome.status != 'complete' or outcome.selection is None:
        return pending('The isotope-selected amide parent has no complete mapped shared-pool result.')
    selected = outcome.selection.selected
    from .component_mapping import fully_covered
    component = selected.construction
    if component is None or not fully_covered(mol, component.atoms, component.embedded_atoms):
        return pending('The isotope-selected amide has incomplete original atom ownership.')
    centers = frozenset(group.center for group in partition.selected_groups)
    decision = MethodDecision(selected.name, 'substitutive', selected.principal_centers,
        component.atoms | component.embedded_atoms, RULE, selected.atom_locants,
        component_roles=(('amide_isotope_partition', partition), ('mapped_selection', outcome.selection)))
    if selected.principal_centers != centers:
        return pending('Partial suffix coverage retains the P-45/P-51 senior-parent multiplication comparison.', decision)
    numbering = ordinary_amide_numbering_proof(mol, Chem, partition, selected)
    from dataclasses import replace
    decision = replace(decision, component_roles=(*decision.component_roles, ('ordinary_numbering_proof', numbering)))
    if numbering.status != 'proved':
        return pending(numbering.reason, decision)
    return NamingMethodOutcome(NamingMethodStatus.COMPLETE, decision, rule=RULE,
        method='isotope-amide', reason='Complete isotope-selected amide suffix/prefix construction.',
        evidence=(('amide_isotope_partition', partition), ('mapped_selection', outcome.selection)))
