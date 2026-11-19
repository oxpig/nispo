"""Immutable evidence between source-qualified candidate search and delivery.

Completeness is scoped to an enumerator. Successful rendering or structural
recovery alone cannot establish it. Original-atom witnesses stay internal;
public summaries contain no atom IDs.
"""
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CandidateCompleteness:
    scope: str
    rules: tuple[str, ...] = ()
    expected: tuple = ()
    accounted: tuple = ()
    reason: str = ""
    proposal_keys: tuple = ()

    @property
    def missing(self):
        accounted = set(self.accounted)
        return tuple(key for key in dict.fromkeys(self.expected) if key not in accounted)

    @property
    def complete(self):
        return bool(self.scope and self.rules and self.expected and self.proposal_keys and not self.missing)


def proposal_identity(proposal):
    """Bind a call-local certificate to its immutable candidate/fact records.

    Fact identities are internal and retain their owning records in the source
    frontier. They are never serialized as part of public justification.
    """
    return (getattr(proposal, 'name', ''), tuple(proposal.parent_atoms),
            frozenset(proposal.principal_centers), tuple(getattr(proposal, 'atom_locants', ())),
            id(proposal.parent_selection))


@dataclass(frozen=True, slots=True)
class SelectionStep:
    stage: str
    rule: str
    status: str
    reason: str


@dataclass(frozen=True, slots=True)
class PreferredNameJustification:
    """A supported-scope proof, separate from availability and a PIN audit."""

    scope: str
    candidate_count: int
    expected_parent_count: int
    accounted_parent_count: int
    steps: tuple[SelectionStep, ...] = ()
    comparisons: tuple = ()
    name: str = ""

    @property
    def complete(self):
        return bool(self.name and self.scope and self.steps and self.comparisons and self.expected_parent_count
                    and self.expected_parent_count == self.accounted_parent_count
                    and all(step.status in {'proved', 'not-applicable'} for step in self.steps))


def parent_justification(completeness, candidates, selected, *, method_rule='',
                         method_proved=False, reason=''):
    """Project graph/search/comparison evidence; never infer method eligibility."""
    from .parent_selection import compare_parents

    candidates = tuple(candidates)
    complete = completeness is not None and completeness.complete
    steps = [
        SelectionStep('principal-group', 'P-41/P-43/P-44.1.1',
                      'proved' if complete else 'pending',
                      'Original principal sites are retained by the source-qualified parent frontier.'
                      if complete else 'Complete source-qualified principal-site coverage is unproved.'),
        SelectionStep('method', method_rule or 'P-51/P-45.1',
                      'proved' if method_proved else 'pending',
                      'Method eligibility and competing-method exclusions are proved in the declared scope.'
                      if method_proved else 'Competing naming methods have not all been excluded or compared.'),
        SelectionStep('candidate-completeness', 'P-44/P-45/P-51',
                      'proved' if complete else 'pending',
                      completeness.reason if completeness is not None else
                      'No source-qualified complete candidate universe was supplied.'),
    ]
    comparisons = ()
    if selected is not None:
        facts = selected.parent_selection
        # One rendering still needs its own applicable numbering facts.
        rows = [compare_parents(facts, row.parent_selection, trace=True)
                for row in candidates]
        rows.append(compare_parents(facts, facts, trace=True))
        unique = {repr(row): row for row in rows}
        comparisons = tuple(unique[key] for key in sorted(unique))
    comparison_complete = bool(comparisons) and all(
        row.preference in {-1, 0} for row in comparisons)
    for stage, rule, text in (
        ('parent-and-numbering', 'P-44/P-45.0/P-45.2', 'Constitutional parent and numbering criteria'),
        ('isotope', 'P-44.4.1.11/P-45.4/P-82.5', 'Parent isotope selection and equivalent numbering'),
        ('stereochemistry', 'P-44.4.1.12/P-45.6/P-92', 'Mapped configuration and configuration numbering'),
    ):
        steps.append(SelectionStep(
            stage, rule, 'proved' if comparison_complete else 'pending',
            text + (' are accounted for by the ordered mapped comparison.'
                    if comparison_complete else ' remain unproved.')))
    if reason:
        steps.append(SelectionStep('selection', 'P-44/P-45', 'pending', reason))
    expected = len(set(completeness.expected)) if completeness is not None else 0
    accounted = (len(set(completeness.expected) & set(completeness.accounted))
                 if completeness is not None else 0)
    return PreferredNameJustification(
        completeness.scope if completeness is not None else '', len(candidates),
        expected, accounted, tuple(steps), comparisons, getattr(selected, 'name', ''))


def carbon_tree_method_proved(mol, groups, selected):
    """Exclude replacement/phane/multiplication for a complete carbon tree.

    A connected carbon tree has no skeletal heteroatom link. Two disjoint
    copies of its selected senior functional parent need disjoint suffix
    sites AND skeletal atoms. Either resource bound can exclude repetition.
    Class-specific functional-class exceptions stay outside this proof.
    """
    from .molecule_iteration import iter_atoms
    groups = tuple(groups)
    if (selected is None or not groups or any(group.kind not in {
            'acid', 'amide', 'nitrile', 'ketone', 'alcohol', 'thiol', 'amine'
            } for group in groups)):
        return False
    atoms = tuple(iter_atoms(mol))
    if any(atom.IsInRing() or atom.GetFormalCharge() or atom.GetNumRadicalElectrons()
           for atom in atoms):
        return False
    carbon = {atom.GetIdx() for atom in atoms if atom.GetAtomicNum() == 6}
    if not carbon or not frozenset(selected.parent_atoms) <= carbon:
        return False
    adjacency = {idx: {n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                       if n.GetIdx() in carbon} for idx in carbon}
    seen, stack = set(), [next(iter(carbon))]
    while stack:
        idx = stack.pop()
        if idx not in seen:
            seen.add(idx)
            stack.extend(adjacency[idx] - seen)
    if seen != carbon or sum(map(len, adjacency.values())) != 2 * (len(carbon) - 1):
        return False
    sites = {group.center for group in groups}
    selected_sites = set(selected.principal_centers)
    if not selected_sites <= sites:
        return False
    # Distinct characteristic groups can share one carbon host.  A center
    # count is only a site count when every group has a different center;
    # otherwise it can both miss and manufacture a copy-exclusion proof.
    # The joined principal-site frontier retains each original OH identity,
    # including geminal groups, and can establish the stronger bound in its
    # independently validated, saturated carbon/OH scope.
    if len(sites) != len(groups):
        actual_sites = _ordinary_alcohol_site_copy_bound(mol, groups, selected)
        return ((actual_sites is not None and 2 * actual_sites[0] > actual_sites[1])
                or 2 * len(set(selected.parent_atoms)) > len(carbon))
    return (2 * len(selected_sites) > len(sites)
            or 2 * len(set(selected.parent_atoms)) > len(carbon))


def _ordinary_alcohol_site_copy_bound(mol, groups, selected):
    """Count original OH sites, never carbon hosts, in a closed tree scope.

    The proposal must carry the full source-qualified selected/omitted site
    partition, exact source groups, and suffix multiplicity.  A source site
    contributes only when its carbon host belongs to the selected parent.
    No name, renderer success, atom index ordering, or unproved feature is
    used to establish this bound.  Other chemistry retains its existing
    method obligation.
    """
    from collections import Counter
    from .molecule_iteration import iter_atoms, iter_bonds

    selected_rows = tuple(getattr(selected, 'selected_site_evidence', ()))
    omitted_rows = tuple(getattr(selected, 'omitted_site_evidence', ()))
    rows = selected_rows + omitted_rows
    if (not selected_rows or len(rows) != len(groups)
            or not all(getattr(group, 'kind', None) == 'alcohol' for group in groups)
            or getattr(getattr(selected, 'parent_selection', None), 'suffix_count', None) != len(selected_rows)):
        return None
    atoms = tuple(iter_atoms(mol))
    if (mol.GetStereoGroups()
            or any(atom.GetAtomicNum() not in {1, 6, 8} or atom.GetIsotope()
                   or int(atom.GetChiralTag()) for atom in atoms)
            or any(bond.GetBondTypeAsDouble() != 1 or int(bond.GetStereo())
                   or int(bond.GetBondDir()) for bond in iter_bonds(mol))):
        return None

    def identity(group):
        return (group.kind, group.center, tuple(sorted(group.atoms)),
                group.owner, group.ligand)

    if Counter(identity(row.source_group) for row in rows) != Counter(identity(group) for group in groups):
        return None
    parent = frozenset(selected.parent_atoms)
    all_keys, oxygens = [], set()
    for row in rows:
        group = row.source_group
        if (row.kind != 'alcohol' or row.principal_center != group.center
                or row.host_atom != group.center or row.class_rank != 17
                or tuple(sorted(row.group_atoms)) != tuple(sorted(group.atoms))):
            return None
        oxygen = tuple(i for i in group.atoms if mol.GetAtomWithIdx(i).GetAtomicNum() == 8)
        if len(oxygen) != 1 or oxygen[0] in oxygens:
            return None
        oxygens.add(oxygen[0])
        atom = mol.GetAtomWithIdx(oxygen[0])
        bond = mol.GetBondBetweenAtoms(row.host_atom, oxygen[0])
        if (bond is None or bond.GetBondTypeAsDouble() != 1 or atom.GetDegree() != 1
                or atom.GetTotalNumHs(includeNeighbors=True) != 1):
            return None
        all_keys.append((row.kind, row.principal_center, row.host_atom,
                         tuple(sorted(row.group_atoms))))
    if (len(set(all_keys)) != len(all_keys)
            or set(all_keys) != set(getattr(selected, 'principal_site_keys', ()))
            or oxygens != {atom.GetIdx() for atom in atoms if atom.GetAtomicNum() == 8}
            or any(row.host_atom not in parent for row in selected_rows)
            or any(row.host_atom in parent for row in omitted_rows)
            or frozenset(row.principal_center for row in selected_rows)
               != frozenset(selected.principal_centers)):
        return None
    # Keep the accompanying branch grammar source-closed: every detached
    # carbon part is a simple rooted alkyl chain with at most one OH, and
    # that OH, if present, is at its distal endpoint. More complicated
    # branch-parent or prefix choices must keep their prior obligations.
    carbon = {atom.GetIdx() for atom in atoms if atom.GetAtomicNum() == 6}
    if not parent <= carbon:
        return None
    remaining = carbon - parent
    while remaining:
        component = {remaining.pop()}
        stack = list(component)
        while stack:
            idx = stack.pop()
            neighbors = {atom.GetIdx() for atom in mol.GetAtomWithIdx(idx).GetNeighbors()}
            added = neighbors & remaining
            remaining -= added
            component.update(added)
            stack.extend(added)
        attachments = [(idx, atom.GetIdx()) for idx in component
                       for atom in mol.GetAtomWithIdx(idx).GetNeighbors()
                       if atom.GetIdx() in parent]
        if len(attachments) != 1:
            return None
        root, _ = attachments[0]
        degree = {idx: sum(atom.GetIdx() in component
                           for atom in mol.GetAtomWithIdx(idx).GetNeighbors())
                  for idx in component}
        if any(value > 2 for value in degree.values()) or degree[root] > 1:
            return None
        hydroxyl_hosts = [row.host_atom for row in omitted_rows if row.host_atom in component]
        if (len(hydroxyl_hosts) > 1
                or hydroxyl_hosts and len(component) > 1
                   and (hydroxyl_hosts[0] == root or degree[hydroxyl_hosts[0]] != 1)):
            return None
    return len(selected_rows), len(rows)


def functional_justification(mol, groups, frontier, *, method_rule='', method_proved=None):
    candidates = (getattr(frontier, 'compared_candidates', ())
                  or getattr(frontier, 'candidates', ()))
    selected = next(iter(getattr(frontier, 'candidates', ())), None)
    if method_proved is None:
        method_proved = carbon_tree_method_proved(mol, groups, selected)
    return parent_justification(
        getattr(frontier, 'candidate_completeness', None), candidates, selected,
        method_rule=method_rule or 'P-51.1/P-45.1.1', method_proved=method_proved)


def acyl_replacement_justification(mol, Chem, proof, decision):
    """Join an existing full-path replacement proof to ordered comparison.

    The producer proves the unique physical path and both fixed numberings.
    Its suffix-writing uncertainty remains independent of that graph proof.
    """
    from types import SimpleNamespace
    from dataclasses import replace
    from .extended_inventory import selection_inventory
    from .parent_selection import graph_parent_facts
    from .component_mapping import hydrogen_bindings

    groups = tuple(selection_inventory(mol).principal)
    path = tuple(proof.parent_path)
    orientations = tuple(dict.fromkeys((path, tuple(reversed(path)))))
    candidates = []
    for order in orientations:
        mapping = {idx: str(pos) for pos, idx in enumerate(order, 1)}
        prefixes = tuple((mapping[owner], text) for _, owner, text in proof.prefixes)
        mapping.update((idx, mapping[owner]) for idx, owner, _ in proof.prefixes)
        for group in groups:
            if group.center not in mapping:
                break
            mapping.update((idx, mapping[group.center]) for idx in group.atoms
                           if idx not in mapping)
        else:
            mapping = hydrogen_bindings(mol, mapping)
            facts = graph_parent_facts(
                mol, order, mapping, principal_class=groups[0].seniority,
                suffix_centers=tuple(group.center for group in groups),
                suffix_locants=tuple(mapping[group.center] for group in groups),
                prefixes=prefixes, replacement=True, name=decision.name,
                allow_annotations=True)
            if facts is not None:
                candidates.append(SimpleNamespace(parent_atoms=order, name=decision.name,
                    parent_selection=facts, principal_centers=decision.principal_centers))
    selected = next((row for row in candidates if row.parent_atoms == path), None)
    completeness = CandidateCompleteness(
        'typed-linear-acyl-replacement-parent',
        ('P-51.1.3', 'P-51.4.1.1', 'P-15.4.1.2', 'P-44.3', 'P-82.5'),
        orientations, tuple(row.parent_atoms for row in candidates),
        'The typed whole-path proof accounts for both replacement-parent numberings.',
        proposal_keys=tuple(proposal_identity(row) for row in candidates),
    )
    result = parent_justification(completeness, candidates, selected,
                                  method_rule=decision.rule, method_proved=True)
    if proof.pin_expectation_status != 'resolved':
        result = replace(result, steps=(*result.steps, SelectionStep(
            'suffix-writing', 'P-14.3.4.1/P-15.4.3.2.3/P-65.1.2.1',
            'pending', proof.pin_expectation_reason)))
    return result
