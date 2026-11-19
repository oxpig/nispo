"""Bounded original-graph frames for the configured carbonyl release incident."""
from functools import lru_cache


@lru_cache(maxsize=1)
def _naphthalene_topology():
    from rdkit import Chem
    labels = ('1', '2', '3', '4', '4a', '5', '6', '7', '8', '8a')
    graph = Chem.RWMol()
    for _ in labels:
        graph.AddAtom(Chem.Atom(6))
    for a, b in zip(range(10), (*range(1, 10), 0)):
        graph.AddBond(a, b, Chem.BondType.SINGLE)
    graph.AddBond(4, 9, Chem.BondType.SINGLE)
    topology = graph.GetMol()
    Chem.SanitizeMol(topology)
    edges = frozenset(frozenset((labels[b.GetBeginAtomIdx()], labels[b.GetEndAtomIdx()]))
                      for b in topology.GetBonds())
    return labels, edges, topology.GetSubstructMatches(topology, uniquify=False)


def configured_naphthalenone_name(mol, template, match):
    """Prove the alpha-tetralone frame before suffix/H/branch composition.

    P-14.4/P-58: compare fixed naphthalene orientations with the carbonyl at
    position 1, then express its actual 3,4-dihydro/added-2H state. No operation
    is inferred from a previously rendered name or from parser acceptance.
    """
    from .template_access import template_record_locants, template_record_name
    if template_record_name(template) not in {'naphthalene', 'tetralin', 'tetraline'} or len(match) != 10:
        return ''
    from .principal_carbonyl import _ring_carbonyl_comparison_sites, configured_methine_ring_junction
    core = frozenset(match)
    groups = _ring_carbonyl_comparison_sites(mol)
    if (len(groups) != 1 or groups[0].kind != 'ketone' or groups[0].center not in core
            or not configured_methine_ring_junction(mol, core)
            or any(a.GetIsotope() or a.GetChiralTag() for a in mol.GetAtoms())
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6
                   or mol.GetAtomWithIdx(i).GetFormalCharge() for i in core)):
        return ''
    locants = template_record_locants(template)
    labels, edges, symmetries = _naphthalene_topology()
    if set(locants.values()) != set(labels) or len(locants) != 10:
        return ''
    old = {str(locants[j]): idx for j, idx in enumerate(match)}
    inverse = {idx: label for label, idx in old.items()}
    actual = frozenset(frozenset((inverse[b.GetBeginAtomIdx()], inverse[b.GetEndAtomIdx()]))
                       for b in mol.GetBonds()
                       if b.GetBeginAtomIdx() in core and b.GetEndAtomIdx() in core)
    if actual != edges:
        return ''
    oxygen = [a.GetIdx() for a in mol.GetAtomWithIdx(groups[0].center).GetNeighbors()
              if a.GetAtomicNum() == 8 and a.GetDegree() == 1
              and mol.GetBondBetweenAtoms(groups[0].center, a.GetIdx()).GetBondTypeAsDouble() == 2]
    if len(oxygen) != 1:
        return ''
    offered = []
    for permutation in symmetries:
        at = {labels[j]: old[labels[permutation[j]]] for j in range(10)}
        if at['1'] != groups[0].center:
            continue
        if any(not mol.GetAtomWithIdx(at[label]).GetIsAromatic()
               for label in ('4a', '5', '6', '7', '8', '8a')):
            continue
        if any(mol.GetBondBetweenAtoms(at[a], at[b]).GetBondTypeAsDouble() != 1
               for a, b in (('1', '2'), ('2', '3'), ('3', '4'), ('4', '4a'), ('8a', '1'))):
            continue
        if any(mol.GetAtomWithIdx(at[label]).GetIsAromatic() for label in ('1', '2', '3', '4')):
            continue
        methine = [b.GetOtherAtom(mol.GetAtomWithIdx(at['2']))
                   for b in mol.GetAtomWithIdx(at['2']).GetBonds()
                   if b.GetOtherAtomIdx(at['2']) not in core and b.GetBondTypeAsDouble() == 2]
        if (len(methine) != 1 or methine[0].GetAtomicNum() != 6
                or methine[0].GetTotalNumHs() != 1 or methine[0].GetDegree() != 2):
            continue
        if any(mol.GetAtomWithIdx(at[label]).GetTotalNumHs() + sum(
                b.GetBondTypeAsDouble() for b in mol.GetAtomWithIdx(at[label]).GetBonds()
                if b.GetOtherAtomIdx(at[label]) not in core) != 2 for label in ('1', '2', '3', '4')):
            continue
        mapping = {idx: label for label, idx in at.items()}
        offered.append(mapping)
    if len(offered) != 1:
        return ''
    mapping = offered[0]
    from .branch_rules import collect_subtree
    from .parent.rule_helpers import branch_prefix, simple_external_prefix
    from .formatting import format_prefixes, join_prefix_parent
    prefixes, covered = [], set(core) | set(oxygen)
    for idx in core:
        for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
            root = neighbor.GetIdx()
            if root in core or root in oxygen:
                continue
            atoms = set(collect_subtree(mol, root, idx))
            if atoms & covered:
                return ''
            text = simple_external_prefix(mol, root, idx) or branch_prefix(mol, root, idx)
            if not text:
                return ''
            prefixes.append((mapping[idx], text))
            covered.update(atoms)
    if covered != set(range(mol.GetNumAtoms())):
        return ''
    name = join_prefix_parent(format_prefixes(prefixes), '3,4-dihydronaphthalen-1(2H)-one')
    from .molecule_cache import mol_cache_dict
    from .principal_carbonyl import record_cited_ring_suffix
    mol_cache_dict(mol, 'rendered_parent_locants')[(name, core)] = tuple(mapping.items())
    record_cited_ring_suffix(mol, core, name, groups)
    return name
