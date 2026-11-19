"""Source-owned carbon-tree diyl/triyl constructions under P-71."""

from .component_mapping import MappedComponent, decorate_component, mapped_branch
from .formatting import attach_suffix, complete_parent_name, prefix_alphabetization_key
from .naming_methods import MethodDecision
from .naming_result import SelectionLimitation


def carbon_tree_polyradical(mol, Chem):
    atoms = tuple(mol.GetAtoms())
    if (not atoms or len(atoms) > 64 or mol.GetStereoGroups() or mol.GetRingInfo().NumRings()
            or len(Chem.GetMolFrags(mol)) != 1
            or any(a.GetAtomicNum() != 6 or a.GetFormalCharge() or int(a.GetChiralTag()) for a in atoms)
            or any(b.GetBondTypeAsDouble() != 1 or int(b.GetStereo()) or int(b.GetBondDir())
                   for b in mol.GetBonds())):
        return None
    centers = frozenset(a.GetIdx() for a in atoms if a.GetNumRadicalElectrons())
    if len(centers) < 2 or any(a.GetNumRadicalElectrons() not in {0, 1}
                              or a.GetTotalValence() != 4 - a.GetNumRadicalElectrons() for a in atoms):
        return None
    graph = {a.GetIdx(): tuple(n.GetIdx() for n in a.GetNeighbors()) for a in atoms}
    from .carbon_radical_methods import _candidate_paths
    paths = [p for p in _candidate_paths(graph, min(centers)) if centers <= set(p)]
    if not paths:
        return None
    longest = max(map(len, paths))
    from .constants import MULT
    multiplier = MULT.get(len(centers))
    if not multiplier:
        return None
    from .branch_rules import parent_chain_name
    candidates = []
    all_atoms = frozenset(graph)
    for raw in paths:
        if len(raw) != longest:
            continue
        for path in (raw, tuple(reversed(raw))):
            bindings = {idx: str(i + 1) for i, idx in enumerate(path)}
            prefixes, embedded, roles = [], set(), []
            failed = False
            for idx in path:
                for root in graph[idx]:
                    if root in bindings:
                        continue
                    text, owned = mapped_branch(mol, Chem, root, idx)
                    if not text or owned & (bindings.keys() | embedded):
                        failed = True
                        break
                    prefixes.append((bindings[idx], text))
                    embedded.update(owned)
                    roles.append(('carbon-substituent', owned, ()))
                if failed:
                    break
            if failed or bindings.keys() | embedded != all_atoms:
                continue
            locants = tuple(sorted((int(bindings[i]) for i in centers)))
            parent = complete_parent_name(parent_chain_name(mol, path))
            parent = attach_suffix(parent, multiplier + 'yl', locants)
            if not parent:
                continue
            name = decorate_component(mol, Chem, MappedComponent(parent, tuple(bindings.items()),
                frozenset(bindings), tuple(prefixes), frozenset(embedded), systematic_parent=True))
            if name:
                key = (locants, -len(prefixes), tuple(sorted(int(p) for p, _ in prefixes)),
                       tuple(int(p) for p, t in sorted(prefixes,
                            key=lambda row: (prefix_alphabetization_key(row[1]), int(row[0])))))
                candidates.append((key, name, bindings, roles))
    if not candidates:
        return None
    best = min(row[0] for row in candidates)
    # A stable general construction does not certify a completed P-71/P-46
    # numbering frontier. Lexical order resolves only this delivery choice.
    _, name, bindings, roles = min((r for r in candidates if r[0] == best), key=lambda r: r[1])
    return MethodDecision(name=name, method='substitutive', principal_centers=centers,
        atoms=all_atoms, rule='P-71.2.1.2/P-71.6/P-71.7/P-29', atom_locants=tuple(bindings.items()),
        component_roles=(('parent-polyradical', frozenset(bindings), tuple(bindings.items())), *roles),
        preference_issues=(SelectionLimitation('polyradical-parent', 'P-71.7/P-46',
            'All radical centers and source atoms are owned; the global preferred radical-parent comparison remains unresolved.'),))
