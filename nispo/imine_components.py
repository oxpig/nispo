"""Mapped ordinary carbon imines under Blue Book P-62.3.

This deliberately bounded renderer handles neutral C=NH and C=N–C imines
whose parent can be represented by an acyclic carbon chain. Ring and other
parent systems remain the responsibility of their dedicated parent routes.
"""

from itertools import permutations, product

RULES = (
    "P-41-class-20",
    "P-44.1.1/P-45.2",
    "P-62.3.1.1",
    "P-62.3.1.2",
    "P-16.9.2/P-16.9.3",
    "P-82.2",
    "P-91.2.1.1/P-93.1",
)


def ordinary_imine(mol, carbon_idx):
    """True for a neutral C=N–H or C=N–C imine with closed local valence."""
    carbon = mol.GetAtomWithIdx(carbon_idx)
    if carbon.GetAtomicNum() != 6 or carbon.GetFormalCharge() or carbon.GetNumRadicalElectrons():
        return False
    doubles = [
        atom for atom in carbon.GetNeighbors()
        if atom.GetAtomicNum() == 7
        and mol.GetBondBetweenAtoms(carbon_idx, atom.GetIdx()).GetBondTypeAsDouble() == 2
    ]
    if len(doubles) != 1:
        return False
    nitrogen = doubles[0]
    single_heavy = [a for a in nitrogen.GetNeighbors() if a.GetIdx() != carbon_idx and a.GetAtomicNum() > 1]
    return (
        not nitrogen.GetFormalCharge()
        and not nitrogen.GetNumRadicalElectrons()
        and not nitrogen.IsInRing()
        and nitrogen.GetTotalValence() == 3
        and len(single_heavy) <= 1
        and all(a.GetAtomicNum() == 6 for a in single_heavy)
        and all(mol.GetBondBetweenAtoms(nitrogen.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() == 1
                for a in single_heavy)
    )


def imine_nitrogen(mol, carbon_idx):
    if not ordinary_imine(mol, carbon_idx):
        return -1
    return next(a.GetIdx() for a in mol.GetAtomWithIdx(carbon_idx).GetNeighbors()
                if a.GetAtomicNum() == 7 and
                mol.GetBondBetweenAtoms(carbon_idx, a.GetIdx()).GetBondTypeAsDouble() == 2)


def principal_scope(mol, groups):
    """Admit only complete, neutral ordinary imine groups in one molecule."""
    return bool(groups) and all(
        group.kind == "imine" and ordinary_imine(mol, group.center)
        for group in groups
    )


def suffix_site_maps(mol, groups, parent, mapping):
    """Yield carbon suffix locants and original N-site locant assignments."""
    by_site = {}
    for group in groups:
        if group.center not in parent or not ordinary_imine(mol, group.center):
            return
        by_site.setdefault(group.center, []).append(imine_nitrogen(mol, group.center))
    sites = tuple(by_site)
    values = tuple(by_site.values())
    for assignments in product(*(permutations(ids) for ids in values)):
        nitrogen_map = {}
        for (site, ids), ordered in zip(by_site.items(), assignments):
            for index, nitrogen in enumerate(ordered):
                label = "N" + (str(mapping[site]) if len(groups) > 1 else "")
                if index:
                    label = "N" + "'" * index + (str(mapping[site]) if len(groups) > 1 else "")
                nitrogen_map[nitrogen] = label
        yield sites, nitrogen_map


def imine_ending(mol, Chem, parent, groups, locants):
    """Build the P-62.3 suffix and atom ownership for the imine nitrogens."""
    from .formatting import attach_suffix, multiplied_suffix

    if not groups or any(not ordinary_imine(mol, g.center) for g in groups):
        return "", frozenset()
    suffix = multiplied_suffix(len(groups), "imine")
    if not suffix:
        return "", frozenset()
    text = attach_suffix(parent, suffix, locants)
    nitrogens = frozenset(imine_nitrogen(mol, group.center) for group in groups)
    return text, nitrogens


def unlocanted_imine_nitrogen(mol, scope=None):
    atoms = mol.GetAtoms() if scope is None else (mol.GetAtomWithIdx(i) for i in scope)
    return (7,) if sum(a.GetAtomicNum() == 7 for a in atoms) == 1 else ()


def imino_prefix(mol, Chem, root, blocked, atoms):
    """Map a subordinate =NH component as P-62.3.1.2 `imino`.

    The component mapper calls with root=N and blocked=its C host. The
    earlier C-root form remains available to callers that cut the host bond.
    N-substituted imines need the P-62.3.1.2 `azanylylidene` context and are
    declined here.
    """
    from .component_mapping import MappedComponent, decorate_component, hydrogen_bindings

    root_atom = mol.GetAtomWithIdx(root)
    if root_atom.GetAtomicNum() == 7:
        if (mol.GetAtomWithIdx(blocked).GetAtomicNum() != 6
                or mol.GetBondBetweenAtoms(root, blocked) is None
                or mol.GetBondBetweenAtoms(root, blocked).GetBondTypeAsDouble() != 2
                or imine_nitrogen(mol, blocked) != root):
            return "", frozenset()
        if any(a.GetAtomicNum() > 1 and a.GetIdx() != blocked for a in root_atom.GetNeighbors()):
            return "", frozenset()
        mapping = {root: ""}
        mapping.update({a.GetIdx(): "" for a in root_atom.GetNeighbors() if a.GetAtomicNum() == 1})
        mapping = hydrogen_bindings(mol, mapping)
        if set(mapping) != set(atoms):
            return "", frozenset()
        text = decorate_component(mol, Chem, MappedComponent(
            "imino", tuple(mapping.items()), frozenset(mapping),
        ))
        return (text, frozenset(atoms)) if text else ("", frozenset())
    edge = mol.GetBondBetweenAtoms(root, blocked)
    if (root_atom.GetAtomicNum() != 6 or edge is None or edge.GetBondTypeAsDouble() != 1):
        return "", frozenset()
    nitrogen = imine_nitrogen(mol, root)
    if nitrogen < 0 or nitrogen not in atoms:
        return "", frozenset()
    n_atom = mol.GetAtomWithIdx(nitrogen)
    if any(a.GetAtomicNum() > 1 and a.GetIdx() != root for a in n_atom.GetNeighbors()):
        return "", frozenset()
    if not set(atoms) <= {root, nitrogen, *(a.GetIdx() for a in n_atom.GetNeighbors() if a.GetAtomicNum() == 1)}:
        return "", frozenset()
    return "imino", frozenset(atoms)


def chain_proposals(mol, Chem, inventory, *, scope=None, principal=None):
    """Generate fully mapped acyclic chain parents ending in imine suffixes."""
    from .multifunctional_parent import _paths
    from .component_mapping import MappedComponent, decorate_component, hydrogen_bindings, mapped_branch
    from .formatting import complete_parent_name
    from .branch_rules import parent_chain_name
    from .parent_selection import graph_parent_facts
    from .cross_route_selection import ParentProposal, _annotation_ownership

    groups = inventory.principal if principal is None else tuple(principal)
    if not groups or inventory.principal[0].kind != "imine" or not principal_scope(mol, groups):
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
            for sites, nmap in suffix_site_maps(mol, covered, seq, carbon_mapping):
                mapping = dict(carbon_mapping)
                mapping.update(nmap)
                prefixes, embedded, failed = [], set(), False
                for idx, locant in tuple(mapping.items()):
                    for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
                        root = neighbor.GetIdx()
                        if root in mapping or neighbor.GetAtomicNum() <= 1 or (scope is not None and root not in scope):
                            continue
                        if mol.GetBondBetweenAtoms(idx, root).GetBondTypeAsDouble() != 1:
                            failed = True
                            break
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
                suffix_locants = () if len(covered) == 1 and len(seq) <= 2 else locants
                ending, suffix_owned = imine_ending(mol, Chem, parent, covered, suffix_locants)
                if not ending:
                    continue
                name = decorate_component(mol, Chem, MappedComponent(
                    ending, tuple(mapping.items()), frozenset(mapping), tuple(prefixes),
                    frozenset(embedded), systematic_parent=True,
                    nitrogen_site_locants=True,
                    isotope_unlocanted_elements=unlocanted_imine_nitrogen(mol, scope),
                ))
                if not name or len(name) > 260:
                    continue
                facts = graph_parent_facts(
                    mol, seq, mapping, principal_class=groups[0].seniority,
                    suffix_centers=[g.center for g in covered], suffix_locants=locants,
                    prefixes=prefixes, kind="chain", name=name, allow_annotations=True,
                )
                if facts is not None:
                    yield ParentProposal(
                        name, "mapped-chain-imine", "imine",
                        frozenset(g.center for g in covered), tuple(seq),
                        tuple(mapping.items()), facts, True,
                    )
