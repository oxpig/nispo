"""Mapped aldehyde and chalcogen-aldehyde suffix decisions.

The admitted family is a neutral acyclic carbon tree with one or two terminal
C=X-H aldehyde groups (X = O, S, Se, Te).  P-66.6 makes aldehydes senior to
ketones; the latter may therefore occur as supported lower prefixes.  The
producer keeps the suffix parent and lower prefixes typed until the ordinary
mapped-component renderer applies isotope and stereochemical annotations.
"""

from dataclasses import dataclass

from .molecule_iteration import iter_atoms, iter_bonds
from .naming_methods import MethodDecision

_ALDEHYDE = {
    8: ("al", "oxo"),
    16: ("thial", "sulfanylidene"),
    34: ("selenal", "selanylidene"),
    52: ("tellanal", "tellanylidene"),
}


@dataclass(frozen=True, slots=True)
class AldehydeSite:
    center: int
    chalcogen: int
    atomic_number: int
    suffix: str
    prefix: str

    @property
    def atoms(self):
        return frozenset((self.center, self.chalcogen))


@dataclass(frozen=True, slots=True)
class AldehydeProposal:
    name: str
    parent_atoms: tuple[int, ...]
    atom_locants: tuple[tuple[int, str], ...]
    sites: tuple[AldehydeSite, ...]
    principal_centers: frozenset[int]
    facts: object


def _single(mol, a, b):
    bond = mol.GetBondBetweenAtoms(a, b)
    return bond is not None and bond.GetBondTypeAsDouble() == 1.0


def recognize_chalcogen_aldehydes(mol):
    """Return neutral terminal C=X-H sites in input atom order."""
    sites = []
    for carbon in iter_atoms(mol):
        if (carbon.GetAtomicNum() != 6 or carbon.GetFormalCharge()
                or carbon.GetNumRadicalElectrons() or carbon.IsInRing()):
            continue
        doubles = [b for b in carbon.GetBonds() if b.GetBondTypeAsDouble() == 2.0]
        if len(doubles) != 1:
            continue
        x = doubles[0].GetOtherAtom(carbon)
        info = _ALDEHYDE.get(x.GetAtomicNum())
        if (info is None or x.GetFormalCharge() or x.GetNumRadicalElectrons()
                or x.GetDegree() != 1 or x.IsInRing()):
            continue
        carbon_neighbors = [n for n in carbon.GetNeighbors()
                            if n.GetIdx() != x.GetIdx() and n.GetAtomicNum() == 6
                            and _single(mol, carbon.GetIdx(), n.GetIdx())]
        explicit_h = [n for n in carbon.GetNeighbors() if n.GetAtomicNum() == 1]
        if len(carbon_neighbors) != 1 or carbon.GetDegree() != 2:
            continue
        # An implicit hydrogen is the only hydrogen representation admitted.
        # Isotope labels on this functional H or X require suffix-local syntax
        # not established by P-82.2.1 for this producer.
        if explicit_h or carbon.GetTotalValence() != 4:
            continue
        suffix, prefix = info
        sites.append(AldehydeSite(carbon.GetIdx(), x.GetIdx(), x.GetAtomicNum(),
                                  suffix, prefix))
    return tuple(sites)


def _carbon_parent_paths(mol, carbon_atoms):
    atoms = set(carbon_atoms)
    if not atoms:
        return ()
    graph = {i: tuple(sorted(n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors()
                             if n.GetIdx() in atoms)) for i in atoms}
    if len(atoms) == 1:
        return (tuple(atoms),)
    if sum(map(len, graph.values())) // 2 != len(atoms) - 1:
        return ()
    leaves = sorted(i for i, row in graph.items() if len(row) <= 1)
    paths = set()
    for p, start in enumerate(leaves):
        for end in leaves[p + 1:]:
            stack = [(start, -1, (start,))]
            while stack:
                current, previous, path = stack.pop()
                if current == end:
                    paths.add(path)
                    break
                for nxt in graph[current]:
                    if nxt != previous:
                        stack.append((nxt, current, path + (nxt,)))
    return tuple(sorted(paths))


def _render_parent(mol, Chem, path, sites, ketones):
    from .branch_rules import parent_chain_name
    from .component_mapping import (MappedComponent, decorate_component,
                                    fully_covered, hydrogen_bindings, mapped_branch)
    from .constants import ROOTS
    from .formatting import (attach_suffix, complete_parent_name,
                             multiplied_suffix)
    from .parent_selection import graph_parent_facts

    if len(path) not in ROOTS or not all(site.center in path for site in sites):
        return None
    mapping = {idx: str(i) for i, idx in enumerate(path, 1)}
    owner_map = hydrogen_bindings(mol, mapping)
    carbon_set = {a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() == 6}
    recognized_x = {s.chalcogen for s in sites} | {s.chalcogen for s in ketones}
    if any(a.GetAtomicNum() != 6 for a in iter_atoms(mol)
           if a.GetAtomicNum() > 1 and a.GetIdx() not in recognized_x):
        return None
    # Functional chalcogens do not yet have proved isotope descriptor syntax.
    if any(mol.GetAtomWithIdx(idx).GetIsotope()
           or any(n.GetAtomicNum() == 1 and n.GetIsotope()
                  for n in mol.GetAtomWithIdx(idx).GetNeighbors())
           for idx in recognized_x):
        return None

    prefixes = []
    for site in ketones:
        if site.center not in mapping:
            return None
        prefixes.append((mapping[site.center], site.prefix))

    # Ordinary carbon branches are rendered by the established graph mapper;
    # a branch containing another C=X site cannot be mistaken for alkyl.
    embedded = set(recognized_x)
    remaining = carbon_set - set(path)
    while remaining:
        root = min(remaining)
        stack, component, attached = [root], set(), []
        while stack:
            idx = stack.pop()
            if idx in component:
                continue
            component.add(idx)
            remaining.discard(idx)
            for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
                other = neighbor.GetIdx()
                if other in path:
                    attached.append((other, idx))
                elif other in remaining:
                    stack.append(other)
        if len(attached) != 1:
            return None
        parent_idx, branch_root = attached[0]
        text, owned = mapped_branch(mol, Chem, branch_root, parent_idx)
        if not text or set(owned) != component or embedded & set(owned):
            return None
        prefixes.append((mapping[parent_idx], text))
        embedded.update(owned)

    stem = parent_chain_name(mol, path)
    if not stem:
        return None
    suffix = sites[0].suffix
    if any(site.suffix != suffix for site in sites):
        return None
    if len(sites) == 1:
        # P-66.6.1.1: ordinary monoaldehyde terminal suffix, without a locant.
        # The chalcogen-analogue suffixes begin consonantically and retain e.
        suffix_parent = stem if suffix == "al" else complete_parent_name(stem)
        name = attach_suffix(suffix_parent, suffix)
    elif len(sites) == 2:
        # P-66.6.1.1 gives dialdehydes; the two included terminal groups have
        # implicit terminal locants, as in pentanedial / pentanedithial.
        parent = complete_parent_name(stem)
        multiple = multiplied_suffix(2, suffix)
        if not multiple:
            return None
        name = attach_suffix(parent, multiple)
    else:
        return None

    if embedded != ({a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1}
                    - set(path)):
        return None
    mapped = MappedComponent(
        name, tuple(owner_map.items()), frozenset(owner_map),
        prefixes=tuple(prefixes), embedded_atoms=frozenset(embedded),
        systematic_parent=True,
    )
    rendered = decorate_component(mol, Chem, mapped)
    if not rendered or not fully_covered(mol, owner_map, embedded):
        return None
    suffix_locants = tuple(str(i + 1) for i, idx in enumerate(path)
                           if any(s.center == idx for s in sites))
    facts = graph_parent_facts(
        mol, path, owner_map, principal_class=(15,),
        suffix_centers=tuple(s.center for s in sites), suffix_locants=suffix_locants,
        prefixes=tuple(prefixes), kind="chain", name=rendered,
        allow_annotations=True,
    )
    if facts is None:
        return None
    return AldehydeProposal(rendered, tuple(path), tuple(owner_map.items()),
                            tuple(sites), frozenset(s.center for s in sites), facts)


def chain_proposals(mol, Chem, inventory=None, *, principal=None, scope=None):
    """Build graph-mapped candidates for the admitted acyclic aldehyde family."""
    from .chalcogen_carbonyl_methods import recognize_chalcogen_carbonyls

    sites = recognize_chalcogen_aldehydes(mol)
    if (not sites or len(sites) > 2 or mol.GetNumHeavyAtoms() > 64
            or mol.GetRingInfo().NumRings() or mol.GetStereoGroups()):
        return ()
    if any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in iter_atoms(mol)):
        return ()
    # Mixed O/S/Se/Te principal aldehyde suffix ranking is not established by
    # the cited rules here; retain only one suffix family per decision.
    if len({site.atomic_number for site in sites}) != 1:
        return ()
    carbon = {a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() == 6}
    # P-66.6.1.2.1 designates acetaldehyde as the PIN and ethanal as its
    # systematic alternative. Leave the two-carbon ordinary aldehyde to the
    # retained-name route, including isotope/stereo substitutions; never
    # publish the non-PIN systematic alternative from this substitutive path.
    if len(carbon) == 2 and len(sites) == 1 and sites[0].atomic_number == 8:
        return ()
    ketones = recognize_chalcogen_carbonyls(mol)
    paths = _carbon_parent_paths(mol, carbon)
    if not paths:
        return ()
    carbon_bonds = [b for b in iter_bonds(mol)
                    if b.GetBeginAtomIdx() in carbon and b.GetEndAtomIdx() in carbon]
    if any(b.GetBondTypeAsDouble() not in {1.0, 2.0, 3.0} for b in carbon_bonds):
        return ()
    admitted_x = {s.chalcogen for s in sites} | {s.chalcogen for s in ketones}
    if any(a.GetAtomicNum() not in {1, 6} | {8, 16, 34, 52}
           for a in iter_atoms(mol)) or any(
            a.GetAtomicNum() in {8, 16, 34, 52} and a.GetIdx() not in admitted_x
            for a in iter_atoms(mol)):
        return ()
    # The aldehyde suffix is senior to ketones (P-66.6.4), regardless of X.
    coverage = {path: sum(site.center in path for site in sites) for path in paths}
    best_coverage = max(coverage.values())
    paths = [p for p in paths if coverage[p] == best_coverage]
    longest = max(map(len, paths))
    paths = [p for p in paths if len(p) == longest]
    if any(not all(s.center in path for s in sites) for path in paths):
        return ()
    proposals = []
    for raw in paths:
        for path in (raw, tuple(reversed(raw))):
            proposal = _render_parent(mol, Chem, path, sites, ketones)
            if proposal:
                proposals.append(proposal)
    if not proposals:
        return ()
    from .parent_selection import parent_frontier
    frontier = parent_frontier(proposals, lambda item: item.facts)
    if len({p.name for p in frontier}) != 1:
        return ()
    return tuple(frontier)


def chalcogen_aldehyde_decision(mol, Chem):
    """Return a complete mapped decision or decline unsupported chemistry."""
    proposals = chain_proposals(mol, Chem)
    if not proposals or len({p.name for p in proposals}) != 1:
        return None
    proposal = min(proposals, key=lambda p: p.atom_locants)
    all_atoms = frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1)
    locants = dict(proposal.atom_locants)
    role_rows = []
    parent_role = set(proposal.parent_atoms)
    for idx in proposal.parent_atoms:
        parent_role.update(n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                           if n.GetAtomicNum() == 1 and n.GetIsotope())
    role_rows.append(("parent_hydride", frozenset(parent_role), proposal.atom_locants))
    outside = {a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() == 6} - set(proposal.parent_atoms)
    remaining = set(outside)
    while remaining:
        root = min(remaining)
        component, stack, attachments = set(), [root], []
        while stack:
            idx = stack.pop()
            if idx in component:
                continue
            component.add(idx)
            remaining.discard(idx)
            for n in mol.GetAtomWithIdx(idx).GetNeighbors():
                other = n.GetIdx()
                if other in proposal.parent_atoms:
                    attachments.append(other)
                elif other in remaining:
                    stack.append(other)
        if len(set(attachments)) != 1:
            return None
        role_atoms = set(component)
        role_atoms.update(n.GetIdx() for idx in component
                          for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                          if n.GetAtomicNum() == 1 and n.GetIsotope())
        attachment = str(locants[attachments[0]])
        role_rows.append(("carbon_substituent", frozenset(role_atoms),
                          tuple((i, attachment) for i in sorted(component) if i in locants)))
    for site in proposal.sites:
        role_rows.append(("aldehyde_chalcogen", frozenset({site.chalcogen}), ()))
    from .chalcogen_carbonyl_methods import recognize_chalcogen_carbonyls
    for site in recognize_chalcogen_carbonyls(mol):
        role_rows.append(("lower_ketone_prefix_chalcogen",
                          frozenset({site.chalcogen}), ()))
    owned = [idx for _, atoms, _ in role_rows for idx in atoms]
    expected = {a.GetIdx() for a in iter_atoms(mol)
                if a.GetAtomicNum() > 1 or a.GetIsotope()}
    if len(owned) != len(set(owned)) or set(owned) != expected:
        return None
    return MethodDecision(proposal.name, "substitutive", proposal.principal_centers,
                          all_atoms, "P-41/P-43/P-44.1/P-44.3/P-66.6",
                          proposal.atom_locants, component_roles=tuple(role_rows))
