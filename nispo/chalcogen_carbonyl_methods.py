"""Mapped substitutive suffixes for acyclic carbon ketone chalcogen analogues.

This producer recognizes neutral internal C=X ketones (X = O, S, Se, Te)
and the terminal C=C=O ketene operation of P-64.2.2.4 in acyclic carbon
skeletons. Ketenes may also carry compulsory halogen prefixes. Aldehydes,
pseudoketones, heterones and rings remain outside scope. Parent selection
and annotation rendering remain graph-mapped operations.
"""

from dataclasses import dataclass

from .molecule_iteration import iter_atoms, iter_bonds
from .naming_methods import MethodDecision

_ELEMENTS = {8: (0, "one", "oxo"), 16: (1, "thione", "sulfanylidene"),
             34: (2, "selone", "selanylidene"),
             52: (3, "tellone", "tellanylidene")}
_HALOGENS = {9: "fluoro", 17: "chloro", 35: "bromo", 53: "iodo"}


@dataclass(frozen=True, slots=True)
class ChalcogenCarbonylSite:
    center: int
    chalcogen: int
    atomic_number: int
    rank: int
    suffix: str
    prefix: str
    is_ketene: bool = False

    @property
    def atoms(self):
        return frozenset((self.center, self.chalcogen))


@dataclass(frozen=True, slots=True)
class ChalcogenCarbonylProposal:
    name: str
    parent_atoms: tuple[int, ...]
    atom_locants: tuple[tuple[int, str], ...]
    sites: tuple[ChalcogenCarbonylSite, ...]
    principal_centers: frozenset[int]
    facts: object


def _single(mol, a, b):
    bond = mol.GetBondBetweenAtoms(a, b)
    return bond is not None and bond.GetBondTypeAsDouble() == 1.0


def recognize_chalcogen_carbonyls(mol):
    """Return exact internal C=X and terminal C=C=O sites in input order."""
    sites = []
    for carbon in iter_atoms(mol):
        if carbon.GetAtomicNum() != 6 or carbon.GetFormalCharge() or carbon.GetNumRadicalElectrons():
            continue
        doubles = [b for b in carbon.GetBonds() if b.GetBondTypeAsDouble() == 2.0]
        if len(doubles) not in {1, 2}:
            continue
        carbonyl_edges = [b for b in doubles
                          if b.GetOtherAtom(carbon).GetAtomicNum() in _ELEMENTS]
        if len(carbonyl_edges) != 1:
            continue
        bond = carbonyl_edges[0]
        x = bond.GetOtherAtom(carbon)
        info = _ELEMENTS.get(x.GetAtomicNum())
        if (info is None or x.GetFormalCharge() or x.GetNumRadicalElectrons()
                or x.GetDegree() != 1 or x.GetIsAromatic() or carbon.IsInRing()
                or x.IsInRing()):
            continue
        is_ketene = (x.GetAtomicNum() == 8 and len(doubles) == 2
                     and carbon.GetDegree() == 2
                     and any(b.GetOtherAtom(carbon).GetAtomicNum() == 6
                             for b in doubles if b.GetIdx() != bond.GetIdx()))
        carbon_neighbors = [n for n in carbon.GetNeighbors()
                            if n.GetIdx() != x.GetIdx() and n.GetAtomicNum() == 6
                            and _single(mol, carbon.GetIdx(), n.GetIdx())]
        if not is_ketene and (len(doubles) != 1 or len(carbon_neighbors) != 2):
            continue
        # The graph and ordinary valence must agree with a ketone site.  This
        # excludes charged/dative and overvalent structures without guessing.
        if carbon.GetDegree() != (2 if is_ketene else 3) or carbon.GetTotalValence() != 4:
            continue
        rank, suffix, prefix = info
        sites.append(ChalcogenCarbonylSite(carbon.GetIdx(), x.GetIdx(),
                                          x.GetAtomicNum(), rank, suffix, prefix, is_ketene))
    return tuple(sites)


def chalcogenylidene_prefix(mol, site):
    """Return the source-backed prefix text and its owned atom ids.

    Isotopic chalcogens and isotopic hydrogens attached to them are outside
    this prefix API's proved grammar and therefore decline.
    """
    atom = mol.GetAtomWithIdx(site.chalcogen)
    if atom.GetIsotope() or any(n.GetAtomicNum() == 1 and n.GetIsotope()
                                for n in atom.GetNeighbors()):
        return None
    return site.prefix, site.atoms


def _carbon_parent_paths(mol, carbon_atoms):
    """Enumerate terminal-to-terminal paths in one carbon tree."""
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


def _render_parent(mol, Chem, path, sites):
    from .branch_rules import parent_chain_name
    from .component_mapping import (MappedComponent, decorate_component,
                                    fully_covered, hydrogen_bindings, mapped_branch)
    from .constants import ROOTS
    from .formatting import attach_suffix, complete_parent_name, multiplied_suffix
    from .parent_selection import graph_parent_facts

    if len(path) not in ROOTS or not all(site.center in path for site in sites):
        return None
    mapping = {idx: str(i) for i, idx in enumerate(path, 1)}
    owner_map = hydrogen_bindings(mol, mapping)
    carbon_set = {a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() == 6}
    has_ketene = any(site.is_ketene for site in sites)
    halogens = tuple(a for a in iter_atoms(mol)
                     if has_ketene and a.GetAtomicNum() in _HALOGENS)
    if any(a.GetIsotope() or a.GetDegree() != 1
           or a.GetNeighbors()[0].GetAtomicNum() != 6
           or a.GetBonds()[0].GetBondTypeAsDouble() != 1 for a in halogens):
        return None
    if any(a.GetAtomicNum() != 6 for a in iter_atoms(mol) if a.GetAtomicNum() > 1
           and a.GetIdx() not in {s.chalcogen for s in sites}
           and a.GetIdx() not in {h.GetIdx() for h in halogens}):
        return None
    if any(mol.GetAtomWithIdx(s.chalcogen).GetIsotope()
           or any(n.GetAtomicNum() == 1 and n.GetIsotope()
                  for n in mol.GetAtomWithIdx(s.chalcogen).GetNeighbors())
           for s in sites):
        return None

    by_rank = {}
    for site in sites:
        by_rank.setdefault(site.rank, []).append(site)
    top_rank = min(by_rank)
    suffix_sites = by_rank[top_rank]
    prefixes = []
    for rank, rows in by_rank.items():
        if rank == top_rank:
            continue
        for site in rows:
            prefixes.append((mapping[site.center], site.prefix))
    locants = sorted(int(mapping[s.center]) for s in suffix_sites)
    embedded = set()
    for halogen in halogens:
        host = halogen.GetNeighbors()[0].GetIdx()
        if host in mapping:
            prefixes.append((mapping[host], _HALOGENS[halogen.GetAtomicNum()]))
            embedded.add(halogen.GetIdx())
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
        expected = component | {h.GetIdx() for h in halogens
                                if h.GetNeighbors()[0].GetIdx() in component}
        if not text or set(owned) != expected or embedded & set(owned):
            return None
        prefixes.append((mapping[parent_idx], text))
        embedded.update(owned)
    stem = parent_chain_name(mol, path)
    if not stem:
        return None
    if len(suffix_sites) == 1:
        suffix_text = suffix_sites[0].suffix
        suffix_parent = (complete_parent_name(stem) if has_ketene else
                         stem if suffix_text[:1] in "aeiou" else complete_parent_name(stem))
        if has_ketene and len(path) > 2 and stem == ROOTS[len(path)] + "en":
            # A standalone alkene may omit 1; the suffix-bearing parent
            # needs both locants (P-14.3.3), including the three-carbon case.
            suffix_parent = ROOTS[len(path)] + "-1-ene"
        # Both the double-bond and suffix positions are intrinsic in the
        # two-carbon ethenone parent. Compulsory prefixes share that property.
        intrinsic_ketene = has_ketene and len(path) == 2
        name = attach_suffix(suffix_parent, suffix_text, () if intrinsic_ketene else locants)
    else:
        parent = complete_parent_name(stem)
        suffix = multiplied_suffix(len(suffix_sites), suffix_sites[0].suffix)
        if not suffix:
            return None
        name = attach_suffix(parent, suffix, locants)
    all_heavy = {a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1}
    chalcogens = {s.chalcogen for s in sites}
    embedded.update(chalcogens)
    if embedded != all_heavy - set(path):
        return None
    # All C=X atoms are locally covered by their suffix or prefix token.  The
    # carbon skeleton, including its isotope/stereo annotations, stays mapped.
    mapped = MappedComponent(
        name, tuple(owner_map.items()), frozenset(owner_map),
        prefixes=tuple(prefixes),
        embedded_atoms=frozenset(embedded), systematic_parent=True,
        prefix_mode="unlocanted" if has_ketene and len(path) == 2 else "locanted",
    )
    rendered = decorate_component(mol, Chem, mapped)
    if not rendered or not fully_covered(mol, owner_map, embedded):
        return None
    suffix_locants = tuple(str(locant) for locant in locants)
    facts = graph_parent_facts(
        mol, path, owner_map, principal_class=(16, top_rank),
        suffix_centers=tuple(s.center for s in suffix_sites),
        suffix_locants=suffix_locants, prefixes=tuple(prefixes), kind="chain",
        name=rendered, allow_annotations=True,
    )
    if facts is None:
        return None
    return ChalcogenCarbonylProposal(rendered, tuple(path), tuple(owner_map.items()),
                                    tuple(sites), frozenset(s.center for s in suffix_sites), facts)


def chain_proposals(mol, Chem, inventory=None, *, principal=None, scope=None):
    """Build mapped proposals for the acyclic carbon-ketone chalcogen family.

    ``inventory``, ``principal`` and ``scope`` are accepted for parent-route
    integration compatibility; this local recognizer remains authoritative.
    """
    sites = recognize_chalcogen_carbonyls(mol)
    if (not sites or mol.GetNumHeavyAtoms() > 64 or mol.GetRingInfo().NumRings()
            or mol.GetStereoGroups()):
        return ()
    carbon = {a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() == 6}
    if any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in iter_atoms(mol)):
        return ()
    # P-44.1 suffix coverage precedes chain length. Enumerate every terminal
    # path, retain maximum coverage of the highest-ranked C=X sites, then the
    # longest eligible parent. Any off-parent C=X unit declines because its
    # acyl/ylidene branch grammar is outside this producer.
    paths = _carbon_parent_paths(mol, carbon)
    carbon_bonds = [b for b in iter_bonds(mol)
                    if b.GetBeginAtomIdx() in carbon and b.GetEndAtomIdx() in carbon]
    if (not paths or any(b.GetBondTypeAsDouble() not in {1.0, 2.0, 3.0}
                         for b in carbon_bonds)):
        return ()
    if not paths:
        return ()
    permitted = {1, 6} | {s.atomic_number for s in sites}
    if any(site.is_ketene for site in sites):
        permitted.update(_HALOGENS)
    if any(a.GetAtomicNum() not in permitted
           for a in iter_atoms(mol) if a.GetAtomicNum() > 1):
        return ()
    principal_rank = min(site.rank for site in sites)
    principal_sites = tuple(site for site in sites if site.rank == principal_rank)
    coverage = {path: sum(site.center in path for site in principal_sites)
                for path in paths}
    best_coverage = max(coverage.values())
    best_paths = [path for path in paths if coverage[path] == best_coverage]
    max_length = max(map(len, best_paths))
    best_paths = [path for path in best_paths if len(path) == max_length]
    # All recognized sites must be inside every surviving parent; otherwise
    # silently dropping an off-parent C=X would change class coverage.
    if any(not all(site.center in path for site in sites) for path in best_paths):
        return ()
    proposals = []
    for raw in best_paths:
        for path in (raw, tuple(reversed(raw))):
            proposal = _render_parent(mol, Chem, path, sites)
            if proposal:
                proposals.append(proposal)
    if not proposals:
        return ()
    from .parent_selection import parent_frontier
    frontier = parent_frontier(proposals, lambda item: item.facts)
    names = {p.name for p in frontier}
    if len(names) != 1:
        return ()
    return tuple(frontier)


def chalcogen_carbonyl_decision(mol, Chem):
    """Return a complete method decision only for the closed acyclic family.

    Any non-carbon heavy atom must be the terminal X of one recognized C=X
    ketone site.  Thus acids, amides, alcohols, amines, and unrelated
    heteroatom chemistry are explicit declines rather than guessed prefixes.
    """
    sites = recognize_chalcogen_carbonyls(mol)
    if not sites:
        return None
    proposals = chain_proposals(mol, Chem)
    if not proposals or len({proposal.name for proposal in proposals}) != 1:
        return None
    proposal = min(proposals, key=lambda item: item.atom_locants)
    atoms = frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1)
    role_rows = []
    parent_role = set(proposal.parent_atoms)
    for idx in proposal.parent_atoms:
        parent_role.update(n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                           if n.GetAtomicNum() == 1 and n.GetIsotope())
    role_rows.append(("parent_hydride", frozenset(parent_role), proposal.atom_locants))
    locants = dict(proposal.atom_locants)
    outside_carbon = {a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() == 6} - set(proposal.parent_atoms)
    unseen = set(outside_carbon)
    while unseen:
        root = min(unseen)
        component, stack, attachments = set(), [root], []
        while stack:
            idx = stack.pop()
            if idx in component:
                continue
            component.add(idx)
            unseen.discard(idx)
            for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
                other = neighbor.GetIdx()
                if other in proposal.parent_atoms:
                    attachments.append(other)
                elif other in unseen:
                    stack.append(other)
        if len(set(attachments)) != 1:
            return None
        branch_atoms = set(component)
        branch_atoms.update(n.GetIdx() for idx in component
                            for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                            if n.GetAtomicNum() in _HALOGENS)
        branch_atoms.update(n.GetIdx() for idx in component
                            for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                            if n.GetAtomicNum() == 1 and n.GetIsotope())
        role_rows.append(("carbon_substituent", frozenset(branch_atoms),
                          tuple((idx, locants[attachments[0]]) for idx in sorted(component)
                                if idx in locants)))
    for site in proposal.sites:
        role_rows.append(("carbonyl_chalcogen", frozenset({site.chalcogen}), ()))
    for atom in iter_atoms(mol):
        if (atom.GetAtomicNum() in _HALOGENS
                and atom.GetNeighbors()[0].GetIdx() in proposal.parent_atoms):
            role_rows.append(("compulsory_halogen_prefix", frozenset({atom.GetIdx()}), ()))
    owned = [idx for _, role_atoms, _ in role_rows for idx in role_atoms]
    expected_owned = {a.GetIdx() for a in iter_atoms(mol)
                      if a.GetAtomicNum() > 1 or a.GetIsotope()}
    if len(owned) != len(set(owned)) or set(owned) != expected_owned:
        return None
    rule = "P-64.6.1/P-64.6.2/P-64.7.3"
    if any(site.is_ketene for site in sites):
        rule += "/P-64.2.2.4/P-61.3/P-14.3.3/P-14.3.4.4"
    return MethodDecision(proposal.name, "substitutive", proposal.principal_centers,
                          atoms, rule,
                          proposal.atom_locants, component_roles=tuple(role_rows))
