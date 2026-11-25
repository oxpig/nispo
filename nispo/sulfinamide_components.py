"""Mapped neutral carbon-hosted sulfinamide suffix proposals (P-66.1.1.2).

This is an additive producer. Shared inventory and principal-selection
integration are owned elsewhere. It supports one ordinary C--S(=O)--N unit
on a saturated all-carbon chain or monocycle, with mapped ordinary carbon
substituents on the terminal amide nitrogen. Recognition preserves original
atom IDs and does not infer a name from external parser output.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from .component_mapping import MappedComponent, decorate_component, fully_covered, hydrogen_bindings, mapped_branch
from .functional_inventory import CharacteristicGroup
from .formatting import complete_parent_name
from .naming_methods import MethodDecision
from .parent_selection import graph_parent_facts, parent_frontier
from .sulfur_selection import sulfur_suffix_locants


RULE = "P-41/P-42.1/P-44/P-45/P-66.1.1.2/P-66.1.1.3.1.1/P-82/P-91"
_SULFINAMIDE_SUFFIX = "sulfinamide"


@dataclass(frozen=True, slots=True)
class SulfinamideSite:
    """One source-shaped group on the original graph.

    The data shape is compatible with ``CharacteristicGroup`` extraction:
    center is S, owner is the carbon parent host, ligand is the amide N, and
    atoms are the S/O/N suffix core (the parent carbon is deliberately not
    included in this characteristic-unit atom set).
    """

    kind: str
    center: int
    atoms: tuple[int, ...]
    owner: int
    ligand: int
    oxo_oxygen: int
    n_substituents: tuple[int, ...] = ()
    n_substituent_atoms: tuple[frozenset[int], ...] = ()


@dataclass(frozen=True, slots=True)
class SulfinamideRecognition:
    status: Literal["not-applicable", "supported", "higher-class-route", "deferred"]
    reason: str
    sites: tuple[SulfinamideSite, ...] = ()
    higher_centers: frozenset[int] = frozenset()
    higher_kinds: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SulfinamideProposal:
    name: str
    site: SulfinamideSite
    parent_atoms: tuple[int, ...]
    atom_locants: tuple[tuple[int, str], ...]
    equivalent_maps: tuple[tuple[tuple[int, str], ...], ...]
    component: MappedComponent
    component_roles: tuple
    priority_facts: object
    rule: str = RULE

    @property
    def atoms(self):
        return self.component.atoms | self.component.embedded_atoms

    def as_method_decision(self) -> MethodDecision:
        """Adapt only a complete source-ranked proposal to the method API."""
        if not self.component_roles:
            raise ValueError("sulfinamide proposal has no ownership proof")
        role_atoms = [frozenset(row[1]) for row in self.component_roles]
        if any(role_atoms[i] & role_atoms[j]
               for i in range(len(role_atoms)) for j in range(i + 1, len(role_atoms))):
            raise ValueError("sulfinamide component roles overlap")
        if frozenset().union(*role_atoms) != self.atoms:
            raise ValueError("sulfinamide component roles do not cover all owned atoms")
        return MethodDecision(
            name=self.name,
            method="substitutive",
            principal_centers=frozenset((self.site.center,)),
            atoms=self.atoms,
            rule=self.rule,
            atom_locants=self.atom_locants,
            component_roles=self.component_roles,
        )


def _single(mol, a: int, b: int) -> bool:
    bond = mol.GetBondBetweenAtoms(a, b)
    return bond is not None and bond.GetBondTypeAsDouble() == 1.0


def _double(mol, a: int, b: int) -> bool:
    bond = mol.GetBondBetweenAtoms(a, b)
    return bond is not None and bond.GetBondTypeAsDouble() == 2.0


def _n_substituent_branch(mol, Chem, nitrogen: int, root: int):
    from .branch_rules import collect_subtree
    from .organyl_prefix_components import organyl_prefix_nature

    atoms = frozenset(collect_subtree(mol, root, nitrogen))
    if not atoms or not _single(mol, nitrogen, root):
        return None
    text, mapped_atoms = mapped_branch(mol, Chem, root, nitrogen)
    if not text or frozenset(mapped_atoms) != atoms:
        return None
    nature = organyl_prefix_nature(mol, root, atoms, text)
    if nature not in {"simple", "compound"}:
        return None
    return text, atoms, nature


def _recognize_site(mol, Chem, sulfur) -> tuple[SulfinamideSite | None, str]:
    sidx = sulfur.GetIdx()
    if sulfur.GetFormalCharge() or sulfur.GetNumRadicalElectrons() or sulfur.IsInRing():
        return None, "sulfur charge/radical/annular topology is outside the local class rule"
    double_oxygens = [n.GetIdx() for n in sulfur.GetNeighbors()
                      if n.GetAtomicNum() == 8 and _double(mol, sidx, n.GetIdx())
                      and n.GetDegree() == 1 and not n.GetFormalCharge()]
    carbon_hosts = [n.GetIdx() for n in sulfur.GetNeighbors()
                    if n.GetAtomicNum() == 6 and _single(mol, sidx, n.GetIdx())]
    nitrogens = [n.GetIdx() for n in sulfur.GetNeighbors()
                 if n.GetAtomicNum() == 7 and _single(mol, sidx, n.GetIdx())]
    if not (len(double_oxygens) == 1 and len(carbon_hosts) == 1 and len(nitrogens) == 1
            and sulfur.GetDegree() == 3 and sulfur.GetTotalValence() == 4):
        return None, "sulfur is not one neutral C--S(=O)--N sulfinamide center"
    oxo, host, nidx = double_oxygens[0], carbon_hosts[0], nitrogens[0]
    oxygen = mol.GetAtomWithIdx(oxo)
    nitrogen = mol.GetAtomWithIdx(nidx)
    if (nitrogen.GetFormalCharge() or nitrogen.GetNumRadicalElectrons()
            or nitrogen.IsInRing()):
        return None, "sulfinamide N charge/radical/annular topology is outside the local class rule"
    roots = []
    for neighbor in nitrogen.GetNeighbors():
        idx = neighbor.GetIdx()
        if idx == sidx:
            continue
        if neighbor.GetAtomicNum() == 1:
            continue
        if neighbor.GetAtomicNum() != 6 or neighbor.GetFormalCharge() or not _single(mol, nidx, idx):
            return None, "N-substituents must be ordinary single-bonded carbon organyls"
        roots.append(idx)
    heavy_neighbors = [neighbor for neighbor in nitrogen.GetNeighbors()
                       if neighbor.GetAtomicNum() > 1]
    explicit_hydrogens = [neighbor for neighbor in nitrogen.GetNeighbors()
                          if neighbor.GetAtomicNum() == 1]
    if (len(roots) > 2 or len(heavy_neighbors) != len(roots) + 1
            or any(not _single(mol, nidx, hydrogen.GetIdx())
                   or hydrogen.GetFormalCharge() for hydrogen in explicit_hydrogens)):
        return None, "terminal sulfinamide N substitution exceeds its ordinary valence"
    if (nitrogen.GetTotalValence() != 3
            or nitrogen.GetTotalNumHs(includeNeighbors=True) != 2 - len(roots)):
        return None, "terminal sulfinamide N does not have ordinary trivalent amide valence"
    # Recognition retains the class center even if annotation or explicit-H
    # roles make this particular output producer decline.
    from .branch_rules import collect_subtree
    branches = []
    for root in roots:
        atoms = frozenset(collect_subtree(mol, root, nidx))
        if not atoms or root not in atoms:
            return None, "N-bound carbon branch traversal is incomplete"
        branches.append(atoms)
    sites = SulfinamideSite(
        "sulfinamide", sidx, (sidx, oxo, nidx), host, nidx, oxo,
        tuple(roots), tuple(branches),
    )
    return sites, "one neutral source-shaped sulfinamide site"


def _all_sites(mol, Chem):
    from .hydrazide_components import recognize_hydrazides
    from .nitrogen_acid_derivatives import sulfonimidamide_site

    sulfur_atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 16]
    hydrazide_centers = ({site.center for site in recognize_hydrazides(mol)}
                         if sulfur_atoms else set())
    if sulfur_atoms:
        from .functional_class_queries import acylhydrazide_query
        hydrazide_centers.update(c for _, c, _, _ in mol.GetSubstructMatches(acylhydrazide_query(Chem))
                                 if mol.GetAtomWithIdx(c).GetAtomicNum() == 16
                                 and mol.GetAtomWithIdx(c).GetDegree() == 3
                                 and mol.GetAtomWithIdx(c).GetTotalValence() == 4)
    sites, failures = [], []
    for sulfur in sulfur_atoms:
        # P-66.3.0: replacing acid OH by NH-NH2 creates a class-12
        # hydrazide, not an unsupported class-11 sulfinamide. Its dedicated
        # producer owns eligibility and any annotation limitations.
        if sulfur.GetIdx() in hydrazide_centers:
            continue
        if sulfur.GetDegree() == 4 and sulfonimidamide_site(sulfur) is not None:
            continue
        site, reason = _recognize_site(mol, Chem, sulfur)
        if site is not None:
            sites.append(site)
        else:
            # A one-oxo S--N unit still creates a class obligation even if its
            # unsupported valence/annotation prevents a local site record.
            oxo_count = sum(n.GetAtomicNum() == 8 and _double(mol, sulfur.GetIdx(), n.GetIdx())
                            for n in sulfur.GetNeighbors())
            has_n = any(n.GetAtomicNum() == 7 and _single(mol, sulfur.GetIdx(), n.GetIdx())
                        for n in sulfur.GetNeighbors())
            has_c = any(n.GetAtomicNum() == 6 and _single(mol, sulfur.GetIdx(), n.GetIdx())
                        for n in sulfur.GetNeighbors())
            if oxo_count == 1 and has_n and has_c:
                failures.append(reason)
    return tuple(sites), tuple(failures)


def _residual_higher_groups(mol, Chem, sites):
    """Find source-known senior groups after projecting away S and terminal oxo."""
    from .extended_inventory import selection_inventory
    from .group_priority import (compare_functional_group_priority,
                                 functional_group_priority_facts, priority_frontier)

    removed = set()
    for site in sites:
        # Project away only the characteristic sulfur and its terminal oxo.
        # Keeping the original amide N and all N-bound branches preserves
        # higher-class functions on those arms for source-priority analysis.
        removed.update((site.center, site.oxo_oxygen))
    keep = [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in removed]
    if not keep:
        return (), None
    rw = Chem.RWMol(mol)
    for idx in sorted(removed, reverse=True):
        rw.RemoveAtom(idx)
    residual = rw.GetMol()
    try:
        Chem.SanitizeMol(residual)
    except (ValueError, RuntimeError):
        return (), None
    original_ids = tuple(keep)
    inventory = selection_inventory(residual)
    if not inventory.supported:
        return (), None
    priority = functional_group_priority_facts(sites[0], mol)
    if priority is None:
        return (), None
    candidates = []
    for group in inventory.groups:
        facts = functional_group_priority_facts(group, residual)
        if facts is None:
            # For recognized legacy groups in a strictly lower P-41 class,
            # the class table itself closes the comparison. Same-class and
            # higher/unranked groups still need semantic evidence.
            try:
                if group.seniority[0] > priority.class_rank:
                    continue
            except (AttributeError, KeyError, TypeError):
                return (), None
            return (), None
        comparison = compare_functional_group_priority(facts, priority)
        if comparison.status == "senior":
            old_center = original_ids[group.center]
            candidates.append((group, facts, old_center))
        elif comparison.status == "unknown":
            return (), None
    if not candidates:
        return (), ()
    winners = priority_frontier(candidates, lambda item: item[1])
    return tuple((group.kind, old_center, facts)
                 for group, facts, old_center in winners), frozenset(old_center for _, _, old_center in winners)


def recognize_sulfinamides(mol, Chem) -> SulfinamideRecognition:
    """Recognize source-shaped S(IV) amides and retain unsupported obligations."""
    if not any(a.GetAtomicNum() == 16 for a in mol.GetAtoms()):
        return SulfinamideRecognition("not-applicable", "no sulfur atom")
    sites, failures = _all_sites(mol, Chem)
    if not sites:
        if failures:
            return SulfinamideRecognition("deferred", "; ".join(sorted(set(failures))))
        return SulfinamideRecognition("not-applicable", "no C--S(=O)--N sulfinamide motif")
    if failures:
        return SulfinamideRecognition(
            "deferred", "; ".join(sorted(set(failures))), sites,
        )
    if len(sites) != 1:
        return SulfinamideRecognition("deferred", "multiple sulfinamide sites need a separate suffix multiplier proof", sites)
    senior, senior_centers = _residual_higher_groups(mol, Chem, sites)
    if senior_centers:
        return SulfinamideRecognition("higher-class-route",
                                      "a source-ranked higher-class group remains after isolating the sulfinamide unit",
                                      sites, frozenset(senior_centers),
                                      tuple(kind for kind, _center, _facts in senior))
    if senior is None:
        return SulfinamideRecognition(
            "deferred", "a remaining characteristic-group obligation is not source-ranked", sites,
        )
    return SulfinamideRecognition("supported", "one source-shaped sulfinamide unit", sites)


def recognize_sulfinamide_sites(mol, Chem) -> tuple[SulfinamideSite, ...]:
    """Return exact C-host site records for reusable inventory construction.

    This is a local graph recognizer, not a naming decision. Callers must also
    honor :func:`recognize_sulfinamides`' disposition so unsupported or
    higher-class obligations cannot fall through to a lower-class route.
    """
    sites, _failures = _all_sites(mol, Chem)
    return sites


def sulfinamide_characteristic_groups(mol, Chem) -> tuple[CharacteristicGroup, ...]:
    """Project raw site facts onto the shared inventory record type."""
    return tuple(
        CharacteristicGroup(site.kind, site.center, site.atoms, site.owner, site.ligand)
        for site in recognize_sulfinamide_sites(mol, Chem)
    )


def _carbon_parent_sequences(mol, Chem, site):
    """Enumerate the complete simple carbon path or saturated monocycle."""
    from .constants import ROOTS
    from .branch_rules import parent_chain_name
    from .formatting import complete_parent_name

    n_branch_atoms = set().union(*site.n_substituent_atoms) if site.n_substituent_atoms else set()
    parent_carbons = frozenset(a.GetIdx() for a in mol.GetAtoms()
                               if a.GetAtomicNum() == 6 and a.GetIdx() not in n_branch_atoms)
    if site.owner not in parent_carbons or not parent_carbons:
        return ()
    edges = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()
             if b.GetBeginAtomIdx() in parent_carbons and b.GetEndAtomIdx() in parent_carbons]
    adjacency = {idx: set() for idx in parent_carbons}
    for a, b in edges:
        if not _single(mol, a, b):
            return ()
        adjacency[a].add(b)
        adjacency[b].add(a)
    seen, stack = set(), [site.owner]
    while stack:
        idx = stack.pop()
        if idx in seen:
            continue
        seen.add(idx)
        stack.extend(adjacency[idx] - seen)
    if seen != set(parent_carbons):
        return ()
    if all(len(adjacency[idx]) <= 2 for idx in parent_carbons) and len(edges) == len(parent_carbons) - 1:
        # The bounded acyclic parent is an unbranched path. Its two source
        # orientations are compared using suffix and prefix facts first.
        ends = [idx for idx in parent_carbons if len(adjacency[idx]) <= 1]
        if len(parent_carbons) == 1:
            paths = ((site.owner,),)
        elif len(ends) == 2:
            def trace(start):
                path, prior, current = [], None, start
                while True:
                    path.append(current)
                    nxt = [x for x in adjacency[current] if x != prior]
                    if not nxt:
                        break
                    prior, current = current, nxt[0]
                return tuple(path)
            paths = (trace(ends[0]), trace(ends[1]))
        else:
            return ()
        result = []
        for path in paths:
            if site.owner not in path:
                continue
            parent = complete_parent_name(parent_chain_name(mol, path))
            if parent:
                result.append(("chain", parent, path))
        return tuple(result)
    if len(edges) != len(parent_carbons) or any(len(adjacency[idx]) != 2 for idx in parent_carbons):
        return ()
    rings = [tuple(ring) for ring in mol.GetRingInfo().AtomRings()
             if set(ring) == set(parent_carbons)]
    if len(rings) != 1 or len(rings[0]) != len(parent_carbons):
        return ()
    if any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in parent_carbons):
        return ()  # Benzene/higher aromatic suffix names are a separate rule.
    if len(parent_carbons) not in ROOTS:
        return ()
    core = rings[0]
    neighbors = adjacency[site.owner]
    result = []
    for start in sorted(neighbors):
        path, previous, current = [site.owner], site.owner, start
        while current != site.owner:
            path.append(current)
            nxt = next(i for i in adjacency[current] if i != previous)
            previous, current = current, nxt
        if len(path) == len(parent_carbons):
            result.append(("ring", "cyclo" + ROOTS[len(path)] + "ane", tuple(path)))
    return tuple(result)


def _candidate(mol, Chem, site, parent_kind, parent, path):
    core_atoms = (site.center, site.oxo_oxygen, site.ligand)
    if (any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms())
            or any(mol.GetAtomWithIdx(i).GetIsotope() for i in core_atoms)
            or any(int(mol.GetAtomWithIdx(i).GetChiralTag()) for i in core_atoms)
            or any(int(bond.GetStereo()) for bond in mol.GetBonds())):
        return None
    locants = {idx: str(position) for position, idx in enumerate(path, 1)}
    suffix_locant = locants.get(site.owner)
    if suffix_locant is None:
        return None
    prefixes, branch_atoms, nature_by_name = [], set(), {}
    for root, atoms in zip(site.n_substituents, site.n_substituent_atoms):
        text, rendered_atoms = mapped_branch(mol, Chem, root, site.ligand)
        if (not text or frozenset(rendered_atoms) != atoms
                or branch_atoms & set(atoms) or set(path) & set(atoms)):
            return None
        prefixes.append(("N", text))
        branch_atoms.update(atoms)
        from .organyl_prefix_components import organyl_prefix_nature
        nature = organyl_prefix_nature(mol, root, atoms, text)
        if nature not in {"simple", "compound"}:
            return None
        if text in nature_by_name and nature_by_name[text] != nature:
            return None
        nature_by_name[text] = nature
    mapping = hydrogen_bindings(mol, locants)
    embedded = frozenset(set(site.atoms) | branch_atoms)
    if not fully_covered(mol, mapping, embedded):
        return None
    component = MappedComponent(
        parent, tuple(mapping.items()), frozenset(mapping), tuple(prefixes), embedded,
        systematic_parent=True, nitrogen_site_locants=bool(prefixes),
        prefix_natures=tuple(nature_by_name.items()),
    )
    from .sulfur_locant_policy import parent_isotope_locant_evidence
    isotope_locants = parent_isotope_locant_evidence(mol, Chem, component)
    if isotope_locants is None:
        return None
    suffix_locant_text = sulfur_suffix_locants(
        count=1, locants=(int(suffix_locant),), cyclic=parent_kind == "ring",
        path_length=len(path), prefix_count=len(prefixes), parent_prefix_count=0,
        force_locants=isotope_locants.required,
    )
    ending = f"{parent}{suffix_locant_text}{_SULFINAMIDE_SUFFIX}"
    component = replace(component, parent=ending)
    name = decorate_component(mol, Chem, component)
    if not name:
        return None
    group = CharacteristicGroup(site.kind, site.center, site.atoms, site.owner, site.ligand)
    from .group_priority import functional_group_priority_facts
    priority_facts = functional_group_priority_facts(group, mol)
    if (priority_facts is None or priority_facts.corresponding_acid is None
            or priority_facts.corresponding_acid.p42_1_family != "sulfinic"):
        return None
    role_rows = [
        ("sulfinamide-parent", frozenset(mapping), tuple(mapping.items())),
        ("sulfinamide-suffix-unit", frozenset(site.atoms), ()),
    ]
    if branch_atoms:
        role_rows.append(("sulfinamide-N-substituents", frozenset(branch_atoms), ()))
    role_atoms = [row[1] for row in role_rows]
    expected = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    if (any(role_atoms[i] & role_atoms[j]
            for i in range(len(role_atoms)) for j in range(i + 1, len(role_atoms)))
            or frozenset().union(*role_atoms) != expected):
        return None
    facts = graph_parent_facts(
        mol, path, locants, principal_class=(11,), suffix_centers=(site.center,),
        suffix_locants=(suffix_locant,), prefixes=tuple(prefixes),
        kind=parent_kind, name=name, allow_annotations=True,
    )
    if facts is None:
        return None
    # The C--S(=O)--N unit is represented as a suffix, not as a nonstandard
    # substituent on the selected carbon parent.
    facts = replace(facts, substituent_nonstandard_key=())
    proposal = SulfinamideProposal(
        name=name, site=site, parent_atoms=tuple(path),
        atom_locants=tuple(mapping.items()), equivalent_maps=(tuple(mapping.items()),),
        component=component, component_roles=tuple(role_rows),
        priority_facts=priority_facts,
    )
    return proposal, facts


def sulfinamide_proposals(mol, Chem) -> tuple[SulfinamideProposal, ...]:
    """Return a complete mapped proposal for the bounded C-host suffix rule."""
    recognition = recognize_sulfinamides(mol, Chem)
    if recognition.status != "supported" or len(recognition.sites) != 1:
        return ()
    site = recognition.sites[0]
    candidates = []
    for kind, parent, path in _carbon_parent_sequences(mol, Chem, site):
        candidate = _candidate(mol, Chem, site, kind, parent, path)
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        return ()
    frontier = parent_frontier(candidates, lambda row: row[1])
    if not frontier or len({proposal.name for proposal, _facts in frontier}) != 1:
        return ()
    chosen, _facts = frontier[0]
    equivalent = tuple(dict.fromkeys(proposal.atom_locants for proposal, _facts in frontier))
    return (replace(chosen, equivalent_maps=equivalent),)


def sulfinamide_decision(mol, Chem) -> MethodDecision | None:
    """Adapt one fully covered source-ranked proposal to the public method API."""
    proposals = sulfinamide_proposals(mol, Chem)
    if len(proposals) != 1:
        from .branched_sulfinamide_methods import branched_sulfinamide_decision
        return branched_sulfinamide_decision(mol, Chem)
    try:
        return proposals[0].as_method_decision()
    except ValueError:
        return None
