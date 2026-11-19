"""Mapped proposals for sulfur suffixes attached through parent nitrogen.

This module recognizes only neutral, unisotopic sulfonamide/sulfinamide units
whose host is a mapped saturated parent nitrogen. It proposes complete names
for the bounded mapped ring parents below; it does not choose among cross-route
parent candidates.
"""

from dataclasses import dataclass
from dataclasses import replace
from types import SimpleNamespace

from .molecule_iteration import iter_atoms, iter_bonds


@dataclass(frozen=True, slots=True)
class HeterohostSulfurProposal:
    kind: str
    sulfur: int
    host: int
    host_locant: str
    ligand: int
    oxo_oxygens: tuple[int, ...]
    hydroxy_oxygen: int | None
    n_substituents: tuple[int, ...]
    owned_atoms: frozenset[int]


@dataclass(frozen=True, slots=True)
class HeterohostSulfurParentProposal:
    name: str
    route: str
    kind: str
    principal_centers: frozenset[int]
    parent_atoms: tuple[int, ...]
    atom_locants: tuple[tuple[int, str], ...]
    suffix_locants: tuple[str, ...]
    facts: object
    complete: bool
    component_roles: tuple
    sulfur_unit: HeterohostSulfurProposal
    construction: object = None


_O_COUNTS = {"sulfonamide": 2, "sulfinamide": 1}


def sulfur_amide_rank(kind):
    """P-41 class-11 ranks ordered by their corresponding acid classes."""
    if kind == "sulfonamide":
        return (11, 1)
    if kind == "sulfinamide":
        # P-41 orders sulfonic before sulfinic acids. Keep sulfinamide between
        # the existing sulfonamide and carbonic-acid-derived urea rank.
        return (11, 1, 1)
    return None


def _is_single(bond):
    return bond is not None and bond.GetBondTypeAsDouble() == 1


def _is_double(bond):
    return bond is not None and bond.GetBondTypeAsDouble() == 2


def _isotope_free_unit(mol, atom_indices):
    for idx in atom_indices:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetIsotope():
            return False
        if any(n.GetAtomicNum() == 1 and n.GetIsotope()
               for n in atom.GetNeighbors()):
            return False
    return True


def _host_is_mapped_ring_nitrogen(mol, host_idx, core, sulfur_idx):
    host = mol.GetAtomWithIdx(host_idx)
    if host.GetAtomicNum() != 7 or host.GetFormalCharge() or host.GetIsotope():
        return False
    if host.GetIsAromatic() or host.GetNumRadicalElectrons() or not host.IsInRing():
        return False
    if not _is_single(mol.GetBondBetweenAtoms(host_idx, sulfur_idx)):
        return False
    core_neighbors = [n.GetIdx() for n in host.GetNeighbors()
                      if n.GetIdx() in core]
    other_neighbors = [n.GetIdx() for n in host.GetNeighbors()
                       if n.GetIdx() != sulfur_idx and n.GetIdx() not in core]
    # A neutral saturated ring amine has two parent bonds; attachment to sulfur
    # replaces its implicit/explicit hydrogen and completes the neutral valence.
    if len(core_neighbors) != 2 or other_neighbors:
        return False
    if any(not _is_single(mol.GetBondBetweenAtoms(host_idx, i))
           for i in core_neighbors):
        return False
    return host.GetDegree() == 3 and host.GetTotalValence() == 3


def _terminal_oxo_oxygens(mol, sulfur_idx):
    result = []
    sulfur = mol.GetAtomWithIdx(sulfur_idx)
    for oxygen in sulfur.GetNeighbors():
        if oxygen.GetAtomicNum() != 8:
            continue
        bond = mol.GetBondBetweenAtoms(sulfur_idx, oxygen.GetIdx())
        if _is_double(bond) and oxygen.GetDegree() == 1 and not oxygen.GetFormalCharge():
            result.append(oxygen.GetIdx())
    return tuple(sorted(result))


def _terminal_amide_n(mol, sulfur_idx, parent_core):
    sulfur = mol.GetAtomWithIdx(sulfur_idx)
    found = []
    for nitrogen in sulfur.GetNeighbors():
        if nitrogen.GetAtomicNum() != 7 or nitrogen.GetFormalCharge() or nitrogen.GetIsotope():
            continue
        if nitrogen.GetIdx() in parent_core:
            continue
        if not _is_single(mol.GetBondBetweenAtoms(sulfur_idx, nitrogen.GetIdx())):
            continue
        carbon_substituents = []
        valid = True
        for neighbor in nitrogen.GetNeighbors():
            if neighbor.GetIdx() == sulfur_idx:
                continue
            if neighbor.GetAtomicNum() != 6 or neighbor.GetFormalCharge():
                valid = False
                break
            if not _is_single(mol.GetBondBetweenAtoms(nitrogen.GetIdx(), neighbor.GetIdx())):
                valid = False
                break
            carbon_substituents.append(neighbor.GetIdx())
        if valid and len(carbon_substituents) <= 2 and nitrogen.GetDegree() <= 3:
            found.append((nitrogen.GetIdx(), tuple(sorted(carbon_substituents))))
    return found[0] if len(found) == 1 else (None, ())


def heterohost_sulfur_proposals(mol, core, atom_locants):
    """Return bounded suffix proposals attached through a mapped ring nitrogen.

    Args:
        mol: Original, unsliced RDKit molecule.
        core: Original atom indices belonging to a candidate parent hydride.
        atom_locants: Mapping from original parent atom index to its rendered
            numeric locant (for example ``{n_index: "1"}``).

    The function preserves original indices and returns no proposal if any
    sulfur/O/N atom or attached hydrogen in the suffix unit is isotopically
    labelled. Unsupported charges, host valences, attachment classes, and
    ambiguous sulfur neighborhoods are declined.
    """
    core = frozenset(core)
    if not core or not set(atom_locants).issuperset(core):
        return ()
    output = []
    for sulfur in iter_atoms(mol):
        if sulfur.GetAtomicNum() != 16 or sulfur.GetFormalCharge() or sulfur.GetIsotope():
            continue
        sulfur_idx = sulfur.GetIdx()
        hosts = [n.GetIdx() for n in sulfur.GetNeighbors()
                 if n.GetIdx() in core and n.GetAtomicNum() == 7]
        if len(hosts) != 1 or not _host_is_mapped_ring_nitrogen(
                mol, hosts[0], core, sulfur_idx):
            continue
        host_idx = hosts[0]
        oxos = _terminal_oxo_oxygens(mol, sulfur_idx)
        n_idx, n_substituents = _terminal_amide_n(mol, sulfur_idx, core)
        kind = None
        ligand = None
        if len(oxos) in (1, 2) and n_idx is not None:
            kind = "sulfonamide" if len(oxos) == 2 else "sulfinamide"
            ligand = n_idx
        if kind is None or len(oxos) != _O_COUNTS[kind]:
            continue
        # Require the whole neutral sulfur coordination sphere to be accounted
        # for: one parent N, the suffix ligand, and the proved oxo set.
        expected = {host_idx, ligand, *oxos}
        if {n.GetIdx() for n in sulfur.GetNeighbors()} != expected:
            continue
        owned = expected
        if not _isotope_free_unit(mol, owned):
            continue
        if any(int(mol.GetAtomWithIdx(i).GetChiralTag()) for i in owned):
            continue  # Characteristic S/N stereochemistry has no proved locant here.
        # Terminal sulfonamide N substitution can be ordinary carbon only.
        if kind.endswith("amide") and len(n_substituents) > 2:
            continue
        output.append(HeterohostSulfurProposal(
            kind=kind,
            sulfur=sulfur_idx,
            host=host_idx,
            host_locant=str(atom_locants[host_idx]),
            ligand=ligand,
            oxo_oxygens=oxos,
            hydroxy_oxygen=None,
            n_substituents=n_substituents,
            owned_atoms=frozenset(owned),
        ))
    return tuple(output)


def _n_substituent_prefixes(mol, Chem, proposal):
    if not proposal.n_substituents:
        return (), frozenset(), ()
    from .component_mapping import mapped_branch

    names = []
    owned = set()
    roles = []
    for root in proposal.n_substituents:
        name, atoms = mapped_branch(mol, Chem, root, proposal.ligand)
        if not name or root not in atoms or owned & set(atoms):
            return None
        # Branch decoration is supported by its own typed producer; the
        # heterohost suffix renderer does not claim to decorate N-branch atoms.
        if any(mol.GetAtomWithIdx(i).GetIsotope() or
               int(mol.GetAtomWithIdx(i).GetChiralTag()) for i in atoms):
            return None
        names.append(name)
        owned.update(atoms)
        roles.append(("amide-n-substituent", frozenset(atoms), ()))
    if len(names) == 1:
        return (("N", names[0]),), frozenset(owned), tuple(roles)
    elif names[0] == names[1]:
        # P-66.1.1.3.1.1 allows ordinary N,N substitution; explicit N-primes
        # are reserved for geminal cases under P-66.1.1.3.1.2.
        return (("N", names[0]), ("N", names[1])), frozenset(owned), tuple(roles)
    else:
        return tuple(("N", name) for name in names), frozenset(owned), tuple(roles)


def heterohost_sulfur_parent_proposals(mol, Chem, *, core=None):
    """Render mapped saturated heteromonocyclic sulfonamide parents.

    Name strings are complete for this bounded route. Original atom indices
    remain in ``parent_atoms``, ``atom_locants`` and ``principal_centers`` for
    cross-route comparison and graph coverage. Parent numbering alternatives
    come only from the shared mapped-ring skeleton generator; suffix locants
    break its alternatives according to the P-66 requirement to locate the
    suffix on the parent hydride.
    """
    from .mapped_ring_parents import ring_systems, skeletons
    from .component_mapping import MappedComponent, decorate_component

    results = []
    if (mol.GetStereoGroups() or any(a.GetFormalCharge() or a.GetNumRadicalElectrons()
            for a in iter_atoms(mol))):
        return ()
    if not mol.GetNumAtoms():
        return ()
    for ring in ring_systems(mol):
        if core is not None and frozenset(core) != ring:
            continue
        # Skeleton discovery supplies reference hydrides. Partially
        # hydrogenated/fused rings need the separate P-58 accommodation
        # operation before a suffix can be added to that reference name.
        if (sum(set(cycle) <= ring for cycle in mol.GetRingInfo().AtomRings()) != 1
                or any(b.GetBondTypeAsDouble() != 1 for b in iter_bonds(mol)
                       if {b.GetBeginAtomIdx(), b.GetEndAtomIdx()} <= ring)):
            continue
        for proposal in heterohost_sulfur_proposals(mol, ring, {
                idx: "" for idx in ring}):
            if proposal.kind not in {"sulfonamide", "sulfinamide"}:
                continue
            n_result = _n_substituent_prefixes(mol, Chem, proposal)
            if n_result is None:
                continue
            n_prefix, n_atoms, n_roles = n_result
            for skeleton in skeletons(mol, Chem, ring):
                parent, sequence = skeleton[0], skeleton[1]
                if set(sequence) != set(ring) or proposal.host not in sequence:
                    continue
                locants = dict(zip(sequence, skeleton[2]))
                host_locant = locants[proposal.host]
                from .component_mapping import mapped_branch
                carbon_prefixes = []
                carbon_atoms = set()
                carbon_roles = []
                valid_branches = True
                for parent_atom in sequence:
                    for neighbor in mol.GetAtomWithIdx(parent_atom).GetNeighbors():
                        idx = neighbor.GetIdx()
                        if idx in ring or idx == proposal.sulfur:
                            continue
                        branch_name, branch_atoms = mapped_branch(mol, Chem, idx, parent_atom)
                        if (not branch_name or idx not in branch_atoms or
                                carbon_atoms & set(branch_atoms)):
                            valid_branches = False
                            break
                        carbon_prefixes.append((locants[parent_atom], branch_name))
                        carbon_atoms.update(branch_atoms)
                        carbon_roles.append(("parent-carbon-substituent",
                                             frozenset(branch_atoms), ()))
                    if not valid_branches:
                        break
                if not valid_branches:
                    continue
                suffix = "sulfonamide" if proposal.kind == "sulfonamide" else "sulfinamide"
                parent_name = f"{parent}-{host_locant}-{suffix}"
                suffix_atoms = frozenset({proposal.sulfur, proposal.ligand,
                                          *proposal.oxo_oxygens})
                component_atoms = frozenset(set(sequence) | set(suffix_atoms))
                embedded_atoms = frozenset(set(n_atoms) | carbon_atoms)
                required_heavy = frozenset(a.GetIdx() for a in iter_atoms(mol)
                                           if a.GetAtomicNum() > 1)
                if (set(sequence) & set(suffix_atoms) or
                        component_atoms & embedded_atoms or
                        component_atoms | embedded_atoms != required_heavy):
                    continue
                atom_locants = (tuple((idx, locants[idx]) for idx in sequence) +
                                ((proposal.sulfur, ""),) +
                                tuple((idx, "") for idx in proposal.oxo_oxygens) +
                                ((proposal.ligand, "N"),))
                component = MappedComponent(
                    parent=parent_name,
                    atom_locants=atom_locants,
                    atoms=component_atoms,
                    embedded_atoms=embedded_atoms,
                    prefixes=tuple(carbon_prefixes) + tuple(n_prefix),
                    systematic_parent=True,
                    nitrogen_site_locants=True,
                )
                name = decorate_component(mol, Chem, component)
                if not name:
                    continue
                rank = sulfur_amide_rank(proposal.kind)
                record = SimpleNamespace(
                    parent=parent, parent_atoms=tuple(sequence),
                    atom_locants=tuple((idx, locants[idx]) for idx in sequence),
                    principal_centers=frozenset({proposal.sulfur}),
                    suffix_locants=(host_locant,), prefixes=component.prefixes,
                    name=name, indicated_h_locants=(), method="heterohost-sulfur",
                    construction=component,
                )
                from .mapped_ring_parents import parent_facts
                facts = parent_facts(mol, record, rank)
                if facts is None:
                    continue
                # The recognized sulfur suffix is part of the principal
                # group, not a substituent evaluated by P-45.3. The generic
                # parent-facts helper sees those S/O atoms in the whole
                # molecule and conservatively leaves this criterion unknown.
                # Prove ordinary substituent chemistry on the remainder,
                # then supply the absent-substituent key explicitly.
                ordinary_substituents = all(
                    mol.GetAtomWithIdx(idx).GetAtomicNum() in
                    {1, 6, 7, 8, 9, 17, 35, 53}
                    for idx in required_heavy - suffix_atoms
                )
                if ordinary_substituents:
                    facts = replace(facts, substituent_nonstandard_key=())
                roles = (("parent-hydride", frozenset(sequence),
                          tuple((idx, locants[idx]) for idx in sequence)),
                         (proposal.kind, suffix_atoms,
                          ((proposal.sulfur, ""),) +
                          tuple((idx, "") for idx in proposal.oxo_oxygens) +
                          ((proposal.ligand, "N"),))) + tuple(n_roles) + tuple(carbon_roles)
                results.append(HeterohostSulfurParentProposal(
                    name=name, route="heterohost-sulfur-ring", kind=proposal.kind,
                    principal_centers=frozenset({proposal.sulfur}),
                    parent_atoms=tuple(sequence),
                    atom_locants=tuple((idx, locants[idx]) for idx in sequence),
                    suffix_locants=(host_locant,),
                    facts=facts, complete=True, component_roles=roles,
                    sulfur_unit=proposal, construction=component,
                ))
    return tuple(results)


def _has_more_senior_group(mol, Chem, sulfur_unit):
    """Inspect recognized groups after removing this sulfur suffix privately.

    Inventory support is deliberately not used as an admission flag: an
    unrelated unsupported sulfur may make an inventory incomplete while its
    already-recognized acid/ester/amide groups remain useful evidence.
    """
    from .extended_inventory import selection_inventory

    remove = {sulfur_unit.sulfur, sulfur_unit.ligand,
              *sulfur_unit.oxo_oxygens}
    editable = Chem.RWMol(mol)
    host_idx = sulfur_unit.host
    for idx in sorted(remove, reverse=True):
        editable.RemoveAtom(idx)
    capped_host = host_idx - sum(idx < host_idx for idx in remove)
    # Removing S leaves the original ring N as its neutral parent-hydride
    # form. Restore its one parent-hydride hydrogen explicitly so the shared
    # inventory can inspect other groups on the molecule.
    host = editable.GetAtomWithIdx(capped_host)
    host.SetNoImplicit(True)
    host.SetNumExplicitHs(1)
    capped = editable.GetMol()
    try:
        Chem.SanitizeMol(capped)
        inventory = selection_inventory(capped)
    except (ValueError, RuntimeError):
        # A graph the shared inventory cannot inspect safely stays outside the
        # bounded route; do not assume it lacks a senior group.
        return True
    rank = sulfur_amide_rank(
        "sulfonamide" if len(sulfur_unit.oxo_oxygens) == 2 else "sulfinamide")
    return any(group.seniority < rank for group in inventory.groups)


def heterohost_sulfur_decision(mol, Chem):
    """Return one unambiguous mapped suffix decision, or ``None``.

    All viable mapped-ring proposals are compared through the shared P-44/P-45
    comparator. A tie with different rendered names is unresolved and declines.
    """
    from .naming_methods import MethodDecision
    from .parent_selection import compare_parents, parent_frontier

    if (not mol.GetNumAtoms() or mol.GetStereoGroups() or
            any(a.GetFormalCharge() or a.GetNumRadicalElectrons()
                for a in iter_atoms(mol))):
        return None
    candidates = heterohost_sulfur_parent_proposals(mol, Chem)
    candidates = tuple(candidate for candidate in candidates
                       if not _has_more_senior_group(mol, Chem,
                                                     candidate.sulfur_unit))
    if not candidates:
        return None
    frontier = tuple(parent_frontier(candidates, lambda item: item.facts))
    if not frontier:
        return None
    names = {item.name for item in frontier}
    if len(names) != 1:
        return None
    # Confirm the shared comparator sees no surviving decisive disagreement.
    for i, left in enumerate(frontier):
        for right in frontier[i + 1:]:
            comparison = compare_parents(left.facts, right.facts)
            if comparison.preference not in {0, None}:
                return None
    chosen = frontier[0]
    all_heavy = frozenset(a.GetIdx() for a in iter_atoms(mol)
                          if a.GetAtomicNum() > 1)
    role_atoms = [idx for _, atoms, _ in chosen.component_roles for idx in atoms]
    if len(role_atoms) != len(set(role_atoms)) or frozenset(role_atoms) != all_heavy:
        return None
    return MethodDecision(
        name=chosen.name,
        method="substitutive",
        principal_centers=chosen.principal_centers,
        atoms=all_heavy,
        rule="P-41/P-44/P-45/P-66.1.1.2",
        atom_locants=chosen.construction.atom_locants,
        component_roles=chosen.component_roles,
    )
