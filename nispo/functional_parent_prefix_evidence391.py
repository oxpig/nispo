"""Pre-render ownership evidence for prefixes on functional parent proposals.

This module deliberately does not select between parent proposals and does
not render names.  It records candidate-local graph facts and exact atom
ownership so a later P-44/P-45 consumer can distinguish a proved empty prefix
set from an unresolved prefix construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .molecule_iteration import iter_atoms, iter_bonds
from .molecule_cache import exact_graph_computation


class PrefixEvidenceStatus(str, Enum):
    COMPLETE = "complete"
    PENDING = "pending"
    OUTSIDE_SCOPE = "outside_scope"


@dataclass(frozen=True, slots=True)
class FunctionalPrefixUnitEvidence:
    """One candidate-local directly attached role or detached component.

    ``atoms`` are atoms owned by the prefix unit and are disjoint from the
    candidate parent and principal suffix atoms.  ``source_group_atoms`` may
    include a parent host (for example, the carbon in an alcohol group); it
    preserves the full source-recognized group without double-owning that
    host.  The component is a graph unit, not a claim about P-45 prefix count.
    """

    kind: str
    host: int
    root: int | None
    atoms: tuple[int, ...]
    source_group_atoms: tuple[tuple[int, ...], ...]
    source_group_identities: tuple[tuple, ...]
    source_rules: tuple[str, ...]
    parent_locant: int
    role_count: int
    rendered_name: str | None = None
    # Present only for a detached functionalized carbon tree. These are the
    # live source-ordered P-46 maxima. The mapped-arm join may supply real
    # nested tokens required by the last citation criterion.
    p46_frontier: tuple = ()


@dataclass(frozen=True, slots=True)
class FunctionalParentPrefixEvidence391:
    status: PrefixEvidenceStatus
    parent_atoms: tuple[int, ...]
    selected_principal_identities: tuple[tuple, ...]
    covered_principal_identities: tuple[tuple, ...]
    omitted_principal_identities: tuple[tuple, ...]
    principal_suffix_atoms: tuple[int, ...]
    p44_principal_site_count: int | None
    p44_skeletal_atom_count: int | None
    # Candidate numbering is an observation attached to this proposal only.
    # It has not been established as preferred among different parents.
    candidate_suffix_locants: tuple[int, ...] | None
    prefix_units: tuple[FunctionalPrefixUnitEvidence, ...] | None
    prefix_component_count: int | None
    prefix_attachment_locants: tuple[int, ...] | None
    name_dependent_citation_facts: None
    original_graph_signature: tuple
    rules: tuple[str, ...]
    reason: str

    # Job391 proves one proposal's partition. It never certifies that the
    # caller supplied the global parent-candidate universe.
    parent_candidate_universe_proved: bool = False
    mapped_branch_tokens: bool = False


_SUPPORTED_PRINCIPAL_KINDS = {"acid", "amide", "nitrile"}
_SUPPORTED_PREFIX_GROUPS = {"alcohol", "amine", "thiol", "nitrile", "ketone", "aldehyde"}


def _identity(group):
    return (group.kind, int(group.center), tuple(int(i) for i in group.atoms),
            int(group.owner), int(group.ligand))


def _signature(mol):
    # Numbered proposals on one input share this exact graph observation.
    # Native annotations, atom maps and temporary modes remain in the cache
    # identity. Validation below still rebuilds evidence on a fresh graph.
    from rdkit import Chem
    return exact_graph_computation(mol, Chem, 'functional_prefix_graph_signature',
                                   (), lambda: _uncached_signature(mol))


def _uncached_signature(mol):
    atoms = tuple(sorted((
        atom.GetIdx(), atom.GetAtomMapNum(), atom.GetAtomicNum(), atom.GetIsotope(),
        atom.GetFormalCharge(), atom.GetNumRadicalElectrons(),
        int(atom.GetChiralTag()), atom.GetIsAromatic(),
        atom.GetTotalValence(), atom.GetNumExplicitHs(), atom.GetNoImplicit(),
    ) for atom in iter_atoms(mol)))
    bond_rows = []
    for bond in iter_bonds(mol):
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if left > right:
            left, right = right, left
        bond_rows.append((
            left, right, str(bond.GetBondType()), bond.GetIsAromatic(),
            int(bond.GetStereo()), int(bond.GetBondDir()),
        ))
    bonds = tuple(sorted(bond_rows))
    groups = getattr(mol, "GetStereoGroups", lambda: ())()
    from rdkit.Chem.rdchem import StereoGroup_vect
    if type(groups) is StereoGroup_vect and len(groups) == 0:
        # Exhausting this native Boost vector invokes its expensive indexed
        # exception path even when it is empty. Its length already proves
        # the same empty observation; custom containers retain iteration.
        stereo_groups = ()
    else:
        stereo_groups = tuple(sorted((
            int(group.GetGroupType()),
            tuple(sorted(atom.GetIdx() for atom in group.GetAtoms())),
        ) for group in groups))
    return atoms, bonds, stereo_groups


def _pending(parent_atoms, signature, reason, *, inventory=None, p44_count=None,
             p44_size=None, suffix_atoms=(), suffix_locants=None,
             covered=(), omitted=()):
    return FunctionalParentPrefixEvidence391(
        PrefixEvidenceStatus.PENDING, tuple(parent_atoms),
        tuple(_identity(g) for g in getattr(inventory, "principal", ())),
        tuple(covered), tuple(omitted), tuple(sorted(set(suffix_atoms))),
        p44_count, p44_size, suffix_locants, None, None, None, None,
        signature, ("P-41", "P-43", "P-44", "P-45.2"), reason,
    )


def _outside(parent_atoms, signature, reason):
    return FunctionalParentPrefixEvidence391(
        PrefixEvidenceStatus.OUTSIDE_SCOPE, tuple(parent_atoms), (), (), (), (),
        None, None, None, None, None, None, None, signature,
        ("P-41", "P-43"), reason,
    )


def _tree_component(mol, start, blocked):
    """Return the heavy-atom component beyond one parent attachment edge."""
    seen = set()
    stack = [start]
    while stack:
        idx = stack.pop()
        if idx in blocked or idx in seen:
            continue
        seen.add(idx)
        stack.extend(n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                     if n.GetAtomicNum() > 1 and n.GetIdx() not in blocked)
    return frozenset(seen)


def _group_host(group, mol):
    """Return the attachment host for an inventoried nonprincipal group."""
    if group.kind in {"alcohol", "ketone", "aldehyde", "acid", "amide"}:
        return int(group.center)
    if group.kind in {"nitrile", "ester", "acid-halide", "sulfonamide"}:
        return int(group.owner)
    if group.kind == "amine":
        neighbors = tuple(n.GetIdx() for n in mol.GetAtomWithIdx(group.center).GetNeighbors()
                          if n.GetAtomicNum() == 6)
        return neighbors[0] if len(neighbors) == 1 else None
    if group.kind == "thiol":
        return int(group.owner)
    return None


def _principal_suffix_shape_is_included(mol, group, parent_set):
    """Check that this bounded P-43 suffix unit is truly on the C parent."""
    center = int(group.center)
    atom = mol.GetAtomWithIdx(center)
    if center not in parent_set or atom.GetAtomicNum() != 6:
        return False
    if group.kind == "nitrile":
        triple_n = tuple(n.GetIdx() for n in atom.GetNeighbors()
                         if n.GetAtomicNum() == 7
                         and mol.GetBondBetweenAtoms(center, n.GetIdx()).GetBondTypeAsDouble() == 3)
        return (len(triple_n) == 1 and set(group.atoms) == {center, triple_n[0]}
                and len(tuple(mol.GetAtomWithIdx(triple_n[0]).GetNeighbors())) == 1)
    carbonyl_oxygens = tuple(n.GetIdx() for n in atom.GetNeighbors()
                             if n.GetAtomicNum() == 8
                             and mol.GetBondBetweenAtoms(center, n.GetIdx()).GetBondTypeAsDouble() == 2)
    if len(carbonyl_oxygens) != 1:
        return False
    if group.kind == "acid":
        single_oxygens = tuple(n.GetIdx() for n in atom.GetNeighbors()
                               if n.GetAtomicNum() == 8
                               and mol.GetBondBetweenAtoms(center, n.GetIdx()).GetBondTypeAsDouble() == 1)
        return (len(single_oxygens) == 1
                and set(group.atoms) == {center, carbonyl_oxygens[0], single_oxygens[0]})
    if group.kind == "amide":
        single_nitrogens = tuple(n.GetIdx() for n in atom.GetNeighbors()
                                 if n.GetAtomicNum() == 7
                                 and mol.GetBondBetweenAtoms(center, n.GetIdx()).GetBondTypeAsDouble() == 1)
        return (len(single_nitrogens) == 1
                and set(group.atoms) == {center, carbonyl_oxygens[0], single_nitrogens[0]})
    return False


def _ordinary_saturated_carbon_component(mol, Chem, component):
    """Prove a neutral saturated C-only tree with a single-bond attachment."""
    component = set(component)
    if not component:
        return False
    edge_count = 0
    attachment_count = 0
    for idx in component:
        atom = mol.GetAtomWithIdx(idx)
        if (atom.GetAtomicNum() != 6 or atom.GetFormalCharge()
                or atom.GetIsotope() or atom.GetNumRadicalElectrons()
                or atom.GetIsAromatic() or atom.IsInRing()
                or atom.GetTotalValence() != 4):
            return False
        for neighbor in atom.GetNeighbors():
            other = neighbor.GetIdx()
            if neighbor.GetAtomicNum() == 1:
                if neighbor.GetIsotope():
                    return False
                continue
            bond = mol.GetBondBetweenAtoms(idx, other)
            if bond.GetBondType() != Chem.BondType.SINGLE:
                return False
            if other in component:
                if idx < other:
                    edge_count += 1
            else:
                attachment_count += 1
    return edge_count == len(component) - 1 and attachment_count == 1


def _source_group_units(mol, groups, parent_set, principal_identities,
                        principal_atoms):
    """Map directly parent-attached nonprincipal groups without rendering."""
    direct = []
    claimed = set(principal_atoms)
    for group in groups:
        host = _group_host(group, mol)
        if host not in parent_set:
            continue
        # Principal groups are handled separately as suffix roles.
        if _identity(group) in principal_identities:
            continue
        if group.kind not in _SUPPORTED_PREFIX_GROUPS:
            return None, f"unsupported source-inventoried role {group.kind!r} touches the parent"
        if group.kind in {"ketone", "aldehyde"}:
            from .functionalized_carbon_prefixes import subordinate_carbonyl_oxygen
            if subordinate_carbonyl_oxygen(mol, group) is None:
                return None, "subordinate carbonyl lacks an ordinary unlabelled oxygen role"
        source_atoms = tuple(sorted(set(int(i) for i in group.atoms)))
        exclusive = tuple(i for i in source_atoms if i not in parent_set)
        if not exclusive:
            return None, "direct nonprincipal group has no separately owned atoms"
        if any(i in claimed for i in exclusive):
            return None, "direct prefix group overlaps principal suffix ownership"
        # Every exclusive source-group atom must attach to the designated host
        # through its own recognized group graph; group inventory is the source
        # identity, while this check establishes its exact local attachment.
        if not any(mol.GetBondBetweenAtoms(host, atom) is not None for atom in exclusive):
            return None, "direct prefix group lacks a mapped host attachment"
        direct.append(FunctionalPrefixUnitEvidence(
            kind=group.kind, host=host, root=None, atoms=exclusive,
            source_group_atoms=(source_atoms,),
            source_group_identities=(_identity(group),),
            source_rules=(("P-64.5.1",) if group.kind == "ketone" else
                          ("P-66.6.1.3",) if group.kind == "aldehyde" else ("P-35.2",)),
            parent_locant=None,
            role_count=1,
        ))
        claimed.update(exclusive)
    return tuple(direct), ""


def build_functional_parent_prefix_evidence391(
    mol, Chem, parent_atoms, principal_groups, *, suffix_mode="chain-included",
    resolve_branch_tokens=False,
):
    """Build source-role and detachable-prefix graph evidence for one proposal.

    ``parent_atoms`` is an ordered original-index path from a proposal, not a
    winner. The returned suffix locants are proposal-local observations.
    This function does not render the enclosing parent, compare parent
    proposals, or declare P-45/P-51 method eligibility. An enclosing writer
    may opt into exact mapped arm tokens to close P-46.1(m); graph-only
    consumers keep the historical default.
    """
    return _build_functional_parent_prefix_evidence391(
        mol, Chem, parent_atoms, principal_groups, suffix_mode=suffix_mode,
        resolve_branch_tokens=resolve_branch_tokens,
    )


def _build_functional_parent_prefix_evidence391(
    mol, Chem, parent_atoms, principal_groups, *, suffix_mode="chain-included",
    resolve_branch_tokens=False, observation=None,
):
    from .extended_inventory import selection_inventory
    from .functionalized_carbon_prefixes import (
        functionalized_carbon_role_partition,
        validate_functionalized_carbon_role_partition,
    )
    from .functionalized_substituent_selection import (
        FunctionalizedSubstituentStatus,
        functionalized_substituent_frontier,
    )

    path = tuple(parent_atoms) if isinstance(parent_atoms, (tuple, list)) else ()
    signature = _signature(mol) if observation is None else observation.signature
    atom_rows, bond_rows, stereo_groups = signature
    atom_count = len(atom_rows)
    if (not path or any(type(i) is not int or not 0 <= i < atom_count for i in path)
            or len(set(path)) != len(path)):
        return _pending(path, signature, "parent proposal is not a unique original-index path")
    path_bonds = []
    for left, right in zip(path, path[1:]):
        bond = mol.GetBondBetweenAtoms(left, right)
        if bond is None:
            return _pending(path, signature, "parent proposal path is not contiguous")
        path_bonds.append(bond)
    locant_by_atom = {atom: locant for locant, atom in enumerate(path, 1)}
    parent_set = set(locant_by_atom)
    if any(atom_rows[i][2] != 6 for i in parent_set):
        return _pending(path, signature, "bounded parent proposal requires a carbon skeleton")
    if len(Chem.GetMolFrags(mol)) != 1 or mol.GetNumAtoms() - mol.GetNumBonds() != 1:
        return _pending(path, signature, "input is outside the connected acyclic graph scope")
    # The exact native-state signature already observed these annotations.
    # Reuse its immutable rows instead of scanning the same atoms and bonds
    # again for every oriented parent proposal. Fresh-copy validation below
    # still builds its own signature and all candidate-local evidence.
    if any(row[3] or row[4] or row[5] or row[6]
           for row in atom_rows if row[2] > 1):
        return _pending(path, signature, "annotations or nonneutral heavy atoms are outside scope")
    if any(row[3] for row in atom_rows if row[2] == 1):
        return _pending(path, signature, "isotopic hydrogen is outside scope")
    if any(row[4] or row[5] for row in bond_rows):
        return _pending(path, signature, "assigned bond stereochemistry is outside scope")
    if stereo_groups:
        return _pending(path, signature, "enhanced stereo groups are outside scope")
    if any(bond.GetBondType() != Chem.BondType.SINGLE for bond in path_bonds):
        return _pending(path, signature, "bounded parent skeleton requires single bonds")

    inventory = (selection_inventory(mol) if observation is None
                 else observation.inventory)
    if inventory.supported and not inventory.groups:
        return _outside(path, signature,
                        "no source-recognized principal-group obligation applies")
    frontier = getattr(inventory, "source_frontier", None)
    requested = tuple(principal_groups)
    if (not inventory.supported or frontier is None
            or frontier.status not in {"selected", "tie"}
            or tuple(getattr(frontier, "unknown_pairs", ()))
            or bool(getattr(frontier, "cycle", False))
            or tuple(frontier.groups) != requested
            or tuple(inventory.principal) != requested
            or len(inventory.source_group_facts) != len(inventory.groups)):
        return _pending(path, signature, "P-41 source principal inventory is incomplete",
                        inventory=inventory)
    if (not requested or any(group.kind not in _SUPPORTED_PRINCIPAL_KINDS for group in requested)
            or suffix_mode != "chain-included"):
        return _pending(path, signature, "principal class or suffix mode is outside this bounded adapter",
                        inventory=inventory)
    selected_ids = tuple(sorted(_identity(group) for group in requested))
    suffix_atoms = set()
    covered = []
    omitted = []
    for group in requested:
        identity = _identity(group)
        center = int(group.center)
        if center in parent_set:
            covered.append(identity)
        else:
            omitted.append(identity)
        atoms = set(int(i) for i in group.atoms)
        if center not in atoms or not atoms:
            return _pending(path, signature, "principal source group has invalid atom ownership",
                            inventory=inventory)
        # The source-defined group remains intact as a role; parent-skeleton
        # atoms overlap that role by construction, but are not re-owned.
        if center in parent_set:
            suffix_atoms.update(atoms - parent_set)
    suffix_locants = tuple(sorted(locant_by_atom[int(group.center)]
                                  for group in requested
                                  if int(group.center) in parent_set))
    p44_count = len(covered)
    p44_size = len(path)
    if any(int(group.center) in parent_set
           and not _principal_suffix_shape_is_included(mol, group, parent_set)
           for group in requested):
        return _pending(path, signature,
                        "P-43 group shape or chain-included suffix ownership is not proved",
                        inventory=inventory, p44_count=p44_count,
                        p44_size=p44_size, suffix_atoms=suffix_atoms,
                        suffix_locants=suffix_locants, covered=covered,
                        omitted=omitted)

    # Exact proposal-local direct source groups attached to a parent host.
    direct, direct_error = _source_group_units(
        mol, inventory.groups, parent_set, set(selected_ids), set(suffix_atoms),
    )
    if direct is None:
        return _pending(path, signature, direct_error, inventory=inventory,
                        p44_count=p44_count, p44_size=p44_size,
                        suffix_atoms=suffix_atoms, suffix_locants=suffix_locants,
                        covered=covered, omitted=omitted)

    direct = tuple(FunctionalPrefixUnitEvidence(
        unit.kind, unit.host, unit.root, unit.atoms, unit.source_group_atoms,
        unit.source_group_identities, unit.source_rules, locant_by_atom[unit.host],
        unit.role_count, unit.rendered_name, unit.p46_frontier,
    ) for unit in direct)
    prefix_units = list(direct)
    owned = set(parent_set) | set(suffix_atoms)
    for unit in direct:
        if owned & set(unit.atoms):
            return _pending(path, signature, "direct prefix ownership overlaps another role",
                            inventory=inventory, p44_count=p44_count, p44_size=p44_size,
                            suffix_atoms=suffix_atoms, suffix_locants=suffix_locants,
                            covered=covered, omitted=omitted)
        owned.update(unit.atoms)

    # Enumerate every off-parent heavy-atom component from its unique path
    # attachment. A connected tree makes each component boundary unique.
    seen_components = set()
    for host in path:
        for neighbor in mol.GetAtomWithIdx(host).GetNeighbors():
            root = neighbor.GetIdx()
            if neighbor.GetAtomicNum() == 1 or root in parent_set:
                continue
            component = _tree_component(mol, root, parent_set)
            if not component or component & seen_components:
                continue
            seen_components.update(component)
            # Directly attached source groups were already assigned as one
            # typed prefix; skip their component only when exact ownership
            # proves that it is precisely that role.
            matched_direct = next((u for u in direct
                                   if frozenset(u.atoms) == component), None)
            if matched_direct is not None:
                continue
            # Carbon-included acid/amide/nitrile suffix atoms can be separate
            # one-atom graph components after the candidate parent is cut.
            # They remain suffix-owned and are never prefix candidates.
            if component <= suffix_atoms:
                continue
            if owned & component:
                return _pending(path, signature, "detached component overlaps a prior role",
                                inventory=inventory, p44_count=p44_count,
                                p44_size=p44_size, suffix_atoms=suffix_atoms,
                                suffix_locants=suffix_locants, covered=covered,
                                omitted=omitted)
            # P-61.3.1 allows standard-bonding halogens as detachable prefix
            # roles. Keep the graph role exact and text-free.
            if (len(component) == 1
                    and mol.GetAtomWithIdx(root).GetAtomicNum() in {9, 17, 35, 53}):
                atom = mol.GetAtomWithIdx(root)
                z = atom.GetAtomicNum()
                if (atom.GetFormalCharge() or atom.GetIsotope()
                        or atom.GetNumRadicalElectrons() or atom.GetTotalValence() != 1
                        or len(tuple(atom.GetNeighbors())) != 1
                        or mol.GetBondBetweenAtoms(root, host).GetBondType() != Chem.BondType.SINGLE):
                    return _pending(path, signature,
                                    "direct halogen is not an unlabelled standard-bonding role",
                                    inventory=inventory, p44_count=p44_count,
                                    p44_size=p44_size, suffix_atoms=suffix_atoms,
                                    suffix_locants=suffix_locants, covered=covered,
                                    omitted=omitted)
                halo_token = {9: "fluoro", 17: "chloro", 35: "bromo", 53: "iodo"}[z]
                prefix_units.append(FunctionalPrefixUnitEvidence(
                    halo_token, host, root, (root,), ((root,),),
                    (("halogen", z, root, host),), ("P-61.3.1",),
                    locant_by_atom[host], 1,
                ))
                owned.add(root)
                continue
            if all(atom_rows[i][2] == 6 for i in component):
                if not _ordinary_saturated_carbon_component(mol, Chem, component):
                    return _pending(path, signature,
                                    "pure-carbon prefix component is not a saturated carbon tree",
                                    inventory=inventory, p44_count=p44_count,
                                    p44_size=p44_size, suffix_atoms=suffix_atoms,
                                    suffix_locants=suffix_locants, covered=covered,
                                    omitted=omitted)
                kind, rules, role_count = "carbon-component", ("P-29.1.2",), 0
                p46 = ()
            else:
                partition, admission = functionalized_carbon_role_partition(
                    mol, Chem, root, host, requested,
                )
                if admission != "complete" or partition is None:
                    return _pending(path, signature,
                                    f"detached component role admission is {admission}",
                                    inventory=inventory, p44_count=p44_count,
                                    p44_size=p44_size, suffix_atoms=suffix_atoms,
                                    suffix_locants=suffix_locants, covered=covered,
                                    omitted=omitted)
                if (partition.scope != component
                        or not validate_functionalized_carbon_role_partition(
                            mol, Chem, partition, requested)):
                    return _pending(path, signature,
                                    "shared role partition does not match the exact detached component",
                                    inventory=inventory, p44_count=p44_count,
                                    p44_size=p44_size, suffix_atoms=suffix_atoms,
                                    suffix_locants=suffix_locants, covered=covered,
                                    omitted=omitted)
                branch_frontier = functionalized_substituent_frontier(
                    mol, Chem, root, host, role_partition=partition,
                    resolve_branch_tokens=resolve_branch_tokens,
                )
                if (branch_frontier.status is not FunctionalizedSubstituentStatus.COMPLETE
                        or not branch_frontier.universe_complete
                        or not branch_frontier.candidates):
                    return _pending(path, signature,
                                    "detached functionalized component lacks a complete P-46 graph frontier",
                                    inventory=inventory, p44_count=p44_count,
                                    p44_size=p44_size, suffix_atoms=suffix_atoms,
                                    suffix_locants=suffix_locants, covered=covered,
                                    omitted=omitted)
                kind = "functionalized-carbon-component"
                rules = tuple(sorted({role.rule for role in partition.roles})) + ("P-29", "P-46")
                role_count = len(partition.roles)
                p46 = tuple(branch_frontier.candidates)
            group_ids = tuple(sorted(
                _identity(group) for group in inventory.groups
                if component.intersection(group.atoms) or group.center in component
                or group.owner in component
            ))
            full_group_atoms = tuple(sorted(
                tuple(sorted(int(i) for i in group.atoms))
                for group in inventory.groups
                if _identity(group) in group_ids
            ))
            if any(not component.issuperset(row) for row in full_group_atoms):
                return _pending(path, signature,
                                "detached source group escapes its component boundary",
                                inventory=inventory, p44_count=p44_count,
                                p44_size=p44_size, suffix_atoms=suffix_atoms,
                                suffix_locants=suffix_locants, covered=covered,
                                omitted=omitted)
            unit = FunctionalPrefixUnitEvidence(
                kind, host, root, tuple(sorted(component)), full_group_atoms,
                group_ids, rules, locant_by_atom[host], role_count, None, p46,
            )
            prefix_units.append(unit)
            owned.update(component)

    # The exact-state signature has already observed every original atom.
    # Reuse its immutable element rows for candidate-local coverage checks.
    all_heavy = {row[0] for row in atom_rows if row[2] > 1}
    # Principal group source atoms outside the parent are suffix ownership;
    # all other heavy atoms must belong to one prefix component.
    if owned != all_heavy:
        return _pending(path, signature,
                        "parent, suffix, and prefix roles do not exactly cover heavy atoms",
                        inventory=inventory, p44_count=p44_count,
                        p44_size=p44_size, suffix_atoms=suffix_atoms,
                        suffix_locants=suffix_locants, covered=covered,
                        omitted=omitted)
    # Every recognized nonprincipal group must be assigned to a direct role or
    # one detached component, including groups that have no useful name yet.
    assigned_identities = {identity for unit in prefix_units
                           for identity in unit.source_group_identities}
    recognized_nonprincipal = {_identity(group) for group in inventory.groups
                               if group not in requested}
    if not recognized_nonprincipal <= assigned_identities:
        return _pending(path, signature,
                        "a recognized nonprincipal source group has no prefix ownership",
                        inventory=inventory, p44_count=p44_count,
                        p44_size=p44_size, suffix_atoms=suffix_atoms,
                        suffix_locants=suffix_locants, covered=covered,
                        omitted=omitted)

    locants = tuple(sorted(unit.parent_locant for unit in prefix_units))
    return FunctionalParentPrefixEvidence391(
        PrefixEvidenceStatus.COMPLETE, path, selected_ids,
        tuple(sorted(covered)), tuple(sorted(omitted)), tuple(sorted(suffix_atoms)),
        p44_count, p44_size, suffix_locants, tuple(prefix_units),
        len(prefix_units), locants, None, signature,
        ("P-41", "P-43", "P-44.1.1", "P-44.3", "P-45.2", "P-29", "P-46"),
        "candidate-local graph roles and atom ownership are complete; names and method eligibility remain open",
        mapped_branch_tokens=bool(resolve_branch_tokens),
    )


def validate_functional_parent_prefix_evidence391(
    mol, Chem, evidence, principal_groups, *, suffix_mode="chain-included",
):
    """Rebuild a candidate-local proof on a fresh graph copy and compare it."""
    if (not isinstance(evidence, FunctionalParentPrefixEvidence391)
            or evidence.original_graph_signature != _signature(mol)):
        return False
    probe = Chem.Mol(mol)
    rebuilt = build_functional_parent_prefix_evidence391(
        probe, Chem, evidence.parent_atoms, tuple(principal_groups),
        suffix_mode=suffix_mode,
        resolve_branch_tokens=evidence.mapped_branch_tokens,
    )
    return rebuilt == evidence


@dataclass(frozen=True, slots=True)
class _FreshPrefixGraphObservation:
    """Immutable native facts independently observed on a fresh graph copy."""

    native_state: bytes
    signature: tuple
    inventory: object


class _FunctionalPrefixEvidenceValidator:
    """Call-local observation reuse while rebuilding every candidate proof.

    This is used only inside one default-writer frontier. Every validation
    still owns a new native Mol with no inherited Python caches. An exact
    native-state comparison permits its immutable graph signature and source
    inventory to be shared across numberings. The rebuilt copy must retain
    that native state; computations with side effects take the public fresh
    validation path. No source Mol or candidate proof is retained here.
    """

    __slots__ = ("observation",)

    def __init__(self):
        self.observation = None

    def validate(self, mol, Chem, evidence, principal_groups, *, suffix_mode="chain-included"):
        if not isinstance(evidence, FunctionalParentPrefixEvidence391):
            return False
        try:
            native_state = mol.ToBinary(Chem.PropertyPickleOptions.AllProps)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return validate_functional_parent_prefix_evidence391(
                mol, Chem, evidence, principal_groups, suffix_mode=suffix_mode,
            )

        # A new native copy is mandatory even when an earlier numbering has
        # already established exactly the same immutable source facts.
        probe = Chem.Mol(mol)
        observation = self.observation
        if observation is None or observation.native_state != native_state:
            from .extended_inventory import selection_inventory

            signature = _signature(probe)
            inventory = selection_inventory(probe)
            # Resolve the lazy source projections on the fresh observation;
            # candidate builders then consume only immutable graph facts.
            inventory.source_frontier
            inventory.principal
            observation = _FreshPrefixGraphObservation(native_state, signature, inventory)
        if evidence.original_graph_signature != observation.signature:
            return False
        rebuilt = _build_functional_parent_prefix_evidence391(
            probe, Chem, evidence.parent_atoms, tuple(principal_groups),
            suffix_mode=suffix_mode,
            resolve_branch_tokens=evidence.mapped_branch_tokens,
            observation=observation,
        )
        try:
            unchanged = probe.ToBinary(Chem.PropertyPickleOptions.AllProps) == native_state
        except (AttributeError, TypeError, ValueError, RuntimeError):
            unchanged = False
        if not unchanged:
            self.observation = None
            return validate_functional_parent_prefix_evidence391(
                mol, Chem, evidence, principal_groups, suffix_mode=suffix_mode,
            )
        self.observation = observation
        return rebuilt == evidence

