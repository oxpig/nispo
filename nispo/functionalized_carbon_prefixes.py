"""Typed P-29/P-46 construction for functionalized carbon substituent prefixes.

Shared pendant-role admission precedes rooted carbon-tree path selection.
Complete maxima are selected graphically, then each live candidate is rendered
with exact mapped ownership before a component proof is returned.
"""

from __future__ import annotations

from dataclasses import dataclass

from .molecule_iteration import iter_atoms, iter_bonds
from .constants import ROOTS
from .formatting import format_prefixes


_HALOGENS = {9: "fluoro", 17: "chloro", 35: "bromo", 53: "iodo"}
_ROLE_TOKENS = {
    "alcohol": "hydroxy", "amine": "amino", "thiol": "sulfanyl",
    "nitrile": "cyano", "ketone": "oxo", "aldehyde": "oxo",
}


@dataclass(frozen=True, slots=True)
class FunctionalizedPendantRole:
    kind: str
    host: int
    atoms: tuple[int, ...]
    token: str
    inventory_identity: tuple
    rule: str


@dataclass(frozen=True, slots=True)
class FunctionalizedCarbonRolePartition:
    """Validated source-role partition before any carbon path selection.

    This is the shared admission layer for the terminal-linear writer and
    later P-46 path selectors. It proves only component ownership, recognized
    pendant roles, and principal-class priority; it does not select or rank a
    carbon skeleton.
    """

    root: int
    parent_host: int
    boundary_cut: tuple[int, int]
    scope: frozenset[int]
    carbon_atoms: frozenset[int]
    roles: tuple[FunctionalizedPendantRole, ...]
    principal_identities: tuple[tuple, ...]
    source_priority_pairs: tuple[tuple, ...]
    original_graph_signature: tuple
    rules: tuple[str, ...] = ("P-29.1.2", "P-35.2", "P-41", "P-42")


@dataclass(frozen=True, slots=True)
class FunctionalizedCarbonPrefixProof:
    status: str
    root: int
    parent_host: int
    scope: frozenset[int]
    carbon_path: tuple[int, ...]
    roles: tuple[FunctionalizedPendantRole, ...]
    principal_identities: tuple[tuple, ...]
    source_priority_pairs: tuple[tuple, ...]
    prefix_rows: tuple[tuple[str, str], ...]
    text: str
    original_graph_signature: tuple
    rules: tuple[str, ...] = ("P-29.3.2.1", "P-29.3.2.2", "P-35.2", "P-41", "P-46.0", "P-46.1", "P-14.3.4.2(a)")
    # P-29.1.2 makes the parent-plus-pendant construction a compound
    # substituent even when the selected parent is methyl and the token has no
    # hyphen or locant. The outer formatter consumes this typed fact directly.
    prefix_nature: str = "compound"
    # Full P-46 maximum frontier and real writer results are included so a
    # fresh graph rebuild can validate the exact late-render handoff.
    selected_path_evidence: tuple = ()
    branch_writer_evidence: tuple = ()


def _identity(group):
    return (group.kind, int(group.center), tuple(int(i) for i in group.atoms),
            int(group.owner), int(group.ligand))


def _graph_signature(mol):
    """Capture atom/bond chemistry so additive proofs reject stale graph caches."""
    atom_rows = tuple(sorted((
        atom.GetIdx(), atom.GetAtomicNum(), atom.GetIsotope(), atom.GetFormalCharge(),
        atom.GetNumRadicalElectrons(), int(atom.GetChiralTag()), atom.GetIsAromatic(),
        atom.GetExplicitValence(), atom.GetNumExplicitHs(), atom.GetNoImplicit(),
    ) for atom in iter_atoms(mol)))
    bond_rows = tuple(sorted((
        min(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
        max(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
        str(bond.GetBondType()), bond.GetIsAromatic(), int(bond.GetStereo()),
        int(bond.GetBondDir()),
    ) for bond in iter_bonds(mol)))
    return atom_rows, bond_rows


def _selected_source_inventory(mol, principal_groups):
    from .extended_inventory import selection_inventory

    inventory = selection_inventory(mol)
    frontier = getattr(inventory, "source_frontier", None)
    if (not inventory.supported or frontier is None
            or frontier.status not in {"selected", "tie"}
            or tuple(getattr(frontier, "unknown_pairs", ()))
            or bool(getattr(frontier, "cycle", False))
            or tuple(frontier.groups) != tuple(principal_groups)
            or tuple(inventory.principal) != tuple(principal_groups)
            or len(inventory.source_group_facts) != len(inventory.groups)):
        return None
    return inventory


def _scope(mol, root, parent_host):
    if (not isinstance(root, int) or not isinstance(parent_host, int)
            or root == parent_host or not 0 <= root < mol.GetNumAtoms()
            or not 0 <= parent_host < mol.GetNumAtoms()):
        return None
    cut = mol.GetBondBetweenAtoms(root, parent_host)
    if cut is None or cut.GetBondTypeAsDouble() != 1:
        return None
    seen, stack = set(), [root]
    while stack:
        idx = stack.pop()
        if idx == parent_host or idx in seen:
            continue
        seen.add(idx)
        stack.extend(n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                     if n.GetAtomicNum() > 1 and n.GetIdx() != parent_host)
    if not seen:
        return None
    cuts = tuple(sorted(
        (atom.GetIdx(), neighbor.GetIdx())
        for atom in iter_atoms(mol) if atom.GetIdx() in seen
        for neighbor in atom.GetNeighbors()
        if neighbor.GetAtomicNum() > 1 and neighbor.GetIdx() not in seen
    ))
    if cuts != ((root, parent_host),):
        return None
    return frozenset(seen)


def _recognized_role_inventory(mol, Chem, inventory, scope, carbon_framework):
    rows, claimed = [], set()
    facts = {id(group): fact for group, fact in
             zip(inventory.groups, inventory.source_group_facts)}
    principal = tuple(inventory.principal)
    priorities = []
    from .group_priority import compare_functional_group_priority

    for group in inventory.groups:
        if group in principal:
            continue
        # Ignore groups wholly outside this branch; a group that touches it
        # must be wholly represented and source-classified here.
        group_atoms = set(group.atoms)
        relevant = bool(group_atoms & scope or group.owner in scope or group.center in scope)
        if not relevant:
            continue
        token = _ROLE_TOKENS.get(group.kind)
        if token is None:
            return None
        if group.kind in {"ketone", "aldehyde"}:
            host = int(group.center)
            oxygen = subordinate_carbonyl_oxygen(mol, group)
            if host not in carbon_framework or oxygen is None:
                return None
            # The carbonyl carbon remains in the candidate carbon skeleton.
            # Its terminal oxygen has separate characteristic-group ownership.
            role_atoms = (oxygen,)
            rule = ("P-64.5.1 oxo/acyl" if group.kind == "ketone"
                    else "P-66.6.1.3 oxo/formyl")
        elif group.kind == "alcohol":
            host = int(group.center)
            oxygens = tuple(i for i in group.atoms if mol.GetAtomWithIdx(i).GetAtomicNum() == 8)
            if len(oxygens) != 1 or host not in scope or set(group.atoms) != {host, oxygens[0]}:
                return None
            role_atoms = (oxygens[0],)
            atom = mol.GetAtomWithIdx(oxygens[0])
            if (atom.GetFormalCharge() or atom.GetIsotope() or atom.GetNumRadicalElectrons()
                    or atom.GetTotalNumHs(includeNeighbors=True) != 1
                    or sum(n.GetAtomicNum() > 1 for n in atom.GetNeighbors()) != 1
                    or mol.GetBondBetweenAtoms(host, atom.GetIdx()).GetBondTypeAsDouble() != 1):
                return None
            rule = "P-35.2.1 hydroxy"
        elif group.kind == "amine":
            n = mol.GetAtomWithIdx(group.center)
            carbon_hosts = tuple(neighbor.GetIdx() for neighbor in n.GetNeighbors()
                                 if neighbor.GetIdx() in carbon_framework)
            host = carbon_hosts[0] if len(carbon_hosts) == 1 else -1
            from .amine_components import ordinary_amine, amino_prefix
            if (host not in scope or set(group.atoms) != {group.center}
                    or not ordinary_amine(mol, group.center)
                    or mol.GetBondBetweenAtoms(host, group.center).GetBondTypeAsDouble() != 1):
                return None
            # P-62.2.3: the entire N-rooted component owns the N ligands.
            # Traversing a C-N-C bond must not extend the carbon parent.
            amino_atoms = _scope(mol, int(group.center), host)
            if (amino_atoms is None or not amino_atoms <= scope
                    or amino_atoms & carbon_framework):
                return None
            token, owned = amino_prefix(mol, Chem, int(group.center), host, amino_atoms)
            if not token or frozenset(owned) != amino_atoms:
                return None
            role_atoms = tuple(sorted(amino_atoms))
            rule = "P-62.2.3 amino"
        elif group.kind == "thiol":
            host, sulfur = int(group.owner), int(group.center)
            atom = mol.GetAtomWithIdx(sulfur)
            if (host not in scope or set(group.atoms) != {sulfur}
                    or atom.GetAtomicNum() != 16 or atom.GetFormalCharge() or atom.GetIsotope()
                    or atom.GetNumRadicalElectrons() or atom.GetTotalNumHs(includeNeighbors=True) != 1
                    or len(tuple(atom.GetNeighbors())) != 1
                    or mol.GetBondBetweenAtoms(host, sulfur).GetBondTypeAsDouble() != 1):
                return None
            role_atoms = (sulfur,)
            rule = "P-35.2.2 sulfanyl"
        else:  # exact terminal nitrile group, attached through its carbon
            nitrile_carbon = int(group.center)
            host = int(group.owner)
            nitrogens = tuple(i for i in group.atoms
                              if mol.GetAtomWithIdx(i).GetAtomicNum() == 7)
            if (host not in scope or len(nitrogens) != 1
                    or set(group.atoms) != {nitrile_carbon, nitrogens[0]}
                    or mol.GetAtomWithIdx(nitrile_carbon).GetFormalCharge()
                    or mol.GetAtomWithIdx(nitrogens[0]).GetFormalCharge()
                    or mol.GetAtomWithIdx(nitrile_carbon).GetIsotope()
                    or mol.GetAtomWithIdx(nitrogens[0]).GetIsotope()
                    or mol.GetBondBetweenAtoms(host, nitrile_carbon) is None
                    or mol.GetBondBetweenAtoms(host, nitrile_carbon).GetBondTypeAsDouble() != 1
                    or mol.GetBondBetweenAtoms(nitrile_carbon, nitrogens[0]) is None
                    or mol.GetBondBetweenAtoms(nitrile_carbon, nitrogens[0]).GetBondTypeAsDouble() != 3
                    or len(tuple(mol.GetAtomWithIdx(nitrogens[0]).GetNeighbors())) != 1):
                return None
            role_atoms = (nitrile_carbon, nitrogens[0])
            rule = "P-35.2 cyano"
        if set(role_atoms) & claimed or set(role_atoms) & set(principal_group_atoms(principal)):
            return None
        fact = facts.get(id(group))
        if fact is None:
            return None
        for selected in principal:
            selected_fact = facts.get(id(selected))
            comparison = compare_functional_group_priority(selected_fact, fact)
            if comparison.status != "senior":
                return None
            priorities.append((selected.center, group.center, comparison.rule, comparison.status))
        claimed.update(role_atoms)
        rows.append(FunctionalizedPendantRole(
            group.kind, host, tuple(sorted(role_atoms)), token, _identity(group), rule,
        ))

    for atom in iter_atoms(mol):
        z = atom.GetAtomicNum()
        if z not in _HALOGENS:
            continue
        idx = atom.GetIdx()
        attached_hosts = tuple(n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() in scope)
        if not attached_hosts:
            continue
        if (len(attached_hosts) != 1 or atom.GetFormalCharge()
                or atom.GetIsotope() or atom.GetNumRadicalElectrons() or atom.GetTotalValence() != 1
                or len(tuple(atom.GetNeighbors())) != 1
                or mol.GetBondBetweenAtoms(idx, attached_hosts[0]).GetBondTypeAsDouble() != 1
                or idx in claimed):
            return None
        host = attached_hosts[0]
        claimed.add(idx)
        rows.append(FunctionalizedPendantRole(
            "halogen", host, (idx,), _HALOGENS[z], ("halogen", z, idx, host),
            "P-35.2.1 " + _HALOGENS[z],
        ))
    return tuple(rows), tuple(priorities), claimed


def subordinate_carbonyl_oxygen(mol, group):
    """Prove one ordinary ketone/aldehyde oxygen without reclassifying C=O.

    Carbonyl-bearing acids, esters, amides and pseudoketones keep their own
    inventory identities. An isotope on the oxygen needs its own descriptor
    operation and is not admitted by this unlabelled role proof.
    """
    if group.kind not in {"ketone", "aldehyde"}:
        return None
    host = int(group.center)
    carbon = mol.GetAtomWithIdx(host)
    if (carbon.GetAtomicNum() != 6 or carbon.GetFormalCharge()
            or carbon.GetNumRadicalElectrons() or carbon.GetIsAromatic()
            or carbon.GetTotalValence() != 4):
        return None
    oxygen = tuple(n for n in carbon.GetNeighbors()
                   if n.GetAtomicNum() == 8
                   and mol.GetBondBetweenAtoms(host, n.GetIdx()).GetBondTypeAsDouble() == 2)
    if len(oxygen) != 1:
        return None
    oxygen = oxygen[0]
    if (set(group.atoms) != {host, oxygen.GetIdx()} or oxygen.GetFormalCharge()
            or oxygen.GetIsotope() or oxygen.GetNumRadicalElectrons()
            or oxygen.GetTotalValence() != 2 or oxygen.GetDegree() != 1
            or int(oxygen.GetChiralTag())):
        return None
    neighbors = tuple(n for n in carbon.GetNeighbors()
                      if n.GetAtomicNum() > 1 and n.GetIdx() != oxygen.GetIdx())
    expected = 2 if group.kind == "ketone" else 1
    if (len(neighbors) != expected or any(n.GetAtomicNum() != 6 for n in neighbors)
            or any(mol.GetBondBetweenAtoms(host,n.GetIdx()).GetBondTypeAsDouble()!=1
                   for n in neighbors)
            or carbon.GetTotalNumHs(includeNeighbors=True) != (group.kind == "aldehyde")):
        return None
    return oxygen.GetIdx()


def principal_group_atoms(groups):
    return {idx for group in groups for idx in group.atoms}


def functionalized_carbon_role_partition(mol, Chem, root, parent_host,
                                         principal_groups):
    """Return a validated typed branch partition and ``complete/blocked/outside``.

    The helper is deliberately topology-neutral after the one-cut component
    is established. Consumers must choose their own admissible carbon paths
    and P-29/P-46 construction from ``carbon_atoms``; they must not infer role
    identity or source priority from rendered component text.
    """
    scope = _scope(mol, root, parent_host)
    if scope is None:
        return None, "outside"
    inventory = _selected_source_inventory(mol, principal_groups)
    if inventory is None:
        # Preserve a fail-closed disposition where recognized source groups
        # touch the component, while leaving unrelated carbon-only branches
        # to their established producers.
        from .extended_inventory import selection_inventory
        inventory = selection_inventory(mol)
        if any(set(group.atoms) & scope or group.center in scope or group.owner in scope
               for group in inventory.groups):
            return None, "blocked"
        return None, "outside"
    if any(set(group.atoms) & set(scope) or group.center in scope or group.owner in scope
           for group in principal_groups):
        return None, "blocked"

    carbon_framework, stack = set(), [root]
    while stack:
        idx = stack.pop()
        if idx in carbon_framework or idx not in scope or mol.GetAtomWithIdx(idx).GetAtomicNum()!=6:
            continue
        carbon_framework.add(idx)
        stack.extend(n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                     if n.GetAtomicNum()==6 and n.GetIdx() in scope)
    rows_result = _recognized_role_inventory(mol, Chem, inventory, scope, carbon_framework)
    if rows_result is None:
        return None, "blocked"
    roles, priority_pairs, role_atoms = rows_result
    if not roles:
        return None, "outside"
    carbon_atoms = frozenset(set(scope) - set(role_atoms))
    if (not carbon_atoms or set(scope) != set(carbon_atoms) | set(role_atoms)
            or any(mol.GetAtomWithIdx(idx).GetAtomicNum() != 6 for idx in carbon_atoms)):
        return None, "blocked"
    # Carbon annotations are decorated on their original numbered skeleton;
    # role-local annotations and explicit isotope H need their own grammar.
    if any((i not in carbon_atoms and (mol.GetAtomWithIdx(i).GetIsotope()
                                      or int(mol.GetAtomWithIdx(i).GetChiralTag())))
           or mol.GetAtomWithIdx(i).GetFormalCharge()
           or mol.GetAtomWithIdx(i).GetNumRadicalElectrons()
           for i in scope):
        return None, "blocked"
    if any(neighbor.GetAtomicNum() == 1 and neighbor.GetIsotope()
           for idx in scope for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors()):
        return None, "blocked"
    identities = tuple(sorted(_identity(group) for group in principal_groups))
    facts = FunctionalizedCarbonRolePartition(
        int(root), int(parent_host), (int(root), int(parent_host)), scope,
        carbon_atoms, roles, identities, priority_pairs, _graph_signature(mol),
    )
    return facts, "complete"


def validate_functionalized_carbon_role_partition(mol, Chem, partition,
                                                   principal_groups):
    """Rebuild role/priority facts on a fresh graph copy and compare exactly."""
    if not isinstance(partition, FunctionalizedCarbonRolePartition):
        return False
    if type(principal_groups) not in {tuple, list}:
        return _validate_functionalized_carbon_role_partition(
            mol, Chem, partition, principal_groups)
    groups = tuple(principal_groups)
    from .molecule_cache import exact_graph_computation
    return exact_graph_computation(
        mol, Chem, 'validated_functionalized_carbon_role_partition',
        (Chem, functionalized_carbon_role_partition, partition, groups),
        lambda: _validate_functionalized_carbon_role_partition(
            mol, Chem, partition, groups),
    )


def _validate_functionalized_carbon_role_partition(mol, Chem, partition,
                                                    principal_groups):
    if (not isinstance(partition, FunctionalizedCarbonRolePartition)
            or partition.original_graph_signature != _graph_signature(mol)):
        return False
    probe = Chem.Mol(mol)
    rebuilt, status = functionalized_carbon_role_partition(
        probe, Chem, partition.root, partition.parent_host, tuple(principal_groups),
    )
    return status == "complete" and rebuilt == partition


def functionalized_carbon_prefix(mol, Chem, root, parent_host, principal_groups):
    """Build a writer-backed prefix from every live P-46 graph maximum."""
    from .molecule_cache import mol_cache_dict

    principal_groups = tuple(principal_groups)
    # Recursive arms and symmetry-equivalent parent paths share exact cuts.
    # Include the live graph signature so mutation cannot reuse old evidence;
    # validation still rebuilds on a fresh same-order molecule below.
    key = (root, parent_host, tuple(_identity(group) for group in principal_groups),
           _graph_signature(mol))
    cache = mol_cache_dict(mol, "functionalized_carbon_prefixes")
    if key not in cache:
        cache[key] = _functionalized_carbon_prefix(
            mol, Chem, root, parent_host, principal_groups,
        )
    return cache[key]


def _functionalized_carbon_prefix(mol, Chem, root, parent_host, principal_groups):
    partition, status = functionalized_carbon_role_partition(
        mol, Chem, root, parent_host, principal_groups,
    )
    if status != "complete":
        return None, status
    from .functionalized_substituent_selection import (
        BranchWriterEvidence, FunctionalizedSubstituentStatus,
        functionalized_substituent_frontier, validate_branch_writer_closure,
    )
    frontier = functionalized_substituent_frontier(
        mol, Chem, root, parent_host, role_partition=partition,
        resolve_branch_tokens=True,
    )
    if frontier.status is FunctionalizedSubstituentStatus.OUTSIDE_SCOPE:
        # Source roles have already made this a required construction. An
        # unsupported skeleton must not re-enter an unproved legacy writer.
        return None, "blocked"
    if (frontier.status is not FunctionalizedSubstituentStatus.COMPLETE
            or not frontier.universe_complete or not frontier.candidates):
        return None, "blocked"

    renderings = []
    writer_evidence = []
    for candidate in frontier.candidates:
        branch_rows = []
        proofs = []
        compound_arms = []
        # N-substituted amino is itself a compound prefix (P-62.2.3), even
        # when its spelling has no punctuation, as in methylamino.
        from types import SimpleNamespace
        compound_arms.extend(SimpleNamespace(text=role.token, prefix_nature="compound")
                             for role in partition.roles
                             if role.kind=="amine" and role.token!="amino"
                             and role.host in candidate.path_atom_ids)
        for obligation in candidate.branch_writer_obligations:
            # An unknown token is a proof gap. Do not accept arbitrary text
            # supplied by a later writer as evidence for a complex branch.
            if obligation.token is None:
                return None, "blocked"
            text, owned = functionalized_arm_prefix(mol, Chem, obligation)
            if (not text or frozenset(owned) != frozenset(obligation.atoms)
                    or text != obligation.token):
                return None, "blocked"
            proof_row = BranchWriterEvidence(
                obligation.root, obligation.host, tuple(sorted(owned)), text,
            )
            branch_rows.append(proof_row)
            proofs.append(proof_row)
            if obligation.role_atoms:
                # Source nature belongs to the completed nested unit:
                # hydroxymethyl is compound, but plain acetyl is simple.
                nested, nested_status = functionalized_carbon_prefix(
                    mol, Chem, obligation.root, obligation.host, principal_groups)
                if nested_status != "complete" or nested.text != text:
                    return None, "blocked"
                if nested.prefix_nature == "compound":
                    compound_arms.append(SimpleNamespace(text=text, prefix_nature="compound"))
        closure = validate_branch_writer_closure(mol, Chem, candidate, tuple(proofs))
        if not closure.complete:
            return None, "blocked"

        chain_root = ROOTS.get(len(candidate.path_atom_ids))
        if not chain_root:
            return None, "blocked"
        path = candidate.path_atom_ids
        orders = tuple(mol.GetBondBetweenAtoms(a,b).GetBondTypeAsDouble()
                       for a,b in zip(path,path[1:]))
        if any(order>1 for order in orders):
            from .unsaturated_substituent_methods import unsaturated_carbon_stem
            stem = unsaturated_carbon_stem(
                len(path), candidate.root_locant,
                tuple(i for i,order in enumerate(orders,1) if order==2),
                tuple(i for i,order in enumerate(orders,1) if order==3))
            if not stem:return None, "blocked"
        elif candidate.method == 1 and candidate.root_locant == 1:
            stem = chain_root + "yl"
        elif candidate.method == 2 and candidate.root_locant > 1:
            stem = f"{chain_root}an-{candidate.root_locant}-yl"
        else:
            return None, "blocked"

        root_carbonyl = next((role for role in partition.roles
                              if role.host == root and role.kind in {"ketone", "aldehyde"}), None)
        if root_carbonyl is not None:
            # A carbonyl-root component is an acyl/formyl group, not a
            # 1-oxoalkyl synonym. Root degree forces locant 1 on every path.
            if candidate.root_locant != 1:
                return None, "blocked"
            if root_carbonyl.kind == "aldehyde":
                if len(path) != 1:
                    return None, "blocked"
                stem = "formyl"
            elif len(path) == 2:
                stem = "acetyl"
            elif len(path) > 2:
                if any(order>1 for order in orders):
                    from .formatting import complete_parent_name, unsaturated_parent
                    parent = complete_parent_name(unsaturated_parent(
                        chain_root,
                        [i for i,order in enumerate(orders,1) if order==2],
                        [i for i,order in enumerate(orders,1) if order==3]))
                    stem = parent[:-1] + "oyl" if parent.endswith("e") else ""
                else:
                    stem = chain_root + "anoyl"
                if not stem:
                    return None, "blocked"
            else:
                return None, "blocked"

        rows = []
        for unit in candidate.units:
            if (root_carbonyl is not None and unit.kind == "pendant-role"
                    and unit.host == root and unit.atoms == root_carbonyl.atoms):
                continue
            if not unit.token:
                return None, "blocked"
            locant = "" if len(candidate.path_atom_ids) == 1 else str(unit.locant)
            rows.append((locant, unit.token))
        if len(candidate.path_atom_ids) == 1:
            from .prefix_sequences import PrefixUnit, mononuclear_prefix_sequence
            counts = {}
            for _locant, token in rows:
                counts[token] = counts.get(token, 0) + 1
            compound_names = {arm.text for arm in compound_arms}
            sequence = mononuclear_prefix_sequence(
                PrefixUnit(token, "compound" if token in compound_names else "simple", count)
                for token, count in counts.items()
            )
            if sequence is None:
                return None, "blocked"
            text = sequence.text + stem
        else:
            from collections import Counter
            from .constants import MULT
            from .formatting import COMPLEX_MULT
            from .substituent_multiplication import multiply_substituent

            counts = Counter(token for _locant, token in rows)
            for token, count in counts.items():
                if count > 1 and count not in MULT:
                    return None, "blocked"
                # P-35 distinguishes repeated -SH groups from an S-S prefix.
                # A missing derived multiplier is not a basic-multiplier tie.
                if (count > 1 and any(role.kind == "thiol" and role.token == token
                                     for role in partition.roles)
                        and count not in COMPLEX_MULT):
                    return None, "blocked"
                if count > 1 and any(unit.kind == "carbon-arm" and unit.token == token
                                     for unit in candidate.units):
                    multiplied = multiply_substituent(token, count, nature="simple")
                    group_rows = [(locant, text) for locant, text in rows if text == token]
                    locants = ",".join(sorted((locant for locant, _text in group_rows),
                                             key=int))
                    if (multiplied is None or format_prefixes(group_rows)
                            != locants + "-" + multiplied.text):
                        return None, "blocked"
            if compound_arms:
                from .component_mapping import _format_prefixes_with_functionalized_compounds
                prefix = _format_prefixes_with_functionalized_compounds(
                    rows, tuple(compound_arms), hierarchical_enclosures=True,
                )
            else:
                prefix = format_prefixes(rows, hierarchical_enclosures=True)
            if rows and not prefix:
                return None, "blocked"
            text = prefix + stem
        # Decorate the numbered carbon skeleton on the original molecule.
        # Pendant roles and off-path arms were already rendered separately;
        # their annotations cannot migrate onto this parent map.
        from .component_mapping import MappedComponent, decorate_component, hydrogen_bindings
        bindings = hydrogen_bindings(mol, {idx:str(i) for i,idx in enumerate(path,1)})
        compound_names = {arm.text for arm in compound_arms}
        component = MappedComponent(
            stem, tuple(bindings.items()), frozenset(bindings), tuple(rows),
            partition.scope-frozenset(bindings), systematic_parent=True,
            compound_prefix_names=tuple(sorted(compound_names)) if len(path)>1 else (),
            prefix_mode='unlocanted' if len(path)==1 else 'locanted',
            prefix_natures=(tuple((token, 'compound' if token in compound_names else 'simple')
                                 for token in sorted({token for _, token in rows}))
                            if len(path)==1 else None),
        )
        text = decorate_component(mol, Chem, component)
        if not text:return None, "blocked"
        renderings.append((text, tuple(rows)))
        writer_evidence.append(tuple(branch_rows))

    # All graph maxima survive source comparison. Accept a component only if
    # every one has a real, exact writer result and the resulting text agrees.
    if len({text for text, _rows in renderings}) != 1:
        return None, "blocked"
    if len({rows for _text, rows in renderings}) != 1:
        return None, "blocked"
    chosen_text, prefix_rows = renderings[0]
    maxima = tuple(frontier.candidates)
    compatibility_path = maxima[0].path_atom_ids if len(maxima) == 1 else ()
    proof = FunctionalizedCarbonPrefixProof(
        "complete", root, parent_host, partition.scope, tuple(compatibility_path),
        partition.roles, partition.principal_identities,
        partition.source_priority_pairs, prefix_rows, chosen_text,
        _graph_signature(mol), selected_path_evidence=maxima,
        branch_writer_evidence=tuple(writer_evidence),
        prefix_nature=("simple" if root_carbonyl is not None and not prefix_rows
                       and not any(mol.GetAtomWithIdx(i).GetIsotope()
                                   or int(mol.GetAtomWithIdx(i).GetChiralTag())
                                   for i in partition.scope) else "compound"),
    )
    return proof, "complete"


def functionalized_arm_prefix(mol, Chem, unit):
    """Render an exact off-path arm through the existing source-bound writers.

    Recursion removes the enclosing path at each cut, so the carbon scope
    strictly decreases. It never selects a smaller parent because a larger
    candidate failed to render. Pure carbon arms retain their established
    P-29/P-46 writer, including retained tert-butyl vocabulary.
    """
    if unit.root is None or not unit.atoms:
        return "", frozenset()
    if unit.role_atoms:
        from .extended_inventory import selection_inventory
        inventory = selection_inventory(mol)
        proof, status = functionalized_carbon_prefix(
            mol, Chem, unit.root, unit.host, tuple(inventory.principal),
        )
        if status == "complete" and proof.scope == frozenset(unit.atoms):
            return proof.text, proof.scope
        return "", frozenset()
    from .substituent_selection import mapped_saturated_carbon_prefix
    result = mapped_saturated_carbon_prefix(
        mol, Chem, unit.root, unit.host, frozenset(unit.atoms),
    )
    if result is None:
        from .unsaturated_substituent_methods import unsaturated_carbon_substituent
        unsaturated = unsaturated_carbon_substituent(mol, Chem, unit.root, unit.host)
        if unsaturated is not None and unsaturated.atoms==frozenset(unit.atoms):
            result = unsaturated.text, unsaturated.atoms
    return result if result is not None else ("", frozenset())


def validate_functionalized_carbon_prefix_proof(mol, Chem, proof):
    """Rebuild and compare all proof fields against the current original graph."""
    if not isinstance(proof, FunctionalizedCarbonPrefixProof):
        return False
    from .molecule_cache import exact_graph_computation
    return exact_graph_computation(
        mol, Chem, 'validated_functionalized_carbon_prefix',
        (Chem, functionalized_carbon_prefix, proof),
        lambda: _validate_functionalized_carbon_prefix_proof(mol, Chem, proof),
    )


def _validate_functionalized_carbon_prefix_proof(mol, Chem, proof):
    if (not isinstance(proof, FunctionalizedCarbonPrefixProof)
            or proof.prefix_nature not in {"simple", "compound"}
            or proof.original_graph_signature != _graph_signature(mol)):
        return False
    # Inventory caches are keyed to mutable RDKit objects. Rebuild source facts
    # on a fresh, same-order copy so a caller's prior cached inventory cannot
    # authorize stale principal identities after an edit.
    probe = Chem.Mol(mol)
    from .extended_inventory import selection_inventory
    inventory = selection_inventory(probe)
    if _selected_source_inventory(probe, tuple(inventory.principal)) is None:
        return False
    frontier = inventory.source_frontier
    rebuilt, status = functionalized_carbon_prefix(
        probe, Chem, proof.root, proof.parent_host, tuple(frontier.groups),
    )
    return status == "complete" and rebuilt == proof
