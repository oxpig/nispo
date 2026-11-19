"""Substitutive PIN construction for partial esters of polybasic acids.

P-65.6.3.3.5 method (1) retains the free carboxylic acid group(s) as the
principal group(s) and names the ester functionality by substitutive prefixes.
This bounded producer delegates mapped acid-parent and prefix construction to
the common functional-parent writer, then requires complete component
ownership before returning a decision.
"""

from dataclasses import dataclass, replace

from .molecule_iteration import iter_atoms, iter_bonds
from .naming_methods import MethodDecision


@dataclass(frozen=True, slots=True)
class _PendantEsterPrefix:
    """A graph-proved ester substituent on a mapped polyacid parent."""

    host: int
    host_locant: str
    carbonyl: int
    carbon_path: tuple[int, ...]
    branch_name: str
    component_atoms: frozenset[int]
    path_locants: tuple[tuple[int, str], ...]
    alkoxy_atoms: frozenset[int]
    alkoxy_name: str


def _carbon_path(mol, start, end, blocked=frozenset()):
    """Return the unique simple carbon path, bounded to the local component."""
    blocked = set(blocked)
    paths = []

    def walk(current, path):
        if len(paths) > 1:
            return
        if current == end:
            paths.append(tuple(path))
            return
        for neighbor in mol.GetAtomWithIdx(current).GetNeighbors():
            idx = neighbor.GetIdx()
            if (idx in blocked or idx in path or neighbor.GetAtomicNum() != 6
                    or mol.GetBondBetweenAtoms(current, idx).GetBondTypeAsDouble() != 1):
                continue
            walk(idx, path + [idx])

    if (mol.GetAtomWithIdx(start).GetAtomicNum() != 6
            or mol.GetAtomWithIdx(end).GetAtomicNum() != 6):
        return None
    walk(start, [start])
    return paths[0] if len(paths) == 1 else None


def _simple_alkoxy_component(mol, Chem, ester):
    """Return a linear saturated alkoxy unit and its graph ownership."""
    from .component_mapping import mapped_branch
    from .formatting import alkyl_to_alkoxy

    oxygen = mol.GetAtomWithIdx(ester.ligand)
    roots = [neighbor.GetIdx() for neighbor in oxygen.GetNeighbors()
             if neighbor.GetIdx() != ester.center]
    if len(roots) != 1 or mol.GetAtomWithIdx(roots[0]).GetAtomicNum() != 6:
        return None
    text, atoms = mapped_branch(mol, Chem, roots[0], ester.ligand)
    if not text or not text.endswith("yl") or not atoms:
        return None
    atom_set = frozenset(atoms)
    if any(mol.GetAtomWithIdx(idx).GetAtomicNum() != 6
           or mol.GetAtomWithIdx(idx).GetIsotope()
           or mol.GetAtomWithIdx(idx).GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
           for idx in atom_set):
        return None
    carbon_atoms = set(atom_set)
    edges = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
             for bond in iter_bonds(mol)
             if bond.GetBeginAtomIdx() in carbon_atoms
             and bond.GetEndAtomIdx() in carbon_atoms]
    if len(edges) != max(0, len(carbon_atoms) - 1):
        return None
    if any(bond.GetBondTypeAsDouble() != 1 for bond in iter_bonds(mol)
           if bond.GetBeginAtomIdx() in carbon_atoms
           and bond.GetEndAtomIdx() in carbon_atoms):
        return None
    # A linear alkyl component has at most two carbon neighbors per atom and
    # exactly one carbon at its bridge-oxygen attachment end.
    if any(sum(n.GetIdx() in carbon_atoms for n in mol.GetAtomWithIdx(idx).GetNeighbors()) > 2
           for idx in carbon_atoms):
        return None
    alkoxy = alkyl_to_alkoxy(text)
    if not alkoxy:
        return None
    return alkoxy, atom_set


def _pendant_ester_prefixes(mol, Chem, candidate, esters, alcohols):
    """Project simple pendant –C(O)OR branches from original atom roles.

    Each admitted ester must have a unique saturated all-carbon path from one
    parent host to the ester carbonyl. The ester must be the terminal carbonyl
    of that branch; the alcohol-derived side must be a mapped linear alkyl.
    """
    from .branch_rules import parent_chain_name
    from .component_mapping import decorate_component
    from .formatting import format_prefixes

    parent_atoms = frozenset(candidate.parent_atoms)
    mapping = dict(candidate.atom_locants)
    parent_locant = {atom: str(loc) for atom, loc in candidate.atom_locants}
    records = []
    for ester in esters:
        if ester.center in parent_atoms or ester.owner in parent_atoms:
            continue
        possible = []
        for host in parent_atoms:
            if mol.GetAtomWithIdx(host).GetAtomicNum() != 6:
                continue
            for neighbor in mol.GetAtomWithIdx(host).GetNeighbors():
                root = neighbor.GetIdx()
                if neighbor.GetAtomicNum() != 6 or root in parent_atoms:
                    continue
                path = _carbon_path(mol, root, ester.center, blocked=parent_atoms)
                if path and len(path) >= 2:
                    possible.append((host, path))
        if len(possible) != 1:
            continue
        host, path = possible[0]
        # The branch is an unbranched carbon path from its attachment root to
        # terminal ester carbonyl. P-46.1.2 prefers the longer substituent
        # chain; the graph restriction proves there is only this chain.
        if any(mol.GetAtomWithIdx(idx).GetIsotope()
               or mol.GetAtomWithIdx(idx).GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
               for idx in path):
            continue
        if any(mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble() != 1
               for a, b in zip(path, path[1:])):
            continue
        if path[-1] != ester.center:
            continue
        # No carbon offshoots or other heteroatoms may be hidden in the branch.
        valid = True
        path_set = set(path)
        for offset, idx in enumerate(path):
            atom = mol.GetAtomWithIdx(idx)
            previous = host if offset == 0 else path[offset - 1]
            following = path[offset + 1] if offset + 1 < len(path) else None
            expected = {previous}
            if following is not None:
                expected.add(following)
            for other in atom.GetNeighbors():
                other_idx = other.GetIdx()
                if other_idx in expected:
                    continue
                if idx == ester.center and other.GetAtomicNum() == 8 and other_idx in ester.atoms:
                    continue
                valid = False
                break
            if not valid:
                break
        if not valid:
            continue
        alkoxy = _simple_alkoxy_component(mol, Chem, ester)
        if alkoxy is None:
            continue
        alkoxy_name, alkoxy_atoms = alkoxy
        branch_name = parent_chain_name(mol, path, as_substituent=True)
        if not branch_name:
            continue
        carbonyl_locant = str(len(path))
        inner_prefix = format_prefixes([
            (carbonyl_locant, alkoxy_name),
            (carbonyl_locant, "oxo"),
        ])
        if not inner_prefix:
            continue
        complete_name = inner_prefix + branch_name
        if any(mol.GetAtomWithIdx(idx).GetAtomicNum() != 6
               for idx in path[:-1]):
            continue
        oxygen_atoms = frozenset(idx for idx in ester.atoms
                                 if mol.GetAtomWithIdx(idx).GetAtomicNum() == 8)
        component_atoms = frozenset(path) | alkoxy_atoms | oxygen_atoms
        if component_atoms & parent_atoms:
            continue
        records.append(_PendantEsterPrefix(
            host=host,
            host_locant=parent_locant[host],
            carbonyl=ester.center,
            carbon_path=path,
            branch_name=complete_name,
            component_atoms=component_atoms,
            path_locants=tuple((idx, str(position))
                               for position, idx in enumerate(path, 1)),
            alkoxy_atoms=alkoxy_atoms,
            alkoxy_name=alkoxy_name,
        ))
    if not records:
        return candidate, ()

    # One independently mapped unit per host is needed to identify the legacy
    # projection to replace. Preserve any separately recognized hydroxy row.
    if len({record.host for record in records}) != len(records):
        return None, ()
    updated_prefixes = list(candidate.prefixes)
    alcohol_hosts = set()
    for alcohol in alcohols:
        if alcohol.center in parent_atoms:
            host = alcohol.center
        else:
            hosts = [n.GetIdx() for n in mol.GetAtomWithIdx(alcohol.center).GetNeighbors()
                     if n.GetIdx() in parent_atoms]
            if len(hosts) != 1:
                return None, ()
            host = hosts[0]
        alcohol_hosts.add(parent_locant[host])
    for record in records:
        indices = [i for i, (loc, _text) in enumerate(updated_prefixes)
                   if str(loc) == record.host_locant]
        hydroxy_rows = [i for i in indices
                        if updated_prefixes[i][1] == "hydroxy"
                        and record.host_locant in alcohol_hosts]
        replaceable = [i for i in indices if i not in hydroxy_rows]
        if len(replaceable) != 1:
            return None, ()
        updated_prefixes[replaceable[0]] = (record.host_locant, record.branch_name)
        # The complete original prefix projection must already own every atom
        # in the typed component before the source form is substituted.
        if not record.component_atoms <= candidate.construction.embedded_atoms:
            return None, ()
    construction = replace(candidate.construction, prefixes=tuple(updated_prefixes))
    name = decorate_component(mol, Chem, construction)
    if not name:
        return None, ()
    candidate = replace(candidate, name=name, prefixes=tuple(updated_prefixes),
                        construction=construction)
    return candidate, tuple(records)


def _mapped_parent_frontier(mol, candidates, acid_sites):
    """Compare candidates with annotation-aware shared parent facts."""
    from .parent_selection import graph_parent_facts, parent_frontier

    facts_by_id = {}
    seniority = acid_sites[0].seniority
    for candidate in candidates:
        parent_atoms = frozenset(candidate.parent_atoms)
        locants = dict(candidate.atom_locants)
        if not parent_atoms or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6
                                   for i in parent_atoms):
            facts_by_id[id(candidate)] = None
            continue
        edge_count = sum(1 for bond in iter_bonds(mol)
                         if bond.GetBeginAtomIdx() in parent_atoms
                         and bond.GetEndAtomIdx() in parent_atoms)
        kind = "ring" if edge_count >= len(parent_atoms) else "chain"
        facts_by_id[id(candidate)] = graph_parent_facts(
            mol, parent_atoms, candidate.atom_locants,
            principal_class=seniority,
            suffix_centers=candidate.principal_centers,
            suffix_locants=candidate.numbering_key[2],
            prefixes=candidate.prefixes,
            kind=kind,
            name=candidate.name,
            allow_annotations=True,
        )
    return parent_frontier(candidates, lambda row: facts_by_id[id(row)])


def _source_prefix_grammar_proved(mol, candidate, esters, alcohols,
                                  pendant_prefixes=()):
    """Check each specialized group against the source-supported prefix form."""
    parent_atoms = frozenset(candidate.parent_atoms)
    atom_locants = dict(candidate.atom_locants)
    prefixes = tuple((str(loc), str(text).strip("()"))
                     for loc, text in candidate.prefixes)
    acid_sites = tuple(group for group in esters
                       if group.owner >= 0 and group.owner in parent_atoms)
    pendant_by_center = {record.carbonyl: record for record in pendant_prefixes}
    for group in acid_sites:
        if group.center in parent_atoms:
            site_loc = str(atom_locants.get(group.center, ""))
            if site_loc.startswith("C") and site_loc[1:].isdigit():
                site_loc = site_loc[1:]
            at_site = [text for loc, text in prefixes if loc == site_loc]
            if (not site_loc.isdigit() or "oxo" not in at_site
                    or not any(text.endswith("oxy") for text in at_site)):
                return False
        else:
            host_loc = str(atom_locants.get(group.owner, ""))
            if host_loc.startswith("C") and host_loc[1:].isdigit():
                host_loc = host_loc[1:]
            if not any(loc == host_loc and text.endswith("oxycarbonyl")
                       for loc, text in prefixes):
                return False
    for group in esters:
        if group in acid_sites:
            continue
        pendant = pendant_by_center.get(group.center)
        if pendant is not None:
            if not any(loc == pendant.host_locant and text == pendant.branch_name
                       for loc, text in prefixes):
                return False
            continue
        if group.center in parent_atoms or group.ligand < 0:
            return False
        roots = [neighbor.GetIdx() for neighbor in mol.GetAtomWithIdx(group.ligand).GetNeighbors()
                 if neighbor.GetIdx() != group.center]
        if len(roots) != 1 or roots[0] not in parent_atoms:
            return False
        site_loc = str(atom_locants.get(roots[0], ""))
        if not any(loc == site_loc and text.endswith("yloxy")
                   for loc, text in prefixes):
            return False
    for group in alcohols:
        if group.center in parent_atoms:
            host = group.center
        else:
            hosts = [neighbor.GetIdx() for neighbor in mol.GetAtomWithIdx(group.center).GetNeighbors()
                     if neighbor.GetIdx() in parent_atoms]
            if len(hosts) != 1:
                return False
            host = hosts[0]
        site_loc = str(atom_locants.get(host, ""))
        if not any(loc == site_loc and text == "hydroxy" for loc, text in prefixes):
            return False
    return True


def partial_acid_ester_decision(mol, Chem):
    """Return a fully mapped P-65.6.3.3.5(1) decision, else ``None``.

    The route is limited to neutral, connected carbon polyacids with at least
    one free carboxylic acid and at least one ordinary ester site. The mapped
    acid parent must own every ester site's acid-side attachment; all other
    atoms must be rendered as part of that parent or its prefixes.
    """
    if (mol.GetNumHeavyAtoms() > 128 or mol.GetStereoGroups()
            or len(Chem.GetMolFrags(mol, asMols=False)) != 1):
        return None
    if any(atom.GetFormalCharge() or atom.GetNumRadicalElectrons()
           for atom in iter_atoms(mol)):
        return None
    # P-82.6.4's ester-oxygen roles are not demonstrated for a polyacid's
    # multiple site-prefix construction. Retain mapped carbon labels only.
    if any(atom.GetIsotope() and atom.GetAtomicNum() != 6 for atom in iter_atoms(mol)):
        return None

    from .extended_inventory import selection_inventory
    inventory = selection_inventory(mol)
    if not inventory.supported:
        return None
    acids = tuple(group for group in inventory.groups if group.kind == "acid")
    esters = tuple(group for group in inventory.groups if group.kind == "ester")
    alcohols = tuple(group for group in inventory.groups if group.kind == "alcohol")
    allowed_kinds = {"acid", "ester", "alcohol"}
    if (not acids or not esters
            or any(group.kind not in allowed_kinds for group in inventory.groups)):
        return None
    principal = tuple(inventory.principal)
    if (not principal or any(group.kind != "acid" for group in principal)
            or frozenset(group.center for group in principal)
            != frozenset(group.center for group in acids)):
        return None

    # P-65.6.3.3.5 applies to one polybasic acid parent. An ester may either
    # attach directly to the selected parent, or be a graph-proved pendant
    # alkoxy...oxo prefix on one of its mapped substituent chains.
    from .complete_functional_parent import (
        FunctionalParent, _render, functional_parent_candidates,
        functional_parent_frontier,
    )
    candidates = None
    if len(acids) > 1:
        # Enumerate P-44/P-45 structural maxima before asking whether their
        # pendant ester prefixes can be written. The frontier API marks the
        # operation blocked if even one maximal skeleton cannot be rendered;
        # this avoids preferring a shorter acid parent just because its branch
        # happens to fit our bounded prefix grammar.
        pendant_by_mapping = {}

        def render_polyacid_maximum(*args, **kwargs):
            rendered = _render(*args, **kwargs)
            if not rendered:
                return None
            record, construction = rendered
            (name, locants, prefix_locs, mapping, prefixes,
             component_name, owners) = record
            path = args[4]
            group_rows = args[2]
            temporary = FunctionalParent(
                name=name,
                principal_centers=frozenset(group.center for group in group_rows),
                atom_locants=tuple(mapping.items()),
                parent_atoms=tuple(path),
                numbering_key=(-len(group_rows), -len(path), locants,
                               (), prefix_locs, name),
                prefix_locants=prefix_locs,
                prefixes=prefixes,
                component_name=component_name,
                ester_owners=owners,
                construction=construction,
            )
            transformed, pendant = _pendant_ester_prefixes(
                mol, Chem, temporary, esters, alcohols,
            )
            if transformed is None or not _source_prefix_grammar_proved(
                    mol, transformed, esters, alcohols, pendant):
                return None
            pendant_by_mapping[(tuple(path), tuple(transformed.atom_locants))] = pendant
            return ((transformed.name, locants, transformed.prefix_locants,
                     dict(transformed.atom_locants), transformed.prefixes,
                     component_name, owners), transformed.construction)

        parent_result = functional_parent_frontier(
            mol, Chem, principal=acids, include_single=False,
            renderer=render_polyacid_maximum,
        )
        if parent_result.status not in {"complete", "tie"}:
            return None
        # Each maximal record carries a fully transformed construction. The
        # pendant descriptors are re-derived below from original atom roles.
        candidates = parent_result.candidates
    else:
        candidates = functional_parent_candidates(
            mol, Chem, principal=acids, include_single=True,
        )
    heavy = frozenset(atom.GetIdx() for atom in iter_atoms(mol)
                      if atom.GetAtomicNum() > 1)
    eligible = []
    for candidate in candidates:
        if candidate.principal_centers != frozenset(group.center for group in acids):
            continue
        parent_atoms = frozenset(candidate.parent_atoms)
        construction = candidate.construction
        if construction is None:
            continue
        if len(acids) > 1:
            # Already projected in the graph-first renderer callback above;
            # retrieve the independently graph-derived role map by its parent
            # path and original atom-to-locant assignment.
            pendant_prefixes = pendant_by_mapping.get(
                (tuple(candidate.parent_atoms), tuple(candidate.atom_locants)),
            )
            if pendant_prefixes is None:
                return None
        else:
            candidate, pendant_prefixes = _pendant_ester_prefixes(
                mol, Chem, candidate, esters, alcohols,
            )
            if candidate is None:
                continue
        parent_atoms = frozenset(candidate.parent_atoms)
        if (not any(group.owner >= 0 and group.owner in parent_atoms
                    for group in esters) and not pendant_prefixes):
            continue
        construction = candidate.construction
        owned = frozenset(construction.atoms) | frozenset(construction.embedded_atoms)
        if owned != heavy:
            continue
        if not _source_prefix_grammar_proved(
                mol, candidate, esters, alcohols, pendant_prefixes):
            continue
        eligible.append((candidate, pendant_prefixes))
    if not eligible:
        return None

    frontier = _mapped_parent_frontier(mol, [row[0] for row in eligible], acids)
    # Equivalent parent orientations can remain on the P-44/P-45 frontier
    # (for example the two directions of a para-disubstituted benzene ring).
    # The locant map is retained in the decision frontier; serialization is
    # safe only when every maximal orientation produces the same complete
    # source-directed name. A differing name remains unresolved.
    frontier_names = {row.name for row in frontier}
    if len(frontier_names) != 1:
        return None
    candidate = frontier[0]
    construction = candidate.construction
    locants = tuple(sorted(construction.atom_locants))
    centers = frozenset(group.center for group in acids)
    roles = [("acid-parent-and-attached-sites",
              frozenset(construction.atoms), locants)]
    # Preserve typed internal locants and original atom ownership for every
    # graph-derived pendant ester unit. Unmapped residual prefix atoms retain
    # the generic component role used by the prior implementation.
    pendant_rows = next((rows for parent, rows in eligible
                         if parent.name == candidate.name
                         and parent.atom_locants == candidate.atom_locants), ())
    pendant_atoms = set()
    for record in pendant_rows:
        if pendant_atoms & set(record.component_atoms):
            return None
        pendant_atoms.update(record.component_atoms)
        roles.append((f"pendant-ester-prefix@{record.host_locant}",
                      record.component_atoms, record.path_locants))
    residual_prefix_atoms = frozenset(construction.embedded_atoms - pendant_atoms)
    if residual_prefix_atoms:
        roles.append(("mapped-prefix-components", residual_prefix_atoms, ()))
    return MethodDecision(
        name=candidate.name,
        method="substitutive",
        principal_centers=centers,
        atoms=heavy,
        rule="P-65.6.3.3.5(1); P-65.6.3.2.3; P-65.6.3.3.6; P-14; P-44/P-45 mapped acid-parent selection; P-46.1.2",
        atom_locants=locants,
        component_roles=tuple(roles),
        functional_parent_frontier=frontier,
    )
