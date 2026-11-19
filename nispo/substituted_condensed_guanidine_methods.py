"""Mapped terminal-N substitution for condensed guanidine diamides.

The bounded construction follows the repeat numbering illustrated in
P-66.4.1.2.2 figure 318: carbon repeat positions are 1, 3, 5, and 7, and
terminal amino/imino N sites are labelled N^position and N′^position. Only
terminal amino and terminal-repeat imino substitution is admitted here.
"""

from __future__ import annotations

from collections import defaultdict

from .molecule_iteration import iter_atoms, iter_bonds
from .component_mapping import mapped_branch
from .formatting import prefix_alphabetization_key
from .naming_methods import MethodDecision
from .substituent_multiplication import multiply_substituent


_RULE = "P-66.4.1.2.2/P-66.4.1.4.1/P-14.4/P-14.5/P-16.9"
_PARENT = {2: "imidodicarbonimidic diamide",
           3: "diimidotricarbonimidic diamide",
           4: "triimidotetracarbonimidic diamide"}


def substituted_condensed_guanidine_decision(mol, Chem):
    """Return a complete mapped decision, or decline outside proved scope."""
    if mol is None or mol.GetNumHeavyAtoms() > 128 or mol.GetStereoGroups():
        return None
    if any(atom.GetAtomicNum() == 1 or atom.GetFormalCharge()
           or atom.GetNumRadicalElectrons() or atom.GetIsotope()
           or int(atom.GetChiralTag()) for atom in iter_atoms(mol)):
        return None
    if any(int(bond.GetStereo()) or int(bond.GetBondDir()) for bond in iter_bonds(mol)):
        return None

    all_carbons = [atom.GetIdx() for atom in iter_atoms(mol) if atom.GetAtomicNum() == 6]
    carbons = []
    for atom in iter_atoms(mol):
        if atom.GetAtomicNum() != 6:
            continue
        orders = []
        for neighbor in atom.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            if neighbor.GetAtomicNum() != 7:
                orders = []
                break
            orders.append(bond.GetBondTypeAsDouble())
        if sorted(orders) == [1.0, 1.0, 2.0]:
            carbons.append(atom.GetIdx())
    if len(carbons) < 2 or len(carbons) > 4:
        return None
    n = len(carbons)
    imino, single_n = set(), set()
    carbon_single = {}
    for cidx in carbons:
        atom = mol.GetAtomWithIdx(cidx)
        doubles, singles = [], []
        for neighbor in atom.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(cidx, neighbor.GetIdx())
            if neighbor.GetAtomicNum() != 7:
                return None
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                doubles.append(neighbor.GetIdx())
            elif bond.GetBondType() == Chem.BondType.SINGLE:
                singles.append(neighbor.GetIdx())
            else:
                return None
        if len(doubles) != 1 or len(singles) != 2 or doubles[0] in imino:
            return None
        imino.add(doubles[0])
        single_n.update(singles)
        carbon_single[cidx] = tuple(singles)
    if len(imino) != n or len(single_n) != n + 1:
        return None
    if len(imino | single_n) + n > mol.GetNumHeavyAtoms():
        return None

    imino_to_core = {}
    for idx in imino:
        atom = mol.GetAtomWithIdx(idx)
        cidxs = [neighbor.GetIdx() for neighbor in atom.GetNeighbors()
                 if neighbor.GetAtomicNum() == 6
                 and mol.GetBondBetweenAtoms(idx, neighbor.GetIdx()).GetBondType()
                 == Chem.BondType.DOUBLE]
        cidx = cidxs[0] if len(cidxs) == 1 else -1
        if (atom.GetDegree() not in {1, 2}
                or cidx not in carbon_single
                or mol.GetBondBetweenAtoms(idx, cidx).GetBondType() != Chem.BondType.DOUBLE):
            return None
        imino_to_core[idx] = cidx

    single_to_c = {idx: [] for idx in single_n}
    for cidx, neighbors in carbon_single.items():
        for nidx in neighbors:
            single_to_c[nidx].append(cidx)
    terminals, bridges = [], []
    for nidx, attached in single_to_c.items():
        if len(attached) == 1:
            terminals.append((nidx, attached[0]))
        elif len(attached) == 2:
            bridges.append((nidx, tuple(attached)))
        else:
            return None
    if len(terminals) != 2 or len(bridges) != n - 1:
        return None
    adjacency = {cidx: set() for cidx in carbons}
    bridge_by_pair = {}
    for nidx, (left, right) in bridges:
        if right in adjacency[left]:
            return None
        adjacency[left].add(right)
        adjacency[right].add(left)
        bridge_by_pair[frozenset((left, right))] = nidx
    ends = [idx for idx, neighbors in adjacency.items() if len(neighbors) == 1]
    if len(ends) != 2 or any(len(neighbors) > 2 for neighbors in adjacency.values()):
        return None
    path = _walk_carbon_path(adjacency, ends[0])
    if len(path) != n:
        return None
    if set(path) != set(carbons):
        return None
    terminal_by_carbon = {carbon: nitrogen for nitrogen, carbon in terminals}
    if set(terminal_by_carbon) != {path[0], path[-1]}:
        return None
    endpoint_imino = {idx for idx, core_carbon in imino_to_core.items()
                      if core_carbon in {path[0], path[-1]}}

    # All remaining nitrogen atoms must be core nitrogens. External arms are
    # accepted only on terminal amino and terminal-repeat imino sites.
    core = set(carbons) | imino | single_n
    if any(atom.GetAtomicNum() not in {6, 7} for atom in iter_atoms(mol)):
        return None
    if len(core) != len(imino) + len(single_n) + len(carbons):
        return None
    allowed = set(terminal_by_carbon.values()) | endpoint_imino
    branch_rows, branch_owned = [], set()
    branch_by_n = defaultdict(list)
    for nidx in allowed:
        atom = mol.GetAtomWithIdx(nidx)
        core_neighbors = [neighbor.GetIdx() for neighbor in atom.GetNeighbors()
                          if neighbor.GetIdx() in core]
        expected_core_neighbors = 1
        if len(core_neighbors) != expected_core_neighbors:
            return None
        roots = [neighbor.GetIdx() for neighbor in atom.GetNeighbors()
                 if neighbor.GetIdx() not in core and neighbor.GetAtomicNum() > 1]
        max_branches = 2 if nidx in terminal_by_carbon.values() else 1
        hydrogens = atom.GetTotalNumHs(includeNeighbors=True)
        if len(roots) > max_branches or hydrogens + len(roots) != max_branches:
            return None
        for root in roots:
            text, owned = mapped_branch(mol, Chem, root, nidx)
            if not text or not owned or owned & (core | branch_owned):
                return None
            if not _simple_organyl_graph(mol, Chem, root, nidx, owned, text):
                return None
            branch_owned.update(owned)
            branch_by_n[nidx].append(text)
            branch_rows.append((nidx, text, frozenset(owned)))
    if not branch_rows:
        return None

    # Any nonterminal imino N and bridge N remains the unsubstituted NH form.
    for nidx in imino - endpoint_imino:
        atom = mol.GetAtomWithIdx(nidx)
        if atom.GetDegree() != 1 or atom.GetTotalNumHs(includeNeighbors=True) != 1:
            return None
    for nidx, attached in bridges:
        atom = mol.GetAtomWithIdx(nidx)
        if atom.GetDegree() != 2 or atom.GetTotalNumHs(includeNeighbors=True) != 1:
            return None
    for nidx, carbon in terminals:
        if nidx not in allowed:
            return None
    heavy = frozenset(atom.GetIdx() for atom in iter_atoms(mol)
                      if atom.GetAtomicNum() > 1)
    if core | branch_owned != set(heavy) or set(all_carbons) != (
            set(carbons) | set(branch_owned)):
        return None

    parent_candidates = []
    for oriented_path in (tuple(path), tuple(reversed(path))):
        position_by_carbon = {cidx: 2 * offset + 1
                              for offset, cidx in enumerate(oriented_path)}
        site_label = {}
        for carbon, nidx in terminal_by_carbon.items():
            site_label[nidx] = (position_by_carbon[carbon], 0)
        for nidx in endpoint_imino:
            carbon = imino_to_core[nidx]
            site_label[nidx] = (position_by_carbon[carbon], 1)
        if set(site_label) != allowed:
            return None
        rows_by_name = defaultdict(list)
        for nidx, text, atoms in branch_rows:
            position, prime = site_label[nidx]
            locant = f"N{'′' if prime else ''}^{position}"
            rows_by_name[text].append((position, prime, locant, nidx))
        names = sorted(rows_by_name, key=prefix_alphabetization_key)
        prefixes, chosen_labels = [], {}
        locants_by_name = {}
        for text in names:
            # Compare the parent-position superscript first; at the same
            # position P-14.3.5 places the primed N locant after unprimed N.
            site_rows = sorted(rows_by_name[text], key=lambda row: _n_locant_key(row[0], row[1]))
            multiplied = multiply_substituent(text, len(site_rows), nature="simple")
            if multiplied is None:
                return None
            from .locant_order import nitrogen_site_display
            labels = [nitrogen_site_display(locant) for _, _, locant, _ in site_rows]
            locants_by_name[text] = tuple(_n_locant_key(position, prime)
                                          for position, prime, _, _ in site_rows)
            prefixes.append(f"{','.join(labels)}-{multiplied.text}")
            chosen_labels.update({nidx: locant for _, _, locant, nidx in site_rows})
        name = "-".join(prefixes) + _PARENT[n]
        # P-14.4(f): compare the complete locant set for detachable
        # alphabetized prefixes. Only then apply (g) to the first-cited
        # substituent class (citation order is established above by P-14.5).
        all_locants = tuple(sorted(key for values in locants_by_name.values()
                                   for key in values))
        first_cited_locants = locants_by_name[names[0]]
        parent_candidates.append(((all_locants, first_cited_locants), name,
                                  tuple(oriented_path), chosen_labels))
    best_score = min(row[0] for row in parent_candidates)
    winners = [row for row in parent_candidates if row[0] == best_score]
    if len({row[1] for row in winners}) != 1:
        return None
    _, name, oriented_path, branch_locants = winners[0]

    atom_locants = []
    for offset, cidx in enumerate(oriented_path):
        position = 2 * offset + 1
        atom_locants.append((cidx, str(position)))
        in_n = next(idx for idx in imino
                    if mol.GetBondBetweenAtoms(cidx, idx) is not None
                    and mol.GetBondBetweenAtoms(cidx, idx).GetBondType()
                    == Chem.BondType.DOUBLE)
        atom_locants.append((in_n, f"N′^{position}"))
        if cidx in terminal_by_carbon:
            atom_locants.append((terminal_by_carbon[cidx], f"N^{position}"))
        if offset < len(oriented_path) - 1:
            bridge = bridge_by_pair[frozenset((cidx, oriented_path[offset + 1]))]
            atom_locants.append((bridge, str(position + 1)))
    roles = [("condensed-guanidine-core", frozenset(core), tuple(atom_locants))]
    for nidx, text, owned in branch_rows:
        locant = branch_locants.get(nidx)
        if not locant:
            return None
        roles.append(("N-substituent-at-" + locant, owned, ()))
    return MethodDecision(name, "substitutive", frozenset(carbons), heavy,
                          _RULE, tuple(atom_locants), component_roles=tuple(roles))


def _n_locant_key(position, prime):
    """Order N locants by parent position, then unprimed before primed."""
    from .locant_order import n_site_key

    return n_site_key(position, prime)


def _walk_carbon_path(adjacency, start):
    output, previous, current = [], None, start
    while current is not None:
        if current in output:
            return ()
        output.append(current)
        following = [other for other in adjacency[current] if other != previous]
        if len(following) > 1:
            return ()
        previous, current = current, following[0] if following else None
    return tuple(output)


def _simple_organyl_graph(mol, Chem, root, attachment, atoms, text):
    atoms = set(atoms)
    if any(mol.GetAtomWithIdx(index).GetAtomicNum() != 6 for index in atoms):
        return False
    ring = {index for index in atoms if mol.GetAtomWithIdx(index).IsInRing()}
    if ring:
        return (len(atoms) == len(ring) == 6
                and all(mol.GetAtomWithIdx(index).GetIsAromatic() for index in atoms)
                and text == "phenyl"
                and all(mol.GetBondBetweenAtoms(left, right).GetBondType()
                        == Chem.BondType.AROMATIC
                        for left in atoms for right in atoms if left < right
                        and mol.GetBondBetweenAtoms(left, right) is not None)
                and all(neighbor.GetIdx() in atoms | {attachment}
                        for index in atoms for neighbor in mol.GetAtomWithIdx(index).GetNeighbors()))
    degrees = {index: sum(neighbor.GetIdx() in atoms
                          for neighbor in mol.GetAtomWithIdx(index).GetNeighbors())
               for index in atoms}
    return (degrees.get(root) == (0 if len(atoms) == 1 else 1)
            and all(degree <= 2 for degree in degrees.values())
            and (len(atoms) == 1 or sum(degree == 1 for degree in degrees.values()) == 2)
            and all(mol.GetBondBetweenAtoms(left, right).GetBondType() == Chem.BondType.SINGLE
                    for left in atoms for right in atoms if left < right
                    and mol.GetBondBetweenAtoms(left, right) is not None)
            and all(neighbor.GetIdx() in atoms | {attachment}
                    for index in atoms for neighbor in mol.GetAtomWithIdx(index).GetNeighbors()))
