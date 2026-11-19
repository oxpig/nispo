"""Mapped constructions for a deliberately bounded P-66.4.1 domain.

This module is a standalone producer. It does not alter functional inventory
or route selection. Unsupported tautomer, ring, and senior-function questions
return ``None`` so the caller can preserve its existing naming routes.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

from .molecule_iteration import iter_atoms, iter_bonds
from .component_mapping import MappedComponent, decorate_component, mapped_branch
from .formatting import attach_suffix
from .naming_methods import MethodDecision


@lru_cache(maxsize=1)
def _amidoxime_query(Chem):
    return Chem.MolFromSmarts("[C;X3;!R](=[N;X2;!R][O;X2])([N;X3;H1,H2;!R;+0])[#6]")


def amidoxime_decision(mol, Chem):
    """P-66.4.4 for one neutral amidoxime on a mapped ring or carbon chain."""
    matches = mol.GetSubstructMatches(_amidoxime_query(Chem))
    if len(matches) != 1 or mol.GetStereoGroups() or any(
            a.GetAtomicNum() == 1 or a.GetIsotope()
            or a.GetNumRadicalElectrons() for a in iter_atoms(mol)):
        return None
    carbon, imino, oxygen, amino, owner = matches[0]
    from .extended_inventory import selection_inventory

    inventory = selection_inventory(mol)
    if (not inventory.supported or inventory.source_frontier.status != 'selected'
            or len(inventory.principal) != 1 or inventory.principal[0].kind != 'amidoxime'
            or inventory.principal[0].center != carbon):
        return None
    from .neutral_charge_components import validate_neutral_charge_role_proofs

    if not validate_neutral_charge_role_proofs(mol, inventory.charge_role_proofs):
        return None
    stereo = ""
    for bond in iter_bonds(mol):
        if not bond.GetStereo():
            continue
        if {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()} != {carbon, imino}:
            if mol.GetAtomWithIdx(owner).IsInRing():
                continue
            return None
        from .stereochemistry import assigned_bond_stereochemistry
        stereo = next((value for a, b, value in assigned_bond_stereochemistry(mol, Chem)
                       if {a, b} == {carbon, imino}), "")
        if stereo not in {"E", "Z"}:
            return None
    oxygen_prefix, oxygen_atoms = mapped_branch(mol, Chem, oxygen, imino)
    if not oxygen_prefix or not oxygen_atoms:
        return None
    unit = frozenset({carbon, imino, amino}) | oxygen_atoms
    n_prefixes = [("N'", oxygen_prefix)]
    for ligand in mol.GetAtomWithIdx(amino).GetNeighbors():
        root = ligand.GetIdx()
        if root == carbon:
            continue
        from .component_mapping import mapped_general_branch

        name, atoms = mapped_general_branch(mol, Chem, root, amino)
        if not name or not atoms or unit & atoms or owner in atoms:
            return None
        n_prefixes.append(("N", name))
        unit |= atoms

    ring_parent = mol.GetAtomWithIdx(owner).IsInRing()
    if ring_parent:
        rings = mol.GetRingInfo().AtomRings()
        rings_at_owner = [ring for ring in rings if owner in ring]
        ring = rings_at_owner[0] if len(rings_at_owner) == 1 else ()
        if (len(n_prefixes) > 1 or len(ring) != 6
                or any(set(ring) & set(other) for other in rings if other != ring)
                or any(not mol.GetAtomWithIdx(i).GetIsAromatic()
                       or mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in ring)):
            return _ring_amidoxime(mol, Chem, carbon, owner, unit, n_prefixes, stereo)
        start = ring.index(owner)
        ring = ring[start:] + ring[:start]
        paths = (ring, (owner, *reversed(ring[1:])))
    else:
        from .branch.preferred_acyl import _carbon_paths
        paths = _carbon_paths(mol, carbon, {imino, amino})
        if not paths:
            return None
        maximum = max(map(len, paths))
        paths = [path for path in paths if len(path) == maximum]
        if len(paths) != 1:
            return None

    # The simple chain and benzene paths do not carry atom stereo maps.
    if any(a.GetChiralTag() for a in iter_atoms(mol)):
        return None
    from .branch_rules import parent_chain_name
    from .formatting import format_prefixes, join_prefix_parent, prefix_alphabetization_key
    candidates = []
    atoms = frozenset(range(mol.GetNumAtoms()))
    for path in paths:
        covered = set(path) | unit
        prefixes = []
        for locant, idx in enumerate(path, 1):
            for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
                root = neighbor.GetIdx()
                if root in covered:
                    continue
                bond = mol.GetBondBetweenAtoms(idx, root)
                text, owned = mapped_branch(mol, Chem, root, idx)
                if not text or owned & covered:
                    return None
                order = bond.GetBondTypeAsDouble()
                if order != 1 and not (not ring_parent and order == 2
                                       and text == "oxo" and owned == {root}
                                       and neighbor.GetAtomicNum() == 8
                                       and neighbor.GetDegree() == 1
                                       and not neighbor.GetFormalCharge()):
                    return None
                prefixes.append((str(locant), text))
                covered.update(owned)
        if covered != atoms:
            return None
        parent = (attach_suffix("benzene", "carboximidamide", ("1",) if prefixes else ())
                  if ring_parent else attach_suffix(parent_chain_name(mol, path), "imidamide"))
        if not parent:
            return None
        ordered = sorted(prefixes, key=lambda row: (prefix_alphabetization_key(row[1]), int(row[0])))
        locants = (tuple(sorted(int(locant) for locant, _ in prefixes)),
                   tuple(int(locant) for locant, _ in ordered))
        prefix_text = format_prefixes([*prefixes, *n_prefixes],
                                      hierarchical_enclosures=True)
        name = join_prefix_parent(prefix_text, parent)
        if stereo:
            name = f"({stereo})-{name}"
        candidates.append((locants, name, tuple((idx, str(i)) for i, idx in enumerate(path, 1))))
    _, name, mapping = min(candidates)
    return MethodDecision(name, "substitutive", frozenset({carbon}), atoms,
                          "P-66.4.4/P-93.4.2.1", mapping)


def _ring_amidoxime(mol, Chem, carbon, owner, unit, n_prefixes, stereo):
    """Reuse the numbered ring and its original-graph branch coverage."""
    from .mapped_ring_parents import records

    atoms = frozenset(range(mol.GetNumAtoms()))
    ring_scope = atoms - unit
    from .mapped_ring_parents import ring_systems

    core = next((core for core in ring_systems(mol) if owner in core), None)
    if core is None:
        return None
    candidates = records(mol, Chem, scope=ring_scope, attachment=owner, core_filter=core,
                         general_branches=True)
    if len(candidates) != 1:
        return None
    record = candidates[0]
    component = record.construction
    if (component is None or not record.hydrogen_budget_proved or not record.annotation_coverage
            or component.atoms | component.embedded_atoms != ring_scope):
        return None
    locant = dict(component.atom_locants).get(owner)
    if not locant:
        return None
    # Replace the typed free-valence suffix while preserving the separate
    # hydrogenation and indicated-H operations of the same numbered hydride.
    yl_ending = f"-{locant}-yl"
    if not component.parent.endswith(yl_ending):
        return None
    stem = component.parent.removesuffix(yl_ending)
    # ``yl`` elides the parent's vowel. This applies equally to an
    # absolute unsaturated ring stem (e.g. hex-2-en) and to a saturated one.
    hydride = stem if stem.endswith('e') else stem + 'e'
    parent = attach_suffix(hydride, 'carboximidamide', (locant,))
    name = decorate_component(mol, Chem, replace(
        component, parent=parent, prefixes=(*component.prefixes, *n_prefixes),
        embedded_atoms=component.embedded_atoms | unit,
    ))
    if not name:
        return None
    if stereo:
        from .stereochemistry import _prepend_stereochemical_descriptors

        name = _prepend_stereochemical_descriptors(name, [stereo])
    return MethodDecision(name, "substitutive", frozenset({carbon}), atoms,
                          "P-66.4.4/P-93.4.2.1", component.atom_locants,
                          preference_issues=record.preference_issues)


def amidine_decision(mol, Chem):
    """Return one complete mapped decision for simple amidines or guanidine.

    Ordinary amidines are limited to a single neutral acyclic ``C(=N)-N``
    group on a unique unbranched saturated carbon parent. Ordinary terminal
    halogens on that path and on N are mapped explicitly. Each amidine N may
    also carry carbon-only acyclic branches. N and N-prime identify
    amino and imino nitrogen, respectively (P-66.4.1.4.1). The only guanidine
    form admitted is unsubstituted ``NC(=N)N``; substituted guanidines need
    the full tautomer-aware prime-locant comparison of P-66.4.1.2.1.2.

    Carbon isotopes and atom stereochemistry on carbon-only N branches are
    kept by the mapped component writer. Core-N labels, charged forms,
    isotope-H, bond geometry, other characteristic groups, and unsupported
    branch graphs decline.
    """
    if mol.GetNumHeavyAtoms() > 128 or mol.GetStereoGroups():
        return None
    if any(a.GetAtomicNum() == 1 or a.GetFormalCharge()
           or a.GetNumRadicalElectrons() for a in iter_atoms(mol)):
        return None
    halogen_names = {9:'fluoro',17:'chloro',35:'bromo',53:'iodo'}
    if any(a.GetAtomicNum() not in {6, 7, *halogen_names} for a in iter_atoms(mol)):
        return None
    if any(a.GetAtomicNum() == 7 and (a.GetIsotope() or int(a.GetChiralTag()))
           for a in iter_atoms(mol)):
        return None
    if any(int(b.GetStereo()) for b in iter_bonds(mol)):
        return None

    sites = []
    for carbon in iter_atoms(mol):
        if carbon.GetAtomicNum() != 6 or carbon.GetFormalCharge() or carbon.IsInRing():
            continue
        cidx = carbon.GetIdx()
        double_n, single_n, carbon_neighbors = [], [], []
        valid = True
        for neighbor in carbon.GetNeighbors():
            nidx = neighbor.GetIdx()
            order = mol.GetBondBetweenAtoms(cidx, nidx).GetBondTypeAsDouble()
            if neighbor.GetAtomicNum() == 7 and order == 2:
                double_n.append(nidx)
            elif neighbor.GetAtomicNum() == 7 and order == 1:
                single_n.append(nidx)
            elif neighbor.GetAtomicNum() == 6 and order == 1:
                carbon_neighbors.append(nidx)
            elif neighbor.GetAtomicNum() > 1:
                valid = False
        if valid and len(double_n) == 1 and len(single_n) in {1, 2}:
            sites.append((cidx, double_n[0], tuple(single_n), tuple(carbon_neighbors)))
    if len(sites) != 1:
        return None
    carbon, imino_n, amino_ns, carbon_neighbors = sites[0]

    # Guanidine has a separate retained preferred parent and N/N'/N'' locants.
    if len(amino_ns) == 2:
        if carbon_neighbors:
            return None
        core = {carbon, imino_n, *amino_ns}
        heavy = {a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1}
        if core != heavy:
            return None
        # The retained parent writer below has no typed isotope path yet.
        if any(mol.GetAtomWithIdx(i).GetIsotope() for i in core):
            return None
        if any(mol.GetAtomWithIdx(i).GetTotalNumHs(includeNeighbors=True) != 2
               for i in amino_ns):
            return None
        imino = mol.GetAtomWithIdx(imino_n)
        if imino.GetTotalNumHs(includeNeighbors=True) != 1:
            return None
        # The source assigns the retained name and three prime locants but
        # does not require a particular atom-order-dependent locant choice for
        # the two equivalent amino nitrogens.
        n_ids = tuple(sorted(amino_ns, key=lambda i: _neighbor_signature(mol, i)))
        mapping = {carbon: "", imino_n: "N′′", n_ids[0]: "N", n_ids[1]: "N′"}
        roles = (("guanidine-parent", frozenset(core), tuple(mapping.items())),)
        return MethodDecision("guanidine", "substitutive", frozenset({carbon}),
                              frozenset(heavy), "P-66.4.1.2.1.1",
                              tuple(mapping.items()), component_roles=roles)

    if len(carbon_neighbors) > 1:
        return None
    # Carbon parent is a unique simple saturated path ending at the amidine C.
    parent_atoms = {carbon}
    stack = [carbon]
    while stack:
        idx = stack.pop()
        for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
            other = neighbor.GetIdx()
            if neighbor.GetAtomicNum() == 1 or other in {imino_n, *amino_ns}:
                continue
            if (neighbor.GetAtomicNum() in halogen_names and neighbor.GetDegree() == 1
                    and mol.GetBondBetweenAtoms(idx,other).GetBondType() == Chem.BondType.SINGLE):
                continue
            if neighbor.GetAtomicNum() != 6:
                return None
            bond = mol.GetBondBetweenAtoms(idx, other)
            if bond.GetBondType() != Chem.BondType.SINGLE or neighbor.IsInRing():
                return None
            if other not in parent_atoms:
                parent_atoms.add(other)
                stack.append(other)
    if carbon_neighbors and carbon_neighbors[0] not in parent_atoms:
        return None
    if any(sum(n.GetIdx() in parent_atoms for n in mol.GetAtomWithIdx(i).GetNeighbors()) > 2
           for i in parent_atoms):
        return None
    ends = [i for i in parent_atoms if sum(n.GetIdx() in parent_atoms
                                            for n in mol.GetAtomWithIdx(i).GetNeighbors()) <= 1]
    if len(parent_atoms) == 1:
        path = (carbon,)
    elif len(ends) == 2:
        paths = (_walk(mol, ends[0], parent_atoms), _walk(mol, ends[1], parent_atoms))
        path = next((p for p in paths if len(p) == len(parent_atoms) and p[0] == carbon), ())
        if not path:
            return None
    else:
        return None
    if path[0] != carbon:
        return None
    from .branch_rules import parent_chain_name
    from .formatting import complete_parent_name

    parent = complete_parent_name(parent_chain_name(mol, path))
    if not parent:
        return None
    parent = attach_suffix(parent, "imidamide")

    core = {carbon, imino_n, amino_ns[0], *path}
    if any(mol.GetAtomWithIdx(i).GetIsotope() for i in (imino_n, amino_ns[0])):
        return None
    mapping = {idx: str(pos) for pos, idx in enumerate(path, 1)}
    mapping[amino_ns[0]] = "N"
    mapping[imino_n] = "N′"
    embedded, prefixes, branch_roles = set(), [], []
    for idx in parent_atoms:
        for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
            other = neighbor.GetIdx()
            if neighbor.GetAtomicNum() in halogen_names:
                if neighbor.GetDegree() != 1 or neighbor.GetIsotope():
                    return None
                prefixes.append((mapping[idx],halogen_names[neighbor.GetAtomicNum()]))
                embedded.add(other)
                branch_roles.append(('carbon-halogen',frozenset({other}),()))
    n_atoms = (amino_ns[0], imino_n)
    n_locants = {amino_ns[0]: "N", imino_n: "N′"}
    for nidx in n_atoms:
        nitrogen = mol.GetAtomWithIdx(nidx)
        if nitrogen.IsInRing() or nitrogen.GetFormalCharge() or nitrogen.GetNumRadicalElectrons():
            return None
        if nidx == imino_n and nitrogen.GetTotalNumHs(includeNeighbors=True) not in {0, 1}:
            return None
        if nidx == amino_ns[0] and nitrogen.GetTotalNumHs(includeNeighbors=True) not in {0, 1, 2}:
            return None
        roots = [a.GetIdx() for a in nitrogen.GetNeighbors()
                 if a.GetIdx() != carbon and a.GetAtomicNum() > 1]
        expected_n_substitution = 1 if nidx == imino_n else 2
        if len(roots) + nitrogen.GetTotalNumHs(includeNeighbors=True) != expected_n_substitution:
            return None
        for root in roots:
            atom = mol.GetAtomWithIdx(root)
            if atom.GetAtomicNum() in halogen_names and atom.GetDegree() == 1 and not atom.GetIsotope():
                text, mapped = halogen_names[atom.GetAtomicNum()], frozenset({root})
                owned = mapped
            else:
                side, owned = _ordinary_carbon_branch(mol, root, nidx, Chem)
                if not side:
                    return None
                text, mapped = mapped_branch(mol, Chem, root, nidx)
            if not text or set(mapped) != set(owned) or set(owned) & (parent_atoms | embedded):
                return None
            prefixes.append((n_locants[nidx], text))
            embedded.update(owned)
            branch_roles.append(("N-substituent-at-" + n_locants[nidx],
                                 frozenset(owned), ()))
    heavy = {a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1}
    if parent_atoms | {imino_n, amino_ns[0]} | embedded != heavy:
        return None
    # In the admitted parent form the imino N is NH and the amino N has one
    # hydrogen before N-substitution. This rejects alternate valence/tautomer
    # encodings rather than treating a graph representation as a naming rule.
    if mol.GetAtomWithIdx(imino_n).GetTotalNumHs(includeNeighbors=True) + len(
            [p for p in prefixes if p[0] == "N′"]) != 1:
        return None
    if mol.GetAtomWithIdx(amino_ns[0]).GetTotalNumHs(includeNeighbors=True) + len(
            [p for p in prefixes if p[0] == "N"]) != 2:
        return None
    loc_bindings = {idx: loc for idx, loc in mapping.items()}
    component = MappedComponent(parent, tuple(loc_bindings.items()), frozenset(loc_bindings),
        tuple(prefixes), frozenset(embedded), nitrogen_site_locants=True)
    name = decorate_component(mol, Chem, component)
    if not name:
        return None
    roles = (("amidine-parent", frozenset(parent_atoms | {imino_n, amino_ns[0]}),
              tuple(mapping.items())), *branch_roles)
    return MethodDecision(name, "substitutive", frozenset({carbon}), frozenset(heavy),
                          "P-66.4.1.1/P-66.4.1.4.1/P-44.3", tuple(mapping.items()),
                          component_roles=roles)


def _walk(mol, start, allowed):
    result, previous, current = [], -1, start
    while True:
        result.append(current)
        children = [n.GetIdx() for n in mol.GetAtomWithIdx(current).GetNeighbors()
                    if n.GetIdx() in allowed and n.GetIdx() != previous]
        if not children:
            return tuple(result)
        if len(children) != 1:
            return ()
        previous, current = current, children[0]


def _ordinary_carbon_branch(mol, root, blocked, Chem):
    atoms, stack = set(), [root]
    while stack:
        idx = stack.pop()
        if idx in atoms:
            continue
        atom = mol.GetAtomWithIdx(idx)
        if (atom.GetAtomicNum() != 6 or atom.GetFormalCharge()
                or atom.GetNumRadicalElectrons() or atom.IsInRing()):
            return None, frozenset()
        atoms.add(idx)
        for neighbor in atom.GetNeighbors():
            other = neighbor.GetIdx()
            if other == blocked or neighbor.GetAtomicNum() == 1:
                continue
            if (neighbor.GetAtomicNum() != 6
                    or mol.GetBondBetweenAtoms(idx, other).GetBondType() != Chem.BondType.SINGLE):
                return None, frozenset()
            stack.append(other)
    return atoms, frozenset(atoms)


def _neighbor_signature(mol, idx):
    atom = mol.GetAtomWithIdx(idx)
    return tuple(sorted((n.GetAtomicNum(), mol.GetBondBetweenAtoms(idx, n.GetIdx()).GetBondTypeAsDouble())
                        for n in atom.GetNeighbors()))
