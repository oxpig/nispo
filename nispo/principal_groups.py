"""Graph-level eligibility and numbering for principal functional parent routes."""

from .molecule_iteration import iter_atoms, iter_bonds
from .branch_rules import collect_subtree, parent_chain_name
from .acyclic.rules import (
    _acyclic_graph_view,
    _path_substituent_prefixes,
    _shortest_path_in_allowed_graph,
)
from .formatting import complete_parent_name, format_prefixes, join_prefix_parent
from .molecule_cache import mol_cache_get, mol_cache_set


def _safe_graph(mol, elements):
    return all(
        a.GetAtomicNum() in elements
        and not (a.GetFormalCharge() or a.GetIsotope() or a.GetNumRadicalElectrons())
        for a in iter_atoms(mol)
    )


def principal_ring_groups(mol):
    """Return the atoms that an eligible alcohol/lactone parent must contain.

    Nonstandard elements, ionic/isotopic states and competing senior groups
    retain their specialized routes. OH multiplicity is counted per oxygen.
    """
    cached = mol_cache_get(mol, "principal_ring_groups")
    if cached is not None:
        return cached
    result = ("", ())
    if mol.GetRingInfo().NumRings() and _safe_graph(mol, {1, 6, 8, 9, 17, 35, 53}):
        carbonyls, alcohols, esters = [], [], []
        unsupported = False
        for oxygen in (a for a in iter_atoms(mol) if a.GetAtomicNum() == 8):
            if oxygen.GetTotalNumHs(includeNeighbors=True) == 1:
                neighbors = list(oxygen.GetNeighbors())
                if len(neighbors) == 1 and neighbors[0].IsInRing():
                    alcohols.append(neighbors[0].GetIdx())
                else:
                    unsupported = True
            for bond in oxygen.GetBonds():
                if bond.GetBondTypeAsDouble() != 2:
                    continue
                carbon = bond.GetOtherAtom(oxygen)
                carbonyls.append(carbon.GetIdx())
                single_o = [
                    a
                    for a in carbon.GetNeighbors()
                    if a.GetAtomicNum() == 8
                    and mol.GetBondBetweenAtoms(
                        carbon.GetIdx(), a.GetIdx()
                    ).GetBondTypeAsDouble()
                    in {1, 1.5}
                ]
                if (
                    carbon.IsInRing()
                    and len(single_o) == 1
                    and single_o[0].IsInRing()
                    and single_o[0].GetDegree() == 2
                ):
                    esters.append((carbon.GetIdx(), single_o[0].GetIdx()))
        if carbonyls and len(esters) == len(carbonyls):
            result = ("lactone", tuple(i for pair in esters for i in pair))
        elif (
            not carbonyls
            and alcohols
            and not unsupported
            and _fused_alcohol_sites(mol, alcohols)
        ):
            result = ("alcohol", tuple(alcohols))
    return mol_cache_set(mol, "principal_ring_groups", result)


def _fused_alcohol_sites(mol, alcohols):
    """Select fused/complex ring alcohols beyond the simple bicyclic route."""
    rings = [set(ring) for ring in mol.GetRingInfo().AtomRings()]
    fused_atoms = set()
    for index, ring in enumerate(rings):
        for other in rings[index + 1 :]:
            shared = ring & other
            if len(shared) == 2 and mol.GetBondBetweenAtoms(*shared) is not None:
                fused_atoms.update(ring | other)
    if set(alcohols) <= fused_atoms:
        return True
    # The simple saturated bicyclic route already selects and numbers its
    # systematic suffix parent. Preserve it rather than reopening retained
    # alternatives through the general principal-ring selector.
    from .polycyclic_rules import _simple_bicyclic_oxygen_suffix_facts
    if _simple_bicyclic_oxygen_suffix_facts(mol, frozenset().union(*rings)):
        return False
    # A connected carbon-only bridged/spiro system also has established
    # hydride numbering; unsaturation does not demote an alcohol to a prefix.
    systems = []
    for ring in rings:
        overlapping = [system for system in systems if system & ring]
        merged = set(ring)
        for system in overlapping:
            merged.update(system)
            systems.remove(system)
        systems.append(merged)
    return any(set(alcohols) <= system
               and sum(r <= system for r in rings) >= 2
               and all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in system)
               for system in systems)


def sole_sulfonamide(mol):
    """Recognize a supported, sole sulfonamide without hiding a senior group."""
    cached = mol_cache_get(mol, "sole_sulfonamide")
    if cached is not None:
        return cached
    result = any(
        a.GetAtomicNum() == 16 for a in iter_atoms(mol)
    ) and _sole_sulfonamide_graph(mol)
    return mol_cache_set(mol, "sole_sulfonamide", result)


def _sole_sulfonamide_graph(mol):
    if not _safe_graph(mol, {1, 6, 7, 8, 9, 16, 17, 35, 53}):
        return False
    if any(int(a.GetChiralTag()) and a.GetAtomicNum() in {7, 16} for a in iter_atoms(mol)):
        return False
    centers = 0
    for atom in iter_atoms(mol):
        if atom.GetAtomicNum() == 6:
            # Ordinary acyclic amides and esters are senior; ring N-linked
            # acyl groups are pseudoketones and do not suppress sulfonamide.
            carbonyl = any(a.GetAtomicNum() == 8 and mol.GetBondBetweenAtoms(atom.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() == 2 for a in atom.GetNeighbors())
            if any(a.GetAtomicNum() == 7 and mol.GetBondBetweenAtoms(atom.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() > 1 for a in atom.GetNeighbors()):
                return False
            if carbonyl and any(a.GetAtomicNum() == 8 and mol.GetBondBetweenAtoms(atom.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() == 1 or a.GetAtomicNum() == 7 and not a.IsInRing() for a in atom.GetNeighbors()):
                return False
        if atom.GetAtomicNum() != 16:
            continue
        neighbors = list(atom.GetNeighbors())
        double_o = [
            a
            for a in neighbors
            if a.GetAtomicNum() == 8
            and mol.GetBondBetweenAtoms(atom.GetIdx(), a.GetIdx()).GetBondTypeAsDouble()
            == 2
        ]
        single = [a for a in neighbors if a not in double_o]
        if len(double_o) != 2 or len(single) != 2:
            return False
        symbols = sorted(a.GetAtomicNum() for a in single)
        if symbols == [6, 7]:
            nitrogen = next(a for a in single if a.GetAtomicNum() == 7)
            if any(a.GetAtomicNum() == 7 for a in nitrogen.GetNeighbors()):
                return False  # P-66.3 hydrazide, not an ordinary class-11 amide.
            centers += 1
        elif symbols != [6, 6]:
            return False  # Sulfonic acids, hydrazides and other S groups.
    return centers == 1


def substituted_alkanesulfonamide_base(mol, sulfur_idx, carbon_idx):
    """Build an alkanesulfonamide parent from its graph, including halo prefixes."""
    subtree = set(collect_subtree(mol, carbon_idx, sulfur_idx))
    if not subtree or not all(
        mol.GetAtomWithIdx(i).GetAtomicNum() in {1, 6, 9, 17, 35, 53}
        and not mol.GetAtomWithIdx(i).IsInRing()
        for i in subtree
    ):
        return ""
    if any(
        mol.GetAtomWithIdx(i).GetFormalCharge()
        or mol.GetAtomWithIdx(i).GetIsotope()
        or mol.GetAtomWithIdx(i).GetNumRadicalElectrons()
        or int(mol.GetAtomWithIdx(i).GetChiralTag())
        for i in subtree
    ) or any(
        int(b.GetStereo()) and {b.GetBeginAtomIdx(), b.GetEndAtomIdx()} <= subtree
        for b in iter_bonds(mol)
    ):
        return ""
    carbons = {i for i in subtree if mol.GetAtomWithIdx(i).GetAtomicNum() == 6}
    if len(carbons) > 40:
        return ""
    view = _acyclic_graph_view(mol)
    endpoints = sorted(
        i for i in carbons if sum(j in carbons for j in view.neighbors[i]) <= 1
    )
    paths = {(i,) for i in carbons}
    for index, a in enumerate(endpoints):
        for b in endpoints[index + 1 :]:
            path = _shortest_path_in_allowed_graph(view, a, b, carbons)
            if path:
                paths.add(path)
    candidates = []
    for path in paths:
        if carbon_idx not in path:
            continue
        for oriented in (path, tuple(reversed(path))):
            locant = oriented.index(carbon_idx) + 1
            prefixes = _path_substituent_prefixes(
                mol, oriented, skip_atoms={sulfur_idx}, skip_plain_hydrogen=True
            )
            if prefixes is None:
                continue
            parent = complete_parent_name(parent_chain_name(mol, oriented))
            if not parent.endswith("ane"):
                continue
            suffix = "sulfonamide" if len(path) <= 2 else f"-{locant}-sulfonamide"
            name = join_prefix_parent(format_prefixes(prefixes), parent + suffix)
            candidates.append(
                (
                    (
                        -len(path),
                        locant,
                        tuple(sorted(int(loc) for loc, _ in prefixes)),
                        name,
                    ),
                    name,
                    parent + suffix,
                    tuple(prefixes),
                )
            )
    if candidates:
        selected = min(candidates)
        mol_cache_set(
            mol, ("alkanesulfonamide_base", sulfur_idx, carbon_idx), selected[2:]
        )
        return selected[1]
    return ""


def alcohol_substituent_prefix(mol, root_idx, parent_idx):
    """Render a neutral OH-bearing carbon branch in prefix mode under a lactone."""
    if (
        mol.GetAtomWithIdx(root_idx).GetAtomicNum() != 6
        or mol.GetAtomWithIdx(root_idx).IsInRing()
    ):
        return ""
    subtree = set(collect_subtree(mol, root_idx, parent_idx))
    oxygens = [
        mol.GetAtomWithIdx(idx)
        for idx in subtree
        if mol.GetAtomWithIdx(idx).GetAtomicNum() == 8
    ]
    if not oxygens or any(
        o.GetDegree() != 1 or o.GetTotalNumHs(includeNeighbors=True) != 1
        for o in oxygens
    ):
        return ""
    if any(
        mol.GetAtomWithIdx(idx).GetAtomicNum() not in {1, 6, 8, 9, 17, 35, 53}
        or mol.GetAtomWithIdx(idx).GetFormalCharge()
        or mol.GetAtomWithIdx(idx).GetIsotope()
        for idx in subtree
    ):
        return ""
    allowed = {
        idx
        for idx in subtree
        if mol.GetAtomWithIdx(idx).GetAtomicNum() == 6
        and not mol.GetAtomWithIdx(idx).IsInRing()
    }
    if len(allowed) > 40:
        return ""
    from .branch.carbon import _locanted_substituent_path_name

    view = _acyclic_graph_view(mol)
    endpoints = sorted(
        i for i in allowed if sum(j in allowed for j in view.neighbors[i]) <= 1
    )
    paths = {(root_idx,)}
    for index, a in enumerate(endpoints):
        for b in endpoints[index + 1 :]:
            path = _shortest_path_in_allowed_graph(view, a, b, allowed)
            if root_idx in path:
                paths.add(path)
    results = []
    for path in paths:
        for oriented in (path, tuple(reversed(path))):
            name = _locanted_substituent_path_name(
                mol, oriented, subtree, root_idx, parent_idx
            )
            if name:
                results.append(((-len(path), oriented.index(root_idx) + 1, name), name))
    return min(results)[1] if results else ""
