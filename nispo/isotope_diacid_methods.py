"""Source-shaped P-82.2.2.2 consumer for geminal cycloalkane diacids."""

from __future__ import annotations

from .characteristic_isotope_selection import (
    CharacteristicIsotopeSelection,
    CharacteristicGroupMap,
    select_characteristic_isotope_frontier,
)
from .naming_methods import MethodDecision
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IsotopeDiacidDecision(MethodDecision):
    """Decision with acid maps and parent-host attachments kept separate."""

    acid_groups: tuple[CharacteristicGroupMap, ...] = ()
    acid_attachment_locants: tuple[tuple[str, int, str], ...] = ()
    selector_result: CharacteristicIsotopeSelection | None = None


def _acid_groups(mol):
    groups = []
    for carbon in mol.GetAtoms():
        if carbon.GetAtomicNum() != 6:
            continue
        center = carbon.GetIdx()
        oxygens = [n for n in carbon.GetNeighbors() if n.GetAtomicNum() == 8]
        double = [n for n in oxygens
                  if mol.GetBondBetweenAtoms(center, n.GetIdx()).GetBondTypeAsDouble() == 2]
        single = [n for n in oxygens
                  if mol.GetBondBetweenAtoms(center, n.GetIdx()).GetBondTypeAsDouble() == 1]
        carbon_neighbors = [n for n in carbon.GetNeighbors()
                            if n.GetAtomicNum() == 6
                            and mol.GetBondBetweenAtoms(center, n.GetIdx()).GetBondTypeAsDouble() == 1]
        if len(double) != 1 or len(single) != 1 or len(carbon_neighbors) != 1:
            continue
        hydroxyl = single[0]
        isotope_h = tuple(sorted(
            n.GetIdx() for n in hydroxyl.GetNeighbors()
            if n.GetAtomicNum() == 1 and n.GetIsotope()
        ))
        if hydroxyl.GetFormalCharge() or (
            hydroxyl.GetTotalNumHs(includeNeighbors=True) == 0 and not isotope_h
        ):
            continue
        roles = (
            ("carboxyl_carbon", (center,)),
            ("carbonyl_oxygen", (double[0].GetIdx(),)),
            ("hydroxyl_oxygen", (hydroxyl.GetIdx(),)),
            ("hydroxyl_hydrogen", isotope_h),
        )
        groups.append({
            "center": center,
            "host": carbon_neighbors[0].GetIdx(),
            "atoms": frozenset(i for _role, ids in roles for i in ids),
            "roles": roles,
        })
    return tuple(groups)


def _cycle_and_host(mol, groups):
    rings = tuple(mol.GetRingInfo().AtomRings())
    if len(rings) != 1 or not 3 <= len(rings[0]) <= 10 or len(groups) != 2:
        return None
    ring = frozenset(rings[0])
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in ring):
        return None
    for idx in ring:
        atom = mol.GetAtomWithIdx(idx)
        ring_neighbors = [n for n in atom.GetNeighbors() if n.GetIdx() in ring]
        if len(ring_neighbors) != 2:
            return None
        if any(mol.GetBondBetweenAtoms(idx, n.GetIdx()).GetBondTypeAsDouble() != 1
               for n in ring_neighbors):
            return None
    host = groups[0]["host"]
    if groups[1]["host"] != host or host not in ring:
        return None
    centers = {group["center"] for group in groups}
    if sum(n.GetIdx() in centers for n in mol.GetAtomWithIdx(host).GetNeighbors()) != 2:
        return None
    if any(len([n for n in mol.GetAtomWithIdx(group["center"]).GetNeighbors()
                if n.GetIdx() == host]) != 1 for group in groups):
        return None
    # The host is the only ring atom with non-ring neighbors, and those are
    # precisely the two mapped carboxyl carbons.
    for idx in ring:
        outside = [n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors() if n.GetIdx() not in ring]
        if set(outside) != (centers if idx == host else set()):
            return None
    return ring, host


def _ring_locants(mol, ring, host):
    """Map a symmetry-equivalent ring direction; it cannot affect the PIN."""
    ring_neighbors = sorted(n.GetIdx() for n in mol.GetAtomWithIdx(host).GetNeighbors()
                            if n.GetIdx() in ring)
    if len(ring_neighbors) != 2:
        return None
    order = [host]
    previous = host
    current = ring_neighbors[0]
    while current != host:
        if current in order:
            return None
        order.append(current)
        choices = [n.GetIdx() for n in mol.GetAtomWithIdx(current).GetNeighbors()
                   if n.GetIdx() in ring and n.GetIdx() != previous]
        if len(choices) != 1:
            return None
        previous, current = current, choices[0]
    if len(order) != len(ring):
        return None
    return tuple((idx, str(i + 1)) for i, idx in enumerate(order))


def isotope_diacid_decision(mol, Chem):
    """Return a mapped mixed prefix/suffix name for one geminal diacid family.

    The graph scope is a neutral saturated cycloalkane (3–10 ring atoms) with
    exactly two carboxyl groups attached to the same ring carbon and no other
    substituents. Only mapped carboxyl-carbon isotopes are composed. Separate
    acid maps and attachment locants preserve ownership: the ring host belongs
    to the parent and is never included in an acid component's atom set.
    """
    if (mol.GetStereoGroups()
            or any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in mol.GetAtoms())
            or any(int(a.GetChiralTag()) for a in mol.GetAtoms())
            or any(b.GetStereo() != Chem.BondStereo.STEREONONE for b in mol.GetBonds())
            or any(b.GetBondDir() != Chem.BondDir.NONE for b in mol.GetBonds())):
        return None
    groups = _acid_groups(mol)
    parent = _cycle_and_host(mol, groups)
    if parent is None:
        return None
    ring, host = parent
    group_atoms = set().union(*(group["atoms"] for group in groups))
    heavy_atoms = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
    if heavy_atoms != ring | group_atoms:
        return None
    if any(a.GetAtomicNum() not in {1, 6, 8} for a in mol.GetAtoms()):
        return None

    centers = {group["center"] for group in groups}
    if any(a.GetIsotope() and a.GetIdx() not in centers for a in mol.GetAtoms()):
        return None
    label_masses = []
    for group in groups:
        labels = [(idx, mol.GetAtomWithIdx(idx).GetAtomicNum(), mol.GetAtomWithIdx(idx).GetIsotope())
                  for idx in group["atoms"] if mol.GetAtomWithIdx(idx).GetIsotope()]
        if len(labels) > 1 or any(z != 6 or idx != group["center"]
                                  for idx, z, _mass in labels):
            return None
        label_masses.append(labels[0][2] if labels else 0)

    maps = tuple(CharacteristicGroupMap(
        f"acid-{i + 1}", group["atoms"], group["roles"]
    ) for i, group in enumerate(groups))
    selection = select_characteristic_isotope_frontier(
        mol, maps, frozenset(group_atoms)
    )
    if selection is None or selection.barrier or len(selection.principal_frontier) != 1:
        return None
    locants = _ring_locants(mol, ring, host)
    if locants is None:
        return None
    parent_name = _cycloalkane_name(len(ring))

    if selection.criterion == "single-profile":
        # P-82.2.2.2 gives multiplicative naming for identically modified
        # groups; the isotope mass is part of the descriptor, not a selector
        # whitelist.
        if not label_masses[0] or label_masses[0] != label_masses[1]:
            return None
        name = f"{parent_name}-1,1-di[({label_masses[0]}C)carboxylic acid]"
        suffix_atoms = frozenset(group_atoms)
        prefix_atoms = frozenset()
        roles = (
            ("parent hydride", ring, locants),
            ("multiplicative acid suffixes", suffix_atoms, ()),
        )
    else:
        cluster = selection.principal_frontier[0]
        member_indices = selection.multiplicity_groups[cluster]
        if len(member_indices) != 1:
            return None
        selected_index = member_indices[0]
        prefix_index = 1 - selected_index
        if label_masses[selected_index] <= 0:
            return None
        selected, prefix = groups[selected_index], groups[prefix_index]
        prefix_mass, suffix_mass = label_masses[prefix_index], label_masses[selected_index]
        prefix_text = "1-" + (f"({prefix_mass}C)" if prefix_mass else "") + "carboxy"
        suffix_text = f"({suffix_mass}C)" if suffix_mass else ""
        name = prefix_text + parent_name + "-1-" + suffix_text + "carboxylic acid"
        suffix_atoms = selected["atoms"]
        prefix_atoms = prefix["atoms"]
        roles = (
            ("parent hydride", ring, locants),
            ("principal acid suffix", suffix_atoms, ()),
            ("carboxy prefix", prefix_atoms, ()),
        )

    if (ring & group_atoms or prefix_atoms & suffix_atoms
            or ring | suffix_atoms | prefix_atoms != heavy_atoms):
        return None
    return IsotopeDiacidDecision(
        name,
        "substitutive",
        frozenset(centers),
        heavy_atoms,
        "P-82.2.2.2/P-65.1.2.2.2",
        locants,
        component_roles=roles,
        acid_groups=maps,
        acid_attachment_locants=tuple((group.group_id, host, "1") for group in maps),
        selector_result=selection,
    )


def _cycloalkane_name(size):
    roots = {3: "prop", 4: "but", 5: "pent", 6: "hex", 7: "hept",
             8: "oct", 9: "non", 10: "dec"}
    if size not in roots:
        raise ValueError("unsupported monocyclic parent size")
    return "cyclo" + roots[size] + "ane"
