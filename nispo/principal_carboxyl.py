"""Select supported neutral carboxylic and carbothioic acids before shorter parents."""

from .molecule_iteration import iter_atoms
from .molecule_cache import mol_cache_get, mol_cache_set


def principal_carboxyl_group(mol):
    cached = mol_cache_get(mol, "principal_carboxyl_group")
    if cached is not None:
        return cached
    groups = []
    atoms = tuple(iter_atoms(mol))
    if all(
        a.GetAtomicNum() in {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}
        and not (a.GetFormalCharge() or a.GetIsotope() or a.GetNumRadicalElectrons())
        for a in atoms
    ):
        for carbon in atoms:
            if carbon.GetAtomicNum() != 6 or carbon.IsInRing():
                continue
            doubles = [
                a
                for a in carbon.GetNeighbors()
                if a.GetAtomicNum() in {8, 16}
                and mol.GetBondBetweenAtoms(
                    carbon.GetIdx(), a.GetIdx()
                ).GetBondTypeAsDouble()
                == 2
                and a.GetDegree() == 1
            ]
            hydroxyl = [
                a
                for a in carbon.GetNeighbors()
                if a.GetAtomicNum() == 8
                and mol.GetBondBetweenAtoms(
                    carbon.GetIdx(), a.GetIdx()
                ).GetBondTypeAsDouble()
                == 1
                and a.GetDegree() == 1
                and a.GetTotalNumHs(includeNeighbors=True) == 1
            ]
            owners = [
                a
                for a in carbon.GetNeighbors()
                if a.GetAtomicNum() == 6
                and mol.GetBondBetweenAtoms(
                    carbon.GetIdx(), a.GetIdx()
                ).GetBondTypeAsDouble()
                == 1
            ]
            if len(doubles) == len(hydroxyl) == len(owners) == 1:
                groups.append(
                    (
                        carbon.GetIdx(),
                        owners[0].GetIdx(),
                        "carbothioic acid"
                        if doubles[0].GetAtomicNum() == 16
                        else "carboxylic acid",
                    )
                )
    result = groups[0] if len(groups) == 1 else ()
    return mol_cache_set(mol, "principal_carboxyl_group", result)


def principal_ring_carboxyl_group(mol):
    """An external acid whose ring owner can take the P-65.1.2 suffix."""
    if mol.HasProp("_nispo_principal_prefix_mode"):
        return ()
    cached = mol_cache_get(mol, "principal_ring_carboxyl_group")
    if cached is not None:
        return cached
    selected = ()
    group = principal_carboxyl_group(mol)
    if group and group[2] == "carboxylic acid" and mol.GetAtomWithIdx(group[1]).IsInRing():
        from .extended_inventory import selection_inventory
        inventory = selection_inventory(mol)
        frontier = inventory.source_frontier
        if (frontier.status == "selected" and len(frontier.groups) == 1
                and frontier.groups[0].kind == "acid"
                and frontier.groups[0].center == group[0]):
            selected = group[:2]
    return mol_cache_set(mol, "principal_ring_carboxyl_group", selected)


def ring_carboxyl_suffix(mol, locants, prefixes, parent):
    """Consume the mapped carboxy prefix once, at the selected acid site."""
    if not any(text == "carboxy" for _, text in prefixes):
        return None
    group = principal_ring_carboxyl_group(mol)
    if not group:
        return None
    center, owner = group
    locant = locants.get(owner)
    prefix = (locant, "carboxy")
    if locant is None or prefix not in prefixes:
        return None
    from .formatting import attach_suffix
    return (attach_suffix(parent, "carboxylic acid", (locant,)),
            [row for row in prefixes if row != prefix], center, locant)


def acyclic_acid_record(mol, Chem, group):
    from .parent_rules import (
        _ester_acyl_parent_paths,
        _generic_acyl_functional_parent_path_name,
    )
    from .acyclic.rules import AcyclicNameRecord

    carbonyl, owner, kind = group
    if kind != "carboxylic acid" or mol.GetAtomWithIdx(owner).IsInRing():
        return AcyclicNameRecord("")
    # Decide the earlier source-ordered structural criteria before asking the
    # branch writer to render anything. An unrenderable preferred path must
    # not promote a less senior path solely through writer availability.
    ranked_paths = []
    for path in _ester_acyl_parent_paths(mol, Chem, carbonyl, owner):
        orders = tuple(mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble()
                       for a, b in zip(path, path[1:]))
        multiple_locs = tuple(i for i, order in enumerate(orders, 2)
                              if order in (2, 3))
        double_locs = tuple(i for i, order in enumerate(orders, 2)
                            if order == 2)
        # P-44.3.2 maximizes chain length before P-44.4.1.1/.2
        # maximize multiple/double bonds. With acid position 1 fixed,
        # P-44.4.1.10 then minimizes unsaturation locants.
        key = (-len(path), -len(multiple_locs), -len(double_locs),
               multiple_locs, double_locs)
        ranked_paths.append((key, path))
    if not ranked_paths:
        return AcyclicNameRecord("")
    best = min(key for key, _ in ranked_paths)
    from .constants import ROOTS
    selected_paths = [(carbonyl, *path) for key, path in ranked_paths if key == best]
    if all(len(path) not in ROOTS for path in selected_paths):
        from .acyclic.rules import _unrenderable_carbon_parent
        return _unrenderable_carbon_parent(selected_paths)
    candidates = []
    for key, path in ranked_paths:
        if key != best:
            continue
        name = _generic_acyl_functional_parent_path_name(
            mol, Chem, carbonyl, path, "acid"
        )
        if name:
            candidates.append(
                (
                    name,
                    tuple(
                        [(carbonyl, "1")]
                        + [(idx, str(i)) for i, idx in enumerate(path, 2)]
                    ),
                )
            )
    if candidates:
        name, locants = min(candidates)
        return AcyclicNameRecord(name, locants)
    from .bounded_retained_carboxylic_parents import (
        alkoxycarbonylamino_acetic_acid_parent, amino_acetic_acid_parent,
        aromatic_acetic_acid_parent,
    )

    selected = (alkoxycarbonylamino_acetic_acid_parent(mol, Chem)
                or amino_acetic_acid_parent(mol, Chem)
                or aromatic_acetic_acid_parent(mol, Chem))
    if selected is not None:
        return AcyclicNameRecord(selected.name, selected.atom_locants)
    return AcyclicNameRecord("")
