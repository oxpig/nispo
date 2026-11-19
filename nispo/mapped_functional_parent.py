"""Mapped decorated acyclic acid, amide and nitrile parent assembly."""

from itertools import combinations

from .molecule_iteration import iter_atoms, iter_bonds
from .component_mapping import (
    MappedComponent,
    chemical_inventory,
    decorate_component,
    fully_covered,
    mapped_branch,
    hydrogen_bindings,
)
from .formatting import complete_parent_name
from .constants import ROOTS


def mapped_functional_parent(mol, Chem):
    if not (
        any(a.GetIsotope() or int(a.GetChiralTag()) for a in iter_atoms(mol))
        or any(int(b.GetStereo()) for b in iter_bonds(mol))
    ):
        return ""
    inventory = chemical_inventory(mol)
    groups = inventory.principal
    if (
        not groups
        or groups[0].kind not in {"acid", "amide", "nitrile"}
        or len(groups) > 2
    ):
        return ""
    kind = groups[0].kind
    if any(g.kind != kind or mol.GetAtomWithIdx(g.center).IsInRing() for g in groups):
        return ""
    if kind == "amide" and len(groups) != 1:
        return ""
    required = {g.center for g in groups}
    allowed = set(required)
    stack = list(required)
    while stack:
        idx = stack.pop()
        for n in mol.GetAtomWithIdx(idx).GetNeighbors():
            if n.GetAtomicNum() == 6 and not n.IsInRing() and n.GetIdx() not in allowed:
                allowed.add(n.GetIdx())
                stack.append(n.GetIdx())
    leaves = [
        i
        for i in allowed
        if sum(n.GetIdx() in allowed for n in mol.GetAtomWithIdx(i).GetNeighbors()) <= 1
    ]
    paths = []
    if len(allowed) == 1:
        paths = [tuple(allowed)]
    for a, b in combinations(leaves, 2):
        path = tuple(Chem.GetShortestPath(mol, a, b))
        if not set(path) <= allowed or not required <= set(path):
            continue
        for oriented in [path, tuple(reversed(path))]:
            if oriented[0] not in required:
                continue
            if len(groups) == 2 and oriented[-1] not in required:
                continue
            paths.append(oriented)
    results = []
    for path in paths:
        if len(path) not in ROOTS:
            continue
        mapping = {idx: str(pos) for pos, idx in enumerate(path, 1)}
        owned = set(path)
        embedded = set()
        skipped = set()
        prefixes = []
        for g in groups:
            loc = mapping[g.center]
            if kind == "nitrile":
                n = next(
                    i for i in g.atoms if mol.GetAtomWithIdx(i).GetAtomicNum() == 7
                )
                mapping[n] = "N" + loc
                owned.add(n)
                skipped.add(n)
            else:
                oxy = [
                    n
                    for n in mol.GetAtomWithIdx(g.center).GetNeighbors()
                    if n.GetAtomicNum() == 8
                ]
                carbonyl = next(
                    n.GetIdx()
                    for n in oxy
                    if mol.GetBondBetweenAtoms(
                        g.center, n.GetIdx()
                    ).GetBondTypeAsDouble()
                    == 2
                )
                mapping[carbonyl] = "O" if kind == "amide" else "O'" + loc
                owned.add(carbonyl)
                skipped.add(carbonyl)
                other = (
                    g.ligand
                    if kind == "amide"
                    else next(n.GetIdx() for n in oxy if n.GetIdx() != carbonyl)
                )
                mapping[other] = "N" if kind == "amide" else "O" + loc
                owned.add(other)
                skipped.add(other)
                if kind == "amide":
                    for n in mol.GetAtomWithIdx(other).GetNeighbors():
                        if n.GetIdx() == g.center or n.GetAtomicNum() <= 1:
                            continue
                        prefix, atoms = mapped_branch(mol, Chem, n.GetIdx(), other)
                        if not prefix or any(
                            mol.GetAtomWithIdx(i).IsInRing() for i in atoms
                        ):
                            break
                        prefixes.append(("N", prefix))
                        embedded.update(atoms)
                    else:
                        continue
                    break
        else:
            failed = False
            for idx in path:
                for n in mol.GetAtomWithIdx(idx).GetNeighbors():
                    if (
                        n.GetIdx() in owned
                        or n.GetIdx() in skipped
                        or n.GetAtomicNum() <= 1
                    ):
                        continue
                    prefix, atoms = mapped_branch(mol, Chem, n.GetIdx(), idx)
                    if (
                        not prefix
                        or set(atoms) & owned
                        or any(mol.GetAtomWithIdx(i).IsInRing() for i in atoms)
                    ):
                        failed = True
                        break
                    prefixes.append((mapping[idx], prefix))
                    embedded.update(atoms)
                if failed:
                    break
            mapping = hydrogen_bindings(mol, mapping)
            owned = set(mapping)
            if failed or not fully_covered(mol, owned, embedded):
                continue
            if len(groups) == 1 and kind in {"amide", "acid"}:
                from .parent_rules import _generic_acyl_functional_parent_text

                parent = _generic_acyl_functional_parent_text(
                    mol, Chem, ROOTS[len(path)], path[1:], kind
                )
            else:
                from .branch_rules import parent_chain_name
                from .multifunctional_parent import _suffix_parent

                parent = complete_parent_name(parent_chain_name(mol, path))
                parent = _suffix_parent(
                    parent,
                    kind,
                    len(groups),
                    [1, len(path)] if len(groups) == 2 else [1],
                    False,
                    len(path),
                )
            if not parent:
                continue
            component = MappedComponent(
                parent,
                tuple(mapping.items()),
                frozenset(owned),
                tuple(prefixes),
                frozenset(embedded),
            )
            name = decorate_component(mol, Chem, component)
            if name:
                results.append(
                    (
                        (
                            -len(path),
                            tuple(
                                sorted(int(loc) for loc, _ in prefixes if loc.isdigit())
                            ),
                            name,
                        ),
                        name,
                    )
                )
    return min(results)[1] if results else ""
