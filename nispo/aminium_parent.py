"""Map a supported acyclic ammonium centre onto its senior carbon chain."""

from .component_mapping import (
    MappedComponent,
    decorate_component,
    fully_covered,
    hydrogen_bindings,
    mapped_branch,
)
from .multifunctional_parent import _paths
from .branch_rules import parent_chain_name
from .formatting import complete_parent_name


def aminium_parent(mol, Chem, inventory):
    # Prefer the independently mapped cation construction.  It compares full
    # parent facts, preserves annotations, and owns every heavy atom.
    from .amine_cation_components import amine_cation_component

    typed = amine_cation_component(mol, Chem)
    if typed is not None:
        heavy = frozenset(a.GetIdx() for a in mol.GetAtoms()
                          if a.GetAtomicNum() > 1)
        if (typed.charge == 1 and typed.atoms == heavy
                and sum(a.GetFormalCharge() for a in mol.GetAtoms()) == 1):
            return typed.name
        return ""
    # The historical route is retained only for unannotated cases outside the
    # typed producer's bounded construction.  It cannot silently erase a
    # mapped isotope or stereochemical descriptor.
    if (any(a.GetIsotope() or int(a.GetChiralTag()) for a in mol.GetAtoms())
            or any(int(b.GetStereo()) for b in mol.GetBonds())
            or mol.GetStereoGroups()):
        return ""
    groups = inventory.principal
    if len(groups) != 1 or groups[0].kind != "ammonium":
        return ""
    nidx = groups[0].center
    n = mol.GetAtomWithIdx(nidx)
    if n.IsInRing() or any(
        a.GetAtomicNum() != 6 for a in n.GetNeighbors() if a.GetAtomicNum() > 1
    ):
        return ""
    # Acylated or imine-bearing ammonium centres need their own functional
    # cation parent, rather than an alkyl-aminium construction.
    if any(
        any(
            mol.GetBondBetweenAtoms(c.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() > 1
            and a.GetAtomicNum() != 6
            for a in c.GetNeighbors()
        )
        for c in n.GetNeighbors()
        if c.GetAtomicNum() == 6
    ):
        return ""
    # Acid/amide carbon atoms become characteristic prefixes when the cation
    # is principal. They do not extend an aminium chain through a carbonyl.
    carbon = {
        a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6 and not a.IsInRing()
    } - {
        g.center
        for g in inventory.groups
        if g.kind in {"acid", "amide", "ester", "nitrile"}
    }
    roots = {a.GetIdx() for a in n.GetNeighbors() if a.GetAtomicNum() == 6}
    results = []
    for path in _paths(mol, carbon):
        for oriented in [path, tuple(reversed(path))]:
            if oriented[0] not in roots:
                continue
            mapping = {i: str(j) for j, i in enumerate(oriented, 1)}
            mapping[nidx] = "N"
            embedded = set()
            prefixes = []
            failed = False
            for idx in tuple(mapping):
                for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
                    other = neighbor.GetIdx()
                    if other in mapping or neighbor.GetAtomicNum() <= 1:
                        continue
                    text, atoms = mapped_branch(mol, Chem, other, idx)
                    if not text or atoms & (set(mapping) | embedded):
                        failed = True
                        break
                    if any(mol.GetAtomWithIdx(i).IsInRing() for i in atoms):
                        failed = True
                        break
                    prefixes.append((mapping[idx], text))
                    embedded.update(atoms)
                if failed:
                    break
            mapping = hydrogen_bindings(mol, mapping)
            if failed or not fully_covered(mol, mapping, embedded):
                continue
            parent = complete_parent_name(parent_chain_name(mol, oriented))
            if not parent.endswith("e"):
                continue
            parent = parent[:-1] + ("-1-aminium" if len(oriented) > 2 else "aminium")
            name = decorate_component(
                mol,
                Chem,
                MappedComponent(
                    parent,
                    tuple(mapping.items()),
                    frozenset(mapping),
                    tuple(prefixes),
                    frozenset(embedded),
                ),
            )
            if name:
                results.append(
                    (
                        (
                            -len(oriented),
                            tuple(
                                sorted(int(loc) for loc, _ in prefixes if loc.isdigit())
                            ),
                            name,
                        ),
                        name,
                    )
                )
    return min(results)[1] if results else ""
