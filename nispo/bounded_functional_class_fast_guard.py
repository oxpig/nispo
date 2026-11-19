"""Fast fallback only for source-proved mapped functional constructions."""


def _complete_partition(mol, decision, center):
    if decision is None or not decision.name or decision.principal_centers != {center}:
        return False
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    roles = [frozenset(atoms) for _role, atoms, _locants in decision.component_roles]
    return (decision.atoms == heavy and bool(roles)
            and sum(len(atoms) for atoms in roles) == len(heavy)
            and frozenset().union(*roles) == heavy)


def _capture_decision(mol, decision, complete):
    """Retain an already proved name on the fast dispatcher's private copy."""
    if complete:
        mol._nispo_fast_functional_decision = decision
    return bool(complete)


def bounded_functional_class_requires_full(mol, Chem):
    """Route mapped amide, urea and carbamate producers without inspecting text.

    Cheap C(=O)NN and C(=O)NO source motifs avoid the full inventory and
    component producers for unrelated inputs. A positive result requires a
    unique senior source class, exact producer center, and disjoint ownership
    of the complete heavy-atom graph. Proved urea and carbamate constructions
    are retained on the private graph for fast delivery; ordinary amides use
    the full writer, including its bond and stereo handling.
    """
    ureas, carbamates, amides = [], [], []
    has_ring = bool(mol.GetRingInfo().NumRings())
    hetero_ring = has_ring and any(a.IsInRing() and a.GetAtomicNum() not in {1, 6}
                                   for a in mol.GetAtoms())
    for carbon in mol.GetAtoms():
        if carbon.GetAtomicNum() != 6 or carbon.IsInRing():
            continue
        oxo = [n for n in carbon.GetNeighbors() if n.GetAtomicNum() == 8
               and mol.GetBondBetweenAtoms(carbon.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 2]
        nitrogens = [n for n in carbon.GetNeighbors() if n.GetAtomicNum() == 7
                     and mol.GetBondBetweenAtoms(carbon.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 1]
        single_oxygen = [n for n in carbon.GetNeighbors() if n.GetAtomicNum() == 8
                         and mol.GetBondBetweenAtoms(carbon.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 1]
        if len(oxo) != 1:
            continue
        if len(nitrogens) == 2 and len(carbon.GetNeighbors()) == 3:
            # Each new producer owns the carbon-rooted N arms separately.
            # The 1+1 extension requires a ring; the 2+1 producer can also
            # prove wholly acyclic arms. The actual decision remains the
            # authoritative graph-partition and seniority proof below.
            arms = [[n for n in nitrogen.GetNeighbors()
                     if n.GetIdx() != carbon.GetIdx() and n.GetAtomicNum() > 1]
                    for nitrogen in nitrogens]
            counts = sorted(len(side) for side in arms)
            if ((counts == [1, 1] and has_ring or counts == [1, 2])
                    and all(n.GetAtomicNum() == 6 and mol.GetBondBetweenAtoms(
                        nitrogen.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 1
                            for nitrogen, side in zip(nitrogens, arms) for n in side)):
                ureas.append(carbon.GetIdx())
        if len(nitrogens) == 1 and len(single_oxygen) == 1 and len(carbon.GetNeighbors()) == 3:
            carbamates.append(carbon.GetIdx())
        if (hetero_ring and len(nitrogens) == 1 and not single_oxygen
                and len(carbon.GetNeighbors()) == 3
                and any(n.GetAtomicNum() == 6 and mol.GetBondBetweenAtoms(
                    carbon.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 1
                        for n in carbon.GetNeighbors())):
            amides.append(carbon.GetIdx())
    if not ureas and not carbamates and not amides:
        return False
    from .extended_inventory import selection_inventory

    inventory = selection_inventory(mol)
    if not inventory.supported or len(inventory.principal) != 1:
        return False
    principal = inventory.principal[0]
    if principal.kind == 'urea' and ureas == [principal.center]:
        frontier = inventory.source_frontier
        if (frontier.status != 'selected' or len(frontier.groups) != 1
                or frontier.groups[0] != principal):
            return False
        from .urea_methods import urea_decision

        decision = urea_decision(mol, Chem)
        return _capture_decision(mol, decision, _complete_partition(mol, decision, principal.center))
    if principal.kind == 'carbamate' and carbamates == [principal.center]:
        from .carbamate_methods import carbamate_decision

        decision = carbamate_decision(mol, Chem)
        return _capture_decision(mol, decision, _complete_partition(mol, decision, principal.center))
    if principal.kind == 'amide' and amides == [principal.center]:
        from .acyclic_amide_general import acyclic_amide_general

        # This producer independently validates the original graph partition
        # and each rooted prefix before returning a decision; unlike the two
        # class producers it does not expose separate component-role tuples.
        decision = acyclic_amide_general(mol, Chem)
        heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
        complete = bool(decision and decision.name and decision.atoms == heavy
                        and decision.principal_centers == {principal.center})
        return complete
    return False
