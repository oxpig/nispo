"""Scoped P-44 seniority for exact neutral nitrogenous fused lactam parents."""

import re


def localize_methylidene_stereo(mol, root_idx, parent_idx, prefix):
    """Carry a single-carbon branch's alkene configuration with its prefix."""
    if not prefix.endswith("methyl") or re.search(r"\([^)]*[EZ][^)]*\)", prefix):
        return prefix
    root = mol.GetAtomWithIdx(root_idx)
    if root.GetAtomicNum() != 6 or root.IsInRing() or root.GetTotalNumHs() != 1:
        return prefix
    if mol.GetBondBetweenAtoms(root_idx, parent_idx).GetBondTypeAsDouble() != 1:
        return prefix
    others = [
        a
        for a in root.GetNeighbors()
        if a.GetIdx() != parent_idx and a.GetAtomicNum() > 1
    ]
    if len(others) != 1 or not others[0].IsInRing() or others[0].GetAtomicNum() != 6:
        return prefix
    bond = mol.GetBondBetweenAtoms(root_idx, others[0].GetIdx())
    if bond.GetBondTypeAsDouble() != 2:
        return prefix
    descriptor = {
        "STEREOE": "E",
        "STEREOTRANS": "E",
        "STEREOZ": "Z",
        "STEREOCIS": "Z",
    }.get(str(bond.GetStereo()))
    return f"({descriptor})-{prefix}" if descriptor else prefix


def fused_lactam_seniority(mol, template, match, name, *, topology_query=False):
    parent_name = str(template["name"])
    if not (parent_name.endswith("one") or parent_name in {"oxindol", "oxindole"}):
        return ()
    # Topology-only shortcuts have not established hydrogen/tautomer state.
    # The rendered name must still use this lactam as its suffix parent.
    preferred_oxindole = parent_name in {"oxindol", "oxindole"} and name.endswith("1,3-dihydro-2H-indol-2-one")
    if topology_query or not (name.endswith(parent_name) or preferred_oxindole):
        return ()
    # Template matching may deliberately relax hydrogenation for other routes.
    # Chemical seniority may only promote an exact core with the hydrogen
    # budget left after its external substitutions have been accounted for.
    query = template["mol"]
    for j, i in enumerate(match):
        actual = mol.GetAtomWithIdx(i)
        expected = query.GetAtomWithIdx(j)
        external_order = sum(b.GetBondTypeAsDouble() for b in actual.GetBonds()
                             if b.GetOtherAtomIdx(i) not in match)
        if (actual.GetIsAromatic() != expected.GetIsAromatic()
                or actual.GetTotalNumHs(includeNeighbors=True) + external_order
                != expected.GetTotalNumHs(includeNeighbors=True)):
            return ()
    from .principal_carbonyl import _is_isocyano_atom
    if any(
        (a.GetFormalCharge() and not _is_isocyano_atom(mol, a))
        or a.GetIsotope() or a.GetNumRadicalElectrons()
        for a in mol.GetAtoms()
    ):
        return ()
    core = set(match)
    rings = [set(r) for r in mol.GetRingInfo().AtomRings() if set(r) <= core]
    if len(rings) < 2:
        return ()
    ring_atoms = set().union(*rings)
    # Restrict the first implementation to one fused N/C ring system, with
    # no spiro connections. Earlier P-44 element criteria then tie.
    reached = set(rings[0])
    while True:
        expanded = reached | set().union(*(r for r in rings if r & reached))
        if expanded == reached:
            break
        reached = expanded
    if reached != ring_atoms or any(
        len(a & b) == 1 for j, a in enumerate(rings) for b in rings[j + 1 :]
    ):
        return ()
    atoms = [mol.GetAtomWithIdx(i) for i in ring_atoms]
    if any(a.GetAtomicNum() not in {6, 7} for a in atoms):
        return ()
    nitrogens = sum(a.GetAtomicNum() == 7 for a in atoms)
    if not nitrogens:
        return ()
    lactams = 0
    for atom in atoms:
        if atom.GetAtomicNum() != 6:
            continue
        carbonyl = any(
            n.GetAtomicNum() == 8
            and mol.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble()
            == 2
            for n in atom.GetNeighbors()
        )
        if not carbonyl:
            continue
        if not any(
            n.GetIdx() in ring_atoms
            and n.GetAtomicNum() == 7
            and mol.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble()
            == 1
            for n in atom.GetNeighbors()
        ):
            return ()
        lactams += 1
    if not lactams:
        return ()
    # Compare only equal functional-group counts. Ring atoms alone determine
    # size and heteroatom seniority; exocyclic O and side-chain N are excluded.
    return lactams, len(rings), len(ring_atoms), nitrogens
