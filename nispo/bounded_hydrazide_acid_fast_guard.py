"""Graph-closed fast handoff for a terminal hydrazide on an acid chain."""


def bounded_hydrazide_acid_requires_full(mol, Chem):
    """Use the full writer only when the complete simple chain is proved.

    The acid is the unique principal site, the carbon chain is saturated and
    unbranched, and the only off-chain heavy atoms form the verified terminal
    hydrazide and its methyl substituents. This checks admission to the full
    writer, not the preferred status of the whole output.
    """
    if (mol is None or mol.GetStereoGroups() or len(Chem.GetMolFrags(mol)) != 1
            or mol.GetRingInfo().NumRings()
            or any(a.GetAtomicNum() not in {6, 7, 8} or a.GetFormalCharge()
                   or a.GetIsotope() or a.GetNumRadicalElectrons()
                   or int(a.GetChiralTag()) for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) for b in mol.GetBonds())):
        return False
    if not any(a.GetAtomicNum() == 7 and any(n.GetAtomicNum() == 7
               for n in a.GetNeighbors()) for a in mol.GetAtoms()):
        return False
    from .branch.hetero_atoms import _terminal_chain_hydrazinyl
    from .extended_inventory import selection_inventory
    from .hydrazide_components import recognize_hydrazides
    from .principal_carboxyl import acyclic_acid_record, principal_carboxyl_group

    inventory = selection_inventory(mol)
    acids = [g for g in inventory.groups if g.kind == 'acid']
    sites = recognize_hydrazides(mol)
    if (not inventory.supported or inventory.source_frontier.status != 'selected'
            or len(acids) != 1 or tuple(inventory.principal) != tuple(acids)
            or len(sites) != 1):
        return False
    acid, site = acids[0], sites[0]
    if not _terminal_chain_hydrazinyl(mol, site.acyl_nitrogen, site.center):
        return False
    if any(g.kind not in {'acid', 'hydrazide', 'amine'} for g in inventory.groups):
        return False
    if any(g.kind == 'amine' and g.center != site.terminal_nitrogen
           for g in inventory.groups):
        return False
    path, previous, current = [], None, acid.center
    while True:
        if current in path or len(path) > 40:
            return False
        path.append(current)
        next_carbons = [n.GetIdx() for n in mol.GetAtomWithIdx(current).GetNeighbors()
                        if n.GetAtomicNum() == 6 and n.GetIdx() != previous]
        if current == site.center:
            if next_carbons:
                return False
            break
        if len(next_carbons) != 1:
            return False
        nxt = next_carbons[0]
        if mol.GetBondBetweenAtoms(current, nxt).GetBondTypeAsDouble() != 1:
            return False
        previous, current = current, nxt
    if len(path) < 3:
        return False
    methyls = {n.GetIdx() for i in (site.acyl_nitrogen, site.terminal_nitrogen)
               for n in mol.GetAtomWithIdx(i).GetNeighbors()
               if n.GetAtomicNum() == 6 and n.GetIdx() != site.center}
    heavy = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
    if set(path) | set(acid.atoms) | set(site.atoms) | methyls != heavy:
        return False
    group = principal_carboxyl_group(mol)
    if not group or group[0] != acid.center:
        return False
    record = acyclic_acid_record(mol, Chem, group)
    if not record.name or dict(record.stereo_locants) != {
            idx: str(position) for position, idx in enumerate(path, 1)}:
        return False
    # Compare the actual ordinary writer with the source-owned chain writer;
    # a different selected construction remains on the prior fast path.
    from . import api

    return api._engine().name(Chem.Mol(mol)) == record.name
