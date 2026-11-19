"""P-66.4.2.3.5 amide-derived prefixes, preserving the N attachment."""


def hydrazonamido_prefix(mol, Chem, root, blocked, scope):
    nitrogen = mol.GetAtomWithIdx(root)
    if (nitrogen.GetAtomicNum() != 7 or nitrogen.IsInRing()
            or mol.GetBondBetweenAtoms(root, blocked).GetBondTypeAsDouble() != 1
            or nitrogen.GetTotalValence() != 3
            or any(mol.GetAtomWithIdx(i).GetIsotope()
                   or mol.GetAtomWithIdx(i).GetFormalCharge()
                   or mol.GetAtomWithIdx(i).GetNumRadicalElectrons()
                   or int(mol.GetAtomWithIdx(i).GetChiralTag()) for i in scope)
            or any(int(b.GetStereo()) for b in mol.GetBonds()
                   if b.GetBeginAtomIdx() in scope and b.GetEndAtomIdx() in scope)):
        return None
    sites = []
    for center in nitrogen.GetNeighbors():
        if center.GetIdx() == blocked or center.GetAtomicNum() not in {6, 16, 34, 52}:
            continue
        if mol.GetBondBetweenAtoms(root, center.GetIdx()).GetBondTypeAsDouble() != 1:
            continue
        for imine in center.GetNeighbors():
            if (imine.GetAtomicNum() != 7 or imine.GetDegree() != 2
                    or mol.GetBondBetweenAtoms(center.GetIdx(), imine.GetIdx()).GetBondTypeAsDouble() != 2):
                continue
            ends = [n for n in imine.GetNeighbors() if n.GetIdx() != center.GetIdx()]
            if (len(ends) == 1 and ends[0].GetAtomicNum() == 7
                    and ends[0].GetDegree() == 1 and ends[0].GetTotalNumHs() == 2
                    and mol.GetBondBetweenAtoms(imine.GetIdx(), ends[0].GetIdx()).GetBondTypeAsDouble() == 1):
                sites.append((center, imine, ends[0]))
    if len(sites) != 1:
        return None
    center, imine, end = sites[0]
    if center.GetAtomicNum() != 6 and sum(n.GetAtomicNum() == 8 and n.GetDegree() == 1
            and mol.GetBondBetweenAtoms(center.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 2
            for n in center.GetNeighbors()) not in {0, 1}:
        return None
    remove = (set(range(mol.GetNumAtoms())) - set(scope)) | {end.GetIdx()}
    rw = Chem.RWMol(mol)
    oxygen = rw.GetAtomWithIdx(imine.GetIdx())
    oxygen.SetAtomicNum(8)
    oxygen.SetNumExplicitHs(0)
    oxygen.SetNoImplicit(False)
    # Cleaving the single attachment restores the amide N-H on the cap.
    cap = rw.GetAtomWithIdx(root)
    if center.GetAtomicNum() != 6:
        if nitrogen.GetDegree() != 2:
            return None
        # The corresponding sulfinic/sulfonic acid establishes the same
        # source parent when the ordinary amide writer cannot yet render it.
        cap.SetAtomicNum(8)
    cap.SetNumExplicitHs(0)
    cap.SetNoImplicit(False)
    for index in sorted(remove, reverse=True):
        rw.RemoveAtom(index)
    probe = rw.GetMol()
    try:
        Chem.SanitizeMol(probe)
    except (ValueError, RuntimeError):
        return None
    if center.GetAtomicNum() == 6:
        # A suffix conversion belongs to this mapped amide, rather than
        # every amide in a multiply functional capped branch.
        from .extended_inventory import selection_inventory
        inventory = selection_inventory(probe)
        mapped_center = center.GetIdx() - sum(index < center.GetIdx() for index in remove)
        amides = [group for group in inventory.groups if group.kind == 'amide']
        if (not inventory.supported or len(amides) != 1
                or amides[0].center != mapped_center
                or tuple(inventory.principal) != (amides[0],)):
            return None
    from .api import mol_to_iupac
    ordinary = mol_to_iupac(probe)
    if center.GetAtomicNum() == 6:
        if ordinary.endswith('formamide'):
            complete = ordinary[:-len('formamide')] + 'methanehydrazonamide'
        elif ordinary.endswith('acetamide'):
            complete = ordinary[:-len('acetamide')] + 'ethanehydrazonamide'
        elif ordinary.endswith('benzamide'):
            complete = ordinary[:-len('benzamide')] + 'benzenecarbohydrazonamide'
        elif ordinary.endswith('carboxamide'):
            complete = ordinary[:-len('carboxamide')] + 'carbohydrazonamide'
        elif ordinary.endswith('amide'):
            complete = ordinary[:-len('amide')] + 'ehydrazonamide'
        else:
            return None
    else:
        endings = {16: ('sulfinic acid', 'sulfonic acid'),
                   34: ('seleninic acid', 'selenonic acid'),
                   52: ('tellurinic acid', 'telluronic acid')}[center.GetAtomicNum()]
        ending = next((word for word in endings if ordinary.endswith(word)), None)
        if ending is None:
            return None
        replacement = ending[:-len('ic acid')] + 'ohydrazonamide'
        complete = ordinary[:-len(ending)] + replacement
    return complete[:-1] + 'o', frozenset(scope)
