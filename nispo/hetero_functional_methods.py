"""Source-closed azinic acids and heterohydride suffix constructions."""

from .naming_methods import MethodDecision
from .functional_replacement_acids import _organyl_units, _plain


_PARENTS = {5: ('borane', 3), 14: ('silane', 4), 15: ('phosphane', 3),
            32: ('germane', 4), 33: ('arsane', 3), 50: ('stannane', 4),
            51: ('stibane', 3), 82: ('plumbane', 4), 83: ('bismuthane', 3)}


def _out(mol, name, center, rule):
    atoms = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    return MethodDecision(name, 'substitutive', frozenset((center,)), atoms, rule, (),
                          component_roles=(('source-functional-parent', atoms, ()),))


def aci_nitro_acid_decision(mol, Chem):
    """P-67.1.6 carbon-ylidene azinic acids with no competing function."""
    if (mol.GetNumHeavyAtoms() > 64 or len(Chem.GetMolFrags(mol)) != 1 or mol.GetStereoGroups()
            or any(a.GetAtomicNum() not in {6, 7, 8} or a.GetIsotope() or a.GetNumRadicalElectrons()
                   or int(a.GetChiralTag()) or a.HasQuery() for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) or b.HasQuery() for b in mol.GetBonds())):
        return None
    ns = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    os = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 8]
    if len(ns) != 1 or len(os) != 2:
        return None
    n = ns[0]
    minus = [a for a in os if a.GetFormalCharge() == -1 and a.GetDegree() == 1]
    hydroxy = [a for a in os if not a.GetFormalCharge() and a.GetDegree() == 1
               and a.GetTotalNumHs(includeNeighbors=True) == 1]
    carbons = [a for a in n.GetNeighbors() if a.GetAtomicNum() == 6
               and mol.GetBondBetweenAtoms(n.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() == 2]
    if (n.GetFormalCharge() != 1 or n.GetDegree() != 3 or len(minus) != 1 or len(hydroxy) != 1
            or len(carbons) != 1 or any(a.GetFormalCharge() for a in mol.GetAtoms()
                                        if a.GetIdx() not in {n.GetIdx(), minus[0].GetIdx()})):
        return None
    if any(mol.GetBondBetweenAtoms(n.GetIdx(), a.GetIdx()) is None
           or mol.GetBondBetweenAtoms(n.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() != 1
           for a in (minus[0], hydroxy[0])):
        return None
    from .component_mapping import mapped_branch

    text, owned = mapped_branch(mol, Chem, carbons[0].GetIdx(), n.GetIdx())
    core = {n.GetIdx(), minus[0].GetIdx(), hydroxy[0].GetIdx()}
    if not text or set(owned) & core or set(owned) | core != {a.GetIdx() for a in mol.GetAtoms()}:
        return None
    return _out(mol, text + 'azinic acid', n.GetIdx(), 'P-67.1.6/P-41')


def aci_nitro_senior_acid_decision(mol, Chem):
    """Render the P-67.1.6 azanylidene prefix on a senior ring carboxylic acid."""
    if (mol.GetNumHeavyAtoms() > 32 or len(Chem.GetMolFrags(mol)) != 1 or mol.GetStereoGroups()
            or any(a.GetAtomicNum() not in {6, 7, 8} or a.GetIsotope() or a.GetNumRadicalElectrons()
                   or int(a.GetChiralTag()) or a.HasQuery() for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) or b.HasQuery() for b in mol.GetBonds())):
        return None
    nitrogens = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    oxygens = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 8]
    rings = mol.GetRingInfo().AtomRings()
    if len(nitrogens) != 1 or len(oxygens) != 4 or len(rings) != 1:
        return None
    ring = frozenset(rings[0])
    if (not 3 <= len(ring) <= 10 or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in ring)
            or any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
            or any(b.GetBondTypeAsDouble() != 1 for b in mol.GetBonds()
                   if b.GetBeginAtomIdx() in ring and b.GetEndAtomIdx() in ring)):
        return None
    nitrogen = nitrogens[0]
    nidx = nitrogen.GetIdx()
    if nitrogen.GetFormalCharge() != 1 or nitrogen.GetDegree() != 3:
        return None
    ring_c = [a for a in nitrogen.GetNeighbors() if a.GetIdx() in ring
              and mol.GetBondBetweenAtoms(nidx, a.GetIdx()).GetBondTypeAsDouble() == 2]
    ominus = [a for a in nitrogen.GetNeighbors() if a.GetAtomicNum() == 8 and a.GetFormalCharge() == -1
              and a.GetDegree() == 1 and mol.GetBondBetweenAtoms(nidx, a.GetIdx()).GetBondTypeAsDouble() == 1]
    hydroxyl = [a for a in nitrogen.GetNeighbors() if a.GetAtomicNum() == 8 and not a.GetFormalCharge()
                and a.GetDegree() == 1 and a.GetTotalNumHs(includeNeighbors=True) == 1
                and mol.GetBondBetweenAtoms(nidx, a.GetIdx()).GetBondTypeAsDouble() == 1]
    if (len(ring_c) != 1 or len(ominus) != 1 or len(hydroxyl) != 1
            or any(a.GetFormalCharge() for a in mol.GetAtoms() if a.GetIdx() not in {nidx, ominus[0].GetIdx()})):
        return None
    acid = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6 or atom.GetIdx() in ring or atom.GetDegree() != 3:
            continue
        oxo = [a for a in atom.GetNeighbors() if a.GetAtomicNum() == 8 and a.GetDegree() == 1
               and mol.GetBondBetweenAtoms(atom.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() == 2]
        oh = [a for a in atom.GetNeighbors() if a.GetAtomicNum() == 8 and a.GetDegree() == 1
              and a.GetTotalNumHs(includeNeighbors=True) == 1
              and mol.GetBondBetweenAtoms(atom.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() == 1]
        attached = [a for a in atom.GetNeighbors() if a.GetIdx() in ring
                    and mol.GetBondBetweenAtoms(atom.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() == 1]
        if len(oxo) == len(oh) == len(attached) == 1:
            acid.append((atom.GetIdx(), oxo[0].GetIdx(), oh[0].GetIdx()))
    if len(acid) != 1 or set(ring) | {nidx, ominus[0].GetIdx(), hydroxyl[0].GetIdx(), *acid[0]} != set(range(mol.GetNumAtoms())):
        return None
    marker = Chem.RWMol(mol)
    replacement = marker.GetAtomWithIdx(nidx)
    replacement.SetAtomicNum(8)
    replacement.SetFormalCharge(0)
    replacement.SetNumExplicitHs(0)
    replacement.SetNoImplicit(True)
    for idx in sorted((ominus[0].GetIdx(), hydroxyl[0].GetIdx()), reverse=True):
        marker.RemoveAtom(idx)
    marker = marker.GetMol()
    try:
        Chem.SanitizeMol(marker)
    except (RuntimeError, ValueError):
        return None
    from .api import mol_to_iupac
    name = mol_to_iupac(marker)
    if name.count('oxo') != 1 or not name.endswith('carboxylic acid'):
        return None
    return _out(mol, name.replace('oxo', '[hydroxy(oxo)-lambda5-azanylidene]'),
                acid[0][0], 'P-67.1.6/P-41')


def heterohydride_carbonitrile_decision(mol, Chem):
    """External C≡N as -carbonitrile on a mononuclear heterohydride."""
    if not _plain(mol, Chem):
        return None
    hosts = [a for a in mol.GetAtoms() if a.GetAtomicNum() in _PARENTS]
    if len(hosts) != 1:
        return None
    host = hosts[0]
    if host.IsInRing() or host.GetTotalValence() != _PARENTS[host.GetAtomicNum()][1]:
        return None
    sites = []
    from .carbonic_cyanide_methods import _cyano
    for atom in host.GetNeighbors():
        if atom.GetAtomicNum() == 6:
            owned = _cyano(mol, atom.GetIdx(), host.GetIdx())
            if owned:
                sites.append((atom.GetIdx(), owned))
    if len(sites) != 1:
        return None
    carbon, nitrile = sites[0]
    roots = [a.GetIdx() for a in host.GetNeighbors() if a.GetIdx() != carbon]
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in roots):
        return None
    seen = {host.GetIdx(), *nitrile}
    prefix = _organyl_units(mol, Chem, host.GetIdx(), roots, seen)
    if prefix is None or seen != {a.GetIdx() for a in mol.GetAtoms()}:
        return None
    return _out(mol, prefix + _PARENTS[host.GetAtomicNum()][0] + 'carbonitrile', carbon,
                'P-68.1.5.1/P-66.5.1.1.3/P-41')


def heteroamine_decision(mol, Chem):
    """Principal -NH2 on a supported heteroatom hydride or saturated N ring."""
    if not _plain(mol, Chem):
        return None
    amino = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7 and a.GetDegree() == 1
             and a.GetTotalNumHs(includeNeighbors=True) == 2 and not a.IsInRing()]
    if len(amino) != 1:
        return None
    n = amino[0]
    host = n.GetNeighbors()[0]
    if (mol.GetBondBetweenAtoms(n.GetIdx(), host.GetIdx()).GetBondTypeAsDouble() != 1
            or host.GetAtomicNum() in {6, 8}):
        return None
    if host.GetAtomicNum() in _PARENTS:
        if host.IsInRing() or host.GetTotalValence() != _PARENTS[host.GetAtomicNum()][1]:
            return None
        roots = [a.GetIdx() for a in host.GetNeighbors() if a.GetIdx() != n.GetIdx()]
        if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in roots):
            return None
        seen = {host.GetIdx(), n.GetIdx()}
        prefix = _organyl_units(mol, Chem, host.GetIdx(), roots, seen)
        if prefix is None or seen != {a.GetIdx() for a in mol.GetAtoms()}:
            return None
        from .formatting import attach_suffix
        return _out(mol, prefix + attach_suffix(_PARENTS[host.GetAtomicNum()][0], 'amine'),
                    host.GetIdx(), 'P-62.2.1.3/P-41')
    if host.GetAtomicNum() != 7 or not host.IsInRing():
        return None
    rings = mol.GetRingInfo().AtomRings()
    if len(rings) != 1 or any(b.GetBondTypeAsDouble() != 1 for b in mol.GetBonds()):
        return None
    ring = frozenset(rings[0])
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() not in {6, 7, 8, 16} for i in ring):
        return None
    outside = [a for a in mol.GetAtoms() if a.GetIdx() not in ring and a.GetIdx() != n.GetIdx()]
    if (len(outside) > 2 or any(a.GetAtomicNum() != 6 or a.GetDegree() != 1
                                or a.GetTotalNumHs(includeNeighbors=True) != 3
                                or a.GetNeighbors()[0].GetIdx() not in ring for a in outside)):
        return None
    marker = Chem.RWMol(mol)
    atom = marker.GetAtomWithIdx(n.GetIdx())
    atom.SetAtomicNum(8)
    atom.SetNumExplicitHs(1)
    atom.SetNoImplicit(True)
    marker = marker.GetMol()
    try:
        Chem.SanitizeMol(marker)
    except (ValueError, RuntimeError):
        return None
    import re
    from .api import mol_to_iupac
    name = mol_to_iupac(marker)
    # With no other principal groups, the hydroxy and amino analogues have
    # identical parent and suffix-site numbering. Keep only supported parent
    # vocabulary and a single, explicit heteroatom suffix locant.
    if not re.search(r'(?:piperidin|piperazin|morpholin|pyrrolidin|azetidin|azepan|aziridin|thiomorpholin)-\d+-ol$', name):
        return None
    name = name[:-2] + 'amine'
    return _out(mol, name, host.GetIdx(), 'P-62.2.1.3/P-41')


def acyclic_heterone_decision(mol, Chem):
    """P-64.4.1 -one on one Si/Ge/Sn/Pb or Group-15 center.

    Sulfoxides, sulfones and their Se/Te analogues follow their separate
    P-63.6 preferred construction and are intentionally outside this route.
    """
    if not _plain(mol, Chem):
        return None
    allowed = {14: 'silane', 32: 'germane', 50: 'stannane', 82: 'plumbane',
               15: 'phosphane', 33: 'arsane', 51: 'stibane', 83: 'bismuthane'}
    hosts = [a for a in mol.GetAtoms() if a.GetAtomicNum() in allowed]
    if len(hosts) != 1 or hosts[0].IsInRing():
        return None
    host = hosts[0]
    hidx = host.GetIdx()
    oxygen = [a for a in host.GetNeighbors() if a.GetAtomicNum() == 8 and a.GetDegree() == 1
              and mol.GetBondBetweenAtoms(hidx, a.GetIdx()).GetBondTypeAsDouble() == 2]
    roots = [a.GetIdx() for a in host.GetNeighbors() if a.GetAtomicNum() == 6
             and mol.GetBondBetweenAtoms(hidx, a.GetIdx()).GetBondTypeAsDouble() == 1]
    if len(oxygen) not in {1, 2} or len(oxygen) + len(roots) != host.GetDegree():
        return None
    group14 = host.GetAtomicNum() in {14, 32, 50, 82}
    if (group14 and (host.GetTotalValence() != 4 or len(oxygen) != 1)
            or not group14 and host.GetTotalValence() not in {3, 5}):
        return None
    seen = {hidx, *(a.GetIdx() for a in oxygen)}
    prefix = _organyl_units(mol, Chem, hidx, roots, seen)
    if prefix is None or seen != {a.GetIdx() for a in mol.GetAtoms()}:
        return None
    from .formatting import attach_suffix
    parent = allowed[host.GetAtomicNum()]
    if not group14 and host.GetTotalValence() == 5:
        parent = 'lambda5-' + parent
        if prefix:
            prefix += '-'
    suffix = 'one' if len(oxygen) == 1 else 'dione'
    return _out(mol, prefix + attach_suffix(parent, suffix), hidx, 'P-64.4.1/P-68/P-41')
