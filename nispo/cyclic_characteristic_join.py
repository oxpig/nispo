"""P-41 class partition joined to complete current cyclic hydride numberings."""

from collections import Counter
from dataclasses import dataclass

from .component_mapping import MappedComponent, decorate_component
from .formatting import attach_suffix, multiplied_suffix
from .naming_methods import MethodDecision
from .naming_result import SelectionLimitation


@dataclass(frozen=True)
class _Unit:
    host: int
    center: int
    atoms: frozenset
    suffix: str
    prefix: str
    rank: int
    hydrogens: int = 1


def _units(mol):
    found = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if atom.GetAtomicNum() == 7 and atom.GetDegree() == 1 and atom.GetTotalNumHs() == 2:
            host = atom.GetNeighbors()[0]
            if host.GetAtomicNum() == 6 and host.IsInRing() and atom.GetBonds()[0].GetBondTypeAsDouble() == 1:
                found.append(_Unit(host.GetIdx(), idx, frozenset({idx}), 'amine', 'amino', 19))
        if atom.GetAtomicNum() == 8 and atom.GetDegree() == 1:
            host = atom.GetNeighbors()[0]
            if host.GetAtomicNum() == 6 and host.IsInRing():
                order = atom.GetBonds()[0].GetBondTypeAsDouble()
                if order == 1 and atom.GetTotalNumHs() == 1:
                    found.append(_Unit(host.GetIdx(), idx, frozenset({idx}), 'ol', 'hydroxy', 17))
                elif order == 2:
                    found.append(_Unit(host.GetIdx(), idx, frozenset({idx}), 'one', 'oxo', 16, 2))
        if atom.GetAtomicNum() != 6 or atom.IsInRing():
            continue
        # The unit is exact and terminal, attached to an original ring carbon.
        neighbors = tuple(atom.GetNeighbors())
        hosts = [n for n in neighbors if n.GetAtomicNum() == 6 and n.IsInRing()
                 and mol.GetBondBetweenAtoms(idx, n.GetIdx()).GetBondTypeAsDouble() == 1]
        if len(hosts) != 1:
            continue
        host = hosts[0].GetIdx()
        ns = [n for n in neighbors if n.GetAtomicNum() == 7 and n.GetDegree() == 1
              and mol.GetBondBetweenAtoms(idx, n.GetIdx()).GetBondTypeAsDouble() == 3]
        if len(neighbors) == 2 and len(ns) == 1:
            found.append(_Unit(host, idx, frozenset({idx, ns[0].GetIdx()}), 'carbonitrile', 'cyano', 14))
        oxo = [n for n in neighbors if n.GetAtomicNum() == 8 and n.GetDegree() == 1
               and mol.GetBondBetweenAtoms(idx, n.GetIdx()).GetBondTypeAsDouble() == 2]
        oh = [n for n in neighbors if n.GetAtomicNum() == 8 and n.GetDegree() == 1
              and n.GetTotalNumHs() == 1
              and mol.GetBondBetweenAtoms(idx, n.GetIdx()).GetBondTypeAsDouble() == 1]
        if len(neighbors) == 3 and len(oxo) == len(oh) == 1:
            found.append(_Unit(host, idx, frozenset({idx, oxo[0].GetIdx(), oh[0].GetIdx()}),
                               'carboxylic acid', 'carboxy', 7))
    return tuple(found)


def cyclic_characteristic_join(mol, Chem, native_numberings):
    if (mol.GetStereoGroups() or len(Chem.GetMolFrags(mol)) != 1
            or mol.GetRingInfo().NumRings() < 2
            or any(int(a.GetChiralTag()) for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) for b in mol.GetBonds())
            or any(a.GetIsotope() or a.GetFormalCharge() or a.GetNumRadicalElectrons()
                   for a in mol.GetAtoms())):
        return None
    units = _units(mol)
    if not units:
        return None
    group = frozenset().union(*(u.atoms for u in units))
    if len(group) != sum(len(u.atoms) for u in units):
        return None
    core = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1) - group
    if (any(u.host not in core for u in units)
            or any(not mol.GetAtomWithIdx(i).IsInRing()
                   or mol.GetAtomWithIdx(i).GetAtomicNum() not in {6, 7, 8} for i in core)
            or any(b.GetBondTypeAsDouble() != 1 for b in mol.GetBonds()
                   if b.GetBeginAtomIdx() in core and b.GetEndAtomIdx() in core)):
        return None
    delta = Counter()
    for unit in units:
        delta[unit.host] += unit.hydrogens
    order = tuple(sorted(core))
    local = {old: new for new, old in enumerate(order)}
    graph = Chem.RWMol()
    for old in order:
        graph.AddAtom(Chem.Atom(mol.GetAtomWithIdx(old).GetAtomicNum()))
    for bond in mol.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if left in core and right in core:
            graph.AddBond(local[left], local[right], Chem.BondType.SINGLE)
    parent = graph.GetMol()
    try:
        Chem.SanitizeMol(parent)
    except (ValueError, RuntimeError):
        return None
    if any(parent.GetAtomWithIdx(new).GetTotalNumHs(includeNeighbors=True) !=
           mol.GetAtomWithIdx(old).GetTotalNumHs(includeNeighbors=True) + delta[old]
           for new, old in enumerate(order)):
        return None
    smiles = Chem.MolToSmiles(parent, canonical=True, isomericSmiles=False)
    canonical = Chem.MolFromSmiles(smiles)
    if canonical is None:
        return None
    matches = canonical.GetSubstructMatches(parent, uniquify=False, useChirality=False, maxMatches=4097)
    if not matches or len(matches) >= 4097:
        return None
    senior = min(u.rank for u in units)
    principal = [u for u in units if u.rank == senior]
    subordinate = [u for u in units if u.rank != senior]
    suffix = multiplied_suffix(len(principal), principal[0].suffix)
    if not suffix:
        return None
    candidates = []
    for parent_name, locants in native_numberings(smiles):
        numbered = dict(locants)
        if set(numbered) != set(range(len(order))):
            continue
        for match in matches:
            bindings = {old: numbered[match[new]] for new, old in enumerate(order)}
            positions = tuple(sorted(int(bindings[u.host]) for u in principal))
            name = attach_suffix(parent_name, suffix, positions)
            prefixes = tuple((bindings[u.host], u.prefix) for u in subordinate)
            complete = dict(bindings)
            complete.update((i, '') for i in group)
            name = decorate_component(mol, Chem, MappedComponent(
                name, tuple(complete.items()), core | group, prefixes=prefixes, systematic_parent=True))
            if name:
                candidates.append((positions, name, bindings))
    if not candidates:
        return None
    _, name, bindings = min(candidates, key=lambda row: (row[0], row[1]))
    roles = [('cyclic-parent', core, tuple(bindings.items()))]
    roles.extend(('principal-suffix' if u.rank == senior else 'subordinate-prefix', u.atoms,
                  ((u.center, bindings[u.host]),)) for u in units)
    return MethodDecision(name=name, method='substitutive',
        principal_centers=frozenset(u.center for u in principal), atoms=core | group,
        rule='P-41/P-62/P-63/P-64/P-65/P-66/P-23/P-9',
        atom_locants=tuple(bindings.items()), component_roles=tuple(roles),
        preference_issues=(SelectionLimitation('cyclic-characteristic-join', 'P-44/P-45/P-51',
            'Every source characteristic unit is partitioned by class and fully constructed; global preferred construction/parent comparison remains unresolved.'),))
