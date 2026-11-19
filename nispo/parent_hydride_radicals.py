"""Mapped P-71 H losses on complete current skeletal hydride constructions."""

from dataclasses import dataclass
import re

from .component_mapping import MappedComponent, decorate_component, hydrogen_bindings, mapped_branch
from .constants import MULT
from .formatting import attach_suffix
from .naming_methods import MethodDecision
from .naming_result import SelectionLimitation


_STANDARD = {5: 3, 6: 4, 7: 3, 8: 2, 9: 1, 14: 4, 15: 3, 16: 2,
             17: 1, 32: 4, 33: 3, 34: 2, 35: 1, 50: 4, 51: 3, 52: 2,
             53: 1, 82: 4, 83: 3, 84: 2, 85: 1}


@dataclass(frozen=True)
class HydrideRadicalEvidence:
    parent_word: str
    parent_smiles: str
    core: frozenset
    atom_locants: tuple
    radical_orders: tuple
    source_prefixes: tuple


_HYDRO = re.compile(r"^((?:[0-9][0-9a-z,'()\-]*-)?(?:" + '|'.join(
    sorted((word for n, word in MULT.items() if n % 2 == 0), key=len, reverse=True))
    + r")hydro)(?=[A-Za-z])(.+)$")


def _precursor(mol, Chem, core, centers):
    order = tuple(sorted(core))
    local = {old: new for new, old in enumerate(order)}
    graph = Chem.RWMol()
    expected = {}
    for old in order:
        original = mol.GetAtomWithIdx(old)
        outside = sum(b.GetBondTypeAsDouble() for b in original.GetBonds()
                      if b.GetOtherAtomIdx(old) not in core and b.GetOtherAtom(original).GetAtomicNum() > 1)
        if outside != int(outside):
            return None
        expected[old] = original.GetTotalNumHs(includeNeighbors=True) + centers.get(old, 0) + int(outside)
        atom = Chem.Atom(original.GetAtomicNum())
        atom.SetIsAromatic(original.GetIsAromatic())
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(expected[old])
        graph.AddAtom(atom)
    for bond in mol.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if left in core and right in core:
            graph.AddBond(local[left], local[right], bond.GetBondType())
    parent = graph.GetMol()
    try:
        Chem.SanitizeMol(parent)
    except (ValueError, RuntimeError):
        return None
    if (len(Chem.GetMolFrags(parent)) != 1 or any(a.GetNumRadicalElectrons() for a in parent.GetAtoms())
            or any(parent.GetAtomWithIdx(local[i]).GetTotalNumHs(includeNeighbors=True) != expected[i] for i in core)):
        return None
    smiles = Chem.MolToSmiles(parent)
    canonical = Chem.MolFromSmiles(smiles)
    matches = canonical.GetSubstructMatches(parent, uniquify=False, maxMatches=4097)
    if len(matches) >= 4097:
        return None
    matches = tuple(match for match in matches if all(
        canonical.GetAtomWithIdx(match[new]).GetTotalNumHs(includeNeighbors=True)
        == parent.GetAtomWithIdx(new).GetTotalNumHs(includeNeighbors=True)
        for new in range(len(order))))
    return (smiles, order, matches) if matches and len(matches) < 4097 else None


def parent_hydride_radicals(mol, Chem, native_numberings):
    if (mol.GetStereoGroups() or len(Chem.GetMolFrags(mol)) != 1
            or any(a.GetAtomicNum() not in ({1} | _STANDARD.keys()) or a.GetFormalCharge()
                   or int(a.GetChiralTag()) for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) for b in mol.GetBonds())):
        return None
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    needed = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1 or a.GetIsotope())
    centers = {i: mol.GetAtomWithIdx(i).GetNumRadicalElectrons() for i in heavy
               if mol.GetAtomWithIdx(i).GetNumRadicalElectrons()}
    if (not centers or any(n not in {1, 2, 3} for n in centers.values())
            or any(mol.GetAtomWithIdx(i).GetTotalValence() + n
                   != _STANDARD[mol.GetAtomWithIdx(i).GetAtomicNum()] for i, n in centers.items())):
        return None
    full = _precursor(mol, Chem, heavy, centers)
    if full is None:
        return None
    smiles, order, matches = full
    branch_source = Chem.Mol(mol)
    for i, count in centers.items():
        atom = branch_source.GetAtomWithIdx(i)
        atom.SetNumRadicalElectrons(0)
        atom.SetNumExplicitHs(atom.GetNumExplicitHs() + count)
        atom.SetNoImplicit(True)
    try:
        Chem.SanitizeMol(branch_source)
    except (ValueError, RuntimeError):
        return None
    scopes = {heavy}
    for _, numbered in native_numberings(smiles):
        numbered = dict(numbered)
        for match in matches:
            scope = frozenset(old for new, old in enumerate(order) if match[new] in numbered)
            if centers.keys() <= scope:
                scopes.add(scope)
    candidates = []
    for core in sorted(scopes, key=lambda scope: (-len(scope), tuple(sorted(scope)))):
        bare = full if core == heavy else _precursor(mol, Chem, core, centers)
        if bare is None:
            continue
        parent_smiles, order, matches = bare
        branches, embedded = [], set()
        failed = False
        for host in sorted(core):
            for neighbor in mol.GetAtomWithIdx(host).GetNeighbors():
                root = neighbor.GetIdx()
                if root in core or neighbor.GetAtomicNum() == 1:
                    continue
                text, owned = mapped_branch(branch_source, Chem, root, host)
                atom = branch_source.GetAtomWithIdx(root)
                if (not text and atom.GetAtomicNum() == 6 and not atom.GetFormalCharge()
                        and not atom.GetNumRadicalElectrons() and atom.GetTotalValence() == 4
                        and atom.GetTotalNumHs(includeNeighbors=True) == 3
                        and [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() > 1] == [host]
                        and branch_source.GetBondBetweenAtoms(root, host).GetBondTypeAsDouble() == 1):
                    child_map = hydrogen_bindings(branch_source, {root: ''})
                    text = decorate_component(branch_source, Chem, MappedComponent(
                        'methyl', tuple(child_map.items()), frozenset(child_map), systematic_parent=True))
                    owned = frozenset(child_map) if text else frozenset()
                if not text or owned & (core | embedded):
                    failed = True
                    break
                embedded.update(owned)
                branches.append((host, text))
            if failed:
                break
        if failed:
            continue
        for word, numbered in native_numberings(parent_smiles):
            numbered = dict(numbered)
            if set(numbered) != set(range(len(order))):
                continue
            for match in matches:
                bindings = hydrogen_bindings(mol, {old: numbered[match[new]] for new, old in enumerate(order)})
                if bindings.keys() | embedded != needed:
                    continue
                hydro = _HYDRO.fullmatch(word)
                hydro_prefix = hydro.group(1) if hydro else ''
                name, key = hydro.group(2) if hydro else word, []
                for count, ending in ((1, 'yl'), (2, 'ylidene'), (3, 'ylidyne')):
                    locants = tuple(sorted((bindings[i] for i, n in centers.items() if n == count),
                                           key=lambda s: (int(''.join(c for c in s if c.isdigit()) or 0), s)))
                    if not locants:
                        continue
                    multiplier = '' if len(locants) == 1 else MULT.get(len(locants))
                    if multiplier is None:
                        name = ''
                        break
                    name = attach_suffix(name, multiplier + ending, locants)
                    key.append(locants)
                if not name:
                    continue
                prefixes = tuple((bindings[host], text) for host, text in branches)
                name = decorate_component(mol, Chem, MappedComponent(
                    name, tuple(bindings.items()), frozenset(bindings), prefixes,
                    frozenset(embedded), hydro_prefix=hydro_prefix, systematic_parent=True))
                if name:
                    evidence = HydrideRadicalEvidence(word, parent_smiles, core, tuple(bindings.items()),
                        tuple(sorted(centers.items())), prefixes)
                    candidates.append((tuple(key), name, bindings, evidence))
    if not candidates:
        return None
    _, name, bindings, evidence = min(candidates, key=lambda row: (row[0], row[1]))
    return MethodDecision(name=name, method='substitutive', principal_centers=frozenset(centers),
        atoms=heavy, rule='P-71.2.1/P-71.2.2/P-71.2.3/P-71.6/P-29/P-82',
        atom_locants=tuple(bindings.items()), heterounits=(evidence,),
        component_roles=(('skeletal-hydride-radical', heavy, tuple(bindings.items())),),
        preference_issues=(SelectionLimitation('skeletal-hydride-radical', 'P-71.7/P-46',
            'The original hydride, radical orders and complete source graph are constructed; preferred radical parent comparison remains unresolved.'),))
