"""Mapped carboxamide suffixes attached to saturated ring nitrogen.

P-66.1.1.1.1.3 includes heterocyclic parent atoms as suffix hosts. The
ring-linked carbonic diamide is not an ordinary carbon-host amide or a hidden
N-acyl pseudoketone. Recognize the complete bounded graph before constructing
the ring-parent suffix and its P-66.1.1.3.1.1 exocyclic N substituents.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from .component_mapping import MappedComponent, decorate_component, mapped_branch
from .heterohost_sulfur import (
    _host_is_mapped_ring_nitrogen, _n_substituent_prefixes, _terminal_amide_n,
)
from .mapped_ring_parents import parent_facts, ring_systems, skeletons
from .molecule_iteration import iter_atoms, iter_bonds
from .naming_methods import MethodDecision
from .parent_selection import locant_key, parent_frontier


RULE = "P-41/P-44/P-45/P-66.1.1.1.1.3/P-66.1.1.3.1.1"


@dataclass(frozen=True, slots=True)
class _AnnularCarboxamide:
    center: int
    host: int
    oxygen: int
    ligand: int
    n_substituents: tuple[int, ...]


def _site(mol, ring):
    """Prove one exocyclic C(=O)(ring-N)(acyclic-N) and its entire scope."""
    doubles = [b for b in iter_bonds(mol) if b.GetBondTypeAsDouble() != 1]
    if len(doubles) != 1 or doubles[0].GetBondTypeAsDouble() != 2:
        return None
    bond = doubles[0]
    ends = (bond.GetBeginAtom(), bond.GetEndAtom())
    carbon = next((a for a in ends if a.GetAtomicNum() == 6), None)
    oxygen = next((a for a in ends if a.GetAtomicNum() == 8), None)
    if (carbon is None or oxygen is None or carbon.IsInRing()
            or oxygen.GetDegree() != 1 or carbon.GetDegree() != 3
            or carbon.GetTotalValence() != 4):
        return None
    center = carbon.GetIdx()
    hosts = [a.GetIdx() for a in carbon.GetNeighbors()
             if a.GetIdx() in ring and a.GetAtomicNum() == 7]
    if len(hosts) != 1 or not _host_is_mapped_ring_nitrogen(mol, hosts[0], ring, center):
        return None
    ligand, roots = _terminal_amide_n(mol, center, ring)
    if ligand is None:
        return None
    nitrogen = mol.GetAtomWithIdx(ligand)
    if (nitrogen.IsInRing() or nitrogen.GetTotalValence() != 3
            or nitrogen.GetTotalNumHs() != 2 - len(roots)):
        return None
    if {a.GetIdx() for a in carbon.GetNeighbors()} != {hosts[0], ligand, oxygen.GetIdx()}:
        return None
    # The complete non-ring remainder consists only of the one suffix and
    # saturated carbon trees. This proves absence of senior groups, other
    # amides, replacement chains and competing cyclic N arms without relying
    # on the ordinary inventory, which intentionally declines this class.
    suffix = {center, ligand, oxygen.GetIdx()}
    if any(a.GetAtomicNum() != 6 for a in iter_atoms(mol)
           if a.GetIdx() not in ring | suffix):
        return None
    return _AnnularCarboxamide(center, hosts[0], oxygen.GetIdx(), ligand, roots)


def annular_carboxamide_decision(mol, Chem):
    """Construct a complete neutral unannotated monocyclic amide, or decline.

    Ring vocabulary/numbering comes from the mapped ring frontier. Every
    outside atom belongs to the suffix or exactly one carbon branch. Charged,
    annotated, explicit-H, unsaturated and polyfunctional inputs keep their
    existing routes; no preferred status is inferred for those inputs here.
    """
    if mol.GetRingInfo().NumRings() >= 2:
        return _spiro_carboxamide_decision(mol, Chem)
    if (mol.GetNumHeavyAtoms() > 64 or mol.GetStereoGroups()
            or mol.GetRingInfo().NumRings() != 1
            or any(a.GetAtomicNum() not in {6, 7, 8, 16} or a.GetFormalCharge()
                   or a.GetNumRadicalElectrons() or a.GetIsotope() or int(a.GetChiralTag())
                   for a in iter_atoms(mol))
            or any(int(b.GetStereo()) for b in iter_bonds(mol))):
        return None
    rings = ring_systems(mol)
    if len(rings) != 1 or not 3 <= len(rings[0]) <= 10:
        return None
    ring = rings[0]
    site = _site(mol, ring)
    if site is None:
        return None
    n_result = _n_substituent_prefixes(mol, Chem, site)
    if n_result is None:
        return None
    n_prefixes, n_atoms, n_roles = n_result
    suffix_atoms = frozenset({site.center, site.oxygen, site.ligand})
    heavy = frozenset(a.GetIdx() for a in iter_atoms(mol))
    candidates = []
    for skeleton in skeletons(mol, Chem, ring):
        parent, sequence, locant_values = skeleton[:3]
        if frozenset(sequence) != ring:
            continue
        mapping = dict(zip(sequence, locant_values))
        prefixes, branches, roles = list(n_prefixes), set(n_atoms), list(n_roles)
        branch_ownership = []
        valid = True
        for idx in sequence:
            for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
                root = neighbor.GetIdx()
                if root in ring or root == site.center:
                    continue
                text, atoms = mapped_branch(mol, Chem, root, idx)
                if (not text or root not in atoms or branches & atoms
                        or atoms & (ring | suffix_atoms)):
                    valid = False
                    break
                prefixes.append((mapping[idx], text))
                branches.update(atoms)
                roles.append(("parent-carbon-substituent", frozenset(atoms), ()))
                branch_ownership.append((root, idx, frozenset(atoms), text))
            if not valid:
                break
        if not valid or branches & (ring | suffix_atoms) or ring | suffix_atoms | branches != heavy:
            continue
        host_locant = mapping[site.host]
        atom_locants = tuple(mapping.items()) + (
            (site.center, ""), (site.oxygen, ""), (site.ligand, "N"))
        component = MappedComponent(
            parent=f"{parent}-{host_locant}-carboxamide",
            atoms=ring | suffix_atoms, embedded_atoms=frozenset(branches),
            atom_locants=atom_locants, prefixes=tuple(prefixes),
            systematic_parent=True, nitrogen_site_locants=True,
            prefix_branch_ownership=tuple(branch_ownership))
        name = decorate_component(mol, Chem, component)
        if not name:
            continue
        record = SimpleNamespace(parent=parent, parent_atoms=tuple(sequence),
            atom_locants=tuple(mapping.items()), principal_centers=frozenset({site.center}),
            suffix_locants=(host_locant,), prefixes=component.prefixes, name=name,
            indicated_h_locants=(), method="annular-carboxamide", construction=component)
        facts = parent_facts(mol, record, (11, 2))
        if facts is None:
            continue
        all_roles = (("parent-hydride", ring, tuple(mapping.items())),
                     ("carboxamide", suffix_atoms, atom_locants[-3:]), *roles)
        owned = [i for _role, atoms, _locants in all_roles for i in atoms]
        if len(owned) != len(set(owned)) or frozenset(owned) != heavy:
            continue
        candidates.append((record, facts, all_roles))
    frontier = parent_frontier(candidates, lambda row: row[1])
    if not frontier or len({row[0].name for row in frontier}) != 1:
        return None
    record, _facts, roles = frontier[0]
    return MethodDecision(record.name, "substitutive", record.principal_centers,
        heavy, RULE, record.construction.atom_locants, component_roles=roles)


def _spiro_carboxamide_decision(mol, Chem):
    """Bind the selected diamide to its saturated spiro ring nitrogen."""
    from .component_mapping import mapped_general_branch
    from .extended_inventory import selection_inventory
    from .formatting import attach_suffix
    from .naming_result import SelectionLimitation

    if (mol.GetNumHeavyAtoms() > 64 or mol.GetStereoGroups()
            or any(a.GetAtomicNum() == 1 or a.GetIsotope() or a.GetFormalCharge()
                   or a.GetNumRadicalElectrons() or a.GetChiralTag() or a.HasQuery() for a in iter_atoms(mol))
            or any(b.GetStereo() or b.HasQuery() for b in iter_bonds(mol))):
        return None
    inventory = selection_inventory(mol)
    if not inventory.supported or len(inventory.principal) != 1 or inventory.principal[0].kind != 'urea':
        return None
    group = inventory.principal[0]
    nitrogens = [i for i in group.atoms if mol.GetAtomWithIdx(i).GetAtomicNum() == 7]
    if len(nitrogens) != 2:
        return None
    hosts = [i for i in nitrogens if mol.GetAtomWithIdx(i).IsInRing()]
    if len(hosts) != 1:
        return None
    host = hosts[0]
    ring = next((core for core in ring_systems(mol) if host in core), frozenset())
    if not _host_is_mapped_ring_nitrogen(mol, host, ring, group.center):
        return None
    ligand, roots = _terminal_amide_n(mol, group.center, ring)
    if ligand is None or len(roots) != 1:
        return None
    root = roots[0]
    text, atoms = mapped_general_branch(mol, Chem, root, ligand)
    suffix = frozenset(group.atoms) - {host}
    scope = frozenset(range(mol.GetNumAtoms()))
    if not text or atoms & (ring | suffix):
        return None
    ring_branches = []
    owned = set(ring | suffix | atoms)
    for idx in ring:
        for atom in mol.GetAtomWithIdx(idx).GetNeighbors():
            branch_root = atom.GetIdx()
            if branch_root in ring or branch_root == group.center:
                continue
            word, branch = mapped_general_branch(mol, Chem, branch_root, idx)
            if not word or branch & owned or branch_root not in branch:
                return None
            ring_branches.append((idx, word, branch))
            owned.update(branch)
    if owned != scope:
        return None
    candidates = []
    for row in skeletons(mol, Chem, ring):
        parent, sequence, locants, _elements, edges, saturated = row[:6]
        if not saturated or len(edges) != len(sequence) + 1 or 'spiro[' not in parent:
            continue
        mapping = tuple(zip(sequence, locants, strict=True))
        atom_locants = (*mapping, (ligand, 'N'))
        by_atom = dict(mapping)
        ending = attach_suffix(parent, 'carboxamide', (by_atom[host],))
        prefixes = (('N', text), *((by_atom[i], word) for i, word, _ in ring_branches))
        name = decorate_component(mol, Chem, MappedComponent(
            ending, atom_locants, ring | {ligand}, prefixes,
            scope - ring - {ligand}, systematic_parent=True, nitrogen_site_locants=True))
        if name:
            key = (locant_key(by_atom[host]), tuple(sorted(locant_key(by_atom[i])
                                                         for i, _, _ in ring_branches)))
            candidates.append((key, name, atom_locants, mapping))
    if not candidates:
        return None
    best = min(row[0] for row in candidates)
    candidates = [row[1:] for row in candidates if row[0] == best]
    if len({row[0] for row in candidates}) != 1:
        return None
    name, atom_locants, mapping = candidates[0]
    return MethodDecision(
        name, 'substitutive', frozenset({group.center}), scope, RULE, atom_locants,
        component_roles=(('parent-hydride', ring, mapping), ('carboxamide', suffix, ((ligand, 'N'),)),
                         ('N-substituent', atoms, ()),
                         *(('ring-substituent', branch, ()) for _, _, branch in ring_branches)),
        preference_issues=(SelectionLimitation('annular-carboxamide', RULE,
            'Complete method comparison for the mapped spiro carboxamide remains open.'),))
