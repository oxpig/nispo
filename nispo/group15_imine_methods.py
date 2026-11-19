"""Mapped, bounded Group 15 imine suffix names under P-68.3."""
from __future__ import annotations

from dataclasses import dataclass
import re

from .component_mapping import MappedComponent, decorate_component, mapped_branch
from .formatting import attach_suffix, enclose_name_unit, prefix_alphabetization_key
from .naming_methods import MethodDecision
from .prefix_sequences import PrefixUnit
from .substituent_multiplication import multiply_substituent


RULE = "P-14/P-16/P-21.1/P-41/P-43/P-44/P-45/P-68.3.2.3.1/P-68.3.3/P-82"
_PARENTS = {15: "phosphane", 33: "arsane", 51: "stibane", 83: "bismuthane"}
_SYMBOLS = {15: "P", 33: "As", 51: "Sb", 83: "Bi"}
_LOCANTED_ALKYL = re.compile(r"^[A-Za-z]+an-[0-9]+-yl$")


@dataclass(frozen=True, slots=True)
class _ImineGraph:
    center: int
    nitrogen: int
    element: int
    center_roots: tuple[int, ...]
    center_groups: tuple[frozenset[int], ...]
    nitrogen_root: int | None
    nitrogen_group: frozenset[int]
    lambda_valence: int | None


def _single(mol, a: int, b: int) -> bool:
    bond = mol.GetBondBetweenAtoms(a, b)
    return bond is not None and bond.GetBondTypeAsDouble() == 1.0


def _carbon_organyl(mol, root: int, host: int) -> frozenset[int] | None:
    """Prove an all-carbon saturated tree or plain phenyl arm."""
    if not _single(mol, root, host):
        return None
    atoms, todo = {root}, [root]
    while todo:
        idx = todo.pop()
        for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
            other = neighbor.GetIdx()
            if other == host or neighbor.GetAtomicNum() == 1:
                continue
            if neighbor.GetAtomicNum() != 6:
                return None
            if other not in atoms:
                atoms.add(other)
                todo.append(other)
    external = {(a.GetIdx(), n.GetIdx()) for a in mol.GetAtoms()
                if a.GetIdx() in atoms for n in a.GetNeighbors()
                if n.GetIdx() not in atoms}
    if external != {(root, host)}:
        return None
    group = frozenset(atoms)
    ring = {i for i in group if mol.GetAtomWithIdx(i).IsInRing()}
    if ring:
        rings = [set(r) for r in mol.GetRingInfo().AtomRings() if set(r) & group]
        if (ring != set(group) or len(group) != 6 or len(rings) != 1
                or rings[0] != set(group)
                or not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in group)
                or not all(b.GetIsAromatic() for b in mol.GetBonds()
                           if b.GetBeginAtomIdx() in group and b.GetEndAtomIdx() in group)):
            return None
    else:
        if any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in group):
            return None
        if any(not _single(mol, i, n.GetIdx()) for i in group
               for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in group):
            return None
        if sum(1 for b in mol.GetBonds()
               if b.GetBeginAtomIdx() in group and b.GetEndAtomIdx() in group) != len(group) - 1:
            return None
    return group


def _recognize(mol) -> _ImineGraph | None:
    centers = [a for a in mol.GetAtoms() if a.GetAtomicNum() in _PARENTS]
    if len(centers) != 1 or mol.GetNumAtoms() > 96 or mol.GetStereoGroups():
        return None
    center = centers[0]
    cidx, element = center.GetIdx(), center.GetAtomicNum()
    if (center.IsInRing() or center.GetFormalCharge() or center.GetNumRadicalElectrons()
            or center.GetIsotope() or int(center.GetChiralTag())
            or any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in mol.GetAtoms())
            or any(a.GetIsotope() or int(a.GetChiralTag()) for a in mol.GetAtoms())
            or any(a.GetAtomicNum() == 1 for a in mol.GetAtoms())):
        return None
    # The P=N geometry is not rendered by this bounded method. Reject E/Z or
    # directional bond annotations rather than silently dropping them.
    if any(int(b.GetStereo()) != 0 or int(b.GetBondDir()) != 0 for b in mol.GetBonds()):
        return None

    imine_n = []
    center_roots, center_groups = [], []
    for neighbor in center.GetNeighbors():
        idx, number = neighbor.GetIdx(), neighbor.GetAtomicNum()
        bond = mol.GetBondBetweenAtoms(cidx, idx)
        if number == 7 and bond.GetBondTypeAsDouble() == 2.0:
            imine_n.append(idx)
        elif number == 6 and bond.GetBondTypeAsDouble() == 1.0:
            group = _carbon_organyl(mol, idx, cidx)
            if group is None:
                return None
            center_roots.append(idx)
            center_groups.append(group)
        else:
            return None
    if len(imine_n) != 1:
        return None
    nitrogen_idx = imine_n[0]
    nitrogen = mol.GetAtomWithIdx(nitrogen_idx)
    if (nitrogen.IsInRing() or nitrogen.GetFormalCharge() or nitrogen.GetNumRadicalElectrons()
            or nitrogen.GetIsotope() or int(nitrogen.GetChiralTag())):
        return None
    n_roots, n_groups = [], []
    for neighbor in nitrogen.GetNeighbors():
        idx = neighbor.GetIdx()
        if idx == cidx:
            continue
        if neighbor.GetAtomicNum() != 6 or not _single(mol, nitrogen_idx, idx):
            return None
        group = _carbon_organyl(mol, idx, nitrogen_idx)
        if group is None:
            return None
        n_roots.append(idx)
        n_groups.append(group)
    if len(n_roots) > 1 or (not n_roots and nitrogen.GetTotalNumHs(includeNeighbors=True) != 1):
        return None
    if n_roots and nitrogen.GetTotalNumHs(includeNeighbors=True) != 0:
        return None
    if not center_roots and not n_roots:
        return None
    # P-74.2.1.5 explicitly extends the phosphine imide method to arsine and
    # stibine imides; the mixed N/center pattern is not extended there to Bi.
    if center_roots and n_roots and element == 83:
        return None
    valence = int(center.GetTotalValence())
    if valence not in {3, 5}:
        return None

    owned = {cidx, nitrogen_idx}
    for group in (*center_groups, *n_groups):
        if owned & group:
            return None
        owned.update(group)
    heavy = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
    if owned != heavy:
        return None
    return _ImineGraph(cidx, nitrogen_idx, element, tuple(center_roots),
        tuple(center_groups), n_roots[0] if n_roots else None,
        n_groups[0] if n_groups else frozenset(), 5 if valence == 5 else None)


def _nature_from_graph(mol, atoms: frozenset[int], text: str) -> str | None:
    """Classify P-16 component nature from mapped structure, not name spelling."""
    if not atoms:
        return None
    ring_atoms = {i for i in atoms if mol.GetAtomWithIdx(i).IsInRing()}
    if ring_atoms:
        return "simple" if len(atoms) == 6 and ring_atoms == set(atoms) else None
    degrees = {i: sum(n.GetIdx() in atoms for n in mol.GetAtomWithIdx(i).GetNeighbors())
               for i in atoms}
    if max(degrees.values(), default=0) <= 2:
        return "simple"
    # Retained tert-butyl is a simple prefix despite its three methyl arms.
    root = next((i for i in atoms if any(n.GetAtomicNum() in (set(_PARENTS) | {7})
                                         and n.GetIdx() not in atoms
                                         for n in mol.GetAtomWithIdx(i).GetNeighbors())), None)
    if root is not None:
        branches = [n.GetIdx() for n in mol.GetAtomWithIdx(root).GetNeighbors()
                    if n.GetIdx() in atoms]
        if len(atoms) == 4 and len(branches) == 3 and all(
                sum(x.GetIdx() in atoms for x in mol.GetAtomWithIdx(branch).GetNeighbors()) == 1
                for branch in branches):
            return "simple" if text == "tert-butyl" else None
    return "compound"


def _locanted_prefix_groups(mol, Chem, graph):
    """Build globally ordered typed prefix units with their own host locants.

    Group locants stay attached to their individual names. This matters for
    heteroimines: P,P,P-triphenyl is one locanted component, while an unlike
    P-ethyl component needs its own P locator rather than a locant list moved
    in front of the whole prefix sequence.
    """
    raw = []
    symbol = _SYMBOLS[graph.element]
    for host, roots, groups in (
        ("N", ((graph.nitrogen_root,) if graph.nitrogen_root is not None else ()),
         ((graph.nitrogen_group,) if graph.nitrogen_root is not None else ())),
        (symbol, graph.center_roots, graph.center_groups),
    ):
        for root, atoms in zip(roots, groups, strict=True):
            text, owned = mapped_branch(mol, Chem, root, graph.nitrogen if host == "N" else graph.center)
            if not text or frozenset(owned) != atoms:
                return None
            nature = _nature_from_graph(mol, atoms, text)
            if nature is None:
                return None
            raw.append((host, text, nature))
    if not raw:
        return None

    # The same component at N and at the parent center requires a source-closed
    # tie-break between two distinct host locants; leave that case unresolved.
    hosts_by_name = {}
    for host, text, _nature in raw:
        hosts_by_name.setdefault(text, set()).add(host)
    if any(len(hosts) > 1 for hosts in hosts_by_name.values()):
        return None
    center_names = [text for host, text, _nature in raw if host == symbol]
    if len(set(center_names)) > 1:
        # The locant/enclosure interaction for unlike substituents on the
        # same mononuclear center is not resolved by the present P-16 proof.
        return None

    grouped = {}
    for host, text, nature in raw:
        key = (host, text, nature)
        grouped[key] = grouped.get(key, 0) + 1
    # A name with one spelling but inconsistent graph-derived nature is not a
    # resolved P-14.5 identity.
    if len({(host, text) for host, text, _nature in grouped}) != len(grouped):
        return None
    keys = [prefix_alphabetization_key(text) for _host, text, _nature in grouped]
    if len(set(keys)) != len(keys):
        return None

    rendered = []
    for (host, text, nature), count in grouped.items():
        multiplied = multiply_substituent(text, count, nature=nature)
        if multiplied is None:
            return None
        component = multiplied.text
        # Parenthesize a single complex or locant-bearing component as a unit;
        # its host locator remains outside the parentheses.
        locant_bearing = bool(_LOCANTED_ALKYL.fullmatch(text))
        if count == 1 and (nature == "compound" or locant_bearing):
            component = enclose_name_unit(component)
        labels = ",".join(host for _ in range(count))
        rendered.append((prefix_alphabetization_key(text), f"{labels}-{component}"))
    rendered.sort(key=lambda item: item[0])
    return "-".join(text for _key, text in rendered)


def group15_imine_decision(mol, Chem) -> MethodDecision | None:
    """Return a complete mapped substitutive Group 15 imine decision."""
    graph = _recognize(mol)
    if graph is None:
        return None
    prefix_text = _locanted_prefix_groups(mol, Chem, graph)
    if prefix_text is None:
        return None
    parent_hydride = _PARENTS[graph.element]
    parent = attach_suffix(parent_hydride, "imine")
    if graph.lambda_valence:
        parent = f"lambda{graph.lambda_valence}-{parent}"
    core = frozenset((graph.center, graph.nitrogen))
    branches = (*graph.center_groups, *((graph.nitrogen_group,) if graph.nitrogen_root is not None else ()))
    embedded = frozenset().union(*branches) if branches else frozenset()
    component = MappedComponent(parent,
        ((graph.center, _SYMBOLS[graph.element]), (graph.nitrogen, "N")), core,
        embedded_atoms=embedded, systematic_parent=True, nitrogen_site_locants=True)
    core_text = decorate_component(mol, Chem, component)
    if not core_text:
        return None

    name = prefix_text + ("-" if graph.lambda_valence else "") + core_text

    roles = [("group15-imine-parent", core,
              ((graph.center, _SYMBOLS[graph.element]), (graph.nitrogen, "N")))]
    for index, atoms in enumerate(graph.center_groups, start=1):
        roles.append((f"center-organyl-{index}", atoms, ()))
    if graph.nitrogen_root is not None:
        roles.append(("nitrogen-organyl", graph.nitrogen_group, ()))
    all_heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    role_sets = [set(atoms) for _, atoms, _ in roles]
    if (any(role_sets[i] & role_sets[j] for i in range(len(role_sets))
            for j in range(i + 1, len(role_sets)))
            or frozenset().union(*(frozenset(x) for x in role_sets)) != all_heavy):
        return None
    return MethodDecision(name=name, method="substitutive",
        principal_centers=frozenset((graph.center,)), atoms=all_heavy,
        rule=RULE, atom_locants=component.atom_locants, component_roles=tuple(roles))
