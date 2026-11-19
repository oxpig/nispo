"""Mapped P-65.6.3.4 pseudoesters with a bounded Group 13 centre.

The accepted core is a single neutral R-C(=O)-O-E unit, E = Al, Ga, In, or
Tl, with exactly two simple carbon organyl arms on ordinary trivalent E.
Boron is excluded because its retained acid names and explicit anhydride
precedents require a different preferred construction. Group 15 is likewise
not handled here: P-65.6.3.4 itself classifies its printed P examples as
anhydrides, ahead of esters in class seniority.
"""

from __future__ import annotations

from dataclasses import dataclass

from .molecule_iteration import iter_atoms, iter_bonds
from .component_mapping import mapped_branch
from .naming_methods import MethodDecision
from .prefix_sequences import PrefixUnit, mononuclear_prefix_sequence


RULE = ("P-41/P-43/P-44/P-45/P-65.6.3.1.2/P-65.6.3.2.1/"
        "P-65.6.3.4.1/P-68.1.1.1/P-68.1.5/P-29.2/P-29.3.1/P-16.3/P-14.5")
_GROUP13 = {13: "alumanyl", 31: "gallanyl", 49: "indiganyl", 81: "thallanyl"}


@dataclass(frozen=True, slots=True)
class TrivalentPseudoesterSite:
    carbonyl: int
    carbonyl_oxygen: int
    bridge_oxygen: int
    center: int
    element: int
    arm_roots: tuple[int, int]
    arm_atoms: tuple[frozenset[int], frozenset[int]]
    acid_carbon_atoms: frozenset[int]


def _single(mol, left, right):
    bond = mol.GetBondBetweenAtoms(left, right)
    return bond is not None and bond.GetBondTypeAsDouble() == 1.0


def _simple_carbon_arm(mol, root, center):
    """Prove a terminal linear alkyl or unsubstituted phenyl E-ligand."""
    if not _single(mol, root, center):
        return None
    seen, pending = {root}, [root]
    while pending:
        idx = pending.pop()
        for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
            other = neighbor.GetIdx()
            if other == center or neighbor.GetAtomicNum() == 1:
                continue
            if neighbor.GetAtomicNum() != 6:
                return None
            if other not in seen:
                seen.add(other)
                pending.append(other)
    external = {(i, n.GetIdx()) for i in seen
                for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() not in seen}
    if external != {(root, center)}:
        return None
    atoms = frozenset(seen)
    ring = {i for i in atoms if mol.GetAtomWithIdx(i).IsInRing()}
    if ring:
        rings = [r for r in mol.GetRingInfo().AtomRings() if set(r) & atoms]
        if (len(atoms) != 6 or len(rings) != 1 or len(rings[0]) != 6
                or set(rings[0]) != set(atoms)
                or not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in atoms)
                or not all(mol.GetBondBetweenAtoms(a, b).GetIsAromatic()
                           for a, b in zip(rings[0], rings[0][1:] + rings[0][:1]))):
            return None
    else:
        if any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in atoms):
            return None
        # Secondary points of attachment need a separately locanted method-2
        # construction, so this component accepts terminal unbranched alkyl.
        if sum(n.GetIdx() in atoms for n in mol.GetAtomWithIdx(root).GetNeighbors()) > 1:
            return None
        if any(not _single(mol, i, n.GetIdx()) for i in atoms
               for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in atoms):
            return None
        if any(sum(n.GetIdx() in atoms for n in mol.GetAtomWithIdx(i).GetNeighbors()) > 2
               for i in atoms):
            return None
        if sum(1 for b in iter_bonds(mol) if b.GetBeginAtomIdx() in atoms
               and b.GetEndAtomIdx() in atoms) != len(atoms) - 1:
            return None
    if any(mol.GetAtomWithIdx(i).GetIsotope()
           or int(mol.GetAtomWithIdx(i).GetChiralTag()) for i in atoms):
        return None
    return atoms


def recognize_trivalent_pseudoester(mol):
    """Recognize the complete neutral Group 13 pseudoester graph."""
    if mol.GetStereoGroups() or mol.GetNumHeavyAtoms() > 64:
        return None
    if any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in iter_atoms(mol)):
        return None
    if any(a.GetIsotope() or int(a.GetChiralTag()) for a in iter_atoms(mol)):
        return None
    if any(int(b.GetStereo()) or int(b.GetBondDir()) for b in iter_bonds(mol)):
        return None
    if any(a.GetAtomicNum() == 1 and (a.GetIsotope() or a.GetDegree() != 1)
           for a in iter_atoms(mol)):
        return None
    heavy = [a for a in iter_atoms(mol) if a.GetAtomicNum() > 1]
    centers = [a for a in heavy if a.GetAtomicNum() in _GROUP13]
    oxygens = [a for a in heavy if a.GetAtomicNum() == 8]
    if len(centers) != 1 or len(oxygens) != 2:
        return None
    center = centers[0]
    if center.IsInRing() or center.GetTotalValence() != 3 or center.GetDegree() != 3:
        return None
    if center.GetTotalNumHs(includeNeighbors=True):
        return None
    bridges = [o for o in center.GetNeighbors() if o.GetAtomicNum() == 8
               and _single(mol, center.GetIdx(), o.GetIdx())]
    if len(bridges) != 1:
        return None
    bridge = bridges[0]
    if bridge.IsInRing() or bridge.GetDegree() != 2:
        return None
    other = [a for a in bridge.GetNeighbors() if a.GetIdx() != center.GetIdx()]
    if len(other) != 1 or other[0].GetAtomicNum() != 6:
        return None
    carbonyl = other[0]
    cidx = carbonyl.GetIdx()
    carbonyl_os = [a for a in carbonyl.GetNeighbors() if a.GetAtomicNum() == 8
                   and mol.GetBondBetweenAtoms(cidx, a.GetIdx()).GetBondTypeAsDouble() == 2]
    if len(carbonyl_os) != 1 or carbonyl.GetFormalCharge():
        return None
    carbonyl_o = carbonyl_os[0]
    remaining = [a for a in carbonyl.GetNeighbors()
                 if a.GetIdx() not in {bridge.GetIdx(), carbonyl_o.GetIdx()}]
    if len(remaining) > 1 or any(a.GetAtomicNum() != 6 for a in remaining):
        return None
    acid_carbons = set()
    if remaining:
        root = remaining[0].GetIdx()
        acid_carbons, pending = {root}, [root]
        while pending:
            idx = pending.pop()
            for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
                other_idx = neighbor.GetIdx()
                if other_idx == cidx or neighbor.GetAtomicNum() == 1:
                    continue
                if neighbor.GetAtomicNum() != 6:
                    return None
                if other_idx not in acid_carbons:
                    acid_carbons.add(other_idx)
                    pending.append(other_idx)
        external = {(i, n.GetIdx()) for i in acid_carbons
                    for n in mol.GetAtomWithIdx(i).GetNeighbors()
                    if n.GetIdx() not in acid_carbons}
        if external != {(root, cidx)}:
            return None
    if len([a for a in carbonyl.GetNeighbors()
            if mol.GetBondBetweenAtoms(cidx, a.GetIdx()).GetBondTypeAsDouble() == 2]) != 1:
        return None
    roots = [a for a in center.GetNeighbors() if a.GetIdx() != bridge.GetIdx()]
    if len(roots) != 2 or any(a.GetAtomicNum() != 6 for a in roots):
        return None
    groups = tuple(_simple_carbon_arm(mol, a.GetIdx(), center.GetIdx()) for a in roots)
    if any(group is None for group in groups):
        return None
    owned = {cidx, carbonyl_o.GetIdx(), bridge.GetIdx(), center.GetIdx()}
    owned.update(acid_carbons)
    for group in groups:
        if owned & set(group):
            return None
        owned.update(group)
    if any(a.GetAtomicNum() not in {6, 8, *_GROUP13} for a in heavy) \
            or owned != {a.GetIdx() for a in heavy}:
        return None
    seen, pending = {cidx}, [cidx]
    while pending:
        idx = pending.pop()
        for atom in mol.GetAtomWithIdx(idx).GetNeighbors():
            if atom.GetAtomicNum() > 1 and atom.GetIdx() not in seen:
                seen.add(atom.GetIdx())
                pending.append(atom.GetIdx())
    if seen != owned:
        return None
    return TrivalentPseudoesterSite(
        cidx, carbonyl_o.GetIdx(), bridge.GetIdx(), center.GetIdx(), center.GetAtomicNum(),
        tuple(a.GetIdx() for a in roots), groups, frozenset(acid_carbons))


def _group13_organyl(mol, Chem, site):
    names, embedded = [], set()
    for root, expected in zip(site.arm_roots, site.arm_atoms, strict=True):
        name, atoms = mapped_branch(mol, Chem, root, site.center)
        if not name or frozenset(atoms) != expected or embedded & set(atoms):
            return None
        names.append(name)
        embedded.update(atoms)
    if embedded != set().union(*site.arm_atoms):
        return None
    sequence = mononuclear_prefix_sequence(PrefixUnit(name, "simple") for name in names)
    if sequence is None:
        return None
    return sequence.text + _GROUP13[site.element], frozenset(embedded)


def trivalent_pseudoester_decision(mol, Chem):
    """Return a complete mapped functional-class proposal, or decline."""
    site = recognize_trivalent_pseudoester(mol)
    if site is None:
        return None
    from .acid_derivative_methods import _acid_component, _anion

    acid = _acid_component(mol, Chem, site.carbonyl, site.bridge_oxygen)
    if acid is None:
        return None
    expected_acid = {site.carbonyl, site.carbonyl_oxygen} | set(site.acid_carbon_atoms)
    if acid.atoms != expected_acid:
        return None
    anion = _anion(acid.name)
    organyl = _group13_organyl(mol, Chem, site)
    if not anion or organyl is None:
        return None
    group_name, arm_atoms = organyl
    heavy = frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1)
    acid_atoms, bridge_atoms = acid.atoms, frozenset({site.bridge_oxygen})
    e_atoms = frozenset({site.center}) | arm_atoms
    if (acid_atoms & bridge_atoms or acid_atoms & e_atoms or bridge_atoms & e_atoms
            or acid_atoms | bridge_atoms | e_atoms != heavy):
        return None
    return MethodDecision(
        name=f"{group_name} {anion}", method="functional-class",
        principal_centers=frozenset({site.carbonyl}), atoms=heavy, rule=RULE,
        atom_locants=acid.atom_locants,
        component_roles=(
            ("acid-anion", acid_atoms, acid.atom_locants),
            ("ester-bridge", bridge_atoms, ()),
            ("Group-13-organyl", e_atoms, ()),
        ),
    )
