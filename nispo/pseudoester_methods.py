"""Mapped functional-class names for a narrow P-65.6.3.4 pseudoester family.

The accepted graph has one carbon-only carboxylic acid component, its single
ester oxygen attached to one ordinary-valence Group 14 atom, and three simple
carbon organyl arms on that atom. Unsupported or annotated graphs decline.
"""

from __future__ import annotations

from dataclasses import dataclass

from .molecule_iteration import iter_atoms, iter_bonds
from .component_mapping import mapped_branch
from .naming_methods import MethodDecision
from .prefix_sequences import PrefixUnit, mononuclear_prefix_sequence


RULE = ("P-41/P-43/P-44/P-45/P-65.6.3.1.2/P-65.6.3.2.1/"
        "P-65.6.3.3.1/P-65.6.3.4.1/P-68.2.2/P-29.2/P-16.3/P-14.5")
_GROUP14 = {14: "silyl", 32: "germyl", 50: "stannyl", 82: "plumbyl"}


@dataclass(frozen=True, slots=True)
class PseudoesterSite:
    carbonyl: int
    carbonyl_oxygen: int
    bridge_oxygen: int
    center: int
    element: int
    arm_roots: tuple[int, int, int]
    arm_atoms: tuple[frozenset[int], frozenset[int], frozenset[int]]
    acid_carbon_atoms: frozenset[int]


def _single(mol, left, right):
    bond = mol.GetBondBetweenAtoms(left, right)
    return bond is not None and bond.GetBondTypeAsDouble() == 1.0


def _simple_carbon_arm(mol, root, center):
    """Prove a saturated linear alkyl or unsubstituted phenyl E-ligand."""
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
    external = {(i, n.GetIdx()) for i in seen for n in mol.GetAtomWithIdx(i).GetNeighbors()
                if n.GetIdx() not in seen}
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
        # This bounded prefix sequence represents an ordinary terminal alkyl
        # arm. A secondary point of attachment needs the separate locanted
        # substituent/multiplication proof and is intentionally deferred.
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


def recognize_pseudoester(mol):
    """Recognize the whole-graph neutral R-C(O)-O-E pseudoester boundary."""
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
    centers = [a for a in heavy if a.GetAtomicNum() in _GROUP14]
    oxygens = [a for a in heavy if a.GetAtomicNum() == 8]
    if len(centers) != 1 or len(oxygens) != 2:
        return None
    center = centers[0]
    if center.IsInRing() or center.GetTotalValence() != 4 or center.GetDegree() != 4:
        return None
    if center.GetTotalNumHs(includeNeighbors=True):
        return None
    # Locate the one E-O-C(=O) ester bridge directly from the original graph.
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
                other = neighbor.GetIdx()
                if other == cidx or neighbor.GetAtomicNum() == 1:
                    continue
                if neighbor.GetAtomicNum() != 6:
                    return None
                if other not in acid_carbons:
                    acid_carbons.add(other)
                    pending.append(other)
        external = {(i, n.GetIdx()) for i in acid_carbons
                    for n in mol.GetAtomWithIdx(i).GetNeighbors()
                    if n.GetIdx() not in acid_carbons}
        if external != {(root, cidx)}:
            return None
    if len([a for a in carbonyl.GetNeighbors()
            if mol.GetBondBetweenAtoms(cidx, a.GetIdx()).GetBondTypeAsDouble() == 2]) != 1:
        return None
    roots = [a for a in center.GetNeighbors() if a.GetIdx() != bridge.GetIdx()]
    if len(roots) != 3 or any(a.GetAtomicNum() != 6 for a in roots):
        return None
    groups = tuple(_simple_carbon_arm(mol, a.GetIdx(), center.GetIdx()) for a in roots)
    if any(group is None for group in groups):
        return None
    if any(mol.GetBondBetweenAtoms(center.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() != 1
           for a in roots):
        return None
    owned = {cidx, carbonyl_o.GetIdx(), bridge.GetIdx(), center.GetIdx()}
    owned.update(acid_carbons)
    for group in groups:
        if owned & set(group):
            return None
        owned.update(group)
    if any(a.GetAtomicNum() not in {6, 8, *_GROUP14}
           for a in heavy) or owned != {a.GetIdx() for a in heavy}:
        return None
    # Enforce a single connected molecular entity.
    seen, pending = {cidx}, [cidx]
    while pending:
        idx = pending.pop()
        for atom in mol.GetAtomWithIdx(idx).GetNeighbors():
            if atom.GetAtomicNum() > 1 and atom.GetIdx() not in seen:
                seen.add(atom.GetIdx())
                pending.append(atom.GetIdx())
    if seen != owned:
        return None
    return PseudoesterSite(cidx, carbonyl_o.GetIdx(), bridge.GetIdx(), center.GetIdx(),
                           center.GetAtomicNum(), tuple(a.GetIdx() for a in roots), groups,
                           frozenset(acid_carbons))


def _group14_organyl(mol, Chem, site):
    from .group14_mononuclear_methods import recognize_group14_mononuclear

    # Keep only the original E-containing component. On sanitization the
    # ester oxygen becomes an OH cap, letting the existing Group 14 recognizer
    # validate this same mapped three-arm topology without a handwritten
    # second recognizer.
    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        atom.SetProp("_pseudoester_origin", str(atom.GetIdx()))
    retained = {site.center, site.bridge_oxygen} | set().union(*site.arm_atoms)
    for idx in reversed(range(rw.GetNumAtoms())):
        if idx not in retained:
            rw.RemoveAtom(idx)
    capped = rw.GetMol()
    cap_oxygen = next(a for a in capped.GetAtoms()
                      if int(a.GetProp("_pseudoester_origin")) == site.bridge_oxygen)
    # A bracketed [O] ester input may carry noImplicit=True despite being an
    # ordinary neutral bridge O. After the acid-side bond is removed, the
    # private parent-search component represents that O as OH; allow RDKit to
    # supply its cap hydrogen without changing the source graph.
    cap_oxygen.SetNoImplicit(False)
    try:
        Chem.SanitizeMol(capped)
    except (ValueError, RuntimeError):
        return None
    recognition = recognize_group14_mononuclear(capped)
    origin = {a.GetIdx(): int(a.GetProp("_pseudoester_origin")) for a in capped.GetAtoms()}
    roots_original = tuple(origin[i] for i in recognition.carbon_roots) \
        if recognition.carbon_roots else ()
    suffix_original = tuple((origin[i], number) for i, number in recognition.suffix_groups)
    if (recognition.status != "supported" or origin.get(recognition.center) != site.center
            or recognition.element != site.element or len(recognition.carbon_roots) != 3
            or suffix_original != ((site.bridge_oxygen, 8),)
            or set(roots_original) != set(site.arm_roots)):
        return None
    arm_names, embedded = [], set()
    for root, expected in zip(site.arm_roots, site.arm_atoms, strict=True):
        name, atoms = mapped_branch(mol, Chem, root, site.center)
        if not name or frozenset(atoms) != expected or embedded & set(atoms):
            return None
        arm_names.append(name)
        embedded.update(atoms)
    if set(embedded) != set().union(*site.arm_atoms):
        return None
    sequence = mononuclear_prefix_sequence(PrefixUnit(name, "simple") for name in arm_names)
    if sequence is None:
        return None
    return sequence.text + _GROUP14[site.element], frozenset(embedded)


def pseudoester_decision(mol, Chem):
    """Return an owned P-65.6.3.4 functional-class decision or ``None``."""
    site = recognize_pseudoester(mol)
    if site is None:
        return None
    from .acid_derivative_methods import _acid_component, _anion

    acid = _acid_component(mol, Chem, site.carbonyl, site.bridge_oxygen)
    if acid is None:
        return None
    # The acid-side component is deliberately restricted to C/H plus its one
    # carbonyl oxygen; the whole-graph recognizer already excludes other
    # heteroatoms and any competing senior acid/anhydride functionality.
    expected_acid = {site.carbonyl, site.carbonyl_oxygen} | set(site.acid_carbon_atoms)
    if acid.atoms != expected_acid:
        return None
    anion = _anion(acid.name)
    if not anion:
        return None
    organyl = _group14_organyl(mol, Chem, site)
    if organyl is None:
        return None
    group_name, arm_atoms = organyl
    heavy = frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1)
    acid_atoms = acid.atoms
    bridge_atoms = frozenset({site.bridge_oxygen})
    e_atoms = frozenset({site.center}) | arm_atoms
    if (acid_atoms & bridge_atoms or acid_atoms & e_atoms or bridge_atoms & e_atoms
            or acid_atoms | bridge_atoms | e_atoms != heavy):
        return None
    return MethodDecision(
        name=f"{group_name} {anion}",
        method="functional-class",
        principal_centers=frozenset({site.carbonyl}),
        atoms=heavy,
        rule=RULE,
        atom_locants=acid.atom_locants,
        component_roles=(
            ("acid-anion", acid_atoms, acid.atom_locants),
            ("ester-bridge", bridge_atoms, ()),
            ("Group-14-organyl", e_atoms, ()),
        ),
    )
