"""Mapped ordinary-valence mononuclear Group 14 parent proposals.

This module is an isolated proposal producer for the source-closed P-68.2
subdomain implemented here.  It requires one whole-graph Si/Ge/Sn/Pb center,
single-bonded saturated carbon-tree/plain-phenyl organyl groups, including
terminal carbon-bound halogens, and at most
one homogeneous family of terminal amino, hydroxy, or sulfanyl suffix units.
Unsupported chemistry returns an explicit typed deferral; it is never reduced
to a hydride name by dropping atoms.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .component_mapping import MappedComponent, decorate_component, fully_covered, mapped_branch
from .formatting import attach_suffix, multiplied_suffix
from .naming_methods import MethodDecision


RULE = "P-41/P-43/P-44/P-45/P-63.1/P-68.2.1.1/P-68.2.4/P-68.2.5/P-68.2.6/P-82/P-91"
_PARENTS = {14: "silane", 32: "germane", 50: "stannane", 82: "plumbane"}
_SUFFIX = {7: "amine", 8: "ol", 16: "thiol"}
_VALENCE = 4


@dataclass(frozen=True, slots=True)
class Group14Recognition:
    status: Literal["supported", "not-applicable", "deferred"]
    reason: str
    center: int | None = None
    element: int | None = None
    carbon_roots: tuple[int, ...] = ()
    carbon_groups: tuple[frozenset[int], ...] = ()
    suffix_groups: tuple[tuple[int, int], ...] = ()
    core_isotope: int | None = None


@dataclass(frozen=True, slots=True)
class Group14Proposal:
    name: str
    method: str
    center: int
    element: int
    parent_atoms: tuple[int, ...]
    atom_locants: tuple[tuple[int, str], ...]
    component: MappedComponent
    suffix_atoms: frozenset[int]
    embedded_organyl_atoms: frozenset[int]
    rule: str = RULE

    @property
    def atoms(self):
        return self.component.atoms | self.component.embedded_atoms

    def as_method_decision(self) -> MethodDecision:
        """Adapt this proposal to the shared typed method-result interface."""
        core_atoms = self.component.atoms
        # The group-14 parent (including terminal suffix atoms represented in
        # its construction) and each prefix-owned organyl are disjoint roles.
        # Prefix branches have their own local numbering in their rendered
        # component, so their host attachment does not fabricate host locants.
        roles = [("group14-parent", core_atoms, self.component.atom_locants)]
        if self.embedded_organyl_atoms:
            roles.append(("organyl-prefixes", self.embedded_organyl_atoms, ()))
        role_atoms = [set(atoms) for _, atoms, _ in roles]
        if any(role_atoms[i] & role_atoms[j]
               for i in range(len(role_atoms)) for j in range(i + 1, len(role_atoms))):
            raise ValueError("Group 14 component roles overlap")
        if frozenset().union(*(frozenset(atoms) for atoms in role_atoms)) != self.atoms:
            raise ValueError("Group 14 component roles do not cover the proposal")
        return MethodDecision(
            name=self.name,
            method="substitutive",
            principal_centers=frozenset((self.center,)),
            atoms=self.atoms,
            rule=self.rule,
            atom_locants=self.atom_locants,
            component_roles=tuple(roles),
        )


def _single(mol, left: int, right: int) -> bool:
    bond = mol.GetBondBetweenAtoms(left, right)
    return bond is not None and bond.GetBondTypeAsDouble() == 1.0


def _carbon_organyl(mol, root: int, center: int):
    """Return a complete carbon-rooted tree/phenyl with typed halo branches."""
    if not _single(mol, root, center):
        return None
    if mol.GetAtomWithIdx(root).GetAtomicNum() != 6:
        return None
    allowed = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in {6, 9, 17, 35, 53}}
    seen, pending = {root}, [root]
    while pending:
        current = pending.pop()
        for neighbor in mol.GetAtomWithIdx(current).GetNeighbors():
            idx = neighbor.GetIdx()
            if idx == center or neighbor.GetAtomicNum() == 1:
                continue
            if idx not in allowed:
                return None
            if neighbor.GetAtomicNum() != 6 and (
                    neighbor.GetTotalValence() != 1 or neighbor.GetTotalNumHs()
                    or sum(n.GetAtomicNum() > 1 for n in neighbor.GetNeighbors()) != 1
                    or mol.GetAtomWithIdx(current).GetAtomicNum() != 6
                    or not _single(mol, current, idx)):
                return None
            if idx not in seen:
                seen.add(idx)
                pending.append(idx)
    # The ligand must have exactly one bond out, namely its root-to-parent
    # bond.  This rejects bridges and carbon branches shared by two centers.
    external = {(a.GetIdx(), n.GetIdx()) for a in mol.GetAtoms() if a.GetIdx() in seen
                for n in a.GetNeighbors() if n.GetIdx() not in seen}
    if external != {(root, center)}:
        return None
    atoms = frozenset(seen)
    carbon_atoms = frozenset(i for i in atoms if mol.GetAtomWithIdx(i).GetAtomicNum() == 6)
    ring_atoms = {i for i in atoms if mol.GetAtomWithIdx(i).IsInRing()}
    if ring_atoms:
        rings = [tuple(ring) for ring in mol.GetRingInfo().AtomRings()
                 if set(ring) & atoms]
        if (len(carbon_atoms) != 6 or len(rings) != 1 or len(rings[0]) != 6
                or set(rings[0]) != set(carbon_atoms)
                or not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in carbon_atoms)
                or not all(mol.GetBondBetweenAtoms(a, b).GetIsAromatic()
                           for a, b in zip(rings[0], rings[0][1:] + rings[0][:1]))):
            return None
    else:
        if any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in atoms):
            return None
        if any(not _single(mol, i, n.GetIdx()) for i in atoms
               for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in atoms):
            return None
        edge_count = sum(1 for b in mol.GetBonds()
                         if b.GetBeginAtomIdx() in atoms and b.GetEndAtomIdx() in atoms)
        if edge_count != len(atoms) - 1:
            return None
    return atoms


def recognize_group14_mononuclear(mol) -> Group14Recognition:
    """Classify this whole graph for the bounded mononuclear producer."""
    centers = [a for a in mol.GetAtoms() if a.GetAtomicNum() in _PARENTS]
    if not centers:
        return Group14Recognition("not-applicable", "no Group 14 hydride center")
    if len(centers) != 1:
        return Group14Recognition("deferred", "multiple Group 14 centers are outside this route")
    center = centers[0]
    cidx, element = center.GetIdx(), center.GetAtomicNum()
    if center.IsInRing() or center.GetFormalCharge() or center.GetNumRadicalElectrons():
        return Group14Recognition("deferred", "center is cyclic, charged, or radical", cidx, element)
    if int(center.GetChiralTag()) or center.GetTotalValence() != _VALENCE:
        return Group14Recognition("deferred", "central stereochemistry or nonordinary valence is unsupported", cidx, element)
    if mol.GetNumAtoms() > 160 or mol.GetStereoGroups():
        return Group14Recognition("deferred", "graph exceeds the bounded atom/stereo-group policy", cidx, element)
    if any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        return Group14Recognition("deferred", "charged or radical atom is outside the neutral rule", cidx, element)
    roots, groups, suffix_groups = [], [], []
    for neighbor in center.GetNeighbors():
        idx, number = neighbor.GetIdx(), neighbor.GetAtomicNum()
        if number == 1:
            if neighbor.GetIsotope():
                return Group14Recognition("deferred", "isotopic hydrogen is not rendered by this component grammar", cidx, element)
            continue
        if not _single(mol, cidx, idx):
            return Group14Recognition("deferred", "multiple center-to-ligand bond is outside the ordinary hydride route", cidx, element)
        if number == 6:
            group = _carbon_organyl(mol, idx, cidx)
            if group is None:
                return Group14Recognition("deferred", "organyl is not a supported carbon tree/phenyl with terminal carbon-bound halogens", cidx, element)
            if any(mol.GetAtomWithIdx(i).GetIsotope() for i in group):
                return Group14Recognition("deferred", "isotope-modified organyl prefixes are not yet rendered by this component grammar", cidx, element)
            if any(int(mol.GetAtomWithIdx(i).GetChiralTag()) for i in group):
                # Branch-local CIP can be added once its source order and
                # mapped descriptor insertion are covered as a separate rule.
                return Group14Recognition("deferred", "carbon stereochemistry in an organyl branch is not yet admitted", cidx, element)
            roots.append(idx)
            groups.append(group)
            continue
        if number in _SUFFIX and not neighbor.IsInRing() and neighbor.GetDegree() == 1:
            expected_h = 2 if number == 7 else 1
            if (neighbor.GetFormalCharge() or neighbor.GetNumRadicalElectrons()
                    or neighbor.GetTotalNumHs() != expected_h
                    or any(n.GetAtomicNum() == 1 and n.GetIsotope() for n in neighbor.GetNeighbors())):
                return Group14Recognition("deferred", "suffix ligand is not an unmodified terminal amino, hydroxy, or sulfanyl group", cidx, element)
            if neighbor.GetIsotope():
                return Group14Recognition("deferred", "suffix-atom isotope placement is outside this component grammar", cidx, element)
            suffix_groups.append((idx, number))
            continue
        return Group14Recognition("deferred", "direct ligand class is outside the mapped hydride/suffix grammar", cidx, element)
    if not roots:
        # With only chalcogen/hydroxy substituents, silicic acid and related
        # non-organic functional parents require the dedicated acid treatment.
        return Group14Recognition("deferred", "no carbon-containing organyl ligand; functional-parent scope is not handled", cidx, element)
    suffix_kinds = {number for _, number in suffix_groups}
    if len(suffix_kinds) > 1:
        return Group14Recognition("deferred", "mixed characteristic-group suffix competition is unproved", cidx, element)
    # Ensure every heavy atom is owned by the center, a recognized suffix unit,
    # or one disjoint organyl.  In particular an attached carboxylic acid may
    # not disappear into a superficially acceptable silicon name.
    owned = {cidx, *(idx for idx, _ in suffix_groups)}
    for group in groups:
        if owned & group:
            return Group14Recognition("deferred", "organyl components overlap", cidx, element)
        owned.update(group)
    heavy = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
    if owned != heavy:
        return Group14Recognition("deferred", "not every heavy atom belongs to one admitted ligand", cidx, element)
    isotope = center.GetIsotope() or None
    return Group14Recognition("supported", "complete ordinary-valence Group 14 graph", cidx,
                              element, tuple(roots), tuple(groups), tuple(suffix_groups), isotope)


def _group14_organyl_prefixes(mol, Chem, recognition):
    prefixes, embedded = [], set()
    for root, atoms in zip(recognition.carbon_roots, recognition.carbon_groups, strict=True):
        parent = recognition.center
        text, owned = mapped_branch(mol, Chem, root, parent)
        if not text or frozenset(owned) != atoms or embedded & set(owned):
            return None
        prefixes.append(("", text))
        embedded.update(owned)
    return tuple(prefixes), frozenset(embedded)


def _name_group14(mol, Chem, parent, prefixes, suffix_groups, recognition):
    """Keep typed prefix multiplication separate from the numbered parent."""
    from .prefix_sequences import PrefixUnit, mononuclear_prefix_sequence

    terms = {}
    for (_, text), atoms in zip(prefixes, recognition.carbon_groups, strict=True):
        # A straight carbon path remains an unsubstituted simple group even
        # when the free valence is internal. Branched groups are compound,
        # except for the retained simple tert-butyl prefix (P-16.3.2(a)).
        path = all(sum(n.GetIdx() in atoms for n in mol.GetAtomWithIdx(i).GetNeighbors()) <= 2
                   for i in atoms)
        substituted = any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in atoms)
        nature = 'simple' if not substituted and (text in {'phenyl', 'tert-butyl'} or path) else 'compound'
        if text in terms and terms[text][1] != nature:
            return None
        count, _ = terms.get(text, (0, nature))
        terms[text] = count + 1, nature
    sequence = mononuclear_prefix_sequence(
        PrefixUnit(text, nature, count) for text, (count, nature) in terms.items()
    ) if terms else None
    if terms and sequence is None:
        return None
    ending = parent
    if suffix_groups:
        suffix = multiplied_suffix(len(suffix_groups), _SUFFIX[suffix_groups[0][1]])
        if not suffix:
            return None
        ending = attach_suffix(parent, suffix)
    return sequence.text if sequence else '', ending


def group14_mononuclear_proposal(mol, Chem) -> Group14Proposal | None:
    """Return a complete mapped proposal, or ``None`` outside this scope."""
    recognition = recognize_group14_mononuclear(mol)
    if recognition.status != "supported":
        return None
    parent = _PARENTS[recognition.element]
    branch_result = _group14_organyl_prefixes(mol, Chem, recognition)
    if branch_result is None:
        return None
    prefixes, embedded = branch_result
    suffix_atoms = frozenset(idx for idx, _ in recognition.suffix_groups)
    owned = frozenset((recognition.center, *suffix_atoms))
    if owned & embedded:
        return None
    locants = {recognition.center: ""}
    locants.update({idx: "" for idx in suffix_atoms})
    # An isotope at the only Group 14 skeleton atom is uniquely attributable
    # and is rendered without an artificial numeric parent locant (P-82).
    bindings = tuple(sorted(locants.items()))
    words = _name_group14(mol, Chem, parent, prefixes, recognition.suffix_groups, recognition)
    if words is None:
        return None
    prefix_text, ending = words
    construction = MappedComponent(
        parent=ending,
        atom_locants=bindings,
        atoms=owned,
        prefixes=(),
        embedded_atoms=embedded,
        systematic_parent=True,
        isotope_unlocanted_elements=(recognition.element,),
    )
    parent_name = decorate_component(mol, Chem, construction)
    if not parent_name or not fully_covered(mol, construction.atoms, construction.embedded_atoms):
        return None
    # P-82.2.1 places a parent isotope descriptor after the separately cited
    # substituent prefixes, as in trichloro(12C)methane. Decorating a string
    # already containing those prefixes would lose this construction boundary.
    name = prefix_text + parent_name
    method = "group14-mononuclear-suffix" if suffix_atoms else "group14-mononuclear-substituted-hydride"
    return Group14Proposal(
        name=name, method=method, center=recognition.center, element=recognition.element,
        parent_atoms=(recognition.center,), atom_locants=bindings,
        component=construction, suffix_atoms=suffix_atoms,
        embedded_organyl_atoms=embedded,
    )

