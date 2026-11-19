"""Mapped ordinary-valence O/S/Se/Te cation components under P-73.1.

This isolated producer covers one hydron-added chalcogen parent-hydride
center with zero to three simple carbon organyl arms.  It does not assemble
counterions or choose among competing principal characteristic groups.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .branch_rules import parent_chain_name
from .constants import ROOTS
from .naming_methods import MethodDecision
from .prefix_sequences import PrefixUnit, mononuclear_prefix_sequence


RULE = "P-73.1.1.2/P-73.1.2.1(2)/P-21.1/P-29.3/P-16.3/P-16.5/P-14.5"
_PARENT = {8: "oxidanium", 16: "sulfanium", 34: "selanium", 52: "tellanium"}
_HEAVY_LIMIT = 64


@dataclass(frozen=True, slots=True)
class ChalcogenCationRecognition:
    status: Literal["supported", "not-applicable", "deferred"]
    reason: str
    center: int | None = None
    element: int | None = None
    carbon_roots: tuple[int, ...] = ()
    carbon_groups: tuple[frozenset[int], ...] = ()
    hydrogen_count: int = 0


@dataclass(frozen=True, slots=True)
class ChalcogenCationProposal:
    name: str
    center: int
    element: int
    atoms: frozenset[int]
    atom_locants: tuple[tuple[int, str], ...]
    carbon_roots: tuple[int, ...]
    carbon_groups: tuple[frozenset[int], ...]
    arm_components: tuple[str, ...]
    charge: int = 1
    rule: str = RULE

    def as_method_decision(self) -> MethodDecision:
        roles = [("chalcogen-cation-parent", frozenset((self.center,)),
                  ((self.center, ""),))]
        for root, atoms, text in zip(self.carbon_roots, self.carbon_groups,
                                     self.arm_components, strict=True):
            roles.append(("organyl-prefix:" + text, atoms, ()))
        if any(roles[i][1] & roles[j][1]
               for i in range(len(roles)) for j in range(i + 1, len(roles))):
            raise ValueError("chalcogen cation component roles overlap")
        if frozenset().union(*(role[1] for role in roles)) != self.atoms:
            raise ValueError("chalcogen cation component roles do not cover atoms")
        return MethodDecision(
            name=self.name,
            method="substitutive",
            principal_centers=frozenset((self.center,)),
            atoms=self.atoms,
            rule=self.rule,
            atom_locants=self.atom_locants,
            component_roles=tuple(roles),
        )


def _single(mol, left: int, right: int, Chem) -> bool:
    bond = mol.GetBondBetweenAtoms(left, right)
    return bond is not None and bond.GetBondType() == Chem.BondType.SINGLE


def _organyl_group(mol, Chem, root: int, center: int) -> frozenset[int] | None:
    """Prove one complete unbranched saturated alkyl or plain phenyl arm."""
    if not _single(mol, root, center, Chem):
        return None
    seen, pending = {root}, [root]
    while pending:
        current = pending.pop()
        for neighbor in mol.GetAtomWithIdx(current).GetNeighbors():
            idx = neighbor.GetIdx()
            if idx == center or neighbor.GetAtomicNum() == 1:
                continue
            if neighbor.GetAtomicNum() != 6:
                return None
            if idx not in seen:
                seen.add(idx)
                pending.append(idx)
    atoms = frozenset(seen)
    external = {(idx, neighbor.GetIdx())
                for idx in atoms
                for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors()
                if neighbor.GetIdx() not in atoms}
    if external != {(root, center)}:
        return None
    ring_atoms = {idx for idx in atoms if mol.GetAtomWithIdx(idx).IsInRing()}
    if ring_atoms:
        rings = [set(ring) for ring in mol.GetRingInfo().AtomRings()
                 if set(ring) & atoms]
        if (len(atoms) != 6 or len(rings) != 1 or rings[0] != set(atoms)
                or not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in atoms)
                or not all(mol.GetAtomWithIdx(i).GetTotalValence() == 4 for i in atoms)
                or any(not mol.GetBondBetweenAtoms(i, n.GetIdx()).GetIsAromatic()
                       for i in atoms for n in mol.GetAtomWithIdx(i).GetNeighbors()
                       if n.GetIdx() in atoms)):
            return None
        return atoms
    if any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in atoms):
        return None
    root_inside = sum(neighbor.GetIdx() in atoms
                      for neighbor in mol.GetAtomWithIdx(root).GetNeighbors())
    if root_inside > 1:
        return None
    edge_count = 0
    for idx in atoms:
        atom = mol.GetAtomWithIdx(idx)
        inside = [neighbor.GetIdx() for neighbor in atom.GetNeighbors()
                  if neighbor.GetIdx() in atoms]
        if atom.GetTotalValence() != 4 or len(inside) > 2 or any(not _single(mol, idx, other, Chem)
                                   for other in inside):
            return None
        edge_count += len(inside)
    if edge_count // 2 != len(atoms) - 1:
        return None
    if sum(1 for idx in atoms
           if sum(n.GetIdx() in atoms for n in mol.GetAtomWithIdx(idx).GetNeighbors()) <= 1) != 2 and len(atoms) > 1:
        return None
    return atoms


def recognize_chalcogen_cation(mol, Chem=None) -> ChalcogenCationRecognition:
    """Classify a complete graph for this narrow hydron-added cation route."""
    if Chem is None:
        from rdkit import Chem as RDKitChem
        Chem = RDKitChem
    centers = [atom for atom in mol.GetAtoms()
               if atom.GetAtomicNum() in _PARENT and atom.GetFormalCharge()]
    if not centers:
        return ChalcogenCationRecognition("not-applicable", "no charged O/S/Se/Te center")
    if len(centers) != 1:
        return ChalcogenCationRecognition("deferred", "multiple charged chalcogen centers are outside this route")
    center = centers[0]
    cidx, element = center.GetIdx(), center.GetAtomicNum()
    def decline(reason):
        return ChalcogenCationRecognition("deferred", reason, cidx, element)
    if (mol.GetNumHeavyAtoms() > _HEAVY_LIMIT or mol.GetNumAtoms() != mol.GetNumHeavyAtoms()
            or len(Chem.GetMolFrags(mol)) != 1 or mol.GetStereoGroups()):
        return decline("graph is disconnected, explicit-H, grouped-stereo, or over the heavy-atom bound")
    if any(int(bond.GetStereo()) or bond.GetBondDir() != Chem.BondDir.NONE
           for bond in mol.GetBonds()):
        return decline("bond stereochemistry or directional annotations are outside this route")
    if (center.GetFormalCharge() != 1 or center.IsInRing() or center.GetIsAromatic()
            or center.GetIsotope() or int(center.GetChiralTag())
            or center.GetNumRadicalElectrons() or center.GetTotalValence() != 3):
        return decline("center is not an unannotated ordinary-valence monocation")
    if any(atom.GetNumRadicalElectrons() or atom.GetFormalCharge()
           for atom in mol.GetAtoms() if atom.GetIdx() != cidx):
        return decline("additional charge or radical is outside this monocation route")
    if sum(atom.GetFormalCharge() for atom in mol.GetAtoms()) != 1:
        return decline("whole-graph charge is not +1")
    neighbors = list(center.GetNeighbors())
    roots = []
    for neighbor in neighbors:
        idx = neighbor.GetIdx()
        if neighbor.GetAtomicNum() != 6 or not _single(mol, cidx, idx, Chem):
            return decline("only single-bonded carbon organyl ligands are admitted")
        group = _organyl_group(mol, Chem, idx, cidx)
        if group is None:
            return decline("ligand is not an unbranched saturated alkyl group or plain phenyl")
        roots.append(idx)
    if len(roots) > 3:
        return decline("ordinary cation parent supports at most three carbon replacements")
    expected_h = 3 - len(roots)
    if center.GetTotalNumHs(includeNeighbors=True) != expected_h:
        return decline("chalcogen hydrogen count does not complete the three-coordinate cation")
    groups = tuple(_organyl_group(mol, Chem, root, cidx) for root in roots)
    if any(group is None for group in groups) or len(set().union(*groups) if groups else set()) != sum(map(len, groups)):
        return decline("organyl arms are incomplete or overlap")
    for group in groups:
        if not any(mol.GetAtomWithIdx(i).IsInRing() for i in group) and len(group) not in ROOTS:
            return decline("alkyl parent root is not available in the configured vocabulary")
    owned = {cidx}
    for group in groups:
        if owned & group:
            return decline("organyl arm overlaps the cation parent")
        owned.update(group)
    heavy = {atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1}
    if owned != heavy:
        return decline("not every heavy atom belongs to the parent center or a proved arm")
    for atom in mol.GetAtoms():
        if atom.GetIsotope() or int(atom.GetChiralTag()):
            return decline("isotopic or stereochemical annotations are outside this bounded grammar")
        if atom.GetIdx() == cidx:
            continue
        if atom.GetAtomicNum() != 6:
            return decline("noncarbon skeletal atoms are outside this route")
    try:
        Chem.SanitizeMol(Chem.Mol(mol))
    except Exception:
        return decline("graph fails ordinary-valence sanitization")
    return ChalcogenCationRecognition("supported", "complete ordinary-valence chalcogen hydron cation",
                                      cidx, element, tuple(roots), groups, expected_h)


def _arm_component(mol, Chem, root: int, group: frozenset[int], center: int) -> str | None:
    if len(group) == 6 and all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in group):
        # mapped_branch owns all six original ring atoms and uses the source
        # preferred simple prefix for a single free valence on benzene.
        from .component_mapping import mapped_branch
        text, mapped = mapped_branch(mol, Chem, root, center)
        return text if text == "phenyl" and frozenset(mapped) == group else None
    path = [root]
    previous = center
    while True:
        choices = [neighbor.GetIdx() for neighbor in mol.GetAtomWithIdx(path[-1]).GetNeighbors()
                   if neighbor.GetIdx() != previous and neighbor.GetIdx() in group]
        if not choices:
            break
        if len(choices) != 1:
            return None
        previous, next_idx = path[-1], choices[0]
        if next_idx in path:
            return None
        path.append(next_idx)
    if frozenset(path) != group or len(path) not in ROOTS:
        return None
    text = parent_chain_name(mol, tuple(path), as_substituent=True)
    if not text:
        return None
    from .substituent_multiplication import multiply_substituent
    return text if multiply_substituent(text, 1, nature="simple") else None


def chalcogen_cation_proposal(mol, Chem=None) -> ChalcogenCationProposal | None:
    """Return a mapped component proposal, or ``None`` outside the proof scope."""
    if Chem is None:
        from rdkit import Chem as RDKitChem
        Chem = RDKitChem
    recognition = recognize_chalcogen_cation(mol, Chem)
    if recognition.status != "supported":
        return None
    texts = []
    for root, group in zip(recognition.carbon_roots, recognition.carbon_groups, strict=True):
        text = _arm_component(mol, Chem, root, group, recognition.center)
        if not text:
            return None
        texts.append(text)
    prefix = mononuclear_prefix_sequence(PrefixUnit(text, "simple") for text in texts) if texts else None
    if texts and prefix is None:
        return None
    name = (prefix.text if prefix else "") + _PARENT[recognition.element]
    heavy = frozenset(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)
    return ChalcogenCationProposal(name, recognition.center, recognition.element,
                                   heavy, ((recognition.center, ""),),
                                   recognition.carbon_roots, recognition.carbon_groups,
                                   tuple(texts))


def chalcogen_cation_decision(mol, Chem=None) -> MethodDecision | None:
    proposal = chalcogen_cation_proposal(mol, Chem)
    return proposal.as_method_decision() if proposal is not None else None
