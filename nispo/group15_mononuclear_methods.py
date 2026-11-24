"""Mapped bounded P-68.3 Group 15 mononuclear hydrides and heterones.

This route names neutral, one-center carbon organyl derivatives of phosphane,
arsane, stibane, and bismuthane.  A single terminal =O, =S, =Se, or =Te unit
may be expressed by the corresponding substitutive suffix.  Unsupported
charge, topology, annotations, or suffix competition declines without
dropping atoms.
"""
from __future__ import annotations

from dataclasses import dataclass

from .component_mapping import MappedComponent, decorate_component, mapped_branch, mapped_general_branch
from .formatting import attach_suffix
from .naming_methods import MethodDecision
from .prefix_sequences import PrefixUnit, mononuclear_prefix_sequence


RULE = "P-41/P-43/P-44/P-45/P-68.3.2.1-.3/P-68.3.3/P-16/P-82"
_PARENTS = {15: "phosphane", 33: "arsane", 51: "stibane", 83: "bismuthane"}
_SUFFIX = {8: "one", 16: "thione", 34: "selone", 52: "tellone"}


@dataclass(frozen=True, slots=True)
class Group15Recognition:
    status: str
    reason: str
    center: int | None = None
    element: int | None = None
    carbon_roots: tuple[int, ...] = ()
    carbon_groups: tuple[frozenset[int], ...] = ()
    suffix_atom: int | None = None
    suffix_element: int | None = None
    lambda_valence: int | None = None
    complex_ligands: bool = False


def _single(mol, left: int, right: int) -> bool:
    bond = mol.GetBondBetweenAtoms(left, right)
    return bond is not None and bond.GetBondTypeAsDouble() == 1.0


def _carbon_organyl(mol, root: int, center: int):
    """Prove one complete saturated carbon tree or an unsubstituted phenyl."""
    if not _single(mol, root, center):
        return None
    atoms, pending = {root}, [root]
    while pending:
        current = pending.pop()
        atom = mol.GetAtomWithIdx(current)
        for neighbor in atom.GetNeighbors():
            idx = neighbor.GetIdx()
            if idx == center or neighbor.GetAtomicNum() == 1:
                continue
            if neighbor.GetAtomicNum() != 6:
                return None
            if idx not in atoms:
                atoms.add(idx)
                pending.append(idx)
    external = {(a.GetIdx(), n.GetIdx()) for a in mol.GetAtoms()
                if a.GetIdx() in atoms for n in a.GetNeighbors()
                if n.GetIdx() not in atoms}
    if external != {(root, center)}:
        return None
    group = frozenset(atoms)
    ring_atoms = {i for i in group if mol.GetAtomWithIdx(i).IsInRing()}
    if ring_atoms:
        rings = [set(r) for r in mol.GetRingInfo().AtomRings() if set(r) & group]
        if (group != ring_atoms or len(group) != 6 or len(rings) != 1
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
        edge_count = sum(1 for b in mol.GetBonds()
                         if b.GetBeginAtomIdx() in group and b.GetEndAtomIdx() in group)
        if edge_count != len(group) - 1:
            return None
    return group


def _complex_carbon_organyl(mol, root, center):
    from .branch_rules import collect_subtree

    atoms = frozenset(collect_subtree(mol, root, center))
    if center in atoms:
        return None
    cuts = {(i, a.GetIdx()) for i in atoms for a in mol.GetAtomWithIdx(i).GetNeighbors()
            if a.GetIdx() not in atoms}
    return atoms if cuts == {(root, center)} else None


def recognize_group15_mononuclear(mol) -> Group15Recognition:
    """Classify the whole graph for this bounded suffix-mode operation."""
    centers = [a for a in mol.GetAtoms() if a.GetAtomicNum() in _PARENTS]
    if not centers:
        return Group15Recognition("not-applicable", "no Group 15 parent center")
    if len(centers) != 1:
        return Group15Recognition("deferred", "multiple Group 15 centers are outside this route")
    center = centers[0]
    cidx, element = center.GetIdx(), center.GetAtomicNum()
    if center.IsInRing() or center.GetFormalCharge() or center.GetNumRadicalElectrons():
        return Group15Recognition("deferred", "cyclic, charged, or radical center", cidx, element)
    if center.GetIsotope() or int(center.GetChiralTag()):
        return Group15Recognition("deferred", "central isotope or stereochemistry is unproved", cidx, element)
    if mol.GetNumAtoms() > 96 or mol.GetStereoGroups():
        return Group15Recognition("deferred", "graph or enhanced-stereo group exceeds bounded scope", cidx, element)
    triorganyl_oxide = element == 15 and center.GetDegree() == 4 and any(
        bond.GetBondTypeAsDouble() == 2 and bond.GetOtherAtom(center).GetAtomicNum() == 8
        for bond in center.GetBonds()
    )
    if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        return Group15Recognition("deferred", "radical atom is outside the neutral route", cidx, element)
    charged = any(a.GetFormalCharge() for a in mol.GetAtoms())
    if charged and not triorganyl_oxide:
        return Group15Recognition("deferred", "charged atom is outside the neutral route", cidx, element)
    if any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
        return Group15Recognition("deferred", "explicit hydrogen atoms are not part of this component grammar", cidx, element)

    suffixes = []
    roots, groups = [], []
    complex_ligands = False
    for neighbor in center.GetNeighbors():
        idx, number = neighbor.GetIdx(), neighbor.GetAtomicNum()
        bond = mol.GetBondBetweenAtoms(cidx, idx)
        if number == 1:
            if neighbor.GetIsotope():
                return Group15Recognition("deferred", "isotopic hydrogen is outside this operation", cidx, element)
            continue
        if number in _SUFFIX and bond.GetBondTypeAsDouble() == 2.0:
            if (neighbor.GetDegree() != 1 or neighbor.IsInRing() or neighbor.GetFormalCharge()
                    or neighbor.GetNumRadicalElectrons() or neighbor.GetIsotope()
                    or neighbor.GetNumImplicitHs() or neighbor.GetNumExplicitHs()
                    or int(neighbor.GetChiralTag())):
                return Group15Recognition("deferred", "suffix atom is not an unmodified terminal chalcogen", cidx, element)
            suffixes.append((idx, number))
            continue
        if number == 6 and bond.GetBondTypeAsDouble() == 1.0:
            group = _carbon_organyl(mol, idx, cidx)
            if group is None and triorganyl_oxide:
                group = _complex_carbon_organyl(mol, idx, cidx)
                complex_ligands = True
            if group is None:
                return Group15Recognition("deferred", "organyl is not a saturated carbon tree or plain phenyl", cidx, element)
            if any(mol.GetAtomWithIdx(i).GetIsotope()
                   or int(mol.GetAtomWithIdx(i).GetChiralTag()) for i in group):
                return Group15Recognition("deferred", "organyl isotope or stereochemistry is not proved", cidx, element)
            roots.append(idx)
            groups.append(group)
            continue
        # P-68.3.2.3.1 also covers =NH, but this operation deliberately limits
        # itself to oxy/chalcogen suffixes until N-substitution mapping is proved.
        return Group15Recognition("deferred", "direct ligand class is outside the typed oxy/chalcogen grammar", cidx, element)
    if len(suffixes) > 1:
        return Group15Recognition("deferred", "multiple typed principal suffix atoms are outside this operation", cidx, element)
    if not roots:
        return Group15Recognition("deferred", "requires at least one carbon organyl", cidx, element)
    if any(a.GetIsotope() for a in mol.GetAtoms()):
        return Group15Recognition("deferred", "isotope grammar outside the bounded parent/suffix operation", cidx, element)
    if complex_ligands or charged:
        from .extended_inventory import selection_inventory
        from .neutral_charge_components import validate_neutral_charge_role_proofs

        inventory = selection_inventory(mol)
        if (not inventory.supported or any(g.seniority[0] <= 16 for g in inventory.groups)
                or not validate_neutral_charge_role_proofs(mol, inventory.charge_role_proofs)):
            return Group15Recognition("deferred", "competing suffix or unproved charge", cidx, element)

    valence = int(center.GetTotalValence())
    if valence not in {3, 5}:
        return Group15Recognition("deferred", "central bonding number is outside proved 3/5 valence forms", cidx, element)
    # P-21.1.2.1 gives λ5 names for PH5, AsH5, and SbH5 generally;
    # P-68.3 applies generalized parent-hydride selection to substitutive
    # pnictogen names. P-68.3.3 independently shows λ5-bismuthane.
    lambda_valence = 5 if valence == 5 else None

    owned = {cidx}
    suffix_atom, suffix_element = (suffixes[0] if suffixes else (None, None))
    if suffix_atom is not None:
        owned.add(suffix_atom)
    for group in groups:
        if owned & group:
            return Group15Recognition("deferred", "organyl branches overlap", cidx, element)
        owned.update(group)
    heavy = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
    if owned != heavy:
        return Group15Recognition("deferred", "not every heavy atom belongs to a mapped parent component", cidx, element)
    return Group15Recognition("supported", "complete neutral mononuclear Group 15 parent-hydride graph",
        cidx, element, tuple(roots), tuple(groups), suffix_atom, suffix_element, lambda_valence,
        complex_ligands)


def _organyl_prefix_text(mol, Chem, recognition):
    terms = {}
    branches = []
    resolver = mapped_general_branch if recognition.complex_ligands else mapped_branch
    for root, atoms in zip(recognition.carbon_roots, recognition.carbon_groups, strict=True):
        text, owned = resolver(mol, Chem, root, recognition.center)
        if not text and recognition.complex_ligands:
            from .component_mapping import _general_branch_stereo_is_cited
            from .parent_rules import _simple_or_branch_prefix, _strip_outer_parentheses

            # The recognition step has already proved this single P-C cut.
            # Reuse the same complete organyl writer as the oxide class name.
            text = _strip_outer_parentheses(_simple_or_branch_prefix(mol, root, recognition.center))
            if text and _general_branch_stereo_is_cited(mol, Chem, atoms, text):
                owned = atoms
        if not text or frozenset(owned) != atoms:
            return None
        path = all(sum(n.GetIdx() in atoms for n in mol.GetAtomWithIdx(i).GetNeighbors()) <= 2
                   for i in atoms)
        simple_path = path and (not recognition.complex_ligands or text.isalpha())
        nature = "simple" if text in {"phenyl", "tert-butyl"} or simple_path else "compound"
        if text in terms and terms[text] != nature:
            return None
        terms[text] = nature
        branches.append((text, atoms, nature))
    sequence = mononuclear_prefix_sequence(
        PrefixUnit(text, nature) for text, _atoms, nature in branches
    )
    if sequence is None:
        return None
    return sequence.text, tuple(branches)


def group15_mononuclear_decision(mol, Chem) -> MethodDecision | None:
    """Return one complete mapped substitutive P-68.3 decision or ``None``."""
    recognition = recognize_group15_mononuclear(mol)
    if recognition.status != "supported":
        return None
    prefix_result = _organyl_prefix_text(mol, Chem, recognition)
    if prefix_result is None:
        return None
    prefix_text, branches = prefix_result
    parent = _PARENTS[recognition.element]
    if recognition.lambda_valence:
        parent = f"lambda{recognition.lambda_valence}-{parent}"
    if recognition.suffix_element is not None:
        parent = attach_suffix(parent, _SUFFIX[recognition.suffix_element])
    core = frozenset((recognition.center,) if recognition.suffix_atom is None
                     else (recognition.center, recognition.suffix_atom))
    embedded = frozenset().union(*(atoms for _, atoms, _ in branches))
    component = MappedComponent(parent, tuple((i, "") for i in sorted(core)), core,
        embedded_atoms=embedded, systematic_parent=True)
    core_text = decorate_component(mol, Chem, component)
    if not core_text:
        return None
    # The λ-convention is a locant-like parent prefix; P-68's examples place
    # a hyphen between the organyl-prefix sequence and λ⁵.
    name = prefix_text + ("-" if prefix_text and recognition.lambda_valence else "") + core_text
    all_heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    roles = [("group15-parent-suffix", core, component.atom_locants)]
    roles.extend((f"organyl-{i+1}", atoms, ()) for i, (_, atoms, _) in enumerate(branches))
    role_atoms = [set(atoms) for _, atoms, _ in roles]
    if (any(role_atoms[i] & role_atoms[j] for i in range(len(role_atoms))
            for j in range(i + 1, len(role_atoms)))
            or frozenset().union(*(frozenset(atoms) for atoms in role_atoms)) != all_heavy):
        return None
    return MethodDecision(name=name, method="substitutive",
        principal_centers=frozenset((recognition.center,)), atoms=all_heavy,
        rule=RULE, atom_locants=component.atom_locants, component_roles=tuple(roles))
