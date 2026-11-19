"""Source-typed P-41/P-42/P-43 priority evidence.

This additive API is intentionally separate from ``CharacteristicGroup.seniority``.
It compares only facts justified by the cited Blue Book criteria and returns an
explicit unknown result when an applicable earlier fact is missing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Status = Literal["senior", "junior", "tie", "unknown"]

# Tuple positions for complete P-43.1 vectors: replacement heteroatom order is
# specified in (a); the latter two vectors use the oxygen/chalcogen atom order
# stated in (b) and (c), respectively.
P43_REPLACEMENT_ELEMENT_ORDER = ("O", "S", "Se", "Te", "=NH", "=NNH2")
P43_PEROXY_ELEMENT_ORDER = ("O", "S", "Se", "Te")
P43_ACIDIC_HYDROXY_ELEMENT_ORDER = ("O", "S", "Se", "Te")


@dataclass(frozen=True, slots=True)
class PriorityComparison:
    status: Status
    rule: str
    reason: str


@dataclass(frozen=True, slots=True)
class PriorityFrontierAnalysis:
    """Conservative result of pairwise source comparisons.

    ``items`` contains the undominated items, except that contradictory
    directed comparisons return the whole input as a barrier. Unknown
    comparisons are reported only when they still occur within the resulting
    frontier, so an unknown lower-priority pair cannot invalidate a winner
    already proved senior to both.
    """

    items: tuple
    status: Literal["selected", "tie", "barrier"]
    unknown_pairs: tuple[tuple[int, int], ...] = ()
    cycle: bool = False
    reason: str = ""


@dataclass(frozen=True, slots=True)
class AcidPriorityFacts:
    """Original-structure facts for one P-42 acid profile.

    ``category`` is one of the source sections ``7a`` through ``7e``. Fields
    not used by that section are left unset. Missing applicable fields are
    barriers; they never mean zero or a tie.
    """

    category: str | None
    p42_1_family: str | None = None
    p42_1_form: str | None = None
    p42_2_family: str | None = None
    central_element: str | None = None
    central_atom_count: int | None = None
    homopolyacid: bool | None = None
    contiguous_central_atoms: bool | None = None
    acidic_hydroxyl_count: int | None = None
    oxidation_number: int | None = None
    source: str = ""


@dataclass(frozen=True, slots=True)
class FunctionalReplacementFacts:
    """Mapped P-43.1 facts, compared only within one known suffix family.

    Each tuple gives counts in the source's stated element order. ``None`` on
    an applicable field means that the graph operation did not establish the
    complete count vector.
    """

    suffix_family: str
    replaceable_atom_counts: tuple[int, ...] | None = None
    peroxy_group_counts: tuple[int, ...] | None = None
    acidic_hydroxy_atom_counts: tuple[int, ...] | None = None
    source: str = "P-43.1"


@dataclass(frozen=True, slots=True)
class FunctionalGroupPriorityFacts:
    """P-41 class plus only source-required within-class evidence."""

    class_name: str | None
    class_rank: int | None
    corresponding_acid: AcidPriorityFacts | None = None
    replacement: FunctionalReplacementFacts | None = None
    leaving_group: str | None = None
    source: str = "P-41"
    # P-63.1.5 directly orders hydroxy compounds and their sulfur analogues.
    # Kept after legacy fields so positional construction stays compatible.
    hydroxy_analogue: str | None = None
    # Isotope annotation on the functional atom or its attached H is a
    # separate P-82 selection issue; this flag is deliberately narrower than
    # whole-molecule isotope annotation.
    functional_labelled: bool = False


_P42_CATEGORY_ORDER = {name: i for i, name in enumerate(("7a", "7b", "7c", "7d", "7e"))}
_P42_1_FAMILIES = (
    "carboxylic", "sulfonic", "sulfinic", "selenonic", "seleninic",
    "telluronic", "tellurinic",
)
_P42_2_FAMILIES = (
    "polycarbonic", "tetracarbonic", "tricarbonic", "dicarbonic",
    "carbonic", "cyanic",
)
_P42_3_CENTRAL_ORDER = ("N", "P", "As", "Sb", "B")
_P42_4_CENTRAL_ORDER = ("P", "As", "Sb", "Si", "B", "S", "Se", "Te")
_P42_5_CENTRAL_ORDER = ("N", "F", "Cl", "Br", "I")
_HALOGEN_ORDER = ("F", "Cl", "Br", "I")
_PSEUDOHALOGEN_ORDER = ("N3", "CN", "NC", "NCO", "NCS", "NCSe", "NCTe", "CNO")
_CLASS_10_LEAVING_GROUP_ORDER = _HALOGEN_ORDER + _PSEUDOHALOGEN_ORDER

# These classes have source-backed class order and a bounded, implemented
# within-class profile. Class 17 is limited to exact O/S identities from
# P-63.1.5; unsupported recognized kinds remain pairwise unknown.
SOURCE_FRONTIER_CLASS_RANKS = frozenset({7, 10, 11, 12, 14, 17})


def _cmp_values(left, right, *, maximize=False, rule="") -> PriorityComparison:
    if left is None or right is None:
        return PriorityComparison("unknown", rule, "An applicable source criterion is unproved.")
    if left == right:
        return PriorityComparison("tie", rule, "The source criterion ties.")
    left_senior = left > right if maximize else left < right
    return PriorityComparison("senior" if left_senior else "junior", rule,
                              "First differing source criterion.")


def _cmp_ordered_values(left, right, order, rule):
    if left is None or right is None:
        return PriorityComparison("unknown", rule, "An applicable source-ordered identity is unproved.")
    if left not in order or right not in order:
        return PriorityComparison("unknown", rule, "A source order for this identity is not established.")
    return _cmp_values(order.index(left), order.index(right), rule=rule)


def compare_replacement_facts(left: FunctionalReplacementFacts | None,
                              right: FunctionalReplacementFacts | None) -> PriorityComparison:
    """Compare complete P-43.1 replacement vectors for the same suffix family."""
    rule = "P-43.1"
    if left is None or right is None or left.suffix_family != right.suffix_family:
        return PriorityComparison("unknown", rule,
                                  "Comparable functional-replacement scope is not proved.")
    for field, clause in (
        ("replaceable_atom_counts", "P-43.1(a)"),
        ("peroxy_group_counts", "P-43.1(b)"),
        ("acidic_hydroxy_atom_counts", "P-43.1(c)"),
    ):
        a, b = getattr(left, field), getattr(right, field)
        if a is None or b is None:
            return PriorityComparison("unknown", clause,
                                      "An applicable replacement count vector is unproved.")
        expected_length = {"replaceable_atom_counts": len(P43_REPLACEMENT_ELEMENT_ORDER),
                           "peroxy_group_counts": len(P43_PEROXY_ELEMENT_ORDER),
                           "acidic_hydroxy_atom_counts": len(P43_ACIDIC_HYDROXY_ELEMENT_ORDER)}[field]
        if len(a) != expected_length or len(b) != expected_length:
            return PriorityComparison("unknown", clause,
                                      "A replacement vector does not match the typed P-43.1 element order.")
        if a != b:
            return _cmp_values(a, b, maximize=True, rule=clause)
    return PriorityComparison("tie", rule, "All supplied P-43.1 replacement criteria tie.")


def compare_acid_priority(left: AcidPriorityFacts | None,
                          right: AcidPriorityFacts | None) -> PriorityComparison:
    """Compare acid profiles by the applicable P-42 source criteria."""
    if left is None or right is None or left.category not in _P42_CATEGORY_ORDER \
            or right.category not in _P42_CATEGORY_ORDER:
        return PriorityComparison("unknown", "P-42", "A source-classified acid profile is missing.")
    category = _cmp_values(_P42_CATEGORY_ORDER[left.category],
                           _P42_CATEGORY_ORDER[right.category], rule="P-42")
    if category.status != "tie":
        return category
    category_name = left.category
    if category_name == "7a":
        family = _cmp_ordered_values(left.p42_1_family, right.p42_1_family,
                                     _P42_1_FAMILIES, "P-42.1")
        if family.status != "tie":
            return family
        # The source places peroxy, imidic, and hydrazonic forms with each
        # corresponding acid; finer chalcogen replacement order is P-43 data.
        known_forms = ("ordinary", "peroxy", "imidic", "hydrazonic")
        form = _cmp_ordered_values(left.p42_1_form, right.p42_1_form,
                                   known_forms, "P-42.1/P-43")
        if form.status != "tie":
            return form
        if left.p42_1_form != "ordinary":
            return PriorityComparison("unknown", "P-43",
                                      "Modified suffix comparison requires typed replacement facts.")
        return PriorityComparison("tie", "P-42.1", "The mapped ordinary acid families tie.")
    if category_name == "7b":
        return _cmp_ordered_values(left.p42_2_family, right.p42_2_family,
                                   _P42_2_FAMILIES, "P-42.2")
    if category_name in {"7c", "7d"}:
        central_order = (_P42_3_CENTRAL_ORDER if category_name == "7c"
                         else _P42_4_CENTRAL_ORDER)
        central = _cmp_ordered_values(left.central_element, right.central_element,
                                      central_order, f"P-42.{3 if category_name == '7c' else 4}(a)")
        if central.status != "tie":
            return central
        for field, clause, maximize in (
            ("central_atom_count", f"P-42.{3 if category_name == '7c' else 4}(b)", True),
            ("homopolyacid", f"P-42.{3 if category_name == '7c' else 4}(c)", True),
            ("contiguous_central_atoms", f"P-42.{3 if category_name == '7c' else 4}(d)", True),
            ("acidic_hydroxyl_count", f"P-42.{3 if category_name == '7c' else 4}(e)", True),
            ("oxidation_number", f"P-42.{3 if category_name == '7c' else 4}(f)", True),
        ):
            comparison = _cmp_values(getattr(left, field), getattr(right, field),
                                     maximize=maximize, rule=clause)
            if comparison.status != "tie":
                return comparison
        return PriorityComparison("tie", f"P-42.{3 if category_name == '7c' else 4}",
                                  "All mapped subclass criteria tie.")
    if category_name == "7e":
        central = _cmp_ordered_values(left.central_element, right.central_element,
                                      _P42_5_CENTRAL_ORDER, "P-42.5(a)")
        if central.status != "tie":
            return central
        return _cmp_values(left.oxidation_number, right.oxidation_number,
                           maximize=True, rule="P-42.5(b)")
    return PriorityComparison("unknown", "P-42", "Unsupported acid subclass.")


def compare_functional_group_priority(
    left: FunctionalGroupPriorityFacts | None,
    right: FunctionalGroupPriorityFacts | None,
) -> PriorityComparison:
    """Compare source-typed group evidence using P-41 then any stated suborder."""
    if left is None or right is None or left.class_rank is None or right.class_rank is None:
        return PriorityComparison("unknown", "P-41", "A P-41 class fact is missing.")
    class_cmp = _cmp_values(left.class_rank, right.class_rank, rule="P-41 Table 4.1")
    if class_cmp.status != "tie":
        return class_cmp
    rank = left.class_rank
    if rank in {7, 11, 12}:
        acid_cmp = compare_acid_priority(left.corresponding_acid, right.corresponding_acid)
        if acid_cmp.status == "unknown" and acid_cmp.rule == "P-43" \
                and left.replacement is not None and right.replacement is not None:
            return compare_replacement_facts(left.replacement, right.replacement)
        if acid_cmp.status != "tie":
            return acid_cmp
        replacements = left.replacement, right.replacement
        if any(value is not None for value in replacements):
            return compare_replacement_facts(*replacements)
        return PriorityComparison("tie", acid_cmp.rule,
                                  "The mapped corresponding-acid priority facts tie.")
    if rank == 10:
        acid_cmp = compare_acid_priority(left.corresponding_acid, right.corresponding_acid)
        if acid_cmp.status != "tie":
            return acid_cmp
        return _cmp_ordered_values(left.leaving_group, right.leaving_group,
                                   _CLASS_10_LEAVING_GROUP_ORDER,
                                   "P-41 class 10 leaving-group order")
    if rank == 17:
        if left.hydroxy_analogue not in {"O", "S"} or right.hydroxy_analogue not in {"O", "S"}:
            return PriorityComparison("unknown", "P-63.1.5",
                                      "A complete recognized hydroxy-analogue identity is missing.")
        if left.hydroxy_analogue == right.hydroxy_analogue:
            return PriorityComparison("tie", "P-41/P-63.1.5",
                                      "Alcohol and phenol have no separate rank; identical analogue atoms tie.")
        if left.functional_labelled or right.functional_labelled:
            return PriorityComparison("unknown", "P-82.2.2.2/P-63.1.5",
                                      "Mixed O/S suffix choice with functional isotope labels is outside the proved interaction scope.")
        return _cmp_ordered_values(left.hydroxy_analogue, right.hydroxy_analogue,
                                   ("O", "S"), "P-63.1.5")
    # P-41 states the class order but supplies no acid-derived suborder here.
    return PriorityComparison("tie", "P-41 Table 4.1",
                              "The source gives no additional within-class priority here.")


def priority_frontier_analysis(items, facts) -> PriorityFrontierAnalysis:
    """Keep every source-undominated item and expose unresolved comparisons.

    ``facts(item)`` must return ``FunctionalGroupPriorityFacts`` or ``None``.
    Tied and incomparable candidates remain together; the helper never uses
    labels, atom indices, or text as a secondary key. A contradictory source
    fact set is treated as a barrier and preserves all candidates.
    """
    items = tuple(items)
    values = tuple(facts(item) for item in items)
    dominated = set()
    seniority_edges = {index: set() for index in range(len(items))}
    unknown_pairs = []
    for left_index in range(len(items)):
        for right_index in range(left_index + 1, len(items)):
            result = compare_functional_group_priority(values[left_index], values[right_index])
            if result.status == "senior":
                dominated.add(right_index)
                seniority_edges[left_index].add(right_index)
            elif result.status == "junior":
                dominated.add(left_index)
                seniority_edges[right_index].add(left_index)
            elif result.status == "unknown":
                unknown_pairs.append((left_index, right_index))

    # Typed callers should supply mutually consistent facts. Still guard
    # against accidental contradictory evidence rather than silently
    # discarding every item in a seniority cycle.
    visiting, visited = set(), set()

    def cyclic(node):
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        if any(cyclic(child) for child in seniority_edges[node]):
            return True
        visiting.remove(node)
        visited.add(node)
        return False

    if any(cyclic(node) for node in seniority_edges):
        return PriorityFrontierAnalysis(items, "barrier", tuple(unknown_pairs), True,
                                        "Contradictory directed priority facts form a cycle.")

    frontier_indices = tuple(index for index in range(len(items)) if index not in dominated)
    frontier_unknowns = tuple(pair for pair in unknown_pairs
                               if pair[0] in frontier_indices and pair[1] in frontier_indices)
    frontier = tuple(items[index] for index in frontier_indices)
    if frontier_unknowns:
        return PriorityFrontierAnalysis(frontier, "barrier", frontier_unknowns, False,
                                        "An unresolved source comparison remains in the frontier.")
    status = "selected" if len(frontier) == 1 else "tie"
    return PriorityFrontierAnalysis(frontier, status, (), False,
                                    "The source frontier is selected or tied without unknown comparisons.")


def priority_frontier(items, facts):
    """Compatibility projection returning only the conservative item tuple."""
    return priority_frontier_analysis(items, facts).items


_CLASS_RANKS = {
    "carboxylate": 4, "carbamate-anion": 4, "sulfonate": 4, "sulfinate": 4,
    "phosphonate": 4, "phosphinate": 4, "phosphate": 4, "sulfur-oxoacid-ester-anion": 4,
    "ammonium": 6,
    "acid": 7, "sulfonic-acid": 7, "sulfinic-acid": 7,
    "carbonic-acid": 7, "phosphonic-acid": 7, "phosphinic-acid": 7,
    "phosphoric-acid": 7,
    "anhydride": 8, "ester": 9, "carbamate": 9, "carbonate": 9,
    "phosphate-acid-ester": 9, "sulfur-oxoacid-ester": 9, "acid-halide": 10, "sulfonyl-halide": 10,
    "amide": 11, "sulfonamide": 11, "sulfonimidamide": 11, "sulfinamide": 11, "sulfamide": 11,
    "selenonamide": 11, "seleninamide": 11,
    "telluronamide": 11, "tellurinamide": 11,
    "urea": 11, "amidine": 11, "amidoxime": 11, "guanidine": 11, "cyanamide": 11,
    "nitramide": 11, "nitrous-amide": 11,
    "hydrazide": 12, "cyanohydrazide": 12, "nitrile": 14, "thiosemicarbazone": 11,
    "aldehyde": 15, "ketone": 16, "lactone": 16, "lactam": 16,
    "pseudoketone": 16, "alcohol": 17, "thiol": 17,
    "peroxol": 18, "amine": 19, "imine": 20, "oxime": 20,
    "oxime-ether": 20,
}


def _acid_for_kind(kind: str, mol, group) -> AcidPriorityFacts | None:
    source_center = (group.owner if kind in {"nitramide", "nitrous-amide"}
                     and group.owner >= 0 else group.center)
    center = mol.GetAtomWithIdx(source_center)
    double_oxygen_count = sum(
        neighbor.GetAtomicNum() == 8
        and mol.GetBondBetweenAtoms(source_center, neighbor.GetIdx()).GetBondTypeAsDouble() == 2
        for neighbor in center.GetNeighbors()
    )
    if kind in {"acid", "amide", "acid-halide", "hydrazide"}:
        if center.GetAtomicNum() == 6 and double_oxygen_count == 1:
            return AcidPriorityFacts("7a", p42_1_family="carboxylic",
                                     p42_1_form="ordinary", source="mapped C(=O) acid correspondence")
    if kind in {"sulfonic-acid", "sulfonamide", "sulfonyl-halide", "hydrazide"}:
        if center.GetAtomicNum() == 16 and double_oxygen_count == 2:
            return AcidPriorityFacts("7a", p42_1_family="sulfonic",
                                     p42_1_form="ordinary", source="mapped S(O)2 acid correspondence")
    if kind in {"sulfinic-acid", "sulfinamide", "hydrazide"}:
        if center.GetAtomicNum() == 16 and double_oxygen_count == 1:
            return AcidPriorityFacts("7a", p42_1_family="sulfinic",
                                     p42_1_form="ordinary", source="mapped S(O) acid correspondence")
    if kind == "sulfamide" and center.GetAtomicNum() == 16 and double_oxygen_count == 2:
        return AcidPriorityFacts("7d", central_element="S", central_atom_count=1,
                                 homopolyacid=False, contiguous_central_atoms=1,
                                 acidic_hydroxyl_count=2, oxidation_number=6,
                                 source="P-67.1.2.6.1 sulfuric diamide")
    if kind in {"carbonic-acid", "urea"}:
        nitrogen_count = sum(neighbor.GetAtomicNum() == 7 for neighbor in center.GetNeighbors())
        if center.GetAtomicNum() == 6 and double_oxygen_count == 1 and nitrogen_count == 2:
            return AcidPriorityFacts("7b", p42_2_family="carbonic",
                                     source="mapped carbonic-acid correspondence")
        if kind == "carbonic-acid" and center.GetAtomicNum() == 6 and double_oxygen_count == 1:
            return AcidPriorityFacts("7b", p42_2_family="carbonic",
                                     source="mapped carbonic-acid correspondence")
    if kind == "thiosemicarbazone" and center.GetAtomicNum() == 6:
        return AcidPriorityFacts("7b", p42_2_family="carbonic",
                                 source="P-68.3.1.2.5 hydrazinecarbothioamide")
    if kind in {'amidine', 'amidoxime'}:
        return AcidPriorityFacts('7a', p42_1_family='carboxylic',
                                 p42_1_form='imidic', source='P-66.4.1 carboximidic acid')
    if kind == 'sulfonimidamide':
        return AcidPriorityFacts('7a', p42_1_family='sulfonic',
                                 p42_1_form='imidic', source='P-66.4.1 sulfonimidic acid')
    if kind in {'guanidine', 'cyanamide', 'cyanohydrazide'}:
        return AcidPriorityFacts('7b', p42_2_family='carbonic' if kind == 'guanidine' else 'cyanic',
                                 source='P-66.1.6/P-66.4.1 corresponding acid')
    if kind in {"nitramide", "nitrous-amide"}:
        terminal_oxygen_count = sum(neighbor.GetAtomicNum() == 8
                                    for neighbor in center.GetNeighbors())
        expected_count = 2 if kind == "nitramide" else 1
        oxidation = 5 if terminal_oxygen_count == 2 else 3 if terminal_oxygen_count == 1 else None
        if center.GetAtomicNum() == 7 and terminal_oxygen_count == expected_count:
            return AcidPriorityFacts("7e", central_element="N", oxidation_number=oxidation,
                                     source="mapped P-42.5 nitric/nitrous-acid profile")
    if kind in {"phosphonic-acid", "phosphinic-acid", "phosphoric-acid"}:
        profile = _mononuclear_phosphorus_acid_profile(kind, mol, group)
        if profile is not None:
            return profile
    # Polyphosphorus and other phosphorus acid profiles need their full
    # P-42.3/.4 structural facts; a scalar legacy class rank cannot supply them.
    return None


def _mononuclear_phosphorus_acid_profile(kind, mol, group):
    """Prove a mononuclear P-42 acid profile, including exact organyl substitution.

    P-67.1.1.2 permits replacement of central P-H atoms by organyl groups. The
    corresponding-acid oxidation number is therefore computed from the exact
    unsubstituted parent hydrogen count, not the derivative's current P-C/P-H
    partition.
    """
    phosphorus = mol.GetAtomWithIdx(group.center)
    if (phosphorus.GetAtomicNum() != 15 or phosphorus.GetFormalCharge()
            or phosphorus.GetNumRadicalElectrons()):
        return None
    neighbors = tuple(phosphorus.GetNeighbors())
    if any(atom.GetAtomicNum() == 15 for atom in neighbors):
        return None
    p_hydrogens = phosphorus.GetTotalNumHs(includeNeighbors=True)
    organyl_count = 0
    oxo = []
    acidic_oh = []
    p_oxygen_bond_order = 0
    for atom in neighbors:
        bond = mol.GetBondBetweenAtoms(group.center, atom.GetIdx())
        order = bond.GetBondTypeAsDouble()
        if atom.GetAtomicNum() == 8:
            p_oxygen_bond_order += int(order)
            if order == 2 and atom.GetDegree() == 1 and not atom.GetFormalCharge():
                oxo.append(atom)
            elif (order == 1 and atom.GetDegree() == 1
                  and atom.GetTotalNumHs(includeNeighbors=True) == 1
                  and not atom.GetFormalCharge()):
                acidic_oh.append(atom)
            elif order != 1:
                return None
        elif atom.GetAtomicNum() == 6 and order == 1:
            organyl_count += 1
        elif atom.GetAtomicNum() != 1 or order != 1:
            return None
    if len(oxo) != 1 or len(acidic_oh) not in {1, 2, 3}:
        return None
    if sum(atom.GetAtomicNum() == 8 for atom in neighbors) != len(oxo) + len(acidic_oh):
        return None
    if kind == "phosphonic-acid":
        parent_hydrogens = 1
        if len(acidic_oh) != 2:
            return None
        category = "7c"
    elif kind == "phosphinic-acid":
        parent_hydrogens = 2
        if len(acidic_oh) != 1:
            return None
        category = "7c"
    else:
        parent_hydrogens = 0
        if len(acidic_oh) != 3:
            return None
        category = "7d"
    if p_hydrogens + organyl_count != parent_hydrogens:
        return None
    corresponding_acid_oxidation = p_oxygen_bond_order - parent_hydrogens
    return AcidPriorityFacts(
        category,
        central_element="P",
        central_atom_count=1,
        homopolyacid=True,
        contiguous_central_atoms=True,
        acidic_hydroxyl_count=len(acidic_oh),
        oxidation_number=corresponding_acid_oxidation,
        source="exact mononuclear P-42.3/.4 corresponding-acid graph; P-67.1.1.2 organyl substitution",
    )


def functional_group_priority_facts(group, mol) -> FunctionalGroupPriorityFacts | None:
    """Build facts only for a bounded set of recognized original-graph groups.

    The function intentionally accepts an inventory-like object to avoid a
    dependency on the shared inventory module. It does not mutate the graph
    and does not decide whether an inventory should admit the group.
    """
    kind = getattr(group, "kind", None)
    class_rank = _CLASS_RANKS.get(kind)
    if class_rank is None:
        return None
    acid = _acid_for_kind(kind, mol, group)
    analogue = None
    functional_labelled = False
    if kind in {"alcohol", "thiol"}:
        analogue, functional_labelled = _hydroxy_analogue_identity(group, mol)
    leaving = None
    if kind in {"acid-halide", "sulfonyl-halide"}:
        attached = [neighbor for neighbor in mol.GetAtomWithIdx(group.center).GetNeighbors()
                    if neighbor.GetAtomicNum() in {9, 17, 35, 53}]
        if len(attached) == 1:
            leaving = {9: "F", 17: "Cl", 35: "Br", 53: "I"}[attached[0].GetAtomicNum()]
    replacement = None
    if kind in {'amidine', 'amidoxime', 'sulfonimidamide', 'guanidine', 'urea'}:
        family = {'amidine': 'carboxylic', 'amidoxime': 'carboxylic',
                  'sulfonimidamide': 'sulfonic'}.get(kind, 'carbonic')
        replacement = FunctionalReplacementFacts(
            family, (int(kind in {'urea', 'sulfonimidamide'}), 0, 0, 0, int(kind != 'urea'), 0),
            (0, 0, 0, 0), (0, 0, 0, 0))
    return FunctionalGroupPriorityFacts(
        class_name=kind,
        class_rank=class_rank,
        corresponding_acid=acid,
        replacement=replacement,
        leaving_group=leaving,
        source="P-41 Table 4.1; P-42 where applicable; P-63.1.5 for class 17",
        hydroxy_analogue=analogue,
        functional_labelled=functional_labelled,
    )


def _hydroxy_analogue_identity(group, mol):
    """Return an exact recognized class-17 identity and local isotope flag.

    This annotates an existing inventory group only; it does not recognize
    new sites. Isotopes on the parent skeleton do not suppress the chemical
    O/S identity. Labels on the functional atom or its attached H are kept
    visible for the separately governed P-82 suffix-choice stage.
    """
    from rdkit import Chem

    kind = getattr(group, "kind", None)
    center = getattr(group, "center", -1)
    if center < 0 or center >= mol.GetNumAtoms():
        return None, False
    atom = mol.GetAtomWithIdx(center)
    atomic_number = 8 if kind == "alcohol" else 16 if kind == "thiol" else None
    if atomic_number is None:
        return None, False
    if kind == "alcohol":
        if len(getattr(group, "atoms", ())) != 2:
            return None, False
        site_atoms = tuple(mol.GetAtomWithIdx(idx) for idx in group.atoms)
        carbon = next((candidate for candidate in site_atoms
                       if candidate.GetAtomicNum() == 6), None)
        oxygen = next((candidate for candidate in site_atoms
                       if candidate.GetAtomicNum() == 8), None)
        if carbon is None or oxygen is None:
            return None, False
        if carbon.GetIdx() != center:
            return None, False
        hydrogen_neighbors = [neighbor for neighbor in oxygen.GetNeighbors()
                              if neighbor.GetAtomicNum() == 1]
        if (carbon.GetAtomicNum() != 6 or oxygen.GetAtomicNum() != 8
                or mol.GetBondBetweenAtoms(carbon.GetIdx(), oxygen.GetIdx()) is None
                or mol.GetBondBetweenAtoms(carbon.GetIdx(), oxygen.GetIdx()).GetBondTypeAsDouble() != 1
                or oxygen.GetDegree() != 1 + len(hydrogen_neighbors)
                or oxygen.GetFormalCharge()
                or oxygen.GetNumRadicalElectrons()
                or oxygen.GetTotalNumHs(includeNeighbors=True) != 1):
            return None, False
        functional_atom = oxygen
    else:
        if (atom.GetAtomicNum() != 16 or atom.GetFormalCharge()
                or atom.GetNumRadicalElectrons()
                or atom.GetTotalNumHs(includeNeighbors=True) != 1):
            return None, False
        carbon_neighbors = [neighbor for neighbor in atom.GetNeighbors()
                            if neighbor.GetAtomicNum() == 6]
        hydrogen_neighbors = [neighbor for neighbor in atom.GetNeighbors()
                              if neighbor.GetAtomicNum() == 1]
        if (len(carbon_neighbors) != 1 or atom.GetDegree() != 1 + len(hydrogen_neighbors)
                or getattr(group, "owner", -1) != carbon_neighbors[0].GetIdx()
                or tuple(getattr(group, "atoms", ())) != (atom.GetIdx(),)):
            return None, False
        bond = mol.GetBondBetweenAtoms(atom.GetIdx(), carbon_neighbors[0].GetIdx())
        if bond is None or bond.GetBondTypeAsDouble() != 1:
            return None, False
        functional_atom = atom
    if any(bond.GetBondType() != Chem.BondType.SINGLE for bond in functional_atom.GetBonds()):
        return None, False
    labelled = bool(functional_atom.GetIsotope())
    labelled = labelled or any(neighbor.GetAtomicNum() == 1 and neighbor.GetIsotope()
                               for neighbor in functional_atom.GetNeighbors())
    return ("O" if atomic_number == 8 else "S"), labelled
