from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass


Gate = Callable[[object], bool]
CandidateProducer = Callable[[object], Iterable[object]]
_PREFILTER_MIN_RULE_COUNT = 6


@dataclass(frozen=True, slots=True)
class RuleSpec:
    """Explicit rule registration record used by the naming engine."""

    family: str
    priority: int
    gates: tuple[Gate, ...]
    producer: CandidateProducer

    def applies(self, ctx) -> bool:
        for gate in self.gates:
            if not gate(ctx):
                return False
        return True

    def candidates(self, ctx) -> Iterable[object]:
        return self.producer(ctx)


def compile_rules_for_context(
    ctx,
    rules: Iterable[RuleSpec],
    *,
    telemetry=None,
) -> tuple[RuleSpec, ...]:
    """Return rules whose declared gates are possible for this molecule.

    This is a conservative graph-fact prefilter. It only skips a rule when an
    existing gate's metadata proves the gate cannot pass; normal gate evaluation
    still remains authoritative.
    """

    rules = rules if isinstance(rules, tuple) else tuple(rules)
    if len(rules) < _PREFILTER_MIN_RULE_COUNT:
        if telemetry is not None:
            route = ctx.route_plan.primary_route
            telemetry.compiled_rule_input_count[route] += len(rules)
            telemetry.compiled_rule_output_count[route] += len(rules)
        return rules
    cache = getattr(ctx, "cache", None)
    cache_key = id(rules)
    cached = None
    if cache is not None:
        cached_entry = cache.compiled_rules.get(cache_key)
        if cached_entry is not None:
            cached_rules, cached = cached_entry
            if cached_rules is not rules:
                cached = None
    if cached is not None:
        compiled = cached
        skipped = len(rules) - len(compiled)
        if telemetry is not None:
            route = ctx.route_plan.primary_route
            telemetry.compiled_rule_input_count[route] += len(rules)
            telemetry.compiled_rule_output_count[route] += len(compiled)
            telemetry.compiled_rule_skip_count[route] += skipped
            telemetry.compiled_rule_cache_hit_count[route] += 1
        return compiled

    compiled = []
    skipped = 0
    facts = ctx.facts
    for rule in rules:
        if _rule_impossible_for_facts(rule, facts):
            skipped += 1
            continue
        compiled.append(rule)
    if telemetry is not None:
        route = ctx.route_plan.primary_route
        telemetry.compiled_rule_input_count[route] += len(rules)
        telemetry.compiled_rule_output_count[route] += len(compiled)
        telemetry.compiled_rule_skip_count[route] += skipped
    result = tuple(compiled)
    if cache is not None:
        cache.compiled_rules[cache_key] = (rules, result)
    return result


def _rule_impossible_for_facts(rule: RuleSpec, facts) -> bool:
    for gate in rule.gates:
        min_ring_count_value = getattr(gate, "_nispo_min_ring_count", None)
        if min_ring_count_value is not None and facts.ring_count < min_ring_count_value:
            return True
        min_ring_system_ring_count_value = getattr(gate, "_nispo_min_ring_system_ring_count", None)
        if (
            min_ring_system_ring_count_value is not None
            and facts.max_ring_system_ring_count < min_ring_system_ring_count_value
        ):
            return True
        min_nonring_heavy_atom_count_value = getattr(gate, "_nispo_min_nonring_heavy_atom_count", None)
        if (
            min_nonring_heavy_atom_count_value is not None
            and facts.nonring_heavy_atom_count < min_nonring_heavy_atom_count_value
        ):
            return True
        if facts.ring_count == 0 and getattr(gate, "_nispo_requires_ring", False):
            return True
        if facts.ring_count != 0 and getattr(gate, "_nispo_requires_acyclic", False):
            return True
        required_fact = getattr(gate, "_nispo_requires_fact", "")
        if required_fact and not bool(getattr(facts, required_fact, False)):
            return True
        required_all_facts = getattr(gate, "_nispo_requires_all_facts", ())
        if required_all_facts and not all(bool(getattr(facts, fact, False)) for fact in required_all_facts):
            return True
        required_any_facts = getattr(gate, "_nispo_requires_any_facts", ())
        if required_any_facts and not any(bool(getattr(facts, fact, False)) for fact in required_any_facts):
            return True
    return False


def acyclic(ctx) -> bool:
    return ctx.mol.GetRingInfo().NumRings() == 0


acyclic._nispo_requires_acyclic = True


def min_ring_count(count: int) -> Gate:
    def gate(ctx) -> bool:
        return ctx.mol.GetRingInfo().NumRings() >= count

    gate._nispo_min_ring_count = count
    gate._nispo_requires_ring = count > 0
    return gate


def min_ring_system_ring_count(count: int) -> Gate:
    def gate(ctx) -> bool:
        return ctx.features["max_ring_system_ring_count"] >= count

    gate._nispo_min_ring_system_ring_count = count
    gate._nispo_requires_ring = count > 0
    return gate


def min_nonring_heavy_atom_count(count: int) -> Gate:
    def gate(ctx) -> bool:
        return ctx.facts.nonring_heavy_atom_count >= count

    gate._nispo_min_nonring_heavy_atom_count = count
    return gate


def exact_ring_system(ctx) -> bool:
    return bool(ctx.features["all_atoms_in_one_ring_system"])


exact_ring_system._nispo_requires_ring = True
exact_ring_system._nispo_requires_fact = "all_atoms_in_one_ring_system"


def nonring_positive_sulfur(ctx) -> bool:
    return ctx.profile.has_nonring_positive_sulfur


nonring_positive_sulfur._nispo_requires_all_facts = ("has_sulfur", "has_formal_charge")


def nonring_positive_nitrogen(ctx) -> bool:
    return ctx.profile.has_nonring_positive_nitrogen


nonring_positive_nitrogen._nispo_requires_all_facts = ("has_nitrogen", "has_formal_charge")


def nonring_positive_phosphorus(ctx) -> bool:
    return ctx.profile.has_nonring_positive_phosphorus


nonring_positive_phosphorus._nispo_requires_all_facts = ("has_phosphorus", "has_formal_charge")


def nonring_neutral_nitrogen(ctx) -> bool:
    return ctx.profile.has_nonring_neutral_nitrogen


nonring_neutral_nitrogen._nispo_requires_fact = "has_nitrogen"


def nonring_neutral_phosphorus(ctx) -> bool:
    return ctx.profile.has_nonring_neutral_phosphorus


nonring_neutral_phosphorus._nispo_requires_fact = "has_phosphorus"


def neutral_phosphorus(ctx) -> bool:
    return ctx.profile.has_neutral_phosphorus


neutral_phosphorus._nispo_requires_fact = "has_phosphorus"


def nonring_neutral_pnictogen(ctx) -> bool:
    return ctx.profile.has_nonring_neutral_pnictogen


nonring_neutral_pnictogen._nispo_requires_any_facts = ("has_nitrogen", "has_phosphorus")


def nonring_neutral_group14(ctx) -> bool:
    return ctx.profile.has_nonring_neutral_group14


nonring_neutral_group14._nispo_requires_fact = "has_group14"


def charged_ring_atom(ctx) -> bool:
    return ctx.profile.has_charged_ring_atom


charged_ring_atom._nispo_requires_ring = True
charged_ring_atom._nispo_requires_fact = "has_charged_ring_atom"


def no_charged_ring_atom(ctx) -> bool:
    return not ctx.profile.has_charged_ring_atom


def formal_charge_signal(ctx) -> bool:
    return ctx.profile.has_formal_charge


formal_charge_signal._nispo_requires_fact = "has_formal_charge"


def boron_signal(ctx) -> bool:
    return ctx.profile.has_boron


boron_signal._nispo_requires_fact = "has_boron"


def iodine_signal(ctx) -> bool:
    return ctx.profile.has_iodine


iodine_signal._nispo_requires_fact = "has_iodine"


def sulfur_signal(ctx) -> bool:
    return ctx.profile.has_sulfur


sulfur_signal._nispo_requires_fact = "has_sulfur"


def oxygen_signal(ctx) -> bool:
    return ctx.facts.has_oxygen


oxygen_signal._nispo_requires_fact = "has_oxygen"


def nitrogen_signal(ctx) -> bool:
    return ctx.facts.has_nitrogen


nitrogen_signal._nispo_requires_fact = "has_nitrogen"


def nitrogen_oxygen_signal(ctx) -> bool:
    return ctx.facts.has_nitrogen and ctx.facts.has_oxygen


nitrogen_oxygen_signal._nispo_requires_all_facts = ("has_nitrogen", "has_oxygen")


def sulfur_oxygen_signal(ctx) -> bool:
    return ctx.facts.has_sulfur and ctx.facts.has_oxygen


sulfur_oxygen_signal._nispo_requires_all_facts = ("has_sulfur", "has_oxygen")


def sulfur_nitrogen_oxygen_signal(ctx) -> bool:
    return ctx.facts.has_sulfur and ctx.facts.has_nitrogen and ctx.facts.has_oxygen


sulfur_nitrogen_oxygen_signal._nispo_requires_all_facts = (
    "has_sulfur",
    "has_nitrogen",
    "has_oxygen",
)


def oxygen_oxygen_single_bond_signal(ctx) -> bool:
    return ctx.profile.has_oxygen_oxygen_single_bond


oxygen_oxygen_single_bond_signal._nispo_requires_fact = "has_oxygen_oxygen_single_bond"


def oxirane_dione_signal(ctx) -> bool:
    return ctx.profile.has_oxirane_dione_signal


oxirane_dione_signal._nispo_requires_fact = "has_oxirane_dione_signal"


def nitrile_oxide_signal(ctx) -> bool:
    return ctx.profile.has_nitrile_oxide_signal


nitrile_oxide_signal._nispo_requires_fact = "has_nitrile_oxide_signal"


def oxime_sulfonate_nitrile_signal(ctx) -> bool:
    return ctx.profile.has_oxime_sulfonate_nitrile_signal


oxime_sulfonate_nitrile_signal._nispo_requires_fact = "has_oxime_sulfonate_nitrile_signal"


def terminal_carboxylate_signal(ctx) -> bool:
    return ctx.profile.has_terminal_carboxylate_signal


terminal_carboxylate_signal._nispo_requires_fact = "has_terminal_carboxylate_signal"


def amide_acyl_signal(ctx) -> bool:
    return ctx.facts.has_amide_acyl_motif


amide_acyl_signal._nispo_requires_fact = "has_amide_acyl_motif"


def dithiocarbamate_retained_signal(ctx) -> bool:
    return ctx.profile.has_dithiocarbamate_signal


dithiocarbamate_retained_signal._nispo_requires_fact = "has_dithiocarbamate_signal"


def small_inorganic_retained_signal(ctx) -> bool:
    return (
        ctx.profile.has_formal_charge
        or ctx.mol.GetNumAtoms() <= 4
        or ctx.profile.has_boron
        or ctx.profile.has_iodine
        or (ctx.profile.has_sulfur and ctx.profile.has_nonring_multiple_bond)
    )


small_inorganic_retained_signal._nispo_requires_fact = "has_small_inorganic_retained_signal"


def sulfur_imino_signal(ctx) -> bool:
    return ctx.profile.has_sulfur and ctx.profile.has_imino_skeleton_signal


sulfur_imino_signal._nispo_requires_all_facts = ("has_sulfur", "has_imino_skeleton_signal")


def imino_skeleton_signal(ctx) -> bool:
    return ctx.profile.has_imino_skeleton_signal


imino_skeleton_signal._nispo_requires_fact = "has_imino_skeleton_signal"


def functional_skeleton_signal(ctx) -> bool:
    return ctx.profile.has_functional_skeleton_signal


functional_skeleton_signal._nispo_requires_fact = "has_functional_skeleton_signal"


def sulfur_oxo_retained_parent_signal(ctx) -> bool:
    return ctx.profile.has_sulfur_oxo_retained_parent_signal


sulfur_oxo_retained_parent_signal._nispo_requires_fact = "has_sulfur_oxo_retained_parent_signal"


def terminal_n_oxide_signal(ctx) -> bool:
    return ctx.profile.has_terminal_n_oxide


terminal_n_oxide_signal._nispo_requires_fact = "has_terminal_n_oxide"


def sulfinate_retained_parent_signal(ctx) -> bool:
    return ctx.profile.has_functional_skeleton_signal or ctx.profile.has_sulfur_oxo_retained_parent_signal


sulfinate_retained_parent_signal._nispo_requires_any_facts = (
    "has_functional_skeleton_signal",
    "has_sulfur_oxo_retained_parent_signal",
)


def dithiocarbonate_signal(ctx) -> bool:
    mol = ctx.mol
    Chem = ctx.Chem
    for carbon in mol.GetAtoms():
        cidx = carbon.GetIdx()
        if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0 or carbon.IsInRing():
            continue
        has_terminal_thioxo = False
        has_oxygen_owner = False
        sulfur_owner_count = 0
        terminal_sulfido_count = 0
        for nbr in carbon.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(cidx, nbr.GetIdx())
            if bond is None:
                continue
            if (
                nbr.GetSymbol() == "S"
                and bond.GetBondType() == Chem.BondType.DOUBLE
                and nbr.GetFormalCharge() == 0
                and _heavy_degree_excluding(nbr, cidx) == 0
            ):
                has_terminal_thioxo = True
            elif (
                nbr.GetSymbol() == "O"
                and bond.GetBondType() == Chem.BondType.SINGLE
                and nbr.GetFormalCharge() == 0
                and _heavy_degree_excluding(nbr, cidx) == 1
            ):
                has_oxygen_owner = True
            elif (
                nbr.GetSymbol() == "S"
                and bond.GetBondType() == Chem.BondType.SINGLE
                and nbr.GetFormalCharge() <= 0
            ):
                heavy_degree = _heavy_degree_excluding(nbr, cidx)
                if heavy_degree == 1:
                    sulfur_owner_count += 1
                elif heavy_degree == 0 and nbr.GetFormalCharge() < 0:
                    terminal_sulfido_count += 1
        if has_terminal_thioxo and (
            (has_oxygen_owner and sulfur_owner_count >= 1)
            or (not has_oxygen_owner and sulfur_owner_count >= 1 and terminal_sulfido_count >= 1)
        ):
            return True
    return False


def _heavy_degree_excluding(atom, excluded_idx: int) -> int:
    return sum(
        1
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != excluded_idx and nbr.GetAtomicNum() > 1
    )


def exact_functional_retained_signal(ctx) -> bool:
    return (
        ctx.profile.has_functional_skeleton_signal
        or ctx.profile.has_charged_ring_atom
        or not ctx.features["all_atoms_in_one_ring_system"]
    )


def acyclic_preference_signal(ctx) -> bool:
    return ctx.profile.may_prefer_acyclic_skeleton


acyclic_preference_signal._nispo_requires_fact = "has_acyclic_preference_signal"


def chain_dominant_acyclic_parent_signal(ctx) -> bool:
    return ctx.facts.may_have_chain_dominant_acyclic_parent


chain_dominant_acyclic_parent_signal._nispo_requires_fact = "may_have_chain_dominant_acyclic_parent"


def conjugated_carbonyl_diene_signal(ctx) -> bool:
    return ctx.profile.has_conjugated_carbonyl_diene_signal


conjugated_carbonyl_diene_signal._nispo_requires_fact = "has_conjugated_carbonyl_diene_signal"


def kekulized_polycycle_signal(ctx) -> bool:
    return ctx.profile.has_kekulized_polycycle_signal


kekulized_polycycle_signal._nispo_requires_fact = "has_kekulized_polycycle_signal"


def generic_bicyclic_topology(ctx) -> bool:
    return ctx.has_generic_bicyclic_topology


generic_bicyclic_topology._nispo_requires_ring = True
generic_bicyclic_topology._nispo_requires_fact = "has_generic_bicyclic_topology"


def generic_polycyclic_topology(ctx) -> bool:
    return ctx.has_generic_polycyclic_topology


generic_polycyclic_topology._nispo_requires_ring = True
generic_polycyclic_topology._nispo_requires_fact = "has_generic_polycyclic_topology"


def spiro_topology(ctx) -> bool:
    return ctx.has_spiro_topology


spiro_topology._nispo_requires_ring = True
spiro_topology._nispo_requires_fact = "has_spiro_topology"


def component_spiro_topology(ctx) -> bool:
    return ctx.has_component_spiro_topology


component_spiro_topology._nispo_requires_ring = True
component_spiro_topology._nispo_requires_fact = "has_component_spiro_topology"


def no_exocyclic_charged_fused_parent(ctx) -> bool:
    return not _has_charged_fused_thioxo_imino_system(ctx)


def _has_charged_fused_thioxo_imino_system(ctx) -> bool:
    mol = ctx.mol
    Chem = ctx.Chem
    if mol.GetRingInfo().NumRings() < 2 or not ctx.profile.has_charged_ring_atom:
        return False
    has_thioxo = False
    has_terminal_imino = False
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "C" or not atom.IsInRing():
            continue
        cidx = atom.GetIdx()
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nbr.IsInRing():
                continue
            bond = mol.GetBondBetweenAtoms(cidx, nidx)
            if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
                continue
            if nbr.GetSymbol() == "S":
                has_thioxo = True
            elif nbr.GetSymbol() == "C":
                terminal_n = [
                    nnbr.GetIdx()
                    for nnbr in nbr.GetNeighbors()
                    if nnbr.GetIdx() != cidx
                    and nnbr.GetSymbol() == "N"
                    and mol.GetBondBetweenAtoms(nidx, nnbr.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
                ]
                if len(terminal_n) == 1:
                    has_terminal_imino = True
    return has_thioxo and has_terminal_imino
