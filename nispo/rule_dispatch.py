from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RulePlan:
    rule: object
    acyclic_only: bool
    min_ring_count: int | None
    min_ring_system_ring_count: int | None
    requires_exact_ring_system: bool
    requires_nonring_positive_sulfur: bool
    requires_nonring_positive_nitrogen: bool
    requires_charged_ring_atom: bool
    excludes_charged_ring_atom: bool
    requires_imino_skeleton_signal: bool
    requires_functional_skeleton_signal: bool
    requires_acyclic_preference_signal: bool
    requires_kekulized_polycycle_signal: bool
    requires_generic_bicyclic_topology: bool
    requires_generic_polycyclic_topology: bool
    requires_spiro_topology: bool
    defer_to_exocyclic_charged_fused_parent: bool


def compile_rule_plan(rule) -> RulePlan:
    return RulePlan(
        rule=rule,
        acyclic_only=bool(getattr(rule, "acyclic_only", False)),
        min_ring_count=getattr(rule, "min_ring_count", None),
        min_ring_system_ring_count=getattr(rule, "min_ring_system_ring_count", None),
        requires_exact_ring_system=bool(getattr(rule, "requires_exact_ring_system", False)),
        requires_nonring_positive_sulfur=bool(getattr(rule, "requires_nonring_positive_sulfur", False)),
        requires_nonring_positive_nitrogen=bool(getattr(rule, "requires_nonring_positive_nitrogen", False)),
        requires_charged_ring_atom=bool(getattr(rule, "requires_charged_ring_atom", False)),
        excludes_charged_ring_atom=bool(getattr(rule, "excludes_charged_ring_atom", False)),
        requires_imino_skeleton_signal=bool(getattr(rule, "requires_imino_skeleton_signal", False)),
        requires_functional_skeleton_signal=bool(getattr(rule, "requires_functional_skeleton_signal", False)),
        requires_acyclic_preference_signal=bool(getattr(rule, "requires_acyclic_preference_signal", False)),
        requires_kekulized_polycycle_signal=bool(getattr(rule, "requires_kekulized_polycycle_signal", False)),
        requires_generic_bicyclic_topology=bool(getattr(rule, "requires_generic_bicyclic_topology", False)),
        requires_generic_polycyclic_topology=bool(getattr(rule, "requires_generic_polycyclic_topology", False)),
        requires_spiro_topology=bool(getattr(rule, "requires_spiro_topology", False)),
        defer_to_exocyclic_charged_fused_parent=bool(
            getattr(rule, "defer_to_exocyclic_charged_fused_parent", False)
        ),
    )


def rule_applies(plan: RulePlan, ctx) -> bool:
    """Return whether a rule family is structurally eligible for this molecule.

    Rule implementations still validate their own chemistry before yielding a
    name. This dispatcher is only a cheap coarse filter, inspired by OPSIN's
    split between token/resource classification and structure construction.
    """

    ring_count = ctx.mol.GetRingInfo().NumRings()
    if plan.acyclic_only and ring_count:
        return False
    if plan.min_ring_count is not None and ring_count < plan.min_ring_count:
        return False
    if plan.min_ring_system_ring_count is not None:
        if ctx.features["max_ring_system_ring_count"] < plan.min_ring_system_ring_count:
            return False
    if plan.requires_exact_ring_system:
        return bool(ctx.features["all_atoms_in_one_ring_system"])
    if plan.requires_nonring_positive_sulfur:
        if not ctx.profile.has_nonring_positive_sulfur:
            return False
    if plan.requires_nonring_positive_nitrogen:
        if not ctx.profile.has_nonring_positive_nitrogen:
            return False
    if plan.requires_charged_ring_atom:
        if not ctx.profile.has_charged_ring_atom:
            return False
    if plan.excludes_charged_ring_atom:
        if ctx.profile.has_charged_ring_atom:
            return False
    if plan.requires_imino_skeleton_signal:
        if not ctx.profile.has_imino_skeleton_signal:
            return False
    if plan.requires_functional_skeleton_signal:
        if not ctx.profile.has_functional_skeleton_signal:
            return False
    if plan.requires_acyclic_preference_signal:
        if not ctx.profile.may_prefer_acyclic_skeleton:
            return False
    if plan.requires_kekulized_polycycle_signal:
        if not ctx.profile.has_kekulized_polycycle_signal:
            return False
    if plan.requires_generic_bicyclic_topology:
        if not ctx.has_generic_bicyclic_topology:
            return False
    if plan.requires_generic_polycyclic_topology:
        if not ctx.has_generic_polycyclic_topology:
            return False
    if plan.requires_spiro_topology:
        if not ctx.has_spiro_topology:
            return False
    if plan.defer_to_exocyclic_charged_fused_parent:
        if _has_charged_fused_thioxo_imino_system(ctx):
            return False
    return True


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
