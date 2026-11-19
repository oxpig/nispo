"""Typed original-graph evidence for bounded subordinate amide admission.

This module does not render a prefix or choose a parent. It validates the
source-class relation needed before the complete-functional-parent operation
may carry a recognized lower-class nitramide, nitrous amide, or sulfinamide.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


AdmissionStatus = Literal["not-required", "admitted", "barrier"]
_N_OXO_KINDS = frozenset({"nitramide", "nitrous-amide"})
_SUBORDINATE_KINDS = _N_OXO_KINDS | {"sulfinamide"}


@dataclass(frozen=True, slots=True)
class OriginalSubordinateRole:
    """One exact original group core admitted as a subordinate prefix role."""

    kind: str
    center: int
    atoms: tuple[int, ...]
    owner: int = -1
    ligand: int = -1
    rule: str = ""


@dataclass(frozen=True, slots=True)
class AdmissionComparison:
    """One principal-to-subordinate source comparison on original groups."""

    principal_center: int
    subordinate_center: int
    status: str
    rule: str
    reason: str


@dataclass(frozen=True, slots=True)
class ParentAdmissionEvidence:
    """Status-bearing evidence for one complete-parent admission request."""

    status: AdmissionStatus
    principal_centers: tuple[int, ...] = ()
    subordinate_roles: tuple[OriginalSubordinateRole, ...] = ()
    comparisons: tuple[AdmissionComparison, ...] = ()
    reason: str = ""
    rule: str = "P-41/P-44.1.1/P-66.1.1.4.2/P-67.1.2.6.3"


def parent_admission_evidence(
    mol,
    Chem,
    inventory,
    groups,
    *,
    nitrogen_oxo_higher_class: bool = False,
) -> ParentAdmissionEvidence:
    """Validate only source-proved N-oxo/sulfinamide subordinate groups.

    A bounded subordinate site must be a member of the complete original
    inventory, have an exact source-shaped core recognition, and compare
    strictly junior to every selected original principal group. Missing or
    unknown facts are barriers. If no target site is subordinate, this helper
    does not impose a new restriction on unrelated chemistry.
    """
    groups = tuple(groups or ())
    inventory_groups = tuple(getattr(inventory, "groups", ()))
    principal = tuple(getattr(inventory, "principal", ()))
    relevant = tuple(group for group in inventory_groups
                     if group.kind in _SUBORDINATE_KINDS and group not in groups)
    if not relevant:
        return ParentAdmissionEvidence(
            "not-required",
            tuple(sorted(group.center for group in principal)),
            reason="No recognized N-oxo or sulfinamide subordinate group is present.",
        )

    principal_centers = tuple(sorted(group.center for group in principal))
    if (not getattr(inventory, "supported", False)
            or not principal
            or groups != principal
            or len(principal_centers) != len(principal)):
        return ParentAdmissionEvidence(
            "barrier", principal_centers,
            reason="Admission requires the complete supported original principal-group set.",
        )
    role_result = _recognized_subordinate_roles(mol, Chem, inventory, relevant)
    if role_result is None:
        return ParentAdmissionEvidence(
            "barrier", principal_centers,
            reason="A candidate subordinate group lacks an exact original-graph role proof.",
        )

    from .group_priority import compare_functional_group_priority

    source_facts = tuple(getattr(inventory, "source_group_facts", ()))
    if len(source_facts) != len(inventory_groups):
        return ParentAdmissionEvidence(
            "barrier", principal_centers, role_result,
            reason="The original inventory has no complete aligned semantic-priority sidecar.",
        )
    fact_by_group_id = {id(group): facts
                        for group, facts in zip(inventory_groups, source_facts)}

    comparisons = []
    for subordinate in relevant:
        subordinate_facts = fact_by_group_id.get(id(subordinate))
        if subordinate_facts is None:
            return ParentAdmissionEvidence(
                "barrier", principal_centers, role_result,
                reason="A subordinate site's source priority facts are unavailable.",
            )
        for selected in principal:
            principal_facts = fact_by_group_id.get(id(selected))
            result = compare_functional_group_priority(principal_facts, subordinate_facts)
            comparisons.append(AdmissionComparison(
                selected.center, subordinate.center, result.status,
                result.rule, result.reason,
            ))
            if result.status != "senior":
                return ParentAdmissionEvidence(
                    "barrier", principal_centers, role_result, tuple(comparisons),
                    reason=("Every selected principal group must be proved senior to "
                            "each admitted subordinate amide group."),
                )

    # The legacy flag remains a request for the same typed operation; it is
    # never itself the evidence that makes the candidate eligible.
    return ParentAdmissionEvidence(
        "admitted", principal_centers, role_result, tuple(comparisons),
        reason="Exact subordinate cores are source-ranked below the complete principal set.",
    )


def _recognized_subordinate_roles(mol, Chem, inventory, groups):
    """Return exact mapped site roles only when every inventory site matches."""
    by_kind = {}
    for group in groups:
        by_kind.setdefault(group.kind, []).append(group)
    roles = []

    if any(kind in _N_OXO_KINDS for kind in by_kind):
        from .functional_inventory import nitrogen_oxoamide_sites

        sites = nitrogen_oxoamide_sites(mol)
        matched = set()
        for kind, n0, n1, oxygens in sites:
            if kind not in by_kind:
                continue
            core = frozenset((n0, n1, *oxygens))
            candidates = [group for group in by_kind[kind]
                          if group.center == n1 and group.owner == n0
                          and group.ligand == n0 and frozenset(group.atoms) == core]
            if len(candidates) != 1:
                return None
            group = candidates[0]
            matched.add(id(group))
            roles.append(OriginalSubordinateRole(
                kind, group.center, tuple(sorted(core)), group.owner, group.ligand,
                "P-67.1.2.6.3 exact nitric/nitrous amide core",
            ))
        if len(matched) != sum(len(by_kind[kind]) for kind in _N_OXO_KINDS if kind in by_kind):
            return None

    if "sulfinamide" in by_kind:
        from .sulfinamide_components import recognize_sulfinamide_sites

        sites = recognize_sulfinamide_sites(mol, Chem)
        matched = set()
        for site in sites:
            core = frozenset(site.atoms)
            candidates = [group for group in by_kind["sulfinamide"]
                          if group.center == site.center and group.owner == site.owner
                          and group.ligand == site.ligand
                          and frozenset(group.atoms) == core]
            if len(candidates) != 1:
                return None
            group = candidates[0]
            # Recheck the source-defined S=O, S-C, S-N roles against original
            # atoms rather than relying on inventory labels alone.
            sulfur = mol.GetAtomWithIdx(site.center)
            oxygen = mol.GetAtomWithIdx(site.oxo_oxygen)
            if (sulfur.GetAtomicNum() != 16 or oxygen.GetAtomicNum() != 8
                    or mol.GetBondBetweenAtoms(site.center, site.oxo_oxygen).GetBondTypeAsDouble() != 2.0
                    or mol.GetAtomWithIdx(site.owner).GetAtomicNum() != 6
                    or mol.GetAtomWithIdx(site.ligand).GetAtomicNum() != 7):
                return None
            matched.add(id(group))
            roles.append(OriginalSubordinateRole(
                "sulfinamide", site.center, tuple(sorted(core)), site.owner,
                site.ligand, "P-66.1.1.2 exact C-S(=O)-N core",
            ))
        if len(matched) != len(by_kind["sulfinamide"]):
            return None

    return tuple(sorted(roles, key=lambda role: (role.kind, role.center, role.atoms)))
