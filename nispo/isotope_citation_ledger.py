"""Typed secondary isotope citations across named components.

Primary atom ownership remains the responsibility of
``isotope_component_coverage``. This module records additional, source-required
citations of that same original isotope atom in another component. It never
looks for nuclide tokens in rendered text and never mutates the input graph.

The initial rule/role table is deliberately closed to two direct Blue Book
constructions: ester bridge oxygen under P-82.6.4, and the phosphorus ester
bridge oxygen cited again on the anion under P-83.1.2.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from .isotope_component_coverage import (
    IsotopeComponentRecord,
    IsotopeCoverageResult,
    validate_isotope_component_coverage,
)


class IsotopeCitationRule(str, Enum):
    P82_6_4_ESTER_OXYGEN = "P-82.6.4 ester oxygen citation"
    P83_1_2_2_PHOSPHORUS_ESTER = "P-83.1.2.2 phosphorus ester citation"


class IsotopeCitationRole(str, Enum):
    ORGANYL_BRIDGE_OXYGEN = "organyl bridge-oxygen qualifier"
    ESTER_ACID_OXYGEN_COUNT = "acid-ending ester oxygen count"
    PHOSPHORUS_ANION_OXYGEN_DESCRIPTOR = "phosphorus-anion oxygen descriptor"


@dataclass(frozen=True, slots=True)
class EsterBridgeOxygenCitationProof:
    """Original-graph proof for the P-82.6.4 ester bridge reference."""

    acyl_carbon: int
    bridge_oxygen: int
    carbonyl_oxygen: int
    organyl_carbon: int


@dataclass(frozen=True, slots=True)
class PhosphorusBridgeOxygenCitationProof:
    """Original-graph proof for the P-83.1.2.2 phosphorus ester reference."""

    phosphorus: int
    bridge_oxygen: int
    organyl_carbon: int
    phosphoryl_oxygen: int


CitationRoleProof = EsterBridgeOxygenCitationProof | PhosphorusBridgeOxygenCitationProof


_ALLOWED_ROLE_PAIRS = {
    IsotopeCitationRule.P82_6_4_ESTER_OXYGEN: frozenset({
        (IsotopeCitationRole.ORGANYL_BRIDGE_OXYGEN,
         IsotopeCitationRole.ESTER_ACID_OXYGEN_COUNT),
    }),
    IsotopeCitationRule.P83_1_2_2_PHOSPHORUS_ESTER: frozenset({
        (IsotopeCitationRole.ORGANYL_BRIDGE_OXYGEN,
         IsotopeCitationRole.PHOSPHORUS_ANION_OXYGEN_DESCRIPTOR),
    }),
}


@dataclass(frozen=True, slots=True)
class IsotopeCitationRequirement:
    """One source-derived obligation, built from a mapped role recognizer.

    This records why one isotope atom must be cited in a second component.
    It does not claim that the target component owns the atom.
    """

    atom_id: int
    source_component_id: str
    target_component_id: str
    rule: IsotopeCitationRule
    source_role: IsotopeCitationRole
    target_role: IsotopeCitationRole
    source_locant: str | None = None
    target_locant: str | None = None
    role_proof: CitationRoleProof | None = None


@dataclass(frozen=True, slots=True)
class IsotopeCitationReference:
    """Writer claim that one already-owned atom is cited at another site."""

    atom_id: int
    element: str
    mass: int
    source_component_id: str
    target_component_id: str
    rule: IsotopeCitationRule
    source_role: IsotopeCitationRole
    target_role: IsotopeCitationRole
    source_locant: str | None = None
    target_locant: str | None = None
    role_proof: CitationRoleProof | None = None


@dataclass(frozen=True, slots=True)
class IsotopeCitationLedger:
    """Primary coverage plus validated repeated citations, without text scans."""

    complete: bool
    errors: tuple[str, ...]
    primary_coverage: IsotopeCoverageResult
    required_references: tuple[IsotopeCitationRequirement, ...]
    references: tuple[IsotopeCitationReference, ...]
    citation_totals: tuple[IsotopeCitationTally, ...] = ()


@dataclass(frozen=True, slots=True)
class IsotopeCitationTally:
    """Typed primary and repeated-citation counts for one original atom."""

    atom_id: int
    primary_count: int
    secondary_count: int
    total_count: int


def _valid_atom_id(value: object, atom_count: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value < atom_count


def _valid_locant(value: object) -> bool:
    return value is None or isinstance(value, str) and bool(value)


def _valid_role_pair(rule, source_role, target_role) -> bool:
    return (isinstance(rule, IsotopeCitationRule)
            and isinstance(source_role, IsotopeCitationRole)
            and isinstance(target_role, IsotopeCitationRole)
            and (source_role, target_role) in _ALLOWED_ROLE_PAIRS.get(rule, ()))


def _bond(mol, first: int, second: int, order: float) -> bool:
    bond = mol.GetBondBetweenAtoms(first, second)
    return bond is not None and bond.GetBondTypeAsDouble() == order


def _valid_role_proof(mol, atom_id: int, rule, proof) -> bool:
    """Check one closed source role map directly on the unmodified graph."""
    try:
        if rule is IsotopeCitationRule.P82_6_4_ESTER_OXYGEN:
            if not isinstance(proof, EsterBridgeOxygenCitationProof):
                return False
            role_ids = (proof.acyl_carbon, proof.bridge_oxygen,
                        proof.carbonyl_oxygen, proof.organyl_carbon)
            if (len(set(role_ids)) != 4
                    or any(not _valid_atom_id(idx, mol.GetNumAtoms()) for idx in role_ids)
                    or proof.bridge_oxygen != atom_id):
                return False
            acyl, bridge, carbonyl, organyl = (mol.GetAtomWithIdx(idx) for idx in role_ids)
            return (
                acyl.GetAtomicNum() == 6 and bridge.GetAtomicNum() == 8
                and carbonyl.GetAtomicNum() == 8 and organyl.GetAtomicNum() == 6
                and bridge.GetDegree() == 2
                and {n.GetIdx() for n in bridge.GetNeighbors()}
                    == {proof.acyl_carbon, proof.organyl_carbon}
                and carbonyl.GetDegree() == 1
                and {n.GetIdx() for n in carbonyl.GetNeighbors()} == {proof.acyl_carbon}
                and _bond(mol, proof.acyl_carbon, proof.bridge_oxygen, 1.0)
                and _bond(mol, proof.bridge_oxygen, proof.organyl_carbon, 1.0)
                and _bond(mol, proof.acyl_carbon, proof.carbonyl_oxygen, 2.0)
            )
        if rule is IsotopeCitationRule.P83_1_2_2_PHOSPHORUS_ESTER:
            if not isinstance(proof, PhosphorusBridgeOxygenCitationProof):
                return False
            role_ids = (proof.phosphorus, proof.bridge_oxygen,
                        proof.organyl_carbon, proof.phosphoryl_oxygen)
            if (len(set(role_ids)) != 4
                    or any(not _valid_atom_id(idx, mol.GetNumAtoms()) for idx in role_ids)
                    or proof.bridge_oxygen != atom_id):
                return False
            phosphorus, bridge, organyl, phosphoryl = (
                mol.GetAtomWithIdx(idx) for idx in role_ids
            )
            return (
                phosphorus.GetAtomicNum() == 15 and bridge.GetAtomicNum() == 8
                and organyl.GetAtomicNum() == 6 and phosphoryl.GetAtomicNum() == 8
                and bridge.GetDegree() == 2
                and {n.GetIdx() for n in bridge.GetNeighbors()}
                    == {proof.phosphorus, proof.organyl_carbon}
                and phosphoryl.GetDegree() == 1
                and {n.GetIdx() for n in phosphoryl.GetNeighbors()} == {proof.phosphorus}
                and not phosphoryl.GetIsotope()
                and _bond(mol, proof.phosphorus, proof.bridge_oxygen, 1.0)
                and _bond(mol, proof.bridge_oxygen, proof.organyl_carbon, 1.0)
                and _bond(mol, proof.phosphorus, proof.phosphoryl_oxygen, 2.0)
            )
    except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
        return False
    return False


def validate_isotope_citation_ledger(
    mol,
    primary_components: Iterable[IsotopeComponentRecord],
    required_references: Iterable[IsotopeCitationRequirement],
    references: Iterable[IsotopeCitationReference],
) -> IsotopeCitationLedger:
    """Validate primary ownership and explicitly required secondary citations.

    Requirements must be supplied by a source-aware, original-atom-mapped
    producer. This validator checks their closed rule/role pair, atom identity,
    primary owner, component endpoints, locant facts, and exact fulfillment by
    a writer reference. It does not derive requirements from text or infer
    structural roles. Each isotope atom still has exactly one primary owner;
    multiple distinct requirements may refer to that atom only if the producer
    supplies separately justified source-role requirements.
    """
    try:
        atom_count = mol.GetNumAtoms()
        records = tuple(primary_components)
        requirements = tuple(required_references)
        supplied = tuple(references)
    except Exception:
        invalid = IsotopeCoverageResult(
            False, ("invalid graph or iterable inputs",), (), (), (),
        )
        return IsotopeCitationLedger(False, ("invalid graph or iterable inputs",),
                                     invalid, (), ())

    primary = validate_isotope_component_coverage(mol, records)
    errors = ["primary ledger: " + error for error in primary.errors]
    owner: dict[int, str] = {}
    primary_descriptor_by_atom = {}
    component_ids: set[str] = set()
    for record in records:
        if not isinstance(record, IsotopeComponentRecord):
            continue
        if isinstance(record.component_id, str) and record.component_id:
            component_ids.add(record.component_id)
        if isinstance(record.atom_ids, frozenset):
            for atom_id in record.atom_ids:
                if _valid_atom_id(atom_id, atom_count):
                    owner.setdefault(atom_id, record.component_id)
        if isinstance(record.isotope_atoms, tuple):
            for descriptor in record.isotope_atoms:
                if isinstance(getattr(descriptor, "atom_id", None), int):
                    primary_descriptor_by_atom.setdefault(descriptor.atom_id, []).append(descriptor)

    def validate_shape(item, index: int, *, reference: bool) -> bool:
        label = "reference" if reference else "requirement"
        valid = True
        if not isinstance(item, IsotopeCitationReference if reference else IsotopeCitationRequirement):
            errors.append(f"{label} {index}: invalid record type")
            return False
        if not _valid_atom_id(item.atom_id, atom_count):
            errors.append(f"{label} {index}: invalid original atom ID")
            valid = False
        for field, value in (("source component", item.source_component_id),
                             ("target component", item.target_component_id)):
            if not isinstance(value, str) or not value:
                errors.append(f"{label} {index}: invalid {field} ID")
                valid = False
        if (isinstance(item.source_component_id, str)
                and isinstance(item.target_component_id, str)
                and item.source_component_id == item.target_component_id):
            errors.append(f"{label} {index}: source and target components must differ")
            valid = False
        if not _valid_role_pair(item.rule, item.source_role, item.target_role):
            errors.append(f"{label} {index}: unsupported source rule/role pair")
            valid = False
        expected_proof = (
            EsterBridgeOxygenCitationProof
            if item.rule is IsotopeCitationRule.P82_6_4_ESTER_OXYGEN
            else PhosphorusBridgeOxygenCitationProof
            if item.rule is IsotopeCitationRule.P83_1_2_2_PHOSPHORUS_ESTER
            else ()
        )
        if not isinstance(item.role_proof, expected_proof):
            errors.append(f"{label} {index}: missing or incompatible original-graph role proof")
            valid = False
        elif (_valid_atom_id(item.atom_id, atom_count)
              and not _valid_role_proof(mol, item.atom_id, item.rule, item.role_proof)):
            errors.append(f"{label} {index}: original graph does not match the typed role proof")
            valid = False
        if not _valid_locant(item.source_locant) or not _valid_locant(item.target_locant):
            errors.append(f"{label} {index}: locants must be nonempty strings or None")
            valid = False
        return valid

    valid_requirements = []
    for index, item in enumerate(requirements):
        if validate_shape(item, index, reference=False):
            valid_requirements.append(item)
            if owner.get(item.atom_id) != item.source_component_id:
                errors.append(f"requirement {index}: source component is not the primary owner")
            descriptors = primary_descriptor_by_atom.get(item.atom_id, ())
            if len(descriptors) != 1 or descriptors[0].printed_locant != item.source_locant:
                errors.append(f"requirement {index}: source locant differs from primary descriptor")
            if item.target_component_id not in component_ids:
                errors.append(f"requirement {index}: target component is absent from primary ledger")
            if _valid_atom_id(item.atom_id, atom_count):
                atom = mol.GetAtomWithIdx(item.atom_id)
                if not atom.GetIsotope():
                    errors.append(f"requirement {index}: original atom is not isotope-labeled")

    valid_references = []
    for index, item in enumerate(supplied):
        if validate_shape(item, index, reference=True):
            valid_references.append(item)
            if not isinstance(item.element, str) or not item.element:
                errors.append(f"reference {index}: element must be a nonempty symbol")
            if not isinstance(item.mass, int) or isinstance(item.mass, bool) or item.mass <= 0:
                errors.append(f"reference {index}: mass must be a positive integer")
            if owner.get(item.atom_id) != item.source_component_id:
                errors.append(f"reference {index}: source component is not the primary owner")
            descriptors = primary_descriptor_by_atom.get(item.atom_id, ())
            if len(descriptors) != 1 or descriptors[0].printed_locant != item.source_locant:
                errors.append(f"reference {index}: source locant differs from primary descriptor")
            if item.target_component_id not in component_ids:
                errors.append(f"reference {index}: target component is absent from primary ledger")
            if _valid_atom_id(item.atom_id, atom_count):
                atom = mol.GetAtomWithIdx(item.atom_id)
                if not atom.GetIsotope():
                    errors.append(f"reference {index}: original atom is not isotope-labeled")
                if isinstance(item.element, str) and atom.GetSymbol() != item.element:
                    errors.append(f"reference {index}: element mismatch for original atom")
                if isinstance(item.mass, int) and not isinstance(item.mass, bool) and atom.GetIsotope() != item.mass:
                    errors.append(f"reference {index}: mass mismatch for original atom")

    def key(item):
        return (item.atom_id, item.source_component_id, item.target_component_id,
                item.rule, item.source_role, item.target_role,
                item.source_locant, item.target_locant, item.role_proof)

    required_keys = [key(item) for item in valid_requirements]
    reference_keys = [key(item) for item in valid_references]
    if len(set(required_keys)) != len(required_keys):
        errors.append("duplicate citation requirement")
    if len(set(reference_keys)) != len(reference_keys):
        errors.append("duplicate secondary citation reference")
    if set(reference_keys) != set(required_keys):
        errors.append("secondary references do not exactly fulfill source-derived requirements")

    secondary_counts: dict[int, int] = {}
    for item in valid_references:
        if _valid_atom_id(item.atom_id, atom_count):
            secondary_counts[item.atom_id] = secondary_counts.get(item.atom_id, 0) + 1
    totals = tuple(
        IsotopeCitationTally(atom_id, 1, secondary_counts.get(atom_id, 0),
                             1 + secondary_counts.get(atom_id, 0))
        for atom_id in primary.covered_atom_ids
    )
    return IsotopeCitationLedger(
        complete=not errors,
        errors=tuple(dict.fromkeys(errors)),
        primary_coverage=primary,
        required_references=tuple(valid_requirements),
        references=tuple(valid_references),
        citation_totals=totals,
    )
