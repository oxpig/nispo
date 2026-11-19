"""Typed primary ownership and secondary references for P-82.6.4 esters.

P-82.6.4 can reference one bridge oxygen twice in the name: its primary
nuclide-O qualifier belongs to the organyl component, while the oxygen count
descriptor on the acid-derived ending also counts that same atom. The latter
is a typed secondary reference, not a second primary atom-coverage claim.
This module is deliberately limited to one ordinary acyclic ester linkage.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from .isotope_component_coverage import (
    IsotopeComponentRecord,
    component_record_from_atom_map,
    validate_isotope_component_coverage,
)


@dataclass(frozen=True, slots=True)
class EsterOxygenRoles:
    """Original-graph roles for one acyl carbonyl and its ester bridge."""

    acyl_carbon: int
    bridge_oxygen: int
    carbonyl_oxygen: int
    organyl_carbon: int


@dataclass(frozen=True, slots=True)
class EsterOxygenReference:
    """A P-82.6.4 secondary reference to an already-owned isotope atom."""

    atom_id: int
    element: str
    mass: int
    source_component_id: str
    target_component_id: str
    target_role: str


@dataclass(frozen=True, slots=True)
class EsterOxygenCoverageResult:
    """Validation status for primary ownership and required P-82.6.4 refs."""

    complete: bool
    errors: tuple[str, ...]
    primary_atom_ids: tuple[int, ...]
    suffix_count_atom_ids: tuple[int, ...]
    suffix_count_primary_atom_ids: tuple[int, ...]
    secondary_references: tuple[EsterOxygenReference, ...]
    suffix_mass_counts: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class EsterAcidComponentWrite:
    """Acid component text and primary claims from its mapped writer."""

    text: str
    coverage: IsotopeComponentRecord
    suffix_count_atom_ids: tuple[int, ...]
    suffix_count_primary_atom_ids: tuple[int, ...]


def _original_id(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def render_ester_acid_component_with_coverage(
    mol,
    Chem,
    component,
    *,
    component_id: str,
    acid_scope: Iterable[int],
    bridge_oxygen_id: int,
    carbonyl_oxygen_id: int,
) -> EsterAcidComponentWrite | None:
    """Render a mapped acid component with explicit P-82.6.4 suffix claims.

    The ordinary mapped writer remains responsible for parent isotope claims.
    A labeled carbonyl oxygen is an explicit local/suffix atom role and is
    added only after the same parent component has been rendered on a private
    copy with that local suffix label cleared. The bridge oxygen is never
    claimed by this acid component; its primary qualifier belongs to the
    organyl component and the secondary acid-count reference is validated by
    :func:`validate_ester_oxygen_coverage`.
    """
    from .component_mapping import MappedComponent
    from .isotope_component_writer import render_mapped_component_with_coverage
    from .oxygen_isotope_suffixes import suffix_oxygen_descriptors

    if not isinstance(component, MappedComponent):
        return None
    if (not isinstance(component.atoms, frozenset)
            or not isinstance(component.embedded_atoms, frozenset)
            or not isinstance(component.local_isotope_atoms, frozenset)):
        return None
    try:
        atom_count = mol.GetNumAtoms()
        scope = tuple(acid_scope)
    except (AttributeError, TypeError, ValueError):
        return None
    if (not isinstance(component_id, str) or not component_id
            or any(not _original_id(idx) or idx < 0 or idx >= atom_count for idx in scope)
            or len(scope) != len(set(scope))):
        return None
    scope_set = frozenset(scope)
    if not scope_set or not component.atoms <= scope_set:
        return None
    if not (component.embedded_atoms <= scope_set
            and scope_set - component.atoms <= component.embedded_atoms):
        return None
    if (not _original_id(bridge_oxygen_id) or not 0 <= bridge_oxygen_id < atom_count
            or not _original_id(carbonyl_oxygen_id) or not 0 <= carbonyl_oxygen_id < atom_count
            or bridge_oxygen_id in scope_set or carbonyl_oxygen_id not in scope_set):
        return None
    bridge = mol.GetAtomWithIdx(bridge_oxygen_id)
    carbonyl = mol.GetAtomWithIdx(carbonyl_oxygen_id)
    if bridge.GetAtomicNum() != 8 or carbonyl.GetAtomicNum() != 8:
        return None
    carbonyl_mass = carbonyl.GetIsotope()
    suffix_count_ids = tuple(sorted(
        idx for idx, atom in ((bridge_oxygen_id, bridge),
                              (carbonyl_oxygen_id, carbonyl))
        if atom.GetIsotope()
    ))
    suffix_primary_ids = (carbonyl_oxygen_id,) if carbonyl_mass else ()
    local_labeled = {idx for idx in component.local_isotope_atoms
                     if _original_id(idx) and 0 <= idx < atom_count
                     and mol.GetAtomWithIdx(idx).GetIsotope()}
    if local_labeled != set(suffix_primary_ids):
        return None
    if any(atom.GetIsotope() and atom.GetAtomicNum() == 8
           and atom.GetIdx() not in {bridge_oxygen_id, carbonyl_oxygen_id}
           for atom in mol.GetAtoms()):
        return None

    base_mol = Chem.Mol(mol)
    if carbonyl_mass:
        base_mol.GetAtomWithIdx(carbonyl_oxygen_id).SetIsotope(0)
    base_write = render_mapped_component_with_coverage(
        base_mol, Chem, component, component_id=component_id,
    )
    if base_write is None:
        return None
    text = (suffix_oxygen_descriptors(
        base_write.text,
        tuple(mol.GetAtomWithIdx(idx).GetIsotope() for idx in suffix_count_ids),
    ) if suffix_count_ids else base_write.text)
    if not text:
        return None

    map_by_atom = dict(base_write.coverage.atom_locants)
    if set(map_by_atom) != set(component.atoms):
        return None
    all_locants = tuple(
        (idx, map_by_atom[idx] if idx in map_by_atom else None)
        for idx in sorted(scope_set)
    )
    parent_claims = tuple(fact.atom_id for fact in base_write.coverage.isotope_atoms)
    descriptor_ids = tuple(sorted(set(parent_claims) | set(suffix_primary_ids)))
    printed = {fact.atom_id: fact.printed_locant
               for fact in base_write.coverage.isotope_atoms}
    printed.update({idx: None for idx in suffix_primary_ids})
    try:
        record = component_record_from_atom_map(
            mol,
            component_id=component_id,
            component_text=text,
            atom_ids=scope_set,
            atom_locants=all_locants,
            descriptor_atom_ids=descriptor_ids,
            printed_locants=printed,
        )
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    return EsterAcidComponentWrite(
        text=text,
        coverage=record,
        suffix_count_atom_ids=suffix_count_ids,
        suffix_count_primary_atom_ids=tuple(suffix_primary_ids),
    )


def validate_ester_oxygen_coverage(
    mol,
    roles: EsterOxygenRoles,
    primary_components: Iterable[IsotopeComponentRecord],
    *,
    organyl_component_id: str,
    acid_component_id: str,
    organyl_qualifier_atom_ids: Iterable[int],
    suffix_count_atom_ids: Iterable[int],
    suffix_count_secondary_atom_ids: Iterable[int],
) -> EsterOxygenCoverageResult:
    """Validate source-role facts for one ordinary ester oxygen construction.

    The caller supplies atom IDs attested by its typed writer construction,
    never IDs inferred from rendered tokens. The primary component records
    retain Job 151's exact-once invariant. The suffix count is separately
    checked as a set of primary atoms plus allowed P-82.6.4 references; a
    bridge oxygen can be primary in the organyl qualifier and secondary in
    the acid-ending count without being counted twice in primary coverage.

    ``organyl_qualifier_atom_ids`` denotes the bridge role named by either
    ``O-`` or ``nuclide-O-``. Only a labeled bridge oxygen is an isotope
    primary claim there. The unmodified bridge oxygen named by ``O-`` when
    the carbonyl oxygen is labeled is not an isotope claim.
    """
    errors: list[str] = []
    try:
        atom_count = mol.GetNumAtoms()
        records = tuple(primary_components)
        supplied_qualifier = tuple(organyl_qualifier_atom_ids)
        supplied_count = tuple(suffix_count_atom_ids)
        supplied_secondary = tuple(suffix_count_secondary_atom_ids)
    except Exception:
        return EsterOxygenCoverageResult(
            False, ("invalid graph or iterable inputs",), (), (), (), (), (),
        )

    if not isinstance(roles, EsterOxygenRoles):
        return EsterOxygenCoverageResult(
            False, ("roles must be an EsterOxygenRoles record",), (), (), (), (), (),
        )
    role_ids = (roles.acyl_carbon, roles.bridge_oxygen,
                roles.carbonyl_oxygen, roles.organyl_carbon)
    if (any(not _original_id(idx) or idx < 0 or idx >= atom_count
                   for idx in role_ids)
            or len(set(role_ids)) != len(role_ids)):
        return EsterOxygenCoverageResult(
            False, ("invalid or overlapping ester role atom IDs",), (), (), (), (), (),
        )
    if (not isinstance(organyl_component_id, str) or not organyl_component_id
            or not isinstance(acid_component_id, str) or not acid_component_id
            or organyl_component_id == acid_component_id):
        return EsterOxygenCoverageResult(
            False, ("invalid component IDs",), (), (), (), (), (),
        )
    valid_inputs: dict[str, bool] = {}
    for key, values in (("organyl qualifier", supplied_qualifier),
                        ("suffix count", supplied_count),
                        ("suffix secondary reference", supplied_secondary)):
        valid = all(_original_id(idx) and 0 <= idx < atom_count for idx in values)
        valid_inputs[key] = valid
        if not valid:
            errors.append(f"{key}: invalid original atom ID")
        elif len(values) != len(set(values)):
            errors.append(f"{key}: duplicate original atom ID")

    acyl = mol.GetAtomWithIdx(roles.acyl_carbon)
    bridge = mol.GetAtomWithIdx(roles.bridge_oxygen)
    carbonyl = mol.GetAtomWithIdx(roles.carbonyl_oxygen)
    organyl = mol.GetAtomWithIdx(roles.organyl_carbon)
    if acyl.GetAtomicNum() != 6 or bridge.GetAtomicNum() != 8:
        errors.append("ester role atom elements do not match carbonyl/bridge oxygen")
    if carbonyl.GetAtomicNum() != 8 or organyl.GetAtomicNum() != 6:
        errors.append("ester role atom elements do not match carbonyl/alkyl carbon")
    role_atoms = (acyl, bridge, carbonyl, organyl)
    if any(atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons()
           for atom in role_atoms):
        errors.append("ester role atoms must be neutral and nonradical")
    try:
        role_valences = (acyl.GetTotalValence(), bridge.GetTotalValence(),
                         carbonyl.GetTotalValence(), organyl.GetTotalValence())
    except (RuntimeError, ValueError):
        role_valences = ()
    if role_valences and role_valences != (4, 2, 2, 4):
        errors.append("ester role valences are outside the ordinary neutral ester construction")
    bridge_bond = mol.GetBondBetweenAtoms(roles.acyl_carbon, roles.bridge_oxygen)
    organyl_bond = mol.GetBondBetweenAtoms(roles.bridge_oxygen, roles.organyl_carbon)
    carbonyl_bond = mol.GetBondBetweenAtoms(roles.acyl_carbon, roles.carbonyl_oxygen)
    if (bridge_bond is None or bridge_bond.GetBondTypeAsDouble() != 1.0
            or organyl_bond is None or organyl_bond.GetBondTypeAsDouble() != 1.0
            or carbonyl_bond is None or carbonyl_bond.GetBondTypeAsDouble() != 2.0):
        errors.append("original bonds do not establish the typed ester oxygen roles")
    if (bridge.GetDegree() != 2
            or {atom.GetIdx() for atom in bridge.GetNeighbors()}
            != {roles.acyl_carbon, roles.organyl_carbon}
            or carbonyl.GetDegree() != 1
            or {atom.GetIdx() for atom in carbonyl.GetNeighbors()}
            != {roles.acyl_carbon}):
        errors.append("oxygen role neighborhoods are not the simple ester bridge/carbonyl pair")
    acyl_oxygen_neighbors = {
        atom.GetIdx() for atom in acyl.GetNeighbors() if atom.GetAtomicNum() == 8
    }
    organyl_carbonyl_neighbors = [
        atom for atom in organyl.GetNeighbors()
        if atom.GetAtomicNum() == 8
        and mol.GetBondBetweenAtoms(roles.organyl_carbon, atom.GetIdx())
        .GetBondTypeAsDouble() == 2.0
    ]
    if acyl_oxygen_neighbors != {roles.bridge_oxygen, roles.carbonyl_oxygen}:
        errors.append("acyl carbon has additional oxygen roles outside this ester grammar")
    if organyl_carbonyl_neighbors:
        errors.append("organyl attachment is an acyl carbon rather than an ordinary alkyl carbon")
    if bridge.GetIsotope() and bridge.GetAtomicNum() != 8:
        errors.append("bridge role has incompatible isotope element")
    if carbonyl.GetIsotope() and carbonyl.GetAtomicNum() != 8:
        errors.append("carbonyl role has incompatible isotope element")

    expected_qualifier = {roles.bridge_oxygen} if (
        bridge.GetIsotope() or carbonyl.GetIsotope()) else set()
    expected_count = {idx for idx in (roles.bridge_oxygen, roles.carbonyl_oxygen)
                      if mol.GetAtomWithIdx(idx).GetIsotope()}
    expected_secondary = {roles.bridge_oxygen} if bridge.GetIsotope() else set()
    if (valid_inputs["organyl qualifier"]
            and set(supplied_qualifier) != expected_qualifier):
        errors.append("organyl O qualifier does not point to the mapped bridge role")
    if valid_inputs["suffix count"] and set(supplied_count) != expected_count:
        errors.append("acid-ending oxygen count omits or adds a labeled ester oxygen role")
    if (valid_inputs["suffix secondary reference"]
            and set(supplied_secondary) != expected_secondary):
        errors.append("secondary count references do not match labeled bridge oxygen roles")
    other_labeled_oxygen = {
        atom.GetIdx() for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 8 and atom.GetIsotope()
        and atom.GetIdx() not in {roles.bridge_oxygen, roles.carbonyl_oxygen}
    }
    if other_labeled_oxygen:
        errors.append("isotope-labeled oxygen lies outside the two proved ester oxygen roles")

    ledger = validate_isotope_component_coverage(mol, records)
    errors.extend("primary ledger: " + message for message in ledger.errors)
    owner: dict[int, str] = {}
    claims_by_atom: dict[int, list[str]] = {}
    for record in records:
        if isinstance(record, IsotopeComponentRecord) and isinstance(record.atom_ids, frozenset):
            for idx in record.atom_ids:
                if _original_id(idx):
                    owner.setdefault(idx, record.component_id)
        if isinstance(record, IsotopeComponentRecord) and isinstance(record.isotope_atoms, tuple):
            for descriptor in record.isotope_atoms:
                if _original_id(getattr(descriptor, "atom_id", None)):
                    claims_by_atom.setdefault(descriptor.atom_id, []).append(record.component_id)

    if bridge.GetIsotope():
        if claims_by_atom.get(roles.bridge_oxygen) != [organyl_component_id]:
            errors.append("labeled bridge oxygen must be primarily owned by the organyl qualifier")
    if carbonyl.GetIsotope():
        if claims_by_atom.get(roles.carbonyl_oxygen) != [acid_component_id]:
            errors.append("labeled carbonyl oxygen must be primarily owned by the acid ending")
    if owner.get(roles.bridge_oxygen) != organyl_component_id:
        errors.append("bridge oxygen is not owned by the organyl component")
    if owner.get(roles.carbonyl_oxygen) != acid_component_id:
        errors.append("carbonyl oxygen is not owned by the acid component")

    references = tuple(
        EsterOxygenReference(
            atom_id=idx,
            element="O",
            mass=mol.GetAtomWithIdx(idx).GetIsotope(),
            source_component_id=organyl_component_id,
            target_component_id=acid_component_id,
            target_role="P-82.6.4 acid-ending oxygen-count descriptor",
        )
        for idx in sorted(expected_secondary)
    )
    mass_counts = Counter(mol.GetAtomWithIdx(idx).GetIsotope()
                          for idx in expected_count)
    return EsterOxygenCoverageResult(
        complete=not errors,
        errors=tuple(dict.fromkeys(errors)),
        primary_atom_ids=tuple(sorted(ledger.covered_atom_ids)),
        suffix_count_atom_ids=tuple(sorted(idx for idx in supplied_count
                                           if _original_id(idx)
                                           and 0 <= idx < atom_count)),
        suffix_count_primary_atom_ids=tuple(sorted(
            idx for idx in expected_count if idx not in expected_secondary
        )),
        secondary_references=references,
        suffix_mass_counts=tuple(sorted(mass_counts.items())),
    )
