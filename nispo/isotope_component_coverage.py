"""Typed accounting for isotope descriptors across named graph components.

This ledger validates primary original-atom coverage and component-map claims.
Exact-once coverage counts isotope-bearing original atoms; it does not claim
that the corresponding nuclide token occurs exactly once in written text.
P-82.6.4 can require one oxygen role to be referenced in more than one part of
an ester name. Such cross-word references need a separate typed reference
proof and are outside this closed component-local API. It does not parse or
certify the spelling of ``component_text``; callers must provide text from a
trusted typed writer and attest which isotope atoms its typed descriptor
construction covers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class IsotopeDescriptorAtom:
    """One original isotope-bearing atom represented by a typed descriptor.

    ``component_locant`` is the locant supplied by the component's original
    atom map. ``printed_locant`` is the locant claimed by the typed writer;
    ``None`` represents a writer-selected locant omission and is not itself a
    proof that the omission follows P-82.6.
    """

    atom_id: int
    element: str
    mass: int
    component_locant: str | None
    printed_locant: str | None


@dataclass(frozen=True, slots=True)
class IsotopeComponentRecord:
    """A trusted writer's component text plus its original-atom proof data."""

    component_id: str
    component_text: str
    atom_ids: frozenset[int]
    # Complete map for this component scope. None denotes an atom without a
    # conventional locant in this component (for example a separately named
    # suffix atom); the map remains explicit for every owned atom.
    atom_locants: tuple[tuple[int, str | None], ...]
    # Per-original-atom claims made by the typed descriptor constructor.
    isotope_atoms: tuple[IsotopeDescriptorAtom, ...] = ()


@dataclass(frozen=True, slots=True)
class IsotopeCoverageResult:
    """Coverage status plus structured diagnostics; no text-token inference."""

    complete: bool
    errors: tuple[str, ...]
    required_atom_ids: tuple[int, ...]
    covered_atom_ids: tuple[int, ...]
    component_texts: tuple[tuple[str, str], ...]


def _atom_id(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _locant(value) -> bool:
    return value is None or isinstance(value, str) and bool(value)


def validate_isotope_component_coverage(
    mol,
    components: Iterable[IsotopeComponentRecord],
) -> IsotopeCoverageResult:
    """Check exact coverage of every specified isotope atom in ``mol``.

    Components have disjoint original atom scopes. Each descriptor-atom claim
    must refer to a labeled atom in its owning scope and agree with the element,
    mass and mapped component locant on the original graph. Each required
    isotope atom must be covered exactly once. Missing, duplicated, or
    out-of-scope claims fail closed. "Exactly once" means each original
    isotope-bearing atom is accounted for as a primary descriptor atom once;
    it is not a count of printed nuclide-token occurrences. Cross-word
    references to one atom require an additional typed reference record and
    are not accepted by this API.

    The ``component_text`` field is carried through for a caller assembling a
    name, but it is deliberately never searched for isotope substrings: textual
    token presence cannot prove atom ownership. Similarly, this function does
    not adjudicate whether a writer's ``printed_locant=None`` is authorized by
    P-82.6; that remains part of the writer's typed rule proof.
    """
    errors: list[str] = []
    try:
        atom_count = mol.GetNumAtoms()
        required = {
            atom.GetIdx()
            for atom in mol.GetAtoms()
            if atom.GetIsotope()
        }
    except Exception:
        return IsotopeCoverageResult(False, ("invalid original molecule",), (), (), ())

    try:
        supplied_records = tuple(components)
    except Exception:
        return IsotopeCoverageResult(
            False, ("components must be an iterable of component records",),
            tuple(sorted(required)), (), (),
        )
    records: list[IsotopeComponentRecord] = []
    for index, record in enumerate(supplied_records):
        if not isinstance(record, IsotopeComponentRecord):
            errors.append(f"record {index}: expected IsotopeComponentRecord")
            continue
        records.append(record)
    ids = [record.component_id for record in records if isinstance(record.component_id, str)]
    if any(not isinstance(record.component_id, str) or not record.component_id
           for record in records):
        errors.append("component ID must be a nonempty string")
    if len(set(ids)) != len(ids):
        errors.append("duplicate component ID")

    owner: dict[int, str] = {}
    component_texts: list[tuple[str, str]] = []
    for record in records:
        component_id = record.component_id if isinstance(record.component_id, str) else "<invalid>"
        if not isinstance(record.component_text, str) or not record.component_text:
            errors.append(f"{component_id}: missing trusted component text")
        else:
            if isinstance(record.component_id, str):
                component_texts.append((record.component_id, record.component_text))

        if not isinstance(record.atom_ids, frozenset):
            errors.append(f"{component_id}: atom_ids must be a frozenset of original atom IDs")
            atom_ids = frozenset()
        else:
            atom_ids = record.atom_ids
        if not atom_ids:
            errors.append(f"{component_id}: empty component scope")
        valid_scope: set[int] = set()
        for atom_id in atom_ids:
            if not _atom_id(atom_id) or atom_id < 0 or atom_id >= atom_count:
                errors.append(f"{component_id}: atom ID outside original graph: {atom_id!r}")
                continue
            valid_scope.add(atom_id)
            if atom_id in owner:
                errors.append(
                    f"atom {atom_id}: overlapping component scopes "
                    f"{owner[atom_id]!r} and {component_id!r}"
                )
            else:
                owner[atom_id] = component_id

        if not isinstance(record.atom_locants, tuple):
            errors.append(f"{component_id}: atom_locants must be a tuple of (atom ID, locant) pairs")
            map_rows = ()
        else:
            map_rows = record.atom_locants
        map_ids = []
        valid_map_rows = []
        for map_row in map_rows:
            if not isinstance(map_row, tuple) or len(map_row) != 2:
                errors.append(f"{component_id}: malformed atom-map row")
                continue
            atom_id, locant_value = map_row
            if not _atom_id(atom_id):
                errors.append(f"{component_id}: atom-map key must be an integer original atom ID")
                continue
            if not _locant(locant_value):
                errors.append(f"{component_id}: locant must be a nonempty string or None")
                continue
            map_ids.append(atom_id)
            valid_map_rows.append((atom_id, locant_value))
        if len(set(map_ids)) != len(map_ids):
            errors.append(f"{component_id}: duplicate atom-map entry")
        if set(map_ids) != set(atom_ids):
            errors.append(f"{component_id}: atom map does not cover its complete scope")
        locant_by_atom = dict(valid_map_rows)

        if not isinstance(record.isotope_atoms, tuple):
            errors.append(f"{component_id}: isotope_atoms must be a tuple of descriptor facts")
            descriptor_rows = ()
        else:
            descriptor_rows = record.isotope_atoms
        valid_descriptors = []
        for descriptor in descriptor_rows:
            if not isinstance(descriptor, IsotopeDescriptorAtom):
                errors.append(f"{component_id}: expected IsotopeDescriptorAtom")
                continue
            atom_id = descriptor.atom_id
            if not _atom_id(atom_id):
                errors.append(f"{component_id}: descriptor atom ID must be an integer original atom ID")
                continue
            if not isinstance(descriptor.element, str) or not descriptor.element:
                errors.append(f"{component_id}: descriptor element must be a nonempty symbol")
                continue
            if not isinstance(descriptor.mass, int) or isinstance(descriptor.mass, bool) or descriptor.mass <= 0:
                errors.append(f"{component_id}: descriptor mass must be a positive integer")
                continue
            if not _locant(descriptor.component_locant) or not _locant(descriptor.printed_locant):
                errors.append(f"{component_id}: descriptor locants must be nonempty strings or None")
                continue
            valid_descriptors.append(descriptor)
            atom_id = descriptor.atom_id
            if atom_id not in valid_scope:
                errors.append(f"{component_id}: isotope atom {atom_id} is outside component scope")
                continue
            if atom_id < 0 or atom_id >= atom_count:
                errors.append(f"{component_id}: isotope atom {atom_id} is outside original graph")
                continue
            atom = mol.GetAtomWithIdx(atom_id)
            if not atom.GetIsotope():
                errors.append(f"{component_id}: atom {atom_id} is not isotope-labeled in original graph")
            if atom.GetSymbol() != descriptor.element:
                errors.append(f"{component_id}: element mismatch for original atom {atom_id}")
            if atom.GetIsotope() != descriptor.mass:
                errors.append(f"{component_id}: mass mismatch for original atom {atom_id}")
            mapped_locant = locant_by_atom.get(atom_id, "__missing__")
            if mapped_locant == "__missing__":
                errors.append(f"{component_id}: isotope atom {atom_id} has no component-map entry")
            elif descriptor.component_locant != mapped_locant:
                errors.append(f"{component_id}: mapped locant mismatch for original atom {atom_id}")
            if descriptor.printed_locant is not None and descriptor.printed_locant != mapped_locant:
                errors.append(f"{component_id}: claimed printed locant mismatch for original atom {atom_id}")

    claimed = [
        descriptor.atom_id
        for record in records
        for descriptor in record.isotope_atoms
        if isinstance(record.isotope_atoms, tuple)
        and isinstance(descriptor, IsotopeDescriptorAtom)
        and _atom_id(descriptor.atom_id)
    ]
    counts: dict[int, int] = {}
    for atom_id in claimed:
        counts[atom_id] = counts.get(atom_id, 0) + 1
    for atom_id in sorted(required):
        if counts.get(atom_id, 0) == 0:
            errors.append(f"required isotope atom {atom_id} is uncovered")
        elif counts[atom_id] > 1:
            errors.append(f"required isotope atom {atom_id} is covered {counts[atom_id]} times")
    for atom_id in sorted(set(counts) - required):
        errors.append(f"unrequired isotope atom {atom_id} was claimed")

    # A covered isotope cannot be claimed by a component that failed the
    # disjoint ownership map, even if the local record otherwise looks valid.
    for record in records:
        if not isinstance(record.isotope_atoms, tuple):
            continue
        for descriptor in record.isotope_atoms:
            if not isinstance(descriptor, IsotopeDescriptorAtom) or not _atom_id(descriptor.atom_id):
                continue
            if owner.get(descriptor.atom_id) != record.component_id:
                errors.append(f"{record.component_id}: no exclusive owner for isotope atom {descriptor.atom_id}")

    covered = tuple(sorted(atom_id for atom_id, count in counts.items() if count == 1))
    return IsotopeCoverageResult(
        complete=not errors,
        errors=tuple(dict.fromkeys(errors)),
        required_atom_ids=tuple(sorted(required)),
        covered_atom_ids=covered,
        component_texts=tuple(component_texts),
    )


def component_record_from_atom_map(
    mol,
    *,
    component_id: str,
    component_text: str,
    atom_ids: Iterable[int],
    atom_locants: Iterable[tuple[int, str | None]],
    descriptor_atom_ids: Iterable[int],
    printed_locants: dict[int, str | None] | None = None,
) -> IsotopeComponentRecord:
    """Build a coverage record from an explicit typed writer claim.

    ``descriptor_atom_ids`` must come from the writer's structural descriptor
    construction (or an explicit local isotope atom set), never from scanning
    ``component_text``. Element, mass and mapped locant are read from the
    original molecule and supplied atom map. ``printed_locants`` may state a
    writer-proved omission with ``None``; this helper does not decide whether
    that omission is allowed by naming rules.
    """
    if not isinstance(component_id, str) or not component_id:
        raise ValueError("component_id must be a nonempty string")
    if not isinstance(component_text, str) or not component_text:
        raise ValueError("component_text must be a nonempty trusted-writer string")
    try:
        scope_values = tuple(atom_ids)
        map_rows = tuple(atom_locants)
        claimed = tuple(descriptor_atom_ids)
    except Exception as exc:
        raise ValueError("scope, map, and descriptor IDs must be iterable") from exc
    if any(not _atom_id(atom_id) for atom_id in scope_values):
        raise ValueError("component scope IDs must be integer original atom IDs")
    if len(scope_values) != len(set(scope_values)):
        raise ValueError("component scope IDs must be unique")
    scope = frozenset(scope_values)
    if not scope:
        raise ValueError("component scope must be nonempty")
    validated_map_rows = []
    map_ids = []
    for row in map_rows:
        if not isinstance(row, (tuple, list)) or len(row) != 2:
            raise ValueError("each atom-map row must be a two-item pair")
        atom_id, locant_value = row
        if not _atom_id(atom_id):
            raise ValueError("atom-map keys must be integer original atom IDs")
        if not _locant(locant_value):
            raise ValueError("atom-map locants must be nonempty strings or None")
        map_ids.append(atom_id)
        validated_map_rows.append((atom_id, locant_value))
    if len(map_ids) != len(set(map_ids)):
        raise ValueError("duplicate atom-map entry")
    mapping = dict(validated_map_rows)
    if set(mapping) != set(scope):
        raise ValueError("atom_locants must map every component-scope atom exactly once")
    if any(not _atom_id(atom_id) for atom_id in claimed):
        raise ValueError("descriptor atom IDs must be integer original atom IDs")
    if len(claimed) != len(set(claimed)):
        raise ValueError("descriptor atom IDs must be unique within the component")
    if printed_locants is None:
        printed_locants = {}
    if not isinstance(printed_locants, dict):
        raise ValueError("printed_locants must be a mapping from descriptor atom ID to locant or None")
    for atom_id, locant_value in printed_locants.items():
        if not _atom_id(atom_id) or atom_id not in claimed or not _locant(locant_value):
            raise ValueError("printed_locants contains an invalid or unclaimed atom entry")
    try:
        atom_count = mol.GetNumAtoms()
    except Exception as exc:
        raise ValueError("invalid original molecule") from exc
    facts = []
    for atom_id in claimed:
        if atom_id < 0 or atom_id >= atom_count:
            raise ValueError(f"descriptor atom {atom_id} is outside original graph")
        if atom_id not in scope:
            raise ValueError(f"descriptor atom {atom_id} is outside component scope")
        atom = mol.GetAtomWithIdx(atom_id)
        if not atom.GetIsotope():
            raise ValueError(f"descriptor atom {atom_id} has no isotope in original graph")
        facts.append(IsotopeDescriptorAtom(
            atom_id=atom_id,
            element=atom.GetSymbol(),
            mass=atom.GetIsotope(),
            component_locant=mapping[atom_id],
            printed_locant=printed_locants.get(atom_id, mapping[atom_id]),
        ))
    return IsotopeComponentRecord(
        component_id=component_id,
        component_text=component_text,
        atom_ids=scope,
        atom_locants=tuple(validated_map_rows),
        isotope_atoms=tuple(facts),
    )
