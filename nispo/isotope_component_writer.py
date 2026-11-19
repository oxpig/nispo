"""Coverage records emitted beside a bounded MappedComponent rendering.

This adapter records only primary isotope descriptors constructed by the
component's mapped parent labels. It never scans rendered text. Prefix-child
descriptors require their own mapped writer record; suffix descriptor parts
without original atom IDs and P-82.6.4 cross-word references are deliberately
outside this adapter.
"""

from __future__ import annotations

from dataclasses import dataclass

from .component_mapping import (
    MappedComponent,
    _mapped_isotope_descriptor_plan,
    decorate_component,
)
from .isotope_component_coverage import (
    IsotopeComponentRecord,
    component_record_from_atom_map,
)


@dataclass(frozen=True, slots=True)
class MappedComponentWrite:
    """Rendered component text and its primary original-isotope coverage."""

    text: str
    coverage: IsotopeComponentRecord


def _primary_descriptor_atoms(mol, component: MappedComponent) -> tuple[int, ...]:
    return tuple(sorted(
        idx for idx in component.atoms
        if mol.GetAtomWithIdx(idx).GetIsotope()
        and idx not in component.local_isotope_atoms
    ))


def _component_shape_is_valid(mol, component: MappedComponent) -> bool:
    """Validate atom ownership and the fields consumed by the decorator."""
    if component.isotope_labelling_kind:
        # This adapter certifies P-82 fixed isotopic molecules only; a P-83
        # distribution needs its distinct labelled-substance result model.
        return False
    try:
        atom_count = mol.GetNumAtoms()
    except (AttributeError, TypeError, ValueError):
        return False

    def valid_ids(values) -> bool:
        return (isinstance(values, frozenset)
                and all(isinstance(idx, int) and not isinstance(idx, bool)
                        and 0 <= idx < atom_count for idx in values))

    if not valid_ids(component.atoms) or not valid_ids(component.embedded_atoms):
        return False
    if not component.atoms:
        return False
    if component.atoms & component.embedded_atoms:
        return False
    if not isinstance(component.atom_locants, tuple):
        return False
    rows = []
    for row in component.atom_locants:
        if (not isinstance(row, tuple) or len(row) != 2
                or not isinstance(row[0], int) or isinstance(row[0], bool)
                or row[0] not in component.atoms
                or not isinstance(row[1], str)):
            return False
        rows.append(row[0])
    if len(rows) != len(set(rows)) or set(rows) != set(component.atoms):
        return False
    if not valid_ids(component.local_isotope_atoms):
        return False
    if not component.local_isotope_atoms <= component.atoms:
        return False
    if not isinstance(component.isotope_descriptor_parts, tuple):
        return False
    if not isinstance(component.parent, str) or not component.parent:
        return False
    if not isinstance(component.prefixes, tuple):
        return False
    for row in component.prefixes:
        if (not isinstance(row, tuple) or len(row) != 2
                or not all(isinstance(value, str) for value in row)):
            return False
    if not isinstance(component.hydro_prefix, str):
        return False
    if not isinstance(component.systematic_parent, bool):
        return False
    if not isinstance(component.nitrogen_site_locants, bool):
        return False
    if (not isinstance(component.isotope_unlocanted_elements, tuple)
            or any(not isinstance(number, int) or isinstance(number, bool)
                   or number <= 0 for number in component.isotope_unlocanted_elements)):
        return False
    if not isinstance(component.prefix_mode, str):
        return False
    if not isinstance(component.omit_methyl_prefix_locant, bool):
        return False
    if component.prefix_natures is not None:
        if not isinstance(component.prefix_natures, tuple):
            return False
        if any(not isinstance(row, tuple) or len(row) != 2
               or not all(isinstance(value, str) for value in row)
               for row in component.prefix_natures):
            return False
    return True


def _written_locants(mol, Chem, component: MappedComponent,
                     atom_ids: tuple[int, ...]) -> dict[int, str | None]:
    """Use the descriptor writer's actual original-atom locant decisions.

    Descriptor atom identity and mapped location come from the graph and the
    component map. This function never inspects the rendered name.
    """
    plan = _mapped_isotope_descriptor_plan(mol, Chem, component, dict(component.atom_locants))
    if plan is None:
        raise ValueError("parent isotope descriptor plan is unavailable")
    result = {atom_id: printed_locant
              for atom_id, _symbol, _mass, _source_locant, printed_locant in plan[1]}
    if set(result) != set(atom_ids):
        raise ValueError("parent isotope descriptor coverage differs from its original atoms")
    return result


def render_mapped_component_with_coverage(
    mol,
    Chem,
    component: MappedComponent,
    *,
    component_id: str,
) -> MappedComponentWrite | None:
    """Render a primary mapped component and return its isotope atom facts.

    The supported closed scope is the isotope population that
    ``decorate_component`` writes from ``component.atoms``. The component map
    must be complete, and isotope atoms excluded through
    ``local_isotope_atoms`` cause a decline because their alternative writer
    part has no original-atom descriptor record here. Nonempty
    ``isotope_descriptor_parts`` also decline: their current tuple format has
    element/mass/locant but no original atom IDs.

    Isotope-bearing ``embedded_atoms`` are not claimed by this primary record;
    their separately rendered typed child must supply another component
    record. The complete ledger is expected to combine those records. A
    repeated cross-word reference to an atom (including P-82.6.4 ester oxygen
    roles) needs a separate typed reference proof and is not accepted here.
    """
    if not isinstance(component, MappedComponent):
        return None
    if not _component_shape_is_valid(mol, component):
        return None
    if component.isotope_descriptor_parts:
        return None
    if any(mol.GetAtomWithIdx(idx).GetIsotope()
           for idx in component.local_isotope_atoms):
        return None
    try:
        text = decorate_component(mol, Chem, component)
    except (TypeError, ValueError, IndexError, KeyError):
        # Malformed typed formatter data declines locally; unexpected
        # exception classes remain visible to callers.
        return None
    if not text:
        return None
    atom_ids = _primary_descriptor_atoms(mol, component)
    locants = tuple((idx, (loc or None)) for idx, loc in component.atom_locants)
    try:
        record = component_record_from_atom_map(
            mol,
            component_id=component_id,
            component_text=text,
            atom_ids=component.atoms,
            atom_locants=locants,
            descriptor_atom_ids=atom_ids,
            printed_locants=_written_locants(mol, Chem, component, atom_ids),
        )
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    return MappedComponentWrite(text=text, coverage=record)
