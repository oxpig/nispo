"""Status-only frontier for bounded neutral linear monoacid O/H isotopes.

This helper exposes mapped acid constructions to a caller that owns public
selection. It never emits a whole preferred name: source-closed constructions
are ``complete`` and in-scope isotope roles without closed construction
evidence are ``pending`` so callers can block a generic fallback.
"""

from __future__ import annotations

from dataclasses import dataclass

from .acid_oxygen_isotope_methods import recognize_linear_monoacid
from .complete_functional_parent import functional_parent_candidates
from .component_mapping import _valid_isotope_role_evidence
from .oxygen_characteristic_suffixes import oxygen_group


@dataclass(frozen=True, slots=True)
class AcidOxygenIsotopeFrontier:
    disposition: str
    reason: str
    constructions: tuple = ()

    @property
    def blocks_generic_fallback(self) -> bool:
        return self.disposition == "pending"


def acid_oxygen_isotope_frontier(mol, Chem) -> AcidOxygenIsotopeFrontier:
    """Classify only isotope-modified acid O/H roles in the exact linear scope.

    Unrelated molecules and carbon-only isotope operations return
    ``outside-scope``. Within scope, the existing functional-parent candidate
    comparison supplies the tied construction frontier. Candidates are exposed
    only when every construction carries validated original-atom evidence and
    no clause-derived same-mass count remains pending.
    """
    group = recognize_linear_monoacid(mol, Chem)
    if group is None:
        return AcidOxygenIsotopeFrontier("outside-scope", "not a neutral linear monoacid")
    typed_group = oxygen_group(mol, group)
    if typed_group is None:
        return AcidOxygenIsotopeFrontier("outside-scope", "acid oxygen roles are not typed")
    role_atoms = typed_group.oxygen_atoms + typed_group.attached_hydrogens
    if not any(mol.GetAtomWithIdx(idx).GetIsotope() for idx in role_atoms):
        return AcidOxygenIsotopeFrontier("outside-scope", "no acid O/H isotope role")

    from .acid_oxygen_isotope_methods import retained_acid_oxygen_record

    retained = retained_acid_oxygen_record(mol, group)
    if retained is not None and retained.status == "clause-derived-count":
        return AcidOxygenIsotopeFrontier(
            "pending", "P-82.2.1 count construction is clause-derived, PIN status pending",
        )

    candidates = functional_parent_candidates(mol, Chem)
    constructions = tuple(
        candidate.construction for candidate in candidates
        if candidate.construction is not None
    )
    if (not candidates or len(constructions) != len(candidates)
            or any(not _valid_isotope_role_evidence(mol, construction)
                   for construction in constructions)):
        return AcidOxygenIsotopeFrontier(
            "pending", "in-scope acid O/H isotope form lacks a complete mapped writer construction",
        )
    if any(construction.isotope_role_evidence.status == "clause-derived-count"
           for construction in constructions):
        return AcidOxygenIsotopeFrontier(
            "pending", "a candidate carries a clause-derived isotope count",
        )
    return AcidOxygenIsotopeFrontier(
        "complete", "mapped construction frontier carries validated acid isotope roles",
        constructions,
    )
