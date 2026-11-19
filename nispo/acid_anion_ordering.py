"""Source-status assessment for isotope-modified acid-anion citations.

This module does not choose, sort, or render acid-anion names. It records the
evidentiary status of a proposed citation order under P-65.6.3.3.3.2 and
P-82.2.2/.2.2.1. In particular, modified-before-unmodified ordering for
otherwise identical anion names is retained only as an explicit inference:
P-82.2.2.1 states that preference for substituents, not for separate
functional-class anion components.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal


OrderStatus = Literal["source_closed", "inferred", "pending"]


@dataclass(frozen=True, slots=True)
class NuclideModification:
    """One explicitly assigned nuclide fact on an anion component."""

    element: str
    mass_number: int
    role: str


@dataclass(frozen=True, slots=True)
class AcidAnionCitationFacts:
    """Caller-proved ordinary and annotated identity facts for one anion.

    ``ordinary_identity`` identifies the unannotated anion constitution;
    ``ordinary_name`` is its source-selected unannotated citation name.
    ``annotated_identity`` identifies the fully annotated component. The
    isotope profile is a set of nuclide/role facts, not rendered text.
    A ``None`` profile means that annotation facts are incomplete; an empty
    profile means the component is known to be isotopically unmodified.
    """

    ordinary_identity: str
    ordinary_name: str
    annotated_identity: str
    isotope_profile: frozenset[NuclideModification] | None


@dataclass(frozen=True, slots=True)
class PairOrderEvidence:
    """Evidence classification for one pair of cited anion components."""

    status: OrderStatus
    basis: str
    rules: tuple[str, ...]
    proposed_convention: str | None = None


@dataclass(frozen=True, slots=True)
class AcidAnionOrderAssessment:
    """Pure assessment; contains no rendered order or component name."""

    status: OrderStatus
    pair_evidence: tuple[PairOrderEvidence, ...]
    rules: tuple[str, ...]
    reason: str


def _assess_pair(
    first: AcidAnionCitationFacts,
    second: AcidAnionCitationFacts,
) -> PairOrderEvidence:
    if first.isotope_profile is None or second.isotope_profile is None:
        return PairOrderEvidence(
            status="pending",
            basis="incomplete-isotope-profile-facts",
            rules=("P-65.6.3.3.3.2", "P-82.2.2"),
        )

    if first.ordinary_identity != second.ordinary_identity:
        if first.ordinary_name != second.ordinary_name:
            return PairOrderEvidence(
                status="source_closed",
                basis="distinct-ordinary-anion-names",
                rules=("P-65.6.3.3.3.2", "P-14.5"),
            )
        return PairOrderEvidence(
            status="pending",
            basis="distinct-ordinary-identities-with-tied-ordinary-names",
            rules=("P-65.6.3.3.3.2", "P-14.5"),
        )

    if first.isotope_profile == second.isotope_profile:
        if first.annotated_identity == second.annotated_identity:
            return PairOrderEvidence(
                status="source_closed",
                basis="identical-ordinary-and-annotation-profiles",
                rules=("P-65.6.3.3.3.1", "P-82.2.2"),
            )
        return PairOrderEvidence(
            status="pending",
            basis="same-isotope-profile-but-annotated-identities-disagree",
            rules=("P-65.6.3.3.3.1", "P-82.2.2"),
        )

    # This is the open tie: the ordinary anion name is the same, but the
    # isotope-modified component identities differ. Keeping the existing
    # modified-first convention is a reproducible implementation choice only.
    return PairOrderEvidence(
        status="inferred",
        basis="same-ordinary-anion-different-isotope-profiles",
        rules=("P-82.2.2", "P-82.2.2.1", "P-65.6.3.3.3.2", "P-14.5"),
        proposed_convention="modified-before-unmodified-inference",
    )


def assess_acid_anion_order(
    components: Iterable[AcidAnionCitationFacts],
) -> AcidAnionOrderAssessment:
    """Classify pairwise acid-anion citation-order evidence.

    The function preserves caller order solely to label pair evidence; it
    never rearranges components. Any pair with an inferred rule makes the
    aggregate ``inferred`` unless a pair is still pending. Missing facts or an
    otherwise unresolved tie make it ``pending``. At least two complete
    components are required.
    """
    items = tuple(components)
    if len(items) < 2:
        return AcidAnionOrderAssessment(
            status="pending",
            pair_evidence=(),
            rules=("P-65.6.3.3.3.2",),
            reason="At least two anion components are required to assess citation order.",
        )
    if not all(
        isinstance(item, AcidAnionCitationFacts)
        and item.ordinary_identity
        and item.ordinary_name
        and item.annotated_identity
        for item in items
    ):
        return AcidAnionOrderAssessment(
            status="pending",
            pair_evidence=(),
            rules=("P-65.6.3.3.3.2", "P-82.2.2"),
            reason="Component identity facts are incomplete or invalid.",
        )

    pairs = tuple(
        _assess_pair(items[i], items[j])
        for i in range(len(items))
        for j in range(i + 1, len(items))
    )
    statuses = {pair.status for pair in pairs}
    if "pending" in statuses:
        status: OrderStatus = "pending"
        reason = "At least one pair lacks a source-closed citation-order determination."
    elif "inferred" in statuses:
        status = "inferred"
        reason = "At least one tied ordinary-anion pair uses an implementation inference."
    else:
        status = "source_closed"
        reason = "Every pair is covered by a direct ordinary-name or identical-unit rule."
    rules = tuple(dict.fromkeys(rule for pair in pairs for rule in pair.rules))
    return AcidAnionOrderAssessment(status, pairs, rules, reason)
