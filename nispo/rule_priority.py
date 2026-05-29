from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RulePriority:
    """Named priority slot for deterministic parent-rule ordering.

    The numeric value preserves current behavior. The metadata records the
    nomenclature reason for that order so the rule base can move toward
    seniority/specificity scoring instead of anonymous edge-case precedence.
    """

    value: int
    seniority_score: int
    parent_class_score: int
    specificity_score: int
    seniority: str
    parent_class: str
    specificity: str
    rationale: str

    @property
    def score_vector(self) -> tuple[int, int, int, int]:
        return (
            self.value,
            self.seniority_score,
            self.parent_class_score,
            self.specificity_score,
        )


PARENT_RULE_PRIORITIES = {
    "senior-functional-exception": RulePriority(
        9,
        seniority_score=0,
        parent_class_score=0,
        specificity_score=0,
        seniority="characteristic group / charge exception",
        parent_class="acyclic retained or functional parent",
        specificity="very high",
        rationale="Specific charged and retained functional parents outrank generic skeleton naming.",
    ),
    "acyclic-parent": RulePriority(
        10,
        seniority_score=1,
        parent_class_score=0,
        specificity_score=1,
        seniority="acyclic parent skeleton",
        parent_class="chain",
        specificity="high",
        rationale="Acyclic molecules should use the strongest chain parent before ring fallback paths.",
    ),
    "functional-acyclic-skeleton": RulePriority(
        13,
        seniority_score=1,
        parent_class_score=1,
        specificity_score=1,
        seniority="functional acyclic skeleton",
        parent_class="chain with ring substituents",
        specificity="high",
        rationale="Functional skeleton signals can outrank retained ring parents when the ring is substituent-like.",
    ),
    "polycyclic-functional-skeleton": RulePriority(
        14,
        seniority_score=1,
        parent_class_score=2,
        specificity_score=1,
        seniority="polycyclic or functional skeleton",
        parent_class="generic graph parent",
        specificity="high",
        rationale="Generic polycyclic and functional skeleton rules run before retained fused fallback.",
    ),
    "preferred-generic-bicyclic": RulePriority(
        19,
        seniority_score=2,
        parent_class_score=2,
        specificity_score=1,
        seniority="preferred bicyclic graph parent",
        parent_class="bicyclic",
        specificity="high",
        rationale="Preferred generic bicyclic names are tried just before retained fused systems.",
    ),
    "whole-molecule-retained-parent": RulePriority(
        19,
        seniority_score=2,
        parent_class_score=2,
        specificity_score=0,
        seniority="exact retained whole molecule",
        parent_class="retained ring or natural product",
        specificity="very high",
        rationale="Standalone exact retained names should outrank generic ring-core decompositions.",
    ),
    "generated-spiro-component-parent": RulePriority(
        19,
        seniority_score=2,
        parent_class_score=2,
        specificity_score=1,
        seniority="generated spiro component parent",
        parent_class="spiro fused component",
        specificity="high",
        rationale="OPSIN-validated component-spiro parents should outrank generic skeletal-replacement aliases for the same spiro ring system.",
    ),
    "component-spiro-parent": RulePriority(
        19,
        seniority_score=2,
        parent_class_score=2,
        specificity_score=1,
        seniority="component-composed spiro parent",
        parent_class="spiro fused/bridged component",
        specificity="high",
        rationale="Graph-composed spiro parents with generic bicyclo/tricyclo components should outrank aromatic skeletal-replacement aliases for saturated spiro systems.",
    ),
    "retained-fused-parent": RulePriority(
        18,
        seniority_score=2,
        parent_class_score=2,
        specificity_score=2,
        seniority="retained fused parent",
        parent_class="fused ring",
        specificity="medium",
        rationale="Retained fused templates are a general parent family and should precede generated variants.",
    ),
    "hydro-retained-fused-parent": RulePriority(
        18,
        seniority_score=2,
        parent_class_score=2,
        specificity_score=3,
        seniority="hydrogenated retained fused parent",
        parent_class="partially saturated fused ring",
        specificity="medium",
        rationale="Hydrogenated retained fused forms are a close retained-parent variant.",
    ),
    "charged-retained-fused-parent": RulePriority(
        18,
        seniority_score=2,
        parent_class_score=2,
        specificity_score=3,
        seniority="charged retained fused parent",
        parent_class="charged fused ring",
        specificity="medium",
        rationale="Charged retained fused adjustments are tried after neutral retained fused names.",
    ),
    "generic-bicyclic-parent": RulePriority(
        19,
        seniority_score=3,
        parent_class_score=2,
        specificity_score=2,
        seniority="generic bicyclic parent",
        parent_class="bicyclic",
        specificity="medium",
        rationale="Generic bicyclic graph nomenclature is preferred for bridged systems before retained fused fallbacks.",
    ),
    "generic-spiro-parent": RulePriority(
        26,
        seniority_score=3,
        parent_class_score=2,
        specificity_score=2,
        seniority="generic spiro parent",
        parent_class="spiro",
        specificity="medium",
        rationale="Spiro graph names are more specific than single-ring fallback.",
    ),
    "spiro-functional-chain-substituent": RulePriority(
        19,
        seniority_score=3,
        parent_class_score=3,
        specificity_score=3,
        seniority="functional chain with spiro substituent",
        parent_class="chain with spiro ring substituent",
        specificity="medium",
        rationale="Functional chains bearing spiro substituents can be more OPSIN-stable than generic spiro parents.",
    ),
    "senior-spiro-functional-chain-substituent": RulePriority(
        13,
        seniority_score=1,
        parent_class_score=1,
        specificity_score=2,
        seniority="senior functional chain with spiro substituent",
        parent_class="chain with spiro ring substituent",
        specificity="high",
        rationale="Senior functional chains bearing spiro substituents can outrank generic polycyclic parents when the chain carries the characteristic group.",
    ),
    "generic-polycyclic-parent": RulePriority(
        28,
        seniority_score=3,
        parent_class_score=2,
        specificity_score=3,
        seniority="generic polycyclic parent",
        parent_class="polycyclic",
        specificity="medium",
        rationale="Generic polycyclic graph names run after retained fused, bicyclic, and spiro preferences and before generated templates.",
    ),
    "generated-parent": RulePriority(
        28,
        seniority_score=3,
        parent_class_score=3,
        specificity_score=3,
        seniority="generated retained-like parent",
        parent_class="generated ring template",
        specificity="medium",
        rationale="Generated templates broaden retained-parent coverage without outranking curated retained forms.",
    ),
    "single-ring-parent": RulePriority(
        30,
        seniority_score=4,
        parent_class_score=4,
        specificity_score=3,
        seniority="single ring parent",
        parent_class="monocycle",
        specificity="low",
        rationale="Single-ring parent naming runs after fused and generated ring-system candidates.",
    ),
    "generated-partial-hydro-parent": RulePriority(
        28,
        seniority_score=4,
        parent_class_score=3,
        specificity_score=4,
        seniority="generated partially saturated parent",
        parent_class="partially saturated generated fused ring",
        specificity="low",
        rationale="Generated partial-hydro variants are fallback candidates after exact generated templates.",
    ),
    "skeletal-replacement-parent": RulePriority(
        28,
        seniority_score=4,
        parent_class_score=3,
        specificity_score=4,
        seniority="skeletal replacement parent",
        parent_class="hetero replacement fused ring",
        specificity="low",
        rationale="Skeletal replacement names are broader fallback names for fused systems.",
    ),
    "exact-retained-parent": RulePriority(
        45,
        seniority_score=5,
        parent_class_score=5,
        specificity_score=5,
        seniority="exact retained parent",
        parent_class="exact retained ring",
        specificity="fallback",
        rationale="Exact retained single-template matches run late to avoid displacing stronger parent families.",
    ),
    "acyclic-with-ring-substituents": RulePriority(
        60,
        seniority_score=6,
        parent_class_score=6,
        specificity_score=6,
        seniority="acyclic skeleton fallback",
        parent_class="chain with ring substituents",
        specificity="fallback",
        rationale="Acyclic-with-ring-substituents is intentionally a broad fallback family.",
    ),
}


def parent_priority(name: str) -> int:
    return PARENT_RULE_PRIORITIES[name].value
