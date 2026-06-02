from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass(slots=True)
class CandidateTelemetry:
    """Optional per-molecule counters for rule and template work.

    Normal API calls leave this disabled.  Benchmark/debug callers can pass one
    through the naming engine to see which rule families produce, render, and
    select candidates without changing naming behavior.
    """

    rule_applies: Counter[str] = field(default_factory=Counter)
    rule_seconds: Counter[str] = field(default_factory=Counter)
    candidate_count: Counter[str] = field(default_factory=Counter)
    rendered_candidate_count: Counter[str] = field(default_factory=Counter)
    rejected_candidate_count: Counter[str] = field(default_factory=Counter)
    selected_family: Counter[str] = field(default_factory=Counter)
    structured_candidate_count: Counter[str] = field(default_factory=Counter)
    unstructured_candidate_count: Counter[str] = field(default_factory=Counter)
    structured_selected_count: Counter[str] = field(default_factory=Counter)
    selected_with_template_id: int = 0
    selected_with_parent_atoms: int = 0
    selected_safety_reasons: Counter[str] = field(default_factory=Counter)
    template_candidate_count: Counter[str] = field(default_factory=Counter)
    substructure_match_calls: int = 0
