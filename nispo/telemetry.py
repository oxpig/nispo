from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable


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
    selected_rule_family: Counter[str] = field(default_factory=Counter)
    structured_candidate_count: Counter[str] = field(default_factory=Counter)
    unstructured_candidate_count: Counter[str] = field(default_factory=Counter)
    structured_selected_count: Counter[str] = field(default_factory=Counter)
    selected_with_template_id: int = 0
    selected_with_parent_atoms: int = 0
    selected_safety_reasons: Counter[str] = field(default_factory=Counter)
    selected_render_rank_count: Counter[str] = field(default_factory=Counter)
    selected_render_rank_total: Counter[str] = field(default_factory=Counter)
    selected_render_rank_bucket: Counter[str] = field(default_factory=Counter)
    candidate_facts_total: Counter[str] = field(default_factory=Counter)
    render_set_size: Counter[str] = field(default_factory=Counter)
    facts_suppressed_before_render: Counter[str] = field(default_factory=Counter)
    fallback_render_all_count: Counter[str] = field(default_factory=Counter)
    branch_prefix_calls: int = 0
    branch_prefix_cache_hits: int = 0
    branch_prefix_cache_misses: int = 0
    branch_prefix_in_progress_rejections: int = 0
    branch_prefix_produced: Counter[str] = field(default_factory=Counter)
    branch_prefix_calls_by_rule: Counter[str] = field(default_factory=Counter)
    branch_prefix_cache_hits_by_rule: Counter[str] = field(default_factory=Counter)
    branch_prefix_cache_misses_by_rule: Counter[str] = field(default_factory=Counter)
    active_rule_family: str = ""
    filter_stage_count: Counter[str] = field(default_factory=Counter)
    template_candidate_count: Counter[str] = field(default_factory=Counter)
    substructure_match_calls: int = 0


_COUNTER_FIELDS = (
    "rule_applies",
    "rule_seconds",
    "candidate_count",
    "rendered_candidate_count",
    "rejected_candidate_count",
    "selected_family",
    "selected_rule_family",
    "structured_candidate_count",
    "unstructured_candidate_count",
    "structured_selected_count",
    "selected_safety_reasons",
    "selected_render_rank_count",
    "selected_render_rank_total",
    "selected_render_rank_bucket",
    "candidate_facts_total",
    "render_set_size",
    "facts_suppressed_before_render",
    "fallback_render_all_count",
    "branch_prefix_produced",
    "branch_prefix_calls_by_rule",
    "branch_prefix_cache_hits_by_rule",
    "branch_prefix_cache_misses_by_rule",
    "filter_stage_count",
    "template_candidate_count",
)

_INTEGER_FIELDS = (
    "selected_with_template_id",
    "selected_with_parent_atoms",
    "branch_prefix_calls",
    "branch_prefix_cache_hits",
    "branch_prefix_cache_misses",
    "branch_prefix_in_progress_rejections",
    "substructure_match_calls",
)

_TIMING_FIELDS = frozenset({"rule_seconds"})


def telemetry_to_dict(telemetry: CandidateTelemetry) -> dict:
    """Return a deterministic JSON-safe representation of telemetry counters."""

    data: dict[str, dict | int] = {}
    for field_name in _COUNTER_FIELDS:
        counter = getattr(telemetry, field_name)
        data[field_name] = _sorted_counter_dict(
            counter,
            round_values=field_name in _TIMING_FIELDS,
        )
    for field_name in _INTEGER_FIELDS:
        data[field_name] = int(getattr(telemetry, field_name))
    return data


def empty_telemetry_dict() -> dict:
    """Return the zero value for serialized telemetry."""

    return merge_telemetry_dicts(())


def merge_telemetry_dicts(items: Iterable[dict | None]) -> dict:
    """Merge serialized telemetry emitted by molecules, chunks, or workers."""

    counters = {field_name: Counter() for field_name in _COUNTER_FIELDS}
    integers = {field_name: 0 for field_name in _INTEGER_FIELDS}
    for item in items:
        if not item:
            continue
        for field_name in _COUNTER_FIELDS:
            counters[field_name].update(item.get(field_name, {}))
        for field_name in _INTEGER_FIELDS:
            integers[field_name] += int(item.get(field_name, 0) or 0)

    merged: dict[str, dict | int] = {}
    for field_name, counter in counters.items():
        merged[field_name] = _sorted_counter_dict(
            counter,
            round_values=field_name in _TIMING_FIELDS,
        )
    merged.update(integers)
    return merged


def compact_telemetry_summary(telemetry: dict, telemetry_path) -> dict:
    """Return compact benchmark summary telemetry and keep full details external."""

    return {
        "path": str(telemetry_path),
        "top_rule_seconds": _top_items(telemetry.get("rule_seconds", {})),
        "top_candidate_count": _top_items(telemetry.get("candidate_count", {})),
        "top_rendered_candidate_count": _top_items(telemetry.get("rendered_candidate_count", {})),
        "top_selected_family": _top_items(telemetry.get("selected_family", {})),
        "top_selected_rule_family": _top_items(telemetry.get("selected_rule_family", {})),
        "top_template_candidate_count": _top_items(telemetry.get("template_candidate_count", {})),
        "top_branch_prefix_produced": _top_items(telemetry.get("branch_prefix_produced", {})),
        "top_branch_prefix_calls_by_rule": _top_items(telemetry.get("branch_prefix_calls_by_rule", {})),
        "top_branch_prefix_cache_hits_by_rule": _top_items(
            telemetry.get("branch_prefix_cache_hits_by_rule", {})
        ),
        "top_branch_prefix_cache_misses_by_rule": _top_items(
            telemetry.get("branch_prefix_cache_misses_by_rule", {})
        ),
        "total_candidate_facts": _sum_values(telemetry.get("candidate_facts_total", {})),
        "total_facts_suppressed_before_render": _sum_values(
            telemetry.get("facts_suppressed_before_render", {})
        ),
        "total_rendered_candidates": _sum_values(telemetry.get("rendered_candidate_count", {})),
        "total_rejected_candidates": _sum_values(telemetry.get("rejected_candidate_count", {})),
        "total_branch_prefix_calls": int(telemetry.get("branch_prefix_calls", 0) or 0),
        "total_branch_prefix_cache_hits": int(telemetry.get("branch_prefix_cache_hits", 0) or 0),
        "total_branch_prefix_cache_misses": int(telemetry.get("branch_prefix_cache_misses", 0) or 0),
        "total_branch_prefix_in_progress_rejections": int(
            telemetry.get("branch_prefix_in_progress_rejections", 0) or 0
        ),
        "total_substructure_match_calls": int(telemetry.get("substructure_match_calls", 0) or 0),
    }


def _sorted_counter_dict(counter, *, round_values: bool = False) -> dict:
    items = sorted(counter.items(), key=lambda item: str(item[0]))
    if round_values:
        return {str(key): round(float(value), 6) for key, value in items if value}
    return {str(key): int(value) for key, value in items if value}


def _top_items(counter: dict, *, limit: int = 10) -> list[dict[str, float | int | str]]:
    items = sorted(
        counter.items(),
        key=lambda item: (-float(item[1]), str(item[0])),
    )[:limit]
    return [{"key": str(key), "value": value} for key, value in items]


def _sum_values(counter: dict) -> int:
    return int(sum(int(value) for value in counter.values()))
