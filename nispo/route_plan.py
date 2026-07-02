from __future__ import annotations

from dataclasses import dataclass

from .molecule_facts import MoleculeFacts


ACYCLIC_SIMPLE = "acyclic_simple"
ACYCLIC_SENIOR_FUNCTIONAL = "acyclic_senior_functional"
SIMPLE_MONOCYCLE = "simple_monocycle"
RETAINED_RING = "retained_ring"
MULTI_MONOCYCLE = "multi_monocycle"
FUSED_POLYCYCLIC = "fused_polycyclic"
SPIRO = "spiro"
BRIDGED = "bridged"
CHARGED = "charged"
HARD_FALLBACK = "hard_fallback"


@dataclass(frozen=True, slots=True)
class RoutePlan:
    """Graph-first routing decision for the current molecule.

    The plan is observational for now: existing legacy rule execution remains
    unchanged while telemetry and future compiled dispatch consume this surface.
    """

    primary_route: str
    routes: tuple[str, ...]
    uses_fused_resources: bool
    uses_polycyclic_search: bool
    legacy_fallback_allowed: bool = True

    @property
    def telemetry_key(self) -> str:
        return "+".join(self.routes)


def plan_routes(facts: MoleculeFacts) -> RoutePlan:
    routes: list[str] = []
    if facts.has_formal_charge:
        routes.append(CHARGED)
    if not facts.allowed_organic_element_class or facts.radical_atom_count or facts.isotope_atom_count:
        routes.append(HARD_FALLBACK)

    if facts.ring_count == 0:
        if facts.has_functional_skeleton_signal or facts.has_imino_skeleton_signal:
            routes.append(ACYCLIC_SENIOR_FUNCTIONAL)
        else:
            routes.append(ACYCLIC_SIMPLE)
    elif facts.has_component_spiro_topology:
        routes.append(SPIRO)
    elif facts.max_ring_system_cycle_rank >= 3:
        routes.append(BRIDGED)
        routes.append(FUSED_POLYCYCLIC)
    elif facts.has_polycyclic_ring_system:
        routes.append(FUSED_POLYCYCLIC)
    elif facts.ring_system_count > 1:
        routes.append(MULTI_MONOCYCLE)
    elif facts.ring_count == 1:
        routes.append(SIMPLE_MONOCYCLE)
    else:
        routes.append(RETAINED_RING)

    routes = _dedupe(routes)
    primary = _primary_route(routes)
    return RoutePlan(
        primary_route=primary,
        routes=tuple(routes),
        uses_fused_resources=any(route in {FUSED_POLYCYCLIC, SPIRO, BRIDGED} for route in routes),
        uses_polycyclic_search=any(route in {FUSED_POLYCYCLIC, SPIRO, BRIDGED} for route in routes),
    )


def _dedupe(routes: list[str]) -> list[str]:
    seen = set()
    result = []
    for route in routes:
        if route in seen:
            continue
        seen.add(route)
        result.append(route)
    return result or [HARD_FALLBACK]


def _primary_route(routes: list[str]) -> str:
    for route in routes:
        if route not in {CHARGED, HARD_FALLBACK}:
            return route
    return routes[0] if routes else HARD_FALLBACK
