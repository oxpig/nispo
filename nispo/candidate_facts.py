from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


GRAPH_INCOMPLETE = "graph_incomplete"
OPSIN_FRAGILE_BENZIMIDAZOLE_ORIENTATION = "opsin_fragile_benzimidazole_orientation"
OPSIN_FRAGILE_BENZOANTHRACENE = "opsin_fragile_benzoanthracene"
OPSIN_FRAGILE_EXOCYCLIC_CHARGED_RING_SUFFIX = "opsin_fragile_exocyclic_charged_ring_suffix"
OPSIN_FRAGILE_GROUP14_HETERO_CARBON_LIGAND = "opsin_fragile_group14_hetero_carbon_ligand"
OPSIN_FRAGILE_OXYIMINO_CARBONYL_PARENT = "opsin_fragile_oxyimino_carbonyl_parent"
OPSIN_FRAGILE_POLYCYCLIC_BRIDGE_UNSATURATION = "opsin_fragile_polycyclic_bridge_unsaturation"
UNSUPPORTED_CHARGE_STATE = "unsupported_charge_state"
UNSUPPORTED_H_STATE = "unsupported_h_state"
UNSUPPORTED_STEREO = "unsupported_stereo"


@dataclass(frozen=True, slots=True)
class CandidateSafety:
    graph_complete: bool = True
    charge_supported: bool = True
    hydrogen_state_supported: bool = True
    stereo_supported: bool = True
    opsin_fragility_reasons: tuple[str, ...] = ()
    unsupported_reasons: tuple[str, ...] = ()

    @property
    def reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if not self.graph_complete:
            reasons.append(GRAPH_INCOMPLETE)
        if not self.charge_supported:
            reasons.append(UNSUPPORTED_CHARGE_STATE)
        if not self.hydrogen_state_supported:
            reasons.append(UNSUPPORTED_H_STATE)
        if not self.stereo_supported:
            reasons.append(UNSUPPORTED_STEREO)
        reasons.extend(self.opsin_fragility_reasons)
        reasons.extend(self.unsupported_reasons)
        return tuple(reasons)


@dataclass(frozen=True, slots=True)
class PrefixCandidate:
    text: str
    atoms: frozenset[int] = frozenset()
    attachment_atom: int | None = None
    parent_atom: int | None = None
    locant: str | None = None
    kind: str = ""
    source: str = ""
    safety: CandidateSafety = CandidateSafety()
    estimated_length: int | None = None


@dataclass(frozen=True, slots=True)
class SuffixCandidate:
    text: str
    atoms: frozenset[int] = frozenset()
    locants: tuple[str, ...] = ()
    kind: str = ""
    source: str = ""
    safety: CandidateSafety = CandidateSafety()


@dataclass(frozen=True, slots=True)
class ChargeEvidence:
    formal_charge: int = 0
    charged_atoms: tuple[int, ...] = ()
    rendered_terms: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class HydrogenEvidence:
    indicated_h_locs: tuple[str, ...] = ()
    saturated_locs: tuple[str, ...] = ()
    unsupported: bool = False


@dataclass(frozen=True, slots=True)
class StereoEvidence:
    atom_descriptors: tuple[tuple[int, str], ...] = ()
    bond_descriptors: tuple[tuple[int, str], ...] = ()
    rendered_prefix: str = ""
    unsupported: bool = False


def normalized_locants(locants: Mapping[int, str] | None) -> tuple[tuple[int, str], ...]:
    if not locants:
        return ()
    return tuple(sorted((int(key), str(value)) for key, value in locants.items()))
