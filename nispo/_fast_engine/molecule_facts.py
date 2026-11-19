from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


_ORGANIC_ROUTE_ELEMENTS = frozenset(
    {
        "B",
        "Br",
        "C",
        "Cl",
        "F",
        "I",
        "N",
        "O",
        "P",
        "S",
        "Se",
        "Si",
    }
)


@dataclass(frozen=True, slots=True)
class MoleculeFacts:
    """Immutable graph facts shared by route planning and future rule dispatch."""

    atom_count: int
    heavy_atom_count: int
    nonring_heavy_atom_count: int
    ring_heavy_atom_count: int
    element_counts: tuple[tuple[str, int], ...]
    ring_count: int
    ring_system_count: int
    max_ring_system_size: int
    max_ring_system_ring_count: int
    max_ring_system_cycle_rank: int
    all_atoms_in_one_ring_system: bool
    aromatic_atom_count: int
    hetero_atom_count: int
    nonhalogen_hetero_atom_count: int
    chiral_atom_count: int
    formal_charge_atom_count: int
    abs_formal_charge_sum: int
    radical_atom_count: int
    isotope_atom_count: int
    has_spiro_topology: bool
    has_component_spiro_topology: bool
    has_functional_skeleton_signal: bool
    has_imino_skeleton_signal: bool
    has_acyclic_preference_signal: bool
    may_have_chain_dominant_acyclic_parent: bool
    has_kekulized_polycycle_signal: bool
    has_charged_ring_atom: bool
    has_generic_bicyclic_topology: bool
    has_generic_polycyclic_topology: bool
    has_small_inorganic_retained_signal: bool
    has_sulfur_oxo_retained_parent_signal: bool
    has_terminal_n_oxide: bool
    has_oxygen_oxygen_single_bond: bool
    has_oxirane_dione_signal: bool
    has_nitrile_oxide_signal: bool
    has_oxime_sulfonate_nitrile_signal: bool
    has_terminal_carboxylate_signal: bool
    has_amide_acyl_motif: bool
    has_dithiocarbamate_signal: bool
    has_conjugated_carbonyl_diene_signal: bool
    has_nonring_multiple_bond: bool
    has_nitrogen: bool
    has_oxygen: bool
    has_sulfur: bool
    has_phosphorus: bool
    has_boron: bool
    has_iodine: bool
    has_group14: bool

    @property
    def element_symbols(self) -> frozenset[str]:
        return frozenset(symbol for symbol, count in self.element_counts if count)

    @property
    def allowed_organic_element_class(self) -> bool:
        return self.element_symbols <= _ORGANIC_ROUTE_ELEMENTS

    @property
    def has_formal_charge(self) -> bool:
        return self.formal_charge_atom_count > 0

    @property
    def has_ring_system(self) -> bool:
        return self.ring_system_count > 0

    @property
    def has_polycyclic_ring_system(self) -> bool:
        return self.max_ring_system_cycle_rank >= 2 or self.max_ring_system_ring_count >= 2


def build_molecule_facts(
    mol,
    *,
    features: dict,
    ring_topologies,
    profile,
) -> MoleculeFacts:
    """Build reusable graph facts from already-centralized molecule views."""

    atoms = tuple(mol.GetAtoms())
    element_counter = Counter(atom.GetSymbol() for atom in atoms)
    nonring_heavy_atom_count = sum(1 for atom in atoms if atom.GetAtomicNum() > 1 and not atom.IsInRing())
    ring_heavy_atom_count = sum(1 for atom in atoms if atom.GetAtomicNum() > 1 and atom.IsInRing())
    element_counts = tuple(
        sorted(
            (symbol, count)
            for symbol, count in element_counter.items()
            if count
        )
    )
    ring_topologies = tuple(ring_topologies)
    max_cycle_rank = max((topology.cycle_rank for topology in ring_topologies), default=0)
    has_spiro_topology, has_component_spiro_topology = _spiro_topology_flags(mol, ring_topologies)
    return MoleculeFacts(
        atom_count=mol.GetNumAtoms(),
        heavy_atom_count=mol.GetNumHeavyAtoms(),
        nonring_heavy_atom_count=nonring_heavy_atom_count,
        ring_heavy_atom_count=ring_heavy_atom_count,
        element_counts=element_counts,
        ring_count=mol.GetRingInfo().NumRings(),
        ring_system_count=len(features.get("ring_systems", ())),
        max_ring_system_size=int(features.get("max_ring_system_size", 0) or 0),
        max_ring_system_ring_count=int(features.get("max_ring_system_ring_count", 0) or 0),
        max_ring_system_cycle_rank=max_cycle_rank,
        all_atoms_in_one_ring_system=bool(features.get("all_atoms_in_one_ring_system", False)),
        aromatic_atom_count=sum(1 for atom in atoms if atom.GetIsAromatic()),
        hetero_atom_count=sum(1 for atom in atoms if atom.GetAtomicNum() not in {1, 6}),
        nonhalogen_hetero_atom_count=sum(
            1 for atom in atoms if atom.GetAtomicNum() not in {1, 6, 9, 17, 35, 53, 85}
        ),
        chiral_atom_count=sum(int(atom.GetChiralTag()) != 0 for atom in atoms),
        formal_charge_atom_count=sum(1 for atom in atoms if atom.GetFormalCharge()),
        abs_formal_charge_sum=sum(abs(atom.GetFormalCharge()) for atom in atoms),
        radical_atom_count=sum(atom.GetNumRadicalElectrons() for atom in atoms),
        isotope_atom_count=sum(1 for atom in atoms if atom.GetIsotope()),
        has_spiro_topology=has_spiro_topology,
        has_component_spiro_topology=has_component_spiro_topology,
        has_functional_skeleton_signal=bool(profile.has_functional_skeleton_signal),
        has_imino_skeleton_signal=bool(profile.has_imino_skeleton_signal),
        has_acyclic_preference_signal=bool(profile.may_prefer_acyclic_skeleton),
        may_have_chain_dominant_acyclic_parent=_may_have_chain_dominant_acyclic_parent(
            mol,
            nonring_heavy_atom_count=nonring_heavy_atom_count,
            ring_heavy_atom_count=ring_heavy_atom_count,
        ),
        has_kekulized_polycycle_signal=bool(profile.has_kekulized_polycycle_signal),
        has_charged_ring_atom=bool(profile.has_charged_ring_atom),
        has_generic_bicyclic_topology=any(topology.cycle_rank == 2 for topology in ring_topologies),
        has_generic_polycyclic_topology=any(2 <= topology.cycle_rank <= 9 for topology in ring_topologies),
        has_small_inorganic_retained_signal=bool(profile.has_small_inorganic_retained_signal),
        has_sulfur_oxo_retained_parent_signal=bool(profile.has_sulfur_oxo_retained_parent_signal),
        has_terminal_n_oxide=bool(profile.has_terminal_n_oxide),
        has_oxygen_oxygen_single_bond=bool(profile.has_oxygen_oxygen_single_bond),
        has_oxirane_dione_signal=bool(profile.has_oxirane_dione_signal),
        has_nitrile_oxide_signal=bool(profile.has_nitrile_oxide_signal),
        has_oxime_sulfonate_nitrile_signal=bool(profile.has_oxime_sulfonate_nitrile_signal),
        has_terminal_carboxylate_signal=bool(profile.has_terminal_carboxylate_signal),
        has_amide_acyl_motif=bool(profile.has_amide_acyl_motif),
        has_dithiocarbamate_signal=bool(profile.has_dithiocarbamate_signal),
        has_conjugated_carbonyl_diene_signal=bool(profile.has_conjugated_carbonyl_diene_signal),
        has_nonring_multiple_bond=bool(profile.has_nonring_multiple_bond),
        has_nitrogen=bool(profile.has_nitrogen),
        has_oxygen=bool(profile.has_oxygen),
        has_sulfur=bool(profile.has_sulfur),
        has_phosphorus=bool(element_counter.get("P", 0)),
        has_boron=bool(profile.has_boron),
        has_iodine=bool(profile.has_iodine),
        has_group14=any(element_counter.get(symbol, 0) for symbol in ("Si", "Ge", "Sn", "Pb")),
    )


def _spiro_topology_flags(mol, ring_topologies) -> tuple[bool, bool]:
    has_route_spiro = False
    has_component_spiro = False
    rings = tuple(frozenset(ring) for ring in mol.GetRingInfo().AtomRings())
    for topology in ring_topologies:
        if topology.cycle_rank < 2:
            continue
        system_rings = [ring for ring in rings if ring <= topology.atoms]
        if any(
            len(left & right) == 1
            for idx, left in enumerate(system_rings)
            for right in system_rings[idx + 1 :]
        ):
            has_component_spiro = True
            if topology.cycle_rank <= 3:
                has_route_spiro = True
            if has_route_spiro and has_component_spiro:
                break
    return has_route_spiro, has_component_spiro


def _may_have_chain_dominant_acyclic_parent(
    mol,
    *,
    nonring_heavy_atom_count: int,
    ring_heavy_atom_count: int,
) -> bool:
    """Cheap graph superset for chain-dominant acyclic parent rendering."""

    if not ring_heavy_atom_count or nonring_heavy_atom_count < 2:
        return False
    if nonring_heavy_atom_count >= ring_heavy_atom_count:
        return True
    if mol.GetRingInfo().NumRings() > 1 and nonring_heavy_atom_count >= 4:
        return True
    nonring_atoms = frozenset(
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() > 1 and not atom.IsInRing()
    )
    if any(mol.GetAtomWithIdx(idx).GetSymbol() != "C" for idx in nonring_atoms):
        return True
    return any(
        bond.GetBondTypeAsDouble() > 1.0
        and bond.GetBeginAtomIdx() in nonring_atoms
        and bond.GetEndAtomIdx() in nonring_atoms
        for bond in mol.GetBonds()
    )
