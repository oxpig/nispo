"""Public domain boundaries from independently declared source operations.

Origin: elementary ribose/glycerol/phosphate condensations, a three-member
glycosyl cycle, the mathematical C60 fullerane cage and malformed operation
declarations. These were authored from the public schemas and general chemical
stoichiometry, not a panel, reference-name list or molecule-level failures.
"""
from __future__ import annotations

import pytest
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


pytestmark = pytest.mark.full_only


def _declined(result, status="invalid-input", stage=None):
    assert result.generation_status == status
    assert not result.name
    assert result.preferred_status == "not-assessed"
    assert result.unresolved
    assert all(issue.stage and issue.rule and issue.reason for issue in result.unresolved)
    if stage is not None:
        assert all(issue.stage == stage for issue in result.unresolved)


def _mapped(result):
    assert result.generation_status == "available", result.unresolved
    assert result.name
    graph = Chem.MolFromSmiles(result.mapped_smiles)
    assert graph is not None
    numbers = [atom.GetAtomMapNum() for atom in graph.GetAtoms()]
    assert all(numbers) and len(set(numbers)) == len(numbers)
    atoms = {atom.GetAtomMapNum(): atom for atom in graph.GetAtoms()}
    unlabelled = Chem.Mol(graph)
    for atom in unlabelled.GetAtoms():
        atom.SetAtomMapNum(0)
    assert Chem.MolToSmiles(unlabelled, isomericSmiles=True) == result.canonical_smiles
    return graph, atoms


def _ribose(anomer="beta"):
    from nispo import CarbohydrateSpec
    return CarbohydrateSpec("D-ribose", ring="furanose", anomer=anomer)


@pytest.mark.parametrize("case", ["too-short", "boolean-count", "zero-port", "unassigned-anomer", "unknown-method", "anomeric-port"])
def test_cyclic_glycosyl_schema_requires_a_nonanomeric_port_and_assigned_ring(case):
    from nispo import CyclicOligosaccharideSpec, cyclic_oligosaccharide_to_iupac_result

    residue = _ribose("unspecified" if case == "unassigned-anomer" else "beta")
    options = {
        "too-short": {"repeat_count": 2},
        "boolean-count": {"repeat_count": True},
        "zero-port": {"acceptor_position": 0},
        "unassigned-anomer": {},
        "unknown-method": {"naming_method": "unlisted"},
        "anomeric-port": {"acceptor_position": 1},
    }[case]
    spec = CyclicOligosaccharideSpec(residue, **({"repeat_count": 3, "acceptor_position": 2} | options))
    _declined(cyclic_oligosaccharide_to_iupac_result(spec), stage="oligosaccharide-construction")


def test_three_ribose_cycle_closes_all_donor_ports_and_loses_three_waters():
    from nispo import CyclicOligosaccharideSpec, cyclic_oligosaccharide_to_iupac_result

    request = CyclicOligosaccharideSpec(_ribose(), 3, 2, "systematic")
    before = repr(request)
    result = cyclic_oligosaccharide_to_iupac_result(request)
    graph, atoms = _mapped(result)
    assert repr(request) == before
    assert rdMolDescriptors.CalcMolFormula(graph) == "C15H24O12"
    assert result.is_cyclic and result.cyclic_repeat_count == 3
    assert result.reducing_residue is None and len(result.glycosidic_bindings) == 3
    assert {binding.donor_residue for binding in result.glycosidic_bindings} == {0, 1, 2}
    assert {binding.acceptor_residue for binding in result.glycosidic_bindings} == {0, 1, 2}
    for binding in result.glycosidic_bindings:
        oxygen = atoms[binding.bridge_oxygen_map]
        assert oxygen.GetAtomicNum() == 8 and oxygen.GetDegree() == 2
        assert {atom.GetAtomMapNum() for atom in oxygen.GetNeighbors()} == {binding.donor_carbon_map, binding.acceptor_carbon_map}


@pytest.mark.parametrize("case", ["unknown-head", "unassigned-frame", "boolean-site", "occupied-site"])
def test_phosphoglycerol_rejects_an_unproved_head_or_already_consumed_oxygen(case):
    from nispo import GlycerolAcyl, PhosphoglycerolSpec, phosphoglycerol_to_iupac_result

    options = {
        "unknown-head": {"head_group": "unlisted"},
        "unassigned-frame": {"configuration": "rac"},
        "boolean-site": {"phosphate_position": True},
        "occupied-site": {"acyl_groups": (GlycerolAcyl(3, "CC(=O)O"),)},
    }[case]
    _declined(phosphoglycerol_to_iupac_result(PhosphoglycerolSpec(**options)), stage="phosphoglycerol-construction")


@pytest.mark.parametrize(("head", "formula", "acid_h"), [
    ("none", "C3H9O6P", 2), ("2-aminoethanol", "C5H14NO6P", 1), ("L-serine", "C6H14NO8P", 1),
])
def test_neutral_phosphate_head_condensation_preserves_oxygen_ports_and_acid_hydrogens(head, formula, acid_h):
    from nispo import PhosphoglycerolSpec, phosphoglycerol_to_iupac_result

    result = phosphoglycerol_to_iupac_result(PhosphoglycerolSpec(head_group=head))
    graph, atoms = _mapped(result)
    assert rdMolDescriptors.CalcMolFormula(graph) == formula
    assert not any(atom.GetFormalCharge() for atom in graph.GetAtoms())
    binding = result.phosphate_binding
    phosphorus = atoms[binding.phosphorus_map]
    assert phosphorus.GetAtomicNum() == 15
    assert result.acid_hydrogen_count == acid_h == len(binding.acidic_atom_maps)
    assert all(atoms[number].GetTotalNumHs() == 1 for number in binding.acidic_atom_maps)
    glycerol_oxygen = atoms[binding.glycerol_oxygen_map]
    assert glycerol_oxygen.GetAtomicNum() == 8 and glycerol_oxygen.GetDegree() == 2
    assert graph.GetBondBetweenAtoms(glycerol_oxygen.GetIdx(), phosphorus.GetIdx()) is not None
    assert (binding.head_oxygen_map is None) == (head == "none")
    if binding.head_oxygen_map is not None:
        oxygen = atoms[binding.head_oxygen_map]
        assert oxygen.GetAtomicNum() == 8 and oxygen.GetDegree() == 2
        assert graph.GetBondBetweenAtoms(oxygen.GetIdx(), phosphorus.GetIdx()) is not None


@pytest.mark.parametrize("case", ["missing-request", "empty-glycerol", "occupied-carrier-port", "missing-sphingoid"])
def test_glycolipid_requires_a_typed_carrier_with_a_free_declared_join_port(case):
    from nispo import GlycerolAcyl, GlycerolSpec, GlycoglycerolSpec, GlycosphingoidSpec, glycolipid_to_iupac_result

    requests = {
        "missing-request": None,
        "empty-glycerol": GlycoglycerolSpec(_ribose(), GlycerolSpec()),
        "occupied-carrier-port": GlycoglycerolSpec(_ribose(), GlycerolSpec((GlycerolAcyl(3, "CC(=O)O"),))),
        "missing-sphingoid": GlycosphingoidSpec(_ribose(), None),
    }
    _declined(glycolipid_to_iupac_result(requests[case]), stage="glycolipid-construction")


@pytest.mark.parametrize(("kind", "formula"), [("glycerol", "C10H18O8"), ("sphingoid", "C23H47NO6")])
def test_single_glycolipid_join_keeps_carrier_oxygen_and_complete_component_partition(kind, formula):
    from nispo import GlycerolAcyl, GlycerolSpec, GlycoglycerolSpec, GlycosphingoidSpec, SphingoidSpec, glycolipid_to_iupac_result

    request = (GlycoglycerolSpec(_ribose(), GlycerolSpec((GlycerolAcyl(1, "CC(=O)O"),))) if kind == "glycerol"
               else GlycosphingoidSpec(_ribose(), SphingoidSpec()))
    result = glycolipid_to_iupac_result(request)
    graph, atoms = _mapped(result)
    assert rdMolDescriptors.CalcMolFormula(graph) == formula
    assert result.lipid_kind == kind
    groups = [set(numbers) for _owner, numbers in result.component_atom_maps]
    assert sum(map(len, groups)) == len(set().union(*groups)) == graph.GetNumAtoms()
    assert set().union(*groups) == set(atoms)
    binding = result.glycosidic_binding
    oxygen = atoms[binding.bridge_oxygen_map]
    assert binding.bridge_oxygen_map in dict(result.lipid_atom_maps).values()
    assert oxygen.GetAtomicNum() == 8 and oxygen.GetDegree() == 2
    assert {atom.GetAtomMapNum() for atom in oxygen.GetNeighbors()} == {binding.anomeric_carbon_map, binding.aglycone_carbon_map}


@pytest.mark.parametrize("case", ["unknown-parent", "invalid-C5", "mutable-operations", "duplicate-homo", "unsupported-element"])
def test_natural_product_schema_declines_ambiguous_or_unavailable_operations_before_construction(case):
    from nispo import NaturalProductHomo, NaturalProductReplacement, NaturalProductSpec, natural_product_to_iupac_result

    options = {
        "unknown-parent": {"parent": "unlisted-parent"},
        "invalid-C5": {"c5_configuration": "opposite"},
        "mutable-operations": {"nor": ["1"]},
        "duplicate-homo": {"homo": (NaturalProductHomo("2"), NaturalProductHomo("2", 2))},
        "unsupported-element": {"replacements": (NaturalProductReplacement("2", "Fe"),)},
    }[case]
    _declined(natural_product_to_iupac_result(NaturalProductSpec(**options)), stage="natural-product-construction")


@pytest.mark.parametrize(("case", "status"), [
    ("missing-request", "invalid-input"), ("unsupported-parent", "unavailable"),
    ("invalid-saturation", "invalid-input"), ("mutable-operations", "invalid-input"), ("boolean-site", "invalid-input"),
])
def test_fullerene_schema_rejects_unavailable_parents_and_nonvertex_operation_declarations(case, status):
    from nispo import FullereneSpec, fullerene_to_iupac_result

    requests = {
        "missing-request": None,
        "unsupported-parent": FullereneSpec(parent="C70"),
        "invalid-saturation": FullereneSpec(saturation="partial"),
        "mutable-operations": FullereneSpec(nor=[1]),
        "boolean-site": FullereneSpec(nor=(True,)),
    }
    _declined(fullerene_to_iupac_result(requests[case]), status)


def test_saturated_C60_cage_has_sixty_carbons_and_one_hydrogen_per_vertex():
    from nispo import FullereneSpec, fullerene_to_iupac_result

    result = fullerene_to_iupac_result(FullereneSpec(saturation="fullerane"))
    graph, _atoms = _mapped(result)
    assert rdMolDescriptors.CalcMolFormula(graph) == "C60H60"
    assert graph.GetNumAtoms() == 60 and graph.GetNumBonds() == 90
    assert all(atom.GetAtomicNum() == 6 and atom.GetDegree() == 3 and atom.GetTotalNumHs() == 1 for atom in graph.GetAtoms())
    assert result.double_bond_count == 0 and result.hydrogen_count == 60
    assert result.graph_atom_count == 60 and len(result.parent_atom_maps) == 60


@pytest.mark.parametrize(("text", "status"), [("   ", "invalid-input"), ("CCO.CCO", "unavailable")])
def test_repeated_parent_does_not_invent_a_connected_core_path(text, status):
    from nispo import repeated_parent_to_iupac_result
    _declined(repeated_parent_to_iupac_result(text), status, "nearest-repeated-parent")
