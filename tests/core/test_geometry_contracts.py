"""Small public geometry contracts with independently authored inputs.

Provenance: the four-atom F-C-C-Cl torsion, terminally halogenated allene,
ortho-halogenated biaryl and ideal tetrahedron/square are elementary geometric
constructions chosen for these contracts. They were not selected from a panel,
benchmark disagreement, reference-name list or generated development corpus.
The tests compare geometry, retained graphs and public status/identity fields.
"""
from dataclasses import replace
from math import sqrt

import pytest


pytestmark = pytest.mark.full_only

_TETRAHEDRON = ((1., 1., 1.), (1., -1., -1.), (-1., 1., -1.), (-1., -1., 1.))
_SQUARE = ((1., 0., 0.), (0., 1., 0.), (-1., 0., 0.), (0., -1., 0.))


def _with_coordinates(smiles, coordinates, *, is_3d=True):
    from rdkit import Chem

    graph = Chem.MolFromSmiles(smiles)
    assert graph is not None and graph.GetNumAtoms() == len(coordinates)
    conformer = Chem.Conformer(graph.GetNumAtoms())
    conformer.Set3D(is_3d)
    for index, point in enumerate(coordinates):
        conformer.SetAtomPosition(index, point)
    graph.AddConformer(conformer)
    graph.SetProp('independent_geometry_control', 'preserve')
    return graph


def _torsion_graph():
    return _with_coordinates('FCCCl', ((1., 0., 0.), (0., 0., 0.),
                                       (0., 0., 1.), (.5, sqrt(3.)/2., 1.)))


def _allene_graph():
    return _with_coordinates('FC=C=CCl', ((1., 0., 0.), (0., 0., 0.),
                                         (0., 0., .5), (0., 0., 1.), (0., 1., 1.)))


@pytest.fixture(scope='module')
def tetrahedral_control():
    from rdkit import Chem
    from nispo import PolyhedralGeometry, describe_polyhedral_configuration

    graph = Chem.MolFromSmiles('[Si](F)(Cl)(Br)I')
    geometry = PolyhedralGeometry('T-4', (1, 2, 3, 4), _TETRAHEDRON)
    return graph, geometry, describe_polyhedral_configuration(graph, 0, geometry)


def test_conformer_retains_source_coordinates_references_and_graph(assert_graphs):
    from rdkit import Chem
    from nispo import TorsionReference, describe_conformer

    graph = _torsion_graph()
    before = graph.ToBinary(Chem.PropertyPickleOptions.AllProps)
    result = describe_conformer(graph, references=(TorsionReference((0, 1, 2, 3)),),
                                axes=(), rings=())
    assert graph.ToBinary(Chem.PropertyPickleOptions.AllProps) == before
    assert result.status == 'available'
    assert result.canonical_smiles == Chem.MolToSmiles(graph)
    assert result.coordinates == tuple(tuple(graph.GetConformer().GetAtomPosition(i)) for i in range(4))
    assert result.torsions[0].reference_atoms == (0, 1, 2, 3)
    assert result.torsions[0].status == 'complete'
    assert result.torsions[0].angle_degrees is not None
    assert result.molecular_result.generation_status in {'available', 'timeout'}
    if result.molecular_result.generation_status == 'available':
        assert result.molecular_name
        assert_graphs(('FCCCl',), (result.molecular_name,))
    else:
        assert not result.molecular_name and result.molecular_result.unresolved


def test_conformer_torsion_survives_rigid_coordinate_changes():
    from nispo import TorsionReference, describe_conformer

    graph = _torsion_graph()
    first = describe_conformer(graph, references=(TorsionReference((0, 1, 2, 3)),), axes=(), rings=())
    conformer = graph.GetConformer()
    for index in range(graph.GetNumAtoms()):
        x, y, z = conformer.GetAtomPosition(index)
        conformer.SetAtomPosition(index, (z+7., x-2., y+3.))
    second = describe_conformer(graph, references=(TorsionReference((0, 1, 2, 3)),), axes=(), rings=())
    assert first.status == second.status == 'available'
    assert first.canonical_smiles == second.canonical_smiles
    assert first.torsions[0].angle_degrees == pytest.approx(second.torsions[0].angle_degrees)
    assert first.coordinates != second.coordinates


@pytest.mark.parametrize('case', [
    'missing-conformer', 'two-dimensional', 'unknown-conformer', 'boolean-conformer',
    'zero-distance-tolerance', 'nonfinite-angular-tolerance', 'unbonded-axis',
    'invalid-reference', 'noncycle-ring',
])
def test_conformer_rejects_invalid_geometry_as_structured_input(case):
    from rdkit import Chem
    from nispo import TorsionReference, describe_conformer

    graph = _torsion_graph()
    options = {'axes': (), 'rings': ()}
    if case == 'missing-conformer':
        graph.RemoveAllConformers()
    elif case == 'two-dimensional':
        graph.GetConformer().Set3D(False)
    elif case == 'unknown-conformer':
        options['conformer_id'] = 999
    elif case == 'boolean-conformer':
        options['conformer_id'] = True
    elif case == 'zero-distance-tolerance':
        options['distance_tolerance'] = 0
    elif case == 'nonfinite-angular-tolerance':
        options['angular_tolerance'] = float('nan')
    elif case == 'unbonded-axis':
        options['axes'] = ((0, 3),)
    elif case == 'invalid-reference':
        options['references'] = (TorsionReference((0, 1, 2, 4)),)
    else:
        options['rings'] = ((0, 1, 2, 3),)
    before = graph.ToBinary(Chem.PropertyPickleOptions.AllProps)
    result = describe_conformer(graph, **options)
    assert result.status == 'invalid-input'
    assert not result.molecular_name and result.limitations
    assert graph.ToBinary(Chem.PropertyPickleOptions.AllProps) == before


def test_degenerate_torsion_is_partial_without_losing_molecular_identity():
    from nispo import TorsionReference, describe_conformer

    graph = _torsion_graph()
    graph.GetConformer().SetAtomPosition(0, (0., 0., -1.))
    result = describe_conformer(graph, references=(TorsionReference((0, 1, 2, 3)),), axes=(), rings=())
    assert result.status == 'partial' and result.canonical_smiles
    assert result.torsions[0].status == 'unavailable'
    assert result.torsions[0].angle_degrees is None and result.limitations


def test_tetrahedral_description_retains_native_stereo_and_constitution(tetrahedral_control):
    from rdkit import Chem

    graph, geometry, result = tetrahedral_control
    assert result.status == 'complete' and not result.limitations
    assert result.center_atom == 0 and result.donor_atoms == geometry.donor_atoms
    assert result.directions == geometry.directions and result.chirality_symbol in {'R', 'S'}
    assert result.maximum_angular_error_degrees == pytest.approx(0., abs=1e-5)
    recovered = Chem.MolFromSmiles(result.native_stereo_smiles)
    assert recovered is not None
    assert any(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for atom in recovered.GetAtoms())
    Chem.RemoveStereochemistry(recovered)
    assert Chem.MolToSmiles(recovered) == Chem.MolToSmiles(graph)


def test_polyhedral_rotation_and_donor_permutation_preserve_configuration(tetrahedral_control):
    from nispo import describe_polyhedral_configuration

    graph, geometry, first = tetrahedral_control
    rotated = replace(geometry, directions=tuple((z, x, y) for x, y, z in geometry.directions))
    permuted = replace(geometry, donor_atoms=tuple(reversed(geometry.donor_atoms)),
                       directions=tuple(reversed(geometry.directions)))
    second = describe_polyhedral_configuration(graph, 0, rotated)
    third = describe_polyhedral_configuration(graph, 0, permuted)
    assert first.status == second.status == third.status == 'complete'
    assert first.descriptor == second.descriptor == third.descriptor
    assert third.priority_numbers == tuple(reversed(first.priority_numbers))


def test_polyhedral_reflection_changes_configuration_not_constitution(tetrahedral_control):
    from nispo import describe_polyhedral_configuration

    graph, geometry, first = tetrahedral_control
    reflected = replace(geometry, directions=tuple((-x, y, z) for x, y, z in geometry.directions))
    second = describe_polyhedral_configuration(graph, 0, reflected)
    assert second.status == 'complete'
    assert {first.chirality_symbol, second.chirality_symbol} == {'R', 'S'}
    assert first.molecular_smiles == second.molecular_smiles
    assert first.native_stereo_smiles != second.native_stereo_smiles


@pytest.mark.parametrize('case', [
    'boolean-center', 'unknown-center', 'missing-model', 'unknown-symbol',
    'duplicate-donors', 'missing-direction', 'nonfinite-direction', 'zero-direction',
    'boolean-tolerance', 'excessive-tolerance',
])
def test_polyhedral_input_errors_do_not_deliver_a_configuration(case, tetrahedral_control):
    from rdkit import Chem
    from nispo import describe_polyhedral_configuration

    graph, geometry, _ = tetrahedral_control
    center = 0
    if case == 'boolean-center':
        center = True
    elif case == 'unknown-center':
        center = graph.GetNumAtoms()
    elif case == 'missing-model':
        geometry = None
    elif case == 'unknown-symbol':
        geometry = replace(geometry, symbol='unknown')
    elif case == 'duplicate-donors':
        geometry = replace(geometry, donor_atoms=(1, 1, 3, 4))
    elif case == 'missing-direction':
        geometry = replace(geometry, directions=geometry.directions[:-1])
    elif case == 'nonfinite-direction':
        geometry = replace(geometry, directions=((float('inf'), 0., 1.),)+geometry.directions[1:])
    elif case == 'zero-direction':
        geometry = replace(geometry, directions=((0., 0., 0.),)+geometry.directions[1:])
    elif case == 'boolean-tolerance':
        geometry = replace(geometry, angular_tolerance_degrees=True)
    else:
        geometry = replace(geometry, angular_tolerance_degrees=30.)
    before = graph.ToBinary(Chem.PropertyPickleOptions.AllProps)
    result = describe_polyhedral_configuration(graph, center, geometry)
    assert result.status == 'invalid-input' and not result.descriptor and result.limitations
    assert graph.ToBinary(Chem.PropertyPickleOptions.AllProps) == before


def test_incompatible_polyhedral_directions_remain_unavailable(tetrahedral_control):
    from nispo import describe_polyhedral_configuration

    graph, geometry, _ = tetrahedral_control
    result = describe_polyhedral_configuration(graph, 0, replace(geometry, directions=_SQUARE))
    assert result.status == 'unavailable' and not result.descriptor and result.limitations


def test_unproved_polyhedral_naming_join_retains_completed_configuration():
    from rdkit import Chem
    from nispo import PolyhedralGeometry, polyhedral_to_iupac_result

    graph = Chem.MolFromSmiles('[Pt](F)(Cl)(Br)I')
    result = polyhedral_to_iupac_result(graph, 0, PolyhedralGeometry('SP-4', (1, 2, 3, 4), _SQUARE))
    assert result.generation_status == 'unavailable' and not result.name
    assert result.unresolved and result.configuration.status == 'complete'
    assert result.configuration.molecular_smiles == Chem.MolToSmiles(graph)


def _assigned_axis(clockwise=True):
    from rdkit import Chem

    graph = Chem.MolFromSmiles('Fc1ccccc1-c1ccccc1Cl')
    axis = next(bond for bond in graph.GetBonds() if not bond.IsInRing()
                and bond.GetBeginAtom().GetIsAromatic() and bond.GetEndAtom().GetIsAromatic())
    axis.SetStereo(Chem.BondStereo.STEREOATROPCW if clockwise else Chem.BondStereo.STEREOATROPCCW)
    return graph


def test_native_axial_configuration_is_retained_without_mutating_input():
    from rdkit import Chem
    from nispo import atropisomer_to_iupac_result

    graph = _assigned_axis()
    before = graph.ToBinary(Chem.PropertyPickleOptions.AllProps)
    result = atropisomer_to_iupac_result(graph)
    assert result.generation_status == 'available' and result.name
    assert len(result.axial_units) == 1 and result.axial_units[0].descriptor in {'M', 'P'}
    recovered = atropisomer_to_iupac_result(Chem.MolFromSmiles(result.isomeric_cxsmiles))
    assert recovered.name == result.name
    assert recovered.axial_units[0].descriptor == result.axial_units[0].descriptor
    assert graph.ToBinary(Chem.PropertyPickleOptions.AllProps) == before


def test_reversing_declared_axial_configuration_keeps_constitution():
    from rdkit import Chem
    from nispo import atropisomer_to_iupac_result

    first = atropisomer_to_iupac_result(_assigned_axis())
    second = atropisomer_to_iupac_result(_assigned_axis(False))
    assert first.name and second.name and first.name != second.name
    assert {first.axial_units[0].descriptor, second.axial_units[0].descriptor} == {'M', 'P'}
    assert Chem.MolToSmiles(Chem.MolFromSmiles(first.isomeric_cxsmiles)) == Chem.MolToSmiles(
        Chem.MolFromSmiles(second.isomeric_cxsmiles))


@pytest.mark.parametrize('value', [None, 'CC'])
def test_axial_api_requires_an_explicit_declared_axis(value):
    from rdkit import Chem
    from nispo import atropisomer_to_iupac_result

    result = atropisomer_to_iupac_result(Chem.MolFromSmiles(value) if value is not None else None)
    assert result.generation_status == 'invalid-input' and not result.name and result.unresolved


def test_explicit_allene_geometry_keeps_axis_and_original_graph():
    from rdkit import Chem
    from nispo import cumulene_to_iupac_result

    graph = _allene_graph()
    before = graph.ToBinary(Chem.PropertyPickleOptions.AllProps)
    result = cumulene_to_iupac_result(graph)
    assert graph.ToBinary(Chem.PropertyPickleOptions.AllProps) == before
    assert result.generation_status == 'available' and result.name
    assert result.axial_units[0].atoms == (1, 2, 3)
    assert result.axial_units[0].reference_atoms == (0, 1, 3, 4)
    assert abs(result.torsion_degrees) == pytest.approx(90.)
    restored = Chem.MolFromSmiles(result.isomeric_cxsmiles)
    assert restored is not None and Chem.MolToSmiles(restored) == Chem.MolToSmiles(graph)


def test_reversing_the_complete_allene_axis_preserves_configuration():
    from nispo import cumulene_to_iupac_result

    graph = _allene_graph()
    first = cumulene_to_iupac_result(graph)
    second = cumulene_to_iupac_result(graph, axis_path=(3, 2, 1))
    assert first.name and second.name == first.name
    assert second.axial_units[0].atoms == (3, 2, 1)
    assert first.axial_units[0].descriptor == second.axial_units[0].descriptor


def test_reflecting_allene_coordinates_changes_its_axial_configuration():
    from nispo import cumulene_to_iupac_result

    graph = _allene_graph()
    first = cumulene_to_iupac_result(graph)
    for index in range(graph.GetNumAtoms()):
        x, y, z = graph.GetConformer().GetAtomPosition(index)
        graph.GetConformer().SetAtomPosition(index, (-x, y, z))
    second = cumulene_to_iupac_result(graph)
    assert first.name and second.name and first.name != second.name
    assert {first.axial_units[0].descriptor, second.axial_units[0].descriptor} == {'M', 'P'}
    assert first.coordinates != second.coordinates


def test_cumulene_axis_must_include_the_complete_declared_path():
    from nispo import cumulene_to_iupac_result

    result = cumulene_to_iupac_result(_allene_graph(), axis_path=(1, 2))
    assert result.generation_status == 'invalid-input' and not result.name and result.unresolved


def test_nonorthogonal_allene_geometry_cannot_invent_axial_configuration():
    from nispo import cumulene_to_iupac_result

    graph = _allene_graph()
    graph.GetConformer().SetAtomPosition(4, (1., 0., 1.))
    result = cumulene_to_iupac_result(graph)
    assert result.generation_status == 'unavailable' and not result.name and result.unresolved


@pytest.mark.parametrize('option,value', [('conformer_id', True), ('distance_tolerance', 0.),
                                        ('angular_tolerance', float('nan'))])
def test_helical_api_rejects_invalid_geometry_parameters(option, value):
    from nispo import helicene_to_iupac_result

    result = helicene_to_iupac_result(_torsion_graph(), **{option: value})
    assert result.generation_status == 'invalid-input' and not result.name and result.unresolved


def test_nonhelical_graph_is_declined_with_supplied_identity_retained():
    from rdkit import Chem
    from nispo import helicene_to_iupac_result

    graph = _torsion_graph()
    result = helicene_to_iupac_result(graph)
    assert result.generation_status == 'unavailable' and not result.name and result.unresolved
    assert result.canonical_smiles == Chem.MolToSmiles(graph)
    assert result.coordinates and result.conformer_id == graph.GetConformer().GetId()
