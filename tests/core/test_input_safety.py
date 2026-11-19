"""Related release contracts consolidated without dropping cases or assertions.

Original module provenance is retained in each section below.
"""


# Originally test_input_safety.py
"""Small public checks that source annotations do not change their meaning.

Provenance: the alcohol/haloalkane controls and ortho-substituted biaryl are
independently authored textbook graph patterns. CX fields are hand-written
boundary inputs, not panel molecules, source names or naming disagreements.
Only the absolute group, three harmless-metadata examples and two nonchiral
relative markers perform real naming. Rejections exit at the input boundary.
"""
import pytest
from rdkit import Chem


def _assert_nonabsolute_decline(result):
    assert not result.name
    assert result.generation_status == 'unavailable'
    assert result.preferred_status == 'unresolved'
    assert any(issue.stage == 'input-stereochemistry' for issue in result.unresolved)


def _biaryl_with_axis(direction):
    stereo = getattr(Chem.BondStereo, direction, None)
    if stereo is None:
        pytest.skip('This RDKit build does not expose native atropisomeric bond stereo')
    # Two different ortho substituents on each side define the biaryl axis.
    mol = Chem.MolFromSmiles('Cc1cccc(Br)c1-c1c(Cl)cccc1F')
    axes = [bond for bond in mol.GetBonds()
            if not bond.GetIsAromatic()
            and bond.GetBeginAtom().GetIsAromatic()
            and bond.GetEndAtom().GetIsAromatic()]
    assert len(axes) == 1
    axes[0].SetStereo(stereo)
    return mol, axes[0]


@pytest.mark.current_only
@pytest.mark.parametrize('extension', (
    'r,atomProp:0.note.safe',
    'atomProp:0.note.safe,r',
    '$alpha,r,beta;;;$,r',
), ids=('leading-field', 'trailing-field', 'real-field-after-comma-label'))
def test_relative_flag_is_recognized_at_top_level_field_boundaries(naming, extension):
    text = f'C[C@H](F)Cl |{extension}|'
    _assert_nonabsolute_decline(naming.result(text))
    assert naming.name(text) == ''


@pytest.mark.current_only
def test_byte_smiles_retains_relative_stereo_intent(naming):
    text = b'C[C@H](F)Cl |r|'
    _assert_nonabsolute_decline(naming.result(text))
    assert naming.name(text) == ''


@pytest.mark.full_only
@pytest.mark.parametrize('group', ('o1:1', '&1:1'), ids=('native-or', 'native-and'))
def test_native_nonabsolute_group_is_guarded_without_original_cx_text(naming, group):
    mol = Chem.MolFromSmiles(f'C[C@H](F)Cl |{group}|')
    assert mol is not None and mol.GetStereoGroups()
    if mol.HasProp('_CXSMILES_Data'):
        mol.ClearProp('_CXSMILES_Data')
    before = mol.ToBinary(Chem.PropertyPickleOptions.AllProps)

    _assert_nonabsolute_decline(naming.molecule_result(mol))

    assert mol.ToBinary(Chem.PropertyPickleOptions.AllProps) == before
    assert mol.GetStereoGroups()
    assert not mol.HasProp('_CXSMILES_Data')


@pytest.mark.full_only
def test_native_absolute_group_is_not_confused_with_a_relative_group(naming, assert_graphs):
    mol = Chem.MolFromSmiles('C[C@H](O)F |a:1|')
    assert mol is not None and mol.GetStereoGroups()
    assert all(group.GetGroupType() == Chem.StereoGroupType.STEREO_ABSOLUTE
               for group in mol.GetStereoGroups())
    if mol.HasProp('_CXSMILES_Data'):
        mol.ClearProp('_CXSMILES_Data')
    before = mol.ToBinary(Chem.PropertyPickleOptions.AllProps)

    result = naming.molecule_result(mol)

    assert result.generation_status == 'available' and result.name
    assert not any(issue.stage == 'input-stereochemistry' for issue in result.unresolved)
    assert mol.ToBinary(Chem.PropertyPickleOptions.AllProps) == before
    assert_graphs(['C[C@H](O)F'], [result.name])


@pytest.mark.full_only
@pytest.mark.parametrize('direction', ('STEREOATROPCW', 'STEREOATROPCCW'))
def test_relative_axial_bond_metadata_cannot_become_an_absolute_member(naming, direction):
    mol, axis = _biaryl_with_axis(direction)
    mol.SetProp('_CXSMILES_Data', '|r|')
    before = mol.ToBinary(Chem.PropertyPickleOptions.AllProps)

    _assert_nonabsolute_decline(naming.molecule_result(mol))

    assert mol.ToBinary(Chem.PropertyPickleOptions.AllProps) == before
    assert mol.GetBondWithIdx(axis.GetIdx()).GetStereo() == getattr(Chem.BondStereo, direction)


@pytest.mark.current_only
@pytest.mark.parametrize('direction', ('STEREOATROPCW', 'STEREOATROPCCW'))
def test_encoded_relative_axial_stereo_keeps_its_intent_at_smiles_entrypoint(naming, direction):
    mol, _axis = _biaryl_with_axis(direction)
    text = Chem.MolToCXSmiles(mol)
    graph, separator, fields = text.partition('|')
    assert separator and fields.endswith('|')
    relative = graph + '|r,' + fields
    parsed = Chem.MolFromSmiles(relative)
    assert parsed is not None
    assert any(str(bond.GetStereo()).startswith('STEREOATROP') for bond in parsed.GetBonds())

    _assert_nonabsolute_decline(naming.result(relative))
    assert naming.name(relative) == ''


@pytest.mark.current_only
@pytest.mark.parametrize('extension', (
    '$alpha,r,beta;;;$',
    '$r;;;$',
    'atomProp:0.note.r',
), ids=('comma-label-data', 'single-letter-label', 'property-value'))
def test_letter_r_inside_metadata_is_not_a_relative_stereo_declaration(naming, assert_graphs, extension):
    plain = 'C[C@H](O)F'
    result = naming.result(f'{plain} |{extension}|')
    assert result.name and result.generation_status == 'available'
    assert not any(issue.stage == 'input-stereochemistry' for issue in result.unresolved)
    assert_graphs([plain], [result.name])


@pytest.mark.current_only
@pytest.mark.parametrize('plain', ('F/C=C/F', 'CC(O)F'), ids=('ez-only', 'unassigned-centre'))
def test_relative_marker_without_an_encoded_chiral_unit_preserves_ordinary_graph(naming, assert_graphs, plain):
    result = naming.result(plain + ' |r|')
    assert result.name and result.generation_status == 'available'
    assert not any(issue.stage == 'input-stereochemistry' for issue in result.unresolved)
    assert_graphs([plain], [result.name])


# Originally test_annotation_contracts.py
"""Independent small annotation controls and explicit unsupported-input safety.

Provenance: the structures are independently authored examples of isotope,
formal-charge, absolute stereo, atom-map and CX stereo annotations.  They do
not originate in a benchmark panel or molecule-level disagreement report.
"""

import pytest
from rdkit import Chem


ABSOLUTE_STEREO = (
    pytest.param(
        ("C[C@H](O)F", "C[C@@H](O)F", "CC[C@H](O)C", "CC[C@@H](O)C"),
        id="acyclic-tetrahedral",
    ),
    pytest.param(
        ("C[C@H]1CC[C@@H](O)CC1", "C[C@H]1CC[C@H](O)CC1", "O=C(O)[C@H]1C[C@H](O)C1", "O=C(O)[C@H]1C[C@@H](O)C1"),
        id="ring-tetrahedral",
    ),
    pytest.param(
        ("C/C=C/C", "C/C=C\\C", "F/C=C/F", "F/C=C\\F"),
        id="double-bond-geometry",
    ),
)


@pytest.mark.parametrize("inputs", ABSOLUTE_STEREO)
def test_absolute_stereochemical_graphs_are_recovered(naming, assert_graphs, inputs):
    names = naming.names(inputs)
    assert len(names) == len(inputs)
    assert all(names)
    assert_graphs(inputs, names)


def test_formal_charge_and_disconnected_components_are_recovered(naming, assert_graphs):
    inputs = (
        "[NH4+]", "C[N+](C)(C)C", "CC(=O)[O-]", "[Na+].[Cl-]",
        "C[NH3+].[Cl-]", "[O-][N+](=O)c1ccccc1",
    )
    names = naming.names(inputs)
    assert len(names) == len(inputs)
    assert all(names)
    assert_graphs(inputs, names)


def test_isotope_labels_are_recovered(naming, assert_graphs):
    inputs = ("[13CH3]CO", "C[18OH]", "[2H]OC", "[15NH2]CC", "[13CH3][13CH2]O", "[14CH4]")
    names = naming.names(inputs)
    assert len(names) == len(inputs)
    assert all(names)
    assert_graphs(inputs, names)


def test_radical_electron_counts_are_recovered(naming, assert_graphs):
    inputs = ("[CH3]", "[CH2]C")
    names = naming.names(inputs)
    assert len(names) == len(inputs)
    assert all(names)
    assert_graphs(inputs, names)


def test_atom_maps_do_not_become_molecular_substituents(naming, assert_graphs):
    mapped = ("[CH3:7][CH2:8][OH:9]", "[CH3:2][C@H:3]([OH:4])[F:5]")
    expected = []
    for text in mapped:
        mol = Chem.MolFromSmiles(text)
        assert mol is not None
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        expected.append(Chem.MolToSmiles(mol, isomericSmiles=True))
    names = naming.names(mapped)
    assert len(names) == len(mapped)
    assert all(names)
    assert_graphs(expected, names)


@pytest.mark.current_only
@pytest.mark.parametrize(
    "text",
    (
        "C[C@H](F)Cl |r|",
        "C[C@H](F)Cl |o1:1|",
        "C[C@H](F)Cl |&1:1|",
        "CC[C@H](O)C |r|",
    ),
    ids=("relative", "or-group", "and-group", "relative-second-center"),
)
def test_nonabsolute_input_is_not_reported_as_an_absolute_member(naming, text):
    result = naming.result(text)
    assert not result.name
    assert result.generation_status == "unavailable"
    assert result.preferred_status == "unresolved"
    assert any(issue.stage == "input-stereochemistry" for issue in result.unresolved)


@pytest.mark.modes("full", "public")
@pytest.mark.current_only
def test_relative_stereo_molecule_keeps_caller_metadata(naming):
    mol = Chem.MolFromSmiles("C[C@H](F)Cl |&1:1|")
    assert mol is not None and mol.GetStereoGroups()
    mol.SetProp("caller_note", "retain")
    before = mol.ToBinary(Chem.PropertyPickleOptions.AllProps)
    result = naming.molecule_result(mol)
    assert not result.name
    assert result.generation_status == "unavailable"
    assert any(issue.stage == "input-stereochemistry" for issue in result.unresolved)
    assert mol.ToBinary(Chem.PropertyPickleOptions.AllProps) == before


@pytest.mark.current_only
def test_nonstereochemical_cx_metadata_does_not_block_ordinary_graphs(naming, assert_graphs):
    inputs = ("CCO |r|", "C[C@H](O)F |$tag;;;label$|")
    names = naming.names(inputs)
    assert all(names)
    assert_graphs(("CCO", "C[C@H](O)F"), names)


# Originally test_process_portability.py
"""Fresh-interpreter naming, using independent elementary ethanol.

This is also a public-release-compatible check that subprocess coverage is real.
The 0.1.7 release itself has no naming-worker API.
"""
import json
import os
from pathlib import Path
import subprocess
import sys


def test_fresh_interpreter_names_the_same_input(naming, assert_graphs, tmp_path):
    import nispo
    source = Path(nispo.__file__).resolve().parent.parent
    code = '''import json,os,sys
from pathlib import Path
import nispo
mode=sys.argv[1]
if mode=='fast':
    from nispo.fast_selection import smiles_to_iupac_fast
    eager_full_engine='nispo.rule_engine' in sys.modules
    name=smiles_to_iupac_fast('CCO')
else:
    eager_full_engine=None
    name=nispo.smiles_to_iupac('CCO',timeout_seconds=None) if mode=='full' else nispo.smiles_to_iupac('CCO')
print(json.dumps({'name':name,'pid':os.getpid(),'package':str(Path(nispo.__file__).resolve()),'eager_full_engine':eager_full_engine}))
'''
    env = os.environ.copy()
    env['PYTHONPATH'] = str(source)+os.pathsep+env.get('PYTHONPATH', '')
    completed = subprocess.run([sys.executable, '-c', code, naming.mode], cwd=tmp_path,
                               env=env, capture_output=True, text=True, check=True, timeout=120)
    result = json.loads(completed.stdout)
    assert result['pid'] != os.getpid()
    assert Path(result['package']) == Path(nispo.__file__).resolve()
    if naming.mode=='fast':
        assert result['eager_full_engine'] is False
    assert_graphs(['CCO'], [result['name']])


# Originally test_public_exports.py
"""Public namespace integrity without molecule or naming-rule fixtures."""
import pytest


@pytest.mark.modes("full", "public")
def test_declared_exports_support_lookup_directory_and_star_import(naming):
    import nispo

    declared = tuple(nispo.__all__)
    assert declared
    assert set(declared).issubset(dir(nispo))
    namespace = {}
    exec('from nispo import *', namespace)
    assert set(namespace)-{'__builtins__'} == set(declared)
    for name in declared:
        assert namespace[name] is getattr(nispo, name)
