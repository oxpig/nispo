"""Related release contracts consolidated without dropping cases or assertions.

Original module provenance is retained in each section below.
"""
from __future__ import annotations


# Originally test_biochemical_contracts.py
"""Small public composition contracts from independent textbook monomers.

Origin: glycine/alanine condensation, Fischer aldoses, adenosine substitution,
neutral phosphate condensation, and glycerol/sphinganine functional groups.
The inputs were authored for these public contracts, not selected from a panel,
reference-name list, or molecule-level benchmark disagreements. Generated name
spelling is deliberately not an oracle; serialized graphs and public bindings
are checked instead. No combinatorial corpus is constructed here.
"""


import pytest
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


pytestmark = pytest.mark.full_only


def _failure(result, status="invalid-input"):
    assert result.generation_status == status
    assert not result.name
    assert result.preferred_status == "not-assessed"
    assert result.unresolved
    assert all(issue.stage and issue.rule and issue.reason for issue in result.unresolved)


def _mapped_graph(result):
    """Check that the public map evidence serializes the reported identity."""
    assert result.generation_status == "available", result.unresolved
    assert result.name
    assert result.preferred_status == "not-assessed"
    graph = Chem.MolFromSmiles(result.mapped_smiles)
    assert graph is not None
    maps = [atom.GetAtomMapNum() for atom in graph.GetAtoms()]
    assert all(maps) and len(set(maps)) == len(maps)
    unmapped = Chem.Mol(graph)
    for atom in unmapped.GetAtoms():
        atom.SetAtomMapNum(0)
    identity = Chem.MolToSmiles(unmapped, isomericSmiles=True)
    candidates = (result.canonical_smiles,) if result.canonical_smiles else result.member_smiles
    assert identity in candidates
    assert not any(atom.GetFormalCharge() or atom.GetNumRadicalElectrons() for atom in graph.GetAtoms())
    return graph, {atom.GetAtomMapNum(): atom for atom in graph.GetAtoms()}


@pytest.mark.parametrize("case", ["empty", "single", "mutable", "unknown", "achiral-D"])
def test_peptide_rejects_incomplete_or_contradictory_sequence_declarations(case):
    from nispo import PeptideResidue, PeptideSequence, peptide_to_iupac_result

    requests = {
        "empty": PeptideSequence(()),
        "single": PeptideSequence(("Gly",)),
        "mutable": PeptideSequence(["Gly", "Ala"]),
        "unknown": PeptideSequence(("Gly", "unknown-residue")),
        "achiral-D": PeptideSequence((PeptideResidue("Gly", "D"), "Ala")),
    }
    _failure(peptide_to_iupac_result(requests[case]))


@pytest.mark.parametrize(
    ("n_end", "c_end", "formula"),
    [("free", "acid", "C4H8N2O3"), ("free", "amide", "C4H9N3O2"),
     ("free", "methyl-ester", "C5H10N2O3"), ("acetyl", "acid", "C6H10N2O4")],
)
def test_dipeptide_terminal_derivatives_preserve_condensation_and_map_evidence(n_end, c_end, formula):
    from nispo import PeptideSequence, peptide_to_iupac_result

    request = PeptideSequence(("Gly", "Gly"), n_terminus=n_end, c_terminus=c_end)
    result = peptide_to_iupac_result(request)
    graph, atoms = _mapped_graph(result)
    assert rdMolDescriptors.CalcMolFormula(graph) == formula
    assert len(result.residue_bindings) == 2
    assert len(result.peptide_bond_maps) == 1
    left, right = result.residue_bindings
    assert result.peptide_bond_maps == ((left.carbonyl_carbon_map, right.nitrogen_map),)
    peptide_edge = result.peptide_bond_maps[0]
    assert [atoms[number].GetAtomicNum() for number in peptide_edge] == [6, 7]
    assert graph.GetBondBetweenAtoms(*(atoms[number].GetIdx() for number in peptide_edge)).GetBondType() == Chem.BondType.SINGLE
    residue_maps = {number for binding in result.residue_bindings for number in binding.atom_maps}
    cap_maps = {number for _label, numbers in result.cap_atom_maps for number in numbers}
    assert not residue_maps.intersection(cap_maps)
    assert residue_maps | cap_maps == set(atoms)
    assert request == PeptideSequence(("Gly", "Gly"), n_terminus=n_end, c_terminus=c_end)


def test_changing_one_residue_configuration_changes_stereo_without_changing_composition():
    from nispo import PeptideResidue, PeptideSequence, peptide_to_iupac_result

    results = [peptide_to_iupac_result(PeptideSequence((PeptideResidue("Ala", configuration), "Gly")))
               for configuration in ("L", "D")]
    graphs = [_mapped_graph(result)[0] for result in results]
    assert [rdMolDescriptors.CalcMolFormula(graph) for graph in graphs] == ["C5H10N2O3"] * 2
    assert results[0].canonical_smiles != results[1].canonical_smiles
    for graph in graphs:
        Chem.RemoveStereochemistry(graph)
    assert Chem.MolToSmiles(graphs[0]) == Chem.MolToSmiles(graphs[1])


@pytest.mark.parametrize("case", ["missing-series", "impossible-ring", "acyclic-anomer", "bad-projection"])
def test_carbohydrate_rejects_missing_or_incompatible_configuration(case):
    from nispo import CarbohydrateSpec, FischerSugar, carbohydrate_to_iupac_result

    requests = {
        "missing-series": CarbohydrateSpec("glucose"),
        "impossible-ring": CarbohydrateSpec(FischerSugar(3, "R"), ring="pyranose"),
        "acyclic-anomer": CarbohydrateSpec("D-glucose", ring="acyclic", anomer="alpha"),
        "bad-projection": CarbohydrateSpec(FischerSugar(5, "RXR"), ring="acyclic"),
    }
    _failure(carbohydrate_to_iupac_result(requests[case]))


def test_fischer_enantiomers_preserve_formula_but_not_absolute_identity():
    from nispo import CarbohydrateSpec, FischerSugar, carbohydrate_to_iupac_result

    results = [carbohydrate_to_iupac_result(CarbohydrateSpec(FischerSugar(3, side), ring="acyclic"))
               for side in ("R", "L")]
    graphs = [_mapped_graph(result)[0] for result in results]
    assert [rdMolDescriptors.CalcMolFormula(graph) for graph in graphs] == ["C3H6O3"] * 2
    assert results[0].canonical_smiles != results[1].canonical_smiles
    assert all(len(Chem.FindMolChiralCenters(graph, includeUnassigned=True)) == 1 for graph in graphs)


def test_ring_closure_retains_sugar_composition_and_reports_the_bridge_oxygen():
    from nispo import CarbohydrateSpec, carbohydrate_to_iupac_result

    result = carbohydrate_to_iupac_result(CarbohydrateSpec("D-glucose", anomer="alpha"))
    graph, atoms = _mapped_graph(result)
    assert rdMolDescriptors.CalcMolFormula(graph) == "C6H12O6"
    bridge = atoms[result.ring_oxygen_map]
    assert bridge.GetAtomicNum() == 8 and bridge.GetDegree() == 2 and bridge.IsInRing()
    assert graph.GetRingInfo().NumRings() == 1
    assert result.ring_closing_position == 5


def test_anomer_pair_is_reported_as_two_members_instead_of_one_assigned_structure():
    from nispo import CarbohydrateSpec, carbohydrate_to_iupac_result

    result = carbohydrate_to_iupac_result(CarbohydrateSpec("D-glucose", anomer="alpha,beta"))
    _mapped_graph(result)
    assert not result.canonical_smiles
    assert len(set(result.member_smiles)) == 2
    assert set(result.member_anomers) == {"alpha", "beta"}
    assert result.proportions_unspecified
    assert all(rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(member)) == "C6H12O6" for member in result.member_smiles)


@pytest.mark.parametrize(
    ("alcohol", "status"),
    [("", "invalid-input"), ("CC(=O)O", "invalid-input"),
     ("CO.CO", "invalid-input"), ("NCCO", "unavailable")],
)
def test_glycoside_declines_nonalcohol_or_out_of_scope_aglycones(alcohol, status):
    from nispo import CarbohydrateSpec, GlycosideSpec, glycoside_to_iupac_result

    request = GlycosideSpec(CarbohydrateSpec("D-glucose", anomer="beta"), alcohol)
    _failure(glycoside_to_iupac_result(request), status)


def test_glycosidic_binding_retains_the_alcohol_oxygen_and_condensation_formula():
    from nispo import CarbohydrateSpec, GlycosideSpec, glycoside_to_iupac_result

    request = GlycosideSpec(CarbohydrateSpec("D-glucose", anomer="beta"), "CO")
    result = glycoside_to_iupac_result(request)
    graph, atoms = _mapped_graph(result)
    assert rdMolDescriptors.CalcMolFormula(graph) == "C7H14O6"
    binding = result.glycosidic_binding
    maps = (binding.anomeric_carbon_map, binding.bridge_oxygen_map, binding.aglycone_carbon_map)
    assert [atoms[number].GetAtomicNum() for number in maps] == [6, 8, 6]
    assert binding.bridge_oxygen_origin == "alcohol"
    assert all(graph.GetBondBetweenAtoms(atoms[left].GetIdx(), atoms[right].GetIdx()) is not None
               for left, right in zip(maps, maps[1:]))


@pytest.mark.parametrize("case", ["unknown-parent", "bad-sugar-site", "occupied-sugar-site", "unsupported-base-operation"])
def test_nucleoside_requires_a_known_parent_and_unambiguous_modification_sites(case):
    from nispo import BaseModification, NucleosideSpec, SugarModification, nucleoside_to_iupac_result

    requests = {
        "unknown-parent": NucleosideSpec("not-a-nucleoside"),
        "bad-sugar-site": NucleosideSpec("adenosine", (SugarModification(1, "deoxy"),)),
        "occupied-sugar-site": NucleosideSpec("adenosine", (SugarModification(2, "deoxy"), SugarModification(2, "fluoro"))),
        "unsupported-base-operation": NucleosideSpec("adenosine", base_modifications=(BaseModification(2, "oxidize"),)),
    }
    _failure(nucleoside_to_iupac_result(requests[case]))


@pytest.mark.parametrize(("operation", "formula"), [("deoxy", "C10H13N5O3"), ("fluoro", "C10H12FN5O3")])
def test_single_sugar_modification_preserves_nucleoside_atom_partition(operation, formula):
    from nispo import NucleosideSpec, SugarModification, nucleoside_to_iupac_result

    result = nucleoside_to_iupac_result(NucleosideSpec("adenosine", (SugarModification(2, operation),)))
    graph, atoms = _mapped_graph(result)
    assert rdMolDescriptors.CalcMolFormula(graph) == formula
    components = [set(numbers) for _owner, numbers in result.component_atom_maps]
    assert sum(map(len, components)) == len(set().union(*components)) == len(atoms)
    assert set().union(*components) == set(atoms)
    assert all(number in atoms for _role, number in result.parent_atom_maps)


@pytest.mark.parametrize(("count", "bridge"), [(1, "oxygen"), (2, "sulfur"), (3, "methylene")])
def test_neutral_phosphate_chain_conserves_phosphorus_and_acid_hydrogens(count, bridge):
    from nispo import NucleosideSpec, NucleotideSpec, nucleotide_to_iupac_result

    result = nucleotide_to_iupac_result(NucleotideSpec(NucleosideSpec("adenosine"), phosphate_count=count,
                                                     bridge_replacement=bridge))
    graph, atoms = _mapped_graph(result)
    assert sum(atom.GetAtomicNum() == 15 for atom in graph.GetAtoms()) == count
    assert result.acid_hydrogen_count == count + 1
    assert len(result.acidic_atom_maps) == count + 1
    assert all(atoms[number].GetTotalNumHs() == 1 for number in result.acidic_atom_maps)
    backbone = [atoms[number] for number in result.phosphate_backbone_maps]
    assert len(backbone) == 2 * count - 1
    assert all(atom.GetAtomicNum() == 15 for atom in backbone[::2])
    assert all(atom.GetAtomicNum() == {"oxygen": 8, "sulfur": 16, "methylene": 6}[bridge] for atom in backbone[1::2])


@pytest.mark.parametrize("case", ["zero-phosphates", "bad-position", "missing-bridge", "ambiguous-thio-site"])
def test_nucleotide_declines_invalid_sites_or_unspecified_multi_phosphate_thio_position(case):
    from nispo import NucleosideSpec, NucleotideSpec, nucleotide_to_iupac_result

    options = {
        "zero-phosphates": {"phosphate_count": 0},
        "bad-position": {"position": 4},
        "missing-bridge": {"bridge_replacement": "sulfur"},
        "ambiguous-thio-site": {"phosphate_count": 2, "phosphorothioate": True},
    }
    result = nucleotide_to_iupac_result(NucleotideSpec(NucleosideSpec("adenosine"), **options[case]))
    _failure(result, "unavailable" if case == "ambiguous-thio-site" else "invalid-input")


@pytest.mark.parametrize(("kind", "formula"), [("acyl", "C5H10O4"), ("ether", "C5H12O3")])
def test_glycerol_functionalization_keeps_the_original_oxygen_in_the_binding(kind, formula):
    from nispo import GlycerolAcyl, GlycerolEther, GlycerolSpec, glycerol_to_iupac_result

    request = (GlycerolSpec((GlycerolAcyl(1, "CC(=O)O"),)) if kind == "acyl" else
               GlycerolSpec(ether_groups=(GlycerolEther(1, "CCO"),)))
    result = glycerol_to_iupac_result(request)
    graph, atoms = _mapped_graph(result)
    assert rdMolDescriptors.CalcMolFormula(graph) == formula
    bindings = result.acyl_bindings if kind == "acyl" else result.ether_bindings
    assert len(bindings) == 1 and bindings[0].position == 1
    oxygen = bindings[0].original_glycerol_oxygen_map
    assert oxygen in dict(result.glycerol_atom_maps).values()
    assert atoms[oxygen].GetAtomicNum() == 8 and atoms[oxygen].GetDegree() == 2
    assert not set(bindings[0].atom_maps).intersection(dict(result.glycerol_atom_maps).values())


@pytest.mark.parametrize("case", ["empty", "occupied-OH", "bad-position", "nonacid"])
def test_glycerol_rejects_missing_substituents_and_conflicting_or_invalid_ports(case):
    from nispo import GlycerolAcyl, GlycerolEther, GlycerolSpec, glycerol_to_iupac_result

    requests = {
        "empty": GlycerolSpec(),
        "occupied-OH": GlycerolSpec((GlycerolAcyl(1, "CC(=O)O"),), ether_groups=(GlycerolEther(1, "CCO"),)),
        "bad-position": GlycerolSpec((GlycerolAcyl(0, "CC(=O)O"),)),
        "nonacid": GlycerolSpec((GlycerolAcyl(1, "CO"),)),
    }
    _failure(glycerol_to_iupac_result(requests[case]))


def test_racemic_glycerol_is_two_equimolar_members_without_a_single_fixed_identity():
    from nispo import GlycerolAcyl, GlycerolSpec, glycerol_to_iupac_result

    result = glycerol_to_iupac_result(GlycerolSpec((GlycerolAcyl(1, "CC(=O)O"),), configuration="rac"))
    _mapped_graph(result)
    assert not result.canonical_smiles
    assert len(set(result.member_smiles)) == 2
    assert result.proportions == ((1, 2), (1, 2))
    members = [Chem.MolFromSmiles(value) for value in result.member_smiles]
    for graph in members:
        Chem.RemoveStereochemistry(graph)
    assert Chem.MolToSmiles(members[0]) == Chem.MolToSmiles(members[1])


def test_sphinganine_parent_preserves_two_declared_centres_and_neutral_composition():
    from nispo import SphingoidSpec, sphingoid_to_iupac_result
    from rdkit.Chem import rdCIPLabeler

    result = sphingoid_to_iupac_result(SphingoidSpec())
    graph, atoms = _mapped_graph(result)
    assert rdMolDescriptors.CalcMolFormula(graph) == "C18H39NO2"
    role = dict(result.parent_atom_maps)
    # SMILES parsing assigns approximate legacy labels; adjacent stereocentres
    # require the accurate CIP implementation for this configuration oracle.
    rdCIPLabeler.AssignCIPLabels(graph)
    assert [atoms[role[position]].GetProp("_CIPCode") for position in ("2", "3")] == ["S", "R"]
    assert not result.acyl_bindings and not result.double_bond_configurations


def test_sphingoid_unsaturation_changes_hydrogen_count_and_retains_declared_geometry():
    from nispo import SphingoidDoubleBond, SphingoidSpec, sphingoid_to_iupac_result

    result = sphingoid_to_iupac_result(SphingoidSpec((SphingoidDoubleBond(4, "E"),)))
    graph, _atoms = _mapped_graph(result)
    assert rdMolDescriptors.CalcMolFormula(graph) == "C18H37NO2"
    doubles = [bond for bond in graph.GetBonds() if bond.GetBondType() == Chem.BondType.DOUBLE]
    assert len(doubles) == 1 and doubles[0].GetStereo() == Chem.BondStereo.STEREOE
    assert result.double_bond_configurations == ((4, "E"),)


@pytest.mark.parametrize("case", ["terminal-E", "cumulated", "occupied-OH"])
def test_sphingoid_rejects_contradictory_unsaturation_or_occupied_hydroxy_sites(case):
    from nispo import SphingoidDoubleBond, SphingoidOAcyl, SphingoidSpec, sphingoid_to_iupac_result

    requests = {
        "terminal-E": SphingoidSpec((SphingoidDoubleBond(17, "E"),)),
        "cumulated": SphingoidSpec((SphingoidDoubleBond(4), SphingoidDoubleBond(5))),
        "occupied-OH": SphingoidSpec(o_acyl_groups=(SphingoidOAcyl(1, "CC(=O)O"), SphingoidOAcyl(1, "CC(=O)O"))),
    }
    _failure(sphingoid_to_iupac_result(requests[case]))


# Originally test_domain_boundaries.py
"""Small public composition boundaries from independent textbook building blocks.

Origin: a two-nucleoside phosphate condensation, a two-ribose O-glycosidic
condensation and a six-carbon inositol drawing. These inputs were authored
from the stated monomer/port chemistry for these tests, independently of any
panel, source-name list or naming disagreement. No name spelling is an oracle.
The largest construction is a dinucleoside; no oligomer enumeration occurs.
"""


import pytest
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


pytestmark = pytest.mark.full_only


def _declined(result, status='invalid-input'):
    assert result.generation_status == status
    assert not result.name and result.preferred_status == 'not-assessed'
    assert result.unresolved
    assert all(issue.stage and issue.rule and issue.reason for issue in result.unresolved)


def _mapped(result):
    assert result.generation_status == 'available', result.unresolved
    assert result.name and result.preferred_status == 'not-assessed'
    graph = Chem.MolFromSmiles(result.mapped_smiles)
    assert graph is not None
    maps = [atom.GetAtomMapNum() for atom in graph.GetAtoms()]
    assert all(maps) and len(maps) == len(set(maps))
    atoms = {atom.GetAtomMapNum(): atom for atom in graph.GetAtoms()}
    plain = Chem.Mol(graph)
    for atom in plain.GetAtoms():
        atom.SetAtomMapNum(0)
    identity = Chem.MolToSmiles(plain, isomericSmiles=True)
    expected = (result.canonical_smiles,) if result.canonical_smiles else result.member_smiles
    assert identity in expected
    components = [set(numbers) for _owner, numbers in result.component_atom_maps]
    assert sum(map(len, components)) == len(set().union(*components)) == len(atoms)
    assert set().union(*components) == set(atoms)
    return graph, atoms


@pytest.mark.parametrize('case', [
    'no-request', 'one-residue', 'mutable-sequence', 'non-residue',
    'duplicate-link', 'zero-link', 'boolean-link', 'terminal-link',
])
def test_oligonucleotide_requires_an_ordered_sequence_and_unique_internal_links(case):
    from nispo import OligonucleotideSequence, oligonucleotide_to_iupac_result

    requests = {
        'no-request': None,
        'one-residue': OligonucleotideSequence(('adenosine',)),
        'mutable-sequence': OligonucleotideSequence(['adenosine', 'uridine']),
        'non-residue': OligonucleotideSequence(('adenosine', object())),
        'duplicate-link': OligonucleotideSequence(('adenosine', 'uridine'), (1, 1)),
        'zero-link': OligonucleotideSequence(('adenosine', 'uridine'), (0,)),
        'boolean-link': OligonucleotideSequence(('adenosine', 'uridine'), (True,)),
        'terminal-link': OligonucleotideSequence(('adenosine', 'uridine'), (2,)),
    }
    _declined(oligonucleotide_to_iupac_result(requests[case]))


@pytest.mark.parametrize(('thio', 'formula'), [(False, 'C19H24N7O12P'), (True, 'C19H24N7O11PS')])
def test_dinucleoside_link_retains_both_sugar_oxygens_and_one_acid_hydrogen(thio, formula):
    from nispo import OligonucleotideSequence, oligonucleotide_to_iupac_result

    # Monomers + phosphoric acid - two waters; thio replaces one phosphate O.
    request = OligonucleotideSequence(('adenosine', 'uridine'), (1,) if thio else ())
    before = repr(request)
    result = oligonucleotide_to_iupac_result(request)
    graph, atoms = _mapped(result)
    assert repr(request) == before
    assert rdMolDescriptors.CalcMolFormula(graph) == formula
    assert len(result.residue_bindings) == 2 and len(result.phosphodiester_bindings) == 1
    left, right = result.residue_bindings
    link = result.phosphodiester_bindings[0]
    assert link.first_sugar_oxygen_map == dict(left.atom_roles)["O3'"]
    assert link.second_sugar_oxygen_map == dict(right.atom_roles)["O5'"]
    phosphorus = atoms[link.phosphorus_map]
    assert phosphorus.GetAtomicNum() == 15
    for number in (link.first_sugar_oxygen_map, link.second_sugar_oxygen_map):
        oxygen = atoms[number]
        assert oxygen.GetAtomicNum() == 8 and oxygen.GetTotalNumHs() == 0 and oxygen.GetDegree() == 2
        assert graph.GetBondBetweenAtoms(oxygen.GetIdx(), phosphorus.GetIdx()).GetBondType() == Chem.BondType.SINGLE
    acid = atoms[link.acidic_atom_map]
    # The public binding identifies the acidic atom; a sulfur replacement
    # need not leave that atom as oxygen. Both sugar bridge atoms stay O.
    assert acid.GetAtomicNum() in ({8, 16} if thio else {8}) and acid.GetTotalNumHs() == 1
    assert graph.GetBondBetweenAtoms(acid.GetIdx(), phosphorus.GetIdx()).GetBondType() == Chem.BondType.SINGLE
    sulfur = [atoms[number] for number in link.atom_maps if atoms[number].GetAtomicNum() == 16]
    assert len(sulfur) == int(thio)
    if sulfur:
        assert sulfur[0].GetDegree() == 1
        assert graph.GetBondBetweenAtoms(sulfur[0].GetIdx(), phosphorus.GetIdx()) is not None
    assert link.phosphorothioate is thio and link.phosphorus_configuration == 'unspecified'
    residue_maps = set(left.atom_maps) | set(right.atom_maps)
    assert not residue_maps.intersection(link.atom_maps)
    assert residue_maps | set(link.atom_maps) == set(atoms)


@pytest.mark.parametrize(('residue', 'position'), [(0, 3), (1, 5)])
def test_oligonucleotide_declines_an_erased_required_sugar_oxygen(residue, position):
    from nispo import NucleosideSpec, OligonucleotideSequence, SugarModification, oligonucleotide_to_iupac_result

    monomers = [NucleosideSpec('adenosine'), NucleosideSpec('uridine')]
    monomers[residue] = NucleosideSpec(monomers[residue].parent, (SugarModification(position, 'deoxy'),))
    _declined(oligonucleotide_to_iupac_result(OligonucleotideSequence(tuple(monomers))))


def _ribose(anomer):
    from nispo import CarbohydrateSpec
    return CarbohydrateSpec('D-ribose', ring='furanose', anomer=anomer)


def test_disaccharide_link_consumes_original_acceptor_oh_and_one_water():
    from nispo import OligosaccharideSpec, SugarLink, oligosaccharide_to_iupac_result

    request = OligosaccharideSpec((_ribose('alpha'), _ribose('beta')), (SugarLink(0, 1, 2),))
    before = repr(request)
    result = oligosaccharide_to_iupac_result(request)
    graph, atoms = _mapped(result)
    assert rdMolDescriptors.CalcMolFormula(graph) == 'C10H18O9'
    assert result.reducing_residue == 1 and result.glycoside_residue is None
    assert len(result.glycosidic_bindings) == 1 and repr(request) == before
    link = result.glycosidic_bindings[0]
    assert link.bridge_oxygen_map == dict(result.residue_atom_maps[1])['O2']
    assert link.donor_carbon_map == dict(result.residue_atom_maps[0])['1']
    assert link.bridge_oxygen_origin == 'acceptor'
    bridge = atoms[link.bridge_oxygen_map]
    assert bridge.GetAtomicNum() == 8 and bridge.GetDegree() == 2 and bridge.GetTotalNumHs() == 0
    assert {atom.GetAtomMapNum() for atom in bridge.GetNeighbors()} == {link.donor_carbon_map, link.acceptor_carbon_map}


def test_anomer_to_anomer_join_removes_reducing_end_without_changing_graph_when_parent_is_rechosen():
    from nispo import OligosaccharideSpec, SugarLink, oligosaccharide_to_iupac_result

    residues = (_ribose('alpha'), _ribose('beta'))
    results = [oligosaccharide_to_iupac_result(OligosaccharideSpec(residues, (SugarLink(0, 1, 1),), parent))
               for parent in (0, 1)]
    for result in results:
        graph, atoms = _mapped(result)
        assert rdMolDescriptors.CalcMolFormula(graph) == 'C10H18O9'
        assert result.reducing_residue is None
        link = result.glycosidic_bindings[0]
        assert atoms[link.donor_carbon_map].GetAtomicNum() == atoms[link.acceptor_carbon_map].GetAtomicNum() == 6
    assert results[0].canonical_smiles == results[1].canonical_smiles
    assert [result.glycoside_residue for result in results] == [0, 1]


def test_reducing_end_mixture_keeps_two_physical_members_instead_of_one_identity():
    from nispo import OligosaccharideSpec, SugarLink, oligosaccharide_to_iupac_result

    result = oligosaccharide_to_iupac_result(OligosaccharideSpec(
        (_ribose('alpha'), _ribose('alpha,beta')), (SugarLink(0, 1, 2),)))
    _mapped(result)
    assert not result.canonical_smiles and result.proportions_unspecified
    assert result.reducing_residue == 1 and len(set(result.member_smiles)) == 2
    assert set(result.member_anomers) == {'alpha', 'beta'}
    assert {rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(value)) for value in result.member_smiles} == {'C10H18O9'}


@pytest.mark.parametrize('case', ['self-link', 'boolean-residue', 'negative-residue', 'duplicate-donor', 'disconnected-cycle'])
def test_oligosaccharide_declines_invalid_or_reused_link_ports(case):
    from nispo import OligosaccharideSpec, SugarLink, oligosaccharide_to_iupac_result

    residues = (_ribose('alpha'), _ribose('beta'))
    links = {
        'self-link': (SugarLink(0, 0, 2),),
        'boolean-residue': (SugarLink(True, 0, 2),),
        'negative-residue': (SugarLink(-1, 1, 2),),
        'duplicate-donor': (SugarLink(0, 1, 2), SugarLink(0, 2, 2)),
        'disconnected-cycle': (SugarLink(0, 1, 2), SugarLink(1, 0, 2)),
    }[case]
    if len(links) == 2:
        residues += (_ribose('alpha'),)
    _declined(oligosaccharide_to_iupac_result(OligosaccharideSpec(residues, links)))


def test_bonded_anomer_must_be_specified_even_when_free_end_may_be_unspecified():
    from nispo import OligosaccharideSpec, SugarLink, oligosaccharide_to_iupac_result

    request = OligosaccharideSpec((_ribose('unspecified'), _ribose('beta')), (SugarLink(0, 1, 2),))
    _declined(oligosaccharide_to_iupac_result(request), 'unavailable')


@pytest.mark.parametrize('case', ['no-request', 'unknown-parent', 'missing-numbered-series', 'occupied-hydroxyl'])
def test_inositol_requires_a_defined_drawing_and_nonconflicting_modification_sites(case):
    from nispo import CarbohydrateModification, InositolSpec, inositol_to_iupac_result

    requests = {
        'no-request': None,
        'unknown-parent': InositolSpec('not-an-inositol-parent'),
        'missing-numbered-series': InositolSpec(series='D'),
        'occupied-hydroxyl': InositolSpec(modifications=(CarbohydrateModification(2, 'deoxy'), CarbohydrateModification(2, 'O-methyl'))),
    }
    _declined(inositol_to_iupac_result(requests[case]))


def test_inositol_enantiomer_drawings_preserve_formula_but_not_absolute_identity():
    from nispo import InositolSpec, inositol_to_iupac_result

    results = [inositol_to_iupac_result(InositolSpec('chiro', series)) for series in ('1L', '1D')]
    graphs = [_mapped(result)[0] for result in results]
    assert all(rdMolDescriptors.CalcMolFormula(graph) == 'C6H12O6' for graph in graphs)
    assert all(result.configuration_descriptor_required for result in results)
    assert results[0].canonical_smiles != results[1].canonical_smiles
    for graph in graphs:
        Chem.RemoveStereochemistry(graph)
    assert Chem.MolToSmiles(graphs[0]) == Chem.MolToSmiles(graphs[1])


def test_inositol_phosphate_keeps_original_ring_oxygen_and_two_acid_hydrogens():
    from nispo import CarbohydrateModification, InositolSpec, inositol_to_iupac_result

    request = InositolSpec(modifications=(CarbohydrateModification(2, 'phosphate'),))
    before = repr(request)
    result = inositol_to_iupac_result(request)
    graph, atoms = _mapped(result)
    assert rdMolDescriptors.CalcMolFormula(graph) == 'C6H13O9P'
    assert repr(request) == before and len(result.phosphate_bindings) == 1
    binding = result.phosphate_bindings[0]
    oxygen = atoms[binding.sugar_oxygen_map]
    assert binding.sugar_oxygen_map == dict(result.parent_atom_maps)['O2']
    assert oxygen.GetAtomicNum() == 8 and oxygen.GetDegree() == 2 and oxygen.GetTotalNumHs() == 0
    assert graph.GetBondBetweenAtoms(oxygen.GetIdx(), atoms[binding.phosphorus_map].GetIdx()) is not None
    assert len(binding.acidic_atom_maps) == 2
    assert all(atoms[number].GetAtomicNum() == 8 and atoms[number].GetTotalNumHs() == 1
               for number in binding.acidic_atom_maps)


# Originally test_api_execution_boundaries.py
"""Public execution boundaries, independent of molecular naming choices.

Provenance: None, empty molecules and wrong Python types are independently
authored API inputs. Ethanol is an elementary transport control; the one
relative-stereo string is an independently authored tetrahedral CX boundary.
No panel graphs, source names or naming disagreements informed these cases.
Only the final test needs a worker. The launch guard observes external work;
it does not replace parsing, molecule validation or the naming algorithm.
"""


import subprocess

import pytest


pytestmark = pytest.mark.full_only


def _invalid_molecule(kind):
    if kind == 'none':
        return None
    if kind == 'empty':
        from rdkit import Chem
        return Chem.Mol()
    return object()


@pytest.fixture
def no_external_work(monkeypatch):
    attempts = []

    def forbidden(*_args, **_kwargs):
        attempts.append(True)
        raise AssertionError('Rejected input must not start an external process')

    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    yield
    # An internally caught exception must not conceal a launch attempt.
    assert not attempts


@pytest.mark.parametrize('kind', ('none', 'empty', 'wrong-type'))
@pytest.mark.parametrize('timeout', (None, 0.5), ids=('direct', 'finite-limit'))
def test_invalid_structured_molecule_is_rejected_before_external_work(kind, timeout, no_external_work):
    from nispo import mol_to_iupac_result

    result = mol_to_iupac_result(_invalid_molecule(kind), timeout_seconds=timeout)

    assert not result.name
    assert result.generation_status == 'invalid-input'
    assert result.preferred_status == 'not-assessed'


@pytest.mark.parametrize('kind', ('none', 'empty', 'wrong-type'))
def test_finite_limit_string_molecule_rejection_needs_no_worker(kind, no_external_work):
    from nispo import mol_to_iupac

    assert mol_to_iupac(_invalid_molecule(kind), timeout_seconds=0.5) == ''


@pytest.mark.parametrize('value', ([], {}), ids=('list', 'dictionary'))
def test_unhashable_scalar_smiles_is_invalid_without_external_work(value, no_external_work):
    from nispo import smiles_to_iupac, smiles_to_iupac_result

    result = smiles_to_iupac_result(value, timeout_seconds=0.5)

    assert not result.name
    assert result.generation_status == 'invalid-input'
    assert result.preferred_status == 'not-assessed'
    assert smiles_to_iupac(value, timeout_seconds=0.5) == ''


@pytest.mark.parametrize('timeout', (None, 0.5), ids=('direct', 'finite-limit'))
def test_empty_structured_iterator_batch_needs_no_worker(timeout, no_external_work):
    from nispo import smiles_to_iupac_batch_results

    assert smiles_to_iupac_batch_results(iter(()), timeout_seconds=timeout) == []


def test_direct_batch_keeps_nonabsolute_rejection_in_its_original_slot():
    from nispo import smiles_to_iupac_batch, smiles_to_iupac_batch_results

    inputs = ('CCO', 'C[C@H](F)Cl |&1:1|', '', 'CCO')
    results = smiles_to_iupac_batch_results(iter(inputs), timeout_seconds=None)
    names = smiles_to_iupac_batch(iter(inputs), timeout_seconds=None)

    assert len(results) == len(names) == len(inputs)
    assert names == [result.name for result in results]
    assert names[0] and results[0] == results[3]
    relative, invalid = results[1:3]
    assert not relative.name and relative.generation_status == 'unavailable'
    assert relative.preferred_status == 'unresolved'
    assert any(issue.stage == 'input-stereochemistry' for issue in relative.unresolved)
    assert not invalid.name and invalid.generation_status == 'invalid-input'


@pytest.mark.workers
def test_finite_public_scalar_and_batch_wrappers_preserve_direct_results():
    from nispo import (
        smiles_to_iupac, smiles_to_iupac_batch, smiles_to_iupac_batch_results,
    )

    inputs = ('CCO', '', 'CCO')
    direct = smiles_to_iupac_batch_results(inputs, timeout_seconds=None)
    assert direct[0].name and direct[0].generation_status == 'available'
    assert not direct[1].name and direct[1].generation_status == 'invalid-input'
    expected_names = [result.name for result in direct]

    # A generous deadline measures real transport without manufacturing a
    # timeout or replacing the worker's algorithm. Calls share the public
    # single-worker owner, so only its first call needs worker initialization.
    options = {'timeout_seconds': 15, 'fallback_to_fast': False}
    assert smiles_to_iupac('CCO', **options) == expected_names[0]
    assert smiles_to_iupac_batch(iter(inputs), **options) == expected_names
    assert smiles_to_iupac_batch_results(iter(inputs), **options) == direct


# Originally test_stereo_declarations.py
"""Small public contracts for explicit stereo-ensemble declarations.

Origin: elementary halogenated chains were authored from tetrahedral valence
and P-93's distinction between relative sets, racemates and epimer mixtures.
They were not selected from a benchmark panel, reference name or naming
disagreement. Most cases reject an incomplete declaration before naming;
only one two-centre epimer case constructs member names.
"""


import pytest


pytestmark = pytest.mark.full_only

_TWO_CENTRES = 'C[C@H:20](F)[C@H:30](Cl)Br'


def _assert_no_invented_members(result, status, stage):
    assert not result.name and result.generation_status == status
    assert result.preferred_status == ('not-assessed' if status == 'invalid-input' else 'unresolved')
    assert result.member_smiles == () and result.mole_fractions == ()
    assert result.unresolved and all(issue.stage == stage for issue in result.unresolved)


@pytest.mark.parametrize('operation', ('ensemble', 'subgroup'))
def test_stereo_declaration_requires_its_explicit_public_value_object(operation):
    from nispo import stereo_ensemble_to_iupac_result, stereo_subgroup_to_iupac_result

    function = (stereo_ensemble_to_iupac_result if operation == 'ensemble'
                else stereo_subgroup_to_iupac_result)
    result = function(None)
    _assert_no_invented_members(result, 'invalid-input', 'stereo-'+operation)


@pytest.mark.parametrize('problem', (
    'unsupported-interpretation', 'non-text-representative', 'empty-representative',
    'disconnected-representative', 'unassigned-centre', 'achiral-representative',
))
def test_whole_ensemble_rejects_incomplete_or_contradictory_declarations(problem):
    from nispo import StereoEnsemble, stereo_ensemble_to_iupac_result

    text, interpretation = {
        'unsupported-interpretation': ('C[C@H](F)Cl', 'ambo'),
        'non-text-representative': (None, 'racemic'),
        'empty-representative': ('', 'relative'),
        'disconnected-representative': ('CC.C[C@H](F)Cl', 'racemic'),
        'unassigned-centre': ('CC(F)Cl', 'relative'),
        'achiral-representative': ('CCO', 'racemic'),
    }[problem]
    result = stereo_ensemble_to_iupac_result(StereoEnsemble(text, interpretation))
    _assert_no_invented_members(result, 'invalid-input', 'stereo-ensemble')


@pytest.mark.parametrize('problem', (
    'unsupported-interpretation', 'non-tuple-group', 'duplicate-group-map',
    'boolean-map', 'absent-map', 'duplicate-source-map', 'unassigned-centre',
    'whole-relative-flag', 'no-fixed-centre-outside-group', 'one-centre-relative-set',
))
def test_subgroup_requires_unique_assigned_maps_and_a_fixed_outside_configuration(problem):
    from nispo import StereoSubgroup, stereo_subgroup_to_iupac_result

    text, interpretation, maps, status = {
        'unsupported-interpretation': (_TWO_CENTRES, 'relative', (20,), 'invalid-input'),
        'non-tuple-group': (_TWO_CENTRES, 'ambo', [20], 'invalid-input'),
        'duplicate-group-map': (_TWO_CENTRES, 'ambo', (20, 20), 'invalid-input'),
        'boolean-map': (_TWO_CENTRES, 'ambo', (True,), 'invalid-input'),
        'absent-map': (_TWO_CENTRES, 'ambo', (99,), 'invalid-input'),
        'duplicate-source-map': ('C[C@H:20](F)[C@H:20](Cl)Br', 'ambo', (20,), 'invalid-input'),
        'unassigned-centre': ('C[CH:20](F)[C@H:30](Cl)Br', 'ambo', (20,), 'invalid-input'),
        'whole-relative-flag': (_TWO_CENTRES+' |r|', 'ambo', (20,), 'unavailable'),
        'no-fixed-centre-outside-group': (_TWO_CENTRES, 'ambo', (20, 30), 'invalid-input'),
        'one-centre-relative-set': (_TWO_CENTRES, 'relative-group', (20,), 'invalid-input'),
    }[problem]
    result = stereo_subgroup_to_iupac_result(StereoSubgroup(text, interpretation, maps))
    _assert_no_invented_members(result, status, 'stereo-subgroup')


def test_ambo_only_delivers_proved_member_graphs_and_never_invents_equal_proportions():
    from rdkit import Chem
    from rdkit.Chem import rdCIPLabeler, rdMolDescriptors
    from nispo import StereoSubgroup, stereo_subgroup_to_iupac_result

    declaration = StereoSubgroup(_TWO_CENTRES, 'ambo', (20,))
    result = stereo_subgroup_to_iupac_result(declaration)
    # The public operation explicitly declines routes without a completed
    # descriptor-to-original-atom map. Do not freeze a route's current level
    # of support; validate either its explicit limitation or proved members.
    if result.generation_status == 'unavailable':
        _assert_no_invented_members(result, 'unavailable', 'stereo-subgroup')
        return
    assert result.name and result.generation_status == 'available'
    assert result.interpretation == 'ambo' and result.membership == 'and-epimer'
    assert result.proportions == 'unspecified' and result.mole_fractions == ()
    assert result.representative_atom_maps == (20,) and len(result.group_locants) == 1
    assert len(result.member_smiles) == 2 and len(set(result.member_smiles)) == 2

    original = Chem.MolFromSmiles(_TWO_CENTRES)
    indices = {atom.GetAtomMapNum(): atom.GetIdx() for atom in original.GetAtoms() if atom.GetAtomMapNum()}
    query = Chem.Mol(original)
    Chem.RemoveStereochemistry(query)
    for atom in query.GetAtoms():
        atom.SetAtomMapNum(0)
    assignments = []
    for value in result.member_smiles:
        member = Chem.MolFromSmiles(value)
        assert member is not None and member.GetNumAtoms() == original.GetNumAtoms()
        assert rdMolDescriptors.CalcMolFormula(member) == rdMolDescriptors.CalcMolFormula(original)
        matches = member.GetSubstructMatches(query, uniquify=False, useChirality=False)
        assert len(matches) == 1, 'The asymmetric control must give a unique atom correspondence'
        rdCIPLabeler.AssignCIPLabels(member)
        assignments.append({number: member.GetAtomWithIdx(matches[0][index]).GetProp('_CIPCode')
                            for number, index in indices.items()})
    assert {row[20] for row in assignments} == {'R', 'S'}
    assert assignments[0][30] == assignments[1][30]


# Originally test_extended_functional_graphs.py
"""Bounded public graph contracts for less common functional operations.

Provenance: these controls were authored directly from ordinary valence and
general P-21/P-65/P-66/P-67/P-68 functional-class definitions on 2026-11-11.
The feature list came from coverage gaps; no panel molecules, source names or
molecule-level naming outcomes were used to construct the graphs. Each group
has four small controls and a single batched name-to-graph verification.
"""


import pytest


pytestmark = pytest.mark.full_only


EXTENDED_FUNCTIONAL_GRAPHS = (
    pytest.param(
        ("O=C(Cl)CC(=O)Cl", "O=C(F)CC(=O)F", "O=C(Br)CCC(=O)Br", "O=C(Cl)CC(=O)F"),
        id="multiple-acyl-halides",
    ),
    pytest.param(
        ("COS(=O)(=O)C", "CCOS(=O)(=O)C", "COS(=O)C", "CCOC(=O)SC"),
        id="sulfur-ester-connectivity",
    ),
    pytest.param(
        ("CC(=O)OCCOC(C)=O", "CCC(=O)OCCOC(C)=O", "CC(=O)OCC(O)COC(C)=O", "CC(=O)OCC(O)CO"),
        id="partial-and-complete-polyol-esters",
    ),
    pytest.param(
        ("C[Se](=O)O", "C[Te](=O)O", "CC[Se](=O)(=O)O", "C[Se](=O)N"),
        id="chalcogen-oxoacids-and-amide",
    ),
    pytest.param(
        ("CS(=N)C", "CS(=O)(=N)C", "CS(=N)(=N)C", "C[Se](=O)(=N)C"),
        id="chalcogen-imide-bond-orders",
    ),
    pytest.param(
        ("C[N+]#[C-]", "CC[N+]#[C-]", "N#CC[N+]#[C-]", "O=C(O)CC[N+]#[C-]"),
        id="isocyano-attachment-and-coexisting-functions",
    ),
    pytest.param(
        ("C[SiH2]C", "C[GeH2]C", "CO[SiH3]", "C[SiH2]OC"),
        id="group-fourteen-hydrogen-budget",
    ),
)


@pytest.mark.parametrize("inputs", EXTENDED_FUNCTIONAL_GRAPHS)
def test_extended_functional_graph_is_recoverable(naming, assert_graphs, inputs):
    """Require the delivered name to preserve every atom, bond and H budget."""
    names = naming.names(inputs)
    assert len(names) == len(inputs)
    assert all(isinstance(name, str) and name for name in names)
    assert_graphs(inputs, names)


@pytest.mark.parametrize(
    "text",
    ("CN#C", "C[SiH4]C"),
    ids=("neutral-overvalent-isocyano-nitrogen", "overvalent-silicon"),
)
def test_invalid_functional_valence_is_not_repaired_into_another_graph(naming, text):
    result = naming.result(text)
    assert result.name == ""
    assert result.generation_status == "invalid-input"
