"""Small public composition contracts with independently constructed inputs.

Provenance: these controls were authored from exact rational stoichiometry,
mapped two-carbon isotope bookkeeping, global electron balance, and elementary
carbonyl/alkene coordination. No panel records or reference names were used.
The cases protect graph and state evidence, never generated name spelling.
"""
from fractions import Fraction

import pytest


pytestmark = pytest.mark.full_only


def _canonical(smiles, *, remove_maps=True):
    from rdkit import Chem

    graph = Chem.MolFromSmiles(smiles)
    assert graph is not None
    if remove_maps:
        for atom in graph.GetAtoms():
            atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(graph, canonical=True, isomericSmiles=True)


def _assert_failure(result, status):
    assert result.generation_status == status
    assert result.name == ""
    assert result.unresolved
    assert all(issue.stage and issue.rule and issue.reason for issue in result.unresolved)


@pytest.mark.parametrize("amount", [0, True, 0.5, "0/0"])
def test_adduct_rejects_nonpositive_or_inexact_amounts(amount):
    from nispo import AdductComponent, adduct_to_iupac_result

    result = adduct_to_iupac_result((AdductComponent("CCO", amount), AdductComponent("O")))
    _assert_failure(result, "invalid-input")
    assert result.components == ()


@pytest.mark.parametrize("shape", ["not-iterable", "empty", "one-species"])
def test_adduct_requires_two_distinct_explicit_species(shape):
    from nispo import AdductComponent, adduct_to_iupac_result

    inputs = {
        "not-iterable": None,
        "empty": (),
        "one-species": (AdductComponent("CCO"), AdductComponent("[CH3:7][CH2:8][OH:9]")),
    }
    _assert_failure(adduct_to_iupac_result(inputs[shape]), "invalid-input")


@pytest.fixture(scope="module")
def exact_adduct():
    from nispo import AdductComponent, adduct_to_iupac_result

    # Atom maps do not create a third chemical species. Two half shares of
    # the same organic component and 3/2 shares of water reduce to 2:3.
    inputs = (
        AdductComponent("CCO", Fraction(1, 2)),
        AdductComponent("[CH3:7][CH2:8][OH:9]", "1/2"),
        AdductComponent("O", "3/2"),
    )
    result = adduct_to_iupac_result(iter(inputs))
    assert result.generation_status == "available" and result.name
    return inputs, result


def test_adduct_merges_identity_and_preserves_reduced_exact_proportions(exact_adduct):
    inputs, result = exact_adduct

    assert {c.canonical_smiles: c.proportion for c in result.components} == {
        _canonical("CCO"): 2,
        _canonical("O"): 3,
    }
    assert [c.amount for c in inputs] == [Fraction(1, 2), "1/2", "3/2"]
    assert result.preferred_status in {"not-assessed", "unresolved"}


def test_adduct_component_names_recover_each_complete_species(exact_adduct, assert_graphs):
    _, result = exact_adduct

    assert_graphs(
        [c.canonical_smiles for c in result.components],
        [c.name for c in result.components],
    )
    assert all(c.result.name == c.name for c in result.components)


def test_adduct_scale_and_input_order_do_not_change_composition(exact_adduct):
    from nispo import AdductComponent, adduct_to_iupac_result

    _, baseline = exact_adduct
    scaled = adduct_to_iupac_result((AdductComponent("O", 6), AdductComponent("CCO", 4)))

    assert scaled.generation_status == "available"
    assert [(c.canonical_smiles, c.proportion) for c in scaled.components] == [
        (c.canonical_smiles, c.proportion) for c in baseline.components
    ]


@pytest.mark.parametrize("problem", ["unknown-kind", "missing-maps", "missing-form", "duplicate-nuclide"])
def test_labelling_rejects_incomplete_or_ambiguous_declarations(problem):
    from nispo import IsotopeLabelling, Nuclide, isotope_labelling_to_iupac_result

    precursor = "[CH3:11][CH3:12]"
    cases = {
        "unknown-kind": IsotopeLabelling(precursor, "unknown"),
        "missing-maps": IsotopeLabelling("CC", "specific", ("[13CH3]C",)),
        "missing-form": IsotopeLabelling(precursor, "specific"),
        "duplicate-nuclide": IsotopeLabelling(
            precursor, "uniform", nuclides=(Nuclide("C", 13), Nuclide("C", 13))
        ),
    }
    _assert_failure(isotope_labelling_to_iupac_result(cases[problem]), "invalid-input")


def test_specific_labelling_preserves_declared_member_and_site():
    from nispo import IsotopeLabelling, isotope_labelling_to_iupac_result

    precursor = "[CH3:11][CH2:12][Cl:13]"
    member = "[13CH3:11][CH2:12][Cl:13]"
    supplied = IsotopeLabelling(precursor, "specific", (member,))
    result = isotope_labelling_to_iupac_result(supplied)

    assert result.generation_status == "available" and result.name
    assert result.includes_unmodified is True
    assert result.isotopologues == (_canonical(member),)
    assert _canonical(result.unmodified_smiles) == _canonical(precursor)
    assert [(s.atom_map, s.element, s.mass, s.maximum_count) for s in result.labelled_sites] == [
        (11, "C", 13, 1)
    ]
    assert supplied.isotopologues == (member,)


@pytest.mark.parametrize("kind", ["nonselective", "deficient", "general", "uniform"])
def test_distribution_labelling_does_not_invent_fixed_members_or_enrichment(kind):
    from nispo import IsotopeLabelling, Nuclide, isotope_labelling_to_iupac_result

    result = isotope_labelling_to_iupac_result(
        IsotopeLabelling("[CH3:11][CH3:12]", kind, nuclides=(Nuclide("C", 13),))
    )

    assert result.generation_status == "available" and result.name
    assert result.kind == kind
    assert result.isotopologues == ()
    assert result.includes_unmodified is None
    assert {(s.atom_map, s.element, s.mass) for s in result.labelled_sites} == {
        (11, "C", 13), (12, "C", 13)
    }
    assert result.position_ratios == (
        "equal-at-declared-positions" if kind == "uniform" else "unspecified"
    )


def test_uniform_labelling_subset_does_not_label_other_positions():
    from nispo import IsotopeLabelling, Nuclide, isotope_labelling_to_iupac_result

    result = isotope_labelling_to_iupac_result(
        IsotopeLabelling("[CH3:11][CH3:12]", "uniform", nuclides=(Nuclide("C", 13),),
                        subset_atom_maps=(12,))
    )

    assert result.generation_status == "available" and result.name
    assert {s.atom_map for s in result.labelled_sites} == {12}


def test_labelling_rejects_a_member_with_changed_connectivity():
    from nispo import IsotopeLabelling, isotope_labelling_to_iupac_result

    _assert_failure(
        isotope_labelling_to_iupac_result(
            IsotopeLabelling("[CH3:11][CH2:12][Cl:13]", "specific",
                            ("[13CH3:11][CH2:12][F:13]",))
        ),
        "invalid-input",
    )


@pytest.mark.parametrize("problem", ["zero-transfer", "boolean-transfer", "incompatible-spin", "charged-parent"])
def test_electron_transfer_requires_a_valid_neutral_precursor_and_electron_state(problem):
    from nispo import ElectronTransfer, electron_transfer_to_iupac_result

    inputs = {
        "zero-transfer": ElectronTransfer("CC", 0),
        "boolean-transfer": ElectronTransfer("CC", True),
        "incompatible-spin": ElectronTransfer("CC", 1, 2),
        "charged-parent": ElectronTransfer("C[CH2+]", 1),
    }
    _assert_failure(electron_transfer_to_iupac_result(inputs[problem]), "invalid-input")


@pytest.mark.parametrize("electrons_added", [-1, 1])
def test_global_electron_transfer_conserves_formula_and_declared_charge(electrons_added):
    from nispo import ElectronTransfer, electron_transfer_to_iupac_result
    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula

    supplied = ElectronTransfer("[CH3:21][CH3:22]", electrons_added)
    result = electron_transfer_to_iupac_result(supplied)

    assert result.generation_status == "available" and result.name
    assert result.electrons_added == electrons_added
    assert result.charge == -electrons_added
    assert result.radical_electrons == 1
    assert result.localization == "global-unspecified"
    assert result.molecular_formula == CalcMolFormula(Chem.MolFromSmiles(supplied.neutral_smiles))
    assert _canonical(result.neutral_smiles) == _canonical(supplied.neutral_smiles)
    assert supplied.neutral_smiles == "[CH3:21][CH3:22]"


def test_global_electron_transfer_declines_an_independently_suffixed_parent():
    from nispo import ElectronTransfer, electron_transfer_to_iupac_result

    _assert_failure(electron_transfer_to_iupac_result(ElectronTransfer("CCO", 1)), "unavailable")


@pytest.mark.parametrize("problem", ["unknown-metal", "invalid-donor", "boolean-charge"])
def test_coordination_rejects_invalid_metal_and_original_index_declarations(problem):
    from nispo import CoordinationLigand, CoordinationSpec, coordination_to_iupac_result

    ligand = CoordinationLigand("[C-]#[O+]", (0,))
    supplied = {
        "unknown-metal": CoordinationSpec("C", (ligand,)),
        "invalid-donor": CoordinationSpec("Ni", (CoordinationLigand("[C-]#[O+]", (2,)),)),
        "boolean-charge": CoordinationSpec("Ni", (ligand,), metal_formal_charge=True),
    }
    _assert_failure(coordination_to_iupac_result(supplied[problem]), "invalid-input")


@pytest.mark.parametrize("problem", ["oxidation-mismatch", "no-carbon-binding"])
def test_coordination_reports_unproved_bonding_models(problem):
    from nispo import CoordinationLigand, CoordinationSpec, coordination_to_iupac_result

    supplied = {
        "oxidation-mismatch": CoordinationSpec(
            "Ni", (CoordinationLigand("[C-]#[O+]", (0,)),), oxidation_state=2
        ),
        "no-carbon-binding": CoordinationSpec("Ni", (CoordinationLigand("N", (0,)),)),
    }
    _assert_failure(coordination_to_iupac_result(supplied[problem]), "unavailable")


def _assert_coordination_graph(result):
    from rdkit import Chem

    assert result.generation_status == "available" and result.name
    graph = Chem.MolFromSmiles(result.mapped_smiles)
    assert graph is not None
    atom_maps = {a.GetAtomMapNum() for a in graph.GetAtoms()}
    assert 0 not in atom_maps and len(atom_maps) == graph.GetNumAtoms()
    ligand_maps = [set(b.ligand_atom_maps) for b in result.bindings]
    assert sum(map(len, ligand_maps)) + 1 == len(atom_maps)
    assert set().union(*ligand_maps) | {result.metal_atom_map} == atom_maps
    assert all(set(b.donor_atom_maps) <= set(b.ligand_atom_maps) for b in result.bindings)
    assert sum(a.GetFormalCharge() for a in graph.GetAtoms()) == result.total_formal_charge
    assert _canonical(result.mapped_smiles) == result.canonical_smiles
    return graph


def test_carbonyl_coordination_preserves_whole_ligands_and_original_donor_indices():
    from nispo import CoordinationLigand, CoordinationSpec, coordination_to_iupac_result
    from rdkit import Chem

    supplied = CoordinationSpec("Ni", (
        CoordinationLigand("[C-:31]#[O+:32]", (0,)),
        CoordinationLigand("[O+:41]#[C-:42]", (1,)),
    ))
    result = coordination_to_iupac_result(supplied)
    graph = _assert_coordination_graph(result)

    assert graph.GetNumAtoms() == 5
    assert len(result.bindings) == 2
    assert {b.original_donor_indices for b in result.bindings} == {(0,), (1,)}
    assert all(b.mode == "kappa" and b.free_ligand_charge == 0 for b in result.bindings)
    assert all(Chem.MolFromSmiles(b.free_ligand_smiles).GetNumAtoms() == 2 for b in result.bindings)
    assert all(b.GetBondType() == Chem.BondType.DATIVE for b in graph.GetBonds()
               if result.metal_atom_map in {
                   b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum()
               })
    assert supplied.ligands[0].smiles == "[C-:31]#[O+:32]"


def test_coordination_total_charge_includes_ligand_charge():
    from nispo import CoordinationLigand, CoordinationSpec, coordination_to_iupac_result

    result = coordination_to_iupac_result(CoordinationSpec(
        "Fe", (CoordinationLigand("[C-]#[O+]", (0,)), CoordinationLigand("[Cl-]", (0,))),
        metal_formal_charge=0, oxidation_state=0,
    ))
    _assert_coordination_graph(result)
    assert result.total_formal_charge == -1
    assert result.metal_formal_charge == result.oxidation_state == 0


def test_full_eta_declaration_preserves_all_carbon_donors_as_one_binding():
    from nispo import CoordinationLigand, CoordinationSpec, coordination_to_iupac_result

    result = coordination_to_iupac_result(
        CoordinationSpec("Ni", (CoordinationLigand("C=C", (0, 1), "eta"),))
    )
    _assert_coordination_graph(result)
    assert len(result.bindings) == 1
    binding = result.bindings[0]
    assert binding.mode == "eta"
    assert set(binding.donor_atom_maps) == set(binding.ligand_atom_maps)
    assert binding.original_donor_indices == (0, 1)


@pytest.mark.parametrize("smiles,status", [(None, "invalid-input"), ("CCC", "unavailable")])
def test_nearest_repeated_parent_does_not_invent_repeated_cores(smiles, status):
    from nispo import repeated_parent_to_iupac_result

    result = repeated_parent_to_iupac_result(smiles)
    _assert_failure(result, status)
    assert not result.multiplicative_name
    assert result.core_bindings == ()
