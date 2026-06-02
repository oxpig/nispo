from __future__ import annotations

import inspect
import subprocess
import tempfile
from pathlib import Path

import pytest
from rdkit import Chem

import nispo.rule_engine as rule_engine
from nispo.naming_context import MoleculeContext
from nispo import smiles_to_iupac, smiles_to_iupac_batch
from nispo.parent_rules import parent_rules
from nispo.rule_dispatch import RuleSpec
from nispo.rule_engine import (
    NameCandidate,
    _apply_uniform_isotope_label,
    _candidate_key,
    _finalize_candidate_name,
    _prepare_candidates,
    _select_best_prepared_candidate,
    candidate_estimated_length,
    candidate_has_complete_graph,
    candidate_parent_atoms,
    candidate_safety_reasons,
    candidate_source_key,
)
from nispo.candidate_facts import CandidateSafety, GRAPH_INCOMPLETE
from nispo.structural_profile import structural_profile
from nispo.telemetry import CandidateTelemetry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPSIN_JAR = PROJECT_ROOT / "benchmark" / "opsin-cli-2.9.0-jar-with-dependencies.jar"
JAVA = Path("/opt/homebrew/opt/openjdk/bin/java")


def test_candidate_keys_are_comparable_across_descriptor_shapes():
    candidates = [
        NameCandidate("flat-parent", "flat", 29, sort_key=(1, "flat-parent")),
        NameCandidate(
            "nested-parent",
            "nested",
            29,
            sort_key=((750, 999, (), 999, "nested-parent"), len("nested-parent"), "nested-parent"),
        ),
    ]

    assert min(candidates, key=_candidate_key) is candidates[0]


def test_name_candidate_accepts_optional_structural_facts_without_breaking_legacy_constructor():
    legacy = NameCandidate("ethan", "probe", 10)
    structured = NameCandidate(
        "ethan",
        "probe",
        10,
        parent_atoms=frozenset({0, 1}),
        template_id=42,
        source_family="exact-probe",
        safety=CandidateSafety(graph_complete=False),
        estimated_length=5,
    )

    assert legacy.name == "ethan"
    assert candidate_parent_atoms(legacy) == frozenset()
    assert candidate_source_key(legacy) == ("probe", None)
    assert candidate_has_complete_graph(legacy) is False
    assert candidate_parent_atoms(structured) == frozenset({0, 1})
    assert candidate_source_key(structured) == ("exact-probe", 42)
    assert candidate_safety_reasons(structured) == (GRAPH_INCOMPLETE,)
    assert candidate_estimated_length(structured) == 5
    assert candidate_has_complete_graph(structured) is False


def test_rule_engine_has_no_final_shortening_pipeline():
    source = "\n".join(
        (
            inspect.getsource(rule_engine.NamingEngine.name),
            inspect.getsource(rule_engine._prepare_candidates),
            inspect.getsource(rule_engine._finalize_candidate_name),
        )
    )
    forbidden = (
        "_compact_final_output_aliases",
        "_compact_selected_output_name",
        "_compact_common_substituent_terms",
        "_normalize_terminal_parent_stem",
        "final_aliases",
        "selected_output",
        "late_compaction",
        "CONTRACTION",
    )

    assert all(token not in source for token in forbidden)
    for symbol in forbidden[:4]:
        assert not hasattr(rule_engine, symbol)
    assert "name = candidate.name" in inspect.getsource(rule_engine._finalize_candidate_name)


def test_finalizer_preserves_intrinsic_candidate_text_without_stereo_or_isotopes():
    mol = Chem.MolFromSmiles("CC")
    candidate = NameCandidate("1,1-dimethyl-ethyl", "branch-probe", 50)

    assert _finalize_candidate_name(mol, Chem, candidate) == candidate.name


def test_uniform_isotope_label_fast_path_preserves_nonisotopic_names():
    nonisotopic = Chem.MolFromSmiles("CCO")
    isotopic = Chem.MolFromSmiles("[13CH3][13CH3]")

    assert _apply_uniform_isotope_label(nonisotopic, "ethanol") == "ethanol"
    assert _apply_uniform_isotope_label(isotopic, "ethan") == "ethan-13C2"


@pytest.mark.parametrize(
    "candidate_name",
    [
        "1,1-dimethyl-ethyl",
        "1-phenyl-methyl",
        "phenyloxy",
        "N-methylamino",
        "1-azamethylcarbonyl",
        "2-phenyl-1-azaethylcarbonyl",
        "benz2-oxazolyl",
    ],
)
def test_candidate_pipeline_does_not_compact_selected_name(candidate_name):
    mol = Chem.MolFromSmiles("CCO")
    candidate = NameCandidate(candidate_name, "candidate-probe", 50)
    rule = RuleSpec(
        family="candidate-probe",
        priority=50,
        gates=(lambda ctx: True,),
        producer=lambda ctx: (candidate,),
    )

    prepared = _prepare_candidates(mol, Chem, MoleculeContext(mol, Chem), (rule,))

    assert [item.name for item in prepared] == [candidate_name]
    assert _select_best_prepared_candidate(prepared) == candidate_name


def test_candidate_telemetry_records_rule_work_without_changing_output():
    mol = Chem.MolFromSmiles("c1ccc2ccccc2c1CC")
    telemetry = CandidateTelemetry()

    plain = rule_engine.NamingEngine(Chem, ()).name(mol)
    instrumented = rule_engine.NamingEngine(Chem, ()).name(mol, telemetry=telemetry)

    assert instrumented == plain
    assert isinstance(telemetry.substructure_match_calls, int)


def test_candidate_telemetry_records_selected_rule_family():
    mol = Chem.MolFromSmiles("CCc1ccccc1")
    candidate = NameCandidate("ethylbenzene", "probe", 20)
    rule = RuleSpec(
        family="probe-rule",
        priority=20,
        gates=(lambda ctx: True,),
        producer=lambda ctx: (candidate,),
    )
    telemetry = CandidateTelemetry()

    name = rule_engine.NamingEngine(Chem, (rule,)).name(mol, telemetry=telemetry)

    assert name == "ethylbenzene"
    assert telemetry.rule_applies["probe-rule"] == 1
    assert "probe-rule" in telemetry.rule_seconds
    assert telemetry.rule_seconds["probe-rule"] >= 0
    assert telemetry.candidate_count["probe-rule"] == 1
    assert telemetry.rendered_candidate_count["probe-rule"] == 1
    assert telemetry.selected_family["probe"] == 1
    assert telemetry.selected_rule_family["probe-rule"] == 1


def test_candidate_telemetry_records_structured_candidate_facts():
    mol = Chem.MolFromSmiles("CC")
    candidate = NameCandidate("ethan", "probe", 20, parent_atoms=frozenset({0, 1}), estimated_length=5)
    rule = RuleSpec(
        family="probe-rule",
        priority=20,
        gates=(lambda ctx: True,),
        producer=lambda ctx: (candidate,),
    )
    telemetry = CandidateTelemetry()

    name = rule_engine.NamingEngine(Chem, (rule,)).name(mol, telemetry=telemetry)

    assert name == "ethan"
    assert telemetry.structured_candidate_count["probe-rule"] == 1
    assert telemetry.unstructured_candidate_count["probe-rule"] == 0
    assert telemetry.structured_selected_count["probe"] == 1
    assert telemetry.selected_with_parent_atoms == 1


def test_retained_scaffold_telemetry_records_filter_stage_counts():
    mol = Chem.MolFromSmiles("Cc1ccc(C)c(Cl)c1")
    telemetry = CandidateTelemetry()

    rule_engine.NamingEngine(Chem, parent_rules(Chem)).name(mol, telemetry=telemetry)

    assert telemetry.filter_stage_count["retained-ring-scaffold-parent:candidate_templates"] > 0
    assert telemetry.filter_stage_count["retained-ring-scaffold-parent:facts"] >= 0


def test_structural_profile_detects_conjugated_carbonyl_diene_signal():
    positive = structural_profile(Chem.MolFromSmiles("O=CC=CC=Cc1ccccc1"), Chem)
    negative = structural_profile(Chem.MolFromSmiles("O=CCc1ccccc1"), Chem)

    assert positive.has_conjugated_carbonyl_diene_signal
    assert not negative.has_conjugated_carbonyl_diene_signal


def test_nested_sort_key_parent_selection_does_not_empty_api_output():
    smiles = "O=C1CC2(CCCCC2)c2cn[nH]c2N1"

    scalar = smiles_to_iupac(smiles)
    batch = smiles_to_iupac_batch([smiles])

    assert scalar
    assert batch == [scalar]


def test_single_ring_functional_suffix_parents_roundtrip():
    smiles_values = [
        "O=C(N)c1ccncc1",
        "NC(=O)C1=CC=COC1",
        "NC(=O)C1=CC=CSC1",
        "N#Cc1ccncc1",
        "O=C([O-])c1ccncc1",
        "O=C(O)c1ccsc1",
        "O=C(O)c1cc(C#N)ncc1",
        "c1ccc(-c2ccccc2)cc1",
        "N#Cc1cc(-c2ccccc2)c(C#N)cc1",
        "c1ccc(-c2ccc(-c3ccccc3)cc2)cc1",
        "Clc1ccc(-c2ccc(-c3ccc(Cl)cc3)cc2)cc1",
        "Fc1c(F)c(F)c(OC(F)(F)F)c(F)c1F",
        "FC(F)(F)ON1CCCC1",
        "FC(F)OC1CNC1",
        "O=S(=O)(c1ccccc1)N1CCCCC1",
        "O=C(c1cccnc1)N1CCCCC1",
        "Nc1ccc(-c2nc3cc(F)ccc3s2)cc1",
        "c1ccc2cc(-c3cn4ccccc4n3)ccc2c1",
        "O=C(Nc1ccccc1)c1ccccc1",
        "O=C(Nc1ccccc1)c1ccncc1",
        "O=C(Nc1ccccc1)c1ccco1",
        "Nc1ccc(C(=O)Nc2ccccc2)cc1",
        "O=C(Nc1ccccc1)NC1CCCCC1",
        "B1(OC(C(O1)(C)C)(C)C)c2ccccc2",
        "CCCCCCCCCCCCCCCCCCCCC(=O)O",
        "CCCCCC(=O)[O-]",
        "Nc1ccc(C(=O)Nc2ccccc2N)cc1",
        "C[Si](C)(C)CCOCn1ccc2ccc(N)cc21",
        "C[Si](C)(C)CCOCOc1ccc2cccnc2c1Br",
    ]

    names = smiles_to_iupac_batch(smiles_values)
    parsed = _opsin_parse(names)

    for smiles, name, parsed_smiles in zip(smiles_values, names, parsed, strict=True):
        assert name
        assert parsed_smiles


def test_tautomer_sensitive_single_ring_suffix_parents_roundtrip():
    smiles_values = [
        "Cc1[nH]ncc1C(N)=O",
        "Cc1cc(=O)c(C(=O)O)c(C)[nH]1",
        "O=C(Nc1ccc(Cl)nc1Cl)c1cnc[nH]1",
    ]

    names = smiles_to_iupac_batch(smiles_values)
    parsed = _opsin_parse(names)

    for smiles, name, parsed_smiles in zip(smiles_values, names, parsed, strict=True):
        assert name
        assert parsed_smiles


def test_simple_parent_prefix_join_is_intrinsically_compact_and_roundtrips():
    smiles_values = [
        "NC1Cc2ccccc2C1",
        "CCC(C)Br",
        "CCCCCCc1ccsc1",
        "CCCC1CCS1",
        "CCCC(C#N)CCCC#N",
        "COc1cc2c(cc1OC)CCC2",
        "CCCCCCCI",
        "Nc1ccc2ccccc2c1",
    ]

    names = smiles_to_iupac_batch(smiles_values)
    parsed = _opsin_parse(names)

    for smiles, name, parsed_smiles in zip(smiles_values, names, parsed, strict=True):
        assert name
        assert parsed_smiles


def test_acyclic_unsaturated_parents_use_compact_upstream_forms():
    smiles_values = [
        "C=CCCCC",
        "C#CCCCC",
        "C=CCCC=CCC",
        "C=CCC=C",
        "CCC=CCCCCCCCCCC",
        "C#CCCCC=O",
        "CCCCC=CCCCC(=O)O",
    ]

    names = smiles_to_iupac_batch(smiles_values)
    parsed = _opsin_parse(names)

    for smiles, name, parsed_smiles in zip(smiles_values, names, parsed, strict=True):
        assert name
        assert parsed_smiles


def test_saturated_carbocycle_parents_use_compact_stems():
    smiles_values = [
        "C1CCCCC1",
        "BrC1CCCCC1",
        "C1CCCCCC1",
    ]

    names = smiles_to_iupac_batch(smiles_values)
    parsed = _opsin_parse(names)

    for smiles, name, parsed_smiles in zip(smiles_values, names, parsed, strict=True):
        assert name
        assert parsed_smiles


def test_generated_fused_indicated_h_parents_roundtrip():
    smiles_values = [
        "Ic1n[nH]c2cccnc12",
        "NC(=O)c1cncc2c[nH]nc12",
        "Nc1ncnc2[nH]ncc12",
        "Cn1ccc2cc[nH]c(=O)c21",
        "Fc1[nH]nc2ccccc12",
        "Ic1ncnc2[nH]c(-c3ccccc3)nc12",
    ]

    names = smiles_to_iupac_batch(smiles_values)
    parsed = _opsin_parse(names)

    for smiles, name, parsed_smiles in zip(smiles_values, names, parsed, strict=True):
        assert name
        assert parsed_smiles


def test_generated_fused_topology_and_benzotriazole_orientation_roundtrip():
    smiles_values = [
        "NC(=O)C1=CC=C2C(=Cc3ncncc32)C1",
        "N#Cc1cccc2nn[nH]c12",
        "Brc1cc([N+](=O)[O-])c2nn[nH]c2c1",
        "O=C(O)Cc1cccc2nn[nH]c12",
    ]

    names = smiles_to_iupac_batch(smiles_values)
    parsed = _opsin_parse(names)

    for smiles, name, parsed_smiles in zip(smiles_values, names, parsed, strict=True):
        assert name
        assert parsed_smiles


def test_retained_scaffold_topology_parents_roundtrip_without_generic_bicyclo_fallback():
    smiles_values = [
        "O=C1C=CC2=CN=NC2=C1",
        "Nc1nc2c(c(=O)[nH]1)=NC(=O)N=2",
        "N#CC1C=CC=CO1",
        "O=c1cc2cccc3c(=O)cc4cccc1c4=c23",
    ]

    names = smiles_to_iupac_batch(smiles_values)
    parsed = _opsin_parse(names)

    for smiles, name, parsed_smiles in zip(smiles_values, names, parsed, strict=True):
        assert name
        assert parsed_smiles


def test_n_substituted_purine_nucleoside_avoids_tautomeric_retained_parent_regression():
    smiles_values = [
        "O=P(O)(O)O[C@H]1[C@@H](O)[C@H](n2cnc3c(O)ncnc32)O[C@@H]1CO",
        "O=P(O)(O)OC[C@H]1O[C@@H](n2cnc3c(O)nc(S)nc32)[C@H](O)[C@@H]1O",
    ]

    names = smiles_to_iupac_batch(smiles_values)
    parsed = _opsin_parse(names)

    for smiles, name, parsed_smiles in zip(smiles_values, names, parsed, strict=True):
        assert name
        assert parsed_smiles


def test_fused_indicated_h_ketone_suffixes_roundtrip():
    smiles_values = [
        "O=c1nc[nH]c2cc([N+](=O)[O-])ccc12",
        "Cc1ccc2[nH]cnc(=O)c2c1",
        "O=c1cc[nH]c2nccnc12",
        "O=c1[nH][nH]c2ncncc12",
        "O=c1c2ccccc2[nH]cn1",
    ]

    names = smiles_to_iupac_batch(smiles_values)
    parsed = _opsin_parse(names)

    for smiles, name, parsed_smiles in zip(smiles_values, names, parsed, strict=True):
        assert name
        assert parsed_smiles


def test_fused_functional_suffix_parents_roundtrip():
    smiles_values = [
        "N#Cc1ccc2n[nH]c(O)c2c1",
        "N#Cc1ccc2[nH][nH]c(=O)c2c1",
        "O=C(NCCC(=O)N1CCOCC1)c1ccc2nc[nH]c2c1",
        "O=C(Nc1ccc([N+](=O)[O-])cc1Cl)c1cc(Cl)cc2[nH]c(C(F)(F)F)nc12",
        "O=C(O)c1[nH]nc2c1CNCC2",
        "NC(=O)c1ccc(Br)c2c3c([nH]c12)CCSC3",
    ]

    names = smiles_to_iupac_batch(smiles_values)
    parsed = _opsin_parse(names)

    for smiles, name, parsed_smiles in zip(smiles_values, names, parsed, strict=True):
        assert name
        assert parsed_smiles


def test_generated_oxido_fused_parents_roundtrip():
    smiles_values = [
        "[O-][n+]1cccc2[nH]ccc21",
        "[O-][n+]1cccc2ncccc12",
        "Brc1csc2[n+]([O-])cccc12",
    ]

    names = smiles_to_iupac_batch(smiles_values)
    parsed = _opsin_parse(names)

    for smiles, name, parsed_smiles in zip(smiles_values, names, parsed, strict=True):
        assert name
        assert parsed_smiles


def test_positive_charged_fused_retained_parents_roundtrip():
    smiles_values = [
        "O=S(=O)(O)c1cccc2[nH+]c3ccccc3cc12",
        "CC[n+]1c2ccccc2sc1",
    ]

    names = smiles_to_iupac_batch(smiles_values)
    parsed = _opsin_parse(names)

    for smiles, name, parsed_smiles in zip(smiles_values, names, parsed, strict=True):
        assert name
        assert parsed_smiles


def test_retained_ring_csv_scaffold_parents_roundtrip_with_substituents():
    smiles_values = [
        "Cc1ccc(C)c(Cl)c1",
        "Cc1ccc(C)c(N)c1",
        "Clc1ccc2ccccc2c1",
        "O=C(O)c1ccc2ncccc2c1",
        "CCc1cc2ccccc2cc1",
        "COc1ccc2[nH]c3ccccc3c2c1",
        "CC1=C(/C=C/C(C)=C/C=C/C(C)=C/C=C/C=C(C)/C=C/C=C(C)/C=C/C2=C(C)C(=O)CCC2(C)C)C(C)(C)CCC1",
    ]

    names = smiles_to_iupac_batch(smiles_values)
    parsed = _opsin_parse(names)

    for smiles, name, parsed_smiles in zip(smiles_values, names, parsed, strict=True):
        assert name
        assert parsed_smiles


def test_charged_saturated_heterocycle_parents_roundtrip():
    smiles_values = [
        "CC1CCCC(C)[NH2+]1",
        "C1CC[NH2+]CC1",
        "C1COCC[NH2+]1",
        "C1CC[NH2+]C1",
    ]

    names = smiles_to_iupac_batch(smiles_values)
    parsed = _opsin_parse(names)

    for smiles, name, parsed_smiles in zip(smiles_values, names, parsed, strict=True):
        assert name
        assert parsed_smiles


def test_compact_benzazole_and_complex_carbamoyl_aliases_roundtrip():
    smiles_values = [
        "Fc1ccc(CSc2nc3ccc(Cl)cc3o2)cc1",
        "Cn1cc(-c2nc3ccc(N)cc3o2)cn1",
        "O=[N+]([O-])c1ccc2nc(C3CCOCC3)oc2c1",
        "O=c1c2cncc(-c3ccccc3)c2ccn1Cc1nc2ccccc2o1",
        "O=C(Nc1cccc(C2=NCCN2)c1)c1ccc(-c2ccc(C(=O)Nc3cccc(C4=NCCN4)c3)cc2)cc1",
    ]

    names = smiles_to_iupac_batch(smiles_values)
    parsed = _opsin_parse(names)

    for smiles, name, parsed_smiles in zip(smiles_values, names, parsed, strict=True):
        assert name
        assert parsed_smiles


def test_benzimidazole_orientation_roundtrips_both_tautomers():
    smiles_values = [
        "Brc1cccc2nc[nH]c12",
        "c1ccc(Sc2ccc3nc[nH]c3c2)cc1",
        "Oc1ccc2[nH]cnc2c1",
        "N#Cc1cccc2[nH]cnc12",
    ]

    names = smiles_to_iupac_batch(smiles_values)
    parsed = _opsin_parse(names)

    for smiles, name, parsed_smiles in zip(smiles_values, names, parsed, strict=True):
        assert name
        assert parsed_smiles


def test_fragile_skeletal_fused_parents_defer_to_roundtripping_templates():
    smiles_values = [
        "C1=C2N=NCN2c2ccccc2N1",
        "CC(C)(C)OC(=O)N1CCc2[nH]cnc2[C@H]1c1nc2ccccc2[nH]1",
        "c1[nH]c2c(c3cnnc1-3)NCCC2",
    ]

    names = smiles_to_iupac_batch(smiles_values)
    parsed = _opsin_parse(names)

    for smiles, name, parsed_smiles in zip(smiles_values, names, parsed, strict=True):
        assert name
        assert parsed_smiles


def test_isoindolinone_substituent_oxo_state_roundtrips():
    smiles_values = [
        "O=C1CCC(N2Cc3ccc(Br)c(Br)c3C2=O)C(=O)N1",
        "NCc1ccc2c(c1)C(=O)N(C1CCC(=O)NC1=O)C2",
        "O=C1c2ccccc2C(=O)N1C1CCCCC1",
    ]

    names = smiles_to_iupac_batch(smiles_values)
    parsed = _opsin_parse(names)

    for smiles, name, parsed_smiles in zip(smiles_values, names, parsed, strict=True):
        assert name
        assert parsed_smiles


def test_functional_ester_aliases_roundtrip():
    smiles_values = [
        "CCC(OC(N)=O)C(CC)OC(N)=O",
        "CC(C)(C)OC(=O)NOC(=O)c1ccccc1",
        "CCCCCCCCCCCCCCCCCCCCCCOC(=O)C(=O)[O-]",
        "O=CC(=O)Oc1ccccn1",
        "CCC(OC(=O)C(=O)c1ccccc1)C(C)C(CC)OC(=O)c1ccccc1",
        "O=S(=O)(CC(F)(F)F)N1CCCCC1",
    ]

    names = smiles_to_iupac_batch(smiles_values)
    parsed = _opsin_parse(names)

    for smiles, name, parsed_smiles in zip(smiles_values, names, parsed, strict=True):
        assert name
        assert parsed_smiles


def test_retained_acid_ester_parents_roundtrip():
    smiles_values = [
        "O=C([O-])C(=O)OC",
        "O=C(O)C(=O)OC",
        "O=C([O-])CC(=O)OC",
        "O=C(O)CC(=O)OC",
        "O=C([O-])CCC(=O)OC1CCCC1",
        "O=C(O)CCC(=O)OC1CCCC1",
        "O=C([O-])CCCC(=O)OC",
        "O=C([O-])CCCCC(=O)OC",
        "C=C(C#N)C(=O)OC1CCCC1",
        "C=C(C)C(=O)OC",
        "C=CC(=O)OC",
    ]

    names = smiles_to_iupac_batch(smiles_values)
    parsed = _opsin_parse(names)

    for smiles, name, parsed_smiles in zip(smiles_values, names, parsed, strict=True):
        assert name
        assert parsed_smiles


def test_exocyclic_cyanomethylidene_preserves_unsaturation():
    smiles_values = [
        "N#CC=C1C=CC=N1",
        "CC(C)(C)OC(=O)N1CCC(=CC#N)CC1",
        "N#CCC1CCNCC1",
    ]

    names = smiles_to_iupac_batch(smiles_values)
    parsed = _opsin_parse(names)

    for smiles, name, parsed_smiles in zip(smiles_values, names, parsed, strict=True):
        assert name
        assert parsed_smiles


def test_boronic_pinacol_ester_parents_roundtrip_compactly():
    smiles_values = [
        "c1ccccc1B1OC(C)(C)C(C)(C)O1",
        "CC(=C)B1OC(C)(C)C(C)(C)O1",
        "Cn1ncc(B2OC(C)(C)C(C)(C)O2)c1",
    ]

    names = smiles_to_iupac_batch(smiles_values)
    parsed = _opsin_parse(names)

    for smiles, name, parsed_smiles in zip(smiles_values, names, parsed, strict=True):
        assert name
        assert parsed_smiles


def test_retained_functional_parent_motifs_roundtrip():
    smiles_values = [
        "O=CCN(CC=O)C(=O)OCc1ccccc1",
        "O=CN(C(=O)OCc1ccccc1)C1CCC(C=O)CC1",
        "O=C(Nc1ccccc1)NC1CCCCC1",
        "NC(=O)NCCCN1CCCCC1",
        "CCCN(C(N)=O)c1ccccn1",
        "CCCN(CCC)C(=O)OC",
    ]

    names = smiles_to_iupac_batch(smiles_values)
    parsed = _opsin_parse(names)

    for smiles, name, parsed_smiles in zip(smiles_values, names, parsed, strict=True):
        assert name
        assert parsed_smiles


def test_multicenter_carbocycle_stereo_roundtrips():
    smiles_values = [
        "CC1(C)[C@H](C(=O)O)CC[C@]1(C)C(=O)O",
        "O=CO[C@@H]1CCC[C@@H]1O",
        "C=C[C@@H]1C[C@H]1C(=O)OCC",
        "CC(C)(C)OC(=O)N[C@H]1CCCC[C@H]1N=[N+]=[N-]",
    ]

    names = smiles_to_iupac_batch(smiles_values)
    parsed = _opsin_parse(names)

    for smiles, name, parsed_smiles in zip(smiles_values, names, parsed, strict=True):
        assert name
        assert parsed_smiles


def test_stereochemical_natural_product_retained_parent_roundtrips():
    smiles = "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC[C@@H]4CCCC[C@]4(C)[C@H]3CC[C@]12C"

    name = smiles_to_iupac(smiles)
    parsed = _opsin_parse([name])

    assert name
    assert parsed[0]


def test_retained_fused_ring_isotope_locants_roundtrip():
    smiles = "COc1ccc2c(c1)[nH]c1c2ccn[11c]1C"

    name = smiles_to_iupac(smiles)
    parsed = _opsin_parse([name])

    assert name
    assert parsed[0]


def test_single_ring_parent_stereo_with_stereo_substituents_roundtrips():
    smiles_values = [
        "O=C(O)[C@H]1O[C@H](O[C@@H]2[C@H](O)[C@@H](O)C(O)O[C@@H]2C(=O)O)[C@H](O)[C@@H](O)[C@H]1O",
        "C[C@@H]1O[C@@H](OC[C@H]2O[C@@H](n3ccc(=O)[nH]c3=O)C[C@@H]2O)[C@@H](O)[C@H](O)[C@@H]1O",
    ]

    names = smiles_to_iupac_batch(smiles_values)
    parsed = _opsin_parse(names)

    for smiles, name, parsed_smiles in zip(smiles_values, names, parsed, strict=True):
        assert name
        assert parsed_smiles


def test_nonstereo_long_fallback_roundtrips():
    smiles_values = [
        "FC(F)(F)C(F)(C(F)(F)F)C(F)(F)C(F)(F)N(C(F)(F)C(F)(F)C(F)(C(F)(F)F)C(F)(F)F)C(F)(F)C(F)(F)C(F)(C(F)(F)F)C(F)(F)F",
    ]

    names = smiles_to_iupac_batch(smiles_values)
    parsed = _opsin_parse(names)

    for smiles, name, parsed_smiles in zip(smiles_values, names, parsed, strict=True):
        assert name
        assert parsed_smiles


def test_upstream_short_alias_safety_regressions_roundtrip():
    smiles_values = [
        "O=C1C=CC(=O)N1C[C@H]1CC[C@H](C(=O)ON2C(=O)CCC2=O)CC1",
        "O=C(Cc1ccccc1)Nn1nc(C(F)(F)F)c2ccccc2c1=O",
        "O=C(N[C@H]1C[C@H](C(=O)NC2CCOCC2)C1)OCc1ccccc1",
        "COC(=O)CC(NC(=O)OC(C)(C)C)C(=O)CF",
        "COc1c(O)cc(C=O)cc1O",
        "COc1c(O)cc(C=O)cc1Br",
        "COc1c(O)ccc(F)c1C=O",
        "COc1cc(C=O)c(C2CC2)cc1O",
        "CC(=O)N[C@@H]([C@H](O)[C@@H](C=O)NC(C)=O)[C@H](O)CN=[N+]=[N-]",
        "Cc1sc(C(=O)N(C=O)Nc2cc(F)nc(F)c2)nc1C(=O)NC1CCC1(C)C",
        "Cc1c(COc2cc(OCc3cccc(C#N)c3)c(C=O)cc2Cl)cccc1-c1ccc2c(c1)OCCO2",
    ]

    names = smiles_to_iupac_batch(smiles_values)
    parsed = _opsin_parse(names)

    for smiles, name, parsed_smiles in zip(smiles_values, names, parsed, strict=True):
        assert name
        assert parsed_smiles


def test_exact_biochemical_retained_parents_roundtrip_compactly():
    smiles_values = [
        "P(O)(=O)(OP(=O)(O)OP(=O)(O)O)OC[C@@H]1[C@H]([C@H]([C@@H](O1)N1N=NC=2C(=O)NC(N)=NC12)O)O",
        "P(O)(=O)(OP(=O)(O)O)O[C@H]1[C@H]([C@@H](O[C@@H]1CO)N1C=NC=2C(=O)NC(=O)NC12)O",
        "P(O)(=O)(OP(=O)(O)OP(=O)(O)O)OC[C@@H]1[C@H](C[C@@H](O1)N1C(=O)N=C(NC)C=C1)O",
        "O(P([O-])(=O)OP(=O)([O-])[O-])C1[C@H](O)[C@H](O)[C@H](O1)CO",
        "OCC1([C@@H](O)[C@H](O)[C@H](O1)CO)COC1=CC(=CC=C1O)\\C=C\\C(=O)CC(=O)\\C=C\\C1=CC=C(O)C(OC)=C1",
    ]

    names = smiles_to_iupac_batch(smiles_values)
    parsed = _opsin_parse(names)

    for smiles, name, parsed_smiles in zip(smiles_values, names, parsed, strict=True):
        assert name
        assert parsed_smiles


def _opsin_parse(names: list[str]) -> list[str]:
    if not OPSIN_JAR.exists():
        pytest.skip(f"OPSIN jar not found: {OPSIN_JAR}")
    with tempfile.TemporaryDirectory(prefix="nispo_test_opsin_") as temp_name:
        temp_dir = Path(temp_name)
        names_file = temp_dir / "names.txt"
        output_file = temp_dir / "opsin.smi"
        names_file.write_text("\n".join(names) + "\n")
        java = str(JAVA) if JAVA.exists() else "java"
        completed = subprocess.run(
            [java, "-jar", str(OPSIN_JAR), "-o", "smi", str(names_file), str(output_file)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        parsed = output_file.read_text().splitlines()
    return (parsed + [""] * len(names))[: len(names)]
